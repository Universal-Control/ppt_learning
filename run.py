import os
import sys
from typing import Union, Optional
import logging

import hydra
from tqdm import trange

import wandb
from omegaconf import OmegaConf

import torch
from torch.utils import data
from torch.utils.data import DataLoader

from ppt_learning.utils import learning, model_utils, logging_utils
from ppt_learning.utils.warmup_lr_wrapper import WarmupLR
from ppt_learning.paths import *

sys.path.append(f"{PPT_DIR}/third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ppt_learning import train_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=f"configs",
    config_name="config",
    version_base="1.2",
)
def run(cfg: OmegaConf) -> None:
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Register custom OmegaConf resolver for mathematical expressions
    OmegaConf.register_new_resolver("eval", eval)

    is_eval = cfg.train.total_epochs == 0

    device = "cuda"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    
    domain = cfg.get("dataset_path", "debug").split("/")[-1] # domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)

    if not cfg.debug:
        run = wandb.init(
            project=domain,
            name=cfg.suffix,
            tags=[cfg.wb_tag],
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=False,
            resume="allow",
        )
        logger.info(f"W&B URL: {wandb.run.get_url()}")

    output_dir_full = cfg.output_dir.split("/")
    output_dir = "/".join(output_dir_full[:-2] + [domain, ""])
    if len(cfg.suffix):
        output_dir += f"{cfg.suffix}"
    else:
        output_dir += "-".join(output_dir_full[-2:])
    if is_eval:
        output_dir += "-eval"
    cfg.output_dir = output_dir

    use_pcd = "pointcloud" in cfg.stem.modalities
    if use_pcd:
        cfg.dataset.use_pcd = use_pcd
        cfg.dataset.pcdnet_pretrain_domain = (
            cfg.rollout_runner.pcdnet_pretrain_domain
        ) = cfg.stem.pointcloud.pcd_domain
        cfg.rollout_runner.pcd_channels = cfg.dataset.pcd_channels
    cfg.dataset.horizon = (
        cfg.dataset.observation_horizon + cfg.dataset.action_horizon - 1
    )
    cfg.dataset.domain = domain

    normalizer = None
    if not is_eval:
        cfg.dataset.dataset_path = (
            cfg.get("dataset_path", "") + "/" + domain_list[0] + ".zarr"
            if len(domain_list) == 1
            else [cfg.get("dataset_path", "") + "/" + domain + ".zarr" for domain in domain_list]
        )
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            **cfg.dataset,
        )
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        pcd_num_points = 1024
        if use_pcd:
            pcd_num_points = dataset.pcd_num_points
            assert pcd_num_points is not None

        train_loader = data.DataLoader(
            dataset, **cfg.dataloader, multiprocessing_context="fork"
        )
        test_loader = data.DataLoader(
            val_dataset, **cfg.val_dataloader, multiprocessing_context="fork"
        )

        logger.info(f"Train size: {len(dataset)}, Test size: {len(val_dataset)}")

        action_dim = dataset.action_dim
        state_dim = dataset.state_dim

    # initialize policy
    if cfg.dataset.get("hist_action_cond", False):
        cfg.head["hist_horizon"] = cfg.dataset.observation_horizon
    cfg.head["output_dim"] = cfg.network["action_dim"] = action_dim

    learning.save_args_hydra(cfg.output_dir, cfg)

    logger.info(f"Configuration: {cfg}")
    logger.info(f"Output directory: {cfg.output_dir}")

    policy = hydra.utils.instantiate(cfg.network)
    cfg.stem.state["input_dim"] = state_dim
    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, cfg.head, normalizer=normalizer)

    # optimizer and scheduler
    policy.finalize_modules()
    logger.info(f"Pretrained directory: {cfg.train.pretrained_dir}")

    loaded_epoch = -1
    if len(cfg.train.pretrained_dir) > 0:
        if "pth" in cfg.train.pretrained_dir:
            assert os.path.exists(
                cfg.train.pretrained_dir
            ), "Pretrained model not found"
            logger.info(f"Loading model from {cfg.train.pretrained_dir}")
            policy.load_state_dict(torch.load(cfg.train.pretrained_dir))
            loaded_epoch = int(
                cfg.train.pretrained_dir.split("/")[-1].split(".")[0].split("_")[-1]
            )
        else:
            assert os.path.exists(
                os.path.join(cfg.train.pretrained_dir, f"model.pth")
            ), "Pretrained model not found"
            policy.load_state_dict(
                torch.load(os.path.join(cfg.train.pretrained_dir, f"model.pth"))
            )

        logger.info("Loaded trunk")
        # policy.load_trunk(os.path.join(cfg.train.pretrained_dir, f"model.pth"))
        if cfg.train.freeze_trunk:
            policy.freeze_trunk()
            logger.info("Trunk frozen")
    else:
        logger.info("Training from scratch")

    policy.to(device)

    total_steps = cfg.train.total_epochs * len(train_loader)
    opt = learning.get_optimizer(cfg.optimizer, policy)
    sch = learning.get_scheduler(cfg.lr_scheduler, opt, num_warmup_steps=cfg.warmup_lr.step, num_training_steps=total_steps)

    # sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)
    # sch = WarmupLR(
    #     sch,
    #     init_lr=cfg.warmup_lr.lr,
    #     num_warmup=cfg.warmup_lr.step,
    #     warmup_strategy="constant",
    # )
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Number of parameters (M): {n_parameters / 1.0e6:.2f}")

    if not is_eval:
        # train / test loop
        pbar = trange(
            loaded_epoch + 1, loaded_epoch + 1 + cfg.train.total_epochs, position=0
        )
        for epoch in pbar:
            train_stats = train_test.train(
                cfg.log_interval,
                policy,
                device,
                train_loader,
                opt,
                sch,
                epoch,
                pcd_npoints=pcd_num_points,
                in_channels=dataset.pcd_channels,
                debug=cfg.debug,
                epoch_size=cfg.train.epoch_iters
            )
            test_loss = train_test.test(
                policy,
                device,
                test_loader,
                epoch,
                pcd_npoints=pcd_num_points,
                in_channels=dataset.pcd_channels,
                debug=cfg.debug,
            )
            train_steps = (epoch + 1) * len(train_loader)

            # Save the policy every epoch
            if epoch % cfg.save_interval == 0:
                policy_path = os.path.join(cfg.output_dir, f"model_{epoch}.pth")
            else:
                policy_path = os.path.join(cfg.output_dir, f"model.pth")
            policy.save(policy_path)
            if "loss" in train_stats:
                pbar.set_description(
                    f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}. Test loss: {test_loss:.4f}"
                )

            if train_steps > cfg.train.total_iters:
                break

        policy.save(policy_path)
        pbar.close()

    # Evaluate jointly trained policy
    if cfg.parallel_eval:
        total_success = train_test.eval_policy_parallel(policy, cfg)
    else:
        total_success = train_test.eval_policy_sequential(policy, cfg)

    logger.info(f"Saved results to: {cfg.output_dir}")
    # save the results
    logging_utils.log_results(cfg, total_success)


if __name__ == "__main__":
    run()
