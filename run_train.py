import os
import sys
from typing import Union, Optional
import logging

import hydra
from tqdm import trange

import wandb
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data import DataLoader, DistributedSampler

from ppt_learning.utils import learning, model_utils, logging_utils
from ppt_learning.paths import *

sys.path.append(f"{PPT_DIR}/../third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ppt_learning import train_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training if multiple GPUs are available."""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(
    config_path=f"configs",
    config_name="config",
    version_base="1.2",
)
def run_train(cfg: OmegaConf) -> None:
    """
    Unified training script for both single-GPU and multi-GPU training.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Setup distributed training
    is_distributed, rank, local_rank, world_size = setup_distributed()
    print(f"Process Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    is_main_process = rank == 0
    
    # Register custom OmegaConf resolver for mathematical expressions
    OmegaConf.register_new_resolver("eval", eval)

    is_eval = cfg.train.total_epochs == 0
    
    # Use appropriate device based on distributed setup
    if is_distributed:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    
    domain = cfg.get("dataset_path", "debug").split("/")[-1]

    # Only initialize wandb on main process
    if not cfg.debug and is_main_process:
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
    
    # Add rank suffix for distributed training
    if is_distributed and not is_main_process:
        output_dir += f"_rank_{rank}"
        
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

        # Setup distributed samplers if using multiple GPUs
        train_sampler = None
        val_sampler = None
        if is_distributed:
            train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            # Don't shuffle in dataloader when using DistributedSampler
            cfg.dataloader.shuffle = False
            cfg.val_dataloader.shuffle = False

        train_loader = DataLoader(
            dataset,
            sampler=train_sampler,
            **cfg.dataloader,
            multiprocessing_context="fork"
        )
        test_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            **cfg.val_dataloader,
            multiprocessing_context="fork"
        )

        if is_main_process:
            logger.info(f"Train size: {len(dataset)}, Test size: {len(val_dataset)}")

        action_dim = dataset.action_dim
        state_dim = dataset.state_dim

    # Initialize policy
    if cfg.dataset.get("hist_action_cond", False):
        cfg.head["hist_horizon"] = cfg.dataset.observation_horizon
    cfg.head["output_dim"] = cfg.network["action_dim"] = action_dim

    if is_main_process:
        learning.save_args_hydra(cfg.output_dir, cfg)
        logger.info(f"Configuration: {cfg}")
        logger.info(f"Output directory: {cfg.output_dir}")

    policy = hydra.utils.instantiate(cfg.network)
    cfg.stem.state["input_dim"] = state_dim
    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, cfg.head, normalizer=normalizer)

    # Optimizer and scheduler
    policy.finalize_modules()
    
    if is_main_process:
        logger.info(f"Pretrained directory: {cfg.train.pretrained_dir}")

    loaded_epoch = -1
    if len(cfg.train.pretrained_dir) > 0:
        if "pth" in cfg.train.pretrained_dir:
            assert os.path.exists(
                cfg.train.pretrained_dir
            ), "Pretrained model not found"
            if is_main_process:
                logger.info(f"Loading model from {cfg.train.pretrained_dir}")
            policy.load_state_dict(torch.load(cfg.train.pretrained_dir, map_location=device))
            loaded_epoch = int(
                cfg.train.pretrained_dir.split("/")[-1].split(".")[0].split("_")[-1]
            )
        else:
            assert os.path.exists(
                os.path.join(cfg.train.pretrained_dir, f"model.pth")
            ), "Pretrained model not found"
            policy.load_state_dict(
                torch.load(os.path.join(cfg.train.pretrained_dir, f"model.pth"), map_location=device)
            )

        if is_main_process:
            logger.info("Loaded trunk")
        if cfg.train.freeze_trunk:
            policy.freeze_trunk()
            if is_main_process:
                logger.info("Trunk frozen")
    else:
        if is_main_process:
            logger.info("Training from scratch")

    policy.to(device)

    # Wrap model with DDP for distributed training
    if is_distributed:
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_without_ddp = policy.module
    else:
        model_without_ddp = policy

    total_steps = cfg.train.total_epochs * len(train_loader)
    
    # Only create optimizer on the actual policy (not DDP wrapper)
    opt = learning.get_optimizer(cfg.optimizer, model_without_ddp)
    sch = learning.get_scheduler(cfg.lr_scheduler, opt, num_warmup_steps=cfg.warmup_lr.step, num_training_steps=total_steps)

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    if is_main_process:
        logger.info(f"Number of parameters (M): {n_parameters / 1.0e6:.2f}")

    if not is_eval:
        # Train / test loop
        if is_main_process:
            pbar = trange(
                loaded_epoch + 1, loaded_epoch + 1 + cfg.train.total_epochs, position=0
            )
        else:
            pbar = range(loaded_epoch + 1, loaded_epoch + 1 + cfg.train.total_epochs)
            
        for epoch in pbar:
            # Set epoch for distributed sampler
            if is_distributed:
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)
            
            train_stats = train_test.train(
                cfg.log_interval,
                policy,
                device,
                train_loader,
                opt,
                sch,
                epoch,
                rank=rank,
                world_size=world_size,
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
                rank=rank,
                world_size=world_size,
                pcd_npoints=pcd_num_points,
                in_channels=dataset.pcd_channels,
                debug=cfg.debug,
            )
            train_steps = (epoch + 1) * len(train_loader)

            # Only save on main process
            if is_main_process:
                if epoch % cfg.save_interval == 0:
                    policy_path = os.path.join(cfg.output_dir, f"model_{epoch}.pth")
                else:
                    policy_path = os.path.join(cfg.output_dir, f"model.pth")
                model_without_ddp.save(policy_path)
                
                if "loss" in train_stats and hasattr(pbar, 'set_description'):
                    pbar.set_description(
                        f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}. Test loss: {test_loss:.4f}"
                    )

            if train_steps > cfg.train.total_iters:
                break

        if is_main_process:
            model_without_ddp.save(policy_path)
            if hasattr(pbar, 'close'):
                pbar.close()

    # Synchronize all processes before evaluation
    if is_distributed:
        dist.barrier()

    # Cleanup distributed training
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    run_train()
