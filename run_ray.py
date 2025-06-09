import os, sys
from typing import Union

import hydra
from hydra import initialize, compose
import argparse
from tqdm import trange, tqdm
import csv
import wandb
import datetime
from omegaconf import DictConfig, OmegaConf
from collections import deque, OrderedDict
import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from ppt_learning.utils import utils, model_utils
from ppt_learning.utils.warmup_lr_wrapper import WarmupLR
from ppt_learning.paths import *
from ppt_learning.utils.utils import dict_apply, batchify, sample_pcd_data, unbatchify

import ray.train.torch
from ray import train

sys.path.append(f"{PPT_DIR}/third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ppt_learning import train_test

info_key = [
    "loss",
    "max_action",
    "max_label_action",
    "max_gradient",
    "max_stem_gradient",
    "max_trunk_gradient",
    "max_head_gradient",
    "batch_time",
    "mean_param",
    "lr",
]

def get_dataloader(dataset, **kwargs):

    # Create DataLoader with the sampler
    dataloader = DataLoader(
        dataset, **kwargs,
    )
    dataloader = ray.train.torch.prepare_data_loader(dataloader)
    return dataloader


def train_one_epoch(
    log_interval,
    model,
    device,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    rank=0,
    world_size=1,
    pcd_npoints=8192,
    in_channels=4,
    log_name="train",
    debug=True,
    epoch_size=None, # Not used for now
):
    """training for one epoch"""
    info_log = {k: deque([], maxlen=20) for k in info_key}
    model.train()
    start_time = time.time()

    # combined_dataloader = train_loader  # WeightedDataLoader(train_loaders)
    epoch_size = len(train_loader)
    assert epoch_size > 0, "empty dataloader"
    
    # Create progress bar for rank 0 only
    pbar = tqdm(
        total=epoch_size,
        position=1,
        leave=True,
        disable=(rank != 0),
        desc=f"Epoch {epoch}"
    )

    # Iterate through batches
    for batch_idx, batch in enumerate(train_loader):
        pbar.update(1)
        # batch["data"] = dict_apply(
        #     batch["data"], lambda x: x.to(device, non_blocking=True).float()
        # )

        batch["data"] = batchify(batch["data"], exclude=["action"])

        if "pointcloud" in batch["data"]:
            sample_pcd_data(
                batch["data"]["pointcloud"],
                npoints=pcd_npoints,
                in_channels=in_channels,
            )

        data_time = time.time() - start_time
        start_time = time.time()
        output = model.module.forward_train(batch)

        target = batch["data"]["action"]
        if rank == 0 and target.shape[1] == 0:
            print("empty target:", target.shape)
            continue

        if isinstance(output, (dict, OrderedDict)):
            domain_loss = output["loss"]
            output = None
        else:
            output = output.reshape(target.shape)
            domain_loss = F.mse_loss(output, target)

        # backward
        optimizer.zero_grad()
        domain_loss.backward()
        optimizer.step()
        scheduler.step(epoch + batch_idx / epoch_size)
        train_step = len(train_loader) * epoch + batch_idx

        model_ = model.module

        # Log stats (only rank 0, with aggregated metrics)
        if rank == 0:
            train_test.log_stat(
                info_log,
                train_step,
                log_interval,
                log_name,
                batch["domain"][0],
                target,
                domain_loss,
                output,
                model_,
                optimizer,
                start_time,
                use_wandb=False,
            )

            step_time = time.time() - start_time
            pbar.set_description(
                f"Epoch: {epoch} Step: {batch_idx}/{epoch_size} Time: {step_time:.3f} {data_time:.3f} Loss: {info_log[batch['domain'][0] + '_loss'][-1]:.3f} Grad: {info_log['max_gradient'][-1]:.3f}"
            )
        start_time = time.time()

    return {k: np.mean(v) for k, v in info_log.items() if len(v) > 1}

def run(config):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """
    # Initialize DDP process group
    cfg: DictConfig = config["cfg"]
    is_eval = cfg.train.total_epochs == 0

    device = train.torch.get_device()
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    
    domain = cfg.get("dataset_path", "debug").split("/")[-1] # domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)

    rank = ray.train.get_context().get_world_rank()
    # Setup wandb (only for rank 0)
    # if rank == 0:
    #     run = wandb.init(
    #         project=domain,
    #         name=cfg.suffix,
    #         tags=[cfg.wb_tag],
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #         reinit=False,
    #         resume="allow",
    #     )
    #     print("wandb url:", wandb.run.get_url())

    output_dir_full = cfg.output_dir.split("/")
    output_dir = "/".join(output_dir_full + [domain, ""])
    if len(cfg.suffix):
        output_dir += f"{cfg.suffix}"
    if is_eval:
        output_dir += "-eval"
    cfg.output_dir = output_dir

    use_pcd = "pointcloud" in cfg.stem.modalities
    if use_pcd:
        cfg.dataset.use_pcd = use_pcd
        cfg.dataset.pcdnet_pretrain_domain = cfg.rollout_runner.pcdnet_pretrain_domain = cfg.stem.pointcloud.pcd_domain
        cfg.rollout_runner.pcd_channels = cfg.dataset.pcd_channels
        
    if cfg.dataset.get("hist_action_cond", False):
        cfg.head["hist_horizon"] = cfg.dataset.observation_horizon
    cfg.dataset.horizon = (
        cfg.dataset.observation_horizon + cfg.dataset.action_horizon - 1
    )
    cfg.dataset.pad_before -= 1
    cfg.dataset.pad_after -= 1
    cfg.dataset.domain = domain

    seed = cfg.seed
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    normalizer = None
    if not is_eval:
        cfg.dataset.dataset_path = (
            [cfg.get("dataset_path", "") + "/" + domain + ".zarr" for domain in domain_list]
        )
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            seed=cfg.seed + rank,
            **cfg.dataset,
        )
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        pcd_num_points = 1024
        if use_pcd:
            pcd_num_points = dataset.pcd_num_points
            assert pcd_num_points is not None
        train_loader = get_dataloader(
            dataset, **cfg.dataloader
        )
        test_loader = get_dataloader(
            val_dataset,
            **cfg.val_dataloader,
        )
        if rank == 0:
            print(f"Train size: {len(dataset)}. Test size: {len(val_dataset)}.")

        action_dim = dataset.action_dim
        state_dim = dataset.state_dim

    # initialize policy
    cfg.head["output_dim"] = cfg.network["action_dim"] = action_dim

    if rank == 0:
        print("cfg: ", cfg)
        print("output dir", cfg.output_dir)

    policy = hydra.utils.instantiate(cfg.network)
    cfg.stem.state["input_dim"] = state_dim

    # Ensure only rank 0 creates output directory
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        utils.save_args_hydra(cfg.output_dir, cfg)
    dist.barrier()  # Wait for rank 0 to create directory

    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, cfg.head, normalizer=normalizer)

    # optimizer and scheduler
    policy.finalize_modules()

    loaded_epoch = -1
    if len(cfg.train.pretrained_dir) > 0:
        if "pth" in cfg.train.pretrained_dir:
            assert os.path.exists(
                cfg.train.pretrained_dir
            ), "Pretrained model not found"
            policy.load_state_dict(torch.load(cfg.train.pretrained_dir))
            loaded_epoch = int(
                cfg.train.pretrained_dir.split("/")[-1].split(".")[0].split("_")[-1]
            )
            if rank == 0:
                print("load model from", cfg.train.pretrained_dir, "loaded_epoch", loaded_epoch)
        else:
            assert os.path.exists(
                os.path.join(cfg.train.pretrained_dir, f"model.pth")
            ), "Pretrained model not found"
            policy.load_state_dict(
                torch.load(os.path.join(cfg.train.pretrained_dir, f"model.pth"))
            )
            if rank == 0:
                print("load model from", cfg.train.pretrained_dir)
        if rank == 0:
            print("loaded trunk")
        # policy.load_trunk(os.path.join(cfg.train.pretrained_dir, f"model.pth"))
        if cfg.train.freeze_trunk:
            policy.freeze_trunk()
            print("trunk frozen")
    else:
        if rank == 0:
            print("train from scratch")

    policy.to(device)
    policy = ray.train.torch.prepare_model(policy)

    if rank == 0:
        print("cfg.train.pretrained_dir:", cfg.train.pretrained_dir)

    opt = utils.get_optimizer(cfg.optimizer, policy)
    sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)

    sch = WarmupLR(
        sch,
        init_lr=cfg.warmup_lr.lr,
        num_warmup=cfg.warmup_lr.step,
        warmup_strategy="constant",
    )
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if rank == 0:
        print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    if not is_eval:
        # train / test loop
        pbar = trange(
            loaded_epoch + 1, loaded_epoch + 1 + cfg.train.total_epochs, position=0
        )
        for epoch in pbar:
            if ray.train.get_context().get_world_size() > 1:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                cfg.log_interval,
                policy,
                device,
                train_loader,
                opt,
                sch,
                epoch,
                rank=rank,
                pcd_npoints=pcd_num_points,
                in_channels=dataset.pcd_channels,
                debug=cfg.debug,
                epoch_size=cfg.train.epoch_iters
            )
            train_steps = (epoch + 1) * len(train_loader)

            # Save policy (only rank 0)
            if rank == 0:
                if epoch % cfg.save_interval == 0:
                    policy_path = os.path.join(cfg.output_dir, f"model_{epoch}.pth")
                else:
                    policy_path = os.path.join(cfg.output_dir, f"model.pth")
                policy.module.save(policy_path)

            if rank == 0 and "loss" in train_stats:
                pbar.set_description(
                    f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}."
                )

            if train_steps > cfg.train.total_iters:
                break

        if rank == 0:
            policy.module.save(policy_path)

        pbar.close()

    print("saved results to:", cfg.output_dir)
    # save the results
    # utils.log_results(cfg, total_success)


def get_args():
    """Filter out DDP-specific arguments to avoid Hydra conflicts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    parser.add_argument("--address", type=str, default="127.0.0.1:6379")
    parser.add_argument("--model", type=str, default="pcd") # ["pcd", "rgb", "PCD"])
    parser.add_argument("--suffix", type=str, default="")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    args = get_args()
    model_type, suffix = args.model, args.suffix

    ray.init(args.address)

    # Initialize Hydra
    with initialize(config_path=f"ppt_learning/experiments/configs", version_base="1.2"):
        cfg = compose(config_name=f"config_ddp_{model_type}")

    # Resolve any remaining interpolations
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)  # Convert back to DictConfig

    cfg.suffix = cfg.suffix if len(cfg.suffix) else suffix
    cfg.wb_tag = cfg.suffix if len(cfg.suffix) else "default"
    # Ensure output_dir has a default value if not set
    if not cfg.get("output_dir"):
        cfg.output_dir = os.path.join("outputs", model_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    cfg.wb_tag = cfg.suffix if len(cfg.suffix) else "default"

    scaling_config = ray.train.ScalingConfig(num_workers=args.num_workers, use_gpu=True)
    # Spawn DDP processes
    trainer = ray.train.torch.TorchTrainer(
        run,
        scaling_config=scaling_config,
        train_loop_config={"cfg": cfg}
    )
    trainer.fit()


if __name__ == "__main__":
    # if "ARNOLD_WORKER_0_HOST" in os.environ:
    #     os.environ["MASTER_IP"] = os.environ["ARNOLD_WORKER_0_HOST"]
    #     os.environ["MASTER_ADDR"] = os.environ["ARNOLD_WORKER_0_HOST"]
    #     Port = os.environ["ARNOLD_WORKER_0_PORT"].split(",")
    #     os.environ["WORLD_SIZE"] = os.environ["NODE_SIZE"] = os.environ[
    #         "ARNOLD_WORKER_NUM"
    #     ]
    #     if True:  # int(os.environ["WORLD_SIZE"]) > 1:
    #         os.environ["MASTER_PORT"] = Port[0]
    #     else:
    #         for p in Port:
    #             if check_port(int(p)) == False:
    #                 os.environ["MASTER_PORT"] = p
    #                 break
    #     os.environ["RANK"] = os.environ["NODE_RANK"] = os.environ["ARNOLD_ID"]
    #     print(f"ARNOLD_WORKER_0_PORT: {os.environ['ARNOLD_WORKER_0_PORT']}")
    # elif "ARNOLD_EXECUTOR_0_HOST" in os.environ:
    #     os.environ["MASTER_IP"] = os.environ["ARNOLD_EXECUTOR_0_HOST"]
    #     os.environ["MASTER_ADDR"] = os.environ["ARNOLD_EXECUTOR_0_HOST"]
    #     Port = os.environ["ARNOLD_EXECUTOR_0_PORT"].split(",")
    #     os.environ["WORLD_SIZE"] = os.environ["NODE_SIZE"] = os.environ[
    #         "ARNOLD_EXECUTOR_NUM"
    #     ]
    #     if True:  # int(os.environ["WORLD_SIZE"]) > 1:
    #         os.environ["MASTER_PORT"] = Port[0]
    #     else:
    #         for p in Port:
    #             if check_port(int(p)) == False:
    #                 os.environ["MASTER_PORT"] = p
    #                 break
    #     os.environ["RANK"] = os.environ["NODE_SIZE"] = os.environ["ARNOLD_ID"]
    #     print(f"ARNOLD_EXECUTOR_0_PORT: {os.environ['ARNOLD_EXECUTOR_0_PORT']}")
    # try:
    #     print(
    #         f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}"
    #     )
    #     print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}, RANK: {os.environ['RANK']}")
    # except:
    #     pass
    main()
