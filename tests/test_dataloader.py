import os, sys
from typing import Union

import hydra
from hydra import initialize, compose
import argparse
from tqdm import trange
import wandb
import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import random
from pathlib import Path
import ppt_learning

PPT_DIR = Path(ppt_learning.__path__[-1])
sys.path.append(f"{PPT_DIR}/../third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_dataloader(dataset, seed, rank, world_size, **kwargs):

    shuffle = kwargs.pop("shuffle", False)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed + rank
    )

    # Create DataLoader with the sampler
    dataloader = DataLoader(
        dataset, **kwargs, sampler=sampler, multiprocessing_context="fork"
    )

    return dataloader


def run(rank: int, world_size: int, cfg: DictConfig):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """
    # Initialize DDP process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    is_eval = cfg.train.total_epochs == 0

    device = torch.device(f"cuda:{rank}")
    domain_list = [d.strip() for d in cfg.domains.split(",")]

    domain = cfg.get("dataset_path", "debug").split("/")[
        -1
    ]  # domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)

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
        cfg.dataset.pcdnet_pretrain_domain = (
            cfg.rollout_runner.pcdnet_pretrain_domain
        ) = cfg.stem.pointcloud.pcd_domain
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
            cfg.get("dataset_path", "") + "/" + domain_list[0] + ".zarr"
            if len(domain_list) == 1
            else [
                cfg.get("dataset_path", "") + "/" + domain + ".zarr"
                for domain in domain_list
            ]
        )
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            seed=cfg.seed + rank,
            **cfg.dataset,
        )
        val_dataset = dataset.get_validation_dataset()
        pcd_num_points = 1024
        if use_pcd:
            pcd_num_points = dataset.pcd_num_points
            assert pcd_num_points is not None
        train_loader = get_dataloader(
            dataset, cfg.seed, rank=rank, world_size=world_size, **cfg.dataloader
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

    cfg.stem.state["input_dim"] = state_dim

    # Ensure only rank 0 creates output directory
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
    dist.barrier()  # Wait for rank 0 to create directory

    if rank == 0:
        print("cfg.train.pretrained_dir:", cfg.train.pretrained_dir)

    epoch_size = len(train_loader) // 8
    loaded_epoch = 0
    pbar = trange(
        loaded_epoch + 1, loaded_epoch + 1 + cfg.train.total_epochs, position=0
    )

    if not is_eval:
        # train / test loop
        for epoch in pbar:
            start_time = time.time()
            pbar_local = tqdm(
                total=epoch_size,
                position=1,
                leave=True,
                disable=(rank != 0),
                desc=f"Epoch {epoch}",
            )

            for i, batch in enumerate(train_loader):
                pbar_local.update(1)
                if i >= (epoch_size):
                    break
            print("Time to finish an epoch")
            # break

    print("saved results to:", cfg.output_dir)
    dist.destroy_process_group()


def filter_ddp_args():
    """Filter out DDP-specific arguments to avoid Hydra conflicts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model", type=str, default="pcd")  # ["pcd", "rgb", "PCD"])
    parser.add_argument("--suffix", type=str, default="")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.local_rank, args.model, args.suffix


def main():
    # Filter DDP arguments
    local_rank, model_type, suffix = filter_ddp_args()

    # Initialize Hydra
    with initialize(config_path=f"../configs", version_base="1.2"):
        cfg = compose(config_name=f"config_ddp_{model_type}")

    # Resolve any remaining interpolations
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)  # Convert back to DictConfig

    cfg.suffix = cfg.suffix if len(cfg.suffix) else suffix
    cfg.wb_tag = cfg.suffix if len(cfg.suffix) else "default"
    # Ensure output_dir has a default value if not set
    if not cfg.get("output_dir"):
        cfg.output_dir = os.path.join(
            "outputs", model_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    cfg.wb_tag = cfg.suffix if len(cfg.suffix) else "default"

    # Determine world size (number of GPUs)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available for DDP training.")

    # Spawn DDP processes
    # mp.spawn(run, args=(world_size, cfg), nprocs=world_size, join=True)
    run(0, 1, cfg)


if __name__ == "__main__":
    if "ARNOLD_WORKER_0_HOST" in os.environ:
        os.environ["MASTER_IP"] = os.environ["ARNOLD_WORKER_0_HOST"]
        os.environ["MASTER_ADDR"] = os.environ["ARNOLD_WORKER_0_HOST"]
        Port = os.environ["ARNOLD_WORKER_0_PORT"].split(",")
        os.environ["WORLD_SIZE"] = os.environ["NODE_SIZE"] = os.environ[
            "ARNOLD_WORKER_NUM"
        ]
        if True:  # int(os.environ["WORLD_SIZE"]) > 1:
            os.environ["MASTER_PORT"] = Port[0]
        else:
            for p in Port:
                if check_port(int(p)) == False:
                    os.environ["MASTER_PORT"] = p
                    break
        os.environ["RANK"] = os.environ["NODE_RANK"] = os.environ["ARNOLD_ID"]
        print(f"ARNOLD_WORKER_0_PORT: {os.environ['ARNOLD_WORKER_0_PORT']}")
    try:
        print(
            f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}"
        )
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}, RANK: {os.environ['RANK']}")
    except:
        pass
    main()
