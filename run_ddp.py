import os, sys
import glob
from typing import Union

import hydra
from hydra import initialize, compose
import argparse
from tqdm import trange
import csv
import wandb
import datetime
from omegaconf import DictConfig, OmegaConf

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ppt_learning.utils import learning, model_utils
from ppt_learning.utils.warmup_lr_wrapper import WarmupLR
from ppt_learning.paths import *

sys.path.append(f"{PPT_DIR}/third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ppt_learning import train_test


def remove_old_checkpoints(output_dir, k):
    """
    Remove the oldest checkpoints to keep only the last k checkpoints.
    """
    # Get all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(output_dir, "model_*.pth")), key=os.path.getmtime)
    
    # Remove the oldest checkpoints if there are more than k
    while len(checkpoint_files) > k:
        oldest_checkpoint = checkpoint_files.pop(0)
        os.remove(oldest_checkpoint)
        print(f"Removed old checkpoint: {oldest_checkpoint}")

def get_dataloader(dataset, seed, rank, world_size, **kwargs):

    shuffle = kwargs.pop("shuffle", False)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed + rank
    )

    # Create DataLoader with the sampler
    dataloader = DataLoader(
        dataset, **kwargs, sampler=sampler# , multiprocessing_context="fork"
    )

    return dataloader


def run(local_rank: int, world_size: int, cfg: DictConfig, node_rank: int = 0):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """
    # Compute global rank
    gpus_per_node = torch.cuda.device_count()
    rank = node_rank * gpus_per_node + local_rank

    # Initialize DDP process group
    print(f"Process {rank} initialized with world size {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    is_eval = cfg.train.total_epochs == 0

    device = torch.device(f"cuda:{local_rank}")
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    
    domain = cfg.get("dataset_path", "debug").split("/")[-1] # domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)

    # Setup wandb (only for rank 0)
    if not cfg.debug and rank == 0:
        run = wandb.init(
            project=domain,
            name=cfg.suffix,
            tags=[cfg.wb_tag],
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=False,
            # resume="allow",
            resume="must",
        )
        print("wandb url:", wandb.run.get_url())

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
            else [cfg.get("dataset_path", "") + "/" + domain + ".zarr" for domain in domain_list]
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
            dataset, cfg.seed, rank=rank, world_size=world_size, **cfg.dataloader
        )
        test_loader = get_dataloader(
            val_dataset,
            cfg.seed,
            rank=rank,
            world_size=world_size,
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
        learning.save_args_hydra(cfg.output_dir, cfg)
    dist.barrier()  # Wait for rank 0 to create directory

    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, cfg.head, normalizer=normalizer)

    # optimizer and scheduler
    policy.finalize_modules()

    loaded_epoch = -1
    if len(cfg.train.pretrained_dir) > 0:
        if ".pth" in cfg.train.pretrained_dir:
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
    policy = DDP(policy, device_ids=[rank])

    if rank == 0:
        print("cfg.train.pretrained_dir:", cfg.train.pretrained_dir)

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
    if rank == 0:
        print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

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
                rank=rank,
                world_size=world_size,
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

                # Get the number of checkpoints to keep from config
                k = cfg.train.get('last_k_checkpoints', 5)  # Default to 5 if not set
                remove_old_checkpoints(cfg.output_dir, k)

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

    dist.destroy_process_group()


def filter_ddp_args():
    """Filter out DDP-specific arguments to avoid Hydra conflicts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model", type=str, default="depth") # ["pcd", "rgb", "PCD"])
    parser.add_argument("--suffix", type=str, default="")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.local_rank, args.model, args.suffix


def main():
    # Filter DDP arguments
    local_rank, model_type, suffix = filter_ddp_args()

    # Initialize Hydra
    with initialize(config_path=f"ppt_learning/experiments/configs", version_base="1.2"):
        cfg = compose(config_name=f"config_ddp_{model_type}")

    # Resolve any remaining interpolations
    # Register custom OmegaConf resolver for mathematical expressions
    OmegaConf.register_new_resolver("eval", eval)

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)  # Convert back to DictConfig

    cfg.suffix = cfg.suffix if len(cfg.suffix) else suffix
    cfg.wb_tag = cfg.suffix if len(cfg.suffix) else "default"
    # Ensure output_dir has a default value if not set
    if not cfg.get("output_dir"):
        cfg.output_dir = os.path.join("outputs", model_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    cfg.wb_tag = cfg.suffix if len(cfg.suffix) else "default"

    # Get world size and rank from environment variables
    world_size = int(os.environ.get('WORLD_SIZE', 1)) * torch.cuda.device_count()
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    node_rank = int(os.environ.get('RANK', 0))

    if world_size < 1:
        raise RuntimeError("No GPUs available for DDP training.")

    # Spawn DDP processes
    if world_size > 1:
        mp.spawn(run, args=(world_size, cfg, node_rank), nprocs=torch.cuda.device_count(), join=True)
    else:
        run(local_rank, world_size, cfg)


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
    elif "ARNOLD_EXECUTOR_0_HOST" in os.environ:
        os.environ["MASTER_IP"] = os.environ["ARNOLD_EXECUTOR_0_HOST"]
        os.environ["MASTER_ADDR"] = os.environ["ARNOLD_EXECUTOR_0_HOST"]
        Port = os.environ["ARNOLD_EXECUTOR_0_PORT"].split(",")
        os.environ["WORLD_SIZE"] = os.environ["NODE_SIZE"] = os.environ[
            "ARNOLD_EXECUTOR_NUM"
        ]
        if True:  # int(os.environ["WORLD_SIZE"]) > 1:
            os.environ["MASTER_PORT"] = Port[0]
        else:
            for p in Port:
                if check_port(int(p)) == False:
                    os.environ["MASTER_PORT"] = p
                    break
        os.environ["RANK"] = os.environ["NODE_SIZE"] = os.environ["ARNOLD_ID"]
        print(f"ARNOLD_EXECUTOR_0_PORT: {os.environ['ARNOLD_EXECUTOR_0_PORT']}")
    try:
        print(
            f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}"
        )
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}, RANK: {os.environ['RANK']}")
    except:
        pass
    main()
