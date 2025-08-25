"""Unified training script for PPT Learning.

This script provides both single-GPU and distributed data parallel (DDP) training
modes for robotic manipulation policies. It intelligently selects the appropriate
training mode based on available resources and configuration.

Key Features:
    - Single-GPU training for development and debugging
    - Multi-GPU distributed training for production
    - Automatic mode selection based on resources
    - Comprehensive logging and monitoring
    - Weights & Biases integration
    - Checkpoint management with automatic cleanup
"""

import os
import sys
import glob
import random
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import logging
from tqdm import trange

import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from ppt_learning.utils import learning, model_utils
from ppt_learning.utils.warmup_lr_wrapper import WarmupLR
from ppt_learning.paths import *
from ppt_learning import train_test

# Configure environment
sys.path.append(f"{PPT_DIR}/../third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Unified trainer for single-GPU and distributed training."""
    
    def __init__(self, cfg: DictConfig, mode: str = "auto"):
        """Initialize the unified trainer.
        
        Args:
            cfg: Hydra configuration object
            mode: Training mode - "single", "distributed", or "auto"
        """
        self.cfg = cfg
        self.mode = self._determine_mode(mode)
        self.is_eval = cfg.train.total_epochs == 0
        
        # Setup configuration
        self._setup_configuration()
        
        logger.info(f"Initialized {self.mode} trainer")
        logger.info(f"Output directory: {self.cfg.output_dir}")
        logger.info(f"Domain: {self.domain}")
        logger.info(f"Evaluation mode: {self.is_eval}")
    
    def _determine_mode(self, mode: str) -> str:
        """Determine training mode based on configuration and resources.
        
        Args:
            mode: Requested mode ("single", "distributed", or "auto")
            
        Returns:
            Determined training mode
        """
        if mode == "auto":
            n_gpus = torch.cuda.device_count()
            use_ddp = getattr(self.cfg, 'use_ddp', False)
            
            if n_gpus > 1 and use_ddp:
                return "distributed"
            else:
                return "single"
        return mode
    
    def _setup_configuration(self) -> None:
        """Setup common configuration for both training modes."""
        # Register custom OmegaConf resolver
        OmegaConf.register_new_resolver("eval", eval)
        
        # Parse domain configuration
        domain_list = [d.strip() for d in self.cfg.domains.split(",")]
        self.domain = self.cfg.get("dataset_path", "debug").split("/")[-1]
        self.domain_list = domain_list
        
        # Setup output directory
        output_dir_full = self.cfg.output_dir.split("/")
        output_dir = "/".join(output_dir_full + [self.domain, ""])
        if len(self.cfg.suffix):
            output_dir += f"{self.cfg.suffix}"
        if self.is_eval:
            output_dir += "-eval"
        self.cfg.output_dir = output_dir
        
        # Configure point cloud settings
        self.use_pcd = "pointcloud" in self.cfg.stem.modalities
        if self.use_pcd:
            self.cfg.dataset.use_pcd = self.use_pcd
            self.cfg.dataset.pcdnet_pretrain_domain = self.cfg.stem.pointcloud.pcd_domain
            self.cfg.rollout_runner.pcdnet_pretrain_domain = self.cfg.stem.pointcloud.pcd_domain
            self.cfg.rollout_runner.pcd_channels = self.cfg.dataset.pcd_channels
        
        # Configure dataset settings
        if self.cfg.dataset.get("hist_action_cond", False):
            self.cfg.head["hist_horizon"] = self.cfg.dataset.observation_horizon
        self.cfg.dataset.horizon = (
            self.cfg.dataset.observation_horizon + self.cfg.dataset.action_horizon - 1
        )
        self.cfg.dataset.domain = self.domain
    
    def remove_old_checkpoints(self, output_dir: str, k: int) -> None:
        """Remove old checkpoints keeping only the last k.
        
        Args:
            output_dir: Directory containing checkpoints
            k: Number of checkpoints to keep
        """
        checkpoint_files = sorted(
            glob.glob(os.path.join(output_dir, "model_*.pth")), 
            key=os.path.getmtime
        )
        
        while len(checkpoint_files) > k:
            oldest_checkpoint = checkpoint_files.pop(0)
            os.remove(oldest_checkpoint)
            logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
    
    def create_dataloader(
        self, 
        dataset: Any, 
        rank: int = 0, 
        world_size: int = 1,
        is_distributed: bool = False,
        **kwargs: Any
    ) -> DataLoader:
        """Create dataloader for training.
        
        Args:
            dataset: Dataset to load
            rank: Process rank for distributed training
            world_size: Total number of processes
            is_distributed: Whether using distributed training
            **kwargs: Additional dataloader arguments
            
        Returns:
            Configured DataLoader
        """
        if is_distributed:
            shuffle = kwargs.pop("shuffle", False)
            sampler = DistributedSampler(
                dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=shuffle, 
                seed=self.cfg.seed + rank
            )
            return DataLoader(dataset, sampler=sampler, **kwargs)
        else:
            return DataLoader(
                dataset, 
                **kwargs, 
                multiprocessing_context="fork"
            )
    
    def setup_dataset(
        self, 
        rank: int = 0, 
        world_size: int = 1
    ) -> Tuple[Any, Any, Any, int, int]:
        """Setup datasets and dataloaders.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            
        Returns:
            Tuple of (train_loader, test_loader, normalizer, action_dim, state_dim)
        """
        if self.is_eval:
            return None, None, None, 7, self.cfg.state_dim
        
        # Configure dataset paths
        if len(self.domain_list) == 1:
            dataset_path = f"{self.cfg.get('dataset_path', '')}/{self.domain_list[0]}.zarr"
        else:
            dataset_path = [
                f"{self.cfg.get('dataset_path', '')}/{domain}.zarr" 
                for domain in self.domain_list
            ]
        self.cfg.dataset.dataset_path = dataset_path
        
        # Create dataset
        dataset_kwargs = dict(self.cfg.dataset)
        if self.mode == "distributed":
            dataset_kwargs['seed'] = self.cfg.seed + rank
        
        dataset = hydra.utils.instantiate(self.cfg.dataset, **dataset_kwargs)
        normalizer = dataset.get_normalizer()
        
        # Create validation dataset
        val_dataset = dataset.get_validation_dataset()
        
        # Get dimensions
        action_dim = dataset.action_dim
        state_dim = dataset.state_dim
        
        # Setup point cloud points
        pcd_num_points = 1024
        if self.use_pcd:
            pcd_num_points = dataset.pcd_num_points
            assert pcd_num_points is not None
        
        # Create dataloaders
        is_distributed = self.mode == "distributed"
        train_loader = self.create_dataloader(
            dataset, 
            rank=rank, 
            world_size=world_size,
            is_distributed=is_distributed,
            **self.cfg.dataloader
        )
        test_loader = self.create_dataloader(
            val_dataset,
            rank=rank,
            world_size=world_size, 
            is_distributed=is_distributed,
            **self.cfg.val_dataloader
        )
        
        if rank == 0:
            logger.info(f"Train size: {len(dataset)}, Test size: {len(val_dataset)}")
        
        return train_loader, test_loader, normalizer, action_dim, state_dim
    
    def create_policy(
        self, 
        normalizer: Optional[Any] = None,
        action_dim: int = 7,
        state_dim: int = 14,
        device: str = "cuda"
    ) -> nn.Module:
        """Create and initialize policy network.
        
        Args:
            normalizer: Data normalizer
            action_dim: Action dimension
            state_dim: State dimension
            device: Device to place model on
            
        Returns:
            Initialized policy module
        """
        # Configure dimensions
        self.cfg.head["output_dim"] = self.cfg.network["action_dim"] = action_dim
        self.cfg.stem.state["input_dim"] = state_dim
        
        # Create policy
        policy = hydra.utils.instantiate(self.cfg.network)
        policy.init_domain_stem(self.domain, self.cfg.stem)
        policy.init_domain_head(self.domain, self.cfg.head, normalizer=normalizer)
        policy.finalize_modules()
        
        # Load pretrained model if specified
        loaded_epoch = self.load_pretrained_model(policy)
        
        # Move to device
        policy.to(device)
        
        # Log model info
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_parameters / 1.0e6:.2f}M")
        
        return policy, loaded_epoch
    
    def load_pretrained_model(self, policy: nn.Module) -> int:
        """Load pretrained model if specified.
        
        Args:
            policy: Policy module to load weights into
            
        Returns:
            Loaded epoch number (-1 if no pretrained model)
        """
        loaded_epoch = -1
        
        if len(self.cfg.train.pretrained_dir) > 0:
            if ".pth" in self.cfg.train.pretrained_dir:
                model_path = self.cfg.train.pretrained_dir
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Pretrained model not found: {model_path}")
                
                policy.load_state_dict(torch.load(model_path))
                loaded_epoch = int(
                    model_path.split("/")[-1].split(".")[0].split("_")[-1]
                )
                logger.info(f"Loaded model from {model_path}, epoch {loaded_epoch}")
            else:
                model_path = os.path.join(self.cfg.train.pretrained_dir, "model.pth")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Pretrained model not found: {model_path}")
                
                policy.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded model from {model_path}")
            
            if self.cfg.train.freeze_trunk:
                policy.freeze_trunk()
                logger.info("Trunk frozen")
        else:
            logger.info("Training from scratch")
        
        return loaded_epoch
    
    def setup_wandb(self, rank: int = 0) -> None:
        """Setup Weights & Biases logging.
        
        Args:
            rank: Process rank (only rank 0 logs to wandb)
        """
        if not self.cfg.debug and rank == 0:
            run = wandb.init(
                project=self.domain,
                name=self.cfg.suffix,
                tags=[self.cfg.wb_tag],
                config=OmegaConf.to_container(self.cfg, resolve=True),
                reinit=False,
                resume="must" if self.mode == "distributed" else "allow",
            )
            logger.info(f"W&B URL: {wandb.run.get_url()}")
    
    def run_single_gpu(self) -> None:
        """Run single-GPU training."""
        logger.info("Starting single-GPU training")
        
        # Setup wandb
        self.setup_wandb()
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set random seeds
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        
        # Setup dataset
        train_loader, test_loader, normalizer, action_dim, state_dim = self.setup_dataset()
        
        # Create policy
        policy, loaded_epoch = self.create_policy(
            normalizer, action_dim, state_dim, device
        )
        
        # Save configuration
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        learning.save_args_hydra(self.cfg.output_dir, self.cfg)
        
        # Setup optimizer and scheduler
        total_steps = self.cfg.train.total_epochs * len(train_loader)
        optimizer = learning.get_optimizer(self.cfg.optimizer, policy)
        scheduler = learning.get_scheduler(
            self.cfg.lr_scheduler, 
            optimizer,
            num_warmup_steps=self.cfg.warmup_lr.step,
            num_training_steps=total_steps
        )
        
        # Training loop
        if not self.is_eval:
            pbar = trange(
                loaded_epoch + 1, 
                loaded_epoch + 1 + self.cfg.train.total_epochs,
                position=0
            )
            
            for epoch in pbar:
                # Train
                train_stats = train_test.train(
                    self.cfg.log_interval,
                    policy,
                    device,
                    train_loader,
                    optimizer,
                    scheduler,
                    epoch,
                    debug=self.cfg.debug,
                    epoch_size=self.cfg.train.epoch_iters
                )
                
                train_steps = (epoch + 1) * len(train_loader)
                
                # Save model
                if epoch % self.cfg.save_interval == 0:
                    policy_path = os.path.join(self.cfg.output_dir, f"model_{epoch}.pth")
                else:
                    policy_path = os.path.join(self.cfg.output_dir, "model.pth")
                
                policy.save(policy_path)
                
                # Clean up old checkpoints
                k = self.cfg.train.get('last_k_checkpoints', 5)
                self.remove_old_checkpoints(self.cfg.output_dir, k)
                
                # Update progress bar
                if "loss" in train_stats:
                    pbar.set_description(
                        f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}."
                    )
                
                # Check if we've reached total iterations
                if train_steps > self.cfg.train.total_iters:
                    break
            
            # Save final model
            policy.save(policy_path)
            pbar.close()
        
        logger.info(f"Results saved to: {self.cfg.output_dir}")
    
    def run_distributed(self) -> None:
        """Run distributed training."""
        logger.info("Starting distributed training")
        
        # Get world size
        world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
        node_rank = int(os.environ.get('RANK', 0))
        
        if world_size < 1:
            raise RuntimeError("No GPUs available for distributed training")
        
        # Spawn distributed processes
        if world_size > 1:
            mp.spawn(
                self._distributed_worker,
                args=(world_size, node_rank),
                nprocs=torch.cuda.device_count(),
                join=True
            )
        else:
            self._distributed_worker(0, world_size, node_rank)
    
    def _distributed_worker(
        self, 
        local_rank: int, 
        world_size: int, 
        node_rank: int = 0
    ) -> None:
        """Worker process for distributed training.
        
        Args:
            local_rank: Local GPU rank
            world_size: Total number of processes
            node_rank: Node rank for multi-node training
        """
        # Compute global rank
        gpus_per_node = torch.cuda.device_count()
        rank = node_rank * gpus_per_node + local_rank
        
        # Initialize process group
        logger.info(f"Process {rank} initialized with world size {world_size}")
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        
        # Setup wandb (only rank 0)
        self.setup_wandb(rank)
        
        # Set random seeds
        seed = self.cfg.seed
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        
        # Setup device
        device = torch.device(f"cuda:{local_rank}")
        
        # Setup dataset
        train_loader, test_loader, normalizer, action_dim, state_dim = self.setup_dataset(
            rank, world_size
        )
        
        # Create policy
        policy, loaded_epoch = self.create_policy(
            normalizer, action_dim, state_dim, device
        )
        
        # Wrap in DDP
        policy = DDP(policy, device_ids=[rank])
        
        # Save configuration (only rank 0)
        if rank == 0:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            learning.save_args_hydra(self.cfg.output_dir, self.cfg)
        
        # Wait for rank 0 to create directory
        dist.barrier()
        
        # Setup optimizer and scheduler
        total_steps = self.cfg.train.total_epochs * len(train_loader)
        optimizer = learning.get_optimizer(self.cfg.optimizer, policy)
        scheduler = learning.get_scheduler(
            self.cfg.lr_scheduler,
            optimizer,
            num_warmup_steps=self.cfg.warmup_lr.step,
            num_training_steps=total_steps
        )
        
        # Training loop
        if not self.is_eval:
            pbar = trange(
                loaded_epoch + 1,
                loaded_epoch + 1 + self.cfg.train.total_epochs,
                position=0,
                disable=(rank != 0)  # Only show progress on rank 0
            )
            
            for epoch in pbar:
                # Set epoch for sampler
                train_loader.sampler.set_epoch(epoch)
                
                # Train
                train_stats = train_test.train(
                    self.cfg.log_interval,
                    policy,
                    device,
                    train_loader,
                    optimizer,
                    scheduler,
                    epoch,
                    rank=rank,
                    world_size=world_size,
                    pcd_npoints=1024,  # TODO: get from dataset
                    in_channels=self.cfg.dataset.get('pcd_channels', 4),
                    debug=self.cfg.debug,
                    epoch_size=self.cfg.train.epoch_iters
                )
                
                train_steps = (epoch + 1) * len(train_loader)
                
                # Save model (only rank 0)
                if rank == 0:
                    if epoch % self.cfg.save_interval == 0:
                        policy_path = os.path.join(self.cfg.output_dir, f"model_{epoch}.pth")
                    else:
                        policy_path = os.path.join(self.cfg.output_dir, "model.pth")
                    
                    policy.module.save(policy_path)
                    
                    # Clean up old checkpoints
                    k = self.cfg.train.get('last_k_checkpoints', 5)
                    self.remove_old_checkpoints(self.cfg.output_dir, k)
                    
                    # Update progress bar
                    if "loss" in train_stats:
                        pbar.set_description(
                            f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}."
                        )
                
                # Check if we've reached total iterations
                if train_steps > self.cfg.train.total_iters:
                    break
            
            # Save final model (only rank 0)
            if rank == 0:
                policy.module.save(policy_path)
            
            pbar.close()
        
        if rank == 0:
            logger.info(f"Results saved to: {self.cfg.output_dir}")
        
        # Clean up
        dist.destroy_process_group()
    
    def run(self) -> None:
        """Run training in the configured mode."""
        if self.mode == "single":
            self.run_single_gpu()
        elif self.mode == "distributed":
            self.run_distributed()
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")


def filter_ddp_args() -> Tuple[int, str, str]:
    """Filter out DDP-specific arguments to avoid Hydra conflicts.
    
    Returns:
        Tuple of (local_rank, model_type, suffix)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model", type=str, default="depth")
    parser.add_argument("--suffix", type=str, default="")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.local_rank, args.model, args.suffix


@hydra.main(
    config_path="configs",
    config_name="config_unified",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """Main training entry point.
    
    Args:
        cfg: Hydra configuration
    """
    # Filter DDP arguments if present
    local_rank, model_type, suffix = filter_ddp_args()
    
    # Update configuration with command-line arguments
    if suffix and not cfg.suffix:
        cfg.suffix = suffix
    cfg.wb_tag = cfg.suffix if len(cfg.suffix) else "default"
    
    # Ensure output_dir has a default value
    if not cfg.get("output_dir"):
        cfg.output_dir = os.path.join(
            "outputs", 
            model_type, 
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    
    # Handle ARNOLD cluster environment variables
    if "ARNOLD_WORKER_0_HOST" in os.environ:
        os.environ["MASTER_IP"] = os.environ["ARNOLD_WORKER_0_HOST"]
        os.environ["MASTER_ADDR"] = os.environ["ARNOLD_WORKER_0_HOST"]
        Port = os.environ["ARNOLD_WORKER_0_PORT"].split(",")
        os.environ["WORLD_SIZE"] = os.environ["NODE_SIZE"] = os.environ["ARNOLD_WORKER_NUM"]
        os.environ["MASTER_PORT"] = Port[0]
        os.environ["RANK"] = os.environ["NODE_RANK"] = os.environ["ARNOLD_ID"]
        logger.info(f"ARNOLD cluster detected: MASTER_ADDR={os.environ['MASTER_ADDR']}, "
                   f"MASTER_PORT={os.environ['MASTER_PORT']}, "
                   f"WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ['RANK']}")
    
    # Determine training mode
    train_mode = getattr(cfg, 'train_mode', 'auto')
    
    # Create and run trainer
    trainer = UnifiedTrainer(cfg, mode=train_mode)
    trainer.run()


if __name__ == "__main__":
    main()