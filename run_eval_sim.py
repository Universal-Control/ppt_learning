"""Unified simulation evaluation script for PPT Learning.

This script provides both sequential and distributed parallel evaluation modes
for robotic manipulation policies. It supports single and multi-model evaluation
with configurable parallelization strategies.

Key Features:
    - Sequential evaluation for single GPU setups
    - Distributed parallel evaluation for multi-GPU setups
    - Comprehensive logging and result collection
    - Support for subtask-level success rate analysis
    - Flexible model loading and evaluation
"""

import os
import sys
import hydra
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
from torch.utils import data
from torch.multiprocessing import JoinableQueue
import torch.multiprocessing as mp
from omegaconf import DictConfig

from ppt_learning.utils import learning
from ppt_learning.utils.learning import dict_apply
from ppt_learning.paths import *

# Configure environment
sys.path.append(f"{PPT_DIR}/../third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Unified evaluation runner for sequential and parallel modes."""
    
    def __init__(self, cfg: DictConfig, mode: str = "auto"):
        """Initialize the evaluation runner.
        
        Args:
            cfg: Hydra configuration object
            mode: Evaluation mode - "sequential", "parallel", or "auto"
        """
        self.cfg = cfg
        self.mode = self._determine_mode(mode)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup output directory
        self.output_dir = self._setup_output_directory()
        
        # Parse domain configuration
        self.domain_list = [d.strip() for d in cfg.domains.split(",")]
        self.domain = self.domain_list[0] if len(self.domain_list) == 1 else "_".join(self.domain_list)
        
        # Configure model dimensions
        self.action_dim = getattr(cfg, 'action_dim', 7)
        self.state_dim = cfg.state_dim
        
        # Configure point cloud settings
        self.use_pcd = "pointcloud" in cfg.stem.modalities
        if self.use_pcd:
            cfg.rollout_runner.pcdnet_pretrain_domain = cfg.stem.pointcloud.pcd_domain
            
        # Setup head configuration
        if cfg.rollout_runner.get("hist_action_cond", False):
            cfg.head["hist_horizon"] = cfg.dataset.observation_horizon
        cfg.head["output_dim"] = cfg.network["action_dim"] = self.action_dim
        cfg.stem.state["input_dim"] = self.state_dim
        
        logger.info(f"Initialized {self.mode} evaluation runner")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Domain: {self.domain}")
        logger.info(f"Using point clouds: {self.use_pcd}")
    
    def _determine_mode(self, mode: str) -> str:
        """Determine evaluation mode based on configuration and resources.
        
        Args:
            mode: Requested mode ("sequential", "parallel", or "auto")
            
        Returns:
            Determined evaluation mode
        """
        if mode == "auto":
            n_procs = getattr(self.cfg, 'n_procs', 1)
            n_gpus = torch.cuda.device_count()
            
            if n_procs > 1 and n_gpus > 1:
                return "parallel"
            else:
                return "sequential"
        return mode
    
    def _setup_output_directory(self) -> str:
        """Setup and create output directory.
        
        Returns:
            Output directory path
        """
        output_dir_full = self.cfg.output_dir.split("/")
        domain_list = [d.strip() for d in self.cfg.domains.split(",")]
        domain = domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)
        
        output_dir = "/".join(output_dir_full[:-2] + [domain, ""])
        if len(self.cfg.suffix):
            output_dir += f"{self.cfg.suffix}"
        else:
            output_dir += "-".join(output_dir_full[-2:])
        
        is_eval = self.cfg.train.total_epochs == 0
        if is_eval:
            output_dir += "-eval"
            
        self.cfg.output_dir = output_dir
        learning.save_args_hydra(output_dir, self.cfg)
        
        return output_dir
    
    def create_policy(self, device: str = None) -> torch.nn.Module:
        """Create and initialize policy.
        
        Args:
            device: Target device for policy
            
        Returns:
            Initialized policy module
        """
        if device is None:
            device = self.device
            
        policy = hydra.utils.instantiate(
            self.cfg.network, 
            max_timestep=self.cfg.rollout_runner.max_timestep
        )
        
        policy.init_domain_stem(self.domain, self.cfg.stem)
        policy.init_domain_head(self.domain, self.cfg.head)
        policy.finalize_modules()
        policy.to(device)
        
        n_parameters = sum(p.numel() for p in policy.parameters())
        logger.info(f"Policy parameters: {n_parameters / 1.0e6:.2f}M")
        
        return policy
    
    def load_model(self, policy: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Load model weights from checkpoint.
        
        Args:
            policy: Policy module to load weights into
            model_name: Name of model checkpoint file
            
        Returns:
            Policy with loaded weights
            
        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
        """
        model_path = os.path.join(self.cfg.train.pretrained_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pretrained model not found: {model_path}"
            )
        
        state_dict = torch.load(model_path, map_location=self.device)
        policy.load_state_dict(state_dict)
        policy.eval()
        
        logger.info(f"Loaded model: {model_name}")
        return policy
    
    def evaluate_single_model(
        self, 
        policy: torch.nn.Module, 
        runner: Any, 
        model_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single model.
        
        Args:
            policy: Policy to evaluate
            runner: Evaluation runner instance
            model_name: Name of the model being evaluated
            
        Returns:
            Tuple of (success_rate, subtask_success_rates)
        """
        logger.info(f"Starting evaluation of {model_name}")
        start_time = time.time()
        
        try:
            result = runner.run(policy, model_name)
            if len(result) == 3:
                success_rate, _, extra_info = result
                if isinstance(extra_info, tuple) and len(extra_info) == 3:
                    _, subtask_success_nums, episode_num = extra_info
                    subtask_success_rates = {
                        key: float(count) / episode_num 
                        for key, count in subtask_success_nums.items()
                    }
                else:
                    subtask_success_rates = {}
            else:
                success_rate = result
                subtask_success_rates = {}
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            success_rate = 0.0
            subtask_success_rates = {}
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        logger.info(f"Model {model_name} evaluation completed")
        logger.info(f"Success rate: {success_rate:.4f}")
        logger.info(f"Evaluation time: {evaluation_time:.2f}s")
        
        for subtask, rate in subtask_success_rates.items():
            logger.info(f"Subtask {subtask} success rate: {rate:.4f}")
        
        return success_rate, subtask_success_rates
    
    def run_sequential(self) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        """Run sequential evaluation on a single process.
        
        Returns:
            Dictionary mapping model names to their evaluation results
        """
        logger.info("Starting sequential evaluation")
        
        policy = self.create_policy()
        runner = hydra.utils.instantiate(self.cfg.rollout_runner)
        results = {}
        
        model_names = self.cfg.train.model_names
        log_file = os.path.join(
            self.cfg.train.pretrained_dir, 
            f"{self.cfg.eval_log_name}.txt"
        )
        
        logger.info(f"Evaluating {len(model_names)} models")
        logger.info(f"Results will be logged to: {log_file}")
        
        for model_name in model_names:
            try:
                self.load_model(policy, model_name)
                success_rate, subtask_rates = self.evaluate_single_model(
                    policy, runner, model_name
                )
                
                # Store results
                results[model_name] = {
                    "total": success_rate,
                    "subtask_sr": subtask_rates
                }
                
                # Log to file
                with open(log_file, "a") as f:
                    f.write(f"success rate of {model_name} is: {success_rate}\\n")
                    for subtask, rate in subtask_rates.items():
                        f.write(f"Subtask success rate for {subtask} is: {rate}\\n")
                        
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = {"total": 0.0, "subtask_sr": {}}
        
        return results
    
    def run_parallel(self) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        """Run parallel evaluation using multiple processes.
        
        Returns:
            Dictionary mapping model names to their evaluation results
        """
        logger.info("Starting parallel evaluation")
        
        world_size = self.cfg.n_procs
        if world_size > torch.cuda.device_count():
            logger.warning(
                f"Requested {world_size} processes but only {torch.cuda.device_count()} GPUs available"
            )
            world_size = torch.cuda.device_count()
        
        logger.info(f"Using {world_size} parallel processes")
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        # Create shared queue for results
        shared_queue = JoinableQueue()
        
        # Launch parallel processes
        mp.spawn(
            self._eval_worker_process,
            args=(world_size, shared_queue),
            nprocs=world_size,
            join=False,
            daemon=True
        )
        
        # Collect results
        results = {}
        model_names = self.cfg.train.model_names
        log_file = os.path.join(
            self.cfg.train.pretrained_dir,
            f"{self.cfg.eval_log_name}.txt"
        )
        
        logger.info(f"Collecting results from {len(model_names)} models")
        
        try:
            for _ in range(len(model_names)):
                model_name, success_rate, subtask_rates = shared_queue.get()
                
                results[model_name] = {
                    "total": success_rate,
                    "subtask_sr": subtask_rates
                }
                
                # Log results
                with open(log_file, "a") as f:
                    f.write(f"success rate of {model_name} is: {success_rate}\\n")
                    for subtask, rate in subtask_rates.items():
                        f.write(f"Subtask success rate for {subtask} is: {rate}\\n")
                
                shared_queue.task_done()
                logger.info(f"Collected results for {model_name}")
                
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user")
        
        return results
    
    def _eval_worker_process(self, rank: int, world_size: int, queue: JoinableQueue) -> None:
        """Worker process for parallel evaluation.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            queue: Shared queue for result communication
        """
        try:
            # Set environment variables for distributed setup
            os.environ["LOCAL_RANK"] = str(rank)
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            
            device = f"cuda:{rank}"
            logger.info(f"Worker {rank} starting on {device}")
            
            # Create policy and runner for this process
            policy = self.create_policy(device)
            runner = hydra.utils.instantiate(
                self.cfg.rollout_runner, 
                world_size=world_size, 
                rank=rank
            )
            
            # Evaluate assigned models (round-robin assignment)
            model_names = [
                name for i, name in enumerate(self.cfg.train.model_names) 
                if i % world_size == rank
            ]
            
            logger.info(f"Worker {rank} assigned {len(model_names)} models")
            
            for model_name in model_names:
                try:
                    self.load_model(policy, model_name)
                    success_rate, subtask_rates = self.evaluate_single_model(
                        policy, runner, model_name
                    )
                    queue.put((model_name, success_rate, subtask_rates))
                    logger.info(f"Worker {rank} completed {model_name}")
                    
                except Exception as e:
                    logger.error(f"Worker {rank} failed on {model_name}: {e}")
                    queue.put((model_name, 0.0, {}))
            
        except Exception as e:
            logger.error(f"Worker {rank} encountered error: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        """Run evaluation in the configured mode.
        
        Returns:
            Dictionary mapping model names to their evaluation results
        """
        logger.info(f"Starting evaluation in {self.mode} mode")
        
        if self.mode == "sequential":
            results = self.run_sequential()
        elif self.mode == "parallel":
            results = self.run_parallel()
        else:
            raise ValueError(f"Unknown evaluation mode: {self.mode}")
        
        # Save results to JSON
        results_file = os.path.join(
            self.cfg.train.pretrained_dir,
            f"{self.cfg.eval_log_name}.json"
        )
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Log summary
        total_models = len(results)
        avg_success_rate = np.mean([r["total"] for r in results.values()])
        
        logger.info(f"Evaluation Summary:")
        logger.info(f"  Total models: {total_models}")
        logger.info(f"  Average success rate: {avg_success_rate:.4f}")
        
        return results


@hydra.main(
    config_path="configs",
    config_name="config_eval_pcd_unified",
    version_base="1.2",
)
def main(cfg: DictConfig) -> float:
    """Main evaluation entry point.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Average success rate across all models
    """
    # Determine evaluation mode
    eval_mode = getattr(cfg, 'eval_mode', 'auto')
    
    # Create and run evaluator
    evaluator = EvaluationRunner(cfg, mode=eval_mode)
    results = evaluator.run()
    
    # Return average success rate for compatibility
    avg_success_rate = np.mean([r["total"] for r in results.values()])
    return avg_success_rate


if __name__ == "__main__":
    main()