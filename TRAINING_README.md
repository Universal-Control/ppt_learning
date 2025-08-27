# PPT Learning Unified Training System

This document describes the unified training system for PPT Learning robotic manipulation policies.

## Overview

The training system has been unified from two separate scripts into a single, flexible training runner that supports both single-GPU and distributed training modes.

### Previous Scripts (Deprecated)
- `run.py` - Single-GPU training
- `run_ddp.py` - Distributed multi-GPU training

### New Unified Script
- `run_train.py` - Unified training supporting both single-GPU and multi-GPU with torchrun

## Features

✅ **Unified Interface**: Single script handles both single-GPU and distributed training  
✅ **Automatic Distributed Detection**: Automatically detects distributed environment variables  
✅ **Comprehensive Logging**: Structured logging with detailed progress information  
✅ **Type Safety**: Full type hints for better IDE support and error detection  
✅ **Flexible Configuration**: Supports all previous configuration options  
✅ **Enhanced Checkpointing**: Automatic checkpoint cleanup with configurable retention  
✅ **Error Handling**: Robust error handling with detailed error messages  
✅ **Backward Compatibility**: Migration tools and compatibility wrappers provided  

## Quick Start

### Basic Usage

```bash
# Single-GPU training
python run_train.py --config-name=config_ddp_depth_ur5_microwave

# Multi-GPU training with torchrun
torchrun --nproc_per_node=4 run_train.py --config-name=config_ddp_depth_ur5_microwave

# Multi-GPU training with 8 GPUs
torchrun --nproc_per_node=8 run_train.py --config-name=config_ddp_depth_ur5_microwave
```

### Configuration

The unified script uses existing config files like `config_ddp_depth_ur5_microwave.yaml`:

```yaml
# Dataset configuration
dataset_path: /mnt/xiaoshen/datasets/ur5_put_bowl_in_microwave_and_close/generate_ur5_close_microwave_717_open_gripper_close_door3_fasterclose_723_2e-4
domains: generate_ur5_close_microwave_717_open_gripper_close_door3_fasterclose.001, generate_ur5_close_microwave_717_open_gripper_close_door3_fasterclose.002

# Dataset setup
dataset:
  _target_: ppt_learning.dataset.multi_sim_traj_dataset.MultiTrajDataset
  action_horizon: 16
  observation_horizon: 1

# Training configuration
train:
  total_epochs: 1200
  total_iters: 1000000000
  epoch_iters: 500
  last_k_checkpoints: 5

# Batch size (scales automatically for multi-GPU)
batch_size: 512
```

## Training Modes

### Single-GPU Mode
- **When**: Single GPU setups, debugging, or when `train_mode=single`
- **Benefits**: Simple, deterministic, easy to debug
- **Usage**: Best for small models or single GPU environments

### Distributed Mode  
- **When**: Multi-GPU setups or when `train_mode=distributed`
- **Benefits**: Faster training, better resource utilization
- **Usage**: Best for large models with multiple GPUs available

### Auto Mode (Default)
- **Logic**: 
  - Uses distributed if `world_size > 1` and multiple GPUs available
  - Falls back to single-GPU otherwise
- **Benefits**: Hands-off operation, optimal resource usage

## Migration Guide

### Step 1: Backup Old Scripts (Optional)
```bash
python migrate_train_scripts.py --backup
```

### Step 2: Update Your Workflow

**Old way:**
```bash
# Single GPU
python run.py --config-name=config

# Distributed  
python run_ddp.py --model=depth --suffix=experiment1
```

**New way:**
```bash
# Automatic mode selection
python run_train.py

# Or specify mode explicitly
python run_train.py train_mode=single
python run_train.py train_mode=distributed world_size=4 suffix=experiment1
```

### Step 3: Update Configuration Files

Add these parameters to your existing configs:

```yaml
train_mode: auto  # "single", "distributed", or "auto"
world_size: 4     # number of GPUs for distributed training
```

### Step 4: Create Compatibility Wrappers (If Needed)
```bash
python migrate_train_scripts.py --create-wrappers
```

## Configuration Options

### Core Training Settings
```yaml
train_mode: auto              # Training mode selection
world_size: 4                 # Number of GPUs for distributed training
seed: 42                      # Random seed
debug: false                  # Debug mode
```

### Training Parameters
```yaml
train:
  total_epochs: 100           # Total training epochs
  total_iters: 100000         # Total training iterations
  epoch_iters: 1000           # Iterations per epoch
  freeze_trunk: false         # Whether to freeze pretrained trunk
  pretrained_dir: ""          # Path to pretrained model
  last_k_checkpoints: 5       # Number of checkpoints to keep
```

### Dataset Configuration
```yaml
domains: "place_crayon_phase2"    # Training domains (comma-separated)
dataset_path: "data/"             # Path to dataset directory
```

### Logging and Checkpointing
```yaml
log_interval: 10              # Logging interval (iterations)
save_interval: 10             # Checkpoint saving interval (epochs)
output_dir: "outputs/models"  # Output directory for checkpoints
```

## Output Structure

### Checkpoints
- `model.pth` - Latest model checkpoint
- `model_{epoch}.pth` - Epoch-specific checkpoints (saved every `save_interval`)
- Automatic cleanup keeps only the last `k` checkpoints

### Logs
- `{job_name}.log` - Training logs
- Weights & Biases integration for experiment tracking

## Advanced Usage

### Custom Configuration
```bash
python run_train.py --config-path=my_configs --config-name=custom_train
```

### Parameter Overrides
```bash
python run_train.py \
  train_mode=distributed \
  world_size=8 \
  train.total_epochs=200 \
  seed=123 \
  suffix=large_scale_experiment
```

### Environment Variables for Distributed Training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500
python run_train.py train_mode=distributed world_size=4
```

### Multi-Node Training
```bash
# Node 0
export MASTER_ADDR=node0.example.com
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0
python run_train.py train_mode=distributed world_size=8

# Node 1
export MASTER_ADDR=node0.example.com
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4
python run_train.py train_mode=distributed world_size=8
```

## Troubleshooting

### Common Issues

**Q: "No CUDA devices available" in distributed mode**
A: Set `train_mode=single` or ensure CUDA is properly installed

**Q: "Model checkpoint not found"**  
A: Check that `train.pretrained_dir` path is correct

**Q: "Training stuck in distributed mode"**
A: Try reducing `world_size` or switching to single-GPU mode

**Q: "Import errors with old scripts"**
A: Use the migration script to create compatibility wrappers

**Q: "Out of memory errors"**
A: Reduce batch size in dataloader configuration or use gradient accumulation

### Performance Tips

1. **Use distributed mode** for large models with multiple GPUs
2. **Adjust world_size** based on available GPUs (typically 1 GPU per process)
3. **Use single mode** for debugging or small model experiments
4. **Set CUDA_VISIBLE_DEVICES** to control GPU usage
5. **Enable mixed precision** training for memory efficiency

## API Reference

### UnifiedTrainer Class

```python
class UnifiedTrainer:
    def __init__(cfg: DictConfig, mode: str = "auto")
    def run() -> None
    def run_single_gpu() -> None
    def run_distributed() -> None
    def _setup_distributed(local_rank: int, world_size: int) -> None
    def _cleanup_distributed() -> None
```

### Main Function

```python
@hydra.main(config_path="configs", config_name="config_unified")
def main(cfg: DictConfig) -> None
```

## Migration Support

For questions or issues during migration:

1. **Check the migration script**: `python migrate_train_scripts.py --help`
2. **Create compatibility wrappers**: For gradual migration
3. **Backup old scripts**: Keep working versions during transition

The unified training system maintains full backward compatibility while providing enhanced functionality, better resource management, and improved maintainability.

## Integration with Evaluation

The unified training script works seamlessly with the unified evaluation system:

```bash
# Train a model
python run_train.py train_mode=distributed world_size=4 suffix=experiment1

# Evaluate the trained model
python run_eval_sim.py \
  eval_mode=parallel \
  n_procs=4 \
  train.pretrained_dir=outputs/models/place_crayon_phase2/experiment1
```

This provides a complete end-to-end workflow for training and evaluating robotic manipulation policies.