# PPT Learning

PPT stands for Proprioceptive Pointcloud Transformer, derived from [GenSim2](https://gensim2.github.io/), yet it is more than point clouds.

PPT serves as a unified robotic manipulation learning framework, providing comprehensive tools for training and evaluating robotic policies.

## Overview

PPT Learning offers a complete pipeline for robotic manipulation tasks including:

✅ **Unified Training System** - Single script supporting both single-GPU and distributed training  
✅ **Unified Evaluation System** - Flexible evaluation with sequential and parallel modes  
✅ **Multi-Modal Support** - RGB, depth, point cloud, and state modalities  
✅ **Distributed Training** - Multi-GPU and multi-node training with PyTorch DDP  
✅ **Comprehensive Logging** - Structured logging with Weights & Biases integration  
✅ **Type Safety** - Full type hints throughout the codebase  

## Quick Start

### Installation

**Basic installation (no point cloud processing):**
```bash
pip install -r requirements.txt
pip install -e .
```

**Full installation (with point cloud support):**
```bash
pip install -r requirements.txt
pip install -e .

cd ppt_learning/third_party
git clone git@github.com:guochengqian/openpoints.git # For PointNext
cd ..
bash install.sh
```

### Training

```bash
# Single-GPU training
python run_train.py --config-name=config_ddp_depth_ur5_microwave

# Multi-GPU training with torchrun
torchrun --nproc_per_node=4 run_train.py --config-name=config_ddp_depth_ur5_microwave

# Multi-GPU training with 8 GPUs
torchrun --nproc_per_node=8 run_train.py --config-name=config_ddp_depth_ur5_microwave
```

#### Training Modes

##### Single-GPU Mode
- **When**: Single GPU setups, debugging, or when `train_mode=single`
- **Benefits**: Simple, deterministic, easy to debug
- **Usage**: Best for small models or single GPU environments

##### Distributed Mode  
- **When**: Multi-GPU setups or when `train_mode=distributed`
- **Benefits**: Faster training, better resource utilization
- **Usage**: Best for large models with multiple GPUs available

##### Auto Mode (Default)
- **Logic**: 
  - Uses distributed if `world_size > 1` and multiple GPUs available
  - Falls back to single-GPU otherwise
- **Benefits**: Hands-off operation, optimal resource usage

#### Configuration Example
```yaml
# Dataset configuration
dataset_path: /path/to/datasets/ur5_put_bowl_in_microwave
domains: generate_ur5_close_microwave_717.001, generate_ur5_close_microwave_717.002

# Training configuration
train:
  total_epochs: 1200
  total_iters: 1000000000
  epoch_iters: 500
  last_k_checkpoints: 5

# Batch size (scales automatically for multi-GPU)
batch_size: 512
```


#### Configuration Options

##### Core Training Settings
```yaml
train_mode: auto              # Training mode selection
world_size: 4                 # Number of GPUs for distributed training
seed: 42                      # Random seed
debug: false                  # Debug mode
```

##### Training Parameters
```yaml
train:
  total_epochs: 100           # Total training epochs
  total_iters: 100000         # Total training iterations
  epoch_iters: 1000           # Iterations per epoch
  freeze_trunk: false         # Whether to freeze pretrained trunk
  pretrained_dir: ""          # Path to pretrained model
  last_k_checkpoints: 5       # Number of checkpoints to keep
```

##### Dataset Configuration
```yaml
domains: "place_crayon_phase2"    # Training domains (comma-separated)
dataset_path: "data/"             # Path to dataset directory
```

##### Logging and Checkpointing
```yaml
log_interval: 10              # Logging interval (iterations)
save_interval: 10             # Checkpoint saving interval (epochs)
output_dir: "outputs/models"  # Output directory for checkpoints
```

#### Output Structure

##### Checkpoints
- `model.pth` - Latest model checkpoint
- `model_{epoch}.pth` - Epoch-specific checkpoints (saved every `save_interval`)
- Automatic cleanup keeps only the last `k` checkpoints

##### Logs
- `{job_name}.log` - Training logs
- Weights & Biases integration for experiment tracking

### Evaluation

```bash
# Automatic mode selection
python run_eval_sim.py

# Sequential evaluation
python run_eval_sim.py eval_mode=sequential

# Parallel evaluation with 4 processes
python run_eval_sim.py eval_mode=parallel n_procs=4
```
#### Evaluation Modes

##### Sequential Mode
- **When**: Single GPU setups, debugging, or when `eval_mode=sequential`
- **Benefits**: Simple, deterministic, easy to debug
- **Usage**: Best for small model sets or single GPU environments

##### Parallel Mode  
- **When**: Multi-GPU setups or when `eval_mode=parallel`
- **Benefits**: Faster evaluation, better resource utilization
- **Usage**: Best for large model sets with multiple GPUs available

##### Auto Mode (Default)
- **Logic**: 
  - Uses parallel if `n_procs > 1` and multiple GPUs available
  - Falls back to sequential otherwise
- **Benefits**: Hands-off operation, optimal resource usage

#### Configuration Example
```yaml
# Evaluation mode: "sequential", "parallel", or "auto"
eval_mode: auto

# Number of parallel processes
n_procs: 8

# Model configuration
train:
  pretrained_dir: "/path/to/models"
  model_names:
    - "model.pth"
    - "model_1490.pth"

# Rollout configuration
rollout_runner:
  episode_num: 50
  save_video: true
  obs_mode: "depth"
  hist_action_cond: true
```

#### Configuration Options

##### Core Evaluation Settings
```yaml
eval_mode: auto              # Evaluation mode selection
n_procs: 4                   # Number of parallel processes
eval_log_name: "results"     # Log file prefix
```

##### Model Configuration
```yaml
train:
  pretrained_dir: "models/"  # Directory containing model checkpoints
  model_names:               # List of model files to evaluate
    - "model_epoch_100.pth"
    - "model_best.pth"
```

##### Rollout Configuration  
```yaml
rollout_runner:
  episode_num: 50            # Episodes per model evaluation
  max_timestep: 1300         # Maximum timesteps per episode
  save_video: false          # Whether to save evaluation videos
  hist_action_cond: true     # Historical action conditioning
  obs_mode: "depth"          # Observation mode
  warmup_step: 30            # Warmup steps before action
```

#### Output Structure

##### Log Files
- `{eval_log_name}.txt` - Human-readable evaluation logs
- `{eval_log_name}.json` - Machine-readable results

##### Result Format
```json
{
  "model_best.pth": {
    "total": 0.85,
    "subtask_sr": {
      "grasp": 0.90,
      "lift": 0.85,
      "place": 0.80
    }
  }
}
```

## Advanced Usage

### Multi-Node Training
```bash
# Setup these variables manually or use the ones in your cluster
# Node 0
export MASTER_ADDR=node0.example.com
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0
torchrun --nproc_per_node=4 run_train.py

# Node 1
export MASTER_ADDR=node0.example.com
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4
torchrun --nproc_per_node=4 run_train.py
```

### Parameter Overrides
```bash
# Training with custom parameters
python run_train.py \
  train.total_epochs=200 \
  seed=123 \
  suffix=large_scale_experiment

# Evaluation with custom parameters
python run_eval_sim.py \
  eval_mode=parallel \
  n_procs=8 \
  rollout_runner.episode_num=100
```

## Documentation

For detailed usage instructions:
- [Training Documentation](TRAINING_README.md)
- [Evaluation Documentation](EVALUATION_README.md)

## Troubleshooting

### Common Issues

**OpenPoints Version Bug:**
In `openpoints/transforms/point_transformer_gpu.py`:
```python
# Change line 282 from:
if isinstance(self.angle, collections.Iterable):

# To:
if isinstance(self.angle, collections.abc.Iterable):
```

**CUDA Compatibility:**
```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Check GPU availability
python -c "import torch; print(torch.cuda.device_count())"
```

**Configuration Issues:**
- Ensure dataset paths are correct
- Check that model checkpoint paths exist
- Verify YAML syntax in configuration files

## Project Structure

```
ppt_learning/
├── run_train.py              # Unified training script
├── run_eval_sim.py           # Unified evaluation script
├── configs/                  # Configuration files
│   ├── config_unified.yaml   # Unified training config
│   └── config_eval_*.yaml    # Evaluation configs
├── ppt_learning/            # Core package
│   ├── dataset/             # Dataset classes
│   ├── models/              # Model implementations
│   ├── utils/               # Utility functions
│   └── third_party/         # External dependencies
└── *_README.md              # Detailed documentation
```

## Acknowledgments

- [GenSim2](https://github.com/GenSim2/GenSim2)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [OpenPoints](https://github.com/guochengqian/openpoints)
