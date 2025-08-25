# PPT Learning Evaluation System

This document describes the unified evaluation system for PPT Learning robotic manipulation policies.

## Overview

The evaluation system has been unified from two separate scripts into a single, flexible evaluation runner that supports both sequential and parallel evaluation modes.

### Previous Scripts (Deprecated)
- `run_eval_sim_sequential.py` - Sequential single-GPU evaluation
- `run_eval_sim_batch_ddp.py` - Parallel multi-GPU evaluation

### New Unified Script
- `run_eval_sim.py` - Unified evaluation with automatic mode selection

## Features

✅ **Unified Interface**: Single script handles both sequential and parallel modes
✅ **Automatic Mode Selection**: Intelligently chooses evaluation mode based on resources
✅ **Comprehensive Logging**: Structured logging with detailed progress information
✅ **Type Safety**: Full type hints for better IDE support and error detection
✅ **Flexible Configuration**: Supports all previous configuration options
✅ **Result Collection**: Enhanced result collection with subtask-level analysis
✅ **Error Handling**: Robust error handling with detailed error messages
✅ **Backward Compatibility**: Migration tools and compatibility wrappers provided

## Quick Start

### Basic Usage

```bash
# Automatic mode selection
python run_eval_sim.py

# Force sequential evaluation
python run_eval_sim.py eval_mode=sequential

# Force parallel evaluation with 4 processes  
python run_eval_sim.py eval_mode=parallel n_procs=4
```

### Configuration

The unified script uses `config_eval_pcd_unified.yaml`:

```yaml
# Evaluation mode: "sequential", "parallel", or "auto"
eval_mode: auto

# Number of parallel processes (only used in parallel mode)
n_procs: 4

# Model configuration
train:
  pretrained_dir: "outputs/models"
  model_names:
    - "model_best.pth"
    - "model_latest.pth"

# Rollout configuration
rollout_runner:
  num_episodes: 50
  save_video: false
```

## Evaluation Modes

### Sequential Mode
- **When**: Single GPU setups, debugging, or when `eval_mode=sequential`
- **Benefits**: Simple, deterministic, easy to debug
- **Usage**: Best for small model sets or single GPU environments

### Parallel Mode  
- **When**: Multi-GPU setups or when `eval_mode=parallel`
- **Benefits**: Faster evaluation, better resource utilization
- **Usage**: Best for large model sets with multiple GPUs available

### Auto Mode (Default)
- **Logic**: 
  - Uses parallel if `n_procs > 1` and multiple GPUs available
  - Falls back to sequential otherwise
- **Benefits**: Hands-off operation, optimal resource usage

## Migration Guide

### Step 1: Backup Old Scripts (Optional)
```bash
python migrate_eval_scripts.py --backup
```

### Step 2: Update Your Workflow

**Old way:**
```bash
# Sequential
python run_eval_sim_sequential.py --config-name=config_eval_pcd_sequential

# Parallel  
python run_eval_sim_batch_ddp.py --config-name=config_eval_pcd_ddp
```

**New way:**
```bash
# Automatic mode selection
python run_eval_sim.py

# Or specify mode explicitly
python run_eval_sim.py eval_mode=sequential
python run_eval_sim.py eval_mode=parallel n_procs=4
```

### Step 3: Update Configuration Files

Add these parameters to your existing configs:

```yaml
eval_mode: auto  # "sequential", "parallel", or "auto"
n_procs: 4       # number of parallel processes
```

### Step 4: Create Compatibility Wrappers (If Needed)
```bash
python migrate_eval_scripts.py --create-wrappers
```

## Configuration Options

### Core Evaluation Settings
```yaml
eval_mode: auto              # Evaluation mode selection
n_procs: 4                   # Number of parallel processes
eval_log_name: "results"     # Log file prefix
```

### Model Configuration
```yaml
train:
  pretrained_dir: "models/"  # Directory containing model checkpoints
  model_names:               # List of model files to evaluate
    - "model_epoch_100.pth"
    - "model_best.pth"
```

### Rollout Configuration  
```yaml
rollout_runner:
  num_episodes: 50           # Episodes per model evaluation
  max_timestep: 1300         # Maximum timesteps per episode
  save_video: false          # Whether to save evaluation videos
  hist_action_cond: false    # Historical action conditioning
```

## Output Structure

### Log Files
- `{eval_log_name}.txt` - Human-readable evaluation logs
- `{eval_log_name}.json` - Machine-readable results

### Result Format
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

### Custom Configuration
```bash
python run_eval_sim.py --config-path=my_configs --config-name=custom_eval
```

### Parameter Overrides
```bash
python run_eval_sim.py \
  eval_mode=parallel \
  n_procs=8 \
  rollout_runner.num_episodes=100 \
  train.pretrained_dir=experiments/run_1/models
```

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_eval_sim.py eval_mode=parallel n_procs=4
```

## Troubleshooting

### Common Issues

**Q: "No CUDA devices available" in parallel mode**
A: Set `eval_mode=sequential` or ensure CUDA is properly installed

**Q: "Model checkpoint not found"**  
A: Check that `train.pretrained_dir` and `train.model_names` are correct

**Q: "Evaluation stuck in parallel mode"**
A: Try reducing `n_procs` or switching to sequential mode

**Q: "Import errors with old scripts"**
A: Use the migration script to create compatibility wrappers

### Performance Tips

1. **Use parallel mode** for multiple models with multiple GPUs
2. **Adjust n_procs** based on GPU memory (typically 1 process per GPU)
3. **Use sequential mode** for debugging or single model evaluation
4. **Set CUDA_VISIBLE_DEVICES** to control GPU usage

## API Reference

### EvaluationRunner Class

```python
class EvaluationRunner:
    def __init__(cfg: DictConfig, mode: str = "auto")
    def run() -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]
    def create_policy(device: str = None) -> torch.nn.Module
    def load_model(policy: torch.nn.Module, model_name: str) -> torch.nn.Module
```

### Main Function

```python
@hydra.main(config_path="configs", config_name="config_eval_pcd_unified")
def main(cfg: DictConfig) -> float
```

## Migration Support

For questions or issues during migration:

1. **Check the migration script**: `python migrate_eval_scripts.py --help`
2. **Create compatibility wrappers**: For gradual migration
3. **Backup old scripts**: Keep working versions during transition

The unified evaluation system maintains full backward compatibility while providing enhanced functionality and better maintainability.