# PPT Learning Evaluation System

This document describes the unified evaluation system for PPT Learning robotic manipulation policies.

## Overview

The evaluation system provides a unified script that supports both sequential and parallel evaluation modes for robotic manipulation policies.

### Script
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

The unified script uses `config_eval_depth_unified.yaml`:

```yaml
# Evaluation mode: "sequential", "parallel", or "auto"
eval_mode: auto

# Number of parallel processes (only used in parallel mode)
n_procs: 8

# Model configuration
train:
  pretrained_dir: "/path/to/models"
  model_names:
    - "model.pth"
    - "model_1490.pth"
    - "model_1480.pth"

# Rollout configuration
rollout_runner:
  episode_num: 50
  save_video: true
  obs_mode: "depth"
  hist_action_cond: true
  warmup_step: 30
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

## Usage Guide

### Running Evaluations

**Automatic mode selection:**
```bash
python run_eval_sim.py
```

**Specify mode explicitly:**
```bash
python run_eval_sim.py eval_mode=sequential
python run_eval_sim.py eval_mode=parallel n_procs=8
```

### Configuration

Ensure your config includes these parameters:

```yaml
eval_mode: auto  # "sequential", "parallel", or "auto"
n_procs: 8       # number of parallel processes
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
  episode_num: 50            # Episodes per model evaluation
  max_timestep: 1300         # Maximum timesteps per episode
  save_video: false          # Whether to save evaluation videos
  hist_action_cond: true     # Historical action conditioning
  obs_mode: "depth"          # Observation mode
  warmup_step: 30            # Warmup steps before action
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
  rollout_runner.episode_num=100 \
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

**Q: "OmegaConf eval interpolation error"**
A: Ensure the script includes `OmegaConf.register_new_resolver("eval", eval)`

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
@hydra.main(config_path="configs", config_name="config_eval_depth_unified")
def main(cfg: DictConfig) -> float
```

## Notes

### Important Considerations

1. **OmegaConf Resolver**: The scripts register an `eval` resolver for mathematical expressions in configs
2. **Device Management**: The rollout runner properly handles device assignment in parallel mode
3. **Result Format**: Both sequential and parallel modes output the same result format
4. **Model Loading**: Models are distributed round-robin across processes in parallel mode

### Best Practices

1. **Use unified script** for new projects and evaluations
2. **Test with sequential mode** first when debugging
3. **Monitor GPU memory** when setting n_procs
4. **Check logs** in the pretrained model directory for detailed results