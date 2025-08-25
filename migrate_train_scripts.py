#!/usr/bin/env python3
"""Migration helper for training scripts.

This script helps users migrate from the old separate training scripts
(run.py and run_ddp.py) to the new unified training script (run_train.py).
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional


def backup_old_scripts(backup_dir: str = "old_train_scripts") -> bool:
    """Backup old training scripts.
    
    Args:
        backup_dir: Directory to store backups
        
    Returns:
        True if backup was successful
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(exist_ok=True)
    
    old_scripts = [
        "run.py",
        "run_ddp.py",
        "run_train.py"  # In case user wants to backup the new one too
    ]
    
    backed_up = []
    for script in old_scripts:
        if os.path.exists(script):
            if script == "run_train.py":
                # Ask user if they want to backup the new unified script
                response = input(f"Found existing {script}. Backup? (y/N): ").lower().strip()
                if response != 'y':
                    continue
            shutil.copy2(script, backup_path / script)
            backed_up.append(script)
            print(f"✓ Backed up {script} to {backup_path}")
    
    if backed_up:
        print(f"\n📁 Old scripts backed up to: {backup_path.absolute()}")
        return True
    else:
        print("ℹ️  No old training scripts found to backup")
        return False


def create_wrapper_scripts() -> None:
    """Create wrapper scripts for backward compatibility."""
    
    # Single-GPU wrapper (run.py replacement)
    single_gpu_wrapper = '''#!/usr/bin/env python3
"""Backward compatibility wrapper for single-GPU training.

This script is deprecated. Please use run_train.py instead.
"""

import sys
import warnings
from run_train import main
import hydra
from omegaconf import DictConfig

warnings.warn(
    "run.py is deprecated. Please use run_train.py with train_mode=single",
    DeprecationWarning,
    stacklevel=2
)

@hydra.main(
    config_path="configs",
    config_name="config_unified",
    version_base="1.2",
)
def run_single_gpu(cfg: DictConfig) -> None:
    """Run single-GPU training (deprecated wrapper)."""
    cfg.train_mode = "single"
    cfg.world_size = 1
    return main.hydra_main(cfg)

if __name__ == "__main__":
    run_single_gpu()
'''
    
    # Distributed wrapper (run_ddp.py replacement)
    distributed_wrapper = '''#!/usr/bin/env python3
"""Backward compatibility wrapper for distributed training.

This script is deprecated. Please use run_train.py instead.
"""

import sys
import warnings
from run_train import main
import hydra
from omegaconf import DictConfig

warnings.warn(
    "run_ddp.py is deprecated. Please use run_train.py with train_mode=distributed",
    DeprecationWarning,
    stacklevel=2
)

@hydra.main(
    config_path="configs", 
    config_name="config_unified",
    version_base="1.2",
)
def run_distributed(cfg: DictConfig) -> None:
    """Run distributed training (deprecated wrapper)."""
    cfg.train_mode = "distributed"
    return main.hydra_main(cfg)

if __name__ == "__main__":
    run_distributed()
'''
    
    # Write wrapper scripts
    with open("run_deprecated.py", "w") as f:
        f.write(single_gpu_wrapper)
    
    with open("run_ddp_deprecated.py", "w") as f:
        f.write(distributed_wrapper)
    
    # Make them executable
    os.chmod("run_deprecated.py", 0o755)
    os.chmod("run_ddp_deprecated.py", 0o755)
    
    print("✓ Created backward compatibility wrappers")
    print("  - run_deprecated.py")
    print("  - run_ddp_deprecated.py")


def show_usage_guide() -> None:
    """Show usage guide for the new unified script."""
    
    guide = """
🚀 PPT Learning Training Script Migration Complete!

The old separate training scripts have been unified into a single script:
run_train.py

📖 Usage Examples:

1. Automatic mode (chooses single/distributed based on config and GPUs):
   python run_train.py

2. Force single-GPU training:
   python run_train.py train_mode=single

3. Force distributed training with 4 GPUs:
   python run_train.py train_mode=distributed world_size=4

4. Use custom config:
   python run_train.py --config-name=my_train_config

5. Override specific parameters:
   python run_train.py train.total_epochs=200 seed=123 suffix=my_experiment

6. DDP with specific environment variables:
   CUDA_VISIBLE_DEVICES=0,1,2,3 python run_train.py train_mode=distributed world_size=4

⚙️  Configuration:

The unified script uses config_unified.yaml which supports:
- train_mode: "auto", "single", or "distributed"  
- world_size: Number of GPUs for distributed training
- All the same parameters as the old scripts

🔄 Migration Benefits:

✅ Single script handles both training modes
✅ Better device management and GPU detection
✅ Enhanced checkpoint management with automatic cleanup  
✅ Comprehensive type hints and documentation
✅ Improved error handling and logging
✅ Flexible configuration system
✅ Backward compatibility wrappers available

📝 Configuration Migration:

Old configs can be adapted by adding these parameters:
```yaml
train_mode: auto  # or "single"/"distributed" 
world_size: 4     # number of GPUs for distributed training
```

🔧 Environment Variables:

For distributed training, the unified script supports:
- CUDA_VISIBLE_DEVICES: Control which GPUs to use
- MASTER_ADDR: Master node address (for multi-node)
- MASTER_PORT: Master node port (for multi-node)
- WORLD_SIZE: Total number of processes
- RANK: Global rank of current process
- LOCAL_RANK: Local rank within node

📋 Common Migration Patterns:

Old way:
```bash
# Single GPU
python run.py --config-name=config

# Distributed  
python run_ddp.py --model=depth --suffix=experiment1
```

New way:
```bash
# Automatic mode selection
python run_train.py

# Explicit modes
python run_train.py train_mode=single
python run_train.py train_mode=distributed world_size=4 suffix=experiment1
```

For help: python run_train.py --help
"""
    
    print(guide)


def main() -> None:
    """Main migration script."""
    parser = argparse.ArgumentParser(description="Migrate training scripts to unified version")
    parser.add_argument(
        "--backup", 
        action="store_true",
        help="Backup old scripts before migration"
    )
    parser.add_argument(
        "--create-wrappers",
        action="store_true", 
        help="Create backward compatibility wrapper scripts"
    )
    parser.add_argument(
        "--backup-dir",
        default="old_train_scripts",
        help="Directory for backing up old scripts"
    )
    
    args = parser.parse_args()
    
    print("🔄 PPT Learning Training Script Migration")
    print("=" * 50)
    
    if args.backup:
        backup_old_scripts(args.backup_dir)
        print()
    
    if args.create_wrappers:
        create_wrapper_scripts() 
        print()
    
    show_usage_guide()


if __name__ == "__main__":
    main()