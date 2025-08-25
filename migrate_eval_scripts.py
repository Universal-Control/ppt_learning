#!/usr/bin/env python3
"""Migration helper for evaluation scripts.

This script helps users migrate from the old separate evaluation scripts
(run_eval_sim_sequential.py and run_eval_sim_batch_ddp.py) to the new
unified evaluation script (run_eval_sim.py).
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional


def backup_old_scripts(backup_dir: str = "old_eval_scripts") -> bool:
    """Backup old evaluation scripts.
    
    Args:
        backup_dir: Directory to store backups
        
    Returns:
        True if backup was successful
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(exist_ok=True)
    
    old_scripts = [
        "run_eval_sim_sequential.py",
        "run_eval_sim_batch_ddp.py"
    ]
    
    backed_up = []
    for script in old_scripts:
        if os.path.exists(script):
            shutil.copy2(script, backup_path / script)
            backed_up.append(script)
            print(f"✓ Backed up {script} to {backup_path}")
    
    if backed_up:
        print(f"\n📁 Old scripts backed up to: {backup_path.absolute()}")
        return True
    else:
        print("ℹ️  No old evaluation scripts found to backup")
        return False


def create_wrapper_scripts() -> None:
    """Create wrapper scripts for backward compatibility."""
    
    # Sequential wrapper
    sequential_wrapper = '''#!/usr/bin/env python3
"""Backward compatibility wrapper for sequential evaluation.

This script is deprecated. Please use run_eval_sim.py instead.
"""

import sys
import warnings
from run_eval_sim import main
import hydra
from omegaconf import DictConfig

warnings.warn(
    "run_eval_sim_sequential.py is deprecated. Please use run_eval_sim.py with eval_mode=sequential",
    DeprecationWarning,
    stacklevel=2
)

@hydra.main(
    config_path="configs",
    config_name="config_eval_pcd_unified",
    version_base="1.2",
)
def run_sequential(cfg: DictConfig) -> float:
    """Run sequential evaluation (deprecated wrapper)."""
    cfg.eval_mode = "sequential"
    cfg.n_procs = 1
    return main.hydra_main(cfg)

if __name__ == "__main__":
    run_sequential()
'''
    
    # Parallel wrapper
    parallel_wrapper = '''#!/usr/bin/env python3
"""Backward compatibility wrapper for parallel evaluation.

This script is deprecated. Please use run_eval_sim.py instead.
"""

import sys
import warnings
from run_eval_sim import main
import hydra
from omegaconf import DictConfig

warnings.warn(
    "run_eval_sim_batch_ddp.py is deprecated. Please use run_eval_sim.py with eval_mode=parallel",
    DeprecationWarning,
    stacklevel=2
)

@hydra.main(
    config_path="configs", 
    config_name="config_eval_pcd_unified",
    version_base="1.2",
)
def run_parallel(cfg: DictConfig) -> float:
    """Run parallel evaluation (deprecated wrapper)."""
    cfg.eval_mode = "parallel" 
    return main.hydra_main(cfg)

if __name__ == "__main__":
    run_parallel()
'''
    
    # Write wrapper scripts
    with open("run_eval_sim_sequential_deprecated.py", "w") as f:
        f.write(sequential_wrapper)
    
    with open("run_eval_sim_batch_ddp_deprecated.py", "w") as f:
        f.write(parallel_wrapper)
    
    # Make them executable
    os.chmod("run_eval_sim_sequential_deprecated.py", 0o755)
    os.chmod("run_eval_sim_batch_ddp_deprecated.py", 0o755)
    
    print("✓ Created backward compatibility wrappers")
    print("  - run_eval_sim_sequential_deprecated.py")
    print("  - run_eval_sim_batch_ddp_deprecated.py")


def show_usage_guide() -> None:
    """Show usage guide for the new unified script."""
    
    guide = """
🚀 PPT Learning Evaluation Script Migration Complete!

The old separate evaluation scripts have been unified into a single script:
run_eval_sim.py

📖 Usage Examples:

1. Automatic mode (chooses sequential/parallel based on config):
   python run_eval_sim.py

2. Force sequential evaluation:
   python run_eval_sim.py eval_mode=sequential

3. Force parallel evaluation with 4 processes:
   python run_eval_sim.py eval_mode=parallel n_procs=4

4. Use custom config:
   python run_eval_sim.py --config-name=my_eval_config

5. Override specific parameters:
   python run_eval_sim.py train.model_names=["model1.pth","model2.pth"] n_procs=2

⚙️  Configuration:

The unified script uses config_eval_pcd_unified.yaml which supports:
- eval_mode: "auto", "sequential", or "parallel"  
- n_procs: Number of parallel processes
- All the same parameters as before

🔄 Migration Benefits:

✅ Single script handles both evaluation modes
✅ Better error handling and logging
✅ Comprehensive type hints and documentation  
✅ Flexible configuration system
✅ Improved result collection and reporting
✅ Backward compatibility wrappers available

📝 Configuration Migration:

Old configs can be adapted by adding these parameters:
```yaml
eval_mode: auto  # or "sequential"/"parallel" 
n_procs: 4       # number of parallel processes
```

For help: python run_eval_sim.py --help
"""
    
    print(guide)


def main() -> None:
    """Main migration script."""
    parser = argparse.ArgumentParser(description="Migrate evaluation scripts to unified version")
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
        default="old_eval_scripts",
        help="Directory for backing up old scripts"
    )
    
    args = parser.parse_args()
    
    print("🔄 PPT Learning Evaluation Script Migration")
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