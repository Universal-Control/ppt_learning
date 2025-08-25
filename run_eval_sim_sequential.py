"""
not parallel yet, just a loop for now
"""
import os, sys
import hydra
import numpy as np
import time
import json
import torch
from torch.utils import data

from ppt_learning.utils import learning
from ppt_learning.utils.learning import dict_apply
from ppt_learning.paths import *

sys.path.append(f"{PPT_DIR}/../third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(
    config_path=f"configs",
    config_name="config_eval_pcd_sequential",
    version_base="1.2",
)
def run(cfg):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """

    is_eval = cfg.train.total_epochs == 0

    device = "cuda"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)

    output_dir_full = cfg.output_dir.split("/")
    output_dir = "/".join(output_dir_full[:-2] + [domain, ""])
    if len(cfg.suffix):
        output_dir += f"{cfg.suffix}"
    else:
        output_dir += "-".join(output_dir_full[-2:])
    if is_eval:
        output_dir += "-eval"
    cfg.output_dir = output_dir
    learning.save_args_hydra(cfg.output_dir, cfg)

    print("cfg: ", cfg)
    print("output dir", cfg.output_dir)

    use_pcd = "pointcloud" in cfg.stem.modalities
    if use_pcd:
        cfg.rollout_runner.pcdnet_pretrain_domain = cfg.stem.pointcloud.pcd_domain

    action_dim = 7 # cfg.action_dim
    state_dim = cfg.state_dim

    # initialize policy
    if cfg.rollout_runner.get("hist_action_cond", False):
        cfg.head["hist_horizon"] = cfg.dataset.observation_horizon
    cfg.head["output_dim"] = cfg.network["action_dim"] = action_dim
    
    policy = hydra.utils.instantiate(cfg.network, max_timestep=cfg.rollout_runner.max_timestep)
    cfg.stem.state["input_dim"] = state_dim
    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, cfg.head)

    # optimizer and scheduler
    policy.finalize_modules()
    print("cfg.train.pretrained_dir:", cfg.train.pretrained_dir)

    policy.to(device)

    print("cfg.train.pretrained_dir:", cfg.train.pretrained_dir)

    model_names = cfg.train.model_names
    model_dir = cfg.train.pretrained_dir
    srs = {}
    runner = hydra.utils.instantiate(cfg.rollout_runner)

    print("============================================")
    print(f'Log will be write to {os.path.join(model_dir, f"{cfg.eval_log_name}.txt")}')
    print("============================================")
    for model_name in model_names:
        assert os.path.exists(
            os.path.join(cfg.train.pretrained_dir, model_name)
        ), f"Pretrained model not found, try to load model from {os.path.join(cfg.train.pretrained_dir, model_name)}"
        
        policy.load_state_dict(
            torch.load(os.path.join(cfg.train.pretrained_dir, model_name))
        )

        n_parameters = sum(p.numel() for p in policy.parameters())
        print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

        policy.eval()

        print("Model initialize successfully")
        start_time = time.time()
        success_rate, _, _ = runner.run(policy, model_name)
        end_time = time.time()
        print(f"Evaluation takes {end_time - start_time} second to finish.")
        print("\n\nThe success rate is {}\n".format(success_rate))
        with open(os.path.join(model_dir, f"{cfg.eval_log_name}.txt"), "at") as t:
            t.write(f"success rate of {model_name} is: {success_rate}\n")
    with open(os.path.join(model_dir, f"{cfg.eval_log_name}.json"), "wt") as j:
        json.dump(srs, j)

    return success_rate

if __name__ == "__main__":
    run()
