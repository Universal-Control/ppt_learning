"""
not parallel yet, just a loop for now
"""
import os, sys
import hydra
import torch
from ppt_learning.utils import utils
from ppt_learning.utils.warmup_lr_wrapper import WarmupLR
from ppt_learning.utils.utils import dict_apply
from ppt_learning.paths import *
import numpy as np
from torch.utils import data
import time
import open3d as o3d
from collections import deque
import json
import argparse
import threading
from torch.multiprocessing import JoinableQueue
import torch.multiprocessing as mp

sys.path.append(f"{PPT_DIR}/third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

hostname = ""  # TODO fill in the hostname
deploy_on_real = True
MAX_EP_STEPS = 500

def eval_in_one_process(rank, world_size, cfg, domain, queue:JoinableQueue):
    try:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        device = f"cuda:{rank}"
        # initialize policy
        if cfg.dataset.get("hist_action_cond", False):
            cfg.head["hist_horizon"] = cfg.dataset.observation_horizon
        policy = hydra.utils.instantiate(cfg.network, max_timestep=cfg.rollout_runner.max_timestep).to(device)
        policy.init_domain_stem(domain, cfg.stem)
        policy.init_domain_head(domain, cfg.head)

        # optimizer and scheduler
        policy.finalize_modules()
        print(f"rank: {rank}, cfg.train.pretrained_dir:", cfg.train.pretrained_dir)
        policy.to(device)

        print(f"Before init at rank {rank}")
        runner = hydra.utils.instantiate(cfg.rollout_runner, world_size=world_size, rank=rank)
        print(f"After init at rank {rank}")

        model_dir = cfg.train.pretrained_dir

        model_names = [name for i, name in enumerate(cfg.train.model_names) if i % world_size == rank]
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
            success_rate, _, (_, subtask_success_nums, episode_num) = runner.run(policy, model_name)
            end_time = time.time()
            print(f"Evaluation takes {end_time - start_time} second to finish.")
            print("\n\nThe success rate is {}\n".format(success_rate))
            subtask_success_sr = {}
            for key in range(subtask_success_nums):
                print(f"Subtask success rate for {key} is: {float(subtask_success_nums) / episode_num}") 
                subtask_success_sr[key] = float(subtask_success_nums) / episode_num
            queue.put((model_name, success_rate, subtask_success_sr))
        
        queue.join()
    except Exception as e:
        print(f"Error in process {rank}: {e}")

# TODO use +prompt "task description" to run specific task
# TODO fill in config_name with config from training
@hydra.main(
    config_path=f"{PPT_DIR}/experiments/configs",
    config_name="config_eval_pcd_ddp",
    version_base="1.2",
)
def run(cfg):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """
    mp.set_start_method('spawn')
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
    utils.save_args_hydra(cfg.output_dir, cfg)

    print("cfg: ", cfg)
    print("output dir", cfg.output_dir)

    action_dim = 7
    state_dim = 21

    cfg.head["output_dim"] = cfg.network["action_dim"] = action_dim
    cfg.stem.state["input_dim"] = state_dim

    model_dir = cfg.train.pretrained_dir
    shared_queue = JoinableQueue()
    print("============================================")
    print(f'Log will be write to {os.path.join(model_dir, f"{cfg.eval_log_name}.txt")}')
    print("============================================")
    
    world_size = cfg.n_procs
    print(f'Parallel on {world_size} gpu(s)')
    mp.spawn(eval_in_one_process, args=(world_size, cfg, domain, shared_queue), nprocs=world_size, join=False, daemon=True)

    idx = 0
    srs = {}
    try:
        while idx < len(cfg.train.model_names):
            model_name, success_rate, subtask_sr = shared_queue.get()
            with open(os.path.join(model_dir, f"{cfg.eval_log_name}.txt"), "at") as t:
                t.write(f"success rate of {model_name} is: {success_rate}\n")
                for key in subtask_sr:
                    t.write(f"Subtask success rate for {key} is: {subtask_sr[key]}\n")
            srs[model_name] = success_rate
            shared_queue.task_done()
            idx += 1
            srs[model_name]["subtask_sr"] = subtask_sr
        with open(os.path.join(model_dir, f"{cfg.eval_log_name}.json"), "wt") as j:
            json.dump(srs, j)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")

if __name__ == "__main__":
    run()
