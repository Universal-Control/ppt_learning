"""
Unified evaluation script that combines sequential and parallel evaluation modes.
"""
import os, sys
import hydra
import numpy as np
import time
import json
import torch
from torch.utils import data
from torch.multiprocessing import JoinableQueue
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from ppt_learning.utils import learning
from ppt_learning.utils.learning import dict_apply
from ppt_learning.paths import *

sys.path.append(f"{PPT_DIR}/../third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Register eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval)


def eval_in_one_process(rank, world_size, cfg, domain, queue: JoinableQueue):
    """Worker process for parallel evaluation."""
    try:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        device = f"cuda:{rank}"

        # initialize policy
        if cfg.rollout_runner.get("hist_action_cond", False):
            cfg.head["hist_horizon"] = cfg.dataset.observation_horizon
        policy = hydra.utils.instantiate(cfg.network, max_timestep=cfg.rollout_runner.max_timestep)
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
            result = runner.run(policy, model_name)
            
            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 3:
                success_rate, _, extra_info = result
                if isinstance(extra_info, tuple) and len(extra_info) == 3:
                    _, subtask_success_nums, episode_num = extra_info
                    subtask_success_sr = {}
                    for key in subtask_success_nums:
                        print(f"Subtask success rate for {key} is: {float(subtask_success_nums[key]) / episode_num}")
                        subtask_success_sr[key] = float(subtask_success_nums[key]) / episode_num
                else:
                    subtask_success_sr = {}
            else:
                success_rate = result
                subtask_success_sr = {}
                
            end_time = time.time()
            print(f"Evaluation takes {end_time - start_time} second to finish.")
            print("\n\nThe success rate is {}\n".format(success_rate))
            
            queue.put((model_name, success_rate, subtask_success_sr))

        queue.join()
    except Exception as e:
        print(f"Error in process {rank}: {e}")
        import traceback
        traceback.print_exc()


def run_sequential_eval(cfg, domain):
    """Run sequential evaluation on a single process."""
    device = "cuda"
    
    use_pcd = "pointcloud" in cfg.stem.modalities
    if use_pcd:
        cfg.rollout_runner.pcdnet_pretrain_domain = cfg.stem.pointcloud.pcd_domain

    action_dim = cfg.action_dim
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
        result = runner.run(policy, model_name)
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            success_rate, _, extra_info = result
            if isinstance(extra_info, tuple) and len(extra_info) == 3:
                _, subtask_success_nums, episode_num = extra_info
                subtask_success_sr = {}
                for key in subtask_success_nums:
                    print(f"Subtask success rate for {key} is: {float(subtask_success_nums[key]) / episode_num}")
                    subtask_success_sr[key] = float(subtask_success_nums[key]) / episode_num
            else:
                subtask_success_sr = {}
        else:
            success_rate = result
            subtask_success_sr = {}
            
        end_time = time.time()
        print(f"Evaluation takes {end_time - start_time} second to finish.")
        print("\n\nThe success rate is {}\n".format(success_rate))
        
        with open(os.path.join(model_dir, f"{cfg.eval_log_name}.txt"), "at") as t:
            t.write(f"success rate of {model_name} is: {success_rate}\n")
            for key in subtask_success_sr:
                t.write(f"Subtask success rate for {key} is: {subtask_success_sr[key]}\n")
                
        srs[model_name] = {"total": success_rate, "subtask_sr": subtask_success_sr}
        
    with open(os.path.join(model_dir, f"{cfg.eval_log_name}.json"), "wt") as j:
        json.dump(srs, j)

    return success_rate


def run_parallel_eval(cfg, domain):
    """Run parallel evaluation using multiple processes."""
    mp.set_start_method('spawn')
    
    use_pcd = "pointcloud" in cfg.stem.modalities
    if use_pcd:
        cfg.rollout_runner.pcdnet_pretrain_domain = cfg.stem.pointcloud.pcd_domain
    
    action_dim = cfg.action_dim
    state_dim = cfg.state_dim
    
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
            srs[model_name] = {"total": success_rate}
            shared_queue.task_done()
            idx += 1
            srs[model_name]["subtask_sr"] = subtask_sr
        with open(os.path.join(model_dir, f"{cfg.eval_log_name}.json"), "wt") as j:
            json.dump(srs, j)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    
    # Return average success rate
    if srs:
        return np.mean([s["total"] for s in srs.values()])
    return 0.0


@hydra.main(
    config_path=f"configs",
    config_name="config_eval_depth_unified",
    version_base="1.2",
)
def run(cfg):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """
    is_eval = cfg.train.total_epochs == 0

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
    
    # Determine evaluation mode
    eval_mode = cfg.get('eval_mode', 'auto')
    
    if eval_mode == 'auto':
        # Auto mode: use parallel if n_procs > 1 and multiple GPUs available
        n_procs = cfg.get('n_procs', 1)
        n_gpus = torch.cuda.device_count()
        use_parallel = n_procs > 1 and n_gpus >= n_procs and len(cfg.train.model_names) > 1
    elif eval_mode == 'sequential':
        use_parallel = False
    elif eval_mode == 'parallel':
        use_parallel = True
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")
    
    # Run evaluation
    if use_parallel:
        print(f"Running PARALLEL evaluation with {cfg.n_procs} processes")
        success_rate = run_parallel_eval(cfg, domain)
    else:
        print("Running SEQUENTIAL evaluation")
        success_rate = run_sequential_eval(cfg, domain)

    return success_rate


if __name__ == "__main__":
    run()