import os, sys

import hydra
import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler

from ppt_learning.utils.rollout_runner import preprocess_obs, update_pcd_transform
from ppt_learning.utils.robot.real_robot_ur5 import RealRobot
from ppt_learning.utils import utils, model_utils
from ppt_learning.utils.warmup_lr_wrapper import WarmupLR
from ppt_learning.utils.video import videoLogger
from ppt_learning.utils.utils import dict_apply
from ppt_learning.paths import *

import numpy as np
import time
import open3d as o3d
from collections import deque
import argparse
from loguru import logger
import threading

sys.path.append(f"{PPT_DIR}/third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

hostname = ""  # TODO fill in the hostname
deploy_on_real = True
MAX_EP_STEPS = 600
FPS = 15
state_keys = [
                "eef_pos",
                "eef_quat",
                "joint_pos",
                "joint_vel",
            ]

# TODO use +prompt "task description" to run specific task
# TODO fill in config_name with config from training
@hydra.main(
    config_path=f"{PPT_DIR}/experiments/configs",
    config_name="config_eval",
    version_base="1.2",
)
def run(cfg):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """
    robot = RealRobot(fps=FPS)
    assert hasattr(
        cfg, "prompt"
    ), "Prompt not found in config, use +prompt 'task description' to run"
    print("============================================")
    print("Current task: ", cfg.prompt)
    print("============================================")

    device = "cuda"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)

    output_dir_full = cfg.output_dir.split("/")
    output_dir = "/".join(output_dir_full[:-2] + [domain, ""])
    if len(cfg.suffix):
        output_dir += f"{cfg.suffix}"
    else:
        output_dir += "-".join(output_dir_full[-2:])

    output_dir += "-eval"

    cfg.output_dir = output_dir
    utils.save_args_hydra(cfg.output_dir, cfg)

    print("cfg: ", cfg)
    print("output dir", cfg.output_dir)

    use_pcd = "pointcloud" in cfg.stem.modalities
    if use_pcd:
        cfg.dataset.use_pcd = use_pcd
        cfg.dataset.pcdnet_pretrain_domain = (
            cfg.rollout_runner.pcdnet_pretrain_domain
        ) = cfg.stem.pointcloud.pcd_domain
        cfg.rollout_runner.pcd_channels = cfg.dataset.pcd_channels
    cfg.dataset.horizon = (
        cfg.dataset.observation_horizon + cfg.dataset.action_horizon - 1
    )

    # initialize policy
    cfg.stem.pointcloud.pretrained_path = None
    cfg.stem.pointcloud.finetune = False
    cfg.head["output_dim"] = cfg.network["action_dim"] = 8
    cfg.stem.state["input_dim"] = 21
    policy = hydra.utils.instantiate(cfg.network)
    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, cfg.head, normalizer=None)  # no normalizer

    # optimizer and scheduler
    policy.finalize_modules()
    print("cfg.train.pretrained_dir:", cfg.train.pretrained_dir)

    model_name = cfg.train.model_name  # TODO fill in the model name
    assert os.path.exists(
        os.path.join(cfg.train.pretrained_dir, model_name)
    ), "Pretrained model not found"
    policy.load_state_dict(
        torch.load(os.path.join(cfg.train.pretrained_dir, model_name))
    )

    policy.to(device)

    n_parameters = sum(p.numel() for p in policy.parameters())
    print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    policy.eval()

    # t1 = threading.Thread(target=robot.log_pose)

    # t1.start()
    run_in_real(policy, cfg, robot)

def get_state(sample):
    res = []
    for key in state_keys:
        res.append(sample[key])
    res = np.concatenate(res, axis=-1)

    return res

def run_in_real(policy, cfg, robot=None):
    print("Running in real")
    pcdnet_pretrain_domain = cfg.dataset.pcdnet_pretrain_domain
    pcd_channels = cfg.dataset.pcd_channels
    pcd_transform, pcd_num_points = update_pcd_transform(pcdnet_pretrain_domain)

    if robot is None:
        robot = RealRobot()
    save_video = cfg.get("save_video", False)
    if save_video:
        video_logger = videoLogger()

    traj_length = 0
    done = False
    policy.reset()
    obs = robot.get_obs(visualize=False)
    openloop_actions = deque()
    init_time = time.time()
    last_time = init_time
    for t in range(MAX_EP_STEPS):
        traj_length += 1
        if done:
            break
        with torch.no_grad():
            if len(openloop_actions) > 0:
                action = openloop_actions.popleft()
            else:
                obs["state"] = get_state(obs["state"])
                obs = preprocess_obs(
                    obs, pcd_transform=pcd_transform, pcd_channels=pcd_channels
                )

                action = policy.get_action(
                    obs,
                    pcd_npoints=pcd_num_points,
                    in_channels=pcd_channels,
                    task_description=cfg.prompt,
                    t=t,
                )

                if len(action.shape) > 1:
                    for a in action[1:]:
                        openloop_actions.append(a)
                    action = action[0]
        action[-1] = 0.0 if action[-1] < 0.5 else 1.0

        # If want to visualize pcd for debugging turn to True
        next_obs = robot.step(action, visualize=False)

        if save_video:
            for key in obs["images"]:
                self.video_logger.extend(key, obs["images"][key].cpu().numpy(), category="color")
            for key in obs["depths"]:
                self.video_logger.extend(key, obs["depths"][key].cpu().numpy(), category="depth")
            self.video_logger.extend(key, obs["pcds"][key].cpu().numpy(), category="pointcloud")

        obs = next_obs
        this_time = time.time()
        time.sleep(max(1 / FPS - (this_time - last_time), 0))
        logger.info(f"At step {t}: FPS this step: {1 / (this_time - last_time)}, FPS average step: {t / (this_time - init_time)}")
        last_time = this_time


if __name__ == "__main__":
    run()
