# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import List, Optional
import numpy as np
import inspect

from hydra.utils import instantiate
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import transforms3d
import copy
import cv2

from multiprocessing import Process, Queue
import multiprocessing as mp

try:
    mp.set_start_method("forkserver", force=True)
except RuntimeError:
    pass
from multiprocessing import Process, Queue

from ppt_learning.utils.utils import dict_apply
from ppt_learning.utils.video import videoLogger
from ppt_learning.utils.pcd_utils import (
    visualize_point_cloud,
    randomly_drop_point,
    add_gaussian_noise,
)
from ppt_learning.paths import *
from pathlib import Path

import collections
try:
    collections.Iterable=collections.abc.Iterable
except:
    pass
from collections import OrderedDict

try:
    from gensim2.env.task.origin_tasks import *
    from gensim2.env.task.primitive_tasks import *
    from gensim2.env.create_task import create_gensim
except Exception as e:
    print("Gensim2 not installed or error due to", e)

try:
    from gensim2.env.utils.rlbench import *
    from rlbench.backend.exceptions import InvalidActionError
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.vision_sensor import VisionSensor
    from pyrep.const import RenderMode
    from pyrep.errors import IKError, ConfigurationPathError
    from pyrep.objects import VisionSensor, Dummy
except Exception as e:
    print("RLBench not installed or error due to", e)

MAX_EP_STEPS = 100
MAX_RLBENCH_EP_STEPS = 100

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm == 0:
        return q  # Handle the zero division case if needed
    return q / norm


def flat_pcd_sample(sample, pcd_channels=4):
    if "obs" in sample.keys():
        for key, val in sample.items():
            sample[key] = val
        del sample
    if "pointcloud" in sample.keys():
        if pcd_channels == 3:
            pass
            # sample['pointcloud']['pos'] = sample['pointcloud']['pos']
        elif pcd_channels == 5:
            sample["pointcloud"]["x"] = np.concatenate(
                [
                    sample["pointcloud"]["pos"],
                    sample["pointcloud"]["heights"],
                    sample["pointcloud"]["seg"][..., 1].unsqueeze(-1),
                ],
                axis=-1,
            )
        elif pcd_channels == 6:
            sample["pointcloud"]["x"] = np.concatenate(
                [sample["pointcloud"]["pos"], sample["pointcloud"]["colors"]], axis=-1
            )
        elif pcd_channels == 4:
            sample["pointcloud"]["x"] = np.concatenate(
                [sample["pointcloud"]["pos"], sample["pointcloud"]["heights"]], axis=-1
            )
        elif pcd_channels == 7:
            sample["pointcloud"]["x"] = np.concatenate(
                [
                    sample["pointcloud"]["pos"],
                    sample["pointcloud"]["colors"],
                    sample["pointcloud"]["heights"],
                ],
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid pcd_channels: {pcd_channels}")
    return sample


def update_pcd_transform(pcdnet_pretrain_domain, pcd_setup_cfg=None):
    from openpoints.transforms import build_transforms_from_cfg

    if pcd_setup_cfg is None:
        from openpoints.utils import EasyConfig

        pcd_setup_cfg = EasyConfig()
        pcd_setup_cfg.load(
            f"{PPT_DIR}/models/pointnet_cfg/{pcdnet_pretrain_domain}/pcd_setup.yaml",
            recursive=True,
        )

    pcd_transform = build_transforms_from_cfg("train", pcd_setup_cfg.datatransforms)
    # pcd_transform = build_transforms_from_cfg("val", pcd_setup_cfg.datatransforms)
    pcd_num_points = pcd_setup_cfg.num_points

    return pcd_transform, pcd_num_points


def preprocess_obs(sample, pcd_aug=None, pcd_transform=None, pcd_channels=4):
    if "pointcloud" in sample.keys():
        sample["pointcloud"]["x"] = sample["pointcloud"]["color"]

        if pcd_aug is not None:
            pcd_aug = lambda x: randomly_drop_point(add_gaussian_noise(x))
            sample["pointcloud"]["pos"] = pcd_aug(sample["pointcloud"]["pos"])
        if pcd_transform is not None and "pointcloud" in sample.keys():
            sample["pointcloud"] = pcd_transform(sample["pointcloud"])

    sample = dict_apply(
        sample, lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    )

    def unsqueeze(x, dim=0):
        if isinstance(x, np.ndarray):
            return np.expand_dims(x, dim)
        elif isinstance(x, torch.Tensor):
            return x.unsqueeze(dim)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

    sample = dict_apply(sample, unsqueeze)
    return flat_pcd_sample(sample, pcd_channels=pcd_channels)


def env_sample(env_names, ret_queue, **kwargs):
    for env_name in env_names:
        env = create_gensim(
            task_name=env_name,
            sim_type=kwargs["sim_type"],
            use_gui=kwargs["render"],
            eval=True,
            obs_mode=kwargs["obs_mode"],
            headless=False,
            asset_id="random",
        )


class RolloutRunner:
    """evaluate policy rollouts"""

    def __init__(
        self,
        env_names,
        episode_num=100,
        save_video=False,
        sim_type="sapien",
        render=False,
        obs_mode="pointcloud",
        pcd_channels=4,
        pcdnet_pretrain_domain="",
        random_reset=True,
        collision_pred=False,
    ):
        self.env_names = env_names
        self.save_video = save_video
        self.pcd_transform, self.pcd_num_points = update_pcd_transform(
            pcdnet_pretrain_domain
        )
        self.pcd_channels = pcd_channels
        self.pcd_aug = lambda x: randomly_drop_point(add_gaussian_noise(x))

        self.episode_num = episode_num
        self.envs = []
        self.env_name_dict = {}

        self.render = render
        self.random_reset = random_reset

        self.collision_pred = collision_pred

        for env_name in env_names:
            env = create_gensim(
                task_name=env_name,
                sim_type=sim_type,
                use_gui=self.render,
                eval=False,
                obs_mode=obs_mode,
                headless=False,
                asset_id="random",
            )
            self.envs.append(env)
            self.env_name_dict[env_name] = env

    @torch.no_grad()
    def run(self, policy, env_name, seed=233):
        episode_num = self.episode_num  # upper bound for number of trajectories
        imgs = OrderedDict()

        total_success = 0
        total_reward = 0
        env = self.env_name_dict[env_name]
        env.seed(seed)

        print(f"selected env name: {env_name}")
        pbar = tqdm(range(episode_num), position=1, leave=True)

        for i in pbar:
            eps_reward = 0
            traj_length = 0
            done = False
            policy.reset()
            obs = env.reset(random=self.random_reset)
            task_description = env.task.sub_task_descriptions[0]

            for task in env.sub_tasks:
                for t in range(MAX_EP_STEPS):
                    traj_length += 1

                    with torch.no_grad():
                        if "pointcloud" in obs.keys():
                            action = policy.get_action(
                                preprocess_obs(
                                    obs,
                                    self.pcd_aug,
                                    self.pcd_transform,
                                    self.pcd_channels,
                                ),
                                pcd_npoints=self.pcd_num_points,
                                in_channels=self.pcd_channels,
                                task_description=task_description,
                                t=t,
                            )
                        else:
                            action = policy.get_action(
                                preprocess_obs(obs, None, None, 3),
                                pcd_npoints=self.pcd_num_points,
                                in_channels=3,
                                task_description=task_description,
                                t=t,
                            )
                        
                    action[-1] = 0.0 if action[-1] < 0.5 else 1.0
                    if self.collision_pred:
                        action[-2] = 0.0 if action[-2] < 0.5 else 1.0
                        ignore_collisions = bool(action[-1])
                        action = action[:-1]
                        next_obs, reward, done, info = env.step(
                            action, ignore_collisions=ignore_collisions
                        )
                    else:
                        next_obs, reward, done, info = env.step(action)
                    if self.render:
                        env.render()
                    if self.save_video:
                        for key, val in env.get_images().items():
                            if key not in imgs:
                                imgs[key] = []
                            imgs[key].append(val)

                    eps_reward += reward
                    obs = next_obs
                    task_description = info["next_task_description"]

                    if done:
                        break

                pbar.set_description(
                    f"{task} success: {info['sub_task_success']}, progress: {info['task_progress']}"
                )

                if not info["sub_task_success"]:
                    break

            total_reward += eps_reward
            total_success += info["success"]

        return total_success / episode_num, total_reward / episode_num, imgs


class RLBenchRolloutRunner:
    """evaluate policy rollouts"""

    def __init__(
        self,
        env_names,
        episode_num=100,
        save_video=False,
        render=False,
        obs_mode="pointcloud",
        pcd_channels=4,
        pcdnet_pretrain_domain="",
        random_reset=True,
        action_mode="joint_positions",
        collision_pred=False,
    ):
        self.env_names = env_names
        self.save_video = save_video

        self.pcd_transform, self.pcd_num_points = update_pcd_transform(
            pcdnet_pretrain_domain
        )
        self.pcd_channels = pcd_channels
        self.pcd_aug = lambda x: randomly_drop_point(add_gaussian_noise(x))

        self.episode_num = episode_num
        self.envs = []
        self.env_name_dict = {}

        self.render = render
        self.random_reset = random_reset
        self.action_mode = action_mode
        self.collision_pred = collision_pred

        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self.env, self.tr = create_rlbench_env(action_mode=action_mode)

    @torch.no_grad()
    def run(self, policy, env_name, seed=233):
        env = self.env
        if self.save_video:
            self.tr._current_snaps = []
            self.tr._cam_motion.save_pose()

        episode_num = self.episode_num  # upper bound for number of trajectories
        imgs = OrderedDict()

        total_success = 0
        total_reward = 0
        np.random.seed(seed)

        print(f"selected env name: {env_name}")
        pbar = tqdm(range(episode_num), position=1, leave=True)

        for i in pbar:
            eps_reward = 0
            traj_length = 0
            done = False
            policy.reset()
            descriptions, obs = env.reset()
            task_description = np.random.choice(descriptions)

            if self.save_video:
                self.tr._cam_motion.restore_pose()

            for t in range(MAX_RLBENCH_EP_STEPS):
                if self.save_video:
                    self.tr.take_snap(obs)
                obs_data = OrderedDict()
                obs_data["state"] = obs.get_low_dim_data()
                obs_data["pointcloud"] = get_pcds(env, obs)

                traj_length += 1

                if done:
                    break
                with torch.no_grad():
                    action = policy.get_action(
                        preprocess_obs(
                            copy.deepcopy(obs_data),
                            None,
                            self.pcd_transform,
                            self.pcd_channels,
                        ),
                        # preprocess_obs(obs_data, self.pcd_aug, self.pcd_transform, self.pcd_channels),
                        pcd_npoints=self.pcd_num_points,
                        in_channels=self.pcd_channels,
                        task_description=task_description,
                        t=t,
                    )
                    # plot_pred(action, obs_data["pointcloud"]["pos"], obs_data["pointcloud"]["colors"], ".")
                    
                action[-1] = 0.0 if action[-1] < 0.5 else 1.0
                if self.collision_pred:
                    action[-2] = 0.0 if action[-2] < 0.5 else 1.0  # -2 is gripper open
                    ignore_collisions = bool(action[-1])
                    action = action[:-1]  # remove ignore_collisions

                if self.action_mode == "gripper_pose" or self.action_mode == "key_pose":
                    # action[3:-1] = normalize_quaternion(action[3:-1])
                    rotation = transforms3d.euler.euler2quat(*action[3:-1])
                    action = np.concatenate(
                        [action[:3], rotation, np.array([action[-1]])]
                    )

                if self.collision_pred:
                    action = np.concatenate([action, np.array([ignore_collisions])])

                try:
                    next_obs, reward, done = env.step(action)
                    success = reward > 0  # done
                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    done = True
                    reward = 0.0
                    success = False

                    if isinstance(e, IKError):
                        self._error_type_counts["IKError"] += 1
                    elif isinstance(e, ConfigurationPathError):
                        self._error_type_counts["ConfigurationPathError"] += 1
                    elif isinstance(e, InvalidActionError):
                        self._error_type_counts["InvalidActionError"] += 1

                    print(e)
                eps_reward += reward
                if done:
                    self.tr.take_snap(obs)
                    break
                if self.render:
                    pass  # TODO
                    # env.render()
                obs = next_obs

            pbar.set_description(f"success: {eps_reward}, {done}, {success}")
            total_reward += eps_reward
            total_success += int(success)  # eps_reward

        if self.save_video:
            self.tr._snaps.extend(self.tr._current_snaps)

        return total_success / episode_num, total_reward / episode_num, self.tr


def _recursive_flat_env_dim(obs: dict):
    flatted_dict = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            flatted_dict[k] = _recursive_flat_env_dim(v)
        else:
            flatted_dict[k] = v[0]
    return flatted_dict

def clip_depth(depth):
    valid_mask = np.logical_and(depth > 0.01, ~np.isnan(depth)) & (~np.isinf(depth))
    if valid_mask.sum() == 0:
        Log.warn(
            "No valid mask in the depth map of {}".format(self.depth_files[index])
        )
    if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0:
        depth[np.isnan(depth)] = depth[valid_mask].max()
    if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0:
        depth[np.isinf(depth)] = depth[valid_mask].max()

    return depth


class IsaacEnvRolloutRunner:
    def __init__(
        self,
        task_name,
        episode_num=100,
        save_video=False,
        headless=True,
        obs_mode="pointcloud",
        pcd_channels=4,
        pcdnet_pretrain_domain="",
        random_reset=True,
        collision_pred=False,
        pose_transform=None,
        num_envs=1,
        device="cuda:0",
        seed=0,
        state_keys=None,
        video_save_dir=None,
        world_size=1,
        rank=0,
        max_timestep=1200,
        warmup_step=10,
        hist_action_cond=False,
        tar_size=(224,224),
        **kwargs,
    ):
        try:
            assert obs_mode == "pointcloud"
            import os
            self.task_name = task_name
            self.save_video = save_video
            self.max_timestep = max_timestep
            self.headless = headless
            self.episode_num = episode_num
            self.pcd_channels = pcd_channels
            self.warmup_step = warmup_step
            self.pcd_transform, self.pcd_num_points = update_pcd_transform(
                pcdnet_pretrain_domain
            )
            self.pose_transform = None
            if pose_transform is not None:
                if self.pose_transform == "pose_to_quat":
                    from ppt_learning.utils.pose_utils import pose_to_quat
                    self.pose_transform = pose_to_quat
            self.hist_action_cond = hist_action_cond
                                        
            self.random_reset = random_reset
            self.collision_pred = collision_pred
            self.device = device
            self.num_envs = num_envs
            self.state_keys = state_keys
            if state_keys is None:
                self.state_keys = [
                    "eef_pos",
                    "eef_quat",
                    "joint_pos",
                    "joint_vel",
                ]
            self.pcd_aug = lambda x: randomly_drop_point(add_gaussian_noise(x))

            from isaaclab.app import AppLauncher

            distributed = (world_size != 1)

            app_launcher_kwargs = {
                "headless": self.headless, "enable_cameras": True,
                "distributed": distributed, "n_procs": world_size, "livestream": -1,
                "xr": False, device:'cuda:0', "cpu": False,
                "verbose": False, "info": False, "experience": '', "rendering_mode": None,
                "kit_args":''
            }
            self.app_launcher = AppLauncher(None, **app_launcher_kwargs)
            self.app = self.app_launcher.app

            import isaaclab_mimic.envs
            import bytemini_sim.tasks
            from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
            from isaaclab.envs import ManagerBasedRLEnv
            import gymnasium as gym

            if world_size > 1:
                self.device = f"cuda:{rank}"

            env_cfg = parse_env_cfg(
                self.task_name, device=self.device, num_envs=self.num_envs
            )
            
            self.success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
            
            env_cfg.seed = seed

            if world_size > 1:
                env_cfg.sim.device = f'cuda:{rank}'
                env_cfg.seed += self.app_launcher.global_rank
                print("self.app_launcher.global_rank:", self.app_launcher.global_rank)

            self.gym_env = gym.make(self.task_name, cfg=env_cfg)
            self.env = self.gym_env.unwrapped
            print("Env created")

            if inspect.isclass(self.success_term.func):
                self.success_term.func = self.success_term.func(cfg=self.success_term, env=self.env)
                self._success_term_class = True
            else:
                self._success_term_class = False

            self.video_save_dir = Path(video_save_dir)
            print(f"Video will be saved to {self.video_save_dir}")
            if self.save_video:
                self.video_logger = videoLogger(self.video_save_dir)
            self.rank = rank
        except Exception as e:
            print(f"Error initializing IsaacEnvRolloutRunner: {e}")
            import traceback
            traceback.print_exc()

    def get_state(self, sample):
        res = []
        for key in self.state_keys:
            if key in sample.keys():
                res.append(sample[key])
        res = torch.cat(res, dim=-1)

        return res

    def _isaac_obs_wrapper(self, obs):
        ppt_obs = {}
        ppt_obs["depth"] = {}
        for key in obs["policy_infer"]["depths"]:
            ppt_obs["depth"][key] = cv2.resize(clip_depth(obs["policy_infer"]["depths"]), self.tar_size, interpolation=cv2.INTER_NEAREST)
        ppt_obs["pointcloud"] = {
            "color": obs["policy_infer"]["pointcloud"][..., 1],
            "pos": obs["policy_infer"]["pointcloud"][..., 0],
        }
        
        ppt_obs = _recursive_flat_env_dim(ppt_obs)
        ppt_obs["state"] = self.get_state(obs["policy"])

        return ppt_obs

    @torch.inference_mode()
    def run(self, policy, policy_name="model"):
        policy.to(self.device)
        try:
            print(f"Begin eval model:{policy_name}")
            episode_num = self.episode_num  # upper bound for number of trajectories
            imgs = OrderedDict()

            total_success = 0
            total_reward = 0
            env = self.env
            subtask_success_nums = {}

            pbar = tqdm(range(episode_num), position=1, leave=True)

            for i in pbar:
                eps_reward = 0
                traj_length = 0
                done = False
                
                if not self._success_term_class:
                    self.success_term.func(env, **self.success_term.params)
                policy.reset()
                obs, _ = env.reset()
                if self._success_term_class:
                    self.success_term.func.reset()
                    
                task_description = ""
                success = False
                subtask_successes = {key: False for key in obs["subtask_terms"]}
                
                # warm up
                for _ in range(self.warmup_step):
                    obs, reward, terminations, timeouts, info = env.step(env.cfg.mimic_config.default_actions[None])

                for t in range(self.max_timestep):
                    if self.save_video:
                        for key in obs["images"]:
                            image = obs["images"][key][0].cpu().numpy()
                            cv2.putText(image, f'{key}: step {t}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            for idx,subtask in enumerate(obs["subtask_terms"].keys()):
                                cv2.putText(image, f'{subtask}: {obs["subtask_terms"][subtask].cpu().numpy()[0]}', (50, 50*(idx+2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            if hasattr(policy, "openloop_actions") and len(policy.openloop_actions) == 0:
                                cv2.putText(image, f'Next step new action trunk!', (50, 50*(idx+3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            self.video_logger.extend(
                                key, image, category="color"
                            )

                    traj_length += 1

                    with torch.no_grad():
                        if (
                            "pointcloud" in obs.keys()
                            or "pointcloud" in obs.get("policy_infer", {}).keys()
                        ):
                            action = policy.get_action(
                                preprocess_obs(
                                    self._isaac_obs_wrapper(obs),
                                    # self.pcd_aug,
                                    None,
                                    self.pcd_transform,
                                    self.pcd_channels,
                                ),
                                pcd_npoints=self.pcd_num_points,
                                in_channels=self.pcd_channels,
                                task_description=task_description,
                                t=t,
                                hist_action_cond=self.hist_action_cond
                            )
                        else:
                            action = policy.get_action(
                                preprocess_obs(
                                    self._isaac_obs_wrapper(obs), None, None, 3
                                ),
                                pcd_npoints=self.pcd_num_points,
                                in_channels=3,
                                task_description=task_description,
                                t=t,
                                hist_action_cond=self.hist_action_cond
                            )
                        
                    action[-1] = 0.0 if action[-1] < 0.5 else 1.0

                    if self.collision_pred:
                        assert False, "Not support collision pred"
                    else:
                        if isinstance(action, np.ndarray):
                            action = torch.from_numpy(action)
                        next_obs, reward, terminations, timeouts, info = env.step(
                            action[None]
                        )
                        for key in subtask_successes:
                            subtask_successes[key] = subtask_successes[key] or next_obs["subtask_terms"][key]
                        if self.success_term.func(env, **self.success_term.params):
                            success = True
                            terminations[0] = True
                        done = torch.logical_or(terminations, timeouts)

                    eps_reward += reward
                    obs = next_obs

                    if done:
                        break

                # if not info["sub_task_success"]:
                #     break
                total_reward += eps_reward
                total_success += success
                for key in subtask_successes:
                    subtask_success_nums[key] = subtask_successes[key] + subtask_success_nums.get(key, 0)

                if self.save_video:
                    postfix = "success" if success else "fail"
                    self.video_logger.save(dir_name=f"{i}-th-{postfix}", model_name=policy_name)
                    self.video_logger.reset()
                print(subtask_success_nums)
                pbar.set_description(f"{self.task_name} total_success: {total_success} at rank: {self.rank}")

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

        return total_success / episode_num, total_reward / episode_num, (imgs, subtask_success_nums, episode_num)


class IsaacEnvWbcRolloutRunner(IsaacEnvRolloutRunner):
    def __init__(
        self,
        task_name,
        episode_num=100,
        save_video=False,
        headless=True,
        obs_mode="pointcloud",
        pcd_channels=4,
        pcdnet_pretrain_domain="",
        random_reset=True,
        collision_pred=False,
        num_envs=1,
        device="cuda:0",
        seed=0,
        state_keys=None,
        video_save_dir=None,
        world_size=1,
        rank=0,
        max_timestep=1200,
        wbc_controller=None,
        until_wbc_reach=False,
        robot_type="ur5e",
        **kwargs,
    ):
        from isaaclab_mimic.utils.robots.mobile_wbc_controller import MobileWbcController
        from isaaclab_mimic.utils.robots.mobile_wbc_controller_dual import MobileDualArmWbcController
        self.wbc_controller: MobileDualArmWbcController | MobileWbcController = wbc_controller
        self.until_wbc_reach = until_wbc_reach
        self.robot_type = robot_type
        super().__init__(
            task_name, episode_num, save_video, headless,
            obs_mode, pcd_channels, pcdnet_pretrain_domain,
            random_reset, collision_pred, num_envs, device,
            seed, state_keys, video_save_dir, world_size,
            rank, max_timestep, **kwargs,)
        self.wbc_controller.cfg.dt = self.env.unwrapped.step_dt

    @torch.inference_mode()
    def run(self, policy, policy_name="model"):
        policy.to(self.device)
        try:
            print(f"Begin eval model:{policy_name}")
            episode_num = self.episode_num  # upper bound for number of trajectories
            imgs = OrderedDict()

            total_success = 0
            total_reward = 0
            env = self.env
            subtask_success_nums = {}

            pbar = tqdm(range(episode_num), position=1, leave=True)

            for i in pbar:
                eps_reward = 0
                traj_length = 0
                done = False
                self.success_term.func(env, **self.success_term.params)

                policy.reset()
                obs, _ = env.reset()
                
                # warm up
                for _ in range(self.warmup_step):
                    obs, reward, terminations, timeouts, info = env.step(env.cfg.mimic_config.default_actions[None])
                
                task_description = ""
                success = False
                subtask_successes = {key: False for key in obs["subtask_terms"]}
                for t in range(self.max_timestep):
                    traj_length += 1

                    with torch.no_grad():
                        if (
                            "pointcloud" in obs.keys()
                            or "pointcloud" in obs["policy_infer"].keys()
                        ):
                            action = policy.get_action(
                                preprocess_obs(
                                    self._isaac_obs_wrapper(obs),
                                    # self.pcd_aug,
                                    None,
                                    self.pcd_transform,
                                    self.pcd_channels,
                                ),
                                pcd_npoints=self.pcd_num_points,
                                in_channels=self.pcd_channels,
                                task_description=task_description,
                                t=t,
                            )
                        else:
                            action = policy.get_action(
                                preprocess_obs(
                                    self._isaac_obs_wrapper(obs), None, None, 3
                                ),
                                pcd_npoints=self.pcd_num_points,
                                in_channels=3,
                                task_description=task_description,
                                t=t,
                            )
                        
                    action[-1] = 0.0 if action[-1] < 0.5 else 1.0
                    if self.collision_pred:
                        assert False, "Temporarily not support collision pred"
                        action[-2] = 0.0 if action[-2] < 0.5 else 1.0
                        ignore_collisions = bool(action[-1])
                        action = action[:-1]
                        next_obs, reward, terminations, timeouts, info = env.step(action)
                        done = torch.logical_or(terminations, timeouts)
                    else:
                        if self.robot_type == "ur5e":
                            joint_action = action[:6]
                            ee_pos = self.wbc_controller.fkine(joint_action)
                            current_joint_pos = obs["policy"]["joint_pos"]
                            current_joint_pos = current_joint_pos.cpu().numpy().squeeze()
                            self.wbc_controller.set_goal(ee_pos)
                            self.wbc_controller.update_joint_pos(current_joint_pos[:6])
                            target_reached, dq = self.wbc_controller.step_robot()
                            target_q = self.wbc_controller.dt * dq + current_joint_pos[:6]
                            action[:6] = target_q

                        if isinstance(action, np.ndarray):
                            action = torch.from_numpy(action)
                        next_obs, reward, terminations, timeouts, info = env.step(
                            action[None]
                        )
                        for key in subtask_successes:
                            subtask_successes[key] = subtask_successes[key] or next_obs["subtask_terms"][key]
                        if self.success_term.func(env, **self.success_term.params):
                            success = True
                            terminations[0] = True
                        done = torch.logical_or(terminations, timeouts)

                    eps_reward += reward
                    if self.save_video:
                        for key in obs["images"]:
                            self.video_logger.extend(
                                key, obs["images"][key][0].cpu().numpy(), category="color"
                            )
                    obs = next_obs

                    if done:
                        break

                # if not info["sub_task_success"]:
                #     break

                total_reward += eps_reward
                total_success += success
                for key in subtask_successes:
                    subtask_success_nums[key] = subtask_successes[key] + subtask_success_nums.get(key, 0)

                if self.save_video:
                    postfix = "success" if success else "fail"
                    self.video_logger.save(dir_name=f"{i}-th-{postfix}", model_name=policy_name)
                    self.video_logger.reset()
                print(subtask_success_nums)
                pbar.set_description(f"{self.task_name} total_success: {total_success} at rank: {self.rank}")

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

        return total_success / episode_num, total_reward / episode_num, (imgs, subtask_success_nums, episode_num)

if __name__ == "__main__":
    # generate for all tasks
    runner = IsaacEnvRolloutRunner("Isaac-UR5-CloseMicroWave-Mimic-v0", headless=True)
    print("Done")
