from PIL import Image
import numpy as np
from collections import OrderedDict
from typing import Dict
import copy
import cv2

from ppt_learning.utils.replay_buffer import ReplayBuffer
from ppt_learning.utils.sampler import SequenceSampler, get_val_mask
from ppt_learning.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ppt_learning.utils.pcd_utils import (
    add_gaussian_noise,
    randomly_drop_point,
    voxelize_point_cloud,
    se3_augmentation,
    create_pointcloud_from_rgbd,
)

from ppt_learning.paths import *

from ppt_learning.utils.pcd_utils import BOUND

try:
    from gensim2.env.utils.rlbench import SCENE_BOUNDS as RLBENCH_BOUNDS
except:
    print("RLBench is not installed. Skip.")

import os
import zarr
import ipdb

import argparse


class TrajDataset:
    """
    Single Dataset class that converts simulation data into trajectory data.
    Explanations of parameters are in config
    """

    def __init__(
        self,
        domain,
        dataset_path,
        state_keys=None,
        mode="train",
        episode_cnt=10,
        step_cnt=100,
        data_augmentation=False,
        se3_augmentation=False,  # pcd in roboframe do not need this (gensim2 & rlbench)
        data_ratio=1,
        use_disk=False,
        horizon=4,
        pad_before=0,
        pad_after=0,
        val_ratio=0.1,
        seed=233,
        action_horizon=1,
        observation_horizon=1,
        resize_img=True,
        img_size=224,
        dataset_postfix="",
        dataset_encoder_postfix="",
        precompute_feat=False,
        env_rollout_fn=None,
        use_multiview=False,
        normalize_state=False,
        from_empty=True,
        use_pcd=False,
        pcdnet_pretrain_domain="",
        pcd_channels=None,
        load_from_cache=True,
        env_names=None,
        voxelization=False,
        voxel_size=0.01,
        ignored_keys=None,
        use_lru_cache=True,
        rank=0,  # Add rank for DDP
        **kwargs,
    ):  
        self.rank = rank
        self.dataset_name = domain
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_augmentation = data_augmentation
        self.episode_cnt = episode_cnt
        self.step_cnt = step_cnt
        self.action_horizon = action_horizon
        self.observation_horizon = observation_horizon
        self.precompute_feat = precompute_feat
        self.data_ratio = data_ratio
        self.use_multiview = use_multiview
        self.normalize_state = normalize_state
        self.resize_img = resize_img
        self.img_size = img_size
        if use_pcd:
            assert pcd_channels is not None, "pcd_channels must be provided for pcd"
        if pcd_channels is not None:
            assert pcd_channels in [
                3,
                4,
                5,
                6,
                7,
            ], "pcd_channels must be one of [3, 4, 5, 6, 7]"
        self.pcd_channels = pcd_channels
        self.mode = mode
        self.use_pcd = use_pcd
        self.pcd_transform = None
        self.pcdnet_pretrain_domain = pcdnet_pretrain_domain
        self.pcd_num_points = None
        self.env_names = env_names
        self.se3_augmentation = se3_augmentation
        self.bounds = BOUND

        self.state_keys = state_keys
        if state_keys is None:
            self.state_keys = [
                "eef_pos",
                "eef_quat",
                "joint_pos",
                "joint_vel",
            ]
        self.ignored_keys = ignored_keys
        if ignored_keys is None:
            self.ignored_keys = ["initial_state", "states", "depths"] # , "images", "color"]
            # self.use_pcd = False

        self.voxelization = voxelization
        self.voxel_size = voxel_size

        self.update_pcd_transform()

        load_from_cache = os.path.exists(dataset_path) and load_from_cache
        print(
            f"\n\n >>>dataset_path: {dataset_path} load_from_cache: {load_from_cache} \n\n"
        )
        self.dataset_path = dataset_path
        self.replay_buffer = None
        
        if use_disk:
            # self.replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(path=dataset_path))
            if load_from_cache:
                if use_lru_cache:
                    store = zarr.DirectoryStore(dataset_path)
                    cache = zarr.LRUStoreCache(store=store, max_size=2**30)
                    group = zarr.open(cache, "r")
                    self.replay_buffer = ReplayBuffer.create_from_group(group, env_names=self.env_names)
                    print("Using lru cache")
                else:
                    self.replay_buffer = ReplayBuffer.create_from_path(
                        dataset_path, self.env_names
                    )
            else:
                self.replay_buffer = ReplayBuffer.create_empty_zarr(
                    storage=zarr.DirectoryStore(path=dataset_path)
                )
        else:
            self.replay_buffer = ReplayBuffer.create_empty_numpy()

        # loading datasets
        if not from_empty:
            if not (load_from_cache and use_disk):
                self.load_dataset()

            self.get_training_dataset(val_ratio, seed)
            self.get_sa_dim()

    def update_pcd_transform(self, pcd_setup_cfg=None):
        if not self.use_pcd:
            return
        assert (
            self.pcdnet_pretrain_domain != ""
        ), "pcdnet_domain must be provided for pcdnet"

        from openpoints.transforms import build_transforms_from_cfg

        if pcd_setup_cfg is None:
            from openpoints.utils import EasyConfig

            pcd_setup_cfg = EasyConfig()
            pcd_setup_cfg.load(
                f"{PPT_DIR}/models/pointnet_cfg/{self.pcdnet_pretrain_domain}/pcd_setup.yaml",
                recursive=True,
            )

        # in case only val or test transforms are provided.
        if self.mode not in pcd_setup_cfg.keys() and self.mode in ["val", "test"]:
            trans_split = "val"
        else:
            trans_split = self.mode
        self.pcd_transform = build_transforms_from_cfg(
            trans_split, pcd_setup_cfg.datatransforms
        )
        self.pcd_num_points = pcd_setup_cfg.num_points

    def get_sa_dim(self):
        self.action_dim = self[0]["data"]["action"].shape[-1]  #  * self.action_horizon
        self.state_dim = self[0]["data"]["state"].shape[-1]

    def get_normalizer(self, mode="limits", **kwargs):
        """action normalizer"""
        data = self._sample_to_data(self.replay_buffer)
        self.normalizer = LinearNormalizer()
        self.normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for k, v in self.normalizer.params_dict.items():
            print(f"normalizer {k} stats min: {v['input_stats'].min}")
            print(f"normalizer {k} stats max: {v['input_stats'].max}")
        return self.normalizer

    def append_episode(self, episode, description="", env_name=None):
        data = OrderedDict()

        def recursive_dict_update(d, u):
            for k, v in u.items():
                if isinstance(v, (dict, OrderedDict)):
                    d[k] = recursive_dict_update(d.get(k, {}), v)
                else:
                    if k not in d:
                        d[k] = []
                    d[k].append(v)
            return d

        def recursive_array(d):
            if isinstance(d, (dict, OrderedDict)):
                for k, v in d.items():
                    d[k] = recursive_array(v)
            elif isinstance(d, list):
                d = np.array(d)
            return d

        for dataset_step in episode:
            recursive_dict_update(data, dataset_step)
        for key, val in data.items():
            data[key] = recursive_array(data[key])

        self.replay_buffer.add_episode(data, description=description, env_name=env_name)

    def get_episode(self, idx):
        return self.replay_buffer.get_episode(idx)

    def _sample_to_data(self, sample):
        data = {"action": sample["action"]} if "action" in sample else {"action": sample["actions"]}
        if self.normalize_state:
            if "state" in sample:
                data["state"] = sample["state"]  # 1 x N
            else:
                data["state"] = self.get_state(sample)
        return data

    def get_training_dataset(self, val_ratio, seed):
        # split into train and test sets
        self.val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        self.train_mask = ~self.val_mask

        # considering hyperparameters and masking
        n_episodes = int(
            self.data_ratio * min(self.episode_cnt, self.replay_buffer.n_episodes)
        )
        self.val_mask[n_episodes:] = False
        self.train_mask[n_episodes:] = False

        # normalize and create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
            ignored_keys=self.ignored_keys,
        )
        print(
            f"{self.dataset_path} size: {len(self.sampler)} episodes: {n_episodes} train: {self.train_mask.sum()} eval: {self.val_mask.sum()}"
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.mode = "val"
        val_set.update_pcd_transform()
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int):
        """normalize observation and actions"""
        # import time
        # start_time = time.time()
        sample = self.sampler.sample_sequence(idx)
        # end_time = time.time()
        # print("Time used of sample:", end_time - start_time)
        if "actions" in sample: # Align the name
            sample["action"] = sample["actions"]
            del sample["actions"]

        # the full horizon is for the trajectory
        def recursive_horizon(data):
            for key, val in data.items():
                if isinstance(val, (dict, OrderedDict)):
                    recursive_horizon(val)
                else:
                    if (key not in ["action", "actions"]) and (key != "action_is_pad"):
                        if key == "language":
                            data[key] = val
                        else:
                            data[key] = val[: self.observation_horizon]
                    else:
                        if key in ["action", "actions"]:
                            data["action"] = val[
                                self.observation_horizon
                                - 1 : self.action_horizon
                                + self.observation_horizon
                                - 1
                            ]
                        elif key == "action_is_pad":
                            data["action_is_pad"] = val[
                                self.observation_horizon
                                - 1 : self.action_horizon
                                + self.observation_horizon
                                - 1
                            ]

        if self.use_pcd:
            if "pointcloud" not in sample["obs"]:
                sample["obs"]["pointcloud"] = {}
                for cam_idx, (cam_name_depth, cam_name_rgb) in enumerate(zip(sample["obs"]["depths"], sample["obs"]["images"])):
                    depth = sample["obs"]["depths"][cam_name_depth]
                    rgb = sample["obs"]["images"][cam_name_rgb]
                    sample["obs"]["pointcloud"][f"camera_{cam_idx}"] = create_pointcloud_from_rgbd(
                        depth=depth[..., 0],
                        rgb=rgb,
                        intrinsic_matrix=self.replay_buffer.meta["camera_info"][f"camera_{cam_idx}"]["intrinsics"][0],
                        position=self.replay_buffer.meta["camera_info"][f"camera_{cam_idx}"]["extrinsics"][0,:3],
                        orientation=self.replay_buffer.meta["camera_info"][f"camera_{cam_idx}"]["extrinsics"][0,3:],
                    )
                camera_nums = len(sample["obs"]["pointcloud"])
                sample["obs"]["pointcloud"]["pos"] = np.concatenate(
                    [
                        sample["obs"]["pointcloud"][f"camera_{cam_idx}"]["pos"]
                        for cam_idx in range(camera_nums)
                    ],
                    axis=1,
                )
                sample["obs"]["pointcloud"]["color"] = np.concatenate(
                    [
                        sample["obs"]["pointcloud"][f"camera_{cam_idx}"]["color"]
                        for cam_idx in range(camera_nums)
                    ],
                    axis=1,
                )
                for cam_idx in range(camera_nums):
                    del sample["obs"]["pointcloud"][f"camera_{cam_idx}"]
                
            if self.data_augmentation:
                sample["obs"]["pointcloud"]["pos"] = self.pcd_aug(
                    sample["obs"]["pointcloud"]["pos"]
                )

            if self.se3_augmentation:
                poses, gripper = sample["action"][:, :6], sample["action"][:, 6:]
                poses, sample["obs"]["pointcloud"]["pos"] = se3_augmentation(
                    poses, sample["obs"]["pointcloud"]["pos"], bounds=self.bounds
                )
                sample["action"] = np.concatenate([poses, gripper], axis=-1)
                sample["obs"]["pointcloud"]["pos"] = pcds

            if self.voxelization:
                sample["obs"]["pointcloud"] = np.array(
                    [
                        voxelize_point_cloud(
                            sample["obs"]["pointcloud"]["pos"][i],
                            sample["obs"]["pointcloud"]["color"][i],
                            self.voxel_size,
                        )
                        for i in range(len(sample["obs"]["pointcloud"]["pos"]))
                    ]
                )
            if self.pcd_transform is not None:
                seq_len, num_points = sample["obs"]["pointcloud"]["pos"].shape[:2]
                for key, val in sample["obs"][
                    "pointcloud"
                ].items():  # Reshape for tranform API
                    sample["obs"]["pointcloud"][key] = val.reshape(
                        seq_len * num_points, -1
                    )
                try:
                    sample["obs"]["pointcloud"] = self.pcd_transform(
                        sample["obs"]["pointcloud"]
                    )
                except Exception as e:
                    print(
                        f"Found error {e}, should not be a problem when initializing the dataset"
                    )
                for key, val in sample["obs"]["pointcloud"].items():
                    sample["obs"]["pointcloud"][key] = val.reshape(
                        seq_len, num_points, -1
                    )

        self.flat_sample(sample)
        self.transform(sample)

        # if "pointcloud" in sample.keys():
        #     assert sample["pointcloud"].shape[-1] == self.pcd_channels, f"pointcloud channel mismatch! expected {self.pcd_channels}, got {sample['pointcloud'].shape[-1]}"
        recursive_horizon(sample)

        return {"domain": self.dataset_name, "data": sample}

    def save_dataset(self):
        if self.rank == 0:  # Only rank 0 saves the dataset
            self.replay_buffer.save_to_path(self.dataset_path)

    def load_dataset(self):
        self.replay_buffer = ReplayBuffer.copy_from_path(self.dataset_path)
        print("Replay buffer keys: ", self.replay_buffer.keys())

    def get_state(self, sample):
        sample["state"] = []
        for key in self.state_keys:
            if key in sample.keys():
                sample["state"].append(sample[key])
                del sample[key]
        sample["state"] = np.concatenate(sample["state"], axis=-1)

        return sample

    def transform(self, sample):
        if self.resize_img and "image" in sample.keys():
            for key, val in sample["image"].items():
                # Image shape N, H, W, C
                resize_image_sequence(val, (self.img_size, self.img_size))

    def flat_sample(self, sample):
        if "obs" in sample.keys():
            for key, val in sample["obs"].items():
                sample[key] = val
            del sample["obs"]
        
        if "state" not in sample:
            sample = self.get_state(sample)

        if "images" in sample.keys() and "image" not in sample.keys():
            sample["image"] = sample.pop("images")

        if not self.use_pcd:
            if "pointcloud" in sample.keys():
                del sample["pointcloud"]
        if self.use_pcd and ("pointcloud" in sample.keys()):
            if self.pcd_channels == 3:
                pass
                # sample['pointcloud']['pos'] = sample['pointcloud']['pos']
            elif self.pcd_channels == 6:
                sample["pointcloud"]["x"] = np.concatenate(
                    [sample["pointcloud"]["pos"], sample["pointcloud"]["colors"]],
                    axis=-1,
                )
            elif self.pcd_channels == 4:
                sample["pointcloud"]["x"] = np.concatenate(
                    [sample["pointcloud"]["pos"], sample["pointcloud"]["heights"]],
                    axis=-1,
                )
            elif self.pcd_channels == 5:
                sample["pointcloud"]["x"] = np.concatenate(
                    [
                        sample["pointcloud"]["pos"],
                        sample["pointcloud"]["heights"],
                        sample["pointcloud"]["seg"][..., 1].unsqueeze(-1),
                    ],
                    axis=-1,
                )
            elif self.pcd_channels == 7:
                sample["pointcloud"]["x"] = np.concatenate(
                    [
                        sample["pointcloud"]["pos"],
                        sample["pointcloud"]["colors"],
                        sample["pointcloud"]["heights"],
                    ],
                    axis=-1,
                )
            else:
                raise ValueError(f"Invalid pcd_channels: {self.pcd_channels}")

    def pcd_aug(self, pcd):
        pcd = add_gaussian_noise(pcd)
        pcd = randomly_drop_point(pcd)

        return pcd


def delete_indices(
    replay_buffer: ReplayBuffer,
    env_name: str,
) -> np.ndarray:
    episode_ends = replay_buffer.episode_ends[:]
    episode_desc = replay_buffer.meta["episode_descriptions"]
    env_names = replay_buffer.meta["env_names"]

    indices = list()
    for i in range(len(episode_ends)):
        if env_names[i] == env_name:
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i - 1]
            end_idx = episode_ends[i]
            eps_description = episode_desc
            episode_length = end_idx - start_idx

            # TODO

            # remove i in replay.meta

            # remove start_idx:end_idx in replay.data


def resize_image_sequence(images, target_size):
    """
    Resize an image sequence using OpenCV
    
    Args:
        images: numpy array of shape (N, H, W, C) where:
               N = number of images
               H = height
               W = width
               C = channels
        target_size: tuple of (height, width)
    
    Returns:
        resized images array of shape (N, new_H, new_W, C)
    """
    N, H, W, C = images.shape
    new_H, new_W = target_size
    
    # Reshape to 2D array of images for faster processing
    reshaped = images.reshape(-1, H, W, C)
    
    # Preallocate output array
    output = np.empty((N, new_H, new_W, C), dtype=images.dtype)
    
    # Resize each image
    for i in range(N):
        output[i] = cv2.resize(images[i], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    
    return output


if __name__ == "__main__":
    import collections

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rlbench_test_keypose")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    dataset = TrajDataset(
        dataset_path="/mnt/bn/robot-minghuan-datasets-lq/xiaoshen/datasets/ur5_close_microwave/ur5_close_microwave_version_4_generated_1000_2.zarr",
        from_empty=False,
        use_disk=True,
        load_from_cache=True,
    )
    dataset.load_dataset()
    print(collections.Counter(dataset.replay_buffer.meta["episode_descriptions"]))
    if "env_names" in dataset.replay_buffer.meta.keys():
        print(collections.Counter(dataset.replay_buffer.meta["env_names"]))

    if args.render:
        # visualize point cloud trajectory with open3d
        import open3d as o3d

        data = dataset.replay_buffer
        pcds_traj = data["obs"]["pointcloud"]["pos"]

        # open a blank o3d visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        for i in range(len(pcds_traj)):
            pcds = pcds_traj[i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcds)
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            vis.run()

            # clean up the visualizer

        vis.destroy_window()
