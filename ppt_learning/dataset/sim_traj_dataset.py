"""Single trajectory dataset for robotics learning.

This module provides the TrajDataset class for managing trajectory data from
simulation environments, supporting data augmentation, normalization, and
efficient sampling for training robotic manipulation policies.

Key Features:
    - Flexible data loading from disk or memory
    - Point cloud generation from RGB-D images
    - Comprehensive data augmentation pipeline
    - State and action normalization
    - Support for distributed training
    - Efficient caching with LRU support
"""

import numpy as np
from collections import OrderedDict
import copy
import cv2
import os
import zarr
import ipdb
import argparse
import time
import albumentations as A
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class TrajDataset:
    """Single trajectory dataset for robotic manipulation learning.
    
    This class manages trajectory data from a single domain/task, handling
    data loading, preprocessing, augmentation, and sampling for training
    neural network policies.
    
    Attributes:
        dataset_name: Name of the dataset/domain
        replay_buffer: Buffer storing trajectory data
        sampler: Sequence sampler for training data
        normalizer: Data normalizer for states/actions
        action_dim: Dimension of action space
        state_dim: Dimension of state space
    """

    def __init__(
        self,
        domain: str,
        dataset_path: Union[str, List[str]],
        mode: str = "train",
        episode_cnt: int = 10,
        step_cnt: int = 100,
        data_ratio: float = 1.0,
        use_disk: bool = False,
        horizon: int = 4,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.1,
        seed: int = 233,
        action_horizon: int = 1,
        observation_horizon: int = 1,
        hist_action_cond: bool = False,
        resize_img: bool = True,
        img_size: Tuple[int, int] = (224, 224),
        augment_pcd: bool = False,
        se3_augmentation: bool = False,  # pcd in roboframe do not need this (gensim2 & rlbench)
        augment_img: bool = True,
        augment_depth: bool = True,
        img_augment_prob: float = 0.7,
        dataset_postfix: str = "",
        dataset_encoder_postfix: str = "",
        precompute_feat: bool = False,
        env_rollout_fn: Optional[Callable] = None,
        use_multiview: bool = False,
        normalize_state: bool = False,
        from_empty: bool = True,
        use_pcd: bool = False,
        pcdnet_pretrain_domain: str = "",
        pcd_channels: Optional[int] = None,
        load_from_cache: bool = True,
        voxelization: bool = False,
        voxel_size: float = 0.01,
        use_lru_cache: bool = True,
        rank: int = 0,  # Add rank for DDP
        ignored_keys: Optional[List[str]] = None,
        state_keys: Optional[List[str]] = None,
        action_key: str = "actions",
        pose_transform: Optional[Any] = None,
        norm_depth: bool = False,
        **kwargs: Any,
    ) -> None:
        self.rank = rank
        self.dataset_name = domain
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.augment_pcd = augment_pcd
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
        self.action_key = action_key
        self.hist_action_cond = hist_action_cond
        # Validate configuration
        self._validate_config(use_pcd, pcd_channels, hist_action_cond)
        self.pcd_channels = pcd_channels
        self.mode = mode
        self.use_pcd = use_pcd
        self.pcd_transform = None
        self.pcdnet_pretrain_domain = pcdnet_pretrain_domain
        self.pcd_num_points = None
        self.bounds = BOUND
        self.se3_augmentation = se3_augmentation
        self.augment_img = augment_img
        self.augment_depth = augment_depth
        self.img_transform = None
        self.depth_transform = None
        self.norm_depth = norm_depth
        self.warp_func = None
        if norm_depth:
            self.warp_func = WarpMinMax()

        # Setup augmentation pipelines
        self.img_transform = self._setup_image_augmentation(augment_img, img_augment_prob)
        self.depth_transform = self._setup_depth_augmentation(augment_depth, img_augment_prob)
        
        self.state_keys = state_keys
        if state_keys is None:
            self.state_keys = [
                "eef_pos",
                "eef_quat",
                "joint_pos",
                # "joint_vel",
                "normalized_gripper_pos"
            ]
        self.ignored_keys = ignored_keys
        if ignored_keys is None:
            self.ignored_keys = [
                "initial_state",
                "states",
                "depths",
            ] # , "images", "color"]
            # self.use_pcd = False

        self.voxelization = voxelization
        self.voxel_size = voxel_size

        self.pose_transform = None
        if pose_transform is not None:
            if pose_transform == "quat_to_pose":
                from ppt_learning.utils.pose_utils import quat_to_pose
                self.pose_transform = quat_to_pose
            else:
                raise NotImplementedError("Pose transform function not assigned!")

        self.update_pcd_transform()

        load_from_cache = os.path.exists(dataset_path) and load_from_cache
        logger.info(
            f"Loading dataset: {dataset_path} (load_from_cache={load_from_cache})"
        )
        self.dataset_path = dataset_path
        self.replay_buffer = None

        if use_disk:
            # self.replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(path=dataset_path))
            if load_from_cache:
                if use_lru_cache:
                    store = zarr.DirectoryStore(dataset_path)
                    cache = zarr.LRUStoreCache(store=store, max_size=2**32)
                    group = zarr.open(cache, "r")
                    self.replay_buffer = ReplayBuffer.create_from_group(
                        group
                    )
                    logger.info(f"Using LRU cache with max_size={2**38}")
                else:
                    self.replay_buffer = ReplayBuffer.create_from_path(
                        dataset_path
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

    def update_pcd_transform(self, pcd_setup_cfg: Optional[Any] = None) -> None:
        """Update point cloud transformation configuration.
        
        Args:
            pcd_setup_cfg: Point cloud setup configuration object
        """
        if not self.use_pcd:
            return
        if not self.pcdnet_pretrain_domain:
            raise ValueError("pcdnet_pretrain_domain must be provided when use_pcd=True")

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

    def get_sa_dim(self) -> None:
        """Calculate and set action and state dimensions from the first sample."""
        self.action_dim = self[0]["data"]["action"].shape[-1]  #  * self.action_horizon
        self.state_dim = self[0]["data"]["state"].shape[-1]

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        """Get action normalizer fitted on the dataset.
        
        Args:
            mode: Normalization mode (e.g., 'limits', 'gaussian')
            **kwargs: Additional keyword arguments for normalization
            
        Returns:
            Fitted LinearNormalizer instance
        """
        data = self._sample_to_data(self.replay_buffer)
        self.normalizer = LinearNormalizer()
        self.normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for k, v in self.normalizer.params_dict.items():
            logger.info(f"Normalizer {k} - min: {v['input_stats'].min:.4f}, max: {v['input_stats'].max:.4f}")
        return self.normalizer

    def append_episode(
        self, 
        episode: List[Dict[str, Any]], 
        description: str = "", 
        env_name: Optional[str] = None
    ) -> None:
        """Append a new episode to the replay buffer.
        
        Args:
            episode: List of timestep dictionaries
            description: Episode description
            env_name: Environment name
        """
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

    def get_episode(self, idx: int) -> Dict[str, Any]:
        """Get an episode by index.
        
        Args:
            idx: Episode index
            
        Returns:
            Episode data dictionary
        """
        return self.replay_buffer.get_episode(idx)

    def _sample_to_data(self, sample: Union[ReplayBuffer, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Convert sample to normalized data dictionary.
        
        Args:
            sample: Sample from replay buffer
            
        Returns:
            Dictionary with action and optionally state data
        """
        data = {"action": sample[self.action_key]}
        if self.normalize_state:
            if "state" in sample:
                data["state"] = sample["state"]  # 1 x N
            else:
                data["state"] = self.get_state(sample["obs"])
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
            action_key=self.action_key,
        )
        logger.info(
            f"Dataset {self.dataset_name}: {len(self.sampler)} samples, "
            f"{n_episodes} episodes ({self.train_mask.sum()} train, {self.val_mask.sum()} val)"
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
        """Get total number of samples in the dataset."""
        if not hasattr(self, 'sampler') or self.sampler is None:
            return 0
        return len(self.sampler)
    
    @property
    def num_episodes(self) -> int:
        """Get number of episodes in the dataset."""
        if not hasattr(self, 'replay_buffer') or self.replay_buffer is None:
            return 0
        return self.replay_buffer.n_episodes
    
    def _validate_config(self, use_pcd: bool, pcd_channels: Optional[int], hist_action_cond: bool) -> None:
        """Validate dataset configuration parameters.
        
        Args:
            use_pcd: Whether point clouds are used
            pcd_channels: Number of point cloud channels
            hist_action_cond: Whether using historical action conditioning
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate point cloud configuration
        if use_pcd:
            if pcd_channels is None:
                raise ValueError("pcd_channels must be provided when use_pcd=True")
            if pcd_channels not in [3, 4, 5, 6, 7]:
                raise ValueError(f"pcd_channels must be one of [3, 4, 5, 6, 7], got {pcd_channels}")
        
        # Validate horizon configuration
        if not hist_action_cond:
            expected_horizon = self.observation_horizon + self.action_horizon - 1
            if self.horizon != expected_horizon:
                raise ValueError(
                    f"Horizon mismatch: got {self.horizon}, expected {expected_horizon} "
                    f"(observation_horizon={self.observation_horizon} + action_horizon={self.action_horizon} - 1)"
                )
    
    def _setup_image_augmentation(self, augment_img: bool, img_augment_prob: float) -> Optional[A.Compose]:
        """Setup image augmentation pipeline.
        
        Args:
            augment_img: Whether to enable image augmentation
            img_augment_prob: Probability of applying augmentation
            
        Returns:
            Albumentations composition or None
        """
        if not augment_img:
            return None
            
        return A.Compose([
            A.OneOf([
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    sigma_limit=(0.1, 2),
                    p=img_augment_prob,
                ),
                A.MotionBlur(p=img_augment_prob),
                A.Defocus(p=img_augment_prob),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=img_augment_prob
                ),
                A.GaussNoise(p=img_augment_prob),
            ]),
        ])
    
    def _setup_depth_augmentation(self, augment_depth: bool, img_augment_prob: float) -> Optional[A.Compose]:
        """Setup depth augmentation pipeline.
        
        Args:
            augment_depth: Whether to enable depth augmentation
            img_augment_prob: Probability of applying augmentation
            
        Returns:
            Albumentations composition or None
        """
        if not augment_depth:
            return None
            
        return A.Compose([
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.25,
                    scale_limit=0.05,
                    rotate_limit=2,
                    p=img_augment_prob,
                ),
            ]),
        ])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Normalize observation and actions for the given index.
        
        Args:
            idx: Index of the trajectory sequence to retrieve
            
        Returns:
            Dictionary containing processed observations, actions, and metadata
        """
        # import time
        # start_time = time.time()
        sample = self.sampler.sample_sequence(idx)
        # print("Time used of sample:", time.time() - start_time)
        action_sub_keys = self.action_key.split('/')
        if len(action_sub_keys) > 1 or action_sub_keys[0] != "action":
            action = sample
            for key in action_sub_keys:
                if isinstance(action, (dict, OrderedDict)):
                    try:
                        action = action[key]
                    except KeyError:
                        logger.error(f"Action key not found: {key}")
                        logger.error(f"Available keys: {list(action.keys())}")
                        raise KeyError(f"Failed to access action key '{key}' in sample")
            sample["action"] = action
            del sample[action_sub_keys[0]]

        # the full horizon is for the trajectory
        def recursive_horizon(data):
            for key, val in data.items():
                if isinstance(val, (dict, OrderedDict)):
                    recursive_horizon(val)
                else:
                    if (key not in self.ignored_keys) and (key != "action") and (key != "action_is_pad"):
                        if key == "language":
                            data[key] = val
                        else:
                            data[key] = val[: self.observation_horizon]
                    else:
                        if self.hist_action_cond:
                            if key in ["action", "action_is_pad"]:
                                data[key] = val[: self.action_horizon] # including hist action
                        else:
                            if key in ["action", "action_is_pad"]:
                                data[key] = val[
                                    self.observation_horizon
                                    - 1 : self.action_horizon
                                    + self.observation_horizon
                                    - 1
                                ]

        if self.use_pcd:
            if "pointcloud" not in sample["obs"]:
                sample["obs"]["pointcloud"] = {}
                for cam_idx, (cam_name_depth, cam_name_rgb) in enumerate(
                    zip(sample["obs"]["depths"], sample["obs"]["images"])
                ):
                    depth = sample["obs"]["depths"][cam_name_depth]
                    rgb = sample["obs"]["images"][cam_name_rgb]
                    sample["obs"]["pointcloud"][f"camera_{cam_idx}"] = (
                        create_pointcloud_from_rgbd(
                            depth=depth[..., 0],
                            rgb=rgb,
                            intrinsic_matrix=self.replay_buffer.meta["camera_info"][
                                f"camera_{cam_idx}"
                            ]["intrinsics"][0],
                            # position=self.replay_buffer.meta["camera_info"][
                            #     f"camera_{cam_idx}"
                            # ]["extrinsics"][0, :3],
                            # orientation=self.replay_buffer.meta["camera_info"][
                            #     f"camera_{cam_idx}"
                            # ]["extrinsics"][0, 3:],
                        )
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

            if self.augment_pcd:
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
                    logger.warning(
                        f"Point cloud transform error (may be expected during initialization): {e}"
                    )
                for key, val in sample["obs"]["pointcloud"].items():
                    sample["obs"]["pointcloud"][key] = val.reshape(
                        seq_len, num_points, -1
                    )
        self.flat_sample(sample)

        # if "pointcloud" in sample.keys():
        #     assert sample["pointcloud"].shape[-1] == self.pcd_channels, f"pointcloud channel mismatch! expected {self.pcd_channels}, got {sample['pointcloud'].shape[-1]}"
        recursive_horizon(sample)

        self.transform(sample, idx)

        return {"domain": self.dataset_name, "data": sample}

    def save_dataset(self):
        if self.rank == 0:  # Only rank 0 saves the dataset
            self.replay_buffer.save_to_path(self.dataset_path)

    def load_dataset(self) -> None:
        """Load dataset from disk into replay buffer."""
        self.replay_buffer = ReplayBuffer.copy_from_path(self.dataset_path)
        logger.info(f"Replay buffer keys: {self.replay_buffer.keys()}")

    def get_state(self, sample: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
        """Extract and concatenate state information from sample.
        
        Args:
            sample: Sample containing state components
            
        Returns:
            Concatenated state array
        """
        res = {"state": []}
        if isinstance(sample, (dict, OrderedDict)):
            res = sample
            res['state'] = []
        for key in self.state_keys:
            if key in sample.keys():
                if len(sample[key].shape) == 1:
                    res["state"].append(np.array(sample[key])[..., None])
                else:
                    res["state"].append(sample[key])
                if isinstance(sample, (dict, OrderedDict)):
                    del sample[key]
        res["state"] = np.concatenate(res["state"], axis=-1)

        return res["state"]

    def transform(self, sample: Dict[str, Any], idx: int) -> None:
        """Apply transformations to sample data in-place.
        
        Includes image resizing, depth normalization, and augmentations.
        
        Args:
            sample: Sample dictionary to transform
            idx: Sample index (for debugging)
        """
        if self.resize_img and "image" in sample.keys():
            for key, val in sample["image"].items():
                # Image shape N, H, W, C
                sample["image"][key] = resize_image_sequence(val, (self.img_size[0], self.img_size[1]))
        if self.resize_img and "depth" in sample.keys():
            for key, val in sample["depth"].items():
                # Image shape N, H, W, C
                clippped_depth, _ = clip_depth(val)
                if not _:
                    logger.warning(f"Invalid depth detected in sample")
                sample["depth"][key] = resize_image_sequence(clippped_depth, (self.img_size[0], self.img_size[1]), interp=cv2.INTER_NEAREST)
                if self.norm_depth:
                    sample["depth"][key] = self.warp_func.warp(sample["depth"][key], sample["depth"][key])
        if self.augment_depth and "depth" in sample.keys():
            for key, val in sample["depth"].items():
                for step_idx in range(val.shape[0]):
                    sample["depth"][key][step_idx] = self.depth_transform(image=val[step_idx])["image"]
        if self.augment_img and "image" in sample.keys():
            for key, val in sample["image"].items():
                for step_idx in range(val.shape[0]):
                    sample["image"][key][step_idx] = self.img_transform(image=val[step_idx])["image"]
        if self.pose_transform is not None: # Last dim is gripper
            if len(sample['action'].shape) == 2:
                N, A = sample['action'].shape
                sample['action'] = np.concatenate([self.pose_transform(sample['action'][..., :-1].reshape(-1, A-1)).reshape(N, -1), sample['action'][..., -1:]], axis=-1)
            elif len(sample['action'].shape) == 3:
                N, L, A = sample['action'].shape
                sample['action'] = np.concatenate([self.pose_transform(sample['action'][..., :-1].reshape(-1, A-1)).reshape(N, L, -1), sample['action'][..., -1:]], axis=-1)
            else:
                raise ValueError(f"Invalid action shape: {sample['action'].shape}")

    def flat_sample(self, sample: Dict[str, Any]) -> None:
        """Flatten nested sample structure in-place.
        
        Moves observation data to top level and handles point cloud channels.
        
        Args:
            sample: Sample dictionary to flatten
        """
        if "obs" in sample.keys():
            for key, val in sample["obs"].items():
                sample[key] = val
            del sample["obs"]

        if "state" not in sample:
            sample["state"] = self.get_state(sample)

        if "images" in sample.keys() and "image" not in sample.keys():
            sample["image"] = sample.pop("images")

        if "depths" in sample.keys() and "depth" not in sample.keys():
            sample["depth"] = sample.pop("depths")

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

    def pcd_aug(self, pcd: np.ndarray) -> np.ndarray:
        """Apply augmentations to point cloud data.
        
        Args:
            pcd: Point cloud array
            
        Returns:
            Augmented point cloud array
        """
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

class WarpMinMax:
    EPS = 1e-3
    def warp(self, depth, reference, **kwargs):
        depth_min, depth_max = (
            reference.reshape(depth.shape[0], -1).min(1, keepdims=True)[0],
            reference.reshape(depth.shape[0], -1).max(1, keepdims=True)[0],
        )
        if ((depth_max - depth_min) < self.EPS).any():
            depth_max[(depth_max - depth_min) < self.EPS] = (
                depth_min[(depth_max - depth_min) < self.EPS] + self.EPS
            )
        return (depth - depth_min[:, None, None]) / (depth_max - depth_min)[
            :, None, None
        ]

    def unwarp(self, depth, reference, **kwargs):
        depth_min, depth_max = (
            reference.reshape(depth.shape[0], -1).min(1, keepdims=True)[0],
            reference.reshape(depth.shape[0], -1).max(1, keepdims=True)[0],
        )
        if ((depth_max - depth_min) < self.EPS).any():
            depth_max[(depth_max - depth_min) < self.EPS] = (
                depth_min[(depth_max - depth_min) < self.EPS] + self.EPS
            )
        return depth * (depth_max - depth_min)[:, None, None] + depth_min[:, None, None]

def clip_depth(depth):
    res = True
    valid_mask = np.logical_and(depth > 0.01, ~np.isnan(depth)) & (~np.isinf(depth))
    if valid_mask.sum() == 0:
        print(
            "No valid mask in the depth map"
        )
        res = False
    if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0:
        depth[np.isnan(depth)] = depth[valid_mask].max()
    if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0:
        depth[np.isinf(depth)] = depth[valid_mask].max()

    return depth, res

def resize_image_sequence(images, target_size, interp=cv2.INTER_AREA):
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
        res = cv2.resize(
            images[i], (new_W, new_H), interpolation=interp
        )
        if C == 1:
            output[i] = res[:, :, np.newaxis]
        else:
            output[i] = res

    return output


if __name__ == "__main__":
    import collections
    import matplotlib.pyplot as plt
    import imageio.v3 as imageio

    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default="rlbench_test_keypose")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    dataset = TrajDataset(
        domain="debug",
        dataset_path="/mnt/bn/robot-minghuan-datasets-lq/xiaoshen/datasets/ur5_put_bowl_in_microwave_and_close/one_camera_higher_627_close_faster.zarr",
        from_empty=False,
        use_disk=True,
        load_from_cache=True,
        use_lru_cache=True,
        val_ratio=0.,
        action_horizon=16,
        observation_horizon=3,
        horizon=18,
        pad_before=2,
        pad_after=15,
        use_pcd=True,
        pcd_channels=4,
        pcdnet_pretrain_domain="scanobjectnn",
        ignored_keys=["initial_state", "states", "images", "color", "abs_gripper_pos"],
        # action_key="wbc_target/r"
    )
    dataset.__getitem__(0)
    # dataset.load_dataset()
    # rigid_pos = dataset.replay_buffer['initial_state']['rigid_object']['bowl']['root_pose'][:, :2]
    # articulation_pos = dataset.replay_buffer['initial_state']['articulation']['microwave']['root_pose'][:, :2]
    # plt.scatter(rigid_pos[:, 0], rigid_pos[:, 1], label="rigid body")
    # plt.scatter(articulation_pos[:, 0], articulation_pos[:, 1], color='b', label="articulation")
    # plt.legend()
    # plt.savefig("pos_visualization.png")
    # images = dataset.replay_buffer.data.obs.images.camera_0[73366:74103]
    import ipdb; ipdb.set_trace()
    # print(collections.Counter(dataset.replay_buffer.meta["episode_descriptions"]))
    # if "env_names" in dataset.replay_buffer.meta.keys():
    #     print(collections.Counter(dataset.replay_buffer.meta["env_names"]))

    if args.render:
        # visualize point cloud trajectory with open3d
        import open3d as o3d

        data = dataset.replay_buffer
        pcds_traj = data["obs"]["pointcloud"]["pos"]

        # open a blank o3d visualizer
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()

        for i in range(len(pcds_traj)):
            pcds = pcds_traj[i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcds)
            o3d.write_point_cloud(f"pcd_{i}.ply", pcd)
            # vis.clear_geometries()
            # vis.add_geometry(pcd)
            # vis.poll_events()
            # vis.update_renderer()
            # vis.run()

            # clean up the visualizer

        # vis.destroy_window()
