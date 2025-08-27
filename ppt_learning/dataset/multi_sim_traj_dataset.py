import numpy as np
from collections import OrderedDict
import copy
import os
import zarr
import ipdb
import albumentations as A
import cv2
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ppt_learning.utils.replay_buffer import ReplayBuffer
import torch
from ppt_learning.utils.sampler import SequenceSampler, get_val_mask
from ppt_learning.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ppt_learning.utils.pcd_utils import (
    add_gaussian_noise,
    randomly_drop_point,
    voxelize_point_cloud,
)

from ppt_learning.paths import *
from ppt_learning.utils.pcd_utils import BOUND
from ppt_learning.dataset.sim_traj_dataset import resize_image_sequence, clip_depth, WarpMinMax

class MultiTrajDataset:
    """Multiple trajectory dataset for handling multi-domain simulation data.
    
    This class manages multiple single trajectory datasets, enabling multi-task
    learning across different domains. It handles data loading, normalization,
    augmentation, and sampling for training multi-domain policies.
    
    Key Features:
        - Multi-domain data loading and management
        - Point cloud generation from RGB-D images
        - Data augmentation for images, depth, and point clouds
        - Flexible state/action normalization
        - Support for distributed training
    """

    def __init__(
        self,
        domain: Union[str, List[str]],
        dataset_path: Union[str, List[str]] = "",
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
        pose_transform: Optional[Union[str, Callable]] = None,
        norm_depth: bool = False,
        **kwargs: Any,
    ) -> None:
        self.dataset_name = domain if isinstance(domain, list) else [domain]
        self.rank = rank
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
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

        # Validate horizon configuration
        if not hist_action_cond:
            expected_horizon = self.observation_horizon + self.action_horizon - 1
            if self.horizon != expected_horizon:
                raise ValueError(
                    f"Horizon mismatch: got {self.horizon}, expected {expected_horizon} "
                    f"(observation_horizon={self.observation_horizon} + action_horizon={self.action_horizon} - 1)"
                )

        if augment_img:
            self.img_transform = A.Compose(
                [
                    A.OneOf(
                        [
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
                        ],
                    ),
                ]
            )
            
        if augment_depth:
            # random translate / random affine
            self.depth_transform = A.Compose(
                [
                    A.OneOf(
                        [
                            A.ShiftScaleRotate(
                                shift_limit=0.15,
                                scale_limit=0.1,
                                rotate_limit=1,
                                p=img_augment_prob,
                            ),
                        ],
                    ),
                ]
            )

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
                raise NotImplementedError(f"Pose transform '{pose_transform}' not implemented")
        
        self.update_pcd_transform()

        # Ensure dataset_path is a list
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        if isinstance(domain, str):
            domain = [domain] * len(dataset_path)
        
        if len(domain) != len(dataset_path):
            raise ValueError(f"Domain list length ({len(domain)}) must match dataset_path length ({len(dataset_path)})")
        
        self.dataset_path = dataset_path
        self.replay_buffer: List[Optional[ReplayBuffer]] = [None] * len(dataset_path) 

        # Initialize replay buffers for each dataset
        for idx, single_dpath in enumerate(dataset_path):
            # Check if dataset exists and cache loading is enabled
            load_from_cache = os.path.exists(single_dpath) and load_from_cache
            if not os.path.exists(single_dpath) and not from_empty:
                raise FileNotFoundError(f"Dataset path not found: {single_dpath}")
            logger.info(
                f"Loading dataset {idx+1}/{len(dataset_path)}: {single_dpath} "
                f"(load_from_cache={load_from_cache})"
            )

            if use_disk:
                # self.replay_buffer[idx] = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(path=dataset_path))
                if load_from_cache:
                    if use_lru_cache:
                        store = zarr.DirectoryStore(single_dpath)
                        cache = zarr.LRUStoreCache(store=store, max_size=2**(39-len(dataset_path)))
                        group = zarr.open(cache, "r")
                        self.replay_buffer[idx] = ReplayBuffer.create_from_group(
                            group,
                        )
                        logger.info(f"Using LRU cache for dataset {idx+1} with max_size={2**(39-len(dataset_path))}")
                    else:
                        self.replay_buffer[idx] = ReplayBuffer.create_from_group(
                            zarr.open(zarr.DirectoryStore(single_dpath), "r"),
                        )
                else:
                    self.replay_buffer[idx] = ReplayBuffer.create_empty_zarr(
                        storage=zarr.DirectoryStore(path=single_dpath)
                    )
            else:
                self.replay_buffer[idx] = ReplayBuffer.create_empty_numpy()

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
        """Calculate and set state and action dimensions from the first sample."""
        first_sample = self[0]
        self.action_dim = first_sample["data"]["action"].shape[-1]
        self.state_dim = first_sample["data"]["state"].shape[-1]
        logger.info(f"Set action_dim={self.action_dim}, state_dim={self.state_dim}")

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        """Get normalizer fitted on combined data from all datasets.
        
        Args:
            mode: Normalization mode (e.g., 'limits', 'gaussian')
            **kwargs: Additional keyword arguments for normalization
            
        Returns:
            Fitted LinearNormalizer instance for multi-domain data
        """
        data = {}
        for idx in range(len(self.replay_buffer)):
            tmp = self._sample_to_data(self.replay_buffer[idx])
            for k, v in tmp.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)
        self.normalizer = LinearNormalizer()
        self.normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # Log normalizer statistics
        for k, v in self.normalizer.params_dict.items():
            logger.info(f"Normalizer {k} - min: {v['input_stats'].min}, max: {v['input_stats'].max}")
        return self.normalizer

    def get_episode(self, idx: int) -> Dict[str, Any]:
        """Get an episode by index across all datasets.
        
        Args:
            idx: Episode index
            
        Returns:
            Episode data dictionary
        """
        dataset_idx = np.searchsorted(self.dataset_episodes, idx, side="right") - 1
        idx = idx - self.dataset_episodes[dataset_idx]
        return self.replay_buffer[dataset_idx].get_episode(idx)

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

    def get_training_dataset(self, val_ratio: float, seed: int) -> None:
        """Initialize training dataset with train/validation split.
        
        Args:
            val_ratio: Ratio of data to use for validation
            seed: Random seed for reproducible splits
        """
        self.val_mask = [None] * len(self.replay_buffer)
        self.train_mask = [None] * len(self.replay_buffer)
        self.sampler = [None] * len(self.replay_buffer)
        for idx, replay_buffer in enumerate(self.replay_buffer):
            # split into train and test sets
            self.val_mask[idx] = get_val_mask(
                n_episodes=self.replay_buffer[idx].n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
            self.train_mask[idx] = ~self.val_mask[idx]

            # considering hyperparameters and masking
            n_episodes = int(
                self.data_ratio
                * min(self.episode_cnt, self.replay_buffer[idx].n_episodes)
            )
            self.val_mask[idx][n_episodes:] = False
            self.train_mask[idx][n_episodes:] = False

            # normalize and create sampler
            self.sampler[idx] = SequenceSampler(
                replay_buffer=self.replay_buffer[idx],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.train_mask[idx],
                ignored_keys=self.ignored_keys,
                action_key=self.action_key,
            )
            # logger.info(
            #     f"Dataset {self.dataset_name[0]}: {len(self.sampler[idx])} samples, "
            #     f"{n_episodes} episodes ({self.train_mask[idx].sum()} train, {self.val_mask[idx].sum()} val)"
            # )

        # Calculate cumulative dataset lengths for indexing
        self.dataset_length = np.cumsum([len(sampler) for sampler in self.sampler])
        self.dataset_length = np.insert(self.dataset_length, 0, 0)
        
        # Calculate cumulative episode counts
        episode_counts = [self.replay_buffer[idx].n_episodes for idx in range(len(self.replay_buffer))]
        self.dataset_episodes = np.cumsum([0] + episode_counts)

    def get_validation_dataset(self) -> List['MultiTrajDataset']:
        """Create validation dataset instances.
        
        Returns:
            List of validation dataset instances for each domain
        """
        val_set = [None] * len(self.replay_buffer)
        for idx, replay_buffer in enumerate(self.replay_buffer):
            val_set[idx] = copy.copy(self)
            val_set[idx].mode = "val"
            val_set[idx].update_pcd_transform()
            val_set[idx].sampler = SequenceSampler(
                replay_buffer=self.replay_buffer[idx],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.val_mask[idx],
            )
            val_set[idx].train_mask = self.val_mask[idx]
        return val_set

    def __len__(self) -> int:
        """Get total number of samples across all datasets."""
        if not hasattr(self, 'sampler') or not self.sampler:
            return 0
        return sum([len(sampler) for sampler in self.sampler])
    
    @property
    def num_datasets(self) -> int:
        """Get number of datasets being managed."""
        return len(self.replay_buffer)
    
    @property
    def total_episodes(self) -> int:
        """Get total number of episodes across all datasets."""
        if not self.replay_buffer or not all(self.replay_buffer):
            return 0
        return sum(rb.n_episodes for rb in self.replay_buffer if rb is not None)
    
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get normalized observation and actions for given index.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary with 'domain' and 'data' keys containing processed sample
        
        Raises:
            KeyError: If required action keys are not found in the sample
        """
        dataset_idx = np.searchsorted(self.dataset_length, idx, side="right") - 1
        idx = idx - self.dataset_length[dataset_idx]
        sample = self.sampler[dataset_idx].sample_sequence(idx)
        
        action_sub_keys = self.action_key.split('/')
        if len(action_sub_keys) > 1 or action_sub_keys[0] != "action":
            action = sample
            for key in action_sub_keys:
                if isinstance(action, (dict, OrderedDict)):
                    try:
                        action = action[key]
                    except KeyError as e:
                        logger.error(f"Action key not found: {key}")
                        logger.error(f"Available keys: {list(action.keys())}")
                        raise KeyError(f"Failed to access action key '{key}' in sample") from e
            sample["action"] = action
            del sample[action_sub_keys[0]]

        # Process horizon for the trajectory
        def recursive_horizon(data: Dict[str, Any]) -> None:
            """Recursively apply horizon windowing to nested data."""
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
        self.transform(sample)

        return {"domain": self.dataset_name, "data": sample}

    def load_dataset(self) -> None:
        """Load all datasets from disk into replay buffers."""
        for idx, dataset_path in enumerate(self.dataset_path):
            self.replay_buffer[idx] = ReplayBuffer.copy_from_path(dataset_path)
            logger.info(f"Dataset {idx+1} replay buffer keys: {self.replay_buffer[idx].keys()}")

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

    def transform(self, sample: Dict[str, Any]) -> None:
        """Apply transformations to sample data in-place.
        
        Includes image resizing, depth normalization, and augmentations.
        
        Args:
            sample: Sample dictionary to transform
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


def main() -> None:
    """Test the MultiTrajDataset class with sample data."""
    import collections

    dataset = MultiTrajDataset(
        domain="debug",
        dataset_path="/mnt/bn/robot-minghuan-datasets-lq/xiaoshen/datasets/ur5_put_bowl_in_microwave_and_close/put_bowl_in_microwave__520_collected_data_retry_random_x015_new_subtask_generated_1gpu.zarr",
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
        # action_key="wbc_target/r"
    )
    # dataset.load_dataset()
    print(collections.Counter(dataset.replay_buffer.meta["episode_descriptions"]))
    if "env_names" in dataset.replay_buffer.meta.keys():
        print(collections.Counter(dataset.replay_buffer.meta["env_names"]))

    import ipdb

    # Debug breakpoint - uncomment if needed
    # ipdb.set_trace()

    from gensim2.env.utils.rlbench import plot_pred

    for i in range(15, 25):
        data = dataset.replay_buffer
        pcds = data["obs"]["pointcloud"]["pos"][i]
        rgbs = data["obs"]["pointcloud"]["colors"][i]
        action = data["action"][i]
        plot_pred(np.array([action]), pcds, rgbs, ".")


if __name__ == "__main__":
    main()
