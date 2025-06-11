import numpy as np
from collections import OrderedDict
import copy
import os
import zarr
import ipdb
import lance
import pickle
import time
import cv2

from ppt_learning.utils.sampler import SequenceSamplerLance, get_val_mask, get_shape
from ppt_learning.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ppt_learning.utils.pcd_utils import (
    add_gaussian_noise,
    randomly_drop_point,
    voxelize_point_cloud,
)

from ppt_learning.paths import *
from ppt_learning.utils.pcd_utils import BOUND
from ppt_learning.dataset.sim_traj_dataset import resize_image_sequence, clip_depth

class MultiTrajDatasetLance:
    """
    Multiple single Dataset class that converts simulation data into trajectory data.
    Explanations of parameters are in config
    """

    def __init__(
        self,
        domain,
        dataset_path="",
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
        hist_action_cond=False,
        resize_img=True,
        img_size=(224,224),
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
        voxelization=False,
        voxel_size=0.01,
        use_lru_cache=True,
        rank=0,  # Add rank for DDP
        ignored_keys=None,
        state_keys=None,
        action_key="actions",
        pose_transform=None,
        keys_in_memory=None,
        **kwargs,
    ):
        self.dataset_name = domain
        self.rank = rank
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_augmentation = data_augmentation
        self.se3_augmentation = se3_augmentation
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
        if use_pcd:
            assert pcd_channels is not None, "pcd_channels must be provided for pcd"
        if pcd_channels is not None:
            assert pcd_channels in [
                3,
                4,
                5,
                6,
                7,
            ], "pcd_channels must be one of [3, 4, 6, 7]"
        self.pcd_channels = pcd_channels
        self.mode = mode
        self.use_pcd = use_pcd
        self.pcd_transform = None
        self.pcdnet_pretrain_domain = pcdnet_pretrain_domain
        self.pcd_num_points = None
        self.bounds = BOUND

        if not hist_action_cond:
            assert self.horizon == self.observation_horizon + self.action_horizon - 1, "Check if your horizon is right"

        self.state_keys = state_keys
        if state_keys is None:
            self.state_keys = [
                "eef_pos",
                "eef_quat",
                "joint_pos",
                "joint_vel",
                "normalized_gripper_pos"
            ]
        self.ignored_keys = ignored_keys
        if ignored_keys is None:
            self.ignored_keys = [
                "initial_state",
                "states",
                "depths",
                "images",
                "pointcloud/color",
                "wbc_step",
                "wbc_target",
                "obs/abs_gripper_pos",
                'obs/last_action',
            ] # , "images", "color"]
            # self.use_pcd = False
        if keys_in_memory is None:
            self.keys_in_memory = [
                'obs/eef_pos',
                'obs/eef_quat',
                'obs/joint_pos', 
                'obs/joint_vel',
                'obs/normalized_gripper_pos',
                "actions"
            ]

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

        self.dataset_path = dataset_path
        self.lance_datasets = [None] * len(dataset_path)
        self.shape_metas = [{}] * len(dataset_path)
        self.state_keys_flat = None
        self.meta_info = None
        for i, single_dataset_path in enumerate(dataset_path):
            self.lance_datasets[i] = lance.dataset(os.path.join(single_dataset_path, "data.lance"))
            with open(os.path.join(single_dataset_path, "meta_info.pkl"), "rb") as f:
                meta_info_this = pickle.load(f)
                if self.meta_info is None:
                    self.meta_info = meta_info_this["meta"]
                raw_shape_meta = meta_info_this["shape_meta"]
            all_keys = self.lance_datasets[i].schema.names
            if self.state_keys_flat is None:
                self.state_keys_flat = ["obs/" + key for key in self.state_keys]
            for key in all_keys:
                if key == "row_id":
                    continue
                self.shape_metas[i][key] = get_shape(key, raw_shape_meta)
        
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
        data = {}
        for idx in range(len(self.lance_datasets)):
            tmp = self._sample_to_data(self.lance_datasets[idx], self.shape_metas[idx])
            for k, v in tmp.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)
        self.normalizer = LinearNormalizer()
        self.normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for k, v in self.normalizer.params_dict.items():
            print(f"normalizer {k} stats min: {v['input_stats'].min}")
            print(f"normalizer {k} stats max: {v['input_stats'].max}")
        return self.normalizer

    def get_episode(self, idx):
        dataset_idx = np.searchsorted(self.dataset_episodes, idx, side="right") - 1
        idx = idx - self.dataset_episodes[dataset_idx]
        return self.sampler[dataset_idx].get_episode(idx)

    def _sample_to_data(self, lance_dataset, shape_meta):
        keys = [self.action_key]
        keys.extend(self.state_keys_flat)
        table = lance_dataset.to_table(columns=keys)
        data = {"action": np.stack(table[self.action_key].to_numpy()).reshape(*shape_meta[self.action_key])}
        keys.pop(0)
        if self.normalize_state:
            sample = {}
            for key in keys:
                sample[key[4:]] = np.stack(table[key].to_numpy()).reshape(*shape_meta[key])
            data["state"] = self.get_state(sample)
        return data

    def get_training_dataset(self, val_ratio, seed):
        self.val_mask = [None] * len(self.lance_datasets)
        self.train_mask = [None] * len(self.lance_datasets)
        self.sampler = [None] * len(self.lance_datasets)
        for idx, lance_dataset in enumerate(self.lance_datasets):
            # split into train and test sets
            self.val_mask[idx] = get_val_mask(
                n_episodes=self.lance_datasets[idx].count_rows(),
                val_ratio=val_ratio,
                seed=seed,
            )
            self.train_mask[idx] = ~self.val_mask[idx]

            # considering hyperparameters and masking
            n_samples = int(
                self.data_ratio
                * min(self.episode_cnt, self.lance_datasets[idx].count_rows())
            )
            self.val_mask[idx][n_samples:] = False
            self.train_mask[idx][n_samples:] = False

            # normalize and create sampler
            self.sampler[idx] = SequenceSamplerLance(
                lance_data_path=self.dataset_path[idx],
                lance_dataset=self.lance_datasets[idx],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.train_mask[idx],
                ignored_keys=self.ignored_keys,
                keys_in_memory=self.keys_in_memory,
            )
            print(
                f"{self.dataset_name[idx]} size: {len(self.sampler[idx])} episodes: {n_samples} train: {self.train_mask[idx].sum()} eval: {self.val_mask[idx].sum()}"
            )

        self.dataset_length = np.cumsum([len(sampler) for sampler in self.sampler])
        self.dataset_length = np.insert(self.dataset_length, 0, 0)
        self.dataset_episodes = [0] + [self.lance_datasets[idx].count_rows() for idx in range(len(self.lance_datasets))]

    def get_validation_dataset(self):
        val_set = [None] * len(self.lance_datasets)
        for idx in range(len(self.lance_datasets)):
            val_set[idx] = copy.copy(self)
            val_set[idx].mode = "val"
            val_set[idx].update_pcd_transform()
            val_set[idx].sampler = SequenceSamplerLance(
                lance_data_path=self.dataset_path[idx],
                lance_dataset=self.lance_datasets[idx],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.val_mask[idx],
                ignored_keys=self.ignored_keys,
                keys_in_memory=self.keys_in_memory,
            )
            val_set[idx].train_mask = self.val_mask[idx]
        return val_set

    def __len__(self) -> int:
        return sum([len(sampler) for sampler in self.sampler])

    def __getitem__(self, idx: int):
        """normalize observation and actions"""
        dataset_idx = np.searchsorted(self.dataset_length, idx, side="right") - 1
        idx = idx - self.dataset_length[dataset_idx]
        sample = self.sampler[dataset_idx].sample_sequence(idx)
        
        action_sub_keys = self.action_key.split('/')
        action = sample
        for key in action_sub_keys:
            if isinstance(action, (dict, OrderedDict)):
                try:
                    action = action[key]
                except:
                    print("Action key not found:", key)
                    import ipdb; ipdb.set_trace()
        sample["action"] = action
        del sample[action_sub_keys[0]]

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

        # if "pointcloud" in sample.keys():
        #     assert sample["pointcloud"].shape[-1] == self.pcd_channels, f"pointcloud channel mismatch! expected {self.pcd_channels}, got {sample['pointcloud'].shape[-1]}"
        recursive_horizon(sample)

        self.transform(sample)

        return {"domain": self.dataset_name, "data": sample}


    def get_state(self, sample,):
        res = {"state": []}
        for state_key in self.state_keys:
            state_key_lst = state_key.split("/")
            state = sample[state_key]
            res_state_dict = res["state"]
            for key in state_key_lst[:-1]:
                if key not in res_state_dict.keys():
                    res_state_dict[key] = {}
                res_state_dict = res_state_dict[key]
            res_state_dict.append(state)
        res["state"] = np.concatenate(res["state"], axis=-1)
        return res["state"]

    def transform(self, sample):
        if self.resize_img and "image" in sample.keys():
            for key, val in sample["image"].items():
                # Image shape N, H, W, C
                sample["image"][key] = resize_image_sequence(val, (self.img_size[0], self.img_size[1]))
        if self.resize_img and "depth" in sample.keys():
            for key, val in sample["depth"].items():
                # Image shape N, H, W, C
                sample["depth"][key] = resize_image_sequence(clip_depth(val), (self.img_size[0], self.img_size[1]), interp=cv2.INTER_NEAREST)
        if self.pose_transform is not None: # Last dim is gripper
            if len(sample['action'].shape) == 2:
                N, A = sample['action'].shape
                sample['action'] = np.concatenate([self.pose_transform(sample['action'][..., :-1].reshape(-1, A-1)).reshape(N, -1), sample['action'][..., -1:]], axis=-1)
            elif len(sample['action'].shape) == 3:
                N, L, A = sample['action'].shape
                sample['action'] = np.concatenate([self.pose_transform(sample['action'][..., :-1].reshape(-1, A-1)).reshape(N, L, -1), sample['action'][..., -1:]], axis=-1)
            else:
                raise ValueError(f"Invalid action shape: {sample['action'].shape}")

    def flat_sample(self, sample):
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

    def pcd_aug(self, pcd):
        pcd = add_gaussian_noise(pcd)
        pcd = randomly_drop_point(pcd)

        return pcd


if __name__ == "__main__":
    import collections

    dataset = MultiTrajDatasetLance(
        domain="debug",
        dataset_path=["/mnt/bn/robot-minghuan-datasets-lq/xiaoshen/datasets/ur5_put_bowl_in_microwave_and_close/one_camera_no_crop_642_no_yaw_lance"],
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
    import ipdb; ipdb.set_trace()

    dataset.get_training_dataset(0.0, 1)

    for i in range(15, 25):
        data = dataset[i]
        print(i)