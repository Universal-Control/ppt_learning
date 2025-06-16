from typing import Optional
import numpy as np
from collections import OrderedDict
from typing import List
import os
import gc
from tqdm import tqdm
import zarr
import copy

from ppt_learning.utils.replay_buffer import ReplayBuffer
from ppt_learning.paths import PPT_DIR


def create_indices(
    episode_ends: np.ndarray,
    episode_descriptions: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
    replay_idx: int | None = None,
    obs: bool = False,
) -> np.ndarray:
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    eps_description = episode_descriptions[0]  # only one description
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue

        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = (
                max(idx, 0) + start_idx
            )  # make sure all start idx is the same
            if obs:
                buffer_end_idx = (
                    min(idx + (pad_before + 1), episode_length) + start_idx
                )  # pad_before+1 because we want to include the history
            else:
                buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx

            start_offset = buffer_start_idx - (idx + start_idx)
            if obs:
                end_offset = (idx + (pad_before + 1) + start_idx) - buffer_end_idx
            else:
                end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset  # make sure all start idx is the same
            if obs:
                sample_end_idx = (
                    pad_before + 1
                ) - end_offset  # pad_before+1 because we want to include the  history
            else:
                sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                ), f"start_offset: {start_offset}, end_offset: {end_offset}, sample_end_idx: {sample_end_idx}, sample_start_idx: {sample_start_idx}, buffer_end_idx: {buffer_end_idx}, buffer_start_idx: {buffer_start_idx}"
            if replay_idx is not None:
                indices.append(
                    [
                        replay_idx,
                        i,
                        buffer_start_idx,
                        buffer_end_idx,
                        sample_start_idx,
                        sample_end_idx,
                        eps_description,
                    ]
                )
            else:
                indices.append(
                    [
                        i,
                        buffer_start_idx,
                        buffer_end_idx,
                        sample_start_idx,
                        sample_end_idx,
                        eps_description,
                    ]
                )
    indices = np.array(indices, dtype=object)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        ignored_keys=[],
        action_key="",
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert sequence_length >= 1
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        episode_descriptions = replay_buffer.episode_descriptions[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            act_indices = create_indices(
                episode_ends,
                episode_descriptions,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
            obs_indices = create_indices(
                episode_ends,
                episode_descriptions,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
                obs=True,
            )
        else:
            # raise ValueError("No episodes to sample from")
            obs_indices = np.zeros((0, 4), dtype=np.int64)
            act_indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.act_indices = act_indices
        self.obs_indices = obs_indices

        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.ignored_keys = ignored_keys
        self.action_key = action_key

    def __len__(self):
        return len(self.act_indices)

    def sample_sequence(self, idx):
        (
            obs_eps_i,
            obs_buffer_start_idx,
            obs_buffer_end_idx,
            obs_sample_start_idx,
            obs_sample_end_idx,
            obs_eps_description,
        ) = self.obs_indices[idx]

        (
            act_eps_i,
            act_buffer_start_idx,
            act_buffer_end_idx,
            act_sample_start_idx,
            act_sample_end_idx,
            act_eps_description,
        ) = self.act_indices[idx]

        result = dict()

        def recursive_sample(data, target_dict):

            for key, input_arr in data.items():
                if key in self.ignored_keys:
                    continue

                if isinstance(input_arr, (dict, OrderedDict, zarr.Group)):
                    target_dict[key] = {}
                    recursive_sample(input_arr, target_dict[key])
                else:
                    if key == self.action_key:
                        eps_i = act_eps_i
                        buffer_start_idx = act_buffer_start_idx
                        buffer_end_idx = act_buffer_end_idx
                        sample_start_idx = act_sample_start_idx
                        sample_end_idx = act_sample_end_idx
                        eps_description = act_eps_description
                    else:
                        eps_i = obs_eps_i
                        buffer_start_idx = obs_buffer_start_idx
                        buffer_end_idx = obs_buffer_end_idx
                        sample_start_idx = obs_sample_start_idx
                        sample_end_idx = obs_sample_end_idx
                        eps_description = obs_eps_description

                    # performance optimization, avoid small allocation if possible
                    if key not in self.key_first_k:
                        sample = input_arr[buffer_start_idx:buffer_end_idx]
                    else:
                        # performance optimization, only load used obs steps
                        n_data = buffer_end_idx - buffer_start_idx
                        k_data = min(self.key_first_k[key], n_data)
                        # fill value with Nan to catch bugs
                        # the non-loaded region should never be used
                        sample = np.full(
                            (n_data,) + input_arr.shape[1:],
                            fill_value=np.nan,
                            dtype=input_arr.dtype,
                        )
                        try:
                            sample[:k_data] = input_arr[
                                buffer_start_idx : buffer_start_idx + k_data
                            ]
                        except Exception as e:
                            import pdb

                            pdb.set_trace()

                    data = sample
                    if (sample_start_idx > 0) or (
                        sample_end_idx < self.sequence_length
                    ):
                        data = np.zeros(
                            shape=(self.sequence_length,) + input_arr.shape[1:],
                            dtype=input_arr.dtype,
                        )
                        if sample_start_idx > 0:
                            data[:sample_start_idx] = sample[0]
                        if sample_end_idx < self.sequence_length:
                            data[sample_end_idx:] = sample[-1]
                        data[sample_start_idx:sample_end_idx] = sample
                    target_dict[key] = data

        recursive_sample(self.replay_buffer, result)

        result["language"] = obs_eps_description
        result["obs_is_pad"] = np.zeros((self.sequence_length, 1), dtype=np.float32)
        result["action_is_pad"] = np.zeros((self.sequence_length, 1), dtype=np.float32)

        if act_sample_start_idx > 0:
            result["obs_is_pad"][:obs_sample_start_idx] = True
            result["action_is_pad"][:act_sample_start_idx] = True

        if act_sample_end_idx < self.sequence_length:
            result["obs_is_pad"][obs_sample_end_idx:] = True
            result["action_is_pad"][act_sample_end_idx:] = True

        return result


def check_key(key, ignored_keys):
    for ignored_key in ignored_keys:
        if ignored_key in key:
            return False
    return True


def get_shape(keys, shape_meta):
    shape = shape_meta
    key_lst = keys.split("/")
    for key in key_lst:
        shape = shape[key]
    return shape


def add_to_dict(d, keys, data):
    key_lst = keys.split("/")
    for key in key_lst[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[key_lst[-1]] = data


class SequenceSamplerLance:
    def __init__(
        self,
        lance_data_path: str,
        sequence_length: int,
        lance_dataset=None,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        keys_in_memory: list = None,
        episode_mask: Optional[np.ndarray] = None,
        ignored_keys=[],
        action_key="",
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """
        import lance
        import pickle

        super().__init__()
        assert sequence_length >= 1
        if lance_dataset is not None:
            self.lance_dataset = lance_dataset
        else:
            self.lance_dataset = lance.dataset(
                os.path.join(lance_data_path, "data.lance")
            )
        with open(os.path.join(lance_data_path, "meta_info.pkl"), "rb") as f:
            self.meta_info = pickle.load(f)
        if keys is None:
            keys = self.lance_dataset.schema.names

        episode_ends = self.meta_info["meta"]["episode_ends"]
        episode_descriptions = self.meta_info["meta"]["episode_descriptions"]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            obs_indices = create_indices(
                episode_ends,
                episode_descriptions,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
                obs=True,
            )
            act_indices = create_indices(
                episode_ends,
                episode_descriptions,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            # raise ValueError("No episodes to sample from")
            obs_indices = np.zeros((0, 4), dtype=np.int64)
            act_indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.obs_indices = obs_indices
        self.act_indices = act_indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        if keys_in_memory is None:
            keys_in_memory = []
        self.keys_in_memory = keys_in_memory
        self.ignored_keys = ignored_keys
        new_keys = []
        raw_shape_meta = self.meta_info["shape_meta"]
        self.shape_meta = {}
        for key in keys:
            if key == "row_id":
                continue
            if check_key(key, ignored_keys):
                new_keys.append(key)
                self.shape_meta[key] = get_shape(key, raw_shape_meta)
        self.keys_in_disk = [
            key
            for key in self.keys
            if (key not in self.keys_in_memory)
            and check_key(key, ignored_keys)
            and key != "row_id"
        ]
        self.load_key_in_memory()
        self.keys = new_keys
        self.action_key = action_key

    def __len__(self):
        return len(self.act_indices)

    def load_key_in_memory(self):
        tmp_data_in_memory = self.lance_dataset.to_table(columns=self.keys_in_memory)
        self.data_in_memory = {}
        for key in self.keys_in_memory:
            self.data_in_memory[key] = np.stack(
                tmp_data_in_memory[key].to_numpy()
            ).reshape(*self.shape_meta[key])

    def sample_sequence(self, idx):
        (
            obs_eps_i,
            obs_buffer_start_idx,
            obs_buffer_end_idx,
            obs_sample_start_idx,
            obs_sample_end_idx,
            obs_eps_description,
        ) = self.obs_indices[idx]

        (
            act_eps_i,
            act_buffer_start_idx,
            act_buffer_end_idx,
            act_sample_start_idx,
            act_sample_end_idx,
            act_eps_description,
        ) = self.act_indices[idx]
        result = dict()

        for key in self.keys_in_disk:
            if key == self.action_key:
                eps_i = act_eps_i
                buffer_start_idx = act_buffer_start_idx
                buffer_end_idx = act_buffer_end_idx
                sample_start_idx = act_sample_start_idx
                sample_end_idx = act_sample_end_idx
                eps_description = act_eps_description
            else:
                eps_i = obs_eps_i
                buffer_start_idx = obs_buffer_start_idx
                buffer_end_idx = obs_buffer_end_idx
                sample_start_idx = obs_sample_start_idx
                sample_end_idx = obs_sample_end_idx
                eps_description = obs_eps_description

            indices = list(range(buffer_start_idx, buffer_end_idx))
            row = self.lance_dataset.take(indices, columns=self.keys_in_disk)
            value = np.stack(row[key].to_numpy()).reshape(-1, *self.shape_meta[key][1:])
            data = np.zeros(
                shape=(self.sequence_length,) + value.shape[1:],
                dtype=value.dtype,
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = value[0]
            if sample_end_idx < self.sequence_length:
                data[sample_end_idx:] = value[-1]
            data[sample_start_idx:sample_end_idx] = value
            add_to_dict(result, key, data)
        for key in self.keys_in_memory:
            if key == self.action_key:
                eps_i = act_eps_i
                buffer_start_idx = act_buffer_start_idx
                buffer_end_idx = act_buffer_end_idx
                sample_start_idx = act_sample_start_idx
                sample_end_idx = act_sample_end_idx
                eps_description = act_eps_description
            else:
                eps_i = obs_eps_i
                buffer_start_idx = obs_buffer_start_idx
                buffer_end_idx = obs_buffer_end_idx
                sample_start_idx = obs_sample_start_idx
                sample_end_idx = obs_sample_end_idx
                eps_description = obs_eps_description
            value = self.data_in_memory[key][indices]
            data = np.zeros(
                shape=(self.sequence_length,) + value.shape[1:],
                dtype=value.dtype,
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = value[0]
            if sample_end_idx < self.sequence_length:
                data[sample_end_idx:] = value[-1]
            data[sample_start_idx:sample_end_idx] = value
            add_to_dict(result, key, data)

        result["language"] = obs_eps_description
        result["obs_is_pad"] = np.zeros((self.sequence_length, 1), dtype=np.float32)
        result["action_is_pad"] = np.zeros((self.sequence_length, 1), dtype=np.float32)

        if act_sample_start_idx > 0:
            result["obs_is_pad"][:obs_sample_start_idx] = True
            result["action_is_pad"][:act_sample_start_idx] = True

        if act_sample_end_idx < self.sequence_length:
            result["obs_is_pad"][obs_sample_end_idx:] = True
            result["action_is_pad"][act_sample_end_idx:] = True

        return result
