#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from collections import OrderedDict, deque

from functools import partial
from types import SimpleNamespace
from typing import Dict, Optional, Any, Union, Tuple, List
import hydra
from omegaconf import DictConfig

import torch
from torch import optim
import torch.nn as nn
import numpy as np

from ppt_learning.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ppt_learning.utils.learning import batchify, unbatchify
from ppt_learning.models.transformer import (
    MultiheadAttention,
    SimpleTransformer,
    CrossAttention,
)
from ppt_learning.utils.learning import (
    dict_apply,
    get_sinusoid_encoding_table,
    EinOpsRearrange,
    get_image_embeddings,
    normalize_image_numpy,
    recursive_get,
    recursive_in,
    sample_pcd_data,
)
from ppt_learning.constants import DEFAULT_K_VALUE
import IPython


def merge_act(
    actions_for_curr_step: torch.Tensor, t: int, k: float = DEFAULT_K_VALUE
) -> torch.Tensor:
    """Merge actions using exponential weighting based on recency.

    Args:
        actions_for_curr_step: Tensor of actions for current timestep, shape (N, action_dim)
        t: Current timestep (unused but kept for API compatibility)
        k: Exponential weight decay constant for temporal weighting

    Returns:
        Merged action tensor with exponential temporal weighting applied
    """
    actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]

    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    exp_weights = exp_weights / exp_weights.sum()
    exp_weights = (
        torch.from_numpy(exp_weights).to(actions_for_curr_step.device).unsqueeze(dim=1)
    )
    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

    return raw_action


class Policy(nn.Module):
    """The stem / trunk / head separation for each policy, which
    respectively consumes low-level, mid-level, high-level representations.
    Different from the the pretraining code, this class should support arbitrary
    heads and stems architecture."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        use_modality_embedding: bool = True,
        token_postprocessing: bool = False,
        observation_horizon: int = 4,
        action_horizon: int = 1,
        openloop_steps: int = 1,
        no_trunk: bool = False,
        temporal_agg: bool = False,
        max_timesteps: int = 1300,
        action_dim: int = 7,
        num_envs: int = 1,  # for parallel evaluation
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        if not no_trunk:
            self.trunk = self._create_policy_trunk(
                embed_dim, num_blocks, num_heads, **kwargs
            )
        self.stems = OrderedDict()
        self.heads = OrderedDict()
        self.normalizer = OrderedDict()  # normalizer
        self.domains = []
        self.use_modality_embedding = use_modality_embedding
        self.crossattn_modalities_latents = OrderedDict()
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.token_postprocessing = token_postprocessing
        self.no_trunk = no_trunk
        self.openloop_steps = openloop_steps
        self.temporal_agg = temporal_agg
        self.max_timesteps = max_timesteps
        self.action_dim = action_dim
        self.train_mode = False  # whether in training mode

        self.num_envs = num_envs

        if (not self.no_trunk) and self.use_modality_embedding:
            self.modalities_tokens = OrderedDict()

    def init_domain_stem(self, domain_name: str, stem_spec: DictConfig) -> None:
        """Initialize an observation stem for each domain.

        Args:
            domain_name: Name of the domain to initialize
            stem_spec: Configuration specification for the stem architecture
        """
        self.stem_spec = stem_spec
        self.modalities = stem_spec.modalities

        def recursive_init_domain_stem(cur_name, modality, crossattn_latent, stem_spec):
            if "/" in modality:  # e.g., 'obs/pcd/pos'
                sub_modality = modality.split("/")[0]
                recursive_init_domain_stem(
                    cur_name + "_" + sub_modality,
                    "/".join(modality.split("/")[1:]),
                    getattr(crossattn_latent, sub_modality),
                    getattr(stem_spec, sub_modality),
                )
            else:
                self.stems[cur_name + "_" + modality] = hydra.utils.instantiate(
                    getattr(stem_spec, modality)
                )
                if self.stem_spec.cross_attention:
                    self.crossattn_modalities_latents[cur_name + "_" + modality] = (
                        nn.Parameter(
                            torch.zeros(
                                1,
                                getattr(crossattn_latent, modality),
                                stem_spec.modality_embed_dim,
                            )  # 1, token_size, embed_dim
                        )
                    )

                    self.stems[cur_name + "_attend_" + modality] = CrossAttention(
                        self.embed_dim,
                        heads=self.stem_spec.num_heads,
                        dim_head=self.stem_spec.dim_head,
                    )  # query_dim
                if (not self.no_trunk) and self.use_modality_embedding:
                    self.modalities_tokens[modality] = nn.Parameter(
                        torch.zeros(1, 1, stem_spec.modality_embed_dim)
                    )

        for modality in self.modalities:
            recursive_init_domain_stem(
                domain_name, modality, self.stem_spec.crossattn_latent, self.stem_spec
            )

    def init_domain_head(
        self,
        domain_name: str,
        head_spec: DictConfig,
        normalizer: Optional[LinearNormalizer] = None,
    ) -> None:
        """Initialize an action head for each domain, along with normalizer.

        Args:
            domain_name: Name of the domain to initialize
            head_spec: Configuration specification for the head architecture
            normalizer: Optional pre-trained normalizer to load
        """
        self.head_spec = head_spec
        self.domains.append(domain_name)
        self.normalizer[domain_name] = LinearNormalizer()
        if normalizer is not None:
            self.normalizer[domain_name].load_state_dict(normalizer.state_dict())
        self.heads[domain_name] = hydra.utils.instantiate(head_spec)

    def finalize_modules(self) -> None:
        """Convert ordered dictionaries to PyTorch ModuleDict/ParameterDict for proper registration."""
        self.stems = nn.ModuleDict(self.stems)
        self.heads = nn.ModuleDict(self.heads)
        self.normalizer = nn.ModuleDict(self.normalizer)
        if self.stem_spec.cross_attention:
            self.crossattn_modalities_latents = nn.ParameterDict(
                self.crossattn_modalities_latents
            )

        if (not self.no_trunk) and self.use_modality_embedding:
            self.modalities_tokens = nn.ParameterDict(self.modalities_tokens)

    def _create_policy_trunk(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        drop_path: float = 0.0,
        weight_init_style: str = "pytorch",
        **kwargs: Any,
    ) -> SimpleTransformer:
        # follow ImageBind
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    (
                        nn.LayerNorm(embed_dim, eps=1e-6)
                        if pre_transformer_ln
                        else nn.Identity()
                    ),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
            )

        # only one trunk. need to attach tokens to each domain and modality
        trunk = instantiate_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=drop_path,
        )

        return trunk

    def process_position_embedding(self, feature):
        """add positional embedding to the features"""
        positional_embedding = get_sinusoid_encoding_table(
            0, feature.shape[1], self.embed_dim
        )
        positional_embedding = positional_embedding.repeat((feature.shape[0], 1, 1)).to(
            feature.device
        )
        feature = feature + positional_embedding
        return feature

    def process_time_embedding(self, feature):
        """add time embedding to the features"""
        time_embedding = get_sinusoid_encoding_table(
            0, feature.shape[1], self.embed_dim
        )
        time_embedding = time_embedding.repeat((feature.shape[0], 1, 1)).to(
            feature.device
        )
        feature = feature + time_embedding.unsqueeze(2)
        return feature

    def preprocess_tokens(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate tokens and add modality tokens. Add positional and time embeddings.

        Args:
            features: List of feature tensors from different modalities

        Returns:
            Concatenated and processed feature tensor with shape (B, L, D)
            where B=batch, L=total sequence length, D=embedding dimension
        """
        processed_features = []
        for idx, (modality, feature) in enumerate(zip(self.modalities, features)):
            if (not self.no_trunk) and self.use_modality_embedding:
                modality_embedding = self.modalities_tokens[modality].repeat(
                    (*feature.shape[:-1], 1)
                )
                feature = feature + modality_embedding
                # B x T x L x D
                feature = self.process_time_embedding(feature)  # denote timesteps
            feature = feature.reshape(feature.shape[0], -1, feature.shape[-1])
            processed_features.append(feature)

        tokens = torch.cat(processed_features, dim=-2)
        if not self.no_trunk:
            tokens = self.process_position_embedding(
                tokens
            )  # global position in the sequence

        return tokens

    def postprocess_tokens(self, trunk_tokens, feats):
        """
        avg pooling as the final features
        N x L x D -> N X D
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "max":
            return trunk_tokens.max(dim=1)
        elif self.token_postprocessing == "concat":
            return trunk_tokens.reshape(trunk_tokens.shape[0], -1)
        elif (
            self.token_postprocessing == "modal_mean"
        ):  # mean pooling for each modality
            res = []
            cur_id = 0
            for data in feats:
                length = data.shape[1]
                res.append(trunk_tokens[:, cur_id : cur_id + length].mean(dim=1))
                cur_id += length
            return torch.cat(res, dim=1)
        else:
            return trunk_tokens  # No postprocessing

    def preprocess_actions(self, domain: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and normalize action data if configured.

        Args:
            domain: Domain name for normalization
            data: Data dictionary potentially containing actions

        Returns:
            Data dictionary with normalized actions (if applicable)
        """
        if (
            self.head_spec.normalize_action
            and ("action" in data)
            and ("action" in self.normalizer[domain].params_dict)
        ):
            data["action"] = self.normalizer[domain]["action"].normalize(data["action"])

        return data

    def preprocess_states(self, domain: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and normalize state observations.

        Args:
            domain: Domain name for normalization
            data: Data dictionary containing state observations

        Returns:
            Data dictionary with normalized state observations
        """
        # normalize actions if needed
        if not self.stem_spec.cross_attention:
            return data

        if (
            self.stem_spec.normalize_state
            and ("state" in data)
            and ("state" in self.normalizer[domain].params_dict)
        ):
            data["state"] = self.normalizer[domain]["state"].normalize(data["state"])

        # (B * T) x 1 x D
        # state = data["state"][:, None]
        # TODO: Is this useful?
        # self.state_positional_embedding = get_sinusoid_encoding_table(
        #     0, data["state"].shape[-1], self.stem_spec.state_embedding_dim
        # ).to(state)
        # self.state_positional_embedding = self.state_positional_embedding.transpose(1, 2)[:, None]
        # state = state + self.state_positional_embedding
        # data["state"] = state
        return data

    def preprocess_images(self, domain, data):
        """pre-process image output"""
        # normalize actions if needed
        if not self.stem_spec.resize_image:
            return data

        raise NotImplementedError

    def postprocess_actions(self, domain, action):
        """postprocess output"""
        # unnormalize actions
        if self.head_spec.normalize_action and (not self.train_mode):
            action = self.normalizer[domain]["action"].unnormalize(action)
        return action

    def stem_process(self, domain, data):
        """pass through the MLP stem and then cross attention."""
        feats = []
        for modality in self.modalities:
            mode_name = domain + "_" + modality
            if "/" in modality:
                mode_name = domain + "_" + modality.replace("/", "_")
            stem = self.stems[mode_name]
            if not recursive_in(data, modality):
                continue

            stem_feat = stem(
                recursive_get(data, modality)
            )  # Should be B*T, L, D, L is the number of tokens
            if self.stem_spec.cross_attention:
                crossattn_param = self.crossattn_modalities_latents[
                    domain + "_" + modality
                ]
                crossattn_layer = self.stems[domain + "_attend_" + modality]
                stem_feat = stem_feat.reshape(
                    stem_feat.shape[0], -1, stem_feat.shape[-1]
                )
                crossattn_param = crossattn_param.repeat(len(stem_feat), 1, 1)
                stem_token = crossattn_layer(crossattn_param, stem_feat)

                if modality == "language":
                    stem_token = stem_token.reshape(
                        stem_token.shape[0], -1, *stem_token.shape[1:]
                    )
                else:
                    stem_token = stem_token.reshape(
                        stem_token.shape[0] // self.observation_horizon,
                        -1,
                        *stem_token.shape[1:],
                    )  # B, T, L, D
            else:
                stem_feat = stem_feat.reshape(
                    stem_feat.shape[0], -1, stem_feat.shape[-1]
                )
                if modality == "language":
                    stem_token = stem_feat.reshape(
                        stem_feat.shape[0], -1, *stem_feat.shape[1:]
                    )
                else:
                    stem_token = stem_feat.reshape(
                        stem_feat.shape[0] // self.observation_horizon,
                        -1,
                        *stem_feat.shape[1:],
                    )  # B, T, L, D
            feats.append(stem_token)
        return feats

    def reset(self, env_id=None):
        """reset the policy's history buffer"""
        if self.num_envs > 1:
            if env_id is None:
                self.history_buffer = OrderedDict(
                    {i: OrderedDict() for i in range(self.num_envs)}
                )
            else:
                self.history_buffer[env_id] = OrderedDict()
        else:
            self.history_buffer = OrderedDict()

        if self.temporal_agg:
            if self.num_envs > 1:
                if env_id is None:
                    self.all_time_actions = torch.zeros(
                        [
                            self.num_envs,
                            self.max_timesteps + self.action_horizon,
                            self.max_timesteps + self.action_horizon,
                            self.action_dim,
                        ],
                        device=next(self.parameters()).device,
                    )
                else:
                    self.all_time_actions[env_id] = torch.zeros(
                        [
                            self.max_timesteps + self.action_horizon,
                            self.max_timesteps + self.action_horizon,
                            self.action_dim,
                        ],
                        device=next(self.parameters()).device,
                    )
            else:
                self.all_time_actions = torch.zeros(
                    [
                        self.max_timesteps + self.action_horizon,
                        self.max_timesteps + self.action_horizon,
                        self.action_dim,
                    ],
                    device=next(self.parameters()).device,
                )
        else:
            if self.num_envs > 1:
                if env_id is None:
                    self.openloop_actions = [deque() for _ in range(self.num_envs)]
                else:
                    self.openloop_actions[env_id] = deque()
            else:
                self.openloop_actions = deque()

    def get_action(
        self,
        data,
        domain=None,
        pcd_npoints=8192,
        in_channels=4,
        task_description="",
        t=0,
        hist_action_cond=False,
    ):
        """Get action in the evaluation setup.
        data should be dictionary
        """
        self.train_mode = False
        if domain is None:
            domain = self.domains[0]

        # return self.postprocess_actions(domain, np.random.uniform(-1, 1, self.action_dim)[np.newaxis,:]).squeeze() # random action

        device = next(self.parameters()).device
        if not hasattr(self, "history_buffer"):
            print(
                "should call policy reset explicitly to avoid problems for evaluation in sequence."
            )
            self.reset()

        def recursive_append(val, key, buffer_dict):
            if isinstance(val, (dict, OrderedDict)):
                if key not in buffer_dict:
                    buffer_dict[key] = OrderedDict()
                for k, v in val.items():
                    recursive_append(v, k, buffer_dict[key])
            else:
                if key not in buffer_dict:
                    buffer_dict[key] = deque(maxlen=self.observation_horizon)
                    for _ in range(self.observation_horizon):
                        buffer_dict[key].append(val)
                else:
                    buffer_dict[key].append(val)

        def recursive_concatenate(buffer_dict):
            result = OrderedDict()
            for k, v in buffer_dict.items():
                if isinstance(v, (dict, OrderedDict)):
                    result[k] = recursive_concatenate(v)
                else:
                    result[k] = np.concatenate(list(v), axis=0)
            return result

        data_np = data
        if self.observation_horizon > 1:
            for key in data_np:
                recursive_append(data_np[key], key, self.history_buffer)
            data_np = recursive_concatenate(self.history_buffer)

        if not self.temporal_agg and len(self.openloop_actions) > 0:
            return self.openloop_actions.popleft()

        data_tensor = dict_apply(
            data_np, lambda x: torch.from_numpy(x).to(device, non_blocking=True).float()
        )
        if "pointcloud" in data_tensor:
            sample_pcd_data(
                data_tensor["pointcloud"], npoints=pcd_npoints, in_channels=in_channels
            )

        data_tensor["language"] = [task_description]
        action_th = self(domain, data_tensor)[0]
        self.policy_action = action_th

        output = self.policy_action.squeeze()
        if self.temporal_agg:
            if hist_action_cond:
                if t - self.observation_horizon + 1 < 0:
                    assert t - self.observation_horizon + 1 + self.action_horizon > 0
                    self.all_time_actions[
                        t, 0 : t - self.observation_horizon + 1 + self.action_horizon
                    ] = output[
                        output.shape[0]
                        - (t - self.observation_horizon + 1 + self.action_horizon) :
                    ]
                else:
                    self.all_time_actions[
                        t,
                        t
                        - self.observation_horizon
                        + 1 : t
                        - self.observation_horizon
                        + 1
                        + self.action_horizon,
                    ] = output
            else:
                self.all_time_actions[[t], t : t + self.action_horizon] = output
            output = merge_act(self.all_time_actions[:, t], t)

        action = output.detach().cpu().numpy()

        if self.temporal_agg:
            return action.squeeze()
        else:
            if hist_action_cond:
                assert (len(action.shape) == 2) and (
                    len(action[0]) >= self.observation_horizon
                ), "condition on history horizon but get too short prediction!"
                for a in action[
                    self.observation_horizon : self.observation_horizon
                    - 1
                    + self.openloop_steps
                ]:
                    self.openloop_actions.append(a)
                return action[self.observation_horizon - 1]
            else:
                if len(action.shape) > 1:
                    # for a in action[self.observation_horizon : self.observation_horizon-1+self.openloop_steps]:
                    for a in action[1 : self.openloop_steps]:
                        self.openloop_actions.append(a)
                return action[0]  # action w.r.t current obs

    def forward_train(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Training loop forward pass.

        Args:
            batch: Training batch containing domain and data information

        Returns:
            Action predictions for the training batch
        """
        self.train_mode = True
        domain = (
            batch["domain"][0]
            if isinstance(batch["domain"][0], str)
            else batch["domain"][0][0]
        )
        return self(domain, batch["data"])

    def forward(
        self, domain: str, data: Dict[str, Any], head_kwargs: Dict[str, Any] = {}
    ) -> torch.Tensor:
        """Main forward pass of the combined policy.

        Args:
            domain: Domain name for the current forward pass
            data: Dictionary containing input observations and actions
            head_kwargs: Additional keyword arguments for the head module

        Returns:
            Action tensor output from the policy head
        """
        # preprocess / normalization
        data = self.preprocess_states(domain, data)
        data = self.preprocess_actions(domain, data)

        # stem pass
        feats = self.stem_process(domain, data)

        for i in range(1, len(feats)):  # For debugging
            assert (
                feats[0].shape[-1] == feats[i].shape[-1]
            ), "embedding dimension mismatch for each feature."

        # combine tokens
        features = self.preprocess_tokens(feats)

        # trunk pass
        if not self.no_trunk:
            features = self.trunk(features)

        # pooling the features
        features = self.postprocess_tokens(features, feats)

        # head pass
        if self.train_mode:
            action = self.heads[domain](
                features,
                target=data["action"],
                action_is_pad=data.get("action_is_pad", None),
                **head_kwargs,
            )
        else:
            action = self.heads[domain](features, **head_kwargs)

        if isinstance(action, (dict, OrderedDict)):  # That should be losses
            return action

        # postprocess
        action = self.postprocess_actions(domain, action)
        return action.unsqueeze(1)

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        batch["data"] = batchify(batch["data"], exclude=["action"])
        output = self.forward_train(batch)
        loss = output["loss"]
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # return self.opt, self.scheduler

    def device(self):
        """get the current device of the model"""
        return next(self.parameters()).device

    def save(self, checkpoint_path: str = "./.checkpoints/full"):
        """save the trunk part of the model"""
        try:
            torch.save(self.state_dict(), checkpoint_path)
        except FileNotFoundError:
            logging.warning(
                f"Could not save module parameters for trunk to {checkpoint_path}."
            )

    def load(self, path):
        """load all parts of the model"""
        self.load_state_dict(torch.load(path))

    def load_trunk(self, path, postfix="_last", extension="pth"):
        """load the trunk part of the model"""
        assert not self.no_trunk, "Cannot load trunk when no_trunk is True."
        self.trunk.load_state_dict(torch.load(path))

    def freeze_trunk(self):
        assert not self.no_trunk, "Cannot freeze trunk when no_trunk is True."
        for param in self.trunk.parameters():
            param.requires_grad = False

    def unfreeze_trunk(self):
        assert not self.no_trunk, "Cannot unfreeze trunk when no_trunk is True."
        for param in self.trunk.parameters():
            param.requires_grad = True

    def freeze_stems(self):
        for param in self.stems.parameters():
            param.requires_grad = False

    def unfreeze_stems(self):
        for param in self.stems.parameters():
            param.requires_grad = True
