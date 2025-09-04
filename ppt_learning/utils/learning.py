import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import numpy as np
import json


import torch
from omegaconf import OmegaConf, DictConfig
import os
from PIL import Image
from types import SimpleNamespace
import numpy as np
from collections import OrderedDict
import einops

import gc
import hydra

from transformers import T5Tokenizer, T5Model
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextConfig
from transformers import CLIPTextModel, CLIPVisionModel, AutoProcessor
from transformers import AutoImageProcessor, Dinov2Model

from diffusers.optimization import (
    Union,
    SchedulerType,
    Optional,
    Optimizer,
    TYPE_TO_SCHEDULER_FUNCTION,
)

from ppt_learning.utils.logging_utils import module_max_gradient, log_results

try:
    import pytorch3d.ops as torch3d_ops
except:
    print("pytorch3d not installed")

try:
    from openpoints.models.layers import furthest_point_sample
except:
    print("openpoints not installed")


# global model cache
clip_model = None
clip_processor = None

clip_vision_model = None
clip_vision_processor = None

dino_vision_model = None
dino_vision_processor = None

clip_text_model = None
clip_text_tokenizer = None

t5_model = None
t5_tok = None
vit_model = None
resnet_model = None

ModalityType = SimpleNamespace(
    VISION="image",
    TEXT="language",
    PROPRICEPTION="state",
    DEPTH="depth",
)


def recursive_in(data: Dict[str, Any], modality: str) -> bool:
    """Check if a nested modality key exists in the data dictionary.

    Args:
        data: Dictionary to search in
        modality: Nested key path separated by '/' (e.g., 'obs/rgb/image')

    Returns:
        True if the modality path exists, False otherwise
    """
    if "/" in modality:
        sub_modality = modality.split("/")[0]
        return recursive_in(data[sub_modality], "/".join(modality.split("/")[1:]))
    return modality in data


def recursive_get(data: Dict[str, Any], modality: str) -> Any:
    """Recursively get a nested value from a dictionary using a path.

    Args:
        data: Dictionary to search in
        modality: Nested key path separated by '/' (e.g., 'obs/rgb/image')

    Returns:
        Value at the specified modality path
    """
    if "/" in modality:
        sub_modality = modality.split("/")[0]
        return recursive_get(data[sub_modality], "/".join(modality.split("/")[1:]))
    return data[modality]


def mkdir_if_missing(dst_dir: str) -> None:
    """Create directory if it doesn't exist.

    Args:
        dst_dir: Directory path to create
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_args_hydra(path: str, cfg: DictConfig) -> None:
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    save_args_json(path, cfg_dict)


def save_args_json(path, args, convert_to_vars=False):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "config.json")
    with open(arg_json, "w") as f:
        if convert_to_vars:
            args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)


def select_task(task_name_list):
    task_configs = []
    for task_name in task_name_list:
        # assert task_name in ALL_TASK_CONFIG
        for task_config in ALL_TASK_CONFIG:
            if task_config[0] == task_name:
                task_configs.append(task_config)
                break

    return task_configs


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


def mkdir_if_missing(dst_dir):
    """make destination folder if it's missing"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_args_hydra(path, cfg):
    # save the current config
    mkdir_if_missing(path)
    with open(os.path.join(path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)


def get_scheduler(
    schduler_spec: DictConfig,
    optimizer: torch.optim.Optimizer,
    **kwargs: Any,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get a learning rate scheduler based on specification.

    Args:
        schduler_spec: Configuration for scheduler instantiation
        optimizer: PyTorch optimizer to schedule
        **kwargs: Additional keyword arguments

    Returns:
        Instantiated learning rate scheduler
    """
    sch = hydra.utils.instantiate(schduler_spec, optimizer)
    # sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=gamma, milestones=milestones, **kwargs)
    return sch


def get_optimizer(
    optimizer_spec: DictConfig,
    policy: torch.nn.Module,
    optimizer_extra: Optional[Any] = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Get an optimizer based on specification.

    Args:
        optimizer_spec: Configuration for optimizer instantiation
        policy: PyTorch model to optimize
        optimizer_extra: Additional optimizer configuration (unused)
        **kwargs: Additional keyword arguments

    Returns:
        Instantiated PyTorch optimizer
    """
    opt_i = hydra.utils.instantiate(optimizer_spec, params=policy.parameters())
    return opt_i


def batchify(
    data: Dict[str, torch.Tensor], exclude: List[str] = []
) -> Dict[str, torch.Tensor]:
    """Merge batchsize, seqlen, horizon into the first dimension.

    Args:
        data: Dictionary of tensors with shape (bs, seq_len, *dim)
        exclude: List of keys to exclude from batchification

    Returns:
        Dictionary with tensors reshaped to merge batch and sequence dimensions
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key not in exclude:
                data[key] = batchify(value)
    else:
        if isinstance(data, (str, list)):
            pass
        else:
            data = data.reshape(-1, *data.shape[2:])
    return data


def unbatchify(data, seq_len):
    """split the first dimension into batchsize, seqlen, horizon"""
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = unbatchify(value, seq_len)
    else:
        if isinstance(data, (str, list)):
            pass
        else:
            data = data.reshape(-1, seq_len, *data.shape[1:])
    return data


def sample_pcd_data(data, npoints, in_channels):
    """
    Refer to https://github.com/guochengqian/PointNeXt/blob/master/examples/classification/train.py#L248
    """
    points = data["x"]
    point_shape = len(points.size())
    if point_shape == 4:
        B, T, N, D = points.shape
        points = points.reshape(B * T, N, D)
    num_curr_pts = points.shape[1]
    if num_curr_pts > npoints:  # point resampling strategy
        if npoints == 1024:
            point_all = 1200
        elif npoints == 2048:
            point_all = 2400
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()
        if points.size(1) < point_all:
            point_all = points.size(1)
        try:
            fps_idx = furthest_point_sample(points[..., :3].contiguous(), point_all)
        except:
            _, fps_idx = torch3d_ops.sample_farthest_points(
                points[..., :3].contiguous(), K=point_all
            )
        fps_idx = fps_idx[..., np.random.choice(point_all, npoints, False)]
        points = torch.gather(
            points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1])
        )
    data["pos"] = points[:, :, :3].contiguous()
    data["x"] = points[:, :, :in_channels].transpose(1, 2).contiguous()
    if "colors" in data:
        data["colors"] = torch.gather(
            data["colors"],
            1,
            fps_idx.unsqueeze(-1).long().expand(-1, -1, data["colors"].shape[-1]),
        )
        data["colors"] = data["colors"].transpose(1, 2).contiguous()
    if point_shape == 4:
        data["pos"] = data["pos"].reshape(B, T, -1, 3)
        data["x"] = data["x"].reshape(B, T, -1, in_channels)
        if "colors" in data:
            data["colors"] = data["colors"].reshape(B, T, -1, in_channels)


def dict_apply(x, func):
    dict_type = type(x)
    if type(x) is not dict_type:
        return func(x)

    result = dict_type()
    for key, value in x.items():
        if isinstance(value, (str, list)):
            result[key] = value
        elif isinstance(value, (dict_type, dict, OrderedDict)):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def get_sinusoid_encoding_table(position_start, position_end, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(position_start, position_end)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def vis_image(images):
    import cv2

    image = unnormalize_image_numpy(images[0])
    cv2.imwrite("test.png", image)
    # cv2.waitKey(0)


def unnormalize_image_numpy(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = image * std + mean
    image = image * 255
    image = image.astype(np.uint8)
    return image


def normalize_image_numpy(image):
    import cv2

    """H x W x 3(uint8) -> imagenet normalized 3 x H x W"""
    image = cv2.resize(image, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image / 255.0

    # convert to array
    image = np.asarray(image)

    # normalize
    image = (image - mean) / std
    return image.transpose(2, 0, 1)


def dict_apply_device(x, device):
    if type(x) is not dict:
        return value.to(device)

    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply_device(value, device)
        else:
            result[key] = value.to(device)
    return result


@torch.no_grad()
def get_clip_embeddings(
    image, language, device="cuda", max_length=77, image_token_size=(3, 3)
):
    """Get CLIP embedding"""
    global clip_vision_model, clip_vision_processor, clip_text_model, clip_text_tokenizer
    if clip_vision_model is None:
        try:
            clip_vision_model = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            clip_vision_processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            clip_text_model = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            clip_text_tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        except:
            clip_vision_model = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32", local_files_only=True
            )
            clip_vision_processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", local_files_only=True
            )
            clip_text_model = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32", local_files_only=True
            )
            clip_text_tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32", local_files_only=True
            )

        clip_text_model = clip_text_model.to(device)
        clip_vision_model = clip_vision_model.to(device)
        print("initialize CLIP Model")

    # language
    if len(language) > max_length:  # one letter and one token
        print("language instruction too long:", language)
        language = language[:max_length]

    text_inputs = clip_text_tokenizer(
        text=[language],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    ).to(device)
    text_outputs = clip_text_model(**text_inputs)
    text_embeds_numpy = text_outputs.last_hidden_state.detach().cpu().numpy()

    # vision
    image = Image.fromarray(image)
    vision_inputs = clip_vision_processor(images=image, return_tensors="pt").to(device)
    vision_outputs = clip_vision_model(**vision_inputs)

    # 16 -> 50 (7*7+1)
    img_embeds = vision_outputs.last_hidden_state
    img_embeds = img_embeds[:, 1:]  # .reshape(1, 7, 7, -1)  # remove cls token

    # # pooling to downsample
    # D = img_embeds.shape[-1]
    # img_embeds = img_embeds.permute(0, 3, 1, 2)  # B C H W
    # img_embeds = torch.nn.functional.interpolate(img_embeds, image_token_size, mode="bilinear")
    # img_embeds = img_embeds.permute(0, 2, 3, 1).reshape(1, -1, D)  # B 4 C
    img_embeds_numpy = img_embeds.detach().cpu().numpy()

    # garbage collection
    del vision_outputs, text_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return text_embeds_numpy[0], img_embeds_numpy[0]


@torch.no_grad()
def get_dino_embeddings(image, device="cuda", image_token_size=(3, 3)):
    """Get CLIP embedding. TODO: replace with local CLIP to process image and language signal separately"""
    global dino_vision_model, dino_vision_processor
    if dino_vision_model is None:
        try:
            dino_vision_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
            dino_vision_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base"
            )
        except:
            dino_vision_model = Dinov2Model.from_pretrained(
                "facebook/dinov2-base", local_files_only=True
            )
            dino_vision_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base", local_files_only=True
            )

        print("initialize Dino Model")

    dino_vision_model = dino_vision_model.to(device)
    vision_inputs = dino_vision_processor(images=image, return_tensors="pt").to(device)
    vision_outputs = dino_vision_model(**vision_inputs)

    # 16 -> 50 (7*7+1)
    img_embeds = vision_outputs.last_hidden_state
    img_embeds = img_embeds[:, 1:]  # remove cls token
    img_embeds_numpy = img_embeds.detach().cpu().numpy()

    # garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    return img_embeds_numpy[0]


@torch.no_grad()
def get_t5_embeddings(language, per_token=False, max_length=128, device="cuda"):
    """Get T5 embedding"""
    global t5_model, t5_tok
    if t5_model is None:
        try:
            t5_model = T5Model.from_pretrained("t5-base")  # small
            t5_tok = T5Tokenizer.from_pretrained("t5-base")
        except:
            t5_model = T5Model.from_pretrained(
                "t5-base", local_files_only=True
            )  # small
            t5_tok = T5Tokenizer.from_pretrained("t5-base", local_files_only=True)
        t5_model = t5_model.to(device)

    enc = t5_tok(
        [language], return_tensors="pt", padding="max_length", max_length=max_length
    ).to(device)
    # forward pass through encoder only
    output = t5_model.encoder(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        return_dict=True,
    )

    if per_token:
        return output.last_hidden_state[0].detach().cpu().numpy()
    else:
        # get the final hidden states. average across tokens.
        emb = output.last_hidden_state[0].mean(dim=0).detach().cpu().numpy()
        return emb


def get_image_embeddings(image, image_encoder):
    if image_encoder == "resnet":
        return get_resnet_embeddings(image)
    if image_encoder == "vit":
        return get_vit_embeddings(image)
    if image_encoder == "clip":
        return get_clip_img_embeddings(image)
    if image_encoder == "dino":
        return get_dino_embeddings(image)
    else:
        raise Exception("missing embedding type")


@torch.no_grad()
def get_resnet_embeddings(image, per_token=False, device="cuda"):
    """Get Resnet embedding.
    H x W x 3 -> 1 x D"""
    global resnet_model
    if resnet_model is None:
        resnet_model = ResNet()

    resnet_model = resnet_model.to(device)

    image = normalize_image_numpy(image)
    resnet_model.eval()
    image_th = torch.FloatTensor(image).to(device)
    if len(image_th.shape) == 3:
        image_th = image_th[None]

    # vis_image(image_th)
    # forward pass through encoder only
    output = resnet_model.net(image_th)  # 1 x 512 x 7 x 7
    output = output.reshape(1, 512, -1).transpose(1, 2)
    return output.detach().cpu().numpy()


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )
