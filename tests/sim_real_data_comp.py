from ppt_learning.dataset.sim_traj_dataset import TrajDataset

import os
from os.path import join
from tqdm import tqdm
import numpy as np
import torch
import collections
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from collections import OrderedDict
from typing import Union
import time
import matplotlib
import cv2
import pickle
import copy

import ranging_anything
from ranging_anything.model import RangeAnything
from ranging_anything.compute_metric import (
    interp_depth_rgb,
    recover_metric_depth_ransac,
)

EPS=1e-3
TensorData = Union[np.ndarray, torch.Tensor]
output_dir = "sim-real"

def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, depth_model):
        super().__init__()
        for para in depth_model.parameters():
            para.requires_grad = False
        self.depth_model = depth_model

        class WarpMinMax:
            def warp(self, depth, reference, **kwargs):
                depth_min, depth_max = (
                    reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0],
                    reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0],
                )
                depth_max[(depth_max - depth_min) < EPS] = (
                    depth_min[(depth_max - depth_min) < EPS] + EPS
                )
                return (depth - depth_min[:, None, None]) / (depth_max - depth_min)[
                    :, None, None
                ]

            def unwarp(self, depth, reference, **kwargs):
                depth_min, depth_max = (
                    reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0],
                    reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0],
                )
                depth_max[(depth_max - depth_min) < EPS] = (
                    depth_min[(depth_max - depth_min) < EPS] + EPS
                )
                return (
                    depth * (depth_max - depth_min)[:, None, None]
                    + depth_min[:, None, None]
                )

        self.warp_func = WarpMinMax()

    def forward(self, x, lowres_depth, msks, absolute=True, align_scale=True):
        depth = self.depth_model(
            (x - self.depth_model._mean) / self.depth_model._std,
            lowres_depth=self.warp_func.warp(lowres_depth, reference=lowres_depth),
        ).unsqueeze(1)
        if absolute:
            depth = (
                self.warp_func.unwarp(depth, reference=lowres_depth)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            depth = self.warp_func.warp(depth, reference=depth)
            depth = depth.detach().cpu().numpy()
        if align_scale:
            res = []
            for idx, pred in enumerate(depth):
                lowres = lowres_depth[idx].detach().cpu().numpy()
                tmp = recover_metric_depth_ransac(
                    pred,
                    lowres,
                    msks[idx],
                )
                res.append(tmp)
            depth = np.stack(res, axis=0)
        else:
            depth = depth.squeeze(1)
        return depth


def save_data(depth_image, color_image, camera_dir, frame_count):
    depth_image = depth_image * 1000
    depth_image = np.nan_to_num(depth_image, 0)
    depth_image[depth_image > 65535] = 65535
    depth_image[depth_image < 1e-5] = 0

    cv2.imwrite(
        str(camera_dir / f"depth_{frame_count}.png"), depth_image.astype(np.uint16)
    )
    cv2.imwrite(str(camera_dir / f"color_{frame_count}.png"), color_image)

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def save_vis_depth(
    sim_depth, sim_rgb, real_depth, real_rgb, name, tag, gt_depth=None, lowres_depth=None
) -> None:
    depth_np = real_depth
    save_name = name

    save_imgs = []
    if gt_depth is not None:
        gt_depth_np = gt_depth
        save_img = colorize_depth_maps(
            depth_np,
            gt_depth_np.min(),
            gt_depth_np.max(),
        )[0].transpose((1, 2, 0))
    else:
        save_img = colorize_depth_maps(
            depth_np,
            depth_np.min(),
            depth_np.max(),
        )[0].transpose((1, 2, 0))
    save_imgs.append(copy.deepcopy(save_img))
    rgb_np = cv2.resize(
        real_rgb,
        (save_img.shape[1], save_img.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    save_img = np.concatenate([rgb_np, save_img], axis=1)
    save_imgs.append(copy.deepcopy(rgb_np))

    depth_np = sim_depth
    if gt_depth is not None:
        gt_depth_np = gt_depth
        depth_img = colorize_depth_maps(
            depth_np,
            gt_depth_np.min(),
            gt_depth_np.max(),
        )[0].transpose((1, 2, 0))
    else:
        depth_img = colorize_depth_maps(
            depth_np,
            depth_np.min(),
            depth_np.max(),
        )[0].transpose((1, 2, 0))
    save_img = np.concatenate([depth_img, save_img], axis=1)
    save_imgs.append(save_img)
    rgb_np = cv2.resize(
        sim_rgb,
        (rgb_np.shape[1], rgb_np.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    save_img = np.concatenate([rgb_np, save_img], axis=1)
    save_imgs.append(rgb_np)

    if lowres_depth is not None:
        lowres_depth_np = lowres_depth
        tar_h, tar_w = depth_np.shape[0], depth_np.shape[1]
        if (
            lowres_depth_np.shape[1] != tar_w
            or lowres_depth_np.shape[0] != tar_h
        ):
            if (lowres_depth_np == 0.0).sum() >= 10:
                u, v = lowres_depth_np.nonzero()
                orig_u, orig_v = u, v
                u, v = (u * tar_h / lowres_depth_np.shape[0]).astype(
                    np.int32
                ), (v * tar_w / lowres_depth_np.shape[1]).astype(np.int32)
                lowres_depth_np_new = np.zeros_like(depth_np)
                lowres_depth_np_new[u, v] = lowres_depth_np[orig_u, orig_v]
                lowres_depth_np = lowres_depth_np_new
            else:
                lowres_depth_np = cv2.resize(
                    lowres_depth_np,
                    (tar_w, tar_h),
                    interpolation=cv2.INTER_NEAREST,
                )
        if gt_depth is not None:
            lowres_depth_np = colorize_depth_maps(
                lowres_depth_np,
                gt_depth_np.min(),
                gt_depth_np.max(),
            )[0].transpose((1, 2, 0))
        else:
            lowres_depth_np = colorize_depth_maps(
                lowres_depth_np,
                (
                    lowres_depth_np.min()
                ),
                lowres_depth_np.max(),
            )[0].transpose((1, 2, 0))
        lowres_depth_np = cv2.resize(
            lowres_depth_np,
            (depth_np.shape[1], depth_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        save_img = np.concatenate(
            [save_img, lowres_depth_np], axis=1
        )
        save_imgs.append(lowres_depth_np)
    depth_diff = sim_depth - real_depth[...,None]
    depth_diff_img = colorize_depth_maps(
            np.abs(depth_diff),
            0.,
            depth_diff.max(),
        )[0].transpose((1, 2, 0))
    save_img = np.concatenate(
        [save_img, depth_diff_img], axis=1
    )
    save_imgs.append(depth_diff_img)
    depth_mask = np.abs(depth_diff) > 0.04
    save_img = np.concatenate(
        [save_img, depth_mask.repeat(3, axis=-1)], axis=1
    )
    save_imgs.append(depth_mask)

    if gt_depth is not None:
        gt_depth_np = gt_depth
        gt_depth_np = colorize_depth_maps(
            gt_depth_np,
            gt_depth_np.min(),
            gt_depth_np.max(),
        )[0].transpose((1, 2, 0))
        gt_depth_np = cv2.resize(
            gt_depth_np,
            (depth_np.shape[1], depth_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        save_img = np.concatenate(
            [save_img, gt_depth_np], axis=1
        )
        save_imgs.append(gt_depth_np)
    img_path = join(output_dir, f"{save_name}")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    ending = ".png"
    if '.exr' in img_path:
        ending = ".exr"
    imageio.imwrite(
        img_path.replace(ending, ".jpg"), (save_img * 255.0).astype(np.uint8)
    )

dataset_sim = TrajDataset(
    domain="debug",
    dataset_path="/mnt/bn/robot-minghuan-datasets-lq/xiaoshen/datasets/ur5_put_bowl_in_microwave_and_close/output_dataset_debug.zarr",
    from_empty=False,
    use_disk=True,
    load_from_cache=True,
    use_lru_cache=True,
    val_ratio=0.,
    action_horizon=2,
    observation_horizon=1,
    horizon=2,
    pad_before=2,
    pad_after=15,
    use_pcd=False,
    pcd_channels=4,
    pcdnet_pretrain_domain="scanobjectnn",
    ignored_keys=["language", "initial_state", "states", "images", "color", "abs_gripper_pos", "pointcloud", "wbc_target", "wbc_step", "last_action"]
)

dataset_real = TrajDataset(
    domain="debug",
    dataset_path="/mnt/bn/robot-minghuan-datasets-lq/xiaoshen/datasets/real/20250618_014003.cartesian.zarr",
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
    use_pcd=False,
    pcd_channels=4,
    pcdnet_pretrain_domain="scanobjectnn",
    action_key="action",
    state_keys=["ee_positions", "ee_rotations", "joint_posistions", "joint_velocities", "gripper_state"],
    ignored_keys=["language", "initial_state", "states", "abs_gripper_pos", "pointcloud", "wbc_target", "wbc_step", "last_action"]
)

sim_depth = dataset_sim.replay_buffer.data.obs.depths.camera_0[5:30]
sim_rgb = dataset_sim.replay_buffer.data.obs.images.camera_0[5:30]
real_depth = dataset_real.replay_buffer.data.obs.depths.camera_0[:3000]
real_rgb = dataset_real.replay_buffer.data.obs.images.camera_0[:3000]

cv2.imwrite("sim-real/sim_rgb.png", sim_rgb[0][..., [2, 1, 0]])

device = "cuda"
depth_model = RangeAnything(
    block_type='featurefusiondepthblock',
    # encoder='vits',
    # features=64,
    # out_channels=[48, 96, 192, 384],
    warp_func=ranging_anything.depth_anything.warp_func.WarpMinMax,
    # load_pretrain_net="/mnt/bn/robot-minghuan-debug/promptanyrobot/outputs/depth_estimation/promptanyrobot_ft_vits_allsim_shape640_435/checkpoints/e024-s102400.ckpt",
    load_pretrain_net="/mnt/bn/robot-minghuan-debug/promptanyrobot/outputs/depth_estimation/promptanyrobot_ft_vitl_allsim_shape640_435/noscannet/checkpoints_25/e025-s103376.ckpt"
)
depth_model = DepthModelWrapper(depth_model)
depth_size = [504, 672]

depth_model.eval()
depth_model.to(device)
print("Depth model applied to camera data.")

all_pred_depths = []
step_len = 10
for idx in tqdm(range(0, 2, step_len)):
    colors = real_rgb[idx:idx+step_len, ..., [2, 1, 0]]  # Convert BGR to RGB
    colors = colors.astype(
        np.float32
    )  # bs, H, W, 3
    depths = real_depth[idx:idx+step_len]
    resized_colors = []
    resized_depths = []
    masks = []
    for i in range(len(depths)):
        color = cv2.resize(
            colors[i], depth_size[::-1], interpolation=cv2.INTER_AREA
        )
        depth = cv2.resize(
            depths[i], depth_size[::-1], interpolation=cv2.INTER_NEAREST
        )
        mask = np.logical_and(depth > 1e-3, ~np.isnan(depth)) & (
            ~np.isinf(depth)
        )
        depth = interp_depth_rgb(
            depth, cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), speed=5, k=4
        )
        resized_colors.append(color)
        resized_depths.append(depth)
        masks.append(mask)
    if len(masks) == 0:
        break
    masks = np.stack(masks, axis=0)  # BS, H, W
    colors = convert_to_torch(
        np.stack(resized_colors, axis=0),
        dtype=torch.float32,
        device=device,
    ).permute(
        0, 3, 1, 2
    )  # (BS, 3, H, W)
    depths = convert_to_torch(
        np.stack(resized_depths, axis=0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    torch.cuda.synchronize() if device == "cuda" else None
    time1 = time.time()
    with torch.no_grad():
        pred_depths = depth_model(
            colors / 255., depths, masks
        )  # , absolute=False, align_scale=False)
    # torch.cuda.synchronize() if device == "cuda" else None
    # print("infer used time", time.time() - time1)
    for i in range(len(pred_depths)):
        all_pred_depths.append(pred_depths[i][..., None])

# all_pred_depths = np.stack(all_pred_depths, axis=0)
# print(all_pred_depths.min(), all_pred_depths.max(), all_pred_depths.mean())
# with open("all_pred_depths.pkl", "wb") as f:
#     pickle.dump(all_pred_depths, f)
# with open("all_pred_depths.pkl", "rb") as f:
#     all_pred_depths = pickle.load(f)

save_vis_depth(sim_depth[0], sim_rgb[0]/255., cv2.resize(all_pred_depths[0], real_rgb.shape[1:3][::-1], interpolation=cv2.INTER_AREA), real_rgb[0]/255., f'0.png', '', gt_depth=real_depth[0])
# save_vis_depth(sim_depth[0], sim_rgb[0]/255., cv2.resize(all_pred_depths[0], sim_rgb.shape[1:3][::-1], interpolation=cv2.INTER_AREA), cv2.resize(real_rgb[0]/255., sim_rgb.shape[1:3][::-1], interpolation=cv2.INTER_AREA), f'0.png', '', gt_depth=cv2.resize(real_depth[0], sim_rgb.shape[1:3][::-1], interpolation=cv2.INTER_NEAREST))

import ipdb; ipdb.set_trace()