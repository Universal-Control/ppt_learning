from typing import Tuple
from omegaconf import DictConfig
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tyro
import os,sys
from os.path import join
from tqdm.auto import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import h5py
import torch.multiprocessing as mp

from ranging_anything.model import get_model as get_pda_model
from ranging_anything.compute_metric import (
    interp_depth_rgb,
    add_noise_to_depth,
    save_vis_depth,
    compute_metrics,
    recover_metric_depth_ransac,
    colorize_depth_maps,
)

import argparse

HDF5_PATH = "/mnt/bn/robot-minghuan-datasets-lq/xiaoshen/code/GR-Isaaclab/datasets/ur5_close_microwave_version_2_generated_37.hdf5"

def load_image(
    image_path: str,
    tar_size: Tuple[int, int] = (756, 1008),
) -> torch.Tensor:
    """
    Load image and resize to target size.
    Args:
        image_path: Path to input image.
        tar_size: Target size (h, w).
    Returns:
        image: Image tensor with shape (1, 3, h, w).
    """
    image = imageio.imread(image_path)
    image = cv2.resize(image, tar_size[::-1], interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return image


def load_depth(depth_path: str) -> torch.Tensor:
    """
    depth is in mm and stored in 16-bit PNG
    """
    depth = imageio.imread(depth_path)
    depth = (depth / 1000.0).astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    return depth


class hdf5Dataset(Dataset):
    def __init__(self, hdf5_path, tar_size=(476, 630), key="third"):
        self.hdf5_path = hdf5_path
        self.tar_size = tar_size
        self.key = key
        self.traj_ends = h5py.File(hdf5_path)["data_info/cum_num"]

    def __len__(self):
        return self.traj_ends[-1]

    def __getitem__(self, idx):
        traj_idx = np.searchsorted(self.traj_ends, idx, side="right") - 1
        data_idx = idx - self.traj_ends[traj_idx - 1] if traj_idx > 0 else idx
        data = h5py.File(self.hdf5_path)[f"data/demo_{traj_idx}"]
        rgb_key = f"obs/images/{self.key}_rgb"
        depth_key = f"obs/depths/{self.key}_depth"

        rgb_s = data[rgb_key]
        depth_s = data[depth_key][()].astype(np.float32) / 1000.0

        rgb = rgb_s[f"{data_idx}"][()]
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        rgb = cv2.resize(rgb, self.tar_size[::-1], interpolation=cv2.INTER_AREA)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        depth = cv2.resize(
            depth_s[data_idx], self.tar_size[::-1], interpolation=cv2.INTER_AREA
        )
        depth = interp_depth_rgb(depth, rgb)
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        return rgb, depth


class hdf5TrajDataset(Dataset):
    def __init__(self, hdf5_path, tar_size=(1008, 756), key="third"):
        self.hdf5_path = hdf5_path
        self.tar_size = tar_size
        self.key = key
        self.traj_ends = h5py.File(hdf5_path)["data_info/cum_num"]

    def __len__(self):
        return len(self.traj_ends) - 1

    def __getitem__(self, idx):
        data = h5py.File(self.hdf5_path)[f"data/demo_{idx}"]
        rgb_key = f"obs/images/{self.key}_rgb"
        depth_key = f"obs/depths/{self.key}_depth"

        rgb_s = data[rgb_key]
        depth_s = data[depth_key][()].astype(np.float32) / 1000.0
        length = len(rgb_s)
        rgbs_lst = []
        depths_lst = []
        for i in range(length):
            rgb = rgb_s[f"{i}"][()]
            rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
            rgb = cv2.resize(rgb, self.tar_size[::-1], interpolation=cv2.INTER_AREA)
            depth = cv2.resize(
                depth_s[i], self.tar_size[::-1], interpolation=cv2.INTER_AREA
            )
            depth = interp_depth_rgb(depth, rgb)
            depths_lst.append(torch.from_numpy(depth).unsqueeze(0).float())
            image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            rgbs_lst.append(image)
        rgbs = torch.stack(rgbs_lst, dim=0)
        depths = torch.stack(depths_lst, dim=0)

        return idx, rgbs, depths


def load_hdf5(
    hdf5_path: str, tar_size: Tuple[int, int] = (476, 630), demo_index=133, key="third"
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = h5py.File(hdf5_path)[f"data/demo_{demo_index}"]
    rgb_key = f"obs/images/{key}_rgb"
    depth_key = f"obs/depths/{key}_depth"
    rgb_s = data[rgb_key]
    depth_s = data[depth_key][()].astype(np.float32) / 1000.0
    length = len(rgb_s)
    rgbs_lst = []
    depths_lst = []
    for i in range(length):
        rgb = rgb_s[f"{i}"][()]
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        rgb = cv2.resize(rgb, tar_size[::-1], interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth_s[i], tar_size[::-1], interpolation=cv2.INTER_AREA)
        depth = interp_depth_rgb(depth, rgb)
        depths_lst.append(torch.from_numpy(depth).unsqueeze(0).float())
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgbs_lst.append(image)
    rgbs = torch.stack(rgbs_lst, dim=0)
    depths = torch.stack(depths_lst, dim=0)
    return rgbs, depths


def plot_depth(
    image: torch.Tensor,
    depth: torch.Tensor,
    lowres_depth: torch.Tensor,
    output_path: str,
) -> None:
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(depth.detach().squeeze(0).squeeze(0).cpu().numpy())
    plt.title("Predicted Depth")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(lowres_depth.detach().squeeze(0).squeeze(0).cpu().numpy())
    plt.title("Input Depth")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)


def write_to_videos(images, depths, lowres_depths, output_path, traj_id="") -> None:
    """
    Write the predicted depth map to a video.
    Args:
        depth: Predicted depth map tensor with shape (B, 1, H, W).
        output_path: Path to the output video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_path = os.path.dirname(output_path)
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()
    depths_np = depths.squeeze(1)
    lowres_depths_np = lowres_depths.squeeze(1)

    depths_colorize_lst = []
    lowres_depths_colorize_lst = []
    min_depth = lowres_depths_np.min()
    max_depth = lowres_depths_np.max()
    for i in range(depths_np.shape[0]):
        depths_colorize_lst.append(
            colorize_depth_maps(depths_np[i], min_depth, max_depth)
        )
        lowres_depths_colorize_lst.append(
            colorize_depth_maps(lowres_depths_np[i], min_depth, max_depth)
        )
    depths_colorize = np.concatenate(depths_colorize_lst, axis=0)
    lowres_depths_colorize = np.concatenate(lowres_depths_colorize_lst, axis=0)
    depths_colorize = np.moveaxis(depths_colorize, 1, 3)
    lowres_depths_colorize = np.moveaxis(lowres_depths_colorize, 1, 3)

    imageio.imwrite(
        os.path.join(output_path, f"rgb{traj_id}.mp4"), np.uint8(images_np * 255.0)
    )
    imageio.imwrite(
        os.path.join(output_path, f"depth{traj_id}.mp4"),
        (depths_colorize * 255.0).astype(np.uint8),
    )
    imageio.imwrite(
        os.path.join(output_path, f"lowres_depth{traj_id}.mp4"),
        (lowres_depths_colorize * 255.0).astype(np.uint8),
    )


def batch_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    lowres_depth: torch.Tensor,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Inference on a batch of images and depth maps.
    Args:
        model: Pre-trained model.
        image: Image tensor with shape (B, 3, H, W).
        lowres_depth: Low-resolution depth map tensor with shape (B, 1, H, W).
        batch_size: Batch size.
    Returns:
        depth: Predicted depth map tensor with
    """
    raw_shape_len = len(image.shape)
    if raw_shape_len > 4:  # traj data
        image = image.reshape(-1, *image.shape[2:])
        lowres_depth = lowres_depth.reshape(-1, *lowres_depth.shape[2:])

    depth = []
    tqdm_bar = tqdm(total=image.shape[0], desc="Inference")
    for i in range(0, image.shape[0], batch_size):
        image_batch = image[i : i + batch_size]
        lowres_depth_batch = lowres_depth[i : i + batch_size]
        if isinstance(model, DDP):
            depth_batch = model.module.inference(
                image=image_batch, lowres_depth=lowres_depth_batch
            )
        else:
            depth_batch = model.inference(
                image=image_batch, lowres_depth=lowres_depth_batch
            )
        depth.append(depth_batch)
        tqdm_bar.update(batch_size)
    tqdm_bar.close()
    depth = torch.cat(depth, dim=0)

    if raw_shape_len > 4:  # traj data
        depth = depth.reshape(image.shape[0], -1, *depth.shape[1:])
    return depth


def plot_metrics_dict(
    metrics_dict, fig_size=(15, 10), n_cols=3, title="Metrics Comparison"
):
    """
    Plot line graphs for each item in a metrics dictionary and combine them into one figure.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary where keys are labels and values are lists of metrics to plot
    fig_size : tuple, optional
        Size of the overall figure (width, height)
    n_cols : int, optional
        Number of columns in the subplot grid
    title : str, optional
        Title for the overall figure
    """
    # Calculate number of plots and determine grid dimensions
    n_plots = len(metrics_dict)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Find global min and max for consistent y-axis scaling
    all_values = []
    for values in metrics_dict.values():
        all_values.extend(values)

    global_min = min(all_values) if all_values else 0
    global_max = max(all_values) if all_values else 1

    # Add some buffer to the min and max
    y_range = global_max - global_min
    y_min = global_min - 0.05 * y_range
    y_max = global_max + 0.05 * y_range

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharey=True)
    fig.suptitle(title, fontsize=16)

    # Flatten axes array for easy indexing
    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols > 1:
        axes = axes  # Already a 1D array
    elif n_cols == 1 and n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make a list for single subplot

    # Plot each metric
    for i, (key, values) in enumerate(metrics_dict.items()):
        if i < len(axes):
            ax = axes[i]
            x = np.arange(len(values))
            ax.plot(x, values)
            ax.set_title(key)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle="--", alpha=0.7)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

    return fig


def print_average_std(dict):
    print_str = ""
    for key, value in dict.items():
        if isinstance(value, list):
            print_str += f"{key}: {np.mean(value):.3f} +- {np.std(value):.3f}\n"
        else:
            print_str += f"{key}: {value:.4f}\n"
    print(print_str)


def model_infer(
    model: str,
    img: torch.Tensor,
    lowres: torch.Tensor,
    align_scale: bool = True,
) -> None:
    """
    Inference on a single image and depth map.
    Args:
        input_image_path: Path to input image.
        input_depth_path: Path to input depth map.
        model_path: Path to the pre-trained model.
    """
    local_metrics = {}
    rank = 0

    img = img.to(f"cuda:{rank}")
    lowres = lowres.to(f"cuda:{rank}")

    # 进行推理
    pred = batch_inference(model, img, lowres)

    pred = pred.detach().cpu().numpy()
    lowres = lowres.detach().cpu().numpy()
    msk = np.logical_and(lowres > 1e-3, ~np.isnan(lowres)) & (~np.isinf(lowres))
    if align_scale:
        output_depth = recover_metric_depth_ransac(
            pred,
            lowres,
            msk,
        )
    else:
        output_depth = pred

    return output_depth

@torch.no_grad()
def eval_traj(
    rank,
    world_size,
    model_type: str,
    hdf5_path: str = HDF5_PATH,
    output_path: str = "results/depth_results.png",
    corrupted_depth: bool = False,
) -> None:
    """
    Inference on a single image and depth map.
    Args:
        input_image_path: Path to input image.
        input_depth_path: Path to input depth map.
        model_path: Path to the pre-trained model.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = get_model(model_type, rank)

    # 准备数据
    dataset = hdf5TrajDataset(hdf5_path)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    for traj_id, img, gt in tqdm(dataloader, position=rank):

        if corrupted_depth:
            lowres = add_noise_to_depth(
                img, gt, gt, value_error_scale_max=0.6, value_error_scale_min=0.3
            )
        else:
            lowres = gt.clone()
        img = img.to(f"cuda:{rank}")
        gt = gt.to(f"cuda:{rank}")
        lowres = lowres.to(f"cuda:{rank}")

        # 进行推理
        pred = batch_inference(model, img, lowres)

        if len(img.shape) > 4:  # traj data
            img = img.reshape(-1, *img.shape[2:])
            gt = gt.reshape(-1, *gt.shape[2:])
            lowres = lowres.reshape(-1, *lowres.shape[2:])
            pred = pred.reshape(-1, *pred.shape[2:])

        pred = pred.detach().cpu().numpy()
        lowres = lowres.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        metrics_dict, pred_depth_align_low = compute_metrics(img, lowres, gt, pred)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plot_metrics_dict(metrics_dict, title="Metrics Comparison").savefig(
            os.path.join(os.path.dirname(output_path), f"metrics_dict_{traj_id}.png")
        )
        # plot_depth(image, depth, lowres_depth, output_path)
        write_to_videos(img, pred_depth_align_low, lowres, output_path)

def get_model(model_path: str, rank: int = 0, model_type = "promptda-robot") -> torch.nn.Module:
    """
    Load the pre-trained model.
    Args:
        model_path: Path to the pre-trained model.
        rank: Rank of the current process.
    Returns:
        model: Pre-trained model.
    """
    if "jit" in model_path:
        model = torch.jit.load(model_path)
    else:
        if model_type == "promptda-robot":
            model = get_pda_model(model_path, rank=rank)
        elif model_type == "promptda":
            model_path = f"{os.environ['workspace']}/cache_models/pda_vitl.ckpt"
            model = get_pda_model(model_path, rank=rank)
        elif model_type == "dav2":
            from depth_anything_v2.metric_depth.depth_anything_v2 import DepthAnythingV2

            # 加载模型
            encoder = "vitl"
            dataset = "hypersim"
            max_depth = 10
            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
                "vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                },
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                },
            }
            model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
            ckpt_path = f'{os.environ["workspace"]}/cache_models/depth_anything_v2_metric_{dataset}_{encoder}.pth'
            model.load_state_dict(torch.load(ckpt_path, map_location=f"cuda:{rank}"))

        model.eval()
        # model = DDP(model, device_ids=[rank])

    return model
