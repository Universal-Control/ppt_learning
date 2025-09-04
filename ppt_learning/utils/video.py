import os
import cv2
import time
import numpy as np
import open3d as o3d
from pathlib import Path
import torch
import matplotlib
from matplotlib import colormaps

import imageio.v3 as imageio


def save_video(images, name, path, fps=10):
    """save video to path"""
    if isinstance(name, tuple):
        name = name[0]
    if not os.path.exists(path):
        os.makedirs(path)
    for key, val in images.items():
        video_path = os.path.join(path, f"{name}_{key}.mp4")
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (val[0].shape[1], val[0].shape[0]),
        )
        for img in val:
            writer.write(img[..., :3])
        writer.release()


class videoLogger:
    def __init__(self, video_save_dir=None):
        self._snaps = {"color": {}, "depth": {}, "pointcloud": {}}

        date_time = time.strftime("%m%d-%H%M%S")
        if video_save_dir is None:
            self.video_save_dir = Path(f"./outputs/video_eval_no_label/{date_time}")
        else:
            self.video_save_dir = video_save_dir

    def normalize_and_visualize_depth(
        self, depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
    ):
        """
        Colorize depth maps.
        """
        depth_map = depth_map.squeeze()
        assert len(depth_map.shape) >= 2, "Invalid dimension"

        if isinstance(depth_map, torch.Tensor):
            depth = depth_map.detach().clone().squeeze().numpy()
        elif isinstance(depth_map, np.ndarray):
            depth = depth_map.copy().squeeze()
        # reshape to [ (B,) H, W ]
        if depth.ndim < 3:
            depth = depth[np.newaxis, :, :]

        # colorize
        cm = colormaps[cmap]
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

        return (np.einsum("chw->hwc", img_colored.squeeze()) * 255).astype(np.uint8)

    def extend(self, key, snaps, category):
        if key not in self._snaps[category]:
            self._snaps[category][key] = []
        if category == "depth":
            self._snaps[category][key].append(
                self.normalize_and_visualize_depth(snaps, 0.0, 1.5)
            )
        else:
            self._snaps[category][key].append(snaps)

    def reset(self):
        self._snaps = {"color": {}, "depth": {}, "pointcloud": {}}

    def save(self, dir_name, model_name):
        for category in self._snaps:
            if len(self._snaps[category]):
                print(f"Saving videos of {category}")
                if category == "pointcloud":
                    # visualize pointcloud and save the video
                    frames = []
                    for key in self._snaps[category]:
                        pcds = self._snaps[category][key]

                        for pcd in pcds:
                            vis = o3d.visualization.Visualizer()
                            vis.create_window(visible=False)
                            vis.add_geometry(pcd)
                            vis.poll_events()
                            vis.update_renderer()
                            frame = vis.capture_screen_float_buffer(False)
                            frames.append(np.asarray(frame) * 255)
                            vis.destroy_window()

                        # Save as video
                        os.makedirs(
                            self.video_save_dir / category / dir_name, exist_ok=True
                        )
                        imageio.imwrite(
                            self.video_save_dir / category / dir_name / f"{key}.mp4",
                            frames,
                            fps=30,
                        )

                else:

                    os.makedirs(
                        self.video_save_dir / category / model_name / dir_name,
                        exist_ok=True,
                    )
                    for key in self._snaps[category]:
                        imageio.imwrite(
                            self.video_save_dir
                            / category
                            / model_name
                            / dir_name
                            / f"{key}.mp4",
                            self._snaps[category][key],
                            fps=30,
                        )
