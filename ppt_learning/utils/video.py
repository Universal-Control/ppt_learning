import os
import cv2
import time
import numpy as np
import open3d as o3d
from pathlib import Path

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
            self.video_save_dir = Path(f"./outputs/video_eval/{date_time}")

    def extend(self, key, snaps, category):
        if key not in self._snaps:
            self._snaps[category][key] = []
        self._snaps[category][key].extend(snaps)

    def save(self, dir_name):
        for category in self._snaps:
            frames = []
            if len(self._snaps[category]):
                print(f"Saving videos of {category}")
                if category == "pointcloud":
                    # visualize pointcloud and save the video
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
                    frames = self._snaps[key]

                os.makedirs(self.video_save_dir / category / dir_name, exist_ok=True)
                for key in self._snaps[category]:
                    imageio.imwrite(
                        self.video_save_dir / category / dir_name / f"{key}.mp4", frames
                    )
