import collections
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import argparse
import zarr
import os
import numpy as np
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--failed_dataset_path", default="/data/datasets/ur5_put_bowl_in_microwave_and_close/put_bowl_in_microwave_8_demos_generated_interrupt_simplify_gripper_retry_more_wait_520.zarr")
parser.add_argument("--dataset_path", default="/mnt/xiaoshen/datasets/ur5_put_bowl_in_microwave_and_close/put_bowl_in_microwave__520_collected_data_retry_random_x015_new_subtask_generated_1gpu.zarr")
# parser.add_argument("--dataset_path", default="/data/datasets/ur5_put_bowl_in_microwave_and_close/put_bowl_in_microwave_8_demos_generated_interrupt_simplify_gripper_retry_more_wait_520.zarr")
parser.add_argument("--type", type=int, default=0) # 0 for initial, 1 for last
args = parser.parse_args()

zarr_store = zarr.DirectoryStore(os.path.expanduser(args.dataset_path))
zarr_root = zarr.group(store=zarr_store)
failed_zarr_path = Path(args.failed_dataset_path).parent / f"{Path(args.failed_dataset_path).stem}_failed.zarr"
failed_zarr_store = zarr.DirectoryStore(failed_zarr_path)
failed_zarr_root = zarr.group(store=failed_zarr_store)
# Get data
if args.type == 0:
    rigid_pos = zarr_root["data"]['initial_state']['rigid_object']['bowl']['root_pose'][:, :2]
    articulation_pos = zarr_root["data"]['initial_state']['articulation']['microwave']['root_pose'][:, :2]

    faild_rigid_pos = failed_zarr_root["data"]['initial_state']['rigid_object']['bowl']['root_pose'][:, :2]
    faild_articulation_pos = failed_zarr_root["data"]['initial_state']['articulation']['microwave']['root_pose'][:, :2]
else:
    last_idx = zarr_root["meta/episode_ends/"][()]
    rigid_pos = zarr_root["data"]['states']['rigid_object']['bowl']['root_pose'][last_idx - 1, :2]
    articulation_pos = zarr_root["data"]['states']['articulation']['microwave']['root_pose'][last_idx - 1, :2]

    last_failed_idx = failed_zarr_root["meta/episode_ends/"][()]
    faild_rigid_pos = failed_zarr_root["data"]['states']['rigid_object']['bowl']['root_pose'][last_failed_idx - 1, :2]
    faild_articulation_pos = failed_zarr_root["data"]['states']['articulation']['microwave']['root_pose'][last_failed_idx - 1, :2]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot scatter points
# ax.scatter(faild_rigid_pos[:, 0], faild_rigid_pos[:, 1], label="failed rigid body", color='r')
ax.scatter(rigid_pos[:, 0], rigid_pos[:, 1], label="success rigid body", color='y')
# ax.scatter(faild_articulation_pos[:, 0], faild_articulation_pos[:, 1], color='b', label="failed articulation")
ax.scatter(articulation_pos[:, 0], articulation_pos[:, 1], color='orange', label="success articulation")

# Calculate mean center of articulation positions
art_center = np.mean(articulation_pos, axis=0)

# Draw rectangle centered at articulation mean (y direction ±21.5cm, x direction ±15cm)
rect_width = 0.30  # ±15cm
rect_height = 0.43  # ±21.5cm
rect = Rectangle((art_center[0] - rect_width/2, art_center[1] - rect_height/2), 
                 rect_width, rect_height, 
                 linewidth=1, edgecolor='g', facecolor='none')
ax.add_patch(rect)

random_x = -0.1, 0.14
random_y = -0.2, 0.15
rigid_center = -0.1, -0.04
rect_width, rect_height = rigid_center[0]+random_x[1] - (rigid_center[0]+random_x[0]), rigid_center[1]+random_y[1] - (rigid_center[1]+random_y[0])
rect = Rectangle((rigid_center[0] + random_x[0], rigid_center[1] + random_y[0]),
                 rect_width, rect_height,
                 linewidth=1, edgecolor='g', facecolor='none')
ax.add_patch(rect)

# Draw circles with radius 3.5cm for each rigid body position
# for pos in rigid_pos:
#     circle = Circle((pos[0], pos[1]), 0.035, fill=False, edgecolor='r', linestyle='--')
#     ax.add_patch(circle)

# for pos in faild_rigid_pos:
#     circle = Circle((pos[0], pos[1]), 0.035, fill=False, edgecolor='y', linestyle='--')
#     ax.add_patch(circle)

# Draw rectangle centered at (0,0), x direction 0.8, y direction 1.2
center_rect = Rectangle((-0.4, -0.6), 0.8, 1.2, linewidth=2, edgecolor='purple', facecolor='none')
ax.add_patch(center_rect)

# Draw circle centered at (-6.25179097e-01, 4.98731775e-02) with radius 0.1
ur5_pos_circle = Circle((-6.25179097e-01, 4.98731775e-02), 0.1, fill=False, edgecolor='orange', linewidth=2)
ax.add_patch(ur5_pos_circle)

# Add legend and labels
ax.legend()
ax.set_xlabel('X axis (forward)')
ax.set_ylabel('Y axis (left)')
ax.set_title(f"Position Distribution Visualization {'Start' if args.type==0 else 'End'}")
ax.axis('equal')  # Ensure x and y axes have the same scale
ax.set_xlim(-0.52, 0.52)
ax.set_ylim(-0.62, 0.62)
# Save figure
plt.savefig(f"pos_visualization_{args.type}.png")
plt.close(fig)
