import zarr
import imageio.v3 as imageio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_path",
    type=str,
    default="/mnt/xiaoshen/datasets/ur5_put_bowl_in_microwave_and_close/put_bowl_in_microwave__520_collected_data_retry_random_x015_new_subtask_generated_1gpu.zarr",
)
parser.add_argument(
    "--start",
    type=int,
    default=0,
)
parser.add_argument(
    "--end",
    type=int,
    default=-1,
)

args = parser.parse_args()

zarr.DirectoryStore(os.path.expanduser(args.file_path))

zarr_store = zarr.DirectoryStore(os.path.expanduser(args.file_path))
zarr_root = zarr.group(store=zarr_store)

images = zarr_root["data/obs/images/camera_0"]

imageio.imwrite("./third.mp4", images[args.start:args.end])