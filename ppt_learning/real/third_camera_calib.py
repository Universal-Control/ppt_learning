from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import copy

from ppt_learning.paths import PPT_DIR
from ppt_learning.utils.calibration import *
from ppt_learning.utils.icp_align import *

os.environ["DISPLAY"] = ":0"
# ROOT = f"{PPT_DIR}/../logs/photos/2025-05-08_00-37-25" # No model
ROOT = f"{PPT_DIR}/../logs/photos/2025-05-08_17-00-08-model" # model

def init_aruco_marker(main_image_with_aruco_marker, main_camera_intr, charuco_dict=None, board=None):
    if charuco_dict is None:
        charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    if board is None:
        board = cv2.aruco.CharucoBoard((8, 5), 0.048, 0.038, charuco_dict)
    image = cv2.imread(main_image_with_aruco_marker)
    cam_2_marker_init = estimate_pose(image, charuco_dict, main_camera_intr, np.zeros(5), board)

    return cam_2_marker_init, charuco_dict, board

def visualize_marker(main_image_with_aruco_marker, charuco_dict, main_camera_intr, board):
    """
    Visualize the charuco origin on the image with aruco marker
    """
    image = cv2.imread(main_image_with_aruco_marker)
    draw_image = draw_charuco_origin(image, charuco_dict, main_camera_intr, np.zeros(5), board)
    while True:
        cv2.imshow("Charuco Origin", draw_image)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def calibration(root_path, time_stamp, main_camera_id, slave_camera_id, load_slave=False, slave_only=False, visualize=False, save_calib=False, slave_path=""):
    """
    1. Load data
    2. Init aruco marker
    3. Align slave to main
    4. Align main to table
    """

    main_rgb_only_table_path = f"{root_path}/color_{main_camera_id}/{time_stamp}.png"
    main_depth_only_table_path = f"{root_path}/depth_{main_camera_id}/{time_stamp}.npy"
    output_path = Path(f"{root_path}/output_pcds")
    main_camera_to_table_npy_path = f"{PPT_DIR}/../logs/photos/2025-05-07_11-34-20/camera_2_table.npy"
    main_image_with_aruco_marker = f"{root_path}/color_{main_camera_id}/{time_stamp}.png"
    slave_image_with_aruco_marker = f"{root_path}/color_{slave_camera_id}/{time_stamp}.png"
    slave_to_camera_npy_path = f"{slave_path}/camera_0_to_camera_1.npy"
    slave_rgb_only_table_path = f"{root_path}/color_{slave_camera_id}/{time_stamp}.png"
    slave_depth_only_table_path = f"{root_path}/depth_{slave_camera_id}/{time_stamp}.npy"
    gt_obj_file_path = f"{PPT_DIR}/../logs/photos/2025-05-07_11-34-20/gt/combine-5-7.stl"

    # Load data
    main_camera_rgb, main_camera_depth, main_camera_pcd, main_camera_pcd_np, main_camera_intr = load_data(root_path, main_rgb_only_table_path, main_depth_only_table_path, main_camera_id, visualize=visualize)
    slave_camera_rgb, slave_camera_depth, slave_camera_pcd, slave_camera_pcd_np, slave_camera_intr = load_data(root_path, slave_rgb_only_table_path, slave_depth_only_table_path, slave_camera_id, visualize=visualize)
    if load_slave:
        slave_to_camera_npy = np.load(slave_to_camera_npy_path)

    # Init aruco
    main_cam_2_marker_init, charuco_dict, board = init_aruco_marker(main_image_with_aruco_marker, main_camera_intr)
    slave_cam_2_marker_init, _, __ = init_aruco_marker(slave_image_with_aruco_marker, slave_camera_intr, charuco_dict, board)
    
    if not load_slave:
        # Align slave to main
        slave_to_main_init = np.linalg.inv(main_cam_2_marker_init) @ slave_cam_2_marker_init
        slave_to_camera_npy, _, __ = perform_icp_align( # perform_color_icp_align
            [slave_camera_pcd, main_camera_pcd],
            slave_to_main_init,
            visualize=visualize,
            max_iteration=20000,
            threshold=0.001,
        )
    if save_calib:
        np.save(Path(slave_rgb_only_table_path).parent.parent / f"camera_{slave_camera_id}_to_camera_{main_camera_id}.npy", slave_to_camera_npy)

    if slave_only:
        return slave_to_camera_npy, slave_to_camera_npy
    
    # Get gt
    gt_pcd = load_obj_as_pointcloud(gt_obj_file_path, sample_points=50000)
    # translation = np.array([(0.3395, 0, 0.529)]) 
    # 创建平移矩阵
    # transform = np.eye(4)  # 创建4x4单位矩阵
    # transform[:3, 3] = translation  # 设置平移部分
    # 应用变换
    # gt_pcd.transform(transform)
    # transform = np.eye(4)  # 创建4x4单位矩阵
    # transform[:3, :3] = R.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
    # gt_pcd.transform(transform)
    if visualize:
        vis = []
        frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.append(gt_pcd)
        vis.append(frame_base)
        o3d.visualization.draw_geometries(vis)

    # Align main to table
    slave_camera_pcd_np_in_main_camera = (slave_to_camera_npy[:3, :3] @ slave_camera_pcd_np.T + slave_to_camera_npy[:3, 3][:, None]).T
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate((main_camera_pcd_np, slave_camera_pcd_np_in_main_camera), axis=0))
    main_to_table_npy, _, __ = perform_icp_align( # perform_color_icp_align
        [merged_pcd, gt_pcd],
        main_cam_2_marker_init,
        visualize=visualize,
        max_iteration=20000,
        threshold=0.0001,
    )
    
    if save_calib:
        np.save(Path(main_rgb_only_table_path).parent.parent / "main_camera_2_table.npy", main_to_table_npy)

    return main_to_table_npy, slave_to_camera_npy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slave_only", action="store_true", help="only calibrate slave camera")
    parser.add_argument("--load_slave", action="store_true", help="load slave camera calibration")
    parser.add_argument("--slave_path", type=str, default=ROOT, help="path of the slave camera calibration")
    parser.add_argument("--visualize", action="store_true", help="visualize the calibration process")
    parser.add_argument("--save_calib", action="store_true", help="save the calibration result")
    parser.add_argument("--save_board", action="store_true", help="save the aruco board")

    parser.add_argument("--photo", type=str, default=ROOT, help="root path of the data")

    args = parser.parse_args()

    if args.save_board:
        charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard((8, 5), 0.028, 0.021, charuco_dict)
        bd_img = board.generateImage((3840, 2160))
        cv2.imwrite("charuco_board.png", bd_img)
        exit(0)

    color_files = os.listdir(f"{args.photo}/color_0")
    color_files.sort()
    time_stamps = [int(color_file.split(".")[0]) for color_file in color_files]
    
    main_to_table_npys, slave_to_camera_npys = [], []

    for time_stamp in time_stamps:
        main_to_table_npy, slave_to_camera_npy = calibration(
            args.photo,
            time_stamp,
            main_camera_id=1,
            slave_camera_id=0,
            visualize=args.visualize,
            save_calib=args.save_calib,
            slave_only=args.slave_only,
            load_slave=args.load_slave,
            slave_path=args.slave_path
        )
        main_to_table_npys.append(main_to_table_npy)
        slave_to_camera_npys.append(slave_to_camera_npy)
    
    # Analyse the variance and the mean of the main_to_table_npys and slave_to_camera_npys
    main_to_table_translations, main_to_table_rotations = [], []
    slave_to_camera_translations, slave_to_camera_rotations = [], []

    for main_to_table_npy, slave_to_camera_npy in zip(main_to_table_npys, slave_to_camera_npys):
        main_to_table_translations.append(main_to_table_npy[:3, 3])
        main_to_table_rotations.append(R.from_matrix(main_to_table_npy[:3, :3]).as_euler("xyz"))
        slave_to_camera_translations.append(slave_to_camera_npy[:3, 3])
        slave_to_camera_rotations.append(R.from_matrix(slave_to_camera_npy[:3, :3]).as_euler("xyz"))

    if args.save_calib:
        np.save(Path(args.photo)/ "main_camera_2_table_all.npy", main_to_table_npys)
        np.save(Path(args.photo) / "slave_camera_2_main_all.npy", slave_to_camera_npys)

    main_to_table_translations = np.array(main_to_table_translations)
    main_to_table_rotations = np.array(main_to_table_rotations)

    slave_to_camera_translations = np.array(slave_to_camera_translations)
    slave_to_camera_rotations = np.array(slave_to_camera_rotations)

    main_to_table_translation_mean = np.mean(main_to_table_translations, axis=0)
    main_to_table_translation_std = np.std(main_to_table_translations, axis=0)
    main_to_table_rotation_mean = np.mean(main_to_table_rotations, axis=0)
    main_to_table_rotation_std = np.std(main_to_table_rotations, axis=0)

    slave_to_camera_translation_mean = np.mean(slave_to_camera_translations, axis=0)
    slave_to_camera_translation_std = np.std(slave_to_camera_translations, axis=0)
    slave_to_camera_rotation_mean = np.mean(slave_to_camera_rotations, axis=0)
    slave_to_camera_rotation_std = np.std(slave_to_camera_rotations, axis=0)

    print("main_to_table_translation_mean", main_to_table_translation_mean)
    print("main_to_table_translation_std", main_to_table_translation_std)
    print("main_to_table_rotation_mean", main_to_table_rotation_mean)
    print("main_to_table_rotation_std", main_to_table_rotation_std)

    print("slave_to_camera_translation_mean", slave_to_camera_translation_mean)
    print("slave_to_camera_translation_std", slave_to_camera_translation_std)
    print("slave_to_camera_rotation_mean", slave_to_camera_rotation_mean)
    print("slave_to_camera_rotation_std", slave_to_camera_rotation_std)

    # Construct the mean transformation matrix
    main_to_table_mean = np.eye(4)
    main_to_table_mean[:3, 3] = main_to_table_translation_mean
    main_to_table_mean[:3, :3] = R.from_euler('xyz', main_to_table_rotation_mean, degrees=True).as_matrix()

    slave_to_camera_mean = np.eye(4)
    slave_to_camera_mean[:3, 3] = slave_to_camera_translation_mean
    slave_to_camera_mean[:3, :3] = R.from_euler('xyz', slave_to_camera_rotation_mean, degrees=True).as_matrix()

    if args.save_calib:
        np.save(Path(args.photo) / "main_camera_2_table_mean.npy", main_to_table_mean)
        np.save(Path(args.photo) / "slave_camera_2_main_mean.npy", slave_to_camera_mean)