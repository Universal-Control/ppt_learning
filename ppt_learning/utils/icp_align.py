from matplotlib import pyplot as plt
import open3d as o3d
import copy
import numpy as np
from pathlib import Path

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])

def perform_icp_align(points, colors, initial_transform_src_to_tgt, visualize=False,
                    max_iteration=10000, threshold=0.01, evaluate=True):
    """
    Args:
    points: list[np.ndarray] (N, 3) list of points, expect 2 pcds
    colors: list[np.ndarray] (N, 3) list of colors, expect 2 pcds
    initial_transform_src_to_tgt: np.ndarray (4, 4) from src_pcd to tgt_pcd
    visualize: bool while to use open3d to visualize the pcds before and after icp
    max_iteration: int, max iteration for icp, increase this number to get better result but slower
    threshold: float, threshold for icp, decrease this number to get better result but slower
    evaluate: bool while to use open3d to evaluate the init transform before the icp 
    Return
    transform_src_to_tgt: np.ndarray (4, 4) from src_pcd to tgt_pcd after icp align
    """
    assert len(points) == len(colors) == 2, "expect 2 pcds!"
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(points[0])
    src_pcd.colors = o3d.utility.Vector3dVector(colors[0])
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(points[1])
    tgt_pcd.colors = o3d.utility.Vector3dVector(colors[1])
    
    if visualize:
        draw_registration_result(src_pcd, tgt_pcd, initial_transform_src_to_tgt)
    print("Initial alignment")
    print(initial_transform_src_to_tgt)
    transform_src_to_tgt = initial_transform_src_to_tgt

    if max_iteration > 1:
        if evaluate:
            cam_2_marker_evaluation = o3d.pipelines.registration.evaluate_registration(
                src_pcd, tgt_pcd, threshold, initial_transform_src_to_tgt)
            print(cam_2_marker_evaluation)
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, threshold, initial_transform_src_to_tgt,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        transform_src_to_tgt = reg_p2p.transformation

    if visualize:
        draw_registration_result(src_pcd, tgt_pcd, reg_p2p.transformation)

    src_pcd = src_pcd.transform(transform_src_to_tgt)
    points = [np.asarray(src_pcd.points), np.asarray(tgt_pcd.points)]
    colors = [np.asarray(src_pcd.colors), np.asarray(tgt_pcd.colors)]

    return transform_src_to_tgt, points, colors

def perform_colored_icp_align(points, colors, initial_transform_src_to_tgt, visualize=False,
                    max_iteration=[3000, 3000, 3000], voxel_radius=[0.01, 0.01, 0.01], evaluate=True):
    """
    Args:
    points: list[np.ndarray] (N, 3) list of points, expect 2 pcds
    colors: list[np.ndarray] (N, 3) list of colors, expect 2 pcds
    initial_transform_src_to_tgt: np.ndarray (4, 4) from src_pcd to tgt_pcd
    visualize: bool while to use open3d to visualize the pcds before and after icp
    max_iteration: list[int] (3,), max iteration for icp, increase this number to get better result but slower
    voxel_radius: list[float] (3,), threshold for icp, decrease this number to get better result but slower
    evaluate: bool while to use open3d to evaluate the init transform before the icp 
    Return
    transform_src_to_tgt: np.ndarray (4, 4) from src_pcd to tgt_pcd after icp align
    """
    assert len(points) == len(colors) == 2, "expect 2 pcds!"
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(points[0])
    src_pcd.colors = o3d.utility.Vector3dVector(colors[0])
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(points[1])
    tgt_pcd.colors = o3d.utility.Vector3dVector(colors[1])
    
    if visualize:
        draw_registration_result_original_color(src_pcd, tgt_pcd, initial_transform_src_to_tgt)
    print("Initial alignment")
    print(initial_transform_src_to_tgt)
    transform_src_to_tgt = initial_transform_src_to_tgt

    if max_iteration.max() > 1:
        if evaluate:
            cam_2_marker_evaluation = o3d.pipelines.registration.evaluate_registration(
                src_pcd, tgt_pcd, min(voxel_radius), initial_transform_src_to_tgt)
            print(cam_2_marker_evaluation)
        for scale in range(3):
            iter = max_iteration[scale]
            radius = voxel_radius[scale]
            print([iter, radius, scale])

            print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = src_pcd.voxel_down_sample(radius)
            target_down = tgt_pcd.voxel_down_sample(radius)

            print("3-2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            print("3-3. Applying colored point cloud registration")
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=iter))
            current_transformation = result_icp.transformation
            print(result_icp)

    if visualize:
        draw_registration_result_original_color(src_pcd, tgt_pcd, result_icp.transformation)

    src_pcd = src_pcd.transform(transform_src_to_tgt)
    points = [np.asarray(src_pcd.points), np.asarray(tgt_pcd.points)]
    colors = [np.asarray(src_pcd.colors), np.asarray(tgt_pcd.colors)]

    return transform_src_to_tgt, points, colors



if __name__ == "__main__":
    pass
