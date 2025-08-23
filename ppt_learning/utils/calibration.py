import numpy as np
import open3d as o3d
import cv2
import copy
from pathlib import Path

def load_data(root_path, rgb_path, depth_path, camera_id, depth_threshold=2., visualize=True):
    root_path = Path(root_path)
    only_table_intrinsics = np.load(root_path / "intrinsics.npz", allow_pickle=True)
    # print(cv2.imread(str(rgb_only_table_path)))
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = np.load(str(depth_path))
    intrinsics = get_intr_matrix(only_table_intrinsics, camera_id)
    pcd, pcd_np, color_filtered = depth_to_pointcloud(rgb, depth, intrinsics, depth_threshold)
    if visualize:
        to_vis = []
        to_vis.append(pcd)
        frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        to_vis.append(frame_base)
        o3d.visualization.draw_geometries(to_vis)
    return color_filtered, depth, pcd, pcd_np, intrinsics

def depth_to_pointcloud(rgb_image, depth_map, intrinsics, depth_max_threshold=1.3):
    """
    Convert RGB image and depth map to colored point cloud
    
    Parameters:
        rgb_image: RGB image, numpy array (H, W, 3)
        depth_map: Depth map, numpy array (H, W)
        intrinsics: Camera intrinsic matrix (3, 3)
    
    Returns:
        point_cloud: Open3D point cloud object
    """
    # Get image dimensions
    height, width = depth_map.shape
    
    # Create pixel coordinate grid
    v, u = np.mgrid[0:height, 0:width]
    
    # Reshape to column vectors
    u = u.reshape(-1)
    v = v.reshape(-1)
    depth = depth_map.reshape(-1)
    
    # Filter invalid depth values (0 or negative)
    valid_indices = np.logical_and(depth > 0,  depth < depth_max_threshold)
    u = u[valid_indices]
    v = v[valid_indices]
    depth = depth[valid_indices]
    
    # Get camera intrinsics
    fx = intrinsics[0, 0]  # focal length x
    fy = intrinsics[1, 1]  # focal length y
    cx = intrinsics[0, 2]  # optical center x
    cy = intrinsics[1, 2]  # optical center y
    
    # Calculate 3D coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # Create point cloud
    points = np.vstack((x, y, z)).T
    
    # Get colors from RGB image
    colors = rgb_image.reshape(-1, 3)[valid_indices] / 255.0
    
    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud, points, colors

def get_intr_matrix(only_table_intrinsics, camera_id):
    """
    Get camera intrinsic matrix
    """
    fx = only_table_intrinsics["arr_0"][camera_id]["fx"]
    fy = only_table_intrinsics["arr_0"][camera_id]["fy"]
    cx = only_table_intrinsics["arr_0"][camera_id]["ppx"]
    cy = only_table_intrinsics["arr_0"][camera_id]["ppy"]
    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
    return intrinsics   


def load_obj_as_pointcloud(obj_file_path, sample_points=0):
    """
    Load 3D model from OBJ file and convert to point cloud
    
    Parameters:
        obj_file_path: OBJ file path
        sample_points: Number of sample points, if 0 use all vertices, if >0 sample the mesh
        
    Returns:
        point_cloud: Open3D point cloud object
    """
    print(f"Loading OBJ file: {obj_file_path}")
    
    # Load OBJ file as mesh
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    
    # Ensure mesh has normals, compute if not available
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    print(f"Mesh loading completed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Get point cloud from mesh
    if sample_points > 0:
        print(f"Sampling mesh: {sample_points} points")
        point_cloud = mesh.sample_points_uniformly(number_of_points=sample_points)
    else:
        # Create point cloud using mesh vertices
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = mesh.vertices
        point_cloud.colors = mesh.vertex_colors if mesh.has_vertex_colors() else None
        point_cloud.normals = mesh.vertex_normals if mesh.has_vertex_normals() else None
    
    print(f"Point cloud creation completed: {len(point_cloud.points)} points")
    
    return point_cloud

def estimate_pose(image, charuco_dict, intrinsics_matrix, dist_coeffs, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(charuco_dict, detector_params)
    corners, ids, _ = aruco_detector.detectMarkers(image)

    charucodetector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(image)
    # print('Detected markers: ', ids)
    print("detect {} markers".format(len(marker_corners)))
    if charuco_ids is not None and len(charuco_corners) > 3:
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, intrinsics_matrix, dist_coeffs, None, None)
        if valid:
            R_target2cam = cv2.Rodrigues(rvec)[0]
            t_target2cam = tvec.reshape(3, 1)
            target2cam = np.eye(4)
            target2cam[:3, :3] = R_target2cam
            target2cam[:3, 3] = t_target2cam.reshape(-1)
            local_frame_change_transform = np.zeros((4, 4))
            local_frame_change_transform[1, 0] = 1
            local_frame_change_transform[0, 1] = 1
            local_frame_change_transform[2, 2] = -1
            local_frame_change_transform[3, 3] = 1
            return local_frame_change_transform @ np.linalg.inv(target2cam)
    return None

def draw_charuco_origin(image, charuco_dict, intrinsics_matrix, dist_coeffs, board, axis_length=0.05):
    """
    Draw the Charuco marker coordinate system origin and axes on the given image.
    This function's interface matches `estimate_pose`.

    Parameters:
    - image: The original image on which the axes will be drawn.
    - charuco_dict: The Charuco dictionary used for detection.
    - intrinsics_matrix: The camera intrinsic matrix.
    - dist_coeffs: The distortion coefficients of the camera.
    - board: The Charuco board object.
    - axis_length: The length of the axes to be drawn in the real-world coordinate system. Default is 0.05 (meters).

    Returns:
    - The image with the axes drawn on it if pose estimation is successful, otherwise the original image.
    """
    # Estimate pose using the same method as the `estimate_pose` function
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(charuco_dict, detector_params)
    corners, ids, _ = aruco_detector.detectMarkers(image)

    charucodetector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(image)

    if charuco_ids is not None and len(charuco_corners) > 3:
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, intrinsics_matrix, dist_coeffs, None, None)
        if valid:
            # Define the 3D points of the axes in the Charuco coordinate system
            axis_points = np.float32([
                [0, 0, 0],  # Origin
                [axis_length, 0, 0],  # X-axis (red)
                [0, axis_length, 0],  # Y-axis (green)
                [0, 0, axis_length]   # Z-axis (blue)
            ]).reshape(-1, 3)

            # Project the 3D points to the 2D image plane
            image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, intrinsics_matrix, dist_coeffs)

            # Convert points to integer for visualization
            image_points = np.int32(image_points).reshape(-1, 2)

            # Draw the axes on the image
            image = cv2.line(image, tuple(image_points[0]), tuple(image_points[1]), (0, 0, 255), 2)  # X-axis (red)
            image = cv2.line(image, tuple(image_points[0]), tuple(image_points[2]), (0, 255, 0), 2)  # Y-axis (green)
            image = cv2.line(image, tuple(image_points[0]), tuple(image_points[3]), (255, 0, 0), 2)  # Z-axis (blue)

            # Draw the origin point
            image = cv2.circle(image, tuple(image_points[0]), 5, (0, 255, 255), -1)  # Origin (yellow)

    return image