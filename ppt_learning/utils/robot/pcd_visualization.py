from ppt_learning.utils.robot.real_robot_ur5 import RealRobot
import open3d as o3d
import numpy as np
import time
import tyro

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # 手动调整视角，按q退出
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()

def visualize_and_capture(pos, color, camera_params_path, output_image_path, visualize=False):
    # 创建点云对象并设置数据
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    
    # 颜色归一化处理（假设输入颜色范围0-255）
    if np.max(color) > 1.0:
        color = color.astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(color)

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    
    # 添加几何体
    vis.add_geometry(pcd)
    
    # 设置视角参数
    params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
    view_ctl = vis.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(params)
    
    # 更新渲染
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # 捕获并保存图像（添加延迟确保渲染完成）
    vis.capture_screen_image(output_image_path, do_render=True)

    if visualize:
        vis.run()
    
    # 关闭窗口
    vis.destroy_window()

def hand_save_view_point():
    robot = RealRobot(camera_only=True)
    obs = robot.get_obs()
    pos = obs["pointcloud"]["pos"]
    color = obs["pointcloud"]["color"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.colors = o3d.utility.Vector3dVector(color)
    save_view_point(pcd, "/home/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json")
    print('=' * 80)
    print("Done")

def general_capture(
    obs,
    view_point="/home/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json",
    output_image="/home/minghuan/ppt_learning/ppt_learning/utils/test/general_capture.png",
    tag="general_capture",
    visualize=False,
):
    pos = obs["pointcloud"]["pos"]
    color = obs["pointcloud"]["color"]
    visualize_and_capture(pos, color, view_point,
                        output_image, visualize=visualize)
    print('=' * 80)
    print(f"Saved {tag} pic")

def capture_no_model_dense(
    view_point="/home/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json",
    output_image_dir="/home/minghuan/ppt_learning/ppt_learning/utils/test/",
    tag="dense_no_model",
    visualize=True,
):
    robot = RealRobot(camera_only=True, npoints=120000)
    obs = robot.get_obs()
    general_capture(obs, view_point, output_image_dir+f"{tag}.png", tag, visualize)

    obs = robot.get_obs(post_icp=True, online_icp=True)
    general_capture(obs, view_point, output_image_dir+f"{tag}_icp.png", tag + "-icp", visualize)
    robot.stop()

def capture_model_dense(
    view_point="/home/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json",
    output_image_dir="/home/minghuan/ppt_learning/ppt_learning/utils/test/",
    tag="model dense",
    visualize=True,
):
    robot = RealRobot(camera_only=True, depth_model_path="/home/minghuan/ppt_learning/models/depth/720p-e049-s204800.ckpt", use_model_depth=True)
    obs = robot.get_obs(num_points=120000)
    general_capture(obs, view_point, output_image_dir+f"{tag}.png", tag, visualize)

    obs = robot.get_obs(num_points=120000, post_icp=True, online_icp=True)
    general_capture(obs, view_point, output_image_dir+f"{tag}_icp.png", tag + "-icp", visualize)

    robot.stop()

def capture_model_sparse(
    view_point="/home/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json",
    output_image_dir="/home/minghuan/ppt_learning/ppt_learning/utils/test/",
    tag="sparse_model",
    visualize=False,
):
    robot = RealRobot(camera_only=True, depth_model_path="/home/minghuan/ppt_learning/models/depth/720p-e049-s204800.ckpt", use_model_depth=True)
    general_capture(obs, view_point, output_image_dir+f"{tag}.png", tag, visualize)

    obs = robot.get_obs(post_icp=True, online_icp=True)
    general_capture(obs, view_point, output_image_dir+f"{tag}_icp.png", tag + "-icp", visualize)

    robot.stop()

def capture_no_model_sparse(
    view_point="/home/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json",
    output_image_dir="/home/minghuan/ppt_learning/ppt_learning/utils/test/",
    tag="sparse_no_model",
    visualize=False,
):
    robot = RealRobot(camera_only=True, use_model_depth=False)
    obs = robot.get_obs()
    general_capture(obs, view_point, output_image_dir+"sparse_no_model.png", tag, visualize)
    
    obs = robot.get_obs(post_icp=True, online_icp=True)
    general_capture(obs, view_point, output_image_dir+"sparse_no_model_icp.png", tag + "-icp", visualize)

    robot.stop()

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {   
            "capture_no_model_dense": capture_no_model_dense,
            "capture_model_dense": capture_model_dense,
            "capture_model_sparse": capture_model_sparse,
            "capture_no_model_sparse": capture_no_model_sparse,
            "hand_save_view_point": hand_save_view_point,
        }
    )