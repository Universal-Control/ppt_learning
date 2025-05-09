import pickle
import os

os.environ["DISPLAY"] = ":0"
import open3d as o3d

with open("/home/minghuan/ppt_learning/test.pkl", "rb") as b:
    data = pickle.load(b)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data["pointcloud"]["pos"])
to_vis = [pcd]
o3d.visualization.draw_geometries(to_vis)
