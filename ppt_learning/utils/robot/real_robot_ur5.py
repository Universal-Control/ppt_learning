import panda_py
from panda_py import libfranka

import time
import threading
import transforms3d
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R

import multiprocessing as mp

from multiprocessing import Process, Queue, Event
from multiprocessing.managers import SharedMemoryManager

from ppt_learning.utils.camera.multi_cam import MultiRealsense
from ppt_learning.utils.shared_memory.shared_memory_queue import SharedMemoryQueue
from ppt_learning.utils.calibration import *
from ppt_learning.utils.robot.utils import *

from ppt_learning.utils.pcd_utils import (
    uniform_sampling,
    fps_sampling,
    pcd_filter_bound,
    BOUND,
)
from ppt_learning.utils.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from ppt_learning.utils.pcd_utils import *
from ppt_learning.utils.calibration import *
from ppt_learning.paths import PPT_DIR, ASSET_ROOT
import urx

# TODO fill in these values
hostname = ""
cameras = {
    "wrist_cam": "",
    "right_cam": "",
    "left_cam": "",
}  # wrist, left, right cam


class RealRobot:
    def __init__(self, ip="192.168.1.243", fps=30, init_pose=None, 
                control_space='joint'):
        print("Intializing robot ...")

        # Initialize robot connection and libraries
        self.robot = urx.Robot(ip)
        self.gripper = None # Gripper TODO
        self.panda.enable_logging(int(10))
        self.control_space = control_space
        self.fps = fps
        self.init_pose = init_pose
        if self.init_pose is None:
            self.init_pose = [
                -3.02692452,
                -2.00677957,
                -1.50796447,
                -1.12242124,
                1.59191481,
                -0.055676
            ]

        self.init_robot()
        # self.init_cameras() # 

        # other
        self.current_step = 0
        self.horizon = 50  # TODO
        self._buffer = {}

        print("Finished initializing robot.")

    def init_cameras(self):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # get initial pose
        init_pose = self.panda.get_pose()

        # same for time
        init_time = time.time()

        self.pose_buffer = SharedMemoryRingBuffer.create_from_examples(
            self.shm_manager,
            examples={
                "pose": init_pose,
                "timestamp": init_time,
            },
            get_time_budget=0.002,
            get_max_k=100,
        )

        for i in range(100):
            self.pose_buffer.put(
                {
                    "pose": init_pose,
                    "timestamp": init_time,
                },
                wait=True,
            )

        self.realsense = MultiRealsense(
            serial_numbers=cameras,
            shm_manager=self.shm_manager,
            resolution=(640, 480),
            capture_fps=30,
            put_fps=None,
            pose_buffer=self.pose_buffer,
        )

        self.realsense.start(wait=True)

        while not self.realsense.is_ready:
            time.sleep(0.1)

    def init_robot(self):
        joint_pose = self.init_pose
        self.robot.get_realtime_monitor()
        self.robot.movej(joint_pose)
        # self.gripper.move(width=0.0, speed=0.1) # Gripper TODO

        # replicate in sim
        action = np.zeros((8,))
        action[:-1] = joint_pose

    def log_pose(self, verbose=False):
        while True:
            start_time = time.time()
            p = self.robot.get_pose()
            pose = np.ascontiguousarray(np.concatenate([
                p.pos.array, p.orientation.quaternion.array
            ])).astype(np.float32)
            init_time = time.time()

            data = {
                "pose": pose,
                "timestamp": init_time,
            }

            self.pose_buffer.put(data)

            elapsed_time = time.time() - start_time
            if elapsed_time < 0.001:
                time.sleep(0.001 - elapsed_time)

    @property
    def tcp_pose(self):
        p = self.robot.get_pose()
        pose = np.ascontiguousarray(np.concatenate([
            p.pos.array, p.orientation.quaternion.array
        ])).astype(np.float32)
        return pose

    def get_robot_state(self):
        """
        Get the real robot state.
        """
        # Gripper TODO
        # gripper_state = self.gripper.read_once()
        # gripper_qpos = gripper_state.width
        gripper_qpos = 0.0
        data = self.robot.rtmon.get_all_data()
        p = data["tcp"]
        obs = {
            "eef_pos": p.pos.array,
            "eef_quat": p.orientation.quaternion.array,
            "joint_pos": np.concatenate([data["qActual"], np.array([0])], axis=-1),
            "joint_vel": np.concatenate([data["qdActual"], np.array([0])], axis=-1),
        }
        return obs

    def get_obs(self, visualize=False):
        """
        Get the real robot observation.

        visualize: bool, whether to visualize the pointcloud.
        """
        # TODO Mix pcds
        pcds = self.realsense.get()
        pcds = pcds["pcds"]
        pcd = {"pos": pcds[..., :3], "colors": pcds[..., 3:]}

        if visualize:
            vis_pcd(pcds)

        state = self.get_robot_state()

        obs = {"state": state, "pointcloud": pcd}

        return obs

    def step(self, action, visualize=False):
        """
        Step robot in the real.
        """
        # Simple motion in cartesian space
        gripper = action[-1]
        quat = action[3:-1]

        pose = np.concatenate([action[:3], quat], axis=0)
        print(pose)

        try:

            self.robot.movel(pose)
            # self.gripper.move(width=gripper, speed=0.3) # Gripper TODO

        except Exception as e:
            print(e)
            print("Failed to generate valid waypoints.")

        return self.get_obs(visualize=visualize)

    def test_sequence(self):
        """
        Test sequence of actions to test the robot.
        """
        for i in range(10):
            joint_pose = [
                -3.02692452,
                -2.00677957,
                -1.50796447,
                -1.12242124,
                1.59191481,
                -0.055676 + i * 0.04
            ]


            self.get_robot_state()


    def end(self):
        self.robot.stop()


if __name__ == "__main__":
    data = RealRobot()

    t2 = threading.Thread(target=data.log_pose)
    t2.start()

    data.test_sequence()
    t2.join()
