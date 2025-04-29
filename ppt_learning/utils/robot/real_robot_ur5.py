import time
import threading
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
from ppt_learning.paths import PPT_DIR
import rtde_control
import rtde_receive
from typing import List
# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# TODO fill in these values
hostname = ""
cameras = {
    "camera_0": "943222070526",
    "camera_1": "233622074344",
}  # wrist, left, right cam

class UR5_IK:
    def __init__(self,
            urdf_path = f"{PPT_DIR}/third_party/ur5_isaac_simulation/robot.urdf",
            init_q: np.ndarray = None,
            ee_link_name: str = "",
            base_link_name: str = ""
    ):
        """初始化 UR5 IK 求解器"""
        # Essential parameters
        self.tensor_args = TensorDeviceType(device=torch.device("cuda"))
        self.ee_link_name = ee_link_name if ee_link_name else "wrist_3_link"
        self.base_link_name = base_link_name if base_link_name else "base_link"
        self.filter_state = None
        
        # Create robot config and IK solver
        self.robot_cfg = RobotConfig.from_basic(
            urdf_path, self.base_link_name, self.ee_link_name, self.tensor_args
        )
        
        # Setup default configuration
        self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)

        self.dof = self.kin_model.get_dof()
        self.full_q0 = np.zeros(self.dof)
        if self.dof >= 6:  # Assuming standard UR5 joint layout
            self.full_q0[0] = -1.57
            self.full_q0[1] = -1.57
        
        # Use provided initial configuration if available
        if init_q is not None:
            assert len(init_q) == len(self.full_q0), f"Initial position length mismatch"
            self.full_q0 = np.copy(init_q)
            
        # Configure and initialize IK solver
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            position_threshold=0.005,
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            high_precision=False,
            use_cuda_graph=True,
            grad_iters=None,
            regularization=False,
        )
        
        self.ik_solver = IKSolver(self.ik_config)
        self.last_shape = None
        self.last_dtype = None
    
    def solve_batch(
        self,
        target_pos,
        target_quat,
        joints_batch,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """批量IK问题并行求解"""
        if target_pos.shape != self.last_shape or target_pos.dtype!= self.last_dtype:
            self.last_shape = target_pos.shape
            self.last_dtype = target_pos.dtype
            infer_mode = False
        else:
            infer_mode = True
        # with torch.inference_mode(infer_mode):
        pose = Pose(target_pos, target_quat)
        result = self.ik_solver.solve_batch(pose, retract_config=joints_batch, seed_config=joints_batch[None], return_seeds=1)
        q_solution: torch.Tensor = result.solution
        return q_solution[:, 0, :]


class RealRobot:
    def __init__(self, ip="192.168.1.243", fps=30, init_q=None, 
                control_space='joint'):
        print("Intializing robot ...")

        # Initialize robot connection and libraries
        self.robot_c = rtde_control.RTDEControlInterface(ip)
        self.robot_r = rtde_receive.RTDEReceiveInterface(ip)
        self.gripper = None # Gripper TODO
        self.control_space = control_space
        self.dt = 1. / fps
        self.velocity = 0.5
        self.acceleration = 0.5
        self.lookahead_time = 0.1
        self.gain = 300
        self.init_q = init_q
        if self.init_q is None:
            self.init_q = [
                -3.02692452,
                -2.00677957,
                -1.50796447,
                -1.12242124,
                1.59191481,
                -0.055676
            ]

        self.ik_helper = UR5_IK(
            init_q = self.init_q,
            ee_link_name="wrist_3_link"
        )
        

        self.init_robot()
        self.init_cameras() # 

        # other
        self.current_step = 0
        self.horizon = 50  # TODO
        self._buffer = {}

        print("Finished initializing robot.")
        self._last_command_j = None

    def init_cameras(self):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # get initial pose
        init_pose = self.tcp_pose

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
        joint_pose = self.init_q
        self.robot_c.moveJ(joint_pose)
        # self.gripper.move(width=0.0, speed=0.1) # Gripper TODO

        # replicate in sim
        action = np.zeros((8,))
        action[:-1] = self.tcp_pose

    def log_pose(self, verbose=False):
        while True:
            start_time = time.time()
            pose = self.tcp_pose
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
        joint = self.robot_r.getActualQ()
        joint_torch = torch.tensor(joint, device="cuda").reshape(1, 6)
        ee_torch = self.ik_helper.kin_model.get_state(joint_torch)
        
        pose = np.ascontiguousarray(np.concatenate([
            ee_torch.ee_position.cpu().numpy()[0], ee_torch.ee_quaternion.cpu().numpy()[0]
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
        ee_pose = self.tcp_pose
        obs = {
            "eef_pos": ee_pose[:3],
            "eef_quat": ee_pose[3:],
            "joint_pos": np.array(self.robot_r.getActualQ() + [0]),
            "joint_vel": np.array(self.robot_r.getActualQd() + [0]),
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
        pcd = {"pos": pcds[..., :3], "color": pcds[..., 3:]}

        if visualize:
            vis_pcd(pcd)

        state = self.get_robot_state()

        obs = {"state": state, "pointcloud": pcd}

        return obs

    def step(self, action, visualize=False):
        """
        Step robot in the real.
        """
        import time
        start_time = time.time()
        current_joint_np = self.robot_r.getActualQ()
        current_joint = torch.tensor(current_joint_np, device="cuda").reshape(1, 6)
        joint = self.ik_helper.solve_batch(
            torch.tensor(action[:3], device="cuda").to(torch.float32),
            torch.tensor(action[3:-1], device="cuda").to(torch.float32),
            current_joint.to(torch.float32)
        )[0].cpu().numpy()
        end_time = time.time()
        print("time to solve ik {}".format(end_time - start_time))
        try:
            t_start = self.robot_c.initPeriod()
            self.robot_c.servoJ(
                joint.tolist(), self.velocity, self.acceleration,
                self.dt, self.lookahead_time, self.gain
            )
            self.robot_c.waitPeriod(t_start)
            # self.robot.movej(joint, vel=.4, acc=0.4)
            # self.robot.servoj(joint, vel=0, acc=0, t=0.008, lookahead_time=0.01, gain=300, threshold=0.1)
            # self.robot.speedx("speedj", (joint - current_joint_np)/0.2, acc=0.4, min_time=0.2)
            # time.sleep(1 / self.fps - (time.time() - end_time))
            print("time to move robot {}".format(time.time() - end_time))
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
        # self.robot.stop()
        pass


if __name__ == "__main__":
    data = RealRobot()

    t2 = threading.Thread(target=data.log_pose)
    t2.start()

    data.test_sequence()
    t2.join()
