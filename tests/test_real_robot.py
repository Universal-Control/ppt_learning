from ppt_learning.real.real_robot_ur5 import RealRobot
import pickle
import numpy as np
import time
import rtde_control
import rtde_receive
from ppt_learning.utils.icp_align import perform_icp_align

ip = "192.168.1.243"

def main():
    rb = RealRobot(depth_model_path="/home/minghuan/ppt_learning/models/depth/480pnoise-e096-s397312.ckpt.jit", use_model_depth=True)

    # print(rb.get_robot_state())
    # rb.get_obs(visualize=True, post_icp=True)
    for i in range(10):
        rb.get_obs()

    default_action = np.array(
        [0.6534, -0.0329,  0.2565,  0.0346, -0.7644, 0.6436, -0.0181, 0.0]
    )
    # x = input("continue move?(y/n)")
    # if x == "y":
        # for i in range(20):
        #     rb.step(default_action + [0, i * 0.01, 0, 0, 0, 0, 0, 0])
        #     time.sleep(0.03)
        # a = rb.get_obs(visualize=True)
        # for i in range(10):
        #     rb.step(default_action + [0, 19 * 0.01, i * 0.01, 0, 0, 0, 0, 0])
        #     time.sleep(0.03)
    while True:
        a = rb.get_obs(visualize=True)
        
# def main2():
#     import URBasic as ub

#     robot_model = ub.RobotModel()

#     robot_ext = ub.UrScriptExt(ip, robot_model)
#     robot_ext.init_realtime_control()
#     print("-----")
#     print(robot_ext.get_actual_tcp_pose())
#     print("done")

def main2():
    rtde_c = rtde_control.RTDEControlInterface(ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    print(type(rtde_r.getActualQ()))

    # # Parameters
    # velocity = 0.5
    # acceleration = 0.5
    # dt = 1.0/500  # 2ms
    # lookahead_time = 0.1
    # gain = 300
    # joint_q = [
    #             -3.02692452,
    #             -2.00677957,
    #             -1.50796447,
    #             -1.12242124,
    #             1.59191481,
    #             -0.055676]

    # # Move to initial joint position with a regular moveJ
    # rtde_c.moveJ(joint_q)

    # # Execute 500Hz control loop for 2 seconds, each cycle is 2ms
    # for i in range(100):
    #     t_start = rtde_c.initPeriod()
    #     rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)
    #     joint_q[0] += 0.001
    #     joint_q[1] += 0.001
    #     rtde_c.waitPeriod(t_start)

    # rtde_c.servoStop()
    # rtde_c.stopScript()
    print("done")



if __name__ == "__main__":
    main()