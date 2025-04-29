from ppt_learning.utils.robot.real_robot_ur5 import RealRobot
import pickle
import numpy as np
import time

def main():
    rb = RealRobot()

    print(rb.get_robot_state())
    a = rb.get_obs(visualize=False)

    default_action = np.array(
        [0.6534, -0.0329,  0.2565,  0.0346, -0.7644,  0.6436, -0.0181, 0.0]
    )
    x = input("continue move?(y/n)")
    if x == "y":
        for i in range(100):
            rb.step(default_action + [0, i * 0.01, 0, 0, 0, 0, 0, 0])
            time.sleep(0.03)


if __name__ == "__main__":
    main()