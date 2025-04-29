from __future__ import division

import time
import math3d as m3d
from math import pi

import urx
import pynput
import numpy as np

ControlVec = np.array([0., 0, 0, 0, 0, 0])
SCALE = 0.01
EULERSCALE = 0.1

def key_board_command(key):
    global ControlVec
    ControlVec = np.array([0., 0, 0, 0, 0, 0])
    if str(key) == "'d'":
        # right
        ControlVec[0] += SCALE
    elif str(key) == "'a'":
        # left
        ControlVec[0] -= SCALE
    elif str(key) == "'w'":
        # forward
        ControlVec[1] -= SCALE
    elif str(key) == "'s'":
        # back 
        ControlVec[1] += SCALE
    elif str(key) == "'q'":
        # up 
        ControlVec[2] += SCALE
    elif str(key) == "'e'":
        # down 
        ControlVec[2] -= SCALE
    elif str(key) == "'i'":
        ControlVec[3] += EULERSCALE
    elif str(key) == "'k'":
        ControlVec[3] -= EULERSCALE
    elif str(key) == "'j'":
        ControlVec[4] -= EULERSCALE
    elif str(key) == "'l'":
        ControlVec[4] += EULERSCALE
    elif str(key) == "'u'":
        ControlVec[5] -= EULERSCALE
    elif str(key) == "'o'":
        ControlVec[5] += EULERSCALE 

class Cmd(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0
        self.btn0 = 0
        self.btn1 = 0

    def get_speeds(self):
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]



class Service(object):
    def __init__(self, robot):
        self.robot = robot
        self.lin_coef = 5000
        self.rot_coef = 5000
        self.l = pynput.keyboard.Listener(on_press=key_board_command)
        self.l.start()

    def loop(self):
        ts = 0 
        btn0_state = 0
        btn_event = None
        cmd = Cmd()
        global ControlVec
        while True:
            time.sleep(0.01)
            cmd.reset()
            cmd.x = ControlVec[0]
            cmd.y = ControlVec[1]
            cmd.z = ControlVec[2]
            cmd.rx = ControlVec[3]
            cmd.ry = ControlVec[4]
            cmd.rz = ControlVec[5]
            
            if (time.time() - ts) > 0.12:
                ts = time.time()
                speeds = cmd.get_speeds()
                if btn0_state:
                    self.robot.speedl_tool(speeds, acc=0.10, min_time=2)
                else:
                    self.robot.speedx("speedl", speeds, acc=0.10, min_time=2)
                btn_event = None
                speeds = cmd.get_speeds()
                #if speeds != [0 for _ in speeds]:
                print("Sending", speeds)


if __name__ == '__main__':
    ip = "192.168.1.243"
    robot = urx.Robot(ip)
    #robot = urx.Robot("localhost")
    # robot.set_tcp((0, 0, 0.27, 0, 0, 0))
    trx = m3d.Transform()
    trx.orient.rotate_zb(pi/4)
    robot.set_csys(trx)
    service = Service(robot)
    try:
        service.loop()
    finally:
        robot.close()



