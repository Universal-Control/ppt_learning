import rtde_control

ip = "192.168.1.243"

rtde_c = rtde_control.RTDEControlInterface(ip)
rtde_c.moveL([-0.143, -0.435, 0.20, -0.001, 3.12, 0.04], 0.5, 0.3)