"""底盘小幅横移测试：发 [VX, VY, WZ] 持续 DURATION 秒，然后发零速停车；用轮子转角估算实际位移。
用法: python - <vx> <vy> <wz> <duration>
轮速关系来自实测的 /chassis/joint_cmd：w1 = -5vx + 8.66vy, w2 = -5vx - 8.66vy, w3 = 10vx（轮半径 0.1 m）。"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

VX, VY, WZ, DURATION = (float(x) for x in sys.argv[1:5])
assert abs(VX) <= 0.1 and abs(VY) <= 0.1 and abs(WZ) <= 0.2 and 0 < DURATION <= 30, "outside test limits"
RATE = 20.0

rclpy.init()
node = Node("chassis_step")
pub = node.create_publisher(Float32MultiArray, "/api/chassis/velocity", 10)
wheel = {}
WRAP = 8 * np.pi  # 轮子转角读数在 ±4π 处回绕


def on_wheel(m):
    raw = np.array(m.position[:3])
    if "raw" in wheel:
        step = (raw - wheel["raw"] + WRAP / 2) % WRAP - WRAP / 2
        wheel["pos"] = wheel["pos"] + step  # 展开后的累计转角
    else:
        wheel["pos"] = raw.copy()
    wheel["raw"] = raw
    wheel["vel"] = np.array(m.velocity[:3])


node.create_subscription(JointState, "/chassis/joint_states", on_wheel, 10)


def send(v, seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        pub.publish(Float32MultiArray(data=[float(x) for x in v]))
        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(1.0 / RATE)


def spin(seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.02)


def estimate(d):
    x = (d[2] - 0.5 * (d[0] + d[1])) * 0.1 * 2.0 / 3.0
    y = (d[0] - d[1]) / (2 * 8.66)
    return x, y, float(d.sum() / 3.0)


try:
    deadline = time.monotonic() + 10.0
    while "pos" not in wheel or pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() > deadline:
            raise RuntimeError("no wheel states or no chassis subscriber")
    spin(0.5)
    p0 = wheel["pos"].copy()
    print("WHEEL_START", np.round(p0, 4).tolist(), flush=True)
    try:
        send((VX, VY, WZ), DURATION)
    finally:
        send((0.0, 0.0, 0.0), 1.0)
    p_stop = wheel["pos"].copy()
    spin(1.5)
    p1 = wheel["pos"].copy()
    d = p1 - p0
    x, y, rot = estimate(d)
    print("WHEEL_END", np.round(p1, 4).tolist(), "residual vel", np.round(wheel["vel"], 3).tolist(), flush=True)
    print("WHEEL_DELTA_RAD", np.round(d, 4).tolist(), "drift after zero cmd", np.round(p1 - p_stop, 4).tolist(), flush=True)
    print(f"COMMANDED vx {VX:+.3f} vy {VY:+.3f} wz {WZ:+.3f} for {DURATION:.2f}s -> nominal dx {VX*DURATION*100:+.1f} cm dy {VY*DURATION*100:+.1f} cm", flush=True)
    print(f"ESTIMATED dx {x*100:+.2f} cm dy {y*100:+.2f} cm, mean wheel rotation {rot:+.4f} rad (0 = no yaw)", flush=True)
except KeyboardInterrupt:
    send((0.0, 0.0, 0.0), 1.0)
    print("INTERRUPTED: zero velocity sent", flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
