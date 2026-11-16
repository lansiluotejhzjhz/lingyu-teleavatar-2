"""把双臂从当前姿态线性插值到训练数据 episode 0 第一帧姿态（夹爪张开），收敛后退出。
逻辑沿用 examples/teleavatar_v2/zero_from_episode.py，只是目标写死（客户端容器里没挂数据集）。"""
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

TARGET = {
    "left_arm": np.array([0.096, 0.854, -0.808, 1.763, 0.450, 0.617, 0.540]),
    "right_arm": np.array([-0.037, -0.808, 0.679, -1.685, -0.268, -0.409, -0.620]),
}
GRIPPER_TRIGGER = 0.0  # 0.0 = 全开（effort +2.0 Nm）
RAMP_S, TOL, TIMEOUT_S = 8.0, 0.05, 10.0

cfg = yaml.safe_load(open("/app/arm_config.yml"))["arms"]
for arm in TARGET:
    lo, hi = np.array(cfg[arm]["lower"]), np.array(cfg[arm]["upper"])
    assert ((TARGET[arm] >= lo) & (TARGET[arm] <= hi)).all(), f"{arm} target outside joint limits"

rclpy.init()
node = Node("goto_first_frame")
pubs = {
    "left_arm": node.create_publisher(JointState, "/api/left_arm/joint_cmd", 10),
    "right_arm": node.create_publisher(JointState, "/api/right_arm/joint_cmd", 10),
    "left_gripper": node.create_publisher(Float32, "/api/left_gripper/cmd", 10),
    "right_gripper": node.create_publisher(Float32, "/api/right_gripper/cmd", 10),
    "enable": node.create_publisher(Float32, "/api/fsm/enable", 10),
}
names = {"left_arm": [f"l_joint{i}" for i in range(1, 8)], "right_arm": [f"r_joint{i}" for i in range(1, 8)]}
current = {}
node.create_subscription(JointState, "/left_arm/joint_states",
                         lambda m: current.update(left_arm=np.array(m.position[:7])), 10)
node.create_subscription(JointState, "/right_arm/joint_states",
                         lambda m: current.update(right_arm=np.array(m.position[:7])), 10)


def publish(pos):
    pubs["enable"].publish(Float32(data=1.0))
    stamp = node.get_clock().now().to_msg()
    for arm in ("left_arm", "right_arm"):
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = arm
        msg.name = names[arm]
        msg.position = [float(p) for p in pos[arm]]
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        pubs[arm].publish(msg)
    for g in ("left_gripper", "right_gripper"):
        pubs[g].publish(Float32(data=GRIPPER_TRIGGER))


def max_err():
    return max(np.abs(current[a] - TARGET[a]).max() for a in TARGET)


try:
    deadline = time.monotonic() + 10.0
    while len(current) < 2:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() > deadline:
            raise RuntimeError("no joint states")
    start = {a: current[a].copy() for a in TARGET}
    print("START L", np.round(start["left_arm"], 3).tolist(), "R", np.round(start["right_arm"], 3).tolist(), flush=True)
    t0 = time.monotonic()
    while (el := time.monotonic() - t0) < RAMP_S:
        al = el / RAMP_S
        publish({a: start[a] * (1 - al) + TARGET[a] * al for a in TARGET})
        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(0.01)
    since, t0, ok = None, time.monotonic(), False
    while time.monotonic() - t0 < TIMEOUT_S:
        publish(TARGET)
        rclpy.spin_once(node, timeout_sec=0)
        if max_err() < TOL:
            since = since or time.monotonic()
            if time.monotonic() - since >= 0.5:
                ok = True
                break
        else:
            since = None
        time.sleep(0.01)
    print("CONVERGED" if ok else "NOT_CONVERGED", "max_err", round(float(max_err()), 3), flush=True)
    print("END   L", np.round(current["left_arm"], 3).tolist(), "R", np.round(current["right_arm"], 3).tolist(), flush=True)
except KeyboardInterrupt:
    print("INTERRUPTED: stopped publishing; robot holds the last commanded pose", flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
