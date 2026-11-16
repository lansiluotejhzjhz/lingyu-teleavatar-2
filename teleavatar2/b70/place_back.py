"""把右手里的沙包放回桌面：回到第一帧姿态（夹爪保持闭合）-> 右臂伸到上次抓取位置上方 -> 张开夹爪 -> 收回第一帧姿态。"""
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

FIRST = {
    "left_arm": np.array([0.096, 0.854, -0.808, 1.763, 0.450, 0.617, 0.540]),
    "right_arm": np.array([-0.037, -0.808, 0.679, -1.685, -0.268, -0.409, -0.620]),
}
REACH_R = np.array([-0.576, -1.663, 1.1, -0.542, -0.969, -0.392, -0.641])  # 本轮夹爪闭合时的右臂目标
PLACE = {"left_arm": FIRST["left_arm"], "right_arm": 0.2 * FIRST["right_arm"] + 0.8 * REACH_R}  # 停在其上方一点
CLOSED, OPEN = 1.0, 0.0  # 右夹爪 trigger
# (目标姿态, 右夹爪 trigger, 插值秒数, 到位后保持秒数)
STEPS = [(FIRST, CLOSED, 6.0, 0.5), (PLACE, CLOSED, 6.0, 0.5), (PLACE, OPEN, 0.0, 2.5), (FIRST, OPEN, 6.0, 0.5)]

cfg = yaml.safe_load(open("/app/arm_config.yml"))["arms"]
for pose in (FIRST, PLACE):
    for arm in pose:
        lo, hi = np.array(cfg[arm]["lower"]), np.array(cfg[arm]["upper"])
        assert ((pose[arm] >= lo) & (pose[arm] <= hi)).all(), f"{arm} target outside joint limits"

rclpy.init()
node = Node("place_back")
pubs = {
    "left_arm": node.create_publisher(JointState, "/api/left_arm/joint_cmd", 10),
    "right_arm": node.create_publisher(JointState, "/api/right_arm/joint_cmd", 10),
    "left_gripper": node.create_publisher(Float32, "/api/left_gripper/cmd", 10),
    "right_gripper": node.create_publisher(Float32, "/api/right_gripper/cmd", 10),
    "enable": node.create_publisher(Float32, "/api/fsm/enable", 10),
}
names = {"left_arm": [f"l_joint{i}" for i in range(1, 8)], "right_arm": [f"r_joint{i}" for i in range(1, 8)]}
cur, grip = {}, {}
node.create_subscription(JointState, "/left_arm/joint_states", lambda m: cur.update(left_arm=np.array(m.position[:7])), 10)
node.create_subscription(JointState, "/right_arm/joint_states", lambda m: cur.update(right_arm=np.array(m.position[:7])), 10)
node.create_subscription(JointState, "/right_gripper/joint_states", lambda m: grip.update(right=float(m.position[0])), 10)


def publish(pos, trig):
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
    pubs["left_gripper"].publish(Float32(data=OPEN))
    pubs["right_gripper"].publish(Float32(data=trig))


def run(seconds, fn, trig):
    t0 = time.monotonic()
    while (el := time.monotonic() - t0) < seconds:
        publish(fn(el), trig)
        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(0.01)


try:
    deadline = time.monotonic() + 10.0
    while len(cur) < 2 or not grip:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() > deadline:
            raise RuntimeError("no joint states")
    last = {a: cur[a].copy() for a in FIRST}  # 从实测姿态出发
    print("START R", np.round(last["right_arm"], 3).tolist(), "right gripper", round(grip["right"], 3), flush=True)
    for i, (target, trig, ramp, hold) in enumerate(STEPS):
        start = last
        if ramp > 0:
            run(ramp, lambda el: {a: start[a] + (target[a] - start[a]) * min(el / ramp, 1.0) for a in target}, trig)
        run(hold, lambda el: target, trig)
        last = {a: np.array(target[a], dtype=float) for a in target}
        err = max(np.abs(cur[a] - target[a]).max() for a in target)
        print(f"STEP {i} done: max_err {err:.3f} right gripper {grip['right']:.3f}", flush=True)
    print("DONE", flush=True)
except KeyboardInterrupt:
    print("INTERRUPTED: stopped publishing; robot holds the last commanded pose", flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
