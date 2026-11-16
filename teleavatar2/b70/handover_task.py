"""人机递交任务（单进程）：手里的沙包递到人手上。使用 B70 上的递交模型（127.0.0.1:8001，容器 ta2-handover-5907）。

流程：
1. （手臂下垂时）后退 -> 抬臂到递交起始姿态 -> 回位；手臂已抬起时直接插值到起始姿态。夹爪张开。
2. 夹爪保持张开 --open-s 秒（默认 5 s），这段时间由人把沙包放进右夹爪；然后闭合夹紧。
3. 启动前自检：做一次推理，模型给的第一个目标离当前姿态太远、或模型认为夹爪应张开，就不启动。
   偏差中等时先慢速过渡到模型给的姿态，避免绝对关节角目标造成手臂突跳。
4. pi0.5 递交回合：夹爪保持夹紧，直到模型输出"张开"（力矩高于 OPEN_ABOVE 连续 3 步）才松开并锁住张开。
5. 松开后再跑 --after-release-s 秒，然后手臂慢速收回起始姿态。

起始姿态取自训练数据 put_224_v1 全部 161 个回合第一帧的均值（各关节标准差 0.03-0.14 rad）。
示范回合长 154-317 帧（45 fps，约 3.4-7 s）：前半段夹着沙包伸出，松手后手臂自己收回。

一键启动： ./run_handover.sh      停止： ./stop_task.sh 或在启动终端按 Ctrl+C
"""
import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/app")
from openpi_client import action_chunk_broker  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402
from openpi_client.runtime.agents import policy_agent  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float32, Float32MultiArray  # noqa: E402

from examples.teleavatar_v2 import env as _env  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--open-s", type=float, default=5.0, help="到起始姿态后夹爪张开等待放沙包的秒数")
p.add_argument("--max-steps", type=int, default=600, help="递交回合步数上限（45 Hz）；示范最长 317 帧，实测模型最晚第 362 步才松手")
p.add_argument("--after-release-s", type=float, default=3.0)
p.add_argument("--back-cm", type=float, default=30.0, help="手臂下垂时，抬臂前后退的距离")
p.add_argument("--speed", type=float, default=0.10)
p.add_argument("--port", type=int, default=8001)
p.add_argument("--img-dir", default="/tmp/claude_task")
p.add_argument("--tag", default="")
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S", force=True)
T0 = time.monotonic()


def log(msg):
    logging.info(f"T+{time.monotonic() - T0:6.1f}s  {msg}")


PROMPT = "Hand the sandbag to the person."
# 递交起始姿态：训练数据 put_224_v1（B70 /root/ta2_lerobot/put_224_v1，161 个回合）第一帧关节角的均值
START_L = np.array([-0.036, 0.764, -0.704, 1.913, 0.374, 0.732, 0.304])
START_R = np.array([0.072, -0.752, 0.549, -1.880, -0.402, -0.433, -0.425])
START14 = np.concatenate([START_L, START_R])
OPEN_NM, CLOSE_NM = 2.0, -1.6
OPEN_ABOVE, OPEN_STEPS = -0.4, 3     # 模型右夹爪力矩高于此值连续 N 步 -> 松开
RAMP_OK, RAMP_ABORT = 0.15, 1.0      # 启动自检：第一个目标与当前姿态的最大关节偏差 (rad)
K = 30.0 / 50.02
CTRL_HZ = 45.0

env = _env.TeleavatarEnvironment(prompt=PROMPT, control_frequency=CTRL_HZ, video_decode="va", enable_chassis=False)
iface = env._ros_interface
chassis_pub = iface.create_publisher(Float32MultiArray, "/api/chassis/velocity", 10)
wheel = {}
WRAP = 8 * np.pi


def _on_wheel(m):
    raw = np.array(m.position[:3])
    if "raw" in wheel:
        wheel["pos"] = wheel["pos"] + ((raw - wheel["raw"] + WRAP / 2) % WRAP - WRAP / 2)
    else:
        wheel["pos"] = raw.copy()
    wheel["raw"] = raw


iface.create_subscription(JointState, "/chassis/joint_states", _on_wheel, 10)
os.makedirs(args.img_dir, exist_ok=True)


def wheel_xy(d):
    """轮子转角变化 -> 按 K 修正后的 (右移, 前进) 厘米"""
    return K * (d[2] - 0.5 * (d[0] + d[1])) * 0.1 * 2 / 3 * 100, K * (d[0] - d[1]) / 17.32 * 100


def chassis(vx, vy):
    chassis_pub.publish(Float32MultiArray(data=[float(vx), float(vy), 0.0]))


def stop_chassis(seconds=0.3):
    t = time.monotonic()
    while time.monotonic() - t < seconds:
        chassis(0, 0)
        time.sleep(0.02)


def drive(name, right_cm=0.0, fwd_cm=0.0, tail=0.3):
    """开环平移：时长 = 距离 / (K x 速度) + 起步补偿；结束后发 tail 秒零速。"""
    dist = abs(right_cm) + abs(fwd_cm)
    if dist < 0.3:
        return
    v = args.speed if dist >= 15 else 0.05
    dur = dist / 100 / (K * v) + (0.08 if v > 0.07 else 0.06)
    vx, vy = v * np.sign(right_cm) if right_cm else 0.0, v * np.sign(fwd_cm) if fwd_cm else 0.0
    p0 = wheel["pos"].copy()
    log(f"{name}: vx {vx:+.2f} vy {vy:+.2f} for {dur:.2f}s")
    t = time.monotonic()
    try:
        while time.monotonic() - t < dur:
            chassis(vx, vy)
            time.sleep(0.02)
    finally:
        stop_chassis(tail)
    x, y = wheel_xy(wheel["pos"] - p0)
    log(f"{name}: done, wheel estimate right {x:+.1f} cm forward {y:+.1f} cm")


def arm_now():
    return iface._current_arm_positions()


def snapshot(tag):
    try:
        img = np.asarray(env.get_observation()["observation/images/head_camera"])
        np.save(os.path.join(args.img_dir, f"{tag}.npy"), img[::2, ::2])
    except Exception as exc:
        log(f"snapshot {tag} failed: {exc!r}")


def command(arm14, grip_l, grip_r):
    a = np.zeros(16)
    a[0:7], a[7], a[8:15], a[15] = arm14[:7], grip_l, arm14[7:], grip_r
    iface.publish_action(a)


def move_arm(target14, grip_r, seconds, hold=0.0):
    start = iface._last_cmd_pos.copy()
    t = time.monotonic()
    while (el := time.monotonic() - t) < seconds + hold:
        al = min(el / seconds, 1.0) if seconds > 0 else 1.0
        command(start + (target14 - start) * al, OPEN_NM, grip_r)
        time.sleep(1 / CTRL_HZ)


def raise_direct(target14, ramp=8.0):
    """下垂姿态略超出 arm_config 限位，不能走带限幅的通道：直接发布关节指令，从实测姿态直线插值到目标，夹爪张开。"""
    start = arm_now()
    names = {"left_arm": iface.left_joint_names, "right_arm": iface.right_joint_names}
    t = time.monotonic()
    while True:
        al = min((time.monotonic() - t) / ramp, 1.0)
        pos = start + (target14 - start) * al
        iface.enable_pub.publish(Float32(data=1.0))
        stamp = iface.get_clock().now().to_msg()
        for arm, sl in (("left_arm", slice(0, 7)), ("right_arm", slice(7, 14))):
            m = JointState()
            m.header.stamp, m.header.frame_id, m.name = stamp, arm, names[arm]
            m.position, m.velocity, m.effort = [float(x) for x in pos[sl]], [0.0] * 7, [0.0] * 7
            iface.action_publishers[arm].publish(m)
        for g in ("left_gripper", "right_gripper"):
            iface.action_publishers[g].publish(Float32(data=0.0))
        if al >= 1.0 and np.abs(arm_now() - target14).max() < 0.05:
            return
        if time.monotonic() - t > ramp + 6:
            raise RuntimeError("arms did not reach the start pose")
        time.sleep(0.01)


status = 1
try:
    policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    t = time.monotonic()
    policy.infer(env.get_observation())  # 预热，结果丢弃
    log(f"handover policy connected (port {args.port}), warm-up inference {time.monotonic() - t:.2f}s")
    while "pos" not in wheel or chassis_pub.get_subscription_count() == 0:
        if time.monotonic() - T0 > 20:
            raise RuntimeError("no wheel states or no chassis subscriber")
        time.sleep(0.05)

    # 1. 到起始姿态，夹爪张开
    err = float(np.abs(arm_now() - START14).max())
    hanging = float(np.abs(arm_now()[[3, 10]]).max()) < 0.6  # 肘关节接近伸直 = 重启后的下垂姿态
    if hanging:
        log(f"arms hanging (max err {err:.2f} rad): back up, raise, return")
        drive("back", fwd_cm=-args.back_cm)
        raise_direct(START14)
        command(START14, OPEN_NM, OPEN_NM)
        drive("return", fwd_cm=args.back_cm)
    else:
        log(f"arms raised (max err {err:.2f} rad): move to the start pose")
        command(arm_now(), OPEN_NM, OPEN_NM)
        time.sleep(0.1)
        move_arm(START14, OPEN_NM, max(3.0, err / 0.25))
    log(f"at start pose, max err {np.abs(arm_now() - START14).max():.3f} rad")

    # 2. 张开等待放沙包，然后夹紧
    for left in range(int(round(args.open_s)), 0, -1):
        log(f"GRIPPER OPEN - put the sandbag in the right gripper now ({left}s left)")
        move_arm(START14, OPEN_NM, 0.0, hold=1.0)
    log("gripper closing")
    move_arm(START14, CLOSE_NM, 0.0, hold=1.5)
    snapshot("h0_holding")

    # 3. 启动自检
    a0 = np.array(policy.infer(env.get_observation())["actions"], dtype=np.float64)[0]
    tgt = np.concatenate([a0[0:7], a0[8:15]])
    dev = np.abs(tgt - arm_now())
    log(f"pre-check: first target vs current pose max {dev.max():.2f} rad (joint index {int(dev.argmax())}), model right gripper effort {a0[15]:+.2f} Nm")
    log(f"pre-check: model first target L {np.round(tgt[:7], 3).tolist()} R {np.round(tgt[7:], 3).tolist()}")
    if a0[15] > OPEN_ABOVE:
        raise RuntimeError("model wants the gripper open at the start (it does not see a held sandbag); not started, gripper stays closed")
    if dev.max() > RAMP_ABORT:
        raise RuntimeError(f"model's first target is {dev.max():.2f} rad from the start pose; not started, gripper stays closed")
    if dev.max() > RAMP_OK:
        secs = max(2.0, dev.max() / 0.25)
        log(f"pre-check: slow transition to the model's first target over {secs:.1f}s")
        move_arm(tgt, CLOSE_NM, secs, hold=0.3)

    # 4. 递交回合
    agent = policy_agent.PolicyAgent(policy=action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=16))
    log("handover: episode starts")
    released, above, t_rel, last_log = False, 0, None, 0.0
    step, nxt = 1 / CTRL_HZ, time.monotonic()
    for i in range(args.max_steps):
        a = np.array(agent.get_action(env.get_observation())["actions"], dtype=np.float64)
        now, raw = time.monotonic(), float(a[15])
        if not released:
            above = above + 1 if raw > OPEN_ABOVE else 0
            if above >= OPEN_STEPS:
                released, t_rel = True, now
                log(f"handover: gripper RELEASE at step {i} (model effort {raw:+.2f} Nm), right arm target {np.round(a[8:15], 3).tolist()}")
        if now - last_log >= 0.5:
            last_log = now
            log(f"  step {i:3d} model right gripper {raw:+.2f} Nm -> {'OPEN' if released else 'CLOSED'}; right arm j1 {a[8]:+.2f} j2 {a[9]:+.2f} j4 {a[11]:+.2f}")
        a[7], a[15] = OPEN_NM, (OPEN_NM if released else min(raw, CLOSE_NM))
        iface.publish_action(a)
        if released and now - t_rel > args.after_release_s:
            log(f"handover: {args.after_release_s:.0f}s after release, episode ends at step {i}")
            break
        nxt += step
        d = nxt - time.monotonic()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.monotonic()
    snapshot("h1_after_episode")
    if not released:
        raise RuntimeError("step limit reached and the model never opened the gripper; arm holds its pose, gripper stays closed")

    # 5. 收回
    log("retract to the start pose")
    move_arm(START14, OPEN_NM, 5.0, hold=0.5)
    log("TASK DONE")
    status = 0
except KeyboardInterrupt:
    log("INTERRUPTED")
finally:
    try:
        stop_chassis(0.5)
    finally:
        env.close()
        log("shutdown complete")
sys.exit(status)
