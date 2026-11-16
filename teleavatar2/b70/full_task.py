"""完整任务（单进程，各环节无缝衔接）：
后退 -> 抬臂到第一帧姿态 -> 回位 -> 左移到抓取位置 -> pi0.5 抓取沙包 -> 保持手臂右移 -> 停留 -> 再右移 -> 放下沙包。

相对分脚本运行的优化：
- 相机、ROS 接口、策略连接在最开始建立一次，并做一次预热推理；到抓取位置后立即开始推理。
- 抓取回合在"夹爪已闭合 + 手臂收回并稳定"时立即结束，不再空等到最大步数。
- 回合结束后接口的 200 Hz 定时器继续保持手臂和夹爪，底盘立刻开始移动；底盘各段之间只留 0.3 s 减速。
- 到抓取位置后用头部相机里沙包的位置做一次有上限的底盘微调（可用 --no-align 关闭）。

一键启动： ./run_task.sh（含启动前检查、日志和图像回收）；中途停止： ./stop_task.sh 或在 run_task.sh 里按 Ctrl+C
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
p.add_argument("--back-cm", type=float, default=30.0)
p.add_argument("--left-cm", type=float, default=144.0)
p.add_argument("--right-cm", type=float, default=72.0)
p.add_argument("--dwell-s", type=float, default=15.0)
p.add_argument("--speed", type=float, default=0.10, help="底盘指令速度 m/s")
p.add_argument("--no-align", action="store_true", help="不做基于沙包图像位置的底盘微调")
p.add_argument("--img-dir", default="/tmp/claude_task")
p.add_argument("--tag", default="", help="仅用于在进程列表里识别本任务（stop_task.sh 按它发 SIGINT）")
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S", force=True)
T0 = time.monotonic()


def log(msg):
    logging.info(f"T+{time.monotonic() - T0:6.1f}s  {msg}")


PROMPT = "Pick up the sandbag from the table."
FIRST_L = np.array([0.096, 0.854, -0.808, 1.763, 0.450, 0.617, 0.540])
FIRST_R = np.array([-0.037, -0.808, 0.679, -1.685, -0.268, -0.409, -0.620])
FIRST14 = np.concatenate([FIRST_L, FIRST_R])
OPEN_NM, CLOSE_NM, CLOSE_BELOW = 2.0, -1.6, -0.8  # 夹爪力矩 (Nm) 和闭合阈值
K = 30.0 / 50.02  # 实际位移 / (指令速度 x 时长)，2026-11-14 横向卷尺实测
CTRL_HZ = 45.0
# 头部相机 (960x960) 里沙包的参考位置（此前抓取成功时的位置）和像素/厘米换算（由底盘小幅移动前后的图像估出）
REF_XY, PX_PER_CM, ALIGN_DEADBAND_CM, ALIGN_MAX_CM = (483.0, 726.0), (10.0, 8.0), 1.5, 10.0

assert 0 < args.speed <= 0.1 and 0 <= args.back_cm <= 40 and 0 <= args.left_cm <= 150 and 0 <= args.right_cm <= 80

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
        return img
    except Exception as exc:  # 存图失败不影响任务
        log(f"snapshot {tag} failed: {exc!r}")
        return None


def raise_arms(ramp=8.0):
    """从实测姿态直线插值到第一帧姿态，夹爪张开。直接发布（下垂姿态略超出 arm_config 限位，不能走带限幅的通道）。"""
    start = arm_now()
    err = float(np.abs(start - FIRST14).max())
    if err < 0.05:
        log(f"raise arms: already at first-frame pose (max err {err:.3f}), skipped")
    else:
        log(f"raise arms: ramp {ramp:.0f}s from max err {err:.2f} rad")
        names = {"left_arm": iface.left_joint_names, "right_arm": iface.right_joint_names}
        t = time.monotonic()
        while True:
            al = min((time.monotonic() - t) / ramp, 1.0)
            pos = start + (FIRST14 - start) * al
            iface.enable_pub.publish(Float32(data=1.0))
            stamp = iface.get_clock().now().to_msg()
            for arm, sl in (("left_arm", slice(0, 7)), ("right_arm", slice(7, 14))):
                m = JointState()
                m.header.stamp, m.header.frame_id, m.name = stamp, arm, names[arm]
                m.position, m.velocity, m.effort = [float(x) for x in pos[sl]], [0.0] * 7, [0.0] * 7
                iface.action_publishers[arm].publish(m)
            for g in ("left_gripper", "right_gripper"):
                iface.action_publishers[g].publish(Float32(data=0.0))
            if al >= 1.0 and np.abs(arm_now() - FIRST14).max() < 0.05:
                break
            if time.monotonic() - t > ramp + 6:
                raise RuntimeError("arms did not reach the first-frame pose")
            time.sleep(0.01)
        log(f"raise arms: done, max err {np.abs(arm_now() - FIRST14).max():.3f}")
    command(FIRST14, OPEN_NM, OPEN_NM)  # 交给接口的 200 Hz 定时器保持


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


def find_sandbag(img):
    """头部相机里红色沙包的中心（全分辨率像素），找不到返回 None。只看桌面区域。"""
    y0, y1, x0, x1 = 500, 860, 300, 760
    r, g, b = (img[y0:y1, x0:x1, i].astype(int) for i in range(3))
    ys, xs = np.nonzero((r > 120) & (g < 100) & (b < 100) & (r - g > 60))
    if len(xs) < 150:
        return None
    return float(np.median(xs)) + x0, float(np.median(ys)) + y0


def align_to_sandbag():
    img = snapshot("3_at_table")
    if args.no_align or img is None:
        return
    c = find_sandbag(img)
    if c is None:
        log("align: sandbag not found in the head image, no correction")
        return
    right, back = (c[0] - REF_XY[0]) / PX_PER_CM[0], (c[1] - REF_XY[1]) / PX_PER_CM[1]
    log(f"align: sandbag at ({c[0]:.0f}, {c[1]:.0f}) px, reference {REF_XY} -> robot should move right {right:+.1f} cm, back {back:+.1f} cm")
    if max(abs(right), abs(back)) > ALIGN_MAX_CM:
        log("align: offset larger than the limit, no correction")
        return
    did = False
    if abs(back) > ALIGN_DEADBAND_CM:
        drive("align fwd/back", fwd_cm=-back)
        did = True
    if abs(right) > ALIGN_DEADBAND_CM:
        drive("align left/right", right_cm=right)
        did = True
    if did:
        time.sleep(0.3)
        img2 = snapshot("3b_aligned")
        c2 = None if img2 is None else find_sandbag(img2)
        log(f"align: sandbag now at {None if c2 is None else (round(c2[0]), round(c2[1]))} px")


# 抓住后模型把右臂收回到的"持物"姿态（2026-11-14/15 多次成功抓取后的实测均值）
HOLD_R = np.array([0.14, -0.76, 0.735, -2.01, -0.78, -0.74, -0.65])


def grasp(agent, max_steps=800, near_hold=0.15, near_time=0.3, max_after_close=12.0):
    """pi0.5 抓取回合。返回 (是否闭合, 闭合时的右臂目标)。夹爪带阈值和锁存。
    提前结束判据：夹爪闭合后，右臂实测关节角进入持物姿态 near_hold rad 以内并保持 near_time 秒。
    （原判据"指令目标 1 s 内变化 < 0.02 rad"不触发：模型每个动作块之间的指令抖动比这个阈值大。）"""
    closed = [False, False]
    t_close, reach, t_near, trace = None, None, None, []
    step, nxt = 1 / CTRL_HZ, time.monotonic()

    def finish(why, i):
        tr = np.array(trace)
        np.save(os.path.join(args.img_dir, "grasp_trace.npy"), tr)
        if len(tr) > 50:  # 说明旧判据为何不触发：闭合后指令目标在 1 s 窗口内的最小抖动幅度
            n = int(CTRL_HZ)
            spans = [np.ptp(tr[k:k + n, 1:8], axis=0).max() for k in range(0, len(tr) - n)]
            log(f"grasp: commanded-target jitter over 1 s windows after closing: min {min(spans):.3f} rad" if spans else "grasp: short trace")
        log(f"grasp: {why}, episode ends at step {i}, {time.monotonic() - t_close:.1f}s after closing")
        return True, reach

    for i in range(max_steps):
        a = np.array(agent.get_action(env.get_observation())["actions"], dtype=np.float64)
        now = time.monotonic()
        for j, k in ((0, 7), (1, 15)):
            if not closed[j] and a[k] < CLOSE_BELOW:
                closed[j] = True
                log(f"grasp: {'left' if j == 0 else 'right'} gripper CLOSE at step {i} (model effort {a[k]:+.2f} Nm)")
                if j == 1:
                    t_close, reach = now, a[8:15].copy()
            a[k] = min(a[k], CLOSE_NM) if closed[j] else OPEN_NM
        iface.publish_action(a)
        if t_close is not None:
            meas = arm_now()[7:]
            trace.append(np.concatenate([[now - t_close], a[8:15], meas]))
            dist = float(np.abs(meas - HOLD_R).max())
            if dist < near_hold and now - t_close > 1.5:
                t_near = t_near or now
                if now - t_near >= near_time:
                    return finish(f"right arm back at the holding pose (max joint error {dist:.2f} rad)", i)
            else:
                t_near = None
            if now - t_close > max_after_close:
                return finish(f"time limit, right arm {dist:.2f} rad from the holding pose", i)
        nxt += step
        d = nxt - time.monotonic()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.monotonic()
    log("grasp: max steps reached without closing the right gripper")
    return False, None


status = 1
try:
    policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)
    t = time.monotonic()
    policy.infer(env.get_observation())  # 预热，结果丢弃
    log(f"policy connected, warm-up inference {time.monotonic() - t:.2f}s")
    agent = policy_agent.PolicyAgent(policy=action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=16))
    while "pos" not in wheel or chassis_pub.get_subscription_count() == 0:
        if time.monotonic() - T0 > 20:
            raise RuntimeError("no wheel states or no chassis subscriber")
        time.sleep(0.05)
    snapshot("0_start")

    drive("back", fwd_cm=-args.back_cm)
    raise_arms()
    drive("return", fwd_cm=args.back_cm)
    snapshot("1_returned")
    drive("left", right_cm=-args.left_cm)
    align_to_sandbag()

    log("grasp: episode starts")
    ok, reach = grasp(agent)
    snapshot("4_after_grasp")
    if not ok:
        raise RuntimeError("grasp did not close on the sandbag; task stopped, arms hold their pose")

    drive("right #1", right_cm=args.right_cm, tail=0.0)
    log(f"dwell {args.dwell_s:.0f}s")
    stop_chassis(args.dwell_s)
    drive("right #2", right_cm=args.right_cm, tail=0.5)
    snapshot("5_before_place")

    place = np.concatenate([FIRST_L, 0.2 * FIRST_R + 0.8 * reach])  # 抓取点上方一点
    log("place: reach out")
    move_arm(place, CLOSE_NM, 4.0, hold=0.3)
    log("place: open gripper")
    move_arm(place, OPEN_NM, 0.0, hold=1.2)
    log("place: retract")
    move_arm(FIRST14, OPEN_NM, 4.0, hold=0.5)
    snapshot("6_done")
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
