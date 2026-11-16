"""搬运 + 递交 组合任务（单进程，无缝衔接）：
后退 -> 抬臂 -> 回位 -> 左移到抓取位置 -> pi0.5 抓取沙包（抓取模型，端口 8000） -> 保持手臂右移 -> 停留 -> 再右移
-> 右臂伸出把沙包放到桌面、夹爪松开 -> 等 --wait-s 秒（默认 5 s） -> 夹爪重新闭合夹起沙包
-> 逆运动收回到递交起始姿态 -> 直接开始递交推理（递交模型，端口 8001） -> 模型输出"张开"时松手 -> 手臂收回。

前半段与 full_task.py 完全相同，后半段的递交回合与 handover_task.py 相同（少了"张开 5 秒等人放沙包"，因为沙包已在夹爪里）。
重新夹起后会读右夹爪开度确认夹到了沙包，没夹到就不开始递交。

一键启动： ./run_combo.sh      停止： ./stop_task.sh 或在启动终端按 Ctrl+C
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
p.add_argument("--wait-s", type=float, default=5.0, help="沙包放到桌面松开后，等多少秒再重新夹起")
p.add_argument("--place-blend", type=float, default=0.92, help="放置高度：抓取姿态的比例（1.0 = 抓取时的高度）")
p.add_argument("--handover-port", type=int, default=8001)
p.add_argument("--handover-max-steps", type=int, default=600)
p.add_argument("--after-release-s", type=float, default=3.0)
p.add_argument("--gesture-dir", default="", help="给了这个目录（内含 gesture_gate.py 和模板图片）就先等手势，识别到才开始任务")
p.add_argument("--gesture-thresh", type=float, default=-1.0, help="手势匹配分数阈值；-1 = 用 gesture_gate.py 里的默认值")
p.add_argument("--gesture-hold-s", type=float, default=1.0, help="手势要连续保持这么久才算识别成功")
p.add_argument("--gesture-timeout-s", type=float, default=600.0, help="等手势的最长时间，超时不启动任务")
p.add_argument("--gesture-delay-s", type=float, default=0.0, help="夹爪示意动作之后再多等几秒才继续")
p.add_argument("--gesture-ack-s", type=float, default=0.8, help="识别成功后夹爪示意：每次闭合、每次张开各持续多少秒（共两次）")
p.add_argument("--gesture-only", action="store_true", help="只测手势识别：识别成功或超时后退出，机器人不动")
p.add_argument("--gesture-record-s", type=float, default=0.0, help="等待期间每隔这么多秒存一帧全分辨率头部图像（0 = 不存），用于标定阈值")
p.add_argument("--no-align", action="store_true", help="不做基于沙包图像位置的底盘微调")
p.add_argument("--no-return-comp", action="store_true", help="回程不扣除抓取前对位多走的量（旧行为：停止位置会偏出对位的量）")
p.add_argument("--img-dir", default="/tmp/claude_task")
p.add_argument("--tag", default="", help="仅用于在进程列表里识别本任务（stop_task.sh 按它发 SIGINT）")
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S", force=True)
T0 = time.monotonic()


def log(msg):
    logging.info(f"T+{time.monotonic() - T0:6.1f}s  {msg}")


PROMPT = "Pick up the sandbag from the table."
HANDOVER_PROMPT = "Hand the sandbag to the person."
# 递交起始姿态：训练数据 put_224_v1 全部 161 个回合第一帧关节角的均值
START14 = np.array([-0.036, 0.764, -0.704, 1.913, 0.374, 0.732, 0.304, 0.072, -0.752, 0.549, -1.880, -0.402, -0.433, -0.425])
OPEN_ABOVE, OPEN_STEPS = -0.4, 3     # 递交：模型右夹爪力矩高于此值连续 N 步 -> 松开
RAMP_OK, RAMP_ABORT = 0.15, 1.0      # 递交启动自检：第一个目标与当前姿态的最大关节偏差 (rad)
HELD_MIN = 0.10                      # 右夹爪闭合后开度大于此值 = 夹着沙包。实测：空夹 -0.05..0.06，夹得浅 0.12..0.15，夹稳 0.3..0.55（2026-11-15 由 0.15 改为 0.10：有一次 0.148 被误判为抓空）
FIRST_L = np.array([0.096, 0.854, -0.808, 1.763, 0.450, 0.617, 0.540])
FIRST_R = np.array([-0.037, -0.808, 0.679, -1.685, -0.268, -0.409, -0.620])
FIRST14 = np.concatenate([FIRST_L, FIRST_R])
OPEN_NM, CLOSE_NM, CLOSE_BELOW = 2.0, -1.6, -0.8  # 夹爪力矩 (Nm) 和闭合阈值
K = 30.0 / 50.02  # 实际位移 / (指令速度 x 时长)，2026-11-14 横向卷尺实测
CTRL_HZ = 45.0
# 头部相机 (960x960) 里沙包的参考位置（此前抓取成功时的位置）和像素/厘米换算（由底盘小幅移动前后的图像估出）
REF_XY, PX_PER_CM, ALIGN_DEADBAND_CM, ALIGN_MAX_CM = (483.0, 726.0), (10.0, 8.0), 1.5, 25.0  # 上限 2026-11-15 由 10 调到 25 厘米

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
grip_pos = {}
iface.create_subscription(JointState, "/right_gripper/joint_states", lambda m: grip_pos.update(right=float(m.position[0])), 10)
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


def wait_for_gesture():
    """原地待命直到头部相机里识别到模板手势。期间本函数不发控制指令。返回 True = 识别成功。"""
    sys.path.insert(0, args.gesture_dir)
    import gesture_gate as gg
    templates = gg.load_templates(args.gesture_dir)
    if not templates:
        raise RuntimeError(f"no gesture templates in {args.gesture_dir}")
    thresh = gg.DEFAULT_THRESHOLD if args.gesture_thresh < 0 else args.gesture_thresh
    log(f"gesture: {len(templates)} template variants loaded, threshold {thresh:.2f}, hold {args.gesture_hold_s:.1f}s; "
        f"WAITING for the gesture (robot stays still, up to {args.gesture_timeout_s:.0f}s)")
    t_start = time.monotonic()
    since = None          # 连续识别到的起始时刻
    t_log = t_rec = 0.0
    n_rec = 0
    best_recent = 0.0
    while time.monotonic() - t_start < args.gesture_timeout_s:
        img = np.asarray(env.get_observation()["observation/images/head_camera"])
        det = gg.detect(img, templates)
        now = time.monotonic()
        hit = det["score"] >= thresh
        best_recent = max(best_recent, det["score"])
        if args.gesture_record_s > 0 and now - t_rec >= args.gesture_record_s and n_rec < 200:
            np.save(os.path.join(args.img_dir, f"gframe_{n_rec:03d}_{int(det['score'] * 100):03d}.npy"), img)
            t_rec, n_rec = now, n_rec + 1
        if hit:
            since = since or now
            if now - since >= args.gesture_hold_s:
                log(f"gesture: RECOGNIZED score {det['score']:.2f} template {det['template']} box {det['box']} "
                    f"after {now - t_start:.1f}s of waiting")
                np.save(os.path.join(args.img_dir, "g_gesture_recognized.npy"), gg.draw(img, det, True)[::2, ::2])
                return True
        else:
            since = None
        if now - t_log >= 2.0:
            log(f"gesture: waiting, best score in the last 2s {best_recent:.2f} (need {thresh:.2f}); now {det['score']:.2f} iou {det['iou']:.2f} chamfer {det['chamfer']:.2f} "
                f"template {det['template']} box {det['box']}, skin blobs {det['n']}")
            t_log, best_recent = now, 0.0
        time.sleep(0.03)
    log("gesture: NOT recognized before the timeout; task stops here")
    return False


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


align_moved = [0.0, 0.0]  # 对位时底盘实际多走的量 (右, 前) cm；带着沙包回去时扣掉，让机器人回到出发位置


def align_to_sandbag():
    """按沙包在头部图像里的位置修正底盘。偏差大时像素换算不准，所以最多修正两轮。"""
    img = snapshot("3_at_table")
    if args.no_align or img is None:
        return
    for rnd in (1, 2):
        c = find_sandbag(img)
        if c is None:
            log("align: sandbag not found in the head image, no correction")
            return
        right, back = (c[0] - REF_XY[0]) / PX_PER_CM[0], (c[1] - REF_XY[1]) / PX_PER_CM[1]
        log(f"align #{rnd}: sandbag at ({c[0]:.0f}, {c[1]:.0f}) px, reference {REF_XY} -> robot should move right {right:+.1f} cm, back {back:+.1f} cm")
        if max(abs(right), abs(back)) > ALIGN_MAX_CM:
            log("align: offset larger than the limit, no correction")
            return
        did = False
        if abs(back) > ALIGN_DEADBAND_CM:
            drive("align fwd/back", fwd_cm=-back)
            align_moved[1] += -back
            did = True
        if abs(right) > ALIGN_DEADBAND_CM:
            drive("align left/right", right_cm=right)
            align_moved[0] += right
            did = True
        if not did:
            return
        time.sleep(0.3)
        img = snapshot("3b_aligned")
        if img is None:
            return
        c2 = find_sandbag(img)
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


def handover(policy, max_steps, after_release_s):
    """递交回合（沙包已在右夹爪里）。先自检，再推理；模型输出"张开"时松手并锁住张开。返回是否松手。"""
    obs = env.get_observation()
    a0 = np.array(policy.infer(obs)["actions"], dtype=np.float64)[0]
    tgt = np.concatenate([a0[0:7], a0[8:15]])
    dev = np.abs(tgt - arm_now())
    log(f"handover pre-check: first target vs current pose max {dev.max():.2f} rad, model right gripper effort {a0[15]:+.2f} Nm")
    if a0[15] > OPEN_ABOVE:
        raise RuntimeError("handover model wants the gripper open at the start; not started, gripper stays closed")
    if dev.max() > RAMP_ABORT:
        raise RuntimeError(f"handover model's first target is {dev.max():.2f} rad away; not started, gripper stays closed")
    if dev.max() > RAMP_OK:
        secs = max(2.0, dev.max() / 0.25)
        log(f"handover pre-check: slow transition to the model's first target over {secs:.1f}s")
        move_arm(tgt, CLOSE_NM, secs, hold=0.3)
    agent = policy_agent.PolicyAgent(policy=action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=16))
    log("handover: episode starts")
    released, above, t_rel, last_log = False, 0, None, 0.0
    step, nxt = 1 / CTRL_HZ, time.monotonic()
    for i in range(max_steps):
        a = np.array(agent.get_action(env.get_observation())["actions"], dtype=np.float64)
        now, raw = time.monotonic(), float(a[15])
        if not released:
            above = above + 1 if raw > OPEN_ABOVE else 0
            if above >= OPEN_STEPS:
                released, t_rel = True, now
                log(f"handover: gripper RELEASE at step {i} (model effort {raw:+.2f} Nm)")
        if now - last_log >= 1.0:
            last_log = now
            log(f"  step {i:3d} model right gripper {raw:+.2f} Nm -> {'OPEN' if released else 'CLOSED'}; right arm j1 {a[8]:+.2f} j2 {a[9]:+.2f} j4 {a[11]:+.2f}")
        a[7], a[15] = OPEN_NM, (OPEN_NM if released else min(raw, CLOSE_NM))
        iface.publish_action(a)
        if released and now - t_rel > after_release_s:
            log(f"handover: {after_release_s:.0f}s after release, episode ends at step {i}")
            break
        nxt += step
        d = nxt - time.monotonic()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.monotonic()
    return released


status = 1
armed = False  # 只测识别（--gesture-only）时全程不发控制指令，包括收尾的底盘零速
try:
    policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)
    t = time.monotonic()
    policy.infer(env.get_observation())  # 预热，结果丢弃
    log(f"policy connected, warm-up inference {time.monotonic() - t:.2f}s")
    agent = policy_agent.PolicyAgent(policy=action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=16))
    handover_policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=args.handover_port)
    t = time.monotonic()
    obs = env.get_observation()
    obs["prompt"] = HANDOVER_PROMPT
    handover_policy.infer(obs)  # 预热，结果丢弃
    log(f"handover policy connected (port {args.handover_port}), warm-up inference {time.monotonic() - t:.2f}s")
    while "pos" not in wheel or chassis_pub.get_subscription_count() == 0:
        if time.monotonic() - T0 > 20:
            raise RuntimeError("no wheel states or no chassis subscriber")
        time.sleep(0.05)
    if args.gesture_dir and args.gesture_only:  # 只测识别：不动机器人
        status = 0 if wait_for_gesture() else 3
        raise SystemExit(status)
    armed = True
    snapshot("0_start")

    drive("back", fwd_cm=-args.back_cm)
    raise_arms()
    drive("return", fwd_cm=args.back_cm)
    snapshot("1_returned")
    if args.gesture_dir:  # 抬臂回位后原地待命（手臂由接口定时器保持姿态），识别到手势才继续
        if not wait_for_gesture():
            status = 3
            raise SystemExit(status)
        log("gesture: acknowledged, both grippers close-open twice")
        for _ in range(2):
            for g_nm in (CLOSE_NM, OPEN_NM):
                t = time.monotonic()
                while time.monotonic() - t < args.gesture_ack_s:
                    command(FIRST14, g_nm, g_nm)
                    time.sleep(1 / CTRL_HZ)
        if args.gesture_delay_s > 0:
            log(f"gesture: task continues in {args.gesture_delay_s:.0f}s")
            time.sleep(args.gesture_delay_s)
    drive("left", right_cm=-args.left_cm)
    align_to_sandbag()

    log("grasp: episode starts")
    ok, reach = grasp(agent)
    snapshot("4_after_grasp")
    if not ok:
        raise RuntimeError("grasp did not close on the sandbag; task stopped, arms hold their pose")

    if not args.no_return_comp and abs(align_moved[1]) > 0.3:
        drive("undo align fwd/back", fwd_cm=-align_moved[1])
    # 对位时横向多走的量在第一段右移里扣掉：这样黄区停留点和绿区终点都回到固定位置（2026-11-15 由第二段改到第一段）
    comp = 0.0 if args.no_return_comp else align_moved[0]
    if abs(comp) > 0.3:
        log(f"right #1: {args.right_cm:.0f} cm minus the {comp:+.1f} cm alignment shift, so the dwell point and the end point stay fixed")
    drive("right #1", right_cm=args.right_cm - comp, tail=0.0)
    log(f"dwell {args.dwell_s:.0f}s")
    stop_chassis(args.dwell_s)
    drive("right #2", right_cm=args.right_cm, tail=0.5)
    snapshot("5_before_place")

    b = args.place_blend
    place = np.concatenate([FIRST_L, (1 - b) * FIRST_R + b * reach])
    log("place: reach out")
    move_arm(place, CLOSE_NM, 4.0, hold=0.3)
    log(f"place: gripper OPEN, sandbag on the table; waiting {args.wait_s:.0f}s")
    move_arm(place, OPEN_NM, 0.0, hold=args.wait_s)
    snapshot("6_placed")
    log("re-grasp: gripper closing")
    move_arm(place, CLOSE_NM, 0.0, hold=1.5)
    held = grip_pos.get("right")
    log(f"re-grasp: right gripper opening {held if held is None else round(held, 3)}")
    if held is None or held < HELD_MIN:
        raise RuntimeError("re-grasp closed on nothing (gripper fully closed); handover not started, arm holds its pose")
    log("reverse motion to the handover start pose")
    move_arm(START14, CLOSE_NM, 4.0, hold=0.5)
    held = grip_pos.get("right")
    if held is None or held < HELD_MIN:
        raise RuntimeError("sandbag lost on the way back to the handover start pose; handover not started")
    snapshot("7_handover_start")

    env.set_prompt(HANDOVER_PROMPT)
    released = handover(handover_policy, args.handover_max_steps, args.after_release_s)
    snapshot("8_after_handover")
    if not released:
        raise RuntimeError("handover: step limit reached and the model never opened the gripper; arm holds, gripper stays closed")
    log("retract to the handover start pose")
    move_arm(START14, OPEN_NM, 5.0, hold=0.5)
    log("TASK DONE")
    status = 0
except KeyboardInterrupt:
    log("INTERRUPTED")
finally:
    try:
        if armed:
            stop_chassis(0.5)
    finally:
        env.close()
        log("shutdown complete")
sys.exit(status)
