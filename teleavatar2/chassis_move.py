#!/usr/bin/python3
# 必须用系统 Python 3.10：rclpy（Humble）不支持 conda 的 Python
"""TeleAvatar 2 底盘速度控制（文档 4.5 / 6.5）。

向 /api/chassis/velocity 发布 std_msgs/Float32MultiArray，data=[vx, vy, wz]
（m/s, m/s, rad/s），持续 duration 秒后发送全零并等待受控停车。

示例：
  ./chassis_move.py --vx 0.05 --duration 2      # 沿 x 正向
  ./chassis_move.py --wz 0.2 --duration 2       # 绕 z 正向旋转
Ctrl+C 随时停止（会发送全零）。
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

TOPIC = '/api/chassis/velocity'
RATE_HZ = 10.0
# 机器人 API 本身不做速度限幅，这里是本脚本的保守上限，可用 --max-lin/--max-ang 放宽
DEFAULT_MAX_LIN = 0.2   # m/s
DEFAULT_MAX_ANG = 0.5   # rad/s
STOP_REPEAT_S = 1.0     # 结束时持续发送全零的时长


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vx', type=float, default=0.0, help='x 方向线速度 m/s')
    p.add_argument('--vy', type=float, default=0.0, help='y 方向线速度 m/s')
    p.add_argument('--wz', type=float, default=0.0, help='绕 z 角速度 rad/s')
    p.add_argument('--duration', type=float, default=2.0, help='运动时长 s')
    p.add_argument('--max-lin', type=float, default=DEFAULT_MAX_LIN, help='线速度上限 m/s')
    p.add_argument('--max-ang', type=float, default=DEFAULT_MAX_ANG, help='角速度上限 rad/s')
    a = p.parse_args()
    for name in ('vx', 'vy', 'wz', 'duration'):
        if not math.isfinite(getattr(a, name)):
            p.error(f'--{name} 必须是有限数值')
    if abs(a.vx) > a.max_lin or abs(a.vy) > a.max_lin:
        p.error(f'线速度超过上限 {a.max_lin} m/s（确认安全后可用 --max-lin 放宽）')
    if abs(a.wz) > a.max_ang:
        p.error(f'角速度超过上限 {a.max_ang} rad/s（确认安全后可用 --max-ang 放宽）')
    if a.duration <= 0:
        p.error('--duration 必须大于 0')
    return a


def main():
    args = parse_args()
    rclpy.init()
    node = Node('chassis_velocity_api_publisher')
    pub = node.create_publisher(Float32MultiArray, TOPIC, 10)

    def send(vx, vy, wz):
        pub.publish(Float32MultiArray(data=[float(vx), float(vy), float(wz)]))

    def run_for(seconds, vx, vy, wz):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            send(vx, vy, wz)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(1.0 / RATE_HZ)

    try:
        # 等 bridge 发现发布者，避免前几帧丢失
        deadline = time.monotonic() + 3.0
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            node.get_logger().warn(
                f'{TOPIC} 没有订阅者：请确认 bridge 在运行、ROS_DOMAIN_ID=29、机器人已启用底盘')
        node.get_logger().info(
            f'发送 vx={args.vx:.3f} vy={args.vy:.3f} wz={args.wz:.3f}，持续 {args.duration:.1f}s')
        run_for(args.duration, args.vx, args.vy, args.wz)
    except KeyboardInterrupt:
        node.get_logger().info('用户中断')
    finally:
        # 全零会立即把目标置零，随后按减速度限制停车
        try:
            if rclpy.ok():
                node.get_logger().info('发送零速度，等待停车')
                run_for(STOP_REPEAT_S, 0.0, 0.0, 0.0)
        except KeyboardInterrupt:
            send(0.0, 0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
