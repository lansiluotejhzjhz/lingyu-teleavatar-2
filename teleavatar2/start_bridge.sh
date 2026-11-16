#!/usr/bin/env bash
# 启动 zenoh bridge 连接机器人（文档 2.4）。此终端需保持运行，关闭即断开。
set -e
source "$(dirname "$(readlink -f "$0")")/env.sh"

if ! command -v zenoh-bridge-ros2dds >/dev/null; then
  echo "未找到 zenoh-bridge-ros2dds，请先安装" >&2; exit 1
fi
if ! nc -z -w 3 "$ROBOT_IP" 9000; then
  echo "无法连接 $ROBOT_IP:9000，请检查机器人是否开机、是否同一网段" >&2; exit 1
fi
echo "连接 tcp/$ROBOT_IP:9000 (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
exec zenoh-bridge-ros2dds -e "tcp/$ROBOT_IP:9000"
