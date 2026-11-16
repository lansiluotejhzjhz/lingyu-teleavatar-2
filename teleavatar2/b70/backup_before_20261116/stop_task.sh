#!/usr/bin/env bash
# 停止正在运行的完整任务：给 full_task.py 发 SIGINT，底盘立即发零速，手臂保持当前姿态。
DIR="$(dirname "$(readlink -f "$0")")"
env SSH_ASKPASS="$DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}" setsid -w \
  ssh -o ConnectTimeout=8 -o PreferredAuthentications=password -o PubkeyAuthentication=no root@192.168.31.196 \
  'docker exec openpi-xpu-client pkill -INT -f "^python - --tag ta2_full_task" && echo "已发送停止信号" || echo "没有找到正在运行的任务"' < /dev/null
