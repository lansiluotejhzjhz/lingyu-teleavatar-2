#!/usr/bin/env bash
# 停止正在运行的完整任务：给 full_task.py 发 SIGINT，底盘立即发零速，手臂保持当前姿态。
DIR="$(dirname "$(readlink -f "$0")")"
# 丢包时停止信号可能发不出去，所以最多发 5 次，直到 B70 有回应。
for n in 1 2 3 4 5; do
  OUT=$(env SSH_ASKPASS="$DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}" setsid -w \
    ssh -o ConnectTimeout=5 -o PreferredAuthentications=password -o PubkeyAuthentication=no root@192.168.31.196 \
    'docker exec openpi-xpu-client pkill -INT -f "^python - --tag ta2_full_task" && echo "已发送停止信号" || echo "没有找到正在运行的任务"' < /dev/null 2>/dev/null)
  if [ -n "$OUT" ]; then echo "$OUT"; exit 0; fi
  echo "… 没有连上 B70（第 $n 次），重试" >&2
done
echo "✗ 5 次都没有连上 B70，停止信号没有送达：任务可能仍在运行，请立即按急停或到 B70 上停止" >&2
exit 1
