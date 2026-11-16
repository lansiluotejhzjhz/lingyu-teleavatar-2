#!/usr/bin/env bash
# 一键递交任务：把右夹爪里的沙包递到人手上（B70 上的递交模型，端口 8001）。全程不需要人工干预，除了往夹爪里放沙包。
#
#   ./run_handover.sh            检查通过后倒数 2 秒启动
#   ./run_handover.sh --check    只做启动前检查，不动机器人
#   ./run_handover.sh -- --open-s 8 --max-steps 400      “--” 之后的参数传给 handover_task.py
#
# 流程：到起始姿态（手臂下垂时先后退 30 cm 抬臂再回位）-> 夹爪张开 5 秒，此时把沙包放进右夹爪 -> 夹紧
#       -> 模型推理递出，模型输出"张开"时松手 -> 手臂收回。约 30-45 秒。
# 中途停止：Ctrl+C 或 ./stop_task.sh。接收的人等手臂停止靠近后再伸手；现场留人守急停。
DIR="$(dirname "$(readlink -f "$0")")"
TASK_SCRIPT=handover_task.py TASK_PORT=8001 exec "$DIR/${RUN_TASK:-run_task.sh}" "$@"
