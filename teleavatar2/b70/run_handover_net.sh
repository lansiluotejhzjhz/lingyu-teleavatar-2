#!/usr/bin/env bash
# run_handover.sh 的"网络保险版"：流程和参数完全相同，只是改用 run_task_net.sh 启动
# （启动前测丢包，检查/传模板/回传日志连不上时自动重试，停止信号重发）。停止用 ./stop_task_net.sh（或 Ctrl+C）。
DIR="$(dirname "$(readlink -f "$0")")"
RUN_TASK=run_task_net.sh exec "$DIR/run_handover.sh" "$@"
