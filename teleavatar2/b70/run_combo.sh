#!/usr/bin/env bash
# 一键组合任务：搬运（run_task.sh 的全部流程）+ 递交。
# 沙包放到第二张桌子、夹爪松开后等 5 秒，重新夹起，收回到递交起始姿态，直接开始递交推理，把沙包递到人手上。
#
#   ./run_combo.sh            检查通过后倒数 2 秒启动，约 2.5 分钟
#   ./run_combo.sh --check    只做启动前检查，不动机器人
#   ./run_combo.sh -- --wait-s 8 --dwell-s 10     “--” 之后的参数传给 combo_task.py
#
# 起始条件与 run_task.sh 相同；另外递交时接收的人要在第二张桌子旁，等手臂停止靠近后再伸手。
# 中途停止：Ctrl+C 或 ./stop_task.sh。
DIR="$(dirname "$(readlink -f "$0")")"
GRASP_PORT="${GRASP_PORT:-8000}"   # 抓取模型端口：8000 = 旧模型（5221 步），8002 = 新模型（9264 步）。例：GRASP_PORT=8002 ./run_combo.sh
ARGS=("$@"); case " ${ARGS[*]} " in *" -- "*) ;; *) ARGS+=(--) ;; esac
case " ${ARGS[*]} " in *" --grasp-only "*|*" --home-only "*) ORIG=0 ;; *) ORIG=1 ;; esac
if [ "$GRASP_PORT" = 8000 ] && [ $ORIG = 1 ]; then
  # 旧模型的完整流程：用跑通全流程的任务脚本 combo_task_v1.py，不带为新模型加的改动
  TASK_SCRIPT=combo_task_v1.py TASK_PORT="8000 8001" exec "$DIR/${RUN_TASK:-run_task.sh}" "${ARGS[@]}"
fi
GRASP_PRIME="${GRASP_PRIME:-$([ "$GRASP_PORT" = 8002 ] && echo 1 || echo 0)}"   # 新模型需要先把右臂带出一段再交给模型，旧模型不需要
ARGS+=(--grasp-port "$GRASP_PORT" --grasp-prime "$GRASP_PRIME")
TASK_SCRIPT=combo_task.py TASK_PORT="$GRASP_PORT 8001" exec "$DIR/${RUN_TASK:-run_task.sh}" "${ARGS[@]}"
