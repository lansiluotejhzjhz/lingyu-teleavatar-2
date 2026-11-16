#!/usr/bin/env bash
# 一键组合任务（手势启动版）：
#   手动启动 -> 后退 30 cm、抬臂、回位 -> 原地待命，头部相机识别模板手势（OK 手势）
#   -> 识别到后两个夹爪闭合-张开两次示意 -> run_combo.sh 的后续流程（左移、抓取、右移、放下、递交）。
#
#   ./run_gesture_combo.sh                 检查通过后倒数 2 秒开始；待命最多 10 分钟
#   ./run_gesture_combo.sh --check         只做启动前检查，不动机器人
#   ./run_gesture_combo.sh --gesture-test  只测手势识别：识别到或超时就退出，机器人全程不动；每 0.5 秒存一帧图像到 logs/
#   ./run_gesture_combo.sh --gesture-capture   只采集 40 秒头部相机图像（每 0.25 秒一帧）用于标定，机器人全程不动
#   ./run_gesture_combo.sh -- --gesture-thresh 0.75 --gesture-hold-s 1.5 --gesture-timeout-s 300    “--” 之后的参数传给 combo_task.py
#
# 模板：/home/jhz/workspace/label/*.png（上传的 8 张）+ 本目录 gesture_templates_cam/*.png（从头部相机画面切出的同一手势 3 张）。
# 换手势时两处都要换（环境变量 GESTURE_TEMPLATES 可指定别的目录，空格分隔）。识别算法见 gesture_gate.py。
# 手要举在深色衣服前面、不要挡住脸，保持约 1 秒。待命超时退出码 3，手臂停在抬起的姿态。
# Ctrl+C 或 ./stop_task.sh 可随时结束。
DIR="$(dirname "$(readlink -f "$0")")"
LABEL_DIR=/home/jhz/workspace/label   # 上传的 8 张模板；这个目录不存在时用本目录下的副本 gesture_templates_label/
[ -d "$LABEL_DIR" ] || LABEL_DIR="$DIR/gesture_templates_label"
export GESTURE_TEMPLATES="${GESTURE_TEMPLATES:-$LABEL_DIR $DIR/gesture_templates_cam}"
ARGS=(); TEST=0
for a in "$@"; do
  case "$a" in --gesture-test) TEST=1 ;; --gesture-capture) TEST=2 ;; *) ARGS+=("$a") ;; esac
done
if [ $TEST != 0 ]; then
  case " ${ARGS[*]} " in *" -- "*) ;; *) ARGS+=(--) ;; esac
  [ $TEST = 1 ] && ARGS+=(--gesture-only --gesture-record-s 0.5)
  # 采集模式：40 秒内每 0.25 秒存一帧，不判定成功（退出码 3 属正常），用来标定阈值
  [ $TEST = 2 ] && ARGS+=(--gesture-only --gesture-record-s 0.25 --gesture-hold-s 999 --gesture-timeout-s 40)
fi
GRASP_PORT="${GRASP_PORT:-8000}"   # 抓取模型端口：8000 = 旧模型（5221 步），8002 = 新模型（9264 步）。例：GRASP_PORT=8002 ./run_gesture_combo.sh
case " ${ARGS[*]} " in *" -- "*) ;; *) ARGS+=(--) ;; esac
if [ "$GRASP_PORT" = 8000 ]; then
  # 旧模型：用 2026-11-15 跑通全流程时的任务脚本原样运行，不带为新模型加的改动（带一段、闭合限制、新的沙包定位）；2026-11-16 起抓取最多两次（--grasp-attempts）
  TASK_SCRIPT=combo_task_v1.py TASK_PORT="8000 8001" TASK_TIMEOUT=930 exec "$DIR/${RUN_TASK:-run_task.sh}" "${ARGS[@]}"
fi
GRASP_PRIME="${GRASP_PRIME:-1}"   # 新模型需要先把右臂带出一段再交给模型
ARGS+=(--grasp-port "$GRASP_PORT" --grasp-prime "$GRASP_PRIME")
TASK_SCRIPT=combo_task.py TASK_PORT="$GRASP_PORT 8001" TASK_TIMEOUT=930 exec "$DIR/${RUN_TASK:-run_task.sh}" "${ARGS[@]}"
