#!/usr/bin/env bash
# 一键完整任务：后退-抬臂-回位 -> 左移到抓取位置 -> pi0.5 抓取沙包 -> 右移-停留-右移 -> 放下沙包。
# 全程不需要人工干预；启动前自动检查，结束后把日志和过程图像收回到 logs/ 下。
#
#   ./run_task.sh            检查通过后倒数 2 秒启动
#   ./run_task.sh --check    只做启动前检查，不动机器人
#   ./run_task.sh -- --right-cm 60 --dwell-s 10 --no-align     “--” 之后的参数传给 full_task.py
#
# 中途停止：按 Ctrl+C（或另开终端运行 ./stop_task.sh）。底盘立即发零速，手臂保持当前姿态。
# 起始条件：机器人停在第二张桌子前（抓取位置右侧 144 cm），沙包在主桌中间略偏右、瓶子那一排前方，路径上无人。
set -u
DIR="$(dirname "$(readlink -f "$0")")"
TAG=ta2_full_task
TASK_SCRIPT="${TASK_SCRIPT:-full_task.py}"   # TASK_SCRIPT=handover_task.py ./run_task.sh 运行递交任务
TASK_PORT="${TASK_PORT:-8000}"               # 该任务用的推理服务端口，可写多个，空格分隔（抓取 8000，递交 8001）
COUNTDOWN="${COUNTDOWN:-2}"                  # 检查通过后倒数几秒再启动（Ctrl+C 可取消）
TASK_TIMEOUT="${TASK_TIMEOUT:-300}"           # 任务脚本的总时限（秒），到时发 SIGINT
GESTURE_TEMPLATES="${GESTURE_TEMPLATES:-}"   # 手势模板目录（可多个，空格分隔，文件名不要重复）；非空 = 先等手势再开始（仅 combo_task.py 支持）
CHECK_ONLY=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "未知参数: $1（传给 full_task.py 的参数放在 -- 之后）" >&2; exit 2 ;;
  esac
  shift
done

remote() {  # remote '<命令>'：在 B70 上执行；stdin 原样传过去
  env SSH_ASKPASS="$DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}" setsid -w \
    ssh -o ConnectTimeout=8 -o PreferredAuthentications=password -o PubkeyAuthentication=no root@192.168.31.196 "$1"
}
C="docker exec openpi-xpu-client /usr/local/bin/openpi-client-entrypoint.sh"

echo "== 启动前检查"
if pgrep -f '[z]enoh-bridge-ros2dds' >/dev/null; then
  echo "✗ 本机的 zenoh bridge 在运行（会成为第二个控制端），请先关掉" >&2; exit 1
fi
CHECK=$(remote "
  nc -z -w 3 192.168.31.244 9000 2>/dev/null && echo ROBOT_OK || echo ROBOT_UNREACHABLE
  ok=1; for port in $TASK_PORT; do [ \"\$(curl -fsS --max-time 3 http://127.0.0.1:\$port/healthz)\" = OK ] || ok=0; done; [ \$ok = 1 ] && echo SERVER_OK || echo SERVER_DOWN
  docker top openpi-xpu-client >/dev/null 2>&1 && echo CLIENT_OK || echo CLIENT_DOWN
  docker top openpi-xpu-client 2>/dev/null | grep -qE '[m]ain\\.py|[z]ero\\.py|$TAG|python -( |\$)' && echo CONTROLLER_RUNNING
  for t in /api/chassis/velocity /api/left_arm/joint_cmd /api/right_arm/joint_cmd; do
    ( $C timeout 10 ros2 topic info \$t 2>/dev/null | grep -q 'Publisher count: 0' || echo \"TOPIC_BUSY \$t\" ) &
  done
  ( $C timeout 6 ros2 topic echo --once --full-length /api/current_mode 2>/dev/null | grep -q '\"enable_chassis\": true' && echo MODE_OK || echo MODE_BAD ) &
  wait
" < /dev/null 2>/dev/null)
fail=0
say() { if echo "$CHECK" | grep -q "$1"; then echo "✓ $2"; else echo "✗ $3"; fail=1; fi; }
[ -n "$CHECK" ] || { echo "✗ 连不上 B70 服务器 192.168.31.196" >&2; exit 1; }
say ROBOT_OK   "机器人 192.168.31.244:9000 可达" "机器人不可达（未开机或断网）"
say SERVER_OK  "pi0.5 推理服务正常（端口 $TASK_PORT）" "推理服务没有响应（端口 $TASK_PORT；抓取模型容器 ta2-sandbag-5221，递交模型容器 ta2-handover-5907）"
say CLIENT_OK  "客户端容器在运行" "客户端容器 openpi-xpu-client 没有运行"
say MODE_OK    "机器人处于 API 模式且底盘已启用，话题有数据" "读不到机器人模式，或底盘未启用（可能需要重启机器人）"
if echo "$CHECK" | grep -q CONTROLLER_RUNNING; then echo "✗ 容器里已有控制程序在运行"; fail=1; else echo "✓ 没有其他控制程序"; fi
if echo "$CHECK" | grep -q TOPIC_BUSY; then echo "✗ 有其他节点在发布控制话题: $(echo "$CHECK" | grep TOPIC_BUSY | tr '\n' ' ')"; fail=1; else echo "✓ 控制话题空闲"; fi
[ $fail -eq 0 ] || { echo "检查未通过，没有启动。" >&2; exit 1; }
if [ -n "$GESTURE_TEMPLATES" ]; then
  [ -f "$DIR/gesture_gate.py" ] || { echo "✗ 缺少 gesture_gate.py" >&2; exit 1; }
  TAR_ARGS=()
  for d in $GESTURE_TEMPLATES; do
    ls "$d"/*.png >/dev/null 2>&1 || { echo "✗ 手势模板目录 $d 里没有 png" >&2; exit 1; }
    echo "✓ 手势模板 $(ls "$d"/*.png | wc -l) 张（$d）"
    TAR_ARGS+=(-C "$d" .)
  done
fi
[ $CHECK_ONLY -eq 0 ] || { echo "检查通过（--check：不启动任务）。"; exit 0; }

STAMP=$(date +%Y%m%d_%H%M%S); OUT="$DIR/logs/$STAMP"; mkdir -p "$OUT"
REMOTE_IMG=/tmp/ta2_task_$STAMP
stop_remote() { remote "docker exec openpi-xpu-client pkill -INT -f \"^python - --tag $TAG\"" < /dev/null >/dev/null 2>&1; }
trap 'echo; echo "== 收到中断，正在停止机器人上的任务..."; stop_remote' INT TERM

echo "== $COUNTDOWN 秒后启动，确认手臂和底盘路径上无人（Ctrl+C 取消）"
for i in $(seq "$COUNTDOWN" -1 1); do printf "%s " "$i"; sleep 1 || exit 130; done; echo
echo "== 任务脚本: $TASK_SCRIPT"
if [ -n "$GESTURE_TEMPLATES" ]; then  # 模板图片和识别代码送进容器的临时目录，任务结束时随 $REMOTE_IMG 一起删除
  tar cf - "${TAR_ARGS[@]}" -C "$DIR" gesture_gate.py \
    | remote "docker exec -i openpi-xpu-client sh -c 'mkdir -p $REMOTE_IMG/gesture && tar xf - -C $REMOTE_IMG/gesture'" \
    || { echo "✗ 手势模板传到 B70 失败，没有启动" >&2; exit 1; }
  EXTRA=(--gesture-dir "$REMOTE_IMG/gesture" "${EXTRA[@]}")
  echo "== 手势版：先后退-抬臂-回位，然后原地待命；对着头部相机做模板手势，夹爪开合两次示意后继续任务"
fi
echo "== 任务开始，日志: $OUT/task.log"
remote "docker exec -i openpi-xpu-client /usr/local/bin/openpi-client-entrypoint.sh timeout -s INT -k 5s ${TASK_TIMEOUT}s python - --tag $TAG --img-dir $REMOTE_IMG ${EXTRA[*]:-}; echo TASK_EXIT=\$?" \
  < "$DIR/$TASK_SCRIPT" 2>&1 | tee "$OUT/task.log" | grep --line-buffered -E "T\+|Traceback|Error|ERROR|TASK_EXIT"
trap - INT TERM
RC=$(grep -o 'TASK_EXIT=[0-9]*' "$OUT/task.log" | tail -1 | cut -d= -f2)

remote "docker exec openpi-xpu-client tar cf - --exclude=./gesture -C $REMOTE_IMG . 2>/dev/null; docker exec openpi-xpu-client rm -rf $REMOTE_IMG" < /dev/null > "$OUT/data.tar" 2>/dev/null \
  && tar xf "$OUT/data.tar" -C "$OUT" 2>/dev/null && rm -f "$OUT/data.tar"
/usr/bin/python3 -I - "$OUT" <<'PY' 2>/dev/null || true
import glob, sys, numpy as np, cv2
for f in glob.glob(sys.argv[1] + "/*.npy"):
    a = np.load(f)
    if a.ndim == 3:  # 过程图像；轨迹等其他数组跳过
        cv2.imwrite(f[:-4] + ".jpg", a[:, :, ::-1])
PY
if [ "${RC:-1}" = 3 ]; then echo "== 等待时限内没有识别到手势，任务停在待命处（手臂保持当前姿态）。日志在 $OUT" >&2; exit 3; fi
if [ "${RC:-1}" = 0 ]; then echo "== 任务完成。日志和过程图像在 $OUT"; else echo "== 任务未完成（退出码 ${RC:-未知}），见 $OUT/task.log" >&2; fi
exit "${RC:-1}"
