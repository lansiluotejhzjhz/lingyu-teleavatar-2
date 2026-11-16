#!/usr/bin/env bash
# 在 B70 服务器的客户端容器 openpi-xpu-client 里运行本目录的一个 Python 脚本（脚本经 stdin 传入，不在服务器上落文件）。
# 用法: ./b70_run.sh <脚本.py> [超时秒数] [脚本参数...]
# 例:   ./b70_run.sh chassis_step.py 60 0.1 0 0 12.08
# 连接信息取自 /home/jhz/workspace/B70.txt（askpass.sh 读取其中的密码）。
set -e
DIR="$(dirname "$(readlink -f "$0")")"
SCRIPT="$1"; TIMEOUT="${2:-60}"; shift; [ $# -gt 0 ] && shift
env SSH_ASKPASS="$DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}" setsid -w \
  ssh -o ConnectTimeout=8 -o PreferredAuthentications=password -o PubkeyAuthentication=no root@192.168.31.196 \
  "docker exec -i openpi-xpu-client /usr/local/bin/openpi-client-entrypoint.sh timeout -s INT -k 5s ${TIMEOUT}s python - $*" \
  < "$DIR/$SCRIPT" 2>&1 | grep -v "not multicast-capable"
