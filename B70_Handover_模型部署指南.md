# B70 沙包递交模型部署指南

这份指南用于换一台 Windows 电脑后，继续操作同一台 B70。Windows 电脑通过 PowerShell SSH 连接，模型和 Docker 服务在 B70 上运行。

| 项目 | 当前配置 |
| --- | --- |
| B70 SSH 地址 | `root@192.168.31.196` |
| TA2 地址 | `192.168.31.244` |
| 递交模型 | `sandbag_handover_224_v1-20261115-1312-5907`，PyTorch，5907 步 |
| 模型配置 | `pi05_teleavatar_v2`，绝对关节角，RTC 关闭 |
| 递交模型服务 | B70 本机 `127.0.0.1:8001`，容器 `ta2-handover-5907` |
| 原抓取模型服务 | B70 本机 `127.0.0.1:8000` |
| 递交任务提示词 | `Hand the sandbag to the person.` |
| 服务端 tmux 会话 | `ta2-handover` |

递交模型已上传并完成 SHA‑256 校验。在录制示范不同阶段做离线推理时，模型预测在递出前夹紧，在松手阶段张开。**递交模型的真机效果仍需现场测试。**

## 1. 换电脑后连接 B70

把新电脑接入 B70 所在局域网，在 **Windows PowerShell** 执行。登录密码由现场提供。

```powershell
ssh root@192.168.31.196
```

在显示 `root@user-KVL-XRP` 的 **B70 终端** 检查递交服务。

```bash
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
```

如果返回 `OK`，可以进入第 3 步。这个服务地址给运行在 B70 上的客户端使用。需要查看服务日志时，在 B70 执行 `docker logs --tail 40 -f ta2-handover-5907`。按 `Ctrl+C` 可以退出日志查看。

## 2. 递交服务停止时重新启动

先在 B70 查看容器和会话。如果容器仍在加载或预热，继续看日志，直到健康检查返回 `OK`。

```bash
docker ps -a --filter name=ta2-handover-5907
tmux list-sessions
```

服务已停止时，用 `tmux attach -t ta2-handover` 返回现有会话。如果没有该会话，新建一个。

```bash
tmux new-session -s ta2-handover -c /root/work/deploy/server/openpi-xpu-env
```

在该会话的 Bash 提示符下粘贴下面的命令。启动脚本会按绝对关节角解释 5907 的输出。原版 `serve_policy.py` 的同名配置默认按增量解释，直接用它会把当前关节角额外加到目标上。

```bash
cd /root/work/deploy/server/openpi-xpu-env && \
PT_DIR=/root/work/checkpoints/sandbag_handover_224_v1-20261115-1312-5907 && \
test -r "$PT_DIR/model.safetensors" && \
test -r "$PT_DIR/assets/path-to-dataset/norm_stats.json" && \
OPENPI_DIR=/root/work/deploy/server/openpi \
ZE_AFFINITY_MASK=1 \
DOCKER_RUN_EXTRA="--name ta2-handover-5907 -p 127.0.0.1:8001:8000 -v $PT_DIR:/ckpt:ro -v /root/work/deploy/server/serve_handover_5907.py:/app/scripts/serve_handover_5907.py:ro" \
bash scripts/run.sh python scripts/serve_handover_5907.py \
  --env TELEAVATAR --warmup \
  policy:checkpoint --policy.config pi05_teleavatar_v2 --policy.dir /ckpt \
  --policy.pytorch-device xpu --policy.pytorch-compile-mode default
```

按 `Ctrl+B`，松开后按 `D`，返回普通 SSH 终端，服务继续在 tmux 中运行。再次执行第 1 步的 `curl`，确认预热完成。

## 3. 检查 TA2 客户端

在 TA2 的 RemoteApp 中，将运行模式设为 **API**，启用左右臂并将两侧都设为 **关节（joint）**，保存并生效。视频推流目标是 **B70 `192.168.31.196`、UDP `8890`**。结束 VR 遥操和其他正在向 TA2 发控制指令的程序。

在 B70 上检查客户端。容器已经运行且 Zenoh 目标正确时，接着检查关节话题。

```bash
cd /root/work/deploy
docker ps --filter name=openpi-xpu-client
docker inspect openpi-xpu-client --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(ZENOH_ENDPOINT|ROS_DOMAIN_ID)='
```

客户端容器没有启动时，在 B70 执行下面的命令。

```bash
cd /root/work/deploy
export OPENPI_DIR="$PWD/client/openpi"
export ROS_DOMAIN_ID=29
chmod +x client/gpu_minfreq.sh
ZENOH_ENDPOINT=tcp/192.168.31.244:9000 bash client/run_client.sh start
```

执行预检。左右臂关节状态的 `Publisher count` 应各为 `1`，运行模式及左右臂控制模式都应为 `1`。最后一条只接收视频，运行结束后会写下解码统计。

```bash
cd /root/work/deploy
curl -fsS --max-time 5 http://127.0.0.1:8001/healthz
bash client/run_client.sh exec ros2 topic info /left_arm/joint_states
bash client/run_client.sh exec ros2 topic info /right_arm/joint_states
bash client/run_client.sh exec timeout 5s ros2 topic echo --once --full-length /api/current_mode
bash client/run_client.sh exec python examples/teleavatar_v2/test.py \
  --video-decode va --duration-s 3 \
  --stats-json "/outputs/handover_video_check_$(date +%Y%m%d_%H%M%S).json"
```

## 4. 真机递交试跑

**下面会驱动机械臂和夹爪。** 递交示范的开场是**右夹爪已经夹稳沙包，双臂在递交示范的起始姿态**。按现场操作流程先完成装持和摆位。通用 `zero.py` 的固定目标与递交示范的起点可能不同。首次上机先在桌面上方确认伸出和松爪位置，现场留人守急停，接收者等手臂停止靠近后再伸手承接。

在 B70 开一个独立客户端会话。若已有同名会话，用 `tmux attach -t ta2-handover-client` 回到它。

```bash
tmux new-session -s ta2-handover-client -c /root/work/deploy
```

确认姿态和现场都就绪后运行一轮。`300` 是本指南给首次试跑设置的步数上限。

```bash
bash client/run_client.sh exec python examples/teleavatar_v2/main.py \
  --remote-host 127.0.0.1 --remote-port 8001 \
  --video-decode va \
  --prompt "Hand the sandbag to the person." \
  --num-episodes 1 --max-episode-steps 300
```

观察右爪能否夹稳沙包、向接收位置送出、在接收位置松开。递交完成或动作异常时按 `Ctrl+C` 停止客户端。tmux 脱离或 SSH 断线后程序可能继续运行，现场出现异常先用急停。需要从另一个 B70 终端结束 `main.py` 时执行下面的命令。

```bash
cd /root/work/deploy
bash client/run_client.sh exec pkill -INT -f examples/teleavatar_v2/main.py
```
