# openpi-xpu-env：openpi Intel GPU 推理环境镜像

本镜像提供在 Intel GPU（Arc Pro B70，Xe2/BMG）上使用 PyTorch 运行 `openpi` 推理所需的运行环境。**镜像不含 `openpi` 源码**，源码留在宿主机并在运行时挂载到容器的 `/app`，修改代码、配置或运行参数均无需重建镜像。

> **代码分支要求**：宿主机 `openpi` 需使用 `feat/teleavatar-v2-server-b70` 分支。

| 组件 | 版本 / 说明 |
| --- | --- |
| 镜像版本 | `openpi-xpu-env:v0.1.0`（基于 `ubuntu:24.04`） |
| Python / uv | Python 3.11.17（`/opt/venv`），uv 0.13.5 |
| 核心依赖 | torch 2.12.1+xpu，torchvision 0.27.1+xpu，triton-xpu 3.7.1，transformers 4.53.2，jax 0.5.3（`JAX_PLATFORMS=cpu`），numpy 1.26.4 |
| Intel GPU UMD | compute-runtime 26.09.37435.12，IGC 2.30.3，Level Zero 1.28.0，gmmlib 22.9.0（与离线安装包 `26.18.8.2` 同版本） |


---

## 1. 前置条件

- **硬件与驱动**：Linux x86_64，Intel Arc Pro B70（或其他 Xe2 独显）。
- **Docker**：支持 `--device /dev/dri`（无需 `nvidia-container-toolkit`）。
- **磁盘与内存**：镜像归档约 6.1 GB，解压加载后占 22.1 GB。推理显存峰值约 7.3 GiB，加载 checkpoint 时系统内存峰值约 15.7 GiB。
- **网络与离线缓存**：首次下载 tokenizer 等资源需联网。离线机器可将已缓存的 `OPENPI_DATA_HOME`（默认 `~/.cache/openpi`）拷入，并加 `-e HF_HUB_OFFLINE=1` 运行。

---

## 2. 快速开始

下文命令均在本仓库（`openpi-xpu-env/`）根目录执行。`/path/to/openpi` 为宿主机代码路径，`/path/to/ckpt` 为转换好的 PyTorch checkpoint 目录。将我们提供的U盘中的镜像拷贝到仓库目录下，然后可以执行下列命令：

### 2.1 校验并加载镜像

```bash
sha256sum -c openpi-xpu-env_v0.1.0.tar.gz.sha256
docker load -i openpi-xpu-env_v0.1.0.tar.gz
# 可选：核对镜像版本 Label
docker image inspect --format '{{json .Config.Labels}}' openpi-xpu-env:v0.1.0
```

### 2.2 冒烟测试

运行下列命令，测试 GPU 可见性（应输出 `2.12.1+xpu True 1 Intel(R) Graphics [0xe223]`）：

```bash
OPENPI_DIR=/path/to/openpi scripts/run.sh python -c \
  "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count(), torch.xpu.get_device_name(0))"
```

运行完整算子与编译检查（通过时末行输出 `SMOKE_SUMMARY {"label": "image", "all_ok": true, …}`）：

```bash
OPENPI_DIR=/path/to/openpi DOCKER_RUN_EXTRA="-v $(pwd)/smoke:/smoke:ro" \
  scripts/run.sh bash -c "python /smoke/verify_deps.py && python /smoke/smoke.py --label image --quick"
```

### 2.3 转换 checkpoint

训练得到的是 JAX checkpoint，B70 上推理需要 PyTorch checkpoint，每个 checkpoint 只需转换一次。转换在 CPU 上进行，需要约 32 GB 内存。挂载训练输出中某一步的目录（含 `params/` 和 `assets/`，例如 `/path/to/outputs/checkpoints/pi05_teleavatar_v2/my_exp/19999`），`--config-name` 填训练时用的配置名，结果写到 `OUTPUT_DIR` 下：

```bash
OPENPI_DIR=/path/to/openpi XPU=0 GPUS=none OUTPUT_DIR=/path/to/converted \
  DOCKER_RUN_EXTRA="-e JAX_PLATFORMS=cpu -v /path/to/jax_checkpoint/19999:/jax:ro" \
  scripts/run.sh python examples/convert_jax_model_to_pytorch.py --checkpoint-dir /jax \
  --config-name pi05_teleavatar_v2 --output-path /outputs/pi05_teleavatar_v2_pytorch --precision bfloat16
```

转换结果 `/path/to/converted/pi05_teleavatar_v2_pytorch` 即下文的 `/path/to/ckpt`。脚本会把归一化统计一起拷到 `assets/<数据集目录名>/`（训练时 `repo_id` 的最后一级目录名），但服务端按配置里固定的名字查找，需要再建一个软链接：

| 服务用的配置 | 软链接名 |
| --- | --- |
| `pi05_teleavatar_v2`、`pi05_teleavatar_v2_rtc` | `path-to-dataset` |
| `pi05_teleavatar_v2_chassis` | `checkout_224_plus_no_scan_224` |

```bash
cd /path/to/ckpt/assets && ls                    # 查看数据集目录名，例如 teleavatar_v2_pick_place
ln -s teleavatar_v2_pick_place path-to-dataset    # 底盘配置把 path-to-dataset 换成 checkout_224_plus_no_scan_224
```

容器以 root 运行，转换结果归 root 所有，非 root 用户建软链接时需加 `sudo`。

### 2.4 启动推理服务

**常规启动（无 RTC）：**

```bash
OPENPI_DIR=/path/to/openpi DOCKER_RUN_EXTRA="-p 8000:8000 -v /path/to/ckpt:/ckpt:ro" \
  scripts/run.sh python scripts/serve_policy.py --env TELEAVATAR --warmup \
  policy:checkpoint --policy.config pi05_teleavatar_v2 --policy.dir /ckpt \
  --policy.pytorch-device xpu --policy.pytorch-compile-mode default
```

**开启 Guided RTC 启动（必须执行绑核操作，详见 4.2 节）：**

可以直接复制下列命令：

```bash
OPENPI_DIR=/path/to/openpi DOCKER_RUN_EXTRA="--cpuset-cpus=3,27 -p 8000:8000 -v /path/to/ckpt:/ckpt:ro" \
  scripts/run.sh python scripts/serve_policy.py --env TELEAVATAR --warmup --rtc.enabled --rtc.mode GUIDED \
  policy:checkpoint --policy.config pi05_teleavatar_v2 --policy.dir /ckpt \
  --policy.pytorch-device xpu --policy.pytorch-compile-mode default
```

**按训练配置选择参数：** 上面的命令以默认的 `pi05_teleavatar_v2` 为例。`--policy.config` 必须与训练时的配置名一致，服务端和客户端的参数也要配套：

| 训练配置 | `--policy.config` | 服务端 RTC 参数 | 客户端 `main.py` |
| --- | --- | --- | --- |
| `pi05_teleavatar_v2`（默认） | `pi05_teleavatar_v2` | 不开，或 `--rtc.enabled --rtc.mode GUIDED`（须绑核） | 服务端开了 RTC 才加 `--rtc` |
| `pi05_teleavatar_v2_rtc` | `pi05_teleavatar_v2_rtc` | `--rtc.enabled --rtc.mode TRAINED` | 加 `--rtc` |
| `pi05_teleavatar_v2_chassis`（带底盘，19 维动作） | `pi05_teleavatar_v2_chassis` | 不开、`GUIDED`（须绑核）或 `TRAINED` 均可 | 加 `--enable-chassis`；服务端开了 RTC 才加 `--rtc` |

- **RTC 要两端配套**：客户端的 `--rtc` 要求服务端以 `--rtc.enabled` 启动，否则客户端报 `Policy server did not return an RTC chunk id`。
- **底盘要两端配套**：服务端用 `pi05_teleavatar_v2_chassis` 时输出 19 维动作，客户端必须加 `--enable-chassis`；不带底盘的配置输出 16 维，客户端不能加。两端不一致时第一个动作就会报错。checkpoint 的归一化统计不足 19 维时，底盘配置的服务端会拒绝启动。

---

## 3. 容器运行配置

### 3.1 使用 `scripts/run.sh`（推荐）

不带参数执行 `OPENPI_DIR=/path/to/openpi scripts/run.sh` 可直接进入容器交互式 `bash`。常用环境变量如下：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `OPENPI_DIR` | `../openpi` | 宿主机代码目录，挂载到 `/app` |
| `ZE_AFFINITY_MASK` | `0` | 选卡（`0` 为第一张卡，`1` 为第二张卡），容器内仅见所选单卡 |
| `XPU_CACHE_DIR` | `../xpu-cache` | `torch.compile` 缓存目录（自动创建），挂载到 `/xpu-cache` |
| `OUTPUT_DIR` | `../outputs` | 输出目录（自动创建），挂载到 `/outputs` |
| `OPENPI_DATA_HOME` | `~/.cache/openpi` | openpi 资产缓存（tokenizer 等），挂载到 `/openpi_assets` |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 缓存，挂载到 `/hf_home` |
| `XPU` | `auto` | `1`/`0` 强制开关 Intel GPU 挂载；`auto` 根据镜像名是否含 `xpu` 判断 |
| `PROXY` | `0` | 设为 `1` 时使用 host 网络并透传宿主机 `HTTPS_PROXY` 等代理变量 |
| `RUN_AS_USER` | `0` | 设为 `1` 时以宿主机当前用户运行（**注意：非 root 下无法自动打 transformers 补丁，PyTorch 模型不可用，推理请保持默认 `0`**） |
| `DOCKER_RUN_EXTRA` | 空 | 追加给 `docker run` 的参数（按空白切分且展开通配符；含空格或特殊字符时请直接用 `docker run`） |
| `IMAGE` / `TAG` | `openpi-xpu-env` / `VERSION` | 镜像名与 tag（`IMAGE=openpi-env` 时兼容 CUDA 镜像启动方式） |

**常见启动日志说明：**
- `[openpi-xpu-env] applied transformers_replace -> …`：补丁已生效，属正常提示。
- `WARNING: /app/uv.lock differs from the lock this image was built from`：属预期现象。`xpu` 依赖组仅维护在镜像内，未回写 `openpi` 仓库。
- `NOTE: no /dev/dri/renderD* in this container...`：仅在指定 `XPU=0 GPUS=none` 的无 GPU 容器（如替身客户端）中出现，可忽略。

### 3.2 等价的 `docker run` 命令

若不使用 `run.sh`，请先在宿主机 `mkdir -p` 创建各挂载目录。**代码必须挂载到 `/app`**（镜像 `PYTHONPATH` 固定指向 `/app/src` 和 `/app/packages/openpi-client/src`）：

```bash
docker run --rm --init --shm-size=8g -w /app -v /path/to/openpi:/app \
  -v /path/to/outputs:/outputs -e WANDB_DIR=/outputs -v ~/.cache/openpi:/openpi_assets \
  -v ~/.cache/huggingface:/hf_home -e HF_HOME=/hf_home \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0 \
  -v /path/to/xpu-cache:/xpu-cache -e TORCHINDUCTOR_CACHE_DIR=/xpu-cache/inductor -e TRITON_CACHE_DIR=/xpu-cache/triton \
  openpi-xpu-env:v0.1.0 python -c "import torch; print(torch.xpu.is_available())"
```

---

## 4. 推理与运维要点

### 4.1 编译模式与缓存（`torch.compile`）

- **`--policy.pytorch-compile-mode`**：推荐使用 `default`（冷缓存首次编译约 65–100s，热缓存启动约 22s；`max-autotune` 编译耗时约 4 倍且推理延迟无明显优势；设置为`none` 则为 eager 模式）。
- **`--warmup`**：在监听端口前先用示例观测完成编译，避免首个客户端请求卡顿（开启 RTC 时会自动预热带约束路径）。
- **编译缓存（`XPU_CACHE_DIR`）**：持久化保存 inductor 和 triton 编译产物，支持多容器共享。**升级 torch、驱动或镜像版本后，请清空该目录重建。**

### 4.2 CPU 绑核（Guided RTC 必须）

- **原因**：Guided RTC 含反向传播，CPU 主线程与 autograd 线程交替工作；在宿主机 `powersave` 调频策略下，不绑核会导致核心降频，单次延迟从 ~235ms 劣化至最高 490ms+。
- **做法**：用 `lscpu -e=CPU,CORE` 查找属于同一个空闲物理核（CORE）的两个逻辑超线程（例如 B70 上的 `3,27`），通过 `DOCKER_RUN_EXTRA="--cpuset-cpus=3,27 ..."` 绑定。
- **提示**：绑核后，冷编译的编译子进程也只能用这两个逻辑 CPU，会慢多少尚未实测。建议先不绑核完成首次编译（生成 `XPU_CACHE_DIR` 缓存），再绑核启动线上服务。

### 4.3 RTC 与模型参数注意事项

- **RTC 参数位置**：`--rtc.enabled`、`--rtc.mode`（`GUIDED` 或 `TRAINED`）属于 `serve_policy.py` 顶层参数，**必须写在 `policy:checkpoint` 之前**，否则 tyro 会报错。
- **Trained RTC**：需配合专用配置（如 `pi05_teleavatar_v2_rtc`）。对已按 RTC 训练的 checkpoint，**切勿**添加 `--rtc.trained-prefix-noise`（会导致动作接缝剧烈跳变）。
- **端到端替身测试（无 GPU 客户端）**：服务端启动后，可在另一终端启动无 GPU 容器进行联调（详见 `openpi` 的 `docs/xpu.md`）：
  ```bash
  OPENPI_DIR=/path/to/openpi XPU=0 GPUS=none DOCKER_RUN_EXTRA="--network host" \
    scripts/run.sh python examples/teleavatar_v2/fake_env_rtc.py --rtc --num-episodes 1 --record /outputs/check.json
  ```

### 4.4 GPU 监控

`xpu-smi` 的卡号与 `ZE_AFFINITY_MASK` 一致，但 **`xpu-smi ps` 无法列出容器内进程**。请通过整卡利用率与显存判断占用情况（空闲显存约 41 MiB）：

```bash
xpu-smi dump -d -1 -m 0,18 -n 1
```

---

## 5. 修改生效与镜像构建

### 5.1 修改如何生效

| 修改内容 | 生效方式 |
| --- | --- |
| 源码、训练/推理配置（`config.py` 等）、命令行参数 | 下次启动容器即生效 |
| PyTorch 补丁（`src/openpi/models_pytorch/transformers_replace/`） | 下次启动容器即生效（入口脚本自动拷入环境） |
| Python 依赖（`deps/pyproject.toml`、`deps/uv.lock`） | **需重建镜像**（切勿在容器内执行 `uv sync` 或 `pip install`） |

> **核对镜像内依赖**：可执行 `OPENPI_DIR=/path/to/openpi scripts/run.sh bash -c "cd /opt/openpi-xpu-env && uv sync --frozen --dry-run --no-default-groups --group xpu --group dev --no-install-project --no-install-workspace"`，应输出 `Would make no changes`。详见 `deps/NOTE.md`。