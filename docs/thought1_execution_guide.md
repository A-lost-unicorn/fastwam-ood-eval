# 思考点 1：实施与验收手册

本文是“标准 LIBERO → LIBERO-Plus 环境扰动鲁棒性评测”的操作手册。它回答“按什么顺序做、每一步看什么证据、什么时候必须停”，不代替研究结论边界或实验结果。

当前状态（2026-07-22）：代码、配置、本地 Conda 环境、官方 Fast-WAM checkpoint/配套 stats、运行时公共模型和 LIBERO-Plus assets 均已准备。单卡 Clean smoke（2/2 completed）、单卡 OOD smoke（4/4 completed）和三卡 pilot（8 completed、1 expected skipped、0 exception）均已通过；正式 manifests 已按当前协议重建，full 尚未运行。smoke/pilot 的小样本成功率只用于链路验收，不是性能结论。

配套文档：

- [阶段报告](thought1_report.md)：研究问题、当前证据、pilot 结果、正式 manifest 和资源预算。
- [研究结论边界](thought1_generalization.md)：本阶段能回答和不能回答什么。
- [完成度审计](thought1_readiness.md)：当前哪些内容已实现、哪些仍待实测。
- [环境配置](environment_setup.md)：Python、PyTorch、CUDA、MuJoCo/EGL 兼容方案。
- [实验协议](experiment_protocol.md)：配对、统计和成功率口径。
- [上游勘察](upstream_notes.md)：锁定的上游 commit、真实 API 与许可证。

## 0. 先锁定协议和安全门禁

本项目采用当前锁定的 LIBERO-Plus 上游仓库所描述的 **10,030 个预生成 task-instance 协议**：正式 OOD 计划逐个枚举选中的官方变体，每个变体运行 1 次。它不是某些其他 wrapper 所采用的“标准 40 task、每 task 多次 rollout”接口；两种结果不能混称为同一协议。

当前范围进一步限定为五类环境扰动：camera、light、background、robot initial state 和 object layout，不含 language 与 sensor noise。因此：

- `10,030` 是七类扰动、四个 suite 的全 benchmark 总数，不是本项目当前配置的 job 数。
- 当前五类有 6,892 条分类记录；其中 121 条 `libero_goal / Light Conditions` 没有官方 difficulty，不能擅自映射到 easy/medium/hard，当前分级主实验明确排除并报告它们。
- 当前锁定分类文件下，四个 suite 的 full OOD 计划应为 6,771 个 runnable job，加 68 个用于审计缺失分层的 `skipped` 占位；Clean 为 800 个 job。上游或配置改变后，必须以新生成的 manifest 为准。
- Clean 可用多个初始化 index/seed 建立稳定基线；当前 full 配置为每个标准 task 20 次。正式 Plus 使用 `variant_selection: all_once` 和 `episodes_per_task: 1`，每个官方变体只运行一次，并与同一基础 task 的 Clean index/seed 0 配对。

安全门禁：

1. 在单卡 Clean smoke、单卡 OOD smoke 和三卡 pilot 全部通过前，不运行 full 配置。
2. 任何配置、分类文件或 planner 代码改变后，都要重新运行 `plan`。`evaluate` 会复用已有 `job_manifest.jsonl`，不会自动判断旧 manifest 已经过期。
3. 每次 `plan` 后先审核 job 数、runnable/skipped 数和抽样记录，再决定是否运行。
4. 不因追求“至少 20 次”而执行 `10,030 × 20`。这既是巨额重复计算，也不符合本项目采用的 upstream task-instance 协议。
5. 不使用 `--overwrite`，除非已经明确决定重跑已有结果。

## 1. 激活环境并做只读预检

每次进入新 shell 后运行：

```bash
cd /data/HDD_16TB_WORK/users/tao/projects/fastwam-ood-eval
source scripts/activate_env.sh
```

先确认当前解释器和基础工程状态：

```bash
which python
python --version
if command -v tree >/dev/null 2>&1; then
  tree -L 3
else
  find . -maxdepth 3 -not -path './.git/*' -print | sort
fi
pytest -q
fastwam-ood plan --config configs/eval_ood_smoke.yaml
```

`scripts/create_env.sh` 已把 `tree` 列为环境工具；早于该改动创建的环境可能没有它。`find` 只用于结构检查，不影响项目运行。

`plan` 只生成/刷新 job manifest，不加载 checkpoint、模型或仿真器。由于外部工件尚未准备，带配置的 `doctor` 此时失败是预期现象；应在下面对应工件准备完成后再次执行。

## 2. 阶段 1：准备 checkpoint 和配套 stats

项目提供的规范入口是：

```bash
bash scripts/download_checkpoints.sh
```

它等价于官方发布页中的 LIBERO 子集下载命令；不要重复执行两种方式：

```bash
mkdir -p checkpoints/fastwam_release

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --local-dir checkpoints/fastwam_release
```

下载后检查文件和哈希：

```bash
ls -lh checkpoints/fastwam_release
sha256sum \
  checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

至少应有：

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
└── libero_uncond_2cam224_dataset_stats.json
```

复现官方 release 基线时，必须固定使用随该 checkpoint 发布的 dataset stats。重新计算 stats 会改变动作归一化条件，只能作为显式的新实验条件，不能冒充官方基线。当前程序会哈希 checkpoint，但还不会自动证明 stats 与 checkpoint 配套，因此要保存两者的 SHA-256 和下载 revision。

完成后执行 Clean 配置门禁：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood doctor --config configs/eval_clean_smoke.yaml
```

只有输出包含 `configuration valid`、`runtime paths present` 和 `configured CUDA inventory available` 时，才进入 Clean smoke。

## 3. 阶段 2：准备 LIBERO-Plus assets

从官方数据仓库下载 `assets.zip`。国内链路不稳定时，项目脚本默认使用非官方的 `hf-mirror.com`，只对这一次公开文件下载生效；它不会把镜像写入全局 Conda 环境，也会关闭 token 的隐式发送。脚本保留 `.incomplete`、使用单 worker 并在断流后自动重试：

```bash
bash scripts/download_libero_plus_assets.sh
```

若镜像不可用，可在不删除断点文件的前提下切回官方源：

```bash
HF_ENDPOINT=https://huggingface.co \
  bash scripts/download_libero_plus_assets.sh
```

下载成功后记录哈希。注意解压目标是 `libero/libero/` 父目录，让压缩包生成 `assets/`；不要解压到已存在的 `assets/` 中，否则可能形成错误的 `assets/assets/`：

```bash
sha256sum third_party/LIBERO-plus/.downloads/assets.zip
unzip third_party/LIBERO-plus/.downloads/assets.zip \
  -d third_party/LIBERO-plus/libero/libero/
```

最终应至少包含：

```text
third_party/LIBERO-plus/libero/libero/assets/
├── articulated_objects/
├── new_objects/
├── stable_hope_objects/
├── stable_scanned_objects/
├── textures/
├── turbosquid_objects/
├── serving_region.xml
├── wall_frames.stl
└── wall.xml
```

验收：

```bash
test -d third_party/LIBERO-plus/libero/libero/assets/articulated_objects
test -d third_party/LIBERO-plus/libero/libero/assets/new_objects
test -d third_party/LIBERO-plus/libero/libero/assets/textures
test -f third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood doctor --config configs/eval_ood_smoke.yaml
```

LIBERO 与 LIBERO-Plus 都导出名为 `libero` 的包。本项目按独立进程选择 checkout，不要在同一个 Python/notebook 进程里先后加载两个 backend。

## 4. 阶段 2.5：准备 FastWAM 运行时公共模型

FastWAM 在应用 release checkpoint 之前还会加载 Wan2.2 VAE、UMT5-XXL
文本编码器和 tokenizer。这些文件不包含在 `yuanty/fastwam` 的两个 release
文件中；缺失时上游加载器会隐式下载约 11.9 GiB。为了避免评测进程在模型初始化时
联网，先显式下载并校验：

```bash
bash scripts/download_fastwam_runtime_models.sh
```

脚本使用 ModelScope 国内源、单 worker 和自动重试。ModelScope SDK 的
`snapshot_download()` 在这里使用仓库分支 `master`；仓库元数据返回的 commit SHA
不能作为该接口的 revision。真正用于复现锁定的是脚本内的逐文件 SHA-256，任何上游
内容变化都会让脚本失败，而不是静默继续。

下载完成后应存在：

```text
checkpoints/
├── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
│   ├── Wan2.2_VAE.safetensors
│   └── models_t5_umt5-xxl-enc-bf16.safetensors
└── Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
    ├── special_tokens_map.json
    ├── spiece.model
    ├── tokenizer.json
    └── tokenizer_config.json
```

若连接中断，直接重跑同一个脚本；不要删除 `._____temp`，ModelScope 会利用其中的
临时文件继续处理。

## 5. 阶段 3：单卡 Clean smoke test

为了无论成功或失败都能肉眼检查视频，计划与执行时都覆盖 `save_failure_video_only=false`：

```bash
fastwam-ood plan \
  --config configs/eval_clean_smoke.yaml \
  --set experiment.save_failure_video_only=false

DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true \
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate \
  --config configs/eval_clean_smoke.yaml \
  --device cuda:0 \
  --set experiment.save_failure_video_only=false
```

这一阶段的验收目标不是成功率，而是完整链路：

| 验收项 | 证据 | 通过条件 |
| --- | --- | --- |
| checkpoint 加载 | 运行日志、episode result 的 `checkpoint_hash` | 无 checkpoint/load exception，hash 非空 |
| 环境 reset | `episode_results.jsonl` | `termination_reason` 不是 `exception` |
| 主相机图像 | `workers/rank_0/videos/*.mp4`、`observation_image_shape` | agent view 清晰、方向正常、shape 合理 |
| 动作有效 | `workers/rank_0/traces/*.jsonl` | action 非空、全部 finite、不是全 0 |
| 机器人运动 | 视频与 trace 中首末 `robot0_eef_pos` | 肉眼或状态差异可见 |
| episode 结束 | `termination_reason` | 为 `success` 或 `max_steps`，不是异常退出 |
| 结果落盘 | worker JSONL、聚合后的 JSONL | 每个 planned job 恰好一条最新结果；worker JSONL 可保留同一 job 的历史重试记录 |

快速检查工件：

```bash
find outputs/clean_smoke/workers/rank_0 -maxdepth 2 -type f -print
fastwam-ood aggregate --experiment-dir outputs/clean_smoke
head -n 1 outputs/clean_smoke/summary/episode_results.jsonl | python -m json.tool
```

如果某个 job 以 `exception` 或 `max_steps` 结束且问题已经修复，只重跑失败项：

```bash
DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true \
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate \
  --config configs/eval_clean_smoke.yaml \
  --device cuda:0 \
  --rerun failed \
  --set experiment.save_failure_video_only=false
```

不要删除旧 JSONL；它是尝试历史。resume 与 aggregate 都会按 `job_id` 采用最后一条
记录，因此修复后的结果会替代旧异常进入汇总，同时保留故障审计证据。

检查 action 是否 finite 且不是全零：

```bash
python - <<'PY'
import json
import math
from pathlib import Path

values = []
for path in Path("outputs/clean_smoke/workers/rank_0/traces").glob("*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        action = json.loads(line)["action"]
        if isinstance(action, list):
            values.extend(float(value) for value in action)

assert values, "no recorded actions"
assert all(math.isfinite(value) for value in values), "NaN/Inf action detected"
assert any(abs(value) > 1e-8 for value in values), "all actions are zero"
print({"action_values": len(values), "max_abs": max(abs(value) for value in values)})
PY
```

当前 recorder 只保存 `agentview_image`，因此视频不能单独证明 wrist camera 正常；原始 observation dump 也尚未实现。若两路相机都是正式验收要求，需要在 full 前补充专门诊断。

## 6. 阶段 4：单卡 OOD smoke test

```bash
fastwam-ood plan \
  --config configs/eval_ood_smoke.yaml \
  --set experiment.save_failure_video_only=false

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate \
  --config configs/eval_ood_smoke.yaml \
  --device cuda:0 \
  --set experiment.save_failure_video_only=false
```

为避免模型加载阶段隐式联网，推荐与 Clean 一样显式设置本地模型目录：

```bash
DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true \
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate \
  --config configs/eval_ood_smoke.yaml \
  --device cuda:0 \
  --set experiment.save_failure_video_only=false
```

除 Clean 的全部验收项外，还要确认：

1. 对照 Clean/OOD 视频，主相机视角、光照或背景确实按所选 variant 改变，而不是只有配置名变化。
2. `job_manifest.jsonl` 和 episode result 中存在 `classification_id`、`variant_name`、`official_category`、`official_difficulty` 和 selection metadata。
3. Clean/OOD episode result 的 `checkpoint_hash` 完全一致。
4. OOD 的 `(suite, base task, episode_seed)` 能在 Clean 结果中找到对应项。
5. 所有 `skipped` 都有明确 `skip_reason`，且不进入成功率分母。

当前 manifest 记录的是官方 variant 身份与分类元数据，不是所有底层相机位姿、光源参数或 XML 属性的规范化展开。数值级“实际扰动参数”尚未实现自动采集；在补齐运行时 introspection 前，必须依靠 variant 名称/ID、上游 commit、任务文件和视频共同审计，不能把这一项写成已自动验证。

2026-07-22 的真实 OOD smoke 结果：camera/light 各 2 个变体，共 4/4 completed、4 success、0 exception。四条轨迹 action 均 finite 且非全零，末端执行器位移约 0.36–0.39 m，四个 MP4 均可解码；首帧抽检确认 camera 构图与 light 明暗/阴影确实变化。Clean/OOD checkpoint SHA-256 完全一致。证据位于 `outputs/ood_smoke/summary/` 和 `outputs/ood_smoke/workers/rank_0/`；该结果只证明链路，不用于估计正式 OOD 成功率。

若修复异常后仅需恢复失败 job，使用：

```bash
DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true \
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate \
  --config configs/eval_ood_smoke.yaml \
  --device cuda:0 \
  --rerun failed \
  --set experiment.save_failure_video_only=false
```

不要删除 worker JSONL 中的旧异常行；aggregate 按 `job_id` 采用最后一条记录，旧行保留为故障审计证据。Plus 的视觉变体经常复用基础任务 init state，因此缺少“与 variant 同名”的 `.pruned_init` 不代表 assets 下载不完整；应由 adapter 按 pinned upstream 规则解析，不能临时复制或伪造 init 文件。

## 7. 阶段 5：三卡小规模 pilot

运行 pilot 只选择三类：

- `camera_viewpoints`：论文和上游报告的高敏感因素。
- `robot_initial_states`：论文和上游报告的高敏感因素。
- `objects_layout`：用于额外验证几何、物体资产和布局链路。

先检查并重新生成 manifest：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 MUJOCO_GL=egl \
  fastwam-ood doctor --config configs/eval_ood_pilot.yaml

fastwam-ood plan --config configs/eval_ood_pilot.yaml
wc -l outputs/ood_pilot/job_manifest.jsonl
```

当前锁定配置计划 9 条：8 条 runnable、1 条 `skipped`。该 skip 来自 task 4 的 `objects_layout/easy` 没有官方候选，不是运行故障；若修改 task 子集，必须重新 plan 并重新记录计数。

审核通过并明确决定启动小规模真实试运行后：

```bash
DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true \
CUDA_VISIBLE_DEVICES=0,1,2 MUJOCO_GL=egl \
torchrun \
  --standalone \
  --nproc_per_node=3 \
  -m fastwam_ood_eval.cli distributed-evaluate \
  --config configs/eval_ood_pilot.yaml
```

launcher 会按 `LOCAL_RANK` 选择 `cuda:0/1/2`，并在 EGL 模式下设置对应的 `MUJOCO_EGL_DEVICE_ID`。验收时检查三个 `workers/rank_*` 目录，合并后 `job_id` 不重复，并用真实 episode 墙钟时间估算 full 成本。

2026-07-22 的真实 pilot 已完成：rank 0/1/2 分别处理 3/4/2 条，最终为 8 completed、1 expected skipped、0 exception；8 条 action trace 均 finite 且非全零，8 个 MP4 均可解码。2 条 success、6 条 `max_steps` 是诊断性策略结果，不是系统故障，也不能用作正式成功率。模型并发加载约 369 秒，总墙钟约 11 分 35 秒，平均 episode 71.53 秒，峰值显存约 23.8 GB/卡。机器汇总见 `outputs/ood_pilot/summary/`。

不要把这个运行 pilot 与 `scripts/plan_thought1_pilot.sh` 混淆：前者是单个 suite、三类 easy 扰动的 3-GPU 真实链路测试；后者只生成四个 suite、五类三档的 64-job 规划矩阵，不启动模型。

## 8. 正式计划与运行口径

前五阶段已经通过，正式计划已用下列命令重新生成：

```bash
bash scripts/plan_thought1.sh
```

对当前 pinned classification 和当前五类分级配置，2026-07-22 实际生成的四个 suite 合计：

```text
Clean:        800 planned
OOD:        6,839 planned = 6,771 runnable + 68 skipped audit rows
Total:      7,639 planned
Rollouts:   7,571 runnable = 800 Clean + 6,771 OOD
Excluded:     121 ungraded Light Conditions rows（单独报告，不擅自分级）
```

每次都应重新从实际 manifest 计算这些数字；不要把它们当作跨上游版本的常量。只执行 6,771 个 OOD job 不能得到 Clean→OOD drop，正式研究结论还需要 800 个 Clean baseline。按 pilot 速度理想线性外推约需 50.1 三卡墙钟小时；考虑 `libero_10` 的 700 steps、尾部不均衡和 I/O，应预留 60–72 小时。正式执行仍需逐个 suite 审核配置、checkpoint/stats hash、输出目录和视频策略，并由用户显式确认后再调用三卡 launcher。

### 8.1 单卡 RTX 4090 全量脚本

单卡入口为 `scripts/run_thought1_single_gpu_full.sh`。它会自动激活项目环境、只暴露指定物理 GPU、依次运行四个 suite、使用 incomplete-only resume、逐目录聚合，并在 `all` 完成后生成 combined report：

```bash
CONFIRM_FULL_EVAL=YES GPU_ID=0 \
  bash scripts/run_thought1_single_gpu_full.sh all
```

阶段参数：

```text
all    800 Clean + 6,771 OOD，最后合并报告
clean  仅运行 800 Clean
ood    仅运行 6,771 OOD
```

标准 24 GB RTX 4090 的余量很小：pilot 峰值约 23.8 GB。脚本默认要求启动时至少 24,000 MiB 空闲显存，并拒绝在同一输出目录启动第二个单卡 full 进程。运行前应确保所选 GPU 没有桌面渲染或其他训练进程。按 pilot 平均时长理想外推约 150 GPU-hours；考虑 `libero_10`、模型重复加载和 I/O，单卡应预留约 7–9 天。

推荐使用 tmux：

```bash
tmux new -s thought1-single

CONFIRM_FULL_EVAL=YES GPU_ID=0 \
  bash scripts/run_thought1_single_gpu_full.sh all
```

按 `Ctrl+B`、再按 `D` 脱离；之后使用 `tmux attach -t thought1-single` 返回。

也可以使用 nohup：

```bash
mkdir -p outputs/logs
run_log="outputs/logs/thought1_single_gpu_$(date +%Y%m%d_%H%M%S).log"

nohup env CONFIRM_FULL_EVAL=YES GPU_ID=0 \
  bash scripts/run_thought1_single_gpu_full.sh all \
  >"${run_log}" 2>&1 &

run_pid=$!
printf '%s\n' "${run_pid}" > outputs/logs/thought1_single_gpu.pid
echo "PID=${run_pid} LOG=${run_log}"
```

监控：

```bash
tail -f "${run_log}"
nvidia-smi
ps -fp "$(cat outputs/logs/thought1_single_gpu.pid)"
```

SSH 中断后 nohup 进程会继续。若任务自身失败或机器重启，重新运行相同命令即可；脚本会读取所有现存 `rank_*` 结果并只补 incomplete job。不要使用 `--overwrite`，也不要常规重跑合法的 `max_steps` 失败。

## 9. 立即停止条件

出现下列任一情况时停止扩大规模并保留现场：

- checkpoint 或 stats 来源/hash 不明确；Clean/OOD checkpoint hash 不同。
- reset、EGL、asset、camera 或 policy inference 出现 exception。
- action 为空、含 NaN/Inf、全零，或 robot state/视频表明机器人不运动。
- OOD 视频与 Clean 无可见差异，或 variant 元数据缺失。
- manifest 的 job 数仍符合旧的重复 20 次模式，或 manifest schema 与当前 planner 不一致。
- 多卡 rank 出现重复 job/缺失结果，或任一运行方式出现显存不足和大量 `exception`。
- 无法提供至少 60–72 小时连续三卡窗口或 7–9 天连续单卡窗口，或磁盘余量尚未检查。

## 10. 每阶段应保留的证据

```text
checkpoint/stats SHA-256
assets.zip SHA-256 与来源 revision
experiment_manifest.json
job_manifest.jsonl
workers/rank_*/episode_results.jsonl
workers/rank_*/traces/*.jsonl
smoke/pilot videos
pytest 输出
doctor 输出
上游三个 commit SHA
```

官方来源：

- [Fast-WAM release checkpoint 说明](https://github.com/yuantianyuan01/FastWAM#inference-with-released-checkpoints)
- [Fast-WAM Hugging Face 模型库](https://huggingface.co/yuanty/fastwam)
- [LIBERO-Plus 安装与评测说明](https://github.com/sylvestf/LIBERO-plus#evaluation)
- [LIBERO-Plus assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main)
