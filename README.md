# fastwam-ood-eval

## 1. 这个项目研究什么

这个项目只回答一个问题：**同一份 Fast-WAM 权重在标准 LIBERO 和环境发生分布外变化的 LIBERO-Plus 中，成功率会下降多少？**

标准评测的心智模型是：

```text
Fast-WAM policy → 标准 LIBERO observation → action chunk → LIBERO environment → success / failure
```

OOD 评测的心智模型是：

```text
Fast-WAM policy → LIBERO-Plus 扰动后的 observation/environment → action chunk → 扰动环境 → success / failure
```

Clean 和 OOD 使用同一个 checkpoint，并为同一个基础任务/episode 分配相同 seed。模型权重不变，唯一改变的是测试环境；这样观察到的差距才尽量可以归因于扰动。本阶段测量的是**鲁棒性**，不是重新训练后的“泛化提升”。

## 2. Fast-WAM、ID、OOD 分别是什么

- Fast-WAM 是被评测的策略。适配器复用官方 checkpoint loader、图像/本体状态预处理和 action 后处理，不复制模型内部实现。
- ID（in-distribution）表示测试条件接近训练或标准评测条件；这里用原版 LIBERO 作为 clean/ID 基线。
- OOD（out-of-distribution）表示相机、光照、背景、机器人初始姿态或物体布局发生变化；这里使用 LIBERO-Plus 官方预生成任务变体。

绝对成功率下降：

```text
clean_success_rate - ood_success_rate
```

相对性能下降：

```text
(clean_success_rate - ood_success_rate) / max(clean_success_rate, epsilon)
```

## 3. 当前阶段不做什么

不训练 Fast-WAM，不修改主模型，不实现 Future Adapter、Joint WAM 或历史记忆，也不把仿真 OOD 结论外推成真机结论。未运行的实验不会填入虚构结果。

## 4. 项目架构图

```text
官方 Fast-WAM checkpoint
            │
            ▼
    FastWAMAdapter
            │
      ┌─────┴─────┐
      ▼           ▼
Clean LIBERO   LIBERO-Plus OOD
      │           │
      └─────┬─────┘
            ▼
       Episode Results
            │
            ▼
       Aggregate & Report
```

原版 LIBERO 与 LIBERO-Plus 都导出名为 `libero` 的 Python 包，不能同时 import。本项目在一个兼容环境中按**进程**选择一个 backend：Clean 命令加载 `third_party/LIBERO`，OOD 命令加载 `third_party/LIBERO-plus`。模型无需通过网络 server/client 通信。

正式并行是 episode-level data parallel：每张 GPU 一个独立进程，`job_id % world_size` 稳定分片；不是模型 DDP。

## 5. 安装

先获取代码并查看已锁定的提交：

```bash
bash scripts/fetch_upstreams.sh
git -C third_party/FastWAM rev-parse HEAD
git -C third_party/LIBERO rev-parse HEAD
git -C third_party/LIBERO-plus rev-parse HEAD
```

推荐创建 Python 3.10 的独立 Conda 环境：

```bash
bash scripts/create_env.sh fastwam-ood
conda activate fastwam-ood
```

LIBERO-Plus 还需要单独下载 assets；无显示服务器需要 `MUJOCO_GL=egl`。精确步骤、兼容性理由和 assets 目录见 [环境文档](docs/environment_setup.md)。不要同时 `pip install -e third_party/LIBERO` 和 `pip install -e third_party/LIBERO-plus`。

## 6. 下载 checkpoint

```bash
bash scripts/download_checkpoints.sh
```

默认配置期望：

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

先检查，不启动实验：

```bash
fastwam-ood doctor --config configs/eval_clean_smoke.yaml
```

没有 checkpoint/GPU 时，可先验证完整 CPU mock 数据链路：

```bash
fastwam-ood plan --config configs/eval_mock_smoke.yaml
fastwam-ood evaluate --config configs/eval_mock_smoke.yaml
fastwam-ood aggregate --experiment-dir outputs/mock_smoke
```

## 7. 单 GPU smoke test

`plan` 只写 job manifest，不加载 checkpoint、模型或环境：

```bash
fastwam-ood plan --config configs/eval_clean_smoke.yaml
fastwam-ood evaluate --config configs/eval_clean_smoke.yaml --device cuda:0 --dry-run
```

检查计划后再执行真实 smoke test（1 task × 2 episodes）：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate --config configs/eval_clean_smoke.yaml --device cuda:0
```

## 8. Clean baseline

```bash
fastwam-ood plan --config configs/eval_clean_full.yaml
fastwam-ood evaluate --config configs/eval_clean_full.yaml --device cuda:0
```

这会使用原版 LIBERO 的官方 task suite、固定 init states、稀疏成功判定和 Fast-WAM 官方预/后处理。

## 9. OOD 测试

第一版选取 LIBERO-Plus 的真实类别 `Camera Viewpoints`、`Light Conditions`、`Background Textures`、`Robot Initial States`、`Objects Layout`。统一等级映射为 easy=官方 1–2、medium=3、hard=4–5；每个 job 都保留真实官方难度、classification ID 和变体名。

```bash
fastwam-ood plan --config configs/eval_ood_smoke.yaml
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate --config configs/eval_ood_smoke.yaml --device cuda:0
```

LIBERO-Plus 的扰动来自其 BDDL、场景 XML、robot class、init state 和 observation wrapper，不在本项目中伪造图像。

## 10. 4 GPU 正式评测

先确认 checkpoint、suite、task 子集、20+ episodes、最大步数、五类扰动、三个等级和输出目录。脚本要求显式确认：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIRM_FULL_EVAL=YES \
  bash scripts/run_4gpu_eval.sh configs/eval_ood_full.yaml
```

等价核心命令：

```bash
torchrun --standalone --nproc_per_node=4 \
  -m fastwam_ood_eval.cli distributed-evaluate \
  --config configs/eval_ood_full.yaml
```

每个 rank 写入 `outputs/<experiment>/workers/rank_N/`。默认 resume 不重复已落盘 job；`--rerun failed` 只重跑 exception/max_steps，`--overwrite` 重跑全部已分配 job。

## 11. 聚合结果

```bash
fastwam-ood aggregate --experiment-dir outputs/ood_full
fastwam-ood report --experiment-dir outputs/ood_full
```

输出位于 `summary/`：JSONL、episode CSV、按任务/扰动/等级 CSV、failures、metrics 和 `report.md`。成功率 CI 使用固定随机种子的 95% bootstrap；若同时聚合 Clean 与 OOD 记录，还会给出配对 seed 的四格计数。Clean 与 OOD 分在两个目录时，建立一个比较输出目录并显式传入两者：

```bash
fastwam-ood aggregate --experiment-dir outputs/clean_vs_ood \
  --input-dir outputs/clean_full \
  --input-dir outputs/ood_full
```

## 12. 查看失败视频

```bash
fastwam-ood review-failures --experiment-dir outputs/ood_full
# 浏览器打开 outputs/ood_full/failure_review/index.html
```

页面不需要后端；标注保存在浏览器 localStorage，并可导出 `annotations.json`。默认只保留失败视频。

## 13. 如何理解结果

报告能够说明 Fast-WAM 对已测扰动是否敏感、哪类/哪个强度下降最大，以及标准分布与 OOD 分布的实测差距。它不能说明显式未来想象一定能修复 OOD、Fast-WAM 完全没有世界建模能力、所有 WAM 都不需要未来想象，或仿真与真机 OOD 等价。详细统计口径见 [实验协议](docs/experiment_protocol.md)。

## 14. 常见报错

- `checkpoint ... missing`：运行下载脚本，或覆盖 `checkpoint.path` 和 `checkpoint.dataset_stats_path`。
- `A different libero package is already loaded`：Clean/OOD 必须分别启动新进程，不要在 notebook 内切换 backend。
- MuJoCo/EGL 初始化失败：设置 `MUJOCO_GL=egl` 与当前 worker 对应的 `MUJOCO_EGL_DEVICE_ID`。
- LIBERO-Plus asset not found：按环境文档将官方 `assets.zip` 解压到正确目录。
- CUDA OOM：保持每 GPU 一个 worker，确认 4090 实际可用显存，并降低并发；不要设置 `workers_per_gpu > 1`。

更多见 [故障排查](docs/troubleshooting.md)。

## 15. 下一阶段路线

先完成所有 suite 的 Clean/OOD 配对基线和人工失败标注；再依据最敏感的扰动与失败类型提出下一阶段假设。任何 Future Adapter、Joint WAM 或历史记忆实验都应作为新的训练/消融项目，不混入本仓库第一阶段基线。
