# Troubleshooting

- **找不到 checkpoint/stats**：使用 `--set checkpoint.path=...` 和 `--set checkpoint.dataset_stats_path=...`，再运行 doctor。
- **同名 libero 冲突**：不要 pip editable-install 两套 LIBERO；每条 Clean/OOD 命令使用新 Python 进程。
- **Plus task 数量看起来不是 10**：正常。Plus 的四个 suite 是数千个预生成 variant；本项目用 classification 映射回 10 个基础任务。
- **某条件 skipped**：查看 `perturbation_parameters` 与 error；classification 中该基础任务/等级可能没有官方 variant。
- **EGL/GL error**：设置 `MUJOCO_GL=egl`，检查 NVIDIA driver 和 EGL library；先只暴露一张 GPU。
- **assets not found**：确认 `third_party/LIBERO-plus/libero/libero/assets`，以及 `.libero/config.yaml` 没有指向旧 checkout。
- **CUDA OOM**：每 GPU 保持一个 worker；检查其他进程和 24 GB 卡的实际空闲量。不要用 Fast-WAM 上游的 `MAX_TASKS_PER_GPU>1`。
- **中断/半行 JSONL**：直接重启同一命令。resume 忽略不完整行并跳过完整 job；用 `--rerun failed` 或 `--overwrite` 控制重跑。
- **report 全是 N/A**：Clean/OOD 记录尚未放在同一个实验目录。聚合器只能比较它实际发现的 worker JSONL。

