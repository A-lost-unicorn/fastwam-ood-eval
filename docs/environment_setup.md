# Environment setup

## 推荐环境

选择 Python 3.10、Fast-WAM 指定的 PyTorch 2.7.1+cu128 与 torchvision 0.22.1+cu128。LIBERO 的 README 是旧训练环境说明；Fast-WAM 当前 README 明确在自己的环境中安装 LIBERO 并使用 MuJoCo 3.3.2。不要安装 LIBERO 自带的旧 PyTorch 1.11。

```bash
bash scripts/fetch_upstreams.sh
bash scripts/create_env.sh fastwam-ood
conda activate fastwam-ood
python -m pip install mujoco==3.3.2
```

脚本不使用 sudo、不修改系统 Python。LIBERO-Plus README 还列出若干系统 library；本项目不会自动 sudo 安装。缺库时由管理员安装 `libexpat`、fontconfig、Python runtime 和 ImageMagick/Wand 对应开发包。

## CUDA 检查

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
fastwam-ood doctor
```

## checkpoint 与 dataset stats

```bash
bash scripts/download_checkpoints.sh
sha256sum checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

评测需要 checkpoint 和配套 dataset stats；训练 dataset 不需要。若要准备 Fast-WAM 训练数据，请按上游 README 从 `yuanty/LIBERO-fastwam` 下载；本阶段不训练。

## LIBERO assets/config

上游首次 import 默认会询问并创建用户级 `.libero/config.yaml`。本项目不会修改该文件：每个实验在 `outputs/<experiment>/runtime/<backend>/config.yaml` 生成隔离配置，并在 import 前通过上游支持的 `LIBERO_CONFIG_PATH` 选择它。最终 `benchmark_root`、`bddl_files`、`init_states`、`datasets`、`assets` 路径会记录到 experiment manifest。Clean 指向 `third_party/LIBERO`，OOD 指向 `third_party/LIBERO-plus`。

从 `Sylvest/LIBERO-plus` 下载 `assets.zip`，解压后的目录必须为：

```text
third_party/LIBERO-plus/libero/libero/assets/
```

不要把 assets、datasets、checkpoint 或视频提交到 Git。

## Headless EGL

```bash
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python -m fastwam_ood_eval.cli doctor --config configs/eval_clean_smoke.yaml
```

torchrun worker 看到重新编号后的本地 GPU。若 MuJoCo 要求每进程独立 EGL ID，可由集群 launcher 按 `LOCAL_RANK` 设置；先用单 GPU smoke test 验证。

## 4 GPU 检查与 smoke test

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-smi --query-gpu=index,name,memory.total --format=csv
pytest -q
fastwam-ood plan --config configs/eval_ood_smoke.yaml
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 bash scripts/run_smoke_test.sh
```

在 smoke test 成功前不要启动 full 配置。当前 checkout 没有 checkpoint/Plus assets 的验证记录；真实接口状态为未验证。
