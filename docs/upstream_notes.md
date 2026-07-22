# Upstream survey (2026-07-21)

本页记录阶段 A 的代码级勘察。三个 checkout 均为 shallow clone；更新上游后必须重新核对本页和真实 smoke test。

| Upstream | Remote | Branch | Commit |
|---|---|---|---|
| Fast-WAM | `https://github.com/yuantianyuan01/FastWAM.git` | `main` | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| LIBERO | `https://github.com/Lifelong-Robot-Learning/LIBERO.git` | `master` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus | `https://github.com/sylvestf/LIBERO-plus.git` | `main` | `4976dc30028e805ff8094b55501d532c48fec182` |

## Fast-WAM

- 安装：README 指定 Python 3.10，PyTorch `2.7.1+cu128`、torchvision `0.22.1+cu128`，然后 `pip install -e .`；`pyproject.toml` 也要求 Python >=3.10 并固定这些 torch 版本。
- LIBERO 评测入口：`experiments/libero/run_libero_manager.py` 生成 suite/task 列表，再调用 `run_libero_parallel_test.sh`，最终每个任务运行 `eval_libero_single.py`。
- checkpoint：Hydra 参数 `ckpt`；真实 loader 是 `model.load_checkpoint(ckpt)`。发布权重和 dataset stats 由 Hugging Face 仓库 `yuanty/fastwam` 下载。
- model/config：发布 LIBERO 任务配置为 `configs/task/libero_uncond_2cam224_1e-4.yaml`；默认 manager 配置在 `configs/sim_libero.yaml`。
- observation：官方 helper 从 `agentview_image` 与 `robot0_eye_in_hand_image` 取图并各旋转 180°，center-crop/resize 后水平拼接；proprio 是 `eef_pos(3) + eef_quat axis-angle(3) + gripper_qpos(2)`，共 8 维。
- action：`model.infer_action()` 返回 `[T, 7]` action chunk；dataset stats 反归一化后，官方代码恢复 gripper 符号并可二值化。默认 `num_frames=33`，所以 action horizon 为 32；执行侧默认每次 replan 10 步。
- success：`OffScreenRenderEnv.step()` 返回的 `done` 被当作成功；suite 最大步数 spatial/object/goal=400，libero_10/90=700；动作前默认 30 次 no-op。
- 并行：manager 默认 8 GPU，通过 shell/tmux 做 task-level 多进程调度，不是 DDP；配置还允许一个 GPU 同时多个 task。这个项目改为确定性的 episode job 分片，并强制每 GPU 一个 worker。
- 已提供：checkpoint loader、LIBERO evaluator、多 GPU manager、rollout/未来视频保存、success 判定。
- 未提供：policy server/client；逐 episode JSONL；配对 seed；稳健 resume；失败视频-only；推理 latency/显存统计。
- 许可证：仓库根目录 `LICENSE` 是 MIT（2026 FastWAM Authors）。

### 测试时未来想象的代码级语义

- release 只提供 `libero_uncond_2cam224.pt`，对应 `FastWAM`/`libero_uncond_2cam224_1e-4`。
- `FastWAM._build_mot_attention_mask()` 的 action→video 区域只开放首帧 token；`infer_action()` 只编码首帧并缓存其 KV，然后去噪 action。因此 release 动作推理不依赖预测未来帧。
- 对 uncond 模型调用 `infer_joint()` 虽会额外生成视频，但相同 attention mask 仍禁止 action 读取未来 token；不能将 `visualize_future_video=true/false` 当作 future imagination on/off。
- `FastWAMJoint` 覆盖 attention mask，让 action 读取全部 video token；其 `infer_action()` 实际共同去噪未来视频 latent 与 action。`FastWAMIDM` 则先生成未来视频，再由动作分支读取视频。
- 三种上游 task 配置分别为 `libero_uncond_*`、`libero_joint_*`、`libero_idm_*`，属于需要各自训练 checkpoint 的模型变体。当前 README/Hugging Face 下载命令只列 uncond release，故本项目不声称已具备 on/off 因果消融条件。
- 2026-07-22 再核对官方 Hugging Face 模型页，文件仍只有 LIBERO/RoboTwin uncond checkpoint。第三方 BadWAM 模型页新提供 Joint/IDM 权重，但只声明使用默认 task 配置（Joint metadata 标 step 21700），未给出足以确认其与官方 uncond 初始化、seed、优化预算和精确数据版本配对的证据，且许可证标为 `other`。本项目只把它们列为可选的相关性基线，不自动下载，也不授予因果解释资格。

## LIBERO

- README 安装示例固定 Python 3.8.13、PyTorch 1.11/cu113，`requirements.txt` 固定 NumPy 1.22.4、Hydra 1.2、robosuite 1.4.0 等。
- 当前 `setup.py` 只声明 `python_requires>=3` 且不自动安装依赖。Fast-WAM README 明确要求在其 Python 3.10/PyTorch 2.7 环境安装官方 LIBERO，并另外固定 MuJoCo 3.3.2。因此本项目以 Fast-WAM 环境为主，LIBERO 的旧 torch 训练栈不安装；这一路径仍需真实 smoke test 验证。
- task suite：`benchmark.get_benchmark_dict()[suite]()`；四个目标 suite 分别是 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，每个 10 个标准任务。
- task：`get_task(task_id)` 返回 name/language/problem_folder/BDDL；`get_task_init_states(task_id)` 返回固定初始状态。
- environment：`OffScreenRenderEnv(bddl_file_name=..., camera_heights=..., camera_widths=...)`，然后 `seed()`、`reset()`、`set_init_state()`；action 为 7 维；稀疏成功由环境任务 `_check_success()`/done 决定。
- path config：上游 import 时读取 `LIBERO_CONFIG_PATH/config.yaml`；缺失时会交互式询问并写用户 home。本项目在 experiment output 内预生成隔离 config，避免多 backend 争用或 headless worker 卡住。
- dataset：`benchmark_scripts/download_libero_datasets.py`；评测发布 checkpoint 不需要训练 dataset，但需要 BDDL、init states 和 assets。
- 许可证：代码 MIT；README 将数据集声明为 CC BY 4.0。

## LIBERO-Plus

- 使用方式不是运行时 `perturb(category, strength)` API。它是原版 LIBERO 的同名替代 package，用 10,030 个预生成 task variant（BDDL、scene XML、robot class、init files 和 wrapper 参数）表达扰动。
- suite 任务数：spatial 2402、object 2518、goal 2591、libero_10 2519。分类文件实际位于 `libero/libero/benchmark/task_classification.json`。
- 七个真实类别：`Objects Layout`、`Camera Viewpoints`、`Robot Initial States`、`Language Instructions`、`Light Conditions`、`Background Textures`、`Sensor Noise`。本阶段只选择其中前述五个环境类扰动，不包含 language/noise。
- 难度是整数 1–5，不是 easy/medium/hard；另有 121 条 Light Conditions 的 difficulty 为 null，本项目不抽取这些记录。统一映射：easy=1–2、medium=3、hard=4–5，并把原始值写入每个 job。
- camera/robot/noise 参数编码在带 `_view_..._initstate_...` 的 BDDL 路径中，由 `env_wrapper.ControlEnv` 解析；noise 确实在 observation wrapper 中实现。背景、光照和物体布局使用生成的 BDDL、scene XML、额外 objects 与 init files。本项目不自行造扰动。
- README 要求每个 Plus task `num_trials_per_task=1`；正式配置以 `all_once` 枚举每个有难度标签的官方 variant 一次，smoke/pilot 才使用确定性 `sample`，不会在同一 variant 上机械重复 20 次。
- assets：README 要求从 Hugging Face `Sylvest/LIBERO-plus` 下载 `assets.zip` 并解压到 `LIBERO-plus/libero/libero/assets`。
- 安装：`setup.py` 导出同名 `libero==0.1.0`，且不声明 dependencies；`requirements.txt` 与原版接近，`extra_requirements.txt` 增加 usd-core、Wand、scikit-image 等。
- 许可证风险：该 commit 根目录没有 `LICENSE`/`COPYING`，README 也未声明代码许可证。不能推定其沿用 LIBERO MIT；分发、修改或公开发布含其代码/资产前，应向上游确认许可证。本项目不复制 Plus 源码。

## README 与代码差异/注意点

1. LIBERO-Plus README 写分类路径为 `.libero/libero/benchmark/task_classification.json`，checkout 中的实际相对路径没有开头的点。
2. Fast-WAM README 将多 GPU 描述为 manager 评测；实际 shell 使用 tmux 动态 task 调度，默认可在一张 GPU 同时跑多个任务。本项目不会复用这一并发策略。
3. Fast-WAM evaluator 使用 `torch.no_grad()`，并无 `torch.inference_mode()`；adapter 在调用官方 helper 外层补充 inference mode。
4. Fast-WAM evaluator无条件保存所有 rollout 视频；本项目 recorder 实现可配置的 failure-only。
5. Fast-WAM 的 simulator seed 是整个 worker 的 `cfg.seed`，并未逐 episode 变化或写入结果；本项目为每个 job 明确分配/记录 seed。

## 集成决策

采用一个 Python 3.10/Fast-WAM 环境，但每个进程只把原版或 Plus 的一个 checkout 放到 import path。Clean 与 OOD 是两个独立命令，使用同一 checkpoint/dataset stats 和同一 seed 公式。只有在这一兼容环境经真实 smoke test 证明不可行后，才考虑 policy server/client 双环境；当前不引入该复杂度。
