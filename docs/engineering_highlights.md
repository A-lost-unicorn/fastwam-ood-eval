# 工程亮点、难点与阻碍台账

本文是持续更新的工程复盘和简历素材库。只记录能由代码、测试或实验工件支持的事实；`plan`、mock 和单元测试不能写成真实机器人任务结果。

## 1. 当前事实快照

截至 2026-07-22：

| 项目 | 状态 | 可用证据 |
| --- | --- | --- |
| 配置、planner、adapter、分片、resume、聚合 | 已实现 | `src/`、`configs/`、`tests/` |
| 自动化测试 | 已通过 | `pytest -q`：43 passed；包含 PyTorch 2.6+ 安全边界、LIBERO-Plus init-state 路由与 10,030 个官方变体全量路径审计 |
| Conda 环境与激活入口 | 已配置 | `scripts/create_env.sh`、`scripts/activate_env.sh` |
| checkpoint/stats | 已下载并人工校验 | checkpoint SHA-256 `1000437c...a49579`；stats SHA-256 `30f81ad7...68638` |
| FastWAM 公共运行时模型 | 已下载并逐文件校验 | `scripts/download_fastwam_runtime_models.sh`；T5、VAE、tokenizer 共约 11.9 GiB |
| LIBERO-Plus assets | 已下载并检查目录结构 | `articulated_objects/`、`new_objects/`、`scenes/`、`textures/` 等已就位 |
| 单卡 Clean smoke | 已通过 | 2026-07-22：2 episodes、2 success、0 exception；仅证明链路可用，不作为成功率估计 |
| 单卡 OOD smoke | 已通过 | 2026-07-22：camera/light 共 4 episodes、4 success、0 exception；仅证明链路可用，不作为成功率估计 |
| 三卡真实 pilot | 已通过 | 2026-07-22：9 planned、8 completed、1 expected skipped、0 exception；三个 rank 均有真实结果 |
| 正式 manifests | 已重建并审计 | 800 Clean；6,839 OOD planned=6,771 runnable+68 skipped；无正式 worker result |
| full OOD 结果和鲁棒性结论 | 未执行 | 不得把 pilot 的 2/8 写成正式成功率或性能下降 |

## 2. 可以对外说明的工程亮点

| 亮点 | 设计与价值 | 实现证据 | 验证状态 |
| --- | --- | --- | --- |
| 不侵入上游的适配层 | 不修改 Fast-WAM、LIBERO、LIBERO-Plus；复用官方 checkpoint loader、观测/动作处理和 success 判定 | `policy/fastwam_adapter.py`、`envs/libero_adapter.py` | Clean 2-episode 与 Plus 4-episode smoke 均已验证 |
| 同名 backend 隔离 | 原版与 Plus 都导出 `libero`；为每个进程生成隔离的 `LIBERO_CONFIG_PATH` 并只加载一个 checkout，避免 import/path 污染 | `envs/libero_adapter.py`、`evaluator.py` | Clean 与 Plus 已分别在真实独立进程验证 |
| 可复现任务规划 | 每个 job 固化 suite、base/upstream task、seed、init index、扰动身份和策略身份；job ID 由规范化内容哈希生成 | `evaluation/jobs.py`、`job_manifest.jsonl` | 已单测 |
| episode-level 多 GPU | 每 GPU 一个独立 evaluator，按 job hash 稳定分片；避免把独立 rollout 错做模型 DDP | `distributed_launcher.py`、`shard_jobs()` | 三卡真实 pilot 已验证 3/4/2 分片，无重复遗漏 |
| 可恢复执行 | worker 逐 episode 追加并 `fsync` JSONL；默认跳过完成 job，支持 failed/all 重跑策略 | `evaluation/resume.py`、`schemas/episode_result.py` | 已单测；Clean smoke 用 `--rerun failed` 从两条真实 exception 恢复成功 |
| 科学比较门禁 | Clean/OOD 共用 seed 公式和 checkpoint；聚合时同一策略 checkpoint hash 不一致则拒绝比较 | `reproducibility.py`、`analysis/aggregate.py` | 已单测；Clean/OOD smoke 的 checkpoint SHA-256 已实测一致 |
| 上游协议显式化 | 区分 Clean 多 seed 与 Plus 每官方变体 1 次；`all_once` 强制 `episodes_per_task=1`，防止 10,030×20 的重复计算 | `config.py`、`jobs.py`、`eval_ood_full.yaml` | 已单测，正式 manifest 需重建 |
| 可审计 OOD 元数据 | 记录官方 category、difficulty、classification ID、variant name、candidate/selection 信息和上游 commit | `jobs.py`、`episode_result.py` | 真实 Plus result 已验证，运行时底层数值参数采集仍有限 |
| 研究结论防越界 | 明确 release Fast-WAM、Joint WAM、IDM 是不同架构/权重；训练配方不匹配时禁止把比较写成未来想象的因果增益 | `config.py`、`thought1_generalization.md` | 配置门禁已单测，匹配权重缺失 |
| 失败分析闭环 | 记录 action/robot state trace、异常、失败视频和聚合统计，提供静态 failure review 页面 | `recording/`、`analysis/review.py` | Clean/OOD smoke 已产出真实 trace/video；失败分类样本仍待积累 |

## 3. 难点、阻碍、方案与剩余风险

### 3.1 Fast-WAM 与 LIBERO 的依赖年代不一致

- 问题：LIBERO README 的旧训练环境与 Fast-WAM 当前 Python/PyTorch/CUDA/MuJoCo 组合冲突；直接照两份安装文档叠加会降级 torch 或污染系统 Python。
- 方案：以 Fast-WAM 的 Python 3.10、PyTorch 2.7.1+cu128 栈为主，在项目目录创建隔离 Conda 环境；不使用 sudo，不安装 LIBERO 的旧 torch 训练栈。
- 证据：`scripts/create_env.sh`、`scripts/activate_env.sh`、`docs/environment_setup.md`。
- 状态：环境已配置；2026-07-22 已通过真实 MuJoCo/Fast-WAM Clean smoke。PyTorch 2.6+ 的 init-state 兼容问题见 3.13。

### 3.2 原版 LIBERO 与 LIBERO-Plus 使用同一个 Python 包名

- 问题：两者都导出 `libero==0.1.0`，同时 editable install 或同进程切换会得到依赖加载顺序相关的结果。
- 方案：不同时安装两套包；adapter 在新进程中选择对应 checkout，并为每个实验写隔离路径配置。
- 取舍：实现简单、无需 policy server；代价是 Clean/OOD 必须分进程运行。
- 状态：Clean 与 Plus 已分别在独立真实评测进程完成 smoke；同一进程切换仍明确禁止。

### 3.3 评测单位容易被误解，可能放大到 200,600 次 rollout

- 问题：原需求的“每条件至少 20 episode”若机械套到 10,030 个 Plus task instance，会产生 `10,030 × 20`，且偏离本项目采用的上游协议。
- 方案：增加 `sample/all_once` 两种明确模式；正式 Plus 使用 `all_once + episodes_per_task=1`，配置校验拒绝不一致组合；Clean 仍可多 seed。
- 结果：当前五类分级范围为 6,771 runnable，而不是 10,030×20。
- 状态：代码和单测完成；磁盘上的旧 12,800-job manifests 已过期，正式运行前必须重新 plan。

### 3.4 公平比较需要跨环境稳定配对

- 问题：如果 condition/category 参与 seed，或 Clean/OOD 使用不同 checkpoint，性能差无法归因于环境扰动。
- 方案：seed 只由 base seed、suite、base task 和 episode index 派生；condition 不进入公式；结果记录 checkpoint SHA-256，聚合器校验一致性。
- 取舍：`all_once` 的多个 OOD variant 都与同一 base task 的 Clean index 0 对照，这是 task-instance 协议下的一对多配对，不等同于每个 variant 有多个重复 seed。
- 状态：单测完成；Clean/OOD smoke 的 checkpoint hash 一致，seed 0/1 可配对。正式统计仍需 full 实验。

### 3.5 海量独立 episode 的分布式调度与断点续跑

- 问题：任务时长不同、进程可能中断；用 DDP 不会提高独立环境 rollout 的资源利用率，粗粒度文件覆盖又会丢失进度。
- 方案：按 job hash 对 rank 稳定分片，每 GPU 一个模型/环境进程；每个 episode 完成即追加 durable JSONL；resume 按 job ID 去重。
- 取舍：静态分片简单可复现，但极端任务时长差异可能造成尾部负载不均；如 pilot 证明明显失衡，再设计动态队列。
- 状态：resume 已经真实 exception→failed-only rerun 验证；三卡 pilot 的 3/4/2 静态分片均完成，无重复遗漏。pilot 样本太小，正式长任务的尾部负载仍需监控。

### 3.6 checkpoint 与 dataset stats 是一组实验条件

- 问题：模型可以成功加载，但错用 stats 会改变动作反归一化，产生难以察觉的错误结果。
- 方案：配置显式要求两条路径，实施手册要求从同一 release 下载并记录二者 SHA-256。
- 剩余风险：当前 doctor 只检查文件存在；程序只哈希 checkpoint，尚未记录 stats hash 和 Hugging Face revision。
- 后续：把 stats hash/revision 纳入 provenance，并为已知 release 建可选校验表。

### 3.7 Headless MuJoCo 与多 GPU EGL 绑定

- 问题：服务器无显示环境，torch 的可见 GPU 编号又会被 `CUDA_VISIBLE_DEVICES` 重映射。
- 方案：使用 `MUJOCO_GL=egl`；单卡显式 `MUJOCO_EGL_DEVICE_ID=0`，torchrun 按 `LOCAL_RANK` 设置 EGL device 和 policy device。
- 状态：单卡 GPU 0/EGL 与三卡 torchrun/EGL 均已通过真实运行；pilot 峰值显存约 23.8 GB/卡，无 OOM。

### 3.8 “实际扰动参数”目前只做到可追溯，尚未完全结构化

- 问题：官方 classification 只保证 task ID、类别和 difficulty；相机、光照、布局细节分散在 BDDL、XML、robot class、init files 和 wrapper 中。
- 当前方案：保存 classification ID、variant name、官方类别/难度、selection metadata 和上游 commit，并用视频肉眼复核。2026-07-22 的 OOD smoke 已确认 camera 构图变化和 light 明暗/阴影变化。
- 缺口：尚未把实际 camera pose、FOV、light properties 等统一解析进 result；因此不能声称“底层数值参数已自动记录”。
- 后续：为五类分别实现 runtime introspection/schema，并保存来源文件路径/hash；无法读取的字段显式记为 `unknown`。

### 3.9 上游分类数据存在未分级记录和空分层

- 问题：五类共 6,892 行，但 121 条 Goal/Light 记录的 difficulty 为 null；另有 68 个 base-task/category/level 笛卡尔分层没有候选。
- 方案：不猜测 null difficulty；主分级实验排除并单独报告 121 条。空分层生成 `skipped` 审计行，不进入成功率分母。
- 后续选择：若要覆盖五类全部 6,892 行，新增明确的 `ungraded` bucket，仍保持每 variant 1 次。
- 状态：分级子集逻辑已实现；是否增加 `ungraded` 是实验协议决策。

### 3.10 视觉与 observation 证据仍不完整

- 问题：当前视频只保存 `agentview_image`；`recording.save_observations` 已进入配置但尚未真正落盘，无法仅靠视频证明 wrist camera 正常。
- 临时方案：smoke 强制保存所有 agent-view 视频，并结合 processed image shape、action trace 和 robot state 验收。2026-07-22 的独立 reset probe 已确认 agent-view 与 wrist camera 都返回 `256×256×3`，但落盘视频仍只有 agent-view。
- 后续：增加双相机 contact sheet/短视频及小尺寸 observation diagnostic，避免保存全量原始 observation 造成 I/O 爆炸。
- 状态：已识别，未解决。

### 3.11 Future imagination 的因果问题缺少匹配 checkpoint

- 问题：Fast-WAM release 的 uncond 动作路径不读取预测未来；Joint WAM/IDM 是不同模型和训练权重。不同来源 checkpoint 的胜负不能归因于“测试时开关未来想象”。
- 方案：配置显式区分模型 variant，检查 checkpoint 命名，并要求相同非空 `training_recipe_id` 才允许因果表述。
- 状态：评测框架已准备；官方匹配的 Joint/IDM checkpoint 不存在，本阶段保持阻塞而不伪造结论。

### 3.12 LIBERO-Plus 许可证边界不清晰

- 问题：当前锁定 commit 根目录没有明确 LICENSE，不能推定沿用原版 LIBERO 的 MIT。
- 方案：仅把上游作为 `third_party` checkout 使用，不复制或重新分发其代码/assets；在 `upstream_notes.md` 记录风险。
- 状态：风险已识别，公开分发前仍需上游确认。

### 3.13 PyTorch 2.6+ 默认安全加载与旧 LIBERO init-state 不兼容

- 现象：release checkpoint 成功加载后，两个 Clean smoke job 都在 `environment.reset()` 报 `_pickle.UnpicklingError`；上游 `suite.get_task_init_states()` 调用未带参数的 `torch.load()`。
- 根因：PyTorch 2.6 起 `torch.load()` 默认 `weights_only=True`，而 LIBERO 的 `.init/.pruned_init` 保存的是 NumPy 数组，不是纯 tensor state dict。
- 方案：不修改 pinned `third_party/LIBERO`；在项目 adapter 中复现最小加载路径，并只对 checkout 内 `init_files/` 下、扩展名为 `.init` 或 `.pruned_init` 的常规文件显式使用 `weights_only=False`。
- 安全取舍：旧 pickle 模式可能执行恶意 payload，因此先 `resolve(strict=True)`，拒绝通过符号链接或 `..` 逃出受信任根目录，并拒绝未知扩展名；该例外不用于 checkpoint 或用户任意路径。
- 验证：新增 3 个回归测试；真实 `libero_spatial` task 0 reset 成功，两路相机正常；随后 `--rerun failed` 完成 2/2 episodes，聚合结果为 0 exception。
- 证据：`src/fastwam_ood_eval/envs/libero_adapter.py`、`tests/test_libero_adapter.py`、`outputs/clean_smoke/workers/rank_0/episode_results.jsonl`、`outputs/clean_smoke/summary/metrics.json`。

### 3.14 LIBERO-Plus 视觉变体不提供同名 init-state 文件

- 现象：Plus checkpoint 和模型加载完成后，4 个 OOD smoke job 都在 `reset()` 立即失败；程序尝试读取带 `_view_...` 或 `_light_...` 后缀的 `.pruned_init`，但这些文件不存在。
- 根因：LIBERO-Plus 的 BDDL task 是独立变体，但相机、光照、语言等视觉扰动有意复用基础任务的 init state；table/background、light 和 new-object/level 还有不同的路径重写规则。为修复 PyTorch 2.6 兼容而绕过上游 `get_task_init_states()` 后，项目只保留了安全加载，却遗漏了 Plus 的路径解析语义。
- 方案：把“解析可信相对路径”和“显式 `weights_only=False` 加载”分层；Plus adapter 复现 pinned upstream 的顺序覆盖规则，new-object/level 文件从 `init_files/libero_newobj/` 读取并 reshape，同时抑制上游每次构造 suite 打印数千 task ID 的调试输出。
- 关键细节：规则必须顺序覆盖而不是互斥 `elif`。例如基础任务名本身可能包含 `_table_center`，后续 `_light_`、`_tb_` 或 `_add_` 规则必须覆盖误匹配，否则仍会生成错误路径。
- 验证：10 个规则/优先级单测，加上对官方 `task_classification.json` 全部 10,030 行的路径存在性审计；真实 reset probe 覆盖两条 camera、两条 light 变体；修复后 failed-only rerun 完成 4/4、0 exception。
- 运行证据：四条 action trace 均为 finite 且非全零，末端执行器首末位移约 0.36–0.39 m；四个 MP4 均可解码，camera/light 首帧可见不同构图或明暗；Clean/OOD checkpoint SHA-256 均为 `1000437c...a49579`。
- 证据：`src/fastwam_ood_eval/envs/libero_plus_adapter.py`、`tests/test_libero_plus_adapter.py`、`outputs/ood_smoke/summary/metrics.json`、`outputs/ood_smoke/summary/report.md`。

## 4. 简历表达素材

### 当前即可使用的版本（不包含虚构实验结果）

- 搭建 Fast-WAM 在 LIBERO/LIBERO-Plus 上的配置驱动 OOD 评测框架，以 adapter 隔离同名仿真 backend，并支持单卡调试与 episode-level 多 GPU 推理。
- 设计确定性 job manifest、哈希分片、逐 episode JSONL 落盘与断点续跑机制，保证大规模机器人 rollout 可复现、可审计、可恢复。
- 将 Clean 多 seed 与 LIBERO-Plus 预生成 task-instance 协议显式分离，通过配置门禁阻止每变体重复采样造成的数量级计算浪费。
- 建立相同 checkpoint/配对 seed 的鲁棒性评测与统计链路，覆盖成功率下降、bootstrap CI、失败分类和跨策略配方一致性约束。

### 真实实验完成后再填写的量化模板

不要在结果出来前填数字：

> 在 3 张 GPU 上评测 Fast-WAM 的 `[N]` 个 Clean/OOD task instances，借助断点续跑和任务分片将 `[指标]` 从 `[基线]` 改善至 `[结果]`；识别 `[最敏感扰动]` 导致绝对成功率下降 `[X]` 个百分点，并基于 `[M]` 个失败视频归纳 `[K]` 类主要失败模式。

推荐补齐的量化证据：

| 指标 | 来源 |
| --- | --- |
| runnable/completed/exception/skipped 数 | manifest 与 episode JSONL |
| 单 episode p50/p95 时长 | episode result |
| 每 GPU 峰值显存、吞吐、负载不均衡 | episode result 与系统监控 |
| Clean/OOD 成功率、绝对/相对下降、95% CI | aggregate/report |
| 最敏感类别和强度趋势 | summary by perturbation/level |
| 断点恢复节省的重复 episode 数 | resume 日志 |
| 失败模式数量和占比 | failure review annotations |

## 5. 面试叙事提纲

可以按下面顺序讲，避免只罗列技术名词：

1. 研究问题：同一 Fast-WAM checkpoint 在环境 shift 下会掉多少，而不是训练一个新模型。
2. 核心约束：两个同名 `libero` backend、24 GB 级显存预算、海量独立 rollout、可中断服务器任务和严格配对要求。
3. 关键决策：adapter 隔离、episode-level data parallel、确定性 manifest/resume、checkpoint+seed 科学门禁。
4. 发现并纠正的协议问题：Plus 的评测单位是预生成 task instance，每个变体 1 次；不能机械执行 10,030×20。
5. 尚未解决但已诚实限定的问题：底层扰动参数、双相机证据、null difficulty、许可证和 future checkpoint 可识别性。
6. 最后用真实 pilot/full 数字回答效果、成本和失败模式；数字未产生前明确说“待实测”。

## 6. 更新规则

每次完成新阶段后更新本文：

- 把状态从“待验证”改为“已验证”时，必须附命令、日期和工件路径。
- 配置、上游 commit 或分类文件变化时，重新记录 manifest 数量和协议差异。
- 失败和回滚也要记录，不能只保留成功路径。
- 简历中的每个数字必须能追溯到 `experiment_manifest.json`、episode JSONL 或聚合报告。
- 不把 mock、dry-run、plan、pytest 或 doctor 成功写成真实策略成功率。
