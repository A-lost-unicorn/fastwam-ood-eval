# 工程亮点、难点与阻碍台账

本文是持续更新的工程复盘和简历素材库。只记录能由代码、测试或实验工件支持的事实；`plan`、mock 和单元测试不能写成真实机器人任务结果。

## 1. 当前事实快照

截至 2026-07-22：

| 项目 | 状态 | 可用证据 |
| --- | --- | --- |
| 配置、planner、adapter、分片、resume、聚合 | 已实现 | `src/`、`configs/`、`tests/` |
| 自动化测试 | 已通过 | `pytest -q`：29 passed；不代表真实仿真通过 |
| Conda 环境与激活入口 | 已配置 | `scripts/create_env.sh`、`scripts/activate_env.sh` |
| checkpoint/stats | 未下载 | `checkpoints/fastwam_release/` 仅有 `.gitkeep` |
| LIBERO-Plus assets | 未下载 | 目标 `third_party/LIBERO-plus/libero/libero/assets/` 不存在 |
| 单卡 Clean/OOD smoke | 未执行 | 无真实 episode result/video |
| 三卡真实 pilot | 未执行 | 只有配置/规划能力，无真实 worker result |
| full OOD 结果和鲁棒性结论 | 未执行 | 不得填写成功率或性能下降 |

## 2. 可以对外说明的工程亮点

| 亮点 | 设计与价值 | 实现证据 | 验证状态 |
| --- | --- | --- | --- |
| 不侵入上游的适配层 | 不修改 Fast-WAM、LIBERO、LIBERO-Plus；复用官方 checkpoint loader、观测/动作处理和 success 判定 | `policy/fastwam_adapter.py`、`envs/libero_adapter.py` | 代码完成，真实 smoke 待验证 |
| 同名 backend 隔离 | 原版与 Plus 都导出 `libero`；为每个进程生成隔离的 `LIBERO_CONFIG_PATH` 并只加载一个 checkout，避免 import/path 污染 | `envs/libero_adapter.py`、`evaluator.py` | 单元链路完成，真实 smoke 待验证 |
| 可复现任务规划 | 每个 job 固化 suite、base/upstream task、seed、init index、扰动身份和策略身份；job ID 由规范化内容哈希生成 | `evaluation/jobs.py`、`job_manifest.jsonl` | 已单测 |
| episode-level 多 GPU | 每 GPU 一个独立 evaluator，按 job hash 稳定分片；避免把独立 rollout 错做模型 DDP | `distributed_launcher.py`、`shard_jobs()` | 分片已单测，三卡真实运行待验证 |
| 可恢复执行 | worker 逐 episode 追加并 `fsync` JSONL；默认跳过完成 job，支持 failed/all 重跑策略 | `evaluation/resume.py`、`schemas/episode_result.py` | 已单测 |
| 科学比较门禁 | Clean/OOD 共用 seed 公式和 checkpoint；聚合时同一策略 checkpoint hash 不一致则拒绝比较 | `reproducibility.py`、`analysis/aggregate.py` | 已单测，真实配对待验证 |
| 上游协议显式化 | 区分 Clean 多 seed 与 Plus 每官方变体 1 次；`all_once` 强制 `episodes_per_task=1`，防止 10,030×20 的重复计算 | `config.py`、`jobs.py`、`eval_ood_full.yaml` | 已单测，正式 manifest 需重建 |
| 可审计 OOD 元数据 | 记录官方 category、difficulty、classification ID、variant name、candidate/selection 信息和上游 commit | `jobs.py`、`episode_result.py` | planner 已验证，运行时参数采集仍有限 |
| 研究结论防越界 | 明确 release Fast-WAM、Joint WAM、IDM 是不同架构/权重；训练配方不匹配时禁止把比较写成未来想象的因果增益 | `config.py`、`thought1_generalization.md` | 配置门禁已单测，匹配权重缺失 |
| 失败分析闭环 | 记录 action/robot state trace、异常、失败视频和聚合统计，提供静态 failure review 页面 | `recording/`、`analysis/review.py` | mock/单测完成，真实失败样本待积累 |

## 3. 难点、阻碍、方案与剩余风险

### 3.1 Fast-WAM 与 LIBERO 的依赖年代不一致

- 问题：LIBERO README 的旧训练环境与 Fast-WAM 当前 Python/PyTorch/CUDA/MuJoCo 组合冲突；直接照两份安装文档叠加会降级 torch 或污染系统 Python。
- 方案：以 Fast-WAM 的 Python 3.10、PyTorch 2.7.1+cu128 栈为主，在项目目录创建隔离 Conda 环境；不使用 sudo，不安装 LIBERO 的旧 torch 训练栈。
- 证据：`scripts/create_env.sh`、`scripts/activate_env.sh`、`docs/environment_setup.md`。
- 状态：环境已配置；真实 MuJoCo/Fast-WAM 联合 smoke 仍是最终兼容性证明。

### 3.2 原版 LIBERO 与 LIBERO-Plus 使用同一个 Python 包名

- 问题：两者都导出 `libero==0.1.0`，同时 editable install 或同进程切换会得到依赖加载顺序相关的结果。
- 方案：不同时安装两套包；adapter 在新进程中选择对应 checkout，并为每个实验写隔离路径配置。
- 取舍：实现简单、无需 policy server；代价是 Clean/OOD 必须分进程运行。
- 状态：实现完成，真实双 backend smoke 待验证。

### 3.3 评测单位容易被误解，可能放大到 200,600 次 rollout

- 问题：原需求的“每条件至少 20 episode”若机械套到 10,030 个 Plus task instance，会产生 `10,030 × 20`，且偏离本项目采用的上游协议。
- 方案：增加 `sample/all_once` 两种明确模式；正式 Plus 使用 `all_once + episodes_per_task=1`，配置校验拒绝不一致组合；Clean 仍可多 seed。
- 结果：当前五类分级范围为 6,771 runnable，而不是 10,030×20。
- 状态：代码和单测完成；磁盘上的旧 12,800-job manifests 已过期，正式运行前必须重新 plan。

### 3.4 公平比较需要跨环境稳定配对

- 问题：如果 condition/category 参与 seed，或 Clean/OOD 使用不同 checkpoint，性能差无法归因于环境扰动。
- 方案：seed 只由 base seed、suite、base task 和 episode index 派生；condition 不进入公式；结果记录 checkpoint SHA-256，聚合器校验一致性。
- 取舍：`all_once` 的多个 OOD variant 都与同一 base task 的 Clean index 0 对照，这是 task-instance 协议下的一对多配对，不等同于每个 variant 有多个重复 seed。
- 状态：单测完成，真实结果待验证。

### 3.5 海量独立 episode 的分布式调度与断点续跑

- 问题：任务时长不同、进程可能中断；用 DDP 不会提高独立环境 rollout 的资源利用率，粗粒度文件覆盖又会丢失进度。
- 方案：按 job hash 对 rank 稳定分片，每 GPU 一个模型/环境进程；每个 episode 完成即追加 durable JSONL；resume 按 job ID 去重。
- 取舍：静态分片简单可复现，但极端任务时长差异可能造成尾部负载不均；如 pilot 证明明显失衡，再设计动态队列。
- 状态：分片/resume 已单测，三卡负载与 EGL 行为待 pilot 测量。

### 3.6 checkpoint 与 dataset stats 是一组实验条件

- 问题：模型可以成功加载，但错用 stats 会改变动作反归一化，产生难以察觉的错误结果。
- 方案：配置显式要求两条路径，实施手册要求从同一 release 下载并记录二者 SHA-256。
- 剩余风险：当前 doctor 只检查文件存在；程序只哈希 checkpoint，尚未记录 stats hash 和 Hugging Face revision。
- 后续：把 stats hash/revision 纳入 provenance，并为已知 release 建可选校验表。

### 3.7 Headless MuJoCo 与多 GPU EGL 绑定

- 问题：服务器无显示环境，torch 的可见 GPU 编号又会被 `CUDA_VISIBLE_DEVICES` 重映射。
- 方案：使用 `MUJOCO_GL=egl`；单卡显式 `MUJOCO_EGL_DEVICE_ID=0`，torchrun 按 `LOCAL_RANK` 设置 EGL device 和 policy device。
- 状态：launcher 已实现，真实三卡 EGL smoke 待验证。

### 3.8 “实际扰动参数”目前只做到可追溯，尚未完全结构化

- 问题：官方 classification 只保证 task ID、类别和 difficulty；相机、光照、布局细节分散在 BDDL、XML、robot class、init files 和 wrapper 中。
- 当前方案：保存 classification ID、variant name、官方类别/难度、selection metadata 和上游 commit，并用视频肉眼复核。
- 缺口：尚未把实际 camera pose、FOV、light properties 等统一解析进 result；因此不能声称“底层数值参数已自动记录”。
- 后续：为五类分别实现 runtime introspection/schema，并保存来源文件路径/hash；无法读取的字段显式记为 `unknown`。

### 3.9 上游分类数据存在未分级记录和空分层

- 问题：五类共 6,892 行，但 121 条 Goal/Light 记录的 difficulty 为 null；另有 68 个 base-task/category/level 笛卡尔分层没有候选。
- 方案：不猜测 null difficulty；主分级实验排除并单独报告 121 条。空分层生成 `skipped` 审计行，不进入成功率分母。
- 后续选择：若要覆盖五类全部 6,892 行，新增明确的 `ungraded` bucket，仍保持每 variant 1 次。
- 状态：分级子集逻辑已实现；是否增加 `ungraded` 是实验协议决策。

### 3.10 视觉与 observation 证据仍不完整

- 问题：当前视频只保存 `agentview_image`；`recording.save_observations` 已进入配置但尚未真正落盘，无法仅靠视频证明 wrist camera 正常。
- 临时方案：smoke 强制保存所有 agent-view 视频，并结合 processed image shape、action trace 和 robot state 验收。
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

