# 工程亮点、难点与阻碍台账

本文是持续更新的工程复盘和简历素材库。只记录能由代码、测试或实验工件支持的事实；`plan`、mock 和单元测试不能写成真实机器人任务结果。

## 1. 当前事实快照

截至 2026-07-23：

| 项目 | 状态 | 可用证据 |
| --- | --- | --- |
| 配置、planner、adapter、分片、resume、聚合 | 已实现 | `src/`、`configs/`、`tests/` |
| 自动化测试 | 已通过 | `pytest -q`：160 passed；覆盖阶段一评测、PyTorch 2.6+ 安全边界、LIBERO-Plus 10,030 行路径审计、阶段二诊断、static calibration、label-blind packet 与 outcome-blind freeze 门禁 |
| Conda 环境与激活入口 | 已配置 | `scripts/create_env.sh`、`scripts/activate_env.sh` |
| checkpoint/stats | 已下载并人工校验 | checkpoint SHA-256 `1000437c...a49579`；stats SHA-256 `30f81ad7...68638` |
| FastWAM 公共运行时模型 | 已下载并逐文件校验 | `scripts/download_fastwam_runtime_models.sh`；T5、VAE、tokenizer 共约 11.9 GiB |
| LIBERO-Plus assets | 已下载并检查目录结构 | `articulated_objects/`、`new_objects/`、`scenes/`、`textures/` 等已就位 |
| 单卡 Clean smoke | 已通过 | 2026-07-22：2 episodes、2 success、0 exception；仅证明链路可用，不作为成功率估计 |
| 单卡 OOD smoke | 已通过 | 2026-07-22：camera/light 共 4 episodes、4 success、0 exception；仅证明链路可用，不作为成功率估计 |
| 三卡真实 pilot | 已通过 | 2026-07-22：9 planned、8 completed、1 expected skipped、0 exception；三个 rank 均有真实结果 |
| 正式 manifests | 已重建并审计 | 800 Clean；6,839 OOD planned=6,771 runnable+68 skipped；无正式 worker result |
| full OOD 结果和鲁棒性结论 | 未执行 | 不得把 pilot 的 2/8 写成正式成功率或性能下降 |
| 阶段二 2A unconditional future | 20-step PILOT 已通过 | smoke 后完成 Clean/OOD 5 episodes、7 probes、14 aligned future frames、0 error；全部媒体可解码 |
| 阶段二标签盲审 | WORKFLOW PILOT 已通过 | 7 cases / 28 media；public/private hash 校验和全量解码通过；human labels 仍为 0/7 |
| 阶段二 outcome-blind 抽样 | FORMAL-DRAFT 已生成 | 200 Clean + 532 OOD，68 unsupported、0 supported shortfall；当前未冻结 |
| 阶段二 2B action-conditioned future | 严格阻塞 | release 配置为 `action_conditioned=false`，且不存在通过 provenance 门禁的匹配 checkpoint |

## 2. 可以对外说明的工程亮点

| 亮点 | 设计与价值 | 实现证据 | 验证状态 |
| --- | --- | --- | --- |
| 不侵入上游的适配层 | 不修改 Fast-WAM、LIBERO、LIBERO-Plus；复用官方 checkpoint loader、观测/动作处理和 success 判定 | `policy/fastwam_adapter.py`、`envs/libero_adapter.py` | Clean 2-episode 与 Plus 4-episode smoke 均已验证 |
| 同名 backend 隔离 | 原版与 Plus 都导出 `libero`；为每个进程生成隔离的 `LIBERO_CONFIG_PATH` 并只加载一个 checkout，避免 import/path 污染 | `envs/libero_adapter.py`、`evaluator.py` | Clean 与 Plus 已分别在真实独立进程验证 |
| 可复现任务规划 | 每个 job 固化 suite、base/upstream task、seed、init index、扰动身份和策略身份；job ID 由规范化内容哈希生成 | `evaluation/jobs.py`、`job_manifest.jsonl` | 已单测 |
| episode-level 多 GPU | 每 GPU 一个独立 evaluator，按 job hash 稳定分片；避免把独立 rollout 错做模型 DDP | `distributed_launcher.py`、`shard_jobs()` | 三卡真实 pilot 已验证 3/4/2 分片，无重复遗漏 |
| 可恢复执行 | worker 逐 episode 追加并 `fsync` JSONL；默认跳过完成 job，支持 failed/all 重跑策略 | `evaluation/resume.py`、`schemas/episode_result.py` | 已单测；Clean smoke 用 `--rerun failed` 从两条真实 exception 恢复成功 |
| 科学比较门禁 | Clean/OOD 共用 seed 公式和 checkpoint；聚合时同一策略 checkpoint hash 不一致则拒绝比较 | `reproducibility.py`、`analysis/aggregate.py` | 已单测；Clean/OOD smoke 的 checkpoint SHA-256 已实测一致 |
| 上游协议显式化 | 区分 Clean 多 seed 与 Plus 每官方变体 1 次；`all_once` 强制 `episodes_per_task=1`，防止 10,030×20 的重复计算 | `config.py`、`jobs.py`、`eval_ood_full.yaml` | 已单测；正式 manifest 已重建并审计 |
| 可审计 OOD 元数据 | 记录官方 category、difficulty、classification ID、variant name、candidate/selection 信息和上游 commit | `jobs.py`、`episode_result.py` | 真实 Plus result 已验证，运行时底层数值参数采集仍有限 |
| 研究结论防越界 | 明确 release Fast-WAM、Joint WAM、IDM 是不同架构/权重；训练配方不匹配时禁止把比较写成未来想象的因果增益 | `config.py`、`thought1_generalization.md` | 配置门禁已单测，匹配权重缺失 |
| 失败分析闭环 | 记录 action/robot state trace、异常、失败视频和聚合统计，提供静态 failure review 页面 | `recording/`、`analysis/review.py` | Clean/OOD smoke 已产出真实 trace/video；失败分类样本仍待积累 |
| 阶段二只读 shadow probe | 先冻结并哈希基线动作，再从同一 checkpoint 单独生成 future；current/predicted/actual/side-by-side 工件写入独立目录 | `policy/fastwam_future_probe.py`、`diagnostics/` | smoke 与 5-episode 20-step pilot 已验证；不会改写阶段一 source manifest/result |
| 诊断语义双门禁 | 将 release 可支持的 unconditional consistency（2A）与需要匹配 action-conditioned checkpoint 的动力学一致性（2B）分开，禁止静默降级 | `config.py`、`fastwam_future_probe.py`、`thought2_upstream_audit.md` | 2A 实测通过；2B 对 release 预期拒绝 |
| Source rerun 精确复现 | 将阶段二每个 probe 的 executed action 与阶段一 trace 按环境 step 对齐核对 | `diagnostics.jsonl`、阶段一 `traces/*.jsonl` | 7/7 probe 逐元素相同（最大绝对差 0），5/5 outcome 相同 |
| 多输入语义安全聚合 | Clean/OOD comparison 单独生成 manifest，锁定 mode、共同 provenance、输入 fingerprint 与 source hash；unknown mode 不再猜测 | `diagnostics/aggregate.py`、`diagnostics/report.py` | 5-episode comparison 实测并可重复聚合 |
| 独立 null-motion 校准 | 不读取 pilot 标签、不调用 policy action；以同帧编码噪声和 0/4/8 no-op residual 建候选阈值，自动检查 200 条 freeze gate | `diagnostics/static_calibration*.py`、独立 YAML/manifest/JSONL | 7/7 真实 Clean/五类 OOD 样本有效；旧阈值 1.0 与候选 `0.013223` 相差约 75.6× |
| 标签盲化媒体审阅 | 将 condition/outcome/metric/source mapping 放入独立 `0600` private key；公开 packet 使用 opaque alias 和逐媒体 SHA-256；盲态导入区分 missing/uncertain/decisive 并计算 pairwise κ | `diagnostics/blind_review*.py`、静态 HTML/CSV/JSON | 7 个真实 probe 的 28 个媒体全部解码，public sensitive key/token 泄漏为 0；agreement 工具由合成双 reviewer 标签验证，真人标签仍为 0 |
| Outcome-blind 正式抽样 | 只从阶段一 job manifest 分层哈希选样，记录 skipped-only cell，并强制 Clean episode-0 anchor；formal runner 拒绝未冻结草案 | `diagnostics/diagnostic_cohort.py`、`require_frozen_cohort` | v2 草案 732 jobs、0 supported shortfall；freeze 因 dirty tree 有意未通过 |

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

### 3.11 Future imagination 必须拆成一致性问题与因果问题

- 问题：Fast-WAM release 的动作路径不读取预测未来，且 `video_expert.action_conditioned=false`。因此它可以离线产生 unconditional future，却不能证明动作依赖该未来，也不能回答“给定这组动作后的未来是否正确”。
- 方案：阶段二拆成两个不混用的 protocol：2A `unconditional_future` 只测同一 checkpoint 的表征/方向一致性；2B `action_conditioned_future` 要求可信的 action-conditioned 参数、完整动作依赖覆盖和训练 provenance。
- 因果边界：Joint WAM/IDM 是不同架构或权重；不同来源 checkpoint 的胜负不能归因于“测试时开关未来想象”。阶段三使用 frozen backbone、null-adapter control 和 K=0/1/2/4 配对训练，才是测试轻量未来输入因果作用的主路径。
- 状态：2A 已完成真实 GPU smoke；2B 因 release 能力和匹配 checkpoint 缺失而严格阻塞，不能把 2A 改名为 2B。

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

### 3.15 阶段二 probe 必须在构造环境前固定同名 LIBERO backend

- 现象：阶段二首次真实运行在加载 checkpoint 后才发现 policy 的官方 evaluator 会先 import `libero`；原 adapter 只在环境构造时选择 Clean/Plus backend，诊断路径因此可能过早导入错误 checkout。
- 难点：工作区可经 `/home/...` 和 `/data/...` 两条等价路径访问，而且顶层 `libero` 是 namespace package，`__file__` 可能为 `None`。仅比较字符串路径或 `module.__file__` 会误判同一 checkout。
- 方案：增加可在 simulator 构造前调用的 backend 配置入口；对路径先 `resolve()`，同时检查 namespace package 的 `__path__`，并拒绝真正混用的 source root。
- 验证：新增 symlink/namespace 回归测试；修复后同一官方 checkpoint 的 2A smoke 完成 1 episode/1 probe，生成 current、predicted、actual 和 side-by-side 工件，动作哈希前后相同。
- 范围：这证明诊断链路和“只读 shadow”约束成立；单样本指标及 `max_steps=10` 的 smoke 终止都不是性能结论。

### 3.16 物理 EGL ID 与 torch 逻辑 GPU ID 语义不同

- 现象：只暴露物理 GPU 1 时使用 `MUJOCO_EGL_DEVICE_ID=0`，robosuite 在 import
  阶段断言失败；模型尚未加载。
- 根因：robosuite 先检查 EGL ID 是否出现在原始 `CUDA_VISIBLE_DEVICES`，
  因而要求物理 ID `1`；PyTorch 随后才将唯一可见卡重映射为逻辑 `cuda:0`。
- 方案：该情形使用
  `CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 --device cuda:0`。
- 验证：修正后同一 Clean output 通过 resume 完成 2/2 episode、0 probe error；
  失败尝试没有 reset、action 或 diagnostic row。

### 3.17 已观察 pilot 不能反向决定 static 阈值

- 问题：20-step future pilot 已经暴露 success/OOD 标签；如果直接按这 7 条
  predicted energy 调阈值，会把 outcome 信息泄漏进指标定义。首版阈值 1.0
  又比实际 energy 高一个数量级以上。
- 方案：增加第三个互斥执行 namespace。每条独立 job 只执行标准 no-op，
  `policy.act()` 从不调用；同时测完全相同帧的重复编码噪声和 settle 后
  offset 0/4/8 的模拟器/render residual。raw sample、completion、帧和
  manifest 均原子落盘并支持跨 rank resume。
- 聚合门禁：Clean/OOD 只有 checkpoint、编码器语义、offset、no-op、实现 hash
  和 freeze 配方一致才可合并；候选取两个 null 分布 99% `higher` 值的较大者。
  自动要求 200 条、Clean/OOD 各 100、五类各 20、无异常、所有 source tree
  显式 clean，且运行时 control frequency/model-frame shape 一致；raw job
  manifest、calibration JSONL 和只读 diagnostic JSONL 均固定 SHA-256，
  并保留人工冻结步骤。
- 真实验证：2 Clean + 五类 OOD 共 7/7 eligible；同帧噪声全为 0，8-step
  no-op energy 最大 `0.013223`。只读敏感性把旧阈值的 7/7 predicted-static
  改为 candidate 下 0/7，源 diagnostics 字节不改写。
- 科学取舍：v1 样本量远不足，且采样前没有把 quantile 插值法写入 source
  manifest，因此状态强制为 `candidate_only`。修复后的协议 hash 与 v1 不同，
  dry-run 和真实运行都会要求新目录，避免静默续跑。

### 3.18 看过 outcome 后再挑案例会制造选择偏差

- 问题：阶段一 failure review 天然暴露失败、扰动、seed 和 termination；如果据此
  选阶段二视频，Clean/OOD 或 success/failure 的一致性差异会混入研究者选择。
- 方案：新增只读 cohort planner，在 source outcome JSONL 出现前只按
  job-manifest metadata、预注册 seed 和 SHA-256 selection key 固定 job ID。
  Clean 每 task 强制包含 episode index 0，保证与所有 index-0 OOD variant 有
  预先定义的 base reference；skipped-only cell 保持显式分母。
- 标注隔离：另将 condition/outcome/metric/action/source identity 放入私有
  unblinding key，公开 HTML/CSV 只包含 opaque case、任务文本和媒体。第一轮不问
  failure hypothesis，第二轮才解盲。
- 统计门禁：导入器要求每份文件对应唯一 reviewer 和完整 case set，逐字段输出
  missing、uncertain、decisive 分母及 exact agreement/pairwise Cohen's κ；全体
  同标签造成的退化边际明确写 `undefined`，不误报 κ=1。原始标签和所有派生文件
  都固定 SHA-256，且分析进程不接收 private key。
- 防误用：`--freeze` 要求 clean tree、source 无 outcome JSONL、0 supported
  shortfall；正式配置设 `require_frozen_cohort=true` 后，draft 会在模型加载和
  reset 前失败。
- 当前证据：真实 7-case packet 的 28 个媒体完成 hash/解码审计；v2 抽样草案为
  200 Clean + 532 OOD，但五类/四类取舍和 clean-tree freeze 尚未完成。

## 4. 简历表达素材

### 当前即可使用的版本（不包含虚构实验结果）

- 搭建 Fast-WAM 在 LIBERO/LIBERO-Plus 上的配置驱动 OOD 评测框架，以 adapter 隔离同名仿真 backend，并支持单卡调试与 episode-level 多 GPU 推理。
- 设计确定性 job manifest、哈希分片、逐 episode JSONL 落盘与断点续跑机制，保证大规模机器人 rollout 可复现、可审计、可恢复。
- 将 Clean 多 seed 与 LIBERO-Plus 预生成 task-instance 协议显式分离，通过配置门禁阻止每变体重复采样造成的数量级计算浪费。
- 建立相同 checkpoint/配对 seed 的鲁棒性评测与统计链路，覆盖成功率下降、bootstrap CI、失败分类和跨策略配方一致性约束。
- 为表征运动指标建立 outcome-independent no-op calibration、自动 freeze gate
  与只读历史敏感性分析，在真实 Clean/五类 OOD pilot 中识别并量化旧阈值的
  数量级错误。
- 实现 outcome-blind 分层抽样与 label-blind 双目录审阅协议，以 source/hash、
  pre-outcome freeze、opaque alias 和公开/私有泄漏校验阻止结果后选样；已完成
  7-case/28-media 真实工作流演练。

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
5. 防选择偏差：正式 cohort 在 outcome 前冻结，人工第一轮隐藏标签，failure
   hypothesis 留到解盲后。
6. 尚未解决但已诚实限定的问题：底层扰动参数、双相机证据、null difficulty、许可证和 future checkpoint 可识别性。
7. 最后用真实 pilot/full 数字回答效果、成本和失败模式；数字未产生前明确说“待实测”。

## 6. 更新规则

每次完成新阶段后更新本文：

- 把状态从“待验证”改为“已验证”时，必须附命令、日期和工件路径。
- 配置、上游 commit 或分类文件变化时，重新记录 manifest 数量和协议差异。
- 失败和回滚也要记录，不能只保留成功路径。
- 简历中的每个数字必须能追溯到 `experiment_manifest.json`、episode JSONL 或聚合报告。
- 不把 mock、dry-run、plan、pytest 或 doctor 成功写成真实策略成功率。
