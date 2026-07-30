# 工程亮点、难点与阻碍台账

本文是持续更新的工程复盘和简历素材库。只记录能由代码、测试或实验工件支持的事实；`plan`、mock 和单元测试不能写成真实机器人任务结果。

## 1. 当前事实快照

截至 2026-07-30：

| 项目 | 状态 | 可用证据 |
| --- | --- | --- |
| 配置、planner、adapter、分片、resume、聚合 | 已实现 | `src/`、`configs/`、`tests/` |
| 自动化测试 | 已通过 | `pytest -q`：386 passed、5 warnings；覆盖阶段一/二回归与 Thought3 配置、cache、Adapter、resume、泄漏、任意 positive-flow grid、E9 audit 与 K=1 online CF 编排 |
| Conda 环境与激活入口 | 已配置 | `scripts/create_env.sh`、`scripts/activate_env.sh` |
| checkpoint/stats | 已下载并人工校验 | checkpoint SHA-256 `1000437c...a49579`；stats SHA-256 `30f81ad7...68638` |
| FastWAM 公共运行时模型 | 已下载并逐文件校验 | `scripts/download_fastwam_runtime_models.sh`；T5、VAE、tokenizer 共约 11.9 GiB |
| LIBERO-Plus assets | 已下载并检查目录结构 | `articulated_objects/`、`new_objects/`、`scenes/`、`textures/` 等已就位 |
| 单卡 Clean smoke | 已通过 | 2026-07-22：2 episodes、2 success、0 exception；仅证明链路可用，不作为成功率估计 |
| 单卡 OOD smoke | 已通过 | 2026-07-22：camera/light 共 4 episodes、4 success、0 exception；仅证明链路可用，不作为成功率估计 |
| 三卡真实 pilot | 已通过 | 2026-07-22：9 planned、8 completed、1 expected skipped、0 exception；三个 rank 均有真实结果 |
| 阶段一正式 full | FORMAL 已完成 | 800 Clean + 6,771 OOD runnable 全部完成；68 expected skipped；0 exception、0 job 重复/遗漏 |
| 阶段一鲁棒性结论 | 可正式报告 | Clean 97.25%→OOD 47.70%，drop 49.55 pp；camera 15.13% 最敏感，light 81.88% 最稳健 |
| 阶段二 2A unconditional future | 正式收集与自动分析完成 | 200 Clean + 532 OOD；732 episodes、1,010 probes、2,020 aligned frames、0 error；4,040 media 全量解码 |
| 阶段二自动关联结论 | post-run analysis 完成 | OOD−Clean cosine `+0.0316 [0.0254,0.0381]`；direction `−0.1898 [−0.2134,−0.1664]`；统计 DRAFT 未预冻结 |
| 阶段二 static/no-op | FORMAL 完成 | 200/200 eligible；Clean/OOD 各 100、五类各 20；diagnostic 前接受阈值 `0.0167421166` |
| 阶段二标签盲审 | WORKFLOW PILOT 已通过 | 7 cases / 28 media；public/private hash 校验和全量解码通过；human labels 仍为 0/7 |
| 阶段二 outcome-blind 抽样 | 五类 exact-ratification 已执行 | 200 Clean + 532 OOD 全部运行；不能追溯称为阶段一 outcome 前预注册，但在 Phase 2 future 指标前锁定原 job ID |
| 阶段二 2B action-conditioned future | 严格阻塞 | release 配置为 `action_conditioned=false`，且不存在通过 provenance 门禁的匹配 checkpoint |
| 阶段三 Phase C 单样本门禁 | SMOKE 已通过 | 真实 K1/K2/K4、upstream parity、zero gate、1 backward、0 backbone grad；执行峰值 12.964 GiB |
| 阶段三 Phase D 真实 cache | SMOKE 已通过 | 32 samples、96 entries、12 shards；paired/checksum/resume/leakage 全通过；0.806 sample/s；0 optimizer step |
| 阶段三 Gate E.2 多样本诊断 | FAILED-GATE，工程轨迹完整 | A0/A1 × 三 LR 共 1,200 step；梯度/resume/checkpoint/frozen SHA/13.0 GiB 全通过；无共同 6/8 stable LR，保留负结果并定位单-flow confound |
| 阶段三 Gate E.3 held-out flow | FAILED-GATE，320/320 probe 完整 | 执行/provenance/zero-weight checks 全通过；E.2 A1 fixed-flow 的 24.19%/40.01% 降幅在 held-out flow 变为 0.025%/−1.31%，阻止不稳定配方进入 OOD |
| 阶段三 Gate E.4 diversified flow | FAILED-GATE，1,200 step 完整 | 200 unique paired slots、480 held-out objectives、108 execution checks 全通过；六条 reduction 转正但仅 0.997%–1.948%，继续阻止 full E/A2/A4 |
| 阶段三 Gate E.5 objective aggregation | FAILED-GATE，1,200 updates 完整 | 9,600 train objectives、480 held-out objectives、120 execution checks 全通过；A1@3e-4 为 19.668%/8-of-8，但同 LR A0 仅 2.638%，无共同 eligible LR |
| 阶段三 Gate E.6 fresh-cohort replication | FAILED-GATE，400 updates 完整 | 新 8-sample cohort、3,200 train objectives、160 held-out objectives；A1 降 14.842%/7-of-8 且相对 A0 final mean 低 13.815%，但 A0 仅 4/8 stable |
| 阶段三 Gate E.7 checkpoint trajectory | COMPLETED READ-ONLY DIAGNOSTIC | 8 个既有 checkpoint、800/800 objectives、0 backward/optimizer/write；工程 Gate 通过，primary 不支持实质晚期退化且无 joint candidate |
| 阶段三 Gate E.8 A0 flow variance | COMPLETED READ-ONLY DIAGNOSTIC | 1,536/1,536 forward、0 backward/optimizer/write；step-200 pooled loss 改善 3.728%，但 full/双 block 稳定性为 4/8、4/8、5/8；分类 `mixed_or_inconclusive` |
| 阶段三 Gate E.9 + Phase 0 audit | ENGINEERING VALID / SCIENTIFIC FAILED | 四轨 6,400 train + 2,048 held-out objectives；CPU-only audit 27/27 checks、父 77 文件 0 write；normalized paired `8.274%<10%`，E9b locked |
| 阶段三 Phase 1 K=1 online CF | VALID ENGINEERING SMOKE / BRANCH A | 单卡 8 sample；B0/null 精确 parity；correct-null、correct-shuffle 与 action hash 均 `8/8`；62/62 工件 SHA 通过；只支持 future-content action sensitivity |

## 2. 可以对外说明的工程亮点

| 亮点 | 设计与价值 | 实现证据 | 验证状态 |
| --- | --- | --- | --- |
| 不侵入上游的适配层 | 不修改 Fast-WAM、LIBERO、LIBERO-Plus；复用官方 checkpoint loader、观测/动作处理和 success 判定 | `policy/fastwam_adapter.py`、`envs/libero_adapter.py` | Clean 2-episode 与 Plus 4-episode smoke 均已验证 |
| 同名 backend 隔离 | 原版与 Plus 都导出 `libero`；为每个进程生成隔离的 `LIBERO_CONFIG_PATH` 并只加载一个 checkout，避免 import/path 污染 | `envs/libero_adapter.py`、`evaluator.py` | Clean 与 Plus 已分别在真实独立进程验证 |
| 可复现任务规划 | 每个 job 固化 suite、base/upstream task、seed、init index、扰动身份和策略身份；job ID 由规范化内容哈希生成 | `evaluation/jobs.py`、`job_manifest.jsonl` | 已单测 |
| episode-level 多 GPU | 每 GPU 一个独立 evaluator，按 job hash 稳定分片；避免把独立 rollout 错做模型 DDP | `distributed_launcher.py`、`shard_jobs()` | 三卡 full 完成 7,571 rollout；八个 source 均为 3 rank，0 重复遗漏 |
| 可恢复执行 | worker 逐 episode 追加并 `fsync` JSONL；默认跳过完成 job，支持 failed/all 重跑策略 | `evaluation/resume.py`、`schemas/episode_result.py` | 已单测；Clean smoke 用 `--rerun failed` 从两条真实 exception 恢复成功 |
| 科学比较门禁 | Clean/OOD 共用 seed 公式和 checkpoint；聚合时同一策略 checkpoint hash 不一致则拒绝比较 | `reproducibility.py`、`analysis/aggregate.py` | 已单测；Clean/OOD smoke 的 checkpoint SHA-256 已实测一致 |
| 上游协议显式化 | 区分 Clean 多 seed 与 Plus 每官方变体 1 次；`all_once` 强制 `episodes_per_task=1`，防止 10,030×20 的重复计算 | `config.py`、`jobs.py`、`eval_ood_full.yaml` | 已单测；正式 manifest 已重建并审计 |
| 可审计 OOD 元数据 | 记录官方 category、difficulty、classification ID、variant name、candidate/selection 信息和上游 commit | `jobs.py`、`episode_result.py` | 真实 Plus result 已验证，运行时底层数值参数采集仍有限 |
| 研究结论防越界 | 明确 release Fast-WAM、Joint WAM、IDM 是不同架构/权重；训练配方不匹配时禁止把比较写成未来想象的因果增益 | `config.py`、`thought1_generalization.md` | 配置门禁已单测，匹配权重缺失 |
| 失败分析闭环 | 记录 action/robot state trace、异常、失败视频和聚合统计，提供静态 failure review 页面 | `recording/`、`analysis/review.py` | full 已保存 7,571 traces 与 3,563 failure videos；0 缺失，人工 taxonomy 待标注 |
| 阶段二只读 shadow probe | 先冻结并哈希基线动作，再从同一 checkpoint 单独生成 future；current/predicted/actual/side-by-side 工件写入独立目录 | `policy/fastwam_future_probe.py`、`diagnostics/` | 732-episode 正式运行完成；1,010/1,010 probe 同次 action hash 不变，不会改写阶段一 source manifest/result |
| 诊断语义双门禁 | 将 release 可支持的 unconditional consistency（2A）与需要匹配 action-conditioned checkpoint 的动力学一致性（2B）分开，禁止静默降级 | `config.py`、`fastwam_future_probe.py`、`thought2_upstream_audit.md` | 2A 实测通过；2B 对 release 预期拒绝 |
| Source rerun 稳定性审计 | 将阶段二每个 probe 的 executed action 与阶段一 trace 按环境 step 对齐核对，并区分同次动作保护与跨运行非确定性 | `source_action_audit.csv`、阶段一 `traces/*.jsonl` | 正式 run：996/1,010 exact、13 mismatch、1 unavailable；730/732 outcome match；同次 1,010/1,010 action hash 不变 |
| 多输入语义安全聚合 | Clean/OOD comparison 单独生成 manifest，锁定 mode、共同 provenance、输入 fingerprint 与 source hash；unknown mode 不再猜测 | `diagnostics/aggregate.py`、`diagnostics/report.py` | 5-episode comparison 实测并可重复聚合 |
| 独立 null-motion 校准 | 不读取 diagnostic outcome、不调用 policy action；以同帧编码噪声和 0/4/8 no-op residual 建阈值，自动检查 200 条 freeze gate | `diagnostics/static_calibration*.py`、独立 YAML/manifest/JSONL | 正式 200/200 eligible、五类各 20；阈值 `0.0167421166` 在 diagnostics 前显式接受 |
| 标签盲化媒体审阅 | 将 condition/outcome/metric/source mapping 放入独立 `0600` private key；公开 packet 使用 opaque alias 和逐媒体 SHA-256；盲态导入区分 missing/uncertain/decisive 并计算 pairwise κ | `diagnostics/blind_review*.py`、静态 HTML/CSV/JSON | 7 个真实 probe 的 28 个媒体全部解码，public sensitive key/token 泄漏为 0；agreement 工具由合成双 reviewer 标签验证，真人标签仍为 0 |
| Outcome-blind 正式抽样 | 只从阶段一 job manifest 分层哈希选样，记录 skipped-only cell，并强制 Clean episode-0 anchor；formal runner 拒绝未冻结草案 | `diagnostics/diagnostic_cohort.py`、`require_frozen_cohort` | 五类 732-job exact-ratification 与运行完成；阶段一 outcome 已出现后不允许追溯升级 |
| 层级化正式分析与审计 | probe→episode→task 聚合、suite-stratified task bootstrap、首 probe sensitivity、outcome mismatch 排除、BH-FDR、媒体/trace 全量审计 | `diagnostics/formal_analysis.py`、`formal_analysis_v1/` | 40 task、10,000 bootstrap；4,040 media 0 decode error；175 tests passed |

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
  200 Clean + 532 OOD，但它在阶段一 outcome 前没有通过 clean-tree freeze。
  现在正式 outcome 已存在，草案不能追溯称为阶段一 outcome 前预注册；新增
  exact-ratification 保留原 draft hash/job ID，并把认证范围限制为 Phase 2
  future metric 前锁定。未 ratify 的 draft 仍会被正式门禁拒绝。

### 3.19 NVIDIA 驱动热更新会让 full runner 在模型加载前失败

- 现象：`nvidia-smi` 报
  `Failed to initialize NVML: Driver/library version mismatch`；最初的 full
  runner 又因 `set -euo pipefail` 在 command substitution 中提前退出，终端没有
  显示根因。
- 根因：内核仍加载 `580.159.03`，但磁盘上的 DKMS module、driver package 和
  NVML 已升级为 `580.173.02`。Conda 和项目代码均未参与这个冲突。
- 方案：把 GPU 查询改为显式捕获 `nvidia-smi -i` 的 stderr，失败时打印物理
  GPU ID 和原始错误；新 DKMS 已为当前 kernel 安装后，保存工作并重启加载
  `580.173.02`。
- 证据：修复 commit `575ba8f`；正式 combined manifest 的三卡 inventory 均为
  driver `580.173.02`，随后 full 全程 0 CUDA/NVML exception。

### 3.20 “7,639 条记录”不能误写成 7,639 个真实 rollout

- 问题：combined report 的 `episodes=7639` 包含 68 条零步 skipped 审计行；
  pairing 表的 6,771 个比较又只使用 40 个唯一 Clean episode-0 anchor。如果只
  抄顶层数字，会同时夸大真实计算量和配对独立样本数。
- 审计：逐 source 比较 job manifest、三个 rank raw JSONL 和 combined JSONL；
  7,639 个 ID 完全相等且唯一。真实 rollout 是 7,571，skipped 是 68。
- 深度校验：流式解析约 0.96 GiB traces，共 2,399,314 个 action step；
  验证 finite、shape、step count、非全零运动和末端位移，并核对 3,563 个
  `max_steps` 与 3,563 个非空失败视频一一对应。
- 统计处理：主表保留预定义的 variant/episode-weighted bootstrap；另做
  40-task 等权/cluster bootstrap。两者的 drop 分别为 49.55 pp 与 49.22 pp，
  方向一致，但 task-cluster CI 更宽 `[42.14, 56.39] pp`。

### 3.21 阶段二 failure 多一个 probe，不能把 clip 数直接当独立样本

- 问题：正式 runner 对成功 episode 在完成后停止，只产生首 probe；失败 episode
  运行到 `max_steps`，产生两个 probe。最终 198/2 个 Clean success/failure
  对应 202 probes，256/276 个 OOD success/failure 对应 808 probes。若直接按
  probe 平均，会把失败轨迹系统性加权更高。
- 方案：主统计先在 episode 内平均，再在 task 内平均，40 个 task 等权；outcome
  association 同时报告全部可用 probe 与仅首 probe。两个 Phase 1/2 outcome
  mismatch 从 outcome association 排除，但保留在 ID/OOD consistency。
- 结果：OOD failure−success cosine 从全部 probe 的
  `+0.0249 [0.0166,0.0328]` 变为首 probe 的
  `+0.0197 [0.0116,0.0282]`，方向不变；direction 对比也保持同方向。
- 边界：最低 cosine-error 四分位仍有 41.67% failure，最高四分位仍有
  34.59% success。Consistency 是关联信号，不是成败判定器；shadow future
  不在 control loop 内，更不能被写成失败原因。
- 统计资格：分析方法与运行前 DRAFT 一致，但 DRAFT 未冻结。机器报告显式写
  `formal_data_collection_post_run_protocol_consistent_not_preregistered`，
  防止把事后分析包装成预注册确认性结果。

### 3.22 不改 6B 上游，给动作分支建立可审计 future 输入

- 问题：基础 Fast-WAM 的 Action DiT 读取当前帧 Video DiT 表征，但不读取生成的
  future；直接修改上游或全量重训会破坏阶段隔离，也超出 3×4090 的现实预算。
- 方案：在项目侧给 `action_encoder` 输出注册受 context 管理的单次 hook，将
  `[B,48,2,14,28]` future 经过 Conv3d projector 和 gated cross-attention 注入
  `[B,32,1024]` action hidden。标量 gate 零初始化，使 A0 与 B0 初始 action
  逐元素相同；默认 trainable 参数精确为 1,371,137。
- 可恢复 cache：同一 base sample 的 K1/K2/K4 共享由 SHA-256 推导、且不含 K 的
  初始 noise；按 safetensors shard 原子提交，保存文件/tensor/逐样本 checksum。
  resume 只跳过完整且校验通过的 shard，单字节损坏会 fail-fast。
- 科学门禁：训练 batch 使用 allowlist，拒绝真实 future、next observation、
  success 和 Thought1/2/LIBERO-Plus 来源；A-shuffle 使用跨 task/episode 的
  确定性一一 derangement，不另训错误-future 模型。
- 恢复证据：Adapter-only checkpoint 绑定 backbone/stats/config/split/cache hash，
  保存 optimizer/global step/sample cursor；CPU mock 的中断+恢复与不中断训练最终
  Adapter semantic hash 相同。
- 边界：上述均为 Phase B `TEST`，没有加载官方 checkpoint 或 GPU，不能写成 OOD
  提升。真实 shape、loss parity、显存和成功率由 Phase C–G 逐级解锁。

### 3.23 用当前观测构建可恢复的真实 K-step latent cache

- 前置门禁：Phase C 在单张 4090 上加载官方 checkpoint，验证 video-only sampler
  与上游 joint video path 逐元素一致、zero gate action bitwise equal、一次真实
  backward 的 backbone gradient 为 0。
- 数据隔离：Phase D 只用标准 `libero_goal` task 0 的 training demonstration；
  42 个完整 episode 先做 37/5 split，再从 32 个不同 episode 各取一个当前帧。
  source loader 每样本只请求一个 timestamp 的两路相机和当前 proprio。
- 真实 cache：生成 32×K1/K2/K4 = 96 条 BF16
  `[48,2,14,28]` latent，按 12 个 safetensors shard 原子提交；全部文件、tensor、
  sample checksum 与 paired seed/initial-state hash 通过。
- 恢复与损坏：同配置的第二次 build 跳过 12/12 shard，且没有重新加载模型；
  `/tmp` 临时副本的单字节翻转被 checksum 拒绝，正式 cache 未变化。
- 泄漏审计：共解码 64 张当前相机帧、0 future RGB，未读 action target、
  actual future、success 或 termination；cache 明确
  `source_kind=model_sampled_from_current`。
- 资源：32-sample generation loop 39.70 s，吞吐 0.806 base sample/s（不含
  888.44 s 冷加载）；K1/K2/K4 sampling mean 为
  127.54/186.62/362.99 ms；执行/加载峰值为 12.677/23.125 GiB。
- 边界：这是 `SMOKE` 级 cache 工程证据。没有 optimizer、训练、robot rollout
  或成功率，不能声称未来提高 OOD，也不能把离线 sampling 当在线总延迟。

### 3.24 让 zero-gated 真实训练可诊断、可确定性恢复

- 问题：zero-init gate 的第 1 step 只允许 gate 获得非零梯度；只记录 Adapter
  总 grad norm 会掩盖 projector/attention 永远为零的断链。同时默认 CUDA
  backward 的极小 reduction 差异会使独立重放最终权重 hash 不同。
- 方案：对 gate、future projector、position/norm、Q/K/V/out attention 和全部
  non-gate 参数分别记录 finite/nonzero/L2；强制 cuBLAS workspace、PyTorch
  deterministic algorithms、math SDP，并关闭 TF32/Flash SDP。
- 证据：A0/A1 第 1 step non-gate nonzero 均为 0，第 2 step 分别达到
  1,321,983 / 1,371,102；两组 50→100 resume 与独立 0→100 的最终 Adapter
  semantic SHA 完全一致。
- 资源：1,371,137 个 trainable 参数，模型/训练峰值约 23.12/12.96 GiB，
  deterministic step mean 约 0.66 s。
- 负面门禁：A1 development 和 v3 A0 fixed train probe 未稳定改善，总 Gate
  保持 failed。该结果阻止了“梯度可传播”被错误升级成“训练有效”或“OOD
  有提升”，并将下一步收缩到单样本 fixed-noise overfit 诊断。

### 3.25 用 held-out action-flow 识别固定目标拟合

- 问题：8-sample Gate E.2 的 optimizer 与 probe 都把每条 sample 固定到同一个
  action noise/timestep；少数高-loss objective 的下降可能伪装成稳定训练收益。
- 方案：冻结六个 checkpoint，在从未进入 optimizer 的
  `flow_step=1..5` 上做 320 个 forward；先在 sample 内跨 flow 平均，再应用原
  `10% mean reduction + 6/8 non-worsened` 门槛。
- 工程纠错：官方 scheduler 在 BF16 `timestep=1000` 的 weight 合法为 0；v1
  非门控 ratio 遥测因零分母失效后，使用新 Run ID 显式记录 weight，并保留零
  objective 于主统计。v2 的 8/8 零权重 rows 全部 exact-zero loss。
- 证据：六条 probe 的 finite、pairing、hidden-scale、catastrophic、memory 和
  frozen SHA 检查全通过，但 A0 最好仅下降 1.35%/2-of-8，A1 最好仅
  0.025%/2-of-8；三个 LR 均未通过。
- 决策：不把 fixed-flow 训练降幅写成 future 收益；冻结下一单变量诊断为每次
  optimizer visit 使用唯一、A0/A1 配对且与 held-out probe 不重叠的 flow slot，
  不放宽门槛、不扩 A2/A4、不启动 OOD。

### 3.26 用 paired diversified-flow 训练验证混淆是否可修复

- 方案：保持八条样本、A0/A1、三档 LR、Adapter、200-step budget、official
  loss、held-out probe 和全部门槛不变，只把 optimizer objective 改为
  `10001..10200` 的 200 个唯一 flow slots。
- 审计：六条轨迹共 1,200 optimizer steps、480 held-out objectives、24 个
  checkpoint 目录；108 个 per-track execution checks、全部 paired/cross checks、
  zero-weight step `[49,142]` 和 frozen Fast-WAM SHA 均通过。
- 结果：E.3 中的部分负 held-out reduction 在 E.4 六条轨迹上都转为正值，但仅
  `0.997%–1.948%`；A1@3e-4 虽有 7/8 sample 不变差，mean reduction 仍只有
  1.787%，故严格保留 failed gate。
- 根因线索：post-run 遥测显示 positive per-step loss `max/min` 达
  `2268×–2342×`，top 10% step 占约 59%，scalar gate gradient 高频换号且净值
  远小于绝对值；下一步收缩到 objective aggregation/effective-batch 单变量，而
  不是放宽阈值或直接扩 K。

### 3.27 把多目标梯度聚合做成可审计单变量诊断

- 决策：在结果前选择 matched-optimizer-update，而非事后在 matched-objective
  与 matched-update 中挑口径；E.4/E.5 都做 200 次 AdamW update，但 E.5 每次
  固定遍历完整 8-sample cohort 并对 `loss/8` 累积梯度。
- 配对：为 200×8 个 objective 冻结 `20001..21600` 唯一 slot，锁定完整
  sample/noise/timestep identity SHA；A0/A1 和三个 LR 共享同一 schedule，且与
  probe/E.4 slots 不相交。
- 遥测：分别落盘 objective 与 optimizer-update JSONL；记录每个 micro loss、
  mean-scaled gate-gradient contribution、累计 gradient、符号及 update-level
  cancellation ratio，可从工件独立重算 mean/sum。
- 恢复：checkpoint 的 cursor 强制为 `update×8`，先原子提交两份 metrics 再保存
  Adapter-only checkpoint；恢复时截断 checkpoint 后的孤立 metrics，不允许
  partial cohort、重复或错序。
- 执行：六条真实轨迹完成 1,200 optimizer updates、9,600 train objectives、
  480 held-out objectives；120/120 execution、21/21 paired 和 7/7 cross
  checks 全部通过，六条 track 工件的 30 个 root-recorded SHA 全量复算匹配。
- 结果：A1 在 `1e-4/3e-4/1e-3` 下分别下降
  `6.452%/19.668%/7.315%`，其中 `A1@3e-4` 为 8/8 non-worsened；对应 A0
  只有 `1.889%/2.638%/2.890%`，因此预注册的共同 LR Gate 有效失败。
- 资源：单卡总耗时 114.65 分钟，model-load/train peak
  `23,679.513/13,277.440 MiB`；每个 8-objective update 平均 4.961 秒。
- 决策：不把强 A1 配对差包装成 future 效果，也不事后选择 3e-4；下一步先在
  未使用 train cohort 上做披露 post-selection 的 A0/A1 序贯复验。

### 3.28 把结果后候选复验做成显式可审计协议

- 污染披露：E.6 工件强制记录 `3e-4` 来自查看 E.5 后选择、不是 E.5
  selected LR，并标记非独立确认性实验。
- 新数据窗口：从冻结 Phase D train inventory 的确定性排序中取第 9–16 条，
  与 E.5 cohort/development 零交集且八条来自不同 episode；cohort 身份单独
  SHA 固化。
- 新随机目标：使用 `31001..32600` 共 1,600 个新 flow slots，预先计算完整
  identity schedule SHA 和 19 个合法 zero-weight endpoints，并与 E.4/E.5
  namespace 做硬不相交检查。
- 三层门槛：分别检查 A1 绝对复现、A0 null-future 稳定性以及 sample-paired
  A1-vs-A0 优势，防止只展示最有利的一条轨迹。
- 执行结果：单卡 43.89 分钟完成 400 updates/3,200 objectives；所有工程检查
  通过。A1 absolute 与 paired superiority 复现，但 A0 只有 4/8 不变差，冻结
  Gate 有效失败；结果只支持工程可重复性，不支持 future/OOD 效果。

### 3.29 把中间 checkpoint 诊断与结果后选择隔离

- 污染边界：E.6 step-200 和旧 held-out flow `1..5` 已知，step
  `50/100/150` outcome 未查看；因此旧 flow 只允许做 continuity reproduction。
- 新主 panel：冻结 flow `6..10` 的 40-objective 网格，完整 RNG identity SHA
  为 `3361f170...2f68`，与训练 namespace 和旧 probe 都不重叠。
- 只读证明：启动前后重算 E.6 root 与八个 checkpoint 的三文件 SHA；运行代码
  位于 `torch.inference_mode()`，不创建 optimizer、不调用 backward、不写
  checkpoint，并检查 Adapter/Fast-WAM grad 为空。
- 判定防漂移：预先冻结“最早稳定 checkpoint”比较、至少下降 2 条 sample 与
  mean loss 增加的 late-degradation 条件，以及三个明确的 not-supported
  分支；不允许结果后挑“最好”的早期 step。
- 候选边界：A0/A1/paired 三门同时满足时只登记
  `post_run_diagnostic_candidate_only`，仍需未使用 cohort 复验。
- 执行结果：单卡 13.98 分钟完成 800/800 objectives；所有
  frozen/RNG/provenance/leakage 检查通过，0 backward/optimizer/write。
  Primary 上 A0 50/100 pass、150/200 fail，但 endpoint mean 比 step 50
  低 5.651%，故不支持实质晚期退化；A1 信号随 step 增强，但无 joint
  candidate。Continuity 的不同 materiality 分类被保留为描述性证据，未覆盖
  primary。

### 3.30 区分真实逐样本尾部风险与小 flow panel 方差

- Result-conditioned disclosure：冻结 E.7 已知的三条 step-200 worsened
  sample 为 target，不允许 E.8 运行后改挑新的最差样本。
- 新 RNG namespace：使用 flow `11..74`，与 fixed/held-out `0..10` 及
  E.4/E.5/E.6 的 `10001+ / 20001+ / 31001+` train slots 硬不相交；512 个
  sample-flow identity 预先 SHA 固化。
- 内部 replication：64 flows 预拆为两个不重叠 32-flow block，每个 block
  都比 E.7 panel 大 6.4 倍；full/两个 block 分别执行原 A0 Gate。
- 统计门禁：20,000 次 paired-flow bootstrap 对 16 个 sample×checkpoint
  contrast 做 family-wise 校正；另用 20,000 次 five-of-64 无放回重采样量化
  五-flow Gate 自身的波动率。
- Fail-closed 分类：至少 2/3 target 跨 full、双 block 与校正 CI 确认才支持
  tail risk；只有 8 条均无确认恶化且三个 panel 都过 Gate 才支持 five-flow
  variance；其余统一 mixed/inconclusive。
- 执行结果：单卡 18.51 分钟完成 1,536/1,536 objectives，全部
  frozen/RNG/provenance/leakage checks 通过，0 backward/optimizer/write。
  step-200 pooled mean 改善 3.728%，但 full/双 block 只有 4/8、4/8、5/8
  sample 不变差；一条 target 与一条非 target 经 FWER 确认恶化，冻结分类为
  `mixed_or_inconclusive`。
- 工程解释：20,000 个 five-of-64 panel 中 86.36% 会因 sample stability
  失败，而只有 3.555% pooled mean 变差；这量化了小 panel 波动，同时保留
  `episode_000012` 跨两个 block 的稳定 +8.881% harm，没有把它抹成纯噪声。

### 3.31 把 sample-tail mitigation 隔离成单一优化变量

- 避免新混淆：normalized A0/A1 之外同时保留 same-flow raw A0/A1，使新
  action-flow draw 不会与 weighting 同时变化。
- 固定尺度：只用 E.8 zero-gate initial loss 计算 inverse-loss weights，
  权重和保持 8；不读取 final loss/harm、不 clipping、不按 E.9 结果再调。
- 训练 provenance：扩展 full-cohort trainer，使 arithmetic mean 的旧协议与
  工件保持兼容，同时为 normalized recipe 记录 weight SHA、逐 objective
  sample weight、weighted backward loss 和 resume manifest。
- 分层门禁：保留原 A0/A1/paired thresholds，再增加四轨 32-comparison
  FWER confirmed-harm；稳定候选与“真正减少 raw tail”分成两种分类。
- 一次性 holdout：在不解码的情况下冻结 train 排序 17–28 的 12 条完整身份、
  cohort SHA、flows `107..138` 和 zero-weight 位置；v2 没有 candidate，
  所以该 cohort 保持未消费且 E.9b 不解锁。
- 失败隔离：E.9a-v1 暴露共用 evaluator 的 `1..5` 硬编码，在 raw/A0
  initial probe 前以 0 objective/0 update 停止；五个工件 SHA 与 frozen
  backbone 不变已归档，v1 runner 永久拒绝 resume。
- 修复验证：v2 将 aggregator/evaluator/outcome 改成协议显式正整数 flow 集，
  对完整 `75..106` 的 256-objective grid 做回归；invocation-scoped wrapper
  令 initial-probe/setup 异常原子落盘为 `failed`，不再残留假 `running`。
- 真实执行：单卡 88.60 分钟完成四轨 800 updates、6,400 train objectives、
  2,048 held-out objectives 和 16 个 checkpoint。raw A0/A1 reduction 为
  4.175%/12.994%，normalized 为 2.983%/11.010%；A0 confirmed harm
  `2→0`，但 normalized paired gain 只有 8.274%，未达 10%。
- 二次工程卡点：probe writer 没有保存 checker 强制要求的三个 RNG identity
  字段，使四轨唯一共同 execution check false。根 Gate 正确 fail closed，
  77 个工件和 SHA 冻结后，由全新 CPU-only audit 恢复 256 个唯一 identity；
  27/27 hard checks 通过、父目录 0 write。
- 审计结论：E9a-v2 可登记为 engineering valid，但 normalized paired gain
  仍为 8.274%<10%，所以科学 Gate failed、E9b locked。
- 结论边界：只能写 tail/mean trade-off、只读 provenance 恢复与科学负 Gate；
  不能写 mitigation 成功、E.9b candidate、future 因果增益或 OOD 效果。

### 3.32 把研究主线从 surrogate Gate 拉回在线动作变量

- 唯一 checkpoint：冻结 E6 A1@3e-4 step-200 的 file/state/config SHA，禁止
  试多个 checkpoint 后挑最大动作差。
- 严格 current-only：HF Dataset 先 `select_columns`，只保留 identity、
  timestamp 和当前 proprio；双相机只解码当前帧，不访问 action target、future
  RGB、training cache、reserve/dev/OOD/success。
- 可归因 null：实现 request-scoped parameter-free bypass；不构造零 tensor、
  不运行 Video DiT、Adapter forward 为 0，并要求与官方 B0 `L∞<=1e-5`。
- 内容反事实：correct 与 shuffle 使用同一 recipient future-noise seed；
  shuffle 只把 other-episode online K=1 latent 注入 target 的 current
  cache/Action DiT，target action noise 和 context 不变。
- 先验 replay floor：每条 B0 重复两次，intervention 前冻结
  `max(1e-7,10×p95 replay L2)`；B0 非确定性时 fail closed，不输出 A/B/C。
- 工件闭环：action/future safetensors、semantic/file SHA、逐 sample JSONL、
  synchronized latency/memory、protocol lock、atomic prefix resume 和自动
  A/B/C 决策报告均由独立真实 runner 生成。
- 研究停止规则：A 才进 28/4 A0/A1；B 最多一次单变量结构修复；C 停止
  Adapter-only。本次真实结果为 A；下一步只允许预注册 28/4 A0/A1，不再新增
  flow/LR/checkpoint/sample-weight Gate。
- 真实证据：B0 replay 与 formal-null 在 8/8 sample 上逐位一致；
  correct-null、correct-shuffle 与 action-hash 均 8/8 过冻结门槛；paired
  correct-null overhead mean `258.95 ms`，62/62 工件 SHA 与 frozen/no-grad/
  no-cache/no-RGB checks 通过。
- 结论边界：这是单 task 八条 train sample 的 engineering action-sensitivity
  smoke；action L2 mean 仅 `0.011052/0.012092`，不能写成 OOD 成功率或控制增益。

### 3.33 把完整 28/4 训练做成双卡单变量、可恢复协议

- 问题：Phase 1 分支 A 后若继续试 LR、sample weight 或 checkpoint，会再次陷入
  surrogate 选择；若 A0/A1 在两卡各自重算权重，又会破坏单变量配对。
- 方案：冻结唯一 normalized recipe，由单卡先在 28 条 train sample、32 个新
  flow 上生成 inverse-initial-loss unit-mean 权重和 SHA；随后 A0/A1 分别占用
  两张卡并行，共享同一 200×28 sample/flow schedule。
- 恢复：每 50 update 保存 Adapter-only safetensors、optimizer 和 manifest，
  checkpoint 同时绑定 calibration、weight、metric-prefix 与 flow-schedule
  SHA；中断后另一条已完成 track 只读校验并跳过。
- 防选择：development 只评 step 0/200，主 checkpoint 固定 step 200、无
  fallback；CPU finalize 永远写 `phase3_unlocked=false`，完整 A1 checkpoint
  的 online correct/null/shuffle recheck 才能继续。
- 验证：config/schema/CLI/runner、两卡 launcher、四 stage no-torch dry-run 和
  58 项定向测试已完成；真实 GPU 状态仍是 `NOT RUN`，不能写成 loss 或 OOD
  结果。

## 4. 简历表达素材

### 当前即可使用的版本

- 搭建 Fast-WAM 在 LIBERO/LIBERO-Plus 上的配置驱动 OOD 评测框架，以 adapter 隔离同名仿真 backend，并支持单卡调试与 episode-level 多 GPU 推理。
- 设计确定性 job manifest、哈希分片、逐 episode JSONL 落盘与断点续跑机制，保证大规模机器人 rollout 可复现、可审计、可恢复。
- 设计并执行 raw/normalized × A0/A1 的 matched Adapter 诊断，完成 6,400
  training 与 2,048 held-out objectives；通过 fail-closed 审计定位 probe
  writer/checker RNG provenance contract 缺陷，并以 CPU-only 只读审计恢复
  256 个唯一 RNG identity、验证父 77 文件 0 改写，同时保留科学负结果。
- 为 K=1 online future-to-action 技术反事实实现官方 B0 replay、无 tensor
  formal null 与 other-episode shuffle，冻结 checkpoint/cohort/noise，
  原子保存 action/future tensor与分段 CUDA telemetry；真实单卡 8-sample
  运行中 B0/null 精确一致、correct-null/correct-shuffle/action-hash 均
  `8/8`，验证 future 内容进入动作，同时明确不宣称 rollout/OOD 效果。
- 将 Clean 多 seed 与 LIBERO-Plus 预生成 task-instance 协议显式分离，通过配置门禁阻止每变体重复采样造成的数量级计算浪费。
- 建立相同 checkpoint/配对 seed 的鲁棒性评测与统计链路，覆盖成功率下降、bootstrap CI、失败分类和跨策略配方一致性约束。
- 在 3 张 GPU 上完成 7,571 个真实 rollout 和 2,399,314 个 action step，
  实现 0 exception、0 job 重复/遗漏；Fast-WAM 从 Clean 97.25% 降至
  LIBERO-Plus OOD 47.70%，定位 camera viewpoint 为最敏感 shift（15.13%）。
- 对 7,571 条 trace 与 3,563 个失败视频做完整性审计，排除 NaN、空动作、
  静止机器人和落盘故障；任务级 OOD 成功率跨度 4.57%–92.63%。
- 为表征运动指标建立 outcome-independent no-op calibration、自动 freeze gate
  与只读历史敏感性分析；正式完成 200-job、Clean/OOD 平衡且五类各 20 的
  null 校准，在 diagnostic 前冻结阈值 `0.0167421166`。
- 实现 outcome-blind 分层抽样与 label-blind 双目录审阅协议，以 source/hash、
  pre-outcome freeze、opaque alias 和公开/私有泄漏校验阻止结果后选样；已完成
  7-case/28-media 真实工作流演练。
- 搭建 Fast-WAM/LIBERO-Plus 多 GPU future-consistency 评测与层级统计管线，
  完成 732 episodes、1,010 probes、2,020 aligned frames 和 4,040 媒体全量
  审计；发现 OOD 下 cosine consistency distance 增加
  `0.0316 [0.0254,0.0381]`、视觉方向一致性下降
  `0.1898 [0.1664,0.2134]`。
- 设计 1.37M 参数 zero-gated Future-to-Action Adapter，并实现 K=1/2/4
  paired latent cache、原子 shard/resume/多级 checksum、Adapter-only
  deterministic resume、跨 task/episode 反事实置换和真实未来泄漏门禁；
  已在真实 Fast-WAM 上缓存 32×3 条 latent，12/12 shard 通过恢复与损坏审计，
  不宣称训练收敛或 OOD 效果。
- 在单张 RTX 4090 上从标准 LIBERO 当前观测生成 96 条真实 K-step future
  latent，以 12 个原子 shard 落盘；实现 12/12 no-op resume 无模型重载、
  单字节损坏检测和 0 future-RGB source audit，执行峰值 12.677 GiB。
- 为真实 zero-gated Adapter 训练实现按模块梯度遥测与确定性 CUDA 重放；
  验证 gate 更新后的第 2 step 有百万级 non-gate 元素获得梯度，A0/A1 的
  checkpoint-resume 与 uninterrupted 最终权重 SHA 完全一致，并用 loss
  fail-closed 门禁阻止未收敛配方进入 OOD 评测。
- 设计并预注册单样本 fixed-noise 训练诊断，在冻结 5B Fast-WAM 的前提下只训练
  1.371M Adapter 参数；A0/A1 的固定真实 action loss 经 200 step 分别下降
  92.93%/99.58%，backbone 全参数 SHA 训练前后逐位一致，单卡峰值
  13.0 GiB。
- 将 fp32 residual、scalar gate 与实际 BF16 hidden delta 分层遥测，发现单样本
  overfit 时 A0/A1 correction 达原 hidden norm 的 1.91×/0.70×；据此拒绝直接
  扩 K，把下一门禁收缩为 8-sample train-only 的 loss/尺度联合诊断。
- 为 Adapter 训练建立 320-objective held-out action-flow 门禁，识别出
  fixed-flow train loss 改善没有跨 noise/timestep 泛化；在全部执行与冻结检查
  通过的情况下仍保留负结果，并用该证据阻止不稳定 checkpoint 进入 OOD rollout。
- 设计 1,200-step paired diversified-flow 训练诊断，以共享 schedule SHA、
  zero-weight 审计、Adapter-only checkpoint 和 108 项 execution gate 隔离单一
  优化变量；发现跨 flow 退化虽被缓解但绝对降幅仅 0.997%–1.948%，据此拒绝将
  小幅工程改善包装成 future/OOD 收益。
- 将每次 Adapter 更新扩展为完整 8-sample/8-objective 算术均值，并建立
  micro-contribution、梯度抵消率和双 JSONL 原子恢复审计；在单卡完成 9,600
  个真实 train objective，识别出 A1@3e-4 的 19.668%/8-of-8 held-out 候选
  信号，同时因 A0 未过预注册共同门而保持 fail-closed、未启动 OOD。
- 建立只读 checkpoint-trajectory 诊断，将已知 endpoint 的 continuity panel
  与新 primary flow 隔离；单卡完成 8 个 checkpoint、800 个 forward objective，
  以 0 backward/optimizer/write 和全量 SHA 证明未改模型，并识别出 pooled
  mean 改善与逐样本稳定性下降之间的 flow-sensitive trade-off。
- 将 A0 stability 诊断扩展到 64 个全新 action-flow、两个独立 32-flow block
  和 20,000 次 FWER paired bootstrap；单卡完成 1,536 个只读 objective，
  分离出“pooled loss 改善 3.728%”与“两条 sample 确认恶化”的混合结构，
  并以 fail-closed 分类阻止事后选择 checkpoint 或降低门槛。
- 为结果后 sample-tail 工程问题预注册四轨 matched 对照，将 inverse-initial-loss
  weighting 与新 flow draw 解耦，并用权重/调度 SHA、逐 objective telemetry、
  32-comparison FWER 和一次性未使用 cohort 边界防止配方与验证集反复选择；
  当前只可表述为协议与工程实现，不可宣称 mitigation 已通过。
- 将 Phase 1 正向动作敏感性转化为可审计的 28/4 A0/A1 训练协议：冻结单一
  normalized weight artifact、200×28 matched flow schedule、step-200 endpoint
  和双卡 track isolation，并实现 Adapter/optimizer/metric-prefix checksum
  resume。真实 Phase 2 尚未运行，因此该条只可作为系统设计与工程实现经历。

### 可直接使用的量化表述

> 在 3 张 GPU 上完成 Fast-WAM 的 7,571 个 Clean/OOD 机器人 rollout，设计
> episode-level 哈希分片、逐条持久化与 incomplete-only resume，实现
> 0 exception、0 重复/遗漏；测得标准 LIBERO 97.25% 到 LIBERO-Plus OOD
> 47.70% 的 49.55 个百分点下降，并定位 camera viewpoint 为最敏感扰动
> （15.13% success rate）。

> 在冻结控制动作的 Fast-WAM shadow diagnostic 中完成 732 个
> Clean/LIBERO-Plus episode 与 1,010 次 future probe，实施
> probe→episode→task 层级聚合、10,000 次 task-cluster bootstrap、首 probe
> 敏感性和 4,040 媒体全量审计；测得 OOD future consistency distance 增加
> 0.0316（95% CI 0.0254–0.0381），并明确限定为非因果关联。

> 为 1.37M 参数 Future-to-Action Adapter 建立真实离线 future-latent 管线：
> 在单张 RTX 4090 上从 32 个标准 LIBERO episode 生成 K=1/2/4 共 96 条 BF16
> latent，以 12 个 safetensors shard 原子提交，并验证 paired noise、12/12
> 无重载断点恢复、单字节损坏检测和 0 future-RGB 泄漏。

> 在冻结 5B Fast-WAM 的单卡真实训练诊断中，只优化 1.371M
> Future-to-Action Adapter 参数；通过 zero-gate 两步梯度门禁、逐模块
> gradient-to-parameter telemetry 和全参数 pre/post SHA，验证 A0/A1 在固定
> LIBERO 目标上的 loss 分别下降 92.93%/99.58%，并识别出 hidden correction
> 尺度膨胀这一多样本训练前的关键风险。

> 为冻结 Fast-WAM checkpoint 预注册并执行 K=1 online future-to-action
> 技术反事实：在 8 条固定 LIBERO train sample 上验证 B0 重放与 parameter-free
> null 逐位一致；other-episode future shuffle 使 8/8 action tensor hash 改变，
> correct-null/correct-shuffle 的 normalized action RMS 差异均超过冻结 floor；
> K=1 paired 在线开销均值为 258.95 ms，并完成 62/62 tensor/provenance SHA
> 审计。该表述只代表动作内容敏感性，不代表 OOD 成功率提升。

在人工标注完成前，不要追加“归纳出 K 类失败模式”。可以写“保存并审计
3,563 个失败视频”，但“reviewed/labelled”分母目前仍为 0。

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
5. 防选择偏差：原 v2 cohort 没有在阶段一 outcome 前完成 clean-tree freeze，
   因而没有追溯包装成预注册；正式 runner 只在 Phase 2 future 指标前 exact-ratify
   原 job ID。raw collection 正式完成，但统计 DRAFT 未预冻结，分析如实标为
   post-run。
6. 尚未解决但已诚实限定的问题：底层扰动参数、双相机证据、null difficulty、许可证和 future checkpoint 可识别性。
7. 用正式数字回答效果与成本；失败机制仍明确写“待分层人工复核”，不把
   `max_steps` 自动等同为某种感知或规划错误。

## 6. 更新规则

每次完成新阶段后更新本文：

- 把状态从“待验证”改为“已验证”时，必须附命令、日期和工件路径。
- 配置、上游 commit 或分类文件变化时，重新记录 manifest 数量和协议差异。
- 失败和回滚也要记录，不能只保留成功路径。
- 简历中的每个数字必须能追溯到 `experiment_manifest.json`、episode JSONL 或聚合报告。
- 不把 mock、dry-run、plan、pytest 或 doctor 成功写成真实策略成功率。
