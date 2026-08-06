# Fast-WAM 项目经历｜详细版

## 一、项目基本信息

**项目名称：** 机器人世界动作模型的 OOD 泛化与 Future Utility 机制证伪<br>
**项目角色：** 独立研究与工程实现<br>
**项目时间：** 2026.07–2026.08<br>
**技术栈：** Python、PyTorch、CUDA/EGL、MuJoCo、Fast-WAM、LIBERO/LIBERO-Plus、torchrun、pytest、YAML、safetensors、Bootstrap

**项目简介：** 围绕“Fast-WAM 在标准环境表现接近饱和时，面对相机、光照、背景、物体布局和机器人初态变化是否仍然可靠，以及显式 future 是否真正改善动作”这一问题，搭建 Failure→Representation→Sensitivity→Geometry→Intervention→Failure analysis 的六层研究系统。原假设是 Camera Equivariance Gap 可通过 Geo-REPA 和 Pose/Ray 修复，并进一步恢复 future utility 与 Camera success；实验不支持这条完整机制链。项目因而冻结负结果、停止追求正结果的调参，并用只读失败分解将下一问题从“几何不够好”收敛为“future utility 具有 condition/noise-stage dependence”，形成完整的 hypothesis–intervention–falsification 闭环。

## 二、可直接放入详细简历的版本

**机器人世界动作模型的 OOD 泛化与 Future Utility 机制证伪**｜独立研究与工程实现｜2026.07–08

- 审计 Fast-WAM 官方推理链路与训练配置，将模糊的“未来想象能否改善泛化”拆成行为鲁棒性、观察关联、动作技术依赖、任务效用、缺口定位与针对性干预六层问题，避免把不同 checkpoint 对比或离线相关性包装成因果收益。
- 从 0 到 1 实现配置驱动的 Clean/OOD 评测框架：以进程级 adapter 隔离原版 LIBERO 与 LIBERO-Plus 的同名 Python 包，复用官方 checkpoint loader、观测/动作预后处理和成功判定；通过受信路径校验兼容 PyTorch 2.6+ 的旧 init-state 加载。
- 设计内容哈希 job ID、配对 seed、episode 级多 GPU 稳定分片、append+fsync JSONL、incomplete-only resume 和聚合完整性门禁；在 3 张 GPU 上完成 `7,571` 次真实 rollout、`2,399,314` 个 action step 和 `3,563` 个失败视频，达到 `0 exception`、`0` job 重复/遗漏。
- 在 4 个 LIBERO suite、40 个基础任务和 5 类官方环境扰动上完成 `800` 次 Clean 与 `6,771` 次 runnable OOD 评测；同一 checkpoint 成功率由 `97.25%` 降至 `47.70%`，绝对下降 `49.55` 个百分点，并定位相机视角为跨 suite 最敏感因素（`15.13%`），光照最稳健（`81.88%`）。
- 构建动作隔离的 shadow-future 诊断链路，先哈希保护即将执行的动作，再旁路生成 unconditional future；在 `732` 个 episode 上完成 `1,010` 次 probe、`2,020` 个对齐 future frame 和 `4,040` 个媒体工件全量审计，并实施 probe→episode→task 层级聚合及 `10,000` 次 suite-stratified task bootstrap。
- 发现 OOD 相对 Clean 的 future consistency distance 增加 `0.0316`（95% CI `[0.0254, 0.0381]`），视觉运动方向一致性下降 `0.1898`（95% CI `[0.1664, 0.2134]`）；通过首 probe 敏感性分析控制失败轨迹 probe 数更多的混杂，并明确将结果限定为 shadow observer 的非因果关联。
- 在不改动 Fast-WAM backbone 的前提下设计 `1,371,137` 参数 zero-gated Future-to-Action Adapter，完成 K=1/2/4 paired latent cache、原子 shard、断点恢复、多级 checksum、Adapter-only checkpoint 和真实未来信息泄漏门禁；`32×3=96` 条真实 latent 的 `12/12` shard 通过恢复与单字节损坏审计。
- 预注册 K=1 online correct/null/shuffle 技术反事实，以 B0 重放确定 replay floor，以 parameter-free null 验证基础动作逐位一致，以 other-episode shuffle 只替换 future 内容；`8/8` 固定样本的 correct-null、correct-shuffle 和 action hash 均超过冻结门槛，证明 future 内容会改变动作，但不将其误写成成功率或任务收益。
- 执行双 GPU、K=0/K=1 matched Adapter 训练，共完成 `11,200` 个训练 objective、8 个可恢复 checkpoint 和 12/12 hard checks；固定 step-200 上 K=0 held-out loss 改善 `1.845%`，K=1 恶化 `1.712%`，K=1 最终 loss 比 K=0 高 `3.624%` 且 `4/4` development sample 更差，遂按预注册规则停止 Phase 3/OOD rollout，避免 outcome-driven 调参。
- 设计冻结 geometry–action diagnosis：在 `64` 个 base state 上构造 `256` 个 exact-state/探索配对样本并捕获 `12,544` 条特征；测得 Video Camera−Clean geometry gap `+0.020273 m`，高于 Lighting `+0.011660 m`，且 rank-3 subspace shuffle 在 `36/36` 次干预中改变动作而 correct control 逐位恢复，将下一假设定位为 `camera_equivariance_gap`，不越界声称新方法或成功率收益。
- 基于 Thought4 诊断实现 `1,335,320` 参数 Fast-WAM-GeoEq：在 action-consumed layer-15 K/V projection 安装 rank-8 LoRA，加入 RayPose/relative-pose residual 与训练期 Geo-REPA，并以 shuffled-target G4 作为 specificity control；Action DiT 冻结、GeoProjector 推理移除、GT depth 不进入策略。
- 在 `3×RTX 4090` 上完成单 task 8-train/4-development/4-test 的 B1/G3/G4 matched Pilot（约 `2h29m`）：G3 Camera gap 缩小 `20.94%<25%`，主 future geometry RMSE 未改善，future utility 为 `−0.005231`，Camera success 与 B1 同为 `1/4`；G4 gap 反而缩小 `25.82%`，正确 Geo-REPA correspondence 的特异贡献未被识别。
- 五项 Gate 全 false 后冻结当前 recipe，停止预计 `28–45 h` formal，不通过更多 LR/checkpoint/outcome 追求正结果；对 25 项冻结输入做 CPU-only 只读分解，发现 G3 Clean utility `−0.015268`、Camera 均值 `+0.004807`，且低 sigma bucket 效用低至 `−0.049520`，将下一假设收窄为 condition/noise-stage dependence。
- 验证 RayPose gate/grad/injection 非零，但因缺少 gate-zero/G1/G2 对照而拒绝独立因果归因；最终得到“Camera gap 存在，但当前 Geo-REPA + Pose/Ray 不能恢复 aggregate future utility/success”的完整证伪闭环，而非一个被调参掩盖的负性能结果。
- 建立覆盖配置、分片、resume、RNG 隔离、cache、Adapter 冻结、反事实、FP32 reconstruction、三卡 worker 和旧 CLI 回归的测试体系；当前全量测试为 `504 passed`、5 个仅与不可用 NVML 相关的环境 warning。

## 三、心路历程与决策考量

### 1. 先问“结论能不能被识别”，再问“模型能不能涨点”

项目最初最直接的路线，是给 Fast-WAM 接入 future prediction，然后比较 OOD 成功率。但上游审计发现，官方 `libero_uncond` release 的基础动作路径只读取当前帧表征，并不会读取旁路生成的 future；同时也没有 provenance 匹配的 action-conditioned checkpoint。此时若拿不同架构或不同权重直接比较，即使成功率有差异，也无法把差异归因于“测试时是否使用 future”。

因此第一项决策不是改模型，而是收紧问题：冻结 checkpoint、动作接口和基础任务，只改变环境条件，先建立可信的 OOD 行为缺口。

### 2. 先建立行为事实，再定位相关信号

阶段一用同一 checkpoint、统一 seed 规则和官方环境变体完成 Clean/OOD 评测，确认模型并非“任务不会做”——Clean 已达 `97.25%`，但 OOD 仅 `47.70%`。这个结果证明鲁棒性问题真实存在，却不能说明问题来自视觉表征、动作规划还是 future prediction。

于是阶段二没有贸然让 future 进入控制环，而是把它设计成 shadow observer：原动作先冻结并哈希，future 只在旁路生成，与动作实际造成的视觉变化做对齐。这样可以回答“OOD 下 future proxy 是否变差、是否与失败相关”，同时明确不能回答“future error 是否导致失败”。

### 3. 从“相关”推进到“动作是否真的读取内容”

观察到 future consistency 在 OOD 下变差后，仍不能说明动作会对 future 内容作出响应。为此项目侧实现轻量 Future-to-Action Adapter，并把干预拆成三组：

```text
B0 / parameter-free null  → 验证新增接口不会自行改变基础动作
correct future            → 注入当前样本生成的 K=1 latent
other-episode shuffle     → 只替换 future 内容，其余 current/context/noise 保持不变
```

zero-initialized gate 保证初始时 Adapter 与基础动作对齐；B0 重放给出非确定性 floor；shuffle 则区分“存在一个 latent/hook”与“future 的具体内容”两种解释。结果证明 future 内容在 8 条固定样本上均会改变动作，但变化幅度较小，因此只登记为技术动作敏感性，不上升为控制收益。

### 4. 用 matched 训练验证“会影响”是否等于“有帮助”

在训练阶段，K=0 与 K=1 共享数据、权重、sample/flow schedule、优化预算和固定 step-200 终点，唯一主变量是有无 K=1 future 内容。项目曾在开发 Gate 中看到过局部正信号，但 held-out flow 暴露出 fixed-noise/timestep 拟合，fresh cohort 又暴露 A0 稳定性不足；因此没有根据结果挑 LR、checkpoint 或降低门槛。

最终完整 28-train/4-development matched 实验得到 K=1 方向为负。这里最重要的决策是停止：不继续尝试 K=2/K=4，也不启动 OOD rollout 再用 outcome 反调 Adapter。最终结论收束为“future sensitivity 不等于 future utility”，而不是“future 普遍无用”。

### 5. 负结果后先做机制定位，而不是继续增加 K

K=1 没有通过 held-out utility 后，项目没有直接训练 K=2/K=4。Thought4 先用
exact-state Camera/Lighting 对照回答“信息根本不可读”还是“相机变化下不稳定”：
Clean geometry/motion 均可线性读取，但 Camera gap 高于 Lighting，rank-3 coordinate
shuffle 又会改变动作。由此把搜索空间从任意 future Adapter 收窄为
`camera_equivariance_gap`，同时用 FP32 arithmetic + bitwise correct control 排除
BF16 reconstruction 伪差。

### 6. 针对性方法也要经过 specificity、utility 与 rollout 三重门禁

进入 Thought5 时，我把原假设冻结成一条必须逐环成立的机制链：

```text
Camera Equivariance Gap
        → Geo-REPA + Pose/Ray 特异修复 representation
        → future geometry 改善
        → correct future utility 转正
        → Camera rollout success 提升
```

任意关键箭头未通过，都不能用前一环的弱正信号替代整链收益。Thought5 因而
没有把方法推荐当成方法有效，而是实现 matched B1/G3/G4：G3 使用正确
Geo-REPA correspondence，G4 只打乱 correspondence，二者共享 RayPose、auxiliary
loss 与 LoRA。G3 虽缩小 20.94% Camera gap，却低于 25% 门槛且 G4 更小；future
utility 仍负、Camera success 没有提升。因此停止预计 28–45 小时 formal。

停止后只解析已有工件，发现伤害集中在 Clean/低 sigma，同时 RayPose 路径虽非零
但缺少独立 ablation。这个步骤没有挽救旧假设，而是生成了一个更窄的新问题：
future utility 是否取决于 condition 与 effective-noise regime。因此，项目的闭环不是
“方法涨点”，而是“假设被充分干预并得到明确否证”。

### 7. 整体决策链

```text
同一模型在 OOD 下是否掉点？
        │  是：97.25% → 47.70%
        ▼
shadow future 在 OOD 下是否更不一致并与失败相关？
        │  是：但仅为非因果关联
        ▼
替换 future 内容是否会改变动作？
        │  是：8/8 技术反事实
        ▼
这种敏感性是否改善 held-out objective？
        │  当前冻结配方下没有
        ▼
Camera 缺口更符合哪类机制？
        │  camera equivariance gap
        ▼
针对性 GeoEq 是否通过 representation/utility/rollout Gate？
        │  没有：弱且非特异、utility<0、success 无 gain
        ▼
冻结负结果并停止 formal；只读定位 condition/noise-stage dependence
        │
        ▼
原机制链被否证，但研究问题得到回答
```

这条链路的核心价值是：前一阶段的弱正证据只用于解锁下一个更强问题，
不用于遮蔽后一阶段的负结果。这是完整的 hypothesis–intervention–falsification
闭环，只是不是正向性能闭环。

## 四、项目重难点、解决方案与亮点

| 重难点 | 关键考量与方案 | 可量化结果 | 可引导的面试问题 |
| --- | --- | --- | --- |
| 原版 LIBERO 与 LIBERO-Plus 都导出 `libero` | 不在同一解释器切换；每个进程只选择一个 checkout，并隔离 `LIBERO_CONFIG_PATH` | Clean/OOD 均完成真实运行 | 为什么不能只改 `PYTHONPATH`？如何防止 import 污染？ |
| 独立 episode 如何使用多 GPU | 每卡复制一份 policy，按 job ID 稳定分片；使用 episode-level data parallel，而非模型 DDP | 3 卡完成 7,571 rollout，0 重复/遗漏 | 为什么 DDP 不适合仿真评测？静态分片有什么尾部风险？ |
| OOD 评测单位容易放大 20 倍 | 区分 Clean 多 seed 与 Plus 官方 variant `all_once` 协议，并用配置门禁拒绝错误组合 | OOD 分母锁定为 6,771，而非 variant×20 | 如何平衡统计重复与遵循 benchmark 协议？ |
| 长时间 rollout 可能中断 | 内容哈希 job ID、逐 episode append+fsync、按完整 ID resume，聚合时核对 manifest | 7,571/7,571 完成，0 缺失、0 多余 | 崩溃在写一半时如何恢复？怎样保证幂等？ |
| Clean/OOD 公平比较 | condition 不进入 seed 公式；固定 checkpoint/stats；结果落盘 checkpoint SHA，聚合器拒绝身份不一致 | Clean/OOD 同权重，49.55 pp drop 可审计 | 哪些变量被控制了？为什么一对多 variant 不能直接做普通 McNemar？ |
| PyTorch 新版与旧 init-state 不兼容 | 只对 pinned checkout 的受信 `init_files`、指定扩展名显式 `weights_only=False`，先做 realpath 边界校验 | 修复真实 reset，且不放宽任意 pickle 路径 | 如何在兼容旧格式时控制反序列化风险？ |
| future 相关性与因果效用容易混淆 | 将 shadow observer、correct/null/shuffle 技术干预和 K=0/K=1 matched utility 分成三层 | 分别得到关联、动作敏感、离线负结果 | 为什么 shadow 指标显著仍不能说 future 导致失败？ |
| 新增 Adapter 可能破坏基础策略 | scalar gate 零初始化，parameter-free null 完全 bypass，冻结 backbone 并核对参数 SHA | B0/null 8/8 逐位一致；仅训练 1.371M 参数 | 为什么 A0、B0、null 不能混为一个基线？ |
| 离线训练容易被 flow/样本选择污染 | 训练与 probe flow 隔离，使用 held-out flow、fresh cohort、固定 endpoint 和预注册停止规则 | 识别 fixed-flow 假改善，最终保留负结果 | 如果继续调参可能涨点，为什么选择停止？ |
| cache 与实验工件需可恢复、可验证 | safetensors 原子 shard、逐文件/逐 tensor/逐样本 checksum，Adapter-only checkpoint 绑定配置与 split SHA | 96 latent、12/12 shard 恢复，单字节损坏 fail-fast | checksum 分层解决了哪些不同故障？ |
| BF16 subspace 干预可能制造伪差 | BF16 capture 升 FP32 做 projection/reconstruction，只在 replacement 前单次 cast；correct 必须逐位恢复 | 36/36 correct bitwise；36/36 shuffle 超 floor | 为什么不能简单放宽 BF16 tolerance？ |
| 三轨 Pilot 的方法和执行易混淆 | B1/G3/G4 matched 参数预算与 seed；三卡只改变 wave；worker import preflight + dtype/device 对齐 | 约 2h29m 完成，0 worker failure | 如何证明并行调度没有改变科学变量？ |
| Pilot 出现弱正信号但总 Gate 为负 | 冻结 recipe/formal；CPU-only 分解 condition/sigma/RayPose/G3-G4，不读取新 outcome | 避免 28–45h formal；25 项源 SHA 前后不变 | 为什么停止比继续调参更有研究价值？ |

## 五、项目亮点总结

### 研究亮点

- 没有把“模型能生成 future”“动作会随 future 改变”“future 提高任务表现”混为同一命题，而是用六层证据逐级验证、定位 Camera 缺口并审计针对性修复。
- 正式报告负结果，并用冻结停止规则阻止事后选择；最终不是“没有做成方法”，而是“完整机制假设已经被实验回答”。
- 将失败分析与调参分开：只读分解只负责将新假设收窄到 condition/noise-stage dependence，不回改旧 Gate 或重启已停止 recipe。
- 对所有结论标记证据等级：正式 rollout、post-run 关联分析、技术 smoke 和离线 development 结果各自使用不同措辞。

### 工程亮点

- 在不修改三个上游仓库的前提下完成策略、双仿真 backend、诊断和 Adapter 的项目侧集成。
- 大规模 GPU 仿真具备计划、分片、落盘、恢复、聚合、审计和失败视频闭环。
- 训练与 cache 工件同时具备 provenance、checksum、原子提交、确定性恢复和信息泄漏门禁。
- 通过 `504` 项测试覆盖旧 CLI、Thought1–5、三卡 worker 与只读分析，避免后续改动污染已冻结证据。

### 结果亮点

- `7,571` 次真实 rollout，Clean→OOD 下降 `49.55 pp`，相机视角成功率仅 `15.13%`。
- `732` episode / `1,010` probe 的 shadow diagnostic 显示 OOD future proxy 系统性变差。
- `1.371M` 参数 Adapter 的反事实证明 future 内容确实改变动作，但完整 matched 训练未观察到 held-out utility，形成“future sensitivity ≠ future utility”的清晰结论。
- `1.335M` 参数 GeoEq Pilot 将 Camera gap 缩小 20.94% 但未过门、future utility 仍为负且 success 无 gain；按规则停止 formal，并将结论从 geometry insufficiency 收窄到 condition/noise-stage-dependent utility。

## 六、面试伏笔与回答提纲

### 伏笔 1：为什么使用 episode-level data parallel，而不是 DDP？

每个仿真 episode 相互独立，计算瓶颈是模型推理与环境步进，参数之间不需要梯度同步。DDP 会增加无意义的同步，而每卡一个 policy/环境进程可以直接扩大 rollout 吞吐。为保证恢复后分配不漂移，我使用稳定 job ID 分片，而不是依赖运行时队列顺序。取舍是静态分片可能有尾部负载不均，因此需要从 pilot 监控各 rank 时长。

### 伏笔 2：49.55 个百分点能否完全归因于 OOD？

在本项目的实验边界内，checkpoint、stats、动作接口、基础任务身份和 seed 规则均固定，主要改变的是 LIBERO-Plus 官方环境变体，因此可以解释为“所测环境 shift 下的鲁棒性下降”。但 OOD variant 与 Clean anchor 是一对多，不能把 6,771 组都视为相互独立配对，也不能外推到 unseen task、真机或所有 OOD。

### 伏笔 3：为什么 future consistency 与失败相关，仍然不能说它导致失败？

shadow future 不反馈控制动作，基础 release 动作路径也不读取该 future。环境 shift 可能同时导致 future proxy 变差和动作失败，二者之间存在共同原因。阶段二只能提供观察关联；只有后续 correct/null/shuffle 的输入干预才能回答动作敏感性，而任务效用还要再经过 matched 训练或 rollout。

### 伏笔 4：8/8 到底代表什么？

它代表 8 条固定、单 task 样本中，correct-null 与 correct-shuffle 的动作差异都超过预先冻结的 replay floor，且 action tensor hash 改变。它不是 8 次机器人成功，更不是“100% 提升”。动作 cosine 仍约为 `0.9997`，所以还需要独立效用验证。

### 伏笔 5：zero-gated Adapter 为什么重要？

零初始化 gate 让新增分支在初始时不改变基础动作，从而把“接入模块造成的扰动”和“future 内容造成的扰动”拆开。A0 仍经过同结构但输入零 latent，B0 是官方基础路径，parameter-free null 则完全 bypass Adapter；三者回答的问题不同。

### 伏笔 6：项目中最有价值的一次“失败”是什么？

开发阶段一度出现较大的 fixed-flow loss 改善，但新建 held-out action-flow 后效果几乎消失，说明模型拟合了固定 noise/timestep objective，而不是形成跨 flow 的稳定收益。这个结果促使我把 flow schedule、probe namespace 和停止门槛全部冻结，也最终避免把局部训练信号包装成 OOD 提升。

### 伏笔 7：为什么最终不继续尝试更多 K、LR 或 checkpoint？

因为 step-200、K=1 配方和停止规则已在结果前冻结。看到 development 方向为负后继续挑 K、LR、checkpoint，或直接看 OOD success 再反调，会让验证集变成调参集。更合理的下一项研究应重新预注册多 task、多 seed 和独立 holdout，而不是覆盖本次负结果。

### 伏笔 8：工程上最棘手的兼容问题是什么？

PyTorch 2.6+ 将 `torch.load` 默认改为 `weights_only=True`，而旧 LIBERO init-state 保存的是 NumPy/pickle 对象，导致模型加载成功但环境 reset 失败。我没有全局关闭安全选项，而是把旧模式限制在 pinned checkout 的 `init_files` 受信目录和指定扩展名中，并先解析真实路径，避免任意路径反序列化。

### 伏笔 9：为什么 G4 的 representation gap 比 G3 更小？

G4 只打乱逐样本 Geo-REPA correspondence，仍保留和 G3 相同的 RayPose、正确
equivariance/pose auxiliary loss 与 LoRA，因此它不是“没有 geometry”的 control。
G3/G4 training total correlation 为 0.999986，Video feature-delta 方向 cosine 均值
0.938498，主 future geometry RMSE 也都没改善。最严谨的结论是正确 correspondence
没有被识别为变化来源；共享 conditioning/regularization 更可能，但缺少 G1/G2，
不能继续归因。

### 伏笔 10：为什么不直接花 28–45 小时跑 formal？

Pilot 的职责就是决定是否值得扩展。G3 在 representation、future geometry、absolute
utility、rollout 与 training 五项 Gate 都未通过；此时继续 formal 只会扩大一个
未达方向门的配方，并把 Pilot 变成调参集。我冻结负结果，只做不加载模型和 outcome
的只读分解；下一方法必须使用新预注册、新 cohort 与单变量对照。

### 伏笔 11：方法没有涨点，为什么这个项目仍然是完整的？

因为项目目标不是为了找到一组能涨点的参数，而是检验一条可证伪机制：Camera
Equivariance Gap 能否由 Geo-REPA + Pose/Ray 修复，并恢复 future utility 和
Camera success。Thought4 支持 gap 诊断，但 Thought5 的特异性、future geometry、
absolute utility 和 rollout Gate 均未通过，所以完整机制链已被回答为“不支持”。
我没有继续调参改写结论，而是冻结负结果并用只读分析生成下一个更窄假设。
因此它是完整的 hypothesis–intervention–falsification 闭环，只是不是正向性能闭环。

## 七、不同岗位的强调方式

### 投递算法/具身智能岗位

优先保留六层证据链、shadow future、correct/null/shuffle、zero-gated Adapter、held-out flow、Camera 等变性诊断和 GeoEq specificity control，突出实验设计、模型理解与因果边界。

### 投递机器学习工程/MLOps 岗位

优先保留多 GPU rollout、确定性 job manifest、断点恢复、原子 shard、checksum、provenance、504 项测试和 0 重复/遗漏，突出长任务可靠性与可复现系统。

### 投递研究工程师岗位

同时保留正式 OOD 指标和停止规则，重点讲清楚如何把研究问题转成可执行门禁，以及如何区分 engineering invalid、valid negative 与未回答问题。

## 八、事实核验与使用边界（面试准备使用，不放入简历）

- 阶段一是正式行为结果：`800` Clean + `6,771` runnable OOD，另有 `68` 条 expected skipped，不进入成功率分母。
- `7,571` 是真实 attempted rollout 数；磁盘总记录还包含 skipped，不能把二者混写。
- 阶段二数据收集完整，但统计计划没有在正式指标生成前冻结；应称为 protocol-consistent post-run association analysis，不称预注册确认性结论。
- 阶段二使用 decoded-frame VAE proxy，不是语义 future 正确率；也不能写成 future error 导致失败。
- 阶段三 Phase 1 的 `8/8` 是单 task 技术动作敏感性 smoke，不是成功率、总体显著性或 OOD 提升。
- 阶段三 Phase 2 只有一个 LIBERO-Goal task、28 train/4 development；`4/4` 是描述性结果，不能推广为所有 future 或所有 WAM 都无用。
- Thought4 的 `36/36` 是离线 action tensor 干预，不是机器人成功；分类为
  `camera_equivariance_gap`，不能写成 Camera rollout improvement。
- Thought5 是单 task 8/4/4 的方向性 Pilot；`1/4` 是每 condition 四条 rollout，
  不能作总体成功率推断。G3 未通过当前 recipe 的 Gate，不等于 Geo-REPA、RayPose
  或 future 普遍无效。
- Thought5 只读分解是 post-hoc exploratory；G3 Camera utility 的 `+0.004807`
  只是点估计，95% CI `[−0.009028, +0.018498]` 跨 0。condition/sigma 结果只能
  生成新假设，不能回改 25% 门槛或解锁 formal；RayPose 路径执行非零也不等于
  其独立因果贡献已识别。
- 当前没有完成失败视频的人工 taxonomy；可以写“保存并审计 3,563 个失败视频”，不能写“人工归纳出若干失败模式”。
- 仿真结果不能外推为真机、跨机器人平台、unseen task 或 unseen object 结论。

## 九、量化证据来源

- OOD 正式结果：`docs/thought1/report.md`
- Future consistency 正式结果：`docs/thought2/formal_results.md`
- K=1 动作反事实：`docs/thought3/phase1_action/report.md`
- 28/4 matched 训练结果：`docs/thought3/phase2_adapter/report.md`
- Camera geometry–action 正式诊断：`docs/thought4/formal_v6_results.md`
- GeoEq 单任务方向 Pilot：`docs/thought5/pilot_v4_results.md`
- Condition/flow/RayPose 只读失败分解：`docs/thought5/pilot_v4_readonly_failure_analysis.md`
- 最近工作时间线与统一口径：`docs/shared/recent_work_2026-07-27_to_2026-08-06.md`
- 工程难点与阶段台账：`docs/shared/engineering_highlights.md`
- 论文证据链：`docs/paper/evidence_chain.md`
- 当前测试结果：2026-08-06 本地执行 `pytest -q`，`504 passed, 5 warnings`
