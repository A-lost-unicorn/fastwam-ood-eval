# 论文证据链：从 OOD 缺口、Future Utility 负结果到相机等变性诊断

更新日期：2026-08-05
适用范围：Fast-WAM 官方 `libero_uncond_2cam224` checkpoint、标准 LIBERO、
LIBERO-Plus、本项目冻结的 K=1 Future-to-Action Adapter 配方，以及 Thought4
冻结的 geometry–action diagnosis，以及 Thought5 的方法、单 task Pilot v4 和
只读失败分解；Thought5 formal 多任务效果仍未运行。

![Thought1 到 Thought3 的证据阶梯](figures/figure5_evidence_chain.svg)

## 1. 总结

整条研究路线不是一次模型对比，而是五层逐级收紧的问题：

1. **行为事实**：Fast-WAM 在环境 OOD 下是否掉点？
2. **观察关联**：模型生成的 shadow future 在 OOD 下是否更不一致，并与失败相关？
3. **技术因果**：替换 future latent 内容是否会改变动作？
4. **任务效用**：让动作读取 K=1 future 后，是否真的改善 held-out 目标或 OOD success？
5. **缺口定位**：Camera OOD 的主要问题更接近 geometry unreadability、
   action interface gap，还是 camera equivariance gap？

前三层分别得到“是、是、是”；第四层在当前冻结配方上的离线答案是“没有观察到
改善”。第五层进一步发现 Clean geometry 在 Video/Action 表征中可读、Camera
扰动造成的 exact-state gap 显著大于 Lighting，且一个冻结的 rank-3 geometry
subspace 干预会改变动作，因此将下一方法假设定位为 `camera_equivariance_gap`。
Thought5 又验证了一个更具体的机制配方：G3 将 Camera representation gap 缩小
20.94%，但未达到 25% 门槛，absolute future utility 仍为负且 Camera rollout
未改善。因此论文最稳健的主结论仍是：

> Future sensitivity is not future utility：future 内容能够改变动作，但这一
> 技术敏感性没有自动转化为 held-out 控制收益。

它不是“未来对 Fast-WAM 永远无用”的证明，也没有回答 future 对 OOD rollout
success 的总体因果效应。

## 2. 总证据矩阵

| ID | 问题/命题 | 设计与分析单位 | 核心结果 | 证据类型 | 判定 |
| --- | --- | --- | --- | --- | --- |
| C1 | 官方 Fast-WAM 对所测环境 shift 敏感 | 冻结 checkpoint；800 Clean + 6,771 runnable OOD rollout | 97.25%→47.70%，−49.55 pp；0 exception | 在线行为评测 | **支持** |
| C2 | 不同 OOD 类型的破坏程度不同 | 五类 LIBERO-Plus 官方变体 | Camera 15.13%，Lighting 81.88% | 在线分层评测 | **支持** |
| C3 | OOD 下 future–realized proxy 变差 | 732 episodes、1,010 shadow probes、40 task 等权 | cosine distance +0.0316；direction cosine −0.1898 | 观察关联 | **支持** |
| C4 | future inconsistency 与 OOD failure 相关 | 255 success / 275 failure，排除 2 个 outcome mismatch | failure distance +0.0249；direction −0.2127 | 观察关联 | **支持** |
| C5 | future error 导致原 Fast-WAM 失败 | shadow future 不反馈动作 | 无可识别干预 | 因果证据 | **不支持/不可识别** |
| C6 | future latent 的具体内容进入 Action DiT | 固定 checkpoint，K=1 correct/null/shuffle，8 条样本 | correct−null、correct−shuffle 均 8/8 超过 replay floor；hash 8/8 改变 | 技术因果干预 | **支持（单 task smoke）** |
| C7 | K=1 Adapter 改善 held-out action objective | 28 train / 4 development；A0/A1 matched；固定 step 200 | A0 +1.845%，A1 −1.712%；A1 比 A0 高 3.624%，4/4 更差 | 预注册离线对照 | **本配方不支持** |
| C8 | K=1 改善 Clean/OOD success | Phase 2 停止规则禁止 rollout | 无结果 | 在线任务效用 | **未回答** |
| C9 | K=2/K=4 形成更优收益–延迟曲线 | 只有 cache/工程 smoke，无正式训练与 rollout | 无结果 | 比较效用 | **未回答** |
| C10 | Fast-WAM 在 OOD 中普遍不需要 future | 现有样本、task、K、seed 和配方均不足 | 无总体可识别证据 | 总体命题 | **不得声称** |
| C11 | Clean geometry/motion 在冻结表征中可线性读取 | 64 base state 的 dev/test split；三 seed linear probe | Video translation error 0.032814 m；Action current 0.021851 m；Action future SE(3) 0.105583，均优于 mean/shuffle control | 冻结表征诊断 | **支持** |
| C12 | Camera shift 具有超出 Lighting 的表征缺口 | exact-state paired Camera/Lighting；冻结层与 probe | Video Camera−Clean +0.020273 m，Lighting +0.011660 m；rank-3 Camera−Lighting 0.146284，95% CI [0.088519, 0.200310] | 配对表征诊断 | **支持** |
| C13 | probe-defined geometry subspace 对动作有技术因果影响 | 12 test state × 3 action seed；correct/shuffle intervention | correct 36/36 逐位恢复；shuffle 36/36 超 replay floor；action L2 mean 0.000768 | 技术因果干预 | **支持** |
| C14 | Robot-init 具有独立于 Camera 的同类缺口 | Robot-init 不是 exact-state control，冻结 distinct-pattern 判据 | `robot_init_pattern_distinct=false` | 探索诊断 | **不支持** |
| C15 | Geo-REPA / relative pose / camera-ray equivariance 改善 OOD success | 单 task B1/G3/G4 matched Pilot；4 episode/condition rollout | G3 Camera gap 缩小 20.94%<25%；absolute future utility −0.005231；Camera success B1=G3=1/4 | 方向性 Pilot | **当前配方不支持；formal 锁定** |
| C16 | Pilot 伤害能否定位到 condition/flow regime；RayPose 能否被单独识别 | 既有工件 post-hoc 只读分解；0 新 forward/outcome | G3 Clean −0.015268、Camera +0.004807；低 sigma 伤害；gate/grad/injection 非零但无 gate-zero ablation | 探索机制诊断 | **condition/noise 关联支持；RayPose 独立因果不可识别** |

“支持”只覆盖表中对应设计，不向其他 task、机器人平台、训练 seed、Adapter
结构或 K 外推。

## 3. 研究阶段一 → 阶段二

### 3.1 阶段一：建立行为缺口

[Thought 1 正式报告](../thought1/report.md)冻结官方权重、stats、动作接口和
任务协议，只改变环境条件：

| 条件 | 成功/分母 | 成功率 | 95% CI | Mean policy latency |
| --- | ---: | ---: | ---: | ---: |
| Clean LIBERO | 778/800 | 97.25% | [96.00%, 98.38%] | 972.06 ms |
| LIBERO-Plus OOD | 3,230/6,771 | 47.70% | [46.55%, 48.90%] | 969.84 ms |

计划中另有 68 个不可运行变体被显式标记为 skipped；7,571 个实际 rollout
全部完成，0 exception。成功率下降并不是由推理变慢解释的，因为两种条件的
单次 policy latency 近似相同。

![Fast-WAM 的 Clean/OOD 成功率](figures/figure1_ood_success.svg)

阶段一支持“存在大幅环境鲁棒性缺口”，但无法区分：

- 当前视觉表征是否失真；
- 动作分支是否选错动作；
- 模型内部的 future proxy 是否也失真；
- 显式 future 能否纠正动作。

### 3.2 阶段二：在不干预动作的前提下定位关联

[Thought 2 正式报告](../thought2/formal_results.md)使用同一 checkpoint，
将 future generation 放在 control loop 外。每个 probe 保存当前帧、预测
future、受保护原动作、实际后续画面与 outcome；同次运行内
1,010/1,010 个动作 hash 前后不变。

| Primary metric（40 task 等权） | Clean | OOD | OOD−Clean | 95% task-bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| Future latent cosine distance ↓ | 0.1025 | 0.1341 | +0.0316 | [0.0254, 0.0381] |
| Future latent L1 ↓ | 0.1431 | 0.1708 | +0.0277 | [0.0238, 0.0317] |
| Motion-direction cosine ↑ | 0.7416 | 0.5518 | −0.1898 | [−0.2134, −0.1664] |

![Future consistency 的 ID/OOD 与成败对比](figures/figure2_future_consistency.svg)

从阶段一到阶段二可以形成以下链条：

```text
同一 checkpoint 在 OOD 下 success 大幅下降
                    +
同一 checkpoint 的 shadow future proxy 在 OOD 下变差
                    +
OOD failure episode 的 proxy 平均更差
                    ↓
OOD 同时给控制与 future prediction 施压
```

最后一步只能写“共同压力与关联”。基础 Fast-WAM 动作没有读取 shadow future，
所以不能将其改写为：

```text
future error → action error → failure
```

此外，自动 latent proxy 测的是局部视觉变化相似性，不等价于语义正确、任务目标
正确或物理可行。正式人工盲审尚未完成。

## 4. Thought 3 Phase 1 → Phase 2

这一段专门补齐 Thought 2 缺失的“future 是否进入动作”与“进入后是否有益”。

### 4.1 Phase 1：技术动作反事实

[K=1 反事实报告](../thought3/phase1_action/report.md)在固定 A1 checkpoint、
固定当前观测、语言、proprio、action noise 和 timestep 的条件下，只改变
future 输入：

- `B0`：原始 Fast-WAM；
- `null`：不提供 future tensor，也不调用 Adapter；
- `correct`：当前样本在线生成的 K=1 future；
- `shuffle`：其他 episode 的 future，只替换 future 内容。

B0 两次 replay 与 formal null 均逐位一致；correct−null 和
correct−shuffle 的 action tensor hash 均在 8/8 样本上改变。normalized action
RMS 差异均值分别为 0.01105 和 0.01209。由此能识别：

> 在该 checkpoint 与八条样本上，future latent 的具体内容而非 hook 存在性会
> 改变 Action DiT 输出。

但 action cosine 仍高达 0.99977/0.99971，变化幅度较小；没有 rollout 时，
动作变化的方向与任务价值未知。

### 4.2 Phase 2：matched held-out utility

[28/4 matched 报告](../thought3/phase2_adapter/report.md)随后冻结单一配方：
一个 `libero_goal` task，28 条 train、4 条 development；A0(K=0) 与 A1(K=1)
使用相同 sample identity、loss weight、flow schedule、LR `3e-4`、seed 3407
和 200 updates。Fast-WAM 主体冻结，只优化 1.371M 参数 Adapter。

| 版本 | Initial development loss | Step-200 final | 相对 initial |
| --- | ---: | ---: | ---: |
| A0 / K=0 | 0.004234104 | 0.004155979 | +1.845% 改善 |
| A1 / K=1 | 0.004234104 | 0.004306583 | −1.712% 恶化 |

A1 final 比 A0 高 0.000150604（3.624%），且四条 development sample
全部为 A1 更差。12/12 hard checks 通过，排除了梯度断链、两轨训练不匹配、
checkpoint 损坏和 backbone 被更新等工程解释。

![动作敏感性与 held-out 效用](figures/figure3_sensitivity_vs_utility.svg)

![Phase 2 逐样本 A1 相对 A0](figures/figure4_phase2_per_sample.svg)

因此 Phase 1→Phase 2 的逻辑结论是：

```text
future content changes action                       已支持
future content changes action in a useful direction 未支持
K=1 improves held-out objective in this recipe      负结果
K=1 improves OOD rollout success                     未测试
```

按预注册停止规则，Phase 3 rollout、A2/A4 和事后 checkpoint/LR/K 选择均未启动。
这使负结果得以保留，也阻止用 OOD outcome 反向调参。

## 5. Thought 1 → Thought 5 的完整链条

| 阶段 | 输入是否改动 | 动作是否改动 | outcome 是否读取 | 获得的最强结论 |
| --- | --- | --- | --- | --- |
| Thought 1 | 环境变为五类 OOD | 否，官方策略 | 是 | Fast-WAM 对所测环境 shift 高度敏感 |
| Thought 2 | 旁路生成 future | 否，hash 保护 | 是，仅关联分析 | OOD future proxy 变差并与失败相关 |
| Thought 3 Phase 1 | correct/null/shuffle future | 是，固定其余随机量 | 否 | future 内容对动作有技术因果影响 |
| Thought 3 Phase 2 | A0/K0 与 A1/K1 训练输入 | 是，matched 训练 | 只读 action target；不读 success/OOD | K=1 未改善本配方 held-out objective |
| Thought 4 formal v6 | exact-state Camera/Lighting 与冻结 geometry subspace | 仅干预离线 action tensor | 否 | 缺口定位为 camera equivariance；尚无新方法效用 |
| Thought 5 Pilot v4 | B1/G3/G4 matched 训练、future counterfactual 与四 condition rollout | 是；G3/G4 训练并在线执行 | 是，但只作单 task 方向 Gate | 当前 G3 recipe 未把弱表征变化转成 positive utility 或 success；formal 停止 |
| Thought 5 只读分解 | 不新增输入；只读 Pilot 工件 | 否 | 不新增 outcome | 伤害集中 Clean/低 sigma；RayPose 路径非零但独立因果未识别 |

该链条逐步排除了三个常见逻辑跳跃：

1. **从 OOD 掉点跳到“需要未来”**：Thought 1 只能发现问题，不能指定解法。
2. **从 future 相关跳到“动作依赖未来”**：Thought 2 的 future 是旁观者；
   直到 Thought 3 Phase 1 才用 intervention 证明动作敏感。
3. **从动作敏感跳到“future 有用”**：Thought 3 Phase 2 的负结果表明两者必须
   分开验证。

## 6. Thought 4：Geometry–Action Gap 定位

[Thought4 formal v6 报告](../thought4/formal_v6_results.md)在不训练 Fast-WAM、
不读取 future RGB、success 或 OOD rollout 的条件下，使用 64 个 base state
构造 256 个配对样本并捕获 12,544 条中间特征。运行前由 smoke v8 在真实
`[1,98,3072]` BF16 hidden 上验证 FP32 subspace arithmetic：correct control
经过 BF16→FP32 reconstruction→BF16 replacement 后逐位恢复，action L2 为 0。

三组跨 seed probe 均通过冻结 readability control：

| Probe | Clean error | Mean control | Shuffle control | 相对 mean 改善 |
| --- | ---: | ---: | ---: | ---: |
| Video current geometry / translation | 0.032814 m | 0.061369 m | 0.067243 m | 46.53% |
| Action current geometry / translation | 0.021851 m | 0.061369 m | 0.077118 m | 64.39% |
| Action future motion / SE(3) composite | 0.105583 | 0.197027 | 0.225015 | 46.41% |

这排除了“Clean geometry 在所测表征中完全不可读”这一简单解释。随后，冻结
Video probe 在 exact-state paired panel 上得到 Camera−Clean RMSE
`+0.020273 m`，Lighting 为 `+0.011660 m`；三个 probe seed 的 Camera 点估计
均高于 Lighting。选定 rank-3 坐标的 Camera−Lighting 配对差为 `0.146284`
（95% bootstrap CI `[0.088519, 0.200310]`）。Robot-init 不是 exact-state
control，且冻结判据为 `robot_init_pattern_distinct=false`，不能写成独立机制。

最后，在 12 个 test state × 3 个 action seed 上，只替换 probe-defined rank-3
geometry subspace：

- correct reconstruction 为 36/36 逐位恢复，排除重建/数值噪声；
- shuffled coordinates 为 36/36 超过 replay floor；
- action L2 mean 为 `0.000768`，translation/rotation difference mean 分别为
  `0.001094/0.000410`，gripper difference 为 0；
- backbone 参数 SHA 前后相同，未训练模型。

因此 Thought4 能支持“Camera-specific representation/equivariance gap 与动作
敏感性并存”，冻结分类为 `camera_equivariance_gap`。它只为下一项独立研究
推荐 `Geo-REPA + relative pose / camera-ray equivariance`，不能写成该方法
已经改善 representation、action 或 rollout success。

## 7. Thought5：定向干预、负 Gate 与失败定位

[Thought5 Pilot v4 结果](../thought5/pilot_v4_results.md)在单一
`libero_goal` task 上 matched 比较 B1、G3 与 shuffled-target G4：

| Endpoint | B1 | G3 | G4 | Pilot 判定 |
| --- | ---: | ---: | ---: | --- |
| Camera representation gap | 0.002246 | 0.001776 | 0.001666 | G3 缩小 20.94%<25%，且 G4 更小 |
| Camera future geometry RMSE | 0.341277 | 0.341320 | 0.341331 | 主指标无改善 |
| Correct future utility | −0.015649 | −0.005231 | −0.011302 | G3 缓解伤害但仍为负 |
| Camera rollout success | 1/4 | 1/4 | 0/4 | G3 对 B1 无提升 |

所有 worker/collector 均 complete，但 training、representation、future geometry、
future utility 和 rollout 五项方向 Gate 全 false，因此 `formal_unlocked=false`。
这是有效停止结果，不是运行失败，也不是对 Geo-REPA 或 future 的多任务总体否证。

[只读失败分解](../thought5/pilot_v4_readonly_failure_analysis.md)进一步发现 G3
future utility 在 Clean 为 −0.015268、Camera 为 +0.004807；22/32 个无序 flow
seed-slot 的均值为负，低 effective sigma 的伤害最强。RayPose 的 final gate、
记录梯度和重建 injection 均非零，说明路径实际执行，但没有 gate-zero ablation，
不能识别其独立动作贡献。G3/G4 训练与 video feature-delta 高度同向，且 G4 只
shuffle Geo-REPA correspondence、仍保留正确 equivariance/pose，因此 G4 更小的
gap 更支持共享 conditioning/auxiliary-loss/regularization 解释，而非正确逐样本
Geo-REPA correspondence 已被证明有效。

## 8. 从 0 到 1 的实施过程

| 时间/里程碑 | 从无到有的能力 | 关键卡点与处理 | 证据资格 |
| --- | --- | --- | --- |
| Thought 1 环境搭建 | 同一策略跨原版/Plus backend 的可恢复多 GPU runner | 两套包同名、EGL、init-state、assets、PyTorch 兼容 | 工程 + 正式 rollout |
| Thought 1 正式运行 | 确定性 job manifest、7,571 traces、3,563 failure videos | 68 个不可运行变体显式从分母分离 | 正式行为证据 |
| Thought 2 shadow path | 不改变下一动作的 future 生成、媒体与时间对齐 | future/action 语义、VAE proxy、static/no-op threshold | 关联诊断 |
| Phase A/B | 1.371M zero-gated Adapter、K schema、cache/checkpoint/mock trainer | 冻结 upstream、旧 CLI 回归、信息泄漏门禁 | CPU/mock 工程证据 |
| Phase C | 一条真实 LIBERO 样本的 K1/2/4、forward/backward/memory | BF16→FP32 LayerNorm、video decoder fallback | 单卡真实 smoke |
| Phase D | 32 样本×K1/2/4，共 96 latent、12 shards | episode split、paired noise、resume/checksum/corruption | 真实 cache smoke |
| Gate E.1–E.9 | 从可拟合性到 multi-flow、fresh cohort、trajectory、tail audit | 区分 invalid run 与 valid negative；禁止降门槛/挑 step | 开发与 fail-closed 证据 |
| Thought 3 Phase 1 | online correct/null/shuffle action intervention | formal null、B0 replay、no-cache/no-target/no-RGB | 技术因果 smoke |
| Thought 3 Phase 2 | 完整 matched A0/A1 单配方训练 | 双 GPU 独立轨、固定 step-200、12 项 hard check | 离线负结果 |
| Thought 4 smoke v8 | 真实 BF16 capture 的 FP32 subspace arithmetic | correct 必须逐位恢复；不靠放宽 tolerance | 数值/干预门禁 |
| Thought 4 formal v6 | exact-state paired probes、probe-first commit、rank-3 intervention | simulator replay 对齐、1,586 工件 manifest、自哈希作用域审计 | 正式离线机制诊断 |
| Thought 5 audit→Pilot v4 | layer-15 Geo-REPA、ray/pose injection、B0–G4、matched future probe 与三层判定器 | dtype、worker import、三卡调度、single-task stop gate | smoke + 有效方向 Pilot；formal 未解锁 |
| Thought 5 只读失败分解 | condition/flow/action、gate/LoRA/trajectory、G3/G4 feature delta | 严格区分 utility、final-action sensitivity 与 inference denoise step | post-hoc 探索诊断；不改 Pilot |

Gate E 的完整协议、失败与恢复记录见
[Gate E 索引](../thought3/gate_e/)。它们的论文价值主要是说明研究流程如何避免：

- 把实现报错当作科学负结果；
- 在八条样本上不断增加 flow 直至得到想要的结论；
- 根据 development 结果挑 step、LR 或门槛；
- 在离线方向为负时仍打开 OOD rollout 并据此调参。

## 9. 论文可写、必须限定与不可写

### 可写

- 官方 release checkpoint 在本次 LIBERO-Plus 五类环境 shift 上从 97.25%
  降至 47.70%。
- OOD 下自动 future–realized consistency proxy 显著变差，并与失败相关。
- 对一个固定 K=1 checkpoint 的八样本技术反事实中，future 内容改变动作。
- 在一个 task 的预注册 28/4 matched 配方中，K=1 未改善 held-out action
  objective，且比 K=0 高 3.624%。
- 现有证据展示了 future sensitivity 与 future utility 的分离。
- Clean geometry/motion 在所测冻结表征中可读；Camera 的 exact-state gap 大于
  Lighting，且 probe-defined rank-3 geometry subspace 会改变动作 tensor。
- Thought4 的冻结分类为 `camera_equivariance_gap`，它是方法选择依据而不是
  新方法效果。
- Thought5 单 task Pilot 中，G3 缩小了负 future utility，但没有使其转正，也没有
  提高 Camera rollout success；当前 recipe 按 Gate 停止。
- 只读分解显示 G3 的伤害集中在 Clean 与低 effective sigma；RayPose 有非零
  gate/gradient/injection，但独立因果贡献尚未识别。

### 必须同时限定

- Thought 2 是 protocol-consistent post-run analysis，不是预冻结 confirmatory
  analysis。
- Thought 2 的 proxy 不是语义 future 正确率。
- Phase 1 的 8/8 是动作 hash/sensitivity，不是成功率。
- Phase 2 的 4/4 是描述性单 task development 结果，不是总体显著性结论。
- Phase 2 没有读取 OOD、rollout success 或真实 future RGB。
- Thought4 只做冻结 probe 与离线 action intervention；36/36 不是 episode
  success，Robot-init 也不是 exact-state control。
- Thought5 是单 task、4 episode/condition 的方向性 Pilot，不是 H1/H2/H3 的正式
  多任务否证；condition/sigma 分解是 post-hoc 探索分析。
- `execution_integrity.json` 的自哈希只覆盖写入时的 11 字段核心 payload；后续
  追加字段由 1,586-entry artifact manifest 的逐文件 SHA 覆盖。该作用域缺陷已
  披露，冻结结果不原地修补。

### 不可写

- future error 导致了 Thought 1 的失败；
- Fast-WAM 的动作在基础版本中读取 shadow future；
- K=1 会降低或提升 OOD 成功率；
- K=2/K=4 无用；
- Fast-WAM、所有 WAM 或所有 OOD 场景都不需要未来想象；
- Geo-REPA、relative pose 或 camera-ray equivariance 已经有效或普遍无效；
- G4 更小的 gap 证明了“只是普通正则化”，或 RayPose 单独导致动作变化；
- 某个 inference denoising step 或 action 尾段导致了 utility 伤害；
- rank-3 action tensor 变化会提高 Camera OOD success；
- 仿真结论等价于真机或跨平台结论。

## 10. 当前论文结论与下一项独立研究

当前论文可以作为一篇**审计与负结果论文**完成：它回答“现有证据是否足以支持
future utility”，答案是否定的，并展示为什么需要将关联、动作依赖与任务效用
拆开。

若未来重启 future utility 路线，不应回用当前四条 development sample 调参。
Thought5 已实际检验该方向的最小 G3 配方并触发停止。若另起新配方，应使用新
预注册与未使用 cohort，优先检验 condition-aware/low-sigma mitigation，并用
gate-zero 或 G1/G2 matched 对照区分 RayPose、Geo-REPA correspondence 与共享
正则化；不能回用本 Pilot 调阈值后冒充确认性结果。任何新研究均不能事后改变
本文 K=1 负结果、Thought4 formal v6 或 Thought5 Pilot v4 的冻结登记。
