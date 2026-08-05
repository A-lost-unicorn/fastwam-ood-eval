# Thought5 Pilot v4 只读失败分解

更新日期：2026-08-05

## 登记性质与边界

本分析是在三卡 Pilot v4 已完成并触发停止规则之后，针对已有工件进行的
**post-hoc、只读、探索性诊断**。它不重跑模型、不使用 GPU、不训练、不渲染、
不执行 rollout，也不读取新的 outcome。分析前后逐文件验证 25 项输入 SHA；派生结果
写入独立 sibling namespace，不修改 Pilot v4 权威目录。

因此原判定保持：

```text
g3_direction_observed=false
formal_unlocked=false
current_recipe=stopped
```

本文中的 bootstrap 只用于定位下一条假设，不替换预注册 Pilot Gate，不用于调阈值，
也不是多任务正式推断。

## 一、伤害发生在哪些 condition？

Future utility 定义仍为 `A0 loss - A1 loss`：正值表示正确 future 有帮助，负值表示
正确 future 比 null 更差。

| Backbone | Clean utility | 95% episode-grouped CI | Camera utility | 95% episode-grouped CI | Clean 负值 | Camera 负值 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | -0.021569 | [-0.035824, -0.013017] | -0.009730 | [-0.014508, -0.006186] | 123/128 | 84/128 |
| G3 | -0.015268 | [-0.029476, -0.002836] | +0.004807 | [-0.009028, +0.018498] | 95/128 | 40/128 |
| G4 | -0.018712 | [-0.031480, -0.010833] | -0.003892 | [-0.017139, +0.015238] | 123/128 | 76/128 |

定位结论：G3 的 aggregate 伤害来自 **Clean**；Camera 均值已经转正，但 4 个 formal
episode 的探索性区间仍跨 0，不能写成稳定收益。G3 的 `Camera-Clean` utility 差为
+0.020076，探索性区间 `[+0.006834,+0.031834]`，说明条件差异值得作为下一条假设，
但这仍是单 task post-hoc 结果。

Future-utility collector 只包含 Clean 与 Camera。Lighting、Robot-init **没有**对应的
A0/A1/AS 工件，不能从现有数据回答其 future utility；不能拿 rollout success 冒充。
已有 4-episode rollout 中，B1/G3 在 Clean、Camera、Lighting、Robot-init 均分别为
0.25、0.25、1.00、1.00；G4 分别为 0.25、0、1.00、1.00。这只说明 rollout 方向，
不提供缺失 condition 的 future-utility 分解。

## 二、伤害发生在哪些 flow/action slots？

### 2.1 Flow objective

`flow_slot=171..202` 只是确定 action-noise 与 timestep seed 的无序 identity slot，
不是动作早/晚位置，也不是 inference denoising iteration。按 slot 聚合：B1、G3、G4
分别有 32/32、22/32、30/32 个 slot 的平均 utility 为负；所以 G3 改善了负值覆盖面，
但未消除尾部风险。

用保存的 `action_timestep_seed` 和 Fast-WAM 的 shift=5 scheduler 重建实际 BF16
effective sigma 后，G3 为：

| Effective sigma | n | Mean utility | 负值比例 | Clean mean | Camera mean |
|---|---:|---:|---:|---:|---:|
| [0, 0.25) | 18 | -0.049520 | 88.9% | -0.059421 | -0.037143 |
| [0.25, 0.50) | 20 | -0.040457 | 85.0% | -0.062250 | +0.010396 |
| [0.50, 0.75) | 60 | +0.002821 | 40.0% | -0.012122 | +0.015048 |
| [0.75, 1.00] | 158 | +0.001216 | 49.4% | -0.002095 | +0.004365 |

G3 的 sigma–utility Pearson 相关为 `+0.446`。这支持“伤害集中在低噪声、接近动作
target 的 objective，Clean 尤其明显”这一诊断；它是关联，不是 denoising 因果。

### 2.2 32-step action chunk

现有 technical counterfactual 保存的是 final action chunk 的逐位置 L2 变化，不是逐位置
loss。G3 correct-vs-null 的变化为：

| Action segment | Mean per-step L2 | 角色 |
|---|---:|---|
| 0–9 | 0.156641 | rollout 实际执行的 prefix |
| 10–20 | 0.153542 | 当前 chunk 未执行中段 |
| 21–31 | 0.173893 | 当前 chunk 未执行尾段 |

尾段比 executed prefix 高约 11.0%，峰值在 index 28（0.194378）。但 B1/G4 也有相似
“尾段略大”模式，且 utility loss 已在完整 32-step chunk 上 reduction，所以不能写成
“后几步动作导致伤害”，只能写“future 对尾段动作的改变略大”。

### 2.3 20-step inference denoising

无法从已有工件定位。保存内容只有最终动作块、20-step 数量和 schedule hash，没有每次
denoising 的 action/latent/loss。任何“某一 denoising step 特别差”的结论都会超出证据。

## 三、RayPoseEncoder 真的被使用了吗？

答案分两层：**执行和更新证据为是；独立因果贡献尚未识别。**

| 指标 | B1 | G3 | G4 |
|---|---:|---:|---:|
| final gate | 0 | -0.002483804 | -0.002483254 |
| tanh(gate) | 0 | -0.002483799 | -0.002483249 |
| step-1 RayPose grad L2 | 0 | 0.004956 | 0.004929 |
| step-2 RayPose grad L2 | 0 | 0.005659 | 0.005700 |
| final injection RMS（48 entries） | 0 | 0.000315677 | 0.000315549 |
| final injection per-token L2 | 0 | 0.017479 | 0.017472 |

补充证据：G3 step 1/2 的 LoRA gradient L2 为 0.003583/0.004931，Geo projector 为
0.230688/0.199767；RayPose step 1 有 3,085 个非零梯度元素，step 2 为全部 441,357。
G3 的 effective LoRA delta Frobenius 为 K=1.230522、V=1.776661。因此该路径并非
dead code，也不是完全闭合 gate。

但 gate 的绝对值很小，只有 step 1/2 保存梯度、只有 step 100 保存 checkpoint；没有
gate/norm 全轨迹，没有 pre-injection hidden tensor，无法计算 injection/hidden 比例。
更关键的是，没有 G3 gate-zero checkpoint/action ablation，而 LoRA 与 RayPose 同时训练，
所以“G3 与 B1 action hash 不同”不能归因于 RayPose 单独造成。

严谨表述应为：`executed_nonzero_but_causal_contribution_not_isolated`。

## 四、为什么 G4 的 representation gap 比 G3 小？

| Variant | Camera representation gap | 相对 B1 减少 |
|---|---:|---:|
| B1 | 0.002246117 | — |
| G3 | 0.001775704 | 0.000470413（20.94%） |
| G4 | 0.001666233 | 0.000579884（25.82%） |

这一现象不支持把 G3 的变化直接解释为“学到了正确的逐样本几何对应”：

- G4 只 shuffle Geo-REPA target，仍保留与 G3 相同的正确 equivariance loss、pose loss、
  RayPose conditioning 和 LoRA 训练；它不是全部 geometry 的 random control。
- G3/G4 训练 total trajectory 相关为 0.999986，original objective 为 0.999999；最终
  RayPose 非 gate 参数余弦为 0.9999996，Geo projector 为 0.999647，LoRA factors 为
  0.999435。
- 对 128 个 video-source probe，G3−B1 与 G4−B1 feature-delta 方向余弦均值为
  0.938498（中位数 0.938681）；说明两者主要沿同一表征方向移动。
- primary future camera geometry RMSE 并未改善：B1=0.3412765、G3=0.3413198、
  G4=0.3413308。

最稳妥结论是：`correct_georepa_correspondence_not_identified_as_cause`。现有证据更偏向
G3/G4 共享的 conditioning、equivariance/pose 辅助目标或普通 LoRA regularization，
但 Pilot 没跑 G1/G2，无法在这些解释中继续选择。

## 五、下一条假设与停止决定

当前 G3 recipe 继续停止，formal 继续锁定。本次只读分解支持下一次另行预注册时优先
考虑两个单变量问题：

1. **condition-aware / low-noise mitigation**：防止 Clean、低 sigma 的 future fusion
   伤害，同时保留 Camera 的正向均值；
2. **RayPose identification**：使用 matched gate-zero 或 G1/G2 对照，将 RayPose、
   Geo-REPA correspondence 与共享正则化拆开。

这不是执行许可，也不能回头选择旧 Pilot checkpoint、放宽 25% 门槛或直接启动 formal。

## 六、复现与工件

执行仅需 CPU，正常约 10 秒，不设置任何确认变量或 GPU：

```bash
bash scripts/run_thought5_pilot_v4_readonly_failure_analysis.sh
```

派生工件目录：

```text
outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4_readonly_failure_v1/
├── analysis_result.json
├── report.md
├── condition_utility.csv
├── flow_objective_rows.csv
├── action_horizon_sensitivity.csv
├── training_trajectory.csv
└── artifact_manifest.json
```

`analysis_result.json` 保存四问的结构化答案、所有 unavailable 边界、源 SHA 和 analyzer
代码 SHA；`artifact_manifest.json` 再冻结派生文件。原 Pilot v4 目录不写入任何文件。
