# Thought4 Formal v6 正式结果

- 结果状态：`FORMAL COMPLETE / VALID DIAGNOSTIC RESULT`
- 分类：`camera_equivariance_gap`
- 唯一后续分支：`Geo-REPA + relative pose / camera-ray equivariance`

本结果定位冻结 Fast-WAM 中的 geometry–action gap，不是新方法结果，也没有运行
policy success rollout。它支持“相机变换下的几何等变性缺口”这一诊断，不支持
“建议方法已经改善 OOD 成功率”。

## 1. 运行身份与规模

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_diagnosis_v6` |
| project commit | `46d03f23e88afef79aa63204c13dea6dd3eb7d19` |
| config fingerprint | `3b14a7d7fd09deda9253bb1cd9950d9c4b5bd0cdf9f124a4dfede22add5c24f6` |
| planned cohort SHA | `9af7cf7c1933fb1e5574099361f6d7dcc7500727480ecb4bbf010089f28d8f04` |
| materialized cohort SHA | `fca0444fc786f88db84b914695ddcd340860fe36cecca76a185774869a668210` |
| checkpoint SHA | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| physical GPU | 2 |
| start / finish | 2026-08-03 05:57:33 / 07:16:21 UTC |
| wall time | 4,728.05 s（1h18m48s） |
| base states | 64：40 train / 12 development / 12 test |
| paired samples | 256：64 states × 4 conditions |
| feature records | 12,544 |
| probe rows | Video 1,080；Action 192；layer summaries 424 |
| intervention comparisons | 36：12 held-out Clean targets × 3 action seeds |

Alignment QC 为 56/64 pass、8/64 fail；原 64 个 state identity 全部保留，
`selection_effect=none_all_planned_base_states_retained`。这 8 条只描述 demonstration
prefix 与 parquet EEF 的 replay fidelity，不参与样本、层或方法选择。

## 2. Smoke v8 与 FP32 correct-control Gate

formal v6 启动前，同一 commit 的 smoke v8 于 05:45:24–05:56:20 UTC 完成：

| 检查 | 结果 |
| --- | --- |
| 真实 capture | `[1,98,3072]`，`torch.bfloat16` |
| arithmetic | BF16 capture → FP32 coordinates/residual/reconstruction → 单次 BF16 cast |
| input/output tensor SHA | 均为 `9a161737...d2487` |
| cast 后 max-abs | `0.0` |
| bitwise correct | `true` |
| reconstructed consumer replacement | PASS；action L2=`0.0`，允许值 `1e-6` |
| backbone SHA before/after | 均为 `ac0dd59d...b4f8` |
| smoke internal result SHA | `b674180826bece52c327e22e56220c9749f15b0be019b29f56e228b8432473f1` |
| smoke file SHA | `c2e1199172e1dda004385ec1c723707fb087e6f935bb5803f324ddfb88c49a02` |

formal smoke gate 的 21 项检查全部为 true，包括 same-project-commit、真实 BF16
bitwise reconstruction、alignment 实体 SHA、Robot-init input-state 和 scope。
因此 v6 不再复用旧 smoke v7 的 raw identity replacement 结论。

## 3. Probe 在 intervention 前已冻结

两组 panel 完成后先写出 `video_probe_results.json`、
`action_probe_results.json`、`layer_summary.json` 和
`probe_stage_result.json`，然后才进入 Phase 4-C。冻结选择为：

```text
mot.video_kv_cache.15.v
layer 15 / spatial_mean / linear
target = eef_object_translation_camera
selection = mean development loss over seeds 4407/4408/4409
development loss mean = 0.1167249829
test_or_ood_read = false
```

`probe_stage_result.json` 状态为 `complete_before_intervention`，result SHA 为
`c2f8b1d198205ae310a183fa301d3573911b26a03c19e8e1a09d96758e7018eb`。

## 4. 冻结表征中的几何与运动可读性

以下是冻结规则选中的三组 linear probe 的跨三 seed Clean 误差。Video/Action
current-geometry 单位为米；Action motion 使用预注册 SE(3) composite。

| Probe | 位置 | Probe error | Train-mean control | Shuffled control | 相对 mean 改善 | 相对 shuffle 改善 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Video current geometry | `mot.video_kv_cache.15.v` | 0.032814 | 0.061369 | 0.067243 | 46.53% | 51.20% |
| Action current geometry | `action_expert.blocks.15.norm1@step19` | 0.021851 | 0.061369 | 0.077118 | 64.39% | 71.67% |
| Action future SE(3) | `action_expert.blocks.15.norm1@step19` | 0.105583 | 0.197027 | 0.225015 | 46.41% | 53.08% |

三者都超过预注册的 mean/shuffle 各 5% 改善门槛。因此：

- Clean Video geometry 可读；
- current geometry 已进入 Action hidden；
- Action hidden 中也存在可读的未来 SE(3) motion structure。

这排除了“Clean Video 根本没有几何”以及“Video 几何完全没有进入 Action”作为本次
优先分类，但 probe 可读性不等于策略会正确使用该几何。

## 5. Exact-state Camera gap

在每个 seed 都先用 development loss 冻结 feature group，再只读 test
Clean/Camera/Lighting exact-state pairs：

| 指标 | Camera | Lighting |
| --- | ---: | ---: |
| paired RMSE gap 跨 seed 均值 | +0.020273 m | +0.011660 m |
| 三 seed 中最保守 95% CI lower | +0.002092 m | +0.000772 m |
| 三 seed中最大 95% CI upper | +0.040799 m | +0.020450 m |
| 每个 seed 的 CI lower > 0 | 3/3 | 3/3 |

Camera gap 的均值比 Lighting 大 0.008613 m（相对 Lighting 大 73.87%），且三个
seed 的 Camera point estimate 都高于对应 Lighting。

在最终被选 rank-3 geometry subspace 内，held-out coordinate shift 为：

| Condition | Coordinate L2 estimate | 95% episode-grouped CI | Shift/Clean norm mean | Exact state |
| --- | ---: | --- | ---: | --- |
| Camera | 0.295093 | [0.246436, 0.348635] | 0.791207 | 是 |
| Lighting | 0.148809 | [0.128494, 0.165712] | 0.376373 | 是 |
| Robot-init | 0.096476 | [0.075430, 0.124942] | 0.241280 | 否 |

Camera−Lighting paired coordinate difference 为 `0.146284`，95% CI
`[0.088519, 0.200310]`，不跨 0。Robot-init 按协议只作非 exact-state 描述；其
pattern 没有满足“区别于 Camera”的冻结判据，不能用来加强 Camera 因果结论。

## 6. Geometry-subspace intervention

SVD 子空间 rank=3，解释 probe weight energy 100%；真实 hidden projection energy
很小，36 次比较的均值为 0.1043%，因此这是低维、局部 intervention，不是整条
hidden replacement。

| 检查/指标 | 结果 |
| --- | ---: |
| Correct reconstruction bitwise | 36/36 |
| Correct vs unhooked action L2 | 36/36 为 0 |
| Replay floor action L2 | 36/36 为 0 |
| Shuffle 超过 replay floor | 36/36（100% > 75% frozen threshold） |
| Shuffle action L2 mean / min / max | 0.000768 / 0.000559 / 0.000913 |
| Translation difference mean | 0.001094 |
| Rotation difference mean | 0.000410 |
| Gripper difference | 0 |
| Intervention/hidden norm mean | 3.283% |
| Coordinate norm ratio range | [0.999999894, 1.000000061] |

因此 probe-defined geometry subspace 对动作具有稳定的技术因果影响。但本阶段没有
在环境中执行这些 action chunks，OSC action delta 也没有被伪装成真实 EEF
trajectory；不能从上述幅度推断成功率方向或任务价值。

## 7. 冻结方法分类

冻结决策输入为：

| Evidence boolean | 值 |
| --- | --- |
| Clean Video geometry readable | true |
| Camera Video gap significant | true |
| Camera gap larger than Lighting | true |
| Action current geometry readable | true |
| Action motion readable | true |
| Geometry subspace action-sensitive | true |
| Robot-init pattern distinct from Camera | false |

按 `thought4.method_rule.v1` 的优先级，唯一分类为：

```text
classification = camera_equivariance_gap
recommendation = Geo-REPA + relative pose / camera-ray equivariance
```

方法选择 SHA：
`a6a9351819b54c26fc3421b65f6e74752b61e3f9dc34a12ec5c1a8a746793b33`。

## 8. 完整性审计与已披露缺陷

2026-08-03 的只读文档审计结果：

- `artifact_manifest.json` 含 1,586 个工件；全部路径存在，byte size 与 file SHA
  完全匹配；manifest 内部 SHA
  `0dbd9d7f8fde401b0eef420f0291c9977e82a5bf4c1b70364a2a7d4cde97d1c9`
  有效；
- planned/materialized cohort、alignment、Video/Action probe、layer summary、
  probe stage、intervention、diagnostic evidence 和 method selection 的内部 SHA
  均可重算；
- Fast-WAM parameter SHA before/after 均为
  `ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8`，
  trainable backbone parameter count=0；
- `future_rgb_read=false`、`success_outcome_read=false`、
  `fastwam_training_performed=false`。

存在一个不改变数据或分类的 provenance 缺陷：writer 先生成核心
`execution_integrity` SHA，随后追加 runtime/smoke/probe/alignment 字段但没有重算。
因此 stored `integrity_sha256=a08ac875...f5e71` 精确匹配核心 11 字段，却不匹配
追加字段后的完整 JSON；完整 JSON 的只读 canonical SHA 应为
`41694c83359872db04d7550e00a19a3798df641b13179456ea2f45c0b475b295`。

原 completed 工件不修改。该文件本身的 file SHA
`be3012602230ed013141c75580ddfd24f33fb57be19bde706f339e97579bd347`
已被有效 artifact manifest 覆盖，追加字段引用的独立工件 SHA 也均有效。因此本次
登记为“有效 formal diagnostic + 已披露的 self-hash scope 缺陷”，而不是科学
invalid run。后续新运行前应修复 writer 并增加 full-object self-hash 回归，但不能
回填或覆盖 v6。

## 9. 能证明与不能证明

可以写：

- 冻结 Fast-WAM 的 Video/Action hidden 中存在 probe-readable geometry/motion；
- exact-state Camera shift 对 Video geometry 的破坏大于 Lighting；
- rank-3 probe-defined geometry subspace 的 matched shuffle 稳定改变动作；
- 冻结规则将缺口定位为 `camera_equivariance_gap`。

不能写：

- Camera geometry gap 是 Thought1 Camera failure 的唯一或充分原因；
- Geo-REPA、relative pose 或 camera-ray equivariance 已经有效；
- 该 intervention 改善或降低了任务成功率；
- Thought4 改变了 Thought3 的 K=1 held-out utility 负结果；
- Fast-WAM 或世界动作模型普遍不需要 future imagination。

下一项独立研究只能实现预注册的
`Geo-REPA + relative pose / camera-ray equivariance`，先做 held-out
representation/SE(3) 指标，再运行全新、预注册的 Clean/Camera/Lighting paired
rollout。新结果不得覆盖本诊断。
