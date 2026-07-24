# 阶段二静态/无动作校准手册

更新日期：2026-07-23

## 1. 为什么需要独立校准

阶段二的 `predicted_motion_energy` 与 `actual_motion_energy` 都定义为：

```text
mean(abs(last independently re-encoded VAE embedding
       - first independently re-encoded VAE embedding))
```

它的单位不是像素、光流、动作幅度或 native temporal latent。首版
`static_motion_threshold=1.0` 只是 schema 初值；20-step pilot 的预测/实际
energy 已达到 `0.217–0.259` / `0.102–0.217`，却仍被全部判为 static。
因此旧 static flag 没有科学解释资格。

阈值不能从这 7 个已看过 success/OOD 标签的 probe 反向选择。本协议建立第三个
完全独立的输出空间，只运行标准 no-op，不调用 `policy.act()`，也不读取
future pilot 的成功/失败标签。

## 2. 固定协议

每个 calibration job：

1. 用独立 task/seed reset 环境，并调用 `policy.reset()` 只初始化官方预处理状态。
2. 执行与标准评测相同的 30 个 settle no-op：
   `[0, 0, 0, 0, 0, 0, -1]`。
3. 在额外 no-op 的 offset `0/4/8` 保存官方双相机 model frame。
4. 对 offset-0 完全相同的帧独立编码 3 次，记录每个 job 的最大 pairwise
   embedding energy，测编码器噪声。
5. 独立编码 `0/4/8` 轨迹，记录 no-op residual motion 和像素 MAE。
6. 如果 no-op 期间环境提前 success/done，则样本标为 excluded，不进入阈值。

候选阈值定义为：

```text
max(
  higher_quantile_0.99(per-job same-frame max pairwise energy),
  higher_quantile_0.99(no-op offset-0 to offset-8 energy)
)
```

`higher` 是保守经验分位法；小样本时不会插值到观测最大值以下。阈值只对应
逐帧独立重编码 embedding。它不会自动覆盖原 diagnostics JSONL，也不会因为
达到样本数就自动变成 frozen；仍需人工审核。

## 3. 三个输出空间互斥

```text
outputs/clean_* / ood_*                    # 阶段一 episode results
outputs/thought2_unconditional_*           # 阶段二 future diagnostics
outputs/thought2_static_calibration_*       # 本校准 raw samples
outputs/thought2_static_*_comparison        # 只读派生聚合
```

`calibrate-static` 会拒绝阶段一或 future-diagnostic 目录及其父子目录。
`plan/evaluate/distributed-evaluate` 也会拒绝
`static_calibration.enabled=true`。每个 job 只写：

```text
calibration_manifest.json
static_calibration_job_manifest.jsonl
workers/rank_N/static_calibration_samples.jsonl
workers/rank_N/completed_jobs.jsonl
workers/rank_N/artifacts/<job_id>/offset_000.png
workers/rank_N/artifacts/<job_id>/offset_004.png
workers/rank_N/artifacts/<job_id>/offset_008.png
```

manifest 固定 checkpoint/config/stats hash、Fast-WAM commit、编码语义、
no-op action、offset、重复编码数、阈值规则和相关实现文件 hash。Clean/OOD
只有 `compatibility_fingerprint` 相同才可合并。

## 4. 命令与断点续跑

PILOT-v1 两个 raw 目录已经完整结束并保持只读。采样后协议新增了
`threshold_quantile_method=higher`，所以当前代码有意拒绝把 v2 续写进 v1。
验证当前协议时先指定全新的 experiment/output：

```bash
fastwam-ood calibrate-static \
  --config configs/studies/thought2_static_calibration_clean.yaml \
  --set experiment.name=thought2_static_calibration_clean_v2 \
  --set experiment.output_dir=outputs/thought2_static_calibration_clean_v2 \
  --dry-run
```

物理 GPU 1 的单卡运行：

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
fastwam-ood calibrate-static \
  --config configs/studies/thought2_static_calibration_clean.yaml \
  --set experiment.name=thought2_static_calibration_clean_v2 \
  --set experiment.output_dir=outputs/thought2_static_calibration_clean_v2 \
  --device cuda:0
```

OOD 使用另一进程运行对应配置，避免同名 `libero` package 混用：

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
fastwam-ood calibrate-static \
  --config configs/studies/thought2_static_calibration_ood.yaml \
  --set experiment.name=thought2_static_calibration_ood_v2 \
  --set experiment.output_dir=outputs/thought2_static_calibration_ood_v2 \
  --device cuda:0
```

对 v2 或后续 FORMAL，正常中断后重复同一命令即可跨 rank resume；
`--rerun failed` 只处理异常记录。
协议、配置或实现 hash 改变时，dry-run 与真实运行都会拒绝复用旧目录。此时必须
改 `experiment.name/output_dir`，不要用 `--overwrite` 混写科学实验。

Clean/OOD 合并并对既有 future pilot 做只读敏感性分析：

```bash
fastwam-ood aggregate-static-calibration \
  --experiment-dir outputs/thought2_static_calibration_pilot_comparison \
  --input-dir outputs/thought2_static_calibration_clean \
  --input-dir outputs/thought2_static_calibration_ood \
  --diagnostic-dir outputs/thought2_unconditional_clean \
  --diagnostic-dir outputs/thought2_unconditional_ood

fastwam-ood report-static-calibration \
  --experiment-dir outputs/thought2_static_calibration_pilot_comparison
```

派生 sensitivity 文件记录所有输入 manifest SHA-256，并显式写
calibration/diagnostic JSONL SHA-256；同时显式写
`source_rows_rewritten=false`。缺失 success 标签的历史 probe 单独计数，
绝不隐式归入 failure 组。

## 5. 2026-07-23 PILOT-v1 真实结果

两组使用同一官方 checkpoint：
`1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。
Fast-WAM/LIBERO/LIBERO-Plus commit 分别为
`45d8e145` / `8f1084e3` / `4976dc30`，三个上游均 clean。项目 HEAD 为
`b3c1be8`，但实现尚未提交，`git_dirty=true`，所以证据等级只能是 PILOT。

Clean/OOD protocol fingerprint 分别为
`d0a3b3db6ae8a1b3f80db0c5fe51c078d2e24db959db18bd76cd99bc01298ea7`
和
`6ec8d51045f0a14df3a18347c8231d5eceb92d5b3a00502a97f8ab8228a3db55`；
共同 compatibility fingerprint 为
`9981dd18a609fc0e28899916b3ee74f9b25495bd81013f7fe60ef10ef17bf072`。

| Condition | Task/category | same-frame max | no-op 0→8 | pixel MAE 0→8 | 状态 |
|---|---|---:|---:|---:|---|
| Clean | task 1 | 0 | 0.01148672 | 0.00050602 | eligible |
| Clean | task 6 | 0 | 0.00829949 | 0.00027833 | eligible |
| OOD | camera | 0 | 0.00642072 | 0.00020116 | eligible |
| OOD | lighting | 0 | 0.00661479 | 0.00013788 | eligible |
| OOD | background | 0 | 0.00580508 | 0.00024179 | eligible |
| OOD | robot init | 0 | 0.01322303 | 0.00076839 | eligible |
| OOD | object layout | 0 | 0.00404918 | 0.00009902 | eligible |

验收：

- 7/7 completed、7/7 eligible、0 exception/excluded/skipped。
- 运行时 control frequency 均为 20 Hz；model frame 均为 `224×448×3`。
- 21 张 offset PNG 全部可解码；contact sheet 目检没有黑帧、错相机或损坏。
- 同帧重复编码噪声 7/7 为 0。
- no-op energy：offset-4 中位数/最大值 `0.00488984/0.01134326`；
  offset-8 为 `0.00661479/0.01322303`。
- offset-8 Clean 中位数 `0.00989311`（n=2），OOD 中位数 `0.00642072`
  （n=5）；样本太小且 task 不同，不能解释为 Clean/OOD 差异。
- 视频组件冷启动 Clean/OOD 分别约 `518.15/521.80 s`；第一条 VAE 编码有
  warm-up，后续每组三帧编码约 `54–86 ms`。冷启动不进入阈值。

## 6. 候选阈值与 pilot 敏感性

7 条 null 的同帧分位数为 0，no-op 99% `higher` 值为观测最大值
`0.0132230342`，因此候选阈值为：

```text
candidate_static_motion_threshold = 0.0132230342
status = candidate_only
```

它不能冻结，原因有两层：

1. 只有 7/200 个有效样本；Clean 2/100、OOD 5/100，五个 OOD 类别各
   1/20。
2. PILOT-v1 采样前固定了 99% 分位数，但没有把 `higher` 插值法写进 source
   manifest。当前代码/配置已补上该字段；任何 v2/FORMAL 运行必须用新目录。

只读重分类结果：

| 口径 | predicted static | actual static |
|---|---:|---:|
| 旧阈值 1.0 | 7/7 | 7/7 |
| 候选阈值 0.013223 | 0/7 | 0/7 |

pilot 中最小 predicted/actual energy 仍分别是候选阈值的约 `16.41×/7.70×`。
这只证明旧阈值量纲明显不合适，并提供 formal 阈值的数量级假设；不能据此声称
7 条 future 都“正确地预测了运动”，更不能改变 L1/cosine/direction 的小样本
解释边界。

权威工件：

- `outputs/thought2_static_calibration_clean/calibration_manifest.json`
- `outputs/thought2_static_calibration_ood/calibration_manifest.json`
- `outputs/thought2_static_calibration_pilot_comparison/summary/static_calibration_summary.json`
- `outputs/thought2_static_calibration_pilot_comparison/summary/static_threshold_sensitivity.json`
- `outputs/thought2_static_calibration_pilot_comparison/summary/static_calibration_report.md`

## 7. FORMAL 冻结门槛

正式阈值至少需要：

- 200 个全部有效的独立 null job；
- Clean ≥100、OOD ≥100；
- 五个 OOD 类别各 ≥20；
- 0 个未解释 exception/excluded/skipped；
- 99% `higher` 方法在采样前进入 manifest；
- 项目和三个上游均 `dirty=false`；
- 所有有效样本显式记录相同的运行时 control frequency 和 model-frame shape；
- 每条样本完整执行预注册 settle/capture，且 raw manifest/JSONL hash 可追溯；
- 盲目检预注册子集后，由人工确认冻结版本。

已提供并通过只读 plan 审计的配置：

- `thought2_static_calibration_formal_clean.yaml`：10 个 task × 10
  init-state/seed = 100，0 skipped；
- `thought2_static_calibration_formal_ood.yaml`：5 个无 easy 空分层的 base
  task × 5 类 × 4 init-state/variant selection = 100，五类各 20，0 skipped。

两份 dry-run 均为 `assigned=100, pending=100`，且没有创建输出目录。OOD 的小
variant pool 是否发生有放回选择会逐行保留在
`perturbation_parameters.selection_with_replacement`；正式报告不能把重复
variant 冒充 100 个不同外观。

达到这些条件只会得到 `eligible_for_manual_freeze`，不会自动修改诊断配置。
人工冻结后应以新 output/protocol fingerprint 重跑或重新聚合正式 diagnostics，
同时保留原始连续 energy，避免二值 flag 成为唯一结果。

## 8. 可用于简历/面试的工程事实

可以陈述：

- 设计了与控制评测和 future pilot 隔离的 no-op calibration protocol，使用
  checkpoint/实现 hash、原子 JSONL 和跨 rank resume 防止实验混写。
- 在真实 Fast-WAM/LIBERO(-Plus) 上完成 7 条跨 Clean/五类 OOD 的 null
  pilot，验证旧阈值比候选数量级高约 75.6 倍，并以只读派生文件重分类历史
  probe。
- 将样本数、condition/category 覆盖和预注册 quantile method 编码为自动
  freeze gate，防止小样本 candidate 被误写成 paper 结论。

不能把 `0.013223` 写成最终阈值，也不能把 0/7 static 写成模型能力提升。
