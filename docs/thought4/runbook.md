# Thought4 运行手册

所有命令从项目根目录执行。真实执行只支持一张空闲 24 GiB 卡；
`THOUGHT4_GPU_ID` 是物理编号，进程内 Fast-WAM 使用重映射后的 `cuda:0`。

## 1. 当前冻结身份

当前协议是 FP32 subspace arithmetic formal v6。它是在 simulator-replay formal
v5 的 BF16 reconstruction 工程失败后登记的数值修复，不是根据 probe、policy、
success 或 OOD 结果选择的。完整预注册见
[formal v6 FP32 修复预注册](fp32_subspace_v6_preregistration.md)。

| 项目 | 值 |
| --- | --- |
| smoke config | `configs/thought4/phase4_geometry_action_smoke_v8.yaml` |
| smoke output | `outputs/thought4/phase4_geometry_action_smoke_v8/` |
| smoke config fingerprint | `81d3885ccb5b58806c1a729e509c039f6e1cb33a34ff242f8fa16785796149d7` |
| smoke planned cohort SHA | `a67ff85321dc684a80b853b58ab133905232a275e6d71255fd1c966b9a3d6c12` |
| formal config | `configs/thought4/phase4_geometry_action_diagnosis_v6.yaml` |
| formal output | `outputs/thought4/phase4_geometry_action_diagnosis_v6/` |
| formal config fingerprint | `3b14a7d7fd09deda9253bb1cd9950d9c4b5bd0cdf9f124a4dfede22add5c24f6` |
| formal planned cohort SHA | `9af7cf7c1933fb1e5574099361f6d7dcc7500727480ecb4bbf010089f28d8f04` |
| subspace arithmetic | FP32 coordinates/residual，单次 BF16 output cast，bitwise correct |
| label source | `simulator_action_replay_from_input_t` |
| alignment policy | disclosure-only，仍使用 3 cm / 15° |
| formal cohort | 40/12/12 states，20/6/6 episodes；原 v4 state identity 全保留 |

planned cohort SHA 包含 config fingerprint，因此 v5→v6 会变化；底层 64 个
episode/frame/split/sample identity SHA 仍为
`9916e0444ccfca08bd3d87ead73c344974f733f9b0c5dcc5cd9996a2618f8e8b`。
禁止用新 SHA 误称为重新抽样。

## 2. CPU/read-only dry-run

```bash
.conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought4-phase4-smoke \
  --config configs/thought4/phase4_geometry_action_smoke_v8.yaml \
  --dry-run

.conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought4-phase4-diagnosis \
  --config configs/thought4/phase4_geometry_action_diagnosis_v6.yaml \
  --dry-run
```

必须看到：

```text
would_load_torch=false
would_load_gpu_model=false
would_construct_simulator=false
would_write=false
trajectory_label_source=simulator_action_replay_from_input_t
demonstration_alignment_policy=disclosure_only_3cm_15deg
subspace_arithmetic=fp32_coordinates_residual_single_output_cast_bitwise_correct
```

dry-run 不会替代真实 smoke。

## 3. 为什么不能直接 resume formal v4

smoke v6 已于 2026-08-01 10:07:40–10:18:42 UTC 在 project commit
`aeb02106c48389d49bd7cac693e68113fa7d245a`、物理 GPU 2 上通过：2 个 base
states × 4 conditions、80 feature records，模型加载 409.889 s，主干 SHA
前后均为 `ac0dd59...b4f8`，identity replacement action L2=0，result SHA
`b260977ae826e8c860074bd3402a3914dbc52e3f887cc090dad5ff3be2bc4c37`。
它是有效的非科学工程 Gate。

formal v4 随后于 10:21:25 UTC 启动，但在第 2 个排序状态
`episode_000031@t34`（development）停止：Clean prefix 与 parquet EEF 的误差为
0.031214 m / 2.153°，超过 3 cm / 15° 中的平移阈值。模型尚未加载，没有
paired manifest、feature、probe、intervention 或科学结果。

对全部原 64 states 的只读 simulator 审计显示：

| 项目 | 结果 |
| --- | ---: |
| 通过 / 失败 | 56 / 8 |
| train / development / test 失败 | 5 / 3 / 0 |
| translation mean / median / p90 / p95 / max | 0.023736 / 0.018507 / 0.031168 / 0.060026 / 0.108324 m |
| rotation mean / median / p90 / p95 / max | 3.079 / 2.073 / 4.056 / 8.792 / 28.918° |

因此不能把 v4 当作“只超了 1.2 mm”的单例，也不能放宽阈值、删 8 条、换样本或
选择 step offset。simulator-replay v5 改变了标签生成代码与 config identity，旧
smoke v6 不能解锁它；必须跑全新 smoke v7，formal v4 目录保持原样。

smoke v7 已于 2026-08-03 02:32:15–02:43:10 UTC 通过；formal v5 随后完成
256 条 paired render/label、12,544 条 feature 和内存中的两组 probe panel，但在
04:13:40 UTC 的首次 subspace intervention 前因 BF16 reconstruction 停止。v5
没有科学 classification，且旧编排未在 intervention 前落盘 probe results。当前
不得 resume v5；只运行下述新身份。

## 4. 运行 smoke v8

先确认代码已提交、worktree clean、卡空闲：

```bash
git status --short
nvidia-smi
```

`git status --short` 必须没有输出。然后执行：

```bash
CONFIRM_THOUGHT4_PHASE4_SMOKE=YES \
THOUGHT4_GPU_ID=2 \
bash scripts/run_thought4_phase4_smoke.sh
```

runner 内部应为：

```text
CUDA_VISIBLE_DEVICES=2
MUJOCO_EGL_DEVICE_ID=2
Fast-WAM device=cuda:0
```

v7 实测 10m55s。v8 多一次 reconstructed-cache Action inference，预计约 12–17
分钟；这是运行预算，不是论文 latency。

查看：

```bash
tail -f outputs/thought4/phase4_geometry_action_smoke_v8/logs/run.log
cat outputs/thought4/phase4_geometry_action_smoke_v8/run_status.json
cat outputs/thought4/phase4_geometry_action_smoke_v8/alignment_audit.json
cat outputs/thought4/phase4_geometry_action_smoke_v8/smoke_result.json
```

smoke v8 只取 2 个 base state，覆盖四条件，验证：

- 从真实 input state `t` 的 simulator action replay；
- exact-state 三条件共享同一 Clean world trajectory；
- Robot-init 使用 condition-specific trajectory；
- alignment pass/fail 完整披露且不筛样本；
- Video layer 15 hidden/K/V 与 Action block 15 真实 hook；
- feature shard/checksum、probe backward、identity replacement/replay parity；
- 真实 BF16 capture 经 FP32 subspace reconstruction 和单次 BF16 cast 后逐位恢复；
- reconstructed tensor 真正 replacement 到 Action consumer，且动作通过 replay gate；
- Fast-WAM backbone SHA 前后不变；future RGB/success 均未读取。

进入 formal 的必要条件包括：`status=complete`、`formal_unlocked=true`、四条件
完整、Robot-init input state 2/2 区别于 Clean、主干 SHA 前后相等，并且
`alignment_audit.json` 的实体内容、canonical SHA、计数、来源与 trajectory pairing
均通过 gate。新增检查还要求 input/output tensor SHA 相同、max-abs=0、
`bitwise_equal_after_output_cast=true` 和 subspace contract SHA 有效。只编辑
`smoke_result.json` 或只设置 formal 确认变量不能绕过。

## 5. 运行 formal v6

smoke v8 通过后，不改代码、配置或文档；同一 project commit 直接执行：

```bash
CONFIRM_THOUGHT4_PHASE4_FORMAL=YES \
THOUGHT4_GPU_ID=2 \
bash scripts/run_thought4_phase4_diagnosis.sh
```

预计单卡 4090 为 2–6 小时，正式 ETA 应在 smoke v8 后根据真实计时更新。不能为了
缩短时间改变 64 states、五层、probe seeds、threshold 或 20-step action schedule。

主要输出：

```text
outputs/thought4/phase4_geometry_action_diagnosis_v6/
  alignment_audit.json
  cohort_manifest.json
  paired_render_manifest.jsonl
  label_manifest.jsonl
  feature_manifest.jsonl
  video_probe_results.json
  action_probe_results.json
  layer_summary.json
  probe_stage_result.json
  intervention_results.json
  diagnostic_evidence.json
  method_selection.json
  execution_integrity.json
  artifact_manifest.json
  report.md
```

formal 不启动 policy rollout，不读取 success/OOD，也不实现新训练方法。

## 6. Resume 与历史 namespace

v1–v8 smoke、v1–v6 formal 均使用独立 namespace。历史失败/中断/通过工件不得
删除、覆盖、拼接到当前结果或改写为 PASS。

formal v5 是冻结的工程失败证据，明确禁止 `--resume`。formal v6 只在其自身
namespace 内、同 commit 且 checksum-identical 的非科学进程中断时允许 resume；
不得把 v5 的 `probe_inputs.pt` 或 feature shard 复制到 v6。

只有代码 commit、配置、pre-validation identity 均未变化，且日志明确是非科学性
进程中断时，才可在当前 runner 的内部 CLI 后加 `--resume`。已有 feature shard 和
`probe_inputs.pt` 必须同时通过 sidecar SHA、逐 tensor SHA 与 metadata 校验；任何
差异 fail closed。`status=complete` 的目录不可 resume。

## 7. 测试

```bash
.conda/envs/fastwam-ood/bin/python -m pytest -q tests/test_thought4_*.py
.conda/envs/fastwam-ood/bin/python -m pytest -q
python scripts/check_docs.py
```

## 8. 禁止操作

- 不在 YAML 外调层、样本、seed、threshold 或 label source；
- 不放宽 3 cm / 15°，不过滤/替换 alignment failed states；
- 不把 Robot-init 写成 exact-state；
- 不让三种 exact-state condition 分别 replay future trajectory；
- 不读取 future RGB、reward/success/OOD 来决定样本、层或方法；
- 不用旧 smoke v7 解锁 formal v6；
- 不放宽 correct reconstruction 阈值；correct tensor 必须逐位恢复；
- 不覆盖或 resume formal v5，不复制其工件到 formal v6；
- 不把 smoke 或 alignment QC 写成论文科学结论；
- 不在 Phase 4 直接实现 Geo-REPA/SE(3)-Align。
