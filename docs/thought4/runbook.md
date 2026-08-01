# Thought4 运行手册

所有命令从项目根目录执行。真实执行只支持一张空闲 24 GiB 卡；`GPU_ID` 是物理
编号，进程内映射为 `cuda:0`。

## 1. CPU/read-only dry-run（已运行）

```bash
.conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought4-phase4-smoke \
  --config configs/thought4/phase4_geometry_action_smoke_v4.yaml \
  --dry-run

.conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought4-phase4-diagnosis \
  --config configs/thought4/phase4_geometry_action_diagnosis_v2.yaml \
  --dry-run
```

判据：`would_load_torch/model/simulator/write=false`。当前冻结输出为：

- smoke v4 fingerprint `204ced0d56e038bbbe30fe14119ab48daa920cf52ea81af39f9dbc86c799a7bc`，
  planned cohort SHA
  `5aea296c67dafe5118baec380d057d7a0e485a34f2930283de439d58ef7d2096`；
  plan 候选 2/2/2，但技术执行只取 2 个 base state；每个 state 四条件；
- formal v2 fingerprint
  `cc60895325cbe5f3c7d7870c926fe82893f0c0cb5fa555d4e9f72c61fef46f21`，
  planned cohort SHA
  `e3a6363f100910b63035821101cf8e153a753e47c1196198042ce47097b275ca`；
  40/12/12 共 64 个 base state、20/6/6 个 episode，每个 state 四条件。

## 2. 真实 smoke（v3 已通过；带 Robot-init 的 v4 当前 NOT RUN）

2026-07-31 的 v1 attempt 在 robosuite import 阶段停止：runner 暴露物理 GPU 1，
却将 `MUJOCO_EGL_DEVICE_ID` 设为逻辑 0。模型未加载、环境未 reset、没有 feature、
action、probe 或科学结果。失败目录
`outputs/thought4/phase4_geometry_action_smoke_v1/` 原样保留，禁止 resume 或覆盖。

2026-07-31 的 v2 attempt 已通过 EGL 映射，完成 2 个 base state × 3 conditions
的 6 条 paired render 与 6 条 label，并开始加载模型；首次 source-A K/V feature
capture 时，hook 读取 `torch.inference_mode()` tensor 的 `_version`，触发
`Inference tensors do not track version counter`。因此没有 feature shard、action、
probe、identity replacement 或 `smoke_result.json`，也没有科学结果。失败目录
`outputs/thought4/phase4_geometry_action_smoke_v2/` 同样原样保留，禁止 resume 或
覆盖。

v3 只修复 inference-tensor cache identity：普通 tensor 继续校验 version counter；
inference tensor 校验 data pointer、shape、dtype、device、stride，并在 scope 结束时
逐值对比首次 clone，仍然 fail closed。同时修复 gripper scalar 的 NumPy 弃用警告。
v3 已在 project commit `fd713d580303...`、物理 GPU 2 上于
2026-07-31 11:16:46–11:27:19 UTC 完成：2 base states × 3 conditions、
60 features、主干 SHA 前后均为 `ac0dd59...b4f8`，identity replacement
action L2=0，result SHA `b0e1cc80...17dbb`。它是有效工程 smoke，但不是科学
结果，也没有覆盖 Robot-init。

随后 formal v1 在第一个 Robot-init reset 后错误地要求 robot observation 立即
区别于 Clean，于模型加载前停止。LIBERO 的 `set_init_state()` 会写入共同的
demonstration flat state，因此 reset 时最大可观测 robot-state 差为 0；执行共同
action prefix 到输入时刻后，同一诊断样本的差为 0.08357，扰动实际已生效。
formal v1 只生成 pre-validation、planned manifest 和 error status，无 paired
render、feature、probe、intervention 或科学结果。

v4 把硬检查移到模型输入时刻 `t`，并把 Robot-init 加入 smoke。formal v2 与 v1
的 cohort、层、probe、seed、阈值和 20-step action schedule 完全相同，只使用新
namespace 和更正后的验证语义。旧 smoke v3 因缺少 Robot-init 证据，不能解锁
formal v2。

真实运行只接受已提交且 `git status --porcelain` 为空的项目快照；当前实现应先由
用户审阅并提交。pre-validation 会把 project commit/clean status 与三套上游
commit 一起写入工件。runner 在创建日志前也会拒绝 dirty worktree 和已经
`complete` 的输出，避免日志破坏 completed run 的不可变性。
smoke v4 与 formal v2 必须来自同一个 project commit：先提交修复，再跑 smoke；
smoke 通过后不要改代码或文档，直接启动 formal。

先确认卡空闲：

```bash
nvidia-smi
```

执行：

```bash
CONFIRM_THOUGHT4_PHASE4_SMOKE=YES \
THOUGHT4_GPU_ID=2 \
bash scripts/run_thought4_phase4_smoke.sh
```

v3 实测为 10m33s，其中模型加载 401.7s；v4 只多 2 条 Robot-init inference，
预计约 11–15 分钟。该估计不是论文 latency 结果。

runner 内部映射必须是：

```text
CUDA_VISIBLE_DEVICES=2
MUJOCO_EGL_DEVICE_ID=2   # robosuite 使用物理 ID
Fast-WAM device=cuda:0   # PyTorch 使用重映射后的逻辑 ID
```

Python preflight 会在创建新 run 之前核对三者；不能手工把 EGL 改回 `0`。

smoke 从冻结 plan 只取 2 个 base state，渲染
Clean/Camera/Lighting/Robot-init 四个
condition，验证一个 Video layer（含 hidden/K/V）和一个 Action layer 的真实
hook call、shape、显式 denoise-step identity、feature shard/checksum、probe
backward、source-A identity replacement/replay parity 与主干 SHA。它不训练正式
probe，不输出方法结论，不是科学结果。identity replacement 的固定边界是
`mot.video_kv_cache.15.v` 的 Action-consumer argument；formal 也只从最终
action-consumed K/V 中选择干预层。

查看：

```bash
tail -f outputs/thought4/phase4_geometry_action_smoke_v4/logs/run.log
cat outputs/thought4/phase4_geometry_action_smoke_v4/run_status.json
cat outputs/thought4/phase4_geometry_action_smoke_v4/smoke_result.json
```

只有 `status=complete`、`formal_unlocked=true`、Robot-init 的
`input_differs_clean_count=2`、前后 backbone SHA 相等才进入 formal。formal
runner 会再次验证 smoke status/stage/config fingerprint/result SHA、四条件覆盖、
Robot-init 输入状态、identity replacement 和前后主干 SHA；只设置 formal 确认
变量不能绕过。
失败后先保留目录排查；`--resume` 只接受未完成且 checksum 可解释的工件，绝不
覆盖 completed 输出。

v1/v2/v3/v4 都使用独立 namespace；本次不能给旧输出加 `--resume`。以后只有
代码、配置、pre-validation identity 均未变化且日志明确表明是
非科学性的进程中断，才可在同一命令末尾加 `--resume`。已有
feature shard 和 `probe_inputs.pt` 必须同时通过 sidecar SHA、逐 tensor SHA、
metadata 与本次冻结提取一致，才会只读复用；任何差异均 fail closed。

## 3. 正式 diagnosis（v1 工程失败；v2 当前 NOT RUN）

smoke 通过并人工检查 manifest 后：

```bash
CONFIRM_THOUGHT4_PHASE4_FORMAL=YES \
THOUGHT4_GPU_ID=2 \
bash scripts/run_thought4_phase4_diagnosis.sh
```

这是唯一 formal 命令。它不启动 success rollout，也不实现新训练方法。

粗略资源预算（尚未实测，不能作为结果）：单卡 4090，模型加载约 6–7 分钟；
64×4 次特征推理、CPU probe panel 和 held-out intervention 合计预计 2–6 小时。
以真实 smoke 的单次推理/渲染计时更新 ETA，不为了缩短时间修改冻结样本、层或
20-step action schedule。

主要结果：

```text
outputs/thought4/phase4_geometry_action_diagnosis_v2/
  cohort_manifest.json
  paired_render_manifest.jsonl
  label_manifest.jsonl
  feature_manifest.jsonl
  video_probe_results.json
  action_probe_results.json
  layer_summary.json
  intervention_results.json
  diagnostic_evidence.json
  method_selection.json
  execution_integrity.json
  artifact_manifest.json
  report.md
```

## 4. 测试

```bash
.conda/envs/fastwam-ood/bin/python -m pytest -q tests/test_thought4_*.py
.conda/envs/fastwam-ood/bin/python -m pytest -q
python scripts/check_docs.py
```

## 5. 禁止操作

- 不在 YAML 之外用 `--set` 调层、样本、seed 或 threshold；
- 不用两张卡并行不同 protocol；
- 不读取 success 来决定 sample/layer；
- 不把 Robot-init 写成 exact-state；
- 不把 smoke 写进论文主表；
- 不在 Phase 4 直接实现 Geo-REPA/SE(3)-Align。
