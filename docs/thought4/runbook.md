# Thought4 运行手册

所有命令从项目根目录执行。真实执行只支持一张空闲 24 GiB 卡；`GPU_ID` 是物理
编号，进程内映射为 `cuda:0`。

## 1. CPU/read-only dry-run（已运行）

```bash
.conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought4-phase4-smoke \
  --config configs/thought4/phase4_geometry_action_smoke_v2.yaml \
  --dry-run

.conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought4-phase4-diagnosis \
  --config configs/thought4/phase4_geometry_action_diagnosis_v1.yaml \
  --dry-run
```

判据：`would_load_torch/model/simulator/write=false`。当前冻结输出为：

- smoke v2 fingerprint `9c412c1ce2cd...`，planned cohort SHA
  `8055511bdab1...`；plan 候选 2/2/2，但技术执行只取 2 个 base state；
  每个 state 三条件、Video layer 15、Action block 15；
- formal fingerprint `62951df5bb36...`，planned cohort SHA
  `340db6c1a151...`；40/12/12 共 64 个 base state、20/6/6 个 episode，
  每个 state 四条件。

## 2. 真实 smoke（v1 工程失败；有效 v2 当前 NOT RUN）

2026-07-31 的 v1 attempt 在 robosuite import 阶段停止：runner 暴露物理 GPU 1，
却将 `MUJOCO_EGL_DEVICE_ID` 设为逻辑 0。模型未加载、环境未 reset、没有 feature、
action、probe 或科学结果。失败目录
`outputs/thought4/phase4_geometry_action_smoke_v1/` 原样保留，禁止 resume 或覆盖。
v2 只修复设备映射和增加 preflight，不改变 cohort、层、probe 或干预协议。

真实运行只接受已提交且 `git status --porcelain` 为空的项目快照；当前实现应先由
用户审阅并提交。pre-validation 会把 project commit/clean status 与三套上游
commit 一起写入工件。runner 在创建日志前也会拒绝 dirty worktree 和已经
`complete` 的输出，避免日志破坏 completed run 的不可变性。

先确认卡空闲：

```bash
nvidia-smi
```

执行：

```bash
CONFIRM_THOUGHT4_PHASE4_SMOKE=YES \
THOUGHT4_GPU_ID=1 \
bash scripts/run_thought4_phase4_smoke.sh
```

runner 内部映射必须是：

```text
CUDA_VISIBLE_DEVICES=1
MUJOCO_EGL_DEVICE_ID=1   # robosuite 使用物理 ID
Fast-WAM device=cuda:0   # PyTorch 使用重映射后的逻辑 ID
```

Python preflight 会在创建新 run 之前核对三者；不能手工把 EGL 改回 `0`。

smoke 从冻结 plan 只取 2 个 base state，渲染 Clean/Camera/Lighting 三个
condition，验证一个 Video layer（含 hidden/K/V）和一个 Action layer 的真实
hook call、shape、显式 denoise-step identity、feature shard/checksum、probe
backward、source-A identity replacement/replay parity 与主干 SHA。它不训练正式
probe，不输出方法结论，不是科学结果。identity replacement 的固定边界是
`mot.video_kv_cache.15.v` 的 Action-consumer argument；formal 也只从最终
action-consumed K/V 中选择干预层。

查看：

```bash
tail -f outputs/thought4/phase4_geometry_action_smoke_v2/logs/run.log
cat outputs/thought4/phase4_geometry_action_smoke_v2/run_status.json
cat outputs/thought4/phase4_geometry_action_smoke_v2/smoke_result.json
```

只有 `status=complete`、`formal_unlocked=true`、前后 backbone SHA 相等才进入
formal。formal runner 会再次验证 smoke status/stage/config fingerprint/result
SHA、identity replacement 和前后主干 SHA；只设置 formal 确认变量不能绕过。
失败后先保留目录排查；`--resume` 只接受未完成且 checksum 可解释的工件，绝不
覆盖 completed 输出。

若日志明确表明是非科学性的进程中断，可在同一命令末尾加 `--resume`。已有
feature shard 和 `probe_inputs.pt` 必须同时通过 sidecar SHA、逐 tensor SHA、
metadata 与本次冻结提取一致，才会只读复用；任何差异均 fail closed。

## 3. 正式 diagnosis（当前 NOT RUN）

smoke 通过并人工检查 manifest 后：

```bash
CONFIRM_THOUGHT4_PHASE4_FORMAL=YES \
THOUGHT4_GPU_ID=1 \
bash scripts/run_thought4_phase4_diagnosis.sh
```

这是唯一 formal 命令。它不启动 success rollout，也不实现新训练方法。

粗略资源预算（尚未实测，不能作为结果）：单卡 4090，模型加载约 6–7 分钟；
64×4 次特征推理、CPU probe panel 和 held-out intervention 合计预计数小时。
以真实 smoke 的单次推理/渲染计时更新 ETA，不为了缩短时间修改冻结样本、层或
20-step action schedule。

主要结果：

```text
outputs/thought4/phase4_geometry_action_diagnosis_v1/
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
