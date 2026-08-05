# Thought5 运行手册

所有命令从仓库根目录执行。真实运行前先提交当前实现；runner 会拒绝 dirty
worktree。真实 smoke、pilot freeze 与 formal 必须使用同一 project commit；跨
commit 的 partial output 也不能 resume。每个阶段写入独立 namespace，completed
工件不可覆盖。

## 1. 审计与 CPU dry-run

```bash
bash scripts/run_thought5_audit.sh
```

该命令不加载 simulator/GPU 模型。它生成审计和 formal-candidate 的 `NOT RUN`
工件骨架。重复执行会复用相同 config fingerprint 的已完成 dry-run。

快速核对：

```bash
cat outputs/thought5/phase5_cpu_dry_run_v2/dry_run_result.json
cat outputs/thought5/phase5_camera_equivariant_geo_repa_v2/run_status.json
```

预期分别为 `status=complete, scientific_result=false` 和 formal `NOT RUN`。
当前 v2 实测为 1.140 s；它不加载真实 Fast-WAM，因此不是 GPU ETA。

## 2. 单卡真实 smoke

先选一张空闲 24 GiB 卡：

```bash
CONFIRM_THOUGHT5_SMOKE=YES \
THOUGHT5_GPU_IDS=0 \
bash scripts/run_thought5_smoke.sh
```

范围固定为一个 task、2 个 exact-state pair、B1/G3、最多 5 step。它检查真实
hook、gradient、loss 分量、adapter-only checkpoint、frozen SHA、推理头移除、
zero-LoRA 官方动作逐位一致、ray/pose inference、无 future RGB/depth 泄漏、
延迟和峰值显存，不形成科学结论。首次真实运行前没有可信 ETA；按既有模型加载
与已完成 smoke v4 的 712.323 s telemetry，smoke v5 预留 12–15 分钟。日志位于
`outputs/thought5/phase5_camera_equivariant_geo_repa_smoke_v5/logs/run.log`。

2026-08-04 的 smoke v2 是一次无效技术运行：B1 完成 2 step 后，G3 在训练专用
FP32 pose auxiliary head 接收 BF16 注入表征时触发 dtype error。它没有生成
`smoke_result.json`、没有解锁 pilot，也不构成科学结果。v2 目录只读保留。
smoke v3 已在旧的双/四卡执行 commit 上完整通过，三卡调度 commit 的 smoke v4
也已完整通过。pilot v3 随后暴露了只影响 fresh worker 的 LIBERO import launcher
缺陷；修复属于新 commit，因此只允许全新运行 smoke v5。v5 与 v4 的样本
identity、seed、模型、损失和门槛逐字段相同。
若中断且工件校验无误：

```bash
CONFIRM_THOUGHT5_SMOKE=YES \
THOUGHT5_GPU_IDS=0 \
bash scripts/run_thought5_smoke.sh --resume
```

## 3. 2/3 卡 pilot

只有 smoke 的 `run_status.json` 为 complete 后运行：

```bash
CONFIRM_THOUGHT5_PILOT=YES \
THOUGHT5_GPU_IDS=0,1,2 \
bash scripts/run_thought5_pilot.sh
```

pilot 使用单 task 的互斥 8/4/4 episode，比较 B1/G3/G4，并只根据 development
方向以及完整 representation/future-geometry/future-utility/rollout panel 决定
是否冻结 formal recipe。三卡同时运行 B1/G3/G4；后续 utility 与 rollout 也
按同一三卡 wave 调度。两卡兼容模式仍是先 B1/G3、再 G4，不得删除 G4。
pilot success 只能标为 PILOT，不能进入论文正式主表。
若 G3 无方向性改善、G4 同等有效或 Clean objective 明显损害，停止，不生成
formal unlock。三卡预估总耗时 2.5–4.5 小时，细分与推导见
[三卡执行调度预注册](three_gpu_execution_preregistration.md)。日志位于
`outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/logs/run.log`。

旧 pilot v2 仅进入 task-0 render 约 6 秒后被人工中断。pilot v3 完成 16 个 base
state/64 个 condition sample 的缓存后，B1/G3/G4 均在导入 LIBERO 时退出；两者
均为 `status=error`，没有训练或科学结果，必须只读保留，不能传 `--resume` 给
pilot v4。pilot v4 启动时应先出现 `panel_worker_import_preflight_complete`，再开始
render。

## 4. 3/4 卡 formal

只有 pilot 已生成且校验通过
`formal_protocol_frozen.json` 才可运行：

```bash
CONFIRM_THOUGHT5_FORMAL=YES \
THOUGHT5_GPU_IDS=0,1,2 \
bash scripts/run_thought5_formal.sh
```

三卡第一波运行 B1/G1/G2，第二波运行 G3/B0；四卡兼容路径第一波运行
B1/G1/G2/G3，第二波运行 B0。B0 始终只读官方 checkpoint。formal
必须一次性执行冻结的 representation、future geometry、future utility 和 paired
rollout panel，随后才生成机制分类与 report。未产生完整 H1/H2/H3 工件时，
finalizer 必须保持 `NOT RUN` 或 error，不能写 complete。

正式输出目录为
`outputs/thought5/phase5_camera_equivariant_geo_repa_v2/`。每个训练、future-
adapter、representation、future-geometry 与 rollout worker 都先写自己的原子工件；
最后才写 mechanism evidence、15 问 report、execution integrity 与 artifact
manifest。当前三卡容量规划为约 28–45 小时；pilot 完成后必须用实际 worker
`elapsed_s` 和 episode 长度重新计算，但不得据此减少样本、step 或 variant。

## 常见门禁

- GPU used memory >1024 MiB：换空闲窗口，不强行启动。
- worktree dirty：先审查、提交；不要用 reset 丢弃实验代码。
- `MUJOCO_EGL_DEVICE_ID`：runner 自动使用对应 physical GPU。
- partial output：先读 `run_status.json` 和 log；只有 checksum-valid 才传 `--resume`。
- complete output：新协议必须使用新版本目录，禁止覆盖。
- v1 scaffold：只读保留；v2 修复了 Phase 5-B matched probe，禁止 resume v1。
- smoke v2：只读保留的 dtype 失败运行，禁止 resume。
- smoke v3/smoke v4：只读保留的完整技术结果；launcher hotfix commit 必须全新运行 smoke v5。
- pilot v2：只读保留的人工中断运行。
- pilot v3：只读保留的 worker import 失败运行；只写 pilot v4，禁止跨 namespace/commit resume。
