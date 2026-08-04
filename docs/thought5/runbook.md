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
与小步训练 telemetry，建议预留 20–40 分钟，并以 log 实际进度为准。日志位于
`outputs/thought5/phase5_camera_equivariant_geo_repa_smoke_v2/logs/run.log`。
若中断且工件校验无误：

```bash
CONFIRM_THOUGHT5_SMOKE=YES \
THOUGHT5_GPU_IDS=0 \
bash scripts/run_thought5_smoke.sh --resume
```

## 3. 双卡 pilot

只有 smoke 的 `run_status.json` 为 complete 后运行：

```bash
CONFIRM_THOUGHT5_PILOT=YES \
THOUGHT5_GPU_IDS=0,1 \
bash scripts/run_thought5_pilot.sh
```

pilot 使用单 task 的互斥 8/4/4 episode，比较 B1/G3/G4，并只根据 development
方向以及完整 representation/future-geometry/future-utility/rollout panel 决定
是否冻结 formal recipe。两卡先并行 B1/G3，再运行 G4；后续 utility 与 rollout
也按双卡 wave 调度。pilot success 只能标为 PILOT，不能进入论文正式主表。
若 G3 无方向性改善、G4 同等有效或 Clean objective 明显损害，停止，不生成
formal unlock。该阶段预计是小时级，首轮 throughput 会写入结果，当前不伪造 ETA。

## 4. 四卡 formal

只有 pilot 已生成且校验通过
`formal_protocol_frozen.json` 才可运行：

```bash
CONFIRM_THOUGHT5_FORMAL=YES \
THOUGHT5_GPU_IDS=0,1,2,3 \
bash scripts/run_thought5_formal.sh
```

四张卡对应独立 matched track B1/G1/G2/G3；B0 只读官方 checkpoint。formal
必须一次性执行冻结的 representation、future geometry、future utility 和 paired
rollout panel，随后才生成机制分类与 report。未产生完整 H1/H2/H3 工件时，
finalizer 必须保持 `NOT RUN` 或 error，不能写 complete。

正式输出目录为
`outputs/thought5/phase5_camera_equivariant_geo_repa_v2/`。每个训练、future-
adapter、representation、future-geometry 与 rollout worker 都先写自己的原子工件；
最后才写 mechanism evidence、15 问 report、execution integrity 与 artifact
manifest。formal 是多小时任务，必须根据 pilot 实测吞吐再估算，不能沿用 CPU
dry-run 数字。

## 常见门禁

- GPU used memory >1024 MiB：换空闲窗口，不强行启动。
- worktree dirty：先审查、提交；不要用 reset 丢弃实验代码。
- `MUJOCO_EGL_DEVICE_ID`：runner 自动使用对应 physical GPU。
- partial output：先读 `run_status.json` 和 log；只有 checksum-valid 才传 `--resume`。
- complete output：新协议必须使用新版本目录，禁止覆盖。
- v1 scaffold：只读保留；v2 修复了 Phase 5-B matched probe，禁止 resume v1。
