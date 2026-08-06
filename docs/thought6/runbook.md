# Phase 6 运行手册

## 1. CPU 审计与 dry-run

```bash
bash scripts/run_thought6_audit.sh

PYTHONPATH=src .conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought6-dry-run --config configs/thought6/phase6_audit.yaml
```

审计预计少于 1 分钟，dry-run 约 5 秒；二者不加载模型、不启动 GPU、不构造 optimizer。

## 2. Phase 6A 单卡技术 smoke

```bash
CONFIRM_THOUGHT6_PHASE6A=YES \
THOUGHT6_GPU_IDS=0 \
bash scripts/run_thought6_phase6a_smoke.sh
```

工程估计为 20–35 分钟，峰值约 23.0–23.5 GiB。它验证 B0/formal-null bitwise parity、F0 20/20、Fsigma 17/20、低 sigma 零 contribution、paired noise 和 checkpoint 不变；不形成论文效果结论。

## 3. Phase 6B 三卡离线 utility

```bash
CONFIRM_THOUGHT6_PHASE6B=YES \
THOUGHT6_GPU_IDS=0,1,2 \
bash scripts/run_thought6_phase6b_utility.sh
```

当前命令会在 GPU 启动前 fail-closed，因为三套 demonstrations 缺失。数据完整后的粗略工程估计为 1.5–3 小时；正式时间应以新 namespace 的 smoke 吞吐重新估算。

## 4. Phase 6C Stage 1

```bash
CONFIRM_THOUGHT6_PHASE6C_STAGE1=YES \
THOUGHT6_GPU_IDS=0,1,2 \
bash scripts/run_thought6_phase6c_stage1.sh
```

只有 Phase 6B 五个 Gate 全通过才可运行。规模为 480 rollouts，三张 4090 粗估 8–16 小时；不得把估计写成实测时间。

## 5. Stage 2 独立确认

```bash
CONFIRM_THOUGHT6_PHASE6C_STAGE2=YES \
THOUGHT6_GPU_IDS=0,1,2 \
bash scripts/run_thought6_phase6c_stage2.sh
```

只有 Stage 1 Camera 差值为正、CI lower 不大于 0、Clean 非劣且 Fsigma 优于 F0 时解锁。新增 state 10–19，使累计规模达到 960 rollouts；额外三卡时间粗估 8–16 小时。Stage 2 永不自动启动。

