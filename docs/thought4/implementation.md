# Thought4 实现与数据流

## 总体数据流

```text
LeRobot episode metadata/actions/EEF
             │
             ├─ outcome-blind episode split + frame plan
             ├─ prefix-to-t EEF alignment QC（披露，不筛样本）
             │
             ▼
LIBERO-Plus Clean state recovery + frozen variant-panel selection
             │ same flat state
             ├─────────────┬──────────────┐
             ▼             ▼              ▼
          Clean         Camera         Lighting       Robot-init
             │             │              │              │
             ├─ one shared Clean world replay ───────────┤ independent replay
             │             │              │              │
             └──── RGB/depth/camera/EEF/object labels ───┘
                                  │
                         release simulator/EGL
                                  ▼
                       frozen Fast-WAM inference
                                  │
                  ┌───────────────┴────────────────┐
                  ▼                                ▼
       Source A: Video hidden/K/V        Source B: Action hidden
                  │                                │
                  └──── linear/MLP + controls ─────┘
                                  │
                development-only intervention layer
                                  ▼
                   geometry-coordinate shuffle
                                  │
                                  ▼
                      exactly one method class
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `schemas.py` | sample/pair/feature 版本化 schema 与 SHA |
| `config.py` | 严格 YAML、官方 commit/checkpoint/layer 边界 |
| `cohort.py` | episode split、frame plan、materialized state manifest |
| `paired_rendering.py` | exact-state 恢复、camera/depth/hash |
| `geometry_labels.py` | 坐标变换、depth/pose/relation/trajectory mask |
| `feature_hooks.py` | 实际调用点 capture/replacement context manager |
| `video_feature_extractor.py` | pooling、ROI mask、feature shard/checksum |
| `action_feature_extractor.py` | Action 四个固定边界 |
| `probe_*` / `pipeline.py` | probe、controls、metrics、bootstrap |
| `geometry_subspace.py` | SVD basis、reconstruction、coordinate replacement |
| `action_intervention.py` | donor、seed contract、action metrics/replay floor |
| `real_runtime.py` | simulator-first/model-second 真实链路 |
| `intervention_runtime.py` | frozen policy 的 matched intervention |
| `decision.py` | 四选一方法规则 |
| `phase4.py` | dry-run、smoke、formal 工件编排 |

`phase4.py` 还生成 `alignment_audit.json`，保存原冻结 cohort 的逐状态误差、
3 cm / 15° pass/fail、split 计数与分位数。formal gate 会读取该实体文件，重算
canonical SHA，并与 `smoke_result.json` 的计数和 SHA 交叉验证；只有汇总字段而
缺少/篡改审计文件不能解锁正式运行。

## Hook 为什么不是 block.forward

Fast-WAM MoT 直接访问每个 block 的子模块，不调用 block 自身的 `forward()`。
因此真实 hook 是：

```text
Video: video_expert.blocks.<i>.norm1 (input)
       mot.video_kv_cache.<i>.k/v
       (forward_action_with_video_cache keyword argument)

Action: action_expert.action_encoder (output)
        action_expert.blocks.15.norm1 (input)
        action_expert.blocks.29.norm1 (input)
        action_expert.head (input)
```

每个 context 退出时检查 call count 并移除 handle。未触发、shape 不符、NaN/Inf、
replacement dtype/device 改变都会 fail closed。

正式干预选择只允许 `mot.video_kv_cache.<i>.k/v`，因为它们是 Action DiT 直接
读取、且 K 已完成 norm/RoPE 的 current-frame cache；`norm1` 只用于可读性分析。
scoped wrapper 在 20 个 denoise calls 中验证 cache pointer/version 不变，替换时
只复制 list/目标 dict，不原地改原 cache。真实 smoke 固定对 layer 15 cache V 做
identity replacement，以验证同类 consumer-boundary replay parity。

smoke 只注册 Video layer 15 与 Action block 15；formal 才注册五个 Video layer
和四个 Action 边界。Action 20 次调用不会被静默覆盖：v1 取最后一次并在 schema
中写 `denoise_step_index=19`。

Action source B 同时走两条互不混淆的 probe 路由：当前 `t` 的
EEF–object translation/orientation geometry 用于直接测 Video→Action geometry transfer；未来
`t+1...t+32` 的 translation/rotation/gripper/SE(3) 用于测可执行运动结构。
方法规则分别选定同名主目标，不跨单位比较 development loss。

Probe 的 feature/target 标准化统计只从 train rows 和 valid labels 计算；模型在
标准化空间训练和 early-stop，评估时回到米、相对深度、rotation-6D、gripper 等
原单位。统计量 SHA 随结果落盘。linear probe 用于 Phase 4-C 时先换算为
`W_raw = diag(target_std) W_norm diag(1/feature_std)`，再做 SVD，保证层比较与
geometry basis 不被 raw activation scale 混淆。
正式层/方法选择不挑“幸运 seed”：先以全部冻结 seeds 的 mean development loss
选择 feature group，再以跨 seed mean 原单位指标判定可读性；每个被选 row SHA 和
逐 seed paired gap 都写入 `diagnostic_evidence.json`。

## FP32 subspace arithmetic 与 correct control

真实 Action-consumed K/V cache 是 BF16。formal v5 的旧实现把 linear-probe SVD
basis 转为 BF16 后执行 `h - P(h) + P(h)`，因此在第一次 intervention 前无法通过
correct reconstruction。formal v6 的生产路径固定为：

```text
captured BF16 hidden
  → hidden.float() + FP32 basis
  → FP32 coordinates / projection / residual / replacement
  → one output cast to captured BF16 dtype
  → Action consumer
```

`geometry_subspace.py` 不再按 hidden dtype 降精度 basis；
`intervention_runtime.py` 对 correct control 强制 shape/dtype/device 不变、
`torch.equal`、input/output SHA 相同和 cast 后 max-abs=0。smoke v8 从真实 capture
构造固定 technical subspace，先执行上述逐位检查，再把 reconstructed BF16 tensor
送入真实 K/V consumer。正式 gate 会重算该子工件的 canonical SHA。动作 replay
floor 仍用于推理复现，但不参与或放宽 tensor 逐位判定。

formal v6 还把 Phase 4-A/B 的四个工件放到 Phase 4-C 调用之前提交：两个 probe
result、layer summary 和 `probe_stage_result.json`。因此 intervention 工程失败不会
再次丢失已完成 probe panel；最终 evidence/method/integrity/report 仍只在
intervention 成功后生成。

## 显存设计

Fast-WAM 已观测加载峰值约 23.68 GiB，接近 4090 上限。真实链路先完成所有
MuJoCo paired rendering 并关闭四个环境，再加载模型；pool 后特征立即转 CPU。
正式 feature 分 shard 保存 `.pt + .sha256`，默认不长期保存所有 3072 维完整
token tensor。中断恢复时，完整 shard 只读加载，并同时核对 sidecar SHA、每个
tensor SHA 及 frozen metadata；`probe_inputs.pt` 使用同样的 checksum 合同。
真实 smoke/formal 在创建 run 工件前要求项目 worktree clean，并把 project commit
写入 static audit；runner 在碰到 completed `run_status.json` 时不会再追加日志。

## simulator-replay 标签边界

formal v5 不再直接使用 parquet future EEF 作为动作—运动标签。collector 在
Clean 输入状态 `t` 只执行一次 `a_t...a_{t+H-1}`，保存 simulator 产生的 world
translation/rotation/gripper/valid mask，并在 `finally` 中恢复 `state_t`。随后：

- Clean、Camera、Lighting 使用同一 world replay，仅按各自 camera extrinsic
  转换；这是 `shared_clean_world_replay_transformed_to_condition_camera`；
- Robot-init 在自己的 input state 独立 replay；这是
  `condition_specific_world_replay`，不能复用 Clean trajectory；
- future observation 的 RGB 字段不读取、不保存、不输入模型；reward/done 也不
  用于样本选择或 success 判定。

Clean prefix 与 parquet EEF 的 3 cm / 15° 对齐仍逐条计算，但 v5 只将其作为
数据质量审计。任何 failed row 仍进入 probe panel，原 64 个 state identity 不变。
这把“parquet 状态是否精确复现”与“从模型实际看到的 state_t 出发会产生什么
运动”分离开来。

在 prefix 前还会核对所有 task object/fixture 的排序 pose snapshot 与 Clean 一致，
robot reset state 只披露、不作为扰动是否生效的判据。执行相同 prefix 后，在实际
模型输入时刻 `t` 核对 Robot-init robot state 和完整 simulator state 均区别于
Clean；Clean/Camera/Lighting 仍必须相同。reset/input 两套 SHA 和 match 标志均
落入 label manifest。Clean/Camera/Lighting 的 input observation 必须全部从冻结
Clean flat state 经相同 `set_state → forward → observable refresh` 路径生成；
不能把 `env.step()` 的缓存 observation 与刷新后的 observation 直接比较。
