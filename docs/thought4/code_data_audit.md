# Thought4 Phase 4：Geometry–Action Gap 代码与数据审计

状态：`AUDIT COMPLETE / IMPLEMENTATION + CPU TESTS COMPLETE / V1 GPU ATTEMPT FAILED PRE-MODEL / VALID V2 NOT RUN`

审计日期：2026-07-31

本审计发生在 Thought4 实现之前。范围仅包括代码、配置、已有工件和本地数据
schema 的只读检查；没有加载 Fast-WAM checkpoint，没有运行 GPU，没有生成
rollout，也没有读取 success/OOD outcome 来选择样本或层。

## 结论先行

Phase 4-A/B/C 可以在不修改 `third_party/FastWAM` 的前提下实现，但有两个必须
显式处理的边界：

1. Fast-WAM 在线动作路径由 `MoT.prefill_video_cache()` 和
   `MoT.forward_action_with_video_cache()` 驱动；它不会调用
   `DiTBlock.forward()`。因此把 hook 直接注册到 `blocks[i]` 会注册成功却不触发。
   Thought4 必须 hook 实际被调用的 `norm1`、`action_encoder` 和 `head`；最终
   K/V 则从 `forward_action_with_video_cache(video_kv_cache=...)` 的真实消费参数
   捕获，并以 replay parity fail closed。
2. 当前本地 LeRobot demonstration 只有 RGB、EEF/joint/gripper、action 和时间
   identity，没有 depth、完整 MuJoCo state、object pose、相机参数。因此它适合
   提供冻结 action/EEF trajectory 标签，却不能单独支持 exact-state
   Clean/Camera/Lighting 重渲染。正式 paired panel 必须由 LIBERO-Plus simulator
   从同一 flat simulator state 重渲染，并在每一 pair 上验证 state hash。

这意味着实现可继续，但 formal 科学结果仍依赖真实 simulator/model smoke 通过。

## 1. Video DiT block 数量和命名

发布配置
`third_party/FastWAM/configs/model/fastwam.yaml` 冻结为：

- `video_dit_config.num_layers = 30`；
- `hidden_dim = 3072`；
- `num_heads = 24`；
- `attn_head_dim = 128`；
- `patch_size = [1, 2, 2]`。

模型路径为：

```text
video_expert.blocks.0
...
video_expert.blocks.29
```

每个 block 的真实可调用子路径包括：

```text
video_expert.blocks.<i>.norm1
video_expert.blocks.<i>.self_attn.q
video_expert.blocks.<i>.self_attn.k
video_expert.blocks.<i>.self_attn.v
video_expert.blocks.<i>.self_attn.o
video_expert.blocks.<i>.cross_attn
video_expert.blocks.<i>.norm2
video_expert.blocks.<i>.ffn
```

正式代表层在看见任何 probe/test/OOD 结果前冻结为
`[0, 7, 15, 22, 29]`，覆盖 early/middle/late/final。

## 2. 当前帧 K/V 的生成函数和 shape

在线 `FastWAM.infer_action()` 的当前图像路径是：

```text
RGB [1,3,224,448]
  -> VAE current latent [1,48,1,14,28]
  -> Video patch tokens [1,98,3072]
  -> MoT.prefill_video_cache()
  -> 30 × {"k": [1,98,3072], "v": [1,98,3072]}
```

K/V 的权威生产函数是：

```text
third_party/FastWAM/src/fastwam/models/wan22/mot.py
MoT.prefill_video_cache()
```

其中 `_build_expert_attention_io()` 在每层从当前层输入 hidden 计算 Q/K/V，
将 K/V 写入逐层 list。Action denoising 的权威消费函数是
`MoT.forward_action_with_video_cache()`；每个 Action query 在每层读取对应
Video K/V，并与当前 Action K/V 拼接。

注意：Thought3 文档里的 future tail `[1,48,2,14,28]` patch 后有 196 token；
在线 current-only cache 只有一个 latent time，因此是 98 token，不能混称。

## 3. Action DiT block 数量和命名

发布配置冻结为：

- `action_dit_config.num_layers = 30`；
- `hidden_dim = 1024`；
- `action_dim = 7`；
- `ffn_dim = 4096`；
- `num_heads = 24`；
- `attn_head_dim = 128`；
- action horizon = 32；
-正式 action denoising steps = 20。

模型路径是：

```text
action_expert.action_encoder
action_expert.blocks.0
...
action_expert.blocks.29
action_expert.head
```

Thought3 的 `ActionEncoderFutureInjector` 属于 Action-side source B：
它只 hook `action_encoder` 输出，不是 Video intermediate 或 Video K/V。

## 4. Action early/mid/late/pre-head hook

只读提取位置冻结为：

| 语义 | 模块路径 | 捕获方式 |
| --- | --- | --- |
| input | `action_expert.action_encoder` | forward output |
| middle | `action_expert.blocks.15.norm1` | forward pre-hook 的 tensor input |
| late | `action_expert.blocks.29.norm1` | forward pre-hook 的 tensor input |
| pre-head | `action_expert.head` | forward pre-hook 的 tensor input |

`norm1` 是 MoT 内部真实调用点；直接 hook `blocks.15` 或 `blocks.29` 无效。
Action feature 每个去噪步都会产生，因此 manifest 必须记录
`denoise_step`，不能把 20 次调用静默覆盖。第一版 probe 冻结使用最后一个
action denoise step，并显式保存零基 `denoise_step_index=19`。

## 5. 只读 hook 支持

PyTorch 的 forward/pre-forward hook 足以完成只读捕获：

- Video layer input：`video_expert.blocks.<i>.norm1` pre-hook；
- Action input/mid/late/pre-head：上述四个真实调用点。

`self_attn.k` module output 之后仍有 `norm_k + RoPE`，所以不能误标为最终 cached
K（V projection output 恰好等于 cache V，但也不单独依赖这一巧合）。实际
action-consumed K/V 使用 scoped method wrapper，从
`mot.forward_action_with_video_cache()` 的 `video_kv_cache` keyword argument
捕获，语义路径记为 `mot.video_kv_cache.<layer>.k/v`。20 个 action denoise calls
必须复用相同 pointer/version；只保存一份 detached tensor。

所有捕获值必须 `detach()`，按冻结 pooling 规则转 CPU，并保存 module path、
layer、source、shape、dtype、sample/condition identity 和 tensor SHA。hook
必须由 context manager 注册/移除；退出后 replay output 必须与无 hook 基线一致。

## 6. Phase 4-C output replacement 支持

源 A 存在不修改上游源码的最小替换点：

- 在 `video_expert.blocks.<i>.norm1` pre-hook 中只替换 probe-defined hidden
  geometry coordinates；该修改影响该层 Q/K/V，而 residual hidden 仍保持原值；
- 在 `forward_action_with_video_cache` 参数边界复制 cache list/目标 layer dict，
  只替换实际 cached K 或 V 的 geometry coordinates，不原地修改原 cache。

第一版将 `norm1` hidden 保留为只读 probe，但 Phase 4-C 的候选集合严格限制为
`mot.video_kv_cache.<layer>.k/v`：它们是 Action DiT 直接消费的 current-frame
cache，且与 linear probe 共用 `hidden_dim=3072`。真实 smoke 固定在 layer 15 的
cache V consumer argument 做
identity replacement，以先验证 formal 会使用的同类边界。正式使用前必须通过：

- identity intervention 与 replay parity；
- correct reconstruction；
- residual projection invariant；
- correct/shuffle norm matching；
- frozen action seed/noise/schedule。

如果真实 smoke 发现该 hook 未触发、shape 不稳定或 replacement replay 不稳定，
Phase 4-C 必须 fail closed；不得静默改成整条 hidden replacement。v1 不自动切换
到 Action-side fallback；若确实需要 `action_encoder`/`head`，必须另开协议版本并
披露 source B 限制。

## 7. Depth、camera、EEF、object pose 标签

本地 LeRobot 数据
`data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot` 包含：

- 两路 512×512 RGB video；
- `observation.state [8]`；
- `observation.states.ee_state [6]`；
- `observation.states.joint_state [7]`；
- `observation.states.gripper_state [2]`；
- `action [7]`；
- timestamp/frame/episode/task identity。

它不包含：

- depth；
- flat MuJoCo state；
- object pose；
- camera intrinsic/extrinsic。

Simulator 侧可获得：

- `camera_depths=True` 的 normalized depth，并通过
  `robosuite.utils.camera_utils.get_real_depth_map()` 转成 metric depth；
- `get_camera_intrinsic_matrix()`；
- `get_camera_extrinsic_matrix()`，语义为 camera-to-world pose；
- observation 中的 `robot0_eef_pos/quat/gripper_qpos`；
- LIBERO `obj_body_id` 对应的 `sim.data.body_xpos/body_xquat`；
- `env.get_sim_state()` / `set_state()`。

因此 depth、camera pose、EEF–object relation 标签必须在 paired simulator
collector 中生成；缺任一字段时 formal collector fail closed。Demonstration
的未来 EEF/action 允许作为 t+1…t+H 标签，必须记录时间戳和 episode mask，且不得
把 future RGB 输入 probe/model。

## 8. Clean/Camera/Lighting exact-state 重渲染

LIBERO-Plus 对 task 0 提供 21 个 Camera、42 个 Lighting variant。Camera BDDL
通过 `_view_..._initstate_0` 参数改变 camera；Lighting BDDL 改变 scene light。
两类 visual variant 都路由回 base init-state 文件。

实现方案是在单一 `third_party/LIBERO-plus` Python package 进程内构造：

1. Plus checkout 中的 clean base BDDL；
2. 冻结的五视角 Camera panel；
3. 冻结的五种光照 Lighting panel。

每个 base state 的具体 variant 仅由冻结 split seed、task、episode、frame hash
决定。Camera/Lighting/Robot-init panel ID 全部写进 YAML，不能按 probe 结果改。

每个 base state 先持久化 flat simulator state，再分别 set/forward/render。只有
以下全部相等才登记 exact-state pair：

- simulator state vector shape；
- canonical float bytes SHA；
- robot qpos/qvel；
- object body world pose；
- task/progress/timestamp identity。

camera/light/XML/renderer metadata则必须按条件不同并另存。若模型 state layout
不兼容，不能只比较 init-state index；formal collection 必须失败。

## 9. Robot-init 的独立物理状态 control

task 0 有 43 个官方 Robot Initial States variant。它们通过
`_view_0_0_100_0_0_initstate_<n>` 选择不同 Panda initial configuration。

Robot-init 明确不是 exact-state visual pair。它只要求：

- 同 task；
- 同冻结 seed namespace；
- prefix 前所有 task object/fixture pose 与 Clean 在 `1e-7` 容差内一致；
- reset robot joint/EEF/gripper state 与 Clean 确实不同；
- 独立 simulator state 和独立 state hash；
- `exact_state_pair=false`。

任何把 Robot-init 与 Clean 标为相同 state 的 record 都必须被 schema/test 拒绝。
同样不能直接给 Robot-init 复制 Clean 的未来 EEF 标签：正式 collector 在该
Robot-init 状态下重放相同 prefix 到 `t`，再执行冻结 demonstration future
actions 构造独立 trajectory label，并在 label manifest 披露来源。

另外，Clean action-prefix 恢复后的 EEF 与 parquet 第 `t` 帧必须通过 3 cm / 15°
输入时间对齐检查；Robot-init 则明确标为不适用。Probe feature/target normalizer
只在 train/valid label 上拟合，预测反标准化后再计算指标；跨 seed 层选择使用
mean development loss，不能选择单一幸运 seed。Phase 4-C 只从 Action 直接消费的
K/V 中选择，并区分 probe weight energy 与真实 hidden projection energy。

## 10. 预计修改文件与实施计划

已实现模块：

```text
src/fastwam_ood_eval/thought4/
  __init__.py
  schemas.py
  config.py
  feature_hooks.py
  video_feature_extractor.py
  action_feature_extractor.py
  paired_rendering.py
  geometry_labels.py
  probe_models.py
  probe_training.py
  probe_evaluation.py
  geometry_subspace.py
  action_intervention.py
  decision.py
  audit.py
  cli.py
  cohort.py
  io_utils.py
  pipeline.py
  real_runtime.py
  intervention_runtime.py
  phase4.py
  report.py
```

新配置与 runner：

```text
configs/thought4/phase4_geometry_action_diagnosis_v1.yaml
configs/thought4/phase4_geometry_action_smoke.yaml       # v1 失败身份，保留
configs/thought4/phase4_geometry_action_smoke_v2.yaml    # 当前 runner
scripts/run_thought4_phase4_smoke.sh
scripts/run_thought4_phase4_diagnosis.sh
```

测试集中在 `tests/test_thought4_*.py`，覆盖目标文件列出的 25 条合同。正式 cohort
将每 episode 上限冻结为 2 帧，使 40/12/12 states 对应 20/6/6 个 episode，避免
held-out grouped bootstrap 只建立在 3 个 episode 上。主 CLI 只
增加 Thought4 dispatch，不改变 Thought1/2/3 语义。

验证映射：

| 目标合同 | 权威测试/运行检查 |
| --- | --- |
| 1–7：hook read-only、A/B capture、非法层、冻结、probe-only、detach | `test_thought4_hooks.py`、真实 smoke backbone SHA |
| 8–13：episode split、exact pair、Robot-init、camera、坐标、future mask | `test_thought4_schemas_geometry.py`、label/render manifest |
| 14–20：shuffle、SVD、correct/shuffle、donor、seed、replay floor | `test_thought4_probe_intervention.py`、真实 identity replacement |
| 21–24：参数 SHA、finite、不可覆盖、dry-run 零加载 | hooks/config tests、smoke/formal hard checks |
| 25：Thought1/2/3 不回归 | 全项目 `432 passed` |

当前专项结果为 `35 passed`；文档检查为 84 个 Markdown、本地链接全部有效且
`docs/` 根目录整洁。上述真实
v1 smoke 在 robosuite import 时因 EGL 物理编号错误停止，模型未加载、环境未
reset；有效 v2 smoke 仍为 **NOT RUN**，不能用 mock 或失败尝试代替。

最短执行顺序：

```text
hooks/labels/schemas
  -> mock paired panel
  -> linear/MLP probes + controls + grouped bootstrap
  -> geometry-subspace intervention
  -> method selection
  -> CPU/mock tests and dry-run
  -> confirmed real smoke
  -> freeze formal cohort/config
  -> user-confirmed formal diagnosis
```

## 审计后的 fail-closed 清单

- 禁止 hook `DiTBlock.forward()` 后假设已捕获 MoT feature；
- 禁止从现有 parquet 伪造 depth/object/camera/sim-state 标签；
- 禁止把 Robot-init 作为 exact-state pair；
- 禁止根据 test/OOD/success 选择层、样本或 probe；
- 禁止保存未 detach 的 backbone tensor；
- 禁止训练 Fast-WAM/Video DiT/Action DiT/VAE；
- 禁止在未设置确认变量时运行真实 smoke 或 formal；
- 禁止把 probe/intervention 结果写成 policy success 或 OOD improvement。
