# Thought4 实验记录与论文表格

正式科学结果尚未运行；真实 smoke v3 已通过，但只属于工程证据。本文件的论文
数字必须从
`outputs/thought4/.../artifact_manifest.json` 对应工件生成，不手抄猜测。

## 运行身份

| 字段 | 值 |
| --- | --- |
| project commit | smoke v1 `c629475bb601...`；v2 `67ea98347ce1...`；v3 `fd713d580303...`；v4 `fb49f57...`；v5 待提交/运行 |
| Fast-WAM commit | `45d8e145...` |
| checkpoint SHA | `1000437c...` |
| smoke v1 config fingerprint | `caef470ae35e0d1dafdc8f07ba4b8a088d6df26acebce787f900bdf7dda9caba` |
| smoke v2 config fingerprint | `9c412c1ce2cde90c9559ae298522bcc3f7dfc21d34c1376cfc565a4a9536146f` |
| smoke v2 planned cohort SHA | `8055511bdab15020224cd0dbfd6f1680fe45f4c8700cdf7ca29b416c29b5733f` |
| smoke v3 config fingerprint | `dbf3d11c354602c2484225e517838ea32ae9a2fa8a681c3e7ead6cceb0dbd0a6` |
| smoke v3 planned cohort SHA | `1a2c8a81349bcc4a8ffb15aaa6a5f8c0ebc1656762c7d4a352f475ef9ef492bc` |
| smoke v4 config fingerprint | `204ced0d56e038bbbe30fe14119ab48daa920cf52ea81af39f9dbc86c799a7bc` |
| smoke v4 planned cohort SHA | `5aea296c67dafe5118baec380d057d7a0e485a34f2930283de439d58ef7d2096` |
| smoke v5 config fingerprint | `fb895d2e3edccc5c966437324fd5a97433b1e0ed2c39a2eb56cb21606ea0be08` |
| smoke v5 planned cohort SHA | `08dfe330e2d91e0e7e436f0a5eb3325f5cdb8787742b0799eae19e7837dabb75` |
| formal v1 config / cohort SHA | `62951df5...ae44` / `340db6c1...708`（工程失败身份） |
| formal v2 config / cohort SHA | `cc608953...6f21` / `e3a6363f...5ca`（未运行，被新代码身份取代） |
| formal v3 config fingerprint | `44639a0229c7899dbc754ed8c2b743fd649f35ce5d729ef525ab99321575a27b` |
| formal v3 planned cohort SHA | `b9268f8ce0e63516e3742638acc68f6e528615743a54b4b8f7ed541981c6e210` |
| physical GPU | smoke v1/v2：1；smoke v3/formal v1/smoke v4：2；v5/v3 待运行 |
| smoke v3 start / finish | 2026-07-31 11:16:46 / 11:27:19 UTC |
| smoke v3 backbone SHA before / after | `ac0dd59...b4f8` / `ac0dd59...b4f8` |
| future RGB read | false |
| success outcome read | false |

## 实施验证（非科学结果）

| 检查 | 结果 |
| --- | --- |
| Thought4 CPU/mock | 40 passed |
| Thought1–4 全项目回归 | 437 passed；5 条 NVML 环境 warning 不影响测试结论 |
| 文档校验 | 84 个 Markdown；本地链接全部有效；`docs/` 根目录整洁 |
| smoke dry-run | PASS；Torch/model/simulator/write 均为 false |
| formal dry-run | PASS；Torch/model/simulator/write 均为 false |
| 真实 GPU smoke | v1/v2 **ENGINEERING FAILED**；v3 **PASSED / NON-SCIENTIFIC**；v4 **ENGINEERING FAILED（pre-model）**；v5 **NOT RUN** |
| 正式 diagnosis | v1 **ENGINEERING FAILED（pre-model）**；v2 未运行且被取代；v3 **NOT RUN** |

## Smoke v1 失败记录（非科学结果）

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_smoke_v1` |
| 时间 | 2026-07-31 10:47:10–10:47:14 UTC |
| project commit | `c629475bb601364a609a7fea69199772424d768f` |
| physical GPU | 1 |
| 停止位置 | `paired_render_started` 后、robosuite import 时 |
| 错误 | `MUJOCO_EGL_DEVICE_ID=0` 不属于 `CUDA_VISIBLE_DEVICES=1` |
| 模型加载 | false |
| 环境 reset / render | false / false |
| feature / action / probe | 0 / 0 / 0 |
| 科学结论 | 无；不能登记为 smoke PASS |

根因是 robosuite 校验物理可见 ID，而 PyTorch 才把唯一可见卡重映射为逻辑
`cuda:0`。v2 runner 固定 `CUDA_VISIBLE_DEVICES=<physical>`、
`MUJOCO_EGL_DEVICE_ID=<physical>`，模型仍使用 `cuda:0`。由于修复产生新 project
commit，v1 的 pre-validation 不能 checksum-identical resume；因此保留 v1，使用
全新 v2 namespace。

## Smoke v2 失败记录（非科学结果）

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_smoke_v2` |
| 时间 | 2026-07-31 10:56:34–11:05:41 UTC |
| project commit | `67ea98347ce106279b6262a864407504950e1498` |
| physical GPU | 1 |
| 已完成 | 2 base states × 3 conditions；6 paired render；6 labels |
| 停止位置 | model load 后，首次 source-A K/V feature capture |
| 错误 | `Inference tensors do not track version counter.` |
| feature shard / probe / intervention | 0 / 0 / 0 |
| `smoke_result.json` | 未生成 |
| 科学结论 | 无；不能登记为 smoke PASS |

根因是只读 hook 用 `_version` 检查 cache tensor identity，而 Fast-WAM 正式推理在
`torch.inference_mode()` 内生成的 tensor 按 PyTorch 设计没有 version counter。
v3 对普通 tensor 保留 version 检查；对 inference tensor 改用 pointer/shape/dtype/
device/stride 并在 scope 退出时与首次 clone 做逐值比较，因此兼容推理同时继续
检测原地修改。v2 的代码身份已变化，不能 resume；保留 v2 并使用全新 v3
namespace。v3 不改变任何科学协议字段。

## Smoke v3 通过记录（非科学结果）

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_smoke_v3` |
| project commit / GPU | `fd713d580303a61571a46b9902f1b4b708a453b5` / 2 |
| 时间 / 总时长 | 2026-07-31 11:16:46–11:27:19 UTC / 10m33s |
| 覆盖 | 2 base states × Clean/Camera/Lighting；60 feature records |
| model load | 401.698 s |
| 主干 SHA before / after | `ac0dd59...b4f8` / `ac0dd59...b4f8` |
| identity replacement | PASS；action L2=0 |
| future RGB / success read | false / false |
| result SHA | `b0e1cc80e4620deed389435463459cbcce6a8a804aa58e7f44af7a5212e17dbb` |

该结果证明真实 hook、feature、probe backward 和 identity replacement 链路可运行；
它没有 Robot-init condition，不是 formal 数据，也不能解锁当前 formal v3。

## Formal v1 失败记录与第一版 input-time 修复（非科学结果）

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_diagnosis_v1` |
| project commit / GPU | `fd713d580303a61571a46b9902f1b4b708a453b5` / 2 |
| 时间 | 2026-07-31 11:31:24–11:32:01 UTC |
| 停止位置 | 第一个 base state 的 Robot-init reset 检查；模型加载前 |
| 错误 | `Robot-init variant did not change the initial robot state` |
| paired render / feature / probe / intervention | 0 / 0 / 0 / 0 |
| 科学结论 | 无 |

根因不是 Robot-init variant 无效，而是判定时机错误：adapter reset 后调用
`set_init_state()`，共享 demonstration flat state 覆盖了当下可观测 qpos。
只读单样本诊断中，reset 最大 robot-state 差为 0；共同 prefix 到 `t=37` 后为
0.08357（30-step settling 参考为 0.23856）。同一 Clean prefix 与 demonstration
的 EEF 对齐误差为 0.01250 m / 0.8694°，仍通过 3 cm / 15° 门槛。

v4/v2 只把 Robot-init 生效检查移到模型输入时刻并双写 reset/input SHA；object
layout 仍在 prefix 前与 Clean 配对，Camera/Lighting 仍是 exact-state，正式科学
协议字段不变。formal v2 未运行；smoke v4 的后续工程失败单独登记如下。

## Smoke v4 失败记录与 v5 修复（非科学结果）

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_smoke_v4` |
| project commit / GPU | `fb49f57` / 2 |
| 时间 | 2026-08-01 06:32:16–06:32:54 UTC |
| 停止位置 | 第一条 Camera input-state check；模型加载前 |
| 错误 | `camera robot state differs from Clean at model input time` |
| paired manifest / feature / probe / intervention | 0 / 0 / 0 / 0 |
| 科学结论 | 无 |

根因是 observation 采样路径不匹配，而非 exact flat-state 配对失败：Clean 保留
`env.step()` 返回的缓存 observation；Camera/Lighting 则在注入同一 Clean state
后强制更新 observables。v5 让三种 exact-state condition 都从同一 Clean state
走相同刷新路径，Robot-init 也从自身 input state 刷新。

修复后的真实 render-only 复验完成 2 base states × 4 conditions 共 8 条样本：
Robot-init reset-match 为 2/2，input-match 为 0/2，input simulator-state
differs 为 2/2；没有加载 Fast-WAM。smoke v5 与 formal v3 保持相同科学参数，
使用全新 namespace，当前均为 **NOT RUN**。

## Table A：Video geometry readability

| Module/layer | Pool | Probe | Target | Clean | Camera | Lighting | Robot-init | Camera−Clean paired 95% CI |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 待运行 | | Linear | depth | | | | | |
| 待运行 | | MLP | camera pose | | | | | |
| 待运行 | | Linear | EEF–object | | | | | |

同时记录 zero、target-mean、shuffled-label control；不能只展示最优 MLP。
所有指标使用反标准化后的原始目标单位；另从每条 probe row 登记 train-only
normalizer SHA、constant-dimension 数和 best development epoch。

## Table B：Action motion readability

| Module/layer | Denoise step | Probe | Target | Clean | Camera | Lighting | Robot-init | Linear−MLP |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 待运行 | 19（第 20 步） | Linear | current EEF–object geometry | | | | | |
| 待运行 | 19（第 20 步） | Linear | translation ADE/FDE | | | | | |
| 待运行 | 19（第 20 步） | MLP | rotation geodesic | | | | | |
| 待运行 | 19（第 20 步） | Linear | gripper | | | | | |
| 待运行 | 19（第 20 步） | MLP | SE(3) composite | | | | | |

## Table C：geometry-subspace intervention

| Layer | Rank | Energy | Norm ratio | Replay floor L2 | Correct–shuffle L2 | Above floor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 待运行 | | | | | | |

记录 target/donor episode、mapping SHA、action noise/schedule/preprocessing SHA 和
每个 timestep 差异。

| Selected feature | Camera coordinate L2 | Lighting coordinate L2 | Camera−Lighting paired 95% CI | Robot-init（非 exact） |
| --- | ---: | ---: | --- | ---: |
| 待运行 | | | | |

## 唯一结论

```text
classification: NOT RUN
recommendation: NOT RUN
```

只有 `method_selection.json` schema/SHA 验证通过后填写。论文表述必须是：
“冻结表征诊断支持/不支持某种 gap”，而不是“方法提高 OOD success”。

## 简历可用工程表达（结果前）

> 为 5B Video DiT + Action DiT 机器人策略设计冻结式 geometry–action gap
> 诊断栈：审计真实 MoT K/V 消费链路，构建同 MuJoCo state 的
> Clean/Camera/Lighting paired rendering、episode-safe SE(3) probes、SVD
> geometry-subspace matched intervention，以及全链路 SHA/不可覆盖工件协议。

真实结果出来后，只可增加已由工件支持的数字。
