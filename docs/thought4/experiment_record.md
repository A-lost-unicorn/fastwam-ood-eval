# Thought4 实验记录与论文表格

有效真实结果尚未运行。本文件是填写模板；数字必须从
`outputs/thought4/.../artifact_manifest.json` 对应工件生成，不手抄猜测。

## 运行身份

| 字段 | 值 |
| --- | --- |
| project commit | v1 `c629475bb601...`；v2 `67ea98347ce1...`；v3 待提交/运行 |
| Fast-WAM commit | `45d8e145...` |
| checkpoint SHA | `1000437c...` |
| smoke v1 config fingerprint | `caef470ae35e0d1dafdc8f07ba4b8a088d6df26acebce787f900bdf7dda9caba` |
| smoke v2 config fingerprint | `9c412c1ce2cde90c9559ae298522bcc3f7dfc21d34c1376cfc565a4a9536146f` |
| smoke v2 planned cohort SHA | `8055511bdab15020224cd0dbfd6f1680fe45f4c8700cdf7ca29b416c29b5733f` |
| smoke v3 config fingerprint | `dbf3d11c354602c2484225e517838ea32ae9a2fa8a681c3e7ead6cceb0dbd0a6` |
| smoke v3 planned cohort SHA | `1a2c8a81349bcc4a8ffb15aaa6a5f8c0ebc1656762c7d4a352f475ef9ef492bc` |
| formal config fingerprint | `62951df5bb364daf5687ac505bad207da19f9626e8038de1471ef4befcfaae44` |
| formal planned cohort SHA | `340db6c1a15111a390b601beb0a29afdbfaec2deac22060f7ee709a83f054708` |
| physical GPU | v1：1；v2：1；v3 待运行 |
| start / finish | 待运行 |
| backbone SHA before / after | 待运行 / 待运行 |
| future RGB read | false |
| success outcome read | false |

## 实施验证（非科学结果）

| 检查 | 结果 |
| --- | --- |
| Thought4 CPU/mock | 37 passed |
| Thought1–4 全项目回归 | 434 passed，NVML 环境 warning 不影响测试结论 |
| 文档校验 | 84 个 Markdown，本地链接全部有效，`docs/` 根目录整洁 |
| smoke dry-run | PASS；Torch/model/simulator/write 均为 false |
| formal dry-run | PASS；Torch/model/simulator/write 均为 false |
| 真实 GPU smoke | v1 **ENGINEERING FAILED（pre-model）**；v2 **ENGINEERING FAILED（first feature inference）**；有效 v3 **NOT RUN** |
| 正式 diagnosis | **NOT RUN** |

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
