# Thought4 实验记录与论文表格

真实结果尚未运行。本文件是填写模板；数字必须从
`outputs/thought4/.../artifact_manifest.json` 对应工件生成，不手抄猜测。

## 运行身份

| 字段 | 值 |
| --- | --- |
| project commit | 待运行 |
| Fast-WAM commit | `45d8e145...` |
| checkpoint SHA | `1000437c...` |
| smoke config fingerprint | `caef470ae35e0d1dafdc8f07ba4b8a088d6df26acebce787f900bdf7dda9caba` |
| formal config fingerprint | `62951df5bb364daf5687ac505bad207da19f9626e8038de1471ef4befcfaae44` |
| formal planned cohort SHA | `340db6c1a15111a390b601beb0a29afdbfaec2deac22060f7ee709a83f054708` |
| physical GPU | 待运行 |
| start / finish | 待运行 |
| backbone SHA before / after | 待运行 / 待运行 |
| future RGB read | false |
| success outcome read | false |

## 实施验证（非科学结果）

| 检查 | 结果 |
| --- | --- |
| Thought4 CPU/mock | 34 passed |
| Thought1–4 全项目回归 | 431 passed，5 个 NVML 环境 warning |
| 文档校验 | 84 个 Markdown，本地链接全部有效，`docs/` 根目录整洁 |
| smoke dry-run | PASS；Torch/model/simulator/write 均为 false |
| formal dry-run | PASS；Torch/model/simulator/write 均为 false |
| 真实 GPU smoke | **NOT RUN** |
| 正式 diagnosis | **NOT RUN** |

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
