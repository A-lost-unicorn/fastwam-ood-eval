# Thought4 实验记录与论文表格

formal v6 冻结诊断已经完成；真实 smoke 仍只属于工程证据。本文件的论文数字必须从
`outputs/thought4/.../artifact_manifest.json` 对应工件生成，不手抄猜测。

## 运行身份

| 字段 | 值 |
| --- | --- |
| project commit | smoke v1 `c629475...`；v2 `67ea983...`；v3 `fd713d5...`；v4 `fb49f57...`；v5 `833071f...`；v6/formal v4 `aeb0210...`；smoke v7/formal v5 `229a0f3...`；smoke v8/formal v6 `46d03f23e88afef79aa63204c13dea6dd3eb7d19` |
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
| smoke v6 config fingerprint | `90d1290e9ec9a644b968e4965deab53052c784c23c717ce1b632cfd7435c2ce3` |
| smoke v6 planned cohort SHA | `84ec20b2f03d59ed1d2c8d2b76f78be456ce408c87a8e2cbdc7f05f3435d9206` |
| smoke v7 config fingerprint | `72eeaed07fb5f2b8457106dd3d5cd89333a47dca19bd4533bb9c6a90b13ebb90` |
| smoke v7 planned cohort SHA | `36bd5f967fbb145b73522cdd90d28b44dfd7ea47ad4f9f84fac93cd9760c8cbd` |
| smoke v8 config fingerprint | `81d3885ccb5b58806c1a729e509c039f6e1cb33a34ff242f8fa16785796149d7` |
| smoke v8 planned cohort SHA | `a67ff85321dc684a80b853b58ab133905232a275e6d71255fd1c966b9a3d6c12` |
| formal v1 config / cohort SHA | `62951df5...ae44` / `340db6c1...708`（工程失败身份） |
| formal v2 config / cohort SHA | `cc608953...6f21` / `e3a6363f...5ca`（未运行，被新代码身份取代） |
| formal v3 config fingerprint | `44639a0229c7899dbc754ed8c2b743fd649f35ce5d729ef525ab99321575a27b` |
| formal v3 planned cohort SHA | `b9268f8ce0e63516e3742638acc68f6e528615743a54b4b8f7ed541981c6e210` |
| formal v4 config fingerprint | `7783f2371fd2c1e781dc673817c4bcbbc2f85a5123e5dc67df29db768102efd1` |
| formal v4 planned cohort SHA | `640af37c911e87f8ae950d648e98cceb20c5c178b50e547864924cd05771a683` |
| formal v5 config fingerprint | `7b2a8e7ba6a51fe5246599324b09983d8df19824bfc906d7e8fd3932276fbb3a` |
| formal v5 planned cohort SHA | `26394c6a856a8292eb5f2a0f125fa307ecec23ec9a907294ad05e9b2bbf5ccda` |
| formal v6 config fingerprint | `3b14a7d7fd09deda9253bb1cd9950d9c4b5bd0cdf9f124a4dfede22add5c24f6` |
| formal v6 planned cohort SHA | `9af7cf7c1933fb1e5574099361f6d7dcc7500727480ecb4bbf010089f28d8f04` |
| physical GPU | smoke v1/v2：1；其余真实 Thought4 smoke/formal：2 |
| smoke v3 start / finish | 2026-07-31 11:16:46 / 11:27:19 UTC |
| smoke v3 backbone SHA before / after | `ac0dd59...b4f8` / `ac0dd59...b4f8` |
| smoke v6 start / finish | 2026-08-01 10:07:40 / 10:18:42 UTC |
| smoke v6 result SHA | `b260977ae826e8c860074bd3402a3914dbc52e3f887cc090dad5ff3be2bc4c37` |
| smoke v7 start / finish | 2026-08-03 02:32:15 / 02:43:10 UTC |
| smoke v7 result SHA | `9d81d79afa9f3efcadf1a015f596f33e60414198b72f7c1e6cfa5a1322a1fbf9` |
| formal v5 start / finish | 2026-08-03 03:03:08 / 04:13:40 UTC；ENGINEERING FAILED |
| smoke v8 start / finish | 2026-08-03 05:45:24 / 05:56:20 UTC；10m55.90s |
| smoke v8 result SHA | `b674180826bece52c327e22e56220c9749f15b0be019b29f56e228b8432473f1` |
| formal v6 start / finish | 2026-08-03 05:57:33 / 07:16:21 UTC；1h18m48.05s |
| formal v6 method-selection SHA | `a6a9351819b54c26fc3421b65f6e74752b61e3f9dc34a12ec5c1a8a746793b33` |
| v7–v6 trajectory source | `simulator_action_replay_from_input_t` |
| future RGB read | false |
| success outcome read | false |

## 实施验证（非科学结果）

| 检查 | 结果 |
| --- | --- |
| Thought4 CPU/mock | 46 passed |
| Thought1–4 全项目回归 | 443 passed；5 条 NVML 环境 warning 不影响测试结论 |
| 文档校验 | 2026-08-03：86 个 Markdown；全部本地链接有效；`docs/` root clean |
| smoke dry-run | PASS；Torch/model/simulator/write 均为 false |
| formal dry-run | PASS；Torch/model/simulator/write 均为 false |
| simulator-replay render-only | PASS；2 states×4 conditions；Clean/Lighting label SHA 逐 state 相同；replay 后 state 精确恢复；audit SHA `30ea849e...94664`；未加载 Fast-WAM |
| 真实 GPU smoke | v1/v2/v4 **ENGINEERING FAILED**；v3/v6/v7/v8 **PASSED / NON-SCIENTIFIC**；v5 **INTERRUPTED / RESUME BUG** |
| 正式 diagnosis | v1/v4 **ENGINEERING FAILED（pre-model）**；v2/v3 未运行；v5 **ENGINEERING FAILED（post-probe）**；v6 **FORMAL COMPLETE** |

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
它没有 Robot-init condition，不是 formal 数据，也不能解锁当前 formal v4。

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

## Smoke v4 失败、v5 中断与 v6 修复（非科学结果）

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
differs 为 2/2；没有加载 Fast-WAM。

v5 于 2026-08-01 06:56:51 UTC 启动，完成 2×4 paired render 和 8 条 label，
06:57:51 进入 `model_load_started`，随后外部中断，没有 feature、probe、
intervention 或 `smoke_result.json`。08:59:31 对 v5 resume 时，static
pre-validation 因内存 tuple 与 JSON 回读 list 直接比较而错误拒绝相同工件，最终
状态为 `error`。该 attempt 只提供前半链路工程记录，没有科学结果。

v6 把配置工件规范化为 canonical JSON 数据类型并增加 write/read/resume 回归。
v5→v6 除 experiment name/output namespace 外协议完全相同；formal v3→v4 同理。
因为修复改变 project commit，v5 不得继续 resume 或手工修补。

## Smoke v6 通过记录（非科学结果）

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_smoke_v6` |
| project commit / GPU | `aeb02106c48389d49bd7cac693e68113fa7d245a` / 2 |
| 时间 / 总时长 | 2026-08-01 10:07:40–10:18:42 UTC / 11m02s |
| 覆盖 | 2 base states × 4 conditions；80 feature records |
| model load | 409.889 s |
| 主干 SHA before / after | `ac0dd59...b4f8` / `ac0dd59...b4f8` |
| identity replacement | PASS；action L2=0 |
| Robot-init | input differs 2/2；simulator state differs 2/2；object layout matched 2/2 |
| future RGB / success read | false / false |
| result SHA | `b260977ae826e8c860074bd3402a3914dbc52e3f887cc090dad5ff3be2bc4c37` |

v6 完成了当时协议定义的全部技术 Gate，`formal_unlocked=true`，但仍不是科学结果。

## Formal v4 alignment 失败与 64-state 只读审计（非科学结果）

| 字段 | 值 |
| --- | --- |
| Run ID | `phase4_geometry_action_diagnosis_v4` |
| project commit / GPU | `aeb02106c48389d49bd7cac693e68113fa7d245a` / 2 |
| 时间 | 2026-08-01 10:21:25–10:22:28 UTC |
| 停止状态 | 第 2 个排序 base state；模型加载前 |
| 失败 state | `episode_000031@t34`；development split |
| 错误 | translation 0.031214 m、rotation 2.153°；超出 3 cm / 15° 中的平移阈值 |
| paired manifest / feature / probe / intervention | 0 / 0 / 0 / 0 |
| 科学结论 | 无 |

随后对同一原始 64-state cohort 做只读 simulator alignment audit；未读取模型、
future RGB、success、OOD 或 policy outcome，也未写入/修改 v4 工件：

| QC 指标 | translation (m) | rotation (degree) |
| --- | ---: | ---: |
| mean | 0.0237362120 | 3.079129 |
| median | 0.0185068857 | 2.072967 |
| p90 | 0.0311675769 | 4.056327 |
| p95 | 0.0600255217 | 8.792372 |
| max | 0.1083243997 | 28.917588 |

总计 56/64 通过、8/64 失败；失败按 split 为 train 5、development 3、test 0。
失败状态为 `episode_000031@t34/t31`、`episode_000033@t101`、
`episode_000018@t116/t106`、`episode_000036@t92`、`episode_000014@t111`、
`episode_000008@t127`。t±2 诊断不支持统一 off-by-one 解释；6/8 位于较长
prefix，另 2 条来自同一低对齐 episode 31。该审计只描述 replay fidelity，不是
policy 或方法效果。

## simulator-replay v5 预注册与执行

在上述失败与审计之后，冻结以下单变量修复：

- 保留原 64 个 episode/frame/split/sample identity，不过滤 8 条失败，不补样；
- 3 cm / 15° 阈值不变，从 hard gate 改为完整 QC disclosure；
- 从模型实际输入状态 `t` 重放 actions `a_t...a_{t+H-1}` 生成 `t+1...t+H`
  运动标签；
- Clean/Camera/Lighting 共享一次 Clean world replay，只做 camera transform；
- Robot-init 从自身 `t` 状态独立 replay；
- 不读取 future RGB、success、OOD、policy outcome，不改变 probe/layer/seed/threshold；
- 新身份为 smoke v7 / formal v5，必须先通过新的 smoke；旧 v6 PASS 不复用。

smoke v7 在 commit `229a0f383d7638b1919aa6a08f5aa3ea999a5cfe`、GPU 2
完成：2 个 base states、8 条四条件样本、80 条 feature、2/2 alignment pass，
backbone SHA 前后相同，raw identity replacement action L2=0；结果 SHA 为
`9d81d79afa9f3efcadf1a015f596f33e60414198b72f7c1e6cfa5a1322a1fbf9`。
该结果是有效技术 Gate，但没有覆盖 subspace projection/reconstruction。

formal v5 随后运行并保留以下工程证据：

| 字段 | 值 |
| --- | --- |
| Run ID / GPU | `phase4_geometry_action_diagnosis_v5` / 2 |
| 时间 | 2026-08-03 03:03:08–04:13:40 UTC |
| config fingerprint | `7b2a8e7ba6a51fe5246599324b09983d8df19824bfc906d7e8fd3932276fbb3a` |
| 已完成 | 64 base states；256 paired render；256 labels；12,544 feature records；Video/Action probe panel 在内存完成 |
| alignment | 56 pass / 8 fail；全部保留；内部 audit SHA `78c181b7dbd094fbb6f420b69523e162079725e09ea266b9278c0122d15dd925` |
| 失败位置 | 第一条 geometry-subspace intervention 的 correct reconstruction |
| 错误 | `correct geometry reconstruction exceeded BF16 tolerance` |
| 科学输出 | 无 intervention、evidence、method、integrity、report；无 classification |

旧代码只在 intervention 之后写 probe JSON，所以 v5 工件中没有可登记的
`video_probe_results.json` 或 `action_probe_results.json`。v5 目录冻结，禁止覆盖、
补写或 resume。

## FP32 subspace smoke v8 / formal v6 结果

formal v5 失败后只登记以下工程修复：

- FP32 coordinates/projection/residual/reconstruction，最终只 cast 一次 BF16；
- correct control 必须 `torch.equal`、input/output SHA 相同、max-abs=0；
- smoke v8 用真实 BF16 cache 和真实 consumer replacement 覆盖该路径；
- probe panel 在 intervention 前原子落盘并由 `probe_stage_result.json` 冻结；
- 新 formal v6 namespace，不覆盖、不 resume v5；
- 64-state cohort、probe、层、seed、threshold、统计和方法规则完全不变。

上述预注册已按顺序完整执行。smoke v8 的真实 BF16 input/output SHA 相同、
max-abs=0、bitwise=true；formal v6 在 intervention 前成功提交 probe stage，随后
完成 36 个 matched intervention 和冻结方法分类。完整数字与完整性审计见
[formal v6 正式结果](formal_v6_results.md)。

结果后只读审计重算了 artifact manifest 的 1,586 个文件，0 mismatch。唯一 caveat
是 `execution_integrity.json` 的内部 SHA 只覆盖先生成的核心字段，没有在追加
smoke/probe/alignment telemetry 后重算；完整文件仍由 artifact manifest 的 file
SHA 覆盖。原 v6 不修改，结果登记为有效 formal diagnostic，并披露该 self-hash
scope 缺陷。

## Table A：Video geometry readability

| Module/layer | Pool | Probe | Target | Clean RMSE | Camera gap | Lighting gap | Robot-init gap | Camera lower-min / upper-max |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `mot.video_kv_cache.15.v` | spatial mean | Linear，3 seeds | EEF–object camera translation | 0.032814 m | +0.020273 m | +0.011660 m | +0.001583 m（非 exact） | 0.002092 / 0.040799 m |

同时记录 zero、target-mean、shuffled-label control；不能只展示最优 MLP。
所有指标使用反标准化后的原始目标单位；另从每条 probe row 登记 train-only
normalizer SHA、constant-dimension 数和 best development epoch。

## Table B：Action motion readability

| Module/layer | Denoise step | Probe | Target | Clean error | Mean control | Shuffle control | Camera gap | Lighting gap |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `action_expert.blocks.15.norm1` | 19（第 20 步） | Linear，3 seeds | current EEF–object geometry | 0.021851 m | 0.061369 m | 0.077118 m | +0.023903 m | +0.002909 m |
| `action_expert.blocks.15.norm1` | 19（第 20 步） | Linear，3 seeds | future SE(3) composite | 0.105583 | 0.197027 | 0.225015 | +0.051516 | +0.012175 |

## Table C：geometry-subspace intervention

| Layer | Rank | Weight energy | Feature energy mean | Norm ratio range | Correct–shuffle action L2 mean | Above floor |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `mot.video_kv_cache.15.v` | 3 | 100% | 0.1043% | [0.999999894, 1.000000061] | 0.000768 | 36/36 |

记录 target/donor episode、mapping SHA、action noise/schedule/preprocessing SHA 和
每个 timestep 差异。

| Selected feature | Camera coordinate L2 | Lighting coordinate L2 | Camera−Lighting paired 95% CI | Robot-init（非 exact） |
| --- | ---: | ---: | --- | ---: |
| layer-15 cache V / spatial mean | 0.295093 | 0.148809 | [0.088519, 0.200310]；estimate 0.146284 | 0.096476 |

## 唯一结论

```text
classification: camera_equivariance_gap
recommendation: Geo-REPA + relative pose / camera-ray equivariance
```

`method_selection.json` schema/SHA 已验证。论文表述必须是：“冻结表征诊断支持
camera-equivariance gap”，而不是“方法提高 OOD success”。

## 简历可用工程表达

> 为 5B Video DiT + Action DiT 策略构建冻结式 geometry–action gap 诊断栈，
> 在 64 个 base state / 256 条四条件样本上训练 1,272 组轻量 probe，并以
> rank-3 matched intervention 验证 36/36 动作敏感；定位 Camera shift 的几何
> 破坏显著高于 Lighting，冻结分类为 camera-equivariance gap。

这里的 `1,272` 是 Video 1,080 + Action 192 probe rows，不是 1,272 个独立 episode；
简历中必须保留“冻结诊断、未做新方法 success rollout”的范围。
