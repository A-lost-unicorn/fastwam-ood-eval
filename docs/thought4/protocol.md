# Phase 4 冻结研究协议

## 1. 唯一问题

官方 Fast-WAM 的 future latent 已被证明会改变动作，但 K=1 Adapter 没有改善
held-out action objective。Phase 4 定位缺口在 Video geometry、Video→Action
interface、camera equivariance，还是几何假设本身不成立。

## 2. 冻结项

- checkpoint：
  `checkpoints/fastwam_release/libero_uncond_2cam224.pt`；
- checkpoint SHA：
  `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`；
- Fast-WAM commit：
  `45d8e1458921d83f8ad6cf9ce993d371208dabd0`；
- Video layers：`0, 7, 15, 22, 29`；
- Action hooks：encoder output、block 15 input、block 29 input、head input；
- action denoising：20 steps；
- action horizon：32；
- cohort：40 train / 12 development / 12 test base states；每 episode 最多 2 帧，
  因而分别覆盖 20 / 6 / 6 个互不跨 split 的 episode；
- split unit：episode；
- task：`libero_goal` task 0；
- conditions：Clean ID 1；Camera panel
  `[691,697,698,706,711]`；Lighting panel
  `[2313,2314,2334,2337,2351]`；Robot-init panel
  `[282,283,284,285,294]`；
- exact-state：只允许 Clean/Camera/Lighting；
- linear/MLP、优化器、epoch、seed、bootstrap 均在 YAML 中冻结；
- intervention target：
  `eef_object_translation_camera`；
- layer selection：只读 development，只在 Action 直接消费的 Video K/V 中，
  按全部冻结 probe seeds 的 mean normalized loss 选择；tie 选更晚层，再按
  module path/feature key 固定；basis 使用第一组冻结 seed；
- test/OOD/success 不参与样本、层或 checkpoint 选择。

真实 smoke 是技术 Gate，不共享 formal 科学结果：只取 2 个 base state、
Clean/Camera/Lighting 三条件、Video layer 15（hidden/K/V）和 Action block 15
input。smoke 还必须让 source-A identity replacement 通过 matched replay floor；
replacement 固定使用 `mot.video_kv_cache.15.v` consumer argument，与 formal
候选边界同类。它不训练
正式 panel，也不生成方法选择。

## 3. 数据生成

LeRobot demonstration 只读取 action、EEF state、episode/frame identity；不读取
future RGB。输入状态 `t` 通过 official init state + demonstration action prefix
恢复，动作 gripper 从数据约定 `g∈[0,1]` 转为 LIBERO 的 `1−2g`。
prefix 后的 Clean EEF 必须与 parquet 第 `t` 帧在 3 cm / 15° 的预注册宽松阈值内
一致，否则时间对齐 fail closed；实际误差和阈值写入 label manifest。Camera 与
Lighting 复用该 exact state。Robot-init 不与 Clean EEF 对齐，而以自身状态生成
未来 simulator-replay 标签，并明确标记 alignment 不适用。

Clean 状态恢复后，将完全相同的 flat MuJoCo state 注入 Camera 与 Lighting
variant。每条记录保存 simulator、RGB、depth、EEF/object、camera、lighting
SHA。variant 由 split seed、task、episode、frame 的哈希确定，不读实验结果。
Robot-init 使用自身状态，`exact_state_pair=false`；它先执行相同 demonstration
action prefix 恢复到时间 `t`，再从该 Robot-init 状态离线重放未来 action 以生成
自身的 EEF trajectory 标签，不能套用 Clean 的未来 EEF。

在执行任何 prefix 前，collector 对所有 task object/fixture 的 position+quaternion
做排序快照：Robot-init 必须在 `1e-7` 容差内与 Clean 保持同一 object layout，
reset robot state 只作披露，不要求此时已经不同。原因是 LIBERO
`set_init_state()` 会把共享 demonstration flat state 写回 simulator，暂时覆盖
variant 的可观测 qpos。真正的硬检查位于相同 action prefix 执行后的模型输入时刻
`t`：Clean/Camera/Lighting 的 robot state 必须相同，Robot-init 的
joint/EEF/gripper state 与完整 simulator-state SHA 必须不同。reset 与 input 两套
SHA/match 标志都写入 label manifest；这不把 Robot-init 伪装成 exact-state pair。
三种 exact-state condition 的 input observation 必须从同一 Clean flat state 走
相同 observable-refresh 路径，禁止混用 step-return cache 与 regenerated snapshot。

动作 prefix 只用于诊断状态恢复，不执行 policy，不读取/记录/筛选 success。

## 4. Phase 4-A

Source A 捕获：

- `video_expert.blocks.<i>.norm1` input；
- Action 消费端 `mot.forward_action_with_video_cache` 收到的同层最终 K/V cache，
  语义路径 `mot.video_kv_cache.<i>.k/v`。

当前帧应为 `[1,98,3072]`。pooling 路由：

| Probe | Pooling |
| --- | --- |
| low-res relative depth 7×7 | spatial / foreground |
| relative camera translation/rotation-6D | spatial |
| EEF–object translation/orientation（camera/world） | spatial / robot-object ROI |

报告 Linear、MLP、shuffled-label、zero、train-target-mean；指标包括 depth
AbsRel/RMSE/δ1/rank correlation、translation RMSE、rotation geodesic error、
condition gap、exact-state paired gap 和 episode-grouped bootstrap。

为避免不同 Video/Action 层的 hidden 尺度、米/角度/轨迹维度尺度污染 probe
优化与 development 层选择，所有 probe 都只用对应 train split 拟合逐维
feature/target mean 与 standard deviation（下限 `1e-6`）。development/test/OOD
不参与统计量拟合；预测先反标准化回原始标签单位，再计算全部指标。每组统计量
的 SHA 写入 probe row。target-mean baseline 同样只使用 train 中 valid-mask 为真的
标签。shuffled control 先冻结原 train statistics，再把 label 与 mask 成对置换。
置换使用无固定点的 deterministic derangement，避免少量样本保留自身标签。

## 5. Phase 4-B

Source B 在每个 action denoise step 触发。v1 冻结取第 20 步
（零基 `denoise_step_index=19`），再对 32 action token 做 temporal mean。
该 step identity 显式进入 feature schema/shard/probe row。标签来自
`t+1…t+32`：

- camera-frame translation trajectory；
- rotation-6D；
- gripper；
- `[xyz, rot6d, gripper]` SE(3) trajectory。

episode 尾部使用 valid mask，绝不跨 episode。指标为 RMSE、ADE/FDE、rotation
geodesic、gripper MAE/accuracy/F1 和披露分量的 composite。

为直接回答 Video geometry 是否进入 Action hidden，同一 source-B feature 还
增加时间 `t` 的 camera/world-frame EEF–object translation probe。该 probe 只
训练轻量 probe，不把 geometry label 输入 Action DiT；它与未来 SE(3) probe
分别报告，不能混合不同单位后再挑“最优”目标。

## 6. Phase 4-C

只做一次 geometry subspace intervention。候选仅限 Action DiT 直接消费的
Video `mot.video_kv_cache.<i>.k/v`；`norm1` hidden 仍报告 probe，但不参与干预层选择。
线性 probe 权重 `W` 做 SVD，取达到
95% weight energy 且 rank≤32 的正交 basis `U`。只替换：

```text
z = h U
h_res = h - z Uᵀ
h_shuffle = h_res + norm_match(z_donor) Uᵀ
```

donor 必须同 task、不同 episode、尽量匹配 progress bin，并是固定 seed 的一一
derangement。correct/shuffle 固定 observation、language、proprio、action seed、
initial noise、denoise schedule、checkpoint 和 preprocessing。输出动作 L1/L2、
cosine、translation/rotation/gripper、per-timestep、trajectory change 和 replay
floor。

`denoise_schedule_sha256` 覆盖 scheduler 类型与完整 config、steps、sigma shift、
horizon 和 rand device。每个 seed 的 correct reconstruction 不仅检查 hidden
误差，也必须在动作空间低于 `max(1e-6, 2×replay_floor+1e-8)`；shuffle 只有超过
同一容差才计为 above floor。

同时分别记录 SVD 截断后的 probe `explained_weight_energy`，以及被选 hidden 在
该子空间内的实际投影能量比 `explained_feature_energy`，不得混称。
对同一被选 pooled feature，还在 held-out exact-state pairs 上投影 Clean/Camera/
Lighting 坐标，报告 Camera/Lighting 相对 Clean 的 L2 与 Camera−Lighting
episode-grouped CI；Robot-init 只作非 exact-state 的独立坐标距离。该步骤复用已
抽取 feature，不增加模型调用。

最终 evidence 同样先按跨 seed mean development loss 选定 feature group，再汇总
所有 seed；不能挑单个最优 seed。完整 SE(3) trajectory 的可读性使用已披露的
translation ADE + rotation geodesic/180 + gripper MAE composite，而不是只看平移。
Camera paired gap 只有在所有 probe seed 的 bootstrap lower bound 均大于 0 时才
记为 significant；Camera>Lighting 也要求所有 seed 同方向。

由于 linear probe 在标准化空间训练，SVD 使用换算回原始 hidden/原始 target
坐标的有效权重 `diag(target_std)·W_norm·diag(1/feature_std)`；因此 intervention
定义的是原始 captured hidden 中的 probe-readable geometry subspace，而不是由
归一化尺度任意改变的子空间。

## 7. 方法选择

方法选择规则在
`src/fastwam_ood_eval/thought4/decision.py` 冻结；输出 schema 只接受四种分类。
正式报告必须同时列出能证明与不能证明的内容。特别是 Phase 4-C 不能证明
correct geometry 提高 success。
