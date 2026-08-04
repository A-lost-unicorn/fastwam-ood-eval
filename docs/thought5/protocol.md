# Thought5 冻结实验协议

## 数据拆分

formal candidate 使用 task-level 隔离：

| split | tasks | episodes/task | base states |
| --- | --- | ---: | ---: |
| train | 0–5 | 24 | 144 |
| development | 6–7 | 12 | 24 |
| untouched formal | 8–9 | 24 | 48 |

每个 base state 生成 Clean/Camera/Lighting exact-state 三元组和一个非
exact-state Robot-init specificity control。样本只由 task、episode、长度和冻结
hash 选择，不读取 success。以下历史样本明确排除：

- Thought3 development：task 0 episodes 3、28、29、32、37；
- Thought4 formal test：task 0 episodes 0、5、6、7、11、19。

pilot 仅用 task 0，但 train/dev/test 分别使用 8/4/4 个互斥 episode。formal
不能读取 pilot test outcome 选择层、loss、checkpoint 或停止时机。

## 对照组

| 组 | LoRA/训练 | Geo-REPA | Ray/Pose | geometry 对应 |
| --- | --- | --- | --- | --- |
| B0 | 无，官方 checkpoint | 无 | 无 | 不适用 |
| B1 | matched | 0 | 0 | 正确但不进入 loss |
| G1 | matched | 开 | 关 | 正确 |
| G2 | matched | 关 | 开 | 正确 |
| G3 | matched | 开 | 开 | 正确 |
| G4 | reduced pilot | 开 | 开 | 确定性错配 |

pilot 运行 B1/G3/G4；formal 运行 B0/B1/G1/G2/G3。动作分支去噪步数固定 20。

## H1：表征层

冻结 layer 15，训练 probe 只读 train，checkpoint/standardization 只读
development，formal cohort 只做一次评测。主要终点：

```text
GapReduction = (CameraGap_B1 - CameraGap_G3) / CameraGap_B1
```

H1 同时要求：相对减少 ≥25%；episode-grouped paired bootstrap 的 G3−B1
95% CI upper <0；task-cluster CI upper <0；Clean 不明显退化；Camera 改善大于
Lighting 一般性改善。报告每 task、episode/seed 与 aggregate。固定附加终点
包括 world-frame relation、depth/relative-depth、rank-3 shift、Action current
geometry 和 Action future SE(3)。

## Phase 5-B：K=1 future geometry

B1、G3（pilot 另含 G4）使用同 observation、K=1 noise seed 与 sampler schedule。
future–actual latent L1 和 motion-direction cosine 沿用 Thought2 指标；depth
relation、EEF–object relation 与 camera-frame 3D geometry 使用独立冻结 probe。

公平 probe 协议固定为：从 `mot.video_kv_cache.15.v` 的两个 future frame、每帧
98 个 token 取特征；用 seed 5597 的 signed projection 从 3072 降到 128 维；
linear ridge 只在 train cohort 拟合，development 只在
`[1e-4, 1e-2, 1, 100]` 中选 alpha，formal 只预测一次。每个 backbone 使用相同
输入/输出维数、projection、候选 alpha 和选择规则。G3 的训练期 GeoProjector
不作为该比较的 evaluator，避免和 B1 的 inactive head 形成不公平对照。

## H2：future utility

在 B1/G3 两个 backbone 上分别运行 A0(null)、A1(correct K=1)、AS(shuffle)，
复用 Thought3 Adapter 配方。每组必须共享 action noise、timestep、denoise
schedule 和 flow slot。

```text
Utility = Loss(A0) - Loss(A1)
Specificity = Loss(AS) - Loss(A1)
```

H2 同时要求 A1-G3 优于 A0-G3、A1-G3 优于 AS-G3、
`Utility_G3-Utility_B1` 的 episode-grouped 与 task-cluster 95% CI lower 均 >0，
且 A0-G3 不得比 A0-B1 异常恶化超过 5%。action changed 仅是 sensitivity，
不能替代上述 held-out loss 条件。

同时保存 20-step fully-denoised `[32,7]` action chunk 的 correct/null/shuffle
技术反事实：A1 bitwise replay、action SHA、L2/cosine 与 translation/rotation/
gripper change。material threshold 固定为
`max(1e-7, 10 × replay_p95)`，replay max-L∞ 必须 ≤`1e-5`。这些只回答
future content 是否真正进入动作，不能替代 H2 utility。

## H3：闭环

所有 checkpoint 和 manifest 冻结后，使用完全匹配的 task/environment seed 比较
Clean、Camera、Lighting、Robot-init。主要终点是
`Success_G3_camera-Success_B1_camera`。H3 同时要求 paired episode 与
task-cluster CI lower >0、Clean 下降不超过 5 percentage points、Camera 改善
大于 Lighting 一般性改善，并单独披露 latency/peak memory。

闭环语义固定为每 episode 最多 400 simulator step、reset 后等待 30 step、每次
执行 action chunk 前 10 个 control step、输入 256×256。B0/B1/G1/G2/G3 使用
完全相同 task/seed/init index；Clean/Camera/Lighting reset 后的物理 state SHA
必须逐项一致，Robot-init 单独报告。episode JSONL 支持校验后断点恢复；rollout
success 永不参与 checkpoint 选择。

## 最终分类

程序只允许：

- `full_mechanism_support`；
- `representation_only_support`；
- `utility_without_closed_loop_support`；
- `closed_loop_without_future_mediation`；
- `mechanism_not_supported`。

H1 失败、G4 与 G3 同等有效、或 B1 能解释全部提升时，必须登记
`mechanism_not_supported`。不得事后创造第六种更有利分类。

## 冻结与停止规则

- formal 前冻结 config、cohort、lambda、training steps、checkpoint rule、统计方法。
- 真实 smoke 记录 clean project commit；pilot 必须与 smoke 同 commit，formal
  必须与 pilot freeze 同 commit；跨 commit 的 partial output 禁止 resume。
- 物理卡数只允许采用[三卡执行调度预注册](three_gpu_execution_preregistration.md)
  中的固定 2/3 卡 pilot 与 3/4 卡 formal 波次；卡数不得改变 variant、seed、
  浮点图、训练量或统计规则。
- 只允许一次 hook/坐标 debug、一次 development-only lambda/stability 调整、一次 pilot recipe。
- formal 不得中途看 success 换 checkpoint 或提前停止。
- H1 formal 失败后，不继续堆 geometry module 追正结果。
- 所有 NaN/Inf、SHA、配对、seed 或 artifact mismatch 都 fail closed。

本页对应 v2 protocol。v1 仅生成过 `NOT RUN` scaffold；由于 v2 修复了
future-geometry evaluator 的 matched-probe 公平性，v1 不得启动或 resume。
