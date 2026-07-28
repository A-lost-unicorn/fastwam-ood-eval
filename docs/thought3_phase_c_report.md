# Thought3 Phase C：单卡真实 tensor/backward/memory 验收报告

状态：**Gate C 通过**
验收日期：2026-07-28
证据等级：`SMOKE`
Run ID：`P3-PHASE-C-v1`

## 1. 结论

Phase C 已在一张 NVIDIA GeForce RTX 4090 上使用一条真实标准 LIBERO
training demonstration 完成。以下工程门禁全部通过：

- 官方 Fast-WAM checkpoint 能在单卡加载并保持冻结；
- 同一当前观测可生成配对初始噪声的 K=1/2/4 原生 future latent；
- video-only sampler 与上游 joint video path 在同输入下逐元素一致；
- zero-init Adapter 在初始状态不改变 current-only action；
- 一次官方 action flow-matching loss backward 有限，且 backbone 无梯度；
- Adapter 输入不读取真实未来 RGB，future-RGB mutation 不改变生成 latent；
- 执行峰值显存和模型加载峰值均低于 43 GiB 硬上限；
- 没有创建 optimizer、没有 optimizer step、没有写真实 cache、没有启动长训练。

这只证明 **Phase D/E 的技术可行性**。它不能证明 Adapter 能收敛、future 能改善
OOD、K 越大越好，或本次单样本 latency 能代表正式分布。

## 2. 冻结来源

| 项目 | 冻结值 |
| --- | --- |
| 项目分支 | `feature/thought3-partial-future-adapter` |
| Phase C 实现 commit | `5c7d9a84a1058f1ca1d01641d02810eae102ea2a` |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| dataset stats SHA-256 | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| model config SHA-256 | `ab3c2ffde9933e7576c747fecce82bd7d28c9c6478c1b53fcac02b3012be416c` |
| LIBERO revision | `117413dc0ca99c7cd64036c4eaa4a316c537d692` |
| dataset archive SHA-256 | `a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea` |
| config fingerprint | `9c5e062bceb9d0bdda25de86c130e35de6dfb72688ae71b12913cc9826d3edaf` |
| physical/logical GPU | GPU 1 / `cuda:0`，visible device count = 1 |

真实样本来自 `libero_goal`：episode 0、frame 0，任务为
`open the middle drawer of the cabinet`。数据集 inventory 为 433 episodes、
52,895 frames；本次只读取其中一条样本。

## 3. Tensor 与 sampler 结果

当前帧 VAE latent 为 `[1,48,1,14,28]`、BF16。三个 future 输出均为
`[1,48,2,14,28]`、BF16、finite，且共享同一 initial-state SHA-256：

`158b9aaa5d10cffd7fa97a1a36cbaf6e33ee13ab9cdf479f8acaae22ac7f5e82`

| K | sigma nodes | 单次采样 latency | 执行峰值显存 | future SHA-256 |
| ---: | --- | ---: | ---: | --- |
| 1 | `[1,0]` | 120.34 ms | 12.602 GiB | `90a7040580b59fbf2b39eb40abb098b8dc4a5145db65e56ce51ad50d1b687654` |
| 2 | `[1,0.833333,0]` | 165.62 ms | 12.602 GiB | `a5feba90e8bc53c4ad964acffbac55d4eb3706e2374e1ff753c73f861f90e2c0` |
| 4 | `[1,0.9375,0.833333,0.625,0]` | 325.30 ms | 12.602 GiB | `d8044d70dc007c977b168582c5b468bfc878d1954bf8c5c8a88ecd17f638ecf7` |

这些 latency 是单样本 CUDA 工程遥测，不含模型加载、数据读取和完整在线 action
denoising，也不是 P50/P95。模型加载本身为 427.06 s，加载峰值 23.125 GiB。

## 4. 上游等价性与 zero gate

| 检查 | 结果 |
| --- | --- |
| video-only sampler vs upstream joint video path | max/mean absolute difference = `0/0` |
| current-only action vs upstream full joint action | max `0.015625`，mean `0.0019098`，通过 `atol=rtol=0.01` 的 combined allclose |
| 当前帧 VAE latent parity | max absolute difference = `0` |
| zero-gate Adapter vs current-only action | bitwise equal，max absolute difference = `0` |

`current-only` 对 `full joint` 的最大差值大于 0.01；通过原因是
`atol + rtol × |reference|` 的 combined tolerance，不能误写成 “max < 0.01”。

## 5. Backward、冻结与显存

- action loss：`0.00027467653853818774`，仅用于 finite/backward 工程检查；
- Adapter 参数：`1,371,137`；
- backward：1 次；microbatch：1；
- optimizer：未创建；optimizer steps：0；
- backbone gradient count：0；
- MoT 参数 hash 在 backward 前后完全相同：
  `48df1e64927298451acae7766c6178ac838a5561d22675527be2bbba96cf6a49`；
- backward latency：721.94 ms；执行峰值显存：12.772 GiB；
- 本次最高执行阶段：full-demo VAE parity encode，12.964 GiB；
- 所有阶段均低于 43 GiB。

zero gate 初始为 0，因此第一步只有标量 `gate` 获得非零梯度：
L2 `1.9073486328125e-05`。其余 Adapter 参数都有 finite grad tensor，但数值为
0；这是 gated residual 的预期链式梯度。Phase E 必须在 gate 被 optimizer
打开后，显式确认 projector/attention 等非 gate 参数开始获得非零梯度。

## 6. 信息泄漏门禁

- sampler API 只接收当前 latent、language context/mask 和 sampled noise；
- `future_frames` 被 batch schema 拒绝；
- training API 报告不使用 GT future RGB；
- 在保持当前帧不变时替换 demonstration 后续 RGB，K=1 future hash 不变；
- 本次没有序列化 future latent。

所以 Gate C 支持的准确结论是：**当前实现的 future 输入来自模型对当前观测的
采样，不来自真实后续 observation。**

## 7. 冻结机器工件

| 工件 | SHA-256 |
| --- | --- |
| `run_status.json` | `581de5813e11fd19c8d7a1433c511c1a32e896900f62677ac3e47330d3f3bc33` |
| `gate_c_result.json` | `ccac9ac39fd7920dc89726313b89a3ae16ab71b5494b072d0b6c6ba6778d3f02` |
| `logs/phase_c.log` | `f09670e9e5bd8bdb9ddd51653d71f7f5759c8f51cc3cb079a1c95993c5e648d2` |

权威路径为 `outputs/thought3/phase_c_single_sample_v1/`。该目录受 `.gitignore`
保护；上表 hash 是文档与机器工件之间的只读锚点。

## 8. 卡点与修复

首次 backward 在 FP32 Adapter 的 query `LayerNorm` 接收 BF16 Fast-WAM hidden
时触发 dtype mismatch。修复是在进入 FP32 query normalization 前显式把 action
hidden 转为 FP32；zero-gate residual 返回时仍转换回原 action dtype。新增回归
测试覆盖 BF16 输入与 bitwise zero-gate identity。失败尝试没有创建 cache、
optimizer 或训练 checkpoint。

LIBERO 视频读取环境中 TorchCodec 未能加载 FFmpeg 动态库，loader 自动回退到
torchvision/PyAV 并成功完成真实样本读取。Phase D inventory 和 cache manifest
需要显式记录该 decode backend；正式大规模 cache 前应固定 PyAV 版本。

Phase C 收口后的仓库全量回归为：

```text
238 passed, 5 warnings in 29.97s
```

5 条 warning 均来自受限测试进程无法初始化 NVML。Thought1/Thought2 八个只读
SHA-256 哨兵与 Phase B 冻结值完全一致。

## 9. 下一门 Gate D

Gate D 只允许：

1. `libero_goal` 的一个 task；
2. 完整 episode identity 上的 90/10 train/development split；
3. 约 32 个当前观测样本；
4. K=1/2/4 BF16 latent cache；
5. build、resume、checksum、paired-noise、shape、泄漏和吞吐验证。

Gate D 通过前不启动 100–500 step Adapter 训练。
