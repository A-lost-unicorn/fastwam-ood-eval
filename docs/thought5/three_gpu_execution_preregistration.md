# Thought5 三卡执行调度预注册

预注册日期：2026-08-04（smoke v4、pilot v3 和 formal 均未运行）

科学角色：**execution-only，非科学结果**

硬件边界：本服务器固定只有 3×RTX 4090；每个 worker 独占一张物理卡。

## 1. 修改原因与边界

原执行器把 pilot 写死为 2 卡、formal 写死为 4 卡。smoke v3 已在 commit
`0346872f76d947503747271ce0942e88c47f979b` 完整通过，但该 commit 无法在本机
启动 formal。为避免先跑 pilot、再改执行器导致 smoke/pilot/formal commit 证据链
断裂，本次在任何有效 pilot 之前预注册 3 卡调度。

本次只允许改变：

- 同时启动多少个独立单卡 worker；
- variant 到物理 GPU 的确定性映射；
- 全新的 smoke/pilot 输出 namespace。

以下变量逐项冻结、不允许改变：模型 checkpoint、Fast-WAM commit、样本 identity、
task/episode split、seed namespace、训练目标、lambda、LR、optimizer、训练步数、
gradient accumulation、checkpoint 选择、future K、action denoise steps、rollout
horizon、统计门槛和停止规则。每个 worker 内仍只看到 `cuda:0`，不会使用 DDP、
跨卡梯度同步或改变浮点运算顺序。

## 2. 既有工件处置

- smoke v3：`status=complete`，耗时 630.780 s，结果 SHA-256
  `d9eedfb0b1f31fe7a1ed297a27b9e4a58c90e4ab4886b3913d31e8aefe40764d`；只读保留。
- pilot v2：仅开始 task-0 render，约 6 秒后收到 `KeyboardInterrupt`；状态为
  `error`，没有训练、checkpoint、utility 或 rollout。其 `run_status.json` SHA-256
  为 `3b41ba2fd72d162f2f0650f6fce41ce8d5724573082e25d61d066fe93c22526b`；
  不 resume、不解释、只读保留。
- smoke v4 与 pilot v3：使用新目录和包含本预注册的同一个 clean commit。
- formal：继续使用从未运行的 v2 科学候选 config；只有 pilot v3 的完整方向门禁
  通过并封存 `formal_protocol_frozen.json` 后才能启动。

namespace 改名之外的配置等价性由回归测试逐字段检查。冻结 identity 为：

| 阶段 | Config fingerprint | Cohort semantic SHA |
| --- | --- | --- |
| smoke v4 | `603e3394ab3ea521a6a54a3cfd5b9e753ae6c6c8c5dd761f4c1875e9be592bf1` | `4c9903f38af549218355f9781b3097d89d98069a7388e615fed491a8c9011035` |
| pilot v3 | `b5817d12a2b791dfe9fae093ad7b1b30039ce77492ee319d0f3255a0ff042631` | `16ee9c53ff9542da6ac9f46b26e3d17fd81c9e7ab14f6638b23b9fad5d5fdac0` |
| formal v2 | `87d11a6b1fdfde08793ff21f0a364686ea781d3f1f129c81853d3d0bd6ef77ca` | `d17a967aa04fa3ceb6447e150361dbab8110adde120bb875aab4e6094106f6c3` |

## 3. 冻结进程波次

### Pilot

三卡是本机主路径：

| 阶段 | GPU 0 | GPU 1 | GPU 2 |
| --- | --- | --- | --- |
| matched track | B1 | G3 | G4 |
| future calibration | B1 | idle | idle |
| future utility | B1 | G3 | G4 |
| paired rollout | B1 | G3 | G4 |

两卡只保留为兼容路径：第一波 B1/G3，第二波 G4；utility 和 rollout 使用相同
2+1 波次。任何结果都必须包含完全相同的 variant 集，不能因卡数减少而删除 G4。

### Formal

| 波次 | GPU 0 | GPU 1 | GPU 2 |
| --- | --- | --- | --- |
| matched track 1 | B1 | G1 | G2 |
| matched track 2 | G3 | B0 | idle |
| future calibration | B1 | idle | idle |
| future utility | B1 | G3 | idle |
| rollout 1 | B0 | B1 | G1 |
| rollout 2 | G2 | G3 | idle |

四卡只作为兼容路径：第一波 B1/G1/G2/G3，第二波 B0。formal 三卡与四卡必须
产生同一 variant 集和相同逐 variant seed；区别只能是 wall-clock overlap。

## 4. 三卡 ETA 预注册

ETA 是容量规划，不是结果，也不是停止规则。它基于以下已测数据：smoke v3
10.513 分钟；Thought3 28-sample Adapter 更新均值 17.295 秒；按 pilot 8 个训练
identity 线性折算为 4.94 秒/update，按 formal 144 个 identity 折算为
88.95 秒/update；Fast-WAM action chunk 单次约 3 秒。

| 阶段 | 三卡 wall-clock 估计 | 主要工作量 |
| --- | ---: | --- |
| smoke v4 | 10–15 分钟 | 单卡，复验同一真实 BF16/gradient/checkpoint contract |
| pilot render/cache | 5–15 分钟 | 16 base states × 4 conditions |
| pilot tracks + feature bundles | 30–60 分钟 | B1/G3/G4 各 100 update，同波 |
| pilot future calibration | 10–20 分钟 | B1、8 samples × 32 flow slots |
| pilot future utility | 65–100 分钟 | 每个 backbone 的 A0/A1/AS 各 200 update，同波 |
| pilot rollout | 40–70 分钟 | 每 variant 16 episodes，最多 400 env steps |
| **pilot 总计** | **2.5–4.5 小时** | finalizer 另需约 1–5 分钟 |
| formal render/cache | 1.5–2.5 小时 | 216 base states × 4 conditions |
| formal tracks + feature bundles | 3–7 小时 | 四个 400-update trainable tracks，3+1 波次；另有 B0 |
| formal calibration | 1–1.5 小时 | B1、144 samples × 32 flow slots |
| formal future utility | 16–20 小时 | B1/G3 并行；每个 backbone 三条 200-update Adapter |
| formal rollout | 7–14 小时 | 5 variants × 192 episodes，两波；成功可提前结束 episode |
| **formal 总计** | **约 28–45 小时** | 只有 pilot gate 为正才允许运行 |

正式 ETA 必须在 pilot 完成后使用各 worker 的真实 `elapsed_s`、episode latency 和
成功终止长度重新计算。不得因耗时超出估计而减少样本、step、flow slot、variant
或 rollout horizon。

## 5. 门禁与停止规则

1. 提交实现后 worktree 必须 clean。
2. smoke v4 必须 `status=complete` 且 `pilot_unlocked=true`。
3. pilot v3 必须与 smoke v4 的 project commit 完全一致。
4. pilot 任一 collector、G4 specificity 或 artifact 校验失败即停止；不得启动 formal。
5. formal 必须与 pilot v3 freeze 的 project commit、config fingerprint 和 cohort SHA
   完全一致。
6. 中断只允许在相同 commit、相同 namespace、checksum-valid 工件上使用
   `--resume`；旧 smoke v3/pilot v2 不得 resume 到本协议。
