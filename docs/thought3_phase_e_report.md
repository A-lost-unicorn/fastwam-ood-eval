# Thought3 Phase E：真实 Adapter 小训练验收报告

状态：**Gate E 未通过；工程子门禁部分通过**
验收日期：2026-07-28
证据等级：`SMOKE / FAILED-GATE`
数据范围：标准 `libero_goal` task 0，28 train / 4 development，单卡

## 1. 结论

Phase E 已真实运行 A0/A1 Adapter 训练，不是 CPU mock。当前可以确认：

- 官方 Fast-WAM backbone 全部冻结，optimizer 只包含 1,371,137 个 Adapter
  参数；
- A0/A1 使用相同初始化、sample 顺序、action noise/timestep、训练预算和
  checkpoint 频率；
- zero-init gate 的第 1 step 只有 gate 获得非零梯度；
- 从第 2 step 起，future projector、position/norm 路径和 attention
  Q/K/V/out 参数均出现 finite、nonzero gradient；
- A0/A1 都完成了 50→100 的真实 checkpoint/optimizer resume，以及独立
  uninterrupted 100-step 重放；
- 启用确定性 CUDA 配置后，两条轨迹的最终 Adapter semantic SHA 逐位相同；
- Adapter-only checkpoint 的保存、checksum、optimizer restore 和 round-trip
  均通过；
- 单步峰值约 12.96 GiB，模型加载峰值约 23.12 GiB，低于 43 GiB；
- 训练输入仍为当前 RGB/proprio、action supervision 和 model-generated cache，
  没有 future RGB、success 或 OOD outcome。

但 Gate E 整体不能标记为通过：

1. v2 中 A1 的固定 development action loss 没有低于初始化；
2. 为避免把“泛化提升”混入工程门禁，v3 新增固定 train-probe，但 A0 的该
   probe 也没有低于初始化；
3. 因 Gate 在训练后 frozen hash 之前 fail-closed，尚缺一次完整的
   frozen-parameter before/after 等值闭环；
4. 因此当前不能扩到 A2/A4，更不能启动 ID/OOD rollout。

这不是“未来无效”的结论。它只说明当前 `lr=1e-3、100 step、zero-gated
Adapter` 配方虽然梯度链路和恢复机制正确，但尚未展示稳定、可诊断的 loss 改善。

## 2. 冻结来源

| 项目 | 值 |
| --- | --- |
| 分支 | `feature/thought3-partial-future-adapter` |
| Phase D 收口 commit | `ba8a46f45f7d8ea28608171a7ded8d999e360559` |
| Phase E 初始实现 | `eb5ec8a15c0ce2bece7de33acfb71958f62281a2` |
| provenance 修复 | `9b51179` |
| progress callback 修复 | `2b42964` |
| 确定性 CUDA v2 | `c4fcadb` |
| train-probe v3 | `dc77bd2` |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| dataset stats SHA-256 | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| cache fingerprint | `63a70e1af38f68bc894fc11d03c84f212e6c6328a5051256c9d045741156d9c5` |
| split fingerprint | `ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb` |
| GPU | physical GPU 1 / logical `cuda:0` |

v3 当前配置 SHA-256 为
`2acabae32aadb5e2abeacf82032147e4c6baa3816a9731a0108f8fa7d102b726`。
该配置仍为 A0/A1 各 100 optimizer step、microbatch 1、LR `1e-3`、
weight decay `1e-2`、step 25 checkpoint、seed 3407。

## 3. 三次 fail-closed 尝试

| Run | 目的 | 到达范围 | 结果 |
| --- | --- | --- | --- |
| v1 | 首次接入真实训练 | A0 100 + 独立 A0 100 | 先修复 split 字段和 callback；随后发现默认 CUDA 重放存在极小 backward 差异，semantic SHA 不同，拒绝 |
| v2 | 强制确定性 CUDA | A0/A1 各 resumed 100 + uninterrupted 100 | 梯度、resume、SHA、显存通过；A1 development loss 未下降，拒绝 |
| v3 | 分离 trainability 与泛化 | A0 resumed 100 + uninterrupted 100 | 固定 train-probe 未低于初始化，拒绝；未重复 A1 |

v1 的差异从 step 2 的极小 gradient reduction 差异开始。修复方式为：

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
torch.use_deterministic_algorithms(True)
TF32=false
cuDNN benchmark=false / deterministic=true
Flash SDP=false
memory-efficient SDP=false
math SDP=true
```

v2/v3 在正式 100-step 前还对 A0/A1 各执行两次独立 2-step 内存重放。
两次 semantic SHA 分别完全一致：

- A0：`d1708ec00f29f57235d9677900166e69ab59d01bc5306cea27cf87ac209870f4`
- A1：`cf897e8c13dd59e39c998328c5858b49378b237d2ddb7dee91d1e451d140f218`

## 4. 梯度门禁

v2 的第 1 step：

| Variant | action loss | gate grad L2 | non-gate nonzero elements |
| --- | ---: | ---: | ---: |
| A0 | 0.0358901 | 0.0209961 | 0 |
| A1 | 0.0358901 | 0.000146866 | 0 |

初始 action loss 逐值相同，符合 zero gate 下 A0/A1 都等于未注入 action
路径。gate gradient 可以不同，因为它取决于各自尚未生效的 residual。

第 2 step：

| Variant | projector nonzero | attention nonzero | all non-gate nonzero |
| --- | ---: | ---: | ---: |
| A0 | 256 | 1,315,327 | 1,321,983 |
| A1 | 49,408 | 1,315,294 | 1,371,102 |

A0 的 projector weight 输入为零，因此主要由 bias 路径获得梯度；A1 的真实 K1
latent 使 projector weight/bias 都获得梯度。两组所有记录的 gradient、loss、
gate 和 memory 都是 finite。

所以用户指定的核心检查——“gate 打开后其他参数是否获得非零梯度”——已经通过。

## 5. Resume、checkpoint 与确定性

v2 的真实最终 Adapter semantic SHA：

| Variant | resumed 50→100 | uninterrupted 0→100 | 是否相同 |
| --- | --- | --- | --- |
| A0 | `67d0735fb6f226e65d33d977edb50ffc47877b6d20081836d4ed7dedd3b2ae00` | 同左 | 是 |
| A1 | `f327127e2f268cf13a17e5df6ad0e944505c6665ade1b3c9af5f7137fb3a8fb1` | 同左 | 是 |

这里比较的是 tensor semantic SHA。两个 safetensors 文件本身可以因 manifest/config
metadata 不同而有不同文件 SHA，不能用文件字节 hash 代替权重语义比较。

每条轨迹保存 step 25/50/75/100 checkpoint，内容只有：

- `adapter.safetensors`；
- `optimizer.pt`；
- provenance/checksum manifest。

checkpoint 不包含 backbone；round-trip 后 Adapter semantic hash 相同，optimizer
state 非空。A0 development-only 选择 step 100，A1 选择 step 75。

## 6. Loss 结果

### 6.1 v2 development

| Variant | initial | best | final | final−initial | 相对变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.01848339 | 0.01831786 | 0.01831786 | −0.00016553 | −0.896% |
| A1 | 0.01848339 | 0.01852790 | 0.01853071 | +0.00004732 | +0.256% |

development 只有 4 个 episode。上表用于工程诊断与 checkpoint 选择，不允许写成
A0 优于 A1、future 有害或 OOD 结论。

### 6.2 v3 固定 train-probe

v3 使用按冻结 sample order 选出的 4 个 train sample，并固定
`evaluation_step_base=80000`，保证每次使用相同 action noise/timestep。

| Variant | initial | step 75 | final step 100 | final−initial |
| --- | ---: | ---: | ---: | ---: |
| A0 | 0.00157760 | 0.00158690 | 0.00159928 | +0.00002168 |

该 probe 没有下降，所以 v3 在 A0 后停止。v2 在线训练日志的首/末 25-step 均值
虽然 A0/A1 分别下降约 5.75%/5.45%，但不同 step 使用不同 noise，不能替代固定
probe，也不能用于把 Gate 改判为通过。

## 7. 数据与信息泄漏

每次 run 都重新校验 Phase D 的七个冻结工件 SHA 和完整 cache checksum。数据准备
必须同时满足，否则在训练前抛错：

```text
base samples                    32
train / development             28 / 4
current camera frames decoded   64
action chunks / rows            32 / 1024
current state rows              32
future RGB frames decoded       0
actual future read              false
uses ground-truth future        false
```

训练 API 中没有 future RGB、next observation、success、termination 或 OOD outcome
字段。A0 使用同 shape 全零 latent；A1 只读取 Phase D 的
`source_kind=model_sampled_from_current` K1 latent。

## 8. 延迟与显存

| 指标 | 实测 |
| --- | ---: |
| 模型加载峰值 | 23,679.51 MiB |
| optimizer step 峰值 | 13,273.17 MiB |
| A0 resumed mean step | 662.97 ms |
| A1 resumed mean step | 662.13 ms |
| trainable parameters | 1,371,137 |

step time 包含 deterministic math SDP 下的一次 forward/backward/update，不是在线
20-step action denoising latency，也不是论文中的 ID/OOD policy latency。

## 9. 冻结机器工件

### v2

| 工件 | SHA-256 |
| --- | --- |
| `run_status.json` | `0631de121b683d0c78a2154476c2768d5a33ef1ae87f28c856716f95845fd56f` |
| `logs/phase_e.log` | `73f227a7db560521261e5985424383194d421b14bb277f9c30d0a76583cb29c1` |
| A0 resumed manifest | `4a5f3c041d67a808cbc3926b3a69137f812924edf7c6ad450d0e226d77bcaf6f` |
| A0 resumed metrics | `a36b43c53633b01c788789d36ba56b1c1bab729cd1139ce6226f31733b67db01` |
| A1 resumed manifest | `b4308f9ee2a4ca16ea00374fb2c75b614bbafad0e043ace7fae48b462c457a95` |
| A1 resumed metrics | `943a4f66ecb82cd8bd89738a6dfe877a34d290f273184b587e4864ce89f795e8` |

### v3

| 工件 | SHA-256 |
| --- | --- |
| `run_status.json` | `ab19301eeaf572b6750389f4de0862d1641669da3b8c02089e3fa7ea6b65bc53` |
| `logs/phase_e.log` | `248fde0261029083ee5bcbca4e91ccd7ede96bb981eb2e56498f19421c679678` |
| A0 resumed manifest | `dbe14f4d32fa8e3ff2bec75411042499d216ce286683be7a2aff749f22b613c6` |
| A0 resumed development/probe metrics | `effd5a6936bc12e40d63416867bf9d89d57d87e472dc5be3c530ab61c5a3c476` |
| A0 uninterrupted manifest | `d8cf00bbe2bff00b555191395233a47dc39744994d90dbef144246dfed3565f3` |

权威目录为：

```text
outputs/thought3/phase_e_training_smoke_v1/  # nondeterminism diagnosis
outputs/thought3/phase_e_training_smoke_v2/  # complete A0/A1 engineering traces
outputs/thought3/phase_e_training_smoke_v3/  # fixed train-probe failure
```

三个目录都必须保留。不要删除失败 run 后复用相同 Run ID。

## 10. 下一步：Gate E.1 优化诊断

在扩到 A2/A4 前，先做不涉及 OOD outcome 的最小诊断：

1. 单条标准 LIBERO train sample、固定 action noise/timestep，分别让 A0/A1
   重复优化 100–300 step，验证是否能 overfit；
2. 记录 gate gradient 符号、gate/residual scale、BF16 注入前后实际 hidden
   delta，以及 projector/attention gradient-to-parameter ratio；
3. 若单样本不能明显下降，优先检查注入尺度、gate optimizer group 和 loss
   连接，不扩大数据或 K；
4. 若单样本能下降，再用 8-sample train-only probe 比较 `1e-4/3e-4/1e-3`
   等预先列出的少量 LR，不读取 development/OOD 结果来挑配方；
5. 冻结一个配方后重新跑 28/4 Gate E，并确保 frozen hash after 与 before
   完全相同；
6. 只有 Gate E 完整通过才解锁 A2/A4 和 Phase F。

当前阶段最重要的负面工程结论是：**梯度能传播，不等于当前优化配方已经学到稳定
有效的 future-to-action 修正。**
