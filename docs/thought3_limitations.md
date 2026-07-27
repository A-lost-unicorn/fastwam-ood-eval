# Thought3 当前限制与未决问题

状态：Phase B 完成后的诚实边界
更新时间：2026-07-27

## 1. 尚无真实阶段三结果

当前没有：

- 真实 Fast-WAM future latent；
- 真实 action loss backward；
- 真实 Adapter checkpoint；
- ID/OOD success rate；
- GPU latency/显存；
- A-shuffle 机器人 rollout。

Phase B 的 mock loss、mock success 和 CPU latency 都是 `TEST`，不支持 future
改善 OOD 的科学结论。

## 2. Native latent contract 仍需运行时确认

`[B,48,2,14,28]` bf16 是由冻结配置和上游代码静态推导的 contract。Phase C
必须在官方 checkpoint 和一条真实 train sample 上确认：

- VAE/current shape；
- full diffusion state `[B,48,3,14,28]`；
- future tail；
- dtype/device；
- 无 VAE decode/re-encode；
- current slice 每 update 不漂移。

## 3. Video-only sampler parity 未关闭

项目侧 K-step loop 已实现完整 shifted schedule与固定 seed，但尚未绑定真实
Video DiT，也未与 upstream joint path 做数值 parity。若 parity 失败，必须先修正
condition、cache、mask 或 scheduler 调用，不能用“看起来合理”的 latent 进入训练。

## 4. Action loss 只完成 mock 语义

mock trainer 使用 action velocity MSE 验证 Adapter 能收到梯度、loss 能下降和 resume
一致。真实 Phase C 必须逐项对照官方：

- timestep/sigma sampling；
- noisy action 公式；
- velocity target 符号；
- action pad mask；
- loss weighting/reduction；
- normalization/denormalization。

在 parity 前不能称“保持官方 action loss 已实证通过”。

## 5. 训练数据尚缺

当前缺标准 LIBERO 正式 LeRobot training directories 和不可变 revision/inventory。
下载后还需确认：

- license 与分发边界；
- demonstration identity；
- frame/action alignment；
- suite×task episode 数足以做 90/10 split；
- 数据没有混入 LIBERO-Plus/test trajectory。

## 6. Hook 依赖上游调用次数

第一版通过 `action_encoder` output hook 避免改 `third_party/FastWAM`，但真实模型可能在：

- gradient checkpoint recompute；
- torch compile；
- 多次 action sub-call；
- DDP wrapping

下改变调用次数。exact-call guard 会 fail-fast，而不是静默多注入；Phase C 要确认训练和
推理各自的合法 count。

## 7. Zero gate 的优化动力学

gate=0 保证初始 parity，但第一步主要只有 gate 获得梯度，projector/attention 信号会在
gate 打开后出现。真实 100–500 step smoke 需要观察：

- gate 是否离开零；
-各子模块 grad norm；
- Adapter 是否只学会常数偏置；
- null/correct/shuffle action sensitivity。

gate 保持接近零是允许结果，不应在正式 OOD 上反复调到“有效”。

## 8. A-shuffle 的部署语义和成本

正式 shuffle 不能读取训练 cache。为了保持计算公平，donor current 也必须在线运行同 K。
这需要预注册 donor observation 获取方式，并区分：

- recipient policy latency；
- donor future generation latency；
-研究性反事实总成本。

A-shuffle 是因果干预诊断，不是可部署策略。

## 9. Cache 容量只完成公式与 mock 校验

纯 bf16 latent 为约 220.5 KiB/sample（K1+K2+K4）；真实 metadata、mask、文件系统和
inventory 规模未知。只有下载数据后才能报告实际总 GiB、throughput、inode 和 20%
空间余量。

## 10. 3×4090 仍需真实验证

现有 Thought2 shadow 峰值约 24,841 MiB，不能直接推断 action backward。Phase C
单卡硬上限 43 GiB；Phase E 后才决定是否需要 gradient checkpoint、CPU offload、
FSDP 或 ZeRO。第一版不同时启用多个回退手段。

## 11. Statistical protocol 仍是 DRAFT

primary K、train seed 数、smallest effect、multiplicity、formal job manifest 和
checkpoint selection 仍未冻结。Phase F 之前保持
`thought3_analysis_protocol_DRAFT.md`；看到 Phase G 结果后再改协议将使分析降级为
post-run exploratory。

## 12. 可能的负面科学结果

以下都不是工程失败：

- A0 与 B0 相近，A1/A2/A4 无提升；
- A0 提升但 future 没有增量；
- K4 比 K1 更慢且更差；
- correct/shuffle 都改变动作但 success 不变；
- future 在 camera/robot-init 下放大错误；
- 部分 task 获益、部分受损；
- OOD 增益不足以覆盖 latency。

论文必须保留这些可能性。
