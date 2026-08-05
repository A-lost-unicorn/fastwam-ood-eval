# Thought5 代码审计与 Fast-WAM-GeoEq 方法

## 审计结论

审计通过，未发现阻断 Phase 5 的缺失路径或标签。机器生成的完整审计见
`outputs/thought5/phase5_audit_report_v2.md`。关键事实如下。

- Thought4 冻结特征为 `mot.video_kv_cache.15.v`，真实生产模块为
  `video_expert.blocks.15.self_attn.v`。
- 张量形状为 `[B,98,3072]`。98 个 token 是单 latent frame 的 7×14
  frame-major/row-major 网格，宽度方向包含 7 个 agent-view 和 7 个 wrist-view token。
- `MoT.prefill_video_cache` 生成该 K/V；
  `MoT.forward_action_with_video_cache` 在 Action block 15 实际消费它。
- Video/Action DiT 各 30 层；第一版只在 Video layer 15 的 K/V projection
  安装 rank-8 LoRA，不训练 Action DiT。
- simulator 可稳定给出 RGB、metric depth、intrinsic、camera-to-world
  extrinsic、其逆变换、EEF/object pose。Clean/Camera/Lighting 可以 exact-state
  配对；Robot-init 改变物理状态，必须独立报告。
- Thought3 的 Adapter 结构、AdamW、`3e-4`、200 step、seed 3407、动作目标和
  fixed-step checkpoint 规则复用；旧 runner 因冻结原 backbone SHA 且没有 AS
  分支，不能原样复用。
- Thought2 的纯 future distance / motion-direction metric 可复用，旧正式
  runner 的 checkpoint provenance 不可复用。

## 方法结构

```text
current RGB + proprio
        │
        ├─ camera intrinsic/extrinsic → token rays + relative pose
        │                              ↓
        │                       RayPoseEncoder ─┐
        │                                      │ residual injection
        └─ Video DiT → layer-15 K/V hidden ────┤
                                               ↓
                                   Action-consumed video cache
                                      │                  │
                                      │ inference        └─ training only
                                      ↓                         ↓
                                  Action DiT                GeoProjector
                                      ↓                         ↓
                                  action chunk        detached geometry targets
```

GeoProjector 从 3072 维 action-consumed token 预测：relative depth、camera-frame
3D point、world-frame 3D point、world-frame EEF-object translation。目标由
simulator depth/rays/extrinsic 构造并 detach。RayPoseEncoder 输入每 token ray
和 12 维 relative pose；K=1 video sampling 的 294 个 frame-major token 复用同一
98-token camera field。它不读取 future RGB 或 GT depth。

Phase 5-B 不直接比较 G3 训练过的 GeoProjector 与 B1 未启用的随机 projector。
每个 backbone 都从 K=1 的 layer-15 future token 经过同一个 seed=5597 的冻结
signed random projection（3072→128），再独立拟合同容量 linear-ridge probe；
probe 只在 train 拟合，只用 development 从 `[1e-4,1e-2,1,100]` 选择 alpha，
formal target 不参与拟合、标准化或选择。这一修正确保 future geometry 改善不能
由“G3 多训练了一个 decoder”解释。

总目标为：

```text
L = L_original_fastwam
  + λ_repa L_geo_repa
  + λ_equiv L_equiv
  + λ_pose_aux L_pose_aux
```

`L_original_fastwam` 始终保留。B1 的三个辅助权重严格为 0；G1 仅 Geo-REPA；
G2 仅 pose/ray；G3 二者同时启用；G4 使用无 fixed point 的 geometry
derangement。B1/G1/G2/G3/G4 使用相同 LoRA、Projector、Encoder 参数预算，
避免“多参数”解释。

## 参数与部署边界

CPU mock 结构的 trainable 参数为 1,335,320，真实模型 smoke 必须重新记录真实
名称、数量、冻结数量、backbone SHA 和峰值显存。白名单仅允许：

- `video_expert.blocks.15.self_attn.k.{lora_A,lora_B}`；
- `video_expert.blocks.15.self_attn.v.{lora_A,lora_B}`；
- `geo_projector.*`；
- `ray_pose_encoder.*`。

adapter-only **训练 checkpoint** 保存 LoRA、RayPoseEncoder 和 GeoProjector，
用于可恢复训练和离线分析；正式 **部署图** 只执行 LoRA 后的 backbone 与
RayPoseEncoder，GeoProjector 不参与 action inference。正式推理输入只允许
current RGB、proprio、intrinsic 和 extrinsic；不加载 geometry teacher，不读取
simulator depth。

## 当前工程证据

- Audit：PASS，但不是科学结果。
- CPU/mock dry-run：14 项 contract 检查全部通过，1.140 s；mock 参数数
  1,335,320；RayPoseEncoder CPU 单次观测 1.579 ms。后两项仅为本机技术
  telemetry，不能替代 GPU P50/P95。
- launcher hotfix 候选全仓回归：501 tests 全通过（包含 58 项 Thought5 测试）。
- GPU smoke v2：B1 完成 2 step 后在 G3 的 FP32 pose auxiliary head 输入边界
  发生 BF16 dtype error；运行无效、无科学结论。
- GPU smoke v3：完整通过，B1/G3 各 2 step，总耗时 630.780 s，训练峰值
  25,216.0 MiB；冻结主干 SHA 前后一致，G3 LoRA/GeoProjector/RayPoseEncoder
  梯度均有限且非零，推理未读取 future RGB/GT depth。
- GPU smoke v4：三卡调度 commit 上完整通过，耗时 712.323 s，
  `pilot_unlocked=true`；仍只是技术门禁。
- pilot v2：仅 render 启动约 6 秒后 `KeyboardInterrupt`，无训练或结果。
- pilot v3：完成 16 个 base state/64 个 condition sample 的 render cache 后，
  B1/G3/G4 fresh worker 均因缺少 LIBERO import path 在权重加载前退出；无训练、
  checkpoint 或科学结果。launcher hotfix 的 smoke v5、pilot v4、formal：
  **NOT RUN**。

v2 权威标识：audit SHA `a880bbd9de9ad36dd1670b341a2bb17fc92a1428558e0b16d17e80f06c1bf959`；
formal config fingerprint `87d11a6b1fdfde08793ff21f0a364686ea781d3f1f129c81853d3d0bd6ef77ca`；
candidate protocol SHA `5be775e8b6dc62a77c517c5a6686aec1c139659e70911a7fa66e034df3f9fa58`；
cohort semantic SHA `d17a967aa04fa3ceb6447e150361dbab8110adde120bb875aab4e6094106f6c3`。
旧 v1 只含未运行 scaffold，保留且不得 resume；三卡执行边界和新 namespace
见[预注册](three_gpu_execution_preregistration.md)，不改变 v2 科学协议；pilot v3
失败审计和纯启动修复边界见[hotfix 预注册](worker_launcher_hotfix_preregistration.md)。
