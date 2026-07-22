# 思考点二上游审计：Fast-WAM 未来预测—动作—真实环境变化一致性

审计日期：2026-07-22  
思考点一基线 commit：`0df5fe224e5c5dd767ed105802821b69c141e041`  
保护 tag：`thought1-baseline-v1`  
实现分支：`feature/thought2-shadow-diagnostics`  
Fast-WAM commit：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`

## 1. 审计结论（实现门禁）

1. 当前 release `libero_uncond_2cam224` checkpoint 可以加载为上游 `FastWAM`，而 `FastWAM.infer_joint()` 具有完整的视频扩散、VAE 解码和视频返回路径。因此它在依赖和基础 Wan 权重齐全时**能够生成 future video**。
2. 但是 release 配置明确设置 `model.video_dit_config.action_conditioned: false`。在该结构中，`infer_joint(action=executed_actions)` 虽然接受并检查 action shape，`WanVideoDiT.pre_dit()` 却不会把 action embedding 加入视频分支；FastWAM 的 MoT mask 也没有 video-to-action 通路。因此该 release 生成的是**当前图像、语言和 proprio 条件下的 unconditional future**，不是 action-conditioned future。
3. 所以本仓库不能把当前 release 的视频称为“给定已执行动作的未来预测”，也不能用它运行正式的 `mode=action_conditioned_future`。真实 probe 必须检查 `model.video_expert.action_conditioned is True`；不满足时明确失败。不能通过把配置强改为 `true` 绕过，因为 release checkpoint 没有匹配训练过的 action-embedding 参数。
4. 32 个 action 的直接 cross-attention 条件确实分为两个 16-action group，但 pinned `video_attention_mask_mode=first_frame_causal` 允许所有 future latent 彼此注意。后组读取的 action 16--31 可经后续层或 denoising step 间接影响前组；所以任意 future frame 的安全依赖闭包是完整 action 0--31，而不是只需首组 0--15。当前 `control_horizon=10` 因此远小于所需的 32。
5. 上游 `load_checkpoint()` 对 `mot` 使用 `strict=False`，所以单看运行时 `action_conditioned=true` 也不足以证明条件参数来自匹配训练。真实 probe 还必须校验 checkpoint 内 action-embedding key/shape/value，并要求 checkpoint hash、Fast-WAM commit 和训练 recipe 位于源码审阅的 matched-checkpoint allowlist；当前 allowlist 为空。
6. 这些门禁不会阻止实现和 CPU mock 验证独立 Shadow Diagnostics 基础设施，但会阻止当前 release 越过单卡模型 smoke 的科研语义门禁。若仅研究 unconditional future，应另立研究模式、问题和报告口径，不能静默降级。

## 2. 已审计材料

- `README.md`
- 用户指定的 `docs/thought1_protocol.md` 在当前 commit 不存在；审计了 README 实际链接的等价协议 `docs/thought1_generalization.md`
- `src/fastwam_ood_eval/policy/fastwam_adapter.py`
- `src/fastwam_ood_eval/evaluation/episode_runner.py`
- `src/fastwam_ood_eval/recording/episode_trace.py`
- `src/fastwam_ood_eval/schemas/episode_result.py`
- `third_party/FastWAM/src/fastwam/models/wan22/fastwam.py`
- `third_party/FastWAM/src/fastwam/models/wan22/wan_video_dit.py`
- `third_party/FastWAM/experiments/libero/eval_libero_single.py`
- `third_party/FastWAM/experiments/libero/libero_utils.py`
- `third_party/FastWAM/configs/sim_libero.yaml`
- `third_party/FastWAM/configs/task/libero_uncond_2cam224_1e-4.yaml`
- `third_party/FastWAM/configs/model/fastwam.yaml`
- `third_party/FastWAM/configs/data/libero_2cam.yaml`
- Fast-WAM processor、normalizer 以及 LIBERO 默认控制频率实现

## 3. `FastWAMAdapter.act()` 完整调用链

加载阶段：

```text
FastWAMAdapter.__init__
  -> Hydra compose(configs/sim_libero.yaml, task=libero_uncond_2cam224_1e-4, ...)
  -> hydra.instantiate(upstream_cfg.model, dtype, device)
  -> 校验实例类为 FastWAM
  -> official._load_model_checkpoint -> model.load_checkpoint
  -> instantiate(upstream processor).eval()
  -> load_dataset_stats_from_json
  -> processor.set_normalizer_from_stats
```

每次重规划：

```text
FastWAMAdapter.act(observation)
  -> torch.inference_mode()
  -> official._predict_action_chunk(...)
       -> DEFAULT_PROMPT.format(task=task_description)
       -> official._obs_to_model_input(...)
       -> model.infer_action(...)
       -> official._denormalize_action(...)
       -> gripper [0,1] -> [-1,1]
       -> invert_gripper_action
       -> optional sign binarization
  -> PolicyOutput(actions=<denormalized executable numpy chunk>, telemetry=...)
```

`episode_runner.run_episode()` 随后只取 `list(output.actions)[:control_horizon]`，逐个调用 `environment.step(action)`。因此 `PolicyOutput.actions` 是已经完成官方后处理、可直接交给 LIBERO 的动作，不是模型归一化 action token。

## 4. Observation 与相机预处理

官方 `get_libero_image()` 从 observation 读取：

- `agentview_image`
- `robot0_eye_in_hand_image`

两幅图都执行 `[::-1, ::-1]` 的 180° 翻转以匹配训练数据。`_obs_to_model_input()` 再按 processor 的每相机 `shape_meta` 做保持覆盖范围的 center-crop resize；release 的两个相机各为 `224 x 224`，并按 `concat_multi_camera: horizontal` 拼成 `224 x 448`。最后变为 `[1, 3, 224, 448]`，转换为模型 dtype/device，并从 uint8 映射到 `[-1, 1]`。

诊断实际帧必须复用这一路径；不能直接拿 simulator 原图和预测的拼接图比较。

## 5. Proprio 预处理

官方 `_extract_sim_state()` 拼接：

```text
robot0_eef_pos (3)
+ quat2axisangle(robot0_eef_quat) (3)
+ robot0_gripper_qpos (2)
= 8 dims
```

`_normalize_proprio()` 使用 processor 的唯一 state key，依次调用：

```text
processor.action_state_transform
processor.normalizer.forward
```

得到 `[1, 8]` 的 normalized proprio。`infer_action()` / `infer_joint()` 再通过模型的 `proprio_encoder` 把它作为额外 context token。诊断 probe 必须复用 `_obs_to_model_input()`，不能自行重写 quaternion 或 state normalization。

## 6. Action normalization / denormalization 与 action 语义

`infer_action()` 返回 normalized model action `[T, 7]`。官方执行后处理为：

1. `processor.normalizer.normalizers["action"][key].backward()`，得到 dataset-space 动作。
2. gripper 从 dataset 的 `[0, 1]` 映射到 `[-1, 1]`。
3. `invert_gripper_action()` 再乘 `-1`，匹配 LIBERO 的 gripper 约定。
4. `EVALUATION.binarize_gripper=true` 时取符号。

因此 `FastWAMAdapter.act()` 返回的 executable action 不能直接传给 `infer_joint(action=...)`。对于当前 release（`action_state_transforms: null`），反向构造模型条件动作必须：

1. 复制动作，禁止原地修改执行 chunk。
2. 把 LIBERO gripper `g_env` 映射回 dataset gripper `(1 - g_env) / 2`；若执行前做过 binarization，这表示实际执行命令，无法恢复原始连续预测值。
3. 通过官方 processor 的 action normalizer `forward()` 转成模型 normalized action。

若未来配置启用 `action_state_transforms`，probe 必须通过 processor 的正式 forward 变换，或在不能证明逆向语义时拒绝运行，不能猜测。

## 7. `infer_joint()` 真实签名和输出

Pinned API 为：

```python
infer_joint(
    prompt,
    input_image,
    num_video_frames,
    action_horizon,
    action=None,
    proprio=None,
    context=None,
    context_mask=None,
    negative_prompt=None,
    text_cfg_scale=1.0,
    num_inference_steps=20,
    sigma_shift=None,
    seed=None,
    rand_device="cpu",
    tiled=False,
    test_action_with_infer_action=True,
) -> dict[str, Any]
```

输出是：

```text
{
  "video": list[PIL.Image.Image],  # uint8 RGB, 由 VAE decode 后 clamp 到 [-1,1] 再映射
  "action": torch.FloatTensor[T, action_dim]  # CPU normalized action
}
```

`action=` 在源码注释中的意图是“用于 conditioning videos 的 ground-truth action，而不是 action expert 的目标”。实际是否生效取决于 `video_expert.action_conditioned`：

- `true`：`WanVideoDiT.pre_dit()` 将 action embedding 加入视频 cross-attention context，并按 temporal group 构造 mask。
- `false`（当前 release）：action 被 shape-check 后传下去，但视频分支忽略它。

诊断不使用 `infer_joint()` 返回的 action，且应设置 `test_action_with_infer_action=False`，避免无意义地额外再运行一次 `infer_action()`。

## 8. `num_video_frames`、`action_horizon` 与约束

- release 数据配置 `num_frames: 33`：动作/状态时间长度为 33 个 observation slots，默认 action horizon 为 `33 - 1 = 32`。
- `FastWAMAdapter.action_horizon` 优先取 `EVALUATION.action_horizon`，为空时同样使用 `data.train.num_frames - 1 = 32`。
- release `action_video_freq_ratio: 4`，官方 `_get_num_video_frames()` 计算 `(33 - 1) / 4 + 1 = 9`。
- 视频输入要求 `T % 4 == 1`，所以 9 合法；非法值必须在调用前报错，不能依赖上游自动修正。
- action condition 要求 `[T_action, 7]` 或 `[1, T_action, 7]`，且 `T_action == action_horizon`。action-conditioned video expert 还要求 action horizon 可被 VAE latent future transitions 整除。

注意：环境通常只执行 action chunk 前 `control_horizon=10` 个动作，而 `infer_joint()` 的 action condition shape 仍要求完整 32 步。进一步审计 `WanVideoDiT.pre_dit()` 发现，action-conditioned 路径不是逐视频帧独立读取动作：9 个视频帧经 temporal factor 4 的 VAE 形成 3 个 latent frame，首 latent 不读 action，余下 2 个 temporal group 把 32 个 action token 分成每组 16 个。也就是说，解码视频帧 1--4 所在的第一个 future latent group 会读取 action 0--15。

但 16 只是**直接**条件组。Pinned `first_frame_causal` video self-attention 只禁止首 latent 读取 future；future latent 之间仍是双向的。每层先做 video/mixed self-attention，后做 action cross-attention，因此后组读取的 action 16--31 可从下一层或下一 denoising step 传播回前组。真实依赖闭包为：frame 0 不依赖 action；所有 future frame 都可能依赖完整 action `[0,32)`。所以把 `control_horizon` 从 10 提到 16 仍不安全，只有完整执行 32 才能闭合该 mask 下的依赖。若模型明确使用 `per_frame_causal`，第 g 组才可按前缀 `[0,(g+1)*group_size)` 判定；未知 mask 一律拒绝。

同时，第一版只支持 temporal DiT patch size 为 1。大于 1 时首 patch 会同时含固定首 latent 与 future latent，当前没有经过证明的逐解码帧 action 依赖映射，必须 fail closed。不能把思考点一的 control horizon 改成 16/32 来绕过，因为那会改变基线控制语义。

## 9. 视频帧与控制步的时间对应

上游不是“一帧等于一步”。训练配置明确给出 `action_video_freq_ratio=4`，官方 evaluator 也按它在执行后的第 4、8、12……个 action step 捕获实际帧：

```text
predicted frame 0 -> action 前的当前 observation，offset 0
predicted frame 1 -> 执行 4 个 action 后，offset 4
predicted frame 2 -> 执行 8 个 action 后，offset 8
...
```

LIBERO 环境默认 `control_freq=20 Hz`，在运行时仍应从环境读取/验证；若保持默认值，相对时间是 `offset / 20` 秒，即每个预测视频间隔约 `0.2 s`。如果运行时 control frequency 不可获得或与默认值无法核验，manifest 必须写 `approximate_alignment=true`，不能静默声称精确 wall-clock 对齐。

对 `control_horizon=10`，单纯从 observation capture 时间看，一个完整重规划段最多具有 offset `0, 4, 8` 三个候选帧；episode 提前成功或异常时还要进一步截断。但“帧在第几步采集”与“该预测帧只条件于实际执行动作”是两项不同检查。上一节的完整 32-action 依赖闭包不通过时，这三个候选帧不能进入正式 action-conditioned 一致性指标。

## 10. VAE 编码和解码

Pinned `FastWAM` 暴露的内部官方包装是：

- `_encode_video_latents(video_tensor)` -> `self.vae.encode(video_tensor, device=self.device, ...)`
- `_encode_input_image_latents_tensor(input_image)` -> 对首帧调用同一 VAE encode
- `_decode_latents(latents)` -> `self.vae.decode(...)`，随后 clamp `[-1,1]`、映射到 uint8，并返回 PIL 帧列表

只计算 pixel MSE 不满足协议，但也不能把当前可对齐的三帧 clip 直接整体送入 `_encode_video_latents()`：Wan VAE encoder 使用 `1 + (T - 1) // 4` 个 chunk，`T=3` 时只编码首帧，后两帧会被静默忽略。原生 temporal latent 的首个非平凡合法长度是 5 个模型视频帧，而当前 `control_horizon=10` 只能收集模型帧 0、1、2（环境 offset 0、4、8）。因此第一版必须逐帧调用同一 VAE 的 `_encode_input_image_latents_tensor()`，把结果明确标为 `approximate/reencoded_frame_embedding_without_temporal_context`；不能称为 native diffusion latent likelihood。若以后比较完整 temporal latent，clip 长度必须满足首帧加 4 的倍数并完整覆盖相同行为条件。

## 11. Release checkpoint 的视频生成能力

本地 release checkpoint：

```text
path: checkpoints/fastwam_release/libero_uncond_2cam224.pt
size: 12,041,735,140 bytes
sha256: 1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579
class: FastWAM
```

已有思考点一真实记录证明该 checkpoint 能加载并生成 `[32,7]` action chunk；本轮审计没有自动运行 GPU future smoke。源码和官方 evaluator 的 `visualize_future_video` 路径证明该结构具备 video generation 调用链，但“本机实际生成并保存视频”仍属于阶段 C，不能在未运行前写成实验事实。

结论应严格表述为：**release 支持 unconditional future generation；不支持已执行动作条件下的 future generation。**

此外，Pinned loader 的 `mot.load_state_dict(..., strict=False)` 不报告 missing keys。Shadow probe 因此不能接受“只把 Hydra flag 改成 true”的模型：它会通过 `torch.load(..., map_location="cpu", mmap=True, weights_only=True)` 只核对 checkpoint 中的 `mixtures.video.action_embedding.*` key、shape 和加载后的精确值，并同时要求项目源码 allowlist 中已有匹配的 checkpoint SHA-256、Fast-WAM commit、model config 与 training recipe。当前仓库没有这样的可信 checkpoint，allowlist 有意保持为空。

## 12. RNG、缓存与模型状态副作用

- `infer_joint(seed=..., rand_device="cpu")` 分别创建局部 `torch.Generator(...).manual_seed(seed)` 生成 video/action 初始噪声，不应消费全局 CPU/CUDA generator。
- `seed=None` 时 generator 为 `None`，会消费所选 `rand_device` 的全局 RNG；诊断禁止这样调用。
- `infer_joint()` 调用 `self.eval()`。Adapter 加载后本来就是 eval，因此当前无状态变化；通用 probe 仍应记录/恢复训练标志或要求 eval model。
- `infer_action()` 的 video KV cache 和 joint denoising tensors 都是局部变量。审计未发现写入持久模型 cache 的代码。
- CUDA kernel、processor 或未来上游改动仍可能引入不可见随机消耗，因此 Shadow Diagnostics 仍必须用 `RngIsolation` 保存并恢复 Python、NumPy、Torch CPU 和所有可见 CUDA RNG state，并用 action hash 做端到端回归。
- CUDA allocator/peak-memory counter不是 RNG。future probe 会改变进程内已分配/保留显存与 peak counter；诊断应独立记录 probe 的增量/peak，不把它冒充思考点一 policy memory。

## 13. 对正式实现的约束

1. 保持 `episode_runner.py`、`BasePolicy.act()`、`PolicyOutput`、已有 eval YAML、job ID identity 完全不变。
2. 独立 diagnostic runner 复用 job manifest、environment、policy 与 resume 规则，但写入新的 experiment output；source experiment 只读。
3. `FastWAMFutureProbe` 只持有已加载 Adapter，复用 `official._obs_to_model_input()`、processor、model 与 upstream config。
4. probe 在真实调用前同时检查 class/API、action shape/dtype、`T % 4 == 1`、`action_video_freq_ratio`、camera concat、output range，以及最关键的 `video_expert.action_conditioned is True`。
5. 对 action-conditioned 模型，还必须从实际 VAE temporal factor、DiT temporal patch、video length、attention mask 和 action horizon 推导传递依赖闭包。Pinned `first_frame_causal` 下所有 future frame 都要求完整 32-action horizon，而基线只执行 10；temporal patch 不等于 1 或未知 mask 也必须拒绝。
6. `action_conditioned=true` 不是充分证据；必须同时通过源码 allowlist 的训练 provenance 与 checkpoint action-embedding 实值加载验证。当前 allowlist 为空。
7. 当前 release 触发 action-conditioning 能力门禁是预期结果，不得自动降级为 unconditional future；也不得通过修改思考点一执行 horizon 绕过条件覆盖门禁。
8. `static_motion_threshold=1.0` 是首版写入 manifest 的初始 representation-space 阈值，尚未经过 no-op/静止轨迹校准；阶段 C/D 未通过前不能把 static flag 当作物理静止结论，latent direction 也不是光流方向。
