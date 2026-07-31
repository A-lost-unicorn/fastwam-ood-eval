# Thought4 卡点与排错

## robosuite 报 MUJOCO_EGL_DEVICE_ID 不在 CUDA_VISIBLE_DEVICES

单卡 runner 中两个变量都使用物理 GPU ID。例如物理卡 1：

```text
CUDA_VISIBLE_DEVICES=1
MUJOCO_EGL_DEVICE_ID=1
```

Fast-WAM 配置仍写 `cuda:0`，因为 PyTorch 会把唯一可见卡重映射为逻辑卡 0。
不要把 EGL ID 写成逻辑 0。Python preflight 会在模型/模拟器 import 前拒绝错误
映射。若旧 attempt 已经写出 `run_status.json`，保留旧目录，使用新 Run ID；不要
在代码 commit 已变化时 `--resume`，否则 pre-validation identity 必然不同。

## 首次 feature capture 报 inference tensor 没有 version counter

报错：

```text
Inference tensors do not track version counter.
```

这是 PyTorch `torch.inference_mode()` 的正常语义，不表示 checkpoint 损坏或显存
不足。旧 hook 直接读取 cache tensor 的 `_version` 来验证 20 次 action denoise
共用同一只读 K/V；inference tensor 不提供该 counter，因此在首次 capture 即停止。

当前 v3 hook 对普通 tensor 仍检查 data pointer + version；对 inference tensor 检查
data pointer、shape、dtype、device、stride，并在 scope 退出时把 live tensor 与
首次 detached clone 逐值比较。这样既兼容真实推理，也不会放弃原地修改检测。
对应回归同时覆盖“可正常 capture”和“发生 mutation 必须 fail closed”。

若旧 v2 已生成 `run_status.json`，不要 `--resume` 或删除旧目录。提交修复后直接运行
同一个 smoke runner；它已经指向新的 `phase4_geometry_action_smoke_v3` namespace。
v2 的 paired render/label 仅用于工程审计，不可拼入 v3 或正式结果。

## Hook 注册成功但 call count 为 0

原因通常是 hook 到 `blocks[i]`。MoT 不调用 block `forward()`。检查路径必须是
`norm1`、实际 `mot.video_kv_cache` consumer argument、`action_encoder` 或
`head`。禁止跳过 call-count
硬检查。

## Camera/Lighting state SHA 不等

不能继续做 paired gap。依次检查：

1. variant 是否属于同一 base task；
2. classification ID 是否是一基、传给 suite 时是否减一；
3. flat state shape 是否一致；
4. set state 后是否 `sim.forward()`；
5. render 前后 state 是否被 callback 修改。

不要改成“不同 episode 平均值”来绕过。

## target object body 找不到

task 0 冻结为 `wooden_cabinet_1`。collector 先查 `obj_body_id`，再查 MuJoCo body
name。两者都没有则 fail closed；不能用任意最近物体替代。

## ROI point 在相机后方

先验证 robosuite extrinsic 是 camera-to-world，且 projection 使用
`K @ inverse(camera_to_world)`。不要通过绝对值或裁剪 z 来掩盖坐标错误。

## Action hook 不是 20 次

确认 formal config 的 `action_denoise_steps=20`，官方 helper 没有切到 joint/future
接口，模型类型是 `FastWAM`。任一不符都不能混入正式 panel。

## 4090 OOM

正确顺序是 render → close all LIBERO env → load Fast-WAM。卡启动前显存必须
≤1024 MiB。不要降低层、样本或 action steps 来救同一 formal run；修复工程后用
新输出目录重新登记。

## Probe 看似很好但 shuffled control 也好

优先检查 episode 泄漏、相邻帧跨 split、label identity、mask 广播和 target mean。
方法选择规则要求 probe 同时优于 target-mean 与 shuffled control 5%。

## Formal 中断

保留 `run_status.json`、log、shard 和 sidecar SHA。completed 输出不可覆盖。
对非科学性中断可追加 `--resume`：runner 会先核对 stage 与 config fingerprint；
已有 feature shard 通过 shard sidecar、逐 feature SHA、identity/module/layer/
pooling/denoise-step 和当前冻结提取的逐项一致性后只读复用；`probe_inputs.pt`
也有独立 sidecar 和逐 tensor 对比。checksum 缺失、metadata/tensor 不同或已有
`status=complete` 都会 fail closed，不会静默重复、拼接或覆盖。

## Robot-init trajectory 与 Clean 不一致

这是预期而不是 pairing 失败。Robot-init 先在自己的机器人模型上重放相同 action
prefix 到 `t`，再离线执行 `t...t+H-1` 的 demonstration action，标签来源写为
`simulator_action_replay_from_robot_init_t`。若误用 Clean demonstration EEF，
会把不同物理状态伪装为同一未来目标，必须停止运行。
