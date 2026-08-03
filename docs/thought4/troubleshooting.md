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

若旧 v2 已生成 `run_status.json`，不要 `--resume` 或删除旧目录。历史 v3 已完成
并证明 inference-safe hook 链路。历史 v6 已通过，但当前 simulator-replay runner
指向 `phase4_geometry_action_smoke_v7`。v2–v6 工件不可拼入 v7 或 formal v5。

## Formal 报 Robot-init initial state 没变化

旧错误：

```text
Robot-init variant did not change the initial robot state
```

formal v1 在 reset observation 上过早判定。LIBERO adapter 先用 variant robot
reset，随后把共同 demonstration flat state 写回 simulator，因此 reset 时
joint/EEF/gripper 可以与 Clean 完全相同。相同 action prefix 执行后，不同 robot
variant 的动力学状态会在真正的模型输入时刻 `t` 分化。

不能删除 Robot-init、放宽阈值或把它改成 exact-state。当前修复：

1. prefix 前只硬检查 object layout 与 Clean 相同；
2. reset robot state 只保存 SHA/match 供披露；
3. prefix 后要求 Clean/Camera/Lighting robot state 相同；
4. prefix 后要求 Robot-init observation 与完整 simulator SHA 均区别于 Clean；
5. 当前 smoke v7 必须逐样本通过，formal v5 gate 才解锁。

formal v1 与 smoke v4 目录原样保留，不加 `--resume`。当前执行必须依次运行
smoke v7 和 formal v5，且两次运行之间不能改变 project commit。

## Smoke 报 Camera robot state 与 Clean 不同

v4 错误：

```text
camera robot state differs from Clean at model input time
```

这次 flat simulator state 没有失配。旧 collector 对 Clean 使用最后一次
`env.step()` 返回的 observation，却对 Camera/Lighting 使用
`set_state → sim.forward → observable refresh` 生成 observation。两份数据虽然
对应同一 state，却来自不同缓存/刷新时点，不能用来做 `1e-7` robot-state 比较。

v5 以后在 prefix 完成并冻结 Clean state 后，对 Clean/Camera/Lighting 全部执行相同
observable refresh；Robot-init 从自己的 input state 执行同一刷新。真实
render-only 回归已通过 2×4 条件，并确认 Robot-init 的 2 条 input state 都区别于
Clean。不要放宽容差或删除 exact-state 检查，也不要 resume v4。

## Resume 报 existing JSON artifact differs

v5 的错误：

```text
existing JSON artifact differs during resume: .../pre_validation_result.json
```

不是 checkpoint、paired render 或配置内容变化。旧 `config_to_dict()` 保留 Python
tuple；首次 JSON 写盘后 tuple 按标准变为 list，resume 回读时却把 list 与新生成的
tuple 直接比较，因此相同协议也必然失败。v6 在工件生成边界统一做 canonical JSON
round trip，并用 tuple 字段的写入—回读测试锁住该行为。

v5 已有 8 条 paired render、8 条 label，并曾进入 `model_load_started`；第一次进程
在没有结果的情况下中断，第二次才触发上述 resume bug。不得手工改
`pre_validation_result.json`、删除 v5 或把 v5 前缀复制进 v6。历史 v6 已从全新
namespace 完成。当前 simulator-replay v7 同样不能复用 v5/v6 前缀；若 v7 被
非科学性外部中断，只有代码 commit、配置和已有工件均未变化时才可 resume。

## Formal 报 demonstration prefix/input-time alignment failed

formal v4 错误示例：

```text
demonstration prefix/input-time alignment failed:
translation=0.031214m, rotation=2.153deg
```

这不是模型加载卡住，也不是 GPU/OOM；它发生在 paired render 和模型加载前。完整
64-state 只读审计发现 56 pass / 8 fail，最大平移 0.108324 m、最大旋转
28.918°，因此不能只把 3 cm 改成 3.2 cm。禁止：

- resume formal v4；
- 放宽/删除 3 cm / 15° 阈值；
- 丢弃 8 条失败状态或补抽 8 条；
- 事后改成 step `t±1/t±2`；
- 将 v4 记为 formal 科学结果。

确认的 simulator-replay v5 修复保留全部 64 个 identity，将对齐保留为 QC
disclosure，并从实际 simulator input state `t` 重放 action 生成运动标签。运行
全新 smoke v7；通过后再运行 formal v5。`alignment_audit.json` 必须存在且 SHA、
计数、label source 和 pairing 与 smoke result 一致，否则 formal gate fail closed。

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
`simulator_action_replay_from_input_t`，pairing 写为
`condition_specific_world_replay`。若误用 Clean trajectory，
会把不同物理状态伪装为同一未来目标，必须停止运行。

Clean/Camera/Lighting 则必须共享一次 Clean world replay，再分别做 camera
transform；若三条件分别执行 simulator future replay，会把数值积分差异混入
exact-state paired gap，同样必须停止。
