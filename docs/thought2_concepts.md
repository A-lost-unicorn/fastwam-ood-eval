# 思考点二概念说明：未来预测与真实变化是否一致

## 1. 思考点一在测什么

思考点一只改变测试环境，不改变策略本身：

```text
observation
    ↓
Fast-WAM action policy
    ↓
executed action
    ↓
environment
    ↓
success / failure
```

它回答的是：冻结同一份 checkpoint 后，从 Clean LIBERO 切换到 LIBERO-Plus OOD 扰动，成功率下降多少。它是正式基线，思考点二不会改动它的 runner、动作、job ID、配置或输出。

## 2. 思考点二想测什么

目标协议把 future predictor 放在控制环之外：

```text
observation + 已经决定要执行的 action chunk
                         ↓
                shadow future predictor
                         ↓
                 predicted future
                         │
                         ├── compare ── action 执行后的 actual future
                         ↓
                  consistency metrics
```

执行顺序非常重要：

1. 先调用原来的 `FastWAMAdapter.act()`，把动作确定下来。
2. Shadow predictor 复制这份动作并预测未来；它只旁观。
3. 环境始终执行第 1 步得到的原始动作，绝不执行 shadow 调用返回的动作。
4. 收集执行后的实际 observation，按上游真实时间比例与预测帧对齐。
5. 比较预测与真实变化，并把结果写进独立 diagnostics 输出。

因此，诊断失败最多使一个诊断工件带有 error，不能悄悄换掉控制动作，也不能把预测未来反馈给策略或环境。

## 3. “预测一致”不等于“动作由预测决定”

这里有两个不同的问题：

- **future consistency**：模型的视频分支对随后发生的变化预测得准不准？
- **future-to-action causality**：策略是否读取显式未来，并因此选择了不同或更好的动作？

思考点二只打算回答第一个问题。即使预测误差与成功率相关，也只能说明一种伴随关系，不能说明未来视频因果地决定了动作，更不能说明打开未来想象一定会提高 OOD 成功率。

真正的 action-follows-future 因果实验需要训练配方匹配的 Joint/IDM checkpoint 或 Future Adapter，并做严格的开关对照。当前阶段不训练这些模型。

## 4. 当前 release checkpoint 的硬限制

Pinned release 是 `libero_uncond`。它的动作分支只读取当前首帧 video token；它不能读取显式 future frames。更关键的是，它的配置为：

```text
video_dit_config.action_conditioned = false
```

所以它虽然能通过 `infer_joint()` 生成视频，但传入的 `action=` 不会进入视频分支。该输出是 observation/language/proprio-conditioned 的 **unconditional future**，不是已执行动作条件下的 future。

因此当前实现对 `mode: action_conditioned_future` 使用硬门禁：真实 release 遇到这个模式必须明确报错，不能把 unconditional video 换个名字当作 action-conditioned 结果。CPU mock 可以验证隔离、对齐、schema、聚合和报告，但不能解除这一科研语义限制。完整证据见 [上游审计](thought2_upstream_audit.md)。

## 5. 时间对齐为什么不能按“一帧一步”处理

Release 的训练配置是 32 个 action 对应 9 个 video frame，比例为 4：

```text
predicted frame 0  <-> action 前，environment offset 0
predicted frame 1  <-> 执行 4 个 action 后，offset 4
predicted frame 2  <-> 执行 8 个 action 后，offset 8
```

默认 `control_horizon=10` 时，一个重规划段从采集时间看只能得到 offset 0、4、8 的候选真实对照。episode 若提前成功，还要继续截断。只有能从运行时确认环境控制频率时，才能把 offset 精确换成秒；否则报告会显式标记近似时间。

还有一层更容易忽略的对齐：action-conditioned 视频专家按 VAE latent group 读取动作，不是每个解码视频帧只读对应的 4 个动作。对审计到的 9-frame / 32-action 结构，直接 action cross-attention 把动作分成两个 16-action 条件组；第一个 future group 直接读取 action 0--15。

但 pinned `first_frame_causal` 只隔离当前首帧，future latent 之间仍可双向传递信息。后组直接读取的 action 16--31 能在后续网络层或 denoising step 间接影响前组，因此任意 future frame 的完整依赖闭包其实是 action 0--31。执行 16 步仍不够；当前 10-step 基线更不满足覆盖。Shadow Diagnostics 会同时检查采集 offset、直接条件组和传递依赖闭包，并在 episode 开始前报错。系统不会把控制 horizon 改为 16/32，因为那会改变思考点一的执行语义。

还有一个 checkpoint 门禁：上游以 `strict=False` 加载 MoT，单纯把配置里的 `action_conditioned` 改成 true 可能留下随机初始化的 action embedding。真实 probe 必须证明这些参数来自 checkpoint，并匹配项目审阅的 checkpoint hash、Fast-WAM commit 和训练 recipe。当前没有可信的 matched checkpoint，批准列表为空。

## 6. 本实验可以与不能支持的结论

在取得真正支持 action conditioning 的匹配 checkpoint、并同时通过动作条件组覆盖门禁后，本实验可以说明：

- 视频分支对已执行动作条件下的未来预测是否准确。
- 预测一致性是否与 episode 成功/失败相关。
- OOD 是否伴随更高的未来预测误差。
- 哪些扰动和哪些案例出现静止预测或明显方向不一致。

本实验不能说明：

- 基础 Fast-WAM 的动作由未来视频因果决定。
- 显式未来一定会提升 OOD 成功率。
- Joint/IDM 一定优于 Fast-WAM。
- 仿真一致性可以直接外推到真机。

对当前 `libero_uncond` release，还必须再收紧一句：它只能生成 unconditional future，不能产生本协议要求的 action-conditioned 科学结果。
