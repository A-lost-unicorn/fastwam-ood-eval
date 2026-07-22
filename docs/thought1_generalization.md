# 思考点 1：unseen 泛化与未来想象

## 结论边界

当前 release 能严谨完成的是“标准 LIBERO → LIBERO-Plus 环境扰动”的 zero-shot OOD 鲁棒性评测。其训练数据配置列出 `libero_spatial`、`libero_object`、`libero_goal` 和 `libero_10`，因此直接在这些 suite 上测试不能证明 unseen-object 或 unseen-task 泛化。LIBERO 与 RoboTwin 又使用不同平台接口和各自训练的 release checkpoint，不能把两个独立的同平台分数称为 cross-platform transfer。

| 研究轴 | 当前状态 | 可报告的结论 |
| --- | --- | --- |
| 跨环境 | 可运行 | 相同 checkpoint 在官方 Plus 相机、光照、背景、机器人初态和物体布局 shift 下的成功率与下降 |
| 跨物体 | 仅诊断 | `libero_object` 的按对象表现；不是 unseen object，因为训练已见该 suite |
| 跨任务 | 仅诊断 | 四个 suite/任务的分层表现；不是 unseen task，因为训练已见这些 suite |
| 跨平台 | 阻塞 | 无；需要同一策略的源/目标平台适配与明确的训练暴露关系 |
| 未来想象 | 阻塞 | 无；需要与 Fast-WAM 训练配方匹配的 Joint WAM/IDM checkpoint |

机器可读版本在 `configs/studies/thought1.yaml`。

## 为什么不能直接开关未来想象

上游 `FastWAM` 的 attention mask 只允许 action token 读取首帧 video token；`infer_action` 因而不生成未来帧。即使调用 `infer_joint` 并保存未来视频，action mask 仍只读取首帧，所以这只是额外可视化开销，不是动作因果路径的消融。

上游 `FastWAMJoint` 改变了 attention mask，使 action token 读取全部未来 video latent；`IDM` 则先生成未来视频再恢复动作。这些是不同的训练变体，必须加载各自 checkpoint。不能把 `libero_uncond` 权重加载进 `joint` 结构，也不能用是否保存视频冒充 on/off。

## 三卡执行顺序

1. `bash scripts/plan_thought1_pilot.sh`：只生成 64 条 pilot job；每个 suite 选择 task 0，覆盖 Clean 与五类扰动×三档强度。
2. 先运行单卡 Clean 和 OOD smoke；检查 action、成功判定、视频、variant metadata 和 resume。
3. 使用 `configs/eval_ood_pilot.yaml` 做三卡真实链路 pilot；当前为 9 planned / 8 runnable / 1 skipped，用它估算每个 episode 的墙钟时间与三卡负载。
4. `bash scripts/plan_thought1.sh`：按当前协议重新生成四个 suite 的 full manifest，不启动执行。当前 pinned 数据预期 7,639 planned（800 Clean；6,771 OOD runnable；68 OOD skipped），不是旧协议的 12,800。
5. 审核新 manifest、排除 121 条无 difficulty 的 Goal/Light 记录并获得确认后，用 `scripts/run_3gpu_eval.sh` 分别运行每个计划单元。每个 rank 独立加载一个模型并按 job ID 分片。
6. 每个策略先聚合 Clean/OOD；若以后取得匹配的 future checkpoint，再把两个策略目录作为 `--input-dir` 合并，读取 `future_imagination_comparisons`。

旧的 12,800-job manifests 来自把 OOD 每个分层重复 20 次的计划，已经过期，不能直接执行。逐阶段命令和验收证据见 [实施与验收手册](thought1_execution_guide.md)。

Joint WAM 的 smoke 模板位于 `configs/ablations/`。它们可以执行 `plan`，但 `doctor` 会在匹配 checkpoint 和 stats 不存在时失败；这是有意的安全门禁。

截至 2026-07-22，官方 [`yuanty/fastwam`](https://huggingface.co/yuanty/fastwam) 仍只发布 LIBERO/RoboTwin 的 uncond checkpoint。第三方 [`LIQIIIII/badwam-libero-joint-wam`](https://huggingface.co/LIQIIIII/badwam-libero-joint-wam) 与 [`LIQIIIII/badwam-libero-idm-wam`](https://huggingface.co/LIQIIIII/badwam-libero-idm-wam) 提供约 12 GB 的 Joint/IDM 权重；其 metadata 说明 Joint 训练到 step 21700 并使用默认 Fast-WAM Joint 配置，但没有证明它与官方 uncond checkpoint 在初始化、训练 seed、优化预算和精确数据版本上配对，且模型页许可证标为 `other`。因此它们最多可作为 exploratory/associational baseline，不能直接支持“未来想象导致 unseen 泛化提升”的因果结论。

## 让其真正回答 unseen object/task/platform 所需的新增证据

- Object：定义训练对象集合和完全不相交的测试对象集合，训练一个不含测试对象的 checkpoint。
- Task：定义语言/技能/场景级 holdout，冻结 split 后训练，不得用 release 的全-suite checkpoint冒充。
- Platform：确定同一动作语义、相机/本体状态映射和目标平台成功判定，并说明训练是否见过目标平台。
- Future imagination：Fast-WAM 与 Joint WAM/IDM 使用相同数据 split、优化步数、初始化/训练 seed 集合和推理预算；至少提供多个训练 seed。仅一个不同来源 checkpoint 的胜负只能作相关性证据。
