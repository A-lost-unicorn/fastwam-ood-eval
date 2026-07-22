# 思考点一完成度审计

审计日期：2026-07-22

本文只回答“哪些能力已经被真实运行验证、哪些结论仍缺数据”。完整阶段报告见 [thought1_report.md](thought1_report.md)。plan、doctor、pytest 和 smoke 都不能替代正式泛化结果。

## 逐项状态

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 本地可复现环境 | Python 3.10.20、PyTorch 2.7.1+cu128、项目内 Conda、可 source 激活脚本 | 已完成 |
| checkpoint/stats/runtime models | 官方 checkpoint 与配套 stats 已校验；Wan VAE/T5/tokenizer 已离线准备 | 已完成 |
| LIBERO-Plus assets | 官方 assets 已解压到正确 checkout；doctor 与真实 reset 通过 | 已完成 |
| 单卡 Clean | 2/2 completed、0 exception，action/运动/视频通过 | Smoke 已完成 |
| 单卡 OOD | camera/light 4/4 completed、0 exception，扰动首帧可见 | Smoke 已完成 |
| 三卡评测 | rank 0/1/2 分别处理 3/4/2 条；8 completed、1 expected skipped、0 exception | Pilot 已完成 |
| 正式任务规划 | 800 Clean；6,839 OOD planned=6,771 runnable+68 skipped | 已生成并审计 |
| 正式 Clean baseline | `outputs/thought1/fastwam/*/clean/` 尚无 worker result | 未执行 |
| 正式 OOD | `outputs/thought1/fastwam/*/ood/` 尚无 worker result | 未执行 |
| Clean/OOD drop | 需要 combined aggregate 的正式配对结果 | 尚不可报告 |
| unseen object/task | release 训练配置包含全部四个评测 suite | 当前 checkpoint 不可识别 |
| cross-platform | 缺少同一策略的跨平台 observation/action 适配与权重 | 阻塞 |
| future imagination 因果比较 | 缺少训练配方匹配的 Joint WAM/IDM checkpoint | 阻塞 |

## 已通过的真实门禁

- checkpoint 加载和 dataset stats 动作反归一化链路正常。
- 原版 LIBERO 与 LIBERO-Plus 能在独立进程隔离加载。
- PyTorch 2.6+ init-state 兼容修复通过 Clean 和 Plus reset。
- LIBERO-Plus 全部 10,030 个 classification row 的 init-state 路径审计通过。
- 三卡 CUDA/EGL、每 GPU 一模型、job hash 分片、worker JSONL 和视频落盘均通过。
- pilot 的 8 条 action trace 均 finite 且非全零，机器人末端执行器均发生明显位移。
- aggregate 正确得到 8 attempted、2 success、6 failure、0 exception、1 skipped。
- `pytest -q`：43 passed；测试证明评测机制，不证明模型泛化性能。

## 正式 manifest 权威计数

| Suite | Clean runnable | OOD runnable | OOD skipped |
| --- | ---: | ---: | ---: |
| libero_spatial | 200 | 1,661 | 24 |
| libero_object | 200 | 1,742 | 13 |
| libero_goal | 200 | 1,681 | 11 |
| libero_10 | 200 | 1,687 | 20 |
| 合计 | **800** | **6,771** | **68** |

121 条没有 difficulty 的 Goal/Light variant 明确排除在 easy/medium/hard 主实验之外。68 条 skipped 是空分层的审计占位，不消耗 rollout。

## 剩余完成条件

正式研究问题不是只剩 6,771 个 OOD rollout。要计算相同 checkpoint 的 Clean→OOD 下降，还需要 800 个 Clean baseline，因此剩余真实计算量为：

```text
800 Clean + 6,771 OOD = 7,571 runnable rollouts
```

完成标准：

1. 八个 suite/condition 实验目录都产生完整 worker results。
2. 7,571 个 runnable job 无遗漏，68 个 skipped 有明确原因。
3. 未解释的 exception 为 0；合法 `max_steps` 保留为失败，不反复重跑美化结果。
4. 对八个目录执行 combined aggregate，生成 Clean/OOD drop、CI、配对四格和分层结果。
5. 抽检扰动视频并完成人工失败分类。

在这些条件完成前，科学状态应写为“正式实验待运行”，而不是“Fast-WAM 已证明泛化”或“pilot 25% 是正式 OOD 成功率”。
