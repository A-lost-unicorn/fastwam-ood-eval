# 思考点一完成度审计

审计日期：2026-07-26

思考点一的工程链路与正式科学实验均已完成。正式结果见
[thought1_report.md](report.md)，机器权威结果见
[combined report](../../outputs/thought1/fastwam/combined/summary/report.md)。

## 逐项状态

| 要求 | 正式证据 | 状态 |
| --- | --- | --- |
| 本地可复现环境 | Python 3.10.20、PyTorch 2.7.1+cu128、项目内 Conda、source 激活脚本 | 已完成 |
| checkpoint/stats/runtime models | 官方 checkpoint、配套 stats 与离线 Wan runtime models 已校验 | 已完成 |
| LIBERO-Plus assets | 官方 assets 位于 pinned checkout；doctor 与真实 reset 通过 | 已完成 |
| 单卡 Clean/OOD | Clean 2/2、OOD 4/4 completed，0 exception | Smoke 已完成 |
| 三卡 pilot | 8 attempted、1 expected skipped、0 exception | Pilot 已完成 |
| 正式任务规划 | 800 Clean；6,839 OOD planned=6,771 runnable+68 skipped | 已审计 |
| 正式 Clean baseline | 800/800 completed，778 success，0 exception | Formal 已完成 |
| 正式 OOD | 6,771/6,771 completed，3,230 success，0 exception | Formal 已完成 |
| Combined aggregate | 7,639 行，7,571 attempted，68 skipped，0 重复/遗漏 | 已完成 |
| Clean/OOD drop | 97.25%→47.70%，绝对下降 49.55 pp，相对下降 50.95% | 可正式报告 |
| 分层分析 | suite、40 tasks、5 categories、3 difficulty levels 均有统计 | 已完成 |
| trace/video 完整性 | 7,571 traces；3,563 failure videos；0 缺失、0 空文件 | 已完成 |
| 自动失败机制分类 | 所有系统失败已排除，但语义失败仍需人工视频 taxonomy | 待人工复核 |
| unseen object/task | release 训练配置包含全部四个评测 suite | 当前 checkpoint 不可识别 |
| cross-platform | 缺少同一策略的跨平台接口与权重 | 阻塞 |
| future imagination 因果比较 | 缺少 recipe-matched Joint WAM/IDM checkpoint | 阻塞 |

## 正式分母与结果

| Suite | Clean success / N | OOD success / N | OOD skipped | Absolute drop |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | 197 / 200 | 926 / 1,661 | 24 | 42.75 pp |
| `libero_object` | 197 / 200 | 1,132 / 1,742 | 13 | 33.52 pp |
| `libero_goal` | 193 / 200 | 531 / 1,681 | 11 | 64.91 pp |
| `libero_10` | 191 / 200 | 641 / 1,687 | 20 | 57.50 pp |
| 合计 | **778 / 800** | **3,230 / 6,771** | **68** | **49.55 pp** |

68 条 skipped 是上游没有对应官方 variant 的空分层审计行，不消耗 rollout，
不进入成功率分母。正式 6,771 条 attempted OOD 记录均有官方 difficulty，
且 1–2/easy、3/medium、4–5/hard 的映射无不一致。

## 已通过的正式门禁

- 八个 source manifest 与结果全部使用 checkpoint
  `1000437c...a49579` 和项目 commit `575ba8f...406c5`。
- 项目、Fast-WAM、LIBERO、LIBERO-Plus 在正式运行时均为 clean source。
- 7,639 个 manifest job ID 与 raw worker result ID 集合完全一致。
- 7,571 个 runnable job 均为 `completed`；0 exception、0 raw duplicate。
- 2,399,314 个 action step 中无 NaN/Inf、空动作、错误维度或全零运动 episode。
- 机器人首末位移最小 0.0385 m；3,563 个失败全部是 `max_steps`。
- 3,563 个失败视频路径全部存在且非空；success-only 视频按配置不保存。
- combined aggregate 正确排除 68 skipped，并输出总体、suite、task、category、
  difficulty、CI 与配对四格。

## 尚未完成但不阻塞思考点一主结论

1. 失败视频的正式人工 taxonomy；当前只能报告成功判定和 max-steps，不能自动
   宣称感知、抓取或规划是哪一种根因。
2. wrist camera 的逐 episode 录制证据；当前 failure video 只保存 agent view。
3. 相机位姿、光源参数等底层扰动值的统一 runtime introspection；正式记录已保留
   classification ID、variant name、上游 commit 和任务文件，可追溯但未完全结构化。

这些缺口限制机制解释和媒体审计，不改变已经完成的 Clean/OOD 成功率测量。
