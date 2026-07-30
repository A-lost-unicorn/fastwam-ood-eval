# 论文复现与工件索引

本文把“重算已有分析”“重绘论文图”“重新执行 GPU 实验”分开。前两项可只读
完成；第三项代价高且必须遵循对应预注册协议，不能覆盖已冻结目录。

## 1. 机器权威工件

| 论文阶段 | 工件 | SHA-256 |
| --- | --- | --- |
| Thought 1 | `outputs/thought1/fastwam/combined/summary/metrics.json` | `0aa1173038a1c37d37123570a83ff9f08667490e3f94276345c802151897dbb5` |
| Thought 1 | `outputs/thought1/fastwam/combined/experiment_manifest.json` | `57dd93f51a2491423f1b14f0d90523f219218698e231a133dcef114caca132ee` |
| Thought 2 | `outputs/thought2/five_category_formal_v1/formal_analysis_v1/formal_analysis.json` | `9d51e0f46c7af73340b390c3acdfd30fa05c8d1e2fa92794ebcae0f112c69f19` |
| Thought 3 Phase 1 | `outputs/thought3/phase1_k1_online_counterfactual_v1/aggregate.json` | `7b6e131fbb01f2eb66fe78a3365fa8b681667b216ecdbd19945fda9b58d899ea` |
| Thought 3 Phase 2 | `outputs/thought3/phase2_full_28_4_a0_a1_v1/phase2_training_result.json` | `5ab57efa2747072a14170ef2ecdfc86cfb7bd36528d138cb14b27fdb17f53d93` |
| Phase 2 A0 sample rows | `outputs/thought3/phase2_full_28_4_a0_a1_v1/tracks/a0/development_final_objectives.jsonl` | `8e370049e467b08fcb720fc664b026fad33ccda824404d70ec7385b7eecf7273` |
| Phase 2 A1 sample rows | `outputs/thought3/phase2_full_28_4_a0_a1_v1/tracks/a1/development_final_objectives.jsonl` | `7f4f02412c0d4121870b0839350e10ec63af239eae8779cb67df0b9e049e7ee1` |

作图脚本会重新计算并记录这些 SHA，见
[figure data manifest](figures/data_manifest.json)。若任一来源 hash 变化，应先
停止并审计，不应静默更新图。

## 2. 固定模型与上游版本

| 项目 | 固定值 |
| --- | --- |
| Fast-WAM checkpoint | `checkpoints/fastwam_release/libero_uncond_2cam224.pt` |
| Checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| LIBERO evaluation commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus commit | `4976dc30028e805ff8094b55501d532c48fec182` |
| Dataset stats SHA-256 | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| Model config SHA-256 | `ab3c2ffde9933e7576c747fecce82bd7d28c9c6478c1b53fcac02b3012be416c` |

环境以 Python 3.10、PyTorch 2.7.1+cu128、CUDA 12.8、MuJoCo/EGL 和项目内
Conda 环境为准。完整安装与同名 LIBERO backend 隔离见
[环境文档](../shared/environment_setup.md)。

## 3. 重绘论文图表

重绘不会加载 Fast-WAM、不会调用 GPU，也不会改写 `outputs/`：

```bash
MPLCONFIGDIR=/tmp/fastwam-paper-mpl \
  .conda/envs/fastwam-ood/bin/python scripts/build_paper_figures.py
```

输出：

- `docs/paper/figures/figure1_ood_success.svg`
- `docs/paper/figures/figure2_future_consistency.svg`
- `docs/paper/figures/figure3_sensitivity_vs_utility.svg`
- `docs/paper/figures/figure4_phase2_per_sample.svg`
- `docs/paper/figures/figure5_evidence_chain.svg`
- `docs/paper/tables/core_results.csv`
- `docs/paper/tables/phase2_per_sample.csv`

图 1 的误差线来自 Thought 1 row-bootstrap CI；图 2 的差值区间来自 40 task
等权、suite-stratified task-cluster bootstrap 10,000 次。图 3 的 Phase 1
黑色横线表示 p95，不是置信区间。图 4 是四条 development sample 各自 32 个
匹配 flow 的均值。

## 4. 只读重算 Thought 2 统计

原命令要求输出目录全新；不要对冻结目录原地覆盖。复制配置并指定新的审计目录：

```bash
source scripts/activate_env.sh
fastwam-ood analyze-thought2-formal \
  --experiment-dir outputs/thought2/five_category_formal_v1 \
  --thought1-summary outputs/thought1/fastwam/combined/summary/episode_results.csv \
  --source-trace-root outputs/thought1/fastwam \
  --output-dir outputs/thought2/five_category_formal_v1/formal_analysis_recheck \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260725 \
  --verify-media
```

重算结果必须与正式分析解释边界一致：

- probe→episode→task 后 40 task 等权；
- suite-stratified task bootstrap；
- outcome mismatch 的两条 episode 从成败关联中排除；
- `all_available` 为 primary，`first_probe` 为敏感性分析；
- 自动 latent proxy 不命名为语义正确率。

## 5. GPU 实验入口

以下命令仅用于确认原协议入口，已经完成的 run 不应无理由重跑或覆盖。

### Thought 3 Phase 1：单卡动作反事实

```bash
CONFIRM_THOUGHT3_K1_ONLINE_CF=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_k1_online_counterfactual.sh
```

协议与全部门禁见
[Phase 1 protocol](../thought3/phase1_action/protocol.md)。该脚本要求单卡，
因为四条件共享一个 live model 和确定性状态；不能写 `GPU_ID=1,2`。

### Thought 3 Phase 2：双卡 matched A0/A1

```bash
CONFIRM_THOUGHT3_PHASE2_FULL=YES \
THOUGHT3_GPU_IDS=1,2 \
bash scripts/run_thought3_phase2_full_28_4.sh
```

中断恢复只能使用原配置与原目录：

```bash
CONFIRM_THOUGHT3_PHASE2_FULL=YES \
THOUGHT3_GPU_IDS=1,2 \
bash scripts/run_thought3_phase2_full_28_4.sh --resume
```

完整规则见
[Phase 2 protocol](../thought3/phase2_adapter/protocol.md)。当前结果已经触发
停止规则；不得事后选择 step 50/100/150、调整 LR/门槛、启动 A2/A4 或用 OOD
outcome 调参。

Thought 1 和 Thought 2 的完整首次运行命令、显存门禁、后台运行及恢复方式分别见
[Thought 1 手册](../thought1/execution_guide.md)和
[Thought 2 手册](../thought2/execution_guide.md)。

## 6. 结果完整性检查

文档结构与本地链接：

```bash
python scripts/check_docs.py
```

代码回归：

```bash
.conda/envs/fastwam-ood/bin/pytest -q
```

格式与路径：

```bash
git diff --check
git status --short
```

任何复现报告都应同时记录：

- 当前项目 commit 与 worktree clean 状态；
- checkpoint、stats、配置和上游 commit；
- physical GPU 与进程内 logical device；
- config/split/cache fingerprint；
- completed/skipped/exception 分母；
- 原始工件 SHA；
- 证据等级及明确的不可写结论。

