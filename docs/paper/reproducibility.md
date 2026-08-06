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
| Thought 4 smoke v8 | `outputs/thought4/phase4_geometry_action_smoke_v8/smoke_result.json` | `c2e1199172e1dda004385ec1c723707fb087e6f935bb5803f324ddfb88c49a02` |
| Thought 4 probe-first commit | `outputs/thought4/phase4_geometry_action_diagnosis_v6/probe_stage_result.json` | `db7f6816b3dbf7a1b5574bd9cd7543a6351d7cf126d4f4f7c1de6c3beb9740ff` |
| Thought 4 diagnostic evidence | `outputs/thought4/phase4_geometry_action_diagnosis_v6/diagnostic_evidence.json` | `0542a72b5c733a018e1fd99341bfdbfe501b8bd05a4a88d16566c5685bc14c6b` |
| Thought 4 intervention | `outputs/thought4/phase4_geometry_action_diagnosis_v6/intervention_results.json` | `8aeaf2fda57870f512787c9f63ff67b86b9c161c30d3f13b3a81af4aaa601b9c` |
| Thought 4 method selection | `outputs/thought4/phase4_geometry_action_diagnosis_v6/method_selection.json` | `8fdd9417803b072ec4af160eb395a06d792d0e613c437d27c68a05ccec68c79b` |
| Thought 4 artifact manifest | `outputs/thought4/phase4_geometry_action_diagnosis_v6/artifact_manifest.json` | `d1c5ef118a0cc5950790b5d425f25c8c89c32ec0183c2d376b2baf01004912af` |
| Thought 5 Pilot direction | `outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/pilot_direction.json` | `1bb2944d196253b7002daaa87f340bbad61c86e4e7e2a3e05f0c0fbe46a98c3d` |
| Thought 5 representation | `outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/representation_results.json` | `5604d8ff8f6de52ae1744feb856f0678dcddb19e3f5e953fdfef6be52b8a5efa` |
| Thought 5 future geometry | `outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/future_geometry_results.json` | `582d836ff2b28eb2febef5627007975c460be7a83c73bbc2f5ebe7b1bbdaa4a2` |
| Thought 5 future utility | `outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/future_utility_results.json` | `612c8f8cecdf9d3ade8e033aa6d996de4dd083d676c8a03bcd382a543d998bba` |
| Thought 5 rollout | `outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/rollout_results.json` | `fac48a4b92a59820c77efcd6ca1e64bdc24ca0640fa26202b0782a2da943aa3e` |
| Thought 5 read-only diagnosis | `outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4_readonly_failure_v1/analysis_result.json` | `a5d5c99808773ea2f1fd84bb198d0c1f034dee4841cef4db84754bd5c7826f1f` |
| Thought 5 derived manifest | `outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4_readonly_failure_v1/artifact_manifest.json` | `c94f924586d033002b7b4143241bc4e138a3e70bcb81800b4767537edcc19911` |

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
- `docs/paper/figures/figure6_thought5_pilot.svg`
- `docs/paper/tables/core_results.csv`
- `docs/paper/tables/phase2_per_sample.csv`
- `docs/paper/tables/thought5_pilot_diagnostics.csv`

Thought4 的冻结数值以 [`tables/thought4_diagnosis.csv`](tables/thought4_diagnosis.csv)
保存；Thought5 Pilot 与只读诊断由脚本生成
[`tables/thought5_pilot_diagnostics.csv`](tables/thought5_pilot_diagnostics.csv)。
两份 CSV 都是论文派生表，不是独立机器权威来源；必须追溯对应 JSON 与 SHA。

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

### Thought 4：smoke v8 与 formal v6

两项均已完成，以下命令只用于确认原入口；不得覆盖或 `--resume` 已完成的 v8/v6
namespace：

```bash
CONFIRM_THOUGHT4_PHASE4_SMOKE=YES \
THOUGHT4_GPU_ID=2 \
bash scripts/run_thought4_phase4_smoke.sh

CONFIRM_THOUGHT4_PHASE4_FORMAL=YES \
THOUGHT4_GPU_ID=2 \
bash scripts/run_thought4_phase4_diagnosis.sh
```

v8 smoke 耗时 655.90 秒；formal v6 耗时 4,728.05 秒，冻结分类为
`camera_equivariance_gap`。完整数值、claim boundary 与下一分支见
[formal v6 结果报告](../thought4/formal_v6_results.md)。

### Thought 5：三卡 Pilot 与 CPU-only 只读分解

Pilot v4 已完成并触发停止规则。以下命令只登记原入口；不得覆盖、resume 或用同一
Pilot endpoint 调参后重跑：

```bash
CONFIRM_THOUGHT5_PILOT=YES \
THOUGHT5_GPU_IDS=0,1,2 \
bash scripts/run_thought5_pilot.sh
```

权威运行约 2 小时 29 分，`formal_unlocked=false`。因此 formal 命令当前禁止执行。
现有工件的失败分解可以 CPU-only 重建到独立 sibling namespace；它不会加载模型、
GPU、仿真或新 outcome：

```bash
bash scripts/run_thought5_pilot_v4_readonly_failure_analysis.sh
```

分析前后验证 25 项 source SHA 不变。完整停止边界见
[Pilot v4 结果](../thought5/pilot_v4_results.md)和
[只读失败分解](../thought5/pilot_v4_readonly_failure_analysis.md)。

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

2026-08-06 的当前全量结果为 `504 passed, 5 warnings`；5 个 warning 均来自
测试环境无法初始化 NVML，不是断言失败。

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

Thought4 的只读审计验证了 artifact manifest 中 1,586 个 entry 的路径、大小与
文件 SHA。需要单独披露：`execution_integrity.json` 的
`integrity_sha256=a08ac875...e59f5e71` 只覆盖写入时的 11 字段核心 payload，
不覆盖后追加的 runtime/smoke/probe 字段；完整文件 SHA
`be301260...bd347` 仍由有效 manifest 覆盖。该缺陷不改变诊断分类，但冻结输出
不得原地修补。
