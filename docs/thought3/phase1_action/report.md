# Thought3 Phase 1：K=1 在线动作反事实结果与审计

状态：`COMPLETED / VALID ENGINEERING SMOKE / BRANCH A`

运行日期：2026-07-30

证据等级：`SMOKE`。本实验是固定 checkpoint、固定八条 train sample 上的
技术动作反事实，不是机器人 rollout，也不是 success、ID/OOD 或 K 排序结果。

## 1. 结论先行

预注册 Phase 1 有效完成，并进入分支 A：

```text
future_content_sensitivity_observed
```

在 B0 重放逐位一致、formal null 与 B0 逐位一致的前提下：

- correct future 相对 null 的动作 RMS 差异在 `8/8` sample 上超过冻结 replay
  floor；
- correct future 相对 other-episode shuffle future 的动作 RMS 差异也在
  `8/8` sample 上超过 floor；
- correct/shuffle 的 action tensor SHA 在 `8/8` sample 上改变；
- frozen Fast-WAM 与 Adapter 参数 SHA 前后不变，0 backward、0 optimizer、
  0 gradient；
- 没有读取 action target、真实 future RGB、training future cache、development、
  OOD、rollout、success。

因此可以登记：

> 对这一个 E6 A1 checkpoint 和同一 task 的八条固定 train sample，在线 K=1
> future latent 的具体内容确实进入并改变了 Action DiT 输出；Adapter 并非只对
> “存在一个 latent”或 hook 本身作响应。

但动作变化幅度较小：correct-null 的 normalized action RMS 差异均值为
`0.01105`，约为对应 B0 action-chunk RMS 的 `2.18%`；correct-shuffle 为
`0.01209`、约 `2.38%`。这还不能说明变化足以改变控制轨迹或提高成功率。

## 2. Run identity

| 项目 | 冻结/实测值 |
| --- | --- |
| 命令 | `CONFIRM_THOUGHT3_K1_ONLINE_CF=YES THOUGHT3_GPU_ID=1 bash scripts/run_thought3_k1_online_counterfactual.sh` |
| 输出 | `outputs/thought3/phase1_k1_online_counterfactual_v1/` |
| 项目预注册/实现 commit | `f5169204d852c118756cec8576005dac72e1bc74` |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| 主 checkpoint | E6 `A1@3e-4`, step 200 |
| Adapter file SHA | `aa55622c03aafea05c1bfedcb8548df398b0912dcecba397741c190c6b01b78f` |
| Adapter semantic SHA | `19f62cf45ba36c72da8dbfd752165cc5ef5678d4212b5ab7bf07635fdc7825d9` |
| Config fingerprint | `e343fc73a7cfb6bbdd85a146d32ee79ea30246b839c8be8d2d91f74d739ee544` |
| Cohort fingerprint | `89ce6b84358c6891b1566ef6051201558a59aaeb3e4c28981bc9847dc9af6f72` |
| Shuffle mapping SHA | `55782357b348ef62620efe73939a0e9f3638d56e080f8ceba8a73257aea7f874` |
| 样本 | `libero_goal/task_0`，8 个不同 episode，均为 frame 0 |
| 设备 | 物理 GPU 1；进程内 `cuda:0`；单卡 |
| 开始/完成 | `03:11:27.013838Z` / `03:26:24.699385Z` |
| 总 wall time | `897.69 s = 14.96 min` |
| 最终状态 | `completed`；`phase2_started=false` |

模型加载耗时 `552.82 s`，约占整次 wall time 的 61.6%；这解释了为什么八条
反事实本身很短，而完整命令仍约 15 分钟。

## 3. 有效性与隔离审计

| 检查 | 结果 |
| --- | --- |
| 项目/Fast-WAM worktree preflight | clean / clean |
| B0×2 replay | `8/8` 的 L1/L2/L∞ 全为 0，action SHA 完全相同 |
| 冻结 replay floor | `max(1e-7, 10×p95 replay L2) = 1e-7` |
| Formal null | 无 tensor、0 Video DiT call、0 Adapter call |
| B0-null parity | `8/8` 逐位相同；L∞ 全为 0 |
| Correct/shuffle paired noise | recipient future-noise seed、action seed相同 |
| Shuffle 边界 | donor 全部来自 other episode；只替换 future latent |
| Correct/shuffle initial state | `8/8` hash 相同 |
| Video DiT | correct/shuffle 每个 sample 各精确调用 1 次 |
| Future RGB | 0 decode、0 read |
| 禁止数据源 | action target、development、OOD、rollout、success、training cache 全为 false |
| Frozen Fast-WAM SHA | 前后均为 `ac0dd59d...ceb4f8` |
| Adapter semantic SHA | 前后均为 `19f62cf...25d9` |
| 梯度/优化 | Adapter/backbone gradient name 均为空；0 backward/optimizer |
| 工件 | manifest 内 62/62 文件重新计算 SHA 全部通过 |

权威 manifest 自身 SHA-256：

```text
a0fb8986cdbfcefebcdea3f2272891fb46c5410fe885026a95911b45bbe5a45c
```

关键工件 SHA：

| 工件 | SHA-256 |
| --- | --- |
| `decision.json` | `362cdc9e4ec34c799b88d9f90a3ea5c099713888df519ac762a0f0dde4c6a44a` |
| `run_status.json` | `fd9acdcc37ba1db497123176d7c64dc996fb79e8964f3d5fa930f24a5bdc7453` |
| `aggregate.json` | `7b6e131fbb01f2eb66fe78a3365fa8b681667b216ecdbd19945fda9b58d899ea` |
| `sample_results.jsonl` | `9aa077030f59f65006548addb2be7f56ace81f68b636a672149fe92aeff7ce55` |
| `execution_integrity.json` | `0fb77d4e1a59b57d44005a24cadf18ddbf6b24deb4be822093c5cc8f5bec97d0` |

## 4. 冻结决策

| 冻结判据 | 要求 | 实测 |
| --- | ---: | ---: |
| B0 replay hard pass | L∞ `<=1e-5` | `0`，通过 |
| B0-null parity | L∞ `<=1e-5` | `0`，通过 |
| correct-null 超过 floor | `>=6/8` | `8/8` |
| correct-shuffle 超过 floor | `>=6/8` | `8/8` |
| correct/shuffle action hash 改变 | `>=6/8` | `8/8` |

冻结分类：

```text
classification = future_content_sensitivity_observed
next_branch     = A
```

这不是事后放宽门槛得到的分类。replay floor、`6/8`、hash 条件和 A/B/C 分支都在
真实动作结果产生前由 commit `f516920` 冻结。

## 5. 动作差异

所有动作均为 `[32,7]` normalized policy chunk；L2 是 224 个元素上的 RMS，
并非末端执行轨迹距离。

| Pair | Action hash 改变 | L1 mean / p50 / p95 | L2 mean / p50 / p95 | L∞ mean / p50 / p95 | Action cosine mean |
| --- | ---: | --- | --- | --- | ---: |
| B0 vs null | `0/8` | `0 / 0 / 0` | `0 / 0 / 0` | `0 / 0 / 0` | `1.000000` |
| correct vs null | `8/8` | `0.008455 / 0.008825 / 0.011992` | `0.011052 / 0.011001 / 0.015738` | `0.039063 / 0.040039 / 0.061621` | `0.999769` |
| correct vs shuffle | `8/8` | `0.009442 / 0.008941 / 0.014797` | `0.012092 / 0.011685 / 0.017690` | `0.041351 / 0.032227 / 0.074072` | `0.999712` |
| null vs shuffle | `8/8` | `0.008529 / 0.008702 / 0.012178` | `0.011345 / 0.011317 / 0.016402` | `0.043823 / 0.041016 / 0.080469` | `0.999746` |

分量差异：

| Pair | Translation L2 mean / p95 | Rotation L2 mean / p95 | Gripper abs mean / p95 |
| --- | --- | --- | --- |
| correct vs null | `0.017740 / 0.025255` | `0.020198 / 0.030469` | `0.002609 / 0.004340` |
| correct vs shuffle | `0.020036 / 0.028557` | `0.022004 / 0.034412` | `0.003036 / 0.004901` |

`correct-null` 与 `correct-shuffle` 两个 delta 向量的方向 cosine 为：

```text
mean   0.445885
p50    0.313007
p95    0.875116
```

方向一致性在 sample 间差异较大，说明“future 会改变动作”成立，但当前还不能把
变化解释成稳定、同方向或任务有益的修正。

### 5.1 逐样本结果

| Sample ID 前 12 位 | correct-null L2 | correct-shuffle L2 | 两个 delta 的 cosine |
| --- | ---: | ---: | ---: |
| `9610d2aed3a6` | 0.009434 | 0.014903 | 0.269445 |
| `75359438f810` | 0.007029 | 0.006335 | 0.033825 |
| `5f82a5db9be7` | 0.012893 | 0.012683 | 0.878197 |
| `8f34793be5e0` | 0.013993 | 0.007001 | 0.233277 |
| `8c00174e9155` | 0.008944 | 0.010686 | 0.085914 |
| `461a673f2745` | 0.016678 | 0.016814 | 0.869392 |
| `739baab48223` | 0.012568 | 0.010149 | 0.840456 |
| `81363feff988` | 0.006880 | 0.018162 | 0.356569 |

八条样本都超过 `1e-7` floor，但 effect size 和方向并不均匀。该异质性必须保留，
不能只报告 `8/8`。

## 6. 在线延迟与显存

下表单位均为 ms，报告 `mean / p50 / p95`。B0 使用官方公共
`infer_action()`，其 action time 是 current encode/cache/Action DiT 的 inclusive
计时；null/correct/shuffle 使用可拆分路径。

| 条件 | K=1 Video DiT | Adapter | Action DiT | Condition total | Policy total |
| --- | --- | --- | --- | --- | --- |
| B0 | `0` | `0` | `4000.26 / 3999.11 / 4013.15` | `4000.26 / 3999.11 / 4013.15` | `4234.42 / 4169.92 / 4527.33` |
| null | `0` | `0` | `3756.73 / 3758.06 / 3767.59` | `3950.91 / 3951.74 / 3961.50` | `4207.77 / 4144.68 / 4502.02` |
| correct | `189.88 / 189.75 / 191.38` | `68.42 / 68.38 / 69.09` | `3759.65 / 3761.20 / 3770.41` | `4209.86 / 4210.77 / 4223.68` | `4466.72 / 4403.35 / 4756.61` |
| shuffle | `189.75 / 189.17 / 191.80` | `68.58 / 68.66 / 69.00` | `3761.06 / 3761.46 / 3773.22` | `4211.95 / 4212.93 / 4224.80` | `4725.68 / 4599.86 / 5150.05` |

更适合部署成本解释的是逐样本 paired overhead：

| 比较 | Mean | P50 | P95 | Mean relative |
| --- | ---: | ---: | ---: | ---: |
| correct − null | `258.95 ms` | `257.33 ms` | `277.80 ms` | `+6.17%` |
| correct − B0 | `232.31 ms` | `235.20 ms` | `254.43 ms` | `+5.50%` |

`correct-null` 是更干净的 future-path 增量；`correct-B0` 受公共 API 与拆分路径的
计时边界差异影响。shuffle 的 policy total 额外包含 donor preprocessing、
context 和 current encoding，所以它是研究性反事实总成本，不是可部署策略延迟。

显存：

| 阶段 | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| 模型加载 | `23,679.51 MiB` | `23,866.00 MiB` |
| 正式 policy 条件 | `13,009.92 MiB` | 最高 `13,266.00 MiB` |

加载与执行均低于本阶段冻结的单卡上限。

## 7. 能回答与不能回答

### 已支持

- 固定 E6 A1 checkpoint 的 Adapter 在八条同 task sample 上使用了 K=1 future
  latent 的具体内容；
- 动作变化不是 B0 数值噪声、null tensor、hook presence、不同 action seed 或
  target current/context 被替换造成的；
- K=1 future path 在这次单卡测量中的 paired mean 增量约为 `259 ms`；
- 分支 A 按预注册规则解锁 Phase 2 的**设计与冻结**。

### 尚不支持

- correct future 比 null/shuffle 更“正确”；
- 动作变化会改善或损害机器人成功率；
- K=1 改善 Clean 或 OOD，或降低成功率下降；
- 八条单 task train sample 的结果可推广到其他 task、development 或 OOD；
- E6 step-200 是正式最优 checkpoint；
- K=1 优于 K=2/K=4；
- future 是阶段一/二失败的原因。

特别要避免把 `8/8` 写成“100% 成功”或“未来必然有用”。这里的 `8/8` 只是
deterministic action-sensitivity 条件计数。

## 8. 冻结后的下一步

Phase 2 已被分支 A **允许设计**，但本次 runner 明确没有启动训练。正确顺序是：

1. 单独预注册完整 `28 train / 4 development` matched A0/A1 配方；
2. 固定 normalized sample-loss recipe，并披露它来自 post-E8 engineering
   development，而不是通过了 E9 科学 Gate；
3. 只使用一个预先固定的 LR、seed、update budget、flow schedule 与 dev-only
   checkpoint rule；不做新 sweep；
4. A0/A1 使用相同 sample/flow schedule，继续禁止 OOD、success、reserve
   17–28 和 future RGB 参与训练选择；
5. 完整 checkpoint 训练完成后，先复验一次同协议 K=1
   correct/null/shuffle 内容敏感性；
6. 只有该复验与工程门禁通过，才冻结 240-rollout
   B0/A0/A1/A-shuffle Clean/camera/robot-init directional pilot。

A2/A4、正式多 seed 和完整 OOD 矩阵仍然锁定。
