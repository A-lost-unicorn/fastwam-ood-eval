# Thought5 子进程 LIBERO 启动修复预注册

预注册日期：2026-08-05（smoke v5、pilot v4 均未运行）

科学角色：**execution-only launcher hotfix，非科学结果**

## 1. 已发生运行的登记

三卡执行 commit `9c4c8b2e92818f20676b6e83070bc616ceb610f7` 上：

- smoke v4 完整通过，`pilot_unlocked=true`，耗时 712.323 s；
  `smoke_result.json` SHA-256 为
  `bfb348e3dc3afa2cb3fb44a3892677ffa13f2352220eb9ffc23885f7d66a9912`。
- pilot v3 完成 16 个 base state、64 个 condition sample 的渲染缓存后，
  B1/G3/G4 三个 worker 均以 exit code 1 退出。顶层 `run_status.json` 为
  `status=error`、`scientific_result=false`，文件 SHA-256 为
  `fe46693401eee61ab94d749ded0fbe0ec7114efde3959a6c4bb5e82f9b25999c`。
- pilot v3 的 render cache 本身完整，语义 SHA 为
  `4b4366e7dee897aa0ac84106ab096cd50d53931f0d8ee3f86cbfa4c36303bad1`；
  manifest 文件 SHA-256 为
  `a83b39d78a54ef47d69d9255df7a975d2ede338a29845c2a92015ae1cf91980a`。
  它只读保留，不跨 commit 复用。

三个 worker 的原子状态均给出完全相同的根因：
`ModuleNotFoundError: No module named 'libero'`。对应 B1/G3/G4 状态文件
SHA-256 分别为
`bbbd8468f2c66d7499d33226e0f27f67570513ec4b4c94886afde56dfad25727`、
`b89da8014b3de18365749c83c607402f33f7fb01ab008e41f9335fa9618869e4`、
`d2e7bf4fcd79d30729f8f5d325f880472cf6c6aa67d098ac544e365c1b9a899e`。
异常发生在导入 Fast-WAM 官方 `libero_utils` 时，早于 checkpoint 权重加载、
optimizer update、checkpoint 生成和任何 evaluator。故 pilot v3 是无效技术运行，
不能解释 B1/G3/G4 效果，也没有解锁 formal。

## 2. 根因与允许修改

父进程渲染时由 `configure_libero_package` 临时修改了父进程的 `sys.path`，所以
render 正常；新启动的 worker 只能继承环境变量，不能继承 Python 进程内的
`sys.path`。原 worker `PYTHONPATH` 没有 `third_party/LIBERO-plus`，因此三个并行
worker 在同一 import 处同时失败。

本 hotfix 只允许：

1. 在 shell runner 和每个 worker 的环境中显式加入仓库绝对路径：`src`、
   `third_party/FastWAM`、`third_party/FastWAM/experiments/libero`、
   `third_party/LIBERO-plus`；
2. 为每个 worker 显式传递仓库输出目录中的 `LIBERO_CONFIG_PATH`；
3. 在任何 simulator render 和 GPU worker 启动之前，用一个全新 Python 子进程
   导入 `libero.libero` 与 Fast-WAM `libero_utils`。预检失败必须 fail closed；
4. 使用全新的 smoke v5、pilot v4 namespace，并让后续 formal 只接受 pilot v4
   的 freeze。

不允许修改 checkpoint、样本 identity、seed namespace、split、条件、variant、
训练目标、参数预算、LR、optimizer、训练步数、gradient accumulation、checkpoint
选择、future K、action denoise steps、rollout 语义、统计门槛或停止规则。

## 3. 冻结身份与等价性

| 阶段 | Config fingerprint | Cohort semantic SHA | 相对上一版本 |
| --- | --- | --- | --- |
| smoke v5 | `e497872dcd30822e9b1c641fac77b4140e4f1933abcbf078b26b5963cced2586` | `4c9903f38af549218355f9781b3097d89d98069a7388e615fed491a8c9011035` | 除 name/output 外逐字段等于 v4 |
| pilot v4 | `ff143479357dce6d1cfe93f99948b40d4af43d25f34939dd2762c646be8e186d` | `16ee9c53ff9542da6ac9f46b26e3d17fd81c9e7ab14f6638b23b9fad5d5fdac0` | 除 name/output 外逐字段等于 v3 |
| formal v2 | `87d11a6b1fdfde08793ff21f0a364686ea781d3f1f129c81853d3d0bd6ef77ca` | `d17a967aa04fa3ceb6447e150361dbab8110adde120bb875aab4e6094106f6c3` | 科学候选不变 |

配置等价性和 fresh-process import contract 必须由回归测试覆盖。hotfix 必须先形成
一个 clean commit；smoke v5、pilot v4 和未来可能运行的 formal v2 必须使用该同一
commit。禁止 `--resume` pilot v3，也禁止把 v3 cache 搬入 v4。

## 4. 执行门禁与 ETA

固定顺序为：

```text
fresh-process import regression
        ↓
single-GPU smoke v5（技术复验）
        ↓ status=complete, pilot_unlocked=true
three-GPU pilot v4（完整重跑）
        ↓ 方向门禁与完整证据成立时
formal v2 freeze
```

smoke v5 依据 v4 实测 11.87 分钟，预留 12–15 分钟。pilot v4 因新 commit/new
namespace 会重新渲染约 13 分钟，总 wall-clock 仍预留 2.5–4.5 小时。formal v2
仍估计 28–45 小时，而且只有 pilot v4 正式解锁后才能运行。
