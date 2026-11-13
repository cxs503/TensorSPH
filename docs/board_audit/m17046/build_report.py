import base64
import json
import pathlib
import subprocess

ROOT = pathlib.Path("/root/tp-agent/work/m17046")
fig = ROOT / "docs/board_audit/m17046/fig1_audit_m17046.png"
recompute = json.loads((ROOT / "docs/board_audit/m17046/independent_recompute.json").read_text())

rows = []
for r in recompute["stored_vs_rederived"]:
    rows.append(
        f"| {r['case'].replace('_c025','')} | {r['reference']} | {r['stored_mae']:.9f} | "
        f"{r['rederived_mae']:.9f} | {r['abs_diff']:.1e} | {100*r['rederived_mae_over_amplitude']:.3f} % | "
        f"{'pass' if r['pass_at_15pct'] else 'not graded / out of tolerance'} |")
table = "\n".join(rows)

md = f"""# REPRO-SPH-MONAGHAN1994 交差审计（m17020）：增量 L2，算例 L4

**结论先行**：m17020（status=succeeded，commit `c670396`，分支 `agent/m17020`）的交付是
**真实可重算的后处理增量（L2）**，但它自称的「误差分解」在算术上**不闭合（27.04 倍）**且把
**负残差归因于离散误差**，Ritter 对照更跑出自身声明的适用域 2.1 倍；该分支**未进入主线**。
因此增量**不记级**为 L3/L4，只记 L2 并列出阻塞项。算例级能力本身仍是 **L4**（由 m16873 算例 +
m16936 审计承载），**不判 L5**（锚点论文数值 WITHHELD、参考对自洽 gate fail）。裁决已落表
`tensor_maturity_verdicts id=479`。

## ① 案例介绍

锚点 `REPRO-SPH-MONAGHAN1994` → Monaghan (1994) *J. Comput. Phys.* **110**:399-406，算例为
Martin & Moyce (1952) 二维溃坝实验构型：水柱宽 a=0.2 m、高 H=0.4 m（H/a=2），水槽 2.0×0.75 m，
ρ0=1000 kg/m³、Tait γ=7、h/dx=1.3、Courant 0.3、c0=25 m/s（Mach 0.122）。判据窗口
τ∈[0.5,3.0]（161 采样点），平台容差 15 % 参考幅值，弱可压限 max ρ/ρ0−1 ≤ 0.10
（`run.py:54-57`、`summary.json:protocol`）。

**基准裁判表实况**：`tensor_benchmark_cases` 对 `anchor_slug='REPRO-SPH-MONAGHAN1994'`
**0 行**（全表 29 anchors / 32 行）。实际裁判 = mission acceptance「quantitative_targets 逐项容差」
+ 上述预注册协议（判据先于运行，写在代码常量里）。

## ② 怎么做的

1. 取证：m17020 的 result JSON、branch commit `c670396`、5 个改动文件、`feature_ledger` 状态。
2. 独立重算：自写 161 点窗口网格 + **二分插值**（不 import `run.py:compare/interp`），从 3 条已提交
   轨迹 + 2 张参考表重算 6 组 MAE/幅值、偏置、max|Δ|、幅值与 `summary.json` 逐值对照（1e-9）。
3. repro_cmd 实跑：`run.py --stage accept`，与已提交 summary 做 `diff`。
4. 增量审计：跑 `error_budget.py`（重跑后 JSON 逐字节相同 → 确定性），核可加性、Ritter 适用域、
   9 个新测试是否约束不变量。
5. 反向对照：把 m17020 工件+文档临时放入审计树，验证新守卫会红（3 failed），移出后 6 passed / 3 skipped。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 独立重算 vs 已提交 summary（差 = 0.0，全部 6 组）

| case | 参考 | stored MAE | 重算 MAE | \\|差\\| | MAE/幅值 | 15 % 容差 |
|---|---|---|---|---|---|---|
{table}

（幅值 2.67；dx=0.01 偏置 +0.11451 in Z/a = +4.29 % 幅值，max|Δ|=0.16537；轨迹元数据 800 粒子 /
2179 步 / dt 1.56e-4 / wall clock 299.4 s / escaped=0。）

### 3.2 repro_cmd 实测

`PYTHONPATH=.:src python benchmarks/repro_monaghan_1994/run.py --stage accept` →
`trajectory_vs_independent: pass (0.04376 ≤ 0.15)`、`compressibility: pass (0.01936 ≤ 0.10)`、
`reference_self_consistency: fail (17.01 > 2.0)`、`resolution_trend_vs_independent: fail`，
且**逐字节重建** `results/summary.json`。

### 3.3 m17020「误差分解」不闭合（阻塞项 F1）

| 项 | ÷幅值 |
|---|---|
| 被分解的总量（dx=0.01 vs crosscheck） | **0.043759（4.376 %）** |
| 「参考不确定度」（两表分歧） | 0.318558（31.86 %） |
| 「Ritter 偏差」 | 0.845170（84.52 %） |
| 「弱可压误差」 | 0.019360（1.94 % of ρ0） |
| **三项之和** | **1.183088 = 27.04× 总量** |
| 工件/文档原文 | `residual (-27.48%) is attributable to discretization and boundary condition effects` |

### 3.4 Ritter 适用域（阻塞项 F2）

`error_budget.py` docstring 声明 τ<√2=1.4142，实现却跑到 τ=2.967（2.1 倍越界）；τ=2.967 处
Ritter 给出 Z/a=5.93，而实测前沿饱和在 ≈2.6 —— 那里度量的是浅水解失效。仅适用域内
（n=9，τ≤1.384）平均偏离仍 **57.61 % 幅值**，说明对 H/a=2 水柱 Ritter 不是可用参考。

## ④ 结果图

图 1（`docs/board_audit/m17046/fig1_audit_m17046.png`，脚本 `make_figure.py`，只读工件）：
(a) 3 个分辨率的前沿 Z/a–τ 与两张数字化表，阴影=预注册窗口；(b) 有符号残差随 τ，三级偏置
−0.125/+0.032/+0.115 **穿过零点**（|MAE| 非单调的真因）；(c) m17020 三项与总量在对数尺度上
的对比：和 1.1831（118.3 % 幅值）对总量 0.04376（4.38 %）= **27×**，即不闭合。

## ⑤ 注意点

- **「一致地错」型风险**：m17020 的结论句由 `error_budget.py:155-160` 的 f-string 生成，同时写进
  JSON 与文档 ⑤b，因此任何「文档↔工件一致」型守卫都不会红；9 个新测试只断言 `budget_total ==
  summary_total`、`mae_Za > 0` 之类，在缺陷成立时**依然全绿**（实测 9 passed in 0.08 s）。
- **落地状态**：`git log --all -- benchmarks/repro_monaghan_1994/error_budget.py` 只有 `c670396`
  （branch `agent/m17020`）；审计树（mainline 线）无该文件，`feature_ledger` 该锚仍
  `status=open, evidence=NULL, mission_id=NULL` → succeeded 状态不等于已落地能力。
- **局限**：未重跑任何仿真（审计时 load15=119/nproc128，按容量纪律只做后处理）；dx=0.01 的
  bit-exact 可重跑性本次未复核（m16936 已实测 0.04/0.02 两级）；Monaghan (1994) 原文与 1952 原文
  仍在付费墙后，未引入任何转述值。
- **处置**：request changes —— 改名为并行诊断并显式声明不可加、删掉负残差归因句、Ritter 限在
  τ≤√2、补可加性/适用域两条不变量测试，修好后再落 mainline。

## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）

| 关键量 | 基准 | 实测 | < 3 %？ |
|---|---|---|---|
| 前沿轨迹 Z/a，dx=0.01，τ∈[0.5,3.0] | 独立数字化表（crosscheck） | MAE/幅值 **4.376 %** | **否**（对 15 % 平台线为 pass，两条线不互替） |
| 前沿轨迹 Z/a，dx=0.01 | bundled 合成表 | 17.271 % | 不适用（参考 WITHHELD，禁作 pass 源） |
| 参考对自洽 | 声明 ±0.05 | 0.8506 = 17.01× | 否 |
| 弱可压闭合 | ≤0.10 | 0.01936 | 是 |
| 锚点论文（Monaghan 1994） | JCP 110:399-406 | **不可得**（closed access） | 不适用——无基准，不用转述值 |

**分级**：m17020 增量 = **L2**（真实代码 + 可重算工件 + 测试；无 L3 调用方、未落地，故不判 L3；
其「误差分析」是无效对照，故不判 L4）；算例级 = **L4**（m16873+m16936 承载，本次复核确认），
**不判 L5**（外部锚点数值 WITHHELD + 参考对自洽 17.01× fail）。
"""

report = {
    "title": "REPRO-SPH-MONAGHAN1994 交差审计（m17020）：增量 L2、算例 L4，误差分解不闭合 27×",
    "summary": ("独立重算 3 分辨率×2 参考共 6 组指标（差 0.0），repro_cmd 逐字节重建 summary；"
                "m17020 的“误差分解”三项之和 = 27.04× 总量且负残差被归因于离散误差，Ritter 对照越出 √2 适用域 "
                "2.1 倍，且该分支未落 mainline → 增量判 L2 不记 L4；算例级维持 L4（4.376 % ≤ 15 %，3 % 线答“否”）。"),
    "scenario_key": "REPRO-SPH-MONAGHAN1994",
    "mission_id": 17046,
    "tags": ["board-audit", "TensorSPH", "REPRO-SPH-MONAGHAN1994", "m17020", "maturity-verdict",
             "dam-break", "error-budget", "reference-self-consistency"],
    "markdown": md,
    "figures": [{
        "name": "fig1_audit_m17046.png",
        "caption": ("(a) 溃坝前沿 Z/a–τ：3 个 SPH 分辨率 vs 两张 Martin & Moyce 数字化表；"
                    "(b) 有符号残差（SPH−crosscheck）随 τ，偏置穿零解释 |MAE| 非单调；"
                    "(c) m17020 误差分解三项（对数尺度）与总量对比：和 = 27× 总量，不闭合。"),
        "png_base64": base64.b64encode(fig.read_bytes()).decode(),
    }],
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(report))
print("report.json", len(json.dumps(report)), "bytes")