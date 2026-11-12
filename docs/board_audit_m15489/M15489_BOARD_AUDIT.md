# 交差审计 m15371 — CASE-CAPILLARY-WAVE（毛细波色散，TensorSPH）

审计单：m15489 · 结论：**L4 / approve with notes** · 审计人：board-audit（独立复算，未修改被审仓任何文件）

被审：m15371（TensorSPH，`succeeded`，PR #494 merged，报告 id 14452，task_key `case-capillary-wave`）

---

## 0. 独立性与范围

* 独立性：审计在**自己的 worktree**（`/root/tp-agent/work/m15489`，main 9a12b6a）上重新执行了复现命令与测试，
  不读取 m15371 的中间态；只把 m15371 的工件当**被检对象**。所有数字由本单脚本重算：
  `audit/m15489_probe_capillary.py` → `audit/capillary_probe_m15489.json`。
* 范围：只审本锚点（`CASE-CAPILLARY-WAVE`）的三件套（run_artifacts / ref vs got / repro_cmd）+ 报告六段 + 平台基准行。
  未审 TensorSPH 其它能力，未做 3-D、粘性、时域积分扩展。
* 未跑项（标 `not_run`，不算 pass）：时域 SPH 模拟同波（算例内无积分器）；跨仓消费（无仓 import 该模块）；
  平台基准行登记（属 case feeder 职责，本单不改共享治理表）。

---

## 1. 三件套取证（逐项核）

| 项 | 结论 | 指针 |
|---|---|---|
| impl_files | 真代码 653+188+192 行 | `benchmarks/case_capillary_wave_dispersion/run.py:178`（measure）、`:242`（case_point）、`ref.py:83`（闭式）、`ref.py:130`（reference） |
| entry | CLI 可执行 | `python benchmarks/case_capillary_wave_dispersion/run.py --out-dir <dir>` |
| run_artifacts | 存在且自洽 | `runs/case_capillary_wave_dispersion/{report,benchmark,ref,summary,meta}.json` + `figures/*.png`；`meta.json` 记录 ref.json sha256 = `fa18af59…edf9d`，本单重算一致 |
| repro_cmd | 可执行、逐位复现 | `OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/case_capillary_wave_dispersion/run.py --out-dir artifacts/audit_m15489/run_repro` → exit 0，wall **11.2 s**，8/8 grid `omega_got` 与原件**9 位小数全等**，gates 字节一致 |
| tests | 6 passed / 5.30 s | `tests/test_case_capillary_wave_dispersion.py:43,49,65,70,74,84` |
| integration | CI 收集 | `.github/workflows/tests.yml:27 python -m pytest -q` |

**外部锚**：Lamb 1932 *Hydrodynamics* 6th ed. Art. 267 `ω²=σk³/ρ` 及其两层有限深推广。
本单用 mpmath dps=50 **独立重算** ω_ref = 47.218369685342571，与工件 47.218369685342566 差 **7.1e-15**；
两条退化极限（ρ₂→0 + 深水 → Lamb；ρ₂→0 有限深 → 重力-毛细 g=0）运行时断言 <1e-12 通过 → 参考值非手抄。

## 2. 基准裁判（平台侧）

`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CAPILLARY-WAVE'`
→ **0 行**（2026-11-12 查询；本仓域 0/16 个算例已登记）。因此**平台级容差不存在**，唯一判据是仓内预声明
`TOL_PCT = 10.0`（`run.py:103`，非事后调参）。表内另记：m15371 交付时 `server_quality_audit.gate_valid=false`（gate 未跑到）。

复算的门（本单重跑，与原件一致）：

| gate | 要求 | 结果 | 判 |
|---|---|---|---|
| A 设计点 | \|ω_got/ω_ref−1\|<10 %（ppw=32，mls） | **−0.225 %** | PASS |
| B 分辨率梯 | ppw 16/24/32/48 全 <10 % | −2.48 / −4.22 / −0.23 / −2.55 %，最差 4.22 % | PASS（对照算子 fit_quadratic：−24.3/−22.2/−17.9/−19.0 %） |
| C δs 单位质量 | Σδs·V/L = 1±5 % | 1.00916 | PASS |
| D σ 线性 | 刚度比随 σ×0.5/1/2 不变（spread<5 %） | 0.000 % | **PASS 但无鉴别力（F2）** |

## 3. Findings（按严重度）

### F1 · material — 设计点的 "−0.23 %" 是两相抵偿，不是精度

`case_point` 对 2×2 点阵相位取均值。逐相位（ppw=32，A/λ=0.02）：

| x_shift | 0 | 0 | 0.5 | 0.5 |
|---|---|---|---|---|
| stiffness ratio | **0.8098** | 0.8137 | **1.2041** | 1.1920 |
| 单相位 ω 误差 | **−10.0 %** | −9.8 % | **+9.8 %** | +9.3 % |

即单相位误差已达**算例容差本身（ppw=32 最差 10.01 %；ppw=16 最差 **12.05 %**，单相位口径会**判 FAIL**）**，均值 +0.24 % 是两个 ±10 % 偏置的抵偿。
std=0.193 写在工件里，但 summary/报告的表达（"误差 −0.23 %"）会被读成算子精度 0.2 %，**不成立**。
可支持的表述：*4 相位系综均值在 ppw 16–48 全档 ≤4.3 %*；不可支持的表述：*单相位/算子误差 ~0.2 %*。
（注：最差 12.05 % 仍 <本平台 15 % 不可接受线，故判"表述过强"，不判物理失败。）

### F2 · material — gate D（σ 线性）是恒等式，零信息

`sig` 同时是 `dp_measured`（run.py:217）与 `dp_theory`（run.py:219）的公因子 → 比值与 σ 无关是**构造性的**。
探针复算：σ×0.5/1/2 三行 `stiffness_ratio` 与 `std` **逐位相同**（1.0048754379199991 / 0.19319821894168854）。
更深一层：因 `ω_got = ω_ref·sqrt(ratio)` 是把测量**锚定在参考 ω 上**，该测量结构上**无法**检出 k 或 ρ 指数错误
（例如算子若正比 k² 也照样"通过"）。四个门里有一个不设防。

### F3 · minor-material — A/λ=0.01 的 −36.7 % 是模态投影零点，不是非线性失谐

逐相位：ppw=32 四相中**两个相位 ratio = −0.0（dp_meas 到机器零）**、另两个 +1.61/+1.59；
ppw=16 时**四相全零**（均值 0 → −100 %）。std≈mean（0.8016）正是 0/1.6 混合的结果。
15 % 量级的"失谐"实为**点阵相位导致的模态求和相消（quadrature null）**。总结里"线性响应带外"的归因是错的；
顺带暴露：该模态投影可对某些点阵相位恒等消零，任何未来用此投影的用法都须报相位敏感性。

### F4 · minor（生态）— 算例未登记为平台基准，原交付 gate 未跑

`tensor_benchmark_cases` 本锚 0 行；`result.evidence.server_quality_audit={gate_valid:false,key_present:false}`。
下游按 `anchor_slug` 查不到本算例，容差只活在仓内常量里。建议 case feeder 补登记（本审计不动共享治理表）。

### F5 · note — 通过只对 mls 曲率算子成立

同梯度的 `fit_quadratic` 对照 −17.9…−24.3 %（超 15 % 不可接受线）。mls 是库默认（两相 Laplace 基准亦用它），
且两方案都如实列出 → 判"适用范围声明"，非方案挑选。

### F6 · note — 无时域波传播："色散"由静态模态应力推得

`measure()` 只建一帧点阵、投影 `−σ κ δs V` 到 cos(kx)，再经 ω=ω_ref·√(dp_meas/dp_theory) 映射。
与被审仓方法论一致（`docs/verification_surface_tension_dispersion.md` 记录自由表面时域路线 honest FAIL：198 %、
相位噪声 std/|mean| 0.4–1.0）；两层构造把 δs 积分从 0.93/1.23（不稳定）修到 **1.009**，是本单的实质增量。
须明说：被验证的是**离散恢复刚度**，不是**被观测到的频率**。

## 4. 判级

| 级 | 判据 | 证据 | 判 |
|---|---|---|---|
| L2 | 真代码 输入→计算→输出 | run.py:178 全粒子链（cell_linked_pairs→cubic_spline_kernel→mls_gradient→mls_divergence→模态投影） | 满足 |
| L3 | 接入主求解器/平台 | 仅用库导出算子 + CI pytest 收集；无求解器时间环集成 | 满足（弱） |
| L4 | 完整算例+误差分析+可复现 | 外部闭式锚 + 8 行误差表 + 2 条灵敏度梯 + 3 图 + sha256 + 逐位复现 | 满足（误差分析强度低于自述，见 F1/F2） |
| L5 | 科研工程成果 | 无收敛阶、无实验对照、无工程交付 | 不主张 |

**verdict = 4**（级别=功能实质；F1/F2 削的是"精度宣称"与门设计，不是算例机器本身）。

## 5. 复核方式（任何人可重跑）

```bash
# 1) 复现被审算例（~11 s）
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/case_capillary_wave_dispersion/run.py \
    --out-dir artifacts/audit_m15489/run_repro
# 2) 本审计的逐相位 + 独立参考复算
OMP_NUM_THREADS=4 PYTHONPATH=src python audit/m15489_probe_capillary.py
# 3) 图的生成
OMP_NUM_THREADS=4 PYTHONPATH=src python audit/m15489_figure.py
# 4) 门断言
OMP_NUM_THREADS=4 PYTHONPATH=src python -m pytest tests/test_case_capillary_wave_dispersion.py -q   # 6 passed
```

产物：`audit/capillary_probe_m15489.json`、`audit/capillary_audit_m15489.png`、
`artifacts/audit_m15489/run_repro/*`、`audit/case_capillary_wave_m15371_verdict.json`。

## 6. 处置

**approve with notes**：L4 成立（算例机器完整、可复现、外部锚真实）；返修建议三条，均不推翻算例本身：
1. 报告/summary 把"−0.23 %"改写为"系综均值 −0.23 %，单相位 −10.0…+9.3 %，ppw 16–48 均值 ≤4.3 %"，并给出逐相位表；
2. gate D 换掉（改成 k 梯度或 ρ 依赖的**非构造性**检验；现判据恒真）；
3. A/λ=0.01 行的归因改为"模态投影相位零点（dp_meas 数值零）"，并加相位零点守卫（null 相位直接标 invalid 而非参与均值）。