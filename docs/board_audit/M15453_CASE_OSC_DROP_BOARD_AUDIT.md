# m15328 · CASE-OSC-DROP（振荡液滴频率 · Rayleigh–Lamb）交差核验审计

- 审计对象：mission **m15328**（TensorSPH，CASE-OSC-DROP 算例池锚点，状态 succeeded）
- 审计单：mission **m15453**（交差核验，非重做）
- 被审交付 commit：`b5280f37b50eab41096511e15fcdc2bcc1ff7d27`（"m15328: matured mission deliverables"）
- 审计基线（本次重跑所在 trunk 检出）：`25fc91541bcd6d13e1e0d9e49ed3dbc0e71cbc4c`（`agent/m15453`，从 main 拉出）
- 平台：`Linux-6.6.52 ... loongarch64`，torch 2.7.1，4 线程
- 审计结论：**级别 = L3（系统集成）**；**L4（基准验证）不成立**——honest FAIL 保持，不得消账为"通过"
- 判定词：**approve with findings（L3 成立；F1/F2 需返修）**

---

## 0. 一句话结论

m15328 交付的是**真代码 + 真工件 + 诚实的 FAIL**：CSF 表面张力算子在主求解器里确实被调用（L3 成立），
但其 Rayleigh–Lamb 模态频率基准在 3 % 容差下**逐模态全灭**（n=2 −4.85 %、n=3 −9.48 %、n=4 −19.32 %，
收敛阶 −4.14 远低于要求的 ≥2），且**基准裁判在 `tensor_benchmark_cases` 中根本不存在**
（`anchor_slug='CASE-OSC-DROP'` → 0 行）。因此该锚点**不能记 L4**；交付本身也没有被合入 trunk
（`b5280f3` 不是 `25fc915` 的祖先），trunk 上相关 pytest 处于红态。

---

## 1. 取证：三件套逐项核验

### 1.1 run_artifacts（工件是否真存在）

| 工件 | 路径 | 状态 |
|---|---|---|
| 报告 | `runs/surface_tension_dispersion/report.json` (987 行) | 存在，sha256 `b4dedf4f…a0a7e0` |
| 基准表 | `runs/surface_tension_dispersion/benchmark.json` (239 行) | 存在，sha256 `3e533e13…213e0c` |
| 摘要 | `runs/surface_tension_dispersion/summary.md` | 存在（含 ref/got 表 + 4 张门表） |
| 溯源 | `runs/surface_tension_dispersion/meta.json` | 存在（git sha / 命令 / 版本 / 哈希 / 工件清单 6 项） |
| 图 | `figures/{capillary_wave_error,lamb_stiffness,curvature_audit,delta_measure}.png` | 4/4 存在 |

哈希自校验：`sha256(report.json)` 与 `sha256(benchmark.json)` **与 meta.json 登记值逐字相符**（审计脚本
`runs/m15453_audit/audit_summary.json` 记录）。`meta.json.artifacts` 列出的 6 个文件**全部存在**，
与 `tests/test_surface_tension_dispersion.py::test_committed_campaign_artifacts_satisfy_the_reported_invariants`
的断言一致。

> 但注意 F2：这批工件**只在 `agent/m15328` 分支**，检出到 trunk 后不存在（见 §4-F2）。

### 1.2 benchmark ref vs got（数值是否在容差内）

审计脚本**不采信** `report.json` 的 `gates` 块，直接从 `rows` 重算门（`scripts/audit/audit_m15453_osc_drop.py`）。

**(a) 液滴模态门 C2（容差 3 %，`TOL_LAMB_PCT`）** —— σ=0.072 N/m，ρ=1000，R=0.02 m，ε=0.10：

| n | R/Δx | ω_got [rad/s] | ω_ref(2-D，代码所用) | 误差 vs 2-D | 3 % 内? | ω_ref(3-D Rayleigh) | 误差 vs 3-D |
|---|---|---|---|---|---|---|---|
| 2 | 16 | 7.32537 | 7.34847 | **−0.31 %** | ✅ | 8.48528 | −13.67 % |
| 2 | 24 | 6.96172 | 7.34847 | −5.26 % | ❌ | 8.48528 | −17.96 % |
| 2 | 32 | 6.99237 | 7.34847 | −4.85 % | ❌ | 8.48528 | −17.59 % |
| 3 | 16 | 13.33570 | 14.69694 | −9.26 % | ❌ | 16.43168 | −18.84 % |
| 3 | 24 | 13.27165 | 14.69694 | −9.70 % | ❌ | 16.43168 | −19.23 % |
| 3 | 32 | 13.30411 | 14.69694 | **−9.48 %** | ❌ | 16.43168 | −19.03 % |
| 4 | 16 | 19.05848 | 23.23790 | −17.99 % | ❌ | 25.45584 | −25.13 % |
| 4 | 24 | 19.01951 | 23.23790 | −18.15 % | ❌ | 25.45584 | −25.28 % |
| 4 | 32 | 18.74919 | 23.23790 | **−19.32 %** | ❌ | 25.45584 | −26.35 % |

**C2 = max|err| = 19.316 %（9 行中仅 1 行通过）→ FAIL。** 审计重算值与提交值逐位一致（19.316335444641226）。
**关键：换成 3-D Rayleigh 分支（报告引文所指）误差更大（−17.6/−19.0/−26.4 %），FAIL 结论对 ref 分支选择稳健。**

**(b) 毛细波门 C1（容差 5 %）**：9 行中 **0 行**通过。k=1：−167.99/−338.81/−198.42 %；
k=2：−114.93/−214.32/−141.17 %；k=3：−79.01/−163.47/−137.53 %（24/32/48 ppw）。
提交的门值 `198.42 %` 只取**最细网格行**，最坏行是 **338.81 %**（见 F4）。两种取法都 FAIL。
负号说明离散表面力给出的是**伪负刚度**（不回复），不是精度不足。

**(c) 收敛门 C3**：要求 ≥2。重算阶：k=1 −0.138、k=2 −0.210、k=3 −0.719、n=2 −4.143、n=3 −0.038、n=4 −0.098，
`min_order = −4.143` → **误差随加密变大 → FAIL**（且 n=2 的 −4.14 与其余 ~−0.1 自相矛盾，指向格点相位噪声主导）。

**(d) σ 敏感性门 C4**：`max_spread_pct = 0.0`、PASS —— **这是代数恒等式，不是证据**（见 F6）。

### 1.3 repro_cmd 可执行性

| 记录的复现命令 | 结果 |
|---|---|
| `meta.json.command` = `python scripts/run_surface_tension_dispersion.py --modes 2,3,4 --rdx 16,24,32 --out-dir runs/surface_tension_dispersion` | **原样不可执行**：`ModuleNotFoundError: No module named 'tensorsph.surface_tension_dispersion'`（缺 `PYTHONPATH=src`；环境里另有 m14539 的 stale editable 安装抢包）。在 m15328 检出与 trunk 检出**都复现了这个失败** |
| `report.json.repro_cmd` = `PYTHONPATH=src python -m tensorsph.surface_tension_dispersion … --out …` | 可执行（`--help` 正常，仅一条 runpy RuntimeWarning） |
| 审计实跑：`PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --modes 2,3,4 --rdx 16,24,32 --out-dir runs/m15453_audit` | **成功，61.7 s**；12/12 行 ω 误差与交付值**逐位相同**（最大差 0.0 个百分点），四门逐位相同 → **可复现性本身成立** |

> 结论：**科学可复现 ✅ / 命令字符串不自洽 ❌**（两条不同的 repro 串 + meta 那条缺环境前缀）。

### 1.4 报告六段是否齐全

`tensor_reports` id=14440（mission_id=15328，7031 字符）与仓库内
`benchmarks/oscillating_droplet/benchmark_report_oscillating_droplet_frequency.md`（173 行）**同文**，
含 ①问题与 ref 出处 ②方法与实现 ③参数表 ④对照表（ref/got/误差/勾叉）⑤收敛性与参数敏感性 ⑥失败模式与工件 + 附 repro_cmd。
**六段齐全，数字与 artifacts 一致**（抽查 n=3@32 ω_ref=14.69694 / ω_got=13.30411 / −9.48 %、n=4@32 18.74919 / −19.32 % 均与
`benchmark.json` 相符）。缺陷：正文写"容差判据 <5%（任务令）"与代码门 `TOL_LAMB_PCT = 3.0` **不一致**（F5），
且未显式回答"是否 <3 %"。

---

## 2. 基准裁判（唯一裁判）核对

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-OSC-DROP';
-- (0 rows)
```

`tensor_benchmark_cases` 全表 16 个锚点（CAP-LEDGER-EVIDENCE / CASE-EULER-VORTEX / GAP-DEM-* / GAP-FEM-* /
GAP-FVM-TWOPHASE / GAP-LBM-* / H1..H6 / REPRO-FVM-SOD1978 / XVAL-FEM-TPB1976），**没有任何 SPH / 液滴锚点**。
即：**本锚点的机器裁判缺席**，3 % 容差是交付自己定的（`surface_tension_dispersion.py:131`）。
按任务令"基准判据是唯一裁判"，此处只能降级为"审计人独立重算 + 解析解"，因此我在 §1.2 做了**独立重算（不采信交付的 gates 块）**
与**独立推导 ref**（§3），结论与交付自评一致：FAIL。

---

## 3. 独立复核 ref：代码用的是 2-D 分支，且 2-D 分支正确

交付代码 `surface_tension_dispersion.py:11-12, 892` 写
`ω² = n(n²−1)σ/(ρR³)`，并把这句挂在"(Rayleigh 1879; Lamb Art. 275)"名下。

审计人**从几何独立重推**（`scripts/audit/audit_m15453_osc_drop.py::exact_curvature_mode`）：
对 `r = R(1 + ε cos nθ)` 取极坐标精确曲率 `κ = (r² + 2r′² − r r″)/(r² + r′²)^{3/2}`，
按 `{1, cos nθ, sin nθ}` 投影，得模态系数前因子 /ε·R：

| n | 数值拟合 (n²−1) | 解析 n²−1 | 相对差 |
|---|---|---|---|
| 2 | 3.008 | 3 | +0.27 % |
| 3 | 7.867 | 8 | −1.67 % |
| 4 | 14.396 | 15 | −4.03 % |

（差值随 n 增大是 ε=0.10 的 O(ε²) 非线性修正，不是公式错。）

配合潜流运动学 `η = n A R^{n−1}/(iω)` 与 Bernoulli `p = −ρ∂φ/∂t`：
`ω² = n(n²−1)σ/(ρR³)` —— **这正是 2-D（柱状）液滴的 Rayleigh–Lamb 支**，与代码 2-D 极坐标几何自洽。
**但 Rayleigh 1879 / Lamb Art. 275 的原文结果是 3-D 球滴 `ω² = n(n−1)(n+2)σ/(ρR³)`**（n=2：8 vs 6）。
两项都成立时正确表述应为"2-D 类比"，引文标注是**张冠李戴**（F7，低危：不改变 FAIL，且仓库另有
`tests/test_oscillating_droplet_lamb.py:66` 用 `sqrt(8/6)=1.1547` 显式守住了 2-D/3-D 之辨，说明作者知道区别）。

---

## 4. 分级列证（L0–L5）

| 级别 | 判据 | 证据 | 判定 |
|---|---|---|---|
| L0 文档 | — | 有报告/摘要（但不能只靠文档） | 满足（不足以定级） |
| L1 数学/单元 | — | `tests/test_surface_tension_dispersion.py`、`test_csf_dispersion.py`、`test_oscillating_droplet_lamb.py`（解析分支 1e-12、2-D/3-D 之比、阻尼拟合 <0.5 %、噪声负对照 residual>0.5、σ=0 不振荡） | 满足 |
| L2 独立功能实现 | 真代码 输入→计算→输出 | `surface_tension_dispersion.py` 966 行（hex_lattice:139 / drop_geometry:205 / csf_state:254 / drop_modal_stiffness:341 / lamb_case:454 / run_validation:790）+ `csf_consistent.py` 539 行（mls_gradient:123 / csf_fields:262 / surface_tension_acceleration:383）+ `csf_curvature.py:78 fit_curvature` + 入口 `scripts/run_surface_tension_dispersion.py:1-289`；实跑 61.7 s 出 12 行数值 + 4 图 | **满足** |
| L3 系统集成 | 接入主求解器 | `surface_tension.py:174 compute_surface_tension_force` → `:231 surface_tension_acceleration`（即被基准测的同一算子）→ 主求解器 `droplet_impact.py:929-934` 每步调用；`threed/csf3.py:56`+`threed/liquid_column.py:720` 3-D 路径；`capillary_dt_limit` 被 `threed/solver.py:40,65-81` 用作第二稳定限 `acoustic_dt` | **满足** |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | 可复现 ✅、误差分析 ✅、工件齐 ✅，**但误差 9.5–19.3 % ≫ 3 %、收敛阶 −4.14 < 2、裁判未登记、ε 无极限外推** | **不成立（honest FAIL）** |
| L5 科研工程 | — | 无 | 不成立 |

**→ 级别 = L3。**（不因有 173 行漂亮报告而上抬；也不因 FAIL 而下压到 L2——算子确实进了主求解器路径。）

---

## 5. 发现清单（按严重度）

**F2 [high] 交付未进 trunk，trunk 上 pytest 红**
`git merge-base --is-ancestor b5280f3 HEAD` → **no**；trunk 检出（`25fc915`）无 `runs/surface_tension_dispersion/`。
实跑 `pytest tests/test_oscillating_droplet_lamb.py tests/test_surface_tension_dispersion.py -q` →
`FAILED tests/test_surface_tension_dispersion.py::test_committed_campaign_artifacts_satisfy_the_reported_invariants`（1 failed, 30 passed）。
影响：`m15328` 的工件指针只在 `agent/m15328` 分支有效；trunk 常规红灯会污染"测试全绿"信号。
返修：把 `runs/surface_tension_dispersion/` 合入 trunk（或把该测试改为 skip-if-missing 并在 CI 声明工件随分支走）。

**F1 [medium] 基准裁判缺席**
`tensor_benchmark_cases` 无 `CASE-OSC-DROP` 行（全表 16 锚点，0 个 SPH 锚点）→ 3 % 容差为被审单自定，门无法机器复核。
返修：登记 `case_name=lamb_mode{2,3,4}`、`metric=omega`、`tolerance=3%`、`ref_source="2-D Rayleigh–Lamb ω²=n(n²−1)σ/(ρR³)，n=2,3,4；R=0.02，σ=0.072，ρ=1000"`、`status=failed`（honest）。

**F3 [medium] repro_cmd 不自洽/原样不可执行**
`meta.json.command` 缺 `PYTHONPATH=src` → 两处检出都 `ModuleNotFoundError`；`report.json.repro_cmd` 又是另一条（`python -m … --out`）。
加了前缀后可逐位复现（12/12 行、四门全同）。
返修：统一为 `PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --modes 2,3,4 --rdx 16,24,32 --out-dir …`，并在 meta 里记 `env`。

**F4 [low] C1/C2 只取最细网格行**
`surface_tension_dispersion.py:867-874` 用 `ppw==max(ppws)` / `rdx==max(rdxs)` 取门值 → 提交 C1=198.42 %，而最坏行 338.81 %。
两值都 FAIL，故不影响结论；但"门通过"只能代表最细网格，不能代表整条 ladder。
返修：门值取 ladder 最坏值（或同时报 finest 与 worst）。

**F5 [low] 报告容差写 5 %，代码门 3 %**
报告 §④"容差判据：<5 %（任务令）"vs `TOL_LAMB_PCT=3.0`（n=2@24 −5.26 % 在 5 % 下也不通过，结论不变）。
返修：统一 3 %，并显式回答"是否 <3 %"。

**F6 [low] C4 门不可失败（构造性通过）**
`max_spread_pct = 0.0` 恰好为零：CSF 牵引力对 σ 线性、ω 取 `sqrt(stiffness_ratio)·ω_theory`，故 `ω/√σ` 是代数不变量。
一个永不失败的门不是证据（4 门中 1 门空转）。返修：σ 敏感性应作用于**真实时间推进**（不同 σ 下测阻尼/相位）或干脆删除。

**F7 [low] 引文张冠李戴**
2-D 公式挂 3-D 引文（Rayleigh 1879 / Lamb Art. 275）。若换 3-D 支，误差 −17.6/−19.0/−26.4 %（更差）。
返修：注明"2-D 柱状类比"，并给出 3-D 支作为交叉参考。

**F8 [low] 有限振幅偏差未界定**
ε=0.10 固定（`DROP_EPS`），无 ε→0 外推；同谐波 O(ε²) 曲率修正已达 0.27/1.67/4.03 %（n=2/3/4），
量级可到 3 % 容差的 1.3 倍 → n=4 的 −19.3 % 不能全部归因于离散误差。返修：加 ε=0.02/0.05/0.10 ladder 做振幅外推。

**F9 [info] 诊断链自洽（交付的优点）**
液滴行 `∫δ_s dV/A = 0.702–0.734`（Brackbill 单位质量缺 28 %）；`local_range` 归一化在波上过冲 1.651；
曲率审计符号未恢复（实测波谷 +0.014 / 波峰 −0.044 vs 精确 ∓0.126 1/m，平界面噪声 RMS 0.405 1/m，SNR 0.31）；
4 种已发布开关组合筛选：best lamb 4.85 %、best wave 93.21 %，**无一通过** → FAIL 不可靠调参掩盖。
这些都被诚实写进 summary.md，属"该报的报了"。

**F10 [info] 绿测试 ≠ 能力成立（semantic gate）**
`tests/test_oscillating_droplet_lamb.py:180-190` 把 mode-2 误差**钉在 −61.69 %**（loongarch64，动力学求解器路线）
并断言 `|err| > 15 %` —— 测试断言的是"这个能力已知不达标"。另有独立动力学基准
`benchmarks/oscillating_droplet/`（同为液滴频率，误差 −61.7 %）远比本单的准静态模态路线差；两条路线都未达标，
**不能互相顶替**。

---

## 6. 未做的检查（不冒充已做）

- 未审计 `benchmarks/oscillating_droplet/`（动力学路线）全量 ladder，只核了 `RECORDED_MODE2_ERR_PCT` 与测试断言。
- 未做 3-D 球滴算例（`threed/csf3.py`）的频率基准；F7 的 3-D 支只做了**解析对照**，没跑数值。
- 未做 OpenFOAM/Basilisk 等外部求解器交叉验证（本机无）。
- 未核 GPU 路径与更大 ppw 收敛（>48）。
- 未判定"2-D 类比能否满足 BP 锚点意图"（领域口径问题，留给燃料/锚点 owner）；本审计只判功能实质。

---

## 7. 处置

**approve with findings —— 级别 L3，L4 拒绝；账不许消，需返修后再验。**

返修顺序（建议）：
1. F2 把 `runs/surface_tension_dispersion/` 与 `benchmarks/oscillating_droplet/benchmark_report_…md` 合入 trunk，trunk 测试转绿；
2. F1 在 `tensor_benchmark_cases` 登记 `CASE-OSC-DROP`（3 行，status=failed，honest）；
3. F3/F5 统一 repro_cmd 与容差表述；
4. F6/F4 修门语义（C4 换真实时间推进敏感性；C1/C2 报 worst-over-ladder）；
5. 物理侧：δ_s 归一化（`legacy` −28 % vs `local_range` 过冲）+ 曲率符号（fit_quadratic 在 ppw≤32 符号未恢复）
   是 n≥3 误差主因，需在 `csf_consistent`/`csf_curvature` 上修；F8 振幅外推用于把离散偏差与非线性偏差分开。

---

## 8. 审计工件与复现

```
# 1) 重跑被审单元（逐位复现，61.7 s）
PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --modes 2,3,4 --rdx 16,24,32 \
    --out-dir runs/m15453_audit
# 2) 独立重算门 + 独立重推 ref + 出图
PYTHONPATH=src python scripts/audit/audit_m15453_osc_drop.py
```

产物：
- `runs/m15453_audit/report.json|benchmark.json|meta.json|summary.md|figures/`（审计人独立重跑）
- `runs/m15453_audit/_source_m15328/{report,benchmark,meta}.json,summary.md`（被审交付件快照，sha256 自校验）
- `runs/m15453_audit/audit_summary.json`（重算门 / 复现差 / ref 重推 / 退化门探针 / δ_s / 收敛阶）
- `docs/board_audit/m15453/fig1_omega_vs_mode.png`（实测 vs 2-D/3-D 两支）
- `docs/board_audit/m15453/fig2_error_vs_resolution.png`（误差 vs R/Δx + ±3 % 带）
- `docs/board_audit/m15453/fig3_diagnostics.png`（模态刚度比 + δ_s 积分）

机器可判项复核命令：

```sql
SELECT repo,scope,mission_id,level,gate_state,gate_level FROM tensor_maturity_verdicts
 WHERE repo='TensorSPH' AND scope='m15328 CASE-OSC-DROP';
```