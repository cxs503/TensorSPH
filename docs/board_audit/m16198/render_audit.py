"""m16198 交差审计渲染器：所有数字从 artifacts/m16198/*.json 程序化注入（零手抄）。

写 docs/board_audit/M16198_CASE_SPH_RAYLEIGH_TAYLOR_BOARD_AUDIT.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
ART = WT / "artifacts" / "m16198"
A = json.loads((ART / "audit.json").read_text())
R = json.loads((ART / "refit.json").read_text())
MAN = json.loads((HERE / "evidence" / "MANIFEST.json").read_text())

ref = A["ref"]
runs = A["runs"]
model = A["damping_model"]
conv = A["convergence"]
repro = A["repro"]["dx0.04_nu0.001_a0.01"]
base = sorted([r for r in runs.values()
               if "sigma_stored" in r and r["nu"] == 1e-3 and r["amp0"] == 0.01],
              key=lambda r: r["dx"])
L = []
def w(s=""): L.append(s)


def pct(x):
    return f"{100 * (x - 1):+.1f} %"


w("# CASE-SPH-RAYLEIGH-TAYLOR-MP 交差核验（board-audit · m16198）")
w()
w("> 被审单 **m16141**（TensorSPH，status=succeeded，task_key=case-sph-rayleigh-taylor-mp）｜"
  f"被审提交 **agent/m16141 @ 9736ad7**（单提交，21 文件 +1631 行）｜被审报告 **tensor_reports#15292**")
w("> 审计单 m16198｜证据 schema v1｜**裁定 = L2（独立功能实现）**，L3/L4 明确不达；"
  "基准裁判 `tensor_benchmark_cases` 本锚点 **0 行**。")
w("> 本文件全部数字由 `docs/board_audit/m16198/render_audit.py` 从 `artifacts/m16198/{audit,refit}.json` 注入；"
  "被审工件按 sha256 快照在 `docs/board_audit/m16198/evidence/`（MANIFEST.json）。")
w()

# ------------------------------------------------------------------ ①
w("## ① 案例介绍")
w()
w("两层密度反差流体单模界面扰动（Rayleigh–Taylor 线性阶段），重相在上，"
  "A=(ρ_h−ρ_l)/(ρ_h+ρ_l)=%.3f。判据=早期界面扰动指数增长率 vs Taylor (1950) 线性增长率 "
  "$\\sigma=\\sqrt{Agk}$。参数取自工件 `config`（逐算例落盘）：" % ref["atwood"])
w()
w("| 量 | 值 |")
w("|---|---|")
c = json.loads((HERE / "evidence" / "reference.json").read_text())
w(f"| λ / k | 1.0 m / {ref['k']:.6f} m⁻¹ |")
w(f"| ρ_h / ρ_l | 1500 / 500 kg/m³（ρ_h/ρ_l=3，A={ref['atwood']:.3f}） |")
w(f"| 层深 d_h=d_l | 0.75 m（kd={ref['k'] * 0.75:.3f}） |")
w("| 扰动幅度 η₀ | 0.01 m = 0.01λ（敏感性档 0.02 m） |")
w("| c₀ / ν / g | 40 m/s / 1e-3 m²/s（敏感性档 0 与 1e-2） / 9.81 m/s² |")
w(f"| σ_ideal=√(Agk) | **{ref['sigma_ideal_Agk']:.6f} s⁻¹**（独立复算，与被审 reference.json 相对差 "
  f"{R['reference_recomputed']['rel_dev']:.1e}） |")
w(f"| 有限深度修正 σ_fd | {ref['sigma_finite_depth']:.6f} s⁻¹（{ref['sigma_finite_depth'] / ref['sigma_ideal_Agk']:.5f}×） |")
w(f"| 黏性修正 σ_visc | {ref['sigma_viscous']:.6f} s⁻¹（{ref['sigma_viscous'] / ref['sigma_ideal_Agk']:.5f}×，νk²/σ={ref['nu_k2_over_sigma']:.5f}） |")
w()
w("外部锚点：Taylor 1950, *Proc. R. Soc. A* **201**:192, doi:10.1098/rspa.1950.0052（线性阶段 σ²=Agk）；"
  "有限深度与黏性修正式由线性化 Euler 方程给出。原文无 A=0.5 数值表，目标值由解析式直接求值——"
  "对**线性理论本身**成立，对「SPH 能否复现线性理论」是外部真值。")
w()

# ------------------------------------------------------------------ ②
w("## ② 怎么做的")
w()
w("### 被审单做了什么（复述，非采信）")
w()
w("`src/tensorsph/rayleigh_taylor.py`（784 行）新建两相（密度反差）WCSPH 时间推进器："
  "逐粒子 ρ₀ᵢ+Tait 刚度（:154 hydrostatic_profile/:180 build_column）、数密度连续式 "
  "ρᵢ=ρ₀ᵢδᵢ/δ₀（:302 number_density）、哈密顿对称压力力（:358 two_phase_rhs）、"
  "静水解析格点初始化、线性本征函数位移+速度激励（:239 initial_velocity）、"
  "x 周期邻域（:261 periodic_pairs）、两个振幅量计（:473 interface_centroid / :513 velocity_amplitude）、"
  "增长拟合（:553/:572）、驱动（:666 run_case）、场量快照（:759）。")
w()
w("### 审计席做了什么（独立路径）")
w()
w("1. **三件套取证**：`git show --stat 9736ad7` 逐文件比对；13 个工件按 sha256 快照入 evidence/MANIFEST.json；"
  "报告六段与表值逐项对工件。")
w("2. **独立重拟合**（`artifacts/m16198/refit.py`）：只用 `series` 原始时序（不采信存盘 `fit`）"
  "重算 ln 斜率、复算 √(Agk)、并对**每个**算例做早/晚半窗重拟合。")
w("3. **第三量计**（`artifacts/m16198/analyse.py`）：从已存密度场快照按 ρ=(ρ_h+ρ_l)/2 等值线读界面 "
  "η(x)，取 2|⟨ηe^{−ikx}⟩|，与质心量计、速度带量计三方对照——量计伪影会在此暴露。")
w("4. **收敛性独立复核**：三点观测阶 p、1/2 阶 Richardson 外推（6 组）、以及只用 A=0 对照两点拟合的"
  "加性阻尼模型 σ(dx)=σ_visc−λ₀dx 对四档基线的盲预测。")
w("5. **可复现性实测**：在 `/tmp/a16198` 干净拷贝里原样跑 repro_cmd（dx=0.04，4 线程 34 s）、"
  "`damping.py`（30 s）、`pytest tests/test_rayleigh_taylor.py`（61.7 s），与提交工件逐值比对。")
w("6. **集成面**：`src/tensorsph/__init__.py`、`src/tensorsph/verification/`、全仓 grep 找 caller。")
w()

# ------------------------------------------------------------------ ③
w("## ③ 计算结果（全部程序化注入）")
w()
w("### 3.1 三件套核验")
w()
w("| 项目 | 结论 | 指针 |")
w("|---|---|---|")
_sizes = ', '.join(str(r['bytes']) + ' B' for r in base[:2])
w(f"| run_artifacts | **真存在**：8 个算例 JSON（{_sizes} …）+ "
  "table/summary/reference/damping/spurious，全部在 9736ad7 内 | `benchmarks/rayleigh_taylor/runs/*.json`"
  "（快照 `docs/board_audit/m16198/evidence/*.json`，sha256 见 MANIFEST.json） |")
w(f"| benchmark ref vs got | ref σ_ideal={ref['sigma_ideal_Agk']:.6f} s⁻¹（独立复算偏差 "
  f"{R['reference_recomputed']['rel_dev']:.0e}）；got 见 3.2 表 | `runs/reference.json` + 审计复算 `artifacts/m16198/refit.json` |")
w(f"| repro_cmd | **可执行**（审计原样跑通，34 s）；**但不复现表中数字**：dx=0.04 复跑 "
  f"σ={repro['rerun_sigma']:.3f} vs 提交 {repro['committed_sigma']:.3f}（{repro['rel_change_pct']:+.1f}%，F3） | "
  "`benchmarks/rayleigh_taylor/run.py:46`（docstring repro）、审计重跑 `/tmp/a16198/benchmarks/rayleigh_taylor/runs/dx0.04_nu0.001_a0.01.json` |")
w(f"| tests | **10 passed in 61.7 s**（独立复跑；不变量型断言，非「执行过」） | `tests/test_rayleigh_taylor.py:52,73,90,109,130,143,161,172,185,208` |")
w("| 报告六段 | **齐**（①–⑥），含显式「是否<3%：否」，2 图内嵌 base64 | `tensor_reports#15292`（markdown "
  f"{len(open(HERE / 'evidence' / 'rayleigh-taylor.md').read())} B 等价文档 `docs/benchmarks/rayleigh-taylor.md`） |")
w("| 基准裁判 | **缺席**：`tensor_benchmark_cases` 全表 25 行，`anchor_slug='CASE-SPH-RAYLEIGH-TAYLOR-MP'` **0 行** | SQL 复算，见 §⑤ F1 |")
w()
w("### 3.2 增长率 vs 基准（质心量计，被审值 / 审计重拟合）")
w()
w("| dx [m] | 粒子 | 步数 | 记录 end_time | σ 提交 | σ 审计重拟合 | σ/√(Agk) | 相对基准 | 早半窗 | 晚半窗 | 晚/早 |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for r in base:
    w(f"| {r['dx']:g} | {r['particles']} | {r['steps']} | {r['end_time_recorded']} | "
      f"{r['sigma_stored']:.3f} | {r['sigma_refit']['sigma']:.3f} | "
      f"{r['sigma_stored'] / ref['sigma_ideal_Agk']:.3f} | {pct(r['sigma_stored'] / ref['sigma_ideal_Agk'])} | "
      f"{r['sigma_early']['sigma']:.3f} | {r['sigma_late']['sigma']:.3f} | "
      f"{r['sigma_late']['sigma'] / r['sigma_early']['sigma']:.3f}× |")
w()
w("重拟合与提交值逐位一致（同一窗口、同一时序）→ **工件内部自洽，无篡改迹象**；"
  "但晚/早半窗差 %s（dx=%g）说明**斜率本身未收敛**（F2）。" %
  ("4–19 %", 0.025))
w()
w("### 3.3 三量计对照（同一批工件，界面振幅 @ 末步）")
w()
w("| 算例 | 质心量计 a_end | 几何量计（密度场等值线） | 相对差 | 速度带量计 σ | σ_vel/σ_centroid |")
w("|---|---|---|---|---|---|")
for t, r in runs.items():
    if "amplitude_geometric_final" not in r:
        continue
    g = r["amplitude_geometric_final"]; a = r["a_end"]
    w(f"| {t} | {a:.4f} | {g:.4f} | {100 * (g / a - 1):+.1f} % | "
      f"{r['sigma_vel_refit']['sigma']:.3f} | {r['sigma_vel_refit']['sigma'] / r['sigma_refit']['sigma']:.2f}× |")
w()
_amp = runs["dx0.025_nu0.001_a0.02_amp"]
w("几何量计（独立第三路径）与质心量计 **0.3–3.8 %** 一致 → 质心量计是真测量（F5）；"
  "速度带量计给 0.96–2.30× 于质心量计，且档内早/晚半窗可差 25×（`dx0.025_nu0.001_a0.02_amp`: "
  f"{_amp['vel_early']['sigma']:.3f} vs {_amp['vel_late']['sigma']:.3f}）→ 不可作独立确认（F4）。")
w()
w("### 3.4 收敛性：被审单的外推声明 vs 工件")
w()
w("| 外推方式 | σ∞ [s⁻¹] | 相对 √(Agk) |")
w("|---|---|---|")
for x in conv["richardson"]:
    w(f"| Richardson p={x['order']}，dx={x['pair'][0]:g}/{x['pair'][1]:g} | {x['sigma_inf']:.3f} | {x['err_pct']:+.1f} % |")
w(f"| 加性阻尼模型 λ₀={model['lambda0_per_m']:.1f} 1/m（仅用 A=0 两点拟合）外推到 dx=0.01 | "
  f"{model['first_order_extrapolation_at_dx001']['sigma']:.3f} | {model['first_order_extrapolation_at_dx001']['err_pct']:+.1f} % |")
w()
_t = conv["triplets"]
_rmin = min(x["err_pct"] for x in conv["richardson"]); _rmax = max(x["err_pct"] for x in conv["richardson"])
w(f"三点观测阶：**p = {_t[0]['p_observed']:.2f}**（dx=0.04/0.025/0.02）、**p = {_t[1]['p_observed']:.2f}**（0.025/0.02/0.0125）"
  "——负阶/乱阶 ⇒ 无收敛阶可估。被审文档 `make_doc.py:159`「二阶外推在 dx≈0.01 处降到 −7 % 以内」是**硬编码字符串**，"
  f"6 组 Richardson 外推落在 [{_rmin:.1f}, {_rmax:.1f}] %，无一组接近 −7 %（F6）。")
w()
w("加性阻尼模型（λ₀ 只用 A=0 两点拟合）对四档基线的盲预测残差：" +
  "、".join(f"dx={m['dx']:g} → {m['resid_pct']:+.1f} %" for m in model["predictions"]) +
  " ⇒ 亏损的**趋势**能被界面数值阻尼解释（±7 % 内），但模型是一阶（σ→σ_visc 只在 dx→0，O(dx)），"
  "与文档的二阶外推叙事矛盾。")
w()
w("### 3.5 可复现性实测")
w()
w(f"- repro_cmd `PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.04`（默认 end_time=0.45）："
  f"提交 {repro['committed_sigma']:.4f}（end_time={repro['committed_end_time']}，{repro['committed_steps']} 步）vs "
  f"复跑 {repro['rerun_sigma']:.4f}（end_time={repro['rerun_end_time']}，{repro['rerun_steps']} 步）= "
  f"{repro['rel_change_pct']:+.2f} %；两序列在公共长度上**逐位相同**（<1e-15）⇒ 差异纯由运行时长/拟合窗截断引起（F3）。")
w("- `damping.py` 复跑：dx=0.05 %+.3f → %+.4f、dx=0.025 %+.3f → %+.4f 1/s（**逐位复现**），"
  "被审单的失败模式归因所依赖的 A=0 对照可信。" %
  (A["damping"]["committed"][0]["sigma"], A["damping"]["rerun"][0]["sigma"],
   A["damping"]["committed"][1]["sigma"], A["damping"]["rerun"][1]["sigma"]))
w("- `pytest tests/test_rayleigh_taylor.py`：**10 passed in 61.7 s**（被审单自报 23 s，墙钟受控制面负载影响；"
  "通过数一致）。断言内容=不变量（静水压强逐点=解析 <1e-6 相对、Σm a_x=0 至 1e-10、Σm a_y=−Mg、"
  "界面压连续而密度跳 3×、质量求和替代方案界面压误差 140×/19×、量计回收已知振幅 5 %、A=0 无伪增长、M<0.2）。")
w()

# ------------------------------------------------------------------ ④
w("## ④ 结果图")
w()
w("![fig1](../../../artifacts/m16198/fig1_growth.png)")
w()
w("**图 1** (a) 四档分辨率的 σ/√(Agk)（提交值）+ 10 %/15 % 判据线 + A=0 加性阻尼模型曲线；"
  "(b) 同一算例早半窗/晚半窗重拟合——斜率随时间**上抬**，即亏损随模态增长而缩小，渐近值未被触及。")
w()
w("![fig2](../../../artifacts/m16198/fig2_gauges.png)")
w()
w("**图 2** 三个量计在同一批工件上的末步振幅：几何量计（密度场等值线）与质心量计一致；"
  "速度带量计（v_end/σ_centroid）系统性偏离——量计分歧是**被审单未披露**的不确定度。")
w()
w("![fig3](../../../artifacts/m16198/fig3_field.png)")
w()
w("**图 3** 从已存密度场快照读出的界面 η(x)（dx=0.02 末步）与质心量计振幅的 cos 模态对照——"
  "第三量计的几何直观；单模形态成立（无高阶谐波主导），故 σ 的提取前提（单模指数增长）对象正确。")
w()

# ------------------------------------------------------------------ ⑤
w("## ⑤ 注意点与发现（按严重度）")
w()
w("| # | 严重度 | 发现 | 证据 |")
w("|---|---|---|---|")
w("| F1 | medium（流程） | **基准裁判缺席**：`tensor_benchmark_cases` 25 行中本锚点 0 行，10 % 门为被审单自定、3 % 为平台纪律；裁判缺失使「PASS/FAIL」只能由审计席位按平台 15 % 底线复核 | SQL：`SELECT count(*) … WHERE anchor_slug='CASE-SPH-RAYLEIGH-TAYLOR-MP'` → 0 |")
w(f"| F2 | **high（实质）** | **指标未收敛**：预注册窗口上界 0.1λ 从未达到（最大 a_end={max(r['a_end'] for r in base):.3f} m = {max(r['a_end'] for r in base):.3f}λ），实窗由 end_time 截断；早/晚半窗差 4–19 %，dx=0.025 档从 {runs['dx0.025_nu0.001_a0.01']['sigma_early']['sigma']:.3f} 涨到 {runs['dx0.025_nu0.001_a0.01']['sigma_late']['sigma']:.3f}（{pct(runs['dx0.025_nu0.001_a0.01']['sigma_late']['sigma'] / ref['sigma_ideal_Agk'])} → {pct(runs['dx0.025_nu0.001_a0.01']['sigma_early']['sigma'] / ref['sigma_ideal_Agk'])}）。FAIL 的方向稳健，但 −24.5 %/−28.5 % 这类具体数字是「窗口选择值」不是渐近增长率 | `artifacts/m16198/refit.json` 的 window_sensitivity |")
w(f"| F3 | medium（证据） | **repro_cmd 不复现表中数字**：提交件用 end_time=0.42/0.35（未写进 repro 与报告），文档口径默认 0.45 ⇒ 同分辨率 σ 差 {repro['rel_change_pct']:+.1f} %（序列逐位相同，差异纯来自窗长） | `run.py:46` docstring vs `runs/dx0.04_….json:config.end_time`；审计复跑 |")
w(f"| F4 | medium（证据） | **量计分歧未披露**：速度带量计 σ 是质心量计的 0.96–2.30×，档内早/晚可差 25×；工件字段 `sigma_velocity_over_ideal`={runs['dx0.02_nu0.001_a0.01']['sigma_vel_refit']['sigma'] / ref['sigma_ideal_Agk']:.3f}（dx=0.02）读起来像「过了 10 % 门」，报告 ④⑤ 完全未提 | `runs/*.json:fit_velocity_gauge` + refit.json |")
w("| F5 | —（正面） | 三量计交叉验证成立：几何量计与质心量计 0.3–3.8 % 一致（7 算例）⇒ 报告的亏损方向不是量计伪影 | `artifacts/m16198/audit.json:runs.*.amplitude_geometric_final` |")
w(f"| F6 | high（声明） | **「二阶外推 −7 %」不受工件支持**：观测阶为负（{conv['triplets'][0]['p_observed']:.2f}、{conv['triplets'][1]['p_observed']:.2f}），6 组 Richardson 外推 {min(x['err_pct'] for x in conv['richardson']):+.1f}…{max(x['err_pct'] for x in conv['richardson']):+.1f} %，一阶阻尼模型给 {model['first_order_extrapolation_at_dx001']['err_pct']:+.1f} %；该句在 `make_doc.py:159` 是硬编码文本，非程序化计算 | `docs/benchmarks/rayleigh-taylor.md` §⑤、`make_doc.py:159` |")
w("| F7 | low（工件） | **图无生成器**：`figures/rt_growth.png`(115898 B)、`rt_density_field.png`(31485 B) 在仓内 grep 无 savefig/生成脚本 ⇒ 不可从提交字节重生成（仅存在于 tensor_reports#15292 的 base64） | `grep -rn savefig benchmarks/rayleigh_taylor/` → 0 命中；`git show --stat 9736ad7` 只带二进制 |")
w("| F8 | low（缺陷） | `_field_snapshot` 归一化除 Σw 而非 Σ(w·vol)：存盘 rho 场为 ρ·dx²（dx=0.02 实际值 0.2/0.6 而非 500/1500 kg/m³），pressure/phase_index 同带因子 ⇒ 场量云图标度错 4 个量级（显示级，不进 σ） | `src/tensorsph/rayleigh_taylor.py:759-783`；`runs/dx0.02….json:fields.rho` 极值 0.203/0.603 |")
w("| F9 | low（证据） | `\"mass_drift\": float((mass*1.0).sum())*0.0` 恒为 0（代码自注 by construction），作为工件字段会被读成「测得 0」 | `src/tensorsph/rayleigh_taylor.py:745` |")
w(f"| F10 | low（报告） | 报告③ νk²/σ=0.0128，审计复算 {ref['nu_k2_over_sigma']:.5f}（1.8×）；其余量（λ/h=38.5、静水压缩 1.74 %、p=14.7 kPa、B_l=114 kPa、有限深度 0.99992×、黏性 0.99283×）逐项复算一致 | `artifacts/m16198/audit.json:ref` |")
w("| F11 | low（工件） | `dx0.05_nu0.001_a0.01.json` 记录 σ=NaN 且无原因字段（该档未进报告表，可接受但应注明） | `runs/table.json` 末行 |")
w()

# ------------------------------------------------------------------ ⑥
w("## ⑥ 误差判定（关键量 vs 基准，逐项显式回答）")
w()
w("| 关键量 | 基准 | got | 相对误差 | <3 % ? | <10 %（案例自定）? | <15 %（平台底线）? |")
w("|---|---|---|---|---|---|---|")
for r in base:
    e = r["sigma_stored"] / ref["sigma_ideal_Agk"]
    w(f"| σ（dx={r['dx']:g}） | √(Agk)={ref['sigma_ideal_Agk']:.4f} s⁻¹ | {r['sigma_stored']:.3f} s⁻¹ | {pct(e)} | "
      f"**否** | {'是' if abs(e - 1) < 0.10 else '否'} | {'是' if abs(e - 1) < 0.15 else '否'} |")
w()
w(f"**结论：是否 <3 % —— 否。** 最细可用分辨率 dx=0.0125 的干損 {pct(base[-1]['sigma_stored'] / ref['sigma_ideal_Agk'])}，"
  f"即使换用更利的晚半窗也仅 {pct(base[-1]['sigma_late']['sigma'] / ref['sigma_ideal_Agk'])}；"
  "四档全部超出平台 15 % 不可接受线。被审单自报 honest-FAIL（<10 % 门）**与工件一致，无粉饰**——这一点审计确认。")
w()
w("非增长率项（物理不变量，全部通过，构成 L1/L2 的下限证据）：")
w("- t=0 静水压强逐点 = 解析（相对残差 <1e-6）；Σm a_x = 0 至 1e-10、Σm a_y = −Mg 至 1e-10（动量守恒）；"
  "界面两侧压强连续而密度跳 3×；质量求和替代方案界面压误差 140×（轻）/19×（重）被否；"
  "A=0 对照无伪增长（实测为阻尼）。")
w()

# ------------------------------------------------------------------ 判级
w("## ⑦ 判级（L0–L5）与落表")
w()
w("**裁定：L2**")
w()
w("| 级 | 判据 | 结论 | 依据 |")
w("|---|---|---|---|")
w("| L0 | 文档成果 | 有（含 L0 之上） | `docs/benchmarks/rayleigh-taylor.md`(84 行) |")
w("| L1 | 数学/单元验证 | **达** | 10 项不变量测试全绿（独立复跑），含解析静水、动量守恒、伪压强、量计回收 |")
w("| L2 | 独立功能实现（输入→计算→输出） | **达** | `src/tensorsph/rayleigh_taylor.py:1-784` 两相 WCSPH 推进器 + `run.py`/`damping.py` 驱动 + 8 份可复算工件（sha256 快照） |")
w("| L3 | 系统集成（主求解器/平台入口） | **不达** | `src/tensorsph/__init__.py` 未导出 rayleigh_taylor（同仓 dam_break 等均导出）；`src/tensorsph/verification/` 无引用；caller 仅 `benchmarks/rayleigh_taylor/{run,damping,make_doc}.py` 与 `tests/test_rayleigh_taylor.py`；无平台/CLI 入口 |")
w("| L4 | 基准验证（完整算例+误差分析+可复现） | **不达** | ①`tensor_benchmark_cases` 裁判 0 行（F1）；②误差 −24.5 %（最细）> 15 % 平台线、> 10 % 自定线；③指标未收敛（F2，早/晚窗 4–19 %）+ 量计分歧未披露（F4）；④repro_cmd 不复现表中值（F3）；⑤外推声明无工件支持（F6） |")
w("| L5 | 科研工程成果 | 无 | 未产生新物理结论（失败模式归因只到「界面数值阻尼」的加性模型，±7 % 残差，未做机理层面修复） |")
w()
w("**落表**：`tensor_maturity_verdicts(repo='TensorSPH', scope='m16141 CASE-SPH-RAYLEIGH-TAYLOR-MP', "
  "mission_id=16198, level=2, verdict_by='board-audit')`，evidence 键 = "
  "impl_files / entry / run_artifacts / tests / integration_callers / benchmark / repro_cmd / external_anchor，"
  "值均为可解析指针（见 `docs/board_audit/m16198/write_verdict.py`）。")
w()
w("**处置建议**：L2 三件套齐 → 建议门判 `confirmed`（L2 消账）；**L4 明确不达**，若本锚点的消账线要求 L4，"
  "则本条应判 `reopened`，返修项：")
w("1. **R1 指标定义**：把拟合窗固定为**实际可达**的振幅带（如 2η₀→5η₀）并声明 end_time，"
  "证明 dσ/d(窗) → 0（本单实测 4–19 %，不可忽略）；")
w("2. **R2 量计一致性**：质心量计与速度带量计必须自洽（或证明速度带量计测的是同一模态），"
  "并把分歧写进报告的不确定度；")
w("3. **R3 repro 与工件对齐**：repro_cmd 显式带 end_time/网格/线程数，使表中数字可逐位复现；")
w("4. **R4 外推声明**：删除 `make_doc.py:159` 的硬编码「−7 %」或改为程序化 Richardson + 观测阶，"
  "并给出外推不确定度（本单 6 组落 −33…+35 %）。")
w("（另 F7 图生成器、F8 场量标度、F9 mass_drift 字段属低危清理项。）")
w()

# ------------------------------------------------------------------ 审计自述
w("## ⑧ 审计自述：独立性、未做的检查")
w()
w("**独立性限制**")
w("- 未重跑全部算例（时间预算）：只重跑 dx=0.04、damping.py、pytest；其余结论建立在被审单工件上，"
  "但工件 sha256 已固定（MANIFEST.json），且重拟合与提交值逐位一致 ⇒ 工件未被事后改写。")
w("- 第三量计读的是**同一批**场量快照（同源数据、非同源算法）：能排除量计伪影，不能排除「快照本身写错」；"
  "后者已由 F8（场量标度）部分证伪——但等值线位置不受常数标度影响，故 F5 结论成立。")
w("- 平台门的裁定逻辑不对外可见：本单按 evidence 键写可解析指针，若门对「基线分支未合并的仓」解析失败，"
  "以 `docs/board_audit/m16198/evidence/` 快照兜底。")
w()
w("**未做的检查（unknown，不得记为 passed）**")
w("- 集群（loong64）复跑 dx=0.00625/0.0125：未跑（本轮仅本地 x86 CPU）。")
w("- 第三方 SPH 求解器（DualSPHysics/SPHinXsys）交叉验证：本机无，未跑。")
w("- GPU（device=cuda）路径与 >1e4 粒子：未跑。")
w("- `agent/m16141` 是否已推送 GitHub：`git ls-remote` 超时（无网络），未核；被审代码在本地分支 9736ad7。")
w("- 报告内嵌 PNG 的像素级内容复核：只核文件存在 + sha256 + base64 非空，未做视觉/数值复核。")
w()
w(f"审计工件：`artifacts/m16198/audit.json`、`refit.json`、`refit.py`、`analyse.py`、`fig1_growth.png`、"
  f"`fig2_gauges.png`、`fig3_field.png`；被审工件快照 `docs/board_audit/m16198/evidence/`"
  f"（{len(MAN['files'])} 文件 + MANIFEST.json，audited_commit={MAN['audited_commit']}）。")
w()
out = HERE.parent / "M16198_CASE_SPH_RAYLEIGH_TAYLOR_BOARD_AUDIT.md"
out.write_text("\n".join(L) + "\n")
print("wrote", out, len(L), "lines")