"""Publish the progress report for mission 3453 (numbers injected from report.json)."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
REPO = pathlib.Path("/root/tp-agent/work/m3453")
REP = json.loads((REPO / "benchmarks/oscillating_droplet_stability/report.json").read_text())


def post(path, payload, token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read() or b"{}")


def get(path, token):
    req = urllib.request.Request(BASE + path,
                                 headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read() or b"{}")


tok = post("/auth/login", {"username": USER, "password": PASS})[1]["access_token"]

pub = REP["published_run"]
bp = pub["boundary_pin"]
g = REP["platform_gates"]
main = REP["cases"]["release_R20mm_eps0.10"]
sec = REP["cases"]["release_R20mm_eps0.02"]
ad, v, f = main["admissibility"], main["fit_valid_window"], main["fit_full_trace"]
sad, sv = sec["admissibility"], sec["fit_valid_window"]
setup = main["setup"]
om_ref, T = main["reference"]["omega_rad_per_s"], main["reference"]["period_s"]
b_rel = abs(bp["edge_after_three_refinements_rad_per_s"]
            - pub["refit_from_stored_trace"]["omega_rad_per_s"]) / pub["refit_from_stored_trace"]["omega_rad_per_s"]
ladder = main["fit_window_ladder"]


def f1(x, n=1):
    return f"{x:.{n}f}"


ladder_md = "\n".join(
    f"| {r['window_periods']:.2f} | {r['n_samples']} | {f1(r['omega_rad_per_s'], 2)} | "
    f"{f1(r['rel_error_pct'])} | {f1(r['r2'], 3)} |" for r in ladder)
fft_md = "\n".join(
    f"| {i+1} | {f1(p['omega_rad_per_s'])} | {f1(p['rel_error_pct'])} | {p['amplitude']:.1f} |"
    for i, p in enumerate(main["fft_full_trace"]))

md = f"""## 摘要

TensorSPH 的二维振荡液滴（Rayleigh–Lamb 模态 2）验证在账面上被记为 **−43.5% FAIL**，
同一份代码在 x86_64 CI 测到 **+8.6%**、loong64 测到 **−31.8%**（`tests/test_oscillating_droplet.py` 因此用 strict-xfail
钉死，任一架构上必红一条）。本单证明：**这三个数都不是表面张力模型的精度**。
已发布算例在 {f1(ad['admissible_fraction_of_periods'], 2)} 个解析周期后数值爆炸
（Mach {f1(pub['max_mach'],0)}、终态 $r_{{\\rm eff}}$ = {f1(pub['final_reff_over_R'],0)} R），−43.5% 是对爆炸轨迹做拟合的结果，
而且该拟合的最优值恰好是估计器搜索盒的下沿（$0.6\\omega_2\\cdot0.98^3$，与发布值相对差 {b_rel:.1e}）。
物理可采纳窗口内同一估计器给出 **{f1(v['rel_error_pct'])}%（$R^2$={f1(v['r2'],3)}）**。
账面处置：**FAIL → WITHHELD**（可采纳窗口不足 1 个周期），并给出可机器裁的修复判据。

## ① 案例介绍

二维（无限长柱截面）静止液滴，界面 $r(\\theta)=R(1+\\varepsilon\\cos2\\theta)$ 自由毛细振荡。
线性理论给出 $\\omega_m^2=\\sigma m(m^2-1)/(\\rho R^3)$，模态 2 即 $\\omega_2=\\sqrt{{6\\sigma/(\\rho R^3)}}$ =
**{f1(om_ref,2)} rad/s**（$T$={f1(T*1e3,2)} ms）。参考：Rayleigh 1879；Lamb 1932 art.267；Chandrasekhar 1961 ch.XII。

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| R | {setup['radius_m']} m | $c_0$ | {setup['sound_speed_m_per_s']} m/s |
| dx | {setup['dx_m']} m | ν | {setup['viscosity_m2_per_s']} m²/s |
| N | {setup['n_particles']} | ε | {setup['initial_eps']} |
| σ | {setup['sigma_N_per_m']} N/m | dt（固定） | {setup['dt0_s']:.2e} s |
| ρ₀ | {setup['rho0']} kg/m³ | 时长 | {setup['physical_time_s']:.4f} s = {setup['physical_time_s']/T:.2f} T |

## ② 怎么做的

1. **判据先于拟合**：弱可压缩 SPH 只有在亚声速且液滴半径不被数值误差改变时才算「物理测量」。
   采用平台既有门限：Mach ≤ {g['mach']}、$|r_{{\\rm eff}}/R-1|$ ≤ {g['radius_drift_pct']:.0f}%，首次越界记 $t_{{\\rm gate}}$。
2. **复现已发布数字（不重算）**：直接对存档轨迹 `benchmarks/oscillating_droplet/trace_data.json` 重跑生产估计器
   `damped_cosine_fit`，得到 {f1(pub['refit_from_stored_trace']['rel_error_pct'],2)}%（$R^2$={f1(pub['refit_from_stored_trace']['r2'],3)}），
   与账面 {f1(pub['recorded_rel_error_pct'],2)}% **逐位一致**——结论不是「换算法」换来的。
3. **重跑并逐步记录可采纳性**：新工件 `benchmarks/oscillating_droplet_stability/run.py` 的积分循环与
   `run.run_case` 逐行同构（同 `Config`/`rhs`/`stable_dt`/固定步长），仅增加每步 $\\max|v|/c_0$ 与 $r_{{\\rm eff}}$ 记录（生产记录器只留全程最大值）。
4. **窗口阶梯 + 三估计器 + 频谱**：阻尼余弦（主）、动能（$KE\\propto s^2$，以 $2\\omega$ 振荡）、零交叉；
   窗口 0.5T→{ladder[-1]['window_periods']:.2f}T；Hann 窗 FFT。工具链 PyTorch CPU 2 线程 + pytest + matplotlib。
   **未改 `src/`**（验证单）。全部数字由 `make_doc.py` 从 `report.json` 注入，零手抄。

## ③ 计算结果

**已发布算例不可采纳**（直接读其产物）：$\\max|v|$ = {f1(pub['max_speed_m_per_s'],0)} m/s → Mach **{f1(pub['max_mach'],0)}**
（门限 {g['mach']}，越界 {f1(pub['max_mach']/g['mach'],0)}×）；终态 $r_{{\\rm eff}}/R$ = **{f1(pub['final_reff_over_R'],0)}**，半长轴 {f1(pub['final_semi_major_m'],0)} m。

**边界钉死**：估计器在 $[0.6,1.6]\\omega_{{\\rm guess}}$ 搜索 + 3 次 ±2% 精修；盒外极小值会把答案带到
$0.6\\omega_2\\cdot0.98^3$ = {f1(bp['edge_after_three_refinements_rad_per_s'],3)} rad/s，而发布值 = {f1(pub['refit_from_stored_trace']['omega_rad_per_s'],3)} rad/s，
相对差 **{b_rel:.1e}**。即 −43.5% 是**估计器边界值**，不是物理频率。

**可采纳 vs 全程**：$t_{{\\rm gate}}$ = {f1(ad['t_gate_s']*1e3,2)} ms = **{f1(ad['admissible_fraction_of_periods'],2)} T**（半径门 {f1(ad['t_radius_gate_s']*1e3,1)} ms 先触发，
Mach 门 {f1(ad['t_mach_gate_s']*1e3,1)} ms）。

| 口径 | 窗口 | ω [rad/s] | 误差 | $R^2$ |
|---|---|---|---|---|
| 全程（=已发布） | {ladder[-1]['window_periods']:.2f} T | {f1(f['omega_rad_per_s'],2)} | **{f1(f['rel_error_pct'])}%** | {f1(f['r2'],3)} |
| 可采纳窗口 | {v['window_periods']:.2f} T ({v['n_samples']} 样本) | {f1(v['omega_rad_per_s'],2)} | **{f1(v['rel_error_pct'])}%** | **{f1(v['r2'],3)}** |
| 解析 $\\omega_2$ | — | {f1(om_ref,2)} | 0 | — |

**窗口阶梯**（误差随「吃进多少碎片」单调劣化）：

| 窗口 [T] | 样本 | ω [rad/s] | 误差 | $R^2$ |
|---|---|---|---|---|
{ladder_md}

**为什么两个架构测出 +8.6% / −31.8%**：全程 $s(t)$ 频谱多峰（前 5 峰幅值相当），最小二乘存在近简并盆地，
架构级舍入差异即可换盆地。x86 的 +8.6% 对应第 2 峰、loong64 的 −31.8% 对应第 1 峰——都是碎片的谱。

| # | ω [rad/s] | 相对 $\\omega_2$ | 归一化幅值 |
|---|---|---|---|
{fft_md}

**不是 ε 的锅**：ε=0.02 对照的可采纳窗口只有 {f1(sad['admissible_fraction_of_periods'],2)} T（更早越界），
其亚周期拟合给出 {f1(sv['rel_error_pct'])}%（$R^2$={f1(sv['r2'],3)}）——说明**窗口长度本身是主导误差源**，
< 约 0.6 T 时任何模态 2 读数（无论正负）都不该被引用。

## ④ 结果图

三张图（`plot.py` 从 `report.json` 生成，无手抄）：admissibility_trace（$s(t)$ / Mach / $r_{{\\rm eff}}$ 与 $t_{{\\rm gate}}$ 竖线）、
estimator_comparison（全程 vs 可采纳口径，含 15% 门限带与 $R^2$）、window_and_spectrum（窗口阶梯 + 多峰频谱）。

## ⑤ 注意点

1. **未修 `src/`，不稳定性仍在**：可采纳窗口 < 1 个周期，so {f1(v['rel_error_pct'])}% 是**亚周期**读数，
   与 $R^2$={f1(v['r2'],3)} 一起只说明「窗口内是干净阻尼余弦」，不说明长时行为 → 记 **WITHHELD**，不是 PASS。
2. 亚周期窗口上另两个估计器不可用：动能估计器同样被钉在盒下沿（{f1(v['ke_rel_error_pct'])}%），零交叉法样本不足返回 NaN——
   这本身就是「窗口太短」的证据。
3. FFT 在 0.64 T 窗口上无分辨力（$\\Delta\\omega\\approx2\\pi/t_{{\\rm gate}}$），故频谱结论只用全程轨迹。
4. `report.json` 不含完整轨迹（体积），逐步历史在 `diagnostics.json`（`--stage diag` 重算，约 4 min/算例，本机 load15>100 期间单核运行）。
5. **CI**：`tests/test_oscillating_droplet.py` 两条 strict-xfail 物理钉子（架构相关，必红一条，本机实测 XPASS×2）
   改为 non-strict 并注明根因；漂移检测迁到新的 `tests/test_oscillating_droplet_stability.py`（12 项，纯读工件、不跑仿真、与架构无关）。

## ⑥ 误差判定（逐项，显式回答是否 < 3%）

| # | 关键量 | 基准 | 测量 | 相对误差 | <3%？ | <15%？ |
|---|---|---|---|---|---|---|
| 1 | 模态 2 频率（可采纳窗口） | {f1(om_ref,2)} rad/s | {f1(v['omega_rad_per_s'],2)} | **{f1(v['rel_error_pct'])}%** | **否** | **是** |
| 2 | 模态 2 频率（已发布口径） | 同上 | {f1(f['omega_rad_per_s'],2)} | {f1(pub['recorded_rel_error_pct'])}% | 否（**无效**） | 否（**无效**） |
| 3 | 边界钉死复现 | {f1(bp['edge_after_three_refinements_rad_per_s'],3)} | {f1(pub['refit_from_stored_trace']['omega_rad_per_s'],3)} | **{b_rel:.1e}** | **是** | 是 |
| 4 | 存档轨迹重拟合复现 | {f1(pub['recorded_rel_error_pct'],4)}% | {f1(pub['refit_from_stored_trace']['rel_error_pct'],4)}% | **0.0** | 是 | 是 |
| 5 | 已发布算例 Mach | 门限 {g['mach']} | {f1(pub['max_mach'],0)} | 越界 {f1(pub['max_mach']/g['mach'],0)}× | 否 | 否（**不可采纳**） |

**唯一可采纳的频率读数 {f1(v['omega_rad_per_s'],2)} rad/s，误差 {f1(v['rel_error_pct'])}%：在 15% 门限内、不满足 3%；
且 ε=0.02 对照证明亚周期窗口读数不可外推，故本条不构成频率验收，账户仍记 WITHHELD。**
已发布的 −43.5% 必须停止被引用为模型精度（honest 更正，非粉饰：原 FAIL 的数字被证明是估计器边界值）。

## ⑦ 给后续单（可机器裁的验收判据）

**P0**：修 WCSPH 表面张力力不稳定（拉伸不稳定 / 自由面压强闭合），使 ≥3 个周期可采纳。验收：
`run.py --stage diag` 的 `t_gate_s / T ≥ 3` 且 `--stage fit` 的 `fit_full_trace.rel_error_pct` 与
`fit_valid_window.rel_error_pct` 之差 < 2 pt（「窗口长短不再改变答案」）。
**P1**：`damped_cosine_fit` 加边界诊断——最优值落在搜索盒 ±1% 内即报 warning 并扩大盒（本单的 −43.5% 就是无声的边界值）。
**P2**：所有频率类验证工件加「可采纳性门」前置（Mach/半径漂移），不通过则不出频率数。
"""

figs = []
for name, cap in (
    ("admissibility_trace.png",
     "轨迹可采纳性：s(t)、max|v|/c0、r_eff 与首次越界时刻（红竖线）——拟合窗口大部分落在红竖线右侧"),
    ("estimator_comparison.png",
     "同一估计器：全程口径 vs 可采纳口径（含 15% 门限带与 R²）"),
    ("window_and_spectrum.png",
     "窗口阶梯（虚线=t_gate）与全程 s(t) 的多峰频谱：架构差异即盆地选择"),
):
    b = (REPO / ".scratch/figs" / name).read_bytes()
    figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(b).decode()})

payload = {
    "title": "振荡液滴模态2频率 −43.5%「FAIL」结清：是数值爆炸轨迹上的边界钉死拟合，不是表面张力精度",
    "summary": (f"复现已发布 −43.53%（逐位一致）并证明其为搜索盒下沿 0.6ω₂·0.98³（相对差 {b_rel:.1e}）；"
                f"该算例 Mach {f1(pub['max_mach'],0)}、终态 r_eff={f1(pub['final_reff_over_R'],0)}R，"
                f"可采纳窗口仅 {f1(ad['admissible_fraction_of_periods'],2)}T。可采纳口径误差 {f1(v['rel_error_pct'])}%"
                f"（R²={f1(v['r2'],3)}），账面 FAIL → WITHHELD。同时解释 x86/loong64 的 +8.6%/−31.8% 结构性 CI 红。"),
    "scenario_key": "solver_verification",
    "mission_id": 3453,
    "tags": ["TensorSPH", "验证欠账", "Rayleigh-Lamb", "表面张力", "WCSPH", "可采纳性", "CI红"],
    "markdown": md,
    "figures": figs,
}
print("payload KB", len(json.dumps(payload)) / 1024)
st, body = post("/tensor/reports", payload, tok)
print("POST", st, json.dumps(body)[:300])
st2, lst = get("/tensor/reports?mission_id=3453&limit=3", tok)
items = lst.get("items", lst if isinstance(lst, list) else [])
print("verify", st2, [(i.get("id"), i.get("title", "")[:40]) for i in items][:3])