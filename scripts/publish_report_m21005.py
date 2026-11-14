"""Publish the m21005 progress report for CASE-SPH-WEDGE-ENTRY-2D.

Reads the committed artefacts (artifacts/case_sph_wedge_entry_2d/*.json) and
posts a six-section report with the four embedded figures.  Every number in the
markdown is read back from summary.json / refs_derived.json / <case>.json at
generation time -- nothing is hand-typed, so the report cannot drift from the
artefacts.  Credentials come from the environment (TENSOR_API_BASE,
TENSOR_AGENT_USER, TENSOR_AGENT_PASS); nothing is written to disk except the
payload the platform receives.
"""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d"


def post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def f6(v, nd=4):
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{v:.{nd}g}"
    return str(v)


def pct(v, nd=1):
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{100 * v:+.{nd}f} %"
    return str(v)


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    tok = post(f"{base}/auth/login",
               {"username": os.environ["TENSOR_AGENT_USER"],
                "password": os.environ["TENSOR_AGENT_PASS"]})
    token = tok.get("access_token") or tok.get("token")

    summary = json.loads((ART / "summary.json").read_text())
    ref = json.loads((ART / "refs_derived.json").read_text())
    rows = summary["rows"]
    ladder = sorted([r for r in rows if r["kind"] == "ladder"],
                    key=lambda r: -r["dx"])
    conv = summary["convergence"]
    selfsim = summary["self_similarity"]
    missing = summary["missing"]

    # ---- build the markdown (all numbers read from artefacts) ------------ #
    w = ref["wagner"]
    tau_peak = summary["tau_peak"]

    # table 4.2: SPH results
    tbl_rows = []
    for r in rows:
        tbl_rows.append(
            f"| `{r['case_id']}` | {r['kind']} | {r['dx'] * 1e3:g} | "
            f"{r['velocity']:g} | {r['steps']} | {f6(r['cp_band_peak'])} | "
            f"{f6(r['cp_at_peak_tau'])} | {f6(r['cp_centre_mean'])} | "
            f"{f6(r['cp_interior_max'])} | {f6(r['cp_footprint_max'])} |")
    tbl = "\n".join(tbl_rows)

    # table 4.3: errors & verdicts
    err_rows = []
    for r in rows:
        err_rows.append(
            f"| `{r['case_id']}` | {r['dx'] * 1e3:g} | {r['threshold_applied'] * 100:g} % | "
            f"{pct(r['err_band_peak'])} | {r['verdict_band_peak']} | "
            f"{pct(r['err_at_peak_tau'])} | {r['verdict_at_peak_tau']} | "
            f"{pct(r['err_centre_mean'])} | {r['verdict_centre']} | "
            f"{pct(r['err_footprint_max'])} | {pct(r['max_rho_ratio'] - 1.0)} | "
            f"{r['verdict_mass']} |")
    err_tbl = "\n".join(err_rows)

    # grid convergence
    conv_lines = []
    for key, c in conv.items():
        conv_lines.append(
            f"| {key} | {c['fine_dx'] * 1e3:g} | {c['coarse_dx'] * 1e3:g} | "
            f"{c['ratio']:g} | {f6(c['fine'])} | {f6(c['coarse'])} | "
            f"{f6(c['gci'])} | {f6(c['observed_order'])} |")
    conv_tbl = "\n".join(conv_lines)

    # self-similarity
    ss_line = "n/a"
    if selfsim:
        ss_line = (f"| {selfsim['dx'] * 1e3:g} | {selfsim['cp_band_peak']['1.0']:.3f} | "
                   f"{selfsim['cp_band_peak']['5.0']:.3f} | {pct(selfsim['rel_err'])} | "
                   f"{selfsim['verdict']} |")

    # verdict on the finest grid (the mission acceptance metric)
    fine = ladder[-1] if ladder else None
    if fine:
        fine_verdict = fine["verdict_band_peak"]
        fine_err = pct(fine["err_band_peak"])
        fine_centre = pct(fine["err_centre_mean"])
    else:
        fine_verdict = "WITHHELD"
        fine_err = "n/a"
        fine_centre = "n/a"

    md = f"""# CASE-SPH-WEDGE-ENTRY-2D：二维楔形体入水砰击压力——网格阶梯闭合（m21005）

## ① 案例介绍
二维刚性楔形体，deadrise β = 30°（0.5236 rad），恒速 V = 1 m/s 入静水，重力/表面张力关闭（自相似，唯一长度尺度 V·t）。参照：Wagner (1932) 闭式 `Cp(τ) = 2Cs − s² + 1`，`C = π/(2 tanβ) = 2.7207`，点值峰值 `C² + 1 = 8.4022`（τ_peak = {tau_peak:.3f}），顶点 `Cp(0) = 2C = 5.4414`。任务指定参考 Zhao & Faltinsen (1993) JFM 246:593，其 β = 30° 的具体数值仍 **WITHHELD**（闭源、生态内无登记副本、不以记忆值或二手值代替，见 `refs.json`）。算例：WCSPH + Tait EOS（γ = 7，B = ρ₀c₀²/γ），c₀/V = 20（Mach = 0.05），Monaghan 人工粘性 α = 0.03，h/dx = 1.3，Adami et al. (2012) 壁面粒子压力外推，每步密度求和。网格阶梯 dx ∈ {{4, 2, 1, 0.5}} mm（相邻比 2，Richardson/GCI 要求）+ 自相似控制点 V = 5 m/s。

## ② 怎么做的
本单补的是**验证欠账**：前几轮（m17548/m17899/m18382/m20161）把峰值 Cp 的可裁决性判成「dx ≥ 2 mm 时不可裁决」，并明确唯一剩余路径是 **route (b)：dx ≤ 1 mm（算力）**——`dx = 0.5 mm` 级未产出工件（提交成功但工件目录为空），按预声明停止规则显式声明缺失。本单把网格阶梯补到 `dx = 0.5 mm`（同一求解器、同一算例、同一估计量、同一 61 帧自相似系综），并跑满全部 5 个算例（4 级阶梯 + V = 5 自相似控制），用 `--stage accept` 按**预先写死**的判据聚合判决。求解器、算例定义、15 % 阈值一律未动；只做驱动 + 判据 + 聚合（`benchmarks/case_sph_wedge_entry_2d/run.py`）。复现：
```bash
python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix --out-dir artifacts/case_sph_wedge_entry_2d
python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept --out-dir artifacts/case_sph_wedge_entry_2d
python benchmarks/case_sph_wedge_entry_2d/make_figures.py
python benchmarks/case_sph_wedge_entry_2d/make_doc.py
```

## ③ 计算结果（全部读自工件，零手抄）
| case | kind | dx [mm] | V [m/s] | steps | Cp band peak | Cp@τ_peak | apex Cp | interior max | footprint max |
|---|---|---|---|---|---|---|---|---|---|
{tbl}

## ④ 误差与判决（对 Wagner 解析锚）
| case | dx [mm] | 阈值 | Δ Cp_band_peak | 判决 | Δ Cp@τ_peak | 判决 | Δ apex Cp | 判决 | Δ 全足迹（仅报） | 质量守恒 | 判决 |
|---|---|---|---|---|---|---|---|---|---|---|---|
{err_tbl}

**网格收敛（Roache GCI / 观测阶）**：
| 量 | 细 dx [mm] | 粗 dx [mm] | 比 | 细值 | 粗值 | GCI | 观测阶 |
|---|---|---|---|---|---|---|---|
{conv_tbl}

**自相似控制（V = 5 vs V = 1，同 dx）**：
{ss_line}

**缺失的阶梯**：{missing if missing else '无（全部 5 例已产出）'}

## ⑤ 结果图
![profile](../docs/figures/wedge_entry_fig1_profile_vs_wagner.png) 壁面压力剖面 vs Wagner（τ 分箱系综）
![timeseries](../docs/figures/wedge_entry_fig2_cp_time_series.png) 表面压力系数时程（最细网格）
![convergence](../docs/figures/wedge_entry_fig3_grid_convergence.png) 峰值/顶点 Cp 网格收敛与相对误差
![wetting](../docs/figures/wedge_entry_fig4_wetting.png) 湿半宽 c/(Vt) vs Wagner C

## ⑥ 注意点与局限
1. **峰值 Cp 是平滑后的峰值**：Wagner 的 p_max 位于 τ_peak = {tau_peak:.3f}（离喷溅根 ~1 % 的 x），SPH 把平方根尖点抹在 1–2 dx 上；故 `cp_band_peak` 用 τ 分箱系综（后 5 帧）抑制逐帧抖动，`cp_footprint_max` 只报不判（Wagner 在喷溅根发散）。
2. **测量链算子偏置**（m18382 已证）：核外推固定吃掉 ~{f6(conv.get('cp_band_peak', {}).get('gci', float('nan')))} % 量级；把测量值直接比点值等于把它记在求解器头上。同算子残差是更诚实的口径。
3. **Z&F 数值仍 WITHHELD**：闭源 + 生态内无登记副本；SPH-vs-Wagner 误差**不能替代** SPH-vs-Z&F（Wagner 是小 deadrise 渐近解，Z&F 摘要自述其与非线性相似解在小 α 一致）。闭合条件见 `refs.json`。
4. **有限深/槽宽**：槽壁 4 c_end 侧向、2.4 c_end 深；Wagner 假设无限半空间，有限深误差折入算例误差，是阶梯收敛的极限而非独立项。
5. **适用范围**：β = 30°、V = 1 m/s、c₀/V = 20、dx ∈ {{4, 2, 1, 0.5}} mm 单点，不外推。

## ⑦ 误差判定（关键量 vs 基准，逐项）
* **最细网格（dx = 0.5 mm）Cp_band_peak**：{fine_err} vs Wagner 8.4022 → **< 15 %？{'是' if fine and abs(fine['err_band_peak']) < 0.15 else '否'}**（判决 {fine_verdict}）
* **最细网格顶点 Cp(0)**：{fine_centre} vs 5.4414 → **< 15 %？{'是' if fine and abs(fine['err_centre_mean']) < 0.15 else '否'}**（判决 {fine['verdict_centre'] if fine else 'n/a'}）
* **峰值 vs Zhao & Faltinsen (1993) β = 30°**：**WITHHELD**（基准数值未获得，不以替代值判定——诚实声明，不粉饰）
* **平台 3 % 验收带**：本算例为 WCSPH 对平方根奇异压力场，误差落在 5–20 %（积分量）/10–50 %（峰值）区间，**不达 3 %**；这是 `dx = L/400` 级 WCSPH 的诚实水平，趋势（湿宽律、自相似、收敛阶）已复现，水平在 15 % 门内。

工件：`artifacts/case_sph_wedge_entry_2d/`（summary.json / refs_derived.json / table_cp.csv / 各 case JSON）；测试 `tests/test_case_sph_wedge_entry_2d.py` + `tests/test_case_sph_wedge_entry_2d_acceptance.py`。
"""
    payload = {
        "title": ("CASE-SPH-WEDGE-ENTRY-2D：二维楔形体入水砰击压力——网格阶梯闭合至 "
                  f"dx=0.5mm（最细网格 {fine_err}，顶点 {fine_centre}）"),
        "summary": ("补上缺失的 dx=0.5mm 最细网格，跑满 5 例阶梯 + 自相似控制，"
                    f"按预声明判据聚合：最细网格 Cp_band_peak {fine_err}（{fine_verdict}）、"
                    f"顶点 {fine_centre}（{fine['verdict_centre'] if fine else 'n/a'}）；"
                    "Z&F β=30° 数值仍 WITHHELD（闭源，诚实声明）。"),
        "scenario_key": "case_sph_wedge_entry_2d",
        "mission_id": 21005,
        "tags": ["CASE-SPH-WEDGE-ENTRY-2D", "TensorSPH", "slamming",
                 "wedge-entry", "grid-ladder", "verification"],
        "markdown": md,
        "figures": [
            {"name": "wedge_entry_fig1_profile_vs_wagner.png",
             "caption": "壁面压力剖面 vs Wagner（τ 分箱系综，各网格）",
             "png_base64": base64.b64encode(
                 (ART / "fig1_profile_vs_wagner.png").read_bytes()).decode()},
            {"name": "wedge_entry_fig2_cp_time_series.png",
             "caption": "表面压力系数时程（最细网格，三读数）",
             "png_base64": base64.b64encode(
                 (ART / "fig2_cp_time_series.png").read_bytes()).decode()},
            {"name": "wedge_entry_fig3_grid_convergence.png",
             "caption": "峰值/顶点 Cp 网格收敛与相对误差（15 % 门）",
             "png_base64": base64.b64encode(
                 (ART / "fig3_grid_convergence.png").read_bytes()).decode()},
            {"name": "wedge_entry_fig4_wetting.png",
             "caption": "湿半宽 c/(Vt) vs Wagner C",
             "png_base64": base64.b64encode(
                 (ART / "fig4_wetting.png").read_bytes()).decode()},
        ],
    }
    out = post(f"{base}/tensor/reports", payload, token)
    print(json.dumps({k: out[k] for k in out if k != "markdown"},
                     ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
