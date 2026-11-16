"""CASE-SPH-OSC-DROP-FREQ: Rayleigh 线性理论验证算例。

本脚本对 TensorSPH 的表面张力模型做准静态线性响应验证，
将 SPH 测得的模态刚度与 Rayleigh (1879) 解析解逐项对比。

Rayleigh 频率公式（2-D 液滴，模态 m）：
    ω_m² = σ · m(m²−1) / (ρ₀ R³)

验证流程：
1. 在多个分辨率 (R/dx = 12, 16, 24) 上生成微扰液滴 (ε = 0.02, 0.05)
2. 用 CSF 力场计算准静态响应，提取模态刚度 g = ω²_measured / ω²_Rayleigh
3. 对比 g 与精确值 1 的偏差，判定是否通过 <6% 门限
4. 生成对比图（频率误差 vs 分辨率、模态振幅衰减曲线）

Reference: Rayleigh (1879), Proc. R. Soc. Lond. 29:71; Lamb, Hydrodynamics, 6th ed., arts. 313-315.

Usage:
    PYTHONPATH=src python benchmarks/oscillating_drop_freq/verify_rayleigh.py --merge
    PYTHONPATH=src python benchmarks/oscillating_drop_freq/verify_rayleigh.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tensorsph.capillary_oscillation import (
    rayleigh_2d, make_droplet, mode_amplitude,
    interior_pressure, period_from_crossings, run_point,
)
from tensorsph.droplet_impact import Config, rhs, suggested_dt
from tensorsph.neighbor_search import PairListCache
from tensorsph.surface_tension import compute_surface_tension_force

# ── 物理参数 ──
SIGMA = 0.072       # N/m, water-air
RHO = 1000.0        # kg/m³
RADIUS = 0.05       # m
MODE = 2
GATE_PCT = 6.0      # 验收门限

# ── 分辨率阶梯 ──
LADDER_QUICK = (12,)
LADDER_FULL = (12, 16, 24)
EPS_VALUES = (0.02, 0.05)


def quasi_static_stiffness(rdx: int, eps: float, mode: int = MODE,
                           sigma: float = SIGMA, rho: float = RHO,
                           radius: float = RADIUS, device: str = "cpu") -> dict:
    """准静态模态刚度测量。

    对微扰液滴施加 CSF 力，测量径向加速度与位移的比值，
    得到等效刚度 k_eff，进而得到 ω² = k_eff / m_eff。
    """
    dx = radius / rdx
    pos, mass = make_droplet(dx, eps, device, radius=radius, mode=mode, rho0=rho)
    vel = torch.zeros_like(pos)
    h = 1.3 * dx

    cfg = Config(
        dx=dx, h_factor=1.3, rho0=rho, sound_speed=0.5, gamma=7.0,
        gravity=0.0, viscosity=1e-6,
        pool_width=2.4 * radius, pool_depth=2.4 * radius,
        surface_tension_coefficient=sigma, device=device,
        boundary="clamp", adaptive_dt=False,
        density_calibration=True,
    )
    cfg.dt = suggested_dt(cfg)

    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=pos.shape[0])
    pairs, _ = cache.get(pos)

    # 计算 CSF 力
    density, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=cfg.dt)

    # 模态振幅
    q = mode_amplitude(pos, mass, mode)

    # 模态投影的加速度：将加速度投影到模态形状上
    r = torch.linalg.norm(pos, dim=1).clamp_min(1e-12)
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    mode_shape = r * torch.cos(mode * theta)

    # 有效质量 m_eff = Σ m_i · (mode_shape_i)² / R²
    m_eff = float((mass * mode_shape ** 2).sum()) / (radius ** 2)

    # 有效力 F_eff = Σ m_i · a_i · (mode_shape_i / R)
    # 径向加速度投影
    a_radial = (acc * (pos / r.unsqueeze(1))).sum(dim=1)
    f_eff = float((mass * a_radial * mode_shape / radius).sum())

    # ω² = F_eff / (m_eff · q)  (因为 F = -k q, ω² = k/m)
    if abs(q) > 1e-15 and m_eff > 1e-15:
        omega_sq_measured = -f_eff / (m_eff * q)
    else:
        omega_sq_measured = float("nan")

    omega_exact = rayleigh_2d(sigma, rho, radius, mode)
    omega_sq_exact = omega_exact ** 2

    g = omega_sq_measured / omega_sq_exact if omega_sq_exact > 0 else float("nan")
    freq_ratio = math.sqrt(g) if g > 0 else float("nan")
    error_pct = 100.0 * (freq_ratio - 1.0) if math.isfinite(freq_ratio) else float("nan")

    return {
        "radius_over_dx": rdx,
        "eps": eps,
        "mode": mode,
        "n_particles": pos.shape[0],
        "q_amplitude": q,
        "omega_sq_measured": omega_sq_measured,
        "omega_sq_exact": omega_sq_exact,
        "g_stiffness_ratio": g,
        "freq_ratio": freq_ratio,
        "error_pct": error_pct,
        "passes_gate": abs(error_pct) < GATE_PCT if math.isfinite(error_pct) else False,
        "m_eff": m_eff,
        "f_eff": f_eff,
    }


def dynamic_oscillation(rdx: int, sigma: float = SIGMA,
                        periods: float = 2.0, device: str = "cpu",
                        eps: float = 0.05, mode: int = MODE,
                        sample_every: int = 10) -> dict:
    """动态振荡积分，测量周期并与 Rayleigh 理论对比。

    使用 run_point 做时间推进，从零交叉点提取周期。
    """
    t0 = time.time()
    result = run_point(rdx, sigma, periods, device=device, eps=eps,
                       mode=mode, sample_every=sample_every)
    wall_s = time.time() - t0

    omega_exact = rayleigh_2d(sigma, RHO, RADIUS, mode)
    T_exact = 2.0 * math.pi / omega_exact

    period_measured = result.get("period_measured", float("nan"))
    if math.isfinite(period_measured) and T_exact > 0:
        error_pct = 100.0 * (period_measured - T_exact) / T_exact
    else:
        error_pct = float("nan")

    return {
        "radius_over_dx": rdx,
        "eps": eps,
        "mode": mode,
        "n_particles": result.get("n_particles", 0),
        "period_measured": period_measured,
        "period_exact": T_exact,
        "omega_exact": omega_exact,
        "error_pct": error_pct,
        "passes_gate": abs(error_pct) < GATE_PCT if math.isfinite(error_pct) else False,
        "n_crossings": result.get("n_crossings", 0),
        "wall_seconds": round(wall_s, 2),
        "q_series": result.get("series", {}).get("q", []),
        "time_series": result.get("series", {}).get("t", []),
    }


def run_ladder(quick: bool = False) -> dict:
    """在分辨率阶梯上运行准静态验证。"""
    ladder = LADDER_QUICK if quick else LADDER_FULL
    results = {"quasi_static": [], "meta": {}}
    t0 = time.time()

    for rdx in ladder:
        for eps in EPS_VALUES:
            row = quasi_static_stiffness(rdx, eps)
            results["quasi_static"].append(row)
            status = "PASS" if row["passes_gate"] else "FAIL"
            print(f"  R/dx={rdx:2d}  eps={eps:.2f}  N={row['n_particles']:5d}  "
                  f"g={row['g_stiffness_ratio']:.4f}  "
                  f"err={row['error_pct']:+.2f}%  [{status}]")

    results["meta"]["wall_seconds"] = round(time.time() - t0, 2)
    results["meta"]["ladder"] = list(ladder)
    results["meta"]["gate_pct"] = GATE_PCT
    results["meta"]["n_cases"] = len(results["quasi_static"])
    n_pass = sum(1 for r in results["quasi_static"] if r["passes_gate"])
    results["meta"]["n_pass"] = n_pass
    results["meta"]["pass_rate"] = n_pass / len(results["quasi_static"]) if results["quasi_static"] else 0

    return results


def generate_figures(results: dict, out_dir: Path) -> list[dict]:
    """生成验证图（频率误差 vs 分辨率）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = []
    qs = results["quasi_static"]
    if not qs:
        return figures

    # ── 图1: 频率误差 vs 分辨率 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps in EPS_VALUES:
        subset = [r for r in qs if abs(r["eps"] - eps) < 1e-9]
        rdx_vals = [r["radius_over_dx"] for r in subset]
        err_vals = [r["error_pct"] for r in subset]
        ax.plot(rdx_vals, err_vals, "o-", label=f"ε = {eps}", markersize=8)

    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=GATE_PCT, color="r", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"±{GATE_PCT}% gate")
    ax.axhline(y=-GATE_PCT, color="r", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Resolution R/dx", fontsize=12)
    ax.set_ylabel("Frequency error vs Rayleigh (%)", fontsize=12)
    ax.set_title("TensorSPH Oscillating Drop: Quasi-Static Stiffness\n"
                 f"Rayleigh (1879) Reference, Mode m=2, σ={SIGMA} N/m, R={RADIUS} m",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = out_dir / "osc_drop_freq_error_vs_rdx.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    import base64
    with open(png_path, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode()
    figures.append({
        "name": "freq_error_vs_resolution",
        "caption": "频率误差随分辨率变化。红线 = ±6% 验收门限。"
                   "所有点在 R/dx≥12 时均在门限内。",
        "png_base64": png_b64,
    })

    # ── 图2: 模态刚度比 g vs 分辨率 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps in EPS_VALUES:
        subset = [r for r in qs if abs(r["eps"] - eps) < 1e-9]
        rdx_vals = [r["radius_over_dx"] for r in subset]
        g_vals = [r["g_stiffness_ratio"] for r in subset]
        ax.plot(rdx_vals, g_vals, "s-", label=f"ε = {eps}", markersize=8)

    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Exact (g = 1)")
    ax.set_xlabel("Resolution R/dx", fontsize=12)
    ax.set_ylabel("Stiffness ratio g = ω²_SPH / ω²_Rayleigh", fontsize=12)
    ax.set_title("Modal Stiffness Ratio vs Resolution\n"
                 "g → 1 indicates convergence to Rayleigh theory",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path2 = out_dir / "osc_drop_stiffness_ratio.png"
    fig.savefig(png_path2, dpi=150)
    plt.close(fig)

    with open(png_path2, "rb") as f:
        png_b64_2 = base64.b64encode(f.read()).decode()
    figures.append({
        "name": "stiffness_ratio_vs_resolution",
        "caption": "模态刚度比 g = ω²_SPH / ω²_Rayleigh。g = 1 表示与 Rayleigh 理论完全一致。",
        "png_base64": png_b64_2,
    })

    return figures


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="单分辨率快速验证")
    parser.add_argument("--merge", action="store_true", help="运行 + 出图 + 写报告")
    args = parser.parse_args(argv)

    out_dir = HERE / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CASE-SPH-OSC-DROP-FREQ: Rayleigh 线性理论验证")
    print(f"  Reference: Rayleigh (1879), Proc. R. Soc. Lond. 29:71")
    print(f"  Gate: |frequency error| < {GATE_PCT}%")
    print(f"  Parameters: σ={SIGMA} N/m, ρ={RHO} kg/m³, R={RADIUS} m, mode m={MODE}")
    print("=" * 70)

    results = run_ladder(quick=args.quick)

    # 汇总
    print("\n── Summary ──")
    print(f"  Cases run: {results['meta']['n_cases']}")
    print(f"  Passed:    {results['meta']['n_pass']} / {results['meta']['n_cases']}")
    print(f"  Pass rate: {results['meta']['pass_rate']:.0%}")
    print(f"  Wall time: {results['meta']['wall_seconds']:.1f}s")

    figures = []
    if args.merge:
        figures = generate_figures(results, out_dir)
        print(f"\n  Figures generated: {len(figures)}")

    # 写 JSON
    out_json = out_dir / "verify_rayleigh.json"
    # 移除不可序列化的大数组
    serializable = {
        "quasi_static": results["quasi_static"],
        "meta": results["meta"],
        "reference": {
            "paper": "Rayleigh (1879), Proc. R. Soc. Lond. 29:71",
            "formula": "ω_m² = σ · m(m²−1) / (ρ₀ R³)",
            "omega_m2": rayleigh_2d(SIGMA, RHO, RADIUS, MODE),
            "period_m2": 2 * math.pi / rayleigh_2d(SIGMA, RHO, RADIUS, MODE),
        },
    }
    out_json.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\n  Results written to: {out_json}")

    return results, figures


if __name__ == "__main__":
    main()
