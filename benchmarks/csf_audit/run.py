"""Error budget of the CSF curvature / delta_s pair on the Young-Laplace case.

This benchmark closes the verification debt left open by
``benchmarks/csf_consistent`` (m2447), which recorded

* curvature ``kappa``        +25 %  (15 % bar)  -> FAIL
* delta-function integral   0.73-0.75 (must be 1) -> FAIL
* cut force ``F_cut/(2 sigma)`` 1.07-1.11 (15 % bar) -> PASS

It answers three questions with numbers, all written to
``error_budget.json`` (figures in ``error_budget.png``):

1. **Is the operator or the interface to blame?**
   Control: the MLS divergence applied to an *exactly linear* field ``x/R``
   must return ``2/R`` to machine precision.  If it does, the +25 % cannot be
   an operator-consistency bug.
2. **Where does the delta_s weight actually live?**
   The kernel-sum colour of a *single-phase* cloud saturates at 1 in the bulk,
   so 20-30 % of ``sum delta_s V`` sits one to three layers inside the
   interface (where ``c`` is already 0.99).  Those particles carry a large
   spurious curvature.  Measured as a radial weight profile.
3. **Does the cut-force test actually certify the force model?**
   ``F_cut/(2 sigma) = kappa_bar * int delta_s dn`` follows from the CSF force
   law, so the cut-force test passes when a +x % curvature error cancels a
   -x % delta_s normalisation error.  The product identity is checked
   explicitly, and the two factors are scored separately.

The same script also scores the fix available through
``cfg.csf_scheme = "consistent_jump"`` (smoothed colour + measured-jump
normalisation) and the dynamic consequence (Laplace pressure jump, spurious
currents) via the relaxation run.

Usage::

    PYTHONPATH=src python benchmarks/csf_audit/run.py             # full
    PYTHONPATH=src python benchmarks/csf_audit/run.py --quick     # static only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch.set_num_threads(2)


def _load_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", ROOT / "benchmarks/laplace_law/run.py")
_csf = _load_sibling("csf_consistent_run", ROOT / "benchmarks/csf_consistent/run.py")
hexagonal_droplet = _law.hexagonal_droplet

from tensorsph.csf_consistent import csf_fields, mls_divergence  # noqa: E402

from tensorsph.droplet_impact import density_from_neighbor_pairs  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

R = 0.02
SIGMA = 7.2
RHO0 = 1000.0
TAU = 2.0 * math.pi
DEFAULT_DX = (0.003, 0.002, 0.0015)
KAPPA_TOL = 0.15
DELTA_TOL = 0.15
CUT_TOL = 0.15

SCHEMES = {
    "consistent": dict(color_smoothing=0, delta_s_mode="half"),
    "consistent_jump": dict(color_smoothing=1, delta_s_mode="jump"),
}


def build(dx: float):
    h = 1.3 * dx
    pos, mass = hexagonal_droplet(R, dx, RHO0)
    mass = mass.to(pos.dtype)
    vol = mass / RHO0
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    return pos, mass, vol, rho, pairs, h


def ds_integral(pos, vol, delta_s, lo=0.6, hi=1.05):
    r = torch.linalg.vector_norm(pos, dim=1)
    m = (r > lo * R) & (r < hi * R)
    return float((delta_s[m] * vol[m] / (TAU * r[m])).sum())


def score(dx: float, scheme: str) -> dict:
    pos, mass, vol, rho, pairs, h = build(dx)
    kw = SCHEMES[scheme]
    st = csf_fields(pos, mass, pairs, h, RHO0, **kw)
    r = torch.linalg.vector_norm(pos, dim=1)
    c = st.color
    active = st.delta_s > 0.05 * float(st.delta_s.max())

    # ---- curvature, three consistent ways of defining "the interface" ------
    kappa_exact = 1.0 / R
    kappa_all = float((st.kappa.abs() * st.delta_s).sum() / st.delta_s.sum())
    band = (c > 0.05) & (c < 0.95)
    kappa_band = (float((st.kappa.abs()[band] * st.delta_s[band]).sum()
                        / st.delta_s[band].sum()) if bool(band.any()) else kappa_all)
    shells = []
    for a, b in ((0.6, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 1.02)):
        sel = (r / R > a) & (r / R <= b)
        if int(sel.sum()) == 0:
            continue
        wsum = float((st.delta_s[sel] * vol[sel]).sum())
        shells.append({
            "r_over_R": [a, b],
            "n_particles": int(sel.sum()),
            "delta_s_weight_fraction": wsum / float((st.delta_s * vol).sum()),
            "color_mean": float(c[sel].mean()),
            "kappa_1_per_m": float((st.kappa.abs()[sel] * st.delta_s[sel]).sum()
                                   / max(float(st.delta_s[sel].sum()), 1e-30)),
        })

    # ---- cut-force identity, plus its two factors --------------------------
    acc = (-SIGMA * st.kappa * st.delta_s / rho)[:, None] * st.normal
    cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
    ratio = cut / (-2.0 * SIGMA)
    int_ds = ds_integral(pos, vol, st.delta_s)
    kappa_bar = float((st.kappa.abs().abs() * st.delta_s).sum() / st.delta_s.sum())
    rho_shell = (float(rho[st.delta_s > 0.05 * float(st.delta_s.max())].mean()) / RHO0)
    product = kappa_bar / kappa_exact * int_ds / rho_shell

    # ---- force applied to bulk particles (colour already saturated) --------
    total_w = float((st.delta_s * vol).sum())
    bulk_share = 1.0 - float((st.delta_s[band] * vol[band]).sum()) / total_w

    mag = torch.linalg.vector_norm(acc, dim=1)
    rhat = pos / r[:, None]
    cos_in = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)
    direction_band = active & (r / R > 0.6) & (r / R <= 1.02)

    return {
        "scheme": scheme,
        "dx_m": dx,
        "h_m": h,
        "R_over_h": R / h,
        "n_particles": int(pos.shape[0]),
        "color": {"bulk": float(c.max()), "surface": float(c.min()),
                  "jump": float(c.max() - c.min())},
        "kappa": {"exact_1_per_m": kappa_exact,
                  "delta_s_weighted_mean_1_per_m": kappa_all,
                  "rel_error_pct": (kappa_all - kappa_exact) / kappa_exact * 100.0,
                  "within_15pct": abs(kappa_all - kappa_exact) / kappa_exact < KAPPA_TOL,
                  "transition_band_mean_1_per_m": kappa_band,
                  "transition_band_rel_error_pct":
                      (kappa_band - kappa_exact) / kappa_exact * 100.0,
                  "transition_band_within_15pct":
                      abs(kappa_band - kappa_exact) / kappa_exact < KAPPA_TOL},
        "delta_s": {"integral_dn": int_ds,
                    "rel_error_pct": (int_ds - 1.0) * 100.0,
                    "within_15pct": abs(int_ds - 1.0) < DELTA_TOL,
                    "bulk_weight_share": bulk_share},
        "cut_force": {"theory_N_per_m": -2.0 * SIGMA,
                      "measured_N_per_m": cut,
                      "ratio_measured_over_theory": ratio,
                      "rel_error_pct": abs(ratio - 1.0) * 100.0,
                      "within_15pct": abs(ratio - 1.0) < CUT_TOL,
                      "predicted_ratio_from_factors": product,
                      "factor_closure_error_pct": (product - ratio) / ratio * 100.0},
        "direction": {"cos_median": float(cos_in[direction_band].median()),
                      "fraction_inward": float((cos_in[direction_band] > 0).float().mean())},
        "shells": shells,
    }


def operator_control(dx: float) -> dict:
    """MLS divergence of an exactly linear field must be exact."""
    pos, _mass, vol, _rho, pairs, h = build(dx)
    field = pos / R                      # div(x/R) = 2/R exactly, every point
    got = mls_divergence(pos, field, vol, pairs, h)
    exact = 2.0 / R
    return {"name": "mls_divergence(linear field x/R)",
            "dx_m": dx,
            "exact_div_1_per_m": exact,
            "mean_1_per_m": float(got.mean()),
            "max_rel_error": float((got - exact).abs().max() / exact),
            "consistent_to_machine_precision": bool(
                float((got - exact).abs().max() / exact) < 1e-6)}


def smoothing_sweep(dx: float) -> list[dict]:
    """kappa / delta_s / F_cut as a function of the colour-smoothing passes."""
    pos, mass, vol, rho, pairs, h = build(dx)
    out = []
    for passes in (0, 1, 2, 4):
        st = csf_fields(pos, mass, pairs, h, RHO0, color_smoothing=passes,
                        delta_s_mode="jump")
        acc = (-SIGMA * st.kappa * st.delta_s / rho)[:, None] * st.normal
        cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
        kbar = float((st.kappa.abs() * st.delta_s).sum() / st.delta_s.sum())
        out.append({"color_smoothing_passes": passes,
                    "kappa_rel_error_pct": (kbar - 1.0 / R) * R * 100.0,
                    "delta_s_integral_dn": ds_integral(pos, vol, st.delta_s),
                    "cut_force_ratio": cut / (-2.0 * SIGMA)})
    return out


def relaxation(dx: float, scheme: str, steps: int) -> dict:
    return _csf.relax(radius=R, dx=dx, sigma=SIGMA, rho0=RHO0, sound_speed=20.0,
                      steps=steps, viscosity=0.01, scheme=scheme, output_every=100)


def make_figures(report: dict, out_dir: Path = HERE) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["static"]
    dxs = sorted({r["dx_m"] for r in rows})
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.3))

    # (a) curvature error with both schemes and both band definitions
    ax = axes[0]
    for scheme, style in (("consistent", "s--"), ("consistent_jump", "o-")):
        sel = [r for r in rows if r["scheme"] == scheme]
        sel.sort(key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sel],
                [abs(r["kappa"]["rel_error_pct"]) for r in sel], style,
                label=f"{scheme}: whole shell")
        ax.plot([r["dx_m"] for r in sel],
                [abs(r["kappa"]["transition_band_rel_error_pct"]) for r in sel],
                style, alpha=0.45,
                label=f"{scheme}: transition band")
    ax.axhline(15.0, color="r", ls="--", label="15 % acceptance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$|\kappa - 1/R|\,/\,(1/R)$  [%]")
    ax.set_title(r"(a) curvature error vs $\kappa=1/R$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (b) delta_s normalisation and the bulk weight share
    ax = axes[1]
    for scheme, style in (("consistent", "s--"), ("consistent_jump", "o-")):
        sel = [r for r in rows if r["scheme"] == scheme]
        sel.sort(key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sel], [r["delta_s"]["integral_dn"] for r in sel],
                style, label=f"{scheme}: $\\int\\delta_s\\,dn$")
    ax.axhline(1.0, color="k", ls=":")
    ax.axhspan(0.85, 1.15, color="g", alpha=0.12, label="15 % band")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\int \delta_s\, dn$")
    ax.set_title("(b) surface delta function normalisation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) cut force identity
    ax = axes[2]
    for scheme, style in (("consistent", "s--"), ("consistent_jump", "o-")):
        sel = [r for r in rows if r["scheme"] == scheme]
        sel.sort(key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sel],
                [r["cut_force"]["ratio_measured_over_theory"] for r in sel], style,
                label=scheme)
        ax.plot([r["dx_m"] for r in sel],
                [r["cut_force"]["predicted_ratio_from_factors"] for r in sel],
                style, alpha=0.45, marker="x", ls=":",
                label=f"{scheme}: $\\bar\\kappa\\int\\delta_s$")
    ax.axhspan(0.85, 1.15, color="g", alpha=0.12, label="15 % band")
    ax.axhline(1.0, color="k", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_title(r"(c) $F_{cut}/(2\sigma)$ and its factorisation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (d) radial weight profile of delta_s
    ax = axes[3]
    width = 0.38
    for k, (scheme, colour) in enumerate((("consistent", "C3"), ("consistent_jump", "C0"))):
        sel = [r for r in rows if r["scheme"] == scheme and r["dx_m"] == dxs[-1]]
        if not sel:
            continue
        shells = sel[0]["shells"]
        xs = [0.5 * (s["r_over_R"][0] + s["r_over_R"][1]) for s in shells]
        ys = [s["delta_s_weight_fraction"] * 100.0 for s in shells]
        ax.bar([x + (k - 0.5) * width for x in xs], ys, width=width, color=colour,
               label=scheme)
    ax.set_xlabel("r / R")
    ax.set_ylabel(r"share of $\sum \delta_s V$  [%]")
    ax.set_title(f"(d) where the interface weight lives (dx = {dxs[-1] * 1e3:.1f} mm)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = out_dir / "error_budget.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="static scoring only")
    ap.add_argument("--dx", type=float, nargs="*", default=list(DEFAULT_DX))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    report = {
        "case": {"radius_m": R, "sigma_N_per_m": SIGMA, "rho0_kg_per_m3": RHO0,
                 "analytic": {"kappa_1_per_m": 1.0 / R,
                              "delta_p_Pa": SIGMA / R,
                              "cut_force_N_per_m": -2.0 * SIGMA},
                 "tolerances_pct": {"kappa": KAPPA_TOL * 100,
                                    "delta_s": DELTA_TOL * 100,
                                    "cut_force": CUT_TOL * 100}},
        "operator_control": [operator_control(dx) for dx in (0.002,)],
        "static": [],
        "smoothing_sweep": [],
        "relaxation": [],
    }
    for dx in args.dx:
        for scheme in SCHEMES:
            row = score(dx, scheme)
            report["static"].append(row)
            print(f"dx={dx * 1e3:5.2f} mm {scheme:16s} "
                  f"kappa={row['kappa']['rel_error_pct']:+7.1f}% "
                  f"(band {row['kappa']['transition_band_rel_error_pct']:+7.1f}%) "
                  f"int_ds={row['delta_s']['integral_dn']:.3f} "
                  f"F_cut={row['cut_force']['ratio_measured_over_theory']:.3f} "
                  f"bulk_w={row['delta_s']['bulk_weight_share'] * 100:.0f}%")
        report["smoothing_sweep"].append({"dx_m": dx, "rows": smoothing_sweep(dx)})
        if not args.quick:
            for scheme in SCHEMES:
                rep = relaxation(dx, scheme, args.steps)
                report["relaxation"].append({
                    "dx_m": dx, "scheme": scheme,
                    "delta_p_measured_Pa": rep["laplace"]["delta_p_measured_Pa"],
                    "delta_p_rel_error_pct": rep["laplace"]["rel_error_pct"],
                    "vmax_over_vref": rep["spurious_current"]["vmax_over_vref_final"],
                    "steps": rep["setup"]["steps"],
                    "t_over_t_sigma": rep["setup"]["t_over_t_sigma"],
                    "rho_core_over_rho0": rep["density"]["rho_core_over_rho0"],
                })
                print(f"  relax {scheme:16s} dx={dx * 1e3:.2f} mm "
                      f"dp={rep['laplace']['delta_p_measured_Pa']:7.2f} Pa "
                      f"({rep['laplace']['rel_error_pct']:+6.1f}%) "
                      f"max|v|/v_ref={rep['spurious_current']['vmax_over_vref_final']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig = make_figures(report, args.out)
    report["figure"] = fig.name
    (args.out / "error_budget.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out / 'error_budget.json'} and {fig}")


if __name__ == "__main__":
    main()
