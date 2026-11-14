"""XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE, 3-D density-ratio arm.

The 2-D arm closed the high-contrast convergence defect with the rho-blind
force-density Poisson (``docs/xval_projection_laplace_density_ratio.md``, round
4).  The 3-D module (``tensorsph.laplace_isph3d``) had no density ratio at all:
every published 3-D number was measured on a 1:1 carrier, so the one term the
ratio multiplies -- the ``1/rho`` inside the projection operator -- was switched
off by construction.  This script turns it on in 3-D.

Gates (declared here, before the sweep, and stored in the JSON):

``G1`` owner gate   ``|dp error| < 15 %`` for **every** measured configuration
``G2`` regression    equal-density ``force_density`` run is bit-identical to the
                     historical uniform-density path (``|d| < 1e-9`` relative)
``G3`` blindness     spread of ``dp`` over the ratio ladder at fixed ``R/dx``
                     (force_density arm) ``< 3 %`` of theory
``G4`` resolution    ``|dp error| < 5 %`` at every ``R/dx`` for the force_density
                     arm at the highest contrast
``G5`` ablation      the naive constant-coefficient arm fed by a ratio-aware
                     source **must** exceed 15 % at the highest contrast, i.e.
                     the ratio must be doing real work

Usage
    python scripts/xval_projection_laplace_3d/density_ratio.py [--quick]

``--quick`` restricts the sweep to the coarse level (used as a smoke check);
the stored artifacts come from the full run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph3d import laplace_isph3d_case  # noqa: E402

# The repository's benchmarks are vectorised torch kernels: on a loaded control
# node the default 128-thread pool thrashes and a 1 s run takes 40 s.  One
# thread is both faster here and keeps the numbers reproducible.
torch.set_num_threads(int(os.environ.get("TENSORSPH_THREADS", "1")))

RADIUS = 0.02
SIGMA = 0.072
RHO0 = 1000.0
OUTER = 0.012
H_FACTOR = 1.3
CURVATURE = "mls"          # the validated 3-D default (2-D round-1 audit)
STENCIL = 2.0

RATIOS = (1.0, 10.0, 100.0, 1000.0)
RATIO_LEVEL = 0.0025       # R/dx = 8
LEVELS = (0.004, 0.0025, 0.0016)     # R/dx = 5, 8, 12.5
ARMS = ("force_density", "density", "uniform")
HIGH_RATIO = 1000.0

GATES = {
    "owner_gate_pct": 15.0,
    "regression_rel_tol": 1e-9,
    "ratio_spread_pct": 3.0,
    "resolution_band_pct": 5.0,
    "flux_crosscheck_pct": 10.0,
}

OUT = Path(__file__).resolve().parent / "results"


def _run(dx, ratio, weighting, **overrides):
    kwargs = {"radius": RADIUS, "sigma": SIGMA, "rho0": RHO0, "outer": OUTER,
              "h_factor": H_FACTOR, "dx": dx,
              "curvature_method": CURVATURE, "curvature_stencil": STENCIL,
              "rho_vapour": RHO0 / ratio, "poisson_weighting": weighting}
    kwargs.update(overrides)
    t0 = time.time()
    rep = laplace_isph3d_case(**kwargs)
    rep["wall_s"] = round(time.time() - t0, 2)
    rep["requested_ratio"] = ratio
    # at 1:1 every arm takes the historical uniform path, so the *effective*
    # label is "uniform" for all of them; the requested arm is what the row means
    rep["requested_weighting"] = weighting
    rep.pop("fields", None)
    return rep


def _keep(rep):
    """The columns the artifact stores (everything else stays in the library)."""
    keys = ("dx_m", "R_over_dx", "n_particles", "n_liquid", "n_active",
            "curvature_method", "curvature_stencil", "requested_ratio",
            "density_ratio", "rho_liquid", "rho_vapour", "poisson_weighting",
            "requested_weighting", "kappaR_delta_weighted",
            "delta_s_integral_over_4pir2",
            "dp_measured_Pa", "dp_theory_Pa", "dp_error_pct",
            "dp_flux_Pa", "dp_flux_over_theory",
            "residual_median_over_mean_csf_acc", "cg_iterations", "wall_s")
    return {k: rep[k] for k in keys}


def ratio_ladder():
    """G3/G5: ratio 1 ... 1000 at fixed R/dx, three operator arms."""
    rows = []
    for weighting in ARMS:
        for ratio in RATIOS:
            rep = _run(RATIO_LEVEL, ratio, weighting)
            rows.append(_keep(rep))
            print(f"[ratio] {weighting:14s} ratio={ratio:7.1f} "
                  f"dp_err={rep['dp_error_pct']:+9.2f}% "
                  f"flux={rep['dp_flux_over_theory']:.4f} "
                  f"{rep['wall_s']}s", flush=True)
    return rows


def resolution_ladder():
    """G4: refinement at the highest contrast, plus the equal-density control."""
    rows = []
    for dx in LEVELS:
        for weighting in ("force_density", "density"):
            # the variable-coefficient arm is the ablation: measure it at the two
            # affordable levels only (the finest 3-D level is 64k particles)
            if weighting == "density" and dx == LEVELS[-1]:
                continue
            for ratio in (1.0, HIGH_RATIO):
                rep = _run(dx, ratio, weighting)
                rows.append(_keep(rep))
                print(f"[res] dx={dx:.4f} {weighting:14s} ratio={ratio:7.1f} "
                      f"dp_err={rep['dp_error_pct']:+7.2f}% "
                      f"{rep['wall_s']}s", flush=True)
    return rows


def controls():
    """G2 (1:1 regression) and the sigma = 0 negative control."""
    out = {}
    hist = _run(RATIO_LEVEL, 1.0, "uniform")
    fd = _run(RATIO_LEVEL, 1.0, "force_density")
    den = _run(RATIO_LEVEL, 1.0, "density")
    rel = lambda a, b: abs(a["dp_measured_Pa"] / b["dp_measured_Pa"] - 1.0)  # noqa: E731
    out["equal_density"] = {
        "historical_uniform_pa": hist["dp_measured_Pa"],
        "force_density_pa": fd["dp_measured_Pa"],
        "density_pa": den["dp_measured_Pa"],
        "rel_diff_force_density": rel(fd, hist),
        "rel_diff_density": rel(den, hist),
        "gate_rel_tol": GATES["regression_rel_tol"],
    }
    print(f"[control] 1:1 regression rel diff fd={out['equal_density']['rel_diff_force_density']:.3e} "
          f"density={out['equal_density']['rel_diff_density']:.3e}", flush=True)
    zero = _run(RATIO_LEVEL, HIGH_RATIO, "force_density", sigma=0.0)
    out["sigma_zero"] = {"dp_pa": zero["dp_measured_Pa"],
                         "dp_flux_pa": zero["dp_flux_Pa"],
                         "density_ratio": zero["density_ratio"]}
    print(f"[control] sigma=0 ratio 1000 dp={zero['dp_measured_Pa']:.3e} Pa", flush=True)
    return out, zero


def profile_figure(ax_a, ax_b, ax_c, ratio_rows, res_rows, field):
    """Panel (a) ratio ladder, (b) resolution ladder, (c) pressure profile."""
    mark = {"force_density": ("o", "C0"), "density": ("s", "C1"),
            "uniform": ("^", "C3")}
    for arm in ARMS:
        pts = [r for r in ratio_rows if r["poisson_weighting"] == arm]
        ax_a.semilogx([r["requested_ratio"] for r in pts],
                      [r["dp_error_pct"] for r in pts],
                      marker=mark[arm][0], color=mark[arm][1], label=arm)
    ax_a.axhspan(-GATES["owner_gate_pct"], GATES["owner_gate_pct"], color="0.85",
                 zorder=0, label="owner gate 15%")
    ax_a.axhspan(-GATES["ratio_spread_pct"], GATES["ratio_spread_pct"],
                 color="0.93", zorder=0)
    ax_a.set_xlabel(r"$\rho_l/\rho_v$")
    ax_a.set_ylabel(r"$\Delta p$ error [%]")
    ax_a.set_title(f"(a) 3-D ratio ladder, $R/\\Delta x$ = {RADIUS / RATIO_LEVEL:.0f}")
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    for arm in ("force_density", "density"):
        for ratio, style in ((1.0, "--"), (HIGH_RATIO, "-")):
            pts = [r for r in res_rows
                   if r["poisson_weighting"] == arm and r["requested_ratio"] == ratio]
            pts.sort(key=lambda r: r["R_over_dx"])
            ax_b.plot([r["R_over_dx"] for r in pts],
                      [r["dp_error_pct"] for r in pts],
                      style, marker=mark[arm][0], color=mark[arm][1],
                      label=f"{arm}, ratio {ratio:.0f}")
    ax_b.axhline(GATES["resolution_band_pct"], color="0.4", ls=":", lw=1)
    ax_b.axhline(-GATES["resolution_band_pct"], color="0.4", ls=":", lw=1)
    ax_b.set_xlabel(r"$R/\Delta x$")
    ax_b.set_ylabel(r"$\Delta p$ error [%]")
    ax_b.set_title(r"(b) refinement at 1000:1 ($\pm5\%$ dotted)")
    ax_b.legend(fontsize=7)
    ax_b.grid(alpha=0.3)

    pos, p, kappa, liquid = field
    x = pos[:, 0]
    sel = (x.abs() < RADIUS * 3.0) & (pos[:, 1].abs() < 0.5 * (RADIUS + OUTER) / 8)
    order = torch.argsort(x[sel])
    ax_c.plot(x[sel][order] / RADIUS, p[sel][order], ".", ms=1.5, color="C0")
    ax_c.axhline(2.0 * SIGMA / RADIUS, color="C3", ls="--", lw=1,
                 label=r"$2\sigma/R$")
    ax_c.axvline(1.0, color="0.5", lw=0.8)
    ax_c.axvline(-1.0, color="0.5", lw=0.8)
    ax_c.set_xlabel(r"$x/R$")
    ax_c.set_ylabel("p [Pa]")
    ax_c.set_title("(c) pressure through the droplet, 1000:1 force-density")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="coarse level only (smoke check)")
    ap.add_argument("--phase", choices=("all", "ratio", "res", "assemble"),
                    default="all",
                    help="split the sweep when the control node has no free slot: "
                         "run 'ratio' and 'res' separately, then 'assemble'")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    if args.quick:
        rows = ratio_ladder()
        payload = {"mode": "quick", "ratio_ladder": rows, "gates": GATES}
        (OUT / "density_ratio_quick.json").write_text(json.dumps(payload, indent=1))
        print(json.dumps({"quick_rows": len(rows)}, indent=1))
        return 0

    if args.phase == "ratio":
        rows = ratio_ladder()
        (OUT / "_ratio.json").write_text(json.dumps({"ratio_ladder": rows}, indent=1))
        print(json.dumps({"ratio_rows": len(rows)}, indent=1))
        return 0
    if args.phase == "res":
        rows = resolution_ladder()
        (OUT / "_res.json").write_text(json.dumps({"resolution_ladder": rows}, indent=1))
        print(json.dumps({"res_rows": len(rows)}, indent=1))
        return 0
    if args.phase == "assemble":
        ratio_rows = json.loads((OUT / "_ratio.json").read_text())["ratio_ladder"]
        res_part = json.loads((OUT / "_res.json").read_text())
        res_rows = res_part["resolution_ladder"]
        ctrl, zero = controls()
    else:
        ratio_rows = ratio_ladder()
        res_rows = resolution_ladder()
        ctrl, zero = controls()

    # field for panel (c): the highest-contrast force-density run
    field_rep = laplace_isph3d_case(radius=RADIUS, sigma=SIGMA, rho0=RHO0,
                                    outer=OUTER, h_factor=H_FACTOR,
                                    dx=RATIO_LEVEL, curvature_method=CURVATURE,
                                    curvature_stencil=STENCIL,
                                    rho_vapour=RHO0 / HIGH_RATIO,
                                    poisson_weighting="force_density",
                                    return_fields=True)
    f = field_rep["fields"]
    ratio_rows = [{**r, "field_of_record": True} if (r["poisson_weighting"] == "force_density"
                                                     and r["requested_ratio"] == HIGH_RATIO)
                  else r for r in ratio_rows]

    # ---------------------------------------------------------------- gates
    fd_ratio = [r for r in ratio_rows if r["poisson_weighting"] == "force_density"]
    naive = [r for r in ratio_rows if r["poisson_weighting"] == "uniform"]
    spread = (max(r["dp_measured_Pa"] for r in fd_ratio)
              - min(r["dp_measured_Pa"] for r in fd_ratio))
    spread_pct = 100.0 * spread / fd_ratio[0]["dp_theory_Pa"]
    fd_res = [r for r in res_rows
              if r["poisson_weighting"] == "force_density"
              and r["requested_ratio"] == HIGH_RATIO]
    den_res = [r for r in res_rows
               if r["poisson_weighting"] == "density"
               and r["requested_ratio"] == HIGH_RATIO]
    # G1 covers the two *candidate* operators; the naive constant-coefficient arm
    # is a declared ablation and is required to fail (G5), so it is excluded.
    all_errs = [abs(r["dp_error_pct"]) for r in ratio_rows
                if r["poisson_weighting"] != "uniform"] + \
               [abs(r["dp_error_pct"]) for r in res_rows]
    verdict = {
        "G1_owner_gate_15pct_all_configs": {
            "scope": "force_density + density arms (the naive ablation must fail, G5)",
            "max_abs_err_pct": max(all_errs),
            "pass": max(all_errs) < GATES["owner_gate_pct"]},
        "G2_equal_density_bit_identical": {
            "rel_diff_force_density": ctrl["equal_density"]["rel_diff_force_density"],
            "pass": ctrl["equal_density"]["rel_diff_force_density"]
                    < GATES["regression_rel_tol"]},
        "G3_ratio_spread_pct": {"value": spread_pct,
                                "pass": spread_pct < GATES["ratio_spread_pct"]},
        "G4_resolution_band_5pct": {
            "by_level": {f"R/dx={r['R_over_dx']:.1f}": r["dp_error_pct"]
                         for r in sorted(fd_res, key=lambda r: r["R_over_dx"])},
            "pass": all(abs(r["dp_error_pct"]) < GATES["resolution_band_pct"]
                        for r in fd_res)},
        "G5_naive_arm_must_fail": {
            "naive_err_pct": {f"ratio={r['requested_ratio']:.0f}": r["dp_error_pct"]
                              for r in naive},
            "pass": any(abs(r["dp_error_pct"]) > GATES["owner_gate_pct"]
                        for r in naive)},
        "flux_crosscheck": {
            "max_abs_dev_pct": max(abs(r["dp_flux_over_theory"] - 1.0) * 100.0
                                   for r in ratio_rows + res_rows),
            "pass": max(abs(r["dp_flux_over_theory"] - 1.0) * 100.0
                        for r in ratio_rows + res_rows)
                    < GATES["flux_crosscheck_pct"]},
        "sigma_zero_control": {"dp_pa": ctrl["sigma_zero"]["dp_pa"],
                               "pass": abs(ctrl["sigma_zero"]["dp_pa"]) < 1e-9},
    }
    verdict["all_pass"] = all(v["pass"] for v in verdict.values() if isinstance(v, dict))

    payload = {
        "case": "3-D projection-SPH Young-Laplace jump at a real density ratio",
        "anchor": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE",
        "params": {"radius_m": RADIUS, "sigma_N_per_m": SIGMA,
                   "rho0_kg_per_m3": RHO0, "outer_m": OUTER,
                   "h_factor": H_FACTOR, "curvature_method": CURVATURE,
                   "curvature_stencil": STENCIL,
                   "ratio_level_dx_m": RATIO_LEVEL,
                   "levels_dx_m": list(LEVELS), "ratios": list(RATIOS),
                   "dp_theory_pa": 2.0 * SIGMA / RADIUS},
        "gates": GATES,
        "ratio_ladder": ratio_rows,
        "resolution_ladder": res_rows,
        "controls": ctrl,
        "derived": {"ratio_spread_pct": spread_pct,
                    "density_arm_refined_err_pct":
                        {f"R/dx={r['R_over_dx']:.1f}": r["dp_error_pct"]
                         for r in sorted(den_res, key=lambda r: r["R_over_dx"])}},
        "verdict": verdict,
        "wall_s": round(time.time() - t_all, 1),
    }

    fig_path = OUT / "density_ratio.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        profile_figure(axes[0], axes[1], axes[2], ratio_rows, res_rows,
                       (f["pos"], f["p"], f["kappa"], None))
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        payload["figure"] = str(fig_path.relative_to(ROOT))
    except Exception as exc:  # pragma: no cover - figure is best effort
        payload["figure_error"] = repr(exc)

    (OUT / "density_ratio.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())