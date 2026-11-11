"""CASE-LAPLACE-SPH: ``Delta p = sigma / R`` on an incompressible (projection) SPH drop.

What this harness scores
------------------------
The analytic reference is the 2-D Young-Laplace law

    Delta p = sigma / R        (circle: the two principal curvatures are 1/R and 0)

read off the *pressure field* of a projection-SPH droplet, i.e. the pressure that
cancels the CSF traction, ``grad p = rho a_CSF``.  Nothing here parses a
compressibility: the jump is carried by the Poisson source, which is why this
route can reach the platform's CASE-LAPLACE-SPH 5 % gate while the weakly
compressible (Tait) route cannot -- see the module docstring of
``tensorsph.laplace_isph`` and the resolvability audit below.

Gates (all thresholds fixed before the run, ``platform_tolerance = 0.05``):

    laplace_error_finest    |dp/dp_theory - 1| <= 5 %   on the finest level
    laplace_error_all       every level <= 5 %
    gauge_self_test         the same Poisson solve with exact kappa and an exact
                            unit-normalisation delta must return 1 within 3 %
                            (instrument error bound)
    cross_check_flux        the independent radial-integral route must agree
                            with the Poisson route within 3 %
    bulk_balance            median momentum residual away from the interface
                            shell <= 5 % of the mean CSF acceleration
    negative_control        sigma = 0 must give a jump below 1e-10 Pa
    spurious_currents       measured, reported, and NOT claimed: the time
                            integration is known to diverge (honest FAIL)

Run::

    PYTHONPATH=src python benchmarks/laplace_sph/run.py            # full
    PYTHONPATH=src python benchmarks/laplace_sph/run.py --quick    # 2 levels
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.laplace_isph import laplace_isph_case, relax_projection

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

RADIUS = 0.02          # m
SIGMA = 0.072          # N/m, water/air at 20 C
RHO0 = 1000.0          # kg/m^3
OUTER = 0.012          # m, thickness of the vapour carrier around the drop
H_FACTOR = 1.3         # h/dx (module default)
TOLERANCE = 0.05       # CASE-LAPLACE-SPH gate


def _levels(quick: bool):
    return (0.002, 0.0015) if quick else (0.002, 0.0015, 0.001)


def ladder(levels, curvature_method="mls"):
    rows = []
    for dx in levels:
        rep = laplace_isph_case(radius=RADIUS, dx=dx, sigma=SIGMA, rho0=RHO0,
                                outer=OUTER, h_factor=H_FACTOR,
                                curvature_method=curvature_method)
        rows.append(rep)
    return rows


def variants(levels, curvature_method="mls"):
    out = []
    for variant in ("model", "kappa_exact", "delta_unit", "gauge"):
        rows = []
        for dx in levels:
            rows.append(laplace_isph_case(radius=RADIUS, dx=dx, sigma=SIGMA,
                                          rho0=RHO0, outer=OUTER,
                                          h_factor=H_FACTOR,
                                          curvature_method=curvature_method,
                                          variant=variant))
        out.append({"variant": variant, "rows": rows})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    levels = _levels(a.quick)
    dp_theory = SIGMA / RADIUS
    t0 = time.time()

    print("=" * 78)
    print("CASE-LAPLACE-SPH  dp = sigma / R")
    print(f"  sigma = {SIGMA} N/m   R = {RADIUS} m   rho0 = {RHO0} kg/m^3")
    print(f"  theory: dp = {dp_theory:.6f} Pa   tolerance {TOLERANCE:.0%}")
    print("=" * 78)

    # ---------------------------------------------------------------- ladder
    mls = ladder(levels, "mls")
    for r in mls:
        print(f"  [mls]  dx={r['dx_m']*1e3:5.2f}mm R/dx={r['R_over_dx']:5.1f} "
              f"N={r['n_particles']:5d} dp={r['dp_measured_Pa']:8.4f} "
              f"ratio={r['dp_over_theory']:.4f} err={r['dp_error_pct']:+6.2f}% "
              f"kappa_err={r['kappa_error_pct']:+6.2f}%")
    fit = ladder(levels, "fit_quadratic")

    # -------------------------------------------------------------- variants
    var = variants(levels, "mls")

    # ------------------------------------------------- instrument sensitivity
    bands = [(1.15, 1.35), (1.25, 1.45), (1.35, 1.55)]
    band_rows = [laplace_isph_case(radius=RADIUS, dx=levels[-1], sigma=SIGMA,
                                   rho0=RHO0, outer=OUTER, h_factor=H_FACTOR,
                                   curvature_method="mls", out_band=b)
                 for b in bands]
    hf_rows = [laplace_isph_case(radius=RADIUS, dx=levels[-1], sigma=SIGMA,
                                 rho0=RHO0, outer=OUTER, h_factor=hf,
                                 curvature_method="mls")
               for hf in (1.2, 1.3, 1.4)]
    stencil_rows = [laplace_isph_case(radius=RADIUS, dx=levels[-1], sigma=SIGMA,
                                      rho0=RHO0, outer=OUTER,
                                      h_factor=H_FACTOR,
                                      curvature_method="fit_quadratic",
                                      curvature_stencil=s)
                    for s in (2.0, 3.0, 4.0, 5.0)]

    # ------------------------------------------------------- negative control
    neg = laplace_isph_case(radius=RADIUS, dx=levels[-1], sigma=0.0, rho0=RHO0,
                            outer=OUTER, h_factor=H_FACTOR,
                            curvature_method="mls")

    # ---------------------------------------------------- optional dynamics
    dyn = None
    if not a.no_dynamics:
        dyn = relax_projection(radius=RADIUS, dx=levels[0], sigma=SIGMA,
                              rho0=RHO0, outer=OUTER, h_factor=H_FACTOR,
                              curvature_method="mls", steps=40,
                              sample_every=5)
        print(f"  [dynamics] diverged={dyn['diverged']} "
              f"steps_to_divergence={dyn.get('steps_to_divergence')}")

    # ---------------------------------------------------------------- verdict
    finest = mls[-1]
    finest_fit = fit[-1]
    all_within = all(abs(r["dp_over_theory"] - 1.0) <= TOLERANCE for r in mls)
    gauge = [v for v in var if v["variant"] == "gauge"][0]["rows"][-1]
    gauge_err = abs(gauge["dp_over_theory"] - 1.0)
    flux_err = abs(finest["dp_flux_over_theory"] - 1.0)
    bulk = finest["residual_bulk_median_over_mean_csf_acc"]
    neg_err = abs(neg["dp_measured_Pa"])

    gates = {
        "laplace_error_finest": {
            "measured": abs(finest["dp_over_theory"] - 1.0),
            "limit": TOLERANCE,
            "pass": abs(finest["dp_over_theory"] - 1.0) <= TOLERANCE,
        },
        "laplace_error_all_levels": {
            "measured": max(abs(r["dp_over_theory"] - 1.0) for r in mls),
            "limit": TOLERANCE, "pass": all_within,
        },
        "gauge_self_test": {"measured": gauge_err, "limit": 0.03,
                            "pass": gauge_err <= 0.03},
        "cross_check_flux": {"measured": flux_err, "limit": 0.03,
                             "pass": flux_err <= 0.03},
        "bulk_balance": {"measured": bulk, "limit": 0.05, "pass": bulk <= 0.05},
        "negative_control_sigma0": {"measured_Pa": neg_err, "limit_Pa": 1e-10,
                                    "pass": bool(neg_err <= 1e-10)},
        "spurious_currents_adami_0p1": {
            "measured": dyn["umax_over_vref_final"] if dyn else None,
            "limit": 0.1,
            "pass": bool(dyn and not dyn["diverged"]
                         and dyn["umax_over_vref_final"] <= 0.1),
            "note": "projection time integration is unstable; reported as honest FAIL",
        },
    }
    obs_order = None
    if len(mls) >= 2:
        e0 = abs(mls[-2]["dp_over_theory"] - 1.0)
        e1 = abs(mls[-1]["dp_over_theory"] - 1.0)
        if e1 > 0 and e0 > 0:
            obs_order = math.log(e0 / e1) / math.log(mls[-2]["dx_m"] / mls[-1]["dx_m"])

    report = {
        "case": "CASE-LAPLACE-SPH",
        "reference": {
            "law": "Young-Laplace, 2-D circular interface: dp = sigma / R",
            "dp_theory_Pa": dp_theory,
            "source": "Brackbill, Kothe & Zemach (1992) JCP 100:335; "
                      "Cummins & Rudman (1999) JCP 152:584 (projection SPH)",
        },
        "setup": {
            "radius_m": RADIUS, "sigma_N_per_m": SIGMA, "rho0": RHO0,
            "outer_carrier_m": OUTER, "h_factor": H_FACTOR,
            "curvature_default": "mls", "tolerance": TOLERANCE,
            "dimensionless": {
                "R_over_dx": [r["R_over_dx"] for r in mls],
                "R_over_h": [r["R_over_h"] for r in mls],
                "Bond_number": 0.0,
                "von_Weber_or_capillary": "not applicable (no inertia in statics)",
                "weber": 0.0,
                "oh_comment": "no viscosity enters the static balance",
                "dp_over_bulk_modulus_WCSPH_c0_20":
                    dp_theory / (RHO0 * 20.0 ** 2 / 7.0),
            },
        },
        "resolvability_audit": {
            "statement": "a Tait EOS with c0 = 20 m/s needs a density ratio of "
                         "dp/B = 1.26e-5 to carry this jump, while the WCSPH "
                         "density field carries ~1e-3 noise: the jump is not "
                         "representable by the EOS route at any resolution.",
            "dp_over_bulk_modulus": dp_theory / (RHO0 * 20.0 ** 2 / 7.0),
            "reference_wcsph_relaxed_error_pct":
                -78.55799550367372,
            "reference_closure_best_error_pct": -20.1,
            "reference_artifacts": [
                "benchmarks/csf_consistent/validation_report.json",
                "docs/verification_laplace_pressure_closure.md",
            ],
        },
        "ladder_mls": mls,
        "ladder_fit_quadratic": fit,
        "variants": var,
        "sensitivity": {
            "out_band": [{"band": list(b), "dp_over_theory": r["dp_over_theory"]}
                         for b, r in zip(bands, band_rows)],
            "h_factor": [{"h_factor": hf, "dp_over_theory": r["dp_over_theory"],
                          "kappa_error_pct": r["kappa_error_pct"]}
                         for hf, r in zip((1.2, 1.3, 1.4), hf_rows)],
            "fit_stencil": [{"stencil_h": s, "dp_over_theory": r["dp_over_theory"],
                             "kappa_error_pct": r["kappa_error_pct"]}
                            for s, r in zip((2.0, 3.0, 4.0, 5.0), stencil_rows)],
        },
        "negative_control": neg,
        "dynamics": dyn,
        "observed_order_finest_pair": obs_order,
        "gates": gates,
        "verdict": {
            "laplace_gate_met": bool(gates["laplace_error_finest"]["pass"]
                                     and all_within),
            "fit_quadratic_also_passes": bool(
                abs(finest_fit["dp_over_theory"] - 1.0) <= TOLERANCE),
            "spurious_current_gate_met": gates["spurious_currents_adami_0p1"]["pass"],
        },
        "wall_clock_s": None,
    }
    report["wall_clock_s"] = round(time.time() - t0, 2)
    (out / "results.json").write_text(json.dumps(report, indent=2))

    print("-" * 78)
    for name, g in gates.items():
        print(f"  {name:32s} {'PASS' if g['pass'] else 'FAIL':4s} "
              f"measured={g.get('measured', g.get('measured_Pa'))}")
    print(f"  observed order (last pair, mls) = {obs_order}")
    print(f"wrote {out / 'results.json'}  ({report['wall_clock_s']} s)")

    if a.figure:
        make_figure(report, out / "laplace_sph_isph.png")


def make_figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from tensorsph.laplace_isph import csf_state_two_phase, hex_block, pressure_poisson
    from tensorsph.csf_consistent import mls_divergence

    h = H_FACTOR * 0.0015
    pos, mass, liquid, half = hex_block(RADIUS, 0.0015, OUTER, RHO0)
    st = csf_state_two_phase(pos, mass, liquid, h, RHO0, SIGMA,
                             curvature_method="mls")
    vol = mass / RHO0
    source = -RHO0 * mls_divergence(pos, st["acc"], vol, st["pairs"], h)
    edge = 1.6 * 0.0015
    fixed = (pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge)
    p, _ = pressure_poisson(pos, source, st["pairs"], vol, h, fixed)
    r = torch.linalg.vector_norm(pos, dim=1).numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    sc = ax[0].scatter(pos[:, 0] * 1e3, pos[:, 1] * 1e3, c=p.numpy(),
                       s=6, cmap="coolwarm")
    circ = plt.Circle((0, 0), RADIUS * 1e3, fill=False, ls="--", lw=1.0,
                      color="k")
    ax[0].add_patch(circ)
    fig.colorbar(sc, ax=ax[0], label="p [Pa]")
    ax[0].set_title("projection-SPH pressure field\n"
                    f"$\\Delta p$ = {report['ladder_mls'][-1]['dp_measured_Pa']:.4f} Pa "
                    f"vs $\\sigma/R$ = {report['reference']['dp_theory_Pa']:.4f} Pa")
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")
    ax[0].axis("equal")

    rs = np.linspace(0.2, 1.6, 60)
    prof = []
    for rr in rs:
        m = np.abs(r - rr * RADIUS) < 0.04 * RADIUS
        prof.append(float(p[m].mean()) if m.any() else np.nan)
    ax[1].plot(rs, prof, "o-", ms=3)
    ax[1].axhline(report["reference"]["dp_theory_Pa"], color="k", ls="--",
                  label="$\\sigma/R$")
    ax[1].axvline(1.0, color="0.6", lw=0.8)
    ax[1].set_xlabel("r / R")
    ax[1].set_ylabel("mean p [Pa]")
    ax[1].legend()
    ax[1].set_title("radial pressure profile")

    for key, label, mk in (("ladder_mls", "MLS $\\nabla\\cdot n$", "o"),
                           ("ladder_fit_quadratic", "fit_quadratic", "s")):
        rows = report[key]
        ax[2].plot([row["R_over_dx"] for row in rows],
                   [row["dp_error_pct"] for row in rows], mk + "-", label=label)
    ax[2].axhline(5, color="g", ls=":", label="5 % gate")
    ax[2].axhline(-5, color="g", ls=":")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("R/dx")
    ax[2].set_ylabel("$\\Delta p$ error [%]")
    ax[2].set_title("resolution ladder")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
