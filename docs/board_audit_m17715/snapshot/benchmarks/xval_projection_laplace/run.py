"""XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE — this repo vs the PySPH reference.

Two arms on *identical* geometry, kernel, lattice and pressure gauge, so the
difference measured is the interface/projection discretisation and nothing else:

``ours``
    shipped ``tensorsph.laplace_isph.laplace_isph_case``: Shepard-smoothed
    colour -> first-order-consistent MLS gradient -> quadratic surface-fit
    curvature -> Brackbill delta ``|grad c|`` -> Cummins & Rudman pair
    Laplacian, matrix-free CG, ``p = 0`` on the outer block layer.

``oss``
    the PySPH chain ported in ``tensorsph.oss_projection``: ``SmoothedColor`` ->
    SY11 number-density colour gradient -> SY11/Morris curvature
    (divergence of normals with the ``sum V W`` correction) ->
    CSF force ``sigma kappa grad C / rho`` -> ``PressureCoeffMatrix``
    operator ``8 m_j/(rho_i+rho_j)^2 (XIJ.DWIJ)/(R2IJ+EPS)``, same gauge.

Reference for both arms: the 2-D Young-Laplace law ``dp = sigma/R`` (circle:
principal curvatures ``1/R`` and ``0``).  OSS reference source:
``/tmp/xv2/lib64/python3.11/site-packages/pysph`` (PySPH 1.0b3.dev0), see the
file:line inventory in ``tensorsph.oss_projection``.

Run::

    PYTHONPATH=src python3 benchmarks/xval_projection_laplace/run.py --quick
    PYTHONPATH=src python3 benchmarks/xval_projection_laplace/run.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.laplace_isph import (
    csf_state_two_phase, hex_block, laplace_isph_case, mls_divergence,
    mls_gradient, pressure_poisson,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.oss_projection import (
    oss_color_gradient, oss_csf_acceleration, oss_curvature,
    oss_divergence, oss_pressure_poisson, oss_smoothed_color,
)

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"

RADIUS = 0.02
SIGMA = 0.072
RHO0 = 1000.0
OUTER = 0.012
TOL = 0.15          # owner tolerance, predeclared


def _levels(quick):
    return (0.002, 0.0015) if quick else (0.002, 0.0015, 0.001)


def oss_case(dx, h_factor=1.3, form="paper", delta_variant="sy11",
             morris=True, radius=RADIUS, sigma=SIGMA, rho0=RHO0, outer=OUTER):
    """Static projection equilibrium with the reference (PySPH) operators."""
    h = h_factor * dx
    pos, mass, liquid, half = hex_block(radius, dx, outer, rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    color = liquid.to(pos.dtype)
    rho = torch.full_like(mass, rho0)
    vol = mass / rho0

    c_smooth = oss_smoothed_color(pos, color, vol, pairs, h)
    grad_c, normal, reliable, d_sy11, d_alt = oss_color_gradient(
        pos, c_smooth, vol, pairs, h, form=form)
    kappa, wij = oss_curvature(pos, normal, reliable, vol, pairs, h,
                               morris_correction=morris, form=form)
    delta = d_sy11 if delta_variant == "sy11" else d_alt
    acc = oss_csf_acceleration(kappa, grad_c, rho, sigma)

    # projection: (1/rho) lap p = div a  ->  M p = div a   (reference stencils)
    div_a = oss_divergence(pos, acc, vol, pairs, h)
    edge = 1.6 * dx
    fixed = ((pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge))
    p, n_cg = oss_pressure_poisson(pos, div_a, pairs, mass, rho, h, fixed)

    r = torch.linalg.vector_norm(pos, dim=1)
    active = delta > 0.05 * delta.max()
    core = r < 0.5 * radius
    band = (r > 1.25 * radius) & (r < 1.45 * radius)
    dp = float(p[core].mean() - p[band].mean())

    # same measuring instrument for both arms: MLS gradient of p
    grad_p = mls_gradient(pos, p, vol, pairs, h)
    resid = acc - grad_p / rho0
    a_scale = float(torch.linalg.vector_norm(acc[active], dim=1).mean())
    kappa_w = float((kappa * delta * vol / (2 * math.pi * r.clamp_min(1e-30))).sum()
                    / max(float((delta * vol / (2 * math.pi * r.clamp_min(1e-30))).sum()), 1e-30))

    return {
        "arm": f"oss[{form},{delta_variant},morris={int(morris)},h={h_factor}]",
        "dx_m": dx, "R_over_dx": radius / dx, "h_factor": h_factor,
        "n_particles": int(pos.shape[0]), "n_pairs": int(pairs.shape[0]),
        "kappa_signed_weighted": kappa_w, "kappa_times_R": kappa_w * radius,
        "kappa_analytic": 1.0 / radius,
        "delta_integral_dn": float((delta * vol / (2 * math.pi * r.clamp_min(1e-30))).sum()),
        "dp_measured_Pa": dp, "dp_theory_Pa": sigma / radius,
        "dp_error_pct": (dp / (sigma / radius) - 1.0) * 100.0,
        "residual_median_over_mean_csf_acc": float(
            torch.linalg.vector_norm(resid[active], dim=1).median() / max(a_scale, 1e-30)),
        "cg_iterations": int(n_cg),
    }


def ours_case(dx, h_factor=1.3, curvature_method="mls"):
    rep = laplace_isph_case(radius=RADIUS, dx=dx, sigma=SIGMA, rho0=RHO0,
                            outer=OUTER, h_factor=h_factor,
                            curvature_method=curvature_method)
    rep["arm"] = f"ours[{curvature_method},h={h_factor}]"
    return rep


def make_figure(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    runs = payload["runs"]
    for arm, label, style in (("ours[", "TensorSPH (ours)", "o-"),
                              ("oss[paper", "PySPH-parity (paper psi)", "s--"),
                              ("oss[code", "PySPH-parity (shipped psi)", "^:")):
        xs = [r["R_over_dx"] for r in runs if r["arm"].startswith(arm)]
        ys = [r["dp_error_pct"] for r in runs if r["arm"].startswith(arm)]
        ax[0].plot(xs, ys, style, label=label)
    ax[0].axhspan(-15, 15, color="0.85", zorder=0, label="+-15% gate")
    ax[0].set_xlabel("R/dx"); ax[0].set_ylabel("dp error vs sigma/R  [%]")
    ax[0].set_yscale("symlog", linthresh=10)
    ax[0].set_title("Young-Laplace jump: XVAL ladder")
    ax[0].legend(fontsize=7, loc="lower left")

    for r in runs:
        if r["arm"].startswith(("ours[", "oss[paper")):
            ax[1].plot(r["dx_m"] * 1000, r["dp_measured_Pa"],
                       "o" if r["arm"].startswith("ours") else "s",
                       label=r["arm"] + f" ({r['dp_error_pct']:+.2f}%)")
    ax[1].axhline(SIGMA / RADIUS, color="k", lw=1.0, label=r"$\sigma/R$ analytic")
    ax[1].set_xlabel("dx [mm]"); ax[1].set_ylabel("measured dp [Pa]")
    ax[1].set_title("dp vs resolution")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    levels = _levels(args.quick)

    timing = {}
    runs = []
    for dx in levels:
        t0 = time.time()
        a = ours_case(dx)
        t1 = time.time()
        b = oss_case(dx, form="paper")
        t2 = time.time()
        c = oss_case(dx, form="code")
        t3 = time.time()
        timing[f"{dx}"] = {"ours_s": t1 - t0, "oss_paper_s": t2 - t1,
                           "oss_code_s": t3 - t2}
        runs += [a, b, c]
        print(f"dx={dx} R/dx={RADIUS/dx:.0f}: ours {a['dp_error_pct']:+.2f}% | "
              f"oss(paper) {b['dp_error_pct']:+.2f}% | oss(code) {c['dp_error_pct']:+.3e}%")

    # ---- sensitivity: h/dx and the curvature/quantity choices -------------- #
    sens = []
    for hf in (1.2, 1.3, 1.5):
        a = ours_case(levels[-1], h_factor=hf)
        b = oss_case(levels[-1], h_factor=hf, form="paper")
        sens.append({"vary": "h_factor", "value": hf,
                     "ours_pct": a["dp_error_pct"], "oss_pct": b["dp_error_pct"]})
    for cm in ("fit_quadratic", "mls"):
        a = ours_case(levels[-1], curvature_method=cm)
        sens.append({"vary": "ours_curvature", "value": cm,
                     "ours_pct": a["dp_error_pct"],
                     "kappaR": a["kappaR_delta_weighted"]})
    for morris in (True, False):
        b = oss_case(levels[-1], form="paper", morris=morris)
        sens.append({"vary": "oss_morris_correction", "value": morris,
                     "oss_pct": b["dp_error_pct"], "kappaR": b["kappa_times_R"]})
    for dv in ("sy11", "alt"):
        b = oss_case(levels[-1], form="paper", delta_variant=dv)
        sens.append({"vary": "oss_delta_variant", "value": dv,
                     "oss_pct": b["dp_error_pct"]})
    for form in ("paper", "code"):
        b = oss_case(levels[-1], form=form)
        sens.append({"vary": "oss_number_density_form", "value": form,
                     "oss_pct": b["dp_error_pct"], "kappaR": b["kappa_times_R"]})

    # ---- manufactured Poisson check on the reference operator ------------- #
    h = 1.3 * levels[0]
    pos, mass, liquid, half = hex_block(RADIUS, levels[0], OUTER, RHO0)
    rho = torch.full_like(mass, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    exact = 0.25 * RHO0 * (pos ** 2).sum(dim=1)
    edge = 1.6 * levels[0]
    fixed = ((pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge))
    p_num, _ = oss_pressure_poisson(pos, torch.ones_like(exact), pairs, mass,
                                    rho, h, fixed)
    inner = (torch.linalg.vector_norm(pos, dim=1) < 0.9 * half) & ~fixed
    lap_err = float((p_num[inner] - exact[inner]).abs().max() / exact[inner].abs().max())

    verdict = {
        "tolerance_pct": TOL * 100,
        "ours_finest_pct": runs[-3]["dp_error_pct"],
        "oss_finest_pct": runs[-2]["dp_error_pct"],
        "ours_pass": abs(runs[-3]["dp_error_pct"]) < TOL * 100,
        "oss_pass": abs(runs[-2]["dp_error_pct"]) < TOL * 100,
        "parity": "see tensorsph.oss_projection.pysph_parity_check()",
        "poisson_manufactured_rel_err": lap_err,
    }
    png = make_figure({"runs": runs}, OUT / "xval_projection_laplace.png")
    payload = {"figure": png, "case": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE",
               "reference": "PySPH 1.0b3.dev0 (pypr/pysph), analytic dp=sigma/R",
               "params": {"radius_m": RADIUS, "sigma_N_per_m": SIGMA,
                          "rho0_kg_per_m3": RHO0, "outer_m": OUTER,
                          "levels_m": list(levels)},
               "runs": runs, "sensitivity": sens, "timing": timing,
               "verdict": verdict}
    (OUT / "xval_projection_laplace.json").write_text(
        json.dumps(payload, indent=2, default=float))
    print(json.dumps(verdict, indent=2))
    return payload


if __name__ == "__main__":
    main()
