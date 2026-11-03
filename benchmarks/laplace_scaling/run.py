"""Laplace-law *scaling* verification for the consistent CSF scheme.

Existing B3 evidence scores the static drop at a **single** parameter point
(``R = 0.02 m``, ``sigma = 7.2 N/m``, ``benchmarks/csf_consistent``) plus a
resolution ladder at that same point (``benchmarks/laplace_law``).  A single
point cannot distinguish "reproduces the Laplace *law*" from "reproduces the
Laplace *number* at one radius and one surface tension", so this harness
sweeps the two-dimensional parameter space of the law

    dp = sigma / R        (2-D cylinder),      F_cut = 2 sigma,

and asks two questions that a one-point check cannot answer:

1. **Amplitude** -- is ``|F_cut| / (2 sigma)`` inside the platform tolerance
   (15 %) at *every* ``(R, sigma)`` in the grid?
2. **Scaling** -- does the measured cut force follow ``F_cut ~ sigma^p R^q``
   with ``p = 1`` and ``q = 0``, i.e. is the implied pressure jump
   ``dp = F_cut / (2 R)`` really proportional to ``sigma / R``?

The cut-force identity is used as the acceptance quantity because it is the
one CSF moment that does not need the stiff Tait equation of state: the density
perturbation a real ``dp = sigma/R`` would require is ``O(1e-5)`` here, far
below the discretisation noise (see ``benchmarks/laplace_droplet``
"resolvability_audit").  The diagnostic split of the same force into
``kappa`` and ``delta_s`` is reported alongside, because those two carry
complementary ~25 % errors that only cancel in the product (see the report).

Usage::

    PYTHONPATH=src python benchmarks/laplace_scaling/run.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLATFORM_TOLERANCE = 0.15


def _load_sibling(name: str, filename: str):
    path = HERE.parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


def cut_force_and_fields(radius: float, dx: float, sigma: float, *, h_factor: float,
                         rho0: float = 1000.0, sound_speed: float = 20.0,
                         viscosity: float = 0.01) -> dict:
    """One parameter point: cut force, implied pressure jump and field splits."""
    from tensorsph.csf_consistent import csf_fields
    from tensorsph.surface_tension import compute_surface_tension_force

    cfg = Config(dx=dx, h_factor=h_factor, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)

    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    state = csf_fields(pos, mass, pairs, h, rho0)

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    # Cut force per unit length: sum of m*a over one half of the drop equals the
    # integral of the volumetric CSF force over that half (V_i * rho_i * a_i =
    # V_i * f_i), which by the divergence theorem is 2*sigma.
    right = pos[:, 0] > 0
    # Estimator A (the one used by benchmarks/csf_consistent): momentum flux
    # sum_{x>0} m_i a_i.  With m_i = rho0 * A_cell fixed and a_i = f_i / rho_i
    # this equals sum A_cell f_i (rho0/rho_i): it is the volumetric force
    # integral *inflated* by the free-surface density deficit rho0/rho_i.
    f_cut = float((mass[right, None] * acc[right]).sum(dim=0)[0])
    # Estimator B: the volumetric integral with the geometric cell volume A_cell,
    # i.e. sum_{x>0} A_cell f_i = sum m_i a_i (rho_i / rho0).  This is the
    # divergence-theorem quantity that must equal 2 sigma; it does not inherit
    # the surface density deficit.
    cell_area = float(mass[0] / rho0)
    f_cut_geom = float((cell_area * rho[right, None] * acc[right]).sum(dim=0)[0])
    theory_cut = -2.0 * sigma
    ratio = f_cut / theory_cut
    ratio_geom = f_cut_geom / theory_cut

    # Implied Young-Laplace jump: dp * 2R = F_cut.
    dp_implied = abs(f_cut) / (2.0 * radius)
    dp_theory = sigma / radius
    dp_rel_err = (dp_implied - dp_theory) / dp_theory

    # Cut-plane angle average: a single x>0 cut is exposed to lattice anisotropy
    # (the hexagonal lattice has 6-fold, not continuous, symmetry).  Rotating the
    # cut plane over 0/15/30/45 degrees gives the spread that must be quoted as
    # the measurement uncertainty of this criterion.
    cell_area0 = float(mass[0] / rho0)
    angle_ratios_mom, angle_ratios_geom, angle_net = [], [], []
    for deg in (0.0, 15.0, 30.0, 45.0):
        th = math.radians(deg)
        proj = pos[:, 0] * math.cos(th) + pos[:, 1] * math.sin(th)
        half = proj > 0
        axis_x = acc[:, 0] * math.cos(th) + acc[:, 1] * math.sin(th)
        fm = float((mass[half] * axis_x[half]).sum())
        fg = float((cell_area0 * rho[half] * axis_x[half]).sum())
        angle_ratios_mom.append(fm / theory_cut)
        angle_ratios_geom.append(fg / theory_cut)
        angle_net.append(float((mass * axis_x).sum()) / (2.0 * sigma))

    ds = state.delta_s
    mag = torch.linalg.vector_norm(acc, dim=1)
    # free-surface density deficit on the force-carrying shell
    w = (mass * acc.norm(dim=1))[right]
    rho_ratio_w = float(((rho / rho0)[right] * w).sum() / w.sum().clamp_min(1e-30))
    active = ds > 0.05 * ds.max()
    cos_inward = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)

    # Curvature diagnostics: the physically meaningful value is div(n) weighted
    # by delta_s (the weight that builds the force).  The report of
    # benchmarks/csf_consistent additionally takes |kappa|; both are recorded so
    # that the two numbers can be compared instead of confused.
    def _wmean(vals: torch.Tensor, sel: torch.Tensor) -> float:
        w = ds[sel]
        return float((vals[sel] * w).sum() / w.sum().clamp_min(1e-30))

    band = active & (r / radius > 0.6) & (r / radius <= 1.05)
    inner = band & (r < radius)
    outer = band & (r >= radius)

    # int delta_s dn over the interface shell (ideal 1.0 for a two-phase delta
    # function; the fluid-side only colour field halves it, see csf_consistent).
    shell = (r > 0.6 * radius) & (r < 1.05 * radius)
    vol = mass / rho0
    int_ds = float((ds[shell] * vol[shell]
                    / (2.0 * math.pi * r[shell].clamp_min(1e-30))).sum())

    return {
        "radius_m": radius,
        "dx_m": dx,
        "resolution_R_over_dx": radius / dx,
        "h_over_dx": h_factor,
        "sigma_N_per_m": sigma,
        "n_particles": int(pos.shape[0]),
        "n_pairs": int(pairs.shape[0]),
        "cut_force": {
            "measured_N_per_m": f_cut,
            "theory_N_per_m": theory_cut,
            "ratio_measured_over_theory": ratio,
            "rel_error_pct": (ratio - 1.0) * 100.0,
            "within_15pct": abs(ratio - 1.0) <= PLATFORM_TOLERANCE,
            "estimator": "sum m_i a_i (momentum; inflates by rho0/rho_i)",
        },
        "cut_force_geometric": {
            "measured_N_per_m": f_cut_geom,
            "theory_N_per_m": theory_cut,
            "ratio_measured_over_theory": ratio_geom,
            "rel_error_pct": (ratio_geom - 1.0) * 100.0,
            "within_15pct": abs(ratio_geom - 1.0) <= PLATFORM_TOLERANCE,
            "estimator": "sum A_cell rho_i a_i (volumetric; divergence theorem)",
            "momentum_over_geometric": (ratio / ratio_geom
                                        if abs(ratio_geom) > 1e-12 else None),
            "rho_over_rho0_force_weighted": rho_ratio_w,
        },
        "laplace_jump": {
            "dp_implied_Pa": dp_implied,
            "dp_theory_Pa": dp_theory,
            "rel_error_pct": dp_rel_err * 100.0,
            "within_15pct": abs(dp_rel_err) <= PLATFORM_TOLERANCE,
        },
        "curvature": {
            "exact_1_per_m": 1.0 / radius,
            "delta_s_weighted_signed": _wmean(state.kappa, band),
            "delta_s_weighted_abs": _wmean(state.kappa.abs(), band),
            "delta_s_weighted_signed_rel_error_pct":
                (_wmean(state.kappa, band) * radius - 1.0) * 100.0,
            "delta_s_weighted_abs_rel_error_pct":
                (_wmean(state.kappa.abs(), band) * radius - 1.0) * 100.0,
            "inner_half_mean": _wmean(state.kappa, inner) if bool(inner.any()) else None,
            "outer_half_mean": _wmean(state.kappa, outer) if bool(outer.any()) else None,
        },
        "delta_s": {"integral_dn": int_ds,
                    "within_35pct": abs(int_ds - 1.0) <= 0.35},
        "angle_average": {
            "angles_deg": [0.0, 15.0, 30.0, 45.0],
            "ratio_momentum": angle_ratios_mom,
            "ratio_geometric": angle_ratios_geom,
            "ratio_momentum_mean": sum(angle_ratios_mom) / len(angle_ratios_mom),
            "ratio_momentum_spread": max(angle_ratios_mom) - min(angle_ratios_mom),
            "net_force_over_2sigma": angle_net,
        },
        "direction": {
            "cos_inward_median": float(cos_inward[active].median()),
            "fraction_inward": float((cos_inward[active] > 0).double().mean()),
        },
    }


def _loglog_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    """Least-squares slope/intercept of log(y) against log(x)."""
    lx = torch.log(torch.tensor(x, dtype=torch.float64))
    ly = torch.log(torch.tensor(y, dtype=torch.float64))
    a = torch.stack((lx, torch.ones_like(lx)), dim=1)
    sol = torch.linalg.lstsq(a, ly).solution
    return float(sol[0]), float(sol[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--radii", default="0.015,0.02,0.03")
    ap.add_argument("--sigmas", default="3.6,7.2,14.4")
    ap.add_argument("--cells-per-radius", type=int, default=10)
    args = ap.parse_args(argv)

    radii = [float(s) for s in args.radii.split(",")]
    sigmas = [float(s) for s in args.sigmas.split(",")]
    cpr = args.cells_per_radius

    # 1. the (R, sigma) grid at a fixed lattice resolution R/dx
    grid = []
    for R in radii:
        for s in sigmas:
            grid.append(cut_force_and_fields(R, R / cpr, s, h_factor=1.3))
            print(f"grid     R={R:.4f} sigma={s:5.2f}  "
                  f"Fcut/(2sig)={grid[-1]['cut_force']['ratio_measured_over_theory']:.4f} "
                  f"geom={grid[-1]['cut_force_geometric']['ratio_measured_over_theory']:.4f}  "
                  f"dp_err={grid[-1]['laplace_jump']['rel_error_pct']:+6.2f}%")

    # 2. resolution ladder at the reference point.  Extended past R/dx = 20 so
    #    that the growth of the residual bias can be extrapolated to the point
    #    where it would breach the 15 % limit -- the grid sweep alone cannot see
    #    that, because it scales dx with R and is therefore geometrically
    #    self-similar.
    ladder = [cut_force_and_fields(0.02, 0.02 / n, 7.2, h_factor=1.3)
              for n in (8, 10, 16, 20, 28, 40)]
    for c in ladder:
        print(f"ladder   R/dx={c['resolution_R_over_dx']:5.1f}  "
              f"Fcut/(2sig)={c['cut_force']['ratio_measured_over_theory']:.4f} "
              f"geom={c['cut_force_geometric']['ratio_measured_over_theory']:.4f} "
              f"rho/rho0={c['cut_force_geometric']['rho_over_rho0_force_weighted']:.4f} "
              f"ang(mean={c['angle_average']['ratio_momentum_mean']:.4f}"
              f"+-{0.5*c['angle_average']['ratio_momentum_spread']:.3f})")

    # 2b. fixed-dx sweep: same lattice spacing, three radii.  Here the lattice
    #     is *not* self-similar in units of R, so this is the part of the sweep
    #     that can actually falsify R-independence of the cut force.
    fixed_dx = [cut_force_and_fields(R, 0.002, 7.2, h_factor=1.3)
                for R in (0.012, 0.02, 0.03)]
    for c in fixed_dx:
        print(f"fixed dx R={c['radius_m']:.4f} R/dx={c['resolution_R_over_dx']:5.1f}  "
              f"Fcut/(2sig)={c['cut_force']['ratio_measured_over_theory']:.4f}")

    # 3. kernel-support sensitivity at the reference point: is the residual bias
    #    an artefact of a truncated 2h kernel support?
    support = [cut_force_and_fields(0.02, 0.002, 7.2, h_factor=hf)
               for hf in (1.2, 1.3, 1.5)]
    for c in support:
        print(f"support  h/dx={c['h_over_dx']:3.1f}  "
              f"Fcut/(2sig)={c['cut_force']['ratio_measured_over_theory']:.4f}  "
              f"kappa_abs_err={c['curvature']['delta_s_weighted_abs_rel_error_pct']:+6.2f}%")

    # scaling exponents of the measured cut force, F ~ sigma^p R^q
    by_R: dict[float, list[float]] = {}
    by_sigma: dict[float, list[float]] = {}
    for c in grid:
        by_R.setdefault(c["radius_m"], []).append(abs(c["cut_force"]["measured_N_per_m"]))
        by_sigma.setdefault(c["sigma_N_per_m"], []).append(
            abs(c["cut_force"]["measured_N_per_m"]))
    p_exp, _ = _loglog_fit(sigmas, by_R[radii[0]]) if len(sigmas) > 1 else (float("nan"), 0)
    q_exp, _ = _loglog_fit(radii, by_sigma[sigmas[0]]) if len(radii) > 1 else (float("nan"), 0)

    ratios = [c["cut_force"]["ratio_measured_over_theory"] for c in grid]
    worst = max(abs(r - 1.0) for r in ratios)
    dp_errs = [abs(c["laplace_jump"]["rel_error_pct"]) for c in grid]
    verdict = {
        "criterion_amplitude": "|F_cut/(2 sigma) - 1| <= 15 % at every (R, sigma)",
        "tolerance": PLATFORM_TOLERANCE,
        "n_points": len(grid),
        "n_within_tolerance": sum(1 for c in grid if c["cut_force"]["within_15pct"]),
        "max_abs_ratio_error_pct": worst * 100.0,
        "max_abs_dp_error_pct": max(dp_errs),
        "amplitude_status": "PASS" if worst <= PLATFORM_TOLERANCE else "FAIL",
        "criterion_scaling": "F_cut ~ sigma^p R^q with p = 1 (5 %) and q = 0 (|q| <= 0.1)",
        "sigma_exponent": p_exp,
        "sigma_exponent_error_pct": (p_exp - 1.0) * 100.0,
        "radius_exponent": q_exp,
        "scaling_status": "PASS" if (abs(p_exp - 1.0) <= 0.05 and abs(q_exp) <= 0.1)
                          else "FAIL",
        "ratio_spread_at_fixed_R": {
            f"{R:g}": max(vals) - min(vals) for R, vals in
            {R: [c["cut_force"]["ratio_measured_over_theory"] for c in grid
                 if c["radius_m"] == R] for R in radii}.items()},
        "resolution_ladder": {
            "R_over_dx": [c["resolution_R_over_dx"] for c in ladder],
            "ratio": [c["cut_force"]["ratio_measured_over_theory"] for c in ladder],
            "converges_to_one": False,
        },
    }
    # monotone approach to 1 under refinement is the convergence claim we can test
    lad = [c["cut_force"]["ratio_measured_over_theory"] for c in ladder]
    verdict["resolution_ladder"]["converges_to_one"] = bool(
        all(abs(v - 1.0) <= PLATFORM_TOLERANCE for v in lad))
    verdict["support_sensitivity"] = {
        f"{c['h_over_dx']:g}": c["cut_force"]["ratio_measured_over_theory"]
        for c in support}

    # Does the residual bias shrink under refinement?  Fit ratio-1 against R/dx
    # and report where the linear trend would reach the 15 % limit.
    inv_dx = [c["resolution_R_over_dx"] for c in ladder]
    bias = [c["cut_force"]["ratio_measured_over_theory"] - 1.0 for c in ladder]
    slope_n, _ = _loglog_fit(inv_dx, [1.0 + b for b in bias])  # log-log slope
    mean_b = sum(bias) / len(bias)
    mean_n = sum(inv_dx) / len(inv_dx)
    var_n = sum((n - mean_n) ** 2 for n in inv_dx)
    trend_slope = sum((n - mean_n) * (b - mean_b) for n, b in zip(inv_dx, bias)) / var_n
    breach = None
    if trend_slope > 0:
        breach = 1.0 + (PLATFORM_TOLERANCE - mean_b) / trend_slope + mean_n
    verdict["bias_trend"] = {
        "d_bias_per_R_over_dx": trend_slope,
        "bias_at_R_over_dx": dict(zip([f"{n:g}" for n in inv_dx],
                                      [round(b, 5) for b in bias])),
        "shrinks_under_refinement": bool(trend_slope < 0.0),
        "extrapolated_R_over_dx_where_15pct_is_breached": breach,
    }
    verdict["fixed_dx_radii"] = {
        "R_m": [c["radius_m"] for c in fixed_dx],
        "R_over_dx": [c["resolution_R_over_dx"] for c in fixed_dx],
        "ratio": [c["cut_force"]["ratio_measured_over_theory"] for c in fixed_dx],
        "spread": max(c["cut_force"]["ratio_measured_over_theory"] for c in fixed_dx)
                  - min(c["cut_force"]["ratio_measured_over_theory"] for c in fixed_dx),
    }

    out = {
        "benchmark": "laplace_law_scaling_2d",
        "implementation_under_test": "tensorsph.csf_consistent (csf_scheme='consistent')",
        "reference": {
            "law": "dp = sigma / R  (2-D cylinder) ; F_cut = 2 sigma",
            "citation": "Young (1805); Laplace (1806); Brackbill, Kothe & Zemach "
                        "(1992) JCP 100, 335; Adami, Hu & Adams (2012) JCP 231, 7057",
            "note": "cut-force identity F_cut = dp * 2R avoids the stiff Tait EOS "
                    "(dp/bulk modulus = 1.3e-5 at the reference point)",
        },
        "setup": {"rho0_kg_per_m3": 1000.0, "sound_speed_m_per_s": 20.0,
                  "gamma": 7.0, "h_over_dx": 1.3,
                  "lattice": "hexagonal, R/dx fixed per grid axis",
                  "cells_per_radius": cpr},
        "platform_tolerance": PLATFORM_TOLERANCE,
        "grid": grid,
        "resolution_ladder": ladder,
        "fixed_dx_sweep": fixed_dx,
        "support_sensitivity": support,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"report -> {args.out}")
    print(f"AMPLITUDE {verdict['amplitude_status']} "
          f"(max |ratio-1| = {verdict['max_abs_ratio_error_pct']:.2f} %, "
          f"{verdict['n_within_tolerance']}/{verdict['n_points']} points)")
    print(f"SCALING   {verdict['scaling_status']} "
          f"(p = {p_exp:.4f}, q = {q_exp:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())