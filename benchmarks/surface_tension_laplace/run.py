"""Young-Laplace verification of the CSF surface-tension model (verification debt).

Why this benchmark exists
-------------------------
``benchmarks/static_droplet`` evaluated the surface-tension force once, checked
that it was non-zero and *roughly* inward, and declared PASS.  It never measured
the quantity the model is supposed to deliver, the **Laplace pressure jump**.
This script closes that gap: it measures the pressure jump implied by the force
field and compares it with the analytic value for a static 2-D droplet.

Theory
------
For a 2-D circular droplet of radius ``R`` (one principal curvature, unit
depth) the Young-Laplace relation is

.. math:: \\Delta p = \\sigma / R

In a static state the radial momentum balance of the solver is

.. math:: \\frac{dp}{dr} = \\rho f_r  \\quad\\Longrightarrow\\quad
          \\Delta p = p(0) - p(R) = -\\int_0^{R} \\rho\\, f_r\\, dr

with ``f_r < 0`` (inward) for a convex interface.  ``f_r`` is the surface-tension
acceleration returned by ``compute_surface_tension_force``.  The integral is the
*static momentum-balance* estimator of the pressure jump: it is the pressure the
model would support if the WCSPH pressure field were noiseless.  It needs one
force evaluation (no time integration), so it isolates the surface-tension model
from WCSPH density noise -- which matters, because with ``c0 = 20 m/s`` the
density excess that carries ``sigma/R = 1.44 Pa`` is ``~4e-6`` relative, i.e.
buried under summation-density error.

What is measured, per resolution
--------------------------------
``dp_implied``  pressure jump from the radial quadrature above,
``err_rel``     ``(dp_implied - sigma/R) / (sigma/R)``,
``inward_frac`` fraction of interface particles whose force points inward
                (dot product of force direction with ``-r_hat`` above 0.5),
``kappa_band``  mean curvature of the interface band (should be ``1/R``),
``ds_max``      max ``|grad c|`` (the CSF delta function; must scale like 1/h
                and integrate to one across the interface),
``a_max``       max acceleration magnitude.

A model that passes verification must show ``err_rel`` small *and* not growing
when ``dx`` is halved.  Non-convergence is itself a failure, whatever the error
at a single resolution.

Usage
-----
    python benchmarks/surface_tension_laplace/run.py            # table + JSON
    python benchmarks/surface_tension_laplace/run.py --plot     # + PNG figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_gradient,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

SIGMA = 0.072      # water-air surface tension [N/m]
RADIUS = 0.05      # droplet radius [m]
RHO0 = 1000.0

# Resolutions used by the study: halving dx at every step.
DX_LIST = (0.005, 0.0025, 0.00125)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def make_droplet(radius: float = RADIUS, dx: float = 0.0025, rho0: float = RHO0):
    """Square-lattice disc centred on the origin.

    The lattice spacing is exactly ``dx`` so that ``mass = rho0 dx^2`` is the
    consistent 2-D particle mass and the position of every particle relative to
    the analytic circle is identical at every resolution.
    """
    n = int(2.0 * radius / dx) + 4
    axis = (torch.arange(n, dtype=torch.float64) - 0.5 * n) * dx
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = torch.linalg.vector_norm(pos, dim=1)
    pos = pos[r <= radius]
    mass = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=torch.float64)
    return pos, mass


# --------------------------------------------------------------------------- #
# Static momentum-balance estimator
# --------------------------------------------------------------------------- #

def implied_pressure_jump(pos, acc, rho, radius: float, dx: float):
    """``-\\int_0^R rho f_r dr`` evaluated on the particle set.

    Bin-free form: a ring of ``n`` particles at radius ``r`` represents an
    annulus of width ``dr = n dx^2 / (2 pi r)``, so

        dp = -sum_i rho_i f_r,i dx^2 / (2 pi r_i)

    which for a complete ring reduces to ``-mean(rho f_r) dx`` and needs no
    arbitrary bin width.  ``profile`` (binned at ``dx``) is returned for the
    shape plot and as a quadrature-sensitivity check.
    """
    r = torch.linalg.vector_norm(pos, dim=1)
    r_safe = r.clamp_min(1e-30)
    f_r = (acc * (pos / r_safe[:, None])).sum(dim=1)
    rho_f_r = rho * f_r

    # The ring measure dx^2/(2 pi r) diverges at r -> 0, where the lattice has
    # a single particle and the radial force vanishes anyway; floor r at half a
    # spacing so the innermost particle carries the weight of its own cell.
    weight = dx * dx / (2.0 * math.pi * r.clamp_min(0.5 * dx))
    dp = -float((rho_f_r * weight).sum())

    n_bins = int(math.ceil((radius + 2.0 * dx) / dx))
    edges = torch.arange(n_bins + 1, dtype=torch.float64) * dx
    profile = []
    dp_bin = 0.0
    for k in range(n_bins):
        sel = (r >= edges[k]) & (r < edges[k + 1])
        count = int(sel.sum())
        if count == 0:
            continue
        value = float(rho_f_r[sel].mean())
        r_mid = float(0.5 * (edges[k] + edges[k + 1]))
        dp_bin -= value * count * dx * dx / (2.0 * math.pi * r_mid)
        profile.append((r_mid / radius, value))
    return dp, dp_bin, profile


def positive_control(dx: float, radius: float = RADIUS, sigma: float = SIGMA,
                     width_factor: float = 1.0, inset_spacings: float = 3.0) -> dict:
    """Recover ``sigma/R`` from a *smooth, analytically normalised* force field.

    The field is ``rho f_r(r) = -(sigma/R) G(r)`` with ``G`` a Gaussian of
    width ``w = width_factor*dx`` centred at ``R - inset_spacings*dx`` and
    normalised so that ``int_0^inf G dr = 1`` **analytically**, i.e. the target
    integral is exact before discretisation.  The centre is inset so that the
    delta lies inside the fluid domain: the estimator integrates over the
    particles that exist, and a delta centred exactly on ``R`` would lose its
    outer half (there is no fluid outside a single-phase droplet).  The particle estimator then has to reproduce it
    from the same lattice and mass used by the model runs.  The residual is the
    estimator's own error, so any larger error seen for the shipped model is
    attributable to the model rather than to the measurement.
    """
    pos, mass = make_droplet(radius, dx)
    rho = torch.full((pos.shape[0],), RHO0, dtype=torch.float64)
    r = torch.linalg.vector_norm(pos, dim=1)

    w = width_factor * dx
    centre = radius - inset_spacings * dx
    g = torch.exp(-((r - centre) ** 2) / (2.0 * w * w))
    # analytic 1-D normalisation of a Gaussian truncated at r=0
    erf_term = 1.0 + math.erf(centre / (math.sqrt(2.0) * w))
    g = g * (2.0 / (w * math.sqrt(2.0 * math.pi) * erf_term))

    a_r = -(sigma / radius) * g / rho            # rho * a_r = -(sigma/R) g
    acc = a_r[:, None] * (pos / r.clamp_min(1e-30)[:, None])
    dp, dp_bin, _ = implied_pressure_jump(pos, acc, rho, radius, dx)
    return {"width_m": w, "centre_m": centre, "dp_recovered_Pa": dp, "dp_bin_Pa": dp_bin,
            "dp_theory_Pa": sigma / radius,
            "err_rel": (dp - sigma / radius) / (sigma / radius),
            "err_rel_binned": (dp_bin - sigma / radius) / (sigma / radius)}


def direction_fidelity(pos, acc):
    """Fraction of active particles whose force points inward (dot > 0.5)."""
    mag = torch.linalg.vector_norm(acc, dim=1)
    active = mag > 0.0
    if not bool(active.any()):
        return 0.0, 0.0
    r_hat = pos[active] / torch.linalg.vector_norm(pos[active], dim=1, keepdim=True)
    acc_hat = acc[active] / mag[active][:, None]
    dot = -(acc_hat * r_hat).sum(dim=1)     # +1 = perfectly inward
    return float((dot > 0.5).to(torch.float64).mean()), float(dot.mean())


# --------------------------------------------------------------------------- #
# Instrumented replica of the shipped heuristic pipeline
# --------------------------------------------------------------------------- #

def heuristic_diagnostics(pos, mass, density, cfg: Config):
    """Re-derive the shipped CSF pipeline and expose the intermediate fields.

    The replica must reproduce ``compute_surface_tension_force`` exactly; the
    caller asserts that, so the reported ``kappa`` / ``delta_s`` / ``color``
    describe the shipped model and not a look-alike.
    """
    h = cfg.h_factor * cfg.dx
    n = pos.shape[0]
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)

    count = torch.zeros(n, dtype=pos.dtype)
    ones = torch.ones_like(i, dtype=pos.dtype)
    count.index_add_(0, i, ones)
    count.index_add_(0, j, ones)
    avg = count.mean()
    is_surface = count < 0.7 * avg
    color = torch.ones(n, dtype=pos.dtype)
    color[is_surface] = count[is_surface] / avg

    # gradient of the colour field (same symmetric form as the shipped code)
    delta = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(delta, h)
    dc = color[j] - color[i]
    vol_j = mass[j] / density[j]
    vol_i = mass[i] / density[i]
    grad_c = torch.zeros_like(pos)
    # NOTE: the shipped compute_color_gradient uses vol_j on *both* entries of
    # the pair; the replica keeps that asymmetry on purpose so that it
    # reproduces the shipped force bit for bit (asserted by the caller).
    grad_c.index_add_(0, i, vol_j[:, None] * dc[:, None] * grad_w)
    grad_c.index_add_(0, j, -vol_j[:, None] * dc[:, None] * grad_w)

    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    normal = torch.zeros_like(pos)
    mask = mag.squeeze(1) > 1e-8
    normal[mask] = -grad_c[mask] / mag[mask]        # outward (shipped convention)

    dn = normal[j] - normal[i]
    div_n = (dn * grad_w).sum(dim=1)
    kappa = torch.zeros(n, dtype=pos.dtype)
    kappa.index_add_(0, i, vol_j * div_n)
    kappa.index_add_(0, j, vol_i * div_n)

    delta_s = mag.squeeze(1)
    acc = (-cfg.surface_tension_coefficient * kappa * delta_s)[:, None] * normal / density[:, None]
    return acc, kappa, delta_s, color, is_surface


# --------------------------------------------------------------------------- #
# Diagnostic variant: textbook single-phase colour-function CSF
# --------------------------------------------------------------------------- #

def color_band_force(pos, mass, density, cfg: Config):
    """Reference *diagnostic* variant (not part of the library).

    Colour field ``c_i = sum_j V_j W_ij`` (Akinci et al. 2013), ``n = -grad c``,
    ``kappa = div n``, ``delta_s = |grad c|``, ``f = -sigma kappa delta_s n``.
    It is included to show whether the failure of the shipped model is a local
    bug or a property of any single-phase CSF on a one-layer interface band.
    """
    h = cfg.h_factor * cfg.dx
    n = pos.shape[0]
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    vol = mass / density
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)

    color = torch.zeros(n, dtype=pos.dtype)
    color.index_add_(0, i, vol[j] * w)
    color.index_add_(0, j, vol[i] * w)
    color += vol * cubic_spline_kernel(torch.zeros(n, dtype=pos.dtype), h)

    grad_w = cubic_spline_gradient(pos[i] - pos[j], h)
    dc = color[j] - color[i]
    grad_c = torch.zeros_like(pos)
    grad_c.index_add_(0, i, vol[j][:, None] * dc[:, None] * grad_w)
    grad_c.index_add_(0, j, -vol[i][:, None] * dc[:, None] * grad_w)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    nrm = torch.zeros_like(pos)
    mask = mag.squeeze(1) > 1e-10
    nrm[mask] = -grad_c[mask] / mag[mask]

    dn = nrm[j] - nrm[i]
    div_n = (dn * grad_w).sum(dim=1)
    kappa = torch.zeros(n, dtype=pos.dtype)
    kappa.index_add_(0, i, vol[j] * div_n)
    kappa.index_add_(0, j, vol[i] * div_n)
    acc = (-cfg.surface_tension_coefficient * kappa * mag.squeeze(1))[:, None] * nrm / density[:, None]
    return acc, kappa, mag.squeeze(1), color


# --------------------------------------------------------------------------- #
# Study driver
# --------------------------------------------------------------------------- #

def run_resolution(dx: float, radius: float = RADIUS, sigma: float = SIGMA,
                   diagnostics: bool = True) -> dict:
    cfg = Config(dx=dx, rho0=RHO0, surface_tension_coefficient=sigma,
                 gravity=0.0, viscosity=0.0)
    pos, mass = make_droplet(radius, dx)
    h = cfg.h_factor * dx
    density = density_from_neighbor_pairs(pos, mass, h)
    pairs = cell_linked_pairs(pos, 2.0 * h)

    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    dp, dp_bin, profile = implied_pressure_jump(pos, acc, density, radius, dx)
    inward_frac, inward_mean = direction_fidelity(pos, acc)
    theory = sigma / radius

    row = {
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "n_active": int((torch.linalg.vector_norm(acc, dim=1) > 0.0).sum()),
        "a_max_m_per_s2": float(torch.linalg.vector_norm(acc, dim=1).max()),
        "a_scale_sigma_over_rho_R": sigma / (RHO0 * radius),
        "dp_implied_Pa": dp,
        "dp_implied_binned_Pa": dp_bin,
        "quadrature_spread_rel": (dp_bin - dp) / dp,
        "dp_theory_Pa": theory,
        "err_rel": (dp - theory) / theory,
        "sigma_eff_over_sigma": dp / theory,
        "inward_frac_dot_gt_0p5": inward_frac,
        "inward_dot_mean": inward_mean,
        "rho_min_over_rho0": float(density.min() / cfg.rho0),
        "profile": profile,
    }

    if diagnostics:
        acc2, kappa, delta_s, color, is_surface = heuristic_diagnostics(pos, mass, density, cfg)
        if not torch.allclose(acc, acc2, atol=1e-12, rtol=1e-10):
            raise RuntimeError("instrumented replica does not reproduce the shipped force")
        band = (kappa.abs() > 0.0) & (delta_s > 0.0)
        row["kappa_mean_band"] = float(kappa[band].mean()) if bool(band.any()) else 0.0
        row["kappa_absmedian_band"] = float(kappa[band].abs().median()) if bool(band.any()) else 0.0
        row["kappa_theory_1_over_R"] = 1.0 / radius
        row["delta_s_max"] = float(delta_s.max())
        row["delta_s_times_h_max"] = float((delta_s.max() * h))
        row["color_min"] = float(color.min())
        row["color_max"] = float(color.max())
        row["n_surface_flag"] = int(is_surface.sum())

        acc_c, kappa_c, ds_c, color_c = color_band_force(pos, mass, density, cfg)
        dp_c, dp_c_bin, prof_c = implied_pressure_jump(pos, acc_c, density, radius, dx)
        band_c = ds_c > 0.05 * float(ds_c.max())
        row["diagnostic_color_variant"] = {
            "dp_implied_Pa": dp_c,
            "dp_implied_binned_Pa": dp_c_bin,
            "err_rel": (dp_c - theory) / theory,
            "sigma_eff_over_sigma": dp_c / theory,
            "kappa_absmedian_band": float(kappa_c[band_c].abs().median()) if bool(band_c.any()) else 0.0,
            "delta_s_max": float(ds_c.max()),
            "color_min": float(color_c.min()),
            "color_max": float(color_c.max()),
            "profile": prof_c,
        }
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", nargs="+", type=float, default=list(DX_LIST))
    ap.add_argument("--out", default="artifacts/surface_tension_laplace/laplace_report.json")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    print("positive control (analytic Gaussian interface delta, 3dx inside R, width dx):")
    controls = []
    for dx in args.dx:
        ctl = positive_control(dx)
        controls.append({"dx_m": dx, **ctl})
        print(f"  dx={dx:8.5f}  recovered {ctl['dp_recovered_Pa']:.4f} Pa vs "
              f"{ctl['dp_theory_Pa']:.4f} Pa  ({ctl['err_rel']*100:+.1f}%)")

    rows = []
    print("")
    print(f"{'dx [m]':>10} {'N':>6} {'dp_implied':>11} {'dp_theory':>10} "
          f"{'err_rel':>9} {'sig_eff/sig':>11} {'inward%':>8} {'|a|max':>9} {'kappa/1/R':>10}")
    for dx in args.dx:
        row = run_resolution(dx)
        rows.append(row)
        ratio = row.get("kappa_mean_band", float("nan")) / (1.0 / RADIUS)
        print(f"{dx:10.5f} {row['n_particles']:6d} {row['dp_implied_Pa']:11.4f} "
              f"{row['dp_theory_Pa']:10.4f} {row['err_rel']*100:8.1f}% "
              f"{row['dp_implied_Pa']/(row['dp_theory_Pa']):11.2f} "
              f"{row['inward_frac_dot_gt_0p5']*100:7.1f}% {row['a_max_m_per_s2']:9.3f} "
              f"{ratio:10.2f}")

    # Convergence verdict.  A verified model must be accurate at every
    # resolution *and* stable under refinement: halving dx may not move
    # Delta-p by more than 15 %.  Error decreasing slowly towards a value that
    # is still wrong is not convergence.
    errs = [abs(r["err_rel"]) for r in rows]
    dps = [r["dp_implied_Pa"] for r in rows]
    steps = [abs(b - a) / abs(a) for a, b in zip(dps, dps[1:])]
    stable = all(s <= 0.15 for s in steps)
    worst = max(errs)
    verdict = {
        "criteria": {
            "err_rel_abs_le_0.15_every_resolution": bool(worst <= 0.15),
            "dp_change_under_refinement_le_0.15": bool(stable),
            "dp_relative_step_max": max(steps) if steps else 0.0,
            "inward_frac_ge_0.9_every_resolution": bool(
                all(r["inward_frac_dot_gt_0p5"] >= 0.9 for r in rows)),
        },
        "max_abs_err_rel": worst,
    }
    verdict["status"] = "PASS" if all(verdict["criteria"].values()) else "FAIL"

    report = {
        "benchmark": "surface_tension_young_laplace_2d",
        "model": "CSF heuristic colour (shipped, surface_tension.compute_surface_tension_force)",
        "setup": {
            "radius_m": RADIUS, "sigma_N_per_m": SIGMA, "rho0_kg_per_m3": RHO0,
            "h_factor": Config().h_factor, "gravity": 0.0,
            "dp_theory_Pa": SIGMA / RADIUS,
            "estimator": "dp = -int_0^R rho f_r dr (static radial momentum balance)",
        },
        "positive_control": controls,
        "resolution_study": rows,
        "verdict": verdict,
        "notes": [
            "the estimator isolates the surface-tension model from WCSPH density noise; "
            "with c0=20 m/s the density excess carrying 1.44 Pa is ~4e-6 relative",
            "the diagnostic colour-function variant is not part of the library; it is "
            "run to test whether the failure is local to the shipped colour heuristic",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print(f"verdict: {verdict['status']}  "
          f"(max |err| = {verdict['max_abs_err_rel']*100:.1f}%, "
          f"refinement-stable = {verdict['criteria']['dp_change_under_refinement_le_0.15']}, "
          f"max step {verdict['criteria']['dp_relative_step_max']*100:.0f}%)")

    if args.plot:
        from .plot import make_figures          # noqa: PLC0415  (optional dependency)
        make_figures(report, out.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())