"""Laplace-law verification of the CSF surface-tension model.

Why this benchmark exists
-------------------------
Surface tension (CSF, Brackbill et al. 1992) was merged into TensorSPH without
ever been checked against the analytic Young-Laplace law.  The older
``benchmarks/static_droplet`` run prints ``Δp_theory = σ/R`` but only tests the
*sign* of the force, so that theory value is decorative.  This harness measures
the pressure jump and compares it, number for number, with ``σ/R``.

Three analytic references (2-D, single-phase drop)
--------------------------------------------------
1. **Young-Laplace**: a circular drop of radius ``R`` has one non-zero
   principal curvature, hence

       Δp = p_in - p_out = σ / R                                    (1)

2. **Cut-force identity**: a cut through the drop exposes two interfaces, so
   the total surface-tension force transmitted across the cut is exactly
   ``2σ`` per unit out-of-plane depth (the classical "surface tension =
   force per unit length").  This scores the *magnitude* of the CSF force
   without relying on any pressure measurement:

       F_cut = Σ_{x>0} m_i a_st,i,x  ->  -2σ                        (2)

3. **Curvature**: the CSF pipeline must return ``κ = 1/R`` on a circle.

The spurious-current level is scored against
``max|v| / sqrt(σ/(ρR)) << 1`` (Adami, Ihmsen & Adams 2012, JCP 231:7057).

Usage::

    PYTHONPATH=src python benchmarks/laplace_law/run.py --mode instant
    PYTHONPATH=src python benchmarks/laplace_law/run.py --mode relax
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import csf_fields as csf_fields_consistent
from tensorsph.surface_tension import (
    compute_surface_tension_force,
)

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius: float, dx: float, rho0: float,
                      center: tuple[float, float] = (0.0, 0.0)):
    """Hexagonally packed circular drop.

    A square lattice has 4 nearest neighbours, hex packing has 6, which halves
    the noise of the neighbour-count surface detector and of the SPH gradient
    operators.  The lattice constant is set so ``mass * number_density = rho0``.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    mass = torch.tensor(rho0 * dx * dy, dtype=torch.float64)
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((center[0] + x, center[1] + y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


# --------------------------------------------------------------------------- #
# CSF pipeline introspection (uses the consistent scheme from csf_consistent)
# --------------------------------------------------------------------------- #
def csf_fields(pos, rho, mass, pairs, cfg):
    """The (color, grad_c, normal, kappa, delta_s, is_surface) fields the CSF
    force is built from, so each ingredient can be scored separately.

    Uses the first-order consistent (MLS) scheme from ``csf_consistent``:
    kernel-sum colour function + MLS gradient/divergence + Shepard-smoothed
    normal.  This is the same scheme that ``compute_surface_tension_force``
    uses by default (``csf_scheme="consistent"``).
    """
    h = cfg.h_factor * cfg.dx
    rho0 = cfg.rho0
    state = csf_fields_consistent(pos, mass, pairs, h, rho0)
    # Legacy interface: return the same tuple the old function returned
    is_surface = state.surface_mask(fraction=0.05)
    return state.color, state.grad_c, state.normal, state.kappa, state.delta_s, is_surface


def radial_band(pos, values, radius, lo=0.6, hi=0.98):
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    m = (r >= lo) & (r <= hi)
    if not bool(m.any()):
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan")}
    v = values[m]
    return {
        "count": int(m.sum()),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "std": float(v.std()) if v.numel() > 1 else 0.0,
    }


def pressure_profile(pos, pressure, radius, nbins=20):
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    edges = torch.linspace(0.0, 1.0, nbins + 1, dtype=pos.dtype)
    prof = []
    for k in range(nbins):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if bool(m.any()):
            prof.append({
                "r_over_R": float(0.5 * (edges[k] + edges[k + 1])),
                "p_mean": float(pressure[m].mean()),
                "p_std": float(pressure[m].std()) if int(m.sum()) > 1 else 0.0,
                "n": int(m.sum()),
            })
    return prof


def instant_diagnostics(radius, dx, sigma, rho0, sound_speed, viscosity):
    """Everything that can be scored on a single static evaluation."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    n = pos.shape[0]
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)

    color, grad_c, normal, kappa, delta_s, is_surface = csf_fields(
        pos, rho, mass, pairs, cfg)
    acc_st = compute_surface_tension_force(pos, rho, mass, pairs, cfg)

    # -- cut-force identity (2): force transmitted across the y-axis cut ------- #
    right = pos[:, 0] > 0.0
    f_cut = float((mass[right, None] * acc_st[right]).sum(dim=0)[0])
    kappa_exact = 1.0 / radius
    f_cut_theory = -2.0 * sigma

    kappa_band = radial_band(pos, kappa.abs(), radius)
    # Interface curvature: the physically meaningful average is weighted by the
    # surface delta function |grad c|, because that is the weight the CSF force
    # itself uses.  An unweighted band average mixes interior particles
    # (kappa ~ 0) with interface particles (kappa large) and is not a fair
    # score of the estimator.
    w = delta_s.abs()
    kappa_w = float((kappa.abs() * w).sum() / w.sum().clamp_min(1e-30))
    ds_band = radial_band(pos, delta_s, radius)
    # inward alignment: a_st should oppose the radial unit vector
    r_hat = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
    mag = torch.linalg.vector_norm(acc_st, dim=1)
    # true cosine: +1 means the CSF force points exactly inward (opposite r_hat)
    cos_in = -(acc_st * r_hat).sum(dim=1) / mag.clamp_min(1e-30)
    aligned = mag > 0
    r_norm = torch.linalg.vector_norm(pos, dim=1) / radius
    band = aligned & (r_norm >= 0.6) & (r_norm <= 0.98)
    cb = cos_in[band]
    cos_in_band = {"count": int(band.sum()),
                   "mean": float(cb.mean()) if cb.numel() else float("nan"),
                   "median": float(cb.median()) if cb.numel() else float("nan")}

    # pressure implied by the cut force, for comparison with (1)
    dp_from_force = abs(f_cut) / (2.0 * radius)

    return {
        "dx_m": dx,
        "h_m": h,
        "n_particles": n,
        "dt_acoustic_s": float(stable_dt(cfg)),
        "n_surface_particles": int(is_surface.sum()),
        "surface_fraction": float(is_surface.float().mean()),
        "curvature": {
            "exact_1_per_m": kappa_exact,
            "band": kappa_band,
            "rel_error_pct": (kappa_band["mean"] - kappa_exact)
                             / kappa_exact * 100.0,
            "delta_s_weighted_mean_1_per_m": kappa_w,
            "delta_s_weighted_rel_error_pct": (kappa_w - kappa_exact)
                                              / kappa_exact * 100.0,
        },
        "delta_s": {
            "band": ds_band,
            "ideal_value_1_per_m": 1.0 / (2.0 * h),   # |grad c| across one h
        },
        "cut_force": {
            "theory_N_per_m": f_cut_theory,
            "measured_N_per_m": f_cut,
            "ratio_measured_over_theory": f_cut / f_cut_theory,
            "implied_delta_p_Pa": dp_from_force,
        },
        "inward_alignment": {
            "cos_band_mean": cos_in_band["mean"],
            "cos_band_median": cos_in_band["median"],
            "fraction_inward": float((cos_in[aligned] > 0).float().mean()),
            "n_aligned": int(aligned.sum()),
        },
        "max_acc_st_m_per_s2": float(mag.max()),
        "mean_acc_st_m_per_s2": float(mag.mean()),
    }


def relax(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
          steps=900, viscosity=0.01):
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    v_ref = math.sqrt(sigma / (rho0 * radius))
    dp_theory = sigma / radius

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    pressure = tait_pressure(rho, cfg)

    hist = {"t": [], "vmax": [], "ke": [], "p_center": [], "rho_center": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % 50 == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            core = r < 0.3 * radius
            hist["t"].append((step + 1) * dt)
            hist["vmax"].append(float(torch.linalg.vector_norm(vel, dim=1).max()))
            hist["ke"].append(float(0.5 * (mass @ (vel * vel).sum(dim=1))))
            hist["p_center"].append(float(pressure[core].mean()))
            hist["rho_center"].append(float(rho[core].mean()))

    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho_f = density_from_neighbor_pairs(pos, mass, h, pairs)
    p_f = tait_pressure(rho_f, cfg)
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    p_in = float(p_f[r < 0.5].mean())
    p_band = float(p_f[(r > 0.8) & (r < 0.95)].mean())
    dp_meas = p_in - p_band
    err = (dp_meas - dp_theory) / dp_theory * 100.0
    vmax = float(torch.linalg.vector_norm(vel, dim=1).max())
    _, _, _, kappa_f, _, _ = csf_fields(pos, rho_f, mass, pairs, cfg)
    comp_needed = (1.0 + dp_theory / (rho0 * sound_speed ** 2 / 7.0)) ** (1 / 7) - 1
    return {
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
            "n_particles": n, "sigma_N_per_m": sigma, "rho0": rho0,
            "sound_speed": sound_speed, "viscosity": viscosity,
            "steps": steps, "physical_time_s": steps * dt,
            "t_over_t_sigma": steps * dt / t_sigma,
            "lattice": "hexagonal", "gravity": 0.0,
        },
        "reference": {
            "delta_p_Pa": dp_theory,
            "curvature_1_per_m": 1.0 / radius,
            "capillary_time_s": t_sigma,
            "capillary_velocity_m_per_s": v_ref,
            "cut_force_N_per_m": -2.0 * sigma,
            "spurious_current_criterion": "max|v|/sqrt(sigma/(rho R)) << 1 "
                                          "(Adami et al. 2012)",
        },
        "laplace": {
            "delta_p_theory_Pa": dp_theory,
            "p_center_Pa": p_in,
            "p_interface_band_Pa": p_band,
            "delta_p_measured_Pa": dp_meas,
            "rel_error_pct": err,
            "within_3pct": abs(err) < 3.0,
            "within_15pct": abs(err) < 15.0,
            "density_compression_at_center": float(
                (rho_f[r < 0.5].mean() - rho0) / rho0),
            "compressibility_theory": comp_needed,
        },
        "spurious_current": {
            "v_max_m_per_s": vmax,
            "v_ref_m_per_s": v_ref,
            "ratio": vmax / v_ref if v_ref > 0 else float("nan"),
            "passes_adami_0p1": (vmax / v_ref) < 0.1 if v_ref > 0 else False,
        },
        "curvature_final": {
            "exact_1_per_m": 1.0 / radius,
            "band_mean_1_per_m": float(kappa_f.abs()[
                (r >= 0.6) & (r <= 0.98)].mean()),
        },
        "history": hist,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Laplace-law verification")
    parser.add_argument("--mode", choices=["instant", "relax", "both"],
                        default="both")
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--dx", type=float, default=0.002)
    parser.add_argument("--sigma", type=float, default=7.2)
    parser.add_argument("--rho0", type=float, default=1000.0)
    parser.add_argument("--sound-speed", type=float, default=20.0)
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = {}
    if args.mode in ("instant", "both"):
        # Multi-resolution instant diagnostics
        results = []
        for dx in [0.003, 0.002, 0.0015, 0.001]:
            d = instant_diagnostics(args.radius, dx, args.sigma, args.rho0,
                                    args.sound_speed, args.viscosity)
            results.append(d)
        out["instant"] = results

    if args.mode in ("relax", "both"):
        out["relax"] = relax(radius=args.radius, dx=args.dx, sigma=args.sigma,
                             rho0=args.rho0, sound_speed=args.sound_speed,
                             steps=args.steps, viscosity=args.viscosity)

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
