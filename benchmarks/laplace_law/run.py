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

from tensorsph.csf_consistent import csf_fields as csf_fields_consistent
from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius: float, dx: float, rho0: float,
                      center: tuple[float, float] = (0.0, 0.0)):
    """Hexagonally packed circular drop."""
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
    """Everything that can be scored on a single static evaluation.

    Uses the **consistent** CSF scheme (kernel-sum colour + MLS operators)
    for all field scoring, so that curvature, delta_s, and cut-force are
    evaluated on the same discretisation that produces the acceleration field.
    """
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    n = pos.shape[0]
    pairs = cell_linked_pairs(pos, 2.0 * h)

    # Use the consistent scheme's csf_fields for all scoring
    csf = csf_fields_consistent(pos, mass, pairs, h, rho0)
    kappa = csf.kappa
    delta_s = csf.delta_s
    is_surface = csf.surface_mask()

    # Density needed for acceleration computation
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc_st = compute_surface_tension_force(pos, rho, mass, pairs, cfg)

    # -- cut-force identity (2): force transmitted across the y-axis cut ------- #
    right = pos[:, 0] > 0.0
    f_cut = float((mass[right, None] * acc_st[right]).sum(dim=0)[0])
    kappa_exact = 1.0 / radius
    f_cut_theory = -2.0 * sigma

    kappa_band = radial_band(pos, kappa.abs(), radius)
    w = delta_s.abs()
    kappa_w = float((kappa.abs() * w).sum() / w.sum().clamp_min(1e-30))
    ds_band = radial_band(pos, delta_s, radius)
    # inward alignment
    r_hat = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
    mag = torch.linalg.vector_norm(acc_st, dim=1)
    cos_in = -(acc_st * r_hat).sum(dim=1) / mag.clamp_min(1e-30)
    aligned = mag > 0
    r_norm = torch.linalg.vector_norm(pos, dim=1) / radius
    band = aligned & (r_norm >= 0.6) & (r_norm <= 0.98)
    cb = cos_in[band]
    cos_in_band = {"count": int(band.sum()),
                   "mean": float(cb.mean()) if cb.numel() else float("nan"),
                   "median": float(cb.median()) if cb.numel() else float("nan")}

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
            "ideal_value_1_per_m": 1.0 / (2.0 * h),
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
                 surface_tension_coefficient=sigma, csf_scheme="consistent")
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
    csf = csf_fields_consistent(pos, mass, pairs, h, rho0)
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
        },
        "curvature_relaxed": {
            "band": radial_band(pos, csf.kappa.abs(), radius),
            "rel_error_pct": (radial_band(pos, csf.kappa.abs(), radius)["mean"]
                              - 1.0 / radius) / (1.0 / radius) * 100.0,
        },
        "spurious_current": {
            "v_ref_m_per_s": v_ref,
            "vmax_final_m_per_s": vmax,
            "ratio": vmax / v_ref,
            "passes_adami_0p1": vmax / v_ref < 0.1,
        },
        "density": {
            "rho_center_over_rho0": float(rho_f[r < 0.3 * radius].mean() / rho0),
            "rho_band_over_rho0": float(
                rho_f[(r > 0.8) & (r < 0.95)].mean() / rho0),
            "compression_required_by_law": comp_needed,
        },
        "pressure_profile_final": pressure_profile(pos, p_f, radius),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["instant", "relax", "both"],
                    default="both")
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--rho0", type=float, default=1000.0)
    ap.add_argument("--sound-speed", type=float, default=20.0)
    ap.add_argument("--viscosity", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=900)
    a = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.mode in ("instant", "both"):
        dxs = [0.003, 0.002, 0.0015, 0.001]
        ladder = []
        for dx in dxs:
            row = instant_diagnostics(a.radius, dx, a.sigma, a.rho0,
                                      a.sound_speed, a.viscosity)
            ladder.append(row)
            print(f"dx={dx:<7g} N={row['n_particles']:<6d} "
                  f"kappa_err={row['curvature']['delta_s_weighted_rel_error_pct']:+7.1f}% "
                  f"Fcut/(2sig)={row['cut_force']['ratio_measured_over_theory']:.3f} "
                  f"cos_med={row['inward_alignment']['cos_band_median']:+.3f}")
        (out_dir / "instant_report.json").write_text(
            json.dumps({"ladder": ladder}, indent=2))

    if a.mode in ("relax", "both"):
        rep = relax(a.radius, a.dx, a.sigma, a.rho0, a.sound_speed, a.viscosity,
                    a.steps)
        (out_dir / "validation_report.json").write_text(
            json.dumps(rep, indent=2))
        print(f"relax Delta p={rep['laplace']['delta_p_measured_Pa']:.1f} Pa "
              f"vs {rep['reference']['delta_p_Pa']:.1f} Pa "
              f"({rep['laplace']['rel_error_pct']:+.1f}%)  "
              f"spurious={rep['spurious_current']['ratio']:.3f}")


if __name__ == "__main__":
    main()
