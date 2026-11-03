"""Candidate-neutral scorer for the CSF surface-tension model.

This script is the instrument of the referee.  It imports ``tensorsph`` by name
(PYTHONPATH decides *which* build: the merged tree or a shadow copy of another
mission's branch) and touches **only the public API**:

    Config, cell_linked_pairs, density_from_neighbor_pairs, tait_pressure,
    stable_dt, rhs, compute_surface_tension_force

so a candidate cannot be scored through fields it computes for its own
benchmark -- whatever the user gets by setting ``surface_tension_coefficient``
is what is measured here.

Three things are measured on a 2-D hexagonal drop at rest:

A. **Cut-force identity** -- ``F_cut = sum_{x>0} m a_st,x`` must equal
   ``-2 sigma kappa R``; with ``kappa = 1/R`` that is ``-2 sigma``.  Dividing
   the measured value by ``-2 sigma`` therefore *is* the curvature ratio
   ``kappa_meas / (1/R)`` when ``int delta_s dn = 1``, and it is the strongest
   single check on the CSF force magnitude.

B. **Laplace pressure jump, two estimators** -- this is where candidate reports
   have diverged, so both are computed on the same relaxed state:

   B1 ``dp_force`` : ``sum_i rho_i (-a_i . n_hat) V_i / (2 pi r_c)``.
      This is an integral of the *surface-tension force* over the fluid.  It is
      analytically ``sigma kappa`` -- i.e. it re-measures the force and is
      equivalent to (A) up to the normalisation of ``delta_s``.
   B2 ``dp_pressure`` : ``p(r<0.5R) - p(0.8R<r<0.95R)`` from the Tait EOS on
      the summation density.  This is the only estimator that tests whether the
      *flow solver* actually sustains the over-pressure the law demands.

C. **Spurious current** -- ``max|v| / sqrt(sigma/(rho R))`` (Adami et al. 2012).

Output: one JSON document per candidate, written by the driver.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import tensorsph as ts
from tensorsph import (Config, cell_linked_pairs, density_from_neighbor_pairs,
                       stable_dt, tait_pressure)
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(4)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius: float, dx: float, rho0: float):
    """Hexagonally packed circular drop; ``mass * number density = rho0``."""
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
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


def st_force(pos, rho, mass, pairs, cfg):
    """Call the candidate's force function without assuming its signature."""
    try:
        return compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    except TypeError:
        return compute_surface_tension_force(pos, rho, mass, pairs, cfg, wall=None)


def make_cfg(dx: float, sigma: float, rho0: float, sound_speed: float,
             viscosity: float) -> Config:
    return Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                  gamma=7.0, gravity=0.0, viscosity=viscosity,
                  surface_tension_coefficient=sigma)


# --------------------------------------------------------------------------- #
# A. instantaneous force diagnostics
# --------------------------------------------------------------------------- #
def instant(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0,
            sound_speed=20.0, viscosity=0.01) -> dict:
    cfg = make_cfg(dx, sigma, rho0, sound_speed, viscosity)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc = st_force(pos, rho, mass, pairs, cfg)

    right = pos[:, 0] > 0.0
    f_cut = float((mass[right, None] * acc[right]).sum(dim=0)[0])
    f_theory = -2.0 * sigma
    # kappa_eff = |F_cut| / (2 sigma R)  (equals 1/R iff int delta_s dn = 1)
    kappa_eff = abs(f_cut) / (2.0 * sigma * radius)

    r_hat = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
    r_norm = torch.linalg.vector_norm(pos, dim=1) / radius
    mag = torch.linalg.vector_norm(acc, dim=1)
    cos_in = -(acc * r_hat).sum(dim=1) / mag.clamp_min(1e-30)
    band = (mag > 0) & (r_norm >= 0.6) & (r_norm <= 0.98)
    cb = cos_in[band]
    return {
        "dx_m": dx,
        "n_particles": int(pos.shape[0]),
        "kappa_exact_1_per_m": 1.0 / radius,
        "kappa_eff_1_per_m": kappa_eff,
        "kappa_rel_error_pct": (kappa_eff - 1.0 / radius) / (1.0 / radius) * 100.0,
        "cut_force_measured_N_per_m": f_cut,
        "cut_force_theory_N_per_m": f_theory,
        "cut_force_ratio": f_cut / f_theory,
        "cos_inward_median": float(cb.median()) if cb.numel() else float("nan"),
        "cos_inward_mean": float(cb.mean()) if cb.numel() else float("nan"),
        "fraction_inward": float((cos_in[mag > 0] > 0).float().mean()),
        "mean_force_mag": float(mag.mean()),
        "max_force_mag": float(mag.max()),
    }


# --------------------------------------------------------------------------- #
# B/C. relaxation in time, then both pressure estimators
# --------------------------------------------------------------------------- #
def relax(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
          viscosity=0.01, steps=900) -> dict:
    cfg = make_cfg(dx, sigma, rho0, sound_speed, viscosity)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    v_ref = math.sqrt(sigma / (rho0 * radius))
    dp_theory = sigma / radius

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vol0 = float(mass[0] / rho0)
    vel = torch.zeros_like(pos)
    rho = tait_pressure  # placeholder to keep linters quiet
    hist = {"t": [], "vmax": [], "p_center": [], "rho_center": [], "rho_band": []}
    avg_from = int(0.75 * steps)
    acc_in, acc_band, n_avg = 0.0, 0.0, 0
    sq_in, sq_band = 0.0, 0.0

    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = ts.rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                   dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step >= avg_from:
            rr = torch.linalg.vector_norm(pos, dim=1) / radius
            m_in = float(pressure[rr < 0.5].mean())
            m_bd = float(pressure[(rr > 0.8) & (rr < 0.95)].mean())
            acc_in += m_in; acc_band += m_bd; n_avg += 1
            sq_in += m_in * m_in; sq_band += m_bd * m_bd
        if step % 50 == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1) / radius
            hist["t"].append((step + 1) * dt)
            hist["vmax"].append(float(torch.linalg.vector_norm(vel, dim=1).max()))
            hist["p_center"].append(float(pressure[r < 0.3].mean()))
            hist["rho_center"].append(float(rho[r < 0.3].mean()))
            hist["rho_band"].append(float(rho[(r > 0.8) & (r < 0.95)].mean()))

    # ---- final state -------------------------------------------------------
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho_f = density_from_neighbor_pairs(pos, mass, h, pairs)
    p_f = tait_pressure(rho_f, cfg)
    acc_f = st_force(pos, rho_f, mass, pairs, cfg)

    r = torch.linalg.vector_norm(pos, dim=1)
    rn = r / radius
    core = rn < 0.5
    bandm = (rn > 0.8) & (rn < 0.95)
    p_in = float(p_f[core].mean())
    p_band = float(p_f[bandm].mean())
    dp_pressure = p_in - p_band
    err_pressure = (dp_pressure - dp_theory) / dp_theory * 100.0

    # B1: force-integral estimator (what some candidate reports score)
    r_hat = pos / r.clamp_min(1e-30).unsqueeze(1)
    w = rho_f * (-(acc_f * r_hat).sum(dim=1)) * vol0       # N/m per particle
    # force-weighted centroid radius (geometry-exact for a thin shell)
    wpos = w.clamp_min(0.0)
    r_c = float((wpos * r).sum() / wpos.sum().clamp_min(1e-30))
    dp_force = float(w.sum()) / (2.0 * math.pi * r_c)
    err_force = (dp_force - dp_theory) / dp_theory * 100.0

    vmax = float(torch.linalg.vector_norm(vel, dim=1).max())
    p_std = float(p_f[bandm].std())
    comp_needed = (1.0 + dp_theory / (rho0 * sound_speed ** 2 / 7.0)) ** (1 / 7) - 1

    return {
        "setup": {"radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
                  "n_particles": int(pos.shape[0]), "sigma_N_per_m": sigma,
                  "rho0": rho0, "sound_speed": sound_speed,
                  "viscosity": viscosity, "steps": steps,
                  "physical_time_s": steps * dt,
                  "t_over_t_sigma": steps * dt / t_sigma},
        "reference": {"delta_p_Pa": dp_theory, "curvature_1_per_m": 1.0 / radius,
                      "cut_force_N_per_m": -2.0 * sigma,
                      "capillary_time_s": t_sigma,
                      "capillary_velocity_m_per_s": v_ref},
        "laplace_pressure": {                       # B2 -- the physical test
            "delta_p_theory_Pa": dp_theory,
            "p_center_Pa": p_in, "p_band_Pa": p_band,
            "delta_p_measured_Pa": dp_pressure,
            "rel_error_pct": err_pressure,
            "within_15pct": abs(err_pressure) < 15.0,
        },
        "laplace_pressure_timeavg": {      # B2 averaged over the last 25 % of steps
            "n_samples": n_avg,
            "delta_p_theory_Pa": dp_theory,
            "p_center_Pa": acc_in / max(n_avg, 1),
            "p_band_Pa": acc_band / max(n_avg, 1),
            "delta_p_measured_Pa": acc_in / max(n_avg, 1) - acc_band / max(n_avg, 1),
            "rel_error_pct": ((acc_in - acc_band) / max(n_avg, 1) - dp_theory)
                             / dp_theory * 100.0,
            "within_15pct": abs(((acc_in - acc_band) / max(n_avg, 1) - dp_theory)
                                / dp_theory) < 0.15,
            "p_center_sample_std_Pa": math.sqrt(max(sq_in / max(n_avg, 1) - (acc_in / max(n_avg, 1)) ** 2, 0.0)),
            "p_band_sample_std_Pa": math.sqrt(max(sq_band / max(n_avg, 1) - (acc_band / max(n_avg, 1)) ** 2, 0.0)),
        },
        "laplace_force_integral": {                 # B1 -- the force re-measure
            "delta_p_theory_Pa": dp_theory,
            "r_centroid_m": r_c,
            "delta_p_measured_Pa": dp_force,
            "rel_error_pct": err_force,
            "within_15pct": abs(err_force) < 15.0,
        },
        "spurious_current": {"v_ref_m_per_s": v_ref,
                             "vmax_final_m_per_s": vmax,
                             "ratio": vmax / v_ref,
                             "passes_adami_0p1": vmax / v_ref < 0.1},
        "density": {"rho_center_over_rho0": float(rho_f[core].mean() / rho0),
                    "rho_band_over_rho0": float(rho_f[bandm].mean() / rho0),
                    "compression_required_by_law": comp_needed,
                    "pressure_noise_band_Pa": p_std,
                    "pressure_noise_over_dp": p_std / dp_theory},
        "history": hist,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="both", choices=["both", "instant", "relax"])
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--dx-relax", type=float, default=0.002)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    a = ap.parse_args()

    doc = {"candidate": a.candidate, "tensorsph_file": ts.__file__}
    try:
        doc["version"] = ts.__version__
    except AttributeError:
        doc["version"] = None

    if a.mode in ("both", "instant"):
        rows = [instant(a.radius, dx, a.sigma) for dx in (0.003, 0.002, 0.0015)]
        doc["instant"] = {"reference_curvature_1_per_m": 1.0 / a.radius,
                          "resolutions": rows}
        for r in rows:
            print(f"[{a.candidate}] instant dx={r['dx_m']:.4f} "
                  f"kappa_err={r['kappa_rel_error_pct']:+7.1f}% "
                  f"Fcut/2s={r['cut_force_ratio']:+6.3f} "
                  f"cos_med={r['cos_inward_median']:+5.3f} "
                  f"inward={r['fraction_inward']:.2f}")

    if a.mode in ("both", "relax"):
        rep = relax(radius=a.radius, dx=a.dx_relax, sigma=a.sigma,
                    steps=a.steps)
        doc["relax"] = rep
        print(f"[{a.candidate}] relax dx={a.dx_relax} t/t_s={rep['setup']['t_over_t_sigma']:.2f} "
              f"dp_pressure_err={rep['laplace_pressure']['rel_error_pct']:+7.1f}% "
              f"dp_force_err={rep['laplace_force_integral']['rel_error_pct']:+7.1f}% "
              f"dp_timeavg_err={rep['laplace_pressure_timeavg']['rel_error_pct']:+7.1f}% "
              f"spurious={rep['spurious_current']['ratio']:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()