"""Long-time stability of the single-phase droplet, and the sigma = 0 control.

Why this file exists
--------------------
The equilibrium ladder (``run.py``) showed that ``p_core`` has no plateau inside
four capillary times: it decays monotonically at every resolution.  A slow decay
could be (a) an under-relaxed transient, or (b) evidence that the configuration
has no static state at all.  This script decides between the two by running one
coarse droplet to twelve capillary times and watching

* ``p_core / (sigma/R)``      -- the quantity the Laplace verification wants;
* ``r_rms``                   -- the radius of gyration of the particle cloud;
* ``rho_excess``              -- interior density relative to ``rho0``;
* ``v_max / v_ref``           -- how far from static the run is.

The ``sigma = 0`` control repeats the identical run with the surface-tension
coefficient switched off.  It separates the two candidate mechanisms: if the
cloud expands with no surface tension either, the expansion is a free-surface
pressure/density effect and *not* something the CSF force causes.

Outputs ``drift.json`` and ``laplace_drift.png``; every number quoted in
``docs/verification_laplace_equilibrium.md`` is read from the JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, rhs, stable_dt
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
SIGMA, RHO0, C0, RADIUS = 7.2, 1000.0, 20.0, 0.02
torch.set_num_threads(2)


def hexagonal_droplet(radius, dx, rho0):
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    pts, row, y = [], 0, -radius
    while y <= radius:
        offset = 0.5 * dx if (row % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        row += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


def trace(dx=0.003, cusp=12.0, nu=0.02, sigma=SIGMA, every=100,
          blowup_v_over_vref=1.0):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=7.0,
                 gravity=0.0, viscosity=nu,
                 surface_tension_coefficient=sigma, csf_scheme="consistent")
    h = cfg.h_factor * dx
    dt = float(stable_dt(cfg))
    t_sig = math.sqrt(RHO0 * RADIUS ** 3 / SIGMA)
    v_ref = math.sqrt(SIGMA / (RHO0 * RADIUS))
    steps = int(round(cusp * t_sig / dt))
    pos, mass = hexagonal_droplet(RADIUS, dx, RHO0)
    vel = torch.zeros_like(pos)
    rows = []
    blew_up = False
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if not torch.isfinite(pos).all():
            blew_up = True
            break
        if step % every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            full = r < RADIUS - 2.5 * h
            v_over = float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref
            rows.append({
                "t_over_t_sigma": (step + 1) * dt / t_sig,
                "p_core_over_theory": float(pressure[full].mean())
                / (sigma / RADIUS) if sigma > 0 else 0.0,
                "r_rms_m": float((r ** 2).mean().sqrt()),
                "r_rms_over_r0": float((r ** 2).mean().sqrt()) / RADIUS,
                "rho_excess": float((rho[full] / RHO0 - 1.0).mean())
                if full.any() else float("nan"),
                "v_over_vref": v_over,
            })
            if v_over > blowup_v_over_vref:
                blew_up = True
                break
    return {
        "setup": {"dx_m": dx, "n_particles": int(pos.shape[0]), "dt_s": dt,
                  "steps_planned": steps, "viscosity_m2_per_s": nu,
                  "sigma_N_per_m": sigma, "rho0": RHO0, "radius_m": RADIUS,
                  "sound_speed_m_per_s": C0, "h_over_radius": 1.3 * dx / RADIUS,
                  "note": "sigma = 0 disables the CSF term entirely (control)"},
        "outcome": {
            "blew_up": blew_up,
            "t_over_t_sigma_last_finite": rows[-1]["t_over_t_sigma"] if rows else 0.0,
            "r_rms_over_r0_first": rows[0]["r_rms_over_r0"] if rows else None,
            "r_rms_over_r0_last": rows[-1]["r_rms_over_r0"] if rows else None,
            "rho_excess_last": rows[-1]["rho_excess"] if rows else None,
            "p_core_over_theory_last": rows[-1]["p_core_over_theory"] if rows else None,
        },
        "trace": rows,
    }


def make_figure(runs: list, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
    for run in runs:
        s, tr = run["setup"], run["trace"]
        label = (f"sigma = {s['sigma_N_per_m']:g} N/m" if s["sigma_N_per_m"]
                 else "sigma = 0 (control)")
        t = [r["t_over_t_sigma"] for r in tr]
        ax[0].plot(t, [r["r_rms_over_r0"] for r in tr], lw=1.6, label=label)
        if s["sigma_N_per_m"]:
            ax[1].plot(t, [r["p_core_over_theory"] for r in tr], lw=1.6,
                       label=label)
        ax[2].plot(t, [r["rho_excess"] for r in tr], lw=1.6, label=label)
    ax[0].axhline(1.0, color="k", ls=":", lw=1)
    ax[0].set_xlabel(r"$t/t_\sigma$")
    ax[0].set_ylabel(r"$r_\mathrm{rms}/R_0$")
    ax[0].set_title("(a) the cloud expands, with and without\nsurface tension")
    ax[1].axhline(1.0, color="k", ls="--", lw=1.1, label=r"$\sigma/R$")
    ax[1].set_xlabel(r"$t/t_\sigma$")
    ax[1].set_ylabel(r"$p_\mathrm{core}/(\sigma/R)$")
    ax[1].set_title("(b) no plateau: the jump decays\n"
                    "(complete-support particles)")
    ax[2].axhline(0.0, color="k", ls=":", lw=1)
    ax[2].set_xlabel(r"$t/t_\sigma$")
    ax[2].set_ylabel(r"$\bar\rho/\rho_0 - 1$")
    ax[2].set_title("(c) interior density deficit grows")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    runs = [trace(sigma=SIGMA), trace(sigma=0.0)]
    for run in runs:
        o = run["outcome"]
        print(f"sigma={run['setup']['sigma_N_per_m']:5g}  "
              f"t_end/t_sigma={o['t_over_t_sigma_last_finite']:.2f}  "
              f"r_rms/R0 {o['r_rms_over_r0_first']:.3f} -> "
              f"{o['r_rms_over_r0_last']:.3f}  "
              f"rho_exc={o['rho_excess_last']:+.4f}  "
              f"blew_up={o['blew_up']}")
    report = {
        "benchmark": "laplace_equilibrium/drift",
        "purpose": "does the single-phase droplet have a static equilibrium?",
        "reference": {"t_sigma_s": math.sqrt(RHO0 * RADIUS ** 3 / SIGMA),
                      "v_ref_m_per_s": math.sqrt(SIGMA / (RHO0 * RADIUS))},
        "runs": runs,
    }
    (HERE / "drift.json").write_text(json.dumps(report, indent=1))
    print("drift ->", HERE / "drift.json")
    print("figure ->", make_figure(runs, HERE / "laplace_drift.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())