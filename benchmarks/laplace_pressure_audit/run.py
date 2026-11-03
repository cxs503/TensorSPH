"""Audit of the Young-Laplace *pressure* criterion in the CSF verification chain.

What this file is for
---------------------
``benchmarks/csf_consistent`` (and ``benchmarks/csf_curvature_fit``) score the
surface-tension model with the quantity

    dp = p(r < 0.5 R)  -  p(0.80 R < r < 0.95 R)                         (A)

and compare it against ``sigma / R`` -- the 2-D Young-Laplace jump.  That
comparison is wrong on two counts, and this harness shows both with numbers:

1. ``0.80 R < r < 0.95 R`` is *inside* the drop.  In the continuum solution the
   pressure of a static drop is uniform inside (``grad p = 0`` there, the
   surface force lives on the interface), so the theoretical value of (A) is
   **zero**, not ``sigma / R``.  Scoring (A) against ``sigma / R`` reports a
   failure that no correct implementation could ever avoid.
2. The Laplace jump is between the interior and the *exterior* (``p_gas = 0``
   here), i.e. the criterion a real implementation must satisfy is

       p(interior) = sigma / R + p_gas = sigma / R .                     (B)

This harness measures (B) -- the interior pressure and its resolved jump across
the interface shell -- together with an honest error budget, and it also
re-measures (A) so the historical numbers stay reproducible.  The radial
profiles expose where the missing pressure lives (shell particles carry the CSF
force but sit at a free-surface density deficit, so the ``p = max(Tait, 0)``
floor pins them at zero).

Measured quantities per case
----------------------------
``p_over_theory(r)``   radial pressure profile divided by ``sigma / R``;
``jump``               ``p(r < 0.4R) - p(r > 0.98R)`` (the resolved jump);
``interior``           time-averaged ``p(r < 0.4R)`` scored against (B);
``legacy``             the (A) measure, scored against ``sigma / R`` *and*
                       against its own theoretical value 0;
``noise``              EOS pressure noise floor ``gamma B * d(rho)/rho0`` from
                       the measured core density spread, plus the temporal
                       standard error of the core pressure;
``shell_flux``         CSF force per radial bin, integrated over r: the CSF
                       force really delivered across the interface
                       (``int f_r dn``, theory ``sigma / R``) -- separates
                       "force is wrong" from "pressure does not respond".

Usage::

    PYTHONPATH=src python benchmarks/laplace_pressure_audit/run.py            # full
    PYTHONPATH=src python benchmarks/laplace_pressure_audit/run.py --quick    # smoke
    PYTHONPATH=src python benchmarks/laplace_pressure_audit/run.py --dx 0.001 --steps 1800
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
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
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# 15 %: platform acceptance bar.  3 %: the "publishable" bar quoted in the
# report template.  Both are scored explicitly so nobody has to guess.
GATE_PCT = 15.0
TIGHT_PCT = 3.0

# Radial bins used for the profile: fractions of the initial radius.
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.7), (0.7, 0.8),
        (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 1.0), (1.0, 1.05)]


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_pressure_audit_law", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


def _pct(value: float, ref: float) -> float:
    return (value - ref) / ref * 100.0


def simulate(radius: float = 0.02, dx: float = 0.002, sigma: float = 7.2,
             rho0: float = 1000.0, sound_speed: float = 20.0,
             viscosity: float = 0.01, steps: int | None = None,
             sample_every: int = 50, curvature: str = "mls",
             scheme: str = "consistent",
             t_end_over_t_sigma: float = 2.8,
             density_renormalization: bool = False) -> dict:
    """Relax a static drop and return the pressure audit for one case.

    The run length is specified in capillary times, not steps: ``dt`` shrinks
    with ``dx``, so an equal step count would compare different physical times
    across the resolution ladder -- and the core pressure is still decaying at
    t ~ 1 t_sigma, which makes such a comparison meaningless.
    """
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme=scheme,
                 csf_curvature=curvature,
                 density_renormalization=density_renormalization)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    dp_theory = sigma / radius
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    if steps is None:
        steps = max(1, int(round(t_end_over_t_sigma * t_sigma / dt)))
    v_ref = math.sqrt(sigma / (rho0 * radius))
    bulk = rho0 * sound_speed ** 2 / cfg.gamma

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0 * h))
    pressure = tait_pressure(rho, cfg)

    t_hist, p_hist, rho_hist, vmax_hist = [], [], [], []
    t0 = time.time()
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        last = step == steps - 1
        if step % sample_every == 0 or last:
            r = torch.linalg.vector_norm(pos, dim=1)
            core = r < 0.4 * radius
            t_hist.append((step + 1) * dt)
            p_hist.append(float(pressure[core].mean()))
            rho_hist.append(float(rho[core].mean()) / rho0)
            vmax_hist.append(float(torch.linalg.vector_norm(vel, dim=1).max())
                             / v_ref)
    wall = time.time() - t0

    # --- time averages -----------------------------------------------------
    t_t = torch.tensor(t_hist)
    p_t = torch.tensor(p_hist)
    rho_t = torch.tensor(rho_hist)
    # Second half of the run: the first capillary transient is discarded.  The
    # window length is reported so the reader can judge how much averaging the
    # numbers rest on.
    t_end = float(t_t[-1])
    late = t_t >= 0.5 * t_end
    p_late, rho_late = p_t[late], rho_t[late]
    n_late = int(p_late.numel())
    p_mean = float(p_late.mean())
    p_std = float(p_late.std()) if n_late > 1 else 0.0
    p_sem = p_std / math.sqrt(max(n_late, 1))
    rho_mean = float(rho_late.mean())
    rho_std = float(rho_late.std()) if n_late > 1 else 0.0
    # A pressure derived from a summation density inherits gamma B drho/rho0.
    noise_pa = cfg.gamma * bulk * max(rho_std, 1e-12)
    # Half-cell lattice bias: the density of a perfect lattice also depends on
    # where the free surface is cut, which shows up as the *particle-to-particle*
    # spread at one instant, not in the time series.
    r_final = torch.linalg.vector_norm(pos, dim=1)
    core_f = r_final < 0.4 * radius
    shelf = (r_final > 0.8 * radius) & (r_final < 0.95 * radius)
    p_core = float(pressure[core_f].mean())
    p_shelf = float(pressure[shelf].mean())
    rho_core_spread = float(rho[core_f].std() / rho0)
    noise_pa_spatial = cfg.gamma * bulk * rho_core_spread

    # --- instantaneous radial profile + force flux -------------------------
    h_pairs = cell_linked_pairs(pos, 2.0 * h)
    force = compute_surface_tension_force(pos, rho, mass, h_pairs, cfg, wall=None)
    f_r = torch.linalg.vector_norm(force, dim=1)  # |a_sf| = |f_sf| / rho
    profile = []
    for lo, hi in BINS:
        m = (r_final > lo * radius) & (r_final <= hi * radius)
        if not bool(m.any()):
            continue
        profile.append({
            "r_over_R": [lo, hi],
            "n_particles": int(m.sum()),
            "p_over_theory": float(pressure[m].mean()) / dp_theory,
            "p_pa": float(pressure[m].mean()),
            "rho_over_rho0": float(rho[m].mean()) / rho0,
            "abs_a_sf": float(f_r[m].mean()),
        })
    # int f_r dn over the interface: sum over particles of rho a_r V / (2 pi r).
    # Theory: int sigma kappa delta_s dn = sigma kappa = sigma / R.
    v_part = mass / rho0
    shell = (r_final > 0.6 * radius)
    # The CSF force points *inward* (toward the centre), so the pressure jump
    # that balances it is minus the radial projection.
    project = -(force[:, 0] * pos[:, 0] + force[:, 1] * pos[:, 1]) \
        / r_final.clamp_min(1e-30)
    flux = float((project[shell] * rho[shell] * v_part[shell]
                  / (2.0 * math.pi * r_final[shell])).sum())

    interior = {
        "criterion": "p(interior) vs sigma/R + p_gas (p_gas = 0, vacuum)",
        "measured_pa": p_mean,
        "theory_pa": dp_theory,
        "rel_error_pct": _pct(p_mean, dp_theory),
        "within_3pct": abs(_pct(p_mean, dp_theory)) < TIGHT_PCT,
        "within_15pct": abs(_pct(p_mean, dp_theory)) < GATE_PCT,
        "note": "time-averaged over the second half; instantaneous one-sample "
                "values scatter by p_std",
    }
    legacy = {
        "criterion": "p(r<0.5R) - p(0.80R<r<0.95R), scored against sigma/R "
                     "(historical benchmarks/csf_consistent)",
        "measured_pa": p_core - p_shelf,
        "rel_error_pct_vs_sigma_over_R": _pct(p_core - p_shelf, dp_theory),
        "own_theory_pa": 0.0,
        "note": "this difference is between two interior points; its continuum "
                "value is 0 because grad p = 0 inside a static drop, so the "
                "sigma/R comparison is not a valid criterion",
    }
    jump = {
        "p_inside_pa": p_core,
        "p_outside_pa": float(pressure[r_final > 0.98 * radius].mean())
        if bool((r_final > 0.98 * radius).any()) else 0.0,
    }
    jump["jump_pa"] = jump["p_inside_pa"] - jump["p_outside_pa"]
    jump["rel_error_pct"] = _pct(jump["jump_pa"], dp_theory)

    return {
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "n_particles": int(pos.shape[0]),
            "r_over_dx": radius / dx, "sigma_N_per_m": sigma, "rho0": rho0,
            "sound_speed": sound_speed, "viscosity": viscosity, "gamma": cfg.gamma,
            "scheme": scheme, "curvature": curvature, "steps": steps,
            "density_renormalization": density_renormalization,
            "sample_every": sample_every, "dt_s": dt,
            "t_end_s": steps * dt, "t_over_t_sigma": steps * dt / t_sigma,
            "window_t_over_t_sigma": [0.5 * steps * dt / t_sigma,
                                      steps * dt / t_sigma],
            "wall_seconds": wall,
        },
        "reference": {
            "delta_p_pa": dp_theory, "v_ref_m_per_s": v_ref,
            "compression_required": dp_theory / bulk, "bulk_modulus_pa": bulk,
            "gate_pct": GATE_PCT, "tight_pct": TIGHT_PCT,
        },
        "interior": interior,
        "resolved_jump": jump,
        "legacy_measure": legacy,
        "noise": {
            "temporal_p_std_pa": p_std,
            "temporal_p_sem_pa": p_sem,
            "n_samples": n_late,
            "rho_core_over_rho0_time_mean": rho_mean,
            "rho_core_over_rho0_time_std": rho_std,
            "rho_core_over_rho0_spatial_std": rho_core_spread,
            "eos_noise_pa_from_time_std": noise_pa,
            "eos_noise_pa_from_spatial_std": noise_pa_spatial,
            "noise_over_theory": noise_pa / dp_theory,
        },
        "shell_flux": {
            "measured_pa": flux, "theory_pa": dp_theory,
            "rel_error_pct": _pct(flux, dp_theory),
            "note": "int f_r dn across the interface shell, from the CSF force "
                    "itself; independent of the pressure field",
        },
        "profile": profile,
        "history": {"t": t_hist, "p_core": p_hist,
                    "rho_core_over_rho0": rho_hist, "vmax_over_vref": vmax_hist},
    }


CASES = [
    # (dx, curvature) -- resolution ladder at fixed curvature, plus one
    # fit_quadratic case to check the audit is insensitive to the operator.
    (0.003, "mls"),
    (0.002, "mls"),
    (0.0015, "mls"),
    (0.002, "fit_quadratic"),
]


def run_all(cases=CASES, radius=0.02, sigma=7.2, rho0=1000.0,
            sound_speed=20.0, viscosity=0.01, t_end_over_t_sigma=2.8) -> dict:
    rows = []
    for dx, curvature in cases:
        row = simulate(radius=radius, dx=dx, curvature=curvature,
                       t_end_over_t_sigma=t_end_over_t_sigma)
        print(f"  dx={dx:<7} R/dx={radius / dx:5.1f} curv={curvature:<14} "
              f"p_int={row['interior']['measured_pa']:7.1f} Pa "
              f"({row['interior']['rel_error_pct']:+7.1f}%) "
              f"flux={row['shell_flux']['rel_error_pct']:+7.1f}% "
              f"[{row['setup']['wall_seconds']:.1f}s]", flush=True)
        rows.append(row)
    return {
        "benchmark": "young_laplace_pressure_audit",
        "reference": {"delta_p_pa": sigma / radius, "gate_pct": GATE_PCT,
                      "tight_pct": TIGHT_PCT},
        "definition": {
            "correct": "p(interior) - p_gas = sigma/R (p_gas = 0 here)",
            "historical": "p(r<0.5R) - p(0.80R<r<0.95R) vs sigma/R: invalid, "
                          "continuum value of the difference is 0",
        },
        "cases": rows,
    }


def make_figures(report: dict, out_dir: Path = HERE) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = report["cases"]
    ref = report["reference"]["delta_p_pa"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    ax = axes[0][0]
    for row in cases:
        prof = row["profile"]
        mid = [0.5 * (b["r_over_R"][0] + b["r_over_R"][1]) for b in prof]
        ax.plot(mid, [b["p_over_theory"] for b in prof], "o-",
                label=f"dx={row['setup']['dx_m']*1000:.1f} mm, "
                      f"{row['setup']['curvature']}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="theory $\\sigma/R$")
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("$r/R$")
    ax.set_ylabel("$p\\,/\\,(\\sigma/R)$")
    ax.set_title("(a) radial pressure profile: flat interior below 0.8R")
    ax.legend(fontsize=7)

    ax = axes[0][1]
    for row in cases:
        h = row["history"]
        ax.plot(h["t"], h["p_core"], "-",
                label=f"dx={row['setup']['dx_m']*1000:.1f} mm")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="theory $\\sigma/R$")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$p_{core}$ [Pa]")
    ax.set_title("(b) core pressure time series (one sample = one step)")
    ax.legend(fontsize=7)

    ax = axes[1][0]
    dxs = [row["setup"]["dx_m"] for row in cases]
    ax.plot(dxs, [row["interior"]["rel_error_pct"] for row in cases], "o-",
            label="p(interior) vs $\\sigma/R$")
    ax.plot(dxs, [row["shell_flux"]["rel_error_pct"] for row in cases], "s-",
            label="$\\int f_r\\,dn$ vs $\\sigma/R$ (force)")
    ax.plot(dxs, [row["legacy_measure"]["rel_error_pct_vs_sigma_over_R"]
                  for row in cases], "^--",
            label="historical $p_{core}-p_{shelf}$ vs $\\sigma/R$")
    ax.axhline(0.0, color="k", lw=1)
    ax.axhline(15.0, color="r", ls=":", lw=1)
    ax.axhline(-15.0, color="r", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("$dx$ [m]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(c) error vs resolution (bands = $\\pm15\\%$ gate)")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    for row in cases:
        prof = row["profile"]
        mid = [0.5 * (b["r_over_R"][0] + b["r_over_R"][1]) for b in prof]
        ax.plot(mid, [1.0 - b["rho_over_rho0"] for b in prof], "o-",
                label=f"dx={row['setup']['dx_m']*1000:.1f} mm")
    ax.set_xlabel("$r/R$")
    ax.set_ylabel("$1-\\rho/\\rho_0$")
    ax.set_yscale("log")
    ax.set_title("(d) free-surface density deficit pins $p$ to 0 in the shell")
    ax.legend(fontsize=7)

    fig.suptitle("Young-Laplace pressure audit of the CSF surface-tension model")
    fig.tight_layout()
    out = out_dir / "laplace_pressure_audit.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one coarse case only")
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--t-sigma", type=float, default=2.8,
                    help="run length in capillary times (ignored with --steps)")
    ap.add_argument("--curvature", default="mls")
    ap.add_argument("--renorm", action="store_true",
                    help="enable Shepard density renormalisation")
    ap.add_argument("--out", default=str(HERE / "validation_report.json"))
    args = ap.parse_args()

    if args.dx is not None:
        row = simulate(dx=args.dx, steps=args.steps, curvature=args.curvature,
                       t_end_over_t_sigma=args.t_sigma,
                       density_renormalization=args.renorm)
        Path(args.out).write_text(json.dumps(row, indent=1))
        print(json.dumps({k: row[k] for k in
                          ("interior", "resolved_jump", "legacy_measure",
                           "shell_flux")}, indent=1))
        return
    cases = [(0.003, "mls")] if args.quick else CASES
    report = run_all(cases, t_end_over_t_sigma=args.t_sigma)
    Path(args.out).write_text(json.dumps(report, indent=1))
    if not args.quick:
        make_figures(report)
    print(f"  report -> {args.out}")


if __name__ == "__main__":
    main()