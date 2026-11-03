"""Curvature operator A/B: ``div n`` (MLS) vs weighted local interface fit.

``docs/verification_surface_tension_consistent.md`` closed the CSF amplitude and
direction debt but left the curvature outside the 15 % gate: ``kappa = div n``
read +25 % on a circular drop.  This harness scores the replacement operator from
``tensorsph.csf_curvature`` against the same analytic references, on the same
lattice droplet, with everything else held fixed:

1. ``kappa = 1 / R``                    (CSF curvature on a circle)
2. ``F_cut = 2 sigma``                  (surface tension = force per unit length)
3. ``int delta_s dn = 1``               (delta-function normalisation)
4. median cos(force, -rhat)             (force direction)
5. ``Delta p = sigma / R`` + spurious currents (relaxation, optional)

Usage::

    PYTHONPATH=src python benchmarks/csf_curvature_fit/run.py            # full
    PYTHONPATH=src python benchmarks/csf_curvature_fit/run.py --instant  # no relax
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields, surface_tension_acceleration
from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMES = ("mls", "fit_quadratic", "fit_circle")


def _load_sibling(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
_consistent = _load_sibling("csf_consistent_run", "csf_consistent/run.py")
hexagonal_droplet = _law.hexagonal_droplet
radial_band = _law.radial_band
delta_s_integral = _consistent.delta_s_integral


def instant(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
            viscosity=0.01, scheme="consistent", curvature="mls"):
    """Static (t = 0) diagnostics of one curvature operator."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme=scheme,
                 csf_curvature=curvature)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))
    state = csf_fields(pos, mass, wide, h, rho0, curvature_scheme=curvature)
    acc, _ = surface_tension_acceleration(pos, rho, mass, wide, h, sigma,
                                          rho0=rho0, curvature_scheme=curvature)

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    mag = torch.linalg.vector_norm(acc, dim=1)
    active = state.delta_s > 0.05 * state.delta_s.max()
    cos_in = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)
    band = active & (r / radius > 0.6) & (r / radius <= 1.02)

    cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
    theory_cut = -2.0 * sigma
    w = state.delta_s
    kappa_w = float((state.kappa.abs() * w).sum() / w.sum())
    kappa_exact = 1.0 / radius
    dsi = delta_s_integral(pos, mass, rho0, state.delta_s, radius)

    kappa_band = radial_band(pos, state.kappa.abs(), radius)
    return {
        "csf_scheme": scheme,
        "curvature": curvature,
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "kappa": {
            "exact_1_per_m": kappa_exact,
            "delta_s_weighted_mean_1_per_m": kappa_w,
            "rel_error_pct": (kappa_w - kappa_exact) / kappa_exact * 100.0,
            "band": kappa_band,
            "band_rel_error_pct": (kappa_band["mean"] - kappa_exact)
            / kappa_exact * 100.0,
            "within_3pct": abs(kappa_w - kappa_exact) / kappa_exact < 0.03,
            "within_15pct": abs(kappa_w - kappa_exact) / kappa_exact < 0.15,
        },
        "delta_s": {
            "integral_dn": dsi,
            "rel_error_pct": (dsi - 1.0) * 100.0,
            "within_15pct": abs(dsi - 1.0) < 0.15,
        },
        "cut_force": {
            "theory_N_per_m": theory_cut,
            "measured_N_per_m": cut,
            "ratio_measured_over_theory": cut / theory_cut,
            "rel_error_pct": abs(cut / theory_cut - 1.0) * 100.0,
            "within_15pct": abs(cut / theory_cut - 1.0) < 0.15,
        },
        "direction": {
            "n_active_band": int(band.sum()),
            "cos_band_median": float(cos_in[band].median()),
            "fraction_inward": float((cos_in[band] > 0).float().mean()),
        },
    }


def relax(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
          steps=900, viscosity=0.01, scheme="consistent", curvature="mls",
          output_every=50):
    """Time integration: does the Laplace jump / spurious current change?"""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme=scheme,
                 csf_curvature=curvature)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    v_ref = math.sqrt(sigma / (rho0 * radius))
    dp_theory = sigma / radius
    comp_needed = dp_theory / (cfg.sound_speed ** 2 * rho0 / cfg.gamma)

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))
    pressure = tait_pressure(rho, cfg)
    hist = {"t": [], "vmax_over_vref": [], "ke": []}

    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % output_every == 0 or step == steps - 1:
            hist["t"].append((step + 1) * dt)
            hist["vmax_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)
            hist["ke"].append(float(
                0.5 * (mass * (vel ** 2).sum(dim=1)).sum()))

    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * radius
    shell = (r > 0.80 * radius) & (r < 0.95 * radius)
    p_core = float(pressure[core].mean())
    p_shell = float(pressure[shell].mean())
    dp = p_core - p_shell
    rel = (dp - dp_theory) / dp_theory * 100.0
    return {
        "csf_scheme": scheme,
        "curvature": curvature,
        "setup": {"radius_m": radius, "dx_m": dx, "h_m": h, "sigma_N_per_m": sigma,
                  "rho0": rho0, "sound_speed": sound_speed, "steps": steps,
                  "dt_s": dt, "t_end_s": steps * dt,
                  "t_over_t_sigma": steps * dt
                  / math.sqrt(rho0 * radius ** 3 / sigma)},
        "reference": {"delta_p_Pa": dp_theory},
        "laplace": {"p_center_Pa": p_core, "p_interface_band_Pa": p_shell,
                    "delta_p_measured_Pa": dp, "rel_error_pct": rel,
                    "within_3pct": abs(rel) < 3.0,
                    "within_15pct": abs(rel) < 15.0},
        "spurious_current": {"v_ref_m_per_s": v_ref,
                             "vmax_over_vref_final": hist["vmax_over_vref"][-1],
                             "passes_adami_0p1": hist["vmax_over_vref"][-1] < 0.1},
        "density": {"rho_core_over_rho0": float(rho[core].mean() / rho0),
                    "compression_required_by_law": comp_needed},
        "history": hist,
    }


def make_figures(payload, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["instant"]
    radius = payload["setup"]["radius_m"]
    relax = payload.get("relax", [])

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    ax = axes[0]
    for c in SCHEMES:
        sub = [r for r in rows if r["curvature"] == c and r["csf_scheme"] == "consistent"]
        sub.sort(key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sub],
                [r["kappa"]["delta_s_weighted_mean_1_per_m"] for r in sub], "o-",
                label=c)
    ax.axhline(1.0 / radius, color="k", ls=":", label=r"analytic $\kappa=1/R$")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\kappa$ [1/m]")
    ax.set_title("(a) curvature vs resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for c in SCHEMES:
        sub = [r for r in rows if r["curvature"] == c and r["csf_scheme"] == "consistent"]
        sub.sort(key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sub],
                [r["kappa"]["rel_error_pct"] for r in sub], "o-", label=c)
    ax.axhline(15.0, color="r", ls="--", label="15% gate")
    ax.axhline(0.0, color="k", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\kappa$ error [%]")
    ax.set_title("(b) curvature error vs gate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels, vals = [], []
    for c in SCHEMES:
        sub = [r for r in rows if r["curvature"] == c and r["csf_scheme"] == "consistent"]
        for r in sub:
            labels.append(f"{c}\n{r['dx_m']:.4f}")
            vals.append(r["cut_force"]["ratio_measured_over_theory"])
    ax.bar(range(len(vals)), vals, color="steelblue")
    ax.axhline(1.0, color="k", ls="--", label=r"$F_{cut}=2\sigma$")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6, rotation=60)
    ax.set_title(r"(c) cut-force identity $F_{cut}/(2\sigma)$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[3]
    if relax:
        for r in relax:
            ax.plot(r["history"]["t"], r["history"]["vmax_over_vref"], "o-", ms=3,
                    label=r["curvature"])
        ax.axhline(0.1, color="r", ls="--", label="Adami threshold 0.1")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$|v|_{max}/\sqrt{\sigma/\rho R}$")
        ax.set_title("(d) spurious current (relaxation)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "relaxation not run (--instant)", ha="center")

    # tight_layout() queries the tick locators of every axis, which fails on an
    # empty log axis; the placeholder panel makes that unnecessary.
    try:
        fig.tight_layout()
    except ValueError:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.12,
                            wspace=0.28)
    path = out_dir / "csf_curvature_fit.png"
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--dx-list", default="0.003,0.002,0.0015")
    ap.add_argument("--instant", action="store_true", help="skip the relaxation")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dxs = [float(x) for x in a.dx_list.split(",")]

    rows = []
    for curvature in SCHEMES:
        for dx in dxs:
            row = instant(a.radius, dx, a.sigma, curvature=curvature)
            rows.append(row)
            print(f"{curvature:14s} dx={dx:.4f} N={row['n_particles']:5d} "
                  f"kappa={row['kappa']['delta_s_weighted_mean_1_per_m']:7.2f} "
                  f"({row['kappa']['rel_error_pct']:+6.1f}%) "
                  f"Fcut/2s={row['cut_force']['ratio_measured_over_theory']:+6.3f} "
                  f"int_ds={row['delta_s']['integral_dn']:.3f} "
                  f"cos={row['direction']['cos_band_median']:+.3f}")

    payload = {
        "benchmark": "csf_curvature_operator_ab",
        "reference": {
            "curvature_1_per_m": 1.0 / a.radius,
            "cut_force_N_per_m": -2.0 * a.sigma,
            "delta_s_integral": 1.0,
            "laplace_delta_p_Pa": a.sigma / a.radius,
            "gate_pct": 15.0,
        },
        "setup": {"radius_m": a.radius, "sigma_N_per_m": a.sigma,
                  "rho0": 1000.0, "sound_speed": 20.0, "h_factor": 1.3,
                  "dx_list": dxs},
        "instant": rows,
    }

    if not a.instant:
        relax_rows = []
        for curvature in SCHEMES:
            rep = relax(a.radius, 0.002, a.sigma, steps=a.steps,
                        curvature=curvature)
            relax_rows.append(rep)
            print(f"relax {curvature:14s} dp={rep['laplace']['delta_p_measured_Pa']:7.2f} "
                  f"Pa ({rep['laplace']['rel_error_pct']:+6.1f}%) "
                  f"spurious={rep['spurious_current']['vmax_over_vref_final']:.3f}")
        payload["relax"] = relax_rows

    (out / "validation_report.json").write_text(json.dumps(payload, indent=2))
    make_figures(payload, out)


if __name__ == "__main__":
    main()
