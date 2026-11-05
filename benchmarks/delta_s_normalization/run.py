"""Surface delta function A/B: ``2 |grad_mls c|`` vs ``2 |dc/dx|`` (kernel sum).

Debt being closed (handoff of ``docs/verification_surface_tension_curvature.md``
§7.1): the CSF force amplitude is ``sigma kappa delta_s n``, so a delta function
that integrates to 0.75 instead of 1 loses 25 % of the force *before* any
curvature error enters.  Measured on the lattice droplet, the historical
estimator ``delta_s = 2 |grad_mls c|`` gives

    int delta_s dn = 0.755 / 0.751 / 0.739   (dx = 3 / 2 / 1.5 mm)

i.e. a *systematic* 25 % deficit that does not shrink with resolution.  The
factor 2 itself is exact: the colour field only carries the fluid half of the
indicator.  The residual loss comes from the MLS gradient, which reconstructs a
linear field exactly but sees a 2h-wide step on the one-sided stencil of the
outermost layer.

This harness scores the replacement estimator
``delta_s = 2 |sum_j V_j grad_i W_ij|`` -- the exact derivative of the same
discrete colour field (:func:`tensorsph.csf_consistent.kernel_sum_gradient`) --
against the same analytic references used by the previous rounds:

1. ``int delta_s dn = 1``                (delta-function normalisation)
2. ``F_cut = 2 sigma``                   (surface tension = force per unit length)
3. ``kappa = 1 / R``                     (curvature, both operators)
4. median cos(force, -rhat)              (force direction)
5. spurious current, optional            (relaxation, Adami et al. 2012)

Usage::

    PYTHONPATH=src python benchmarks/delta_s_normalization/run.py
    PYTHONPATH=src python benchmarks/delta_s_normalization/run.py --instant
"""
from __future__ import annotations

import argparse
import importlib.util
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
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# (delta_s scheme, force density, curvature): the first row is what shipped
# before this round; the last row is the combination this round recommends.
COMBOS = (
    ("two_sided", "local", "fit_quadratic"),
    ("two_sided", "reference", "fit_quadratic"),
    ("kernel_sum", "local", "fit_quadratic"),
    ("kernel_sum", "reference", "fit_quadratic"),
)
DELTA_S_SCHEMES = ("two_sided", "kernel_sum")
CURVATURES = ("mls", "fit_quadratic")


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
            viscosity=0.01, curvature="fit_quadratic", delta_s="two_sided",
            force_density="local"):
    """Static (t = 0) diagnostics for one discretisation combination."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature=curvature, csf_delta_s=delta_s,
                 csf_force_density=force_density)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))

    # production path: exactly what rhs() calls
    acc = compute_surface_tension_force(pos, rho, mass, wide, cfg)
    from tensorsph.csf_consistent import surface_tension_acceleration
    acc_ref, state = surface_tension_acceleration(
        pos, rho, mass, wide, h, sigma, rho0=rho0, curvature_scheme=curvature,
        curvature_stencil=4.0, delta_s_scheme=delta_s,
        force_density=force_density)
    assert torch.allclose(acc, acc_ref), "production path disagrees with csf_fields"

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
    # Exact vector form of the same identity: for a normalised delta function
    # sum_{x>0} delta_s n_x dV = integral over the half circle n_x ds = 2 R.
    half = pos[:, 0] > 0
    vol = mass / rho0
    s_vec = float((state.delta_s[half] * state.normal[half, 0] * vol[half]).sum())
    s_vec_ratio = s_vec / (2.0 * radius)
    shell = (r / radius > 0.85) & (r / radius <= 1.02)
    rho_shell = float(rho[shell].mean() / rho0)
    kappa_band = radial_band(pos, state.kappa.abs(), radius)
    return {
        "curvature": curvature,
        "delta_s_scheme": delta_s,
        "force_density": force_density,
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
            "within_3pct": abs(dsi - 1.0) < 0.03,
            "within_15pct": abs(dsi - 1.0) < 0.15,
            "vector_identity": s_vec_ratio,
            "vector_rel_error_pct": (s_vec_ratio - 1.0) * 100.0,
            "vector_within_3pct": abs(s_vec_ratio - 1.0) < 0.03,
            "vector_within_15pct": abs(s_vec_ratio - 1.0) < 0.15,
        },
        "interface_density_ratio": rho_shell,
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


def relax(radius=0.02, dx=0.004, sigma=7.2, rho0=1000.0, sound_speed=20.0,
          steps=380, viscosity=0.01, curvature="fit_quadratic",
          delta_s="two_sided", force_density="local", output_every=40):
    """One capillary time of free relaxation: spurious currents and the jump."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature=curvature, csf_delta_s=delta_s,
                 csf_force_density=force_density)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    v_ref = math.sqrt(sigma / (rho0 * radius))
    dp_theory = sigma / radius

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))
    pressure = tait_pressure(rho, cfg)
    hist = {"t": [], "vmax_over_vref": [], "ke": [], "p_core": []}

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
            hist["ke"].append(float(0.5 * (mass * (vel ** 2).sum(dim=1)).sum()))

    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * radius
    shell = (r > 0.80 * radius) & (r < 0.95 * radius)
    p_core = float(pressure[core].mean())
    p_shell = float(pressure[shell].mean())
    dp = p_core - p_shell
    rel = (dp - dp_theory) / dp_theory * 100.0
    return {
        "curvature": curvature,
        "delta_s_scheme": delta_s,
        "force_density": force_density,
        "setup": {"radius_m": radius, "dx_m": dx, "h_m": h,
                  "sigma_N_per_m": sigma, "rho0": rho0,
                  "sound_speed": sound_speed, "steps": steps, "dt_s": dt,
                  "t_end_s": steps * dt,
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
        "history": hist,
    }


def make_figures(payload, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["instant"]
    colors = {"two_sided": "tab:red", "kernel_sum": "tab:blue"}
    styles = {"local": "o--", "reference": "o-"}

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    ax = axes[0]
    for s in DELTA_S_SCHEMES:
        sub = sorted([r for r in rows if r["delta_s_scheme"] == s],
                     key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sub],
                [r["delta_s"]["vector_identity"] for r in sub], "o-",
                color=colors[s], label=f"delta_s = {s}")
    ax.axhline(1.0, color="k", ls=":", label=r"$\sum_{x>0}\delta_s n_x dV = 2R$")
    ax.axhspan(0.85, 1.15, color="k", alpha=0.07, label="15 % gate")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\sum_{x>0}\delta_s n_x\,dV / 2R$")
    ax.set_title("(a) delta-function normalisation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for s in DELTA_S_SCHEMES:
        for d in ("local", "reference"):
            sub = sorted([r for r in rows if r["delta_s_scheme"] == s
                          and r["force_density"] == d], key=lambda r: r["dx_m"])
            ax.plot([r["dx_m"] for r in sub],
                    [r["cut_force"]["ratio_measured_over_theory"] for r in sub],
                    styles[d], color=colors[s], ms=4, label=f"{s} / {d}")
    ax.axhline(1.0, color="k", ls=":", label=r"$F_{cut} = 2\sigma$")
    ax.axhspan(0.85, 1.15, color="k", alpha=0.07)
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$F_{cut}/(2\sigma)$")
    ax.set_title("(b) cut-force identity")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for s in DELTA_S_SCHEMES:
        sub = sorted([r for r in rows if r["delta_s_scheme"] == s
                      and r["force_density"] == "reference"],
                     key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sub],
                [r["kappa"]["rel_error_pct"] for r in sub], "o-",
                color=colors[s], label=f"fit_quadratic / {s}")
    ax.axhline(0.0, color="k", ls=":")
    ax.axhspan(-15.0, 15.0, color="k", alpha=0.07, label="15 % gate")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\kappa$ error [%]")
    ax.set_title(r"(c) curvature $\kappa = 1/R$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    relax = payload.get("relax", [])
    if relax:
        for r in relax:
            ax.plot(r["history"]["t"], r["history"]["vmax_over_vref"], "-",
                    ms=3, color=colors[r["delta_s_scheme"]],
                    ls=("-" if r["force_density"] == "reference" else "--"),
                    label=f"{r['delta_s_scheme']} / {r['force_density']}")
        ax.axhline(0.1, color="r", ls="--", label="Adami threshold 0.1")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$|v|_{max}/\sqrt{\sigma/\rho R}$")
        ax.set_title("(d) spurious current (relaxation)")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3, which="both")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "relaxation not run (--instant)", ha="center")

    try:
        fig.tight_layout()
    except ValueError:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.12,
                            wspace=0.28)
    path = out_dir / "delta_s_normalization.png"
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--dx-list", default="0.003,0.002,0.0015")
    ap.add_argument("--relax-dx", type=float, default=0.004)
    ap.add_argument("--steps", type=int, default=380)
    ap.add_argument("--instant", action="store_true", help="skip the relaxation")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dxs = [float(x) for x in a.dx_list.split(",")]

    rows = []
    for delta_s, force_density, curvature in COMBOS:
        for dx in dxs:
            row = instant(a.radius, dx, a.sigma, curvature=curvature,
                          delta_s=delta_s, force_density=force_density)
            rows.append(row)
            print(f"{delta_s:10s} {force_density:9s} dx={dx:.4f} "
                  f"N={row['n_particles']:5d} "
                  f"vec={row['delta_s']['vector_identity']:6.3f} "
                  f"int={row['delta_s']['integral_dn']:6.3f} "
                  f"Fcut={row['cut_force']['ratio_measured_over_theory']:6.3f} "
                  f"kappa_err={row['kappa']['rel_error_pct']:+6.1f}% "
                  f"cos={row['direction']['cos_band_median']:+.3f} "
                  f"rho_shell={row['interface_density_ratio']:.2f}")

    relax_rows = []
    if not a.instant:
        for delta_s, force_density, curvature in COMBOS:
            row = relax(a.radius, a.relax_dx, a.sigma, steps=a.steps,
                        curvature=curvature, delta_s=delta_s,
                        force_density=force_density)
            relax_rows.append(row)
            print(f"relax {delta_s:10s} {force_density:9s} dx={a.relax_dx:.4f} "
                  f"dp={row['laplace']['delta_p_measured_Pa']:.1f} Pa "
                  f"({row['laplace']['rel_error_pct']:+.1f}%) "
                  f"vmax/vref={row['spurious_current']['vmax_over_vref_final']:.3f}")

    payload = {
        "case": "2-D static droplet: delta_s normalisation, cut-force identity, "
                "CSF force density",
        "setup": {"radius_m": a.radius, "sigma_N_per_m": a.sigma,
                  "dx_list_m": dxs, "rho0": 1000.0, "sound_speed": 20.0,
                  "gamma": 7.0, "h_over_dx": 1.3, "gravity": 0.0,
                  "combos": [{"delta_s_scheme": c[0],
                              "force_density": c[1],
                              "curvature": c[2]} for c in COMBOS]},
        "reference": {"kappa_1_per_m": 1.0 / a.radius,
                      "delta_s_integral_dn": 1.0,
                      "delta_s_vector_identity_over_2R": 1.0,
                      "cut_force_N_per_m": -2.0 * a.sigma,
                      "delta_p_Pa": a.sigma / a.radius},
        "instant": rows,
        "relax": relax_rows,
        "gates": {"tolerance_15pct": "platform standard",
                  "note": "within_3pct is the stricter research bar"},
    }
    (out / "validation_report.json").write_text(json.dumps(payload, indent=2))
    make_figures(payload, out)
    print(f"report -> {out / 'validation_report.json'}")


if __name__ == "__main__":
    main()
