"""Surface delta-function normalisation: ``2 |grad c|`` vs ``|grad c| / (1 - c_s)``.

``docs/verification_surface_tension_curvature.md`` left one systematic debt: the
delta function integrated to ``int delta_s dn = 0.74`` instead of 1 at every
resolution (0.755 / 0.751 / 0.739 at dx = 0.003 / 0.002 / 0.0015 m), and the
cutting-force identity ``F_cut = 2 sigma`` was 10.7 % low mostly because of it.
The document's hand-off said the deficit was a smoothed-out colour gradient at
the outermost layer and suggested a renormalisation.

This harness measures what the deficit actually is.  Both normalisations are
scored on the same lattice droplet, with the curvature operator held fixed:

1. ``int delta_s dn = 1``            (delta-function normalisation; Brackbill)
2. ``F_cut = 2 sigma``               (surface tension as force per unit length)
3. ``kappa = 1 / R``                 (must not move: this knob is not curvature)
4. median ``cos(force, -rhat)``      (must not move: this knob is not direction)
5. ``Delta p = sigma / R``           (relaxation, optional)

Usage::

    PYTHONPATH=src python benchmarks/csf_delta_s/run.py            # instant only
    PYTHONPATH=src python benchmarks/csf_delta_s/run.py --relax    # + time step
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields, surface_tension_acceleration
from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMES = ("factor2", "influid")


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
            viscosity=0.01, curvature="fit_quadratic", delta_s_scheme="factor2"):
    """Static (t = 0) diagnostics of one delta-function normalisation."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature=curvature, csf_delta_s=delta_s_scheme)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))
    state = csf_fields(pos, mass, wide, h, rho0, curvature_scheme=curvature,
                       delta_s_scheme=delta_s_scheme)
    acc, fstate = surface_tension_acceleration(
        pos, rho, mass, wide, h, sigma, rho0=rho0,
        curvature_scheme=curvature, delta_s_scheme=delta_s_scheme)

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    mag = torch.linalg.vector_norm(acc, dim=1)
    active = state.delta_s > 0.05 * state.delta_s.max()
    cos_in = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)
    band = active & (r / radius > 0.6) & (r / radius <= 1.02)

    # Cutting-force identity: total force transmitted across the y axis.
    cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
    theory_cut = -2.0 * sigma
    # Same identity measured on the *force density* ``f = -sigma kappa delta_s n``
    # instead of the acceleration ``f/rho``: the free-surface kernel-sum density
    # is deficient, so ``a = f/rho`` inflates the force it is supposed to carry.
    f_density = (-sigma * fstate.kappa * fstate.delta_s)[:, None] * fstate.normal
    cut_density = float((mass[pos[:, 0] > 0, None] / rho0
                         * f_density[pos[:, 0] > 0]).sum(dim=0)[0])
    # delta_s-weighted, because the force is carried by the delta_s-weighted
    # particles, not by the median particle of the band.
    wact = state.delta_s[active]
    shell_rho = (float(((rho[active] / rho0) * wact).sum() / wact.sum())
                 if bool(active.any()) and float(wact.sum()) > 0 else float("nan"))

    w = state.delta_s
    kappa_w = float((state.kappa.abs() * w).sum() / w.sum())
    kappa_exact = 1.0 / radius
    dsi = delta_s_integral(pos, mass, rho0, state.delta_s, radius)
    kappa_band = radial_band(pos, state.kappa.abs(), radius)

    # Colour-branch diagnostic: the in-fluid drop decides the normalisation.
    color = state.color
    bulk = float(color[color > 0.97].median()) if bool((color > 0.97).any()) else 1.0
    shell = color[(color < 0.97) & (color > 0.05)]
    c_surface = float(shell.median()) if shell.numel() else float("nan")

    return {
        "delta_s_scheme": delta_s_scheme,
        "curvature": curvature,
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "color": {
            "bulk_level": bulk,
            "interface_band_colour_median": c_surface,
            "surface_level_analytic": float((0.5 * (1.0 + (mass / rho0)
                                                    * _self_kernel(h))).mean()),
            "surface_level_half_lattice": half_lattice_surface_color(dx, h),
        },
        "kappa": {
            "exact_1_per_m": kappa_exact,
            "delta_s_weighted_mean_1_per_m": kappa_w,
            "rel_error_pct": (kappa_w - kappa_exact) / kappa_exact * 100.0,
            "band": kappa_band,
            "within_15pct": abs(kappa_w - kappa_exact) / kappa_exact < 0.15,
        },
        "delta_s": {
            # in-fluid branch of the colour: int_fluid |grad c| dn = 1 - c_s
            "grad_integral_dn": delta_s_integral(
                pos, mass, rho0,
                torch.linalg.vector_norm(state.grad_c, dim=1), radius),
            "surface_level_from_drop": 1.0 - delta_s_integral(
                pos, mass, rho0,
                torch.linalg.vector_norm(state.grad_c, dim=1), radius),
            "integral_dn": dsi,
            "rel_error_pct": (dsi - 1.0) * 100.0,
            "within_3pct": abs(dsi - 1.0) < 0.03,
            "within_15pct": abs(dsi - 1.0) < 0.15,
        },
        "cut_force": {
            "theory_N_per_m": theory_cut,
            "measured_N_per_m": cut,
            "ratio_measured_over_theory": cut / theory_cut,
            "rel_error_pct": abs(cut / theory_cut - 1.0) * 100.0,
            "within_3pct": abs(cut / theory_cut - 1.0) < 0.03,
            "within_15pct": abs(cut / theory_cut - 1.0) < 0.15,
        },
        "cut_force_density": {
            "theory_N_per_m": theory_cut,
            "measured_N_per_m": cut_density,
            "ratio_measured_over_theory": cut_density / theory_cut,
            "rel_error_pct": abs(cut_density / theory_cut - 1.0) * 100.0,
            "within_3pct": abs(cut_density / theory_cut - 1.0) < 0.03,
            "within_15pct": abs(cut_density / theory_cut - 1.0) < 0.15,
        },
        "density": {
            "shell_rho_over_rho0_weighted": shell_rho,
            "acc_inflation_factor": 1.0 / shell_rho if shell_rho == shell_rho else None,
        },
        "direction": {
            "n_active_band": int(band.sum()),
            "cos_band_median": float(cos_in[band].median()),
            "fraction_inward": float((cos_in[band] > 0).float().mean()),
            "cos_error_pct": abs(float(cos_in[band].median()) - 1.0) * 100.0,
        },
        "force_density": {
            "max_abs_acc": float(mag.max()),
            "n_force_particles": int((fstate.delta_s > 0).sum()),
        },
    }


def half_lattice_surface_color(dx, h, rows=40, cols=60):
    """Colour of the top row of a hex half-lattice: the model's own ``c_s``.

    ``delta_s = 2 |grad c|`` assumes this is 1/2.  It is not: the particle keeps
    its self term and also the neighbours of its own lattice row.
    """
    from tensorsph.csf_consistent import csf_fields
    dy = dx * math.sqrt(3.0) / 2.0
    pts = [((col + 0.5 * (row % 2)) * dx, row * dy)
           for row in range(-rows, 1) for col in range(-cols, cols + 1)]
    pos = torch.tensor(pts)
    vol = dx * dx * math.sqrt(3.0) / 2.0
    mass = torch.full((pos.shape[0],), vol * 1000.0)
    d = torch.cdist(pos, pos)
    i, j = torch.nonzero((d <= 2.0 * h) & (d > 0.0), as_tuple=True)
    keep = i < j
    pairs = torch.stack([i[keep], j[keep]], dim=1)
    state = csf_fields(pos, mass, pairs, h, 1000.0)
    top = pos[:, 1] > -1.0e-12
    return float(state.color[top].median())


def _self_kernel(h: float) -> float:
    from tensorsph.droplet_impact import cubic_spline_kernel
    return float(cubic_spline_kernel(torch.zeros(()), h))


def relax(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
          steps=900, viscosity=0.01, curvature="fit_quadratic",
          delta_s_scheme="factor2", output_every=50):
    """Time integration: does the Laplace jump / spurious current change?"""
    from tensorsph.droplet_impact import rhs, stable_dt, tait_pressure

    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature=curvature, csf_delta_s=delta_s_scheme)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    v_ref = math.sqrt(sigma / (rho0 * radius))
    dp_theory = sigma / radius
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))
    hist = []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if (step + 1) % output_every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            core = r < 0.5 * radius
            shell = (r > 0.80 * radius) & (r < 0.95 * radius)
            p_core = float(pressure[core].mean())
            p_shell = float(pressure[shell].mean())
            vmax = float(torch.linalg.vector_norm(vel, dim=1).max())
            hist.append({
                "t_s": (step + 1) * dt,
                "t_over_t_sigma": (step + 1) * dt / t_sigma,
                "p_core_Pa": p_core,
                "p_shell_Pa": p_shell,
                "delta_p_Pa": p_core - p_shell,
                "spurious_vmax_over_vref": vmax / v_ref,
                "rho_core_over_rho0": float(rho[core].mean() / rho0),
            })
    last = hist[-1]
    rel = (last["delta_p_Pa"] - dp_theory) / dp_theory * 100.0
    return {
        "delta_s_scheme": delta_s_scheme,
        "curvature": curvature,
        "dx_m": dx,
        "dt_s": dt,
        "steps": steps,
        "dp_theory_Pa": dp_theory,
        "delta_p_Pa": last["delta_p_Pa"],
        "delta_p_rel_error_pct": rel,
        "delta_p_within_15pct": abs(rel) < 15.0,
        "spurious_vmax_over_vref": last["spurious_vmax_over_vref"],
        "spurious_within_0p1": last["spurious_vmax_over_vref"] < 0.1,
        "history": hist,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, nargs="+", default=[0.003, 0.002, 0.0015])
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--curvature", default="fit_quadratic")
    ap.add_argument("--relax", action="store_true")
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args(argv)

    report = {
        "case": "2-D static droplet, Young-Laplace + cutting force",
        "radius_m": args.radius,
        "sigma_N_per_m": args.sigma,
        "rho0_kg_per_m3": 1000.0,
        "sound_speed_m_per_s": 20.0,
        "curvature": args.curvature,
        "dx_list_m": args.dx,
        "instant": [],
        "relax": [],
    }
    for dx in args.dx:
        for scheme in SCHEMES:
            row = instant(radius=args.radius, dx=dx, sigma=args.sigma,
                          curvature=args.curvature, delta_s_scheme=scheme)
            report["instant"].append(row)
            print(f"dx={dx:.4f} {scheme:8s} "
                  f"int_ds={row['delta_s']['integral_dn']:.4f} "
                  f"Fcut_acc={row['cut_force']['ratio_measured_over_theory']:.4f} "
                  f"Fcut_f={row['cut_force_density']['ratio_measured_over_theory']:.4f} "
                  f"rho_shell={row['density']['shell_rho_over_rho0_weighted']:.3f} "
                  f"kappa_err={row['kappa']['rel_error_pct']:+.1f}% "
                  f"cos={row['direction']['cos_band_median']:+.4f}")
    if args.relax:
        for scheme in SCHEMES:
            row = relax(radius=args.radius, dx=0.002, sigma=args.sigma,
                        steps=args.steps, curvature=args.curvature,
                        delta_s_scheme=scheme)
            report["relax"].append(row)
            print(f"relax {scheme:8s} dp={row['delta_p_Pa']} "
                  f"err={row['delta_p_rel_error_pct']:+.1f}% "
                  f"vmax={row['spurious_vmax_over_vref']:.4f}")

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2))
    for row in report["instant"]:
        tag = f"dx{row['dx_m']:.4f}_{row['delta_s_scheme']}".replace(".", "p")
        (HERE / f"{tag}.json").write_text(json.dumps(row, indent=2))
    return report


if __name__ == "__main__":
    main()