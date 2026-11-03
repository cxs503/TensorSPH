"""Verification of the CSF delta-function closure: ``delta_s = 2 |grad c|`` vs ghost.

``docs/verification_surface_tension_curvature.md`` §7.1 left the delta-function
normalisation as the next debt: ``int delta_s dn = 0.74-0.76`` instead of 1, so
even with an exact curvature the CSF force was ~25 % too weak.  This harness
scores the two closures against the analytic references on the same lattice
droplet:

1. ``int delta_s dn = 1``                       (delta-function normalisation)
2. ``F_cut = 2 sigma``, integrated with the rest volume ``V = m / rho0``
   (the volume convention the per-volume delta function is defined with)
3. ``F_cut = 2 sigma``, integrated with the summation density (the shipped
   acceleration path) -- kept separate because it also measures the
   free-surface density deficit, not only delta_s
4. ``kappa = 1 / R`` and the force direction (unchanged, sanity)
5. relaxation: spurious currents and the Laplace jump (does the stronger force
   destabilise the drop?)

Usage::

    PYTHONPATH=src python benchmarks/csf_delta_s/run.py               # everything
    PYTHONPATH=src python benchmarks/csf_delta_s/run.py --no-relax    # static only
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
SCHEMES = ("gradient", "ghost")
CURVATURES = ("mls", "fit_quadratic")
DXS = (0.003, 0.002, 0.0015)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load("laplace_law_run", "benchmarks/laplace_law/run.py")
_consistent = _load("csf_consistent_run", "benchmarks/csf_consistent/run.py")
_fit = _load("csf_curvature_fit_run", "benchmarks/csf_curvature_fit/run.py")
hexagonal_droplet = _law.hexagonal_droplet
delta_s_integral = _consistent.delta_s_integral


def instant(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
            viscosity=0.01, curvature="fit_quadratic", delta_s="ghost"):
    """Static diagnostics for one (curvature, delta_s) combination."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature=curvature, csf_delta_s=delta_s)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    wide = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0 * h))
    rho_ref = torch.full_like(rho, rho0)

    state = csf_fields(pos, mass, wide, h, rho0, curvature_scheme=curvature,
                       delta_s_scheme=delta_s)
    acc, _ = surface_tension_acceleration(pos, rho, mass, wide, h, sigma,
                                          rho0=rho0, curvature_scheme=curvature,
                                          delta_s_scheme=delta_s)
    acc_ref, _ = surface_tension_acceleration(pos, rho_ref, mass, wide, h, sigma,
                                              rho0=rho0, curvature_scheme=curvature,
                                              delta_s_scheme=delta_s)

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    mag = torch.linalg.vector_norm(acc, dim=1)
    active = state.delta_s > 0.05 * float(state.delta_s.max())
    band = active & (r / radius > 0.6) & (r / radius <= 1.02)
    cos_in = -(acc * rhat).sum(dim=1) / mag.clamp_min(1e-30)

    right = pos[:, 0] > 0
    cut_ship = float((mass[right, None] * acc[right]).sum(dim=0)[0])
    cut_ref = float((mass[right, None] * acc_ref[right]).sum(dim=0)[0])
    theory = -2.0 * sigma
    dsi = delta_s_integral(pos, mass, rho0, state.delta_s, radius)

    # How much of the interface shell has lost kernel support (colour deficit)?
    colour = state.color
    shell = active & (r / radius > 0.9)
    return {
        "curvature": curvature,
        "delta_s_scheme": delta_s,
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "kappa": {
            "exact_1_per_m": 1.0 / radius,
            "delta_s_weighted_mean_1_per_m": float(
                (state.kappa.abs() * state.delta_s).sum() / state.delta_s.sum()),
        },
        "delta_s": {
            "integral_dn": dsi,
            "rel_error_pct": (dsi - 1.0) * 100.0,
            "within_15pct": abs(dsi - 1.0) < 0.15,
            "max_over_source": float(
                (2.0 * torch.linalg.vector_norm(state.grad_c, dim=1)).max()),
        },
        "cut_force_rest_volume": {
            "theory_N_per_m": theory,
            "measured_N_per_m": cut_ref,
            "ratio": cut_ref / theory,
            "rel_error_pct": abs(cut_ref / theory - 1.0) * 100.0,
            "within_15pct": abs(cut_ref / theory - 1.0) < 0.15,
        },
        "cut_force_summation_density": {
            "theory_N_per_m": theory,
            "measured_N_per_m": cut_ship,
            "ratio": cut_ship / theory,
            "rel_error_pct": abs(cut_ship / theory - 1.0) * 100.0,
            "within_15pct": abs(cut_ship / theory - 1.0) < 0.15,
        },
        "colour_deficit": {
            "mean_colour_outer_shell": float(colour[shell].mean()) if bool(shell.any()) else float("nan"),
            "mean_rho_outer_shell_over_rho0": float(rho[shell].mean() / rho0) if bool(shell.any()) else float("nan"),
        },
        "direction": {
            "cos_band_median": float(cos_in[band].median()),
            "fraction_inward": float((cos_in[band] > 0).float().mean()),
            "n_band": int(band.sum()),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--out", default=str(HERE / "validation_report.json"))
    args = ap.parse_args(argv)

    rows = [instant(dx=dx, curvature=cv, delta_s=ds)
            for ds in SCHEMES for cv in CURVATURES for dx in DXS]
    for r in rows:
        print(f"{r['delta_s_scheme']:8s} {r['curvature']:14s} dx={r['dx_m']:.4f} "
              f"N={r['n_particles']:4d} int_ds={r['delta_s']['integral_dn']:.4f} "
              f"({r['delta_s']['rel_error_pct']:+6.2f}%) "
              f"cut_ref={r['cut_force_rest_volume']['ratio']:+.3f} "
              f"cut_ship={r['cut_force_summation_density']['ratio']:+.3f} "
              f"c_shell={r['colour_deficit']['mean_colour_outer_shell']:.3f} "
              f"rho_shell={r['colour_deficit']['mean_rho_outer_shell_over_rho0']:.3f} "
              f"cos={r['direction']['cos_band_median']:+.3f}")

    payload = {
        "case": {
            "radius_m": 0.02,
            "sigma_N_per_m": 7.2,
            "rho0_kg_per_m3": 1000.0,
            "sound_speed_m_per_s": 20.0,
            "h_factor": 1.3,
            "gravity": 0.0,
            "dx_m": list(DXS),
            "references": {"delta_p_Pa": 7.2 / 0.02, "kappa_1_per_m": 50.0,
                           "cut_force_N_per_m": 14.4, "int_delta_s_dn": 1.0},
        },
        "instant": rows,
    }
    if not args.no_relax:
        relax = []
        for ds in SCHEMES:
            rep = _fit.relax(dx=0.002, curvature="fit_quadratic", delta_s=ds)
            rep["delta_s_scheme"] = ds
            relax.append(rep)
            print(f"relax {ds:8s} dp={rep['laplace']['delta_p_measured_Pa']:8.2f} Pa "
                  f"({rep['laplace']['rel_error_pct']:+7.2f}%) "
                  f"vmax/vref={rep['spurious_current']['vmax_over_vref_final']:.4f}")
        payload["relax"] = relax
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return payload


if __name__ == "__main__":
    main()