"""Effective surface tension of the CSF model on a static circular droplet.

Verification debt (m3188): the pin in
``tests/test_laplace_pressure.py::test_effective_surface_tension_at_the_documented_resolution``
was set at 0.79 +/- 0.06 when the default curvature operator was ``mls``.
Mission m2979 changed the default to ``fit_quadratic`` (curvature error
+25 % -> < 6 %) and the measured ratio moved to ~0.71, leaving the pin red.

Analytic reference
------------------
For a circular droplet of radius R the CSF traction is
``f = -sigma * kappa * delta_s * n`` with ``kappa = 1/R`` and outward normal
``n``.  Integrated over the right half of the droplet (per unit depth)

    F_x = sigma * Int (1/R) * n_x dA = sigma * Int_{-pi/2}^{pi/2} cos t dt
        = 2 sigma,

so the ratio ``|F_x| / (2 sigma)`` must tend to **1.0** as ``h -> 0`` for a
consistent discretisation.  Anything that plateaus below 1.0 is a systematic
operator bias, not a resolution effect.

This benchmark is a *static* force evaluation (no time integration): it is the
cheapest possible probe of the surface-tension operator and is the quantity
the Young-Laplace pressure balance ultimately depends on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorsph import Config
from tensorsph.csf_consistent import csf_fields
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

import importlib.util

_BENCH = Path(__file__).resolve().parents[1] / "laplace_pressure" / "run.py"
_spec = importlib.util.spec_from_file_location("laplace_benchmark", _BENCH)
laplace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(laplace)

OUT = Path(__file__).resolve().parent / "out"

SIGMA = 0.072
RADIUS = 0.04
RHO0 = 1000.0
SOUND_SPEED = 20.0
H_FACTOR = 1.3  # Config default


def effective_sigma_case(dx: float, curvature: str, radius: float = RADIUS,
                         sigma: float = SIGMA) -> dict:
    """One static evaluation of the CSF force on a circular droplet."""
    cfg = Config(dx=dx, rho0=RHO0, sound_speed=SOUND_SPEED, gravity=0.0,
                 surface_tension_coefficient=sigma, csf_curvature=curvature)
    h = cfg.h_factor * cfg.dx
    pos, mass, density = laplace.circular_droplet(radius, dx)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx)
    t0 = time.time()
    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    wall = time.time() - t0

    force = mass[:, None] * acc
    right = pos[:, 0] > 0.0
    f_right = force[right].sum(dim=0)
    ratio = float(torch.linalg.vector_norm(f_right) / (2.0 * sigma))

    # Diagnostics on the interface band (delta_s above the flat-field mask),
    # re-derived through the same public path the force used (the fit
    # operators widen the stencil to 4h, mirroring surface_tension_acceleration).
    diag_pairs = (cell_linked_pairs(pos, 4.0 * h)
                  if curvature != "mls" else pairs)
    state = csf_fields(pos, mass, diag_pairs, h, RHO0,
                        curvature_scheme=curvature, curvature_stencil=4.0)
    d_s = state.delta_s
    iface = d_s > 1.0e-3 * d_s.max()
    kappa_iface = float(state.kappa[iface].mean()) if iface.any() else 0.0
    ds_iface = float(d_s[iface].mean()) if iface.any() else 0.0
    # Continuum identity for a circle: Int kappa*delta_s dV -> (1/R) * pi R^2
    # = pi R over the whole droplet (delta_s dV -> dA).
    integral = float((state.kappa * d_s * (mass / density)).sum())
    return {
        "dx": dx,
        "curvature": curvature,
        "R_over_h": radius / h,
        "R_over_dx": radius / dx,
        "n_particles": int(pos.shape[0]),
        "ratio": ratio,
        "force_right_half_N": float(torch.linalg.vector_norm(f_right)),
        "target_force_N": 2.0 * sigma,
        "kappa_iface_mean": kappa_iface,
        "delta_s_iface_mean": ds_iface,
        "kappa_delta_s_integral": integral,
        "kappa_delta_s_target": 3.141592653589793 * radius,
        "wall_time_s": wall,
    }


def run_ladder(dx_list, curvatures=("fit_quadratic", "mls"),
               radius: float = RADIUS, sigma: float = SIGMA) -> dict:
    rows = []
    for curvature in curvatures:
        for dx in dx_list:
            rep = effective_sigma_case(dx, curvature, radius, sigma)
            rows.append(rep)
            print(f"{curvature:14s} dx={dx*1e3:7.3f} mm  R/h={rep['R_over_h']:6.2f}  "
                  f"N={rep['n_particles']:6d}  ratio={rep['ratio']:.4f}  "
                  f"({rep['wall_time_s']:.1f}s)")
    return {"setup": {"radius_m": radius, "sigma": sigma, "rho0": RHO0,
                      "sound_speed": SOUND_SPEED, "h_factor": H_FACTOR,
                      "curvatures": list(curvatures)},
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-list", type=float, nargs="+",
                    default=[0.005, 0.0025, 0.00125, 0.000625])
    ap.add_argument("--curvatures", nargs="+",
                    default=["fit_quadratic", "mls"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = run_ladder(a.dx_list, a.curvatures)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
