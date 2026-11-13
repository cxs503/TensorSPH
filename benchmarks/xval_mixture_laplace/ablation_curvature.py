"""Ablation: which curvature operator to use on the mixture colour band.

The mixture colour of :mod:`tensorsph.csf_mixture` resolves the interface over a
~2h shell, and the normal is a first-order-consistent (MLS) gradient of that
field. Three curvature operators are compared on the identical colour field, at
``R/dx = 16``, with everything else held fixed:

``div n``            ``kappa = div n`` with the MLS divergence (library default)
``fit_circle``       local circle fit to the interface (``csf_curvature``)
``fit_quadratic``    local paraboloid fit (``csf_curvature``)

for the fit operators the stencil radius (in ``h``) is scanned. Written to
``ablation_curvature.json``; consumed by :mod:`make_doc`.

Run::

    PYTHONPATH=src python benchmarks/xval_mixture_laplace/ablation_curvature.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorsph.csf_curvature import fit_curvature
from tensorsph.csf_mixture import (
    MixtureMaterial, MixtureMaterialSet, mixture_csf_fields, particle_volume,
)
from tensorsph.neighbor_search import cell_linked_pairs

RADIUS = 0.04
SIGMA = 0.072
H_FACTOR = 1.3

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)


def build(dx: float, rho_in: float = 1000.0, rho_out: float = 1000.0,
          outer: float = 0.04):
    extent = 2.0 * (RADIUS + outer) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    keep = r <= RADIUS + outer
    pos, r = pos[keep], r[keep]
    phase = (r >= RADIUS).to(torch.long)
    rho = torch.where(phase == 0, torch.tensor(rho_in), torch.tensor(rho_out))
    mats = MixtureMaterialSet((MixtureMaterial("A", rho_in),
                               MixtureMaterial("B", rho_out)), {(0, 1): SIGMA})
    return pos, rho * dx * dx, phase, mats


def main() -> None:
    dx = RADIUS / 16.0
    h = H_FACTOR * dx
    pos, mass, phase, mats = build(dx)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    wide = cell_linked_pairs(pos, 4.0 * h)
    vol = particle_volume(mass, phase, mats)
    state = mixture_csf_fields(pos, vol, phase, pairs, h, SIGMA, 0,
                               curvature_scheme="mls")
    shell = state.delta_s > 0.10 * float(state.delta_s.max())

    def row(method: str, stencil: float | None, kappa: torch.Tensor) -> dict:
        f = (-SIGMA * kappa * state.delta_s)[:, None] * state.normal
        half = pos[:, 0] > 0.0
        resultant = float(torch.linalg.vector_norm(
            (f[half] * vol[half, None]).sum(dim=0)))
        return {
            "method": method,
            "stencil_h": stencil,
            "kappa_shell_mean": float(kappa[shell].mean()),
            "kappa_error_pct": (float(kappa[shell].mean()) * RADIUS - 1.0) * 100.0,
            "sigma_eff_over_sigma": resultant / (2.0 * SIGMA),
        }

    rows = [row("div n (mls)", None, state.kappa)]
    for method in ("fit_circle", "fit_quadratic"):
        for stencil in (2.0, 3.0, 4.0):
            kappa = fit_curvature(pos, state.normal, state.delta_s, wide, h,
                                  fallback=state.kappa, method=method,
                                  stencil=stencil)
            rows.append(row(method, stencil, kappa))

    out = {
        "dx_m": dx, "R_over_dx": RADIUS / dx, "R_over_h": RADIUS / h,
        "n_particles": int(pos.shape[0]),
        "kappa_theory": 1.0 / RADIUS,
        "rows": rows,
    }
    path = Path(__file__).resolve().parent / "ablation_curvature.json"
    path.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['method']:14s} st={r['stencil_h']}  kerr={r['kappa_error_pct']:7.1f}%  "
              f"sigma_eff/sigma={r['sigma_eff_over_sigma']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()