"""Matched comparison of the two CSF chains on the same two geometries.

Question
--------
The repository carries **two** colour-gradient / curvature chains:

* ``tensorsph.csf_consistent`` -- ``mls_gradient`` + ``fit_curvature`` (this is
  what ``csf_fields`` ships, and hence what ``surface_tension_dispersion`` and
  the 3-D CSF module consume);
* ``tensorsph.surface_tension`` -- ``compute_color_gradient`` +
  ``compute_curvature`` (the pair-wise antisymmetric SPH form; what
  ``droplet_impact.compute_surface_tension_force`` uses).

Their two Laplace benches disagree by a factor of seven on the *same* physics:

* ``benchmarks/surface_tension_laplace`` (chain B) : 4.4-4.6 % error,
  resolved colour jump ``int |grad c| dr`` = 0.985;
* ``benchmarks/two_phase_laplace``     (chain A) : 33.7-37.5 % error,
  ``int |grad c| dV`` = 0.70 x 2 pi R.

``docs/verification_two_phase_laplace.md`` and the "decisive control" entry in
``build_ledger.py`` read those two as one story: "the CSF operators are correct
in principle; the ~29 % deficit of the one-sided droplet is free-surface
truncation".  That story is only admissible if the deficit is a property of the
*geometry*.  This script measures the 2 x 2 matrix (geometry x chain) with one
and the same metric, so the deficit can be attributed instead of argued.

Metric (identical for every cell)
---------------------------------
``sigma_eff/sigma`` = ``|sum_{x>0} sigma kappa |grad c| n dV| / (2 sigma)``:
the exact surface-tension resultant on the right half of a circular interface
of either geometry is ``2 sigma``.  ``d_s_mass`` = ``sum |grad c| dV / (2 pi R)``
is the fraction of the unit mass the Brackbill delta function must carry.

Run::

    python benchmarks/two_phase_laplace/matched_chains.py [--dx 0.0025]

Writes ``benchmarks/two_phase_laplace/matched_chains.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import kernel_sum_color, mls_divergence, mls_gradient
from tensorsph.csf_curvature import fit_curvature
from tensorsph.surface_tension import (
    compute_color_gradient, compute_curvature, compute_normal,
)

RADIUS = 0.04  # overwritten by --radius
SIGMA = 0.072
RHO0 = 1000.0
H_FACTOR = 1.3


def build_cloud(dx: float, two_phase: bool, outer_shell: float = 0.04):
    """Lattice cloud: a disc, optionally surrounded by particles on the outside."""
    extent = 2.0 * (RADIUS + outer_shell) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n, dtype=torch.float64) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    keep = (r <= RADIUS + outer_shell) if two_phase else (r <= RADIUS)
    pos = pos[keep]
    count = pos.shape[0]
    mass = torch.full((count,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((count,), RHO0, dtype=torch.float64)
    return pos, mass, density


def indicator_colour(pos, mass, density, pairs, h) -> torch.Tensor:
    """Shepard-smoothed 0/1 indicator -- the colour a two-phase cloud supports."""
    vol = mass / density
    c_raw = (pos.norm(dim=1) < RADIUS).to(pos.dtype)
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    zero = torch.zeros(pos.shape[0], dtype=pos.dtype)
    num = zero.index_add(0, i, w * vol[j] * c_raw[j]) + zero.index_add(
        0, j, w * vol[i] * c_raw[i])
    den = zero.index_add(0, i, w * vol[j]) + zero.index_add(0, j, vol[i] * w)
    den = den + vol * cubic_spline_kernel(zero.new_zeros(()), h)
    return (num / den.clamp_min(1e-30)).clamp(0.0, 1.0)


def effective_sigma_ratio(pos, vol, normal, kappa, d_s) -> float:
    force = (-SIGMA * kappa * d_s)[:, None] * normal
    half = (force * vol[:, None])[pos[:, 0] > 0].sum(dim=0)
    return float(torch.linalg.vector_norm(half) / (2.0 * SIGMA))


def cell(dx: float, two_phase: bool, chain: str) -> dict:
    h = H_FACTOR * dx
    pos, mass, density = build_cloud(dx, two_phase)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    pairs_wide = cell_linked_pairs(pos, 4.0 * h)
    vol = mass / density

    if two_phase:
        color = indicator_colour(pos, mass, density, pairs, h)
    else:
        # The shipped free-surface definition: c = sum_j V_j W_ij
        color = kernel_sum_color(pos, mass, pairs, h, RHO0)

    if chain == "csf_consistent":
        grad = mls_gradient(pos, color, vol, pairs, h)
        d_s = torch.linalg.vector_norm(grad, dim=1)
        mag = d_s[:, None].clamp_min(1e-30)
        raw = -grad / mag
        n = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-30)
        kappa_fit = fit_curvature(pos, n, d_s, pairs_wide, h,
                                 fallback=mls_divergence(pos, n, vol, pairs, h),
                                 method="fit_quadratic", stencil=4.0)
        kappa_mls = mls_divergence(pos, n, vol, pairs, h)
    elif chain == "surface_tension":
        grad = compute_color_gradient(pos, color, mass, density, pairs, h)
        d_s = torch.linalg.vector_norm(grad, dim=1)
        n = compute_normal(-grad)
        kappa_fit = compute_curvature(pos, n, mass, density, pairs, h)
        kappa_mls = kappa_fit
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(chain)

    target = 2.0 * math.pi * RADIUS
    out = {
        "dx_m": dx,
        "R_over_h": RADIUS / h,
        "geometry": "two_phase" if two_phase else "one_sided",
        "chain": chain,
        "n_particles": int(pos.shape[0]),
        "d_s_mass": float((d_s * vol).sum() / target),
        "sigma_eff_over_sigma": effective_sigma_ratio(pos, vol, n, kappa_fit, d_s),
        "curvature_shell_mean_per_m": float(
            kappa_fit[d_s > 0.5 * d_s.max()].mean()),
        "curvature_analytic_per_m": 1.0 / RADIUS,
    }
    if chain == "csf_consistent":
        out["sigma_eff_over_sigma_mls_curvature"] = effective_sigma_ratio(
            pos, vol, n, kappa_mls, d_s)
    return out


def _set_radius(radius: float) -> None:
    """Point the module-level interface radius at ``radius``."""
    global RADIUS
    RADIUS = radius


#: The 2 x 2 matrix, plus the MLS-curvature variant of chain A.
MATRIX = [(True, "csf_consistent"), (True, "surface_tension"),
          (False, "csf_consistent"), (False, "surface_tension")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.0025)
    ap.add_argument("--radius", type=float, default=0.04,
                    help="interface radius in m; R/h is the controlling ratio")
    ap.add_argument("--out", default="benchmarks/two_phase_laplace/matched_chains.json")
    a = ap.parse_args()

    _set_radius(a.radius)
    cells = [cell(a.dx, two_phase, chain) for two_phase, chain in MATRIX]

    print("=" * 92)
    print(f"matched chains, dx = {a.dx*1e3:.3f} mm  R/dx = {RADIUS/a.dx:.0f}  "
          f"analytic kappa = {1.0/RADIUS:.3f} 1/m, target sigma_eff/sigma = 1")
    print("=" * 92)
    print(f"{'geometry':10s} {'chain':16s} {'N':>6s} {'d_s mass':>9s} "
          f"{'kappa':>8s} {'s_eff/s':>8s} {'(MLS kappa)':>12s}")
    for c in cells:
        mls = c.get("sigma_eff_over_sigma_mls_curvature")
        print(f"{c['geometry']:10s} {c['chain']:16s} {c['n_particles']:6d} "
              f"{c['d_s_mass']:9.4f} {c['curvature_shell_mean_per_m']:8.3f} "
              f"{c['sigma_eff_over_sigma']:8.4f} "
              f"{(f'{mls:12.4f}' if mls is not None else ' ' * 12)}")

    report = {
        "setup": {"radius_m": a.radius, "sigma": SIGMA, "rho0": RHO0,
                  "h_factor": H_FACTOR, "dx_m": a.dx,
                  "metric": "|sum_{x>0} sigma kappa |grad c| n dV| / (2 sigma)"},
        "cells": cells,
        "readings": {
            "deficit_follows_geometry_not_chain": bool(
                abs(cells[0]["sigma_eff_over_sigma"]
                    - cells[2]["sigma_eff_over_sigma"])
                > 2.0 * abs(cells[0]["sigma_eff_over_sigma"]
                            - cells[1]["sigma_eff_over_sigma"])),
            "chain_gap_two_phase": cells[1]["sigma_eff_over_sigma"]
            - cells[0]["sigma_eff_over_sigma"],
            "chain_gap_one_sided": cells[3]["sigma_eff_over_sigma"]
            - cells[2]["sigma_eff_over_sigma"],
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()