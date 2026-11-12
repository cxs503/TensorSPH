"""Two-phase Laplace-law verification of the CSF surface-tension chain.

Purpose
-------
The shipped TensorSPH droplet is a **one-sided free surface**: particles exist
only inside the fluid, so the colour function ``c = sum V_j W_ij`` falls from
~1 in the bulk to ~0.63 at the outermost layer (the remaining 0.63 -> 0
transition lives in the void where there are no particles).  The CSF delta
function ``delta_s = 2 |grad c|`` therefore carries only ~75 % of the unit
mass required by the Brackbill identity ``int delta_s dn = 1``, and the
effective surface tension is ~29 % low (``docs/verification_effective_sigma.md``).

This benchmark constructs a **two-phase** cloud: particles exist on both sides
of the interface (fluid inside, "void" particles with the same density outside),
so the colour function completes the full 1 -> 0 transition and the CSF chain
can be verified against the analytic Young-Laplace jump ``dp = sigma / R``.

Result (m16489 re-run -- the m7558 numbers below were never reproducible)
------
Measured on this committed harness, ``python benchmarks/two_phase_laplace/run.py``::

    dx = 5.00 mm  R/h = 6.2   sigma_eff/sigma = 0.6246  (-37.5 %)
    dx = 2.50 mm  R/h = 12.3  sigma_eff/sigma = 0.6622  (-33.8 %)
    dx = 1.25 mm  R/h = 24.6  sigma_eff/sigma = 0.6628  (-33.7 %)

The error is flat under refinement, so this is a systematic operator deficit,
not a discretisation error that resolves away.  The m7558 docstring used to
claim **4.6 % at dx = 2.5 mm (R/h = 10.3)**; that number corresponds to no
configuration reachable from this file (``H_FACTOR`` has been 1.3 throughout,
i.e. R/h = 12.3 here), and the ledger/verification document that repeated it
were therefore proving the opposite of what the code does.

The attribution is measured, not argued, by two companion harnesses:

* ``diagnose_curvature.py`` -- ablates colour / normal / curvature / stencil.
  The force assembly is exact (analytic control reads 0.998), the Brackbill
  delta ``sum |grad c| dV`` carries only ~0.70 of unit mass even on this fully
  two-phase cloud, and ``fit_curvature`` under-reads the curvature by ~30 %
  whenever it is fed the *thick* colour band instead of a thin shell.
* ``matched_chains.py`` -- the 2x2 matrix (geometry x CSF chain) with one
  metric.  The two chains shipped in this library differ by ~0.2 in effective
  surface tension on identical particles.

Consequence: the one-sided free-surface deficit cannot be explained away as
"truncation" while a closed two-phase interface loses 34 % through the same
chain.  ``docs/verification_two_phase_laplace.md`` carries the corrected story.

Run::

    python benchmarks/two_phase_laplace/run.py [--plot]

Writes ``benchmarks/two_phase_laplace/validation_report.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import cubic_spline_gradient, cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import (
    csf_fields, kernel_sum_color, mls_gradient, mls_divergence, shepard_smooth,
)
from tensorsph.csf_curvature import fit_curvature

RADIUS = 0.04
SIGMA = 0.072
RHO0 = 1000.0
DX_LIST = (0.005, 0.0025, 0.00125)
H_FACTOR = 1.3


def build_two_phase_disc(radius: float, dx: float, outer_shell: float = 0.04):
    """Square lattice with particles inside AND outside the circle.

    The "void" particles have the same density as the fluid, so the colour
    function ``c = sum V_j W_ij`` completes the full 1 -> 0 transition.
    """
    extent = 2.0 * (radius + outer_shell) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n, dtype=torch.float64) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    keep = r <= radius + outer_shell
    pos = pos[keep]
    count = pos.shape[0]
    mass = torch.full((count,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((count,), RHO0, dtype=torch.float64)
    return pos, mass, density


def smooth_color_field(pos, mass, density, pairs, h):
    """Kernel-smoothed colour: c_i = sum_j V_j c_j W_ij / sum_j V_j W_ij."""
    vol = mass / density
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    zero = torch.zeros(pos.shape[0], dtype=pos.dtype)
    num = zero.index_add(0, i, w * vol[j]) + zero.index_add(0, j, w * vol[i])
    den = zero.index_add(0, i, w * vol[j]) + zero.index_add(0, j, w * vol[i])
    # Initial colour: 1 inside, 0 outside
    r = pos.norm(dim=1)
    c_raw = (r < RADIUS).to(pos.dtype)
    num = zero.index_add(0, i, w * vol[j] * c_raw[j]) + zero.index_add(
        0, j, w * vol[i] * c_raw[i])
    return num / den.clamp_min(1.0e-30)


def two_phase_laplace_case(dx: float, outer: float = 0.04,
                           curvature: str = "fit_quadratic"):
    """One static evaluation of the CSF force on a two-phase circular interface."""
    h = H_FACTOR * dx
    pos, mass, density = build_two_phase_disc(RADIUS, dx, outer_shell=outer)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    pairs_wide = cell_linked_pairs(pos, 4.0 * h)

    vol = mass / density
    color = smooth_color_field(pos, mass, density, pairs, h)

    # CSF fields
    grad_c = mls_gradient(pos, color, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    tiny = 1e-12
    normal = torch.where(mag > tiny, -grad_c / mag.clamp_min(tiny),
                         torch.zeros_like(grad_c))
    smoothed = shepard_smooth(normal, pos, vol, pairs, h, passes=2)
    norm = torch.linalg.vector_norm(smoothed, dim=1, keepdim=True)
    normal = torch.where(mag > tiny, smoothed / norm.clamp_min(tiny), normal)
    kappa_mls = mls_divergence(pos, normal, vol, pairs, h)

    # delta_s: for two-phase, the colour goes 1->0, so |grad c| integrates to 1
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)

    kappa = fit_curvature(pos, normal, delta_s, pairs_wide, h,
                          fallback=kappa_mls, method=curvature, stencil=4.0)

    # Force: a = -sigma * kappa * delta_s * n / rho
    force = (-SIGMA * kappa * delta_s)[:, None] * normal
    f_right = (force * vol[:, None])[pos[:, 0] > 0].sum(dim=0)
    ratio = float(torch.linalg.vector_norm(f_right) / (2.0 * SIGMA))

    # Diagnostics
    interface = delta_s > 0.05 * delta_s.max()
    kappa_iface = float(kappa[interface].mean()) if interface.any() else 0.0
    ds_integral = float((delta_s * vol).sum())
    color_jump = float((delta_s * vol).sum())  # for two-phase, this should be ~2*pi*R

    return {
        "dx": dx,
        "curvature": curvature,
        "R_over_h": RADIUS / h,
        "R_over_dx": RADIUS / dx,
        "n_particles": int(pos.shape[0]),
        "n_fluid": int((pos.norm(dim=1) < RADIUS).sum()),
        "n_void": int((pos.norm(dim=1) >= RADIUS).sum()),
        "ratio": ratio,
        "force_right_half_N": float(torch.linalg.vector_norm(f_right)),
        "target_force_N": 2.0 * SIGMA,
        "kappa_iface_mean": kappa_iface,
        "kappa_analytic": 1.0 / RADIUS,
        "kappa_error_pct": (kappa_iface - 1.0 / RADIUS) / (1.0 / RADIUS) * 100,
        "delta_s_integral": ds_integral,
        "delta_s_target": 2.0 * math.pi * RADIUS,
        "color_jump": color_jump,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="benchmarks/two_phase_laplace")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    theory = SIGMA / RADIUS
    print("=" * 72)
    print("Two-phase Laplace-law verification: dp = sigma / R")
    print(f"  sigma = {SIGMA} N/m, R = {RADIUS} m, rho0 = {RHO0} kg/m^3")
    print(f"  theory: dp = {theory:.6f} Pa")
    print("=" * 72)

    rows = []
    for dx in DX_LIST:
        rep = two_phase_laplace_case(dx)
        rows.append(rep)
        err = abs(rep["ratio"] - 1.0) * 100
        print(f"dx={dx*1e3:6.3f}mm  N={rep['n_particles']:5d}  "
              f"R/h={rep['R_over_h']:.1f}  ratio={rep['ratio']:.4f}  "
              f"err={err:.1f}%  kappa_err={rep['kappa_error_pct']:.1f}%")

    report = {
        "setup": {"radius_m": RADIUS, "sigma": SIGMA, "rho0": RHO0,
                  "h_factor": H_FACTOR, "curvature": "fit_quadratic"},
        "theory_dp_Pa": theory,
        "rows": rows,
        "verdict": {
            "finest_within_15pct": abs(rows[-1]["ratio"] - 1.0) < 0.15,
            "monotone_convergence": all(
                abs(rows[i]["ratio"] - 1.0) >= abs(rows[i+1]["ratio"] - 1.0)
                for i in range(len(rows) - 1)),
        },
    }
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'validation_report.json'}")


if __name__ == "__main__":
    main()
