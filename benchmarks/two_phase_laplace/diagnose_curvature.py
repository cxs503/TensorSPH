"""Ablation of the two-phase Laplace deficit: where does sigma_eff/sigma = 0.66 come from?

Background
----------
``benchmarks/two_phase_laplace/run.py`` documents ``sigma_eff/sigma = 0.95+``
(Laplace error 4.6 % at ``dx = 2.5 mm``, ``R/h = 10.3``) and
``docs/verification_two_phase_laplace.md`` uses that number as the *control*
that proves the CSF operators are "correct in principle", blaming the ~29 %
one-sided deficit on free-surface truncation.  Running the committed harness
does not reproduce it: every resolution gives ``sigma_eff/sigma ~ 0.66``
(Laplace error 33.7-37.5 %) while ``int delta_s dV`` is within 1.1 % of
``2 pi R`` -- i.e. the colour/delta_s mass is right and the curvature is not.

This script ablates the measurement so the deficit can be attributed to a
single stage instead of being blamed on a rounded pair of numbers:

  * ``delta_s``       : analytic thin-shell vs colour-function ``|grad c|``
  * ``normal``        : analytic ``-r_hat`` vs colour-derived, with 0..3 Shepard passes
  * ``curvature``     : ``fit_quadratic`` / ``fit_circle`` / ``mls`` (divergence)
  * ``stencil``       : fit radius in units of ``h`` (2, 3, 4, 5)

Each variant reports the shell-mean curvature error and the force ratio
``|sum_{x>0} sigma kappa delta_s n dV| / (2 sigma)``; 2 sigma is the exact
surface-tension resultant on the right half of a circular interface.

Run::

    python benchmarks/two_phase_laplace/diagnose_curvature.py [--dx 0.0025]

Writes ``benchmarks/two_phase_laplace/diagnose_curvature.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import (
    kernel_sum_color, mls_divergence, mls_gradient, shepard_smooth,
)
from tensorsph.csf_curvature import fit_curvature

RADIUS = 0.04
SIGMA = 0.072
RHO0 = 1000.0
H_FACTOR = 1.3


def build_two_phase_disc(radius: float, dx: float, outer_shell: float = 0.04):
    extent = 2.0 * (radius + outer_shell) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n, dtype=torch.float64) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    pos = pos[r <= radius + outer_shell]
    count = pos.shape[0]
    mass = torch.full((count,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((count,), RHO0, dtype=torch.float64)
    return pos, mass, density


def indicator_colour(pos, mass, density, pairs, h, radius=RADIUS):
    """Two-phase colour: Shepard-smoothed indicator of the fluid side.

    ``c_i = sum_j V_j c_j W_ij / sum_j V_j W_ij`` with ``c_j`` the raw 0/1
    indicator.  This is the *only* colour definition that makes sense on a
    two-phase cloud: because particles exist on both sides of the interface the
    plain kernel sum ``sum_j V_j W_ij`` (``kernel_sum_color``, the one-sided
    definition) saturates at 1 everywhere inside the cloud and carries no
    interface at all.
    """
    vol = mass / density
    c_raw = (pos.norm(dim=1) < radius).to(pos.dtype)
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    zero = torch.zeros(pos.shape[0], dtype=pos.dtype)
    num = zero.index_add(0, i, w * vol[j] * c_raw[j]) + zero.index_add(
        0, j, w * vol[i] * c_raw[i])
    den = zero.index_add(0, i, w * vol[j]) + zero.index_add(0, j, w * vol[i])
    den = den + vol * cubic_spline_kernel(zero.new_zeros(()), h)
    return (num / den.clamp_min(1e-30)).clamp(0.0, 1.0)


def colour_fields(pos, mass, density, pairs, h, source="indicator"):
    """Colour, its MLS gradient and the colour-derived normal."""
    vol = mass / density
    if source == "indicator":
        color = indicator_colour(pos, mass, density, pairs, h)
    else:  # "kernel_sum" -- one-sided volume sum, kept as a negative control
        color = kernel_sum_color(pos, mass, pairs, h, RHO0)
    grad = mls_gradient(pos, color, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad, dim=1, keepdim=True)
    ok = mag > 1.0e-12
    return vol, color, grad, mag, ok


def smoothed_normal(pos, vol, grad, mag, ok, pairs, h, passes: int):
    n = torch.where(ok, -grad / mag.clamp_min(1e-30), torch.zeros_like(grad))
    if passes:
        n = shepard_smooth(n, pos, vol, pairs, h, passes=passes)
    norm = torch.linalg.vector_norm(n, dim=1, keepdim=True)
    return torch.where(ok, n / norm.clamp_min(1e-30), n)


def analytic_normal(pos):
    r = pos.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return pos / r


def force_ratio(pos, vol, normal, kappa, delta_s):
    force = (-SIGMA * kappa * delta_s)[:, None] * normal
    f_right = (force * vol[:, None])[pos[:, 0] > 0].sum(dim=0)
    return float(torch.linalg.vector_norm(f_right) / (2.0 * SIGMA))


def shell_mean(kappa, delta_s, fraction: float = 0.5):
    mask = delta_s > fraction * delta_s.max()
    if not bool(mask.any()):
        return float("nan"), int(mask.sum())
    return float(kappa[mask].mean()), int(mask.sum())


def run_case(dx: float, passes: int = 2):
    """Full ablation matrix for one resolution."""
    h = H_FACTOR * dx
    pos, mass, density = build_two_phase_disc(RADIUS, dx)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    pairs_wide = cell_linked_pairs(pos, 4.0 * h)
    r = pos.norm(dim=1)

    kappa_mls_of = {}
    delta_s_of = {}
    normal_of = {}

    # ---- colours -----------------------------------------------------------
    for source in ("indicator", "kernel_sum"):
        vol, color, grad, mag, ok = colour_fields(pos, mass, density, pairs, h,
                                                  source=source)
        delta_s_of[source] = mag.squeeze(1)
        normal_of[source] = smoothed_normal(pos, vol, grad, mag, ok, pairs, h, passes)
        kappa_mls_of[source] = mls_divergence(pos, normal_of[source], vol, pairs, h)

    vol = mass / density

    # analytic delta_s: thin shell with unit mass across the surface
    w = 0.5 * dx
    shell = torch.exp(-((r - RADIUS) / w) ** 2)
    ds_analytic = shell * (2.0 * math.pi * RADIUS) / (shell * vol).sum()

    n_exact = analytic_normal(pos)
    kappa_exact = torch.full_like(r, 1.0 / RADIUS)

    # normal fidelity: 1.0 means exactly radial
    align = {}
    for source in ("indicator", "kernel_sum"):
        n = normal_of[source]
        mask = delta_s_of[source] > 0.5 * delta_s_of[source].max()
        align[source] = float((n[mask] * n_exact[mask]).sum(dim=1).mean())

    ds_set = {"analytic": ds_analytic,
              "indicator_ds": delta_s_of["indicator"],
              "kernel_sum_ds": delta_s_of["kernel_sum"]}
    n_set = {"analytic": n_exact,
             "indicator_n": normal_of["indicator"],
             "kernel_sum_n": normal_of["kernel_sum"]}

    rows = []

    def add(label, kappa, ds_name, n_name):
        kappa = kappa
        ds = ds_set[ds_name]
        nm = n_set[n_name]
        km, n_shell = shell_mean(kappa, ds)
        rows.append({
            "variant": label,
            "kappa_shell_mean": km,
            "kappa_error_pct": (km - 1.0 / RADIUS) / (1.0 / RADIUS) * 100.0,
            "shell_particles": n_shell,
            "delta_s_integral": float((ds * vol).sum()),
            "delta_s_over_2piR": float((ds * vol).sum() / (2.0 * math.pi * RADIUS)),
            "normal_radial_alignment": align.get(n_name.replace("_n", ""), None),
            "force_ratio": force_ratio(pos, vol, nm, kappa, ds),
        })

    # 0. analytic control: the force assembly itself
    add("control/all_analytic", kappa_exact, "analytic", "analytic")
    # 1. estimator quality on a thin shell with exact normals (isolates stencil)
    for st in (2.0, 3.0, 4.0, 5.0):
        k = fit_curvature(pos, n_exact, ds_analytic, pairs_wide, h,
                          fallback=kappa_mls_of["indicator"], method="fit_quadratic",
                          stencil=st)
        add(f"fit_quadratic/analytic_n/analytic_ds/st{st:g}", k, "analytic", "analytic")
    # 2. every cross-combination of normal source x delta_s source
    for n_name in ("analytic", "indicator_n", "kernel_sum_n"):
        for ds_name in ("indicator_ds", "kernel_sum_ds"):
            key = n_name.replace("_n", "")
            for st in (3.0, 4.0, 5.0):
                k = fit_curvature(pos, n_set[n_name], ds_set[ds_name], pairs_wide, h,
                                  fallback=kappa_mls_of.get(key, torch.zeros_like(r)),
                                  method="fit_quadratic", stencil=st)
                add(f"fit_quadratic/{n_name}/{ds_name}/st{st:g}", k, ds_name, n_name)
    # 3. fit_circle on the shipped pairing
    for st in (3.0, 4.0, 5.0):
        k = fit_curvature(pos, n_set["indicator_n"], ds_set["indicator_ds"],
                          pairs_wide, h, fallback=kappa_mls_of["indicator"],
                          method="fit_circle", stencil=st)
        add(f"fit_circle/indicator_n/indicator_ds/st{st:g}", k,
            "indicator_ds", "indicator_n")
    # 4. MLS divergence curvature
    add("mls_divergence/indicator", kappa_mls_of["indicator"],
        "indicator_ds", "indicator_n")

    return {
        "dx": dx,
        "R_over_h": RADIUS / h,
        "R_over_dx": RADIUS / dx,
        "n_particles": int(pos.shape[0]),
        "shepard_passes": passes,
        "normal_radial_alignment": align,
        "colour_range": {},
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.0025)
    ap.add_argument("--out", default="benchmarks/two_phase_laplace/diagnose_curvature.json")
    a = ap.parse_args()

    case = run_case(a.dx)
    print("=" * 96)
    print(f"dx = {a.dx*1e3:.3f} mm   R/h = {case['R_over_h']:.2f}   "
          f"N = {case['n_particles']}   kappa_analytic = {1.0/RADIUS:.3f} 1/m")
    print("=" * 96)
    print(f"{'variant':52s} {'kappa':>9s} {'err%':>8s} {'ds_int':>9s} {'ratio':>7s}")
    for r in case["rows"]:
        print(f"{r['variant']:52s} {r['kappa_shell_mean']:9.3f} "
              f"{r['kappa_error_pct']:8.2f} {r['delta_s_integral']:9.5f} "
              f"{r['force_ratio']:7.4f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(case, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()