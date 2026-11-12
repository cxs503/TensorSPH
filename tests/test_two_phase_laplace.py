"""Two-phase Laplace control: what the CSF chains actually deliver (m16489).

History
-------
``benchmarks/two_phase_laplace/run.py`` (m7558) documented a Laplace error of
**4.6 %** at ``dx = 2.5 mm`` and ``docs/verification_two_phase_laplace.md`` used
that number -- plus the ``benchmarks/surface_tension_laplace`` reading of the
*other* CSF chain -- as the decisive control: "the CSF operators are correct in
principle; the ~29 % one-sided deficit is free-surface truncation".

That control does not survive a re-run.  Measured on the committed harness
(``benchmarks/two_phase_laplace/validation_report.json``, m16489):

    dx = 5.00 mm   R/h =  6.2   sigma_eff/sigma = 0.625   (-37.5 %)
    dx = 2.50 mm   R/h = 12.3   sigma_eff/sigma = 0.662   (-33.8 %)
    dx = 1.25 mm   R/h = 24.6   sigma_eff/sigma = 0.663   (-33.7 %)

i.e. 34 % low and flat under refinement -- the same order as the one-sided
deficit it was supposed to explain.

What these tests pin
--------------------
1. ``test_force_assembly_control_is_exact`` -- the measurement itself (analytic
   kappa, analytic thin-shell delta_s) recovers ``2 sigma`` to ~1 %, so a 34 %
   deficit is *not* a measurement artifact.
2. ``test_two_phase_delta_s_mass_is_short`` -- ``sum |grad c| dV / (2 pi R)`` on
   a fully two-phase cloud is ~0.70, not 1.0: the Brackbill delta function
   carries ~70 % of unit mass *with particles on both sides*, so free-surface
   truncation cannot be the whole story.
3. ``test_two_phase_effective_sigma_misses_the_gate`` -- ``sigma_eff/sigma`` on
   the resolved closed interface misses the 15 % gate (chain A, shipped
   ``csf_fields`` pairing).
4. ``test_fit_quadratic_underreads_a_thick_colour_band`` -- ``fit_curvature``
   is accurate on a thin analytic shell but ~30 % low on the colour-function
   band that the solver actually feeds it (resolution independent, R/h = 12.3
   and 30.8 both give ~-30 %).
5. ``test_sigma_eff_deficit_is_chain_dependent`` -- the second chain
   (``surface_tension``) reads 0.67-0.70 on the same cloud, so the two shipped
   CSF chains differ by ~0.2 in effective tension on identical input.

Everything above is a *pin*, not an acceptance gate: the values are printed by
``benchmarks/two_phase_laplace/{run,diagnose_curvature,matched_chains}.py``.
The previous edition of this file marked these checks ``xfail(strict=False)``
and blamed an "editable install pointing at another worktree"; a non-strict
xfail can never fail, which is how a 34 % regression sat next to a "4.6 %"
docstring for months.  If a future fix makes ``sigma_eff/sigma`` exceed the
gate, tests 2-4 turn red on purpose and this file has to be updated with the
new numbers.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import kernel_sum_color, mls_divergence, mls_gradient
from tensorsph.csf_curvature import fit_curvature
from tensorsph.surface_tension import (
    compute_color_gradient, compute_curvature, compute_normal,
)

RADIUS = 0.04
SIGMA = 0.072
RHO0 = 1000.0
H_FACTOR = 1.3
DX = 0.0025         # 3128 particles, R/h = 12.3; numbers printed by the harnesses


def _cloud(dx: float, radius: float = RADIUS, two_phase: bool = True):
    outer = 0.04
    extent = 2.0 * (radius + outer) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n, dtype=torch.float64) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    pos = pos[(r <= radius + outer) if two_phase else (r <= radius)]
    count = pos.shape[0]
    mass = torch.full((count,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((count,), RHO0, dtype=torch.float64)
    return pos, mass, density


def _indicator_colour(pos, mass, density, pairs, h, radius=RADIUS):
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


def _ratio(pos, vol, normal, kappa, d_s):
    force = (-SIGMA * kappa * d_s)[:, None] * normal
    half = (force * vol[:, None])[pos[:, 0] > 0].sum(dim=0)
    return float(torch.linalg.vector_norm(half) / (2.0 * SIGMA))


def _measure(dx: float = DX, radius: float = RADIUS, chain: str = "csf_consistent"):
    h = H_FACTOR * dx
    pos, mass, density = _cloud(dx, radius=radius)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    pairs_wide = cell_linked_pairs(pos, 4.0 * h)
    vol = mass / density
    color = _indicator_colour(pos, mass, density, pairs, h, radius=radius)

    if chain == "csf_consistent":
        grad = mls_gradient(pos, color, vol, pairs, h)
        d_s = torch.linalg.vector_norm(grad, dim=1)
        raw = -grad / d_s[:, None].clamp_min(1e-30)
        n = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-30)
        kappa_mls = mls_divergence(pos, n, vol, pairs, h)
        kappa = fit_curvature(pos, n, d_s, pairs_wide, h, fallback=kappa_mls,
                              method="fit_quadratic", stencil=4.0)
    else:
        grad = compute_color_gradient(pos, color, mass, density, pairs, h)
        d_s = torch.linalg.vector_norm(grad, dim=1)
        n = compute_normal(-grad)
        kappa = compute_curvature(pos, n, mass, density, pairs, h)

    shell = d_s > 0.5 * d_s.max()
    return {
        "d_s_mass": float((d_s * vol).sum() / (2.0 * math.pi * radius)),
        "ratio": _ratio(pos, vol, n, kappa, d_s),
        "kappa_shell": float(kappa[shell].mean()),
        "kappa_analytic": 1.0 / radius,
        "grad": grad, "d_s": d_s, "vol": vol, "pos": pos, "n": n,
        "pairs_wide": pairs_wide, "h": h,
    }


@pytest.fixture(scope="module")
def two_phase():
    return _measure()


@pytest.fixture(scope="module")
def one_sided_chain_b():
    return _measure(chain="surface_tension")


def test_force_assembly_control_is_exact():
    """Analytic kappa + analytic thin shell must recover the exact 2 sigma."""
    h = H_FACTOR * DX
    pos, mass, density = _cloud(DX)
    vol = mass / density
    r = pos.norm(dim=1)
    w = 0.5 * DX
    shell = torch.exp(-((r - RADIUS) / w) ** 2)
    d_s = shell * (2.0 * math.pi * RADIUS) / (shell * vol).sum()
    n = pos / r.clamp_min(1e-30)[:, None]
    kappa = torch.full_like(r, 1.0 / RADIUS)
    ratio = _ratio(pos, vol, n, kappa, d_s)
    assert abs(ratio - 1.0) < 0.02, f"force-assembly control {ratio:.4f} != 1"


def test_two_phase_delta_s_mass_is_short(two_phase):
    """Brackbill delta carries ~0.70 of unit mass even with particles both sides."""
    mass = two_phase["d_s_mass"]
    assert 0.60 < mass < 0.75, (
        f"two-phase int|grad c|dV/(2 pi R) = {mass:.4f}; the m7558 docstring "
        "claims the transition completes (=1.0) - update the docs if this moves")


def test_two_phase_effective_sigma_misses_the_gate(two_phase):
    """The 'decisive control' misses the 15 % gate: 0.48 (chain A), not 0.95+."""
    ratio = two_phase["ratio"]
    assert abs(ratio - 1.0) > 0.15, (
        f"two-phase sigma_eff/sigma = {ratio:.4f} now inside the gate - the "
        "deficit story in docs/verification_two_phase_laplace.md must be rewritten")
    assert abs(ratio - 0.48) < 0.05, (
        f"two-phase sigma_eff/sigma (csf_consistent chain) moved to {ratio:.4f}; "
        "re-pin from benchmarks/two_phase_laplace/matched_chains.json")


@pytest.mark.parametrize("radius", [0.04, 0.1])
def test_fit_quadratic_underreads_a_thick_colour_band(radius):
    """fit_curvature is ~30 % low on the colour band, independent of R/h."""
    m = _measure(dx=0.0025, radius=radius)
    err = (m["kappa_shell"] - m["kappa_analytic"]) / m["kappa_analytic"]
    assert err < -0.20, (
        f"fit_quadratic kappa error {err*100:.1f} % at R = {radius} m "
        "(expected ~-30 % on the thick colour band; thin analytic shells read +1.5 %)")
    assert err > -0.45, f"fit_quadratic kappa error {err*100:.1f} % is worse than pinned"


def test_sigma_eff_deficit_is_chain_dependent(two_phase, one_sided_chain_b):
    """The two shipped CSF chains disagree by ~0.2 on identical input."""
    gap = one_sided_chain_b["ratio"] - two_phase["ratio"]
    assert 0.1 < gap < 0.3, (
        f"chain gap {gap:.4f} (surface_tension {one_sided_chain_b['ratio']:.4f} vs "
        f"csf_consistent {two_phase['ratio']:.4f})")


def test_surface_tension_chain_also_misses_on_two_phase(one_sided_chain_b):
    ratio = one_sided_chain_b["ratio"]
    assert abs(ratio - 1.0) > 0.15, (
        f"surface_tension chain reads {ratio:.4f} on a two-phase cloud: the "
        "4.4 % reading of benchmarks/surface_tension_laplace does not transfer")