"""Two-phase Laplace-law verification (regression test, m7558).

Pins the CSF model's effective surface tension on a two-phase circular interface
where the colour function completes the full 1 -> 0 transition.  This proves the
operators are correct in principle; the one-sided deficit is a free-surface
truncation effect.
"""
from __future__ import annotations

import torch

from tensorsph.droplet_impact import cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import mls_gradient, mls_divergence, shepard_smooth
from tensorsph.csf_curvature import fit_curvature

RADIUS = 0.04
SIGMA = 0.072
RHO0 = 1000.0
DX = 0.005
H_FACTOR = 1.3


def _build_two_phase(radius, dx, outer=0.04):
    extent = 2.0 * (radius + outer) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n, dtype=torch.float64) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    keep = r <= radius + outer
    pos = pos[keep]
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((pos.shape[0],), RHO0, dtype=torch.float64)
    return pos, mass, density


def _smooth_color(pos, mass, density, pairs, h):
    vol = mass / density
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    c_raw = (pos.norm(dim=1) < RADIUS).to(pos.dtype)
    zero = torch.zeros(pos.shape[0], dtype=pos.dtype)
    num = zero.index_add(0, i, w * vol[j] * c_raw[j]) + zero.index_add(0, j, w * vol[i] * c_raw[i])
    den = zero.index_add(0, i, w * vol[j]) + zero.index_add(0, j, w * vol[i])
    return num / den.clamp_min(1e-30)


def _effective_sigma_ratio(dx=DX, curvature="fit_quadratic"):
    h = H_FACTOR * dx
    pos, mass, density = _build_two_phase(RADIUS, dx)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    pairs_wide = cell_linked_pairs(pos, 4.0 * h)
    vol = mass / density
    color = _smooth_color(pos, mass, density, pairs, h)
    grad_c = mls_gradient(pos, color, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    normal = torch.where(mag > 1e-12, -grad_c / mag.clamp_min(1e-12), torch.zeros_like(grad_c))
    smoothed = shepard_smooth(normal, pos, vol, pairs, h, passes=2)
    norm = torch.linalg.vector_norm(smoothed, dim=1, keepdim=True)
    normal = torch.where(mag > 1e-12, smoothed / norm.clamp_min(1e-12), normal)
    kappa_mls = mls_divergence(pos, normal, vol, pairs, h)
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    kappa = fit_curvature(pos, normal, delta_s, pairs_wide, h, fallback=kappa_mls,
                          method=curvature, stencil=4.0)
    force = (-SIGMA * kappa * delta_s)[:, None] * normal
    f_right = (force * vol[:, None])[pos[:, 0] > 0].sum(dim=0)
    return float(torch.linalg.vector_norm(f_right) / (2.0 * SIGMA))


def test_two_phase_effective_sigma_within_gate():
    """Two-phase CSF effective sigma must be within 15 % of analytic.

    This is the decisive control: with a resolved closed interface the CSF
    chain delivers the correct traction, proving the operators are sound.
    """
    ratio = _effective_sigma_ratio()
    assert abs(ratio - 1.0) < 0.15, f"two-phase ratio {ratio:.4f} outside 15 % gate"


def test_two_phase_ratio_better_than_one_sided():
    """Two-phase ratio must be closer to 1.0 than the one-sided baseline (0.71).

    This guards against regressions that would make the two-phase case worse
    than the already-documented one-sided deficit.
    """
    ratio = _effective_sigma_ratio()
    assert abs(ratio - 1.0) < abs(0.7106 - 1.0), (
        f"two-phase ratio {ratio:.4f} not better than one-sided 0.7106")
