"""Quantitative Young-Laplace regression gate for the CSF model.

Motivation: the surface-tension tests that shipped with the model only checked
that the force was non-zero and "predominantly inward". Those assertions pass
for a curvature field whose magnitude is two orders off, which is exactly what
the model did before the gradient-symmetry bug was fixed (kappa median 0.0 with
a maximum of 200 1/m on a circle where kappa = 1/R = 10 1/m, and the normal
direction split 50/50 between inward and outward).

These tests pin the analytic reference instead:

    circle curvature        kappa = 1/R
    Young-Laplace jump      Delta p = sigma / R

Measured kappa and the CSF force integral are compared against those values and
the relative errors are bounded. The bounds are regression bounds for the
current discretisation, *not* a statement that the model meets the 15 % platform
accuracy standard -- see docs/validation/young_laplace_csf.md for the measured
errors and for the remaining delta_s under-integration deficit.
"""
from __future__ import annotations

import torch
import pytest

from tensorsph.droplet_impact import Config, cubic_spline_kernel
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    compute_color_gradient,
    compute_curvature,
    compute_surface_tension_force,
)


def _droplet(radius: float, dx: float = 0.005, center=(0.2, 0.2), rho0=1000.0):
    """Circular droplet of particles on a regular square lattice."""
    extent = 2.0 * center[0]
    nx = int(round(extent / dx))
    x = torch.linspace(0.0, extent, nx, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    dist = ((pos[:, 0] - center[0]) ** 2 + (pos[:, 1] - center[1]) ** 2).sqrt()
    pos = pos[dist <= radius]
    n = pos.shape[0]
    mass = torch.full((n,), rho0 * dx ** 2, dtype=torch.float64)
    density = torch.full((n,), rho0, dtype=torch.float64)
    return pos, mass, density


def _measure(radius: float, sigma: float = 0.072, rho0: float = 1000.0, dx: float = 0.005):
    pos, mass, density = _droplet(radius, dx=dx, rho0=rho0)
    cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=sigma)
    h = cfg.h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)

    i, j = pairs.unbind(dim=1)
    vol_i = mass[i] / density[i]
    vol_j = mass[j] / density[j]
    w_pair = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    color = torch.zeros(pos.shape[0], dtype=pos.dtype).index_add(0, i, vol_j * w_pair)
    color = color.index_add(0, j, vol_i * w_pair)
    bulk = color > 0.9 * color.max()
    color = color / color[bulk].mean()
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    dS = grad_c.norm(dim=1)
    normal = -grad_c / dS.clamp_min(1e-30)[:, None]
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)

    rvec = pos - torch.tensor([0.2, 0.2], dtype=pos.dtype)
    r = rvec.norm(dim=1)
    rhat = rvec / r[:, None]
    shell = dS > 0.3 * dS.max()
    w = dS[shell] * (mass[shell] / density[shell])
    w = w / w.sum()

    kappa_dS = float((kappa[shell] * w).sum())
    vol = mass / density
    f_radial = ((density[:, None] * acc) * rhat).sum(dim=1) * vol
    dp = float(abs(f_radial.sum()) / (2.0 * torch.pi * radius))
    cos_nr = float((normal[shell] * rhat[shell]).sum(dim=1).median())
    return kappa_dS, dp, cos_nr


@pytest.mark.parametrize("radius", [0.05, 0.10])
def test_interface_normal_points_outward(radius):
    """The colour-gradient normal must align with the radial direction.

    A sign error in the symmetric gradient term splits this 50/50 (median ~0).
    """
    _, _, cos_nr = _measure(radius)
    assert cos_nr > 0.95, f"normal misaligned: median cos(n, r_hat) = {cos_nr:.3f}"


@pytest.mark.parametrize("radius", [0.05, 0.10])
def test_curvature_matches_circle(radius):
    """Measured curvature must be within 25 % of the analytic 1/R."""
    kappa_dS, _, _ = _measure(radius)
    rel = abs(kappa_dS * radius - 1.0)
    assert rel < 0.25, f"kappa = {kappa_dS:.3f} 1/m vs 1/R = {1.0 / radius:.3f} 1/m ({100 * rel:.1f}%)"


@pytest.mark.parametrize("radius", [0.05, 0.10])
def test_laplace_pressure_jump_regression(radius):
    """CSF force integral must reproduce sigma/R within the known deficit.

    The single-phase free surface truncates the colour function at ~0.6 instead
    of 0, so |grad c| integrates to ~0.27 of the interface length and Delta p
    comes out ~73 % low. The bound below is a regression gate for that known
    state; it tightens to 15 % once delta_s is normalised.
    """
    _, dp, _ = _measure(radius)
    theory = 0.072 / radius
    rel = abs(dp / theory - 1.0)
    assert rel < 0.80, f"Delta p = {dp:.4f} Pa vs sigma/R = {theory:.4f} Pa ({100 * rel:.1f}%)"