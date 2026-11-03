"""Verification gates for the merged CSF surface-tension model.

Two kinds of test live here.

*What the model does do* is pinned as a normal test: on an isolated droplet the
acceleration is predominantly radial and inward, and the internal force field
carries almost no net momentum.

*What the model does not do* is pinned with ``xfail(strict=True)``. The
Young-Laplace law ``dp = sigma / R`` is the one analytic statement a capillary
model has to satisfy, and the merged implementation does not satisfy it: the
surface delta function integrates to about 0.4 instead of 1, the curvature of a
circular interface comes out an order of magnitude low with the wrong sign at
low resolution, and the resulting pressure jump does not converge with
resolution. Those tests are marked so that the suite stays green while the gap
is open, and so that a fix fails loudly: when someone reworks the model, the
``strict`` xfail turns into an XPASS failure and this file has to be updated
with the new measured numbers. See docs/benchmarks/laplace-verification.md.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

RADIUS = 0.05
SIGMA = 0.072
RHO0 = 1000.0
LEVELS = 10                       # 318 particles: fast enough for the suite


def _benchmark_module():
    path = (Path(__file__).resolve().parents[1]
            / "benchmarks" / "laplace" / "run.py")
    spec = importlib.util.spec_from_file_location("laplace_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def droplet():
    """Lattice disc plus the CSF acceleration from the production path."""
    bench = _benchmark_module()
    dx = RADIUS / LEVELS
    cfg = Config(dx=dx, rho0=RHO0, surface_tension_coefficient=SIGMA,
                 gravity=0.0)
    pos, mass = bench.lattice_disc(RADIUS, dx, RHO0)
    rho = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    return bench, cfg, pos, mass, rho, acc


def test_surface_tension_force_is_radial_and_inward(droplet):
    """The one property the merged benchmark did check, pinned as a test."""
    _, cfg, pos, mass, rho, acc = droplet
    r = torch.linalg.vector_norm(pos, dim=-1)
    outward = pos / r[:, None]
    tangential = torch.stack((-outward[:, 1], outward[:, 0]), dim=1)
    a_r = (acc * outward).sum(dim=1)
    a_t = (acc * tangential).sum(dim=1)
    band = (r > 0.9 * RADIUS) & (torch.linalg.vector_norm(acc, dim=-1) > 0.0)
    assert int(band.sum()) > 20
    radial_fraction = float((a_r[band].abs()
                             / (a_r[band].abs() + a_t[band].abs())).mean())
    inward_fraction = float((a_r[band] < 0).to(torch.float64).mean())
    # Measured at this resolution: 0.57 radial, 0.73 inward. The bounds leave
    # room for the lattice-orientation noise the model is known to have, while
    # still failing if the force stops pointing at the centre.
    assert radial_fraction > 0.45
    assert inward_fraction > 0.6


def test_surface_tension_net_momentum_is_near_zero(droplet):
    """CSF is a single-particle force, so conservation is approximate, not exact."""
    _, cfg, pos, mass, rho, acc = droplet
    momentum = (mass[:, None] * acc).sum(dim=0)
    scale = (mass * torch.linalg.vector_norm(acc, dim=-1)).sum()
    residual = float(torch.linalg.vector_norm(momentum) / scale)
    # Measured 8.6e-3 at R/dx = 10.
    assert residual < 0.05


def _discrete_laplace(bench, pos, mass, rho, acc, cfg):
    """Colour-jump integral, band curvature and the integrated pressure jump."""
    _, kappa, gmag, is_surface = bench.csf_diagnostics(
        pos, mass, rho, cfg, SIGMA)
    r = torch.linalg.vector_norm(pos, dim=-1)
    a_r = (acc * pos / r[:, None]).sum(dim=1)
    edges = torch.linspace(0.0, 1.2 * RADIUS, max(12, LEVELS))
    centre = 0.5 * (edges[1:] + edges[:-1])
    delta = edges[1:] - edges[:-1]
    gmag_shell = bench.radial_bin(gmag, r, edges)
    ar_shell = bench.radial_bin(a_r, r, edges)
    finite = torch.isfinite(gmag_shell)
    colour_jump = float((gmag_shell[finite] * delta[finite]).sum())
    integrand = torch.where(torch.isfinite(ar_shell), -RHO0 * ar_shell,
                            torch.zeros_like(ar_shell))
    dp = float(torch.trapezoid(integrand, centre))
    band = is_surface & (r > 0.8 * RADIUS)
    kappa_band = float(kappa[band].mean()) if band.any() else float("nan")
    return colour_jump, kappa_band, dp


@pytest.mark.xfail(strict=True, reason=(
    "open gap: the colour field has no exterior phase, so |grad c| integrates "
    "to ~0.4 instead of 1 across the interface (docs/benchmarks/"
    "laplace-verification.md)"))
def test_colour_jump_integral_is_unity(droplet):
    bench, cfg, pos, mass, rho, acc = droplet
    colour_jump, _, _ = _discrete_laplace(bench, pos, mass, rho, acc, cfg)
    assert abs(colour_jump - 1.0) <= 0.05


@pytest.mark.xfail(strict=True, reason=(
    "open gap: the discrete curvature of a circle is ~-0.14/R at R/dx=10 "
    "instead of 1/R (docs/benchmarks/laplace-verification.md)"))
def test_curvature_of_a_circle_is_one_over_R(droplet):
    bench, cfg, pos, mass, rho, acc = droplet
    _, kappa_band, _ = _discrete_laplace(bench, pos, mass, rho, acc, cfg)
    assert abs(kappa_band * RADIUS - 1.0) <= 0.15


@pytest.mark.xfail(strict=True, reason=(
    "open gap: the integrated CSF pressure jump is +27% at R/dx=10 and does "
    "not converge with resolution (docs/benchmarks/laplace-verification.md)"))
def test_laplace_pressure_equals_sigma_over_R(droplet):
    bench, cfg, pos, mass, rho, acc = droplet
    _, _, dp = _discrete_laplace(bench, pos, mass, rho, acc, cfg)
    assert abs(dp / (SIGMA / RADIUS) - 1.0) <= 0.15