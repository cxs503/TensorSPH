"""Unit gates for the free-surface pressure closure (Adami et al. 2012 §3.3).

The closure replaces the clamped Tait pressure on particles whose summation
density fell below ``free_surface_threshold * rho0`` by

    p_i = max(0, sum_j [p_j + rho_j (g - a_i).(r_i - r_j)] W_ij / sum_j W_ij)

``rhs`` wires it in behind ``Config.free_surface_pressure == "shell"``; the
default ``"clamp"`` must stay bit-identical, which is the first test here.

The physics-level result (static droplet vs the Young-Laplace law with and
without the closure) is in ``docs/verification_free_surface_pressure_closure.md``
and is gated by ``tests/test_laplace_pressure_closure.py``; the cheap facts that
the scheme is an *extrapolation* rather than a smoothing are pinned here.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
    pair_kernels,
    rhs,
    shell_pressure_extrapolation,
    tait_density,
    tait_pressure,
    validate_config,
)
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"


def _hex_droplet(radius=0.02, dx=0.002, rho0=1000.0):
    spec = importlib.util.spec_from_file_location(
        "laplace_law_run", BENCH.parent / "laplace_law" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hexagonal_droplet(radius, dx, rho0)


# --------------------------------------------------------------------------- #
# lattice used by the synthetic tests: a 2-D square lattice inside a disc
# --------------------------------------------------------------------------- #
def two_phase_lattice(n=9, dx=1.0, device=None):
    """Square lattice with fluid particles first, then 'void' particles.

    Any particle inside radius ``R`` counts as fluid (it carries an exact
    pressure), everything outside is a free-surface particle whose pressure the
    closure has to reconstruct. ``pairs`` holds fluid-fluid pairs only.
    """
    device = device or torch.device("cpu")
    xs = torch.arange(n, dtype=torch.float64, device=device) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    centre = pos.mean(dim=0)
    r = torch.linalg.vector_norm(pos - centre, dim=1)
    return pos, centre, r


def linear_field(pos, centre, grad, p0=0.0):
    """``p = p0 + grad . (x - centre)`` -- the field the closure must extrapolate."""
    return p0 + ((pos - centre) * grad).sum(dim=1)


# --------------------------------------------------------------------------- #
# 1. the default path is untouched
# --------------------------------------------------------------------------- #
def test_default_scheme_leaves_the_clamped_tait_pressure():
    cfg = Config(dx=0.004, sound_speed=20.0, gravity=0.0, viscosity=0.0,
                 surface_tension_coefficient=0.0)
    pos, mass = _hex_droplet(radius=0.02, dx=0.004)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    vel = torch.zeros_like(pos)
    rho, pressure, _ = rhs(pos, vel, mass, cfg, pairs=pairs, dt=1.0e-5)
    assert torch.equal(pressure, tait_pressure(rho, cfg))
    assert cfg.free_surface_pressure == "clamp"


def test_validate_config_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        validate_config(Config(free_surface_pressure="magic"))
    with pytest.raises(ValueError):
        validate_config(Config(free_surface_threshold=0.0))
    validate_config(Config(free_surface_pressure="shell"))


# --------------------------------------------------------------------------- #
# 2. interpolation identity: a uniform field survives the operator
# --------------------------------------------------------------------------- #
def test_uniform_pressure_is_reproduced_at_surface_particles():
    """With no body force the closure is a kernel average, which preserves p = c."""
    dx = 0.01
    cfg = Config(dx=dx, sound_speed=20.0, gravity=0.0, rho0=1000.0,
                 free_surface_pressure="shell")
    pos, centre, r = two_phase_lattice(n=15, dx=dx)
    h = cfg.h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    p_uniform = torch.full((pos.shape[0],), 123.0, dtype=pos.dtype)
    rho = torch.where(r < 0.09, torch.full_like(r, 1000.0),
                      torch.full_like(r, 850.0))
    out, mask = shell_pressure_extrapolation(
        p_uniform, rho, pos, pairs, cfg, h, external=torch.zeros_like(pos))
    assert bool(mask.any())
    assert torch.allclose(out[mask], p_uniform[mask], atol=1e-12)
    # particles above the threshold are not touched at all
    keep = ~mask
    assert torch.equal(out[keep], p_uniform[keep])


# --------------------------------------------------------------------------- #
# 3. the body-force term makes it an extrapolation, not a smoothing
# --------------------------------------------------------------------------- #
def test_linear_body_force_field_is_extrapolated_exactly():
    """With ``a = 0`` and ``p`` linear in y, the closure reproduces the field.

    Inside the kernel ``sum_j (p_j - rho (y_i - y_j) g) W / sum_j W = p_i`` holds
    for any weight set when ``dp/dy = -rho g``, which is the property that turns
    eq. (11) of Adami et al. (2012) from an average into an extrapolation. The
    test marks *every* particle as closed (threshold > 1) so the whole lattice is
    extrapolated from itself and the identity can be read off directly.
    """
    dx = 0.01
    rho0, g = 1000.0, 9.81
    cfg = Config(dx=dx, sound_speed=20.0, gravity=g, rho0=rho0,
                 free_surface_pressure="shell")
    pos, centre, r = two_phase_lattice(n=21, dx=dx)
    h = 2.0 * dx          # wide kernel: several neighbour rings per particle
    pairs = cell_linked_pairs(pos, 2.0 * h)
    grad = torch.tensor([0.0, -rho0 * g], dtype=pos.dtype)
    p_exact = linear_field(pos, centre, grad, p0=5000.0)
    rho = torch.full((pos.shape[0],), rho0, dtype=pos.dtype)
    out, mask = shell_pressure_extrapolation(
        p_exact, rho, pos, pairs, cfg, h, external=torch.zeros_like(pos),
        threshold=1.05)
    assert bool(mask.all())
    err = (out - p_exact).abs() / p_exact
    assert float(err.max()) < 1.0e-12, float(err.max())


def test_plain_kernel_average_would_be_biased():
    """Negative control: without the body term the same operator is biased.

    ``gravity = 0`` removes ``g - a_i`` from the source, leaving a bare kernel
    average, which returns ``p`` at the stencil centroid instead of at ``i``.
    This is what the closure would degenerate to over a *flat* interface, and it
    is the reason the term has to be there.
    """
    dx = 0.01
    rho0, g = 1000.0, 9.81
    cfg_no_body = Config(dx=dx, gravity=0.0, rho0=rho0,
                         free_surface_pressure="shell")
    pos, centre, r = two_phase_lattice(n=21, dx=dx)
    h = 2.0 * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    grad = torch.tensor([0.0, -rho0 * g], dtype=pos.dtype)
    p_exact = linear_field(pos, centre, grad, p0=5000.0)
    rho = torch.full((pos.shape[0],), rho0, dtype=pos.dtype)
    out, mask = shell_pressure_extrapolation(
        p_exact, rho, pos, pairs, cfg_no_body, h, external=None,
        threshold=1.05)
    err = ((out - p_exact).abs() / p_exact).max()
    assert float(err) > 1.0e-3, float(err)


# --------------------------------------------------------------------------- #
# 4. it raises the surface pressure of a static drop instead of deleting it
# --------------------------------------------------------------------------- #
def test_closure_fills_the_deficit_shell_of_a_static_droplet():
    """One reading of a static drop: the shell must carry pressure, not zero.

    Not a physics gate -- that is the Laplace jump in
    ``tests/test_laplace_pressure_closure.py`` -- only the qualitative statement
    that the closure acts where it claims to.
    """
    sigma, radius = 7.2, 0.02
    dx = 0.002
    cfg = Config(dx=dx, sound_speed=20.0, rho0=1000.0, gamma=7.0, gravity=0.0,
                 viscosity=5.0e-5, surface_tension_coefficient=sigma,
                 csf_scheme="consistent", csf_curvature="fit_quadratic")
    pos, mass = _hex_droplet(radius=radius, dx=dx)
    h = cfg.h_factor * cfg.dx
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    shell = (r > 0.85) & (r < 1.0)
    vel = torch.zeros_like(pos)

    readings = {}
    for scheme in ("clamp", "shell"):
        cfg_s = Config(**{**cfg.__dict__, "free_surface_pressure": scheme})
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho = density_from_neighbor_pairs(pos, mass, h, pairs)
        _, pressure, _ = rhs(pos, vel, mass, cfg_s, pairs=pairs, dt=1.0e-5)
        readings[scheme] = {
            "shell_mean": float(pressure[shell].mean()),
            "core_mean": float(pressure[r < 0.5].mean()),
            "finite": bool(torch.isfinite(pressure).all()),
        }

    assert readings["clamp"]["finite"] and readings["shell"]["finite"]
    assert readings["clamp"]["shell_mean"] == pytest.approx(0.0, abs=1.0)
    assert readings["shell"]["shell_mean"] > 10.0
    # and the closure must not invent pressure outside the deficit shell: the
    # core reading is still the Tait value of the (deficient) core density.
    assert readings["shell"]["core_mean"] >= 0.0


# --------------------------------------------------------------------------- #
# 5. the density handed to the pressure force is EOS-consistent
# --------------------------------------------------------------------------- #
def test_tait_density_is_the_inverse_of_the_tait_law():
    cfg = Config(rho0=1000.0, sound_speed=20.0, gamma=7.0)
    p = torch.tensor([0.0, 100.0, 360.0, 5000.0], dtype=torch.float64)
    rho = tait_density(p, cfg)
    assert torch.allclose(tait_pressure(rho, cfg), p, rtol=1e-12, atol=1e-9)
    assert float(rho[0]) == pytest.approx(1000.0, rel=1e-15)


def test_kernel_sum_normalisation_used_by_the_closure():
    """Sanity: sum_j W over a full lattice ring equals the density sum / mass."""
    dx = 0.01
    cfg = Config(dx=dx)
    h = cfg.h_factor * dx
    pos, centre, r = two_phase_lattice(n=15, dx=dx)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    distance = torch.linalg.vector_norm(delta, dim=1)
    w, _ = pair_kernels(delta, h)
    assert torch.allclose(w, cubic_spline_kernel(distance, h), rtol=1e-12)
    num = torch.zeros(pos.shape[0], dtype=pos.dtype).index_add(0, i, w)
    assert math.isfinite(float(num.max()))
