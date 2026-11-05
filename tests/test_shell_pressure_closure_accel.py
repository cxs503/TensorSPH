"""Gates for the *non-pressure* body term of the shell pressure closure.

``free_surface_pressure`` inverts the density deficiency of a free surface by
extrapolating the bulk pressure onto the shell. Adami, Ihmsen & Adams (2012)
eq. (11) writes that extrapolation with ``(g - a_i)``, where ``a_i`` is the
*non-pressure* acceleration of the receiver: for a static wall ``a_w = 0`` and
the body term is gravity alone, but a free-surface particle of a drop carries
the continuum surface force, and dropping it there removes exactly the term that
has to transport the capillary jump across the shell. In zero gravity the
gravity-only form degenerates to a kernel average that mixes the bulk value with
the clamped zeros, so the shell can never hold ``sigma/R``.

The gates below keep the operator algebraic (no relaxation), so they are exact
statements rather than tolerances on a measurement:

* the body term must reproduce a *linear* pressure field with a non-zero
  ``external`` acceleration to round-off, and must fail to when the term is
  missing or mis-signed;
* ``external=None`` must reproduce the gravity-only operator bit for bit, so
  every previously published number stays reproducible;
* the inverse Tait density must invert :func:`tait_pressure`, which is what
  makes ``surface_pressure_density="tait"`` self-consistent.

``tests/test_free_surface_pressure_closure.py`` pins the gravity-only operator;
this file adds the ``external``/``tait`` axis and the production wiring.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    free_surface_pressure,
    pair_kernels,
    rhs,
    rhs_all_pairs,
    tait_density,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

RHO0 = 1000.0


def _column(ny: int = 20, nx: int = 5, dx: float = 0.005,
            dtype=torch.float64):
    xs = dx * torch.arange(nx, dtype=dtype)
    ys = dx * torch.arange(ny, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)


def _operators(pos, h: float, cfg_h_factor: float = 1.3):
    pairs = cell_linked_pairs(pos, 2.0 * cfg_h_factor * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    wij, _ = pair_kernels(delta, h)
    return delta, wij, i, j


def _linear_case(ay: float = 3.0, p0: float = 150.0, nx: int = 5, ny: int = 20,
                 dx: float = 0.005, free_rows: int = 3):
    """Analytic field with ``grad p = rho0 * A``, ``g = 0``, ``a_ext = -A``.

    ``A = g - a_ext = (0, +ay)``, so ``p(y) = p0 + rho0 * ay * y`` and the
    Adami body term must reproduce it exactly for a first-layer receiver; the
    donor density is ``rho0`` there, which is what makes the statement exact.
    """
    pos = _column(ny=ny, nx=nx, dx=dx)
    h = 1.3 * dx
    p = p0 + RHO0 * ay * pos[:, 1]
    rho = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    order = torch.argsort(pos[:, 1], descending=True)
    free = torch.zeros(pos.shape[0], dtype=torch.bool)
    free[order[: nx * free_rows]] = True
    rho[free] = 0.97 * RHO0                 # only used as the receiver mask
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, gravity=0.0,
                 surface_pressure_closure=True, surface_pressure_layers=1)
    ext = torch.zeros_like(pos)
    ext[:, 1] = -ay                         # a_ext = (0, -ay)
    return pos, rho, p, free, h, cfg, ext


# --------------------------------------------------------------------------- #
# the body term: (g - a_i), not g
# --------------------------------------------------------------------------- #
def test_external_acceleration_reproduces_a_linear_field_exactly():
    """First-layer receivers of a linear field are exact to round-off.

    One pass only: after the first pass the filled particles donate with their
    deficient (0.97 rho0) density, which is a property of the donor set, not of
    the body term (the same dilution exists for the gravity-only operator).
    """
    pos, rho, p, free, h, cfg, ext = _linear_case()
    delta, wij, i, j = _operators(pos, h)
    out = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg,
                                external=ext)
    filled = free & (p > 1.0)
    assert int(filled.sum()) > 0
    rel = (out[filled] - p[filled]) / p[filled]
    assert float(rel.abs().max()) < 1e-10
    assert torch.equal(out[~free], p[~free])


def test_missing_external_acceleration_breaks_the_exactness():
    """Negative control: the same field without (or with a flipped) a_i fails."""
    pos, rho, p, free, h, cfg, ext = _linear_case()
    delta, wij, i, j = _operators(pos, h)
    filled = free & (p > 1.0)
    good = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg,
                                 external=ext)
    dropped = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)
    flipped = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg,
                                    external=-ext)
    reference = p[filled]
    assert float((good[filled] - reference).abs().max()) < 1e-8
    assert float((dropped[filled] - reference).abs().max()) > 1.0
    assert float((flipped[filled] - reference).abs().max()) > 1.0
    # and the error is of the order of the body term it is missing
    missed = float((dropped[filled] - reference).abs().max())
    assert 0.5 * RHO0 * 3.0 * h < missed < 4.0 * RHO0 * 3.0 * h


def test_zero_external_is_bit_for_bit_the_gravity_only_operator():
    """``external=None`` and a zero acceleration are the same operator."""
    pos, rho, p, free, h, cfg, _ = _linear_case()
    cfg_g = Config(**{**cfg.__dict__, "gravity": 9.81})
    delta, wij, i, j = _operators(pos, h)
    none = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg_g)
    zeros = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg_g,
                                  external=torch.zeros_like(pos))
    assert torch.equal(none, zeros)


def test_gravity_only_closure_extends_the_field_without_a_gradient():
    """Zero gravity: the gravity-only operator copies, it does not slope.

    Uniform bulk pressure ``p_bulk`` on the interior, ``g = 0``: with only the
    interior as donor every receiver gets ``p_bulk`` back, i.e. a *constant*
    extension (zero gradient) -- the shell then carries no information about the
    capillary force, so nothing pulls the outer particles towards the p = 0
    boundary condition. Feeding the non-pressure acceleration into the body term
    gives the first-order ramp ``rho_j A.(r_i - r_j)`` instead, which is the
    term that transports ``sigma/R`` across the shell.
    """
    pos, rho, p, free, h, cfg, _ = _linear_case(ay=0.0, p0=0.0)
    p_bulk = 360.0                              # sigma/R of the drop case
    p = torch.where(free, torch.zeros_like(p), torch.full_like(p, p_bulk))
    delta, wij, i, j = _operators(pos, h)
    flat = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)
    filled = free & (flat > 0.0)
    assert int(filled.sum()) > 0
    assert float((flat[filled] - p_bulk).abs().max()) < 1e-9   # no gradient

    ay = 5.0e3                                  # a_ext = (0, -ay), g = 0
    accel = torch.zeros_like(pos)
    accel[:, 1] = ay
    ramp = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg,
                                 external=-accel)
    rise = ramp[filled] - p_bulk
    assert float(rise.min()) > 0.0                      # outward gradient
    assert float(rise.max()) > 0.5 * RHO0 * ay * h      # of order rho*A*h


# --------------------------------------------------------------------------- #
# inverse Tait density
# --------------------------------------------------------------------------- #
def test_tait_density_inverts_tait_pressure():
    cfg = Config(rho0=RHO0, sound_speed=20.0, gamma=7.0)
    ratio = torch.linspace(0.95, 1.03, 64, dtype=torch.float64)
    rho = RHO0 * ratio
    p = tait_pressure(rho, cfg)
    back = tait_density(p, cfg)
    compressed = rho >= RHO0                    # where Tait is not clamped
    assert bool(compressed.any())
    rel = (back[compressed] - rho[compressed]) / rho[compressed]
    assert float(rel.abs().max()) < 1e-12
    # the clamp maps every tensile value onto rho0, never below it
    tensile = tait_density(torch.full((4,), -50.0, dtype=torch.float64), cfg)
    assert torch.allclose(tensile, torch.full((4,), RHO0, dtype=torch.float64))


def test_closed_pressure_pairs_with_an_isentrope_density():
    """``p / rho(tait)^gamma`` of a filled particle equals the bulk value."""
    cfg = Config(rho0=RHO0, sound_speed=20.0, gamma=7.0)
    bulk = RHO0 * cfg.sound_speed ** 2 / cfg.gamma
    p = torch.full((8,), 360.0, dtype=torch.float64)
    rho = tait_density(p, cfg)
    assert torch.allclose(p, bulk * ((rho / RHO0).pow(cfg.gamma) - 1.0))
    # pairing the same pressure with the *summation* density would be a
    # different ratio, i.e. an inconsistent 1/rho^2 in the pressure force
    assert not torch.allclose(p, bulk * ((rho * 0.98 / RHO0).pow(cfg.gamma) - 1.0))


# --------------------------------------------------------------------------- #
# production wiring
# --------------------------------------------------------------------------- #
def _hex_drop(radius: float = 0.02, dx: float = 0.002, dtype=torch.float32):
    """Hexagonal drop whose mass is calibrated so the core sums to rho0."""
    k = int(radius / dx) + 2
    xs = (torch.arange(-k, k + 1, dtype=dtype) * dx)
    ys = (torch.arange(-k, k + 1, dtype=dtype) * dx * math.sqrt(3.0) / 2.0)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    x = gx.clone()
    x[1::2] += 0.5 * dx
    pos = torch.stack((x.reshape(-1), gy.reshape(-1)), dim=1)
    keep = torch.linalg.vector_norm(pos, dim=1) <= radius
    pos = pos[keep].contiguous()
    h = 1.3 * dx
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx * math.sqrt(3.0) / 2.0,
                      dtype=dtype)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    from tensorsph.droplet_impact import density_from_neighbor_pairs
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    mass = mass / (rho[r < 0.5].mean() / RHO0)
    return pos, mass, h


DROP_CFG = dict(h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                gravity=0.0, viscosity=0.01, surface_tension_coefficient=7.2,
                csf_scheme="consistent", csf_curvature="fit_quadratic")


def test_production_rhs_fills_the_shell_only_with_the_nonpressure_term():
    """Single CSF evaluation on a lattice drop, no time stepping.

    The un-relaxed drop has ``p = 0`` everywhere (its core sums to ``rho0``),
    so the shell value is produced by the closure's body term alone: the
    gravity-only operator returns exactly zero, the non-pressure operator
    returns the Adami ramp, of the order of ``sigma/R``.
    """
    pos, mass, h = _hex_drop()
    radius = 0.02
    cfg_off = Config(dx=0.002, **DROP_CFG)
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    shell = (r >= 0.9) & (r < 1.05)
    sigma_over_r = 7.2 / radius

    results = {}
    for name, body in (("gravity", "gravity"), ("nonpressure", "nonpressure")):
        cfg = Config(dx=0.002, surface_pressure_closure=True,
                     surface_pressure_body_force=body, **DROP_CFG)
        _, pressure, acc = rhs(pos, torch.zeros_like(pos), mass, cfg)
        assert bool(torch.isfinite(acc).all())
        results[name] = pressure
    assert float(results["gravity"][shell].max()) < 0.05 * sigma_over_r
    assert float(results["nonpressure"][shell].max()) > 0.25 * sigma_over_r
    assert float(results["nonpressure"][shell].max()) < 5.0 * sigma_over_r
    # untouched configuration: the closure is off by default and the new knobs
    # must not move a single number when it is
    _, p_off, acc_off = rhs(pos, torch.zeros_like(pos), mass, cfg_off)
    cfg_knobs = Config(dx=0.002, surface_pressure_body_force="nonpressure",
                       surface_pressure_density="tait", **DROP_CFG)
    _, p_knobs, acc_knobs = rhs(pos, torch.zeros_like(pos), mass, cfg_knobs)
    assert torch.equal(p_off, p_knobs) and torch.equal(acc_off, acc_knobs)


@pytest.mark.parametrize("donors,layers,body,density",
                         [("interior", 2, "nonpressure", "sum"),
                          ("interior", 2, "nonpressure", "tait"),
                          ("all", 1, "nonpressure", "tait"),
                          ("interior", 1, "gravity", "tait")])
def test_every_closure_combination_is_finite_and_dense_path_agrees(
        donors, layers, body, density):
    pos, mass, h = _hex_drop(radius=0.01, dx=0.001)
    cfg = Config(dx=0.001, surface_pressure_closure=True,
                 surface_pressure_donors=donors,
                 surface_pressure_layers=layers,
                 surface_pressure_body_force=body,
                 surface_pressure_density=density, **DROP_CFG)
    rho, pressure, acc = rhs(pos, torch.zeros_like(pos), mass, cfg)
    assert bool(torch.isfinite(pressure).all()) and bool(torch.isfinite(acc).all())
    assert float(pressure.min()) >= 0.0
    rho_d, pressure_d, acc_d = rhs_all_pairs(pos, torch.zeros_like(pos), mass,
                                             cfg)
    if donors == "interior":
        # The dense reference is an independent formulation of the same
        # operator (its a_i is built from the upper-triangular pair set), so the
        # two pressure fields have to agree to the noise that is left. The
        # literal-"all" variant is dominated by the free particles' own body
        # terms and is only checked for finiteness here.
        scale = max(float(pressure_d.abs().max()), 1.0)
        assert float((pressure - pressure_d).abs().mean()) < 0.02 * scale, (
            donors, body, density)


def test_config_validation_rejects_unknown_closure_knobs():
    from tensorsph.droplet_impact import validate_config

    with pytest.raises(ValueError):
        validate_config(Config(surface_pressure_body_force="cfd"))
    with pytest.raises(ValueError):
        validate_config(Config(surface_pressure_density="kernel"))
    validate_config(Config(surface_pressure_body_force="nonpressure",
                           surface_pressure_density="tait"))
