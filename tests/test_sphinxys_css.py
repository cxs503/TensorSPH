"""Tests for the SPHinXsys surface-tension stress port (``tensorsph.sphinxys_css``).

The tests pin the three things the port has to get right *before* its physics can
be trusted, each against an exact reference rather than a recorded number:

* the mixture weighting ``f + f' = 1`` (contact fraction),
* the bulk is blind to the interface (``grad c = 0`` inside a phase),
* the ported flux conserves momentum exactly (structure of the upstream model),
* the surface delta normalisation and the Laplace reading at the upstream
  density ratio (regression lock on the measured values),
* the capillary time-step formula of ``SurfaceTensionTimeStep``.
"""
from __future__ import annotations

import math

import torch

from tensorsph import sphinxys_css as sc
from tensorsph.neighbor_search import cell_linked_pairs

RADIUS = 0.1
RATIO = 16          # R/dx; small enough for a unit test, big enough to resolve 2h
DX = RADIUS / RATIO
H = sc.SPHINXSYS_H_SPACING_RATIO * DX
SIGMA = 0.072
RHO_W = 1000.0
RHO_A = 1.0


def _setup(rho_air: float = RHO_A, rings: int = 6):
    span = RADIUS + (rings + 1) * DX
    axis = torch.arange(-span, span + 0.5 * DX, DX, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    dist = torch.linalg.vector_norm(pts, dim=1)
    keep = dist <= RADIUS + rings * DX + 1e-12
    pos = pts[keep].clone()
    r = dist[keep]
    phase = (r > RADIUS + 1e-12).to(torch.int64)
    rho0 = torch.where(phase == 0, torch.tensor(RHO_W, dtype=pos.dtype),
                       torch.tensor(rho_air, dtype=pos.dtype))
    vol = torch.full((pos.shape[0],), DX * DX, dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    i, j = pairs.unbind(dim=1)
    same = phase[i] == phase[j]
    frac = sc.contact_fraction(rho0, torch.where(phase == 0, rho_air, RHO_W))
    return pos, r, phase, rho0, vol, pairs[same], pairs[~same], frac


def test_contact_fraction_sums_to_one():
    a = torch.tensor([1000.0, 1.0, 7.0], dtype=torch.float64)
    b = torch.tensor([1.0, 1000.0, 3.0], dtype=torch.float64)
    total = sc.contact_fraction(a, b) + sc.contact_fraction(b, a)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-15)


def test_bulk_colour_gradient_vanishes():
    """A particle deeper than 2h inside the water sees no air: grad c must be 0."""
    pos, r, phase, rho0, vol, _pw, pa, frac = _setup()
    grad = sc.contact_colour_gradient(pos, vol, pa, frac, H)
    mag = torch.linalg.vector_norm(grad, dim=1)
    bulk = (phase == 0) & (r < RADIUS - 2.0 * H)
    assert bool(bulk.any())
    assert float(mag[bulk].max()) == 0.0


def test_delta_normalisation_is_density_ratio_dependent():
    """``int |grad c| dn`` is the quantity the Laplace law needs to be 1.

    Measured values (dx = R/16): 0.96 at 1000:1 and 0.49 at 1:1 -- the literal
    ``2 f_k`` prefactor of the upstream model is normalised only in the
    one-sided (high density ratio) limit.  The test locks the measurement so a
    future change to the weighting cannot silently move it.
    """
    out = {}
    for rho_air in (RHO_A, RHO_W):
        pos, _r, phase, rho0, vol, _pw, pa, frac = _setup(rho_air=rho_air)
        grad = sc.contact_colour_gradient(pos, vol, pa, frac, H)
        mag = torch.linalg.vector_norm(grad, dim=1)
        water = phase == 0
        out[rho_air] = float((mag[water] * vol[water]).sum()) / (2.0 * math.pi * RADIUS)
    assert abs(out[RHO_A] - 0.9605) < 0.02
    assert abs(out[RHO_W] - 0.4894) < 0.02
    assert out[RHO_A] / out[RHO_W] > 1.8     # factor ~2 = the 2 f_k difference


def test_stress_flux_conserves_momentum_without_hourglass():
    """The upstream two-point stress flux is momentum conserving by construction."""
    pos, _r, _phase, rho0, vol, pw, pa, frac = _setup()
    mass = rho0 * vol
    _acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pw, pa, SIGMA, H, frac=frac, hourglass=False)
    acc_in = sc.inner_stress_force(pos, mass, rho0, vol, pw, fields, SIGMA, H, hourglass=False)
    acc_co = sc.contact_stress_force(pos, mass, rho0, vol, pa, fields, frac, SIGMA, H,
                                     hourglass=False)
    net = (mass[:, None] * (acc_in + acc_co)).sum(dim=0)
    scale = float((mass * torch.linalg.vector_norm(acc_in + acc_co, dim=1)).sum())
    assert float(torch.linalg.vector_norm(net)) / scale < 1e-12


def test_hourglass_penalty_conserves_momentum():
    """The penalty is conservative *if* the pair is assembled antisymmetrically.

    ``H_ij`` is even under ``i <-> j`` (both ``T_i + T_j`` and ``M ∝ (g.e) g (x) e``
    are), so ``H . grad W`` is odd -- the same parity as ``(S_i + S_j) . grad W``.
    Both terms must therefore enter the ``j`` end with the same sign; upstream does
    exactly that (its symmetric loop plays the ``j`` role), so a residual here is a
    port defect, not a property of the model.  Before the round-3 fix this test
    measured 0.73 (see ``out_single_phase_completion/gate.json``,
    ``hourglass_pair_order.pair_assembly.legacy_pair_order_leak``).
    """
    pos, _r, _phase, rho0, vol, pw, pa, frac = _setup()
    mass = rho0 * vol
    for hourglass in (False, True):
        acc, _fields = sc.surface_tension_acceleration(
            pos, mass, rho0, vol, pw, pa, SIGMA, H, frac=frac, hourglass=hourglass)
        net = (mass[:, None] * acc).sum(dim=0)
        scale = float((mass * torch.linalg.vector_norm(acc, dim=1)).sum())
        assert float(torch.linalg.vector_norm(net)) / scale < 1e-12, hourglass


def test_pair_penalty_is_antisymmetric_for_any_state():
    """Structural: a *pair* cancels for a generic (non-physical) field state.

    Three particles, two pairs, arbitrary colour gradients and no interface: the
    hourglass flux of the two ends must cancel to round-off.  This is the property
    the pre-round-3 ``-w (flux - hg)`` assembly violated.
    """
    h = H
    pos = torch.tensor([[0.0, 0.0], [0.8 * h, 0.0], [0.0, 0.6 * h]], dtype=torch.float64)
    vol = torch.full((3,), DX * DX, dtype=torch.float64)
    rho = torch.full((3,), RHO_W, dtype=torch.float64)
    mass = rho * vol
    pairs = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    grad = torch.tensor([[1.0, 0.3], [-0.2, 1.0], [0.5, -0.7]],
                        dtype=torch.float64) * 1.0e3
    fields = sc.surface_stress(grad, SIGMA)
    acc = sc.inner_stress_force(pos, mass, rho, vol, pairs, fields, SIGMA, h,
                               hourglass=True)
    net = (mass[:, None] * acc).sum(dim=0)
    scale = float((mass * torch.linalg.vector_norm(acc, dim=1)).sum())
    assert float(torch.linalg.vector_norm(net)) / scale < 1e-14


def test_stress_reading_reproduces_laplace_within_gate():
    """``Delta p = int |S_t| dn / R`` must be within 15% of ``sigma/R``."""
    pos, _r, phase, rho0, vol, pw, pa, frac = _setup()
    mass = rho0 * vol
    _acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pw, pa, SIGMA, H, frac=frac)
    water = phase == 0
    interface_area = 2.0 * math.pi * RADIUS
    delta_integral = float((fields.delta_s[water] * vol[water]).sum()) / interface_area
    dp = SIGMA * delta_integral / RADIUS
    assert abs(dp - SIGMA / RADIUS) / (SIGMA / RADIUS) < 0.15


def test_capillary_dt_matches_upstream_formula():
    h = H
    got = sc.capillary_dt_limit(RHO_W, h, SIGMA)
    assert abs(got - 0.6 * math.sqrt(RHO_W * h ** 3 / (2.0 * math.pi * SIGMA))) < 1e-18
    # TensorSPH convention (cfl 0.25, no 2 pi) differs by cfl ratio sqrt(2 pi)/... :
    tensorph = 0.25 * math.sqrt(RHO_W * h ** 3 / SIGMA)
    assert abs(got / tensorph - 1.0) < 0.2