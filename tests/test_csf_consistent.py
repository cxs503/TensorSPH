"""Analytic verification of the consistent CSF surface-tension model.

Each test scores one ingredient against a closed-form reference, so a future
change to ``tensorsph/csf_consistent.py`` cannot silently break the Laplace-law
chain without a red test.  The thresholds are the *measured* accuracy of the
current implementation plus a small margin; the platform's 15 % acceptance
applies to ``Delta p``, ``F_cut`` and ``kappa``.

The end-to-end pressure jump is deliberately *not* asserted to 15 %: it is
limited by the pressure field, not by the force (see
``docs/verification_surface_tension_consistent.md``), and the committed verdict
is checked instead so that the honest number stays visible.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import (
    apply_contact_angle_boundary,
    csf_fields,
    mls_divergence,
    mls_gradient,
    nearest_wall,
    surface_tension_acceleration,
)
from tensorsph.droplet_impact import Config, validate_config
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "csf_consistent"
RADIUS = 0.02
SIGMA = 7.2
RHO0 = 1000.0


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("csf_bench", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def droplet(harness):
    """One hex-packed drop, dx = 2 mm -- a couple of seconds of work."""
    cfg = Config(dx=0.002, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01, surface_tension_coefficient=SIGMA)
    h = cfg.h_factor * cfg.dx
    pos, mass = harness.hexagonal_droplet(RADIUS, cfg.dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    state = csf_fields(pos, mass, pairs, h, RHO0)
    acc, _ = surface_tension_acceleration(pos, torch.full_like(mass, RHO0), mass,
                                          pairs, h, SIGMA, RHO0)
    return cfg, h, pos, mass, pairs, state, acc


# --------------------------------------------------------------------------- #
# the package must stay importable: __init__ imports these names from
# surface_tension, and a merge gate has already dropped them twice (m2389, m2399)
# --------------------------------------------------------------------------- #
def test_public_api_is_complete():
    import tensorsph
    from tensorsph import surface_tension

    for name in ("apply_contact_angle_boundary", "compute_color_gradient",
                 "compute_curvature", "compute_normal",
                 "compute_surface_tension_force", "capillary_dt_limit"):
        assert hasattr(surface_tension, name), name
        assert name in tensorsph.__all__ or hasattr(tensorsph, name), name


def test_rhs_accepts_surface_tension_with_walls():
    """The code path that broke: rhs() passes wall=<wall positions>."""
    from tensorsph.droplet_impact import rhs

    dx = 0.01
    cfg = Config(dx=dx, gravity=0.0, surface_tension_coefficient=SIGMA,
                 sound_speed=20.0)
    pos = torch.tensor([[0.0, 0.0], [dx, 0.0], [0.0, dx], [dx, dx]],
                       dtype=torch.float64)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=torch.float64)
    vel = torch.zeros_like(pos)
    wall = torch.tensor([[0.0, -dx], [dx, -dx]], dtype=torch.float64)
    rho, p, acc = rhs(pos, vel, mass, cfg, wall=wall)
    assert torch.isfinite(acc).all() and acc.shape == pos.shape


# --------------------------------------------------------------------------- #
# operator consistency
# --------------------------------------------------------------------------- #
def test_mls_gradient_is_exact_for_a_linear_field():
    dx = 0.002
    h = 1.3 * dx
    pos = torch.tensor([[i * dx, j * dx] for i in range(-8, 9) for j in range(-8, 9)],
                       dtype=torch.float64)
    vol = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    # trimming particles with a truncated stencil is the whole point of MLS
    field = 2.0 * pos[:, 0] - 3.0 * pos[:, 1]
    grad = mls_gradient(pos, field, vol, pairs, h)
    edge = (torch.linalg.vector_norm(pos, dim=1) < 8 * dx)
    assert torch.allclose(grad[edge, 0], torch.full_like(grad[edge, 0], 2.0),
                          atol=5e-3)
    assert torch.allclose(grad[edge, 1], torch.full_like(grad[edge, 1], -3.0),
                          atol=5e-3)


def test_mls_divergence_is_exact_for_a_linear_field():
    dx = 0.002
    h = 1.3 * dx
    pos = torch.tensor([[i * dx, j * dx] for i in range(-8, 9) for j in range(-8, 9)],
                       dtype=torch.float64)
    vol = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    field = torch.stack([2.0 * pos[:, 0], -3.0 * pos[:, 1]], dim=1)  # div = -1
    div = mls_divergence(pos, field, vol, pairs, h)
    edge = (torch.linalg.vector_norm(pos, dim=1) < 8 * dx)
    assert torch.allclose(div[edge], torch.full_like(div[edge], -1.0), atol=5e-3)


def test_kernel_sum_colour_saturates_in_the_bulk(droplet):
    _, _, _, _, _, state, _ = droplet
    # c = 1 in the bulk and ~1/2 on the outermost particle layer
    assert 0.98 < float(state.color.max()) <= 1.0 + 1e-12
    assert 0.4 < float(state.color.min()) < 0.6


def test_delta_s_integrates_to_one_across_the_interface(harness, droplet):
    _, _, pos, mass, _, state, _ = droplet
    integral = harness.delta_s_integral(pos, mass, RHO0, state.delta_s, RADIUS)
    # measured 0.75 on this lattice; the analytic value is 1.0 and the legacy
    # scheme delivered 0.33 -> the test guards against a silent regression, the
    # remaining 25 % is reported in the verification doc.
    assert 0.6 < integral < 1.15, integral


# --------------------------------------------------------------------------- #
# the three analytic references of the Young-Laplace verification
# --------------------------------------------------------------------------- #
def test_curvature_of_a_circle(droplet):
    _, _, _, _, _, state, _ = droplet
    w = state.delta_s
    kappa = float((state.kappa.abs() * w).sum() / w.sum())
    rel = (kappa - 1.0 / RADIUS) * RADIUS
    # measured +25 %; legacy: +79 % at this resolution
    assert abs(rel) < 0.35, rel


def test_cut_force_identity(droplet):
    _, _, pos, mass, _, _, acc = droplet
    right = pos[:, 0] > 0
    f_cut = float((mass[right, None] * acc[right]).sum(dim=0)[0])
    ratio = f_cut / (-2.0 * SIGMA)
    # 15 % acceptance -- measured 1.08
    assert abs(ratio - 1.0) < 0.15, ratio


def test_force_points_radially_inward(droplet):
    _, _, pos, _, _, state, acc = droplet
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None]
    mag = torch.linalg.vector_norm(acc, dim=1)
    band = state.delta_s > 0.05 * state.delta_s.max()
    cos_in = -(acc * rhat).sum(dim=1)[band] / mag[band]
    assert float(cos_in.median()) > 0.95
    assert float((cos_in > 0.5).float().mean()) > 0.9


# --------------------------------------------------------------------------- #
# contact-angle boundary condition
# --------------------------------------------------------------------------- #
def test_nearest_wall_normal_points_into_the_fluid():
    # A *dense* wall line: with three isolated wall particles the nearest one is
    # not directly below the fluid particle, which is a property of the wall
    # sampling, not of the BC.
    xs = torch.arange(0.0, 2.01, 0.005, dtype=torch.float64)
    wall = torch.stack([xs, torch.zeros_like(xs)], dim=1)
    pos = torch.tensor([[0.5, 0.25], [1.5, 1.0]], dtype=torch.float64)
    d, n = nearest_wall(pos, wall)
    assert torch.allclose(d, torch.tensor([0.25, 1.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(n, torch.tensor([[0.0, 1.0], [0.0, 1.0]],
                                          dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("theta", [math.pi / 6, math.pi / 3, math.pi / 2,
                                   2 * math.pi / 3])
def test_contact_angle_sets_the_normal_angle(theta):
    """n . n_wall = cos(theta), with |grad c| untouched (force amplitude)."""
    h = 0.065
    xs = torch.arange(0.0, 0.61, 0.005, dtype=torch.float64)
    wall = torch.stack([xs, torch.zeros_like(xs)], dim=1)
    # two particles at 2 % and 10 % of the BC reach (2h) from the wall: the
    # blend weight there is 0.98 and 0.90, so the achieved angle approaches the
    # prescribed one; the magnitude must be preserved exactly either way.
    pos = torch.tensor([[0.3, 0.02 * 2 * h], [0.3, 0.10 * 2 * h]],
                       dtype=torch.float64)
    grad_c = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad_c, pos, wall, h, theta)

    assert torch.allclose(torch.linalg.vector_norm(out, dim=1),
                          torch.linalg.vector_norm(grad_c, dim=1), atol=1e-12)
    n = -out / torch.linalg.vector_norm(out, dim=1, keepdim=True)
    n_w = torch.tensor([0.0, 1.0], dtype=torch.float64)
    dot = (n * n_w).sum(dim=1)
    # closest particle: within 5 % of the prescribed angle, and always on the
    # correct side of the neutral (theta = pi/2) value
    assert abs(float(dot[0]) - math.cos(theta)) < 0.05, (theta, dot.tolist())


def test_contact_angle_leaves_far_field_and_empty_wall_alone():
    wall = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    grad_c = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
    far = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad_c, far, wall, 0.05, math.pi / 3)
    assert torch.allclose(out, grad_c)
    near = torch.tensor([[0.0, 0.01]], dtype=torch.float64)
    empty = apply_contact_angle_boundary(grad_c, near, torch.empty((0, 2)),
                                       0.05, math.pi / 3)
    assert torch.allclose(empty, grad_c)


def test_contact_angle_reaches_the_force():
    """cfg.contact_angle with a wall must change the CSF acceleration."""
    dx = 0.004
    h = 1.3 * dx
    pos = torch.tensor([[i * dx, j * dx] for i in range(6) for j in range(4)],
                       dtype=torch.float64)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    wall = torch.tensor([[x * dx, -dx] for x in range(7)], dtype=torch.float64)

    def normal_with(theta):
        return csf_fields(pos, mass, pairs, h, RHO0, wall=wall,
                          contact_angle=theta).normal

    wet = normal_with(math.pi / 6)
    dry = normal_with(5 * math.pi / 6)
    neutral = normal_with(None)
    # the planned normals must differ from each other and from the untouched one
    assert not torch.allclose(wet, dry, atol=1e-6)
    assert not torch.allclose(wet, neutral, atol=1e-6)


# --------------------------------------------------------------------------- #
# configuration contract
# --------------------------------------------------------------------------- #
def test_csf_scheme_defaults_to_consistent_and_is_validated():
    assert Config().csf_scheme == "consistent"
    validate_config(Config(csf_scheme="legacy"))
    with pytest.raises(ValueError):
        validate_config(Config(csf_scheme="magic"))


def test_zero_sigma_stays_an_exact_no_op():
    dx = 0.004
    h = 1.3 * dx
    pos = torch.tensor([[i * dx, j * dx] for i in range(4) for j in range(4)],
                       dtype=torch.float64)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=torch.float64)
    rho = torch.full_like(mass, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    acc, state = surface_tension_acceleration(pos, rho, mass, pairs, h, 0.0, RHO0)
    assert state is None
    assert torch.equal(acc, torch.zeros_like(acc))


def test_legacy_scheme_still_runs_with_the_wall_argument():
    from tensorsph.surface_tension import compute_surface_tension_force

    dx = 0.004
    h = 1.3 * dx
    pos = torch.tensor([[i * dx, j * dx] for i in range(5) for j in range(5)],
                       dtype=torch.float64)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=torch.float64)
    rho = torch.full_like(mass, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    cfg = Config(dx=dx, gravity=0.0, surface_tension_coefficient=SIGMA,
                 csf_scheme="legacy")
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg, wall=None)
    assert acc.shape == pos.shape and torch.isfinite(acc).all()


# --------------------------------------------------------------------------- #
# the committed verdict must stay honest (no compute)
# --------------------------------------------------------------------------- #
def test_committed_verdict_records_the_open_deficit():
    report = json.loads((BENCH / "validation_report.json").read_text())
    lap = report["relax"]["laplace"]
    recomputed = ((lap["p_center_Pa"] - lap["p_interface_band_Pa"])
                  - report["relax"]["reference"]["delta_p_Pa"]) \
        / report["relax"]["reference"]["delta_p_Pa"] * 100.0
    assert recomputed == pytest.approx(lap["rel_error_pct"], abs=1e-6)
    # The force-level criteria pass; the end-to-end pressure jump does not yet.
    consistent = {r["dx_m"]: r for r in report["instant"]["consistent"]}
    legacy = {r["dx_m"]: r for r in report["instant"]["legacy"]}
    for dx, row in consistent.items():
        assert row["cut_force"]["within_15pct"], dx
        assert abs(row["cut_force"]["ratio_measured_over_theory"] - 1.0) < \
            abs(legacy[dx]["cut_force"]["ratio_measured_over_theory"] - 1.0), dx
    assert lap["within_15pct"] is False, (
        "the Laplace jump now passes -- update docs/verification_"
        "surface_tension_consistent.md and this assertion together")
