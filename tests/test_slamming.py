"""Verification surface for the slamming module (campaign node p129w1r2-n08).

The tests are grouped exactly as the module's three layers:

* analytic Wagner added-mass / force-coefficient identities (tests 1-6),
* the delta-SPH and MLS operators, each with a *negative control* that has to
  fail for the test to mean anything (tests 7-12),
* the solver wiring: delegation to the parent when both switches are off, the
  force integral, mass bookkeeping and the noise metric (tests 13-19).
"""

import math

import pytest
import torch

from tensorsph.droplet_impact import cubic_spline_kernel, pair_kernels
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.slamming import (
    SlammingConfig,
    SlammingImpact,
    added_mass_force,
    added_mass_law,
    added_mass_per_length,
    curved_section_width,
    delta_sph_density_rate,
    mls_gradient,
    pressure_noise,
    pressure_shape_l2,
    slamming_force_coefficient,
    wagner_shape,
)
from tensorsph.wedge_entry import WedgeImpact, wagner_pmax, wet_half_width

DTYPE = torch.float64


def lattice(dx=0.01, half=20, h_factor=1.3):
    h = h_factor * dx
    xs = torch.arange(-half, half + 1, dtype=DTYPE) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    _, gw = pair_kernels(delta, h)
    m = 1000.0 / float(cubic_spline_kernel(torch.zeros(1, dtype=DTYPE), h)[0])
    vol = torch.full((pos.shape[0],), m / 1000.0, dtype=DTYPE)
    return pos, i, j, delta, gw, vol, m, h


# --------------------------------------------------------------------------- #
# 1-6  Wagner added-mass layer
# --------------------------------------------------------------------------- #

def test_wetted_width_is_linear_in_time():
    """Wagner: c(t) = C V t, so the log-log exponent is 1, not 1/2."""
    V, beta = 2.0, math.radians(20.0)
    c1 = wet_half_width(V, beta, 1.0e-3)
    c2 = wet_half_width(V, beta, 2.0e-3)
    assert c1 > 0.0
    assert c2 / c1 == pytest.approx(2.0, rel=1e-12)


def test_added_mass_scales_as_t_squared():
    """(pi/2) rho c^2 with c ~ t gives m_a ~ t^2 -- the campaign sheet's sqrt(t)
    is *not* the straight-wedge law (see the next test for what is)."""
    law = added_mass_law()
    assert law["c_exponent"] == pytest.approx(1.0, abs=1e-9)
    assert law["added_mass_exponent"] == pytest.approx(2.0, abs=1e-9)


def test_sqrt_t_law_belongs_to_curved_sections():
    """sqrt(2RVt) ~ sqrt(t): the half-power law is the curved-section wetting."""
    V, R = 1.0, 0.5
    w1 = curved_section_width(V, R, 1.0e-3)
    w2 = curved_section_width(V, R, 4.0e-3)
    assert w2 / w1 == pytest.approx(2.0, rel=1e-12)


def test_force_coefficient_is_pi2_over_tan_beta():
    rho, V, beta = 1000.0, 2.0, math.radians(20.0)
    for t in (1.0e-4, 1.0e-3, 5.0e-3):
        cs = slamming_force_coefficient(rho, V, beta, t)
        assert cs == pytest.approx(math.pi ** 2 / math.tan(beta), rel=1e-12)


def test_force_is_the_time_derivative_of_added_mass_momentum():
    rho, V, beta = 1000.0, 2.0, math.radians(30.0)
    t, dt = 1.0e-3, 1.0e-9
    mom = lambda s: added_mass_per_length(rho, V, beta, s) * V   # noqa: E731
    numeric = (mom(t + dt) - mom(t - dt)) / (2.0 * dt)
    assert numeric == pytest.approx(added_mass_force(rho, V, beta, t), rel=1e-6)


def test_added_mass_force_matches_stagnation_scaling():
    """F ~ pi rho C^2 V^3 t: the V^3 / tan^2 scaling the campaign asks about."""
    rho, beta = 1000.0, math.radians(20.0)
    t = 2.0e-3
    f2 = added_mass_force(rho, 2.0, beta, t)
    f4 = added_mass_force(rho, 4.0, beta, t)
    assert f4 / f2 == pytest.approx(8.0, rel=1e-12)
    f_b30 = added_mass_force(rho, 2.0, math.radians(30.0), t)
    ratio = f2 / f_b30
    expect = (math.tan(math.radians(30.0)) / math.tan(beta)) ** 2
    assert ratio == pytest.approx(expect, rel=1e-12)


# --------------------------------------------------------------------------- #
# 7-12  delta-SPH and MLS operators, each with a negative control
# --------------------------------------------------------------------------- #

def test_mls_gradient_is_exact_on_a_linear_field():
    pos, i, j, delta, gw, vol, _, _ = lattice()
    g = torch.tensor([3.0, -7.0], dtype=DTYPE)
    p = (pos * g).sum(-1)
    grad, bad = mls_gradient(p, i, j, delta, gw, vol, n_fluid=pos.shape[0])
    inner = torch.linalg.vector_norm(pos, dim=-1) < 0.15
    assert int(bad[inner].sum()) == 0
    err = float((grad[inner] - g).abs().max())
    assert err < 1.0e-5          # machine precision up to the kernel sum tolerance


def test_plain_gradient_control_fails_on_the_same_field():
    """Negative control: the antisymmetric SPH form is not first-order exact."""
    pos, i, j, delta, gw, vol, m, _ = lattice()
    rho = torch.full((pos.shape[0],), 1000.0, dtype=DTYPE)
    g = torch.tensor([3.0, -7.0], dtype=DTYPE)
    p = (pos * g).sum(-1)
    bracket = p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2
    contrib = (m * bracket).unsqueeze(-1) * gw
    acc = torch.zeros((pos.shape[0], 2), dtype=DTYPE)
    acc = acc.index_add(0, i, -contrib).index_add(0, j, contrib)
    grad_plain = -acc * rho[0]
    inner = torch.linalg.vector_norm(pos, dim=-1) < 0.15
    err = float((grad_plain[inner] - g).abs().max())
    assert err > 1.0             # this is why the MLS switch exists


def test_delta_sph_is_a_laplacian_with_a_grid_independent_constant():
    """D ~ k_eff delta c0 h lap(rho) with k_eff > 0 and independent of dx."""
    consts = []
    for dx in (0.01, 0.005):
        pos, i, j, delta, gw, vol, _, h = lattice(dx=dx, half=int(round(0.15 / dx)))
        a = 2.0
        rho = 1000.0 + 0.5 * a * pos.square().sum(-1)      # lap(rho) = 2a
        D = delta_sph_density_rate(rho, i, j, delta, gw, vol,
                                   n_fluid=pos.shape[0], c0=20.0, h=h,
                                   delta_coef=0.1)
        inner = pos.abs().max(-1).values < 0.5 * pos.abs().max()
        k_eff = float((D[inner] / (0.1 * 20.0 * h * 2.0 * a)).mean())
        assert k_eff > 0.0                                 # smoothing, not sharpening
        consts.append(k_eff)
    assert consts[0] == pytest.approx(consts[1], rel=1e-6)


def test_delta_sph_conserves_mass():
    pos, i, j, delta, gw, vol, _, h = lattice()
    rho = 1000.0 * (1.0 + 0.05 * torch.exp(-pos.square().sum(-1) / 0.0225))
    D = delta_sph_density_rate(rho, i, j, delta, gw, vol, n_fluid=pos.shape[0],
                               c0=20.0, h=h, delta_coef=0.1)
    total = float((vol * D).sum())
    scale = float((vol * D).abs().sum())
    assert scale > 0.0
    assert abs(total) < 1.0e-10 * scale     # exact by the volume-symmetric form


def test_delta_sph_vanishes_when_disabled():
    pos, i, j, delta, gw, vol, _, h = lattice()
    rho = 1000.0 + 10.0 * torch.sin(pos[:, 0] / 0.05)
    D0 = delta_sph_density_rate(rho, i, j, delta, gw, vol, n_fluid=pos.shape[0],
                                c0=20.0, h=h, delta_coef=0.0)
    assert float(D0.abs().max()) == 0.0
    D1 = delta_sph_density_rate(rho, i, j, delta, gw, vol, n_fluid=pos.shape[0],
                                c0=20.0, h=h, delta_coef=0.1)
    assert float(D1.abs().max()) > 0.0


def test_delta_sph_smooths_a_noisy_field():
    """The operator must lower the grid-scale content of a checkerboard."""
    pos, i, j, delta, gw, vol, _, h = lattice()
    k = (pos[:, 0] / 0.01).round().to(torch.int64)
    noise = torch.where(k.remainder(2) == 0, torch.tensor(1.0, dtype=DTYPE),
                        torch.tensor(-1.0, dtype=DTYPE))
    rho = 1000.0 + 5.0 * noise
    D = delta_sph_density_rate(rho, i, j, delta, gw, vol, n_fluid=pos.shape[0],
                               c0=20.0, h=h, delta_coef=0.1)
    # a diffusion term is negative where the field has a positive peak
    inner = pos.abs().max(-1).values < 0.15
    corr = float(torch.corrcoef(torch.stack((D[inner], noise[inner])))[0, 1])
    assert corr < -0.9


# --------------------------------------------------------------------------- #
# 13-19  solver wiring
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tiny_cfg():
    def make(**kw):
        return SlammingConfig(beta_deg=20.0, velocity=2.0, dx=4.0e-3,
                              half_width=0.10, depth=0.05, headroom=0.04,
                              c_end=0.02, frames=5, **kw)
    return make


def test_switches_off_delegates_to_the_parent_solver(tiny_cfg):
    """delta_sph = 0 and mls_gradient = False must reproduce the parent RHS."""
    cfg = tiny_cfg()
    sub = SlammingImpact(cfg)
    base = WedgeImpact(cfg)
    for _ in range(5):
        sub.advance(sub.dt())
        base.advance(base.dt())
    a = sub._evaluate(0.0)
    b = base._evaluate()
    for key in ("rho", "pressure", "acc", "p_wall"):
        assert torch.equal(a[key], b[key]), key


def test_switches_on_change_the_rhs(tiny_cfg):
    """Positive control for the A/B design: the switches must do something."""
    a = SlammingImpact(tiny_cfg())
    b = SlammingImpact(tiny_cfg(delta_sph=0.1, mls_density=True))
    for _ in range(5):
        a.advance(a.dt())
        b.advance(b.dt())
    ra = a._evaluate(a.dt())
    rb = b._evaluate(b.dt())
    assert not torch.equal(ra["pressure"], rb["pressure"])
    assert not torch.equal(ra["acc"], rb["acc"])


def test_force_integral_is_zero_before_contact_and_positive_after(tiny_cfg):
    sim = SlammingImpact(tiny_cfg())
    before = sim.wall_profile()
    assert before["force"] == 0.0 and before["c"] == 0.0
    steps = 0
    while sim.time < 0.6 * sim.cfg.resolved_t_end() and steps < 4000:
        sim.advance(sim.dt())
        steps += 1
    after = sim.wall_profile()
    assert after["c"] > 0.0
    assert after["force"] > 0.0
    # the force is the one-face pressure integral, F = 2 dx sum p
    assert after["force"] == pytest.approx(2.0 * sim.dx * float(after["p"].sum()),
                                          rel=1e-12)


def test_mass_is_conserved_exactly(tiny_cfg):
    sim = SlammingImpact(tiny_cfg(delta_sph=0.1, mls_density=True))
    m0 = float(sim.fluid_mass.sum())
    for _ in range(20):
        sim.advance(sim.dt())
    assert abs(float(sim.fluid_mass.sum()) - m0) / m0 <= 1.0e-12


def test_pressure_noise_is_small_on_the_analytic_profile():
    """The noise metric must not flag the physical Wagner distribution."""
    c = 0.05
    x = torch.linspace(0.002, 0.99 * c, 200, dtype=DTYPE)
    p = wagner_shape(x, c, math.radians(20.0)) * 1.0e4
    assert pressure_noise(x, p, float(p.max()), c) < 0.02


def test_pressure_noise_negative_control_on_a_sawtooth():
    c = 0.05
    x = torch.linspace(0.002, 0.99 * c, 200, dtype=DTYPE)
    p = torch.where(torch.arange(200).remainder(2) == 0,
                    torch.tensor(1.0e4, dtype=DTYPE),
                    torch.tensor(-1.0e4, dtype=DTYPE))
    assert pressure_noise(x, p, float(p.abs().max()), c) > 0.5


def test_pressure_shape_l2_zero_on_the_reference_and_positive_otherwise():
    c, beta = 0.05, math.radians(20.0)
    x = torch.linspace(0.001, 0.99 * c, 300, dtype=DTYPE)
    ref = wagner_shape(x, c, beta)
    assert pressure_shape_l2(x, ref, c, beta) == pytest.approx(0.0, abs=1e-10)
    flat = torch.ones_like(ref)
    assert pressure_shape_l2(x, flat, c, beta) > 0.2


def test_reference_pressure_stays_below_the_wagner_peak(tiny_cfg):
    """The measured wall pressure cannot exceed the Wagner peak by a lot."""
    sim = SlammingImpact(tiny_cfg())
    pmax_w = wagner_pmax(sim.V, sim.beta, sim.rho0)
    peak = 0.0
    steps = 0
    while sim.time < 0.7 * sim.cfg.resolved_t_end() and steps < 4000:
        sim.advance(sim.dt())
        steps += 1
        prof = sim.wall_profile()
        if prof["p"].numel():
            peak = max(peak, float(prof["p"].max()))
    assert peak > 0.0
    assert peak < 3.0 * pmax_w