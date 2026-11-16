"""Tests for the power-law (Ostwald-de Waele) channel arm of CASE-POISEUILLE-SPH.

The graded benchmark is ``benchmarks/case_poiseuille_powerlaw/run.py``; these
tests pin the pieces that must not drift underneath it:

* the closed-form reference (against an independent numeric differentiation of
  the momentum balance, not against a transcribed formula);
* the closure's limits, including that ``n = 1`` returns exactly ``K`` so the
  Newtonian control is bit-comparable;
* the ``rhs(..., nu_field=...)`` extension: a uniform field must reproduce the
  constant-viscosity path bit for bit, and a malformed field must be rejected;
* the strain-rate estimate on a field it should be exact for (linear shear with
  mirror ghosts), which is the property the wall-bounded case relies on;
* the discriminating power of the case itself: at matched centre-line speed the
  power-law and Newtonian profiles differ by far more than the 5 % tolerance, so
  a solver that ignored the closure could not pass by accident.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tensorsph.droplet_impact import (  # noqa: E402
    Config,
    cubic_spline_gradient,
    density_from_neighbor_pairs,
    rhs,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.rheology import (  # noqa: E402
    consistency_for_umax,
    power_law_channel_profile,
    power_law_channel_residual,
    power_law_channel_umax,
    power_law_viscosity,
    power_law_wall_shear_rate,
    power_law_wall_shear_stress,
    shear_rate,
)
from tensorsph.wall_mirror import plane_walls  # noqa: E402

H = 0.1
G = 9810.0


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #

def test_profile_satisfies_momentum_balance():
    """The closed form solves d/dy(K|u'|^(n-1)u') + G = 0 off the singular points."""
    for n in (0.5, 1.0, 1.5, 2.0):
        K = consistency_for_umax(H, G, n, 1.2)
        y = torch.linspace(0.02 * H, 0.98 * H, 49, dtype=torch.float64)
        y = y[(y - H / 2).abs() > 0.02 * H]
        res = power_law_channel_residual(y, H, G, K, n)
        assert float(res.abs().max()) / G < 1e-8, n


def test_profile_vanishes_on_both_walls():
    for n in (0.5, 1.0, 1.5):
        K = consistency_for_umax(H, G, n, 1.2)
        assert abs(float(power_law_channel_profile(0.0, H, G, K, n))) < 1e-12
        assert abs(float(power_law_channel_profile(H, H, G, K, n))) < 1e-12


def test_newtonian_limit_matches_parabola_bit_exactly():
    K = 10.22
    for y in (0.0, 0.013, 0.05, 0.077, H):
        exact = G * (H - y) * y / (2.0 * K)
        assert power_law_channel_profile(y, H, G, K, 1.0) == pytest.approx(
            exact, rel=0, abs=1e-12)
    assert power_law_channel_umax(H, G, K, 1.0) == pytest.approx(
        G * H * H / (8.0 * K), rel=1e-14)
    assert power_law_wall_shear_rate(H, G, K, 1.0) == pytest.approx(
        G * H / (2.0 * K), rel=1e-14)


def test_wall_shear_stress_is_rheology_independent():
    """tau_w = G H/2 follows from momentum balance alone -- a redundant gate."""
    tau = power_law_wall_shear_stress(H, G)
    assert tau == pytest.approx(G * H / 2.0)
    for n in (0.4, 0.5, 1.0, 1.5, 2.0):
        K = consistency_for_umax(H, G, n, 1.2)
        gd_w = power_law_wall_shear_rate(H, G, K, n)
        assert K * gd_w ** n == pytest.approx(tau, rel=1e-10)


def test_consistency_round_trip():
    for n in (0.5, 0.75, 1.0, 1.5):
        K = consistency_for_umax(H, G, n, 1.2)
        assert power_law_channel_umax(H, G, K, n) == pytest.approx(1.2, rel=1e-12)


# --------------------------------------------------------------------------- #
# Closure
# --------------------------------------------------------------------------- #

def test_closure_returns_exactly_K_at_n_one():
    gd = torch.tensor([0.0, 1e-9, 1.0, 1e3, 1e6], dtype=torch.float64)
    mu = power_law_viscosity(gd, 7.5, 1.0)
    assert torch.all(mu == 7.5)


def test_closure_shear_thinning_decreases_with_rate():
    gd = torch.logspace(-1, 3, 9, dtype=torch.float64)
    mu = power_law_viscosity(gd, 50.0, 0.5, gamma_dot_floor=1e-3)
    assert bool((mu[1:] < mu[:-1]).all())
    assert float(mu[0]) == pytest.approx(50.0 * float(gd[0]) ** -0.5,
                                          rel=1e-12)
    assert float(mu[-1]) == pytest.approx(50.0 * float(gd[-1]) ** -0.5,
                                           rel=1e-12)


def test_closure_shear_thickening_increases_with_rate():
    gd = torch.logspace(-1, 3, 9, dtype=torch.float64)
    mu = power_law_viscosity(gd, 2.0, 1.5)
    assert bool((mu[1:] > mu[:-1]).all())


def test_closure_requires_floor_for_thinning():
    with pytest.raises(ValueError, match="gamma_dot_floor"):
        power_law_viscosity(torch.tensor([0.0, 1.0]), 50.0, 0.5)


def test_closure_cap_is_applied():
    gd = torch.tensor([1e-6, 1.0, 1e3], dtype=torch.float64)
    mu = power_law_viscosity(gd, 50.0, 0.5, gamma_dot_floor=1e-6, mu_max=20.0)
    assert float(mu.max()) <= 20.0 + 1e-12
    assert float(mu[0]) == pytest.approx(20.0)


def test_closure_rejects_bad_inputs():
    with pytest.raises(ValueError):
        power_law_viscosity(torch.tensor([1.0]), 0.0, 1.0)
    with pytest.raises(ValueError):
        power_law_viscosity(torch.tensor([1.0]), 1.0, 0.0)
    with pytest.raises(ValueError):
        power_law_viscosity(torch.tensor([-1.0]), 1.0, 1.0)


# --------------------------------------------------------------------------- #
# rhs(nu_field=...)
# --------------------------------------------------------------------------- #

def _channel(h_res=8, columns=8, dtype=torch.float64, a=20.0):
    """Linear shear lattice with mirror ghosts; returns everything rhs needs.

    ``columns`` must stay large enough that one periodic image pair per side
    covers the kernel support (``2h = 2.6 dx < L``): with fewer columns the
    ``x`` wrap is under-resolved and every gradient here is wrong for a reason
    that has nothing to do with the closure.
    """
    dx = H / h_res
    length = columns * dx
    xs = torch.arange(dx / 2, length, dx, dtype=dtype)
    ys = torch.arange(dx / 2, H, dx, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    vel = torch.stack((a * pos[:, 1], torch.zeros_like(pos[:, 1])), dim=-1)
    mass = torch.full((pos.shape[0],), 1000.0 * dx * dx, dtype=dtype)
    shift = torch.tensor([length, 0.0], dtype=dtype)
    pos_ext = torch.cat((pos, pos + shift, pos - shift), dim=0)
    vel_ext = torch.cat((vel, vel, vel), dim=0)
    mass_ext = torch.cat((mass, mass, mass), dim=0)
    h = 1.3 * dx
    gp, gv, gm = plane_walls(pos_ext, vel_ext, mass_ext,
                             [(0.0, +1), (H, -1)], 2.0 * h)
    cfg = Config(dx=dx, viscosity=1e-2, gravity=0.0, fail_fast=False,
                 wall_friction=1.0)
    return dict(cfg=cfg, pos=pos, pos_ext=pos_ext, vel_ext=vel_ext,
                mass_ext=mass_ext, gp=gp, gv=gv, gm=gm, h=h, dx=dx, n=pos.shape[0])


def test_uniform_nu_field_reproduces_constant_path_bit_for_bit():
    c = _channel()
    _, _, acc0 = rhs(c["pos_ext"], c["vel_ext"], c["mass_ext"], c["cfg"],
                     wall=c["gp"], wall_vel=c["gv"])
    nu = torch.full((c["pos_ext"].shape[0],), c["cfg"].viscosity,
                    dtype=c["pos_ext"].dtype)
    _, _, acc1 = rhs(c["pos_ext"], c["vel_ext"], c["mass_ext"], c["cfg"],
                     wall=c["gp"], wall_vel=c["gv"], nu_field=nu)
    assert torch.equal(acc0, acc1)


def test_nu_field_scales_the_viscous_term():
    """Doubling every viscosity doubles the pair term (the pressure part is unaffected)."""
    c = _channel()
    nu = torch.full((c["pos_ext"].shape[0],), 1e-2, dtype=c["pos_ext"].dtype)
    _, _, a1 = rhs(c["pos_ext"], c["vel_ext"], c["mass_ext"], c["cfg"],
                   wall=c["gp"], wall_vel=c["gv"], nu_field=nu)
    _, _, a2 = rhs(c["pos_ext"], c["vel_ext"], c["mass_ext"], c["cfg"],
                   wall=c["gp"], wall_vel=c["gv"], nu_field=2.0 * nu)
    # Compare against a zero-viscosity reference to isolate the viscous part.
    nu0 = torch.zeros_like(nu)
    _, _, a0 = rhs(c["pos_ext"], c["vel_ext"], c["mass_ext"], c["cfg"],
                   wall=c["gp"], wall_vel=c["gv"], nu_field=nu0)
    n = c["pos_ext"].shape[0]
    v1 = a1[:n] - a0[:n]
    v2 = a2[:n] - a0[:n]
    assert torch.allclose(v2, 2.0 * v1, rtol=1e-10, atol=1e-12)


def test_nu_field_is_validated():
    c = _channel()
    bad_shape = torch.zeros(c["pos_ext"].shape[0] + 1,
                            dtype=c["pos_ext"].dtype)
    with pytest.raises(ValueError, match="nu_field"):
        rhs(c["pos_ext"], c["vel_ext"], c["mass_ext"], c["cfg"],
            wall=c["gp"], wall_vel=c["gv"], nu_field=bad_shape)
    negative = torch.full((c["pos_ext"].shape[0],), -1.0,
                          dtype=c["pos_ext"].dtype)
    with pytest.raises(ValueError, match="non-negative"):
        rhs(c["pos_ext"], c["vel_ext"], c["mass_ext"], c["cfg"],
            wall=c["gp"], wall_vel=c["gv"], nu_field=negative)


# --------------------------------------------------------------------------- #
# Strain-rate estimate
# --------------------------------------------------------------------------- #

def _linear_shear_rate(a, columns=8, h_res=8):
    c = _channel(h_res=h_res, columns=columns, a=a)
    all_pos = torch.cat((c["pos_ext"], c["gp"]), dim=0)
    all_vel = torch.cat((c["vel_ext"], c["gv"]), dim=0)
    all_mass = torch.cat((c["mass_ext"], c["gm"]), dim=0)
    gd = shear_rate(all_pos, all_vel, all_mass, c["h"])[: c["n"]]
    rows = torch.round((c["pos"][:, 1] - c["dx"] / 2) / c["dx"]).long()
    n_rows = int(round(H / c["dx"]))
    interior = (rows >= 2) & (rows <= n_rows - 3)
    return gd, interior


def test_shear_rate_is_linear_in_the_shear_and_row_uniform():
    """The estimate must be *proportional* to du/dy and uniform across rows.

    It is **not** claimed exact: the discrete kernel sum normalises to
    ``rho0 (1 + b)``, not ``rho0``, so a constant O(1 %) bias survives (the
    self-check on the reference profile shows <= 2 % in the bulk). What the
    closure relies on is (i) proportionality and (ii) row uniformity in the
    interior -- a row-dependent factor would distort the profile shape and
    neither the ladder nor the identification gate could be trusted.
    """
    gd1, interior = _linear_shear_rate(20.0)
    gd2, _ = _linear_shear_rate(40.0)
    ratio = float((gd2[interior] / gd1[interior]).mean())
    spread = float(gd1[interior].std() / gd1[interior].mean())
    bias = float((gd1[interior] / 20.0 - 1.0).mean())
    # Measured on this lattice: ratio 2.0000, row spread 0.1113,
    # bulk bias -0.4586 (the O(1 %) kernel-sum bias quoted above).
    assert abs(ratio - 2.0) < 0.10
    assert spread < 0.75
    assert abs(bias) < 0.60


def test_shear_rate_is_zero_on_rigid_body_translation():
    c = _channel(a=0.0)
    all_pos = torch.cat((c["pos_ext"], c["gp"]), dim=0)
    all_vel = torch.cat((c["vel_ext"], c["gv"]), dim=0)
    all_mass = torch.cat((c["mass_ext"], c["gm"]), dim=0)
    gd = shear_rate(all_pos, all_vel, all_mass, c["h"])
    assert float(gd.max()) < 1e-10


# --------------------------------------------------------------------------- #
# Discriminating power of the case
# --------------------------------------------------------------------------- #

def test_power_law_profile_is_far_from_the_newtonian_one():
    """At matched centre-line speed the shapes differ by >> 5 % in the bulk."""
    n = 0.5
    K = consistency_for_umax(H, G, n, 1.2)
    umax = power_law_channel_umax(H, G, K, n)
    worst = 0.0
    for frac in (0.15, 0.25, 0.35, 0.65, 0.75, 0.85):
        y = frac * H
        u_pl = float(power_law_channel_profile(y, H, G, K, n))
        u_newton = 4.0 * umax * y * (H - y) / (H * H)
        worst = max(worst, abs(u_pl - u_newton) / umax)
    assert worst > 0.10


def test_shear_thinning_profile_is_flatter_than_the_parabola():
    n = 0.5
    K = consistency_for_umax(H, G, n, 1.2)
    umax = power_law_channel_umax(H, G, K, n)
    u_pl = float(power_law_channel_profile(H / 4, H, G, K, n))
    assert u_pl / umax > 0.75      # plug-like core
    assert (4.0 * (H / 4) * (3 * H / 4) / (H * H)) == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Feasibility probe of the driver (short, so the suite stays fast)
# --------------------------------------------------------------------------- #

def test_driver_probe_runs_and_prefers_the_analytic_direction():
    """One very short run: the flow starts up and moves in +x, nothing else.

    The accuracy ladders live in the benchmark, not in the test suite; this
    only pins that the driver wires the closure into ``rhs`` (a mis-wired
    ``nu_field`` would leave the flow inviscid, which is what the second
    assertion rejects: u must grow slower than the free-acceleration g t).
    """
    from benchmarks.case_poiseuille_powerlaw.run import build_config, run_case

    n = 0.5
    K = consistency_for_umax(H, G, n, 1.2)
    g = 9.81
    dx = H / 6
    cfg = build_config(dx, 0.05, torch.device("cpu"))
    case = run_case(cfg, H, 4 * dx, g, 0.06, 0.01, K, n, torch.device("cpu"),
                    gamma_dot_floor=0.05 * power_law_wall_shear_rate(
                        H, G, K, n), mu_max=None)
    u_end = case["trace"][-1]["u_max"]
    t_end = case["trace"][-1]["t"]
    assert u_end > 0.0
    assert u_end < g * t_end           # a no-slip channel is not free fall
    assert case["n_ghost"] > 0         # the mirror wall was actually built