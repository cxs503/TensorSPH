"""Unit gates for the free-surface pressure closure (P0 of backlog_update_m2971).

``tait_pressure`` clamps at ``p >= 0``, so on a hex lattice every particle whose
sum-density is deficient sits at exactly zero pressure and the interface cannot
carry a Young-Laplace jump.  ``free_surface_pressure_closure`` replaces the
clamped value on those particles only, with the kernel-weighted extrapolation of
Adami, Ihmsen & Adams (2012) §3.3, eq. (11).

What is tested here is the *operator*, on the geometry it is applied to (the hex
drop of ``benchmarks/laplace_pressure_floor``) but with a synthetic pressure
field: a drop whose surface tension is not switched on carries ``p = 0``
everywhere, so an ``rhs``-driven test would measure nothing.  The gates are:

* the formula against a hand-computed two-particle case, with and without the
  body-force term, so a sign error cannot hide;
* inert interior, lifted shell, and no overshoot above the interior maximum;
* ``sweeps=0`` reproduces the legacy clamped field bit for bit, which is what
  makes the change safe to leave opt-in;
* the sweep ladder is monotone *but not a contraction* - recorded as such,
  because the end-to-end error crosses zero between 4 and 8 sweeps;
* the pruned pair path and the dense reference path agree with the closure
  enabled: the dense kernel matrix carries a diagonal self weight that the pair
  path does not have, and including it silently damps the extrapolation back to
  zero.

The end-to-end acceptance criterion (every recorded case inside 15 %) lives with
the benchmark: ``tests/test_laplace_pressure_floor.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    free_surface_pressure_closure,
    rhs,
    rhs_all_pairs,
    validate_config,
)
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = (Path(__file__).resolve().parents[1]
         / "benchmarks" / "laplace_pressure_floor" / "run.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location("lpf_run", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_harness()

RADIUS = 0.02
RHO0 = 1000.0
DX = 0.002
DP = 360.0            # sigma/R of the benchmark drop, Pa


def _cfg(**kwargs) -> Config:
    base = dict(dx=DX, h_factor=1.3, rho0=RHO0, sound_speed=10.0, gamma=7.0,
                gravity=0.0, viscosity=0.0, surface_tension_coefficient=0.0)
    base.update(kwargs)
    return Config(**base)


@pytest.fixture(scope="module")
def drop():
    """Calibrated hex drop: interior at rho0, deficient free-surface shell."""
    pos, mass, h, pairs, _, _ = H.calibrated_droplet(DX)
    return pos, mass, h, pairs


def _synthetic_pressure(rho):
    """What the clamped field looks like when the interior carries sigma/R."""
    interior = rho >= 0.99 * RHO0
    return torch.where(interior, torch.full_like(rho, DP), torch.zeros_like(rho))


def _pairs_of(pos, h):
    i, j = cell_linked_pairs(pos, 2.0 * h).unbind(dim=1)
    delta = pos[i] - pos[j]
    return i, j, torch.clamp(1.0 - (torch.linalg.vector_norm(delta, dim=1)
                                    / (2.0 * h)) ** 3, min=0.0), delta


# --------------------------------------------------------------------------- #
# the formula, checked by hand
# --------------------------------------------------------------------------- #
def test_formula_matches_hand_computed_kernel_average():
    """One deficient particle with one neighbour: pure kernel average."""
    rho = torch.tensor([900.0, RHO0])
    pressure = torch.tensor([0.0, 100.0])
    i = torch.tensor([0])          # particle 0 is the deficient one
    j = torch.tensor([1])
    wij = torch.tensor([1.0])
    delta = torch.tensor([[1.0e-3, 0.0]])   # pos[i] - pos[j]

    out = free_surface_pressure_closure(rho, pressure, RHO0, 0.0, i, j, wij,
                                        delta, 2, sweeps=1)
    # particle 0 is deficient -> it takes particle 1's pressure; particle 1 is
    # complete -> it keeps its own.
    assert out[0].item() == pytest.approx(100.0)
    assert out[1].item() == pytest.approx(100.0)


def test_body_force_term_lowers_the_extrapolated_pressure():
    """g.(r_i - r_j) must enter with the sign of a hydrostatic gradient."""
    dy, rho_j, p_j, g = 1.0e-3, RHO0, 100.0, 10.0
    rho = torch.tensor([900.0, RHO0])
    pressure = torch.tensor([0.0, p_j])
    i = torch.tensor([0])
    j = torch.tensor([1])
    wij = torch.tensor([1.0])
    delta = torch.tensor([[0.0, dy]])        # pos[0] - pos[1]: 0 sits above 1

    out = free_surface_pressure_closure(rho, pressure, RHO0, g, i, j, wij,
                                        delta, 2, sweeps=1)
    # p_0 = p_1 + rho_1 * g.(r_0 - r_1) = p_1 - rho_1 * g * dy
    assert out[0].item() == pytest.approx(p_j - rho_j * g * dy)
    assert out[0].item() < p_j


def test_closure_is_clamped_at_zero_pressure():
    """A large upward offset cannot manufacture a tensile pressure."""
    rho = torch.tensor([900.0, RHO0])
    pressure = torch.tensor([0.0, 1.0])
    out = free_surface_pressure_closure(rho, pressure, RHO0, 1.0e4,
                                        torch.tensor([0]), torch.tensor([1]),
                                        torch.tensor([1.0]),
                                        torch.tensor([[0.0, 1.0]]), 2, sweeps=1)
    assert out[0].item() == 0.0


# --------------------------------------------------------------------------- #
# behaviour on the real drop geometry
# --------------------------------------------------------------------------- #
def test_complete_particles_are_untouched_and_the_shell_is_lifted(drop):
    pos, mass, h, _ = drop
    rho = density_from_neighbor_pairs(pos, mass, h)
    pressure = _synthetic_pressure(rho)
    i, j, wij, delta = _pairs_of(pos, h)
    out = free_surface_pressure_closure(rho, pressure, RHO0, 0.0, i, j, wij,
                                        delta, pos.shape[0], sweeps=4)

    deficient = rho < 0.99 * RHO0
    complete = ~deficient
    assert bool(deficient.any()) and bool(complete.all() | ~complete.all())
    # complete particles keep their value exactly
    torch.testing.assert_close(out[complete], pressure[complete], rtol=0.0, atol=0.0)
    # the clamped shell (exactly 0) is lifted to a positive pressure
    assert float(pressure[deficient].abs().max()) == 0.0
    assert float(out[deficient].min()) > 0.0


def test_the_closure_does_not_overshoot_the_interior(drop):
    """No particle may be pushed above the largest pressure in the fluid."""
    pos, mass, h, _ = drop
    rho = density_from_neighbor_pairs(pos, mass, h)
    pressure = _synthetic_pressure(rho)
    i, j, wij, delta = _pairs_of(pos, h)
    out = free_surface_pressure_closure(rho, pressure, RHO0, 0.0, i, j, wij,
                                        delta, pos.shape[0], sweeps=4)
    assert float(out.max()) <= float(pressure.max()) * (1.0 + 1.0e-9)


def test_zero_sweeps_reproduces_the_legacy_field_bit_for_bit(drop):
    pos, mass, h, _ = drop
    rho = density_from_neighbor_pairs(pos, mass, h)
    pressure = _synthetic_pressure(rho)
    i, j, wij, delta = _pairs_of(pos, h)
    out = free_surface_pressure_closure(rho, pressure, RHO0, 0.0, i, j, wij,
                                        delta, pos.shape[0], sweeps=0)
    torch.testing.assert_close(out, pressure, rtol=0.0, atol=0.0)


def test_sweeps_are_monotone_but_not_a_contraction(drop):
    """More sweeps keep raising the shell pressure: the iteration does not settle.

    This is a recorded property, not a wish.  Each sweep re-applies the
    extrapolation to the same deficient set, so pressure keeps being carried
    further into the shell; the end-to-end error therefore crosses zero
    (benchmarks/laplace_pressure_floor: sweeps 1/2/4/8 -> -20.1/-6.5/-0.8/+9.1 %)
    and ``sweeps`` is a calibrated parameter, not a converged limit.  What this
    test protects is that the direction of travel stays the same and that the
    value stays inside the physical bound.
    """
    pos, mass, h, _ = drop
    rho = density_from_neighbor_pairs(pos, mass, h)
    pressure = _synthetic_pressure(rho)
    i, j, wij, delta = _pairs_of(pos, h)
    deficient = rho < 0.99 * RHO0
    means = {}
    for sweeps in (1, 2, 4, 8, 16):
        out = free_surface_pressure_closure(rho, pressure, RHO0, 0.0, i, j, wij,
                                            delta, pos.shape[0], sweeps=sweeps)
        means[sweeps] = float(out[deficient].mean())
    steps = [1, 2, 4, 8, 16]
    assert all(means[a] <= means[b] + 1e-9 for a, b in zip(steps, steps[1:])), means
    # there is real movement (not a one-shot correction) ...
    assert means[16] > 1.05 * means[1], means
    # ... bounded by the interior value it extrapolates
    assert means[16] <= DP * (1.0 + 1.0e-9), means


# --------------------------------------------------------------------------- #
# the option itself
# --------------------------------------------------------------------------- #
def test_dense_and_pair_paths_agree_with_the_closure():
    """A lattice with a real pressure (uncalibrated) exercises the closure."""
    pos, mass = H.hexagonal_droplet(RADIUS, DX, RHO0)
    vel = torch.zeros_like(pos)
    cfg = _cfg(free_surface_pressure="average")
    rho_p, p_p, acc_p = rhs(pos, vel, mass, cfg)
    rho_d, p_d, acc_d = rhs_all_pairs(pos, vel, mass, cfg)
    assert float(p_p.max()) > 0.0            # the closure is actually active
    torch.testing.assert_close(rho_p, rho_d, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(p_p, p_d, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(acc_p, acc_d, rtol=1e-8, atol=1e-8)


def test_default_is_the_legacy_clamp():
    """The closure is opt-in; a silent default flip would re-baseline every case."""
    assert Config().free_surface_pressure == "clamp"
    assert Config().free_surface_pressure_sweeps >= 1


def test_unknown_mode_is_rejected():
    cfg = Config(free_surface_pressure="adami")
    with pytest.raises(ValueError, match="free_surface_pressure"):
        validate_config(cfg)


def test_negative_sweeps_are_rejected():
    with pytest.raises(ValueError, match="sweeps"):
        validate_config(Config(free_surface_pressure_sweeps=-1))