"""Unit tests for the free-surface pressure closure (Adami 2012, Sec. 3.3).

The closure is what :mod:`tensorsph.fs_pressure` adds on top of the Tait clamp
``p >= 0``: a free-surface fluid particle (summation density deficient by more
than ``fs_pressure_threshold``) takes the kernel-weighted mean pressure of its
*complete-kernel* neighbours instead of zero.

Two layers are tested here:

* the pure pair-level function, against hand-computable synthetic pairs, so the
  direction of the interpolation, the exclusion of free-surface sources and the
  wall-particle hand-off are pinned independently of any simulation;
* the ``rhs`` wiring, including the backward-compatibility guarantee that the
  default configuration is bit-for-bit the pre-existing behaviour.
"""
import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    make_particles,
    resolve_device,
    rhs,
    tait_pressure,
    validate_config,
)
from tensorsph.fs_pressure import free_surface_mask, shell_pressure
from tensorsph.neighbor_search import cell_linked_pairs


# --------------------------------------------------------------------------- #
# Pure function: direction of the interpolation
# --------------------------------------------------------------------------- #

def test_free_surface_mask_uses_the_deficit_threshold():
    rho = torch.tensor([1000.0, 995.0, 988.0, 985.0], dtype=torch.float64)
    assert free_surface_mask(rho, 1000.0).tolist() == [False, False, True, True]
    assert free_surface_mask(rho, 1000.0, threshold=0.986).tolist() == [
        False, False, False, True]


@pytest.mark.parametrize("free_is_first", [True, False])
def test_pressure_is_extrapolated_to_the_free_surface_particle(free_is_first):
    """Both pair orderings must move the value to the *free* partner."""
    rho = torch.tensor([980.0, 1000.0], dtype=torch.float64)
    pressure = torch.tensor([0.0, 360.0], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]] if free_is_first else [[1, 0]])
    out, stats = shell_pressure(rho, pressure, pairs, torch.ones(1,
                                dtype=torch.float64), 2, rho0=1000.0)
    torch.testing.assert_close(out, torch.tensor([360.0, 360.0],
                                                 dtype=torch.float64))
    assert stats["n_updated"] == 1 and stats["n_starved"] == 0


def test_free_surface_particles_are_not_used_as_sources():
    """Two deficient neighbours carry no information: both keep the clamp."""
    rho = torch.tensor([980.0, 980.0, 1000.0], dtype=torch.float64)
    pressure = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    pairs = torch.tensor([[0, 1], [1, 2]])
    out, stats = shell_pressure(rho, pressure, pairs,
                                torch.ones(2, dtype=torch.float64), 3,
                                rho0=1000.0)
    torch.testing.assert_close(out, torch.zeros(3, dtype=torch.float64))
    # Particle 1 has a complete-kernel neighbour (2) that happens to be at p = 0,
    # so it is *updated* to zero; particle 0 has none and keeps the clamp.
    assert stats["n_free"] == 2
    assert stats["n_updated"] == 1 and stats["n_starved"] == 1


def test_kernel_weighted_mean_and_clamping_to_non_negative():
    rho = torch.tensor([980.0, 1000.0, 980.0], dtype=torch.float64)
    pressure = torch.tensor([0.0, 100.0, 0.0], dtype=torch.float64)
    # Particle 1 is interior (rho = rho0) and sits between two free particles.
    pairs = torch.tensor([[0, 1], [1, 2]])
    weights = torch.tensor([2.0, 1.0], dtype=torch.float64)
    out, stats = shell_pressure(rho, pressure, pairs, weights, 3, rho0=1000.0)
    torch.testing.assert_close(out, torch.tensor([100.0, 100.0, 100.0],
                                                 dtype=torch.float64))
    assert stats["mean_lift_Pa"] == pytest.approx(100.0)


def test_starved_particle_keeps_the_clamp_value():
    rho = torch.tensor([980.0], dtype=torch.float64)
    pressure = torch.tensor([0.0], dtype=torch.float64)
    out, stats = shell_pressure(rho, pressure, torch.zeros((0, 2), dtype=torch.long),
                                torch.zeros(0, dtype=torch.float64), 1,
                                rho0=1000.0)
    torch.testing.assert_close(out, pressure)
    assert stats["n_free"] == 1 and stats["n_starved"] == 1


def test_wall_particles_are_valid_sources_and_never_targets():
    # 1 fluid particle (deficient) + 2 wall particles carrying 250 Pa.
    rho = torch.tensor([980.0], dtype=torch.float64)
    pressure = torch.tensor([0.0], dtype=torch.float64)
    wall_pressure = torch.tensor([250.0, 250.0], dtype=torch.float64)
    pairs = torch.tensor([[0, 1], [0, 2]])
    out, stats = shell_pressure(rho, pressure, pairs,
                                torch.ones(2, dtype=torch.float64), 1,
                                rho0=1000.0, wall_pressure=wall_pressure)
    torch.testing.assert_close(out, torch.tensor([250.0], dtype=torch.float64))
    assert stats["n_updated"] == 1


def test_identical_pressure_field_is_a_fixed_point():
    generator = torch.Generator().manual_seed(7)
    pos = torch.rand((40, 2), generator=generator, dtype=torch.float64)
    h = 0.3
    rho = density_from_neighbor_pairs(pos, torch.full((40,), 0.01,
                                                      dtype=torch.float64), h)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    from tensorsph.droplet_impact import pair_kernels
    wij, _ = pair_kernels(delta, h)
    uniform = torch.full((40,), 42.0, dtype=torch.float64)
    out, stats = shell_pressure(rho, uniform, pairs, wij, 40,
                                rho0=float(rho.max()))
    # A uniform field must survive the weighted average wherever it is applied.
    torch.testing.assert_close(out, uniform)


def test_shape_and_threshold_guards():
    rho = torch.ones(3, dtype=torch.float64)
    pressure = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        shell_pressure(rho, pressure, torch.zeros((0, 2), dtype=torch.long),
                       torch.zeros(0, dtype=torch.float64), 3, rho0=1000.0,
                       threshold=1.0)
    with pytest.raises(ValueError):
        shell_pressure(rho, pressure, torch.tensor([[0, 1, 2]]),
                       torch.ones(1, dtype=torch.float64), 3, rho0=1000.0)
    with pytest.raises(ValueError):
        free_surface_mask(torch.ones((2, 2), dtype=torch.float64), 1000.0)


# --------------------------------------------------------------------------- #
# Wiring into rhs
# --------------------------------------------------------------------------- #

def _small_drop():
    cfg = Config(dx=0.02, h_factor=1.3, rho0=1000.0, sound_speed=10.0,
                 gamma=7.0, gravity=0.0, viscosity=0.01, device="cpu",
                 surface_tension_coefficient=0.072)
    device = resolve_device("cpu")
    pos, mass = make_particles(cfg, device)
    pos = pos.to(torch.float64)
    # The default lattice sums to slightly *below* rho0 over the whole pool, so
    # every particle clamps to zero and the closure would have nothing to lift.
    # 1.05 puts the interior above rho0, which is the configuration of interest.
    mass = mass.to(torch.float64) * 1.05
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    vel = torch.zeros_like(pos)
    return cfg, pos, mass, vel, pairs, rho


def test_default_configuration_is_unchanged():
    """Backward compatibility: "clamp" must reproduce the plain Tait clamp."""
    cfg, pos, mass, vel, pairs, rho = _small_drop()
    assert cfg.free_surface_pressure == "clamp"
    rho_a, p_a, acc_a = rhs(pos, vel, mass, cfg, pairs=pairs)
    torch.testing.assert_close(p_a, tait_pressure(rho_a, cfg))

    cfg_shell = Config(**{**cfg.__dict__, "free_surface_pressure": "shell"})
    validate_config(cfg_shell)
    rho_b, p_b, acc_b = rhs(pos, vel, mass, cfg_shell, pairs=pairs)
    torch.testing.assert_close(rho_a, rho_b)
    # The closure lifts the pressure of deficient particles and perturbs the
    # pressure gradient, so both the field and the acceleration must change.
    assert bool((p_b >= p_a).all())
    changed = p_b > p_a
    assert int(changed.sum()) > 0
    assert bool(free_surface_mask(rho_a, cfg.rho0)[changed].all())
    # Lifting the pressure of the deficient shell changes the discrete gradient.
    assert float((acc_a - acc_b).abs().max()) > 0.0


def test_the_closure_only_touches_deficient_particles():
    cfg, pos, mass, vel, pairs, rho = _small_drop()
    cfg_shell = Config(**{**cfg.__dict__, "free_surface_pressure": "shell"})
    rho, p_clamp, _ = rhs(pos, vel, mass, cfg, pairs=pairs)
    _, p_shell, _ = rhs(pos, vel, mass, cfg_shell, pairs=pairs)
    free = free_surface_mask(rho, cfg.rho0, cfg.fs_pressure_threshold)
    assert bool(free.any()) and not bool(free.all())
    torch.testing.assert_close(p_shell[~free], p_clamp[~free])


def test_unknown_closure_is_rejected():
    cfg = Config(free_surface_pressure="ghost")
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = Config(fs_pressure_threshold=1.5)
    with pytest.raises(ValueError):
        validate_config(cfg)

# --------------------------------------------------------------------------- #
# The published verification record (machine-checked, not prose)
# --------------------------------------------------------------------------- #

def test_recorded_laplace_gate_is_closed_by_the_closure():
    """Re-check ``benchmarks/fs_pressure/report.json`` as the verification record.

    This is the regression gate for the Young-Laplace claim in
    ``docs/verification_fs_pressure.md``: every ``shell`` case must sit inside the
    15 % platform limit, and every ``clamp`` case must sit outside it (otherwise
    the closure would not be doing anything and the "improvement" would be a
    re-reading of the same number).
    """
    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "benchmarks" / "fs_pressure"
            / "report.json")
    report = json.loads(path.read_text())
    shell = [c for c in report["cases"] if c["closure"] == "shell"]
    clamp = [c for c in report["cases"]
             if c["closure"] == "clamp" and c["sigma_N_per_m"] == 7.2]
    assert report["summary"]["gate_15pct_closed"] is True
    assert len(shell) == 6 and len(clamp) == 4
    for case in shell:
        tail = case["absolute_time_average_tail"]
        assert abs(tail["rel_error_pct"]) < 15.0, case["dx_m"]
    for case in clamp:
        tail = case["absolute_time_average_tail"]
        assert abs(tail["rel_error_pct"]) > 15.0, case["dx_m"]
    # The sigma ladder must show that the residual is decoupled from sigma:
    # clamp degrades as sigma grows, shell does not.
    clamp_err = {c["sigma_N_per_m"]: abs(c["absolute_time_average_tail"]["rel_error_pct"])
                 for c in report["cases"] if c["closure"] == "clamp"
                 and c["sound_speed"] == 20.0 and c["dx_m"] == 0.002}
    assert sorted(clamp_err) == [3.6, 7.2, 14.4]
    assert clamp_err[14.4] > clamp_err[3.6] + 10.0
