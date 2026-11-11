"""Solver-level tests: EOS, time step, boundaries, conservation."""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.wcsph import kernels
from tensorsph.wcsph.solver import (Solver, WCSPHConfig, mirror_images,
                                    sound_speed_for, stable_dt, tait_pressure)


def _cfg(**kw):
    base = dict(dim=2, dx=0.01, c0=20.0, alpha_av=0.0, gravity=0.0,
                boundary="none")
    base.update(kw)
    return WCSPHConfig(**base)


def test_tait_pressure_is_zero_at_reference_density():
    cfg = _cfg()
    rho = torch.tensor([1000.0], dtype=torch.float64)
    assert float(tait_pressure(rho, cfg)) == 0.0


def test_tait_pressure_is_monotone_and_uses_gamma_seven():
    cfg = _cfg()
    rho = torch.tensor([1000.0, 1010.0, 1020.0], dtype=torch.float64)
    p = tait_pressure(rho, cfg)
    assert bool((p[1:] > p[:-1]).all())
    expected = cfg.bulk_modulus * ((1.01) ** 7 - 1.0)
    assert float(p[1]) == pytest.approx(expected, rel=1e-12)
    assert cfg.gamma == 7.0


def test_tait_pressure_is_clamped_on_the_free_surface():
    cfg = _cfg()
    rho = torch.tensor([900.0, 100.0], dtype=torch.float64)
    assert float(tait_pressure(rho, cfg).abs().max()) == 0.0


def test_sound_speed_requirement_is_ten_times_the_flow():
    assert sound_speed_for(2.0) == pytest.approx(20.0)
    assert sound_speed_for(0.0) == pytest.approx(1.0)


def test_cfl_step_scales_with_h_over_c0():
    cfg = _cfg()
    dt0 = stable_dt(cfg, 0.0)
    dt1 = stable_dt(cfg, cfg.c0)
    assert dt0 == pytest.approx(cfg.cfl * cfg.h0 / cfg.c0)
    assert dt1 == pytest.approx(cfg.cfl * cfg.h0 / (2.0 * cfg.c0))
    assert dt1 < dt0


def test_gravity_branch_of_the_step_limit_is_active_for_a_quiet_flow():
    cfg = _cfg(c0=1.0, cfl=0.5, gravity=9.81)
    dt = stable_dt(cfg, 0.0)
    assert dt <= 0.25 * math.sqrt(cfg.h0 / 9.81) + 1.0e-15


def test_validation_rejects_impossible_configurations():
    with pytest.raises(ValueError):
        _cfg(alpha_av=1.5).validate()
    with pytest.raises(ValueError):
        _cfg(kernel="quartic").validate()
    with pytest.raises(ValueError):
        _cfg(boundary="periodic").validate()
    with pytest.raises(ValueError):
        _cfg(dim=1).validate()
    with pytest.raises(ValueError):
        _cfg(cfl=0.9).validate()
    with pytest.raises(ValueError):
        _cfg(c0=-1.0).validate()
    with pytest.raises(ValueError):
        _cfg(viscosity=-1.0).validate()
    with pytest.raises(ValueError):
        _cfg(beta_av=-1.0).validate()


def test_mirror_boundary_needs_a_plane():
    cfg = _cfg(boundary="mirror")
    with pytest.raises(ValueError):
        Solver(cfg, torch.zeros((2, 2), dtype=torch.float64),
               torch.zeros((2, 2), dtype=torch.float64),
               torch.ones(2, dtype=torch.float64))


def test_mirror_images_reflect_position_and_normal_velocity():
    pos = torch.tensor([[0.005, 0.02], [0.03, 0.02]], dtype=torch.float64)
    vel = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    mass = torch.tensor([1.0, 1.0], dtype=torch.float64)
    gpos, gvel, gmass, src = mirror_images(pos, vel, mass, [(0, 0.0)], 0.04)
    assert src.tolist() == [0, 1]
    assert float(gpos[0, 0]) == pytest.approx(-0.005)
    assert float(gpos[0, 1]) == pytest.approx(0.02)
    assert float(gvel[0, 0]) == pytest.approx(-1.0)
    assert float(gvel[0, 1]) == pytest.approx(2.0)


def test_mirror_images_are_culled_beyond_the_support():
    pos = torch.tensor([[0.5, 0.02]], dtype=torch.float64)
    vel = torch.zeros((1, 2), dtype=torch.float64)
    mass = torch.ones(1, dtype=torch.float64)
    gpos, _, _, src = mirror_images(pos, vel, mass, [(0, 0.0)], 0.04)
    assert gpos.numel() == 0 and src.numel() == 0


def _lattice(nx, ny, dx, origin=(0.0, 0.0)):
    xs = origin[0] + dx * (torch.arange(nx, dtype=torch.float64) + 0.5)
    ys = origin[1] + dx * (torch.arange(ny, dtype=torch.float64) + 0.5)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def test_free_cloud_momentum_is_conserved_to_round_off():
    """No walls, no gravity: only the antisymmetric pair force acts."""
    cfg = _cfg(alpha_av=0.1, c0=20.0)
    pos = _lattice(6, 6, cfg.dx)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx**2, dtype=torch.float64)
    vel = torch.zeros_like(pos)
    vel[:, 0] = 0.2 * torch.sin(pos[:, 1] / cfg.dx)
    solver = Solver(cfg, pos, vel, mass)
    p0 = solver.momentum()
    for _ in range(50):
        solver.step()
    drift = float((solver.momentum() - p0).abs().max() / p0.abs().max())
    assert drift < 1.0e-12


def test_hydrostatic_column_pressure_matches_rho_g_h():
    """A settled column at c0 = 20 m/s must recover p = rho g h within 5 %."""
    cfg = _cfg(gravity=9.81, c0=20.0, alpha_av=0.1, boundary="dynamic")
    dx, depth = 0.01, 0.20
    pos = _lattice(5, 20, dx)
    from tensorsph.wcsph.cases import box_walls
    wall = box_walls([0.0, 0.0], [5 * dx, depth], dx, 2, layers=2)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx**2, dtype=torch.float64)
    solver = Solver(cfg, pos, torch.zeros_like(pos), mass, wall_pos=wall,
                    box=(0.0, 5 * dx, 0.0, depth))
    for _ in range(400):
        solver.step()
    bottom = solver.state.pressure[solver.state.pos[:, 1] < 1.5 * dx]
    expected = cfg.rho0 * cfg.gravity * depth
    mean = float(bottom.mean())
    assert abs(mean - expected) / expected < 0.05


def test_wall_pressure_is_positive_under_a_column():
    cfg = _cfg(gravity=9.81, c0=20.0, alpha_av=0.1, boundary="dynamic")
    dx = 0.01
    pos = _lattice(4, 10, dx)
    from tensorsph.wcsph.cases import box_walls
    wall = box_walls([0.0, 0.0], [4 * dx, 10 * dx], dx, 2, layers=2)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx**2, dtype=torch.float64)
    solver = Solver(cfg, pos, torch.zeros_like(pos), mass, wall_pos=wall,
                    box=(0.0, 4 * dx, 0.0, 10 * dx))
    for _ in range(60):
        solver.step()
    assert float(solver.state.wall_pressure.max()) > 0.5 * cfg.rho0 * cfg.gravity * 10 * dx


def test_box_clamp_keeps_particles_inside():
    cfg = _cfg()
    pos = torch.tensor([[0.5, 0.5], [-0.01, 0.02]], dtype=torch.float64)
    vel = torch.tensor([[1.0, 0.0], [-5.0, 0.0]], dtype=torch.float64)
    mass = torch.full((2,), cfg.rho0 * cfg.dx**2, dtype=torch.float64)
    solver = Solver(cfg, pos, vel, mass, box=(0.0, 1.0, 0.0, 1.0))
    solver._apply_box()
    assert float(solver.state.pos.min()) >= 0.0
    assert float(solver.state.vel[1, 0]) == 0.0


def test_adaptive_h_updates_from_the_measured_density():
    cfg = _cfg(h_adaptive=True, gravity=9.81, alpha_av=0.1)
    pos = _lattice(5, 5, cfg.dx)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx**2, dtype=torch.float64)
    solver = Solver(cfg, pos, torch.zeros_like(pos), mass)
    solver.step()
    h = solver.state.h
    assert float(h.min()) >= 0.8 * cfg.h0 - 1e-12
    assert float(h.max()) <= 1.25 * cfg.h0 + 1e-12
    assert not torch.equal(h, torch.full_like(h, cfg.h0))


def test_viscous_branch_decreases_kinetic_energy():
    """Same start, two runs: the viscous one must end with less kinetic energy."""
    pos = _lattice(6, 6, 0.01)
    mass = torch.full((pos.shape[0],), 1000.0 * 0.01**2, dtype=torch.float64)
    vel = torch.zeros_like(pos)
    vel[:, 0] = 0.2 * torch.sin(pos[:, 1] / 0.01)
    energies = {}
    for label, mu in (("inviscid", 0.0), ("viscous", 0.01)):
        cfg = _cfg(viscosity=mu, alpha_av=0.0)
        solver = Solver(cfg, pos.clone(), vel.clone(), mass.clone())
        for _ in range(30):
            solver.step()
        energies[label] = solver.kinetic_energy()
    assert energies["viscous"] < energies["inviscid"]

def test_mirror_boundary_runs_and_keeps_the_fluid_above_the_floor():
    cfg = _cfg(boundary="mirror", gravity=9.81, alpha_av=0.1)
    pos = _lattice(5, 6, cfg.dx, origin=(0.0, 0.0))
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx**2, dtype=torch.float64)
    solver = Solver(cfg, pos, torch.zeros_like(pos), mass,
                    mirror_planes=[(1, 0.0), (0, 0.0)])
    for _ in range(20):
        solver.step()
    assert torch.isfinite(solver.state.pos).all()
    assert float(solver.state.pos[:, 1].min()) > -cfg.dx


def test_density_renormalisation_removes_the_lattice_bias():
    plain = _cfg()
    renorm = _cfg(density_renormalisation=True)
    pos = _lattice(10, 10, plain.dx)
    mass = torch.full((pos.shape[0],), plain.rho0 * plain.dx**2, dtype=torch.float64)
    a = Solver(plain, pos, torch.zeros_like(pos), mass)
    b = Solver(renorm, pos, torch.zeros_like(pos), mass)
    a.force(0.0)
    b.force(0.0)
    interior = torch.ones(pos.shape[0], dtype=torch.bool)
    interior[[0, 9, 90, 99]] = False
    bias_plain = float((a.state.rho[interior] / plain.rho0 - 1.0).abs().max())
    bias_renorm = float((b.state.rho[interior] / renorm.rho0 - 1.0).abs().max())
    assert bias_renorm < bias_plain
