"""Regression checks for marine wall loads and flexural-gravity propagation."""
import math

import pytest
import torch

from tensorsph.dam_break_3d import DamBreak3DConfig, rhs_3d
from tensorsph.kleefsman_obstacle_3d import Kleefsman3DConfig
from tensorsph.threed.kernel3 import calibrated_particle_mass
from tensorsph.wave_ice_interaction import G, wave_number_dispersion, wave_number_under_ice


def test_uniform_interior_includes_kernel_self_weight():
    cfg = DamBreak3DConfig(dx=.02, gravity=0, viscosity=0)
    axis = torch.arange(7, dtype=torch.float64)*cfg.dx
    pos = torch.cartesian_prod(axis, axis, axis)
    mass = pos.new_full((len(pos),), calibrated_particle_mass(cfg.dx, cfg.h_factor*cfg.dx,
                                                             cfg.rho0))
    density, _, _, loads = rhs_3d(pos, pos*0, mass, cfg, return_wall_force=True)
    assert density[len(pos)//2].item() == pytest.approx(cfg.rho0, rel=1e-12)
    assert loads.shape == (0, 3)


@pytest.mark.parametrize("friction", [0., 1.])
def test_single_fluid_wall_pair_exchanges_opposite_force(friction):
    cfg = DamBreak3DConfig(dx=.02, sound_speed=24., gravity=9.81, wall_friction=friction,
                           viscosity=.001, surface_pressure_closure=True)
    pos = torch.tensor([[0., .01, 0.]], dtype=torch.float64)
    wall = pos.new_tensor([[0., -.01, 0.]])
    vel = pos.new_tensor([[1., 0., 0.]])
    mass = pos.new_tensor([cfg.rho0*cfg.dx**3])
    baseline = rhs_3d(pos, vel, mass, cfg, wall=wall)
    *fluid, loads = rhs_3d(pos, vel, mass, cfg, wall=wall, return_wall_force=True)
    for observed, default in zip(fluid, baseline):
        torch.testing.assert_close(observed, default)
    gravity = pos.new_tensor([0., -cfg.gravity, 0.])
    torch.testing.assert_close((fluid[2]-gravity)*mass[:, None], -loads)
    assert loads[0, 1] < 0
    assert (loads[0, 0].item() > 0) == bool(friction)


def test_published_box_and_gauge_share_the_tank_centreline():
    cfg = Kleefsman3DConfig()
    assert cfg.obstacle_x == pytest.approx(2.3955)
    assert cfg.station_x() == pytest.approx(2.228)
    assert cfg.obstacle_z + cfg.obstacle_width/2 == pytest.approx(.5)


def test_ice_limits_and_mass_loading_above_open_water_wavenumber():
    w, depth, h, rho = 2., 2000., .5, .9
    open_k = wave_number_dispersion(w, depth)
    assert wave_number_under_ice(w, 0., depth) == open_k
    expected = w*w/(G-rho*h*w*w)
    assert wave_number_under_ice(w, h, depth, E=0, rho_water=1000,
                                 rho_ice=900) == pytest.approx(expected, rel=1e-11)
    assert expected > open_k
    with pytest.raises(ValueError, match="no propagating"):
        wave_number_under_ice(20., h, depth, E=0)


def test_finite_depth_plate_against_independent_forward_frequency():
    k, h, depth, modulus, nu = 1.82, .05, .55, 107e6, .33
    # Eq. D.3, Li et al. 2021, solved in the forward direction.
    bending = modulus*h**3/(12*(1-nu**2))
    frequency = math.sqrt((1025*G+bending*k**4) /
                          (917*h + 1025/(k*math.tanh(k*depth))))
    assert wave_number_under_ice(frequency, h, depth, modulus, nu) == pytest.approx(k, rel=1e-11)


def test_resume_wall_load_csv_only_keeps_the_checkpoint_prefix(tmp_path):
    import csv
    from pathlib import Path
    import runpy

    entry = Path(__file__).resolve().parents[1]/"benchmarks"/"kleefsman_obstacle_3d"/"run.py"
    driver = runpy.run_path(str(entry))
    columns = driver["SERIES_COLUMNS"]
    path = tmp_path/"loads.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for t in (0., .1, .2):
            writer.writerow([f"{t:.6f}"] + [0.]*(len(columns)-1))
    driver["resume_series"](path, [0., .1])
    assert len(path.read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="missing"):
        driver["resume_series"](path, [0., .1, .2])
