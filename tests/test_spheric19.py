from pathlib import Path

import pytest
import torch

from tensorsph.spheric19 import GateMotion, simulate


DATA = Path(__file__).resolve().parents[1] / "benchmarks" / "spheric19"


def test_gate_interpolates_and_stops_above_water(tmp_path):
    data = tmp_path / "gate.dat"
    data.write_text("# seconds, metres\n0 0\n0.1 0.02\n0.15 0.12\n")
    trace = GateMotion.read(data)
    assert trace.at(.125) == pytest.approx((.07, 2.))
    assert trace.at(.1) == pytest.approx((.02, 2.))
    assert trace.at(1.) == (.12, 0.)
    with pytest.raises(ValueError):
        GateMotion((0., .2, .2), (0., .1, .3))
    with pytest.raises(ValueError):
        GateMotion((0., .1), (0., float("nan")))
    figure = GateMotion.read(DATA / "gate_fig2.dat")
    assert figure.at(0.)[0] == 0. and figure.at(.15)[0] > .6


@pytest.mark.parametrize("method", ["wcsph", "delta-plus"])
def test_separate_gate_leaves_downstream_strip_dry_at_start(method):
    # This rigid limit exercises the fluid adapter without the optional FEM checkout.
    class FixedStrip:
        length = .09

        def station_weights(self, arcs):
            return torch.as_tensor(arcs, dtype=torch.float64)

        def sample(self, arcs):
            points = torch.stack((torch.zeros_like(arcs), arcs), dim=1)
            return points, torch.zeros_like(points)

        def advance(self, arcs, forces, dt):
            assert forces.shape == (len(arcs), 2)
            assert forces.abs().max() == 0

    history, info = simulate(GateMotion.read(DATA / "gate_fig2.dat"), FixedStrip(),
                             dx=.02, end=.002, method=method)
    assert history[-1] == (.002, 0., 0., 0.)
    assert info["fluid_particles"] == 150
    assert info["method"] == method
    assert info["steps"] > 1
    assert info["max_tank_escape_m"] == 0
    if method == "delta-plus":
        _, soft = simulate(GateMotion.read(DATA / "gate_fig2.dat"), FixedStrip(),
                           dx=.02, end=.0001, sound_speed=10., method=method)
        assert soft["weak_compressibility_gate"] == "fail"
        assert soft["first_weak_compressibility_exit"]["t_s"] == 0.


@pytest.mark.parametrize(("method", "end"), [("wcsph", .30), ("delta-plus", .36)])
def test_real_fem_receives_impact_and_returns_moving_markers(method, end):
    module = pytest.importorskip("tensorfem.dynamic_strip")
    strip = module.DynamicStrip2D(length=.09, thickness=.004, young=3.5e6,
                                  density=1161.54, elements=4)
    gate = GateMotion.read(DATA / "gate_fig2.dat")
    rows, info = simulate(gate, strip, dx=.02, end=end, method=method)
    top = [r[1] for r in rows]
    assert info["absolute_station_work_J"] > 0
    assert info["partition_work_residual_J"] == pytest.approx(
        info["plate_work_on_fluid_J"] + info["plate_work_on_structure_J"])
    assert max(top) > .005
    assert max(top) > max(r[3] for r in rows)
    assert rows[-1][0] == pytest.approx(end)
    if method == "delta-plus":
        assert info["weak_compressibility_gate"] == "pass"
        assert 0 < info["max_shift_over_dx"] < .1
        assert info["max_horizontal_force_imbalance_N"] < 1.e-8


def test_tank_contact_impulse_leaves_tangential_motion_and_open_top_free():
    from tensorsph.spheric19 import tank_contact

    pos = torch.tensor([[.81, -.01], [-.01, .12], [.81, .61]], dtype=torch.float64)
    vel = torch.tensor([[2., -3.], [3., 4.], [2., 4.]], dtype=torch.float64)
    mass = torch.tensor([.1, .2, .3], dtype=torch.float64)
    previous = torch.tensor([[.79, .01], [.001, .12], [.79, .61]], dtype=torch.float64)
    impulse = tank_contact(previous, pos, vel, mass, .8, .6)
    assert impulse == pytest.approx(.5)
    assert vel.tolist() == [[0., 0.], [3., 4.], [2., 4.]]
    assert pos.tolist() == [[.8, 0.], [0., .12], [.81, .61]]
    outside = pos.clone()
    outside[2, 1] = .59
    assert tank_contact(pos, outside, vel, mass, .8, .6) == 0.
    assert outside[2, 0] == .81
