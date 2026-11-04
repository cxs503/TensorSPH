"""Pin the initial-density offset of a summation-density lattice.

These tests hold the *initial condition* to the acceptance criterion of
``backlog_update_m2971.md`` (item C2 / P1)::

    max|p| / (rho0 c0^2) < 1e-4   after initialisation

and pin the four facts the diagnosis rests on:

1. the offset ``b`` is a pure lattice constant - identical at every spacing,
   so refinement cannot remove it;
2. the square lattice at ``h = 1.3 dx`` is the one already-balanced packing
   (which is why the droplet-impact production case never showed it);
3. the calibration is exact in one pass (density is linear in mass) and
   therefore reaches machine precision, at the documented mass cost
   ``-b/(1+b)``;
4. the absolute offset scales as ``c0^2`` while the normalised criterion does
   not move at all.

Numbers are taken from ``benchmarks/init_conditioning/report.json`` where the
benchmark already measured them, and recomputed here where the test is cheap
enough to stand alone.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config, density_from_neighbor_pairs, make_particles
from tensorsph.initial_conditions import (
    TAIT_ACCEPTANCE_LIMIT,
    calibrate_lattice_mass,
    measure_density_bias,
    spurious_pressure,
    supported_mask,
)
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parents[1]
RUN = HERE / "benchmarks" / "init_conditioning" / "run.py"
REPORT = HERE / "benchmarks" / "init_conditioning" / "report.json"

RHO0 = 1000.0
RADIUS = 0.02
SIGMA = 7.2
C0 = 20.0
GAMMA = 7.0
SIGNAL_PA = SIGMA / RADIUS


def _load_run():
    spec = importlib.util.spec_from_file_location("init_conditioning_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load_run()


@pytest.fixture(scope="module")
def hex_case():
    pos, mass, _ = R.droplet_lattice(RADIUS, 0.002, "hex", rho0=RHO0)
    h = 1.3 * 0.002
    pairs = cell_linked_pairs(pos, 2.0 * h)
    return pos, mass, h, pairs


def test_lattice_offset_is_resolution_independent():
    """b does not move with the spacing: it is the packing, not the mesh."""
    biases = []
    for dx in (0.004, 0.002, 0.001):
        pos, mass, _ = R.droplet_lattice(RADIUS, dx, "hex", rho0=RHO0)
        stats = measure_density_bias(pos, mass, 1.3 * dx, RHO0)
        biases.append(stats["bias"])
    assert max(biases) - min(biases) < 1e-6
    assert biases[0] == pytest.approx(0.00588, abs=5e-5)


def test_supported_mask_excludes_the_surface_layer():
    """Only the saturated interior may enter the fit."""
    pos, mass, _ = R.droplet_lattice(RADIUS, 0.002, "hex", rho0=RHO0)
    h = 1.3 * 0.002
    mask = supported_mask(pos, h)
    r = torch.linalg.vector_norm(
        pos - torch.tensor([R.CENTER, R.CENTER], dtype=torch.float64), dim=1)
    # Selected particles reach the kernel radius 2h = 2.6 dx in from the rim
    # (R - 2.6*0.002 = 0.0148 m = 0.74 R), so the rim ring must be excluded
    # while the plateau is not.
    assert float(r[mask].max()) < 0.85 * RADIUS
    assert not bool(mask[r > 0.9 * RADIUS].any())
    assert 0.3 < int(mask.sum()) / pos.shape[0] < 0.95


def test_square_lattice_at_h13_is_already_balanced():
    """The production packing has no offset - the reason the bug stayed hidden."""
    pos, mass, _ = R.droplet_lattice(RADIUS, 0.002, "square", rho0=RHO0)
    stats = measure_density_bias(pos, mass, 1.3 * 0.002, RHO0)
    assert abs(stats["bias"]) < 5e-4


def test_uncalibrated_hex_initial_state_fails_the_criterion(hex_case):
    """The honest 'before' state: 6.65 x the sigma/R signal, criterion failed."""
    pos, mass, h, pairs = hex_case
    out = spurious_pressure(pos, mass, h, RHO0, C0, GAMMA, pairs=pairs)
    assert out["p_max_over_rho0c0sq"] > TAIT_ACCEPTANCE_LIMIT * 10
    assert not out["passes_acceptance"]
    # absolute offset, and its ratio to the Young-Laplace signal
    assert out["p_abs_max"] == pytest.approx(2395.4, rel=1e-3)
    assert out["p_abs_max"] / SIGNAL_PA == pytest.approx(6.65, abs=0.05)


def test_calibration_reaches_machine_precision(hex_case):
    """Density is linear in mass, so one pass is exact - and provably so."""
    pos, mass, h, pairs = hex_case
    mass_cal, info = calibrate_lattice_mass(pos, mass, h, RHO0, pairs=pairs)
    assert info["bias_after"] == pytest.approx(0.0, abs=1e-12)
    out = spurious_pressure(pos, mass_cal, h, RHO0, C0, GAMMA, pairs=pairs)
    assert out["passes_acceptance"]
    assert out["p_max_over_rho0c0sq"] < 1e-12
    # documented trade-off: density is preserved, total mass is not
    b = info["bias"]
    assert info["mass_change"] == pytest.approx(-b / (1.0 + b), rel=1e-12)
    assert abs(info["mass_change"]) < 0.01


def test_offset_scales_as_c0_squared_but_criterion_does_not(hex_case):
    """Lowering c0 shrinks the pressure, never the relative error."""
    pos, mass, h, pairs = hex_case
    norms, absolutes = [], []
    for c0 in (5.0, 10.0, 20.0, 40.0):
        out = spurious_pressure(pos, mass, h, RHO0, c0, GAMMA, pairs=pairs)
        norms.append(out["p_max_over_rho0c0sq"])
        absolutes.append(out["p_abs_max"])
    assert max(norms) - min(norms) < 1e-9           # identical, by construction
    # p ~ c0^2: a factor 8 in c0 from 5 to 40 m/s is a factor 64 in pressure
    assert absolutes[-1] / absolutes[0] == pytest.approx(64.0, rel=2e-2)


def test_config_flag_wires_the_calibration_into_make_particles():
    """The flag exists, is off by default, and lowers the initial pressure."""
    assert Config().lattice_mass_calibration is False
    base = Config(dx=0.05, steps=1)
    cal = Config(dx=0.05, steps=1, lattice_mass_calibration=True)
    pos_a, mass_a = make_particles(base, torch.device("cpu"))
    pos_b, mass_b = make_particles(cal, torch.device("cpu"))
    assert torch.equal(pos_a, pos_b)                 # geometry untouched
    assert not torch.allclose(mass_a, mass_b)        # masses rescaled
    ratio = float(mass_b.sum() / mass_a.sum())
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_report_pins_the_measured_numbers():
    """The doc's numbers must come from the committed report, not from prose."""
    if not REPORT.exists():
        pytest.skip("run benchmarks/init_conditioning/run.py to regenerate")
    report = json.loads(REPORT.read_text())
    su = report["summary"]
    assert su["hex_bias_dx_independent"] is True
    assert su["all_pass_after"] is True
    assert su["worst_p_norm_after"] < TAIT_ACCEPTANCE_LIMIT
    assert su["hex_p_norm_before"][0] > 10 * TAIT_ACCEPTANCE_LIMIT
    assert su["hex_mass_change_percent"] == pytest.approx(-0.585, abs=0.01)
    # the impact probe: C2 is not the cause of the force-integral gap
    impact = report["laplace_impact"][0]
    assert abs(impact["dp_ratio_delta_points"]) < 2.0
    assert impact["n_delta_calibrated"] == pytest.approx(
        impact["n_delta_uncalibrated"], rel=1e-9)