from pathlib import Path
import runpy

import pytest
import torch

from tensorsph.ice_thermal import properties as props
from tensorsph.ice_thermal.sessile_drop import _laplacian, simulate


def test_conservative_graph_has_unit_three_dimensional_diffusivity():
    x = torch.arange(7, dtype=torch.float64)*.1
    pos = torch.cartesian_prod(x, x, x)
    op, _ = _laplacian(pos, .1)
    quadratic = pos.square().sum(1)
    interior = ((pos > .11) & (pos < .49)).all(1)
    rate = op @ quadratic
    torch.testing.assert_close(rate[interior], torch.full_like(rate[interior], 6.))
    assert abs(float(rate.sum())) < 1e-10
    assert float((op@torch.ones_like(quadratic)).abs().max()) < 1e-11


@pytest.mark.parametrize("glass", [0., .001])
def test_plate_energy_and_post_nucleation_progress(glass):
    result = simulate(cells=8, glass_depth=glass, end_s=18.)
    record = result.metrics
    assert 0 < record["t99_s"] <= record["last_particle_frozen_s"] < 18
    assert record["sampled_energy_residual_over_latent"] < 1e-11
    fractions = [r["frozen_mass_fraction"] for r in result.history]
    assert fractions[0] == 0 and fractions[-1] == pytest.approx(1)
    assert all(b >= a-1e-12 for a, b in zip(fractions, fractions[1:]))
    assert result.positions.shape == (record["water_particles"], 3)
    assert all(f.shape == (int(result.section_mask.sum()),) for _, f in result.sections)
    if glass:
        assert record["last_particle_frozen_s"] > 2*simulate(cells=8, glass_depth=0.).metrics[
            "last_particle_frozen_s"]


def test_supercooled_liquid_retains_its_initial_energy_at_assumed_nucleation():
    r = simulate(cells=4, initial_water_c=-5, glass_depth=0., end_s=1e-4)
    assert r.history[0]["enthalpy_change_J"] == 0
    assert r.history[0]["frozen_mass_fraction"] == pytest.approx(
        props.C_FRESH*5/(props.L_FUSION+.05*props.C_ICE))


def test_run_stays_incomplete_on_a_warm_plate_or_short_clock():
    r = simulate(cells=4, plate_c=0., end_s=.2)
    assert r.metrics["last_particle_frozen_s"] is None
    assert r.metrics["t99_s"] is None
    assert r.history[-1]["extracted_J"] == 0 and not r.solid_fraction.any()
    with pytest.raises(ValueError):
        simulate(height=.003)

def test_three_dimensional_slab_against_one_phase_stefan_solution():
    benchmark = Path(__file__).resolve().parents[1]/"benchmarks"/"sessile_freezing"/"run.py"
    anchor = runpy.run_path(str(benchmark))["verify_slab"](8)
    assert abs(anchor["relative_error"]) < .03
