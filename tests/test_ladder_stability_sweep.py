"""Guards for the method-A resolution-ladder sweep (XVAL-SPH-METHOD-SELECTION-2026-11-12).

The sweep driver turns driver records into the ladder table, and the ladder table
is what the report claims from.  These tests pin the three places where such a
table can silently lie:

* the step size and the resolution knob must be the ones the *shipped* driver
  uses (``run.py::_step``), otherwise the reported blow-up *time* is wrong;
* a configuration the solver rejects is not an instability, and a blow-up is not
  a rejection — the two verdicts must not collapse into "failed";
* the 0.60 s horizon must stay shorter than the first wave arrival at the
  gauges, otherwise the sweep could be read as an accuracy measurement it never
  made.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks.xval_method_selection import ladder_stability_sweep as sweep  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wave_run():
    return _load("wave_flume_run", REPO / "benchmarks" / "wave_flume" / "run.py")


def test_step_size_matches_the_shipped_driver(wave_run):
    for depth_cells in (13, 16, 20, 26):
        args = wave_run.argparse.Namespace(
            dt=sweep.DT_REQUESTED, depth=sweep.DEPTH_M,
            depth_cells=depth_cells, max_courant=sweep.MAX_COURANT,
            sound_speed=sweep.SOUND_SPEED)
        assert sweep.step_size(depth_cells) == pytest.approx(wave_run._step(args), rel=1e-12)


def test_resolution_knob_reproduces_the_pre_registered_cells():
    # run_cell.py: cell 0 is H/dx = 3.20 (20 depth cells), cell 1 is 2.08 (13).
    assert sweep.h_over_dx(20) == pytest.approx(3.20, abs=1e-9)
    assert sweep.h_over_dx(13) == pytest.approx(2.08, abs=1e-9)


def test_fine_cell_record_is_graded_as_a_blow_up_with_its_time():
    record = {"failed": True, "error": "FloatingPointError: non-finite acceleration at step 3890"}
    row = sweep.summarise_point("ladder_d20", 20, 0.60, record, 2, "")
    assert row["verdict"] == "blew_up"
    assert row["onset_step"] == 3890
    # 3890 steps of dt = 1.354e-4 s is the t = 0.53 s reported for cell 0.
    assert row["onset_time_s"] == pytest.approx(0.527, abs=5e-3)
    assert row["stable"] is False


def test_rejected_configuration_is_not_reported_as_an_instability():
    record = {"failed": True,
              "error": "ValueError: wave_height/dx = 1.60: the crest is not resolved"}
    row = sweep.summarise_point("ladder_d10", 10, 0.60, record, 2, "")
    assert row["verdict"] == "rejected"
    assert row["onset_time_s"] is None
    assert row["stable"] is False


def test_survived_rows_are_split_by_the_stability_gates():
    stable = {"failed": False, "diagnostics": {
        "simulated_time": 0.6, "max_speed": 0.04, "max_density_compression": 0.003,
        "escaped": 0}}
    row = sweep.summarise_point("ladder_d13", 13, 0.60, stable, 0, "")
    assert row["verdict"] == "survived_stable" and row["stable"] is True
    assert row["speed_ratio"] == pytest.approx(0.04 / sweep.SOUND_SPEED)

    # Same run, but compression above the 10 % gate: survives the horizon yet is
    # not a stable cell (this is exactly the archived h = 0.20 m record's state).
    compressible = {"failed": False, "diagnostics": {
        "simulated_time": 0.6, "max_speed": 0.6, "max_density_compression": 0.449,
        "escaped": 144}}
    row = sweep.summarise_point("ladder_d20", 20, 0.60, compressible, 0, "")
    assert row["verdict"] == "survived_unstable" and row["stable"] is False


def test_missing_record_is_never_silently_graded():
    row = sweep.summarise_point("ladder_d26", 26, 0.60, None, 255, "free(): corrupted chunks")
    assert row["verdict"] == "no_record"
    assert row["stable"] is False
    assert "corrupted" in row["error"]


def test_sweep_labels_are_unique_and_mechanism_probes_keep_their_knobs():
    labels = [point[0] for point in sweep.SWEEP]
    assert len(labels) == len(set(labels))
    knobs = {point[0]: point[3] for point in sweep.SWEEP}
    assert knobs["mech_d20_xsph"] == ["--xsph", "0.05"]
    assert "--density-regularization" in knobs["mech_d20_comp"]
    # Compression regularization and δ-SPH diffusion are mutually exclusive in the
    # driver, so the probe must switch the diffusion off explicitly.
    assert knobs["mech_d20_comp"][:2] == ["--density-diffusion", "0"]


def test_horizon_stays_short_of_the_first_wave_arrival():
    """The sweep may not claim wave height: no wave reaches the gauges inside it."""
    from tensorsph.wave_flume import (
        WaveFlumeConfig, default_probe_positions, flume_wave_spec, wavelength)

    depth_cells = min(point[1] for point in sweep.SWEEP)
    horizon = max(point[2] for point in sweep.SWEEP)
    dx = sweep.DEPTH_M / depth_cells
    celerity = wavelength(1.0, sweep.DEPTH_M, 9.81) / 1.0
    cfg = WaveFlumeConfig(flume_length=3.2, depth=sweep.DEPTH_M, wave_type="regular",
                          wave_height=sweep.HEIGHT_RATIO * sweep.DEPTH_M, wave_period=1.0,
                          dx=dx, dt=sweep.step_size(depth_cells), steps=1,
                          sound_speed=sweep.SOUND_SPEED,
                          sponge_length=0.9)
    spec = flume_wave_spec(cfg)
    first_probe = min(default_probe_positions(cfg, spec))
    arrival = first_probe / celerity
    assert horizon < arrival, (
        f"horizon {horizon} s reaches probe x = {first_probe:.2f} m at {arrival:.2f} s; "
        "a longer horizon needs an accuracy claim (15 % gate), not a stability sweep")