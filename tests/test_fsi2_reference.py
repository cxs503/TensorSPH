"""Archived SPHinXsys ``test_2d_fsi2`` records: parsing, units, acceptance gate.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The tests pin (a) the parsed content of the verbatim archive, (b) the unit
scaling derived from the data, and (c) the *negative* result the audit reports:
the OSS record is ~9x the benchmark's beam-tip amplitude, so it cannot be used
as the acceptance reference for Turek-Hron FSI2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import fsi2_reference as ref  # noqa: E402


def test_archive_is_present_and_complete():
    assert ref.REFERENCE_DIR.is_dir(), ref.REFERENCE_DIR
    for run in ("0", "4", "9"):
        n, pos = ref.load_tip_series(run)
        assert n == 107, (run, n)
        assert pos.shape == (107, 2), (run, pos.shape)
    mv = ref.load_mean_variance()
    assert any(k.startswith("TotalViscousForceFromFluid_") for k in mv["mean"])


def test_length_scale_is_recovered_from_the_data():
    """A straight beam tip sits at 0.6 m, stored at ~6.0 -> length scale 10."""
    for run in ("0", "4", "9"):
        _, pos = ref.load_tip_series(run)
        assert ref.infer_length_scale(pos) == pytest.approx(ref.LENGTH_SCALE, abs=0.5)


def test_tip_statistics_pin_the_record():
    """Regression pinning of the values the audit quotes (physical units)."""
    _, pos = ref.load_tip_series("0")
    s = ref.summarise_tip(pos)
    assert s["tip_y_amplitude"] == pytest.approx(0.0796685, rel=1e-6)
    assert s["tip_x_amplitude"] == pytest.approx(0.0155770, rel=1e-6)
    assert s["y_mean"] == pytest.approx(0.2028119, rel=1e-6)
    assert s["n_samples"] == 107


def test_reference_runs_are_reproducible():
    """The three upstream runs agree to <10 % on the tip amplitude."""
    amps = [ref.summarise_tip(ref.load_tip_series(r)[1])["tip_y_amplitude"]
            for r in ("0", "4", "9")]
    assert (max(amps) - min(amps)) / np.mean(amps) < 0.10, amps


def test_oss_record_fails_the_benchmark_gate():
    """Finding: the OSS record is 4.6x / 8.7x the benchmark amplitudes."""
    b = ref.bundle()
    gate = b["tip_gate"]
    assert not gate["passed_15"]
    assert not gate["passed_3"]
    rows = {r["quantity"]: r for r in gate["rows"]}
    assert rows["tip_x_amplitude"]["rel_error"] == pytest.approx(4.5632, rel=1e-3)
    assert rows["tip_y_amplitude"]["rel_error"] == pytest.approx(8.7157, rel=1e-3)
    assert gate["worst_rel_error"] == pytest.approx(8.7157, rel=1e-3)


def test_force_unit_conversion_and_friction_cross_check():
    """Scaled force -> N/m -> Cd, checked against a laminar flat-plate estimate."""
    mv = ref.load_mean_variance()
    f_scaled = next(iter(mv["mean"].values()))
    assert f_scaled[0] == pytest.approx(ref.OSS_VISCOUS_FORCE_SCALED[0], rel=1e-12)
    assert f_scaled[1] == pytest.approx(ref.OSS_VISCOUS_FORCE_SCALED[1], rel=1e-10)
    f_phys = ref.viscous_force_physical(f_scaled)
    assert f_phys[0] == pytest.approx(45.1157, rel=1e-6)  # x100 (rho U^2 / L_scale)
    cd = ref.drag_coefficient(f_phys)
    assert cd[0] == pytest.approx(0.902314, rel=1e-6)
    est = ref.independent_friction_estimate()
    # order-of-magnitude agreement only: factor < 3 between the SPH record and
    # the laminar flat-plate estimate for the same body
    assert 1.0 < cd[0] / est["Cd_friction"] < 3.0, (cd[0], est)
    assert cd[1] > 0.0  # transverse (lift-like) viscous component is small


def test_compare_reports_per_quantity_gates():
    summary = {"tip_x_amplitude": 0.00285, "tip_y_amplitude": 0.0081}
    out = ref.compare_to_benchmark(summary)
    assert out["passed_15"] and out["passed_3"]
    summary = {"tip_x_amplitude": 0.0035, "tip_y_amplitude": 0.0081}
    out = ref.compare_to_benchmark(summary)  # 25 % on x -> fails the 15 % gate
    assert not out["passed_15"]
    assert out["rows"][0]["rel_error"] == pytest.approx(0.25, rel=1e-12)


def test_malformed_record_is_rejected():
    with pytest.raises(ValueError):
        ref.parse_tip_series("<result><nothing/></result>")
    with pytest.raises(ValueError):
        ref.parse_tip_series(
            '<Snapshot number_of_snapshot_for_local_result_="2" />'
            '<Particle_0 snapshot_0="~[1,2]" />')
    with pytest.raises(ValueError):
        ref.parse_mean_variance("<meanvariance></meanvariance>")


def test_missing_archive_raises_instead_of_silently_passing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ref.load_tip_series("0", directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        ref.load_mean_variance(directory=tmp_path)