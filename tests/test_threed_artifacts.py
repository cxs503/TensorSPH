"""Guard the committed 3-D artifacts, not just the code that made them.

Each benchmark writes a machine-readable report whose criteria are evaluated at
merge time.  These tests re-read the *committed* report and check the criteria as
well as a few invariants that would catch a silently regenerated artifact:

* the operator ladder must be monotone in ``k h`` for the corrected Laplacian
  (a non-monotone ladder means the measurement, not the operator, is broken);
* the shear-wave momentum drift must stay at round-off in every case;
* the hydrostatic per-layer table must contain the odd-even mode explicitly, so
  the headline number (bottom load) cannot be read as a per-layer accuracy claim.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "benchmarks" / "threed_kernel" / "report.json"
SHEAR = REPO / "benchmarks" / "threed_shear" / "report.json"
HYDRO = REPO / "benchmarks" / "threed_hydrostatic" / "report.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        pytest.fail(f"missing 3-D benchmark artifact {path}; run "
                    f"`PYTHONPATH=src python {path.parent}/run.py --merge`")
    return json.loads(path.read_text())


def test_kernel_report_meets_its_criteria():
    report = _load(KERNEL)
    assert report["missing_cases"] == []
    assert report["verdict"] == "pass", report["criteria"]
    assert all(report["criteria"].values())

    quadrature = report["quadrature"]
    assert abs(quadrature["kernel_integral_error"]) < 1e-12
    assert abs(quadrature["second_moment_error"]) < 1e-6
    assert abs(quadrature["laplacian_at_zero"]
               - quadrature["laplacian_at_zero_analytic"]) < 1e-6 * abs(
                   quadrature["laplacian_at_zero_analytic"])

    # C is a lattice constant: the dx ladder must be bit-identical.
    assert report["dx_independence"]["spread"] < 1e-14

    ladder = report["operator_ladder"]
    corrected = [row["laplacian_corrected_rel_l2"] for row in ladder]
    assert corrected == sorted(corrected, reverse=True), "ladder must be monotone"
    for row in ladder:
        assert row["second_moment_K_spread"] < 1e-12
        assert 2.0 < row["second_moment_K"] < 2.5


def test_shear_report_meets_its_criteria():
    report = _load(SHEAR)
    assert report["missing_cases"] == []
    assert report["verdict"] == "pass", report["criteria"]
    assert all(report["criteria"].values())

    for row in report["resolution_ladder"]:
        cons = row["conservation"]
        assert cons["momentum_final_over_scale"] < 1e-12
        assert cons["mass_relative_drift"] == 0.0
    finest = min(report["resolution_ladder"], key=lambda r: r["kh"])
    assert abs(finest["decay"]["rel_error"]) < 0.03

    pair = report["raw_vs_renormalised"]
    assert abs(pair["renormalised"]["decay"]["rel_error"]) < abs(
        pair["raw"]["decay"]["rel_error"])


def test_hydrostatic_report_meets_its_criteria():
    report = _load(HYDRO)
    assert report["missing_cases"] == []
    assert report["verdict"] == "pass", report["criteria"]
    assert all(report["criteria"].values())

    for run in report["runs"]:
        assert run["conservation"]["wall_displacement"] == 0.0
        # The per-layer table is part of the record: it is where the odd-even mode
        # shows up, and the headline number must not be read without it.
        assert len(run["profile"]) == run["nz"]
        assert run["pressure"]["odd_even_amplitude_over_scale"] is not None

    adami = [r for r in report["runs"] if r["wall_model"] == "adami"]
    tait = [r for r in report["runs"] if r["wall_model"] == "tait"]
    assert adami and tait
    # The control run is kept on purpose: the Tait-only wall model does not hold
    # the column up, and that has to stay visible in the artifact.
    assert abs(tait[0]["pressure"]["bottom_rel_error"]) > abs(
        adami[0]["pressure"]["bottom_rel_error"])
