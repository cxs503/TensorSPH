"""Regression guards for the B2 dam-break reference audit.

Two things are pinned here, both recorded in
``benchmarks/dam_break/reference_audit.json`` and
``docs/benchmarks/dam-break-reference-audit.md``:

1. **The bundled "Martin & Moyce (1952)" table is not a digitisation.** Its
   abscissa is an exact arithmetic sequence and every ordinate is an exact
   multiple of 0.05, so it must not be used as a validation target. The table is
   kept (it is still a plot reference) but flagged in the JSON itself; the tests
   fail if that flag is ever removed without new provenance, and they fail if
   the numbers are replaced by something *with* scatter without updating the
   flag -- in both directions the honest state has to be rewritten by hand.
2. **The analytic reference is implemented correctly.** Ritter's coefficient is
   ``c = 2`` in ``d(Z/a)/d tau``; the harness used ``c = 1`` until this audit.
   The solver's measured coefficient is pinned per resolution and per floor
   boundary condition so a solver change cannot silently move it.

Both layers of the audit are covered, because both are pinned to the committed
artifact:

* layer A (lattice audit, Ritter identity, solver front coefficient) -- the
  numbers in ``payload["table_audit"]``, ``payload["ritter"]``,
  ``payload["table_slope"]`` and ``payload["runs"]``;
* layer B (structure, kinematics and cross-source spread of the two
  transcriptions, plus the three gates and the ``withheld`` verdict) -- the
  numbers in ``payload[{"tables", "grid_stats", "cross_source_spread", "gates",
  "verdict"}]``.

The error metric used by the resolution ladder is pinned separately with a
self-consistency case: comparing a curve against an interpolant of a table must
return zero error.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.dam_break.reference_audit import (  # noqa: E402
    _interp,
    cross_source_spread,
    evaluate,
    grid_stats,
    load_bundled,
    load_crosscheck,
)
from benchmarks.dam_break import reference_ladder  # noqa: E402
from benchmarks.dam_break.reference_ladder import error_vs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"
ARTIFACT = ROOT / "benchmarks" / "dam_break" / "reference_audit.json"
DOC = ROOT / "docs" / "benchmarks" / "dam-break-reference-audit.md"

# measured least-squares d(Z/a)/d tau; rerun reference_audit.py and edit these
# together with the artifact
C_PINNED = {
    (0.04, 0.00): 1.2016,
    (0.04, 0.25): 1.1099,
    (0.04, 0.50): 1.0580,
    (0.04, 0.75): 1.0123,
    (0.04, 1.00): 0.9682,
    (0.02, 0.00): 1.2468,
    (0.02, 1.00): 0.8959,
}
C_TOLERANCE = 0.03
RITTER_COEFFICIENT = 2.0


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text())


@pytest.fixture(scope="module")
def payload():
    assert ARTIFACT.exists(), f"missing {ARTIFACT.relative_to(ROOT)}"
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------- #
# 1. provenance of the validation target
# --------------------------------------------------------------------------- #


HERE = Path(__file__).resolve().parent.parent / "benchmarks" / "dam_break"


@pytest.fixture(scope="module")
def bundled():
    return load_bundled()


@pytest.fixture(scope="module")
def crosscheck():
    return load_crosscheck()


@pytest.fixture(scope="module")
def report(bundled, crosscheck):
    return evaluate({"bundled": bundled, "crosscheck": crosscheck})


# --------------------------------------------------------------------------- #
# tables load with the same convention
# --------------------------------------------------------------------------- #
def test_crosscheck_table_converts_to_the_same_frame(crosscheck):
    assert crosscheck["name"] == "crosscheck"
    assert len(crosscheck["tau"]) == 14
    # X = x_front/a, so the front at t = 0 sits one column width from the wall.
    assert crosscheck["za"][0] == pytest.approx(0.0, abs=1e-12)
    assert crosscheck["tau"] == sorted(crosscheck["tau"])
    assert crosscheck["za"] == sorted(crosscheck["za"])
    assert crosscheck["claimed_uncertainty"] == pytest.approx(0.05)


def test_both_tables_cover_the_same_aspect_ratio_window(bundled, crosscheck):
    assert bundled["tau"][0] == crosscheck["tau"][0] == 0.0
    overlap = min(bundled["tau"][-1], crosscheck["tau"][-1])
    assert overlap >= 2.9


# --------------------------------------------------------------------------- #
# structural facts of the bundled table
# --------------------------------------------------------------------------- #
def test_bundled_table_is_a_piecewise_linear_sketch(bundled):
    g = grid_stats(bundled["tau"], bundled["za"])
    assert g["tau_grid_is_arithmetic"] is True
    assert g["tau_step_spread"] < 1e-9
    assert g["ordinate_quantum"] == pytest.approx(0.05)
    assert g["n_slope_levels"] == 4
    # the four levels are separated by much more than the grouping tolerance
    levels = g["slope_levels"]
    assert levels == sorted(levels)
    assert levels[-1] / levels[0] > 2.0


def test_bundled_table_has_no_acceleration_phase(bundled, crosscheck):
    gb = grid_stats(bundled["tau"], bundled["za"])
    gc = grid_stats(crosscheck["tau"], crosscheck["za"])
    # The bundled curve starts at 80 % of its peak front speed
    assert gb["first_slope_to_peak"] > 0.75
    # ... the independent transcription starts at ~20 % and accelerates 6x
    assert gc["first_slope_to_peak"] < 0.25
    assert gc["slope_peak"] / gc["slope_first"] > 4.0
    # and it is not an arithmetic grid read off a round grid
    assert gc["ordinate_quantum"] <= 0.01


def test_cross_source_spread_dwarfs_the_claimed_uncertainty(bundled, crosscheck):
    spread = cross_source_spread(bundled, crosscheck)
    assert spread["n_overlap"] >= 13
    assert spread["max_abs_delta"] > 0.5
    assert spread["rms_delta"] > 0.4
    # The two transcriptions agree only at the end of the common window.
    last = spread["rows"][-1]
    assert abs(last["delta"]) < 0.1
    # ... and disagree strongly in the middle of it.
    assert max(abs(r["delta"]) for r in spread["rows"]) > 0.5


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def test_all_three_structural_gates_fail_on_the_bundled_table(report):
    gates = report["gates"]
    assert gates["bundled_is_pointwise"]["result"] == "fail"
    assert gates["acceleration_phase_present"]["result"] == "fail"
    assert gates["claimed_uncertainty_consistent"]["result"] == "fail"
    assert gates["claimed_uncertainty_consistent"]["measured"] > 10.0


def test_verdict_withholds_physical_validation(report):
    assert report["verdict"]["physical_validation_status"] == "withheld"
    assert "does not support" in report["verdict"]["reason"]


def test_committed_audit_artifact_is_reproducible(report):
    """The JSON in the repo must equal what the code computes now."""
    path = HERE / "reference_audit.json"
    assert path.exists(), "run benchmarks/dam_break/reference_audit.py"
    on_disk = json.loads(path.read_text())
    for name, gate in report["gates"].items():
        assert on_disk["gates"][name]["result"] == gate["result"]
    fresh = report["cross_source_spread"]
    disk = on_disk["cross_source_spread"]
    for key in ("max_abs_delta", "rms_delta", "mean_signed_delta"):
        assert disk[key] == pytest.approx(fresh[key])
    assert disk["n_overlap"] == fresh["n_overlap"]
    assert on_disk["verdict"]["physical_validation_status"] == \
        report["verdict"]["physical_validation_status"]


# --------------------------------------------------------------------------- #
# the error metric itself
# --------------------------------------------------------------------------- #
def test_error_metric_is_zero_against_its_own_interpolation(crosscheck):
    """A curve that *is* the reference interpolant must measure as exact."""
    tau = [0.005 * i for i in range(int(crosscheck["tau"][-1] / 0.005) + 1)]
    za = [_interp(crosscheck["tau"], crosscheck["za"], t) for t in tau]
    err = error_vs(tau, za, crosscheck)
    assert err["n_points"] == len(crosscheck["tau"])
    assert err["mae"] == pytest.approx(0.0, abs=1e-12)
    assert err["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_error_metric_detects_a_known_bias(crosscheck):
    """Add a constant offset: MAE must return exactly that offset."""
    tau = [0.005 * i for i in range(int(crosscheck["tau"][-1] / 0.005) + 1)]
    za = [_interp(crosscheck["tau"], crosscheck["za"], t) + 0.25 for t in tau]
    err = error_vs(tau, za, crosscheck)
    assert err["mae"] == pytest.approx(0.25, abs=1e-12)
    assert err["mean_signed"] == pytest.approx(0.25, abs=1e-12)


def test_gate_table_marks_the_limit_correctly():
    fake = [{"dx": 0.04, "particles": 50,
             "error_vs_crosscheck": {"mae": 0.10, "rmse": 0.11,
                                     "within_claimed_uncertainty": 0.5}},
            {"dx": 0.02, "particles": 200,
             "error_vs_crosscheck": {"mae": 0.20, "rmse": 0.21,
                                     "within_claimed_uncertainty": 0.3}}]
    table = reference_ladder.gate_table(fake, "crosscheck", 0.15)
    assert [r["result"] for r in table["rows"]] == ["pass", "fail"]
    assert table["finest"]["result"] == "fail"
    assert table["all_pass"] is False

def test_table_is_flagged_as_unverified(table):
    audit = table.get("integrity_audit")
    assert audit is not None, "the provenance flag was dropped from the table"
    assert audit["grade"] == "WITHHELD"
    assert audit["status"] == "provenance_unverified"


def test_table_abscissa_is_an_exact_arithmetic_sequence(table):
    taus = [p["tau"] for p in table["data"]]
    step = (taus[-1] - taus[0]) / (len(taus) - 1)
    assert max(abs(t - i * step) for i, t in enumerate(taus)) < 1e-12
    assert step == pytest.approx(0.226, abs=5e-4)


def test_table_ordinates_are_quantised(table):
    """Every Z/a is a multiple of 0.05 to round-off: a generated table."""
    za = [p["Z_over_a"] for p in table["data"]]
    assert max(abs(z / 0.05 - round(z / 0.05)) for z in za) < 1e-9


def test_table_has_no_measurement_scatter(table):
    """Second differences stay inside the 0.05 quantum: the residual structure is
    the lattice, not noise. The exact lattice and quantum are the decisive
    evidence; this only shows nothing contradicts them."""
    za = [p["Z_over_a"] for p in table["data"]]
    second = [za[i + 1] - 2 * za[i] + za[i - 1] for i in range(1, len(za) - 1)]
    rms = math.sqrt(sum(d * d for d in second) / len(second))
    assert 0.0 < rms < 0.60 * 0.05
    assert max(abs(d) for d in second) <= 0.05 + 1e-12


def test_audit_agrees_with_the_table_flag(payload, table):
    audit = payload["table_audit"]
    assert audit["usable_as_validation_target"] is False
    assert audit["grade"] == "WITHHELD"
    assert audit["tau_lattice_residual"] < 1e-12
    assert audit["ordinate_quantum_residual"] < 1e-9
    assert audit["grade"] == table["integrity_audit"]["grade"]


# --------------------------------------------------------------------------- #
# 2. the analytic reference
# --------------------------------------------------------------------------- #
def test_ritter_identity_residual_is_zero(payload):
    r = payload["ritter"]
    assert r["H_over_a"] == 2.0
    assert r["max_identity_reldiff"] < 1e-12, "Z/a = 2 tau no longer holds"
    assert r["max_helper_reldiff"] < 1e-12, "shipped helper disagrees with Ritter"


def test_table_slope_is_well_below_ritter(payload):
    c = payload["table_slope"]["table_coefficient_c"]
    assert 0.5 < c < 1.0
    assert c / RITTER_COEFFICIENT < 0.5


# --------------------------------------------------------------------------- #
# 3. the solver's dimensionless front coefficient
# --------------------------------------------------------------------------- #
def test_measured_coefficients_are_pinned(payload):
    measured = {(r["dx"], r["wall_friction"]): r["front_coefficient_c"]
                for r in payload["runs"]}
    assert set(measured) == set(C_PINNED)
    for key, pinned in C_PINNED.items():
        assert measured[key] == pytest.approx(pinned, abs=C_TOLERANCE), (
            f"dx={key[0]}, friction={key[1]}: c moved from {pinned}; if the "
            f"solver changed on purpose, rerun reference_audit.py and update "
            f"C_PINNED and the document")


def test_every_run_is_bounded_and_incompressible(payload):
    for r in payload["runs"]:
        assert r["escaped"] == 0
        assert r["mach_number"] < 0.25
        assert r["max_density_compression"] < 0.05


def test_coefficient_falls_with_floor_friction(payload):
    """No-slip floor must slow the front: the boundary model is doing work."""
    rows = sorted((r for r in payload["runs"] if r["dx"] == 0.04),
                  key=lambda r: r["wall_friction"])
    cs = [r["front_coefficient_c"] for r in rows]
    assert all(b < a for a, b in zip(cs, cs[1:])), cs


def test_friction_alone_cannot_reproduce_the_bundled_table(payload):
    """Extrapolating c(friction) to friction = 1 still overshoots the table.

    The table's slope is ~0.735; the free-slip and no-slip runs bracket it from
    above at both resolutions, so the residual cannot be blamed on the missing
    bed friction alone -- and, since the table is unverified, it must not be
    blamed on the solver either. Both statements are pinned so that neither
    drifts into folklore.
    """
    table_c = payload["table_slope"]["table_coefficient_c"]
    for dx in (0.04, 0.02):
        cs = [r["front_coefficient_c"] for r in payload["runs"]
              if r["dx"] == dx and r["wall_friction"] in (0.0, 1.0)]
        if len(cs) < 2:
            continue
        assert min(cs) > table_c


def test_document_is_generated_from_the_artifact(payload):
    assert DOC.exists(), f"missing {DOC.relative_to(ROOT)}"
    text = DOC.read_text()
    for r in payload["runs"]:
        assert "%.4f" % r["front_coefficient_c"] in text, (
            "the document must quote the artifact numbers verbatim; rerun "
            "make_audit_doc.py")
