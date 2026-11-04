"""Architecture-independent pins for the mode-2 (oscillating drop) evidence.

Why these exist
---------------
The two physics-layer pins in ``tests/test_oscillating_droplet.py`` compare a
*re-simulated* frequency against a single recorded number, and that number is
architecture-dependent (+8.6 % on x86_64 CI, -31.8 % on loong64) because the
release case diverges before the fit window ends -- so one of the two
architectures is always red.  The facts that are *not* architecture-dependent
are recorded in ``benchmarks/oscillating_droplet_stability/report.json``:

* the published release run is numerically inadmissible (its own recorder
  records Mach ~9e3 and an area-equivalent radius ~5e3 R),
* the published -43.5 % is the lower edge of the estimator's search box after
  its three +-2 % refinements, i.e. boundary-pinned, not a physical frequency,
* re-fitting the *stored* trace reproduces the published number exactly, and
  the same estimator on the admissible window is a different (and well-fitted)
  answer.

These tests read only that JSON: no simulation, no arch assumption, so they run
identically on every CI machine.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPORT = (Path(__file__).resolve().parents[1]
          / "benchmarks" / "oscillating_droplet_stability" / "report.json")
DOC = Path(__file__).resolve().parents[1] / "docs" / "verification_mode2_admissibility.md"
MAIN = "release_R20mm_eps0.10"
SECOND = "release_R20mm_eps0.02"


@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip(f"artefact {REPORT} missing; run "
                    "benchmarks/oscillating_droplet_stability/run.py --stage all")
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def published(report):
    return report["published_run"]


@pytest.fixture(scope="module")
def main_case(report):
    return report["cases"][MAIN]


# --------------------------------------------------------------------------- #
# 1. the published run is not an admissible measurement
# --------------------------------------------------------------------------- #
def test_published_release_run_is_numerically_inadmissible(published):
    """Mach ~1e3 and r_eff ~1e3 R: the published verdict came from debris."""
    assert published["admissible"] is False
    assert published["max_mach"] > 100.0, published["max_mach"]
    assert published["final_reff_over_R"] > 100.0, published["final_reff_over_R"]


def test_published_fit_explains_almost_nothing(published):
    """A damped cosine on a diverging trace must not have a healthy R^2."""
    assert published["refit_from_stored_trace"]["r2"] < 0.5


def test_published_verdict_string_is_recorded_as_fail(published):
    """The historical verdict is kept verbatim: this is a correction, not a
    quiet rewrite of the ledger."""
    assert published["recorded_verdict"]["worst_within_15pct"] is False
    assert published["recorded_rel_error_pct"] < -40.0


# --------------------------------------------------------------------------- #
# 2. the reported number is the estimator's search-box boundary
# --------------------------------------------------------------------------- #
def test_reported_frequency_is_the_lower_search_box_edge(published):
    """``omega = 0.6 * omega_ref * 0.98**3`` to <1e-6 relative.

    ``damped_cosine_fit`` searches ``[0.6, 1.6] * omega_guess`` and refines
    three times by +-2 %.  Landing exactly on that ladder's lowest reachable
    value means the unconstrained optimum is outside the box, so the printed
    frequency is a property of the estimator, not of the flow.
    """
    edge = published["boundary_pin"]["edge_after_three_refinements_rad_per_s"]
    got = published["refit_from_stored_trace"]["omega_rad_per_s"]
    assert got == pytest.approx(edge, rel=1e-6), (got, edge)


def test_boundary_pin_matches_the_recorded_ledger_number(published):
    """The boundary value and the ledger number are the same quantity."""
    assert published["refit_from_stored_trace"]["reproduces_recorded"] is True
    assert abs(published["refit_from_stored_trace"]["rel_error_pct"]
               - published["recorded_rel_error_pct"]) < 1e-6


# --------------------------------------------------------------------------- #
# 3. admissible window vs full trace
# --------------------------------------------------------------------------- #
def test_gate_trips_before_one_period_in_every_case(report):
    """No case survives a full analytic period: the deficiency is numerical."""
    for key, case in report["cases"].items():
        ad = case["admissibility"]
        assert ad["gate_tripped"] is True, key
        assert ad["admissible_fraction_of_periods"] < 1.0, (key, ad)
        assert ad["t_gate_s"] == pytest.approx(
            min(ad["t_mach_gate_s"], ad["t_radius_gate_s"]))


def test_admissible_window_fit_is_well_posed(main_case):
    """Inside the window the trace *is* a damped cosine (R^2 > 0.99)."""
    v = main_case["fit_valid_window"]
    assert v["r2"] > 0.99, v["r2"]
    assert v["window_periods"] < 1.0          # honest: sub-period window
    assert abs(v["rel_error_pct"]) < 15.0, v["rel_error_pct"]


def test_quality_degrades_monotonically_once_the_window_eats_debris(main_case):
    """R^2 falls monotonically and the answer collapses to -43.5 %."""
    ladder = main_case["fit_window_ladder"]
    r2 = [row["r2"] for row in ladder]
    assert all(b <= a + 1e-9 for a, b in zip(r2, r2[1:])), r2
    assert abs(ladder[-1]["rel_error_pct"] - main_case["fit_full_trace"]["rel_error_pct"]) < 1e-6
    assert ladder[-1]["rel_error_pct"] == pytest.approx(-43.5, abs=0.1)
    # the trend is monotone in |error| too, once the gate has been passed
    err = [abs(row["rel_error_pct"]) for row in ladder]
    assert err[-1] > 4.0 * max(err[0], 1e-9)


def test_full_trace_spectrum_is_multi_modal(main_case):
    """Several comparable peaks: the basin the fit lands in is arbitrary."""
    peaks = main_case["fft_full_trace"]
    assert len(peaks) >= 4
    top = peaks[0]["amplitude"]
    comparable = [p for p in peaks if p["amplitude"] > 0.5 * top]
    assert len(comparable) >= 4, [round(p["amplitude"], 2) for p in peaks]
    # the loong64 and x86_64 readings both sit on debris peaks, not on omega_2
    assert not any(abs(p["rel_error_pct"]) < 5.0 for p in peaks)


def test_smaller_epsilon_does_not_extend_the_admissible_window(report):
    """An initial-condition study cannot rescue the case: not a linearisation
    artefact."""
    if SECOND not in report["cases"]:
        pytest.skip("epsilon control case not recorded")
    a = report["cases"][MAIN]["admissibility"]["admissible_fraction_of_periods"]
    b = report["cases"][SECOND]["admissibility"]["admissible_fraction_of_periods"]
    assert b < a, (a, b)
    # its own admissible-window fit is also unusable -> window length dominates
    v = report["cases"][SECOND]["fit_valid_window"]
    assert not (abs(v["rel_error_pct"]) < 15.0 and v["r2"] > 0.99), v


# --------------------------------------------------------------------------- #
# 4. the doc cannot drift from the artefacts
# --------------------------------------------------------------------------- #
def test_doc_quotes_the_artefact_numbers(main_case, published):
    if not DOC.exists():
        pytest.skip("doc not rendered")
    text = DOC.read_text()
    v = main_case["fit_valid_window"]
    assert f"{v['rel_error_pct']:.1f}" in text
    assert f"{published['max_mach']:.0f}" in text
    assert "WITHHELD" in text
    # the invalid number must never be presented as the model's accuracy
    assert "已发布的 −43.5% 必须停止被引用" in text


def test_no_platform_gate_is_silently_relaxed(report):
    gates = report["platform_gates"]
    assert gates["mach"] == pytest.approx(0.10)
    assert gates["radius_drift_pct"] == pytest.approx(5.0)
    assert math.isfinite(gates["mach"])