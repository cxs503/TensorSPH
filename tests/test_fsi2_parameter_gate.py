"""Gate tests for the FSI2 benchmark target (anchor XVAL-SPH-FSI2-...).

The point of these tests is that the *target side* of every FSI2 ratio in this
repository is a checked object, not a hand-typed dict: they bind the gate to the
survey document and to the archived upstream block on disk, so a later edit of
either cannot silently desynchronise a comparison again.
"""
from __future__ import annotations

import pytest

from tensorsph import fsi2_parameter_gate as gate
from tensorsph.fsi2_similarity import BENCHMARK_FSI2, similarity_report
from tensorsph.fsi2_turek_hron import FSI2Case


def _check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


# --------------------------------------------------------------------------- #
# binding to the declared sources on disk
# --------------------------------------------------------------------------- #
def test_survey_table_is_parsed_from_the_document():
    row = gate.parse_survey_table()
    assert row["rho_f"] == 1000.0
    assert row["mu_f"] == 1.0
    assert row["u_mean"] == 0.2
    assert row["cylinder_d"] == 0.1
    assert row["beam_length"] / row["beam_thickness"] == pytest.approx(17.5)
    assert row["Re"] == 100.0


def test_archived_upstream_block_is_parsed_verbatim():
    oss = gate.parse_oss_block()
    assert oss["rho0_s"] == 10.0          # fsi2.h:36
    assert oss["Ae"] == 1.4e3             # fsi2.h:38
    assert oss["poisson_s"] == 0.4        # fsi2.h:37
    assert oss["Re"] == 100.0             # fsi2.h:31
    assert oss["beam_length"] == pytest.approx(3.5)
    assert oss["beam_thickness"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# the defect: the declared target is not reproducible from the declared table
# --------------------------------------------------------------------------- #
def test_survey_table_breaks_its_own_reynolds_identity():
    report = gate.check_row(gate.parse_survey_table())
    re_check = _check(report, "reynolds")
    assert re_check["status"] == "fail"
    assert re_check["computed"] == pytest.approx(20.0)
    assert re_check["declared"] == pytest.approx(100.0)
    assert re_check["rel_error"] == pytest.approx(-0.8)
    assert "reynolds" in report["failed"]


def test_case_defaults_break_the_declared_stiffness_and_poisson():
    report = gate.check_row(gate.case_row(FSI2Case()))
    ae = _check(report, "normalised_stiffness")
    assert ae["status"] == "fail"
    assert ae["rel_error"] == pytest.approx(16.857, rel=1e-3)
    assert _check(report, "reynolds")["status"] == "pass"
    assert _check(report, "poisson_s")["status"] == "fail"
    assert set(report["failed"]) == {"normalised_stiffness", "poisson_s"}


def test_archived_upstream_differs_from_target_on_exactly_two_groups():
    report = gate.check_row(gate.oss_row())
    assert set(report["failed"]) == {"normalised_stiffness", "density_ratio"}
    assert _check(report, "normalised_stiffness")["rel_error"] == pytest.approx(-0.75)
    assert _check(report, "density_ratio")["computed"] == pytest.approx(10.0)


def test_absent_inputs_are_not_run_and_never_pass():
    report = gate.check_row({"Re": 100.0})
    assert report["passed"] is False
    assert "reynolds" in report["not_run"]


# --------------------------------------------------------------------------- #
# the two coherent candidates and the verdict flip
# --------------------------------------------------------------------------- #
def test_candidates_are_each_self_consistent():
    cands = gate.candidate_points()
    assert set(cands) == {"paper_target", "upstream_consistent"}
    for row in cands.values():
        assert row["Re"] == pytest.approx(100.0, rel=1e-12)
        assert row["u_mean"] == pytest.approx(1.0)          # from the Re identity
        assert row["u_max"] == pytest.approx(1.5)
        assert row["slenderness"] == pytest.approx(17.5)
        assert row["Ae"] == pytest.approx(row["beam_E"] / (row["rho_f"] * row["u_mean"] ** 2))
    # the two branches differ only on the (E, rho_s/rho_f) pair
    a, b = cands["paper_target"], cands["upstream_consistent"]
    assert a["mu_f"] == b["mu_f"] and a["poisson_s"] == b["poisson_s"]
    assert a["rho_s_over_rho_f"] == pytest.approx(1.0)
    assert b["rho_s_over_rho_f"] == pytest.approx(10.0)
    assert a["beam_E"] / b["beam_E"] == pytest.approx(4.0)


def _case_from(row):
    return FSI2Case(rho0=row["rho_f"], viscosity=row["mu_f"],
                    inlet_u_max=row["u_max"], beam_E=row["beam_E"],
                    beam_rho=row["rho_s"], beam_nu=row["poisson_s"])


def test_similarity_verdict_is_target_dependent():
    """The 'different physics problem' verdict is a property of the target."""
    cands = gate.candidate_points()
    upstream = {k: v for k, v in gate.oss_row().items() if k != "U_f_scaled"}

    same = similarity_report(_case_from(cands["upstream_consistent"]), target=upstream)
    assert same["passed"] is True and same["failed"] == []

    against_declared = similarity_report(_case_from(cands["upstream_consistent"]),
                                        target=dict(BENCHMARK_FSI2))
    assert set(against_declared["failed"]) == {"Ae", "rho_s_over_rho_f"}

    paper = similarity_report(_case_from(cands["paper_target"]),
                             target=dict(BENCHMARK_FSI2))
    assert paper["passed"] is True


def test_provenance_matrix_covers_every_source_and_labels_it():
    matrix = gate.provenance_matrix()
    assert set(matrix) == {"survey", "case_defaults", "declared_target",
                           "archived_upstream"}
    for key, block in matrix.items():
        assert block["class"] in {"VERBATIM", "DERIVED", "SECONDARY", "DEFAULT"}
        assert block["source"]
        assert block["values"]
    assert matrix["archived_upstream"]["class"] == "VERBATIM"
    assert matrix["survey"]["gate"]["failed"]


# --------------------------------------------------------------------------- #
# time-coverage gate
# --------------------------------------------------------------------------- #
def test_reference_period_is_measured_from_the_archived_record():
    per = gate.reference_period("0")
    assert per["n_samples"] == 107
    assert per["duration"] == pytest.approx(10.7)
    assert 0.2 < per["period_fft"] < 0.4      # ~3.3 Hz dominant peak
    assert per["period_used"] == pytest.approx(
        max(per["period_fft"], per["period_zero_crossing"]))
    assert per["y_amplitude"] == pytest.approx(0.0796685, rel=1e-6)


def test_short_probe_is_void_by_the_coverage_gate():
    per = gate.reference_period("0")
    dt = FSI2Case(dx=0.02).dt()
    short = gate.coverage(dt, 40, per["period_used"])
    assert short["status"] == "fail"
    assert short["cycles"] < 0.05
    # the number of steps that a 3-cycle amplitude comparison needs
    n = int(-(-gate.MIN_CYCLES_AMPLITUDE * per["period_used"] // dt))
    assert gate.coverage(dt, n, per["period_used"])["status"] == "pass"


def test_coverage_verdict_does_not_depend_on_the_period_estimate():
    per = gate.reference_period("0")
    dt = FSI2Case(dx=0.02).dt()
    for key in ("period_fft", "period_zero_crossing", "period_strouhal_expectation"):
        assert gate.coverage(dt, 40, per[key])["status"] == "fail"