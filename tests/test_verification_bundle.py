"""Tests for the artefact bundle and the canonical entry point (P482-N04).

Two things are pinned here, and neither of them is a physical number:

``the bundle cannot ship a hand-typed comparison``
    A benchmark row must resolve to a metric or to a series element that the run
    recorded; a row that claims a number the run did not produce is rejected.
``the entry point really produces the artefacts the maturity review asks for``
    ``impl_files`` / ``entry`` / ``run_artifacts`` / ``integration_callers`` /
    ``benchmark`` / ``repro_cmd`` are *derived* from the run directory by
    :func:`tensorsph.verification.bundle.assess_level`, and the end-to-end test
    asserts each of them is non-empty and internally consistent (hashes match,
    every row resolves, the recorded command exists).

The physical verdicts are deliberately not asserted: they belong to the run
artefacts under ``runs/sph_free_surface_verification/`` and change when the
solver changes.  A gate that cannot fail would not be a gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tensorsph.verification import cli
from tensorsph.verification.bundle import (BenchmarkRow, Bundle, BundleError,
                                           Case, Metric, assess_level)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# the maturity ladder
# --------------------------------------------------------------------------- #
def test_an_empty_manifest_is_level_zero():
    level = assess_level({})
    assert level["level"] == "L0"
    assert "impl_files" in level["missing"]["L1"]


def test_each_rung_needs_all_of_its_keys():
    impl = {"impl_files": ["a.py"]}
    assert assess_level(impl)["level"] == "L1"
    entry_and_runs = dict(impl, entry={"path": "e.py"},
                          run_artifacts=[{"path": "results.json"}])
    assert assess_level(entry_and_runs)["level"] == "L2"
    callers = dict(entry_and_runs, integration_callers=[{"path": "t.py"}])
    assert assess_level(callers)["level"] == "L3"
    full = dict(callers, benchmark={"rows": 3}, repro_cmd="python -m x")
    assert assess_level(full)["level"] == "L4"
    assert all(not v for v in assess_level(full)["missing"].values())


def test_a_missing_benchmark_keeps_the_level_at_three():
    full = {"impl_files": ["a"], "entry": {"p": 1}, "run_artifacts": [{"p": 1}],
            "integration_callers": [{"p": 1}], "repro_cmd": "cmd"}
    level = assess_level(full)
    assert level["level"] == "L3"
    assert level["missing"]["L4"] == ["benchmark"]


# --------------------------------------------------------------------------- #
# the anti-hand-typing rule
# --------------------------------------------------------------------------- #
def _bundle(tmp_path) -> Bundle:
    b = Bundle(tmp_path)
    b.add_case(Case(name="case", phase="M1",
                    metrics={"m": Metric(value=1.0, source="f()")},
                    series={"points": [{"got": 1.0}, {"got": 2.0}]}))
    return b


def test_a_row_backed_by_a_metric_passes_the_cross_check(tmp_path):
    b = _bundle(tmp_path)
    b.add_benchmark(quantity="q", case="case", reference="r", reference_value=1.0,
                    reference_grade="ANALYTIC", computed_value=1.0, tolerance=0.1,
                    metric="m")
    b.check_benchmark_consistency()


def test_a_row_backed_by_a_series_element_passes_the_cross_check(tmp_path):
    b = _bundle(tmp_path)
    b.add_benchmark(quantity="q", case="case", reference="r", reference_value=2.0,
                    reference_grade="ANALYTIC", computed_value=2.0, tolerance=0.1,
                    series_ref={"series": "points", "index": 1, "field": "got"})
    b.check_benchmark_consistency()


def test_a_hand_typed_number_is_rejected(tmp_path):
    b = _bundle(tmp_path)
    b.add_benchmark(quantity="q", case="case", reference="r", reference_value=1.0,
                    reference_grade="ANALYTIC", computed_value=1.5, tolerance=0.1,
                    metric="m")
    with pytest.raises(BundleError):
        b.check_benchmark_consistency()


def test_a_row_without_a_recorded_source_is_rejected(tmp_path):
    b = _bundle(tmp_path)
    b.add_benchmark(quantity="q", case="case", reference="r", reference_value=1.0,
                    reference_grade="ANALYTIC", computed_value=1.0, tolerance=0.1)
    with pytest.raises(BundleError):
        b.check_benchmark_consistency()


def test_a_withheld_reference_never_reports_pass_or_fail():
    row = BenchmarkRow(quantity="q", case="c", reference="r", reference_value=1.3,
                       reference_grade="WITHHELD", computed_value=1.0, tolerance=0.1)
    assert row.rel_error == pytest.approx(0.23076923076923078)
    assert row.verdict == "WITHHELD"


def test_a_one_sided_row_uses_the_inequality_not_the_ratio():
    row = BenchmarkRow(quantity="order", case="c", reference="at least first order",
                       reference_value=1.0, reference_grade="ANALYTIC",
                       computed_value=1.9, direction="lower")
    assert row.verdict == "PASS"
    assert row.rel_error == pytest.approx(0.9)

    failing = BenchmarkRow(quantity="order", case="c", reference="at least first order",
                           reference_value=1.0, reference_grade="ANALYTIC",
                           computed_value=0.9, direction="lower")
    assert failing.verdict == "FAIL"


def test_an_unknown_verdict_is_a_bug_not_a_default():
    with pytest.raises(BundleError):
        Metric(value=1.0, verdict="OK")


# --------------------------------------------------------------------------- #
# integration points
# --------------------------------------------------------------------------- #
def test_the_solver_entry_points_are_reachable_and_located():
    callers = cli.solver_callers()
    assert callers, "the face must drive the production solver"
    for caller in callers:
        assert caller["resolved"], caller
        assert caller["defined_in"] and (REPO_ROOT / caller["defined_in"]).exists()
        assert caller["line"] and caller["line"] > 0


def test_the_face_has_callers_outside_its_own_package():
    hits = cli.integration_callers()
    paths = {h["path"] for h in hits}
    assert len(paths) >= 3, paths
    assert any(p.startswith("tests/") for p in paths), paths
    assert any(p.startswith("benchmarks/") for p in paths), paths
    assert not any(p.startswith("src/tensorsph/verification/") for p in paths)


def test_the_repro_command_names_a_real_entry_point():
    command = cli.repro_command("smoke", REPO_ROOT / "runs" / "x")
    assert command.startswith("python -m tensorsph.verification")
    assert "--tier smoke" in command
    assert (REPO_ROOT / "scripts" / "run_sph_free_surface_verification.py").exists()
    assert (Path(cli.__file__).parent / "__main__.py").exists()


# --------------------------------------------------------------------------- #
# end to end: the smoke tier produces a consistent bundle
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def smoke_bundle(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("sph_smoke")
    manifest = cli.run("smoke", out, threads=2, figures=True, quiet=True)
    return {"dir": out, "manifest": manifest}


@pytest.mark.slow
def test_smoke_bundle_reaches_the_top_of_the_ladder(smoke_bundle):
    manifest = smoke_bundle["manifest"]
    maturity = manifest["maturity"]
    assert manifest["maturity_level"]["level"] == "L4", manifest["maturity_level"]
    assert maturity["impl_files"]
    assert maturity["entry"]
    assert maturity["run_artifacts"]
    assert maturity["integration_callers"]
    assert maturity["benchmark"]["rows"] > 0
    assert maturity["repro_cmd"]


@pytest.mark.slow
def test_smoke_bundle_artefact_hashes_match_the_files(smoke_bundle):
    out = smoke_bundle["dir"]
    manifest = smoke_bundle["manifest"]
    for artifact in manifest["maturity"]["run_artifacts"]:
        path = out / artifact["path"]
        assert path.exists(), artifact["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"], artifact["path"]
        assert artifact["bytes"] == path.stat().st_size


@pytest.mark.slow
def test_smoke_bundle_has_the_four_faces_and_a_comparison_table(smoke_bundle):
    out = smoke_bundle["dir"]
    results = json.loads((out / "results.json").read_text())
    phases = {case["phase"] for case in results["cases"].values()}
    assert phases == {"M1", "M2", "M3", "M4"}

    benchmark = json.loads((out / "benchmark.json").read_text())
    assert benchmark["summary"]["n_rows"] == len(benchmark["rows"]) > 0
    for row in benchmark["rows"]:
        assert row["reference"] and row["reference_grade"]
        assert row["computed_value"] is not None
        if row["reference_grade"] == "WITHHELD":
            assert row["verdict"] == "WITHHELD"
    grades = {row["reference_grade"] for row in benchmark["rows"]}
    assert "ANALYTIC" in grades

    figures = list((out / "figures").glob("*.png"))
    assert figures, "the report needs figures drawn from the recorded series"


@pytest.mark.slow
def test_smoke_bundle_records_every_gate_as_withheld_or_decisive(smoke_bundle):
    """Outside the campaign tier a gate is WITHHELD, never silently promoted."""
    gates = smoke_bundle["manifest"]["gates"]
    assert set(gates) == {"J1_front", "J1_height", "J1_control_gravity_off",
                          "J2_pressure", "J2_control_zero_density",
                          "J3_mass_closed", "J3_mass_open_and_overflow_control",
                          "J4_convergence", "J4_control_truncated_support"}
    for name, gate in gates.items():
        assert gate["verdict"] == "WITHHELD", (name, gate)


@pytest.mark.slow
def test_smoke_metrics_csv_covers_every_recorded_metric(smoke_bundle):
    out = smoke_bundle["dir"]
    results = json.loads((out / "results.json").read_text())
    csv_rows = (out / "metrics.csv").read_text().strip().splitlines()[1:]
    n_metrics = sum(len(case["metrics"]) for case in results["cases"].values())
    assert len(csv_rows) == n_metrics
    assert all(row.count(",") >= 6 for row in csv_rows)


@pytest.mark.slow
def test_the_dam_break_case_keeps_the_field_snapshot(smoke_bundle):
    results = json.loads((smoke_bundle["dir"] / "results.json").read_text())
    case = next(c for n, c in results["cases"].items()
                if n.startswith("dam_break_2d_dx"))
    snapshot = case["series"]["final_snapshot"]
    assert len(snapshot["x"]) == len(snapshot["y"]) == len(snapshot["speed"]) > 0
    assert snapshot["tau"] > 0
