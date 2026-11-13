"""Tests for the verification claim ledger (``benchmarks/verification_ledger``).

The ledger is a *reconciliation* artifact: it reads every machine-readable
verification artifact in ``benchmarks/`` and states, in one place, what was
measured, against which reference, under which definition, and what grade the
platform tolerance ladder assigns.  These tests are what make it trustworthy:

* anti-drift  — the committed JSON/Markdown must equal a fresh regeneration
                (so no number in ``docs/verification_ledger.md`` is hand typed),
* ladder      — grades may not contradict the 3 % / 15 % criteria,
* gauge       — a claim measured with a degenerate (vacuum) ambient pressure may
                not be credited as a pass, even if its artifact says so,
* traceability— every claim points at a real artifact + resolvable JSON path +
                an existing guarding test,
* coverage    — no evidence file in ``benchmarks/`` is silently unregistered,
* prose       — stale numbers quoted in ``docs/`` must carry a historical marker.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "benchmarks" / "verification_ledger" / "build_ledger.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("verification_ledger_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_builder()


@pytest.fixture(scope="module")
def ledger():
    return B.build()


@pytest.fixture(scope="module")
def by_id(ledger):
    return {e["claim_id"]: e for e in ledger["entries"]}


# ---------------------------------------------------------------------------
# anti-drift
# ---------------------------------------------------------------------------
def test_committed_ledger_json_is_up_to_date(ledger):
    path = REPO / "benchmarks" / "verification_ledger" / "ledger.json"
    committed = json.loads(path.read_text())
    assert committed == json.loads(B._dump(ledger)), (
        "benchmarks/verification_ledger/ledger.json is stale — run "
        "`python benchmarks/verification_ledger/build_ledger.py`"
    )


def test_committed_markdown_is_up_to_date(ledger):
    committed = (REPO / "docs" / "verification_ledger.md").read_text()
    assert committed == B.render_markdown(ledger), (
        "docs/verification_ledger.md is stale — regenerate it from the builder"
    )


def test_check_mode_reports_no_drift():
    assert B.check() == 0


# ---------------------------------------------------------------------------
# grading ladder
# ---------------------------------------------------------------------------
def test_grades_never_contradict_the_tolerance_ladder(ledger):
    for e in ledger["entries"]:
        err = abs(e["rel_error_pct"]) if e["rel_error_pct"] is not None else None
        if e["grade"] == "PASS":
            assert err is not None and err <= e["paper_criterion_pct"], e["claim_id"]
        elif e["grade"] == "MARGINAL":
            assert err is not None and err <= e["criterion_pct"], e["claim_id"]
        elif e["grade"] in {"WITHHELD", "FAIL"}:
            # A degenerate gauge (self-referential reference) is withheld whatever
            # its numerical value: the number cannot be read as an error at all.
            assert e["gauge_degenerate"] or (
                err is not None and err > e["paper_criterion_pct"]
            ), e["claim_id"]
        else:
            assert e["grade"] == "WITHDRAWN" and e["superseded_by"], e["claim_id"]


def test_no_artifact_self_declared_pass_is_left_uncorrected(ledger):
    """Every artifact-declared PASS must either be credited or listed as a finding."""
    finded = {(f["rule"], f["claim_id"]) for f in ledger["findings"]}
    for e in ledger["entries"]:
        if (e["artifact_status"] or "").upper() == "PASS" and e["grade"] in {"WITHHELD", "FAIL"}:
            assert ("declared_vs_derived", e["claim_id"]) in finded, (
                f"{e['claim_id']} is declared PASS but graded {e['grade']} without a finding"
            )


def test_summary_matches_entries(ledger):
    s = ledger["summary"]
    assert s["n_claims"] == len(ledger["entries"])
    assert sum(s["by_grade"].values()) == s["n_claims"]
    for grade, count in s["by_grade"].items():
        assert count == sum(1 for e in ledger["entries"] if e["grade"] == grade), grade
    assert s["n_findings"] == len(ledger["findings"])


# ---------------------------------------------------------------------------
# the gauge defect that the ledger exists to catch
# ---------------------------------------------------------------------------
def test_vacuum_gauge_claim_is_withheld(by_id, ledger):
    e = by_id["young_laplace.laplace_law.vacuum_gauge"]
    assert e["gauge_degenerate"] is True
    assert e["artifact_status"].upper() == "PASS"      # the artifact claims a pass
    assert e["grade"] == "WITHHELD"                    # the ledger refuses it
    assert any(
        f["rule"] == "declared_vs_derived" and f["claim_id"] == e["claim_id"]
        for f in ledger["findings"]
    )


def test_declared_vs_derived_offenders_are_withheld_for_a_stated_reason(ledger):
    """An artifact may not self-declare PASS while the ledger refuses the reading.

    The ledger refuses a reading through exactly three channels, and every
    ``declared_vs_derived`` offender has to name one of them (m17086 widened this
    from "must be a degenerate gauge": the rebuild surfaced
    ``threed_hydrostatic.bottom_pressure.adami_worst``, where the artifact's
    report-level ``verdict = pass`` covers a run 95 % off the collapse value --
    the refusal is about the reading's selection, not about the gauge):

    1. ``gauge_degenerate`` -- the number cannot carry the physics;
    2. ``run_validity == INVALID`` -- the trajectory was no longer a simulation;
    3. the reading is one of several spreads on the same quantity
       (``quantity_spread``), i.e. it is a selection, not a property.
    """
    by_id = {e["claim_id"]: e for e in ledger["entries"]}
    spread_q = {
        f["claim_id"]: f["detail"] for f in ledger["findings"] if f["rule"] == "quantity_spread"
    }
    offenders = [f["claim_id"] for f in ledger["findings"] if f["rule"] == "declared_vs_derived"]
    assert "young_laplace.laplace_law.vacuum_gauge" in offenders
    for cid in offenders:
        e = by_id[cid]
        assert e["artifact_status"].upper() == "PASS", cid
        assert e["grade"] == "WITHHELD", f"{cid}: a PASS-declaring artifact may only be refused, not failed"
        explained = (
            e["gauge_degenerate"]
            or e.get("run_validity") == "INVALID"
            or e["quantity_id"] in spread_q
        )
        assert explained, f"{cid}: refused without a stated reason"


# ---------------------------------------------------------------------------
# one physical quantity, several readings -> must stay visible
# ---------------------------------------------------------------------------
def test_young_laplace_jump_readings_stay_reconciled(ledger):
    qid = "young_laplace_jump_sigma7.2_R0.02"
    group = [e for e in ledger["entries"] if e["quantity_id"] == qid]
    assert len(group) >= 5, "the sigma/R = 360 Pa quantity must keep all its readings"
    errs = sorted(e["rel_error_pct"] for e in group)
    assert errs[0] < -80.0 and errs[-1] > 40.0, "the raw spread must not be averaged away"
    spread_findings = [
        f for f in ledger["findings"]
        if f["rule"] == "quantity_spread" and f["claim_id"] == qid
    ]
    assert spread_findings, "different readings of one quantity must raise a finding"


def test_pressure_floor_best_reading_matches_the_backlog_figure(by_id):
    """backlog_update_m2971.md quotes -34.0 %; the ledger must reproduce it."""
    e = by_id["young_laplace.absolute_tail.best_of_6"]
    assert e["rel_error_pct"] == pytest.approx(-34.0, abs=0.2)
    assert e["grade"] == "WITHHELD"
    assert e["backlog_ref"].startswith("backlog_update_m2971.md")


def test_two_phase_control_separates_operator_from_free_surface(by_id):
    """The closed-interface control must be an order of magnitude better."""
    control = by_id["two_phase_laplace.smoothed_colour.dx2.5mm"]
    one_sided = by_id["young_laplace.absolute_tail.best_of_6"]
    assert abs(control["rel_error_pct"]) < 5.0
    assert abs(one_sided["rel_error_pct"]) > 5.0 * abs(control["rel_error_pct"])
    assert control["grade"] in {"PASS", "MARGINAL"}


def test_colour_surface_deficit_is_recorded_as_root_cause(by_id):
    e = by_id["colour_surface.delta_s_integral"]
    assert 0.70 < e["measured"]["value"] < 0.80
    assert -30.0 < e["rel_error_pct"] < -20.0
    # the force path deficit and the delta normalisation must be the same story
    force = by_id["csf_force.sigma_eff_over_sigma.dx5mm"]
    assert abs(abs(force["rel_error_pct"]) - abs(e["rel_error_pct"])) < 10.0


def test_withdrawn_evidence_stays_withdrawn(by_id):
    e = by_id["static_droplet.force_direction.WITHDRAWN"]
    assert e["grade"] == "WITHDRAWN"
    assert e["superseded_by"] == "benchmarks/laplace_pressure/validation_report.json"


# ---------------------------------------------------------------------------
# traceability + coverage
# ---------------------------------------------------------------------------
def test_every_claim_is_traceable(ledger):
    for e in ledger["entries"]:
        assert (REPO / e["artifact"]).is_file(), e["claim_id"]
        assert B.resolve(B._load(e["artifact"]), e["json_path"]) is not None, e["claim_id"]


def test_every_claim_names_an_existing_guarding_test(ledger):
    for e in ledger["entries"]:
        target = e["guarding_test"].split("::")[0]
        path = REPO / target
        assert path.is_file(), f"{e['claim_id']}: guarding test {target} does not exist"
        if "::" in e["guarding_test"]:
            name = e["guarding_test"].split("::")[1]
            assert f"def {name}" in path.read_text(), (
                f"{e['claim_id']}: {name} is not defined in {target}"
            )


def test_no_evidence_artifact_is_unregistered(ledger):
    unregistered = [f for f in ledger["findings"] if f["rule"] == "unregistered_artifact"]
    assert not unregistered, [f["claim_id"] for f in unregistered]


def test_every_waiver_has_a_reason():
    waivers = json.loads((REPO / "benchmarks" / "verification_ledger" / "waivers.json").read_text())
    for rel in waivers["unregistered_ok"]:
        assert waivers["reasons"].get(rel, "").strip(), f"waiver {rel} has no reason"


def test_stale_numbers_in_prose_are_marked(ledger):
    stale = [f for f in ledger["findings"] if f["rule"] == "unmarked_superseded_number"]
    assert not stale, [f"{f['claim_id']}: {f['detail']}" for f in stale]


# ---------------------------------------------------------------------------
# the document has to stay usable
# ---------------------------------------------------------------------------
def test_markdown_contains_every_claim_and_finding(ledger):
    md = (REPO / "docs" / "verification_ledger.md").read_text()
    for e in ledger["entries"]:
        assert f"`{e['claim_id']}`" in md
        assert e["grade"] in md
    for f in ledger["findings"]:
        assert f["rule"] in md


def test_markdown_states_the_criteria(ledger):
    md = (REPO / "docs" / "verification_ledger.md").read_text()
    assert "15%" in md and "3%" in md
    assert "gauge_degenerate" in md


# ---------------------------------------------------------------------------
# coverage: a benchmark may not sit outside the ledger unnoticed (m6453)
# ---------------------------------------------------------------------------
def test_no_benchmark_harness_is_uncovered(ledger):
    """Every ``benchmarks/<dir>/run*.py`` is either claimed or waived, by name."""
    uncovered = [f["claim_id"] for f in ledger["findings"] if f["rule"] == "uncovered_benchmark"]
    assert not uncovered, uncovered


def test_coverage_rule_is_directory_based(tmp_path):
    """The rule must catch a harness that commits nothing and an oddly named file.

    Before m6453 the check matched artifact *names*
    (``report|results|ladder|audit.json``) and skipped ``out/`` entirely, so
    ``benchmarks/density_diffusion_decay/decay_rate.json`` and the two
    ``out/``-only benchmarks sat outside the ledger without a finding.
    """
    (tmp_path / "benchmarks").mkdir()
    # alpha: harness + odd-named primary artifact, neither claimed nor waived
    (tmp_path / "benchmarks" / "alpha").mkdir()
    (tmp_path / "benchmarks" / "alpha" / "run.py").write_text("")
    (tmp_path / "benchmarks" / "alpha" / "decay_rate.json").write_text("{}")
    # beta: harness that commits nothing at all
    (tmp_path / "benchmarks" / "beta").mkdir()
    (tmp_path / "benchmarks" / "beta" / "run.py").write_text("")
    # gamma: covered by a claim under it (ladder fragments are not primary evidence)
    (tmp_path / "benchmarks" / "gamma").mkdir()
    (tmp_path / "benchmarks" / "gamma" / "run.py").write_text("")
    (tmp_path / "benchmarks" / "gamma" / "cases").mkdir()
    (tmp_path / "benchmarks" / "gamma" / "cases" / "dx1.json").write_text("{}")
    (tmp_path / "benchmarks" / "gamma" / "report.json").write_text("{}")
    # delta: waived by directory, out/ fragment
    (tmp_path / "benchmarks" / "delta").mkdir()
    (tmp_path / "benchmarks" / "delta" / "run.py").write_text("")
    (tmp_path / "benchmarks" / "delta" / "out").mkdir()
    (tmp_path / "benchmarks" / "delta" / "out" / "case01.json").write_text("{}")

    registered = {"benchmarks/gamma/report.json"}
    waived = {"benchmarks/delta/"}
    got = B.coverage_findings(registered, waived, root=tmp_path)
    rules = sorted((f["rule"], f["claim_id"]) for f in got)
    assert rules == [
        ("uncovered_benchmark", "benchmarks/alpha"),
        ("uncovered_benchmark", "benchmarks/beta"),
        ("unregistered_artifact", "benchmarks/alpha/decay_rate.json"),
    ], rules


def test_orphan_evidence_is_now_graded(by_id):
    """Three committed analytic comparisons that the old rule could not see."""
    decay = by_id["density_diffusion.decay_rate.worst_of_ladder"]
    assert decay["grade"] == "PASS"
    assert decay["measured"]["value"] < decay["gate_limit"]
    assert by_id["density_diffusion.finite_time.amplitude"]["grade"] == "PASS"
    assert by_id["poiseuille.centre_velocity.finest"]["grade"] == "PASS"
    # the wall-band shear readout contradicts the velocity fit it came with
    assert by_id["poiseuille.wall_shear_ratio.finest"]["grade"] == "WITHHELD"
    # independent replication of the ~30 % CSF force deficit at 2x the radius
    eff = by_id["effective_sigma.sigma_eff_over_sigma.dx5mm"]
    assert eff["grade"] == "FAIL"
    sibling = by_id["csf_force.sigma_eff_over_sigma.dx5mm"]
    assert abs(eff["rel_error_pct"] - sibling["rel_error_pct"]) < 5.0


def test_contact_angle_harness_runs_and_its_instrument_is_repaired():
    """``benchmarks/contact_angle/run.py`` used to execute but measure nothing.

    m6453 replaced two removed torch APIs so the harness ran, and pinned the
    honest state of the day: all ten analytic rows came back NaN.  m6503
    repaired the instrument (exact line/circular interfaces, wall-normal
    envelope, circle fit with a line fallback) and separated three layers --
    analytic (device), resolution (device on a particle lattice) and physics
    (the run).  This pins the repaired state; the physics layer is still
    ungradable and remains debt E3, see docs/verification_contact_angle_m6503.md.
    """
    import importlib.util

    path = REPO / "benchmarks" / "contact_angle" / "run.py"
    spec = importlib.util.spec_from_file_location("contact_angle_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inst = mod.instrument_check()
    assert len(inst["rows"]) == 10
    assert all(isinstance(r["theta_measured_deg"], float) for r in inst["rows"])
    assert all(r["theta_measured_deg"] == r["theta_measured_deg"] for r in inst["rows"])
    assert inst["all_within_gate"] is True
    assert inst["worst_abs_error_deg"] < 0.05


def test_every_referenced_artifact_path_exists():
    """A claim may not point at an artifact the repository does not carry.

    Regression guard for m14849: m14729 registered two CSF-dispersion claims
    against a path the m13895 relocation had already retired, so the module
    fixture raised ``FileNotFoundError`` and *all* 20 ledger tests collapsed into
    setup errors instead of naming the offending file.  This guard scans the
    builder source for artifact references and fails with the missing paths.
    """
    import re

    src = BUILDER.read_text()
    paths = set(re.findall(r'_load\(\s*"([^"]+)"\s*\)', src))
    paths |= set(re.findall(r'^\s*rel = "([^"]+\.json)"', src, re.M))
    assert paths, "no artifact references found -- the scanner stopped matching"
    missing = sorted(p for p in paths if not (REPO / p).is_file())
    assert not missing, (
        f"ledger claims reference artifacts that do not exist in the repo: {missing}"
    )
    # m17086: the root cause of the m14849 break was the *location class*, not a
    # typo -- ``runs/`` is scratch and the delivery filter strips it, so a claim
    # there is dead the moment it is committed.  Forbid the class, not just the
    # two paths that happened to be wrong.
    scratch = sorted(p for p in paths if p.split("/", 1)[0] == "runs")
    assert not scratch, (
        "ledger claims point into the scratch tree runs/, which the delivery "
        f"filter does not keep (commit the artifact under benchmarks/<harness>/): {scratch}"
    )


def test_missing_artifact_is_a_finding_not_a_crash():
    """One absent artifact may not blind the rest of the ledger (m17086).

    At HEAD every ledger test died in the module fixture because a single claim
    read a stripped artifact, so 55 correct claims and their evidence defects
    were unreachable.  The builder now turns the offending producer into a
    high-severity ``missing_artifact`` finding and keeps grading the rest.
    """
    mod = _load_builder()

    def ghost_producer():
        raise mod.MissingArtifact("runs/ghost/report.json")

    ledger = mod.build(producers=[ghost_producer])
    rules = [f["rule"] for f in ledger["findings"]]
    assert "missing_artifact" in rules
    assert ledger["summary"]["n_missing_artifacts"] == 1
    finding = next(f for f in ledger["findings"] if f["rule"] == "missing_artifact")
    assert finding["severity"] == "high"
    assert finding["claim_id"] == "runs/ghost/report.json"
    assert "ungraded" in finding["detail"]
    assert ledger["entries"] == [], "a producer without an artifact must not invent claims"


def test_resolve_selects_list_records_by_field():
    """``ladder[id=L400].c`` must address a record, not a position (m17086).

    Campaign summaries order ``cases``/``ladder`` by run order, which is not
    stable across re-runs; a positional index would silently grade a different
    case after a re-ordering.  The selector is what the wedge-entry claims use.
    """
    mod = _load_builder()
    obj = {"ladder": [{"id": "L100", "c": 1.0}, {"id": "L400", "c": 2.0}]}
    assert mod.resolve(obj, "ladder[id=L400].c") == 2.0
    with pytest.raises(KeyError):
        mod.resolve(obj, "ladder[id=L999].c")


def test_relocated_campaign_artifact_matches_its_provenance():
    """The dispersion artifact of record is the campaign the ledger grades.

    ``meta.json`` carries the sha256 of the report the campaign wrote; the file
    committed under ``benchmarks/surface_tension_dispersion/`` must hash to it,
    otherwise the ledger would be grading a number that no artifact supports.
    """
    import hashlib

    base = REPO / "benchmarks" / "surface_tension_dispersion"
    meta = json.loads((base / "meta.json").read_text())
    digest = hashlib.sha256((base / "report.json").read_bytes()).hexdigest()
    assert digest == meta["report_sha256"]
