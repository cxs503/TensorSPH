"""Gates for the round-2 capillary-wave study (the resolution criterion).

Round 1 (`benchmarks/capillary_wave/open_surface_audit.json`) recorded *that*
the capillary standing wave is not gradable at HEAD.  Round 2
(`benchmarks/capillary_wave/resolution_criterion.json`) separates the two
candidate causes -- the CSF force on the open free surface vs the wall boundary
condition -- and turns "what would it take?" into numbers.

These tests make that record load-bearing:

* the committed JSON, the page and the figure must exist and agree,
* both the page and the figure must be fresh renderings of the JSON (no
  hand-typed numbers, no stale binaries),
* the wall-boundary comparison must stay the way it was measured: if the bare
  clamp box ever completes the horizon, this suite fails and forces the record
  to be rewritten instead of being patched,
* the three structural findings (force budget on the walls, curvature floor that
  does not vanish, ladder shortfall) must stay inside the bands quoted in the
  page, so a real improvement -- or a regression -- cannot slip through.

References: Lamb (1932) Art. 267/348; Adami, Ihmsen & Adams (2012) JCP 231:7057;
Adami, Hu & Adams (2012) JCP 231:7057-7075.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "capillary_wave"
AUDIT = BENCH / "resolution_criterion.json"
DOC = REPO / "docs" / "verification_capillary_wave_resolution_criterion.md"
FIGURE = REPO / "docs" / "figures" / "capillary_wave_resolution_criterion.png"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return json.loads(AUDIT.read_text())


@pytest.fixture(scope="module")
def make_doc():
    return _load("capillary_make_doc_round2", BENCH / "make_doc.py")


# ---------------------------------------------------------------------------
# existence and anti-drift
# ---------------------------------------------------------------------------
def test_artifact_page_and_figure_exist():
    assert AUDIT.is_file(), "the study artifact is the evidence"
    assert DOC.is_file()
    assert FIGURE.is_file(), (
        f"{FIGURE.relative_to(REPO)} is missing -- the page embeds it, so a "
        "missing figure is a broken record.  Regenerate with "
        "`make_doc.py --figures` (it needs only the committed JSON)."
    )
    assert FIGURE.stat().st_size > 10_000


def test_every_image_link_in_the_page_resolves():
    """A record that embeds a path nobody carries is not a record."""
    import re

    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", DOC.read_text())
    assert links, "the page must embed the study figure"
    for link in links:
        target = (DOC.parent / link).resolve()
        assert target.is_file(), f"page links {link}, which does not exist"
    assert FIGURE.parent == REPO / "docs" / "figures"


def test_page_is_a_fresh_rendering_of_the_artifact(audit, make_doc):
    assert DOC.read_text() == make_doc.render_resolution(audit), (
        "docs/verification_capillary_wave_resolution_criterion.md is stale -- "
        "run benchmarks/capillary_wave/make_doc.py --figures"
    )


def test_figure_is_a_fresh_rendering_of_the_artifact(audit, make_doc, tmp_path):
    """The committed figure must be reproducible from the committed JSON."""
    fresh = make_doc.render_resolution_figure(audit, tmp_path / "fresh.png")
    committed = FIGURE.read_bytes()
    if fresh.read_bytes() != committed:
        # matplotlib/metadata drift between versions: fall back to pixels
        import matplotlib.image as mpimg
        import numpy as np

        a = mpimg.imread(fresh)
        b = mpimg.imread(FIGURE)
        assert a.shape == b.shape
        assert np.array_equal(a, b), (
            "the figure does not match a fresh rendering of the study JSON -- "
            "run `make_doc.py --figures`"
        )


def test_record_declares_its_predeclared_criteria(audit):
    pre = audit["predeclared"]
    assert pre["thresholds"]["dispersion_rel_error_pct"] == 3.0
    assert pre["thresholds"]["adami_spurious_current"] == 0.10
    assert pre["tuned_after_results"] is False
    assert "escape" in pre["stopping_rule"]
    assert audit["environment"]["deterministic"] is True


# ---------------------------------------------------------------------------
# finding 1: the initial CSF force lives on the walls
# ---------------------------------------------------------------------------
def test_force_budget_is_dominated_by_the_walls(audit):
    budget = audit["force_budget"]
    zones = budget["zones"]
    assert set(zones) == {"wall_adjacent", "free_surface", "interior"}
    assert budget["wall_share"] == pytest.approx(
        zones["wall_adjacent"]["share_of_total_abs_a"], rel=1e-12)
    assert budget["wall_share"] > 0.7, (
        "the wall/corner colour truncation no longer dominates the CSF budget: "
        "the round-1 explanation must be re-measured, not patched away")
    # the corners are the worst case: two truncated directions at once
    assert budget["corner"]["n_particles"] >= 4
    assert budget["corner"]["a_max_m_per_s2"] == pytest.approx(
        budget["a_max_m_per_s2"], rel=1e-9)
    # and the whole thing is unphysically scaled against omega^2 A
    assert budget["a_max_over_physical"] > 5.0


# ---------------------------------------------------------------------------
# finding 2: the wall boundary condition carries the run
# ---------------------------------------------------------------------------
def test_wall_bc_rescues_the_case(audit):
    clamp = audit["wall_bc"]["clamp_box_no_wall"]
    wall = audit["wall_bc"]["static_wall_particles"]
    assert clamp["boundary"] == "clamp_box_no_wall"
    assert wall["boundary"] == "static_wall_particles"
    assert wall["n_wall_particles"] > 0
    assert clamp["status"] == "aborted", (
        "the bare clamp box now completes the horizon: retire the round-1 "
        "explanation and rewrite docs/verification_capillary_wave_open_surface.md")
    assert clamp["escape_step"] is not None
    assert wall["status"] == "completed", (
        "static wall particles no longer carry the run: the abort was not (only) "
        "a wall artefact and the page must be rewritten")
    assert 0 < clamp["escape_fraction_of_horizon"] < 1.0
    assert audit["summary"]["wall_bc_rescues_case"] is True


# ---------------------------------------------------------------------------
# finding 3: the curvature floor is a bias, not noise
# ---------------------------------------------------------------------------
def test_curvature_floor_does_not_vanish(audit):
    cells = audit["curvature_floor"]["cells"]
    assert [c["points_per_wavelength"] for c in cells] == [20, 40, 80]
    for cell in cells:
        assert cell["n_interface_particles"] > 10
        assert cell["kappa_flat_mean_1_per_m"] > 20.0, (
            "a flat interface now reports (nearly) zero curvature: the "
            "estimator defect is fixed and this record must be rewritten")
        assert cell["kappa_flat_rms_1_per_m"] > cell["kappa_flat_mean_1_per_m"]
    # refinement does not remove it (the mean is a bias, the rms grows)
    assert cells[-1]["kappa_flat_rms_1_per_m"] > 0.8 * cells[0][
        "kappa_flat_rms_1_per_m"]
    assert cells[-1]["kappa_flat_mean_1_per_m"] > 0.5 * cells[0][
        "kappa_flat_mean_1_per_m"]


def test_amplitude_criterion_is_out_of_reach_for_the_harness(audit):
    crit = audit["amplitude_criterion"]
    harness_amp_over_dx = audit["case"]["amp_over_dx"]
    assert harness_amp_over_dx == 0.5
    assert crit["required_A_over_dx_10x_snr"] > 2.0 * harness_amp_over_dx, (
        "the measured floor is now small enough for the harness amplitude: the "
        "dispersion claim may be gradable -- re-run and rewrite the verdict")
    for cell in crit["cells"]:
        assert cell["floor_source"] in ("interior", "all_shell")
        assert cell["required_A_over_dx_for_10x"] == pytest.approx(
            10.0 * cell["required_A_over_dx_for_parity"], rel=1e-9)


# ---------------------------------------------------------------------------
# finding 4: the ladder and the verdict
# ---------------------------------------------------------------------------
def test_ladder_has_a_completed_run_and_scores_it(audit):
    rows = audit["ladder"]
    assert len(rows) >= 3
    completed = [r for r in rows if r["status"] == "completed"]
    assert completed, "every ladder cell aborted; nothing was measured"
    assert any(row["steepness_kA"] < 1.0 for row in rows), (
        "the ladder must contain a linear cell -- only those can grade the "
        "linear dispersion relation")
    for row in rows:
        if row["steepness_kA"] >= 1.0:      # nonlinear: never gradable
            assert row["graded"] is False, row["variant"]
        assert row["admissible_adami"] == (row["v_max_over_u_ref"] < 0.10)
        assert row["graded"] == bool(
            row["status"] == "completed" and row["admissible_adami"]
            and abs(row["omega_relative_error_pct"])
            < audit["predeclared"]["thresholds"]["dispersion_rel_error_pct"]
            and row["fit_residual_rel"]
            < audit["predeclared"]["thresholds"]["fit_residual_rel"])
    assert audit["summary"]["n_completed"] == len(completed)


def test_verdict_agrees_with_the_ladder(audit):
    rows = audit["ladder"]
    graded = [r for r in rows if r["graded"]]
    verdict = audit["verdict"]
    assert verdict["grade"] in ("REAL", "WITHHELD")
    if graded:
        assert verdict["grade"] == "REAL"
        assert sorted(verdict["graded_variants"]) == sorted(
            r["variant"] for r in graded)
    else:
        assert verdict["grade"] == "WITHHELD"
        assert "3 %" in verdict["reason"]


def test_page_answers_the_3pct_question_explicitly():
    text = DOC.read_text()
    assert "### Is the key quantity within 3 % of the reference?" in text
    assert ("**No.**" in text) or ("**Yes**" in text), (
        "the page must answer the 3 % question in words, not imply it")
    assert "WITHHELD" in text or "REAL" in text
