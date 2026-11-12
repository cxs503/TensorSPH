"""Gates for the open-free-surface audit of the capillary-wave harness.

The audit (``benchmarks/capillary_wave/open_surface_audit.json``) records a
defect: a linear capillary standing wave cannot be carried by the current
discrete surface-tension force on an open free surface.  These tests make that
record load-bearing:

* the committed JSON, the page and the figure must exist and agree,
* the page must be a fresh rendering of the JSON (no hand-typed numbers),
* the harness must still be *known* not to run at HEAD -- if it starts running,
  this test fails and forces the verification record to be rewritten,
* the root-cause numbers must stay inside the bands quoted in the page, so a
  real improvement (or a regression) cannot slip through unnoticed.

References: Lamb (1932) Art. 267/348; Adami, Ihmsen & Adams (2012) JCP 231:7057.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "capillary_wave"
AUDIT = BENCH / "open_surface_audit.json"
DOC = REPO / "docs" / "verification_capillary_wave_open_surface.md"
#: The figure is a rendering of the artifact, so it is committed where generated
#: binaries actually survive the merge path (83 ``*.png`` under ``docs/``, zero
#: under ``benchmarks/`` at HEAD) and regenerated from the JSON by the gate below.
FIGURE = REPO / "docs" / "figures" / "capillary_wave_open_surface.png"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return json.loads(AUDIT.read_text())


# ---------------------------------------------------------------------------
# existence and anti-drift
# ---------------------------------------------------------------------------
def test_artifact_page_and_figure_exist():
    assert AUDIT.is_file(), "the audit artifact is the evidence debt payment"
    assert DOC.is_file()
    assert FIGURE.is_file(), (
        f"{FIGURE.relative_to(REPO)} is missing -- the page embeds it, so a "
        "missing figure is a broken evidence record.  Regenerate with "
        "`make_doc.py --figures` (it needs only the committed JSON)."
    )
    assert FIGURE.stat().st_size > 10_000


def test_every_image_link_in_the_page_resolves():
    """A record that embeds a path nobody carries is not a record.

    This is the gate for the defect that made this suite red on ``main``: the
    figure was written next to the JSON under ``benchmarks/``, four branches
    committed it, and ``main`` never received it, so the page rendered a broken
    image while every other assertion passed.
    """
    import re

    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", DOC.read_text())
    assert links, "the page must embed the audit figure"
    for link in links:
        target = (DOC.parent / link).resolve()
        assert target.is_file(), f"page links {link}, which does not exist"
    # generated binaries ship under docs/ so that the merge path keeps them
    assert FIGURE.parent == REPO / "docs" / "figures"


def test_figure_is_a_fresh_rendering_of_the_artifact(audit, tmp_path):
    """The committed figure must be reproducible from the committed JSON."""
    import hashlib

    make_doc = _load("capillary_make_doc", BENCH / "make_doc.py")
    fresh = make_doc.render_figure(audit, tmp_path / "fresh.png")
    committed = FIGURE.read_bytes()
    if fresh.read_bytes() != committed:
        # matplotlib/metadata drift between versions: fall back to pixels
        import matplotlib.image as mpimg
        import numpy as np

        a = mpimg.imread(fresh)
        b = mpimg.imread(FIGURE)
        assert a.shape == b.shape
        assert np.array_equal(a, b), (
            "docs figure does not match a fresh rendering of the audit JSON -- "
            "run `make_doc.py --figures`"
        )
    else:
        assert hashlib.sha256(committed).hexdigest()  # determinism, no exception


def test_page_is_a_fresh_rendering_of_the_artifact(audit):
    make_doc = _load("capillary_make_doc", BENCH / "make_doc.py")
    assert DOC.read_text() == make_doc.render(audit), (
        "docs/verification_capillary_wave_open_surface.md is stale -- "
        "run benchmarks/capillary_wave/make_doc.py"
    )


def test_artifact_references_and_verdict(audit):
    assert audit["benchmark"] == "capillary_wave_open_surface_audit"
    assert "Lamb (1932)" in audit["references"]["dispersion"]
    assert "Adami" in audit["references"]["admissibility"]
    assert audit["verdict"]["grade"] == "WITHHELD"


# ---------------------------------------------------------------------------
# the defect itself
# ---------------------------------------------------------------------------
def test_harness_is_recorded_as_not_running_at_head(audit):
    """If this ever passes, the page must be rewritten -- that is the point."""
    head = audit["harness_at_head"]
    assert head["status"] == "aborted", (
        "the harness now completes at HEAD: retire the WITHHELD verdict and "
        "grade the dispersion claim instead of patching this assertion away"
    )
    assert head["escape_step"] is not None
    assert 0 < head["escape_step"] / (audit["horizon_periods"] * 683) < 1.0


def test_csf_force_exceeds_the_physical_restoring_acceleration(audit):
    acc = audit["initial_acceleration"]
    a_phys = acc["a_physical_omega2A_m_per_s2"]
    # sanity: omega^2 A for sigma/R at this k
    k = audit["curvature"]["k_1_per_m"]
    assert acc["omega_rad_per_s"] == pytest.approx(
        math.sqrt(7.2 * k**3 / 1000.0 * math.tanh(k * 0.02)), rel=1e-6)
    assert a_phys == pytest.approx(acc["omega_rad_per_s"] ** 2
                                   * audit["curvature"]["amplitude_m"], rel=1e-3)
    assert acc["csf_excess_over_physical"] > 10.0, (
        "the CSF excess is no longer an order of magnitude above the signal: "
        "re-measure and update the verdict"
    )
    # the excess is the difference of the two measured fields, not a fit
    assert acc["csf_excess_a_max_m_per_s2"] == pytest.approx(
        acc["csf_on"]["a_max_m_per_s2"] - acc["csf_off"]["a_max_m_per_s2"], rel=1e-9)


def test_curvature_noise_is_above_the_perturbation_signal(audit):
    curv = audit["curvature"]
    k2a = curv["analytic_signal_kappa_1_per_m"]
    assert k2a == pytest.approx(curv["k_1_per_m"] ** 2 * curv["amplitude_m"], rel=1e-6)
    for cell in curv["cells"]:
        assert cell["kappa_rms_1_per_m"] > k2a, cell
        assert 1.0 < cell["noise_over_signal"] < 100.0, cell
        assert cell["n_interface_particles"] > 10, cell


def test_no_variant_is_admissible(audit):
    sweep = audit["variant_sweep"]
    assert len(sweep) >= 8
    assert not any(row["admissible_adami"] for row in sweep)
    for row in sweep:
        assert row["v_max_over_u_ref"] > 0.1
        assert row["rho_min_over_rho0"] <= 1.05
    assert audit["summary"]["n_aborted"] >= 1


def test_amplitude_ladder_keeps_a_completed_run_to_score(audit):
    """The ladder must contain at least one finished run, else nothing was measured."""
    ladder = audit["amplitude_ladder"]
    completed = [row for row in ladder if row["status"] == "completed"]
    assert completed, "every ladder run aborted; the audit has no reading to report"
    for row in completed:
        assert row["fit_residual_rel"] is not None
        assert abs(row["omega_rel_error_pct"]) > 3.0 or row["fit_residual_rel"] > 0.2, (
            f"{row['variant']} now recovers the analytic frequency: the capability "
            "changed and the page must be rewritten"
        )
        assert row["amplitude_over_dx"] in (0.5, 2.0)


def test_page_states_the_3pct_answer_explicitly():
    text = DOC.read_text()
    assert "not** within 3 % of Lamb" in text or "not within 3 %" in text.lower()
    assert "WITHHELD" in text
    assert "no admissibility" not in text


# ---------------------------------------------------------------------------
# the record must stay a measurement of *this* tree
# ---------------------------------------------------------------------------
# Everything above reads the committed JSON.  That is enough to catch a stale
# page or a doctored number, but not enough to catch the *solver* moving: if
# ``csf_consistent`` / ``droplet_impact`` / ``neighbor_search`` change, the JSON
# stays green while the page silently understates the defect (worse solver) or
# claims a defect that no longer exists (fixed solver).  The gates below
# re-measure instead of re-reading, so the verdict has to be rewritten when the
# capability changes -- in either direction.
REPRO = BENCH / "reproducibility.py"
ATTESTATION = BENCH / "reproducibility.json"
REPRO_DOC = REPO / "docs" / "dev_report_m14576_open_surface_reproducibility.md"
#: Predeclared band (see ``reproducibility.py``).  It is three orders of
#: magnitude above the measured run-to-run noise (0.0) and far below the
#: smallest effect the page quotes (3.7x), so it can fail only on real drift.
RTOL = 1e-6


def test_committed_numbers_reproduce_at_head(audit):
    """Re-measure the load-bearing subset (~10 s) against the committed JSON."""
    repro = _load("capillary_reproducibility", REPRO)
    fresh = repro.measure()
    report = repro.compare({key: audit[key] for key in repro.SUBSET_KEYS}, fresh)
    assert report["rtol"] == RTOL
    assert report["n_float"] >= 90, (
        f"only {report['n_float']} float fields compared -- the subset shrank, "
        "so this gate no longer covers the verdict"
    )
    assert report["failures"] == [], (
        "the committed audit no longer matches a fresh run:\n"
        + "\n".join(f"  {f['path']}: committed={f['committed']!r} "
                    f"fresh={f['fresh']!r}" for f in report["failures"][:5])
        + "\nif the solver really changed, re-run "
        "`benchmarks/capillary_wave/open_surface_audit.py`, refresh the page with "
        "`make_doc.py` and rewrite the verdict; do not widen rtol"
    )
    # the abort is the headline reading: it must not move at all
    assert fresh["harness_at_head"]["status"] == "aborted"
    assert fresh["harness_at_head"]["escape_step"] == audit["harness_at_head"]["escape_step"]


def test_attestation_is_in_step_with_the_artifact():
    """A refreshed audit artifact must carry a fresh attestation."""
    import hashlib

    att = json.loads(ATTESTATION.read_text())
    digest = hashlib.sha256(AUDIT.read_bytes()).hexdigest()
    assert att["audited_artifact_sha256"] == digest, (
        "open_surface_audit.json changed after the reproducibility attestation "
        "was written -- re-run `reproducibility.py --full --report ...`"
    )
    assert att["audited_verdict"] == "WITHHELD"
    assert att["subset"]["rtol"] == RTOL
    assert att["subset"]["failures"] == []
    assert att["subset"]["n_float"] >= 90
    assert att["full_rerun"] is not None and att["full_rerun"]["failures"] == [], (
        "the full re-run is not clean; the record is drifting"
    )
    assert att["environment"]["git_head"], "the attestation must name the tree it measured"


def test_reproducibility_report_is_a_fresh_rendering(monkeypatch):
    """The report is generated from the attestation, not hand-typed."""
    repro = _load("capillary_reproducibility", REPRO)
    assert REPRO_DOC.is_file(), (
        f"{REPRO_DOC.relative_to(REPO)} is missing -- regenerate with "
        "`reproducibility.py --full --report ...`"
    )
    attestation = json.loads(ATTESTATION.read_text())
    assert REPRO_DOC.read_text() == repro.render_report(attestation), (
        "the reproducibility report is stale -- regenerate it from the attestation"
    )
