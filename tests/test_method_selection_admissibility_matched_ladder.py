"""Guards for the admissibility-matched Couette ladder (mission 23211).

The point of these tests is not that the script runs: it is that the classifier
can say *no* to a trend the artefacts do not support, can say *yes* when a
matched ladder really exists (negative control), and that the numbers it prints
are recomputed from the committed artefact rather than copied.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "method_selection"))

import admissibility_matched_ladder as aml  # noqa: E402

BASE = ROOT / "benchmarks" / "couette" / "out_design" / "viscous_time_convergence.json"
SCRIPT = ROOT / "benchmarks" / "method_selection" / "admissibility_matched_ladder.py"
CLAIM = aml.CLAIMS[0]


# --------------------------------------------------------------------------- #
# Synthetic ladder helper: the schema load_points() produces
# --------------------------------------------------------------------------- #
def make_points(rows):
    """rows = (rung, tier, bottom_pct, top_pct, drift)."""
    points = []
    for rung, tier, bottom, top, drift in rows:
        points.append({
            "label": f"r{rung}-vt{tier:g}",
            "rung": int(rung),
            "tier": float(tier),
            "t_end": float(tier) * 0.5,
            "bottom_pct": float(bottom),
            "top_pct": float(top),
            "steady_drift": float(drift),
            "admissible": float(drift) <= aml.DEFAULT_DRIFT_GATE,
            "source": "synthetic",
        })
    return points


def registry_record(pct=17.0, method="wcsph_wall_no_slip"):
    return {CLAIM["extractor"]: {"extractor": CLAIM["extractor"],
                                 "method": method,
                                 "measured_pct": pct,
                                 "quantity": "synthetic",
                                 "status": "banked"}}


# --------------------------------------------------------------------------- #
# 1. The numbers are recomputed from the committed artefact
# --------------------------------------------------------------------------- #
def test_readings_are_recomputed_not_copied():
    doc = json.loads(BASE.read_text())
    point = next(p for p in doc["points"] if p["label"] == "r16-vt3.6")
    assert aml.WALL_FIELDS["top"] == "top_pct"
    points, _ = aml.load_points([BASE])
    row = next(p for p in points if p["label"] == "r16-vt3.6")
    assert row["top_pct"] == pytest.approx(abs(1.0 - point["wall_shear_ratio_top"]) * 100.0,
                                          rel=0, abs=1e-12)
    assert row["bottom_pct"] == pytest.approx(
        abs(1.0 - point["wall_shear_ratio_bottom"]) * 100.0, rel=0, abs=1e-12)


def test_admissibility_flag_matches_the_runs_own_gate():
    doc = json.loads(BASE.read_text())
    gate = float(doc["gates"]["steady_drift"])
    points, _ = aml.load_points([BASE])
    for point in points:
        assert point["admissible"] == (point["steady_drift"] <= gate)


# --------------------------------------------------------------------------- #
# 2. The real claim: the exclusive evidence is a single admissible rung
# --------------------------------------------------------------------------- #
def test_coarse_rung_never_reaches_steady_state():
    """If H/dx=8 did have an admissible point, the mixed-class finding dies."""
    points, _ = aml.load_points([BASE])
    rows = aml.ladder(points, "top")["8"]
    assert [r["admissible"] for r in rows] == [False] * len(rows)
    assert min(r["steady_drift"] for r in rows) > aml.DEFAULT_DRIFT_GATE


def test_steady_end_falls_back_to_a_transient_reading():
    points, _ = aml.load_points([BASE])
    reading = aml.steady_end(aml.ladder(points, "top")["8"])
    assert reading["fallback_to_last_point"] is True
    assert reading["n_admissible"] == 0


def test_bank_claim_is_classified_as_untestable_not_as_a_trend():
    result = aml.build()
    claim = next(c for c in result["claims"] if c["claim_id"] == CLAIM["claim_id"])
    assert claim["class"] == "trend_untestable"
    assert claim["admissible_rung_count"] < 2
    assert claim["matched_trend"] is None
    assert "NOT ESTABLISHED" in claim["verdict"]


def test_bank_claim_is_the_published_exclusion_basis():
    """If this record stops being cited by a decision, the audit's scope changes."""
    result = aml.build()
    claim = next(c for c in result["claims"] if c["claim_id"] == CLAIM["claim_id"])
    labels = [row["label"] for row in claim["decision_linkage"]["rows"]]
    assert labels, "the record is no longer cited by any strict-bound decision"


def test_time_insensitive_exceedance_is_measured_not_asserted():
    """At the rung that has an admissible reading, no tier is inside the standard."""
    result = aml.build()
    claim = next(c for c in result["claims"] if c["claim_id"] == CLAIM["claim_id"])
    assert claim["time_insensitive_exceedance"] is True
    fine = next(r for r in claim["robustness"] if r["rung"] == CLAIM["fine_rung"])
    assert fine["min_pct"] > aml.OWNER_TOLERANCE_PCT
    assert fine["all_tiers_over_standard"] is True


# --------------------------------------------------------------------------- #
# 3. Negative controls: the classifier must be able to say "yes"
# --------------------------------------------------------------------------- #
def test_matched_ladder_is_reported_as_a_trend():
    """Two rungs, both admissible at the same tier -> a real trend."""
    points = make_points([
        (8, 1.2, 12.0, 9.0, 0.004),
        (8, 3.6, 10.0, 8.0, 0.005),
        (16, 1.2, 6.0, 11.0, 0.003),
        (16, 3.6, 4.0, 10.0, 0.004),
    ])
    entry = aml.adjudicate_claim(CLAIM, points, registry_record())
    assert entry["class"] == "matched_trend"
    assert entry["matched_trend"]["worsens"] is True
    assert entry["matched_trend"]["tier"] == pytest.approx(3.6)


def test_admissible_rungs_without_a_shared_tier_are_mixed():
    """Both rungs admissible, but never at the same terminal time."""
    points = make_points([
        (8, 1.2, 9.0, 9.0, 0.004),
        (16, 3.6, 4.0, 16.0, 0.004),
    ])
    entry = aml.adjudicate_claim(CLAIM, points, registry_record())
    assert entry["class"] == "mixed_admissibility"
    assert entry["matched_trend"] is None
    assert "different admissibility classes" in entry["verdict"]


def test_improving_matched_ladder_flips_the_sign():
    """The guard reports what the data says, not the published conclusion."""
    points = make_points([
        (8, 3.6, 9.0, 20.0, 0.005),
        (16, 3.6, 4.0, 11.0, 0.004),
    ])
    entry = aml.adjudicate_claim(CLAIM, points, registry_record())
    assert entry["class"] == "matched_trend"
    assert entry["matched_trend"]["worsens"] is False
    assert entry["time_insensitive_exceedance"] is False


# --------------------------------------------------------------------------- #
# 4. Artefact hygiene
# --------------------------------------------------------------------------- #
def test_conflicting_duplicates_are_rejected(tmp_path):
    """Same label, different reading -> a stale artefact, never silently merged."""
    doc = json.loads(BASE.read_text())
    stale = json.loads(BASE.read_text())
    stale["points"][0]["wall_shear_error_pct_top"] += 1.0
    first = tmp_path / "viscous_time_convergence_a.json"
    second = tmp_path / "viscous_time_convergence_b.json"
    first.write_text(json.dumps(doc))
    second.write_text(json.dumps(stale))
    with pytest.raises(SystemExit):
        aml.load_points([first, second])


def test_committed_artefacts_are_a_fresh_build():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, (
        "docs/reports/method_selection_admissibility_matched_ladder.{json,md} "
        "are stale; re-run the producer and commit the result.\n"
        + proc.stdout[-2000:] + proc.stderr[-2000:])


# --------------------------------------------------------------------------- #
# 5. The endorsed design reading, checked against the same sweep
# --------------------------------------------------------------------------- #
def test_endorsed_design_reading_is_reproduced_and_its_steadiness_reported():
    result = aml.build()
    design = result["design_reading"]
    assert design is not None
    assert design["recomputed_vs_banked_pct"] < 1e-6
    assert design["class"] in {"admissible_design_reading",
                               "design_reading_not_admissible"}
    if design["admissible_reading"]:
        assert design["admissible_reading"]["pct"] <= aml.OWNER_TOLERANCE_PCT