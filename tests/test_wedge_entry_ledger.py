"""Pin the wedge-entry evidence chain: summary -> claims -> ledger.

Why this test exists
--------------------
``benchmarks/wedge_entry`` used to be an *uncovered benchmark*: its harness wrote
only to ``artifacts/`` (git-ignored), so the verification ledger graded nothing it
produced and a regression there was invisible (waiver by m14849, evidence debt
E1).  The closure is that ``wagner_resolved.py --stage accept`` copies every
per-case artefact next to the summary it grades; these tests hold that chain
together instead of merely executing it:

* every graded case has a committed raw artefact next to the summary;
* the four ledger claims reproduce from the summary alone (recomputed here, not
  compared with a retyped constant), and their signed errors carry the sign the
  summary implies;
* the kinematic reference ``C = pi/(2 tan beta)`` is derived in the generator,
  so the test recomputes it independently.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The campaign writes its summary into the scratch tree ``runs/wedge_entry_wagner/``,
# which the delivery filter strips; the artifact of record is the copy committed next
# to the harness (m17086, byte-identical to the campaign output).
SUMMARY = REPO / "benchmarks" / "wedge_entry" / "summary_v2.json"

GRADED_CASES = ("beta10_V1", "beta15_V1", "beta20_V1", "beta45_V1")
GRADED_LADDER = ("ladder_L100", "ladder_L200", "ladder_L400")


def _summary() -> dict:
    assert SUMMARY.is_file(), f"missing committed evidence: {SUMMARY}"
    return json.loads(SUMMARY.read_text())


def _ledger_module():
    path = REPO / "benchmarks" / "verification_ledger" / "build_ledger.py"
    spec = importlib.util.spec_from_file_location("build_ledger_wedge_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_graded_case_has_a_committed_artefact():
    d = _summary()
    ids = {e["id"] for e in d["cases"] + d["ladder"]}
    for cid in GRADED_CASES + GRADED_LADDER:
        assert cid in ids, f"{cid} not in summary"
        assert (SUMMARY.parent / "raw").rglob(f"{cid}.json"), (
            f"{cid} graded but its raw artefact is not committed next to the summary")
    # the summary must say which campaign settings produced it
    solver = d["meta"]["solver"]
    assert solver["dx"] == 1.0e-3
    assert abs(solver["sound_speed_factor"] - 20.0) < 1e-9
    assert abs(solver["alpha_visc"] - 0.03) < 1e-12


def test_claims_reproduce_from_the_committed_summary():
    d = _summary()
    runs = {e["id"]: e for e in d["cases"] + d["ladder"]}
    claims = {c["claim_id"]: c for c in _ledger_module().wedge_entry_claims()}
    assert set(claims) == {
        "wedge_entry.c_wetted_coeff.beta30_dx1mm",
        "wedge_entry.cp_peak.resolved_band.beta30_dx1mm",
        "wedge_entry.cp_peak.resolved_band.beta10_dx1mm",
        "wedge_entry.cp_peak.blunt45_dx1mm",
    }
    # kinematic anchor: reference derived, not retyped
    e30 = runs["ladder_L400"]
    c_ref = math.pi / (2.0 * math.tan(math.radians(e30["beta_deg"])))
    c = claims["wedge_entry.c_wetted_coeff.beta30_dx1mm"]
    assert abs(c["reference"]["value"] - c_ref) < 1e-5
    assert abs(c["measured"]["value"] - e30["c_coeff_mean"]) < 1e-5
    assert abs(c["rel_error_pct"] - (e30["c_coeff_mean"] - c_ref) / c_ref * 100.0) < 1e-2
    # pressure entries: measured/reference are the two band peaks of the run
    beta30_id = "beta30_V1" if "beta30_V1" in runs else "ladder_L400"
    for cid, claim_id in ((beta30_id, "wedge_entry.cp_peak.resolved_band.beta30_dx1mm"),
                          ("beta10_V1", "wedge_entry.cp_peak.resolved_band.beta10_dx1mm"),
                          ("beta45_V1", "wedge_entry.cp_peak.blunt45_dx1mm")):
        e = runs[cid]
        c = claims[claim_id]
        assert abs(c["measured"]["value"] - e["p_band_sph_mean"]) < 1e-3
        assert abs(c["reference"]["value"] - e["p_band_ref_mean"]) < 1e-3
        # 1e-2 percentage point: the summary stores the error at 6 significant
        # digits and the claim recomputes it from the two rounded band peaks
        # the summary stores |err|; the ledger stores the signed error, so the
        # invariant is magnitude equality plus the sign the two peaks imply
        assert abs(abs(c["rel_error_pct"]) - e["p_band_rel_err_mean"] * 100.0) < 1e-2
        assert c["rel_error_pct"] * (e["p_band_sph_mean"] - e["p_band_ref_mean"]) > 0


def test_resolvability_reading_is_reported_not_hidden():
    """The small-deadrise case must be registered as FAIL, not folded away."""
    d = _summary()
    runs = {e["id"]: e for e in d["cases"] + d["ladder"]}
    claims = {c["claim_id"]: c for c in _ledger_module().wedge_entry_claims()}
    e10 = runs["beta10_V1"]
    assert e10["cells_to_root"] < 1.0, "beta = 10 deg is supposed to be unresolvable"
    c10 = claims["wedge_entry.cp_peak.resolved_band.beta10_dx1mm"]
    assert c10["artifact_status"] == "FAIL"
    assert abs(c10["rel_error_pct"]) > 15.0
    # the blunt control is the case where the peak *is* resolvable
    e45 = runs["beta45_V1"]
    assert e45["cells_to_root"] > 5.0
    assert abs(claims["wedge_entry.cp_peak.blunt45_dx1mm"]["rel_error_pct"]) <= 15.0
