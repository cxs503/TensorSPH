"""The free-surface pressure closure must be *visible in the claim ledger*.

``benchmarks/laplace_pressure_closure`` fixed the P0 defect of
``backlog_update_m2971.md``: with the Adami/Ihmsen/Adams (2012) shell-pressure
extrapolation switched on, the Young-Laplace read-out of the same drop goes from
about -38 % to -3.4 %.  For nine missions that number lived only inside that
benchmark's own document, while the canonical claim surface
(``docs/verification_ledger.md``) still told every reader that the platform sat
at -34 % and that "the free-surface pressure closure" was the open defect.  The
evidence was islanded, not missing.

These tests pin the promotion, and they pin its *limits*:

* provenance — every registered closure row is re-derived here from
  ``raw/relax_ladder*.json``, not from the ledger and not from the benchmark's
  own summary; a hand-edited ledger entry fails,
* honesty    — the closure is PROVISIONAL: two of the four tested ``(dx, c0)``
  points violate the spurious-current gate, so their in-band numbers stay
  WITHHELD and raise an ``inadmissible_run`` finding.  A future edit that
  "cleans up" the ledger by dropping those rows, or that rounds -3.41 % up to a
  PASS, fails here,
* separation — the static force-balance read-out (which never touches the
  pressure field) is registered next to it, so the -38 % can be attributed to
  the pressure assembly rather than to the surface-tension model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "laplace_pressure_closure"
BUILDER = REPO / "benchmarks" / "verification_ledger" / "build_ledger.py"

CRITERION_PCT = 15.0
PAPER_PCT = 3.0
ADMISSIBILITY_LIMIT = 0.1


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


@pytest.fixture(scope="module")
def raw():
    """Every recorded ladder case, keyed by (dx_m, c0, case-name) — read from disk."""
    out = {}
    for path in sorted((BENCH / "raw").glob("relax_ladder*.json")):
        d = json.loads(path.read_text())
        for case in d["cases"]:
            out[(d["dx_m"], d["sound_speed"], case["case"])] = (path, d, case)
    assert out, "no raw closure evidence committed"
    return out


def _closure_entries(ledger):
    return [e for e in ledger["entries"] if e["claim_id"].startswith("young_laplace.closure.")]


# ---------------------------------------------------------------------------
# provenance: the ledger row must be the run, not a retelling of it
# ---------------------------------------------------------------------------
def test_closure_claim_tracks_the_raw_ladder(ledger, raw):
    entries = _closure_entries(ledger)
    assert entries, "the closure evidence is not registered in the ledger"
    for e in entries:
        # claim_id encodes the point; the raw file is the authority for the number
        variant = e["claim_id"].split(".")[2]
        hits = [v for (dx, c0, name), v in raw.items() if name == variant]
        path, d, case = min(
            hits,
            key=lambda v: abs(e["rel_error_pct"] - v[2]["tail_rel_error_pct"]),
        )
        assert abs(e["rel_error_pct"] - case["tail_rel_error_pct"]) < 1e-6, (
            f"{e['claim_id']}: ledger says {e['rel_error_pct']:+} %, raw {path.name} "
            f"says {case['tail_rel_error_pct']:+} % — the ledger must not be edited by hand"
        )
        assert e["artifact"].endswith(path.name)
        # and the error must be the error of that run against sigma/R, recomputed here
        recomputed = 100.0 * (
            case["tail_mean_Pa"] - d["analytic_dp_Pa"]
        ) / d["analytic_dp_Pa"]
        assert e["rel_error_pct"] == pytest.approx(recomputed, abs=1e-3)


def test_closure_claim_carries_the_run_identity(by_id):
    """A number without its (dx, c0, particles, steps) is not evidence."""
    for e in _closure_entries_values(by_id):
        definition = e["measured"]["definition"]
        assert "dx = " in definition and "c0 = " in definition
        assert "particles" in definition and "steps" in definition


def _closure_entries_values(by_id):
    return [e for cid, e in by_id.items() if cid.startswith("young_laplace.closure.")]


# ---------------------------------------------------------------------------
# honesty: admissibility travels with the reading
# ---------------------------------------------------------------------------
def test_inadmissible_closure_points_are_withheld(by_id, ledger):
    flagged = {
        f["claim_id"] for f in ledger["findings"] if f["rule"] == "inadmissible_run"
    }
    closure_inadmissible = [
        e for e in _closure_entries_values(by_id)
        if e["admissibility"] and not e["admissibility"]["pass"]
        and not e["superseded_by"]          # duplicates are provenance, not witnesses
    ]
    assert closure_inadmissible, (
        "the two closure points that violate Adami's spurious-current gate must stay "
        "in the ledger — dropping them would hide why the closure is only provisional"
    )
    for e in closure_inadmissible:
        assert e["admissibility"]["value"] >= ADMISSIBILITY_LIMIT
        assert e["grade"] == "WITHHELD"
        assert e["artifact_status"] != "PASS"
        assert e["claim_id"] in flagged, "an uncredited in-band reading needs a finding"


def test_admissible_closure_points_close_the_historical_deficit(by_id):
    admissible = [
        e for e in _closure_entries_values(by_id)
        if e["admissibility"] and e["admissibility"]["pass"]
        and abs(e["rel_error_pct"]) <= CRITERION_PCT
    ]
    labels = sorted(e["claim_id"] for e in admissible)
    assert len(admissible) >= 2, labels
    best = min(admissible, key=lambda e: abs(e["rel_error_pct"]))
    # the closure moves the read-out from the historical -38 % into the band ...
    assert abs(best["rel_error_pct"]) < 5.0, labels
    # ... but not into the 3 % paper band, so PASS would be an overstatement
    assert best["rel_error_pct"] != pytest.approx(0.0, abs=PAPER_PCT)
    for e in admissible:
        assert e["grade"] == "MARGINAL", e["claim_id"]


def test_ledger_says_the_closure_is_provisional(by_id):
    """No reader may take the best point for the capability."""
    best = by_id["young_laplace.closure.closure_on_interior_layers2.dx1.5mm_c20"]
    assert "PROVISIONAL" in best["notes"]
    assert "two of the four" in best["notes"]
    for cid in (
        "young_laplace.closure.closure_on_interior_layers2.dx2mm_c20",
        "young_laplace.closure.closure_on_interior_layers2.dx2mm_c5",
    ):
        assert "INADMISSIBLE" in by_id[cid]["notes"], cid


def test_duplicate_donor_layers_are_not_counted_as_evidence(by_id):
    """layers=4 repeats layers=2 bit for bit — provenance, not a second witness."""
    twin = by_id["young_laplace.closure.closure_on_interior_layers4.dx2mm_c20"]
    assert twin["superseded_by"] == (
        "young_laplace.closure.closure_on_interior_layers2.dx2mm_c20"
    )
    assert twin["grade"] == "WITHDRAWN"


# ---------------------------------------------------------------------------
# separation: pressure assembly vs surface-tension force
# ---------------------------------------------------------------------------
def test_force_balance_integral_is_registered_and_paper_grade(by_id):
    e = by_id["young_laplace.force_balance.flux_integral.dx2mm"]
    ref = e["reference"]["value"]
    raw = json.loads((BENCH / "report_force_balance.json").read_text())
    fs = raw["cases"][0]["force_side"]
    assert e["measured"]["value"] == pytest.approx(fs["delta_p_force_Pa"], abs=1e-5)
    assert ref == pytest.approx(raw["reference"]["analytic_dp_Pa"])
    assert abs(e["rel_error_pct"]) <= PAPER_PCT
    assert e["grade"] == "PASS"
    # the force path is within the band while the closure-off pressure path is not:
    # that is the statement that the historical -38 % was a pressure-assembly defect
    floor = by_id["young_laplace.absolute_tail.best_of_6"]
    assert floor["grade"] == "WITHHELD"
    assert abs(floor["rel_error_pct"]) > 5.0 * abs(e["rel_error_pct"])


# ---------------------------------------------------------------------------
# coverage: the closure evidence must not go back behind a waiver
# ---------------------------------------------------------------------------
def test_closure_evidence_is_not_waived(ledger):
    waiver = json.loads(
        (REPO / "benchmarks" / "verification_ledger" / "waivers.json").read_text()
    )
    waived = set(waiver["unregistered_ok"])
    registered = {e["artifact"] for e in ledger["entries"]}
    closure_artifacts = {
        str(p.relative_to(REPO))
        for p in list((BENCH / "raw").glob("relax_ladder*.json"))
        + [BENCH / "report_force_balance.json"]
    }
    for rel in sorted(closure_artifacts):
        assert rel not in waived, (
            f"{rel} is waived again: the closure benchmark's evidence may not be exempted "
            "on the promise that a claim exists elsewhere — register it or explain it"
        )
    # the artifact the retired waiver named is now a claim source
    assert "benchmarks/laplace_pressure_closure/raw/relax_ladder.json" in registered
    unregistered = [
        f for f in ledger["findings"]
        if f["rule"] == "unregistered_artifact" and "laplace_pressure_closure" in f["claim_id"]
    ]
    assert not unregistered
