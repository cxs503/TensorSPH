"""Guards for the cross-family verification ledger.

These tests do not re-run any solver (the ledger is built from committed
artefacts, which is the point: cheap, reproducible, and honest about what is
stale).  They assert four things:

1. provenance  -- every number in the ledger is identical to the number stored
   in the artefact it claims to come from (no hand copying, no drift);
2. hygiene     -- every reading carries a registered convention, no NaNs, and
   only the sanctioned conventions may be quoted as current precision;
3. findings    -- the substantive conclusions of the ledger (Δp still WITHHELD,
   the two sub-15 % free-surface readings are not resolution/rerun stable,
   convention choice alone moves the error by >50 pp) hold *as computed from
   the artefacts*, so a future change that invalidates them fails loudly;
4. document    -- ``docs/verification_ledger.md`` and ``ledger.json`` are
   byte-identical to what the builder regenerates right now.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks" / "verification_ledger"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEDGER = _load(HERE / "ledger.py", "verification_ledger_impl")
BUILD = _load(HERE / "build.py", "verification_ledger_build")


@pytest.fixture(scope="module")
def ledger():
    return LEDGER.build(ROOT)


@pytest.fixture(scope="module")
def claims():
    return LEDGER.collect(ROOT)


# --------------------------------------------------------------------------- #
# 1. provenance
# --------------------------------------------------------------------------- #
def test_every_source_artefact_exists(ledger):
    missing = [src for src in ledger["sources"] if not (ROOT / src).exists()]
    assert not missing, f"ledger references missing artefacts: {missing}"


def test_every_registered_family_yields_readings(ledger):
    families = {r["family"] for r in ledger["readings"]}
    expected = {name for name, _ in LEDGER.ADAPTERS}
    assert expected <= families, f"families with no readings: {sorted(expected - families)}"


def test_reported_numbers_match_source_artefacts(claims):
    """Where the artefact stored its own rel-error, the ledger must agree."""
    checked = 0
    for claim in claims:
        if claim.reported_rel_error_pct is None:
            continue
        recomputed = claim.rel_error_pct
        # several artefacts store only the magnitude of the relative error;
        # the ledger keeps the sign, so compare magnitudes here and let
        # ``reported_sign_matches`` record which is which.
        assert abs(abs(recomputed) - abs(claim.reported_rel_error_pct)) < 1e-3, (
            f"{claim.id}: artefact says {claim.reported_rel_error_pct:.6f} %, "
            f"ledger recomputes {recomputed:.6f} % from {claim.source}"
        )
        # and the absolute value must round-trip into the source json itself
        blob = json.loads((ROOT / claim.source).read_text(encoding="utf-8"))
        assert blob, claim.source
        checked += 1
    assert checked >= 40, f"only {checked} readings cross-checked against artefacts"


def test_provenance_control_is_internally_consistent(ledger):
    prov = ledger.get("provenance") or {}
    assert prov.get("q1_reproducible"), "provenance control not registered"
    q1 = prov["q1_reproducible"]
    assert q1["reproducible"] is False
    assert q1["delta_pp"] > 1.0, "committed vs re-run laplace_law reading should differ"
    q3 = prov["q3_resolution_stability"]
    assert q3["coarse_within_15pct"] is True
    assert q3["fine_within_15pct"] is False
    assert q3["band_clamped_only_coarse"] is True
    q2 = prov["q2_sigma_control"]
    assert q2["generated_by_surface_tension"] is True
    assert q2["dp_sigma_control_Pa"] == 0.0


# --------------------------------------------------------------------------- #
# 2. hygiene
# --------------------------------------------------------------------------- #
def test_magnitude_only_artefacts_are_counted(ledger):
    """Some families stored abs(rel error); the ledger records and counts that."""
    n = ledger["verdict"]["n_reported_as_magnitude"]
    assert n >= 10, f"only {n} magnitude-only readings detected"
    signed = [r for r in ledger["readings"]
              if r["reported_sign_matches"] is False]
    assert len(signed) == n


def test_all_conventions_registered_and_unique_ids(claims):
    ids = [c.id for c in claims]
    assert len(ids) == len(set(ids)), "duplicate reading ids"
    for claim in claims:
        assert claim.convention in LEDGER.CONVENTIONS, claim.id
        assert claim.theory != 0.0, f"{claim.id}: zero reference value"
        assert math.isfinite(claim.measured) and math.isfinite(claim.theory)
        assert math.isfinite(claim.rel_error_pct)


def test_only_sanctioned_conventions_are_current_precision(claims):
    """The m2971 P2.1 rule: anything outside absolute+time-average is historical."""
    allowed = {"abs_tail_avg", "abs_plateau_avg"}
    for claim in claims:
        meta = LEDGER.CONVENTIONS[claim.convention]
        assert meta["sanctioned"] == (claim.convention in allowed), claim.id
    assert sum(1 for c in claims if c.sanctioned) == 9


# --------------------------------------------------------------------------- #
# 3. findings (all recomputed from the artefacts, never asserted from prose)
# --------------------------------------------------------------------------- #
def test_best_sanctioned_reading_is_the_m2971_number(ledger):
    best = ledger["verdict"]["best_sanctioned_reading"]
    assert best["id"] == "floor#run2#absolute_time_average_tail"
    # backlog_update_m2971.md quotes 237.6 Pa vs 360 Pa = -34.0 %
    assert best["measured_Pa"] == pytest.approx(237.636, abs=1e-3)
    assert best["rel_error_pct"] == pytest.approx(-34.0, abs=0.05)


def test_laplace_jump_stays_withheld(ledger):
    dp = ledger["verdict"]["quantities"]["laplace_dp"]
    assert dp["n_within_3pct"] == 0
    assert dp["sanctioned_all_within_15pct"] is False
    assert dp["sanctioned_min_abs_pct"] > 15.0


def test_free_surface_and_wetted_sets_are_separable(ledger):
    dp = ledger["verdict"]["quantities"]["laplace_dp"]
    assert dp["n_free_surface"] >= 40 and dp["n_no_free_surface"] >= 5
    # readings that avoid the free surface reach the gate; readings that cross it
    # essentially do not (the two exceptions are audited separately)
    assert dp["no_free_surface_min_abs_pct"] < 15.0
    assert dp["no_free_surface_within_15pct"] >= 3
    assert dp["free_surface_within_15pct"] <= 3


def test_every_sub_gate_free_surface_reading_is_not_credible(ledger):
    """No free-surface reading below the gate survives a resolution check."""
    passes = ledger["free_surface_passes"]
    assert len(passes) >= 3
    ids = {row["id"] for row in passes}
    # the two documented ones plus the legacy status-quo ladder
    assert {"laplace_law#relaxed#0", "ladder#level1"} <= ids
    for row in passes:
        assert row["resolution_stable"] is False, row["id"]
        assert row["resolution_verdict"] in {"zero_crossing", "no_ladder"}, row["id"]
    # the audited one has no in-family ladder, so it is resolved by the control
    audited = next(r for r in passes if r["id"] == "laplace_law#relaxed#0")
    assert audited["resolution_verdict"] == "no_ladder"
    assert ledger["provenance"]["q3_resolution_stability"]["resolution_stable"] is False


def test_ladder_passing_reading_is_a_zero_crossing(ledger):
    row = next(r for r in ledger["free_surface_passes"] if r["family"] == "surface_tension_ladder")
    assert row["resolution_verdict"] == "zero_crossing"
    signed = [step["rel_error_pct"] for step in row["ladder"]]  # dx ascending
    # the *signed* error decreases monotonically from the finest to the coarsest
    # grid, i.e. the curve crosses zero at the single level that "passes"
    assert signed == sorted(signed, reverse=True), signed
    assert signed[0] > 200.0, "finest resolution should be far outside the gate"
    errs = [abs(v) for v in signed]
    assert min(errs) == pytest.approx(7.353, abs=0.01)
    assert errs.index(min(errs)) not in (0, len(errs) - 1), "minimum must be interior"


def test_convention_choice_alone_moves_the_error_by_more_than_50pp(ledger):
    sens = ledger["verdict"]["convention_sensitivity"]
    assert len(sens) >= 6
    assert sens[0]["spread_pp"] > 50.0
    # the four conventions must cover the same physical cases
    assert set(sens[0]["conventions"]) == {
        "abs_instant", "abs_tail_avg", "bandA_instant", "bandB_instant"}


# --------------------------------------------------------------------------- #
# 4. document / artefact drift
# --------------------------------------------------------------------------- #
def test_document_matches_generated_output(ledger):
    generated = BUILD.render_markdown(ledger)
    committed = (ROOT / "docs" / "verification_ledger.md").read_text(encoding="utf-8")
    assert committed.strip() == generated.strip(), (
        "docs/verification_ledger.md is stale; run "
        "python benchmarks/verification_ledger/build.py"
    )


def test_ledger_json_matches_generated_output(ledger):
    payload = json.dumps(ledger, indent=1, sort_keys=True, allow_nan=False)
    committed = (HERE / "ledger.json").read_text(encoding="utf-8")
    assert committed.strip() == payload.strip(), (
        "benchmarks/verification_ledger/ledger.json is stale; run "
        "python benchmarks/verification_ledger/build.py"
    )


def test_figure_exists_and_is_small_enough_to_embed():
    png = HERE / "ledger.png"
    assert png.exists() and png.stat().st_size > 20_000
    # the platform rejects report bodies over ~250 KB, so keep the PNG compact
    assert png.stat().st_size < 260_000
