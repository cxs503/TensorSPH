"""Convergence analysis for XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION.

Reads the committed benchmark payloads (no simulation): ``grid_dx004`` /
``grid_dx003`` (per-variant round-1 runs) and ``ladder_dense`` (the four-level
ladder dx = 0.04 / 0.03 / 0.02 / 0.01 m), plus the generated ``convergence``
block of ``benchmarks/xval_sphinx_tvc/validation_report.json``.

What is asserted, in the order a reviewer needs it:

1. TVC beats the uncorrected baseline at both round-1 resolutions;
2. the *coarse* pair improves with refinement (dx = 0.04 -> 0.03);
3. the finer pairs **reverse** it: at fixed ``alpha = 0.10`` the ladder is
   non-monotone, its minimum sits at ``dx = 0.03`` m and the finest level is
   worse than the coarsest -- the "TVC converges onto the experiment" reading
   of the old best-rung-only report is falsified, and the report says so;
4. the ``convergence`` block is a pure function of the committed data
   (regenerating it produces no drift), so no number in the report can be
   hand-edited;
5. the verification ledger grades **every** rung, not the best one, and those
   grades obey the platform tolerance ladder (a rung is credited only when both
   the owner relative gate and the ledger absolute gate are met);
6. auxiliary invariants: mass drift 0, density compression < 5 %, bulk-only
   scope is worse than all-particle scope, alpha = 0.10 is the round-1 optimum.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "xval_sphinx_tvc" / "data"
REPORT = ROOT / "benchmarks" / "xval_sphinx_tvc" / "validation_report.json"
BUILDER = ROOT / "benchmarks" / "xval_sphinx_tvc" / "build_convergence_block.py"
LEDGER_BUILDER = ROOT / "benchmarks" / "verification_ledger" / "build_ledger.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _convergence() -> dict:
    return json.loads(REPORT.read_text())["convergence"]


def _ladder_record(dx: float) -> dict:
    """``ladder_dense`` record for the reference (TVC alpha=0.10) variant."""
    conv = _convergence()
    variant = conv["reference_variant"]["variant"]
    ladder = json.loads((DATA_DIR / "ladder_dense.json").read_text())
    for rec in ladder["records"]:
        if rec["variant"] == variant and abs(rec["dx"] - dx) < 1e-12:
            return rec
    raise KeyError(f"ladder_dense has no {variant} record at dx={dx}")


def _load(dx_str: str) -> dict:
    return json.loads((DATA_DIR / f"grid_dx{dx_str}.json").read_text())


def _mae(record: dict) -> float:
    return record["score"]["mae"]


def _rel(record: dict) -> float:
    return record["score"]["mean_rel_error"]


def _find(records: list, variant: str) -> dict:
    for r in records:
        if r["variant"] == variant:
            return r
    raise KeyError(f"Variant {variant} not found")


@pytest.fixture(scope="module")
def dx004():
    return _load("004")


@pytest.fixture(scope="module")
def dx003():
    return _load("003")


def test_tvc_beats_baseline_at_both_resolutions(dx004, dx003):
    """TVC α=0.10 must have lower MAE than baseline at both dx."""
    for d in [dx004, dx003]:
        baseline = _find(d["records"], "baseline")
        tvc = _find(d["records"], "tvc_a010_all")
        assert _mae(tvc) < _mae(baseline), (
            f"TVC MAE {_mae(tvc):.4f} >= baseline MAE {_mae(baseline):.4f} "
            f"at dx={d['records'][0]['dx']}"
        )


def test_tvc_improves_over_the_coarse_pair(dx004, dx003):
    """TVC α=0.10 MAE decreases from dx=0.04 to dx=0.03 (coarse pair only).

    This is the *only* pair that improves; see
    ``test_ladder_is_non_monotone_for_fixed_alpha`` for the reversal at
    dx = 0.02 / 0.01. Reading this test as "the closure converges" is exactly the
    over-claim the four-level ladder refutes.
    """
    mae_004 = _mae(_find(dx004["records"], "tvc_a010_all"))
    mae_003 = _mae(_find(dx003["records"], "tvc_a010_all"))
    assert mae_003 < mae_004, (
        f"TVC MAE did not improve with refinement: "
        f"dx=0.04 → {mae_004:.4f}, dx=0.03 → {mae_003:.4f}"
    )


def test_baseline_does_not_improve_enough(dx004, dx003):
    """Baseline MAE should also decrease but remain above the gate."""
    mae_004 = _mae(_find(dx004["records"], "baseline"))
    mae_003 = _mae(_find(dx003["records"], "baseline"))
    assert mae_003 > 0.15, (
        f"Baseline MAE at dx=0.03 is {mae_003:.4f}, unexpectedly below gate"
    )


def test_tvc_passes_gates_at_fine_resolution(dx003):
    """TVC α=0.10 at dx=0.03 must pass both owner gates."""
    tvc = _find(dx003["records"], "tvc_a010_all")
    assert tvc["score"]["gate_abs_pass"], (
        f"TVC absolute gate failed: MAE={_mae(tvc):.4f} >= 0.15"
    )
    assert tvc["score"]["gate_rel_pass"], (
        f"TVC relative gate failed: rel_err={_rel(tvc):.4f} >= 0.15"
    )


def test_alpha_optimum_consistent(dx004, dx003):
    """α=0.10 must be the best among tested alphas at both resolutions."""
    for d in [dx004, dx003]:
        variants = [r for r in d["records"]
                    if r["variant"].startswith("tvc_a")
                    and "_all" in r["variant"]
                    and "c5" not in r["variant"]]
        best = min(variants, key=_mae)
        assert best["variant"] == "tvc_a010_all", (
            f"Best alpha at dx={d['records'][0]['dx']} is "
            f"{best['variant']} (MAE={_mae(best):.4f}), not tvc_a010_all"
        )


def test_bulk_scope_loses_benefit(dx004, dx003):
    """Bulk-only scope must be worse than all-particles scope."""
    for d in [dx004, dx003]:
        bulk = _find(d["records"], "tvc_a005_bulk")
        all_p = _find(d["records"], "tvc_a005_all")
        assert _mae(bulk) > _mae(all_p), (
            f"Bulk scope MAE {_mae(bulk):.4f} <= all-scope MAE {_mae(all_p):.4f} "
            f"at dx={d['records'][0]['dx']}"
        )


def test_distribution_quality_improves(dx004):
    """TVC α=0.10 must reduce close-pair fraction and improve median nn."""
    baseline = _find(dx004["records"], "baseline")
    tvc = _find(dx004["records"], "tvc_a010_all")
    b_clump = baseline.get("clumping", {})
    t_clump = tvc.get("clumping", {})

    assert t_clump["close_pair_fraction"] < b_clump["close_pair_fraction"], (
        f"TVC close-pair frac {t_clump['close_pair_fraction']:.3f} >= "
        f"baseline {b_clump['close_pair_fraction']:.3f}"
    )
    assert t_clump["median_nn_over_dx"] > b_clump["median_nn_over_dx"], (
        f"TVC median nn {t_clump['median_nn_over_dx']:.3f} <= "
        f"baseline {b_clump['median_nn_over_dx']:.3f}"
    )
    assert abs(t_clump["median_nn_over_dx"] - 1.0) < abs(
        b_clump["median_nn_over_dx"] - 1.0
    ), "TVC did not improve median nn distance toward 1.0"


def test_mass_integrity_preserved(dx004, dx003):
    """TVC must not introduce mass drift."""
    for d in [dx004, dx003]:
        for r in d["records"]:
            assert r["mass_drift"] == 0.0, (
                f"Mass drift in {r['variant']}: {r['mass_drift']}"
            )


def test_density_compression_within_gate(dx004, dx003):
    """TVC must keep density compression within the 5% gate."""
    for d in [dx004, dx003]:
        tvc = _find(d["records"], "tvc_a010_all")
        assert tvc["max_density_compression"] < 0.05, (
            f"TVC density compression {tvc['max_density_compression']:.4f} >= 0.05 "
            f"at dx={d['records'][0]['dx']}"
        )
        if d["records"][0]["dx"] <= 0.03:
            baseline = _find(d["records"], "baseline")
            assert tvc["max_density_compression"] <= baseline["max_density_compression"], (
                f"TVC density compression {tvc['max_density_compression']:.4f} > "
                f"baseline {baseline['max_density_compression']:.4f} "
                f"at dx={d['records'][0]['dx']}"
            )

# ---------------------------------------------------------------------------
# the four-level ladder: what the report must say, and what it must not
# ---------------------------------------------------------------------------
def test_convergence_block_is_a_function_of_the_committed_data():
    """Regenerating ``validation_report.json::convergence`` must not drift."""
    mod = _load_module("build_convergence_block_test", BUILDER)
    assert _convergence() == mod.build(), (
        "convergence block drifted from its data sources: run "
        "`python benchmarks/xval_sphinx_tvc/build_convergence_block.py`"
    )


def test_ladder_is_non_monotone_for_fixed_alpha():
    """The honest negative result: fixed α=0.10 does *not* converge in dx."""
    conv = _convergence()
    summary = conv["ladder_summary"]
    assert summary["n_levels"] == 4
    assert summary["mae_monotone_under_refinement"] is False
    assert any(summary["mae_improves_at_each_refinement"]), (
        "the ladder summary reports no improving refinement step at all -- "
        "the coarse pair does improve, so this is a bookkeeping error"
    )
    assert summary["mae_improves_at_each_refinement"][0] is True   # 0.04 -> 0.03
    assert summary["finest_level_is_best"] is False
    assert summary["best_dx"] == 0.03
    assert summary["worst_dx"] == 0.01
    # the finest level is worse than *every* coarser one, not just than the best
    mae = {lvl["dx"]: lvl["variants"][conv["reference_variant"]["variant"]]["mae"]
           for lvl in conv["levels"]}
    assert mae[0.01] > max(mae[0.04], mae[0.02])
    assert conv["convergence_verdict"]["fixed_alpha_convergence"] == "falsified"


def test_only_the_best_rung_meets_both_gates():
    conv = _convergence()
    gates = conv["gates"]
    meeting = []
    for lvl in conv["levels"]:
        row = lvl["variants"][conv["reference_variant"]["variant"]]
        both = row["mae"] < gates["abs_mae"] and row["mean_rel_error"] < gates["rel"]
        assert lvl["tvc_meets_both_gates"] is both
        if both:
            meeting.append(lvl["dx"])
    assert meeting == [0.03]
    assert conv["ladder_summary"]["n_levels_meeting_both_gates"] == 1
    # dx = 0.04 passes the absolute gate but misses the owner relative gate:
    # that is why the ledger grades rungs on the worst of the two criteria.
    row04 = [lvl for lvl in conv["levels"] if lvl["dx"] == 0.04][0]["variants"]
    ref = row04[conv["reference_variant"]["variant"]]
    assert ref["gate_abs_pass"] is True and ref["gate_rel_pass"] is False


def test_self_convergence_reverses_on_the_finest_pair():
    """Cauchy self-convergence (no experimental input) must not be claimed."""
    conv = _convergence()
    variant = conv["reference_variant"]["variant"]
    block = conv["self_convergence"][variant]
    cauchy = [row["cauchy_l2"] for row in block["cauchy_l2"]]
    assert cauchy[-1] > cauchy[-2], (
        "the finest Cauchy pair shrinks in this data set: re-derive the "
        "self-convergence verdict instead of keeping the old one"
    )
    assert block["observed_order_l2"][-1]["observed_order_l2"] < 0
    assert block["richardson_limit"] is None     # no limit can be extrapolated
    assert block["in_asymptotic_range"] is False


def test_ladder_and_round1_payloads_describe_the_same_physics():
    """dx = 0.03 in the ladder must reproduce the round-1 grid payload."""
    ladder = _ladder_record(0.03)["score"]
    round1 = _find(_load("003")["records"], "tvc_a010_all")["score"]
    rel = abs(ladder["mae"] - round1["mae"]) / round1["mae"]
    assert rel < 0.01, (
        f"ladder dx=0.03 MAE {ladder['mae']:.6f} disagrees with the round-1 "
        f"grid payload {round1['mae']:.6f} by {rel * 100:.2f} %"
    )


# ---------------------------------------------------------------------------
# the verification ledger must grade every rung, not the best one
# ---------------------------------------------------------------------------
def test_ledger_grades_every_ladder_rung():
    """One ledger claim per rung, graded on the worst of the two criteria.

    Also the drift guard between the report and the ledger: the ledger's reading
    must equal ``max(MAE/0.15, mean_rel_error/0.15)`` computed from the report,
    and its grade must obey the platform ladder (PASS <= 3 %, MARGINAL <= 15 %,
    FAIL otherwise), so a rung cannot be credited by picking the kinder gate.
    """
    conv = _convergence()
    abs_gate = conv["gates"]["abs_mae"]
    rel_gate = conv["gates"]["rel"]
    variant = conv["reference_variant"]["variant"]
    ledger = _load_module("build_ledger_tvc_test", LEDGER_BUILDER).build()
    entries = {e["claim_id"]: e for e in ledger["entries"]}

    for lvl in conv["levels"]:
        dx = lvl["dx"]
        row = lvl["variants"][variant]
        claim_id = f"dam_break.tvc_alpha010.ladder_dx{dx:g}"
        assert claim_id in entries, f"rung dx={dx:g} is not graded by the ledger"
        e = entries[claim_id]
        worst = max(row["mae"] / abs_gate, row["mean_rel_error"] / rel_gate)
        assert abs(e["rel_error_pct"] - (worst - 1.0) * 100.0) < 1e-6
        assert e["gate_result"] == ("pass" if lvl["tvc_meets_both_gates"] else "fail")
        err = abs(e["rel_error_pct"])
        if e["grade"] == "PASS":
            assert err <= e["paper_criterion_pct"]
        elif e["grade"] == "MARGINAL":
            assert err <= e["criterion_pct"]
        else:
            assert e["grade"] == "FAIL" and err > e["paper_criterion_pct"]
        assert e["artifact"] == "benchmarks/xval_sphinx_tvc/validation_report.json"

    credited = [e["claim_id"] for e in entries.values()
                if e["claim_id"].startswith("dam_break.tvc_alpha010.ladder_dx")
                and e["grade"] in {"PASS", "MARGINAL"}]
    assert credited == ["dam_break.tvc_alpha010.ladder_dx0.03"], (
        f"the ledger credits {credited} instead of only the dx=0.03 rung"
    )

    summary = conv["ladder_summary"]
    cov = entries["dam_break.tvc_alpha010.ladder_gate_coverage"]
    assert abs(cov["measured"]["value"]
               - summary["n_levels_meeting_both_gates"] / summary["n_levels"]) < 1e-9
    assert cov["gate_result"] == "fail"

    variant_cauchy = conv["self_convergence"][variant]["cauchy_l2"]
    sc = entries["dam_break.tvc_alpha010.self_convergence"]
    expected = variant_cauchy[-1]["cauchy_l2"] / variant_cauchy[-2]["cauchy_l2"]
    assert abs(sc["measured"]["value"] - expected) < 1e-6
    assert sc["gate_result"] == ("fail" if expected >= 1.0 else "pass")
