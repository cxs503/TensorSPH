"""Contract tests for the phase-ensemble / refinement-ladder consistency audit
(m23321).

The round makes four claims about CASE-SPH-WEDGE-ENTRY-2D's adjudication:

1. the refinement ladder (dx = 4 / 2 / 1 / 0.5 mm) is **not** a convergence
   sequence -- its residual changes sign twice over 8x refinement;
2. the unperturbed (``scale = 0``-phase) lattice is a **>2 sigma outlier** of its
   own grid's phase ensemble, and in *opposite* directions on 4 mm and 2 mm;
3. therefore a single grid landing inside the +/-15 % gate has a large
   ``P_chance`` of doing so by luck, and its PASS is not evidence;
4. the metric's *typical* decision uncertainty exceeds the gate.

Claims 1-3 are only meaningful if the rules can also say "no".  So every rule is
a pure function and these tests pin it on synthetic inputs in **both**
directions:

* negative control A: a genuinely monotone, decaying ladder must come out
  ``monotone=True`` / ``gci_applicable=True``;
* negative control B: a symmetric lattice that really *is* central in its
  ensemble must come out ``is_outlier=False`` and ``direction_consistent`` must
  stay silent (no outliers = no direction claim);
* negative control C: a tight ensemble centred on zero must make a gate hit
  ``pass_is_evidence=True`` (a chance PASS is then unlikely);
* the arithmetic itself (sign changes, z, normal tail, median of the uncertainty
  pool) is checked against closed-form values.

The committed-artefact tests assert reproduction, not new physics: the same
peaks, the same same-operator truths, and the recorded decision.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
BENCH = HERE / "benchmarks/case_sph_wedge_entry_2d"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE / "src"))

import phase_ladder_consistency as plc  # noqa: E402

ART = HERE / "docs/artifacts/case_sph_wedge_entry_2d_phase_ladder/phase_ladder_consistency.json"


# --------------------------------------------------------------------------- #
# P1 -- refinement monotonicity
# --------------------------------------------------------------------------- #
def test_sign_changes_counts_only_real_crossings():
    assert plc.sign_changes([0.1, 0.2, 0.05]) == 0
    assert plc.sign_changes([0.4, -0.1, -0.05, 0.15]) == 2
    # zeros are skipped, not counted as a sign
    assert plc.sign_changes([0.1, 0.0, 0.2]) == 0
    assert plc.sign_changes([0.1, 0.0, -0.2]) == 1


def test_ladder_verdict_accepts_a_monotone_sequence():
    """Negative control A: a real convergence sequence must pass."""
    v = plc.ladder_verdict([0.40, 0.20, 0.10, 0.05])
    assert v["monotone"] is True
    assert v["gci_applicable"] is True
    assert v["n_sign_changes"] == 0
    assert v["abs_monotone_decreasing"] is True
    assert v["coarsest_to_finest_ratio"] == pytest.approx(8.0)


def test_ladder_verdict_rejects_a_sign_change():
    v = plc.ladder_verdict([0.46, -0.19, -0.12, 0.16])
    assert v["monotone"] is False
    assert v["gci_applicable"] is False
    assert v["n_sign_changes"] == 2
    assert v["finest_outside_gate"] is True


def test_ladder_verdict_rejects_non_decaying_abs():
    """Same sign but growing -> still not a convergence sequence."""
    v = plc.ladder_verdict([0.05, 0.10, 0.20])
    assert v["abs_monotone_decreasing"] is False
    assert v["monotone"] is False


def test_ladder_span_is_the_signed_excursion_in_gate_units():
    v = plc.ladder_verdict([0.46, -0.19], gate=0.15)
    assert v["span_pp"] == pytest.approx(65.0)          # max - min, not abs span
    assert v["span_over_gate"] == pytest.approx(0.65 / 0.15)


def test_ladder_verdict_needs_two_grids():
    with pytest.raises(ValueError):
        plc.ladder_verdict([0.1])


# --------------------------------------------------------------------------- #
# P2 -- is the unperturbed lattice representative?
# --------------------------------------------------------------------------- #
def test_ensemble_position_z_is_closed_form():
    pos = plc.ensemble_position(0.46, [0.26, 0.26, 0.09, 0.18])
    mean = (0.26 + 0.26 + 0.09 + 0.18) / 4
    sd = math.sqrt(sum((x - mean) ** 2 for x in [0.26, 0.26, 0.09, 0.18]) / 3)
    assert pos["mean"] == pytest.approx(mean)
    assert pos["sd"] == pytest.approx(sd)
    assert pos["z"] == pytest.approx((0.46 - mean) / sd)
    assert pos["is_outlier"] is True
    assert pos["direction"] == "high"


def test_ensemble_position_accepts_a_central_symmetric_sample():
    """Negative control B: a representative symmetric run is NOT an outlier."""
    pos = plc.ensemble_position(0.20, [0.18, 0.20, 0.22, 0.19])
    assert pos["is_outlier"] is False
    assert abs(pos["z"]) < 2.0
    assert pos["reject_sd_of_mean"] == pytest.approx(pos["sd"] / 2.0)


def test_ensemble_position_zero_spread_does_not_claim_an_outlier():
    """A degenerate ensemble (all draws equal) must not fabricate a z."""
    pos = plc.ensemble_position(0.20, [0.20, 0.20, 0.20])
    assert pos["sd"] == 0.0
    assert pos["is_outlier"] is False


def test_direction_consistency_fires_only_on_a_common_sign():
    assert plc.direction_consistency([1, 1])["systematically_high"] is True
    assert plc.direction_consistency([-1, -1])["systematically_low"] is True
    mixed = plc.direction_consistency([1, -1])
    assert mixed["direction_consistent"] is False
    assert mixed["systematically_high"] is False
    # no outlier at all -> no direction claim either way
    empty = plc.direction_consistency([])
    assert empty["direction_consistent"] is False
    assert empty["systematically_high"] is False


# --------------------------------------------------------------------------- #
# P3 -- chance-PASS probability
# --------------------------------------------------------------------------- #
def test_chance_pass_is_the_normal_gate_mass():
    draws = [-0.20, 0.0, 0.20, 0.10, -0.10]
    got = plc.chance_pass(draws, gate=0.15)
    mu = sum(draws) / len(draws)
    sd = math.sqrt(sum((x - mu) ** 2 for x in draws) / (len(draws) - 1))
    phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))  # noqa: E731
    want = phi((0.15 - mu) / sd) - phi((-0.15 - mu) / sd)
    assert got["p_chance"] == pytest.approx(want)
    assert got["mu"] == pytest.approx(mu)


def test_chance_pass_rejects_a_biased_ensemble():
    """Tight but biased outside the gate: hits are rare *and* uninformative."""
    got = plc.chance_pass([0.50, 0.51, 0.52, 0.49], gate=0.15)
    assert got["p_chance"] == pytest.approx(0.0, abs=1e-12)
    assert got["mean_ci95"][0] > 0.15          # the mean itself is outside
    assert got["pass_is_evidence"] is False


def test_chance_pass_blesses_a_tight_zero_centred_ensemble():
    """Negative control C: when scatter is tiny and centred, a PASS *is* evidence."""
    got = plc.chance_pass([0.00, 0.005, -0.005, 0.002], gate=0.15)
    assert got["p_chance"] == pytest.approx(1.0, abs=1e-9)
    assert got["pass_is_evidence"] is True
    assert -0.15 <= got["mean_ci95"][0] and got["mean_ci95"][1] <= 0.15


def test_chance_pass_flags_a_straddling_ensemble():
    """Wide spread around a mean near the gate -> a single hit is a coin flip."""
    got = plc.chance_pass([-0.05, 0.15, 0.35, 0.20], gate=0.15)
    assert 0.2 < got["p_chance"] < 0.9
    assert got["pass_is_evidence"] is False
    assert got["mean_ci_half_width"] > 0.05


def test_chance_pass_needs_three_draws():
    with pytest.raises(ValueError):
        plc.chance_pass([0.1, 0.2])


# --------------------------------------------------------------------------- #
# P4 -- optimistic floor vs typical resolution
# --------------------------------------------------------------------------- #
def test_honest_floor_separates_best_case_from_typical():
    got = plc.honest_floor([0.065, 0.235, 0.430], [0.063, 0.151])
    assert got["u_floor"] == pytest.approx(0.063)
    assert got["u_typical"] == pytest.approx(0.151)
    assert got["metric_can_meet_gate"] is True      # best case inside 15 %
    assert got["metric_resolution_stable"] is False  # typical case is not


def test_honest_floor_ignores_nan_inputs():
    got = plc.honest_floor([float("nan"), 0.30], [0.20])
    assert got["n_inputs"] == 2
    assert got["u_floor"] == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# committed artefacts -- reproduction, not new physics
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def committed() -> dict:
    if not ART.exists():
        pytest.skip("run phase_ladder_consistency.py to regenerate the artefact")
    return json.loads(ART.read_text())


def test_ladder_readings_match_the_regrade_artefact():
    reg = json.loads((HERE / "docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json").read_text())
    by_tag = {g["tag"]: g for g in reg["ladder_grids"]}
    com = reg["designation"]["reference_q_units"]
    desig = {g["tag"]: g for g in reg["designation"]["grids"]}
    for row in plc.load_ladder():
        g = by_tag[row["tag"]]
        assert row["obs_peak"] == pytest.approx(g["obs_peak"], rel=1e-12)
        assert row["synth_peak"] == pytest.approx(g["synth_peak"], rel=1e-12)
        # both residual definitions are checked against the committed artefact
        assert row["residual_same_operator"] == pytest.approx(g["residual"], rel=1e-12)
        assert row["residual_vs_com"] == pytest.approx(row["obs_peak"] / com - 1.0, rel=1e-12)
        if row["tag"] in desig:
            assert row["residual_vs_com"] == pytest.approx(
                desig[row["tag"]]["residual_rel"], rel=1e-12)


def test_committed_scale_zero_is_excluded_from_the_ensembles():
    """The A2 amplitude sweep must never leak into a phase ensemble."""
    reals = plc.load_realisations(scale=1.0)
    for tag, items in reals.items():
        assert items, tag
        assert all(p["file"].endswith(".json") for p in items)
    all_names = {p["file"] for v in reals.values() for p in v}
    assert not any("scale0" in n for n in all_names)
    with_quarter = plc.load_realisations(scale=0.25)
    assert any("scale0.25" in p["file"] for v in with_quarter.values() for p in v)


def test_committed_verdict(committed):
    v = committed["verdict"]
    assert v["ladder_is_convergence_sequence"] is False
    assert v["gci_applicable"] is False
    assert v["symmetric_lattice_representative"] is False
    assert v["symmetric_direction_consistent"] is False
    assert v["single_grid_pass_is_evidence"] is False
    assert v["decision"] == "NOT_DECIDABLE_BY_REFINEMENT_AT_MEASURED_RANGE"
    # honest-FAIL: the round must not hand out a credible PASS
    assert committed["closure"]["pass_credible"] is False


def test_committed_ensemble_outlier_directions_are_opposite(committed):
    g = committed["P2_per_grid"]
    z4 = g["dx4mm"]["ensemble"]["z"]
    z2 = g["dx2mm"]["ensemble"]["z"]
    assert z4 > 2.0 and z2 < -2.0

    assert committed["P2_direction"]["outlier_directions"] == [1, -1]


def test_committed_phase_means_agree_across_grids(committed):
    """Phase-averaging removes the phase noise: the two grid means must agree."""
    g = committed["P2_per_grid"]
    m4, m2 = g["dx4mm"]["ensemble"]["mean"], g["dx2mm"]["ensemble"]["mean"]
    assert abs(m4 - m2) < 0.05


def test_committed_chance_pass_is_not_evidence(committed):
    p = committed["P3_chance_pass"]
    assert p["n"] == 11
    assert p["p_chance"] > 0.05
    assert p["pass_is_evidence"] is False
    assert p["mean_ci95"][1] > 0.15


def test_doc_is_rendered_and_quotes_the_committed_decision(committed):
    doc = HERE / "docs/case_sph_wedge_entry_2d_phase_ladder_consistency_m23321.md"
    assert doc.exists()
    text = doc.read_text()
    assert committed["verdict"]["decision"] in text
    # six required sections
    for marker in ("① 案例与问题", "② 怎么做的", "③ 计算结果", "④ 图",
                   "⑤ 注意点", "⑥ 误差判定", "⑦ 结论"):
        assert marker in text, marker
    # no hand-typed numbers: the ladder peaks must appear as printed by the renderer
    for row in committed["ladder"]:
        assert f"{row['obs_peak']:.4f}" in text