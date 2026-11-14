"""Full resolution ladder validation for XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION.

This test suite validates the complete four-level resolution ladder
(dx = 0.04, 0.03, 0.02, 0.01 m) and the alpha sweep at dx = 0.01,
capturing both the positive findings (TVC beats baseline at every
resolution, owner standard met at dx = 0.03/0.02) and the negative
findings (fixed-alpha convergence is non-monotone, the constant-rate
scaling hypothesis is falsified).

Data sources (all committed under docs/xval_sphinx_tvc/data/):
- ladder_dense.json: four-level ladder, baseline + TVC alpha=0.10
- sweep_dx001_alpha.json: alpha sweep at dx=0.01 (0.15/0.30)
- convergence_ladder.json: Cauchy differences, observed orders, Richardson
- distribution_ladder.json: particle distribution metrics at three levels
- grid_dx004.json, grid_dx003.json: round-1 per-variant payloads

Gate: owner standard, mean relative error < 15 % (OWNER_GATE_REL),
and the ledger absolute gate MAE < 0.15 in Z/a.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "xval_sphinx_tvc" / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def _find(records: list, variant: str) -> dict:
    for r in records:
        if r["variant"] == variant:
            return r
    raise KeyError(f"Variant {variant} not found in records")


def _mae(record: dict) -> float:
    return record["score"]["mae"]


def _rel(record: dict) -> float:
    return record["score"]["mean_rel_error"]


# ---------------------------------------------------------------------------
# Fixtures: load all data once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ladder():
    return _load("ladder_dense.json")


@pytest.fixture(scope="module")
def sweep():
    return _load("sweep_dx001_alpha.json")


@pytest.fixture(scope="module")
def convergence():
    return _load("convergence_ladder.json")


@pytest.fixture(scope="module")
def distribution():
    return _load("distribution_ladder.json")


@pytest.fixture(scope="module")
def dx004():
    return _load("grid_dx004.json")


@pytest.fixture(scope="module")
def dx003():
    return _load("grid_dx003.json")


# ---------------------------------------------------------------------------
# 1. Full ladder: TVC beats baseline at every resolution
# ---------------------------------------------------------------------------

class TestTvcBeatsBaselineEverywhere:
    """TVC alpha=0.10 must have lower MAE than baseline at all four levels."""

    @pytest.mark.parametrize("dx", [0.04, 0.03, 0.02, 0.01])
    def test_tvc_mae_below_baseline(self, ladder, dx):
        b = [r for r in ladder["records"]
             if r["variant"] == "baseline" and r["dx"] == dx][0]
        t = [r for r in ladder["records"]
             if r["variant"] == "tvc_a010_all" and r["dx"] == dx][0]
        assert _mae(t) < _mae(b), (
            f"TVC MAE {_mae(t):.4f} >= baseline MAE {_mae(b):.4f} at dx={dx}"
        )

    @pytest.mark.parametrize("dx", [0.04, 0.03, 0.02, 0.01])
    def test_tvc_rel_below_baseline(self, ladder, dx):
        b = [r for r in ladder["records"]
             if r["variant"] == "baseline" and r["dx"] == dx][0]
        t = [r for r in ladder["records"]
             if r["variant"] == "tvc_a010_all" and r["dx"] == dx][0]
        assert _rel(t) < _rel(b), (
            f"TVC rel {_rel(t):.4f} >= baseline rel {_rel(b):.4f} at dx={dx}"
        )

    @pytest.mark.parametrize("dx", [0.04, 0.03, 0.02, 0.01])
    def test_improvement_factor(self, ladder, dx):
        """TVC must improve by at least 2x at every resolution."""
        b = [r for r in ladder["records"]
             if r["variant"] == "baseline" and r["dx"] == dx][0]
        t = [r for r in ladder["records"]
             if r["variant"] == "tvc_a010_all" and r["dx"] == dx][0]
        factor = _mae(b) / _mae(t)
        assert factor >= 2.0, (
            f"TVC improvement factor {factor:.2f}x < 2.0x at dx={dx}"
        )


# ---------------------------------------------------------------------------
# 2. Owner gate: which levels pass?
# ---------------------------------------------------------------------------

class TestOwnerGate:
    """Owner standard: mean relative error < 15 %, MAE < 0.15."""

    @pytest.mark.parametrize("dx,expected_pass", [
        (0.04, False),   # 17.1 % — marginal fail
        (0.03, True),    # 10.6 % — pass
        (0.02, True),    # 12.5 % — pass
        (0.01, False),   # 16.4 % — fail (fixed alpha)
    ])
    def test_rel_gate(self, ladder, dx, expected_pass):
        t = [r for r in ladder["records"]
             if r["variant"] == "tvc_a010_all" and r["dx"] == dx][0]
        passes = _rel(t) < 0.15
        assert passes == expected_pass, (
            f"TVC rel gate at dx={dx}: got {passes}, expected {expected_pass} "
            f"(rel={_rel(t):.4f})"
        )

    @pytest.mark.parametrize("dx,expected_pass", [
        (0.04, True),    # MAE 0.147 < 0.15
        (0.03, True),    # MAE 0.142 < 0.15
        (0.02, False),   # MAE 0.156 > 0.15
        (0.01, False),   # MAE 0.280 > 0.15
    ])
    def test_abs_gate(self, ladder, dx, expected_pass):
        t = [r for r in ladder["records"]
             if r["variant"] == "tvc_a010_all" and r["dx"] == dx][0]
        passes = _mae(t) < 0.15
        assert passes == expected_pass, (
            f"TVC abs gate at dx={dx}: got {passes}, expected {expected_pass} "
            f"(MAE={_mae(t):.4f})"
        )


# ---------------------------------------------------------------------------
# 3. Non-monotone convergence (the falsified claim)
# ---------------------------------------------------------------------------

class TestNonMonotoneConvergence:
    """The 'improves convergence' claim is falsified for fixed alpha.

    At fixed alpha=0.10, the error does NOT decrease monotonically:
    dx=0.04 -> 0.03: improves (0.147 -> 0.142)
    dx=0.03 -> 0.02: degrades slightly (0.142 -> 0.156)
    dx=0.02 -> 0.01: degrades significantly (0.156 -> 0.280)
    """

    def test_error_increases_at_finest_level(self, ladder):
        """MAE at dx=0.01 must be HIGHER than at dx=0.02 (non-monotone)."""
        t_002 = [r for r in ladder["records"]
                 if r["variant"] == "tvc_a010_all" and r["dx"] == 0.02][0]
        t_001 = [r for r in ladder["records"]
                 if r["variant"] == "tvc_a010_all" and r["dx"] == 0.01][0]
        assert _mae(t_001) > _mae(t_002), (
            f"Non-monotone convergence not confirmed: "
            f"dx=0.01 MAE {_mae(t_001):.4f} <= dx=0.02 MAE {_mae(t_002):.4f}"
        )

    def test_rel_error_increases_at_finest_level(self, ladder):
        """Relative error at dx=0.01 must be HIGHER than at dx=0.02."""
        t_002 = [r for r in ladder["records"]
                 if r["variant"] == "tvc_a010_all" and r["dx"] == 0.02][0]
        t_001 = [r for r in ladder["records"]
                 if r["variant"] == "tvc_a010_all" and r["dx"] == 0.01][0]
        assert _rel(t_001) > _rel(t_002), (
            f"Non-monotone convergence not confirmed (rel): "
            f"dx=0.01 rel {_rel(t_001):.4f} <= dx=0.02 rel {_rel(t_002):.4f}"
        )

    def test_baseline_also_non_monotone(self, ladder):
        """Baseline is also non-monotone (model error dominates)."""
        b_004 = [r for r in ladder["records"]
                 if r["variant"] == "baseline" and r["dx"] == 0.04][0]
        b_003 = [r for r in ladder["records"]
                 if r["variant"] == "baseline" and r["dx"] == 0.03][0]
        # Baseline gets worse from 0.04 to 0.03
        assert _mae(b_003) > _mae(b_004), (
            f"Baseline non-monotone not confirmed: "
            f"dx=0.03 MAE {_mae(b_003):.4f} <= dx=0.04 MAE {_mae(b_004):.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Alpha sweep at dx=0.01
# ---------------------------------------------------------------------------

class TestAlphaSweepDx001:
    """Alpha sweep at dx=0.01: optimum moves up but not by dx^-1.

    The sweep file has tvc_a015_all and tvc_a030_all.
    The tvc_a010_all at dx=0.01 comes from the ladder file.
    """

    def _tvc_a010_at_001(self, ladder):
        return [r for r in ladder["records"]
                if r["variant"] == "tvc_a010_all" and r["dx"] == 0.01][0]

    def test_alpha_015_beats_alpha_010(self, sweep, ladder):
        """alpha=0.15 must beat alpha=0.10 at dx=0.01."""
        a010 = self._tvc_a010_at_001(ladder)
        a015 = _find(sweep["records"], "tvc_a015_all")
        assert _mae(a015) < _mae(a010), (
            f"alpha=0.15 MAE {_mae(a015):.4f} >= alpha=0.10 MAE {_mae(a010):.4f}"
        )

    def test_alpha_015_passes_rel_gate(self, sweep):
        """alpha=0.15 must pass the 15 % relative gate at dx=0.01."""
        a015 = _find(sweep["records"], "tvc_a015_all")
        assert _rel(a015) < 0.15, (
            f"alpha=0.15 rel {_rel(a015):.4f} >= 0.15 gate"
        )

    def test_alpha_030_degrades(self, sweep):
        """alpha=0.30 (constant-rate scaling) must be worse than alpha=0.15."""
        a015 = _find(sweep["records"], "tvc_a015_all")
        a030 = _find(sweep["records"], "tvc_a030_all")
        assert _mae(a030) > _mae(a015), (
            f"alpha=0.30 MAE {_mae(a030):.4f} <= alpha=0.15 MAE {_mae(a015):.4f} "
            f"(constant-rate scaling hypothesis falsified)"
        )

    def test_alpha_030_still_better_than_baseline(self, sweep, ladder):
        """alpha=0.30 is worse than alpha=0.15 but still better than baseline.

        This corrects an earlier claim in RESOLUTION_LADDER.md that alpha=0.30
        "loses more than the baseline gains" — the data shows MAE=0.410 vs
        baseline MAE=0.572, so the correction still helps, just not as much
        as the optimum.
        """
        a030 = _find(sweep["records"], "tvc_a030_all")
        b = [r for r in ladder["records"]
             if r["variant"] == "baseline" and r["dx"] == 0.01][0]
        assert _mae(a030) < _mae(b), (
            f"alpha=0.30 MAE {_mae(a030):.4f} >= baseline MAE {_mae(b):.4f} "
            f"(overshoot should still be better than baseline)"
        )

    def test_optimum_scaling_exponent(self, sweep):
        """The optimum alpha moves by ~1.5x over 3x refinement, not 3x.

        alpha_opt ~ dx^-0.37, not dx^-1.
        At dx=0.03: alpha_opt = 0.10
        At dx=0.01: alpha_opt = 0.15
        Ratio: 1.5x over 3x refinement -> exponent = log(1.5)/log(3) ≈ 0.37
        """
        # The optimum at dx=0.01 is 0.15, at dx=0.03 it was 0.10
        exponent = math.log(0.15 / 0.10) / math.log(0.03 / 0.01)
        assert 0.2 < exponent < 0.6, (
            f"Optimum scaling exponent {exponent:.3f} outside [0.2, 0.6] "
            f"(expected ~0.37, not 1.0)"
        )


# ---------------------------------------------------------------------------
# 5. Convergence ladder: Cauchy differences and observed orders
# ---------------------------------------------------------------------------

class TestConvergenceLadder:
    """Validate the Cauchy self-convergence analysis."""

    def test_tvc_second_order_at_003_002(self, convergence):
        """TVC shows ~2nd order between dx=0.03 and dx=0.02."""
        tvc = convergence["variants"]["tvc_a010_all"]
        order_003_002 = tvc["orders"][0]["observed_order_l2"]
        assert 1.5 < order_003_002 < 2.5, (
            f"TVC observed order {order_003_002:.2f} not in [1.5, 2.5] "
            f"for dx=0.03->0.02 interval"
        )

    def test_tvc_order_reverses_at_002_001(self, convergence):
        """TVC order reverses (negative) between dx=0.02 and dx=0.01."""
        tvc = convergence["variants"]["tvc_a010_all"]
        order_002_001 = tvc["orders"][1]["observed_order_l2"]
        assert order_002_001 < 0, (
            f"TVC observed order {order_002_001:.2f} not negative "
            f"for dx=0.02->0.01 interval (reversal not confirmed)"
        )

    def test_baseline_order_at_finest_pair(self, convergence):
        """Baseline shows ~1.8 order between dx=0.02 and dx=0.01."""
        baseline = convergence["variants"]["baseline"]
        order_002_001 = baseline["orders"][1]["observed_order_l2"]
        assert 1.0 < order_002_001 < 2.5, (
            f"Baseline observed order {order_002_001:.2f} not in [1.0, 2.5] "
            f"for dx=0.02->0.01 interval"
        )

    def test_richardson_limit_still_far_from_experiment(self, convergence):
        """Richardson extrapolation of baseline is still 37.6 % from experiment."""
        baseline = convergence["variants"]["baseline"]
        rich = baseline["richardson"]
        assert rich is not None
        assert rich["vs_experiment"]["mean_rel_error"] > 0.30, (
            f"Richardson limit rel error {rich['vs_experiment']['mean_rel_error']:.4f} "
            f"not > 0.30 (model error dominates)"
        )

    def test_tvc_richardson_not_extrapolable(self, convergence):
        """TVC Richardson extrapolation is null (order < 0)."""
        tvc = convergence["variants"]["tvc_a010_all"]
        assert tvc["richardson"] is None, (
            "TVC Richardson should be None (order < 0, not extrapolable)"
        )


# ---------------------------------------------------------------------------
# 6. Distribution ladder validation
# ---------------------------------------------------------------------------

class TestDistributionLadder:
    """Validate the particle distribution metrics across resolutions."""

    def test_tvc_reduces_close_pairs_at_dx004(self, distribution):
        """TVC must reduce close-pair fraction at dx=0.04."""
        dx004_records = [r for r in distribution["records"] if r["dx"] == 0.04]
        baseline = _find(dx004_records, "baseline")
        tvc = _find(dx004_records, "tvc_a010_all")
        tvc_cpf = tvc["metrics"]["close_pair_fraction"]
        baseline_cpf = baseline["metrics"]["close_pair_fraction"]
        assert tvc_cpf < baseline_cpf, (
            f"TVC close-pair frac {tvc_cpf:.3f} >= baseline {baseline_cpf:.3f}"
        )

    def test_tvc_improves_median_nn_at_dx004(self, distribution):
        """TVC must improve median nn/dx toward 1.0 at dx=0.04."""
        dx004_records = [r for r in distribution["records"] if r["dx"] == 0.04]
        baseline = _find(dx004_records, "baseline")
        tvc = _find(dx004_records, "tvc_a010_all")
        tvc_nn = tvc["metrics"]["nn_median_over_dx"]
        baseline_nn = baseline["metrics"]["nn_median_over_dx"]
        tvc_dev = abs(tvc_nn - 1.0)
        baseline_dev = abs(baseline_nn - 1.0)
        assert tvc_dev < baseline_dev, (
            f"TVC median nn deviation {tvc_dev:.3f} >= baseline {baseline_dev:.3f}"
        )

    def test_distribution_ladder_has_three_levels(self, distribution):
        """Distribution ladder must have data at dx=0.04, 0.02, 0.01."""
        dx_values = set(r["dx"] for r in distribution["records"])
        assert 0.04 in dx_values
        assert 0.02 in dx_values
        assert 0.01 in dx_values


# ---------------------------------------------------------------------------
# 7. Mass integrity and density compression
# ---------------------------------------------------------------------------

class TestPhysicalIntegrity:
    """TVC must not introduce mass drift or excessive density compression."""

    @pytest.mark.parametrize("dx", [0.04, 0.03, 0.02, 0.01])
    def test_no_mass_drift(self, ladder, dx):
        for variant in ["baseline", "tvc_a010_all"]:
            r = [rec for rec in ladder["records"]
                 if rec["variant"] == variant and rec["dx"] == dx][0]
            assert r["mass_drift"] == 0.0, (
                f"Mass drift in {variant} at dx={dx}: {r['mass_drift']}"
            )

    @pytest.mark.parametrize("dx", [0.04, 0.03, 0.02, 0.01])
    def test_density_compression_within_gate(self, ladder, dx):
        t = [r for r in ladder["records"]
             if r["variant"] == "tvc_a010_all" and r["dx"] == dx][0]
        assert t["max_density_compression"] < 0.05, (
            f"TVC density compression {t['max_density_compression']:.4f} >= 0.05 "
            f"at dx={dx}"
        )


# ---------------------------------------------------------------------------
# 8. Cross-consistency: ladder_dense vs grid_dx004/grid_dx003
# ---------------------------------------------------------------------------

class TestCrossConsistency:
    """Ensure the ladder data is consistent with the round-1 payloads.

    Note: the ladder and grid files were produced by separate runs with
    different thread counts, so we allow a small tolerance (0.2 %) rather
    than bit-identical matching.
    """

    def test_dx004_baseline_mae_consistent(self, ladder, dx004):
        """Baseline MAE at dx=0.04 must be within 0.2 % between ladder and grid."""
        ladder_mae = [r for r in ladder["records"]
                      if r["variant"] == "baseline" and r["dx"] == 0.04][0]["score"]["mae"]
        grid_mae = _find(dx004["records"], "baseline")["score"]["mae"]
        rel_diff = abs(ladder_mae - grid_mae) / grid_mae
        assert rel_diff < 0.002, (
            f"Baseline MAE mismatch at dx=0.04: ladder={ladder_mae}, grid={grid_mae} "
            f"(rel_diff={rel_diff:.4e})"
        )

    def test_dx003_tvc_mae_consistent(self, ladder, dx003):
        """TVC MAE at dx=0.03 must be within 0.2 % between ladder and grid."""
        ladder_mae = [r for r in ladder["records"]
                      if r["variant"] == "tvc_a010_all" and r["dx"] == 0.03][0]["score"]["mae"]
        grid_mae = _find(dx003["records"], "tvc_a010_all")["score"]["mae"]
        rel_diff = abs(ladder_mae - grid_mae) / grid_mae
        assert rel_diff < 0.002, (
            f"TVC MAE mismatch at dx=0.03: ladder={ladder_mae}, grid={grid_mae} "
            f"(rel_diff={rel_diff:.4e})"
        )
