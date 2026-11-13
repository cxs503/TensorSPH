"""Convergence analysis for XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION.

Uses the existing benchmark JSON data (dx=0.04, dx=0.03) to verify:
1. TVC improves over baseline at both resolutions
2. The improvement is monotone with refinement (for the best variant)
3. The optimal alpha is consistent across resolutions

This is a *data analysis* test — it reads pre-computed artifacts, not a
heavy simulation. The full convergence ladder (dx=0.02/0.01) requires
cluster execution and is tracked separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "xval_sphinx_tvc" / "data"


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


class TestTVCConvergence:
    """Convergence properties verified from existing benchmark data."""

    @pytest.fixture(scope="class")
    def dx004(self):
        return _load("004")

    @pytest.fixture(scope="class")
    def dx003(self):
        return _load("003")

    def test_tvc_beats_baseline_at_both_resolutions(self, dx004, dx003):
        """TVC α=0.10 must have lower MAE than baseline at both dx."""
        for d in [dx004, dx003]:
            baseline = _find(d["records"], "baseline")
            tvc = _find(d["records"], "tvc_a010_all")
            assert _mae(tvc) < _mae(baseline), (
                f"TVC MAE {_mae(tvc):.4f} >= baseline MAE {_mae(baseline):.4f} "
                f"at dx={d['records'][0]['dx']}"
            )

    def test_tvc_improves_with_refinement(self, dx004, dx003):
        """TVC α=0.10 MAE must decrease from dx=0.04 to dx=0.03."""
        mae_004 = _mae(_find(dx004["records"], "tvc_a010_all"))
        mae_003 = _mae(_find(dx003["records"], "tvc_a010_all"))
        assert mae_003 < mae_004, (
            f"TVC MAE did not improve with refinement: "
            f"dx=0.04 → {mae_004:.4f}, dx=0.03 → {mae_003:.4f}"
        )

    def test_baseline_does_not_improve_enough(self, dx004, dx003):
        """Baseline MAE should also decrease but remain above the gate."""
        mae_004 = _mae(_find(dx004["records"], "baseline"))
        mae_003 = _mae(_find(dx003["records"], "baseline"))
        # Baseline should improve slightly (or stay similar) but remain bad
        assert mae_003 > 0.15, (
            f"Baseline MAE at dx=0.03 is {mae_003:.4f}, unexpectedly below gate"
        )

    def test_tvc_passes_gates_at_fine_resolution(self, dx003):
        """TVC α=0.10 at dx=0.03 must pass both owner gates."""
        tvc = _find(dx003["records"], "tvc_a010_all")
        assert tvc["score"]["gate_abs_pass"], (
            f"TVC absolute gate failed: MAE={_mae(tvc):.4f} >= 0.15"
        )
        assert tvc["score"]["gate_rel_pass"], (
            f"TVC relative gate failed: rel_err={_rel(tvc):.4f} >= 0.15"
        )

    def test_alpha_optimum_consistent(self, dx004, dx003):
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

    def test_bulk_scope_loses_benefit(self, dx004, dx003):
        """Bulk-only scope must be worse than all-particles scope."""
        for d in [dx004, dx003]:
            bulk = _find(d["records"], "tvc_a005_bulk")
            all_p = _find(d["records"], "tvc_a005_all")
            assert _mae(bulk) > _mae(all_p), (
                f"Bulk scope MAE {_mae(bulk):.4f} <= all-scope MAE {_mae(all_p):.4f} "
                f"at dx={d['records'][0]['dx']}"
            )

    def test_distribution_quality_improves(self, dx004):
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
        # TVC should move median nn closer to 1.0 (perfect lattice)
        assert abs(t_clump["median_nn_over_dx"] - 1.0) < abs(
            b_clump["median_nn_over_dx"] - 1.0
        ), "TVC did not improve median nn distance toward 1.0"

    def test_mass_integrity_preserved(self, dx004, dx003):
        """TVC must not introduce mass drift."""
        for d in [dx004, dx003]:
            for r in d["records"]:
                assert r["mass_drift"] == 0.0, (
                    f"Mass drift in {r['variant']}: {r['mass_drift']}"
                )

    def test_density_compression_within_gate(self, dx004, dx003):
        """TVC must keep density compression within the 5% gate."""
        for d in [dx004, dx003]:
            tvc = _find(d["records"], "tvc_a010_all")
            assert tvc["max_density_compression"] < 0.05, (
                f"TVC density compression {tvc['max_density_compression']:.4f} >= 0.05 "
                f"at dx={d['records'][0]['dx']}"
            )
            # At the finer resolution, TVC should be better than baseline
            if d["records"][0]["dx"] <= 0.03:
                baseline = _find(d["records"], "baseline")
                assert tvc["max_density_compression"] <= baseline["max_density_compression"], (
                    f"TVC density compression {tvc['max_density_compression']:.4f} > "
                    f"baseline {baseline['max_density_compression']:.4f} "
                    f"at dx={d['records'][0]['dx']}"
                )
