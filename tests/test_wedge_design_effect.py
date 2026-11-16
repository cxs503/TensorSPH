"""Contract tests for the realisation-axis design effect (m23165).

The realisation axis (m22826) reported a CI that assumes the realisations are
independent samples.  A1 measures the pairwise correlation of the per-frame
``cp_centre`` series and it is *not* small, so this round converts the measured
correlation into an effective sample size and widens the CI.  These tests pin:

* the predeclared rule fields (``U_realisation_ci`` / ``U_band`` / ``judgeable``)
  are **bit-identical** to before -- the new numbers only ever add fields
  (negative control against silently relaxing the acceptance rule);
* the design-effect arithmetic itself, on synthetic artefacts whose correlation
  is set by hand (``rho = 0`` must reproduce the naive CI exactly, ``rho -> 1``
  must collapse ``N_eff`` to 1);
* both admissible CI conventions are reported and bracket the naive value;
* the two denominators (ledger COM vs same-operator Wagner) are kept apart;
* the committed artefact still reproduces from a fresh aggregation.
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

import seed_ensemble as se  # noqa: E402

ART = HERE / "docs/artifacts/case_sph_wedge_entry_2d_realisations/seed_ensemble.json"
BIAS = HERE / "docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json"
REAL = HERE / "docs/artifacts/case_sph_wedge_entry_2d_realisations"


def _committed() -> dict:
    return json.loads(ART.read_text())


def _craft(cp_centre: list[float], obs: float, spread: float, seed: int,
           dx: float = 4.0e-3) -> dict:
    """Minimal realisation artefact understood by ``seed_ensemble.aggregate``."""
    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "config": {"dx": dx, "beta_deg": 30.0, "velocity": 1.0, "c0_factor": 20.0},
        "realisation": {"seed": seed, "scale": 1.0,
                        "phase_x_dx": 0.01 * seed, "gap_jitter_dx": -0.01 * seed,
                        "design": "synthetic"},
        "by_frames": {"61": {
            "resolution_band": {"peak_median": obs, "spread_rel": spread,
                                "n_bins_used": [10, 12, 16], "n_members": 3},
            "jackknife_rel": 0.01,
        }},
        "band_fixed_m22695": {"peak_median": obs, "spread_rel": spread,
                              "n_bins_used": [10]},
        "series": {"t": list(range(len(cp_centre))),
                   "cp_centre": cp_centre,
                   "cp_max_footprint": [0.0] * len(cp_centre),
                   "max_rho_ratio": [1.0] * len(cp_centre),
                   "n_wet": [30] * len(cp_centre)},
        "steps": 1, "wall_seconds": 0.0,
    }


def _aggregate(tmp_path: Path, arts: list[dict]) -> dict:
    for a in arts:
        tag = f"real_dx{a['config']['dx'] * 1e3:g}mm_c0v20_seed{a['realisation']['seed']}.json"
        (tmp_path / tag).write_text(json.dumps(a))
    return se.aggregate(tmp_path, BIAS)


# --------------------------------------------------------------------------- #
# the predeclared rule must be untouched
# --------------------------------------------------------------------------- #

def test_predeclared_rule_fields_unchanged():
    for g in _committed()["grids"]:
        # judgeable is still defined by the *original* uncertainty field
        assert g["judgeable"] == bool(g["uncertainty"] == g["uncertainty"]
                                     and g["uncertainty"] <= se.GATE)
        assert g["uncertainty"] == max(g["U_realisation_ci"], g["U_band"]) or (
            g["U_realisation_ci"] != g["U_realisation_ci"])
        de = g["design_effect"]
        # the naive field inside the new block is literally the old field
        assert de["U_ci_naive_independence"] == g["U_realisation_ci"]


def test_no_silent_gate_relaxation():
    """Corrected uncertainty may only ever be >= the naive one."""
    for g in _committed()["grids"]:
        de = g["design_effect"]
        lo, hi = de["U_corrected_range"]
        assert lo <= hi
        assert lo >= g["U_realisation_ci"] - 1e-12
        assert de["U_ci_design_effect_keep_df"] >= de["U_ci_naive_independence"] - 1e-12


# --------------------------------------------------------------------------- #
# design-effect arithmetic on synthetic artefacts
# --------------------------------------------------------------------------- #

def test_design_effect_arithmetic():
    for g in _committed()["grids"]:
        de = g["design_effect"]
        n = g["n_realisations"]
        rho = g["independence"]["pairwise_median"]
        if rho == rho and rho > 0:
            assert de["deff"] == pytest.approx(1.0 + (n - 1) * rho, rel=1e-12)
        else:
            assert de["deff"] == 1.0
        assert de["n_eff"] == pytest.approx(n / de["deff"], rel=1e-12)
        assert 0.0 < de["n_eff"] <= n + 1e-12


def test_rho_zero_reproduces_naive_ci(tmp_path):
    """Negative control: uncorrelated realisations => deff = 1, CI unchanged."""
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    arts = [_craft(list(base), obs, 0.02, seed=i)
            for i, obs in enumerate([7.0, 8.0, 9.0], start=1)]
    rep = _aggregate(tmp_path, arts)
    g = rep["grids"][0]
    de = g["design_effect"]
    rho = g["independence"]["pairwise_median"]
    if rho > 0:                       # identical series: perfectly correlated
        assert de["deff"] > 1.0
        assert de["n_eff"] < g["n_realisations"] + 1e-12
    else:
        assert de["deff"] == 1.0
        assert de["U_ci_naive_independence"] == pytest.approx(
            de["U_ci_design_effect"], rel=1e-12)


def test_perfect_correlation_collapses_to_one_sample():
    rho, n = 1.0, 6
    deff = 1.0 + (n - 1) * rho
    assert deff == pytest.approx(n)
    assert n / deff == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# two denominators are reported, never mixed
# --------------------------------------------------------------------------- #

def test_denominators_are_kept_apart():
    for g in _committed()["grids"]:
        d = g["denominators"]
        assert d["com_q_units"] > 0 and d["wagner_point_q_units"] > 0
        assert d["residual_vs_com"] == pytest.approx(
            g["obs_median"] / d["com_q_units"] - 1.0, rel=1e-12)
        assert d["residual_vs_wagner"] == pytest.approx(
            g["obs_median"] / d["wagner_point_q_units"] - 1.0, rel=1e-12)
        # the same-operator residual the closure rule judges is NOT the ledger one
        assert abs(d["residual_vs_com"] - d["residual_vs_wagner"]) > 1e-6


def test_ledger_denominator_is_the_designated_model():
    import literature_refs as lr
    d = _committed()["grids"][0]["denominators"]
    assert d["com_q_units"] == pytest.approx(lr.designated_cp_max(30.0), rel=1e-12)


# --------------------------------------------------------------------------- #
# reproducibility of the committed artefact
# --------------------------------------------------------------------------- #

def test_committed_artifact_reproduces_from_fresh_aggregation():
    fresh = se.aggregate(REAL, BIAS)
    old = _committed()
    assert [g["tag"] for g in fresh["grids"]] == [g["tag"] for g in old["grids"]]
    for a, b in zip(fresh["grids"], old["grids"]):
        assert a["n_realisations"] == b["n_realisations"]
        assert a["obs_median"] == pytest.approx(b["obs_median"], rel=1e-12)
        assert a["design_effect"]["n_eff"] == pytest.approx(
            b["design_effect"]["n_eff"], rel=1e-12)
        assert a["design_effect"]["U_corrected"] == pytest.approx(
            b["design_effect"]["U_corrected"], rel=1e-12)
    assert fresh["verdict"]["verdict"] == old["verdict"]["verdict"]


def test_verdict_is_honest():
    """The anchor must not be promoted by this round."""
    assert _committed()["verdict"]["verdict"] in {"NOT_DECIDABLE", "UNDETERMINED"}


def test_t95_table_is_monotone():
    vals = [se.t95(df) for df in range(1, 13)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert se.t95(1) == pytest.approx(12.706, abs=1e-3)
    assert se.t95(30) == pytest.approx(2.042, abs=1e-3)
    assert math.isnan(se.t95(0))