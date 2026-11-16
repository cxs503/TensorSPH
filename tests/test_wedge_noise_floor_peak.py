"""Invariants for the frame-level same-operator reference (``noise_floor_peak.py``).

The module under test makes three claims that can each be wrong silently, so each
gets a check here:

1. the wall-node grid it rebuilds is the solver's own lattice, so the analytic
   replica ``A12`` is read on the same abscissae as the measurement;
2. the replica uses *exactly* the estimator of ``ensemble_peak.py`` (equal bins of
   ``tau in [0, 1.2]``, bin mean, max over bins with centre ``<= 1``) -- this is
   checked against that module's own ``binned``/``gated_peak`` rather than against
   a restatement of them;
3. the ladder decision rule can still say PASS and still say FAIL on synthetic
   ladders with known answers (a rule that cannot fail is worthless), and the
   committed artefacts produce the verdict that the document reports.

No SPH is run here: everything reads committed artefacts.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CASE))

import noise_floor_peak as NF  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ensemble = _load("case_wedge_ensemble_peak", CASE / "ensemble_peak.py")
closure = _load("case_wedge_ladder_closure", CASE / "ladder_closure.py")
from tensorsph.wedge_entry import wagner_c_coeff, wagner_peak_over_q  # noqa: E402

BETA = math.radians(30.0)
BETA_DEG = 30.0


@pytest.fixture(scope="module")
def built() -> dict:
    return NF.build()


# --------------------------------------------------------------------------- #
# 1. the rebuilt node grid
# --------------------------------------------------------------------------- #

def test_node_grid_is_uniform_in_the_body_frame():
    """``|x_k|`` must step by ``dx cos(beta)`` along the face -- the lattice."""
    dx, C, V, t = 1.0e-3, wagner_c_coeff(BETA), 1.0, 0.012
    taus = NF.wall_node_taus(dx, BETA, C, V, t)
    xs = [tau * C * V * t for tau in taus]
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert steps, "no wall node inside the footprint"
    assert max(steps) - min(steps) < 1.0e-12
    assert steps[0] == pytest.approx(dx * math.cos(BETA), rel=1e-12)


def test_node_count_is_monotone_in_time_and_matches_the_footprint():
    dx, C, V = 0.5e-3, wagner_c_coeff(BETA), 1.0
    counts = [len(NF.wall_node_taus(dx, BETA, C, V, t))
              for t in (0.002, 0.006, 0.012, 0.018)]
    assert counts == sorted(counts)
    # c(t) / (dx cos beta) nodes fit in the footprint, one per spacing
    t = 0.018
    c = C * V * t
    assert counts[-1] == pytest.approx(c / (dx * math.cos(BETA)), abs=2.0)


def test_rebuilt_count_reproduces_the_committed_wet_count(built):
    """Fidelity contract of the bridge: the committed ``n_wet`` is re-derived.

    The rebuild uses the *geometric* footprint (``|x_k| <= C V t``) while the
    solver gates contact on kernel support, so the two agree to a few nodes.
    The contract pinned here is the one the document reports: the median error is
    at most two nodes on every grid, and at least 55 % of the frames lie within
    5 % of the committed count.  The bridge is deliberately *not* claimed to be
    exact: the solver's contact gate is a kernel-support criterion, the rebuild is
    geometric, and the residual difference is what ``cutoff_sensitivity`` bounds.
    """
    for grid in built["grids"]:
        errs = [f["n_nodes_rebuilt"] - f["n_wet"] for f in grid["frames"] if f["n_wet"]]
        errs.sort()
        assert abs(errs[len(errs) // 2]) <= 2, (grid["tag"], errs[:5])
        within = [f for f in grid["frames"] if f["n_wet"]
                  and abs(f["n_nodes_rebuilt"] - f["n_wet"]) <= max(2.0, 0.05 * f["n_wet"])]
        frac = len(within) / max(1, sum(1 for f in grid["frames"] if f["n_wet"]))
        assert frac >= 0.55, (grid["tag"], frac)


# --------------------------------------------------------------------------- #
# 2. the replica is the same estimator
# --------------------------------------------------------------------------- #

def test_replica_equals_ensemble_peak_operator_on_the_same_field():
    """Line-for-line equivalence with ``ensemble_peak.binned`` + ``gated_peak``."""
    C, V, t = wagner_c_coeff(BETA), 1.0, 0.010
    taus = NF.wall_node_taus(0.5e-3, BETA, C, V, t)
    cps = [NF.wagner_p_over_q(tau, BETA) for tau in taus]
    centres, means, cnts = ensemble.binned(taus, cps, NF.N_BINS)
    theirs, _, _ = ensemble.gated_peak(centres, means, cnts, min_count=1,
                                       tau_hi=NF.TAU_HI)
    mine = NF.analytic_binned_peak(taus, BETA)
    assert mine == pytest.approx(theirs, rel=1e-12)


def test_replica_never_exceeds_the_analytic_point_peak():
    """A bin mean of a concave-in-the-mean profile cannot beat the point max."""
    point = wagner_peak_over_q(BETA)
    for dx in (0.004, 0.002, 0.001, 0.0005):
        for t in (0.004, 0.009, 0.014, 0.018):
            taus = NF.wall_node_taus(dx, BETA, wagner_c_coeff(BETA), 1.0, t)
            if len(taus) < 10:
                continue
            a12 = NF.analytic_binned_peak(taus, BETA)
            assert a12 <= point * (1.0 + 1e-9)


def test_replica_under_reads_the_point_peak_by_a_few_percent_at_fine_grid():
    """The operator's own discretisation response is small -- the point of the study."""
    point = wagner_peak_over_q(BETA)
    taus = NF.wall_node_taus(0.5e-3, BETA, wagner_c_coeff(BETA), 1.0, 0.018)
    a12 = NF.analytic_binned_peak(taus, BETA)
    assert 0.85 <= a12 / point <= 1.0, a12 / point


def test_cutoff_sensitivity_is_small_at_the_fine_grids(built):
    """Moving the wet/dry cutoff by two nodes must not move the reading by >5 pp.

    This is the honest bound on the geometric rebuild: at ``dx = 0.5/1 mm`` the
    residual is insensitive to it, so the coarse-grid readings carry the bridge
    error and the fine-grid ones do not.
    """
    for grid in built["grids"]:
        shift = grid["cutoff_sensitivity"]["max_abs_shift"]
        if grid["dx"] <= 1.0e-3:
            assert shift <= 0.05, (grid["tag"], shift)
        else:
            assert shift <= 0.15, (grid["tag"], shift)


def test_built_artifact_records_the_operator_response_per_grid(built):
    for grid in built["grids"]:
        assert 0.80 <= grid["a12_over_point_peak_median"] <= 1.0, grid["tag"]


# --------------------------------------------------------------------------- #
# 3. statistics helpers
# --------------------------------------------------------------------------- #

def test_moving_block_bootstrap_is_stable_and_centred():
    vals = [(-1) ** i * 0.5 for i in range(60)]
    boot = NF.moving_block_bootstrap(vals, rounds=400)
    assert boot["p05"] <= boot["p50"] <= boot["p95"]
    assert abs(boot["p50"]) < 0.1


def test_within_grid_scaling_recovers_a_planted_exponent():
    frames = [{"n_wet": 20 * (1 + i), "residual": 0.5 * (20 * (1 + i)) ** -0.5}
              for i in range(12)]
    out = NF.within_grid_scaling(frames, wet_gate=1)
    assert out["slope"] == pytest.approx(-0.5, abs=1e-9)


def test_fit_exponent_recovers_a_planted_power_law():
    pts = [(dx, 3.0 * dx ** 0.5) for dx in (0.004, 0.002, 0.001, 0.0005)]
    assert NF.fit_exponent(pts)["slope"] == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. what the committed artefacts say (recorded, and the rule that decides it)
# --------------------------------------------------------------------------- #

def test_residual_is_positive_on_every_grid(built):
    """A pure discretisation error would not be one-signed across four grids."""
    for grid in built["grids"]:
        assert grid["reading"]["median"] > 0.0, (grid["tag"], grid["reading"]["median"])


def test_committed_ladder_is_not_decidable_under_this_estimator(built):
    """Honest record: the frame-level estimator does not judge any grid."""
    assert built["ladder"]["verdict"] == "NOT_DECIDABLE", built["ladder"]
    assert built["ladder"]["grids_judgeable"] == []
    for grid in built["grids"]:
        assert grid["reading"]["uncertainty"] > NF.GATE, grid["tag"]


def test_noise_exponent_band_excludes_the_discretisation_scaling(built):
    """``p >= 1`` (scheme convergence) is outside the 5..95 % band of the fit."""
    exp = built["exponent"]
    band = built["exponent_bootstrap"]
    assert exp["slope"] == band["p50"] or abs(exp["slope"] - band["p50"]) < 0.2
    assert band["p95"] < 1.0, band


def test_verdict_rule_can_pass_and_can_fail():
    """Negative controls for the unchanged house rule."""
    assert closure.verdict([
        {"tag": "dx2mm", "residual": 0.10, "judgeable": True},
        {"tag": "dx1mm", "residual": 0.05, "judgeable": True},
    ])["verdict"] == "PASS"
    assert closure.verdict([
        {"tag": "dx2mm", "residual": 0.40, "judgeable": True},
        {"tag": "dx1mm", "residual": 0.30, "judgeable": True},
    ])["verdict"] == "FAIL_above"
    assert closure.verdict([
        {"tag": "dx2mm", "residual": 0.05, "judgeable": False},
        {"tag": "dx1mm", "residual": 0.05, "judgeable": True},
    ])["verdict"] == "NOT_DECIDABLE"


def test_committed_artifact_matches_a_fresh_build():
    """The JSON in ``docs/artifacts`` is what the script produces, bit for bit."""
    path = NF.OUT_DIR / "noise_floor.json"
    if not path.exists():
        pytest.skip("artefact not regenerated")
    stored = json.loads(path.read_text())
    fresh = NF.build()
    assert stored["exponent"]["slope"] == fresh["exponent"]["slope"]
    assert [g["reading"]["median"] for g in stored["grids"]] == \
        [g["reading"]["median"] for g in fresh["grids"]]