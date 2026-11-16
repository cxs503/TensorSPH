"""Failure test: the marched projection-SPH arm does not saturate -- it blows up.

The shipped claim in ``docs/xval_projection_laplace_residual_growth.md`` is that
``max|u|/v_ref`` over a 120-step march (``R/dx = 8, 12``) is a *saturating*
steady current ("plateau", straight-line ``R^2 = 0.84``).  The 600-step horizon
record and the ``R/dx = 16`` ladder record that landed afterwards say otherwise,
and this module pins the refutation down so that a later unit cannot re-assert
the plateau without noticing:

* ``dx12_horizon`` (600 steps declared) diverges at step 181: the last finite
  sample is still growing (0.461 at step 180 against 0.286 at step 120), i.e. the
  "plateau" ended after the window closed;
* ``dx16_nu01`` diverges on its **third** step with a NaN -- at a *finer* level
  than the two the plateau claim was fitted on;
* the ladder summariser therefore reports the declared gates as **failed**, and
  must keep doing so: a skipped or silently-passing gate is the defect this test
  is guarding.

These are *record* assertions, not re-runs: they cost nothing and they fail
loudly if the records are ever replaced by numbers that contradict them.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "benchmarks" / "laplace_sph" / "results_dynamic_ladder"
SUMMARIZER = ROOT / "benchmarks" / "laplace_sph" / "run_dynamic_ladder.py"


def _load(name: str) -> dict:
    p = RECORDS / f"{name}.json"
    if not p.exists():
        pytest.skip(f"record {name} not in the tree")
    return json.loads(p.read_text())


def _finite_samples(rec: dict) -> list:
    return [h for h in rec["history"] if not h.get("diverged")]


def test_horizon_march_diverges_although_the_primary_window_looks_saturated():
    """The 120-step window is not a plateau: the 600-step march blows up at 181."""
    horizon = _load("dx12_horizon")
    assert horizon["role"] == "horizon"
    assert horizon["diverged"] is True
    broke = [h for h in horizon["history"] if h.get("diverged")]
    assert broke, "a diverged record must carry the step it broke at"
    assert broke[-1]["step"] == 181, broke[-1]
    assert broke[-1]["umax_over_vref"] > 1.0e3      # it is a blow-up, not a wobble

    fin = _finite_samples(horizon)
    at120 = [h for h in fin if h["step"] == 120]
    at180 = [h for h in fin if h["step"] == 180]
    assert at120 and at180
    # the number the plateau claim was fitted to, and what the longer march says
    # the value the 120-step record ends on (both runs agree to ~1e-8 rel.)
    assert at120[0]["umax_over_vref"] == pytest.approx(0.286, rel=2e-3)
    assert at180[0]["umax_over_vref"] > 1.5 * at120[0]["umax_over_vref"]
    # no sample after step 120 is lower than the step-120 value: not a plateau
    late = [h["umax_over_vref"] for h in fin if h["step"] > 120]
    assert min(late) >= at120[0]["umax_over_vref"] * 0.98


def test_primary_and_horizon_records_agree_where_they_overlap():
    """Two independent runs of the same physics must agree on the shared window.

    ``dx12_nu01`` (threads=6, local) and ``dx12_horizon`` (threads=24, cluster)
    share the first 120 steps.  They agree to a few % -- i.e. the divergence is a
    property of the scheme, not of one machine.
    """
    prim = _load("dx12_nu01")
    horiz = _load("dx12_horizon")
    for step in (10, 60, 120):
        a = [h for h in _finite_samples(prim) if h["step"] == step]
        b = [h for h in _finite_samples(horiz) if h["step"] == step]
        assert a and b
        assert a[0]["umax_over_vref"] == pytest.approx(b[0]["umax_over_vref"], rel=0.25)


def test_finer_level_diverges_immediately_so_the_ladder_cannot_be_extended():
    finer = _load("dx16_nu01")
    assert finer["R_over_dx"] == 16.0
    assert finer["diverged"] is True
    assert finer["steps_completed"] == 0
    assert math.isnan(finer["umax_over_vref_final"])


def test_summariser_reports_the_declared_gates_as_failed():
    """A gate that cannot be evaluated must read as FAIL, never as a pass."""
    if not SUMMARIZER.exists():
        pytest.skip("summariser not in the tree")
    sys.path.insert(0, str(SUMMARIZER.parent))
    import importlib

    mod = importlib.import_module("run_dynamic_ladder")
    summary = mod.summarize()
    gates = summary["gate_verdicts"]
    assert summary["n_jobs_present"] == 4 and summary["n_jobs_expected"] == 7
    assert gates["no_divergence_viscous"]["pass"] is False
    assert gates["horizon_drift_hold"]["pass"] is False
    assert gates["laplace_all_levels_15pct"]["pass"] is False   # a NaN level poisons it
    assert gates["projection_exact"]["pass"] is True            # the solve itself is fine
    assert gates["spurious_gate_finest"]["pass"] is False
    assert gates["spurious_decay_order_min"]["pass"] is False

def test_lattice_health_reports_a_breakage_signal_on_a_healthy_lattice():
    """The instrument must be off by default and quantitative when switched on."""
    import torch
    from tensorsph.projection_adjoint import (csf_state_two_phase, hex_block,
                                              lattice_health)
    from tensorsph.neighbor_search import cell_linked_pairs

    torch.set_num_threads(2)
    R, SIGMA, RHO, HF = 0.02, 0.072, 1000.0, 1.3
    dx = R / 8.0
    h = HF * dx
    pos, mass, liquid, _ = hex_block(R, dx, 0.012, RHO)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st = csf_state_two_phase(pos, mass, liquid, h, RHO, SIGMA, curvature_method="mls")
    out = lattice_health(pos, pairs, st, dx, R, math.sqrt(SIGMA / (RHO * R)))
    assert out["pair_min_over_dx"] == pytest.approx(1.0, abs=0.05)
    assert out["nn_min"] >= 1.0
    assert out["nn_median"] > out["nn_min"]
    assert 0.0 <= out["color_min"] and out["color_max"] <= 1.0
    assert out["kappa_R_max"] == pytest.approx(1.0, abs=1.0)   # O(1) on a resolved drop
    assert out["acc_over_aref_max"] > 0.0
    assert lattice_health(pos, pairs[:0], st, dx, R, 1.0) == {}
