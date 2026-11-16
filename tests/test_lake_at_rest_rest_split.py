"""Gates for the well-balanced hydrostatic-split closure (``CASE-SPH-LAKE-AT-REST``).

The record is ``tests/data/lake_at_rest/rest_split.json``, produced by
``benchmarks/lake_at_rest/rest_split.py``.  What is asserted here is what the
document claims, no more:

* the rest state is an **exact** discrete equilibrium of the split closure
  (``t = 0`` residual below the anchor tolerance) and the *clamped* wall
  continuation is not -- the difference between the two arms is the only change
  of input, so the residual is attributable to it;
* the wall continuation is the analytic continuation on every wall particle that
  touches the fluid (``R3``), i.e. the equality above is not a coincidence of one
  particle;
* the operator is **not blind**: its residual at a perturbed state grows linearly
  with the perturbation, while the identity arm stays at roundoff;
* the shipped record's own verdict is honest: the anchor is *not* claimed as met
  while the explicit time integration diverges (``persists`` is expected to be
  false), and the memory of that failure must not be silently flipped without a
  new record.

A fresh recompute of one row keeps the record from being a stale artefact.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "tests" / "data" / "lake_at_rest" / "rest_split.json"
BENCH = REPO / "benchmarks" / "lake_at_rest"

sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

from tensorsph.lake_at_rest import LakeConfig  # noqa: E402
from tensorsph.lake_rest_split import SplitOptions, rest_run  # noqa: E402

TOLERANCE = 1.0e-9


@pytest.fixture(scope="module")
def record():
    assert RECORD.exists(), (
        f"missing {RECORD}; regenerate with PYTHONPATH=src python3 "
        "benchmarks/lake_at_rest/rest_split.py --json "
        "tests/data/lake_at_rest/rest_split.json --doc --figures")
    return json.loads(RECORD.read_text())


def closed(record, arm):
    return [row for row in record["rest_ladder"]
            if row["arm"] == arm and not row["periodic_x"]]


def test_rest_state_is_an_exact_discrete_equilibrium(record):
    rows = closed(record, "split/signed")
    assert rows, "no closed-tank row for the split closure"
    for row in rows:
        assert row["a0_max_g"] < TOLERANCE, row
        assert row["a0_max_ax_g"] < TOLERANCE, row


def test_clamped_wall_extension_is_the_defect(record):
    """Same operator, same lattice, one change: the sign of the wall extension."""
    for row in closed(record, "split/clamped"):
        assert row["a0_max_g"] > 1e-2, row


def test_signed_extension_is_the_analytic_continuation(record):
    defect = record["wall_defect"]
    assert defect["n_wall_in_support"] > 0
    assert defect["max_defect_over_pmax"] < 1e-8, defect


def test_identity_arm_is_blind_and_the_scheme_is_not(record):
    response = record["perturbation_response"]
    scheme = {row["amplitude_over_dx"]: row for row in response
              if row["arm"] == "split/signed"}
    identity = {row["amplitude_over_dx"]: row for row in response
                if row["arm"] == "prescribed/signed"}
    assert scheme[1.0]["max_a_g"] > 1e-2
    assert identity[1.0]["max_a_g"] < TOLERANCE
    ratio = scheme[0.5]["max_a_g"] / scheme[0.25]["max_a_g"]
    assert 1.5 < ratio < 2.5, ratio


def test_anchor_is_not_claimed_before_the_integration_is_stable(record):
    """The honest state of the anchor is in the artefact; it must stay honest."""
    summary = record["summary"]
    assert summary["equilibrium_exact"] is True
    assert summary["persists"] is False, (
        "persistence is now reported as holding -- that would need a stable "
        "integration and a new record, not an edit of this one")
    assert summary["anchor_met"] is False
    assert summary["dynamics_certified"] is False
    for row in closed(record, "split/signed"):
        assert row["t_exceed_tolerance"] is not None


def test_fresh_recompute_reproduces_the_headline(record):
    """A stale JSON is not a verdict: one row is re-derived here."""
    torch.set_default_dtype(torch.float64)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = LakeConfig(dx=0.050)
        fresh = rest_run(cfg, horizon=1.0e-3, options=SplitOptions(),
                         sample_every=4)
        clamped = rest_run(cfg, horizon=1.0e-3,
                           options=SplitOptions(wall_extension="clamped"),
                           sample_every=4)
    finally:
        torch.set_num_threads(previous)
    assert fresh["a0_max_g"] < TOLERANCE
    assert clamped["a0_max_g"] > 1e-2
    shipped = [row for row in closed(record, "split/signed")
               if abs(row["dx"] - 0.050) < 1e-12][0]
    assert fresh["a0_max_g"] == pytest.approx(shipped["a0_max_g"], rel=1e-6,
                                              abs=1e-18)