"""Scope of the single-phase interface-completion closure (round-2 follow-up).

Locks the invariants of
``benchmarks/surface_tension/single_phase_completion_transfer.py``:

1. the hexagonal lattice construction is a genuine triangular lattice (cell area
   ``dx**2 sqrt(3)/2``, area filling consistent with the disc area);
2. the round-2 closure weight transfers *quantitatively* to lattices the fit
   never saw -- a hex lattice and a lattice offset by half a cell;
3. the weight that nulls the Laplace reading is **lattice dependent**: the
   coefficient is a discretisation calibration, not a physical constant.  This
   test is the guard against the ledger claiming more than the evidence;
4. the completed flux stays momentum conservative on every lattice;
5. the round-2 baseline number itself is unchanged (-30 % on the square
   lattice), i.e. this file adds evidence and does not re-tune the closure.

The cheapesr rungs are used so the file stays inside the local test budget; the
full ladder is produced by the benchmark and stored in
``out_single_phase_completion_transfer/gate.json``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "benchmarks" / "surface_tension") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "surface_tension"))

import single_phase_completion_transfer as X  # noqa: E402

GATE = X.GATE * 100.0
ALPHA = X.ALPHA_STAR
RUNG = 16


@pytest.fixture(scope="module", autouse=True)
def _dtype():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# 1. the arrangement itself
# --------------------------------------------------------------------------- #
def test_hex_lattice_is_triangular():
    dx = X.RADIUS / RUNG
    pos, cell = X.lattice_points("hex", dx, X.RADIUS)
    assert cell == pytest.approx(dx * dx * math.sqrt(3.0) / 2.0)
    # the body is the disc: every site inside, and the sum of the cells is the
    # disc area up to the unavoidable boundary strip
    r = torch.linalg.norm(pos, dim=1)
    assert float(r.max()) <= X.RADIUS + 1e-12
    assert float(r.min()) < dx
    area = pos.shape[0] * cell
    assert 0.90 * math.pi * X.RADIUS ** 2 < area < 1.02 * math.pi * X.RADIUS ** 2
    # the second row is offset by half a cell, so no two rows share an x
    ys = torch.unique(pos[:, 1])
    assert ys.numel() > 3


def test_offset_moves_the_lattice_not_the_body():
    dx = X.RADIUS / RUNG
    a, _ = X.lattice_points("square", dx, X.RADIUS, (0.0, 0.0))
    b, _ = X.lattice_points("square", dx, X.RADIUS, (0.5, 0.0))
    for pts in (a, b):
        assert float(torch.linalg.norm(pts, dim=1).max()) <= X.RADIUS + 1e-12
    # the two arrangements cannot be related by a translation of a lattice site
    assert a.shape[0] != b.shape[0] or not torch.allclose(
        torch.sort(a[:, 1])[0], torch.sort(b[:, 1])[0])


# --------------------------------------------------------------------------- #
# 2. quantitative transfer of the round-2 weight
# --------------------------------------------------------------------------- #
def test_round2_baseline_unchanged():
    """The square-lattice vacuum reduction is still ~-30 % (round-2 open item)."""
    rec = X.laplace(X.RADIUS, X.RADIUS / 32.0, 0.0, "square")
    assert -32.0 < rec["err_pct"] < -28.0


@pytest.mark.parametrize("kind,offset", [
    ("square", (0.5, 0.0)),        # interface not centred on a site
    ("square", (0.3, 0.37)),       # arbitrary incommensurate offset
    ("hex", (0.0, 0.0)),           # different neighbour geometry
])
def test_closure_transfers_to_unseen_lattice(kind, offset):
    rec = X.laplace(X.RADIUS, X.RADIUS / RUNG, ALPHA, kind, offset)
    assert abs(rec["err_pct"]) < GATE, rec


def test_closure_leaves_no_net_force_on_any_lattice():
    """The completed flux is a symmetric two-point flux: sum f = 0 to round-off."""
    for kind in ("square", "hex"):
        rec = X.laplace(X.RADIUS, X.RADIUS / RUNG, ALPHA, kind)
        assert rec["momentum_residual"] < 1e-12
        assert rec["momentum_net_over_sum"] < 1e-12


# --------------------------------------------------------------------------- #
# 3. the weight is a calibration, not a constant
# --------------------------------------------------------------------------- #
def test_null_weight_is_lattice_dependent():
    """alpha that nulls the reading differs between square and hex lattices.

    This is the finding the file exists for: the closure *works* on unseen
    arrangements (previous test) but its weight does **not** transport, so the
    capability entry must call ``alpha*`` a discretisation calibration fitted on
    a square lattice rather than a physical closure constant.
    """
    scan = {kind: [X.laplace(X.RADIUS, X.RADIUS / RUNG, a, kind)
                   for a in (0.2, 0.3, 0.4, 0.5)]
            for kind in ("square", "hex")}
    nulls = {k: X.null_weight(v) for k, v in scan.items()}
    a_sq = nulls["square"]["alpha_null"]
    a_hex = nulls["hex"]["alpha_null"]
    assert a_sq == a_sq and a_hex == a_hex          # both crosses were bracketed
    assert 0.3 < a_sq < 0.6
    assert 0.2 < a_hex < 0.6
    assert abs(a_hex - a_sq) > 0.10 * a_sq, (a_sq, a_hex)
    # the error is monotone in alpha on both lattices, which is what makes the
    # single-crossing interpolation meaningful
    for kind, recs in scan.items():
        errs = [r["err_pct"] for r in recs]
        assert errs == sorted(errs), (kind, errs)


def test_round2_weight_is_inside_the_gate_on_every_lattice():
    """Even though it is a calibration, the round-2 weight is not a knife edge.

    ``alpha*`` is applied unchanged on the hex lattice and on offset lattices;
    the residual reading stays inside the 15 % ledger gate there, so adopting it
    as the default single-phase closure cannot *break* another arrangement.
    """
    worst = 0.0
    for kind, offset in (("square", (0.0, 0.0)), ("square", (0.5, 0.0)),
                         ("hex", (0.0, 0.0))):
        rec = X.laplace(X.RADIUS, X.RADIUS / RUNG, ALPHA, kind, offset)
        worst = max(worst, abs(rec["err_pct"]))
    assert worst < GATE, worst


# --------------------------------------------------------------------------- #
# 4. the functional indicator and the dynamic probe run and stay finite
# --------------------------------------------------------------------------- #
def test_mode_stiffness_indicator_is_finite_and_alpha_sensitive():
    low = X.mode_stiffness(X.RADIUS, X.RADIUS / RUNG, 0.0, "square")
    high = X.mode_stiffness(X.RADIUS, X.RADIUS / RUNG, ALPHA, "square")
    for rec in (low, high):
        assert math.isfinite(rec["omega2_measured"])
        assert rec["omega2_measured"] > 0.0
    # adding the completion stiffens the mode: the sign of the effect matches
    # the static reading (the closure adds the missing half of the flux)
    assert high["omega2_measured"] > low["omega2_measured"]


def test_dynamic_probe_reports_containment():
    rec = X.dynamic_probe(X.RADIUS, 10, ALPHA, model="css", periods=0.05, sample=10)
    assert rec["steps"] > 10
    assert math.isfinite(rec["containment_max_r_over_R"])
    assert isinstance(rec["contained"], bool)
    assert rec["particles"] > 100