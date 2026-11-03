"""Pins the curvature-convergence study of mission m2551.

The laplace_law benchmark scores one resolution and reports "+25 % curvature".
These tests keep the *reason* visible: the shipped operator converges when it
is fed a smooth colour field, the raw kernel-sum field makes the error grow
under refinement, and two Shepard passes on the colour field restore
convergence -- but they widen the interface, so the knob stays off by default.

See benchmarks/laplace_law/curvature_convergence.py (data + figure) and
docs/verification_curvature_convergence.md (numbers injected from the JSON).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_law"


def _bench():
    spec = importlib.util.spec_from_file_location(
        "curvature_convergence", BENCH / "curvature_convergence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench()


@pytest.fixture(scope="module")
def report():
    return json.loads((BENCH / "curvature_convergence.json").read_text())


@pytest.fixture(scope="module")
def mid(bench):
    """One static evaluation at R/h ~ 15 -- about a second."""
    return bench.run_resolution(0.001)


# --------------------------------------------------------------------------- #
# committed artefacts (no compute -- keeps the record from going stale silently)
# --------------------------------------------------------------------------- #
def test_committed_table_is_self_consistent(report):
    exact = report["setup"]["radius_m"] ** -1
    assert len(report["rows"]) >= 4
    rs = sorted(report["rows"], key=lambda r: r["dx_m"])   # finest first
    assert rs[0]["R_over_h"] > rs[-1]["R_over_h"]
    for row in rs:
        assert row["exact_1_per_m"] == pytest.approx(exact)
        for key in ("analytic_field", "raw_kernel_sum_field", "smoothed_field"):
            s = row[key]
            recomputed = (s["median_1_per_m"] - exact) / exact * 100.0
            assert recomputed == pytest.approx(s["median_rel_error_pct"], abs=1e-6)


def test_committed_verdicts(report):
    v = report["verdict"]
    assert v["operator_converges_on_smooth_field"] is True
    assert v["raw_field_error_diverges"] is True
    assert v["smoothed_field_within_15pct_at_R_over_h_30"] is True
    # honest grade of the shipped pointwise curvature must stay visible
    assert v["pointwise_kappa_evidence_grade"] == "WITHHELD"
    assert v["force_level_identity_evidence_grade"] == "REAL"


# --------------------------------------------------------------------------- #
# live recomputation of the three claims
# --------------------------------------------------------------------------- #
def test_operator_converges_on_a_smooth_colour_field(mid):
    """Fed an analytic smooth profile the shipped operator is inside the 15 %
    platform bar at R/h ~ 15.  This is what rules out "the MLS operator or the
    normal smoothing is wrong" as the cause of the laplace_law +25 %."""
    err = mid["analytic_field"]["median_rel_error_pct"]
    assert abs(err) < 15.0
    assert mid["R_over_h"] > 10.0


def test_raw_kernel_sum_field_degrades_under_refinement(mid, report):
    """The shipped colour field carries lattice noise; div(n) amplifies it by
    1/h, so refining makes the curvature error *worse* (this is the divergence
    the earlier verification rounds were chasing)."""
    rs = sorted(report["rows"], key=lambda r: r["dx_m"])   # finest first
    fine = rs[0]["raw_kernel_sum_field"]["median_rel_error_pct"]
    coarse = rs[-1]["raw_kernel_sum_field"]["median_rel_error_pct"]
    assert fine > coarse + 40.0          # measured +43 % -> +104 %
    assert fine > 50.0
    # and at the resolution used by benchmarks/laplace_law the raw field is bad
    assert abs(mid["raw_kernel_sum_field"]["median_rel_error_pct"]) > 15.0


def test_smoothing_restores_convergence(mid):
    """Two Shepard passes on c before differentiating bring the same
    resolution inside the 15 % bar and beat the raw field."""
    raw = abs(mid["raw_kernel_sum_field"]["median_rel_error_pct"])
    smooth = abs(mid["smoothed_field"]["median_rel_error_pct"])
    assert smooth < 15.0
    assert smooth < raw


# --------------------------------------------------------------------------- #
# invariants: the diagnostic knob must not silently recalibrate the force
# --------------------------------------------------------------------------- #
def test_color_smoothing_is_off_by_default(bench):
    """The default csf_fields must stay bit-identical to color_smoothing=0 --
    the force calibration (cut-force identity 2 sigma) is tied to the raw
    field, and the knob halves |grad c| (measured F_cut ratio 1.07 -> 0.44)."""
    from tensorsph.csf_consistent import csf_fields
    from tensorsph.neighbor_search import cell_linked_pairs

    dx = 0.002
    pos, mass = bench.hexagonal_droplet(0.01, dx, 1000.0)
    h = 1.3 * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    default = csf_fields(pos, mass, pairs, h, 1000.0)
    explicit = csf_fields(pos, mass, pairs, h, 1000.0, color_smoothing=0)
    assert torch.equal(default.kappa, explicit.kappa)
    assert torch.equal(default.delta_s, explicit.delta_s)
    smoothed = csf_fields(pos, mass, pairs, h, 1000.0, color_smoothing=2)
    assert not torch.equal(smoothed.kappa, default.kappa)


def test_shepard_smoothing_accepts_a_scalar_field(bench):
    """shepard_smooth used to broadcast a 1-D field into (N, N); the curvature
    study needs it on the scalar colour function."""
    from tensorsph.csf_consistent import shepard_smooth
    from tensorsph.neighbor_search import cell_linked_pairs

    dx = 0.002
    pos, mass = bench.hexagonal_droplet(0.01, dx, 1000.0)
    h = 1.3 * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    vol = mass / 1000.0
    constant = torch.full((pos.shape[0],), 0.25, dtype=pos.dtype)
    out = shepard_smooth(constant, pos, vol, pairs, h, passes=1)
    assert out.shape == constant.shape              # was (N, N) before the fix
    assert torch.allclose(out, constant, atol=1e-12)
    # a smooth field is smoothed, not destroyed: still bounded by its own range
    field = torch.linspace(0.0, 1.0, pos.shape[0], dtype=pos.dtype)
    out = shepard_smooth(field, pos, vol, pairs, h, passes=1)
    assert out.shape == field.shape
    assert float(out.min()) >= -1e-9 and float(out.max()) <= 1.0 + 1e-9