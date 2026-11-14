"""Anchor grading test for CASE-SPH-COUETTE-TRANSIENT (committed artifacts only).

Anchor: CASE-SPH-COUETTE-TRANSIENT -- ref: exact start-up series of plane Couette
flow (Morris, Fox & Zhu, JCP 136 (1997) 214, Sec. V), tolerance < 8 %.

The test grades ``benchmarks/couette_transient/out/results.json`` without running
the solver, and does the jobs an ordinary "read the number and compare" test does
not:

1. **Binding** -- every published error number is recomputed here from the raw
   per-row ``(y, u)`` samples by an independently written closed form (this file's
   ``_exact_profile``), and must agree to 1e-9.  The verdict therefore cannot
   drift away from the data it claims to grade.

2. **No over-certification** -- the verdict may not be better than the
   predeclared gates: every gated quantity outside 8 % must appear in
   ``verdict.reasons``, ``PASS`` is impossible while a violation exists, and the
   coarse rung that misses the tolerance must stay listed in
   ``coarse_level_failures``.  Because the measured outcome *is* a FAIL (the
   wall-shear gate at t* = 0.1 is outside 8 % at every resolution, and the
   near-steady profile residual does not converge), the test also pins that
   negative result: a future run cannot silently upgrade it.

Regenerate the artifacts with::

    PYTHONPATH=src:. python benchmarks/couette_transient/run.py --sweep \
        --resolutions 10 16 24 32
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "couette_transient" / "out" / "results.json"

TOL = 0.08          # pool tolerance 8 %
REF_RES = 16        # gating resolution of the predeclared hypothesis
T_STAR_GATED = 0.1
T_STAR_NEAR_STEADY = 0.5


def _exact_profile(y, t, height, wall_speed, viscosity):
    """Closed-form start-up profile, written independently of run.py."""
    y = np.asarray(y, dtype=float)
    n = np.arange(1, 4001, dtype=float)
    decay = np.exp(-viscosity * (n * math.pi / height) ** 2 * t)
    keep = decay > 1e-16
    n, decay = n[keep], decay[keep]
    series = np.zeros_like(y)
    for ni, di in zip(n, decay):
        series += (-1.0) ** ni / ni * np.sin(ni * math.pi * y / height) * di
    return wall_speed * y / height + 2.0 * wall_speed / math.pi * series


def _flux_exact(t_star, height, wall_speed):
    k = np.arange(0, 20000, dtype=float)
    odd = 2 * k + 1
    s = float(np.sum(np.exp(-odd ** 2 * math.pi ** 2 * t_star) / odd ** 2))
    return 0.5 * wall_speed * height * (1.0 - 8.0 / math.pi ** 2 * s)


@pytest.fixture(scope="module")
def data():
    assert RESULTS.exists(), f"missing committed artifact {RESULTS}"
    return json.loads(RESULTS.read_text())


def _runs(data):
    out = list(data["ladder"])
    for key in ("sound_speed", "dt_factor"):
        out += data["sweeps"].get(key, [])
    return out


def _first_gated(run):
    cand = [m for m in run["metrics"].values()
            if not m.get("missing") and m["t_star"] >= T_STAR_GATED - 1e-12]
    return min(cand, key=lambda m: m["t_star"])


def _violations(data):
    """Every gated quantity outside the tolerance, recomputed from the metrics."""
    bad = []
    for run in data["ladder"]:
        if run["resolution"] < REF_RES:
            continue
        for key, m in run["metrics"].items():
            if m.get("missing") or m["t_star"] < T_STAR_GATED - 1e-12:
                continue
            if m["l2_rel"] >= TOL:
                bad.append(f"{run['label']} t*={m['t_star']:.3f} L2")
            if abs(m["flux_rel"]) >= TOL:
                bad.append(f"{run['label']} t*={m['t_star']:.3f} flux")
        if abs(_first_gated(run)["shear_rel"]) >= TOL:
            bad.append(f"{run['label']} t*={_first_gated(run)['t_star']:.3f} shear")
    return bad


def test_published_errors_are_recomputable(data):
    """Every stored L2/flux number must follow from the stored raw rows."""
    height = data["height"]
    nu = data["viscosity"]
    U = data["wall_speed"]
    lo, hi = data["interior_band"]
    checked = 0
    for run in _runs(data):
        for key, prof in run["profiles"].items():
            if prof is None:
                continue
            y = np.asarray(prof["y"], dtype=float)
            u = np.asarray(prof["u"], dtype=float)
            t = prof["t_mean"]
            u_ex = _exact_profile(y, t, height, U, nu)
            band = (y / height >= lo) & (y / height <= hi)
            l2 = float(np.linalg.norm(u[band] - u_ex[band])
                       / np.linalg.norm(u_ex[band]))
            q_num = float(np.trapezoid(u, y) + 0.5 * y[0] * u[0]
                          + 0.5 * (height - y[-1]) * (u[-1] + U))
            q_ex = _flux_exact(nu * t / height ** 2, height, U)
            got = run["metrics"][key]
            assert abs(got["l2_rel"] - l2) < 1e-9, (run["label"], key, got["l2_rel"], l2)
            assert abs(got["flux_rel"] - (q_num / q_ex - 1.0)) < 1e-9, (run["label"], key)
            checked += 1
    assert checked >= 25, f"only {checked} sampled profiles graded"


def test_verdict_is_not_better_than_the_gates(data):
    """PASS is impossible while a gate is violated, and every violation is listed."""
    bad = _violations(data)
    verdict = data["verdict"]
    assert bad, ("no violation found -- if the solver really reached 8 % everywhere, "
                 "re-derive the gate, do not assume the test is stale")
    assert verdict["status"] == "FAIL"
    for item in bad:
        assert any(item.split()[-1] in reason and
                   item.split()[0].replace("ladder-", "") in reason
                   for reason in verdict["reasons"]), (item, verdict["reasons"])
    assert len(verdict["reasons"]) >= len(bad)


def test_coarse_level_failures_stay_visible(data):
    """A coarse rung that misses 8 % must be recorded, not dropped."""
    coarse = [r for r in data["ladder"] if r["resolution"] < REF_RES]
    assert coarse, "the ladder must contain a coarse level"
    reported = data["verdict"].get("coarse_level_failures", [])
    miss = [r["label"] for r in coarse if r["gate"]["status"] != "PASS"]
    hit = {r["label"] for r in coarse
           if any(r["label"] in reason for reason in reported)}
    assert hit == set(miss), (sorted(hit), sorted(miss))


def test_profile_error_converges_at_the_first_gated_instant(data):
    """t* = 0.1 must converge with H/dx (this is what does behave)."""
    ladder = sorted(data["ladder"], key=lambda r: r["resolution"])
    assert [r["resolution"] for r in ladder] == [10, 16, 24, 32]
    errs = [_first_gated(r)["l2_rel"] for r in ladder]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < TOL
    assert errs[1] >= TOL - 1e-9 or errs[1] < TOL      # documented, not gated here


def test_near_steady_residual_does_not_converge(data):
    """Pinned negative result: the t* = 0.5 profile residual is a dx floor."""
    ladder = sorted(data["ladder"], key=lambda r: r["resolution"])

    def late(run):
        return next(m["l2_rel"] for m in run["metrics"].values()
                    if not m.get("missing")
                    and abs(m["t_star"] - T_STAR_NEAR_STEADY) < 1e-3)

    vals = [late(r) for r in ladder]
    assert not all(b <= a for a, b in zip(vals, vals[1:])), vals   # non-monotone
    assert min(vals) > 0.04, vals      # a ~5-7 % floor, reported not hidden


def test_free_slip_control_does_not_reproduce_the_flow(data):
    control = data["control"]
    assert control is not None, "the negative control must be part of the artifacts"
    assert control["wall_friction"] == 0.0
    late = max((m for m in control["metrics"].values() if not m.get("missing")),
               key=lambda m: m["t_star"])
    assert late["t_star"] >= 0.4
    assert (1.0 + late["flux_rel"]) < 0.20, late


def test_parameters_are_the_declared_ones(data):
    assert data["reynolds"] == pytest.approx(data["wall_speed"] * data["height"]
                                            / data["viscosity"])
    assert data["gates"]["l2_rel"] == pytest.approx(TOL)
    assert data["t_star_gated"] == pytest.approx(T_STAR_GATED)
    for run in _runs(data):
        assert run["max_mach"] < data["gates"]["max_mach"]