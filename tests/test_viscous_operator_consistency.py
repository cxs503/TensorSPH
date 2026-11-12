"""Pins for the viscous-operator consistency audit.

Three things are asserted, and they are deliberately different in kind:

1. **Reproduction** -- the 2-D measurements stored in ``results.json`` are
   recomputed here from the operators themselves, so the committed artifact
   cannot drift away from the code.  Only the coarse (``dx = 0.05``) rung is
   recomputed: it is the whole 2-D ladder's shape in one cheap call.
2. **The finding** -- the 2-D family is consistent (spread < 2 %) while the 3-D
   family is not (spread > 10 % at the spacings the 3-D cases use), and every
   operator reproduces the *shape* of the Laplacian (residual < 1 %).  These are
   the claims the document makes; if a future change to any operator moves them,
   the test fails and the document has to be re-rendered.
3. **Document/artifact agreement** -- the numbers quoted in
   ``docs/verification_viscous_operator_consistency.md`` are formatted from
   ``results.json``, so a hand-edited table is caught here.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "viscous_operator_consistency"
DOC = REPO / "docs" / "verification_viscous_operator_consistency.md"


def _load_bench():
    spec = importlib.util.spec_from_file_location("viscous_op_run",
                                                  BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads((BENCH / "results.json").read_text())


@pytest.fixture(scope="module")
def bench():
    return _load_bench()


def test_recomputed_2d_operators_match_the_artifact(payload, bench):
    """Recompute the coarse 2-D rung and match the stored numbers."""
    stored = {(r["impl"]): r for r in payload["rows"]
              if r["dim"] == 2 and abs(r["dx"] - 0.05) < 1e-12}
    assert set(stored) == {"tgv_morris", "droplet_prod", "droplet_dense",
                           "isph_morris"}

    pos, mass, n = bench.lattice2d(0.05)
    vel = bench.tgv_field(pos)
    lap = bench.tgv_laplacian(pos)
    mask = bench.interior_mask_2d(n, bench.INTERIOR_CELLS)

    acc = bench.op_tgv_morris(pos, mass, 0.05)
    got = bench.operator_readout(acc, lap, bench.NU, mask)["nu_eff_over_nu"]
    assert got == pytest.approx(stored["tgv_morris"]["interior"]["nu_eff_over_nu"],
                                rel=1e-9)

    acc = bench.op_droplet(bench.rhs, pos, mass, 0.05, vel)
    got = bench.operator_readout(acc, lap, bench.NU, mask)["nu_eff_over_nu"]
    assert got == pytest.approx(stored["droplet_prod"]["interior"]["nu_eff_over_nu"],
                                rel=1e-6)

    acc = bench.op_droplet(bench.rhs_all_pairs, pos, mass, 0.05, vel)
    got = bench.operator_readout(acc, lap, bench.NU, mask)["nu_eff_over_nu"]
    assert got == pytest.approx(stored["droplet_dense"]["interior"]["nu_eff_over_nu"],
                                rel=1e-6)

    acc = bench.op_isph_morris(pos, mass, 0.05, vel)
    got = bench.operator_readout(acc, lap, bench.NU, mask)["nu_eff_over_nu"]
    assert got == pytest.approx(stored["isph_morris"]["interior"]["nu_eff_over_nu"],
                                rel=1e-6)


def test_every_operator_reproduces_the_shape_of_the_laplacian(payload):
    """Residual < 1 % everywhere: the disagreements are levels, not stencils."""
    for r in payload["rows"]:
        assert r["interior"]["residual_pct"] < 1.0, (r["impl"], r["dx"],
                                                    r["interior"]["residual_pct"])


def test_two_d_family_is_consistent(payload):
    """The 2-D Morris implementations agree to < 2 % at every resolution."""
    for dx, rec in payload["summary"]["morris_spread_pct"].items():
        assert rec["spread_pct"] < 2.0, (dx, rec["spread_pct"])
        for value in rec["values"].values():
            assert 0.9 < value < 1.1, (dx, value)


def test_2d_family_converges_to_the_requested_viscosity(payload):
    """Refinement drives every 2-D operator to nu_eff/nu = 1 (second order)."""
    rows = [r for r in payload["rows"] if r["dim"] == 2]
    # ``isph_morris`` is exempt from the monotonicity half of the check on
    # purpose: its eta regulariser is a fixed fraction of h, so it converges to a
    # biased limit (1.0088) and its finest error is larger than its middle one.
    # That bias is a documented finding, not a regression, so only its finest
    # error is gated.
    monotone = {"tgv_morris", "droplet_prod", "droplet_dense"}
    for impl in {r["impl"] for r in rows}:
        # descending dx, so ``errors[0]`` is the coarsest and ``errors[-1]``
        # the finest rung
        sub = sorted((r for r in rows if r["impl"] == impl),
                     key=lambda r: -r["dx"])
        errors = [abs(r["interior"]["nu_eff_over_nu"] - 1.0) for r in sub]
        # ``droplet_dense`` stops at dx = 0.025 (O(N^2) cost), so its finest rung
        # is coarser than the other three and gets the coarser bound.
        bound = 0.015 if len(sub) >= 3 else 0.03
        assert errors[-1] < bound, (impl, errors[-1], bound)
        if impl in monotone:
            assert all(b <= a for a, b in zip(errors, errors[1:])), (impl, errors)


def test_3d_family_ambiguity_is_real(payload):
    """The 3-D spread is the finding: > 10 % wherever the 3-D cases operate."""
    for dx, rec in payload["summary"]["threed_spread_pct"].items():
        assert rec["spread_pct"] > 10.0, (dx, rec["spread_pct"])
    # ... and the order-4 variant is the accurate one, at the finest spacing.
    rows = [r for r in payload["rows"]
            if r["impl"].startswith("threed_") and abs(r["dx"] - 0.04) < 1e-12]
    errors = {r["impl"]: abs(r["interior"]["nu_eff_over_nu"] - 1.0) for r in rows}
    assert errors["threed_renorm4"] < errors["threed_renorm2"]
    assert errors["threed_renorm4"] < errors["threed_raw"]
    assert errors["threed_renorm4"] < 0.03


def test_document_quotes_the_artifact(payload):
    """No hand-typed numbers: the document's tables are formatted from the JSON."""
    text = DOC.read_text()
    for r in payload["rows"]:
        assert f"| {r['interior']['nu_eff_over_nu']:.4f} |" in text, r
        assert f"| {r['interior']['residual_pct']:.2f} |" in text, r
    for key in ("morris_spread_pct", "threed_spread_pct"):
        for rec in payload["summary"][key].values():
            assert f"| {rec['spread_pct']:.2f} |" in text
    # The stated convergence orders are derived, so only their presence is pinned.
    assert "apparent orders" in text


def test_figure_exists():
    fig = BENCH / "figures" / "nu_eff_vs_dx.png"
    assert fig.exists() and fig.stat().st_size > 10_000