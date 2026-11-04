"""Verification of the shallow-water reference used by the B2 finite-reservoir line.

Two layers, both machine-checkable and cheap (the whole file runs in a few
seconds at the resolutions used here):

1. **The reference generator itself** (``tensorsph.shallow_water``) is scored
   against the exact Ritter solution in the regime where Ritter *is* exact.  A
   numerical reference that has not been verified cannot arbitrate anything, so
   these tests run before any finite-reservoir number is quoted.
2. **The archived artifact** (``benchmarks/dam_break/shallow_water_reference.json``)
   is pinned: the finite-reservoir front slope, the attribution of the SPH-vs-
   Ritter deviation, and the fact that
   ``docs/benchmarks/dam-break-finite-reservoir-reference.md`` is a rendering of
   the artifact rather than hand-edited numbers.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tensorsph import ritter
from tensorsph.shallow_water import (
    DRY_DEPTH,
    FRONT_LEVELS,
    dam_break,
    front_coefficient,
    front_position,
    ritter_profile,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "dam_break" / "shallow_water_reference.json"
DOCUMENT = ROOT / "docs" / "benchmarks" / "dam-break-finite-reservoir-reference.md"

GRAVITY = 9.81
H = 0.4
A = 0.2


@pytest.fixture(scope="module")
def payload():
    assert ARTIFACT.exists(), f"missing artifact {ARTIFACT.relative_to(ROOT)}"
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# 1. the generator vs the exact solution
# --------------------------------------------------------------------------
def _semi_infinite(cells=800, time=0.1, order=2, dam_x=3.0):
    run = dam_break(cells=cells, tank_width=2.0 * dam_x, column_width=dam_x,
                    column_height=H, t_end=time, order=order)
    t_end, x, h, u = run.profiles[-1]
    h_exact, u_exact = ritter_profile(x, t_end, H, dam_x=dam_x)
    return run, t_end, x, h, u, h_exact, u_exact


def test_leading_edge_matches_ritter_within_5pct():
    """Estimator-matched leading edge, travel frame, second order.

    The 0.01 level is the calibrated one: on the same grid and with the same
    estimator the scheme under-reads the travel by ~4.5 %, and that residual is
    carried as the reference's uncertainty rather than corrected away.
    """
    _, t_end, x, h, _, h_exact, _ = _semi_infinite()
    dam_x = 3.0
    numeric = front_position(x, h, H, 1.0e-2)
    exact = front_position(x, h_exact, H, 1.0e-2)
    error = (numeric - dam_x) / (exact - dam_x) - 1.0
    assert abs(error) < 0.05, f"leading edge off by {100 * error:+.2f} %"


def test_second_order_beats_first_order_on_the_profile():
    run1, _, x, h1, _, h_exact, _ = _semi_infinite(order=1)
    run2, _, x2, h2, _, h_exact2, _ = _semi_infinite(order=2)
    l1 = float((h1 - h_exact).abs().sum() / h_exact.sum())
    l2 = float((h2 - h_exact2).abs().sum() / h_exact2.sum())
    assert l2 < l1
    assert l2 < 0.01, f"profile L1 {100 * l2:.2f} % > 1 %"


def test_mass_is_conserved_to_round_off():
    run, *_ = _semi_infinite()
    assert run.mass_error < 1.0e-10
    assert run.extra["front_reached_wall"] is False


def test_depth_at_the_dam_section_approaches_four_ninths():
    """``h(0, t) = 4H/9``: the dam section does not keep its depth."""
    errors = []
    for cells in (400, 800, 1600):
        run, _, x, h, _, h_exact, _ = _semi_infinite(cells=cells)
        index = int(3.0 / run.dx - 0.5)
        errors.append(abs(float(h[index]) - 4.0 * H / 9.0) / (4.0 * H / 9.0))
    assert errors[0] > errors[-1], errors        # first order convergence
    assert errors[-1] < 0.10, errors


def test_riemann_invariant_holds_across_the_fan():
    """``u + 2 sqrt(g h) = 2 c0`` through the whole rarefaction."""
    _, t_end, x, h, u, _, _ = _semi_infinite()
    c0 = math.sqrt(GRAVITY * H)
    # The invariant is a relative measure and degenerates at the drying tip
    # (h -> 0, u -> 2c0, u + 2sqrt(gh) -> 2c0 but with the numerical tip noise of
    # a first/second-order scheme); score it where the fan actually carries depth.
    fan = ((x > 3.0 - c0 * t_end) & (x < 3.0 + 2.0 * c0 * t_end)
           & (h > 0.01 * H))
    invariant = u[fan] + 2.0 * (GRAVITY * h[fan]).sqrt()
    worst = float(((invariant - 2.0 * c0).abs() / (2.0 * c0)).max())
    # Pinned as a regression guard: the measured worst case on this grid is
    # 19.4 % (the fan spans the smeared dam section, where a first-order flux
    # is at its worst), and the bound is that value plus ~25 % of headroom.
    assert worst < 0.243, worst


def test_dry_cells_carry_no_momentum():
    run, *_ = _semi_infinite()
    _, x, h, u = run.profiles[-1]
    assert bool((h >= 0.0).all())
    assert float(u[h <= DRY_DEPTH].abs().max()) == 0.0


def test_rejects_nonsense_configurations():
    with pytest.raises(ValueError):
        dam_break(cells=4)
    with pytest.raises(ValueError):
        dam_break(cfl=0.9)
    with pytest.raises(ValueError):
        dam_break(column_width=2.0, tank_width=2.0)
    with pytest.raises(ValueError):
        dam_break(order=3)


# --------------------------------------------------------------------------
# 2. the finite-reservoir result the artifact carries
# --------------------------------------------------------------------------
def test_finite_reservoir_front_is_slower_than_ritter(payload):
    """Reservoir depletion can only remove momentum: the slope must be < 2."""
    setup = payload["reference"]["setup"]
    assert setup["ritter_slope"] == 2.0
    for run in payload["reference"]["runs"]:
        coefficients = run["front_coefficients"]
        assert all(value < 2.0 for value in coefficients.values()), coefficients
        assert 1.5 < coefficients["0.01"] < 1.9, coefficients


def test_front_slope_is_resolution_converged(payload):
    """The quoted slope must not move when the grid is refined."""
    slopes = [run["front_coefficients"]["0.01"] for run in payload["reference"]["runs"]]
    assert max(slopes) - min(slopes) < 0.03, slopes


def test_reference_generator_verification_is_recorded(payload):
    """The artifact must carry its own verification, with the numbers in range."""
    rows = payload["verification"]["rows"]
    assert any(row["order"] == 2 for row in rows)
    worst = max(row["profile_l1_relative"] for row in rows)
    assert worst < 0.02, worst
    second_order = [row for row in rows if row["order"] == 2]
    errors = [abs(row["front_errors"]["0.01"]["travel_fraction_error"])
              for row in second_order]
    assert max(errors) < 0.05, errors
    assert all(row["mass_error"] < 1.0e-9 for row in rows)


def test_attribution_is_evaluated_and_geometry_is_the_smaller_part(payload):
    """The point of the exercise: Ritter's inapplicability is *not* the whole story."""
    attribution = payload["attribution"]
    assert attribution["status"] == "evaluated", attribution
    total = attribution["sph_minus_ritter_mean_rel"]
    geometry = attribution["reference_minus_ritter_mean_rel"]
    assert total < -0.2            # the SPH front really is slow
    assert geometry < 0.0          # the reference is slower than Ritter
    assert abs(geometry) < abs(total), (geometry, total)
    # geometry can explain at most ~40 % of the SPH deficit; if a future fix
    # moves this, the narrative in the doc must be rewritten with it
    share = attribution["geometry_share_of_deviation"]
    assert 0.0 < share < 0.45, share


def test_document_is_a_rendering_of_the_artifact(payload):
    assert DOCUMENT.exists(), DOCUMENT.relative_to(ROOT)
    spec = importlib.util.spec_from_file_location(
        "make_sw_reference_doc",
        ROOT / "benchmarks" / "dam_break" / "make_sw_reference_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render(payload)
    on_disk = DOCUMENT.read_text()
    assert rendered.strip() == on_disk.strip(), (
        "the doc is stale: regenerate it with "
        "python benchmarks/dam_break/make_sw_reference_doc.py")