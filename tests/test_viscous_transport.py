"""Guards on the viscous-transport verification artifact.

Three kinds of test, in the contract the repository already uses for archived
benchmarks (``tests/test_laplace_law.py``, ``tests/test_dam_break_resolution_ladder.py``):

1. **The reference itself** -- the exact Stokes-decay mode must satisfy the
   boundary conditions it is supposed to satisfy, and the energy identity must
   agree with the modal decay. A wrong reference invalidates everything below.
2. **The harness** -- a short run of the shipped operator must decay, must stay
   finite, and must not let the mode grow. This is a live test, not a replay.
3. **The archived finding** -- ``benchmarks/viscous_transport/results.json`` is
   pinned: the effective viscosity at each resolution, the two controls, and the
   *finding that the error shrinks under refinement*. A future fix that removes
   the bias must fail here and force ``docs/verification_viscous_transport.md``
   to be rewritten, instead of leaving a stale "it is 16 % low" claim behind.
4. **The document is a rendering** -- re-rendering the artifact must reproduce
   the checked-in markdown, so no number in the document can be hand-edited.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "viscous_transport"
ARTIFACT = BENCH / "results.json"
DOCUMENT = ROOT / "docs" / "verification_viscous_transport.md"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load("viscous_transport_run", "run.py")


@pytest.fixture(scope="module")
def results():
    if not ARTIFACT.exists():  # pragma: no cover - artifact is checked in
        pytest.skip(f"missing artifact {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------- #
# 1. The reference
# --------------------------------------------------------------------------- #

def test_exact_mode_satisfies_the_boundary_conditions_it_claims(bench):
    """No slip at both plates: u(0) = 0 and u(H) = 0, for every time."""
    H = bench.HEIGHT
    for t in (0.0, 0.05, 0.37):
        assert bench.exact_velocity(0.0, t) == pytest.approx(0.0, abs=1e-15)
        assert abs(bench.exact_velocity(H, t)) < 1e-15 * bench.U0
    # The mode must be the sine that fits the channel: one half wave in H.
    assert bench.wave_number() == pytest.approx(math.pi / H)
    mid = bench.exact_velocity(H / 2, 0.0)
    assert mid == pytest.approx(bench.U0, rel=1e-12)


def test_exact_mode_decays_at_the_configured_viscosity(bench):
    """The modal solution must reproduce nu from its own decay rate."""
    k = bench.wave_number()
    ts = [0.0, 0.05, 0.1, 0.2]
    values = [bench.exact_velocity(bench.HEIGHT / 2, t) for t in ts]
    slope = bench._least_squares_slope(ts, [math.log(v) for v in values])
    assert -slope / (k * k) == pytest.approx(bench.NU, rel=1e-9)


def test_energy_identity_follows_the_modal_decay(bench):
    """E_k(t)/E_k(0) must be the square of the modal amplitude ratio."""
    for t in (0.0, 0.03, 0.11, 0.25):
        ratio = bench.exact_energy_fraction(t)
        modal = math.exp(-bench.NU * bench.wave_number() ** 2 * t)
        assert ratio == pytest.approx(modal * modal, rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. The harness, live
# --------------------------------------------------------------------------- #

def test_a_short_run_decays_and_stays_finite(bench):
    """The shipped operator plus the shipped Euler step must dissipate the mode.

    A coarse slab and a short window keep this in the fast suite; the physics
    claim lives in the archived ladder, this only proves the harness is wired to
    the solver and that the solver does not blow up on the case.
    """
    run = bench.run_case(0.05, t_end=0.06, sample_every=2)
    ratios = [s["kinetic_energy_ratio"] for s in run["samples"]]
    assert all(math.isfinite(r) for r in ratios)
    assert ratios[-1] < 1.0, "a viscous shear mode must lose energy"
    exact = bench.exact_energy_fraction(run["t_end"])
    assert 0.5 * exact < ratios[-1] < 2.0 * exact, (ratios[-1], exact)
    assert run["boundary_events"] == 0, "no particle may leave the domain"
    assert run["peak_density_compression"] < 0.05
    assert run["nu_from_modal"] > 0.0


def test_pressure_forces_conserve_momentum_to_round_off(bench):
    """Without walls and without viscosity every force is pairwise antisymmetric.

    The pressure term is the invariant the pair form has to satisfy exactly. The
    viscous term is *not* in the same class -- see
    ``test_the_viscous_term_is_known_not_to_conserve_momentum``.
    """
    run = bench.run_case(0.02, viscosity=0.0, t_end=0.06, sample_every=3,
                         with_wall=False)
    assert run["momentum_x_relative_drift"] < 1.0e-5, run["momentum_x_relative_drift"]


def test_the_rate_scales_with_the_configured_viscosity(bench):
    """Half the viscosity must halve the decay rate; the operator must be linear."""
    full = bench.run_case(0.02, t_end=0.08, sample_every=2)
    half = bench.run_case(0.02, viscosity=bench.NU / 2, t_end=0.08, sample_every=2)
    assert half["nu_from_modal"] / full["nu_from_modal"] == pytest.approx(0.5, abs=0.08)


# --------------------------------------------------------------------------- #
# 3. The archived finding
# --------------------------------------------------------------------------- #

def test_the_ladder_measures_a_viscosity_error_that_shrinks_with_dx(results):
    """Pins the finding, not just the numbers.

    The effective viscosity is below the configured one and the deficit falls
    under refinement. If a future operator fix removes the deficit, this test
    fails on purpose: the verification record then has to be rewritten.
    """
    ladder = sorted([r for r in results["ladder"] if r["idealisation_ok"]],
                    key=lambda r: r["dx"], reverse=True)
    assert len(ladder) >= 3
    for run in ladder:
        assert 0.0 < run["nu_modal_ratio"] < 1.2, (run["dx"], run["nu_modal_ratio"])
    errors = [abs(run["nu_modal_ratio"] - 1.0) for run in ladder]
    assert errors[0] > errors[-1], f"the error must shrink with dx: {errors}"
    # Two estimators of the same rate must agree; if they diverge, either the
    # mode is contaminated or the fit window is wrong.
    for run in ladder:
        assert abs(run["nu_modal_ratio"] - run["nu_energy_ratio"]) < 0.05, run["nu_modal_ratio"]
    # The finest run has to be the one that decides the verdict.
    assert ladder[-1]["rows_across_channel"] >= 4 * ladder[0]["rows_across_channel"]
    # The conclusion is a deficit, not a match: a 1 % agreement would mean the
    # operator is fine and both this test and the document must be rewritten.
    assert errors[-1] > 0.03, errors


def test_the_viscous_term_is_known_not_to_conserve_momentum(results):
    """Pins finding 8.2: m_j/rho_j is antisymmetric only for uniform density.

    Without walls the fluid's `sum m u_x` can only change through the viscous
    weighting, so the measured drift is the non-conservation of the shipped
    form. If a future rewrite makes the form conservative this test fails and
    the finding has to be rewritten, not silently kept.
    """
    inviscid = results["momentum_invariant"]["inviscid_without_walls"]["relative_drift"]
    viscous = results["momentum_invariant"]["viscous_without_walls"]["relative_drift"]
    assert inviscid < 1.0e-5, inviscid
    assert viscous > 100 * inviscid, (viscous, inviscid)


def test_the_controls_in_the_artifact_pass(results):
    linearity = results["viscosity_control"]
    assert linearity["ratio_to_base"] == pytest.approx(0.5, abs=0.05)
    temporal = results["dt_control"]
    assert temporal["ratio_to_base"] == pytest.approx(1.0, abs=0.05)
    assert results["momentum_invariant"]["inviscid_without_walls"]["relative_drift"] < 1.0e-5


def test_the_idealisation_assumptions_hold_within_the_stated_bounds(results):
    """The exact solution assumes a horizontal, incompressible, subsonic flow."""
    usable = [r for r in results["ladder"] if r["idealisation_ok"]]
    assert usable, "at least one ladder point must satisfy the idealisation"
    worst_u_y = max(r["peak_vertical_velocity"] for r in usable)
    worst_compression = max(r["peak_density_compression"] for r in usable)
    worst_mach = max(r["peak_mach"] for r in usable)
    assert worst_u_y < 0.01 * results["case"]["u0"], worst_u_y
    assert worst_compression < 0.01, worst_compression
    assert worst_mach < 0.10, worst_mach
    # Every excluded run must say which gate it failed, not just be dropped.
    for run in results["ladder"]:
        if not run["idealisation_ok"]:
            assert any(not ok for ok in run["idealisation"].values())


# --------------------------------------------------------------------------- #
# 4. The document is a rendering
# --------------------------------------------------------------------------- #

def test_document_is_generated_from_the_artifact(results):
    make_doc = _load("viscous_transport_make_doc", "make_doc.py")
    assert DOCUMENT.exists(), f"missing {DOCUMENT}"
    assert make_doc.render(results) == DOCUMENT.read_text(), (
        "docs/verification_viscous_transport.md is not the rendering of "
        "benchmarks/viscous_transport/results.json; run make_doc.py")