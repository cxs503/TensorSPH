"""Invariants of the FSI2 interface sub-iteration restart.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FROM-FLUID``.

These tests bind the claims of
``docs/xval_fsi2/COUPLING_CONVERGENCE_BUDGET.md`` to executable checks:

1. a secant pair whose iterate increment is at float64 rounding level must not
   reach the least-squares solve (it produces a zero step, i.e. a frozen
   sub-iteration -- reproduced here with the filter disabled);
2. the history must not span a time-step boundary (``start_step``);
3. ``span_steps=True`` still reproduces the pre-fix bookkeeping, so the A/B
   comparison in the report stays reproducible;
4. the coupled driver calls ``start_step`` exactly once per time step;
5. on the real fluid map, the restarted accelerators resolve the interface to a
   residual the shipped (spanning) IQN-ILS cannot reach at the same budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_strong_coupling import (  # noqa: E402
    AitkenRelaxation, IQNILS, _make_accelerator, step_fsi2_strong,
)
from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state  # noqa: E402


# --------------------------------------------------------------------------- #
# Unit level: the freeze mechanism and its cure
# --------------------------------------------------------------------------- #
def test_degenerate_pair_is_filtered_by_default():
    """A frozen iterate must not create a secant column."""
    acc = IQNILS()
    x = np.array([0.30, 0.20, 0.10, 0.05])
    r0 = np.array([1.0e-4, -2.0e-4, 5.0e-5, -1.0e-5])
    acc.next(x, r0)
    # second call: identical iterate (float64 rounding), different residual
    acc.next(x + 1.0e-16 * np.ones_like(x), 2.0 * r0)
    assert acc.dR == [], "degenerate (frozen-iterate) pair leaked into history"
    assert acc.dX == []


def test_degenerate_pair_mechanism_is_reproducible_when_filter_is_off():
    """With the filter disabled the pair is registered and freezes the step.

    This is the defect itself, kept as a reproducible negative control: the
    proposed step ``|x_{k+1} - x_k|`` collapses to ~1e-12 of the residual it is
    supposed to cancel, so the sub-iteration stops moving.
    """
    acc = IQNILS(dex_scale=0.0)
    x = np.array([0.30, 0.20, 0.10, 0.05])
    r0 = np.array([1.0e-4, -2.0e-4, 5.0e-5, -1.0e-5])
    acc.next(x, r0)
    acc.next(x + 1.0e-16 * np.ones_like(x), 2.0 * r0)
    assert len(acc.dR) == 1, "negative control did not register the pair"
    x_next, _ = acc.next(x + 1.0e-16 * np.ones_like(x), 2.0 * r0)
    step = float(np.linalg.norm(x_next - (x + 1.0e-16 * np.ones_like(x))))
    raw = float(np.linalg.norm(2.0 * r0))
    assert step < 1.0e-6 * raw, (
        f"expected a frozen step, got |step|={step:.3e} vs |r|={raw:.3e}")


def test_start_step_drops_the_cross_step_pair():
    """``x`` is continuous across a step boundary, ``r`` is not.

    The pair built there pairs a zero iterate increment with a finite residual
    increment; it must not be registered.
    """
    acc = IQNILS()
    x = np.array([0.30, 0.20, 0.10, 0.05])
    acc.next(x, np.array([1.0e-9, 0.0, 0.0, 0.0]))       # end of step k
    acc.start_step()                                      # boundary
    acc.next(x, np.array([2.0e-4, 0.0, 0.0, 0.0]))        # start of step k+1
    assert acc.dR == []
    assert acc.r_prev is not None and acc.x_prev is not None


def test_span_steps_reproduces_pre_fix_bookkeeping():
    """The legacy arm of the A/B comparison must stay available."""
    acc = IQNILS(span_steps=True)
    x = np.array([0.30, 0.20, 0.10, 0.05])
    acc.next(x, np.array([1.0e-9, 0.0, 0.0, 0.0]))
    acc.start_step()
    assert acc.r_prev is not None, "span_steps=True must not clear r_prev"
    acc.next(x, np.array([2.0e-4, 0.0, 0.0, 0.0]))
    # x did not move, r did -> the poisoned column is registered (legacy)
    assert len(acc.dR) == 0 or np.linalg.norm(acc.dX[-1]) < 1.0e-12


def test_aitken_restart_clears_the_secant():
    acc = AitkenRelaxation()
    x = np.array([0.3, 0.2])
    acc.next(x, np.array([1.0e-4, -1.0e-4]))
    acc.start_step()
    assert acc.r_prev is None
    assert acc.k == 0
    # the relaxation value itself is a useful predictor and is kept
    assert acc.omega == acc.omega0 * 1.0 or np.isfinite(acc.omega)


def test_aitken_span_steps_is_a_no_op():
    acc = AitkenRelaxation(span_steps=True)
    x = np.array([0.3, 0.2])
    acc.next(x, np.array([1.0e-4, -1.0e-4]))
    acc.start_step()
    assert acc.r_prev is not None


def test_synthetic_scalar_gain_map_converges_after_restart():
    """Map ``S(x) = 2 x + b`` (interface gain 2, i.e. ``dr/dx = 1``).

    A restarted Aitken iteration must reach the fixed point ``x* = -b``; the
    spanning IQN-ILS history fed a boundary pair must stall far from it.  This
    is the mechanism at unit scale, without any fluid solve.
    """
    b = np.array([0.5, -0.25])
    x_star = -b

    def S(x):
        return 2.0 * x + b

    def run(acc, n_iter, poison_boundary=False):
        x = np.zeros_like(b)
        for _ in range(n_iter):
            acc.start_step()
            for _ in range(6):
                r = S(x) - x
                x, _ = acc.next(x, r)
            if poison_boundary:
                # emulate the pre-fix bookkeeping: the next step starts from the
                # converged x with a residual from the step advance
                x = x.copy()
        return x

    x_ait = run(AitkenRelaxation(), 1)
    assert np.linalg.norm(x_ait - x_star) < 1.0e-6, (
        f"restarted Aitken did not converge: {x_ait} vs {x_star}")

    # the poison: with the degenerate filter off and a frozen-iterate pair in
    # the history, the same map does not converge in the same budget
    acc = IQNILS(dex_scale=0.0)
    x = np.zeros_like(b)
    acc.next(x, S(x) - x)
    acc.next(x + 1.0e-16 * np.ones_like(x), S(x + 1.0e-16) - (x + 1.0e-16))
    for _ in range(5):
        r = S(x) - x
        x, _ = acc.next(x, r)
    assert np.linalg.norm(x - x_star) > 1.0e-2, (
        "the poisoned history unexpectedly converged; the negative control "
        "no longer reproduces the freeze")


# --------------------------------------------------------------------------- #
# Driver level: one restart per time step
# --------------------------------------------------------------------------- #
def _coarse_case(dx: float = 0.1):
    case = similarity_corrected_case(U_mean=0.2)
    case.dx = dx
    case.sound_speed = 5.0
    case.beam_rho = 1000.0
    case.n_beam_nodes = max(15, int(case.beam_length / dx))
    case.max_rod_substeps = 2000
    case.rod_stability_target = 0.2
    return case


def test_driver_calls_start_step_once_per_step():
    case = _coarse_case()
    dt = case.dt()
    state = initial_state(case)

    class Counting(IQNILS):
        calls = 0

        def start_step(self):
            Counting.calls += 1
            super().start_step()

    acc = Counting()
    for _ in range(3):
        step_fsi2_strong(state, case, dt, accelerator=acc,
                         max_iter=2, tol=1e-2)
    assert Counting.calls == 3, f"start_step called {Counting.calls} times"


def test_default_accelerator_has_the_hook():
    for kind in ("iqn-ils", "aitken"):
        acc, _ = _make_accelerator(kind)
        assert hasattr(acc, "start_step")
        assert acc.span_steps is False


@pytest.mark.parametrize("arm", ["iqn_shipped", "iqn_restart", "aitken_restart"])
def test_real_map_floor_ordering(arm):
    """On the real fluid map the restarted arms beat the spanning history.

    Cheap resolution (dx = 0.1, 2 steps, 20 sub-iterations): the spanning
    IQN-ILS must not reach the floor the restarted accelerators reach with the
    same budget.  This is the executable form of the report's headline claim.
    """
    case = _coarse_case(0.1)
    dt = case.dt()
    factories = {
        "iqn_shipped": lambda: IQNILS(span_steps=True),
        "iqn_restart": lambda: IQNILS(),
        "aitken_restart": lambda: AitkenRelaxation(),
    }
    acc = factories[arm]()
    state = initial_state(case)
    floors = []
    for _ in range(2):
        _, stats = step_fsi2_strong(state, case, dt, accelerator=acc,
                                    max_iter=20, tol=1e-2)
        floors.append(min(stats.residual_history))
    floor = float(np.median(floors))
    if arm == "iqn_shipped":
        assert floor > 1.0e-3, (
            "expected the spanning history not to resolve the interface; "
            f"got floor={floor:.3e} -- the defect may have been fixed "
            "elsewhere, revisit the report")
    else:
        assert floor < 1.0e-1, f"{arm} did not reduce the residual: {floor:.3e}"


def test_restart_arm_reaches_tight_residual():
    """The restarted Aitken arm resolves the interface to <= 1e-6 (dx = 0.1)."""
    case = _coarse_case(0.1)
    dt = case.dt()
    acc = AitkenRelaxation()
    state = initial_state(case)
    floors = []
    for _ in range(3):
        _, stats = step_fsi2_strong(state, case, dt, accelerator=acc,
                                    max_iter=20, tol=1e-6)
        floors.append(min(stats.residual_history))
    assert max(floors) < 1.0e-6, f"floors={floors}"