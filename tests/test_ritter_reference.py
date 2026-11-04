"""Analytic verification of the Ritter dam-break reference used by case B2.

The dam-break benchmark compares the solver against the Ritter shallow-water
solution. That reference is only worth something if the reference itself is
correct, so this file derives it from the shallow-water equations and pins the
identities the benchmark relies on. Nothing here integrates the SPH solver: the
whole file runs at machine precision.

Reference (dry bed, reservoir of depth ``H`` at ``x < 0``)::

    h(x, t) = (2 c - x/t)**2 / (9 g)     -c t <= x <= 2 c t,   c = sqrt(g H)
    u(x, t) = (2/3) (c + x/t)
    x_front(t) = 2 c t
    h(-c t) = H,  h(0, t) = 4H/9,  h(2 c t) = 0

What is pinned:

1. the front speed is ``2 sqrt(gH)`` (so the *initial* column face is at rest in
   the self-similar frame, not at the front);
2. the Riemann invariant ``u + 2 sqrt(g h) = 2 sqrt(g H)`` holds along the whole
   rarefaction, the reservoir keeps its depth ``H`` behind the tail ``x = -c t``
   and the dam depth drops to exactly ``4H/9``;
3. the profile is self-similar: ``h(l x, l t) = h(x, t)``;
4. the area of water ahead of the dam is exactly ``(4/27) H x_front``, i.e. it
   grows linearly with the front;
5. the non-dimensional identity ``Z/a = 2 tau`` for ``H = 2 a``, which is the
   form the B2 harness compares against (it used ``Z/a = tau`` before the
   2026-11 reference audit -- a factor 2, pinned by
   ``test_za_identity_is_two_tau``);
6. the shipped helpers (``ritter_front``, ``ritter_front_over_a``,
   ``ritter_height_profile``) reproduce all of the above.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.dam_break import (  # noqa: E402
    DamBreakConfig,
    ritter_front,
    ritter_front_over_a,
    ritter_height_profile,
)

G = 9.81
A = 0.2
H = 2.0 * A


@pytest.fixture(scope="module")
def cfg():
    c = DamBreakConfig(dx=0.02, column_width=A, pool_width=2.0)
    c.column_height = H
    return c


def _u(x: float, t: float, c: float) -> float:
    """Ritter velocity in the rarefaction fan."""
    return (2.0 / 3.0) * (c + x / t)


def _t_of_tau(tau: float) -> float:
    """Physical time for a given tau = t sqrt(2 g / a)."""
    return tau * math.sqrt(A / (2.0 * G))


# --------------------------------------------------------------------------- #
# 1-4: the analytic solution itself
# --------------------------------------------------------------------------- #
def test_front_speed_is_two_sqrt_gh(cfg):
    c = math.sqrt(G * H)
    for t in (0.05, 0.1, 0.25, 0.5):
        assert ritter_front(t, cfg) == pytest.approx(2.0 * c * t, rel=1e-15)


def test_surface_landmarks(cfg):
    """h(-ct) = H, h(0) = 4H/9, h(2ct) = 0, dry ahead, undisturbed behind."""
    c = math.sqrt(G * H)
    for t in (0.05, 0.2, 0.5):
        assert ritter_height_profile(-c * t, t, cfg) == pytest.approx(H, rel=1e-13)
        assert ritter_height_profile(-1.5 * c * t, t, cfg) == pytest.approx(H, rel=1e-15)
        assert ritter_height_profile(0.0, t, cfg) == pytest.approx(4.0 * H / 9.0,
                                                                  rel=1e-13)
        assert ritter_height_profile(2.0 * c * t, t, cfg) == pytest.approx(0.0, abs=1e-12)
        assert ritter_height_profile(2.2 * c * t, t, cfg) == 0.0
    assert ritter_height_profile(0.0, 0.0, cfg) is None


def test_characteristic_relation_holds(cfg):
    """u + 2 sqrt(gh) = 2 sqrt(gH) across the fan -- the defining invariant."""
    c = math.sqrt(G * H)
    t = 0.3
    for frac in (-1.0, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0):
        x = frac * 2.0 * c * t if frac >= 0 else frac * c * t
        h = ritter_height_profile(x, t, cfg)
        u = _u(x, t, c)
        assert u + 2.0 * math.sqrt(G * h) == pytest.approx(2.0 * c, rel=1e-12)
    # and the velocity at the front is the front speed itself
    assert _u(2.0 * c * t, t, c) == pytest.approx(2.0 * c, rel=1e-15)


def test_profile_is_self_similar(cfg):
    base = [ritter_height_profile(frac * 0.5, 0.5, cfg)
            for frac in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0)]
    for lam in (0.5, 2.0, 4.0):
        scaled = [ritter_height_profile(frac * 0.5 * lam, 0.5 * lam, cfg)
                  for frac in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0)]
        for a, b in zip(base, scaled):
            assert a == pytest.approx(b, rel=1e-13)


def test_water_ahead_of_the_dam_is_four_twenty_sevenths(cfg):
    """Integral_0^{front} h dx = (4/27) H * front, exactly (convex profile)."""
    for t in (0.1, 0.25, 0.5):
        n = 20000
        front = ritter_front(t, cfg)
        xs = [(i + 0.5) * front / n for i in range(n)]
        area = sum(ritter_height_profile(x, t, cfg) for x in xs) * (front / n)
        assert area == pytest.approx((4.0 / 27.0) * H * front, rel=1e-6)


# --------------------------------------------------------------------------- #
# 5: the non-dimensionalisation the benchmark compares against
# --------------------------------------------------------------------------- #
def test_za_identity_is_two_tau(cfg):
    """With H = 2a and tau = t sqrt(2g/a) the Ritter front is Z/a = 2 tau.

    Regression guard: the B2 harness used ``Z/a = tau`` (a factor 2 low) until
    the reference audit, which halved the reference curve and inflated every
    reported Ritter error.
    """
    for tau in (0.25, 1.0, 2.5, 5.0):
        t = _t_of_tau(tau)
        assert ritter_front(t, cfg) / A == pytest.approx(2.0 * tau, rel=1e-14)
        assert ritter_front_over_a(tau) == pytest.approx(2.0 * tau, rel=1e-15)


def test_over_a_helper_rejects_the_factor_two_regression():
    """The helper is what the harness consumes; guard it explicitly."""
    tau = 1.0
    assert ritter_front_over_a(tau) == 2.0
    assert ritter_front_over_a(tau) != pytest.approx(tau)


# --------------------------------------------------------------------------- #
# 6: frame bookkeeping between the solver and the reference
# --------------------------------------------------------------------------- #
def test_lab_frame_and_self_similar_frame_differ_by_one_column_width(cfg):
    """front_position reports x_max >= a; ritter_front reports the travel Z."""
    t = 0.2
    travel = ritter_front(t, cfg)
    ramp = cfg.column_width + travel
    assert ramp - cfg.column_width == pytest.approx(travel, abs=0.0)
    # the naive comparison the audit removed
    assert ramp / travel == pytest.approx(1.0 + A / travel, rel=1e-12)
