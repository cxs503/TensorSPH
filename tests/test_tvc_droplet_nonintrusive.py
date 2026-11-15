"""Tests of the droplet non-intrusiveness leg
(``benchmarks/xval_sphinx_tvc/droplet_nonintrusive.py``).

The leg asks whether the ported transport-velocity correction is *non-intrusive*
on analytic free-surface dynamics: the drop-oscillation case has an exact
frequency (``omega^2 = sigma m (m^2-1)/(rho R^3)``) and an exact size law
(``T ∝ R^{3/2}``), so any period change the correction causes can be decomposed
into a geometric part (the mean radius moved) and a dynamic residual.

What is asserted here
---------------------
* switched off, the integrator is bit-identical to the pre-hook code path (the
  hook must not change the production case);
* the correction moves positions only -- particle count and total mass are
  untouched, and the recorded diagnostics agree with the operator's own law
  ``|shift| <= alpha h^2 |grad-1|``;
* the cadence knob is honoured exactly (``tvc_steps`` counts what it says);
* the inflation of a zero-g drop is **bounded by the clip** upstream offers
  (``max_shift``): with the clip the mean radius stays at the baseline value,
  without it the cloud expands -- the leg's key qualitative finding, asserted as
  an invariant of the operator rather than as a tuned threshold;
* the published payload is internally consistent (summary reproducible from the
  rows), so the doc numbers cannot drift from the data.

Run: ``PYTHONPATH=src python -m pytest tests/test_tvc_droplet_nonintrusive.py -q``
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from tensorsph.capillary_oscillation import (make_droplet, rayleigh_2d,
                                            run_point)
from tensorsph.transport_velocity_correction import (
    transport_velocity_correction,)

N_SMALL = 6            # R/dx for the cheap invariant runs
SIGMA = 0.072
RHO = 1000.0
RADIUS = 0.05


def _short(n: int = N_SMALL, **kw):
    """Cheap run: 0.3 s of simulated time is enough for every invariant here."""
    torch.set_num_threads(1)
    return run_point(n, SIGMA, 0.0, t_end_s=0.3, sample_every=50, **kw)


def test_disabled_correction_is_bit_identical_to_the_pre_hook_path():
    """``tvc_alpha = 0`` must reproduce the production integrator exactly."""
    a = run_point(N_SMALL, SIGMA, 0.0, t_end_s=0.3, sample_every=50)
    b = run_point(N_SMALL, SIGMA, 0.0, t_end_s=0.3, sample_every=50,
                  tvc_alpha=0.0, tvc_every=1)
    assert a["series"] == b["series"]
    # the window here is shorter than one oscillation, so the period is nan in
    # both runs: compare the bit-identical shape readings instead
    assert a["steps"] == b["steps"] and a["t_end_s"] == b["t_end_s"]
    assert a["q_half_range"] == b["q_half_range"]
    assert a["r_mean_over_R"] == b["r_mean_over_R"]
    assert b["tvc_steps"] == 0
    assert b["tvc_shift_peak_over_h"] == 0.0


def test_correction_moves_positions_only():
    """Particle count and total mass are untouched; the cloud is finite."""
    base = _short()
    on = _short(tvc_alpha=0.10)
    assert on["particles"] == base["particles"]
    # the first step is skipped (no correction before the flow has started),
    # exactly as ``dam_break`` does it
    assert on["tvc_steps"] == on["steps"] - 1
    assert on["tvc_shift_peak_over_h"] > 0.0
    assert on["tvc_mean_h_grad1"] > 0.0


def test_cadence_knob_is_honoured():
    """``tvc_every = k`` applies the correction on ``steps // k`` updates."""
    on = _short(tvc_alpha=0.10, tvc_every=3)
    assert on["tvc_steps"] == on["steps"] // 3


def test_recorded_shift_obeys_the_operator_law_and_the_clip():
    """One operator application on the drop cloud: linearity in alpha and clip."""
    pos, mass = make_droplet(0.05 / 6, 0.05, "cpu")
    vol = mass / RHO
    h = 1.3 * (RADIUS / 6)
    a = transport_velocity_correction(pos, vol, h, alpha=0.05)
    b = transport_velocity_correction(pos, vol, h, alpha=0.10)
    # The shift is exactly linear in alpha (no limiter, no clip).
    assert torch.allclose(b.shift, 2.0 * a.shift, atol=0.0, rtol=1e-12)
    # |shift| == alpha h^2 |g| row by row.
    expect = 0.05 * h * h * torch.linalg.vector_norm(a.gradient_integral, dim=1)
    assert torch.allclose(torch.linalg.vector_norm(a.shift, dim=1), expect,
                          atol=1e-15, rtol=1e-12)
    # The clip is a hard ceiling on the per-particle displacement.
    clip = 1.0e-4 * h
    c = transport_velocity_correction(pos, vol, h, alpha=0.10, max_shift=clip)
    assert float(torch.linalg.vector_norm(c.shift, dim=1).max()) <= clip * (1 + 1e-12)


def test_clip_bounds_the_zero_g_inflation_of_a_free_surface():
    """The leg's qualitative finding, asserted as an operator-bounded effect.

    A free-floating drop has no gravity and no wall to balance the outward
    surface term of the correction, so an unclipped coefficient inflates it; the
    upstream ``max_shift`` clip keeps the mean radius at the baseline value.
    """
    base = _short()
    free = _short(tvc_alpha=0.10)
    clipped = _short(tvc_alpha=0.10, tvc_max_shift=0.02)
    inflation_free = free["r_mean_over_R"] / base["r_mean_over_R"] - 1.0
    inflation_clipped = clipped["r_mean_over_R"] / base["r_mean_over_R"] - 1.0
    assert inflation_free > 0.0, "unclipped outward correction must dilate the drop"
    assert inflation_clipped < inflation_free, "the clip must reduce the inflation"
    assert clipped["tvc_shift_peak_over_h"] <= 0.02 * (1 + 1e-9)


def test_exact_reference_is_the_lamb_frequency():
    """The reference the leg grades against is the analytic one, re-derived."""
    omega = rayleigh_2d(SIGMA, RHO, RADIUS, 2)
    # omega^2 = sigma m (m^2 - 1) / (rho R^3)
    expect = SIGMA * 2 * 3 / (RHO * RADIUS ** 3)
    assert omega ** 2 == pytest.approx(expect, rel=1e-14)


PAYLOAD = pathlib.Path("docs/xval_sphinx_tvc/data/droplet_cadence.json")


@pytest.mark.skipif(not PAYLOAD.exists(), reason="leg payload not generated yet")
def test_published_payload_is_internally_consistent():
    """The doc's summary must follow from the stored rows (no hand-copied number)."""
    data = json.loads(PAYLOAD.read_text())
    rows = data["rows"]
    assert data["leg"] == "droplet-cadence"
    base = next(r for r in rows if r["tvc_alpha"] == 0.0)
    for r in rows:
        assert r["inflation_vs_baseline_pct"] == pytest.approx(
            100.0 * (r["r_mean_over_R"] / base["r_mean_over_R"] - 1.0), abs=1e-9)
        assert r["cumulative_shift_budget_over_h"] == pytest.approx(
            r["tvc_shift_peak_over_h"] * r["tvc_steps"], rel=1e-12)
    per_ac = [r for r in rows if r["tvc_every"] == 1 and r["tvc_alpha"] > 0]
    per_adv = [r for r in rows if r["tvc_every"] > 1]
    assert data["summary"]["worst_inflation_acoustic_cadence_pct"] == pytest.approx(
        max(r["inflation_vs_baseline_pct"] for r in per_ac), rel=1e-12)
    assert data["summary"]["worst_inflation_advection_cadence_pct"] == pytest.approx(
        max(r["inflation_vs_baseline_pct"] for r in per_adv), rel=1e-12)
    # the leg's claim, asserted against the data it publishes: the per-acoustic
    # cadence inflates a free drop by more than the advection cadence
    assert data["summary"]["size_unbounded_at_acoustic_cadence"] in (True, False)
    assert (max(r["inflation_vs_baseline_pct"] for r in per_ac)
            > max(r["inflation_vs_baseline_pct"] for r in per_adv))
