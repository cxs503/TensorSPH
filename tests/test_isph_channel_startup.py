"""Guards for the transient start-up verification of the ISPH channel.

Two layers:

* the exact solution itself (``analytic_startup_profile``) is checked for the
  identities that make it the solution of the start-up problem -- otherwise a
  wrong series would silently become the reference everything else is graded
  against;
* one short integration is checked against it, so the benchmark cannot rot away
  from the solver it grades.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.isph_channel import (
    ChannelConfig,
    analytic_startup_centerline,
    analytic_startup_profile,
    diffusive_time,
    run_case,
)


def steady(cfg: ChannelConfig, y):
    return cfg.body_force * y * (cfg.height - y) / (2.0 * cfg.viscosity)


def test_series_starts_from_rest():
    """t = 0 initial condition: ``u(y, 0) = 0`` everywhere.

    The series is written as ``u = u_s - transient`` and at ``t = 0`` the
    transient is the full Fourier expansion of ``y (H - y)``, i.e. ``u_s``.  So
    the identity this test really pins is the sine expansion
    ``y (H - y) = (8 H^2 / pi^3) sum_{n odd} sin(n pi y / H) / n^3``: a wrong
    prefactor or a wrong exponent leaves a residual of order ``u_s``.
    """
    cfg = ChannelConfig()
    y = torch.linspace(0.0, cfg.height, 41, dtype=torch.float64)
    u0 = analytic_startup_profile(y, 0.0, cfg.height, cfg.body_force,
                                  cfg.viscosity, n_terms=4000)
    assert float(torch.max(torch.abs(u0))) / cfg.u_max_analytic < 1e-3


def test_series_satisfies_the_diffusion_equation():
    """Independent PDE-residual check: ``u_t - nu u_yy - g = 0`` in the interior.

    Nothing here reuses the series; the residual is formed with central
    differences on the analytic field, so a wrong time constant or a wrong mode
    weight shows up as a non-zero residual.
    """
    cfg = ChannelConfig()
    nu, g, H = cfg.viscosity, cfg.body_force, cfg.height
    y = torch.linspace(0.15 * H, 0.85 * H, 15, dtype=torch.float64)
    t = 0.4 * diffusive_time(cfg)
    dt, dy = 1e-5 * H * H / nu, 1e-3 * H

    def u(tt, yy):
        return analytic_startup_profile(yy, tt, H, g, nu, n_terms=400)

    ut = (u(t + dt, y) - u(t - dt, y)) / (2 * dt)
    uyy = (u(t, y + dy) - 2 * u(t, y) + u(t, y - dy)) / dy ** 2
    residual = ut - nu * uyy - g
    assert float(torch.max(torch.abs(residual))) / g < 1e-4


def test_series_decays_to_the_steady_parabola_at_large_time():
    cfg = ChannelConfig()
    y = torch.linspace(0.05 * cfg.height, 0.95 * cfg.height, 11, dtype=torch.float64)
    u = analytic_startup_profile(y, 12.0 * diffusive_time(cfg), cfg.height,
                                 cfg.body_force, cfg.viscosity)
    ref = steady(cfg, y)
    rel = float(torch.max(torch.abs(u - ref)) / torch.max(ref))
    assert rel < 1e-4, f"series has not decayed to the parabola: {rel}"


def test_series_satisfies_no_slip_walls_exactly():
    cfg = ChannelConfig()
    y = torch.tensor([0.0, cfg.height], dtype=torch.float64)
    for t_over_tau in (0.0, 0.1, 1.0, 5.0):
        u = analytic_startup_profile(y, t_over_tau * diffusive_time(cfg),
                                     cfg.height, cfg.body_force, cfg.viscosity)
        assert float(torch.max(torch.abs(u))) < 1e-9


def test_series_is_monotone_in_time_and_positive():
    """Start-up from rest can only accelerate towards the steady state."""
    cfg = ChannelConfig()
    times = [f * diffusive_time(cfg) for f in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    vals = [analytic_startup_centerline(t, cfg) for t in times]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] <= cfg.u_max_analytic * (1.0 + 1e-9)
    assert vals[0] > 0.0


def test_series_truncation_is_converged():
    cfg = ChannelConfig()
    t = 0.3 * diffusive_time(cfg)
    a = analytic_startup_centerline(t, cfg, n_terms=50)
    b = analytic_startup_centerline(t, cfg, n_terms=800)
    assert abs(a - b) / b < 1e-8


def test_diffusive_time_matches_the_slowest_series_mode():
    """At ``t = tau`` the centreline deficit is ``(e^-1 - e^-9/27 + ...)/(pi^3/32)``.

    The ``pi^3/32`` is the t = 0 value of the same alternating sine sum, so this
    is a closed form derived from the series identity, not a fit.
    """
    cfg = ChannelConfig()
    y = torch.tensor([0.5 * cfg.height], dtype=torch.float64)
    ref = steady(cfg, y).item()
    u0 = analytic_startup_profile(y, 0.0, cfg.height, cfg.body_force, cfg.viscosity,
                                  n_terms=2000).item()
    u_tau = analytic_startup_profile(y, diffusive_time(cfg), cfg.height,
                                     cfg.body_force, cfg.viscosity,
                                     n_terms=2000).item()
    measured = (ref - u_tau) / (ref - u0)
    expected = (math.exp(-1) - math.exp(-9) / 27 + math.exp(-25) / 125) / (math.pi ** 3 / 32)
    assert abs(measured - expected) / expected < 1e-3, (measured, expected)


def test_trace_exposes_the_centreline_readout():
    cfg = ChannelConfig(resolution=8, columns=6, t_end=0.2 * diffusive_time(ChannelConfig()),
                        t_average=0.01 * ChannelConfig().height)
    result = run_case(cfg, torch.device("cpu"))
    keys = {k for row in result["trace"] for k in row}
    assert "u_centre_rows" in keys
    vals = [row["u_centre_rows"] for row in result["trace"]]
    assert all(math.isfinite(v) for v in vals)


def test_short_startup_run_tracks_the_exact_series():
    """Integration guard: one cheap run, 15 % acceptance band on the centreline.

    The measured value is the row-averaged centreline of the trace, compared to
    the exact series at the same instant.  The band is the platform acceptance
    criterion; the benchmark ladder resolves the actual numbers.
    """
    cfg = ChannelConfig(resolution=8, columns=6, t_end=2.0 * diffusive_time(ChannelConfig()),
                        t_average=0.01 * ChannelConfig().height)
    result = run_case(cfg, torch.device("cpu"))
    rows = [r for r in result["trace"] if r["t"] >= 0.9 * cfg.t_end]
    assert rows, "no trace samples in the last 10 % of the run"
    measured = sum(r["u_centre_rows"] for r in rows) / len(rows)
    exact = analytic_startup_centerline(rows[-1]["t"], cfg)
    rel = abs(measured - exact) / exact
    assert rel < 0.15, f"centreline off by {100 * rel:.1f} % at t = {rows[-1]['t']:.4f} s"


@pytest.mark.parametrize("n_terms", [1, 2])
def test_series_is_finite_for_a_truncated_sum(n_terms):
    cfg = ChannelConfig()
    val = analytic_startup_centerline(0.1 * diffusive_time(cfg), cfg, n_terms=n_terms)
    assert math.isfinite(val) and val >= 0.0
