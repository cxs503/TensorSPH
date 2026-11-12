"""pytest cases for the standalone Taylor-Green benchmark (TensorSPH).

The verification ladder already pins the TGV decay rate against the analytic
``4 nu k^2`` (rung C2).  This module adds the *benchmark* claims that the
standalone driver ``benchmarks/taylor_green/run.py`` makes: resolution
convergence and the Mach-number validity window.
"""
from __future__ import annotations

import pytest
import torch

from tensorsph.vv.tgv import TGVConfig, run_tgv


@pytest.fixture(scope="module", autouse=True)
def _bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_tgv_converges_with_resolution():
    """The decay-rate error must shrink as dx shrinks (second-order-ish)."""
    err_coarse = run_tgv(TGVConfig(dx=0.05, t_end=0.6, U0=0.2, sound_speed=10.0))["error_pct"]
    err_fine = run_tgv(TGVConfig(dx=0.025, t_end=0.6, U0=0.2, sound_speed=10.0))["error_pct"]
    assert err_fine < err_coarse, (err_coarse, err_fine)
    assert err_fine < 8.0, err_fine  # the benchmark's tolerance


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_tgv_mach_window_breaks_above_0_05():
    """The decay-rate error is flat for M <= 0.02 and collapses above M ~ 0.05."""
    err_low = run_tgv(TGVConfig(dx=0.05, t_end=0.6, U0=0.02, sound_speed=10.0))["error_pct"]
    err_high = run_tgv(TGVConfig(dx=0.05, t_end=0.6, U0=0.5, sound_speed=10.0))["error_pct"]
    assert err_low < 8.0, err_low
    assert err_high > 50.0, err_high  # the window has collapsed
