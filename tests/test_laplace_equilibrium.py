"""Regression tests for the Young-Laplace *measurement protocol* (m2545).

``benchmarks/laplace_equilibrium`` replaces the protocol of the two earlier
verification rounds (``laplace_law`` m2174, ``csf_consistent`` m2447) by

* a density-consistent initial lattice (the shipped hex lattice carries
  ``rho/rho0 = 1.0059`` -> ``p = 2395 Pa = 6.6 sigma/R`` at t = 0), and
* a vacuum-referenced estimator ``Delta p = <p>_(r<0.7R)`` instead of the
  shipped ``0.80 R < r < 0.95 R`` shell band, which sits inside the free-surface
  density-deficiency layer where the Tait pressure is clamped to zero.

These tests pin the two protocol defects, the falsification of the standing
``p0 = sigma/R`` (Adami 2012) backlog item, and the residual model error, so the
verification record cannot silently drift.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_equilibrium" / "run.py"

SIGMA_R = 7.2 / 0.02      # 360 Pa, the analytic Young-Laplace jump


def _bench():
    spec = importlib.util.spec_from_file_location("laplace_equilibrium_run", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench()


@pytest.fixture(scope="module")
def run_coarse(bench):
    """Full window at dx = 3 mm (~10 s); shared by the physics assertions."""
    return bench.relax(0.003, consistent=True)


# --------------------------------------------------------------------------- #
# protocol defect 1: the shipped initial lattice is 6.6x over-pressured
# --------------------------------------------------------------------------- #
def test_shipped_lattice_is_over_pressured(bench):
    _, _, info = bench.lattice_droplet(0.003, consistent=False)
    assert 1.005 < info["rho_core_over_rho0"] < 1.007
    # p = B[(rho/rho0)^gamma - 1] with B = rho0 c0^2/gamma = 4e5/7
    assert 2200.0 < info["p_core_initial_Pa"] < 2600.0
    assert info["p_core_initial_Pa"] > 5.0 * SIGMA_R


def test_consistent_init_starts_at_zero_pressure(bench):
    _, mass, info = bench.lattice_droplet(0.003, consistent=True)
    assert abs(info["rho_core_over_rho0"] - 1.0) < 1e-9
    assert abs(info["p_core_initial_Pa"]) < 1e-6
    assert torch.all(mass > 0)


# --------------------------------------------------------------------------- #
# protocol defect 2: the shipped estimator samples the clamped deficiency layer
# --------------------------------------------------------------------------- #
def test_shell_band_estimator_samples_the_deficiency_layer(run_coarse):
    first = run_coarse["history"][0]
    assert first["rho_shell_band_over_rho0"] < 0.95
    assert first["frac_clamped_shell_band"] > 0.5
    # and it is not a pressure band at all: p is clamped to 0 there
    assert first["p_shell_band_Pa"] < 0.25 * SIGMA_R


# --------------------------------------------------------------------------- #
# falsification of backlog P0 item #1 (Tait background pressure p0 = sigma/R)
# --------------------------------------------------------------------------- #
def test_p0_bias_is_masked_by_the_zero_clamp(bench):
    a = bench.relax(0.003, consistent=True, steps=400)
    b = bench.relax(0.003, consistent=True, steps=400, p0=7.2 / 0.02)
    pa = a["final"]["p_interior_Pa"]
    pb = b["final"]["p_interior_Pa"]
    assert abs(pb - pa) < 0.02 * SIGMA_R, (pa, pb)
    # the clamp is what masks it: the shell band keeps p = 0 in both runs
    assert b["final"]["frac_clamped_shell_band"] > 0.3
    assert b["final"]["p_shell_band_Pa"] < 0.5 * SIGMA_R


# --------------------------------------------------------------------------- #
# residual model error (honest FAIL, pinned so it cannot drift silently)
# --------------------------------------------------------------------------- #
def test_jump_falls_short_of_young_laplace(run_coarse):
    rel = run_coarse["quasi_static"]["rel_error_pct"]
    final = run_coarse["final"]["p_interior_Pa"]
    assert final < SIGMA_R, "a passing jump means the model was fixed: update the record"
    assert -0.70 < rel / 100.0 < -0.35, rel


def test_delta_normalisation_decays_below_one(run_coarse):
    nd0 = run_coarse["history"][0]["n_delta"]
    nd1 = run_coarse["final"]["n_delta"]
    assert 0.5 < nd0 < 0.85, nd0          # already 15-35 % short of Brackbill's 1
    assert nd1 < 0.7 * nd0, (nd0, nd1)    # interface shell diffuses during the run


def test_jump_tracks_the_delta_normalisation(run_coarse):
    """In the quasi-static window the decaying jump tracks the decaying
    Brackbill delta-function normalisation almost linearly (measured slope
    ~1.9); that is the quantitative link between the two observables."""
    hist = run_coarse["history"]
    t_end = hist[-1]["t"]
    hist = [h for h in hist if h["t"] > 0.3 * t_end]
    x = [h["n_delta"] for h in hist]
    y = [h["p_interior_Pa"] / SIGMA_R for h in hist]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx
    r = sxy / math.sqrt(sxx * syy)
    assert r > 0.95, r
    assert 1.2 < slope < 2.6, slope


def test_spurious_currents_break_the_static_equilibrium(run_coarse):
    """Not a pass/fail criterion: the drop never becomes static, which is why
    the measured jump is time dependent (the reason for the WITHHELD grade)."""
    vmax = max(h["vmax_over_v_ref"] for h in run_coarse["history"])
    vend = run_coarse["final"]["vmax_over_v_ref"]
    assert vmax > 0.01
    assert vend > 0.01, vend