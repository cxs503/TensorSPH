"""Guards for the Young-Laplace measurement protocol and the density bias.

Three things are pinned here, each of which was silently wrong before:

1. ``lattice_kernel_sum`` is the exact factor by which the raw density
   summation over-estimates ``rho0`` on a hexagonal lattice, so the calibration
   ``density_normalization="lattice"`` removes the bulk offset instead of
   hiding it (a resting drop must read ``p = 0``, not 2.4 kPa).
2. The Laplace jump has to be measured as ``p_inside - p_outside``. The
   historical ``p_inside - p(0.80R..0.95R)`` band lies inside the liquid and
   therefore understates the jump by construction; a synthetic field with a
   known 360 Pa jump checks both metrics.
3. A relaxed drop under the corrected protocol reproduces ``sigma / R`` within
   the platform's 15 % gate, while the same run read with the historical band
   does not -- the pass/fail verdict of the end-to-end criterion was decided by
   the definition, not by the physics.

See ``docs/verification_laplace_protocol.md``.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    lattice_kernel_sum,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_protocol"
RADIUS = 0.02
SIGMA = 7.2
RHO0 = 1000.0
C0 = 20.0
DP_REF = SIGMA / RADIUS


def _load_bench():
    spec = importlib.util.spec_from_file_location("laplace_protocol_run",
                                                  BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


def _resting_drop(dx=0.003, h_factor=1.3, normalization="lattice"):
    h = h_factor * dx
    pos, mass = bench.hexagonal_droplet(RADIUS, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    if normalization == "lattice":
        rho = rho / lattice_kernel_sum(h, dx)
    return pos, mass, pairs, rho


def test_lattice_sum_is_the_bulk_density_bias():
    """``lattice_kernel_sum`` reproduces the measured bulk summation ratio."""
    dx = 0.003
    pos, mass, _pairs, rho = _resting_drop(dx, normalization="summation")
    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * RADIUS
    measured = float(rho[core].mean() / RHO0)
    predicted = lattice_kernel_sum(1.3 * dx, dx)
    assert measured == pytest.approx(predicted, rel=2e-3)
    # The default h = 1.3 dx is biased high: +0.59 % in density, which a
    # gamma = 7 Tait law turns into +4.2 % in pressure.
    assert predicted > 1.0
    assert (predicted - 1.0) * 7.0 == pytest.approx(0.041, abs=2e-3)


def test_lattice_normalisation_removes_the_resting_pressure_offset():
    """A resting lattice must hold no pressure; the raw summation holds 2.4 kPa."""
    dx = 0.003
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=7.0,
                 gravity=0.0, density_normalization="summation")
    _pos, _mass, _pairs, rho_raw = _resting_drop(dx, normalization="summation")
    offset = float(tait_pressure(rho_raw, cfg).max())
    assert offset > 2000.0, "the raw summation offset should be about 2395 Pa"

    cfg.density_normalization = "lattice"
    pos, mass, pairs, _rho = _resting_drop(dx)
    _rho, pressure, _acc = rhs(pos, torch.zeros_like(pos), mass, cfg, pairs=pairs)
    assert float(pressure.max()) == 0.0

    # And the whole point: 2395 Pa is 6.6x the 360 Pa the drop must hold.
    assert offset / DP_REF > 6.0


def test_corrected_metric_recovers_the_jump_the_legacy_band_hides():
    """Synthetic field with an exact 360 Pa jump through a 1-shell interface."""
    dx = 0.003
    pos, _mass = bench.hexagonal_droplet(RADIUS, dx, RHO0)
    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
    # A CSF-like smeared jump: 360 Pa inside, linear ramp to 0 across the
    # outermost shell (0.90 .. 1.00 R), 0 outside.
    ramp = ((1.0 - r) / 0.10).clamp(0.0, 1.0)
    pressure = DP_REF * ramp
    p_in, _ = bench.band_mean(pressure, r, 0.0, 0.5)
    p_leg, n_leg = bench.band_mean(pressure, r, 0.80, 0.95)
    p_out, n_out = bench.band_mean(pressure, r, 1.02, 1.20)
    assert n_leg > 0
    assert n_out == 0, "a free drop has no particles outside; exterior p is 0"
    assert math.isnan(p_out)
    assert p_in == pytest.approx(DP_REF, rel=1e-9)
    corrected = p_in - 0.0
    legacy = p_in - p_leg
    assert corrected == pytest.approx(DP_REF, rel=1e-6)
    assert legacy < 0.75 * DP_REF, "the legacy band must understate the jump"


@pytest.mark.slow
def test_relaxed_drop_satisfies_young_laplace_with_the_corrected_protocol():
    """End-to-end: corrected reading inside the 15 % gate, legacy reading not."""
    case, _hist, _pos, _press, _prof = bench.relax(
        dx=0.003, normalization="lattice", t_over_t_sigma=2.0, quiet=True)
    final = case["final"]
    corrected = final["rel_error_corrected_pct"]
    legacy = final["rel_error_legacy_pct"]
    assert abs(corrected) < 15.0, (
        f"corrected Young-Laplace error {corrected:+.1f}% is outside the gate")
    assert legacy < -20.0, (
        "the legacy band definition should understate the jump; if it no longer "
        "does, the measurement protocol changed and the docs need updating")
    assert final["p_outside_Pa"] == 0.0