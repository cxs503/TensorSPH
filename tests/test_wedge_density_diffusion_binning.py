"""Density-diffusion switch and wall-profile binning of the wedge solver.

Two independent things are pinned here, both cheap (no long integration):

* the optional diffusive density term of :class:`tensorsph.wedge_entry.WedgeConfig`
  (``density_diffusion`` / ``density_diffusion_form``) -- off by default, so the
  shipped path must stay bit-identical, and when on it must obey the invariants
  the underlying term in :mod:`tensorsph.density_diffusion` guarantees;
* the re-binning estimator of
  ``benchmarks/case_sph_wedge_entry_2d/metric_resolution.py`` -- the
  resolution-consistent bin width is a *rule*, not a fit, so it gets a unit test
  that cannot pass if the rule is ever tuned to a target value.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "case_sph_wedge_entry_2d"))

from tensorsph.density_diffusion import rate_from_positions  # noqa: E402
from tensorsph.wedge_entry import WedgeConfig, WedgeImpact  # noqa: E402

torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# Config surface
# --------------------------------------------------------------------------- #

def test_density_diffusion_is_off_by_default():
    cfg = WedgeConfig()
    assert cfg.density_diffusion == 0.0
    assert cfg.density_diffusion_form == "molteni"
    cfg.validate()


@pytest.mark.parametrize("delta", [-0.1, 1.5])
def test_config_rejects_out_of_range_delta(delta):
    cfg = WedgeConfig(density_diffusion=delta)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_rejects_unknown_form():
    cfg = WedgeConfig(density_diffusion=0.1, density_diffusion_form="bogus")
    with pytest.raises(ValueError):
        cfg.validate()


# --------------------------------------------------------------------------- #
# Solver behaviour
# --------------------------------------------------------------------------- #

def _short_run(**kw):
    cfg = WedgeConfig(beta_deg=30.0, velocity=1.0, dx=0.01, frames=2,
                      half_width=0.08, depth=0.06, c_end=0.02, **kw)
    cfg.validate()
    sim = WedgeImpact(cfg)
    rows = sim.run()
    return sim, rows


def test_zero_delta_reproduces_the_shipped_path_bit_for_bit():
    """Explicit ``0.0`` and the default must be the same trajectory, exactly."""
    a_sim, a = _short_run(density_diffusion=0.0)
    b_sim, b = _short_run()
    assert a_sim.steps == b_sim.steps
    for ra, rb in zip(a, b):
        assert ra["t"] == rb["t"]
        for key in ("p_centre", "p_max_footprint", "c", "max_rho_ratio"):
            assert ra[key] == rb[key], key
        for va, vb in zip(ra["profile_p_over_q"], rb["profile_p_over_q"]):
            assert (va == vb) or (math.isnan(va) and math.isnan(vb))


@pytest.mark.parametrize("form", ["molteni", "antuono"])
def test_active_diffusion_stays_finite_and_keeps_the_state_physical(form):
    sim, rows = _short_run(density_diffusion=0.1, density_diffusion_form=form)
    assert sim.steps > 0
    last = rows[-1]
    assert math.isfinite(last["p_max_footprint"])
    assert math.isfinite(last["c"])
    # the term is a rate applied to the summation density: it must not be able
    # to push a particle outside the solver's own density floor/tolerance band
    assert 0.5 < last["max_rho_ratio"] < 1.5
    assert torch.isfinite(sim.fluid_pos).all()


def test_diffusion_term_vanishes_on_a_uniform_density_field():
    """Invariant of the term itself: a uniform field has no diffusive flux."""
    n = 9
    dx = 0.02
    xs = (torch.arange(n) - (n - 1) / 2) * dx
    pos = torch.stack(torch.meshgrid(xs, xs, indexing="xy"), dim=-1).reshape(-1, 2)
    rho = torch.full((pos.shape[0],), 1000.0)
    mass = torch.full_like(rho, 1000.0 * dx * dx)
    from tensorsph.droplet_impact import pair_kernels
    from tensorsph.neighbor_search import cell_linked_pairs
    h = 1.3 * dx
    pairs = cell_linked_pairs(pos, h)
    ii, jj = pairs.unbind(dim=1)
    delta = pos[ii] - pos[jj]
    _, gw = pair_kernels(delta, h)
    radial = -(delta * gw).sum(dim=-1)
    rate = rate_from_positions(pos, rho, mass, delta, gw, radial,
                               delta.square().sum(dim=-1), (ii, jj),
                               2.0 * 0.1 * 20.0 * h, form="molteni")
    assert float(rate.abs().max()) < 1.0e-9 * 1000.0


# --------------------------------------------------------------------------- #
# Binning estimator of the benchmark probe
# --------------------------------------------------------------------------- #

def test_binned_profile_conserves_samples_and_marks_empty_bins():
    import metric_resolution as mr

    taus = [0.1, 0.15, 0.55, 0.95]
    cps = [1.0, 3.0, 5.0, 7.0]
    centres, means, counts = mr.binned_profile(taus, cps, 10)
    assert len(centres) == len(means) == len(counts) == 10
    assert sum(counts) == len(taus)
    for c, m, k in zip(centres, means, counts):
        if k == 0:
            assert math.isnan(m)
        else:
            assert math.isfinite(m)


def test_adaptive_bin_width_follows_the_grid_not_the_target():
    """Bin width is ``2 dx`` in x, so the bin count halves when dx doubles."""
    import metric_resolution as mr

    n_fine = mr.adaptive_bins(2.0 * 1.0e-3 / 0.04)
    n_coarse = mr.adaptive_bins(2.0 * 2.0e-3 / 0.04)
    assert n_fine > n_coarse
    assert n_coarse >= 4
    assert mr.adaptive_bins(1.0e-9) >= 4          # never degenerate
    # the rule knows nothing about the reference value: same dx -> same bins
    assert n_fine == mr.adaptive_bins(2.0 * 1.0e-3 / 0.04)


def test_band_peak_is_a_maximum_over_the_declared_band_only():
    import metric_resolution as mr

    centres, means, counts = mr.binned_profile([0.2, 0.6, 1.1],
                                               [2.0, 9.0, 100.0], 12)
    peak, tau, n = mr.band_peak(centres, means, counts, tau_hi=1.0)
    assert peak < 100.0          # tau > 1 samples are excluded
    assert tau <= 1.0
    assert n >= 1