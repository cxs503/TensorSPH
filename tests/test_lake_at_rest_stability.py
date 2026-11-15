"""Gates for the rest-horizon study of ``CASE-SPH-LAKE-AT-REST``.

The published PASS (``docs/verification_lake_at_rest_periodic.md``) certifies
``max|v|/c0 = 2.6e-14`` over ``t_end = 0.50 s``.  ``docs/verification_lake_at_rest_horizon.md``
measures the horizon of that statement.  This file is the machine-decidable
part of it, and it is written so that it can fail:

* the shipped artefact ``tests/data/lake_at_rest/stability.json`` must be
  *internally re-derivable*: the tolerance horizon and the growth rate are
  recomputed here from the raw per-sample series and compared with the recorded
  verdict (no hand-copied number is trusted);
* the growth rate is an exponential fit in the declared window and must be a
  good fit (``R^2 >= 0.98``) -- otherwise the ``lambda`` of the artefact is a
  meaningless slope;
* two controls that decide the *interpretation* are asserted with windows that
  were declared before the run: halving ``dt`` keeps ``lambda`` within 25 %
  (a mode of the discrete equilibrium, not an integrator artefact) and the seed
  amplitude does not change ``lambda`` (linear instability) within 10 %;
* one fast re-derivation runs the physics here, at a coarse lattice, without the
  artefact: a seeded lake must depart exponentially, and the rate must be
  ``dt``-independent there too.

If someone fixes the operator so the lake stays still, the fast re-derivation
and the internal-consistency gates keep working and only the direction of the
finding changes; nothing here asserts a timing that could not be re-measured.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import rhs
from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                   reconstruct)

ARTEFACT = Path(__file__).parent / "data" / "lake_at_rest" / "stability.json"
TOLERANCE = 1.0e-9


def fit_lambda(samples: list[dict], low: float, high: float,
               field: str = "v_over_c0"):
    """Least-squares slope of ``ln(field)`` against ``t`` in the window."""
    window = [(s["t"], s[field]) for s in samples
              if low <= s[field] <= high and s[field] > 0]
    if len(window) < 4:
        return None, None, 0
    n = len(window)
    t = [w[0] for w in window]
    y = [math.log(w[1]) for w in window]
    mt, my = sum(t) / n, sum(y) / n
    sxx = sum((ti - mt) ** 2 for ti in t)
    sxy = sum((ti - mt) * (yi - my) for ti, yi in zip(t, y))
    slope = sxy / sxx
    icpt = my - slope * mt
    ss_res = sum((yi - (icpt + slope * ti)) ** 2 for ti, yi in zip(t, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    return slope, (1.0 - ss_res / ss_tot if ss_tot else None), n


@pytest.fixture(scope="module")
def artefact():
    assert ARTEFACT.is_file(), (
        f"{ARTEFACT} is missing; reproduce with "
        "`PYTHONPATH=src python3 benchmarks/lake_at_rest/stability.py "
        "--json tests/data/lake_at_rest/stability.json --figures docs/figures`")
    return json.loads(ARTEFACT.read_text())


def arm(artefact, name):
    for entry in artefact["arms"]:
        if entry["arm"] == name:
            return entry
    raise AssertionError(f"arm {name} missing from the artefact")


def test_artefact_is_the_declared_case(artefact):
    assert artefact["case"] == "CASE-SPH-LAKE-AT-REST"
    assert artefact["tolerance"] == TOLERANCE
    low, high = artefact["fit_window"]
    assert low < high
    for entry in artefact["arms"]:
        assert entry["run"]["steps"] > 0
        assert entry["run"]["samples"]


def test_reported_horizon_follows_from_the_raw_series(artefact):
    """t* recorded in the verdict must be the first crossing of the tolerance."""
    for entry in artefact["arms"]:
        samples = entry["run"]["samples"]
        crossing = None
        worst = 0.0
        for sample in samples:
            worst = max(worst, sample["v_over_c0"])
            if crossing is None and worst > TOLERANCE:
                crossing = sample["t"]
        recorded = entry["run"]["certified_horizon_s"]["v"]
        if crossing is None:
            assert recorded is None, entry["arm"]
        else:
            assert recorded is not None, entry["arm"]
            # the running worst is sampled, so the recorded crossing must sit
            # inside the first interval that contains a crossing sample
            step = entry["run"]["sample_every"] * entry["run"]["dt"]
            assert crossing - step - 1e-12 <= recorded <= crossing + 1e-12, (
                entry["arm"], recorded, crossing)


def test_growth_rate_is_reproducible_from_the_raw_series(artefact):
    low, high = artefact["fit_window"]
    for entry in artefact["arms"]:
        slope, r2, points = fit_lambda(entry["run"]["samples"], low, high)
        recorded = entry["growth"]["lambda_per_s"]
        if recorded is None:
            assert slope is None, entry["arm"]
            continue
        assert points >= 4
        assert math.isclose(slope, recorded, rel_tol=1e-9), entry["arm"]
        assert r2 is not None and r2 >= 0.98, (entry["arm"], r2)


def test_departure_is_exponential_not_power_law(artefact):
    """The declared window must be described by the exponential law it reports."""
    base = arm(artefact, "base")
    assert base["growth"]["lambda_per_s"] is not None
    assert base["growth"]["lambda_per_s"] > 0
    assert base["growth"]["r2_exp"] >= 0.98
    # a power law is the alternative hypothesis; it must fit worse
    assert base["growth"]["r2_exp"] > base["growth"]["r2_power"]


def test_the_published_horizon_is_not_the_horizon(artefact):
    """The 0.50 s PASS must not be reported as a bound on how long rest lasts."""
    base = arm(artefact, "base")
    assert base["run"]["t_end"] >= 2.0
    horizon = base["run"]["certified_horizon_s"]["v"]
    assert horizon is not None, "the rest state persisted; revisit the finding"
    assert horizon < base["run"]["t_end"]
    assert artefact["verdict"]["persists_to_claimed_horizon"] is True
    assert base["run"]["max_v_over_c0"] > 1e-6      # it really does move


def test_halving_dt_keeps_the_growth_rate(artefact):
    """Declared window: the rate is a property of the equilibrium, not of dt."""
    base = arm(artefact, "base")
    half = arm(artefact, "courant_half")
    ratio = half["growth"]["lambda_per_s"] / base["growth"]["lambda_per_s"]
    assert 0.75 <= ratio <= 1.33, ratio
    assert half["run"]["dt"] == pytest.approx(0.5 * base["run"]["dt"], rel=1e-9)


def test_seed_amplitude_does_not_change_the_rate(artefact):
    """Declared window: linear instability means the slope is seed-independent."""
    coarse = arm(artefact, "seed_1e-6")["growth"]["lambda_per_s"]
    fine = arm(artefact, "seed_1e-8")["growth"]["lambda_per_s"]
    assert coarse is not None and fine is not None
    assert 0.90 <= coarse / fine <= 1.10, (coarse, fine)


def test_rebuilding_the_pair_list_does_not_remove_it(artefact):
    """The frozen neighbour list is not what manufactures the departure."""
    base = arm(artefact, "base")["growth"]["lambda_per_s"]
    rebuilt = arm(artefact, "pairs_rebuilt")["growth"]["lambda_per_s"]
    assert rebuilt is not None
    assert rebuilt / base >= 0.5


def test_saturation_reaches_a_finite_amplitude(artefact):
    """After the exponential phase the lake must move at a visible amplitude."""
    long_run = arm(artefact, "saturation_8s")
    assert long_run["run"]["t_end"] >= 8.0
    assert long_run["run"]["max_dy_over_H"] > 1.0e-3
    assert long_run["run"]["max_v_over_c0"] > 1.0e-3


def test_coarse_lattice_rederivation():
    """Fast, artefact-free: a seeded lake departs, and dt does not set the rate."""
    torch.set_default_dtype(torch.float64)
    reports = {}
    for courant in (0.30, 0.15):
        cfg = LakeConfig(dx=0.04, depth=0.24, width=0.32, periodic_x=True,
                         max_courant=courant, t_end=0.30)
        state = build_state(cfg)
        solution = reconstruct(state, cfg, prior=analytic_profile(state, cfg)["mass"])
        assert solution["a_max_over_g"] < 1e-9        # still the certified rest
        mass = solution["mass"]
        pos = state["pos"].clone()
        pos[:, 1] = pos[:, 1] + 1.0e-6 * cfg.dx      # seed in units of dx
        vel = torch.zeros_like(pos)
        dt = cfg.dt()
        steps = int(round(cfg.t_end / dt))
        y0 = state["pos"][:, 1].clone()
        samples = []
        for step in range(steps):
            _rho, _p, acc = rhs(pos, vel, mass, state["config"],
                                pairs=state["pairs"], wall=state["wall"],
                                wrap_width=state["wrap_width"])
            vel = vel + dt * acc
            pos = pos + dt * vel
            samples.append({"t": (step + 1) * dt,
                            "v_over_c0": float(vel.abs().max()) / cfg.sound_speed})
        reports[courant] = (fit_lambda(samples, 1e-10, 1e-5)[0],
                            float((pos[:, 1] - y0).abs().max()) / cfg.depth)
    lam_fine_dt, dy_fine_dt = reports[0.30]
    lam_coarse_dt, _ = reports[0.15]
    assert lam_fine_dt is not None and lam_coarse_dt is not None
    assert lam_fine_dt > 0 and lam_coarse_dt > 0        # it departs
    assert dy_fine_dt > TOLERANCE                       # and it is visible
    assert 0.5 <= lam_coarse_dt / lam_fine_dt <= 1.6    # no dy-driven rate