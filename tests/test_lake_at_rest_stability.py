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
BAND_ARTEFACT = (Path(__file__).parent / "data" / "lake_at_rest"
                 / "stability_band_fit.json")
TOLERANCE = 1.0e-9
#: Declared exponential band of ``benchmarks/lake_at_rest/band_fit.py``: the
#: window in which the departure of *every* arm is exponential (seven decades
#: above the roundoff floor, one below saturation).  The legacy single window
#: ``1e-10 .. 1e-5`` is kept only to show that it is not a rate estimator for an
#: arm that starts moving -- its fit spans the start-up transient and returns
#: ``R^2 < 0.6`` there.
BAND = (1.0e-6, 1.0e-3)
R2_MIN = 0.98


def samples_of(entry):
    """Raw series of an arm (shipped at arm level; older form under ``run``)."""
    series = entry.get("samples") or entry.get("run", {}).get("samples")
    assert series, f"arm {entry.get('arm')} carries no raw series"
    return series


@pytest.fixture(scope="module")
def band_artefact():
    assert BAND_ARTEFACT.is_file(), (
        f"{BAND_ARTEFACT} is missing; reproduce with "
        "`PYTHONPATH=src python3 benchmarks/lake_at_rest/band_fit.py`")
    return json.loads(BAND_ARTEFACT.read_text())


def band_arm(band_artefact, name):
    for entry in band_artefact["arms"]:
        if entry["arm"] == name:
            return entry
    raise AssertionError(f"arm {name} missing from the band artefact")


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
        assert samples_of(entry)


def test_reported_horizon_follows_from_the_raw_series(artefact):
    """t* recorded in the verdict must be the first crossing of the tolerance."""
    for entry in artefact["arms"]:
        samples = samples_of(entry)
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
        slope, r2, points = fit_lambda(samples_of(entry), low, high)
        recorded = entry["growth"]["lambda_per_s"]
        if recorded is None:
            assert slope is None, entry["arm"]
            continue
        assert points >= 4
        assert math.isclose(slope, recorded, rel_tol=1e-9), entry["arm"]
        assert r2 is not None
        if entry["run"]["seed_in_dx"]:
            # A seeded arm enters the legacy window at t = dt: the fit averages
            # the start-up transient (measured R^2 0.29 / 0.58).  Those numbers
            # may be recorded, they may not be quoted as a rate -- the rate is
            # the band estimate gated below.
            assert r2 < R2_MIN, (
                f"{entry['arm']}: legacy-window R^2={r2:.3f} -- if a seeded arm "
                f"now fits this window, the estimator changed; revisit it")


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


def test_halving_dt_keeps_the_growth_rate(artefact, band_artefact):
    """Declared window: the rate is a property of the equilibrium, not of dt."""
    base = arm(artefact, "base")
    half = arm(artefact, "courant_half")
    ratio = (band_arm(band_artefact, "courant_half")["band"]["lambda_per_s"]
             / band_arm(band_artefact, "base")["band"]["lambda_per_s"])
    assert 0.75 <= ratio <= 1.33, ratio
    assert band_artefact["verdict"]["dt_independent"] is True
    assert half["run"]["dt"] == pytest.approx(0.5 * base["run"]["dt"], rel=1e-9)


def test_band_fit_is_reproducible_from_the_raw_series(artefact, band_artefact):
    """The band estimate must be the fit of the shipped series, arm by arm.

    Re-derived with this file's own least squares (not the producer's), so a
    frozen number cannot be a transcription.
    """
    assert band_artefact["band"]["window"] == list(BAND)
    for entry in band_artefact["arms"]:
        raw = samples_of(arm(artefact, entry["arm"]))
        slope, r2, points = fit_lambda(raw, *BAND)
        recorded = entry["band"]["lambda_per_s"]
        if recorded is None:
            continue
        assert points >= 4
        assert math.isclose(slope, recorded, rel_tol=1e-9), entry["arm"]
        assert r2 >= R2_MIN, (entry["arm"], r2)
        assert entry["band"]["valid"] is True


def test_seed_amplitude_does_not_change_the_rate(band_artefact):
    """Declared window: linear instability means the rate is seed-independent.

    Measured in the exponential band.  Over the legacy window the two seeded
    arms give ``0.887`` and ``3.72`` 1/s (``R^2`` 0.29 / 0.58): a transient
    average, which is why this gate failed when it used that window.
    """
    coarse = band_arm(band_artefact, "seed_1e-6")["band"]["lambda_per_s"]
    fine = band_arm(band_artefact, "seed_1e-8")["band"]["lambda_per_s"]
    assert coarse is not None and fine is not None
    ratio = coarse / fine
    assert 0.90 <= ratio <= 1.10, (coarse, fine, ratio)
    base = band_arm(band_artefact, "base")["band"]["lambda_per_s"]
    for lam in (coarse, fine):
        assert abs(lam / base - 1.0) <= 0.10, (lam, base)


def test_legacy_window_is_rejected_where_it_is_a_transient(band_artefact):
    """A slope fitted across the start-up transient must not ship as a rate."""
    assert set(band_artefact["verdict"]["legacy_window_invalid_arms"]) == {
        "seed_1e-6", "seed_1e-8"}, band_artefact["verdict"]
    for name in ("seed_1e-6", "seed_1e-8"):
        entry = band_arm(band_artefact, name)
        assert entry["legacy"]["r2_exp"] < R2_MIN
        assert entry["legacy"]["lambda_per_s"] < 0.5 * entry["band"]["lambda_per_s"]


def test_band_rate_is_not_a_window_artefact(band_artefact):
    """Four declared windows must give the same rate on the base arm (< 5 %)."""
    spread = band_arm(band_artefact, "base")["sensitivity"]["spread_rel"]
    assert spread is not None and spread < 0.05, spread


def test_rebuilding_the_pair_list_does_not_remove_it(band_artefact):
    """The frozen neighbour list is not what manufactures the departure."""
    base = band_arm(band_artefact, "base")["band"]["lambda_per_s"]
    rebuilt = band_arm(band_artefact, "pairs_rebuilt")["band"]["lambda_per_s"]
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
                         max_courant=courant, t_end=3.0)
        state = build_state(cfg)
        solution = reconstruct(state, cfg, prior=analytic_profile(state, cfg)["mass"])
        assert solution["a_max_over_g"] < 1e-9        # still the certified rest
        mass = solution["mass"]
        pos = state["pos"].clone()
        pos[:, 1] = pos[:, 1] + 1.0e-8 * cfg.dx      # seed in units of dx
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
        reports[courant] = (fit_lambda(samples, *BAND)[0],
                            float((pos[:, 1] - y0).abs().max()) / cfg.depth)
    lam_fine_dt, dy_fine_dt = reports[0.30]
    lam_coarse_dt, _ = reports[0.15]
    assert lam_fine_dt is not None and lam_coarse_dt is not None
    assert lam_fine_dt > 0 and lam_coarse_dt > 0        # it departs
    assert dy_fine_dt > TOLERANCE                       # and it is visible
    assert 0.8 <= lam_coarse_dt / lam_fine_dt <= 1.25   # no dt-driven rate