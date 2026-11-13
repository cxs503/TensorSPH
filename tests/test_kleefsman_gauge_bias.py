"""Gauge-operator tests for the Kleefsman dam-break-with-obstacle case.

The case-pool acceptance is a percentage against an experiment, so the *gauge*
itself has to be free of a resolution-dependent bias larger than the tolerance
it is judged with.  These tests pin the three operators down on surfaces whose
exact level is known analytically:

* ``water_height_at``  -- legacy top-``k`` particle mean (the estimator every
  earlier round of this case used);
* ``surface_level``    -- the same, with the ``(k - 1)/2 dx`` row-offset removed;
* ``slab_level``       -- fixed-aperture volume gauge (aperture in metres, so the
  same physical gauge is used on every grid).

The bias envelopes asserted below are the ones measured on a still lattice while
sweeping the free surface across one row; they are the numbers quoted in
``docs/verification_kleefsman_gauge_bias.md``.
"""
import math

import pytest
import torch

from tensorsph.kleefsman_obstacle import (
    KleefsmanConfig,
    gauge_series,
    run_case,
    slab_level,
    surface_level,
    validate_case,
    water_height_at,
)

STATION = 1.5
APERTURE = 0.08


def still_lattice(height: float, dx: float, x0: float = 1.0, x1: float = 2.0,
                  shift: float = 0.0) -> torch.Tensor:
    """Body of still water of depth ``height`` on a square lattice of spacing dx."""
    xs = torch.arange(x0 + shift + 0.5 * dx, x1, dx)
    ys = torch.arange(0.5 * dx, height, dx)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def make_cfg(dx: float, **kw) -> KleefsmanConfig:
    return KleefsmanConfig(dx=dx, gauge_aperture=APERTURE, dt=1.0e-4, **kw)


def phase_sweep(operator, dx: float, rows: int = 19):
    """Estimator error / dx over a sweep of the surface phase inside one row."""
    cfg = make_cfg(dx)
    errs = []
    for j in range(1, rows + 1):
        h = (26.0 + j / rows) * dx
        errs.append((operator(still_lattice(h, dx), cfg, STATION) - h) / dx)
    return errs


def test_legacy_estimator_has_systematic_negative_dx_bias():
    """The historical estimator sits 0.4-1.4 dx *below* the water level, always."""
    for dx in (0.04, 0.02, 0.01):
        err = phase_sweep(lambda p, c, x: water_height_at(p, c, x=x), dx)
        assert max(err) <= -0.3, (dx, max(err))       # never unbiased upwards
        assert min(err) >= -1.4, (dx, min(err))
        assert sum(err) / len(err) < -0.5, (dx, sum(err) / len(err))


def test_corrected_surface_is_phase_unbiased_within_half_a_row():
    """Removing the row offset shrinks the envelope to roughly +/-0.5 dx."""
    for dx in (0.04, 0.02, 0.01):
        err = phase_sweep(lambda p, c, x: surface_level(p, c, x=x), dx)
        assert min(err) >= -0.4 and max(err) <= 0.7, (dx, min(err), max(err))
        assert abs(sum(err) / len(err)) < 0.3, (dx, sum(err) / len(err))


def test_slab_gauge_is_dx_consistent_on_a_planar_surface():
    """Same physical aperture, three grids, same reading: no O(dx) drift."""
    dxs = (0.04, 0.02, 0.01)
    for phase in (0.0, 0.31, 0.77):
        readings = []
        for dx in dxs:
            h = (26.0 + phase) * dx
            got = slab_level(still_lattice(h, dx), make_cfg(dx), STATION)
            assert abs(got - h) <= 0.55 * dx, (dx, phase, got, h)
            readings.append(got / dx)                  # normalise to rows
        spread = max(readings) - min(readings)
        assert spread <= 0.55, (phase, readings)


def test_gauge_bias_versus_the_arrival_threshold():
    """Quantifies why arrival times were biased: gauge bias vs a 0.02 m threshold."""
    threshold = 0.02
    for dx in (0.04, 0.02):
        legacy = max(abs(e) * dx for e in
                     phase_sweep(lambda p, c, x: water_height_at(p, c, x=x) - 0, dx))
        cfg = make_cfg(dx)
        h = 26.5 * dx
        slab = abs(slab_level(still_lattice(h, dx), cfg, STATION) - h)
        # the slab gauge stays inside half a row; the legacy bias does not fit
        # into the 0.02 m threshold on any grid coarser than 0.01 m
        assert slab <= 0.5 * dx + 1e-5, (dx, slab)
        if dx >= 0.02:
            assert legacy > threshold, (dx, legacy)


def test_isolated_spray_is_not_counted_by_any_operator():
    """A ballistic droplet above the water must not raise the gauge reading."""
    dx = 0.02
    cfg = make_cfg(dx)
    h = 0.52
    body = still_lattice(h, dx, shift=0.013)
    droplet = torch.tensor([[STATION, h + 4.0 * dx]])
    wet = torch.cat((body, droplet), dim=0)
    legacy = water_height_at(body, cfg, x=STATION)
    legacy_spray = water_height_at(wet, cfg, x=STATION)
    # contiguity filter must reject it; without the filter it jumps by >= 3 dx
    assert abs(legacy_spray - legacy) <= 0.6 * dx
    cfg_nofilter = make_cfg(dx, robust_surface=False)
    assert water_height_at(wet, cfg_nofilter, x=STATION) - legacy > 1.0 * dx
    slab = abs(slab_level(wet, cfg, STATION) - slab_level(body, cfg, STATION))
    # one particle of volume dx^2 over an APERTURE-wide column: tiny, bounded
    assert slab <= dx * dx / APERTURE + 1e-6


def test_gauge_series_covers_three_operators_and_all_stations():
    cfg = KleefsmanConfig(dx=0.02, t_end=0.03, dt=1.0e-4, output_every=10,
                          gauge_aperture=APERTURE, device="cpu")
    validate_case(cfg)
    run = run_case(cfg)
    stations = cfg.station_x()
    expect = {f"{op}@x{xs:.3f}" for op in ("height", "surface", "slab") for xs in stations}
    assert set(run.series) == expect
    assert set(run.metrics) == expect
    for key, values in run.series.items():
        assert len(values) == len(run.times), key
        assert all(math.isfinite(v) for v in values), key
        assert set(run.metrics[key]) == {"peak_m", "peak_time_s", "arrival_time_s"}


def test_validate_rejects_aliasing_aperture_and_station_inside_the_column():
    with pytest.raises(ValueError, match="narrower than one particle"):
        validate_case(KleefsmanConfig(dx=0.05, gauge_aperture=0.04, dt=1.0e-4))
    with pytest.raises(ValueError, match="outside"):
        validate_case(KleefsmanConfig(dx=0.02, gauge_aperture=APERTURE, dt=1.0e-4,
                                     stations_offset=(0.05, 1.30)))


def test_station_geometry_matches_the_box_face():
    cfg = make_cfg(0.02)
    xs = cfg.station_x()
    assert xs[0] == pytest.approx(cfg.obstacle_x - 0.05)
    assert xs[1] == pytest.approx(cfg.obstacle_x - 0.248)
    assert all(x < cfg.obstacle_x for x in xs)
    # the wide slab gauge must sit entirely upstream of the box face
    assert cfg.station_x()[0] - 0.5 * cfg.resolve_gauge_aperture() > cfg.column_length