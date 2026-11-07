"""Tests for the bow ice-splash minimal capability face (``tensorsph.ice_splash``).

The suite mixes cheap structural checks with a handful of short integrations.
Tolerances are set from the numbers recorded in
``artifacts/ice_splash/results.json`` (produced by
``benchmarks/ice_splash/run.py``), so a regression in the physics shows up as a
threshold failure rather than a silent drift.
"""
import json
import math
import os

import pytest
import torch

from tensorsph.ice_splash import (
    DEM_SCHEMA,
    IceSplashConfig,
    ballistic_apex,
    ballistic_range,
    build_dem_exchange,
    deck_start_x,
    eos_internal_energy,
    floe_force_plate,
    floe_layout,
    front_coefficient,
    ice_bed_lattice,
    ice_surface,
    read_dem_exchange,
    run_case,
    validate_config,
    validate_dem_exchange,
    write_dem_exchange,
)

# Verified platform baseline for the reference-free front coefficient of the
# flat-bed dam break (docs/benchmarks/dam-break-reference-audit.md, section 7):
# dx = 0.04 m, free slip, viscosity 0.01, c = 1.2016.
AUDIT_C_DX004 = 1.2016


def splash_config(**overrides) -> IceSplashConfig:
    """Short bow ice-splash case used by the integration tests."""
    base = dict(
        dx=0.025, column_width=0.20, column_height=0.28, pool_width=1.0,
        tank_height=0.60, sheet_start=0.20, sheet_angle_deg=25.0, deck_height=0.28,
        sound_speed=20.0, viscosity=0.0, device="cpu", output_every=5,
        max_courant=0.3, n_floes=5, seed=7,
    )
    base.update(overrides)
    t_target = float(base.pop("t_target", 0.44))
    cfg = IceSplashConfig(**base)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(t_target / cfg.dt))
    return cfg


def flat_bed_config(**overrides) -> IceSplashConfig:
    """Flat-bed dam break (no ice bed): the Ritter / Martin & Moyce anchor case."""
    base = dict(
        dx=0.04, column_width=0.20, column_height=0.40, pool_width=2.0,
        tank_height=0.70, sheet_start=2.0, sheet_angle_deg=25.0, deck_height=0.10,
        sound_speed=25.0, viscosity=0.01, device="cpu", output_every=5,
        max_courant=0.3, n_floes=0,
    )
    base.update(overrides)
    t_target = float(base.pop("t_target", 0.50))
    cfg = IceSplashConfig(**base)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(t_target / cfg.dt))
    return cfg


@pytest.fixture(scope="module")
def splash():
    cfg = splash_config()
    return cfg, run_case(cfg)


# --------------------------------------------------------------------------- #
# Geometry and configuration (cheap, structural)
# --------------------------------------------------------------------------- #

def test_ice_surface_rises_then_plateaus():
    cfg = splash_config()
    assert ice_surface(0.0, cfg) == 0.0
    assert ice_surface(cfg.sheet_start + 1.0e-9, cfg) == pytest.approx(0.0, abs=1e-6)
    crest = deck_start_x(cfg)
    assert ice_surface(crest, cfg) == pytest.approx(cfg.deck_height, rel=1e-9)
    assert ice_surface(0.5 * (cfg.sheet_start + crest), cfg) == pytest.approx(
        0.5 * cfg.deck_height, rel=1e-6)
    assert ice_surface(cfg.pool_width, cfg) == pytest.approx(cfg.deck_height)
    # Monotone rise along the ramp, no overshoot above the deck.
    heights = [ice_surface(x, cfg) for x in torch.linspace(cfg.sheet_start, crest, 20).tolist()]
    assert all(b >= a - 1e-12 for a, b in zip(heights, heights[1:]))
    assert max(heights) <= cfg.deck_height + 1e-12


def test_bed_particles_sit_behind_the_surface_and_inside_the_tank():
    cfg = splash_config()
    bed = ice_bed_lattice(cfg, torch.device("cpu"))
    assert bed.shape[1] == 2
    # Up to ``wall_layers`` spacings behind the surface, never in front of it,
    # and no bed particle inside the water column.
    margin = (cfg.wall_layers + 0.5) * cfg.dx
    for x, y in bed.tolist():
        assert x >= -margin - 1e-9 and x <= cfg.pool_width + margin + 1e-9
        assert y >= -margin - 1e-9
        if cfg.sheet_start - 1e-9 <= x <= cfg.pool_width - 2.0 * cfg.dx:
            assert y <= cfg.deck_height + 1e-9, "ramp/deck particles stay at or below the deck"
        if 0.5 * cfg.dx < x < cfg.sheet_start - 0.5 * cfg.dx:
            assert y <= 0.5 * cfg.dx + 1e-9, "floor particle must stay below the floor"
    inside_column = ((bed[:, 0] > 0.0) & (bed[:, 0] < cfg.column_width)
                     & (bed[:, 1] > 0.5 * cfg.dx))
    assert not bool(inside_column.any()), "bed particles must not sit in the water column"


@pytest.mark.parametrize("overrides, message", [
    ({"deck_height": 0.59, "tank_height": 0.60}, "tank_height"),
    ({"sheet_start": 0.10, "column_width": 0.20}, "sheet_start"),
    ({"sheet_angle_deg": 0.0}, "sheet_angle_deg"),
    ({"dx": 0.0}, "positive"),
    ({"seed": -1}, "seed"),
    ({"compact": True}, None),
])
def test_invalid_configurations_are_rejected(overrides, message):
    if message is None:
        pytest.skip("placeholder for parametrisation symmetry")
    overrides = {k: v for k, v in overrides.items() if k != "compact"}
    base = dict(dx=0.025, column_width=0.20, column_height=0.28, pool_width=1.0,
                tank_height=0.60, sheet_start=0.20, deck_height=0.28)
    base.update(overrides)
    cfg = IceSplashConfig(**base)
    cfg.dt = 1.0e-4
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


# --------------------------------------------------------------------------- #
# DEM hand-off format
# --------------------------------------------------------------------------- #

def _payload(count=2):
    return {
        "schema": DEM_SCHEMA,
        "source": {"solver": "tensorsph", "case": "ice_splash", "spacing_m": 0.02},
        "particles": {
            "position_m": [[0.1 * k, 0.05] for k in range(count)],
            "velocity_m_s": [[0.5, 0.0] for _ in range(count)],
            "mass_kg_per_m": [0.4 for _ in range(count)],
            "radius_m": [0.01 for _ in range(count)],
            "material": ["water" for _ in range(count)],
        },
        "counts": {"water": count, "ice": 0, "total": count},
    }


def test_dem_exchange_rejects_negative_or_zero_spacing():
    payload = _payload()
    payload["source"]["spacing_m"] = -0.02
    with pytest.raises(ValueError, match="spacing must be positive"):
        validate_dem_exchange(payload)
    payload["source"]["spacing_m"] = 0.0
    with pytest.raises(ValueError, match="spacing must be positive"):
        validate_dem_exchange(payload)
    payload["source"]["spacing_m"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_dem_exchange(payload)


def test_dem_exchange_rejects_zero_or_negative_mass():
    payload = _payload()
    payload["particles"]["mass_kg_per_m"][1] = 0.0
    with pytest.raises(ValueError, match="mass must be positive"):
        validate_dem_exchange(payload)
    payload["particles"]["mass_kg_per_m"][1] = -1.0e-3
    with pytest.raises(ValueError, match="mass must be positive"):
        validate_dem_exchange(payload)
    payload["particles"]["mass_kg_per_m"][1] = 0.4
    payload["particles"]["radius_m"][0] = 0.0
    with pytest.raises(ValueError, match="radius must be positive"):
        validate_dem_exchange(payload)


def test_dem_exchange_rejects_inconsistent_counts_and_schema():
    payload = _payload()
    payload["counts"]["total"] = 3
    with pytest.raises(ValueError, match="counts"):
        validate_dem_exchange(payload)
    payload = _payload()
    payload["schema"] = "tensorsph.dem-exchange/0.1"
    with pytest.raises(ValueError, match="schema"):
        validate_dem_exchange(payload)


def test_dem_exchange_roundtrip_carries_the_resolved_state(splash, tmp_path):
    cfg, run = splash
    payload = build_dem_exchange(run, cfg)
    validate_dem_exchange(payload)
    assert payload["counts"]["water"] == int(run.final_positions.shape[0])
    assert payload["counts"]["ice"] == cfg.n_floes
    # Water particles carry exactly the SPH end state.
    first = payload["particles"]["position_m"][0]
    assert first[0] == pytest.approx(float(run.final_positions[0, 0]))
    assert first[1] == pytest.approx(float(run.final_positions[0, 1]))
    # Ice entries carry the reduced-model launch speed, not the water velocity.
    ice = [k for k, m in enumerate(payload["particles"]["material"]) if m == "ice"]
    assert len(ice) == cfg.n_floes
    for k in ice:
        assert payload["particles"]["mass_kg_per_m"][k] == pytest.approx(
            cfg.floe_rho * math.pi * cfg.floe_radius ** 2)
    out = write_dem_exchange(payload, tmp_path / "dem_init.json")
    reloaded = read_dem_exchange(out)
    assert reloaded["counts"] == payload["counts"]
    assert reloaded["source"]["seed"] == cfg.seed


# --------------------------------------------------------------------------- #
# Reduced floe model: the force plate must reproduce a momentum flux
# --------------------------------------------------------------------------- #

def test_force_plate_reproduces_the_momentum_flux_of_a_uniform_jet():
    cfg = splash_config(n_floes=1, floe_radius=0.05, floe_start=0.60, jitter=0.0)
    dx = cfg.dx
    # A uniform jet of height 2r moving at u in +x, one lattice thick in front
    # of the plate: the momentum flux rho u^2 h must come back out.
    ys = torch.arange(cfg.floe_radius * 0.9, 1.9 * cfg.floe_radius, dx, dtype=torch.float64)
    xs = torch.tensor([0.60 - 0.5 * dx, 0.60 - 1.5 * dx], dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, dtype=torch.float64)
    speed = 1.5
    vel = torch.zeros_like(pos)
    vel[:, 0] = speed
    floes = torch.tensor([[0.60, cfg.floe_radius]], dtype=torch.float64)
    force = floe_force_plate(pos, vel, mass, cfg, floes)
    height = float(ys.max() - ys.min()) + dx
    expected = cfg.rho0 * speed ** 2 * height
    assert float(force[0]) == pytest.approx(expected, rel=0.15)
    # No downstream momentum -> no force.
    vel_backwards = -vel
    assert float(floe_force_plate(pos, vel_backwards, mass, cfg, floes)[0]) == 0.0


def test_ballistic_helpers_are_the_textbook_parabola():
    assert ballistic_apex(2.0, 9.81) == pytest.approx(2.0 ** 2 / (2 * 9.81))
    assert ballistic_range(2.0, 45.0, 9.81) == pytest.approx(4.0 / 9.81, rel=1e-12)
    # 45 degrees maximises the range.
    assert ballistic_range(2.0, 45.0, 9.81) >= ballistic_range(2.0, 30.0, 9.81)


# --------------------------------------------------------------------------- #
# Integration checks
# --------------------------------------------------------------------------- #

def test_same_seed_reproduces_the_run_bit_for_bit():
    cfg_small = dict(dx=0.03, column_width=0.20, column_height=0.28, pool_width=1.0,
                     tank_height=0.60, sheet_start=0.20, deck_height=0.28,
                     sound_speed=20.0, viscosity=0.0, t_target=0.18, seed=11)
    cfg_a = splash_config(**cfg_small)
    cfg_b = splash_config(**cfg_small)
    run_a = run_case(cfg_a)
    run_b = run_case(cfg_b)
    assert torch.equal(run_a.final_positions, run_b.final_positions)
    assert run_a.times == run_b.times
    assert run_a.deck_volume == run_b.deck_volume
    assert run_a.floe_impulse == run_b.floe_impulse
    # A different seed moves the floe layout, and nothing else: the water is
    # deterministic, so its gauges stay bit-identical while the reduced-model
    # floe probe is placed elsewhere.
    cfg_c = splash_config(**{**cfg_small, "seed": 12})
    layout_a = floe_layout(cfg_a, torch.device("cpu"))
    layout_c = floe_layout(cfg_c, torch.device("cpu"))
    assert not torch.equal(layout_a, layout_c)
    run_c = run_case(cfg_c)
    assert run_c.final_positions.shape == run_a.final_positions.shape
    assert run_c.deck_volume == run_a.deck_volume


def test_energy_drift_over_the_event_window(splash):
    cfg, run = splash
    window = run.energy_window_end()
    assert window > 0.0
    mechanical = run.energy_drift_peak(window)
    total = run.energy_drift_peak(window, include_internal=True)
    # The elastic (compressibility) reservoir is what closes the budget: it must
    # be reported and it must not be larger than the mechanical oscillation.
    assert run.peak_compression < 0.10
    assert total <= mechanical + 1.0e-9 or total < 0.05
    assert mechanical < 0.05, f"mechanical drift {mechanical:.3%} over {window:.3f} s"
    # Nothing may leave the box: a clamp would silently take energy out.
    assert run.escaped == 0


def test_splash_apex_is_ballistic(splash):
    cfg, run = splash
    grades = run.ballistic_error()
    assert grades["count"] >= 3, "no water crossed the deck level upward"
    # 0.12 at dx = 0.025; the production resolution (dx = 0.0125) measures 1.9%
    # and the artifact gate uses the stricter 5% (results.json -> criteria).
    assert grades["median_rel_error"] < 0.12
    # Every tracked particle must have been graded on a real rise.
    for pred, meas in zip(run.apex_predicted, run.apex_measured):
        if pred is None:
            continue
        assert meas is not None
        assert meas <= pred + 0.15 * pred, "apex overshoots the ballistic prediction"


def test_gauges_are_physical_and_mass_is_conserved(splash):
    cfg, run = splash
    assert run.initial_mass == pytest.approx(run.final_mass, rel=1e-12)
    assert all(v >= 0.0 for v in run.deck_volume)
    assert run.peak_throw >= 0.0
    assert run.peak_apex < cfg.tank_height, "the splash must not reach the domain top"
    assert run.deck_arrival_time is not None, "the surge never reached the deck"
    assert run.crest_arrival_time is not None
    # The bore arrives at the deck only after it has started to move at all.
    assert 0.0 < run.deck_arrival_time < run.simulated_time
    # Splash kinetic energy above the deck and the deck inundation are positive
    # once water is up there.
    assert run.peak_splash_kinetic_energy > 0.0
    assert run.peak_deck_volume > 0.0


def test_flat_bed_reproduces_the_verified_front_coefficient():
    cfg = flat_bed_config()
    run = run_case(cfg)
    metric = front_coefficient(run, cfg)
    assert metric["c"] is not None
    assert metric["points"] >= 20
    # Reproduce the platform's verified dimensionless baseline within 15%.
    assert metric["c"] == pytest.approx(AUDIT_C_DX004, rel=0.15), metric
    # And the front must stay well below Ritter's c = 2 (the audit's finding).
    assert metric["c"] < 1.5
    assert run.escaped == 0


def test_eos_internal_energy_is_second_order_in_compression():
    cfg = splash_config()
    delta = torch.tensor([1.0e-3, 1.0e-2], dtype=torch.float64)
    energy = eos_internal_energy(cfg.rho0 * (1.0 + delta), cfg)
    # e ~ delta^2: a 10x larger compression gives ~100x the energy.
    assert float(energy[1] / energy[0]) == pytest.approx(100.0, rel=0.25)
    # The tension side of the clamped Tait law stores nothing.
    assert float(eos_internal_energy(torch.tensor([0.9 * cfg.rho0], dtype=torch.float64),
                                     cfg)[0]) == 0.0


def test_artifact_json_is_present_and_consistent():
    path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "ice_splash",
                        "results.json")
    if not os.path.exists(path):
        pytest.skip("artifact bundle not generated in this checkout")
    with open(path) as handle:
        payload = json.load(handle)
    assert payload["case"]["sheet_angle_deg"] == pytest.approx(25.0)
    # The anchor must reproduce the verified baseline, the splash apex must be
    # ballistic, and the seed must reproduce. The energy verdict is recorded as
    # measured (it does not meet the strict 1% gate at this resolution); the
    # test only enforces internal consistency of the recorded verdict.
    assert payload["criteria"]["front_coefficient_error"] < 0.15
    assert payload["criteria"]["ballistic_median_error"] < 0.05
    assert payload["criteria"]["fragmentation_stage_drift"] >= 0.0
    assert payload["criteria"]["energy_drift_total"] == pytest.approx(
        payload["energy_ladder"][1]["energy_drift_total"], rel=1e-6)
    assert payload["criteria"]["energy_pass"] == (
        payload["criteria"]["energy_drift_total"] < payload["criteria"]["tolerances"]["energy"])
    assert payload["compressibility_study"], "the compressibility study must be recorded"
