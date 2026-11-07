"""Guards for the startup-quiescence probe (``benchmarks/startup_quiescence``).

The benchmark separates two layers, and both are pinned here:

* **contract gates** -- the shipped configuration must start at the analytic
  hydrostatic pressure and must not move more than the regression envelope;
  the live cases below re-measure them, so the gate cannot rot into a number in
  a JSON file nobody re-derives.
* **open findings** -- ``F1_flume_resting_noise`` is a defect register entry,
  not a red test.  It is pinned in both directions: it must still be recorded
  as ``OPEN`` *and* its measured value must still reproduce.  When somebody
  fixes the flume damping the test fails and asks for the register to be
  updated, which is the point.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config, apply_boundary, make_particles, rhs, wall_lattice
from tensorsph.wave_flume import (
    WaveFlumeConfig,
    apply_flume_boundary,
    flume_particle_mass,
    flume_water_lattice,
    paddle_lattice,
    static_wall_lattice,
)

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "benchmarks" / "startup_quiescence" / "report.json"


@pytest.fixture(scope="module")
def report() -> dict:
    assert REPORT.exists(), (
        "benchmarks/startup_quiescence/report.json is missing; regenerate with "
        "`PYTHONPATH=src python benchmarks/startup_quiescence/run.py "
        "--json-out benchmarks/startup_quiescence/report.json`")
    return json.loads(REPORT.read_text())


def test_contract_gates_pass(report):
    failed = [g["id"] for g in report["gates"] if not g["pass"]]
    assert not failed, f"contract gates failed: {failed}"
    assert report["all_pass"] is True


def test_static_metric_sees_the_lattice_mass_bias(report):
    """G2 in the report: the probe must stay able to fail."""
    gate = next(g for g in report["gates"] if g["id"] == "G2_sensitivity")
    assert gate["measured"] >= gate["limit"] >= 10.0


def test_calibration_sets_the_initial_pressure_not_the_startup_transient(report):
    """The finding that corrects the backlog's attribution.

    Mass calibration moves the static initial state by orders of magnitude but
    the *dynamic* peak is essentially unchanged, so the self-generated velocity
    is not a mass-unit criterion.
    """
    pool = {(c["geometry"]["h_over_dx"], c["settings"]["density_calibration"]): c
            for c in report["cases"] if c["state"] == "pool"}
    on, off = pool[(1.3, True)], pool[(1.3, False)]
    assert off["initial_pressure_deviation"] > 10.0 * on["initial_pressure_deviation"]
    assert off["v_peak"] > 0.5 * on["v_peak"]


def test_flume_resting_noise_finding_is_registered(report):
    finding = next((f for f in report["findings"]
                    if f["id"] == "F1_flume_resting_noise"), None)
    assert finding is not None, (
        "F1_flume_resting_noise disappeared; if the flume damping was fixed, "
        "update the register (and docs/verification_wave_flume.md) instead of "
        "deleting the finding silently")
    assert finding["status"] == "OPEN"
    assert finding["measured"] > finding["limit"]
    # the ablation must explain the noise, not just describe it
    ablation = finding["ablation"]["all"]
    calmest = min(a["interior_v_late_rms"] for a in ablation)
    shipped = next(a["interior_v_late_rms"] for a in ablation
                   if a["label"].endswith("(paddle frozen at rest)"))
    assert calmest < 0.25 * shipped, "ablation no longer separates the noise source"


def test_live_pool_initial_state_meets_the_gate():
    """Re-derive G1 from a fresh run instead of trusting the JSON."""
    cfg = Config(dx=0.02, h_factor=1.3, hydrostatic_init=True,
                 density_calibration=True, boundary="wall", device="cpu",
                 pool_width=0.30, pool_depth=0.20)
    device = torch.device("cpu")
    pos, mass = make_particles(cfg, device)
    keep = pos[:, 1] <= cfg.pool_depth
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    wall = wall_lattice(cfg, device)
    rho, pressure, _ = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
    h = cfg.h_factor * cfg.dx
    mask = ((pos[:, 0] > 3 * h) & (pos[:, 0] < cfg.pool_width - 3 * h)
            & (pos[:, 1] > 3 * h) & (pos[:, 1] < cfg.pool_depth - 3 * h))
    assert bool(mask.any())
    p_ref = cfg.rho0 * cfg.gravity * (cfg.pool_depth - pos[:, 1])
    scale = cfg.rho0 * cfg.gravity * cfg.pool_depth
    deviation = float(((pressure[mask] - p_ref[mask]).abs() / scale).max())
    assert deviation <= 1e-3, f"initial pressure deviation {deviation:.3e}"


def test_live_flume_initial_state_meets_the_gate():
    """The t = 0 state of a graded wave run must be a discrete equilibrium."""
    cfg = WaveFlumeConfig(depth=0.12, dx=0.012, flume_length=0.36, device="cpu")
    device = torch.device("cpu")
    pos = flume_water_lattice(cfg, device)
    mass = flume_particle_mass(cfg, pos, device)
    wall = torch.cat((static_wall_lattice(cfg, device),
                      paddle_lattice(cfg, device)), dim=0)
    rho, pressure, _ = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
    h = cfg.h_factor * cfg.dx
    mask = ((pos[:, 0] > 3 * h) & (pos[:, 0] < cfg.flume_length - 3 * h)
            & (pos[:, 1] > 3 * h) & (pos[:, 1] < cfg.depth - 3 * h))
    assert bool(mask.any())
    p_ref = cfg.rho0 * cfg.gravity * (cfg.depth - pos[:, 1])
    scale = cfg.rho0 * cfg.gravity * cfg.depth
    deviation = float(((pressure[mask] - p_ref[mask]).abs() / scale).max())
    assert deviation <= 1e-3, f"flume initial pressure deviation {deviation:.3e}"


def test_live_flume_starts_at_rest_to_the_regression_envelope():
    """One short integration of the resting column (0.01 s, coarse)."""
    cfg = WaveFlumeConfig(depth=0.12, dx=0.012, flume_length=0.36, device="cpu")
    cfg.dt = min(cfg.dt, cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed)
    steps = max(int(round(0.01 / cfg.dt)), 2)
    device = torch.device("cpu")
    pos = flume_water_lattice(cfg, device)
    mass = flume_particle_mass(cfg, pos, device)
    wall = torch.cat((static_wall_lattice(cfg, device),
                      paddle_lattice(cfg, device)), dim=0)
    vel = torch.zeros_like(pos)
    peak = 0.0
    for _ in range(steps):
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        assert bool(torch.isfinite(acc).all())
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        apply_flume_boundary(pos, vel, cfg, cfg.depth)
        peak = max(peak, float(torch.linalg.vector_norm(vel, dim=1).max()))
    # the shipped flume is not quiescent (finding F1); the envelope here only
    # catches a regression that makes the startup transient *worse* than the
    # documented ablation range
    assert peak < 0.30, f"flume startup peak {peak:.3f} m/s exceeds the envelope"