"""Startup spurious-current probe (Q1): how still is a "resting" pool, and what drives it?

Why this benchmark exists
-------------------------
``backlog_update_m3211.md`` P2.2 proposed the velocity a *zero-force* pool
spontaneously develops as a cheap smoke criterion for any change to the mass
unit, the free-surface treatment or the pressure closure: the pool has no
forcing at all, so whatever motion appears is spurious by construction.  The
same backlog quoted one number (0.096 m/s at ``h/dx = 1.5`` with an
uncalibrated lattice mass) and attributed it to that mass bias.

This benchmark tests that attribution instead of repeating it, and it records
what actually limits quiescence:

``initial_pressure_deviation``
    ``max_i |p_i - rho0 g (H - y_i)| / (rho0 g H)`` over interior particles
    (at least ``2 h`` from every boundary) at ``t = 0``.  This is the static
    quality of the initial state and it *is* controlled by the lattice mass
    unit: ``density_calibration = True`` puts it at ``1.5e-4``, the uncalibrated
    ``h/dx = 1.5`` lattice puts it at ``0.61``.
``initial_acceleration``
    ``max_i |a_i|`` at ``t = 0``, reported for all particles, for interior
    particles, and with the height at which the maximum sits.  This is the
    *driver* of the transient.
``v_*``
    ``max_i |v_i|`` (all particles and interior only) sampled every
    ``--sample-every`` steps: peak, final value, and RMS over the last quarter.

Two initial states are probed, because the platform ships two: the
:mod:`tensorsph.droplet_impact` free-surface pool, and the
:mod:`tensorsph.wave_flume` resting column with the paddle frozen at its rest
position -- the ``t = 0`` state every graded wave run starts from.

Gates (exit status is non-zero when one fails)
---------------------------------------------
``G1_initial_state_quality``   calibrated pool: ``initial_pressure_deviation <= 1e-3``
``G2_sensitivity``             ``initial_pressure_deviation(uncalibrated) /
                               (calibrated) >= 10`` -- a gate that cannot fail is
                               not a gate
``G3_noise_floor``             interior ``v_late_rms <= --v-floor`` m/s
``G4_flume_initial_state``     flume resting column: same two criteria as G1/G3

``G3``/``G4``'s velocity bound is a **regression envelope** pinned just above
today's measured value, not a physics acceptance criterion: no published
acceptance level exists for "how still is still" (see
``docs/verification_startup_quiescence.md`` §5).

Usage::

    PYTHONPATH=src python benchmarks/startup_quiescence/run.py \\
        --json-out benchmarks/startup_quiescence/report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.droplet_impact import (  # noqa: E402
    Config,
    apply_boundary,
    make_particles,
    rhs,
    wall_lattice,
)
from tensorsph.wave_flume import (  # noqa: E402
    WaveFlumeConfig,
    apply_flume_boundary,
    flume_particle_mass,
    flume_water_lattice,
    paddle_lattice,
    static_wall_lattice,
    linear_dispersion_omega,
    wavenumber_from_period,
)

POOL_WIDTH = 0.60
POOL_DEPTH = 0.24


INTERIOR_MARGIN_H = 3.0


def interior_mask(pos: torch.Tensor, h: float,
                  x_min: float, x_max: float,
                  y_min: float, y_max: float) -> torch.Tensor:
    """Particles at least ``3 h`` away from every boundary.

    ``2 h`` (the kernel support) is the completeness requirement for the volume
    integral, but not for the vertical direction: particles between ``2 h`` and
    ``3 h`` below the free surface still carry a pressure deviation of
    ``3.9e-2 * rho0 g H`` (measured, flume at ``dx = 1 mm``, see the
    ``transition_band_*`` fields), which swamps the ``1e-3`` initial-state gate.
    The band is therefore reported separately instead of being blamed on the
    mass unit.
    """
    margin = INTERIOR_MARGIN_H * h
    return ((pos[:, 0] > x_min + margin) & (pos[:, 0] < x_max - margin)
            & (pos[:, 1] > y_min + margin) & (pos[:, 1] < y_max - margin))


def _max_speed(vel: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        vel = vel[mask]
    if vel.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(vel, dim=1).max())


def _max_accel(acc: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor) -> dict:
    mag = torch.linalg.vector_norm(acc, dim=1)
    peak = int(mag.argmax())
    return {
        "all_max": float(mag.max()),
        "interior_max": float(mag[mask].max()) if bool(mask.any()) else None,
        "all_argmax_y_m": float(pos[peak, 1]),
        "gravity": None,  # filled by the caller
    }


def _summarise(series: list[float]) -> dict:
    if not series:
        return {"v_peak": 0.0, "v_final": 0.0, "v_late_rms": 0.0}
    late = series[-max(len(series) // 4, 1):]
    return {
        "v_peak": max(series),
        "v_final": series[-1],
        "v_late_rms": math.sqrt(sum(v * v for v in late) / len(late)),
    }


def h_at(cfg) -> float:
    """Smoothing length of the case."""
    return cfg.h_factor * cfg.dx


def _initial_state(pos, vel, mass, cfg, wall, mask, surface_height) -> dict:
    rho, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall)
    p_ref = cfg.rho0 * cfg.gravity * (surface_height - pos[:, 1])
    scale = cfg.rho0 * cfg.gravity * surface_height
    deviation = (pressure - p_ref).abs() / scale
    # Transition band: inside the 2h-3h shell below the free surface, reported
    # separately so the interior gate is not silently relaxed to hide it.
    band = (~mask) & (pos[:, 1] > surface_height - 2.0 * h_at(cfg))
    record = {
        "initial_density_deviation": float((rho[mask] / cfg.rho0 - 1.0).abs().max()),
        "initial_pressure_deviation": float(deviation[mask].max()),
        "initial_pressure_deviation_mean": float(deviation[mask].mean()),
        "transition_band_pressure_deviation": (
            float(deviation[band].max()) if bool(band.any()) else None),
    }
    accel = _max_accel(acc, mask, pos)
    accel["gravity"] = cfg.gravity
    record["initial_acceleration"] = accel
    return record


def run_pool_case(h_factor: float, calibrate: bool, dx: float, seconds: float,
                  sample_every: int) -> dict:
    """Resting 2-D pool with gravity, free surface and walls: no forcing at all."""
    cfg = Config(dx=dx, h_factor=h_factor, hydrostatic_init=True,
                 density_calibration=calibrate, boundary="wall", device="cpu",
                 pool_width=POOL_WIDTH, pool_depth=POOL_DEPTH)
    cfg.dt = cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed
    steps = max(int(round(seconds / cfg.dt)), 2)
    device = torch.device("cpu")
    h = cfg.h_factor * cfg.dx

    pos, mass = make_particles(cfg, device)
    keep = pos[:, 1] <= cfg.pool_depth  # pool only: nothing is falling in
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    wall = wall_lattice(cfg, device)
    vel = torch.zeros_like(pos)

    mask = interior_mask(pos, h, 0.0, cfg.pool_width, 0.0, cfg.pool_depth)
    initial = _initial_state(pos, vel, mass, cfg, wall, mask, cfg.pool_depth)

    started = time.perf_counter()
    all_series: list[float] = []
    interior_series: list[float] = []
    escaped = 0
    for step in range(steps):
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        if not bool(torch.isfinite(acc).all()):
            raise SystemExit("ERROR: non-finite acceleration in the quiescence probe")
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        escaped += apply_boundary(pos, vel, cfg)
        if step % sample_every == 0:
            all_series.append(_max_speed(vel))
            interior_series.append(_max_speed(vel, mask))
    elapsed = time.perf_counter() - started

    record = {
        "case": "pool",
        "label": (f"pool h/dx={h_factor:g}, density_calibration={calibrate}"),
        "state": "pool",
        "geometry": {"width_m": cfg.pool_width, "depth_m": cfg.pool_depth,
                     "dx_m": cfg.dx, "h_over_dx": cfg.h_factor, "h_m": h,
                     "particles": int(pos.shape[0]),
                     "wall_particles": int(wall.shape[0])},
        "settings": {"seconds": steps * cfg.dt, "steps": steps, "dt": cfg.dt,
                     "sound_speed": cfg.sound_speed, "gravity": cfg.gravity,
                     "rho0": cfg.rho0, "hydrostatic_init": cfg.hydrostatic_init,
                     "density_calibration": calibrate},
        "interior_particles": int(mask.sum()),
        "escaped_particles": int(escaped),
        "wall_seconds": elapsed,
        "sample_every": sample_every,
        "v_series": all_series,
    }
    record.update(initial)
    for prefix, series in (("", all_series), ("interior_", interior_series)):
        record.update({f"{prefix}{k}": v for k, v in _summarise(series).items()})
    return record


def run_flume_case(depth: float, dx: float, flume_length: float,
                   seconds: float, sample_every: int,
                   overrides: dict | None = None) -> dict:
    """The wave flume's resting column at ``t = 0`` (paddle frozen at rest).

    ``overrides`` lets the ablation sweep change one numerical setting at a
    time (viscosity, density diffusion) without touching the shipped defaults.
    """
    overrides = dict(overrides or {})
    cfg = WaveFlumeConfig(depth=depth, dx=dx, flume_length=flume_length,
                          device="cpu", **overrides)
    cfg.dt = min(cfg.dt, cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed)
    steps = max(int(round(seconds / cfg.dt)), 2)
    device = torch.device("cpu")
    h = cfg.h_factor * cfg.dx

    pos = flume_water_lattice(cfg, device)
    mass = flume_particle_mass(cfg, pos, device)
    wall = torch.cat((static_wall_lattice(cfg, device),
                      paddle_lattice(cfg, device)), dim=0)
    vel = torch.zeros_like(pos)

    mask = interior_mask(pos, h, 0.0, cfg.flume_length, 0.0, cfg.depth)
    initial = _initial_state(pos, vel, mass, cfg, wall, mask, cfg.depth)

    started = time.perf_counter()
    all_series: list[float] = []
    interior_series: list[float] = []
    escaped = 0
    for step in range(steps):
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        if not bool(torch.isfinite(acc).all()):
            raise SystemExit("ERROR: non-finite acceleration in the flume probe")
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        escaped += apply_flume_boundary(pos, vel, cfg, cfg.depth)
        if step % sample_every == 0:
            all_series.append(_max_speed(vel))
            interior_series.append(_max_speed(vel, mask))
    elapsed = time.perf_counter() - started

    record = {
        "case": "flume",
        "label": ("flume resting column (paddle frozen at rest)"
                  + ("" if not overrides else
                     ", overrides " + ",".join(f"{k}={v}" for k, v in
                                               sorted(overrides.items())))),
        "state": "flume",
        "geometry": {"length_m": cfg.flume_length, "depth_m": cfg.depth,
                     "dx_m": cfg.dx, "h_over_dx": cfg.h_factor, "h_m": h,
                     "particles": int(pos.shape[0]),
                     "wall_particles": int(wall.shape[0])},
        "settings": {"seconds": steps * cfg.dt, "steps": steps, "dt": cfg.dt,
                     "sound_speed": cfg.sound_speed, "gravity": cfg.gravity,
                     "rho0": cfg.rho0, "hydrostatic_init": cfg.hydrostatic_init,
                     "density_calibration": cfg.density_calibration,
                     "density_diffusion": cfg.density_diffusion},
        "interior_particles": int(mask.sum()),
        "escaped_particles": int(escaped),
        "wall_seconds": elapsed,
        "sample_every": sample_every,
        "v_series": all_series,
    }
    record.update(initial)
    for prefix, series in (("", all_series), ("interior_", interior_series)):
        record.update({f"{prefix}{k}": v for k, v in _summarise(series).items()})
    return record


def wave_signal_context(depth: float, height: float, period: float) -> dict:
    """The orbital velocity a regular wave of this size carries (Airy theory).

    Used only to state the signal the resting-state noise has to be compared
    with; it is the flume's own analytic anchor, not a result of this benchmark.
    """
    k = wavenumber_from_period(period, depth)
    omega = linear_dispersion_omega(k, depth)
    u_surface = 0.5 * height * omega / math.tanh(k * depth)
    return {"depth_m": depth, "height_m": height, "period_s": period,
            "omega_rad_s": omega, "wavenumber_1_per_m": k,
            "u_orbital_surface_m_s": u_surface,
            "sqrt_gH_m_s": math.sqrt(9.81 * depth)}


def grade(cases: list[dict], v_floor: float, min_ratio: float,
          signal: dict) -> dict:
    """Turn the case records into gates; every gate names the case it reads."""
    by_key = {(c["state"], c["geometry"]["h_over_dx"],
               c["settings"]["density_calibration"]): c for c in cases}
    gates = []

    calibrated = [c for c in cases
                  if c["state"] == "pool" and c["settings"]["density_calibration"]]
    worst_state = (max(calibrated, key=lambda c: c["initial_pressure_deviation"])
                   if calibrated else None)
    gates.append({
        "id": "G1_initial_state_quality",
        "reads": worst_state["label"] if worst_state else None,
        "quantity": "max |p_i - rho0 g (H - y_i)| / (rho0 g H), interior, t = 0",
        "measured": (worst_state["initial_pressure_deviation"]
                     if worst_state else None),
        "limit": 1e-3,
        "limit_side": "upper",
        "pass": bool(worst_state
                     and worst_state["initial_pressure_deviation"] <= 1e-3),
        "rationale": "the static initial state must already be at the analytic "
                     "hydrostatic pressure before any dynamics are trusted",
    })

    off = by_key.get(("pool", 1.5, False))
    on = by_key.get(("pool", 1.5, True))
    ratio = None
    if off and on and on["initial_pressure_deviation"] > 0.0:
        ratio = (off["initial_pressure_deviation"]
                 / on["initial_pressure_deviation"])
    gates.append({
        "id": "G2_sensitivity",
        "reads": "pool h/dx=1.5 uncalibrated / calibrated",
        "quantity": "initial_pressure_deviation(off) / (on)",
        "measured": ratio,
        "limit": min_ratio,
        "limit_side": "lower",
        "pass": bool(ratio is not None and ratio >= min_ratio),
        "rationale": "the static metric must still detect the lattice mass bias "
                     "it was built to guard, otherwise the probe is decorative",
    })

    worst_floor = (max(calibrated, key=lambda c: c["interior_v_late_rms"])
                   if calibrated else None)
    gates.append({
        "id": "G3_noise_floor",
        "reads": worst_floor["label"] if worst_floor else None,
        "quantity": "interior max_i |v_i| RMS over the last quarter (m/s)",
        "measured": worst_floor["interior_v_late_rms"] if worst_floor else None,
        "limit": v_floor,
        "limit_side": "upper",
        "pass": bool(worst_floor and worst_floor["interior_v_late_rms"] <= v_floor),
        "rationale": "regression envelope pinned just above today's measured "
                     "value (see docs/verification_startup_quiescence.md §5); "
                     "it is not a physics acceptance criterion",
    })

    flume = next((c for c in cases
                  if c["state"] == "flume"
                  and c["label"].endswith("(paddle frozen at rest)")), None)
    gates.append({
        "id": "G4_flume_initial_state",
        "reads": flume["label"] if flume else None,
        "quantity": "initial_pressure_deviation, interior, t = 0",
        "measured": (flume["initial_pressure_deviation"] if flume else None),
        "limit": 1e-3,
        "limit_side": "upper",
        "pass": bool(flume and flume["initial_pressure_deviation"] <= 1e-3),
        "rationale": ("the t = 0 state of every graded wave run (the paddle is "
                      "frozen, so any motion is spurious); the *dynamic* part of "
                      "that state is reported as finding F1, not as a contract "
                      "gate, because fixing it needs a decision on the flume's "
                      "damping, not a code change in this benchmark"),
    })
    findings = []
    flumes = [c for c in cases if c["state"] == "flume"]
    shipped = next((c for c in flumes if c["label"].endswith("(paddle frozen at rest)")),
                   flumes[0] if flumes else None)
    if shipped and shipped["interior_v_late_rms"] > v_floor:
        calmest = min(flumes, key=lambda c: c["interior_v_late_rms"])
        findings.append({
            "id": "F1_flume_resting_noise",
            "status": "OPEN",
            "reads": shipped["label"],
            "quantity": "interior max_i |v_i| RMS over the last quarter (m/s)",
            "measured": shipped["interior_v_late_rms"],
            "limit": v_floor,
            "ratio_to_orbital_velocity": (shipped["interior_v_late_rms"]
                                         / signal["u_orbital_surface_m_s"]),
            "ablation": {
                "shipped": shipped["interior_v_late_rms"],
                "calmest": {"label": calmest["label"],
                            "interior_v_late_rms": calmest["interior_v_late_rms"]},
                "all": [{"label": c["label"],
                         "interior_v_late_rms": c["interior_v_late_rms"],
                         "v_peak": c["v_peak"]} for c in flumes],
            },
            "reading": (
                "the resting column of the shipped flume configuration does not "
                "ring down: the noise does not decay over the probed window and "
                "is of the same order as the orbital velocity of a regular wave "
                "of the shipped signal height, so it is not a usable baseline for "
                "graded wave runs until the ablation setting that calms it is "
                "adopted or the flume's damping is justified"),
            "suggested_fix": ("identify whether the shipped viscosity (1e-6, "
                              "physical water) is the intended numerical setting "
                              "for the flume; the ablation case that adds the "
                              "droplet_impact numerical viscosity rings down"),
        })
    return {"gates": gates, "findings": findings,
            "all_pass": all(g["pass"] for g in gates),
            "wave_signal": signal}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.02,
                        help="pool particle spacing (default 0.02 m)")
    parser.add_argument("--seconds", type=float, default=0.10,
                        help="simulated pool seconds (default 0.10)")
    parser.add_argument("--flume-dx", type=float, default=0.01)
    parser.add_argument("--flume-depth", type=float, default=0.20)
    parser.add_argument("--flume-length", type=float, default=0.60)
    parser.add_argument("--flume-seconds", type=float, default=0.06)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--v-floor", type=float, default=0.06,
                        help="regression envelope for the interior spurious "
                             "speed (m/s)")
    parser.add_argument("--min-ratio", type=float, default=10.0,
                        help="required sensitivity of the static metric (G2)")
    parser.add_argument("--signal-height", type=float, default=0.03)
    parser.add_argument("--signal-period", type=float, default=0.8)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--cases", default="all",
                        help="'all' or a comma list of pool h/dx factors, e.g. "
                             "'1.3' (the flume case always runs)")
    args = parser.parse_args(argv)

    wanted = (None if args.cases == "all"
              else {float(v) for v in args.cases.split(",") if v.strip()})
    cases: list[dict] = []
    for h_factor in (1.3, 1.5):
        if wanted is not None and h_factor not in wanted:
            continue
        for calibrate in (True, False):
            cases.append(run_pool_case(h_factor, calibrate, args.dx,
                                       args.seconds, args.sample_every))
            last = cases[-1]
            print(f"pool h/dx={h_factor:g} calib={calibrate}: "
                  f"p_dev={last['initial_pressure_deviation']:.3e} "
                  f"a0_max={last['initial_acceleration']['all_max']:.2f} "
                  f"v_peak={last['v_peak']:.4f} "
                  f"v_late_rms(int)={last['interior_v_late_rms']:.4f} m/s "
                  f"({last['wall_seconds']:.1f} s wall)", flush=True)
    flume_variants = (("shipped", {}),
                      ("viscosity=0.01", {"viscosity": 0.01}),
                      ("density_diffusion=0", {"density_diffusion": 0.0}),
                      ("viscosity=0.01,density_diffusion=0",
                       {"viscosity": 0.01, "density_diffusion": 0.0}))
    for name, overrides in flume_variants:
        cases.append(run_flume_case(args.flume_depth, args.flume_dx,
                                    args.flume_length, args.flume_seconds,
                                    args.sample_every, overrides))
        last = cases[-1]
        print(f"flume {name}: p_dev={last['initial_pressure_deviation']:.3e} "
              f"v_peak={last['v_peak']:.4f} "
              f"v_late_rms(int)={last['interior_v_late_rms']:.4f} m/s "
              f"({last['wall_seconds']:.1f} s wall)", flush=True)

    signal = wave_signal_context(args.flume_depth, args.signal_height,
                                 args.signal_period)
    verdict = grade(cases, args.v_floor, args.min_ratio, signal)
    report = {
        "benchmark": "startup_quiescence",
        "question": ("how still is a zero-force pool / flume, which quantity "
                     "controls it, and what is its noise floor worth?"),
        "definitions": {
            "initial_pressure_deviation": (
                "max_i |p_i - rho0 g (H - y_i)| / (rho0 g H) over particles at "
                "least 2h from every boundary, t = 0"),
            "v_late_rms": ("RMS of max_i |v_i| over the last quarter of the "
                           "run, sampled every --sample-every steps"),
            "interior": "particles at least 2h from every boundary",
        },
        "gate_parameters": {"v_floor_m_s": args.v_floor,
                            "min_sensitivity_ratio": args.min_ratio},
        "cases": cases,
        "flume_ablation": [{"label": c["label"],
                            "interior_v_late_rms": c["interior_v_late_rms"],
                            "v_peak": c["v_peak"]}
                           for c in cases if c["state"] == "flume"],
        **verdict,
    }
    text = json.dumps(report, indent=1)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n")
        print(f"wrote {args.json_out}")
    for gate in verdict["gates"]:
        print(f"{gate['id']}: measured={gate['measured']} limit={gate['limit']}"
              f" -> {'pass' if gate['pass'] else 'FAIL'}")
    for finding in verdict["findings"]:
        print(f"{finding['id']} [{finding['status']}]: "
              f"measured={finding['measured']:.4f} limit={finding['limit']}")
    noise = max(c["interior_v_late_rms"] for c in cases)
    print(f"noise floor / u_orbital = {noise / signal['u_orbital_surface_m_s']:.3f}"
          f" (u_orbital={signal['u_orbital_surface_m_s']:.3f} m/s)")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())