"""Wave-tank benchmark: case matrix, gates and JSON evidence.

Each case is a self-contained WCSPH wave-tank run built from
``tensorsph.wave_tank``. The driver is per-case (``--case NAME``) so that the
matrix can be sharded over separate processes; it writes one JSON file per case,
which ``summarise.py`` turns into the verdict matrix of the report.

Cases
-----
``hydrostatic``      still water, band-averaged pressure against ``rho0 g (h-y)``
``dispersion_kh*``   piston-made regular wave, two-probe phase-speed measurement
``decay10``          wave height after ten wavelengths of propagation
``reflection``       sponge layer on: residual reflection coefficient
``reflection_control`` sponge layer off: the same tank, R should be near 1
``ladder_dp*``       resolution ladder at fixed kh, dispersion error against dp
``csweep_c*``        artificial sound speed sweep at fixed kh
``solitary``         Boussinesq solitary wave: celerity, waveform, mass

The gates are the ones stated in the campaign node; a case reports
``passed: false`` rather than adjusting its own criterion.

Usage
-----
    python benchmarks/wave_tank/run.py --case dispersion_kh3 --threads 1
    python benchmarks/wave_tank/run.py --list
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorsph.wave_tank import (           # noqa: E402
    WaveTankConfig,
    crest_time,
    harmonic_fit,
    reflection_coefficient,
    run_tank,
    solitary_beta,
    solitary_celerity,
    two_probe_dispersion,
    wave_number,
    waveform_error,
)

LAMBDA_OVER_H = 2.0 * math.pi / 3.0        # wavelength in depths at kh = 3


# --------------------------------------------------------------------------- #
# Case matrix
# --------------------------------------------------------------------------- #

CASES: dict[str, dict] = {
    # (a) hydrostatic: still water, pressure against rho g h
    "hydrostatic": dict(
        kind="regular", maker="none", dx_frac=0.02, kh=3.0, length_waves=3.0,
        probes=(0.5, 1.0, 1.5, 2.0), height_frac=0.1, sponge_strength=0.0,
        periods=1.6, ramp_periods=0.0, pressure_bands=6, courant=0.45,
        viscosity=1.0e-2,
    ),
    # (b) dispersion at four kh values (relaxation-zone maker, 5 % H/h)
    "dispersion_kh0.5": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=0.5, length_waves=2.4,
        probes=(0.4, 0.7, 1.0, 1.3), height_frac=0.1, sponge_wavelengths=0.9,
        periods=2.4, ramp_periods=0.5, courant=0.45,
    ),
    "dispersion_kh1": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=1.0, length_waves=2.4,
        probes=(0.4, 0.7, 1.0, 1.3), height_frac=0.1, sponge_wavelengths=0.9,
        periods=2.4, ramp_periods=0.5, courant=0.45,
    ),
    "dispersion_kh2": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=2.0, length_waves=2.4,
        probes=(0.4, 0.7, 1.0, 1.3), height_frac=0.1, sponge_wavelengths=0.9,
        periods=2.4, ramp_periods=0.5, courant=0.45,
    ),
    "dispersion_kh3": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=3.0, length_waves=2.4,
        probes=(0.4, 0.7, 1.0, 1.3), height_frac=0.1, sponge_wavelengths=0.9,
        periods=2.4, ramp_periods=0.5, courant=0.45,
    ),
    # (c) ten wavelengths of propagation, seven probes along the tank
    "decay10": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=3.0, length_waves=11.6,
        probes=(0.5, 3.0, 6.0, 9.0, 9.4, 9.8, 10.2), height_frac=0.1,
        sponge_wavelengths=0.9, periods=11.0, ramp_periods=0.5, courant=0.5,
        c0_factor=8.0,
    ),
    # (e) reflection: sponge on (gate) and off (control)
    "reflection": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=3.0, length_waves=2.4,
        probes=(0.4, 0.7, 1.0, 1.3), height_frac=0.1, sponge_wavelengths=0.9,
        periods=2.4, ramp_periods=0.5, courant=0.45,
    ),
    "reflection_control": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=3.0, length_waves=2.4,
        probes=(0.4, 0.7, 1.0, 1.3), height_frac=0.1, sponge_wavelengths=0.9,
        sponge_strength=0.0, periods=2.4, ramp_periods=0.5, courant=0.45,
    ),
    # (g) resolution ladder at fixed kh = 3
    "ladder_dp0.020": dict(
        kind="regular", maker="relaxation", dx_frac=0.020, kh=3.0, length_waves=2.0,
        probes=(0.4, 0.7, 1.0), height_frac=0.1, sponge_wavelengths=0.8,
        periods=2.2, ramp_periods=0.5, courant=0.45,
    ),
    "ladder_dp0.015": dict(
        kind="regular", maker="relaxation", dx_frac=0.015, kh=3.0, length_waves=2.0,
        probes=(0.4, 0.7, 1.0), height_frac=0.1, sponge_wavelengths=0.8,
        periods=2.2, ramp_periods=0.5, courant=0.45,
    ),
    "ladder_dp0.010": dict(
        kind="regular", maker="relaxation", dx_frac=0.010, kh=3.0, length_waves=2.0,
        probes=(0.4, 0.7, 1.0), height_frac=0.1, sponge_wavelengths=0.8,
        periods=2.2, ramp_periods=0.5, courant=0.45,
    ),
    "ladder_dp0.0075": dict(
        kind="regular", maker="relaxation", dx_frac=0.0075, kh=3.0, length_waves=2.0,
        probes=(0.4, 0.7, 1.0), height_frac=0.1, sponge_wavelengths=0.8,
        periods=2.2, ramp_periods=0.5, courant=0.45,
    ),
    # artificial sound-speed sweep at fixed kh
    "csweep_c5": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=3.0, length_waves=2.0,
        probes=(0.4, 0.7, 1.0), height_frac=0.1, sponge_wavelengths=0.8,
        periods=2.2, ramp_periods=0.5, courant=0.45, c0_factor=5.0,
    ),
    "csweep_c20": dict(
        kind="regular", maker="relaxation", dx_frac=0.02, kh=3.0, length_waves=2.0,
        probes=(0.4, 0.7, 1.0), height_frac=0.1, sponge_wavelengths=0.8,
        periods=2.2, ramp_periods=0.5, courant=0.45, c0_factor=20.0,
    ),
    # piston maker kept as the wavemaker-transfer anchor (measured H/S)
    "piston_transfer": dict(
        kind="regular", maker="piston", dx_frac=0.02, kh=3.0, length_waves=2.0,
        probes=(0.4, 0.7, 1.0), height_frac=0.1, sponge_wavelengths=0.8,
        periods=2.4, ramp_periods=0.5, courant=0.45, viscosity=1.0e-2,
    ),
    # (d) solitary wave, initial value problem, relaxation zone off
    "solitary": dict(
        kind="solitary", maker="none", dx_frac=0.02, kh=3.0, length_m=7.0,
        probes=(2.0, 4.6), height_frac=0.2, sponge_wavelengths=0.25,
        courant=0.45, end_seconds=2.2, snapshot_offsets=(0.55, 0.8),
    ),
}

DEFAULT_DEPTH = 0.5


def build_config(name: str, depth: float = DEFAULT_DEPTH,
                 end_time_scale: float = 1.0) -> WaveTankConfig:
    spec = CASES[name]
    kh = spec["kh"]
    wavelength = 2.0 * math.pi * depth / kh
    if spec["kind"] == "solitary":
        length = spec["length_m"]
    else:
        length = spec["length_waves"] * wavelength
    cfg = WaveTankConfig(
        dx=spec["dx_frac"] * depth,
        depth=depth,
        length=length,
        kh=kh,
        wave_height=spec["height_frac"] * depth,
        maker=spec["maker"],
        wave_kind=spec["kind"],
        max_courant=spec["courant"],
        viscosity=spec.get("viscosity", 1.0e-4),
        c0_factor=spec.get("c0_factor", 10.0),
        sponge_wavelengths=spec.get("sponge_wavelengths", 1.0),
        sponge_strength=spec.get("sponge_strength", 3.0),
        ramp_periods=spec.get("ramp_periods", 1.0),
        probe_offsets=spec["probes"],
        pressure_bands=spec.get("pressure_bands", 0),
    )
    if spec["kind"] == "solitary":
        end = spec["end_seconds"] * end_time_scale
    else:
        end = spec["periods"] * cfg.period() * end_time_scale
    cfg.end_time = end
    cfg.steps = int(math.ceil(end / cfg.dt))
    return cfg


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def steady_window(run, cfg, periods: float = 1.2) -> list[int]:
    """Indices of the record's tail used by every harmonic fit."""
    if run.probe_x and cfg.wave_kind == "regular":
        window = min(periods * cfg.period(), 0.55 * run.simulated_time)
    else:
        window = 0.6 * run.simulated_time
    return [index for index, t in enumerate(run.times) if t >= run.times[-1] - window]


def analyse_regular(run, cfg) -> dict:
    """Probe amplitudes, two-probe dispersion, reflection and decay."""
    index = steady_window(run, cfg)
    times = [run.times[i] for i in index]
    fits = [harmonic_fit(times, [run.probes[k][i] for i in index], cfg.omega())
            for k in range(len(run.probe_x))]
    out = {
        "probe_x": list(run.probe_x),
        "probe_fits": [fit.as_dict() for fit in fits],
        "probe_heights": [fit.height for fit in fits],
        "omega_imposed": cfg.omega(),
        "wavelength_imposed": cfg.wavelength(),
        "wavelength_theory": cfg.wavelength(),
        "k_theory": cfg.wave_number_k(),
        "celerity_theory_imposed": cfg.celerity(),
        "stroke": cfg.stroke(),
        "transfer_theory": cfg.transfer_ratio(),
    }
    # Two-probe dispersion on the first pair (separation < lambda / 2).
    separation = run.probe_x[1] - run.probe_x[0]
    pair = two_probe_dispersion(times, [run.probes[0][i] for i in index],
                               [run.probes[1][i] for i in index],
                               separation, cfg.omega(), cfg.depth, cfg.gravity)
    out["dispersion"] = pair.as_dict()
    out["dispersion"]["separation"] = separation
    out["dispersion"]["k_error"] = abs(pair.k_measured - cfg.wave_number_k()) / cfg.wave_number_k()
    out["dispersion"]["k_theory_from_omega"] = wave_number(cfg.omega(), cfg.depth, cfg.gravity)

    # Reflection from the spatial envelope. The campaign formula
    # R = (H_max - H_min)/(H_max + H_min) over the probe array is only valid if
    # the incident wave does not decay between the probes; the decay that SPH adds
    # makes that formula report decay as reflection. A second, decay-corrected
    # estimate is therefore reported next to it: fit log H against x, divide it
    # out and take the envelope of the residual ratio.
    heights = [fit.height for fit in fits]
    reflection, high, low = reflection_coefficient(heights)
    detrended = None
    decay_per_wavelength = None
    if len(heights) >= 3 and all(value > 0.0 for value in heights):
        positions = [run.probe_x[index] for index in range(len(heights))]
        logs = [math.log(value) for value in heights]
        mean_x = sum(positions) / len(positions)
        mean_y = sum(logs) / len(logs)
        denominator = sum((xi - mean_x) ** 2 for xi in positions)
        slope = (sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(positions, logs))
                 / denominator) if denominator else 0.0
        ratios = [value / math.exp(mean_y + slope * (xi - mean_x))
                  for xi, value in zip(positions, heights)]
        detrended = (max(ratios) - min(ratios)) / (max(ratios) + min(ratios))
        decay_per_wavelength = math.exp(slope * cfg.wavelength())
    out["reflection"] = {"R": reflection, "R_detrended": detrended,
                         "height_max": high, "height_min": low,
                         "height_ratio_per_wavelength": decay_per_wavelength,
                         "probes_used": list(run.probe_x)}
    if cfg.maker in ("piston", "flap") and cfg.stroke() > 0.0:
        out["transfer_measured"] = fits[0].height / cfg.stroke()
        out["transfer_theory"] = cfg.transfer_ratio()
        out["transfer_error"] = (abs(out["transfer_measured"] - cfg.transfer_ratio())
                                 / cfg.transfer_ratio())
    else:
        out["transfer_measured"] = None
        out["transfer_theory"] = None
        out["transfer_error"] = None

    # Amplitude decay between the first and the last probe.
    out["decay"] = {"h_first": fits[0].height, "h_last": fits[-1].height,
                    "ratio": fits[-1].height / fits[0].height if fits[0].height else float("nan"),
                    "wavelengths_between": (run.probe_x[-1] - run.probe_x[0]) / cfg.wavelength()}
    return out


def analyse_hydrostatic(run, cfg) -> dict:
    """Band-averaged pressure against the analytical ``rho0 g (depth - y)``."""
    if not run.band_history:
        raise ValueError("no band history: set pressure_bands > 0")
    half = [ratios for t, ratios in run.band_history
            if t >= 0.5 * run.band_history[-1][0]]
    count = len(run.band_history[0][1])
    means, maxima = [], []
    for band in range(count):
        values = [row[band] for row in half if math.isfinite(row[band])]
        if not values:
            means.append(float("nan"))
            maxima.append(float("nan"))
            continue
        means.append(sum(values) / len(values))
        maxima.append(max(abs(v) for v in values))
    # The topmost band is truncated by the free surface, as in
    # benchmarks/hydrostatic: it is reported but excluded from the gate.
    interior = [abs(v) for v in means[:-1] if math.isfinite(v)]
    return {"band_mean_error": means, "band_max_abs_error": maxima,
            "interior_mean_abs_error": sum(interior) / len(interior) if interior else float("nan"),
            "bands": count, "averaged_samples": len(half)}


def analyse_solitary(run, cfg, snapshot_fraction: float = 0.55) -> dict:
    """Phase speed from crest arrival, waveform L2 and mass conservation."""
    t1, e1 = crest_time(run.times, run.probes[0])
    t2, e2 = crest_time(run.times, run.probes[1])
    separation = run.probe_x[1] - run.probe_x[0]
    measured = separation / (t2 - t1)
    theory = solitary_celerity(cfg.wave_height, cfg.depth, cfg.gravity)
    out = {"crest_times": [t1, t2], "crest_heights": [e1, e2],
           "celerity_measured": measured, "celerity_theory": theory,
           "celerity_error": abs(measured - theory) / theory,
           "separation": separation,
           "beta": solitary_beta(cfg.wave_height, cfg.depth),
           "wave_width": 1.0 / solitary_beta(cfg.wave_height, cfg.depth)}
    if run.snapshots:
        target = snapshot_fraction * run.simulated_time
        snapshot = min(run.snapshots, key=lambda item: abs(item[0] - target))
        t_snap, centres, eta, _, _ = snapshot
        crest = float(centres[int(torch.argmax(eta))])
        l2, samples = waveform_error(centres, eta, cfg.wave_height, cfg.depth, crest)
        # Amplitude actually present at the station, from a local fit window.
        local = (centres - crest).abs() < 2.0 / solitary_beta(cfg.wave_height, cfg.depth)
        amplitude = float(eta[local].max())
        out["waveform"] = {"snapshot_time": t_snap, "crest": crest,
                           "l2_error": l2, "samples": samples,
                           "crest_height": amplitude,
                           "crest_error": abs(amplitude - cfg.wave_height) / cfg.wave_height}
    out["mass"] = {"mass_error": run.mass_error, "volume_error": run.volume_error,
                   "volume_initial": run.volume_initial, "volume_final": run.volume_final}
    return out


def gates(name: str, cfg, run, analysis) -> list[dict]:
    """Evaluate the campaign criteria that this case owns."""
    checks = []
    if name == "hydrostatic":
        error = analysis["interior_mean_abs_error"]
        checks.append({"criterion": "(a) hydrostatic pressure vs rho g h <= 1%",
                       "value": error, "threshold": 0.01, "passed": error <= 0.01})
    if name.startswith("dispersion_"):
        error = analysis["dispersion"]["celerity_error"]
        checks.append({"criterion": "(b) linear dispersion celerity error <= 3%",
                       "value": error, "threshold": 0.03, "passed": error <= 0.03})
        k_error = analysis["dispersion"]["k_error"]
        checks.append({"criterion": "(b') wave number error <= 3%",
                       "value": k_error, "threshold": 0.03, "passed": k_error <= 0.03})
    if name == "decay10":
        ratio = analysis["decay"]["ratio"]
        checks.append({"criterion": "(c) H/H0 after ten wavelengths >= 0.95",
                       "value": ratio, "threshold": 0.95, "passed": ratio >= 0.95})
    if name == "reflection":
        value = analysis["reflection"]["R"]
        checks.append({"criterion": "(e) reflection coefficient <= 5%",
                       "value": value, "threshold": 0.05, "passed": value <= 0.05})
    if name == "reflection_control":
        value = analysis["reflection"]["R"]
        checks.append({"criterion": "(e') control without sponge: R >= 0.8 (the probe "
                                    "array must be able to see a reflector)",
                       "value": value, "threshold": 0.8, "passed": value >= 0.8})
    if name.startswith("ladder_") or name.startswith("csweep_"):
        error = analysis["dispersion"]["celerity_error"]
        checks.append({"criterion": "diagnostic: celerity error (input to the "
                                    "convergence verdict)", "value": error,
                       "threshold": None, "passed": True})
    if name == "solitary":
        checks.append({"criterion": "(d) solitary phase speed error <= 5%",
                       "value": analysis["celerity_error"], "threshold": 0.05,
                       "passed": analysis["celerity_error"] <= 0.05})
        l2 = analysis["waveform"]["l2_error"]
        checks.append({"criterion": "(d) solitary waveform L2 error <= 8%",
                       "value": l2, "threshold": 0.08, "passed": l2 <= 0.08})
        mass = analysis["mass"]["mass_error"]
        checks.append({"criterion": "(d) mass conservation <= 1%", "value": mass,
                       "threshold": 0.01, "passed": mass <= 0.01})
    return checks


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_case(name: str, threads: int = 1, end_time_scale: float = 1.0,
             depth: float = DEFAULT_DEPTH, dry_run: bool = False) -> dict:
    torch.set_num_threads(threads)
    cfg = build_config(name, depth=depth, end_time_scale=end_time_scale)
    cfg.validate()
    spec = CASES[name]
    payload = {"case": name, "spec": {k: str(v) for k, v in spec.items()},
               "config": {"depth": cfg.depth, "dx": cfg.dx, "dx_over_depth": cfg.dx / cfg.depth,
                          "length": cfg.length, "kh": cfg.kh, "wave_height": cfg.wave_height,
                          "sound_speed": cfg.sound_speed, "c0_factor": cfg.c0_factor,
                          "dt": cfg.dt, "steps": cfg.steps, "end_time": cfg.end_time,
                          "period": cfg.period(), "wavelength": cfg.wavelength(),
                          "sponge_start": cfg.sponge_start(),
                          "sponge_strength": cfg.sponge_strength,
                          "probe_x": cfg.probe_positions(),
                          "height_over_depth": cfg.wave_height / cfg.depth,
                          "max_courant": cfg.max_courant,
                          "threads": threads}}
    if dry_run:
        payload["dry_run"] = True
        return payload

    started = time.time()
    run = run_tank(cfg, snapshot_times=spec.get("snapshot_offsets", ()))
    payload["run"] = run.as_dict()
    payload["run"]["wall_seconds"] = time.time() - started
    if run.final_positions is not None:
        payload["run"]["particles"] = int(run.final_positions.shape[0])

    if cfg.pressure_bands > 0:
        analysis = analyse_hydrostatic(run, cfg)
    elif cfg.wave_kind == "solitary":
        analysis = analyse_solitary(run, cfg,
                                    snapshot_fraction=spec.get("snapshot_offsets", (0.55,))[0])
    else:
        analysis = analyse_regular(run, cfg)
    payload["analysis"] = analysis
    payload["gates"] = gates(name, cfg, run, analysis)
    payload["passed"] = all(gate["passed"] for gate in payload["gates"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--end-time-scale", type=float, default=1.0)
    parser.add_argument("--depth", type=float, default=DEFAULT_DEPTH)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, spec in CASES.items():
            print(f"{name:22s} {spec['kind']:10s} kh={spec['kh']:.2f} "
                  f"dx={spec['dx_frac']:.3f}h")
        return 0
    if args.case is None:
        parser.error("--case or --list is required")

    payload = run_case(args.case, threads=args.threads,
                       end_time_scale=args.end_time_scale, depth=args.depth,
                       dry_run=args.dry_run)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    if payload.get("gates") and not payload.get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
