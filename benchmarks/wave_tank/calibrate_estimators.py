"""Calibrate the wave-tank measurement chain on synthetic signals with a known answer.

The gate matrix of ``benchmarks/wave_tank/summarise.py`` is built from three
estimators in ``tensorsph.wave_tank``:

* ``two_probe_dispersion``  -> criterion (b), and the ``celerity_error`` column
* ``spatial_wave_number``   -> criterion (b) gate value (``celerity_error_spatial``)
* ``detrended_reflection``  -> criterion (e) gate value (``R_detrended``)

Those numbers are what the campaign verdicts are made of, yet none of the three
had ever been fed a signal whose answer is known: the tests cover the dispersion
*relation*, the harmonic fit and the envelope formula separately, but not the
chain ``probe series -> harmonic fit -> estimator`` that produces the gate.
This script closes that gap. It builds probe time series from a closed-form wave
field (known ``k``, known spatial decay ``alpha``, known reflection coefficient
``R`` and standing-wave phase ``psi``), pushes them through the same estimator
calls that ``run.py`` / ``summarise.py`` use, and reports the recovered values.

Five experiments:

``positive_control``     noiseless, decaying, non-reflecting train -> both
                         estimators must return the imposed ``k`` and ``alpha``
                         to round-off, the envelope ``R`` must be non-zero
                         (that is the documented defect) and ``R_detrended`` zero.
``reflection_sampling``  known ``R`` in 0.02..0.4, standing-wave phase swept over
                         a half wavelength: the envelope estimators can only see
                         the extremes they happen to sample, so what is measured
                         is a *biased* ``R``. Reports the bias envelope.
``noise_floor``          white noise on the probe records; the induced celerity
                         error is the instrument floor of criterion (b), compared
                         against the analytic phase-noise prediction.
``alias``                probe spacing above ``lambda/2``: the wrapped phase
                         difference aliases silently, which is what ``spatial_wave_number``
                         does today on any probe set spaced wider than half a
                         wavelength (``decay10`` uses 2.5 wavelengths).
``span_sensitivity``     how many probes / what span is needed for a given
                         precision -- the input for choosing a probe layout.

Usage
-----
    python benchmarks/wave_tank/calibrate_estimators.py            # full sweep
    python benchmarks/wave_tank/calibrate_estimators.py --quick     # CI-sized

Writes ``runs/wave_tank/estimator_calibration.json`` and, when matplotlib is
available, ``docs/figures/wave_tank_estimator_calibration.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorsph.wave_tank import (            # noqa: E402
    angular_frequency,
    detrended_reflection,
    harmonic_fit,
    reflection_coefficient,
    spatial_wave_number,
    two_probe_dispersion,
)

DEPTH = 0.5          # m, same as benchmarks/wave_tank/run.py
GRAVITY = 9.81
KH = 3.0             # deep-water-ish case used by criteria (b)/(c)/(e)
HEIGHT = 0.05        # m, H/h = 0.1


# --------------------------------------------------------------------------- #
# Synthetic wave field
# --------------------------------------------------------------------------- #

def wave_number(kh: float = KH, depth: float = DEPTH) -> float:
    return kh / depth


def probe_series(times, positions, k, omega, amplitude=HEIGHT / 2.0,
                 alpha=0.0, reflection=0.0, psi=0.0):
    """Probe records of a decaying incident train plus a reflected train.

    ``eta(x, t) = a e^{-alpha x} cos(omega t - k x)
                + R a e^{-alpha x} cos(omega t + k x + psi)``

    The reflection term is given the same spatial decay factor as the incident
    one: the aim is a *known* answer for the estimator, not a model of a
    particular reflector. ``psi`` is the phase of the reflected wave at ``x = 0``,
    which is what decides where the standing-wave nodes sit relative to the probes.
    """
    t = np.asarray(times, dtype=float)
    out = []
    for x in positions:
        decay = math.exp(-alpha * x)
        incident = decay * np.cos(omega * t - k * x)
        reflected = reflection * decay * np.cos(omega * t + k * x + psi)
        out.append(amplitude * (incident + reflected))
    return out


def measure(positions, series, times, omega, *, use_spatial=True):
    """Run the production chains on one synthetic probe set."""
    fits = [harmonic_fit(times, values, omega) for values in series]
    heights = [fit.height for fit in fits]
    phases = [fit.phase for fit in fits]
    result = {"heights": heights, "phases": phases,
              "amplitude_residual": [fit.residual for fit in fits]}
    if len(positions) >= 2:
        separation = positions[1] - positions[0]
        try:
            pair = two_probe_dispersion(times, series[0], series[1], separation,
                                        omega, DEPTH, GRAVITY)
            result["two_probe"] = pair.as_dict()
        except ValueError as error:                 # aliased layout, rejected
            result["two_probe"] = {"rejected": str(error)}
    if use_spatial and len(positions) >= 3:
        try:
            result["spatial"] = spatial_wave_number(positions, heights, phases,
                                                    omega, DEPTH, GRAVITY)
        except ValueError as error:                 # aliased layout, rejected
            result["spatial"] = {"rejected": str(error)}
    envelope, high, low = reflection_coefficient(heights)
    detrended, per_wavelength = detrended_reflection(
        positions, heights, 2.0 * math.pi / abs(wave_number()))
    result["reflection"] = {
        "envelope": envelope, "detrended": detrended,
        "height_max": high, "height_min": low,
        "height_ratio_per_wavelength": per_wavelength,
    }
    return result


def sample_times(omega, periods=1.2, samples_per_period=200):
    period = 2.0 * math.pi / omega
    dt = period / samples_per_period
    count = int(round(periods * samples_per_period))
    return dt * np.arange(count, dtype=float)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def experiment_positive_control(probe_offsets=(0.4, 0.7, 1.0, 1.3)):
    """Noiseless decaying, non-reflecting train: the answer is known exactly."""
    k = wave_number()
    omega = angular_frequency(k, DEPTH, GRAVITY)
    wavelength = 2.0 * math.pi / k
    positions = [off * wavelength for off in probe_offsets]
    times = sample_times(omega)
    alpha = 2.0 / (wavelength * 10.0)      # 20 % amplitude loss per wavelength
    series = probe_series(times, positions, k, omega, alpha=alpha)
    got = measure(positions, series, times, omega)
    return {
        "positions": positions, "k_imposed": k, "alpha_imposed": alpha,
        "omega": omega,
        "two_probe_k_error": abs(got["two_probe"]["k_measured"] - k) / k,
        "spatial_k_error": abs(got["spatial"]["k"] - k) / k,
        "spatial_alpha_error": abs(got["spatial"]["alpha"] - alpha) / alpha,
        "celerity_error_two_probe": got["two_probe"]["celerity_error"],
        "celerity_error_spatial": got["spatial"]["celerity_error"],
        "envelope_R": got["reflection"]["envelope"],
        "detrended_R": got["reflection"]["detrended"],
        "decay_ratio_per_wavelength": got["reflection"]["height_ratio_per_wavelength"],
        "analytic_decay_ratio_per_wavelength": math.exp(-alpha * wavelength),
    }


def experiment_decay_bias(probe_offsets=(0.4, 0.7, 1.0, 1.3)):
    """How much of the *envelope* reflection reading is pure spatial decay.

    The campaign formula ``(Hmax - Hmin)/(Hmax + Hmin)`` has no way to tell a
    decaying incident wave from a standing one; the detrended estimator should
    remove it. This quantifies both, with zero true reflection.
    """
    k = wave_number()
    omega = angular_frequency(k, DEPTH, GRAVITY)
    wavelength = 2.0 * math.pi / k
    positions = [off * wavelength for off in probe_offsets]
    times = sample_times(omega)
    rows = []
    for loss_per_wavelength in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40):
        alpha = -math.log(1.0 - loss_per_wavelength) / wavelength
        series = probe_series(times, positions, k, omega, alpha=alpha)
        got = measure(positions, series, times, omega)
        rows.append({
            "loss_per_wavelength": loss_per_wavelength,
            "alpha": alpha,
            "envelope_R": got["reflection"]["envelope"],
            "detrended_R": got["reflection"]["detrended"],
            "recovered_ratio": got["reflection"]["height_ratio_per_wavelength"],
            "true_ratio": math.exp(-alpha * wavelength),
        })
    return {"true_R": 0.0, "rows": rows, "probe_offsets": list(probe_offsets)}


def experiment_reflection_sampling(true_R=(0.02, 0.05, 0.10, 0.20, 0.40),
                                   probe_offsets=(0.4, 0.7, 1.0, 1.3),
                                   phases=12, alpha_per_wavelength=0.05,
                                   noise=0.0, seed=7):
    """Bias of both reflection estimators as a function of the standing-wave phase.

    With probes 0.3 wavelengths apart the standing-wave envelope has period
    ``lambda/2``, i.e. 1.67 probe spacings: whether the maximum and the minimum
    are sampled at all depends on where the probes happen to sit relative to the
    nodes. Sweeping ``psi`` over half a wavelength turns that into a distribution.
    """
    k = wave_number()
    omega = angular_frequency(k, DEPTH, GRAVITY)
    wavelength = 2.0 * math.pi / k
    positions = [off * wavelength for off in probe_offsets]
    times = sample_times(omega)
    alpha = -math.log(1.0 - alpha_per_wavelength) / wavelength
    rng = np.random.default_rng(seed)
    rows = []
    for value in true_R:
        envelope, detrended = [], []
        for step in range(phases):
            psi = math.pi * step / phases
            series = probe_series(times, positions, k, omega, alpha=alpha,
                                 reflection=value, psi=psi)
            if noise:
                series = [record + rng.normal(0.0, noise * HEIGHT / 2.0,
                                              size=record.size)
                          for record in series]
            got = measure(positions, series, times, omega)
            envelope.append(got["reflection"]["envelope"])
            detrended.append(got["reflection"]["detrended"])
        rows.append({
            "true_R": value,
            "envelope_mean": float(np.mean(envelope)),
            "envelope_min": float(np.min(envelope)),
            "envelope_max": float(np.max(envelope)),
            "detrended_mean": float(np.mean(detrended)),
            "detrended_min": float(np.min(detrended)),
            "detrended_max": float(np.max(detrended)),
            "detrended_bias": float(np.mean(detrended) - value),
            "detrended_undercount": float(1.0 - np.mean(detrended) / value),
        })
    return {"probe_offsets": list(probe_offsets), "alpha_per_wavelength":
            alpha_per_wavelength, "noise_relative": noise, "phases": phases,
            "rows": rows}


def experiment_noise_floor(levels=(0.002, 0.005, 0.01, 0.02, 0.05),
                           probe_offsets=(0.4, 0.7, 1.0, 1.3),
                           realisations=200, seed=11):
    """Celerity error induced by white noise on the probe records.

    This is the number that decides whether a 3 % celerity gate can be resolved
    at all: if the probe records carry ``sigma`` relative noise, ``k`` cannot be
    measured better than the phase noise divided by the probe separation.

    The analytic line is the least-squares projection error of a sinusoid in
    white noise, ``sigma_phi ~ sigma / A * sqrt(2/N)`` with ``N`` samples, which
    turns into ``delta c / c = sigma_phi / (k * separation)``.
    """
    k = wave_number()
    omega = angular_frequency(k, DEPTH, GRAVITY)
    wavelength = 2.0 * math.pi / k
    positions = [off * wavelength for off in probe_offsets]
    times = sample_times(omega)
    samples = times.size
    rng = np.random.default_rng(seed)
    rows = []
    for level in levels:
        tp, sp, spatial_k = [], [], []
        for _ in range(realisations):
            series = probe_series(times, positions, k, omega)
            series = [record + rng.normal(0.0, level * HEIGHT / 2.0,
                                          size=record.size) for record in series]
            got = measure(positions, series, times, omega)
            tp.append(got["two_probe"]["celerity_error"])
            sp.append(got["spatial"]["celerity_error"])
            spatial_k.append(abs(got["spatial"]["k"] - k) / k)
        separation = positions[1] - positions[0]
        sigma_phi = level * math.sqrt(2.0 / samples)
        analytic = sigma_phi / (k * separation)
        rows.append({
            "relative_noise": level,
            "two_probe_median": float(np.median(tp)),
            "two_probe_p95": float(np.percentile(tp, 95)),
            "spatial_median": float(np.median(sp)),
            "spatial_p95": float(np.percentile(sp, 95)),
            "spatial_k_median": float(np.median(spatial_k)),
            "analytic_two_probe": analytic,
            "fraction_two_probe_over_3pct": float(np.mean(np.array(tp) > 0.03)),
            "fraction_spatial_over_3pct": float(np.mean(np.array(sp) > 0.03)),
        })
    return {"probe_offsets": list(probe_offsets), "samples_per_record": int(samples),
            "realisations": realisations, "rows": rows}


def experiment_alias(spacings=(0.3, 0.45, 0.5, 0.6, 0.8, 1.5, 2.5),
                     probes=4, noise=0.0):
    """Silent aliasing of the wrapped phase difference above half a wavelength."""
    k = wave_number()
    omega = angular_frequency(k, DEPTH, GRAVITY)
    wavelength = 2.0 * math.pi / k
    times = sample_times(omega)
    rng = np.random.default_rng(3)
    rows = []
    for spacing in spacings:
        positions = [0.4 * wavelength + spacing * wavelength * index
                     for index in range(probes)]
        series = probe_series(times, positions, k, omega)
        if noise:
            series = [record + rng.normal(0.0, noise * HEIGHT / 2.0,
                                          size=record.size) for record in series]
        got = measure(positions, series, times, omega)
        resolved = spatial_wave_number(positions, got["heights"], got["phases"],
                                       omega, DEPTH, GRAVITY, k_prior=k)
        delta = ((got["phases"][1] - got["phases"][0] + math.pi)
                 % (2.0 * math.pi)) - math.pi
        rejected = "rejected" in got["spatial"]
        rows.append({
            "spacing_over_wavelength": spacing,
            "wrapped_phase_step": abs(delta),
            "unambiguous": abs(delta) < math.pi,
            "rejected_without_prior": rejected,
            "prior_k_error": abs(resolved["k"] - k) / k,
            "alias_risk": resolved["alias_risk"],
            "branch_tolerance": resolved["branch_tolerance"],
            "prior_celerity_error": resolved["celerity_error"],
        })
    return {"probe_count": probes, "rows": rows}


def experiment_reflection_contamination(values=(0.0, 0.02, 0.05, 0.10, 0.20),
                                      probe_offsets=(0.4, 0.7, 1.0, 1.3),
                                      phases=8, alpha_per_wavelength=0.05):
    """Dispersion error caused by a reflected component the array cannot average.

    The probes span 0.9 wavelengths, i.e. less than one period of the
    ``lambda/2`` standing-wave pattern, so a reflected wave is not averaged out by
    the spatial fit but read as a phase tilt. This is the error the estimator
    *cannot* remove, and it sets what the sponge layer has to achieve for
    criterion (b) to mean anything.
    """
    k = wave_number()
    omega = angular_frequency(k, DEPTH, GRAVITY)
    wavelength = 2.0 * math.pi / k
    positions = [off * wavelength for off in probe_offsets]
    times = sample_times(omega)
    alpha = -math.log(1.0 - alpha_per_wavelength) / wavelength
    rows = []
    for value in values:
        tp, sp = [], []
        for step in range(phases):
            psi = math.pi * step / phases
            series = probe_series(times, positions, k, omega, alpha=alpha,
                                 reflection=value, psi=psi)
            got = measure(positions, series, times, omega)
            tp.append(got["two_probe"]["celerity_error"])
            sp.append(got["spatial"]["celerity_error"])
        rows.append({"true_R": value,
                     "two_probe_median": float(np.median(tp)),
                     "two_probe_max": float(np.max(tp)),
                     "spatial_median": float(np.median(sp)),
                     "spatial_min": float(np.min(sp)),
                     "spatial_max": float(np.max(sp)),
                     "fraction_over_3pct": float(np.mean(np.array(sp) > 0.03))})
    return {"probe_offsets": list(probe_offsets), "phases": phases, "rows": rows}


def experiment_span_sensitivity(probe_counts=(2, 3, 4, 6, 8), noise=0.01,
                                realisations=100, seed=19):
    """Celerity-error floor versus the number of probes in the same span."""
    k = wave_number()
    omega = angular_frequency(k, DEPTH, GRAVITY)
    wavelength = 2.0 * math.pi / k
    times = sample_times(omega)
    rng = np.random.default_rng(seed)
    rows = []
    for count in probe_counts:
        offsets = [0.4 + 0.9 * index / max(count - 1, 1) for index in range(count)]
        positions = [off * wavelength for off in offsets]
        errors = []
        for _ in range(realisations):
            series = probe_series(times, positions, k, omega)
            series = [record + rng.normal(0.0, noise * HEIGHT / 2.0,
                                          size=record.size) for record in series]
            got = measure(positions, series, times, omega, use_spatial=count >= 3)
            errors.append(got["spatial"]["celerity_error"] if count >= 3
                          else got["two_probe"]["celerity_error"])
        rows.append({
            "probes": count, "span_over_wavelength": 0.9,
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
        })
    return {"relative_noise": noise, "realisations": realisations, "rows": rows}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_all(quick=False):
    payload = {
        "description": "wave-tank measurement-chain calibration (synthetic known answer)",
        "case": {"depth_m": DEPTH, "gravity": GRAVITY, "kh": KH,
                 "wave_height_m": HEIGHT},
        "probe_layout_production": [0.4, 0.7, 1.0, 1.3],
    }
    payload["positive_control"] = experiment_positive_control()
    payload["decay_bias"] = experiment_decay_bias()
    payload["reflection_sampling"] = experiment_reflection_sampling(
        phases=6 if quick else 12)
    payload["noise_floor"] = experiment_noise_floor(
        realisations=40 if quick else 200)
    payload["alias"] = experiment_alias()
    payload["reflection_contamination"] = experiment_reflection_contamination()
    payload["span_sensitivity"] = experiment_span_sensitivity(
        realisations=30 if quick else 100)
    return payload


def figure(payload, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:                       # pragma: no cover
        print("figure skipped:", error)
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    ax = axes[0, 0]
    rows = payload["decay_bias"]["rows"]
    loss = [100.0 * row["loss_per_wavelength"] for row in rows]
    ax.plot(loss, [row["envelope_R"] for row in rows], "o-",
            label="envelope (campaign formula)")
    ax.plot(loss, [row["detrended_R"] for row in rows], "s-",
            label="detrended (log-linear fit)")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("amplitude loss per wavelength [%]")
    ax.set_ylabel("reported reflection coefficient R")
    ax.set_title("(a) true R = 0: decay is reported as reflection")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    rows = payload["reflection_sampling"]["rows"]
    true = [row["true_R"] for row in rows]
    ax.plot(true, true, "k:", label="identity")
    ax.errorbar(true, [row["detrended_mean"] for row in rows],
                yerr=[[row["detrended_mean"] - row["detrended_min"] for row in rows],
                      [row["detrended_max"] - row["detrended_mean"] for row in rows]],
                fmt="o-", capsize=3, label="detrended")
    ax.errorbar(true, [row["envelope_mean"] for row in rows],
                yerr=[[row["envelope_mean"] - row["envelope_min"] for row in rows],
                      [row["envelope_max"] - row["envelope_mean"] for row in rows]],
                fmt="s--", capsize=3, label="envelope")
    ax.set_xlabel("imposed R")
    ax.set_ylabel("recovered R (mean, min-max over standing-wave phase)")
    ax.set_title("(b) reflection recovery, 4 probes over 0.9$\\lambda$")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    rows = payload["noise_floor"]["rows"]
    noise = [100.0 * row["relative_noise"] for row in rows]
    ax.loglog(noise, [100.0 * row["two_probe_median"] for row in rows], "o-",
              label="two-probe (median)")
    ax.loglog(noise, [100.0 * row["two_probe_p95"] for row in rows], "o:",
              label="two-probe (p95)")
    ax.loglog(noise, [100.0 * row["spatial_median"] for row in rows], "s-",
              label="4-probe spatial fit (median)")
    ax.loglog(noise, [100.0 * row["analytic_two_probe"] for row in rows], "k--",
              label="analytic $\\sigma_\\phi/(k\\Delta x)$")
    ax.axhline(3.0, color="r", lw=1.0, ls="--", label="gate (b) 3 %")
    ax.set_xlabel("relative noise on the probe record [%]")
    ax.set_ylabel("celerity error [%]")
    ax.set_title("(c) instrument floor of the dispersion gate")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    rows = payload["alias"]["rows"]
    spacing = [row["spacing_over_wavelength"] for row in rows]
    ax.semilogy(spacing, [100.0 * max(row["prior_k_error"], 1e-16) for row in rows],
                "o-", label="with imposed $k$ prior (resolved)")
    ax.plot([row["spacing_over_wavelength"] for row in rows if row["rejected_without_prior"]],
            [100.0 for row in rows if row["rejected_without_prior"]], "rx",
            label="rejected without prior")
    ax.axvline(0.5, color="r", lw=1.0, ls="--", label="$\\lambda/2$ limit")
    ax.axvline(2.5, color="grey", lw=1.0, ls=":", label="decay10 layout")
    ax.set_xlabel("probe spacing [$\\lambda$]")
    ax.set_ylabel("relative $k$ error [%]")
    ax.set_title("(d) silent aliasing above $\\lambda/2$")
    ax.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="fewer realisations (CI)")
    parser.add_argument("--json-out", default="runs/wave_tank/estimator_calibration.json")
    parser.add_argument("--figure", default="docs/figures/wave_tank_estimator_calibration.png")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    payload = run_all(quick=args.quick)
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as handle:
        json.dump(payload, handle, indent=2)
    if not args.no_figure:
        path = figure(payload, args.figure)
        if path:
            payload["figure"] = path
            with open(args.json_out, "w") as handle:
                json.dump(payload, handle, indent=2)

    print("== positive control (known k, alpha, R = 0) ==")
    control = payload["positive_control"]
    print(f"  two-probe k error     : {control['two_probe_k_error']:.3e}")
    print(f"  spatial k error       : {control['spatial_k_error']:.3e}")
    print(f"  alpha error           : {control['spatial_alpha_error']:.3e}")
    print(f"  envelope R            : {control['envelope_R']:.4f}")
    print(f"  detrended R           : {control['detrended_R']:.4f}")
    print("== decay -> reflection confusion ==")
    for row in payload["decay_bias"]["rows"]:
        print(f"  loss {100 * row['loss_per_wavelength']:5.1f} %/lam -> "
              f"envelope R {row['envelope_R']:.4f}  detrended R {row['detrended_R']:.4f}"
              f"  ratio {row['recovered_ratio']:.4f} vs {row['true_ratio']:.4f}")
    print("== reflection sampling bias ==")
    for row in payload["reflection_sampling"]["rows"]:
        print(f"  R {row['true_R']:.2f} -> detrended {row['detrended_mean']:.4f}"
              f" [{row['detrended_min']:.4f},{row['detrended_max']:.4f}]"
              f" undercount {100 * row['detrended_undercount']:6.1f} %")
    print("== noise floor ==")
    for row in payload["noise_floor"]["rows"]:
        print(f"  noise {100 * row['relative_noise']:4.1f} % -> two-probe "
              f"{100 * row['two_probe_median']:6.2f} % (analytic "
              f"{100 * row['analytic_two_probe']:6.2f} %) spatial "
              f"{100 * row['spatial_median']:6.2f} %"
              f"  P(>3%) two-probe {100 * row['fraction_two_probe_over_3pct']:5.1f} %"
              f" spatial {100 * row['fraction_spatial_over_3pct']:5.1f} %")
    print("== aliasing ==")
    for row in payload["alias"]["rows"]:
        print(f"  spacing {row['spacing_over_wavelength']:4.2f} lam -> "
              f"wrapped step {row['wrapped_phase_step']:.3f} rad  "
              f"rejected without prior {row['rejected_without_prior']!s:5}  "
              f"prior k error {100 * row['prior_k_error']:7.3f} %  "
              f"alias_risk {row['alias_risk']!s:5}  tolerance "
              f"{row['branch_tolerance']}")
    print("== reflection contamination of criterion (b) ==")
    for row in payload["reflection_contamination"]["rows"]:
        print(f"  R {row['true_R']:.2f} -> celerity error spatial "
              f"{100 * row['spatial_median']:5.2f} % "
              f"[{100 * row['spatial_min']:5.2f},{100 * row['spatial_max']:6.2f}]"
              f" two-probe {100 * row['two_probe_median']:5.2f} %"
              f"  P(>3%) {100 * row['fraction_over_3pct']:5.1f} %")
    print("== span sensitivity (noise 1 %) ==")
    for row in payload["span_sensitivity"]["rows"]:
        print(f"  {row['probes']} probes over 0.9 lam -> median "
              f"{100 * row['median']:.2f} %  p95 {100 * row['p95']:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
