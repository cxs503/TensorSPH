"""Grading of a wave-flume record against literature wave-height references.

The flume itself (:mod:`tensorsph.wave_flume`) produces gauge time series; this
module turns them into the numbers the case-pool entry ``CASE-WAVE-FLUME`` is
graded on:  the *measured* wave height at each gauge, the *attenuation* of that
height along the flume, and the agreement with the literature values of both.

Three references, each with a source
-----------------------------------
1. **Target height of a piston wavemaker.**  The run prescribes a wave height
   ``H`` and derives the paddle stroke ``S`` from the first-order transfer
   function ``H/S = 2 (cosh 2kh - 1) / (sinh 2kh + 2kh)`` (Dean & Dalrymple
   1991, *Water Wave Mechanics for Engineers and Scientists*, World Scientific,
   eq. 6.38; originally Biesel & Suquet 1951).  A flume that is *numerically*
   faithful must deliver that ``H`` in the test section; the deficit is what
   ``CASE-WAVE-FLUME`` calls "wave-height attenuation" and its tolerance is 10 %.
2. **Physical attenuation over a flat bed.**  Even a perfect flume loses height
   to the laminar bottom boundary layer.  With the time-averaged dissipation per
   unit bed area ``D = (1/4) rho u_b^2 sqrt(nu sigma)`` (Mei 1983, *The Applied
   Dynamics of Ocean Surface Waves*, Wiley, §4.2 / Dean & Dalrymple 1991 §5.4,
   from Putnam & Johnson 1949) the amplitude decays as ``a = a0 exp(-mu x)``
   with

       mu = sigma^2 sqrt(nu sigma) / (4 g c_g sinh^2(kh)),

   where ``c_g`` is the group velocity.  For a 1 s wave in 0.25 m of water this
   is ~1.4e-3 1/m, i.e. 0.3 % over a 2 m test section: the *physical* budget of
   a laboratory flume is far below the 10 % tolerance, so any larger measured
   attenuation is numerical.
3. **Which of the two dominates** is the diagnostic question; the report answers
   it with the two numbers side by side rather than assuming.

Estimators, and why they are not the measurement
------------------------------------------------
``H`` is read wave-by-wave between interpolated upward mean crossings (a
time-domain definition that a noisy gauge record survives), and cross-checked
with ``H = 2 sqrt(2) sigma_eta`` and with the spectral peak amplitude.  The three
answers agreeing is what separates "the solver lost height" from "the topmost
particle estimator wobbled".  A synthetic-signal test pins all three; see
``tests/test_wave_flume_grade.py``.
"""
from __future__ import annotations

import math

import torch

from .wave_flume import (
    WaveSpec,
    dominant_omega,
    linear_dispersion_omega,
    separate_incident_reflected,
    wavenumber_from_period,
    wavenumber_from_two_probes,
    zero_crossing_period,
)

__all__ = [
    "airgap_check",
    "attenuation_rate",
    "grade_regular_record",
    "laminar_boundary_layer_damping",
    "spectral_amplitude",
    "wave_height",
    "wave_height_target",
]

GRAVITY = 9.81


# --------------------------------------------------------------------------- #
# Reference values
# --------------------------------------------------------------------------- #

def wave_height_target(spec) -> float:
    """Target wave height ``H`` of the run, straight from its ``WaveSpec``.

    ``WaveSpec.height`` is the height the paddle stroke was derived for (the
    inverse of the piston transfer function), so it *is* the literature
    reference: the flume is graded on how close the gauges come to it.
    """
    if isinstance(spec, dict):
        value = float(spec["height"])
    else:
        value = float(spec.height)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"target wave height must be positive, got {value!r}")
    return value


def laminar_boundary_layer_damping(period, depth, gravity: float = GRAVITY,
                                   kinematic_viscosity: float = 1.0e-6) -> dict:
    """Amplitude decay coefficient ``mu`` [1/m] of a wave over a flat bed.

    Laminar bottom boundary layer, small amplitude; the derivation is in the
    module docstring.  Returns the coefficient together with the intermediate
    quantities so a report can show them without recomputing:

    ``D = (1/4) rho u_b^2 sqrt(nu sigma)`` per unit bed area,
    ``u_b = a sigma / sinh(kh)``, flux ``= (1/2) rho g a^2 c_g``,
    ``mu = sigma^2 sqrt(nu sigma) / (4 g c_g sinh^2(kh))``.
    """
    period = float(period)
    depth = float(depth)
    if period <= 0.0 or depth <= 0.0 or kinematic_viscosity <= 0.0:
        raise ValueError("period, depth and viscosity must be positive")
    k = wavenumber_from_period(period, depth, gravity)
    sigma = 2.0 * math.pi / period
    c = sigma / k
    kh = k * depth
    sinh_kh = math.sinh(kh)
    cg = 0.5 * c * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    mu = (sigma ** 2 * math.sqrt(kinematic_viscosity * sigma)
          / (4.0 * gravity * cg * sinh_kh ** 2))
    return {
        "mu_per_m": mu,
        "k": k,
        "omega": sigma,
        "celerity": c,
        "group_velocity": cg,
        "sinh_kh": sinh_kh,
        "kinematic_viscosity": float(kinematic_viscosity),
        "source": ("Mei 1983 The Applied Dynamics of Ocean Surface Waves §4.2; "
                   "Dean & Dalrymple 1991 §5.4 (Putnam & Johnson 1949)"),
    }


# --------------------------------------------------------------------------- #
# Wave height from a gauge record
# --------------------------------------------------------------------------- #

def _window(time, eta, start=None, end=None):
    t = torch.as_tensor(time, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(eta, dtype=torch.float64).reshape(-1)
    if t.numel() != e.numel():
        raise ValueError("time and elevation series must be the same length")
    finite = torch.isfinite(e)
    t, e = t[finite], e[finite]
    if start is not None:
        keep = t >= float(start)
        t, e = t[keep], e[keep]
    if end is not None:
        keep = t <= float(end)
        t, e = t[keep], e[keep]
    if t.numel() < 8:
        raise ValueError("window holds too few finite samples for a wave height")
    return t, e


def _smooth(e, samples: int):
    if samples <= 1 or e.numel() <= samples:
        return e
    kernel = torch.ones(samples, dtype=e.dtype, device=e.device) / samples
    pad = samples // 2
    padded = torch.nn.functional.pad(e.reshape(1, 1, -1), (pad, samples - 1 - pad),
                                     mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel.reshape(1, 1, -1)).reshape(-1)


def harmonic_amplitude(time, eta, omega, start=None, end=None) -> dict:
    """Least-squares amplitude of a sinusoid of known ``omega`` in a record.

    Solves ``eta ~ A cos(omega t) + B sin(omega t) + C`` over the window; the
    amplitude is ``hypot(A, B)`` and the wave height ``2 amplitude``.  The
    estimate averages the gauge noise down like ``sqrt(2/N)`` instead of taking
    a max minus a min, which is what makes it survive a topmost-particle surface
    estimator whose jitter can be a sizeable fraction of the wave amplitude.

    ``residual_fraction`` is the norm of the fit residual over the norm of the
    demeaned record: a value near one means the record is *not* that sinusoid
    (a broken wave, a reflection, a lost surface) and the amplitude is a fit to
    the wrong model.
    """
    t, e = _window(time, eta, start, end)
    omega = float(omega)
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega!r}")
    matrix = torch.stack((torch.cos(omega * t), torch.sin(omega * t),
                          torch.ones_like(t)), dim=1)
    centred = e - e.mean()
    solution = torch.linalg.lstsq(matrix, centred).solution
    a, b = float(solution[0]), float(solution[1])
    amplitude = math.hypot(a, b)
    residual = matrix @ solution - centred
    norm = float(torch.linalg.vector_norm(centred))
    return {
        "amplitude": amplitude,
        "height": 2.0 * amplitude,
        "phase": math.atan2(-b, a),
        "residual_fraction": (float(torch.linalg.vector_norm(residual)) / norm
                              if norm > 0.0 else float("inf")),
        "samples": int(t.numel()),
    }


def wave_height(time, eta, start=None, end=None, smooth_samples: int = 5,
                omega=None, period=None) -> dict:
    """Wave height of a gauge record, three ways.

    The reported ``height`` is the harmonic fit (:func:`harmonic_amplitude`) at
    ``omega`` -- or, when ``omega``/``period`` is not given, at the spectral peak
    of the record.  The time-domain definition ``height_crossings`` (mean
    crest-to-trough over the waves delimited by interpolated *upward mean
    crossings*, after an optional moving average of ``smooth_samples`` samples)
    is returned next to it, together with two model-free cross-checks:
    ``h_rms`` (``2 sqrt(2) sigma_eta``, exact for a sinusoid) and
    ``h_crest_to_trough`` (max - min, an upper bound that a single spike
    inflates).

    Disagreement between ``height`` and ``height_crossings`` is the signature of
    gauge noise (spurious crossings fragment the waves) or of a non-sinusoidal
    record; the grader reports both instead of choosing silently.
    """
    t, e = _window(time, eta, start, end)
    if omega is None:
        omega = (2.0 * math.pi / float(period) if period is not None
                 else dominant_omega(t, e))
    fit = harmonic_amplitude(t, e, omega)
    smoothed = _smooth(e - e.mean(), int(smooth_samples))
    sign = smoothed >= 0.0
    crossings = torch.nonzero((~sign[:-1]) & sign[1:]).reshape(-1)
    heights = []
    for i in range(crossings.numel() - 1):
        lo, hi = int(crossings[i]), int(crossings[i + 1])
        segment = smoothed[lo:hi + 1]
        if segment.numel() < 3:
            continue
        heights.append(float(segment.max() - segment.min()))
    sigma = float((e - e.mean()).std(unbiased=False))
    return {
        "height": fit["height"],
        "height_crossings": (float(sum(heights) / len(heights)) if heights
                             else float("nan")),
        "n_waves": len(heights),
        "per_wave": heights,
        "omega": float(omega),
        "residual_fraction": fit["residual_fraction"],
        "h_rms": 2.0 * math.sqrt(2.0) * sigma,
        "h_crest_to_trough": float(e.max() - e.min()),
        "samples": int(t.numel()),
        "window": [float(t[0]), float(t[-1])],
        "smooth_samples": int(smooth_samples),
    }


def spectral_amplitude(time, eta, start=None, end=None) -> dict:
    """Peak angular frequency and one-sided peak amplitude of a gauge record."""
    t, e = _window(time, eta, start, end)
    omega = dominant_omega(t, e)
    spectrum = torch.fft.rfft(e - e.mean())
    return {
        "omega": omega,
        "amplitude": 2.0 * float(spectrum.abs().max()) / e.numel(),
        "period": 2.0 * math.pi / omega if omega > 0.0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Attenuation along the gauge array
# --------------------------------------------------------------------------- #

def attenuation_rate(positions, heights) -> dict:
    """Fit ``H(x) = H0 exp(-mu x)`` to the gauge heights.

    Least squares on ``ln(H/2)`` against ``x``; returns the decay coefficient
    ``mu`` [1/m], the fractional loss over the array span (``1 - exp(-mu span)``)
    and the per-pair rates so a single bad gauge is visible instead of averaged
    away.  A negative ``mu`` is growth (reflection contaminating a downstream
    gauge), reported as such rather than clamped.
    """
    xs = torch.as_tensor(positions, dtype=torch.float64).reshape(-1)
    hs = torch.as_tensor(heights, dtype=torch.float64).reshape(-1)
    if xs.numel() != hs.numel() or xs.numel() < 2:
        raise ValueError("need at least two gauges, one height each")
    if not bool((hs > 0.0).all()):
        raise ValueError("wave heights must be positive")
    slope = torch.linalg.lstsq(torch.stack((xs, torch.ones_like(xs)), dim=1),
                               torch.log(hs)).solution
    mu = -float(slope[0])
    span = float(xs.max() - xs.min())
    pairs = [
        {
            "x1": float(xs[i]), "x2": float(xs[i + 1]),
            "h1": float(hs[i]), "h2": float(hs[i + 1]),
            "mu_per_m": -math.log(float(hs[i + 1] / hs[i]))
                          / float(xs[i + 1] - xs[i]),
        }
        for i in range(xs.numel() - 1)
    ]
    return {
        "mu_per_m": mu,
        "span": span,
        "loss_over_span": 1.0 - math.exp(-mu * span),
        "pairs": pairs,
        "h0": float(math.exp(float(slope[1]))),
    }


def airgap_check(time, eta, depth) -> dict:
    """Share of samples where the gauge lost the surface, and the motion bounds.

    The topmost-particle estimator returns ``nan`` for an empty bin; a record
    with many ``nan`` samples is a resolution problem, not a wave.
    ``dry_fraction`` is computed on the raw series (before the window drops the
    non-finite samples, which is what ``wave_height`` needs).
    """
    t = torch.as_tensor(time, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(eta, dtype=torch.float64).reshape(-1)
    if t.numel() != e.numel():
        raise ValueError("time and elevation series must be the same length")
    finite = torch.isfinite(e)
    count = int(finite.sum())
    return {
        "dry_fraction": float(1.0 - count / max(1, int(e.numel()))),
        "samples": int(e.numel()),
        "eta_max": float(e[finite].max()) if count else float("nan"),
        "eta_min": float(e[finite].min()) if count else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Whole-record grading
# --------------------------------------------------------------------------- #

def grade_regular_record(record: dict, start=None, end=None,
                         tolerance: float = 0.10,
                         kinematic_viscosity: float = 1.0e-6) -> dict:
    """Grade one regular-wave flume record against the literature references.

    ``record`` is the JSON written by ``benchmarks/wave_flume/run.py``.  The
    graded quantity of the case-pool entry is the relative error of the measured
    wave height against the target height of the piston transfer function,
    ``|H_meas - H_target| / H_target``, with ``tolerance`` (10 %) as the gate.

    Returns a dict with the per-gauge table, the attenuation fit, the physical
    damping reference, the dispersion check (measured peak period and two-probe
    wave number against ``omega^2 = g k tanh(kh)``), the reflection coefficient
    and ``passed``.
    """
    if record.get("failed"):
        return {"failed": True, "error": record.get("error", "run failed"),
                "passed": False}
    spec = record["spec"]
    cfg = record["config"]
    depth = float(spec["depth"])
    gravity = float(spec["gravity"])
    target = wave_height_target(spec)
    times = record["times"]
    probe_x = [float(x) for x in record["probe_x"]]
    series = record["probe_eta"]
    end = float(times[-1]) if end is None else float(end)

    gauges = []
    for x, eta in zip(probe_x, series):
        entry = {"x": x}
        try:
            h = wave_height(times, eta, start=start, end=end,
                            period=float(spec["period"]))
            entry.update(h)
        except ValueError as exc:
            entry.update({"height": float("nan"), "error": str(exc)})
        dry = airgap_check(times, eta, depth)
        entry["dry_fraction"] = dry["dry_fraction"]
        gauges.append(entry)

    heights = [g["height"] for g in gauges]
    valid = [g for g in gauges if math.isfinite(g["height"])]
    if len(valid) < 2:
        return {"failed": True, "error": "fewer than two usable gauges",
                "gauges": gauges, "passed": False, "target_height": target}

    for g in gauges:
        if math.isfinite(g["height"]):
            g["error_vs_target"] = (g["height"] - target) / target
            g["within_tolerance"] = abs(g["error_vs_target"]) <= tolerance

    attenuation = attenuation_rate([g["x"] for g in valid],
                                   [g["height"] for g in valid])
    theory = laminar_boundary_layer_damping(spec["period"], depth, gravity,
                                            kinematic_viscosity)
    measured_loss = attenuation["loss_over_span"]
    theory_loss = 1.0 - math.exp(-theory["mu_per_m"] * attenuation["span"])

    # Dispersion: the measured peak period and the two-probe wave number must
    # satisfy the same relation the target height was derived from.
    dispersion = {}
    try:
        omega_meas = dominant_omega(times, series[0], start=start, end=end)
        period_meas = 2.0 * math.pi / omega_meas
        period_zero = zero_crossing_period(times, series[0], start=start, end=end)[0]
        dispersion["period_spectral"] = period_meas
        dispersion["period_zero_crossing"] = period_zero
        dispersion["period_target"] = float(spec["period"])
        dispersion["period_error"] = (period_meas - float(spec["period"])) / float(spec["period"])
        if len(valid) >= 2:
            spacing = abs(probe_x[1] - probe_x[0])
            k_meas = wavenumber_from_two_probes(times, series[0], series[1],
                                                spacing, start=start, end=end)
            k_theory = wavenumber_from_period(float(spec["period"]), depth, gravity)
            dispersion["k_two_probe"] = abs(k_meas)
            dispersion["k_theory"] = k_theory
            dispersion["wavelength_two_probe"] = 2.0 * math.pi / abs(k_meas)
            dispersion["wavelength_theory"] = 2.0 * math.pi / k_theory
            dispersion["wavelength_error"] = (dispersion["wavelength_two_probe"]
                                              / dispersion["wavelength_theory"] - 1.0)
    except ValueError as exc:
        dispersion["error"] = str(exc)

    reflection = {}
    try:
        if len(valid) >= 3:
            omega = dominant_omega(times, series[0], start=start, end=end)
            k = wavenumber_from_period(2.0 * math.pi / omega, depth, gravity)
            split = separate_incident_reflected(
                times, [g_series for g_series in series],
                probe_x, omega, k, start=start, end=end)
            reflection = {key: split[key] for key in
                          ("incident_amplitude", "reflected_amplitude", "ratio",
                           "relative_residual", "samples", "gauges")}
    except ValueError as exc:
        reflection["error"] = str(exc)

    worst = max(abs(g.get("error_vs_target", float("inf"))) for g in valid)
    passed = worst <= tolerance
    return {
        "failed": False,
        "target_height": target,
        "tolerance": tolerance,
        "height_cells": float(cfg.get("height_cells", float("nan"))),
        "depth_cells": float(cfg.get("depth_cells", float("nan"))),
        "gauges": gauges,
        "worst_error_vs_target": worst,
        "attenuation": attenuation,
        "theory": theory,
        "measured_loss_over_span": measured_loss,
        "theory_loss_over_span": theory_loss,
        "excess_loss_over_span": measured_loss - theory_loss,
        "dispersion": dispersion,
        "reflection": reflection,
        "window": [float(start) if start is not None else float(times[0]), end],
        "passed": passed,
    }