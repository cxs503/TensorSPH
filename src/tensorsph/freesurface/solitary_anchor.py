"""Solitary-wave propagation: sech^2 amplitude, celerity and shape against theory.

Physical problem
----------------
A first-order solitary wave of height ``H`` on still water of depth ``h`` is

    eta(x, t) = H sech^2(alpha (x - c t - x0)),   alpha = sqrt(3H / (4 h^3)),
    c         = sqrt(g (h + H)).

It is non-dispersive in the KdV sense -- all of its harmonics travel at the same
speed -- which is exactly why it is the standard long-wave fidelity test: any
numerical scheme that disperses or dissipates the short components will change
the wave's shape, and any scheme with numerical damping will shrink its height.

What is measured
----------------
For gauges at ``x_1 < x_2 < x_3`` ahead of the wavemaker:

``amplitude_ratio``   ``max(eta_2) / max(eta_1)`` -- survivability of the crest;
``decay_per_wavelength``  the same ratio rescaled to one characteristic
                      wavelength ``lambda = 2 pi / alpha``;
``decay_10_lambda``   ``1 - ratio**(10 lambda / (x_2 - x_1))``: the linear
                      attenuation model extrapolated to ten wavelengths;
``celerity_error``    time-of-flight between gauges against ``sqrt(g (h + H))``;
``shape_l2``          relative L2 of the measured profile against
                      ``H sech^2`` at the instant the crest passes ``x_2``,
                      crest-aligned (a phase error is a propagation error and is
                      scored by ``celerity_error``; scoring it twice would
                      double-count it).

Honest scope -- and why the 10-wavelength number is an extrapolation
--------------------------------------------------------------------
``lambda = 2 pi / alpha`` is of the order of 20 water depths for ``H/h = 0.1``.
Ten of those is a 40-50 m flume: 4-6 times the length that fits the compute
budget of this repository, and far outside what any WCSPH run in the literature
covers. The trivial way out -- shortening ``lambda`` by increasing ``H/h`` -- is
not available either, because the first-order sech^2 solution is only valid for
``H/h << 1`` and the wave breaks around ``H/h ~ 0.5``.

So the far-field number is **extrapolated** from the measured attenuation per
wavelength, and labelled as such everywhere it appears. A measured attenuation
that is already at the 5 % level over the traversed distance is an honest FAIL
regardless of what the extrapolation says.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .. import wave_flume
from . import profiles

__all__ = ["SolitaryCase", "SolitaryResult", "run_solitary"]

GRAVITY = 9.81


@dataclass
class SolitaryCase:
    """Geometry, numerics and gauge layout of one propagation run."""

    depth: float = 0.20
    height: float = 0.05                # H/h = 0.25: crest resolved by 2 dx at
                                        # 8 cells per depth, still first order
    flume_length: float = 3.00
    cells_per_depth: float = 8.0
    sound_speed_ratio: float = 21.4     # c_s / sqrt(g h) (2.2 % compressibility)
    viscosity: float = 1.0e-4
    density_diffusion: float = 0.10
    xsph: float = 0.20
    sponge_length: float = 0.50
    sponge_strength: float = 4.0
    duration: float | None = None        # seconds; default: one and a half transits
    output_per_transit: int = 200
    gauge_positions: tuple = (0.50, 1.20, 1.90)
    device: str = "cpu"
    gravity: float = GRAVITY
    label: str = ""

    @property
    def dx(self) -> float:
        return float(self.depth / self.cells_per_depth)

    @property
    def alpha(self) -> float:
        return math.sqrt(3.0 * self.height / (4.0 * self.depth ** 3))

    @property
    def wavelength(self) -> float:
        """Characteristic wavelength ``lambda = 2 pi / alpha``."""
        return 2.0 * math.pi / self.alpha

    @property
    def celerity(self) -> float:
        return wave_flume.solitary_wave_celerity(self.height, self.depth, self.gravity)

    @property
    def sound_speed(self) -> float:
        return float(self.sound_speed_ratio * math.sqrt(self.gravity * self.depth))

    @property
    def dt(self) -> float:
        return float(0.25 * 1.3 * self.dx / self.sound_speed)

    @property
    def run_time(self) -> float:
        if self.duration is not None:
            return float(self.duration)
        # the crest has to cross the last gauge, and the paddle has to deliver
        # the whole stroke: allow two transits plus a margin
        return float(1.5 * self.flume_length / self.celerity)

    def config(self) -> wave_flume.WaveFlumeConfig:
        cfg = wave_flume.WaveFlumeConfig(
            flume_length=float(self.flume_length),
            depth=float(self.depth),
            wave_type="solitary",
            wave_height=float(self.height),
            dx=float(self.dx),
            sound_speed=float(self.sound_speed),
            viscosity=float(self.viscosity),
            density_diffusion=float(self.density_diffusion),
            xsph=float(self.xsph),
            sponge_length=float(self.sponge_length),
            sponge_strength=float(self.sponge_strength),
            probes=tuple(self.gauge_positions),
            device=self.device,
            hydrostatic_init=True,
        )
        cfg.dt = self.dt
        cfg.steps = int(math.ceil(self.run_time / cfg.dt))
        cfg.output_every = max(1, int(round(
            self.wavelength / cfg.dt / self.output_per_transit)))
        return cfg


@dataclass
class SolitaryResult:
    case: SolitaryCase
    times: list = field(default_factory=list)
    gauges: list = field(default_factory=list)      # x positions
    elevations: list = field(default_factory=list)  # per gauge, list of eta
    crest: list = field(default_factory=list)       # per gauge: (height, time)
    shapes: list = field(default_factory=list)      # per-gauge shape diagnostics
    steps_taken: int = 0
    simulated_time: float = 0.0
    wall_clock: float = 0.0
    particle_count: int = 0
    escaped: int = 0
    mass_drift: float = 0.0
    max_density_compression: float = 0.0
    max_speed: float = 0.0
    config_summary: dict = field(default_factory=dict)
    paddle_stroke: float = 0.0

    def summary(self) -> dict:
        gauge_rows = []
        for x, elev, (height, time) in zip(self.gauges, self.elevations, self.crest):
            gauge_rows.append({"x": float(x), "crest_height": float(height),
                               "crest_time": float(time)})
        first = gauge_rows[0]["crest_height"]
        last = gauge_rows[-1]["crest_height"]
        distance = gauge_rows[-1]["x"] - gauge_rows[0]["x"]
        ratio = last / first if first > 0 else float("nan")
        lam = self.case.wavelength
        per_lambda = 1.0 - ratio ** (lam / distance) if distance > 0 else float("nan")
        ten_lambda = 1.0 - ratio ** (10.0 * lam / distance) if distance > 0 else float("nan")
        # time of flight between the first and last gauge
        travel_time = gauge_rows[-1]["crest_time"] - gauge_rows[0]["crest_time"]
        celerity = distance / travel_time if travel_time > 0 else float("nan")
        shape_l2 = float("nan")
        shape_correlation = float("nan")
        if self.shapes:
            middle = self.shapes[min(1, len(self.shapes) - 1)]
            shape_l2 = float(middle["shape_l2"])
            if middle.get("shape_correlation") is not None:
                shape_correlation = float(middle["shape_correlation"])
        return {
            "label": self.case.label,
            "depth": self.case.depth,
            "height": self.case.height,
            "height_over_depth": self.case.height / self.case.depth,
            "wavelength": lam,
            "characteristic_width": 1.0 / self.case.alpha,
            "celerity_analytic": self.case.celerity,
            "celerity_measured": celerity,
            "celerity_error": (abs(celerity - self.case.celerity) / self.case.celerity
                               if celerity == celerity else float("nan")),
            "propagation_distance": distance,
            "distance_in_wavelengths": distance / lam,
            "amplitude_ratio": ratio,
            "decay_per_wavelength": per_lambda,
            "decay_10_lambda_extrapolated": ten_lambda,
            "shape_l2": shape_l2,
            "shape_correlation": shape_correlation,
            "shape_by_gauge": [{k: v for k, v in shape.items()
                                if k not in ("t", "eta_measured", "eta_analytic")}
                               for shape in self.shapes],
            "dx": self.case.dx,
            "cells_per_depth": self.case.cells_per_depth,
            "dt": self.case.dt,
            "steps": self.steps_taken,
            "particles": self.particle_count,
            "simulated_time": self.simulated_time,
            "wall_clock": self.wall_clock,
            "escaped": self.escaped,
            "mass_drift_relative": self.mass_drift,
            "max_density_compression": self.max_density_compression,
            "max_speed": self.max_speed,
            "gauges": gauge_rows,
            "time_series": {"times": [float(t) for t in self.times],
                            "elevations": [[float(v) for v in e]
                                           for e in self.elevations]},
            "shape_records": self.shapes,
        }


def _crest(signal: list[float], times: list[float]) -> tuple[float, float]:
    """Peak elevation and its time, refined by a parabola through three samples."""
    if not signal:
        return float("nan"), float("nan")
    index = max(range(len(signal)), key=lambda i: signal[i])
    height = float(signal[index])
    time = float(times[index])
    if 0 < index < len(signal) - 1:
        y0, y1, y2 = float(signal[index - 1]), height, float(signal[index + 1])
        denominator = y0 - 2.0 * y1 + y2
        if denominator < 0.0:
            offset = 0.5 * (y0 - y2) / denominator
            offset = min(max(offset, -0.5), 0.5)
            dt = float(times[index + 1] - times[index])
            height = y1 - 0.25 * (y0 - y2) * offset
            time = time + offset * dt
    return height, time


def shape_l2_from_gauge(times, elevations, case, index: int = 1) -> dict:
    """Compare a gauge record with ``H sech^2(alpha c (t_crest - t))``.

    A solitary wave is steady in its own frame, so the *time* record at a fixed
    gauge is the spatial profile stretched by the celerity:
    ``eta(x_g, t) = H sech^2(alpha (x_g - c(t - t_crest)))``. Comparing the
    record avoids a second simulation to snapshot the profile at the instant the
    crest passes, and it is what a wave gauge measures in an experiment.

    The analytic curve is scaled by the **measured** crest height before the
    norm is taken, so this number scores the *shape* (width, symmetry, tail) and
    not the amplitude -- amplitude is scored separately by the crest decay and
    the celerity by the time of flight.
    """
    if not elevations or index >= len(elevations):
        return {}
    series = elevations[index]
    height, crest_time = _crest(series, times)
    if not (height == height) or height <= 0.0:
        return {}
    half_width = 2.0 / (case.alpha * case.celerity)
    keep = [i for i, t in enumerate(times) if abs(t - crest_time) <= half_width]
    if len(keep) < 8:
        return {}
    measured = [series[i] for i in keep]
    analytic = [height * wave_flume._sech2(
        case.alpha * case.celerity * (crest_time - times[i])) for i in keep]
    correlation = None
    try:
        a = torch.tensor(measured, dtype=torch.float64)
        b = torch.tensor(analytic, dtype=torch.float64)
        a = a - a.mean()
        b = b - b.mean()
        denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
        if denominator > 0.0:
            correlation = float(torch.dot(a, b)) / denominator
    except Exception:      # pragma: no cover - only a diagnostic
        correlation = None
    return {
        "gauge_index": int(index),
        "gauge_x": float(case.gauge_positions[index]),
        "crest_height": float(height),
        "crest_time": float(crest_time),
        "window_half_width": float(half_width),
        "shape_l2": profiles.relative_l2(measured, analytic, scale=float(height)),
        "shape_correlation": correlation,
        "t": [float(times[i]) for i in keep],
        "eta_measured": [float(v) for v in measured],
        "eta_analytic": [float(v) for v in analytic],
    }


def run_solitary(case: SolitaryCase) -> SolitaryResult:
    """Generate the wave with Goring's paddle and record the three gauges."""
    import time as _time

    cfg = case.config()
    started = _time.perf_counter()
    run = wave_flume.run_case(cfg, probe_positions=tuple(case.gauge_positions))
    elapsed = _time.perf_counter() - started

    result = SolitaryResult(
        case=case,
        times=list(run.times),
        gauges=[float(x) for x in run.probe_x],
        elevations=[list(map(float, series)) for series in run.probe_eta],
        steps_taken=int(run.steps_taken),
        simulated_time=float(run.simulated_time),
        wall_clock=float(elapsed),
        escaped=int(run.escaped),
        max_density_compression=float(run.max_density_compression),
        max_speed=float(run.max_speed),
        config_summary=dict(run.config),
        paddle_stroke=float(run.spec.stroke),
        particle_count=int(run.config.get("particle_count", 0)),
    )
    masses = float(run.mass)
    result.mass_drift = (abs(float(run.mass_final) - masses) / masses
                         if masses > 0 else float("nan"))
    for series in result.elevations:
        result.crest.append(_crest(series, result.times))
    for index in range(len(result.gauges)):
        shape = shape_l2_from_gauge(result.times, result.elevations, case, index)
        if shape:
            result.shapes.append(shape)
    return result


