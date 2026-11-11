"""Breaking-wave verification: energy dissipation above the breaking threshold.

Physics
-------
A progressive wave on finite depth ``h`` breaks when its steepness ``H/L``
exceeds approximately 0.14 (McCowan 1894; Miche 1944).  Above that threshold
the crest becomes unstable, overturns, and a significant fraction of the
mechanical energy is dissipated by turbulence (which SPH models through
numerical viscosity and the WCSPH pressure-dilatation work).

This module drives a wave tank at two steepnesses -- one safely below the
breaking limit and one above -- and measures the energy decay of the
propagating wave train.  The *verification claim* is not that SPH reproduces
the detailed turbulent bore (it cannot at the resolutions used here), but that:

1. The sub-critical wave propagates with little amplitude loss (the sponge
   layer is excluded from the measurement window).
2. The super-critical wave loses significantly more energy than the
   sub-critical one, demonstrating that the solver captures the onset of
   breaking-induced dissipation.
3. Mass is conserved in both cases (breaking redistributes momentum but does
   not destroy mass).

Analytic anchors
~~~~~~~~~~~~~~~~
* Linear dispersion: ``omega^2 = g k tanh(kh)`` (used to set the wave number).
* Breaking threshold: ``H/L > 0.14`` (McCowan 1894).  For the tank depth used
  here (``h = 0.5 m``) and ``kh = 2.0`` (``L ~ 1.57 m``), the breaking height
  is ``H_break ~ 0.22 m``, i.e. ``H/h ~ 0.44``.
* Energy flux of a linear wave: ``P = (1/8) rho g H^2 c_g`` where
  ``c_g = c (1/2 + kh/sinh(2kh))`` is the group velocity.

What is NOT claimed
~~~~~~~~~~~~~~~~~~~
* The detailed breaking geometry (plunging jet, splash-up) is resolution-
  dependent and not validated against experiment at this stage.
* The turbulence model is the WCSPH artificial viscosity; no LES or SGS model
  is active.  The dissipation rate is therefore a combined physical + numerical
  quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Tuple

import torch

from ..wave_tank import (
    WaveTankConfig,
    WaveTankRun,
    angular_frequency,
    run_tank,
)

__all__ = [
    "BreakingCase",
    "BreakingResult",
    "breaking_threshold_height",
    "group_velocity",
    "linear_energy_flux",
    "measure_wave_height",
    "run_breaking_case",
]


GRAVITY = 9.81
RHO0 = 1000.0


def breaking_threshold_height(depth: float, wavelength: float,
                              gravity: float = GRAVITY) -> float:
    """McCowan (1894) breaking criterion: H_break = 0.14 * L.

    This is the deep-water limit; for finite depth the Miche (1944) criterion
    ``H/L = 0.142 tanh(kh)`` is more appropriate, but for ``kh >= 2`` the two
    agree to within 2 %.
    """
    return 0.14 * wavelength


def group_velocity(k: float, depth: float, gravity: float = GRAVITY) -> float:
    """Group speed ``c_g = c * n`` with ``n = 1/2 + kh/sinh(2kh)``."""
    omega = angular_frequency(k, depth, gravity)
    c = omega / k
    kh = k * depth
    if kh > 100:
        n = 0.5  # deep-water limit
    else:
        n = 0.5 + kh / math.sinh(2.0 * kh)
    return c * n


def linear_energy_flux(height: float, k: float, depth: float,
                       rho: float = RHO0, gravity: float = GRAVITY) -> float:
    """Energy flux of a linear (Airy) wave: P = (1/8) rho g H^2 c_g."""
    cg = group_velocity(k, depth, gravity)
    return 0.125 * rho * gravity * height ** 2 * cg


def measure_wave_height(probe_series: list, times: list,
                        skip_fraction: float = 0.3) -> float:
    """Significant wave height from a gauge time series.

    The first ``skip_fraction`` of the record is discarded (transient / ramp),
    then H = 4 * std(eta) is returned (significant height for a Gaussian
    process; for a monochromatic wave this is the same as the crest-to-trough
    height to within a few percent).
    """
    if not probe_series or not times:
        return float("nan")
    n = len(probe_series)
    start = int(skip_fraction * n)
    if start >= n - 4:
        return float("nan")
    eta = probe_series[start:]
    eta_t = torch.tensor(eta, dtype=torch.float64)
    return 4.0 * float(eta_t.std())


@dataclass
class BreakingCase:
    """One wave-tank run at a prescribed steepness.

    Parameters
    ----------
    depth : float
        Still-water depth [m].
    kh : float
        Non-dimensional wave number (sets the wavelength).
    wave_height : float
        Target wave height [m].
    cells_per_depth : float
        Resolution: ``dx = depth / cells_per_depth``.
    tank_lengths : float
        Tank length in multiples of the depth.
    """
    depth: float = 0.5
    kh: float = 2.0
    wave_height: float = 0.05
    cells_per_depth: float = 15.0
    tank_lengths: float = 10.0
    device: str = "cpu"

    @property
    def dx(self) -> float:
        return self.depth / self.cells_per_depth

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi * self.depth / self.kh

    @property
    def steepness(self) -> float:
        return self.wave_height / self.wavelength

    @property
    def is_breaking(self) -> bool:
        return self.steepness > 0.14

    @property
    def breaking_height(self) -> float:
        return breaking_threshold_height(self.depth, self.wavelength)

    def config(self, end_time_factor: float = 4.0) -> WaveTankConfig:
        """Build a WaveTankConfig for this case."""
        length = self.tank_lengths * self.depth
        k = self.kh / self.depth
        omega = angular_frequency(k, self.depth)
        period = 2.0 * math.pi / omega
        cfg = WaveTankConfig(
            dx=self.dx,
            depth=self.depth,
            length=length,
            kh=self.kh,
            wave_height=self.wave_height,
            maker="relaxation",
            wave_kind="regular",
            end_time=end_time_factor * period,
            probe_offsets=(0.4, 0.7, 1.0),
        )
        cfg.device = self.device
        return cfg


@dataclass
class BreakingResult:
    """Result of a breaking-wave verification run."""
    case: BreakingCase
    wave_heights: list = field(default_factory=list)   # per-probe
    probe_positions: list = field(default_factory=list)
    mass_error: float = float("nan")
    max_compression: float = float("nan")
    max_speed: float = float("nan")
    steps_taken: int = 0
    simulated_time: float = 0.0
    wall_clock: float = 0.0
    escaped: int = 0
    energy_decay: float = float("nan")  # (H_last - H_first) / H_first

    def summary(self) -> dict:
        return {
            "depth": self.case.depth,
            "kh": self.case.kh,
            "wave_height": self.case.wave_height,
            "wavelength": self.case.wavelength,
            "steepness": self.case.steepness,
            "is_breaking": self.case.is_breaking,
            "breaking_height": self.case.breaking_height,
            "dx": self.case.dx,
            "cells_per_depth": self.case.cells_per_depth,
            "wave_heights_measured": self.wave_heights,
            "probe_positions": self.probe_positions,
            "mass_error": self.mass_error,
            "max_compression": self.max_compression,
            "max_speed": self.max_speed,
            "steps_taken": self.steps_taken,
            "simulated_time": self.simulated_time,
            "wall_clock": self.wall_clock,
            "escaped": self.escaped,
            "energy_decay": self.energy_decay,
        }


def run_breaking_case(case: BreakingCase) -> BreakingResult:
    """Run the wave tank and measure the wave height at each probe."""
    import time as _time

    cfg = case.config()
    started = _time.perf_counter()
    run: WaveTankRun = run_tank(cfg)
    elapsed = _time.perf_counter() - started

    result = BreakingResult(
        case=case,
        mass_error=run.mass_error,
        max_compression=run.max_compression,
        max_speed=run.max_speed,
        steps_taken=run.steps_taken,
        simulated_time=run.simulated_time,
        wall_clock=elapsed,
        escaped=run.escaped,
    )

    # Measure wave height at each probe
    for i, px in enumerate(run.probe_x):
        times_list, series = run.probe_series(i)
        h_meas = measure_wave_height(series, list(times_list))
        result.wave_heights.append(h_meas)
        result.probe_positions.append(float(px))

    # Energy decay: relative change from first to last probe
    if len(result.wave_heights) >= 2:
        h0 = result.wave_heights[0]
        hN = result.wave_heights[-1]
        if h0 > 0 and h0 == h0:
            result.energy_decay = (hN - h0) / h0

    return result


def run_breaking_pair(case_sub: BreakingCase,
                      case_super: BreakingCase) -> dict:
    """Run a sub-critical and super-critical case and compare energy decay.

    The verification claim is that the super-critical case dissipates
    significantly more energy than the sub-critical one.
    """
    r_sub = run_breaking_case(case_sub)
    r_super = run_breaking_case(case_super)

    decay_sub = r_sub.energy_decay
    decay_super = r_super.energy_decay

    # The differential dissipation: super should lose more energy
    differential = float("nan")
    if (decay_sub == decay_sub and decay_super == decay_super):
        differential = decay_super - decay_sub

    return {
        "sub_critical": r_sub.summary(),
        "super_critical": r_super.summary(),
        "differential_dissipation": differential,
        "breaking_onset_detected": (differential == differential
                                    and differential > 0.05),
    }
