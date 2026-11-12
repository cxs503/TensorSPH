"""Case table for the sloshing-resonance benchmark (CASE-SLOSHING).

Everything that a reader needs to reproduce a number lives here: the geometry,
the two dimensionless groups that the result is reported against
(``H/L`` for the tank, ``A/H`` for the released wave), the resolution ladder and
the predeclared acceptance criterion. ``run.py`` only executes what is listed
here, so the table *is* the experimental plan.

Predeclared (fixed before any of these runs was looked at)
----------------------------------------------------------
* **Decision**: is the first-mode sloshing resonance frequency produced by the
  TensorSPH weakly-compressible solver inside the mission tolerance of 10 % of
  linear theory, and how does that error behave under refinement and under a
  change of the two dimensionless controls?
* **Metric**: ``|omega_sim - omega_th| / omega_th`` with
  ``omega_th = sqrt(g k_1 tanh(k_1 H))``, ``k_1 = pi/L`` -- the linear-theory
  resonance frequency, computed in code by
  :func:`tensorsph.freesurface.sloshing_anchor.omega_n` (closed form) and
  cross-checked by :func:`~tensorsph.freesurface.sloshing_anchor.omega_n_fd`
  (independent FD eigensolve). No number is transcribed from a table by hand.
* **Threshold**: 0.10 for every case (mission criterion). A case above it is a
  FAIL, reported as such: no threshold is moved after seeing a result.
* **Stopping rule**: a case that does not reach ``periods`` first-mode periods
  inside its wall-clock budget, or whose record does not contain two upward
  mean crossings, is reported ``UNMEASURED`` and counts as not passed.
* **Controls**: the resolution ladder (``cells_per_depth``) and the geometry
  sweep (``H/L``) are the two controlled variables; ``A/H`` and
  ``cs/sqrt(gH)`` are swept one at a time around the baseline, everything else
  held at the baseline value (``cells_per_depth = 12``, ``cs/sqrt(gH) = 80``,
  ``A/H = 0.08``), so each sensitivity number has a single cause.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "Case",
    "CASES",
    "BASELINE",
    "TOLERANCE",
    "by_label",
    "baseline",
]

TOLERANCE = 0.10            # mission criterion on the frequency error
GRAVITY = 9.81
TANK_LENGTH = 0.60          # m, fixed for every case
BASELINE_CPD = 12.0         # cells per still-water depth
BASELINE_SOUND_RATIO = 80.0  # c_s / sqrt(g H)
BASELINE_AMPLITUDE_RATIO = 0.08  # A / H (surface displacement at the wall)


@dataclass(frozen=True)
class Case:
    """One released-first-mode run."""

    label: str
    water_depth: float
    tank_height: float
    amplitude: float
    cells_per_depth: float = BASELINE_CPD
    sound_speed_ratio: float = BASELINE_SOUND_RATIO
    periods: float = 3.0
    output_per_period: int = 60
    viscosity: float = 1.0e-3
    family: str = "baseline"
    note: str = ""

    @property
    def tank_length(self) -> float:
        return TANK_LENGTH

    @property
    def depth_ratio(self) -> float:
        return self.water_depth / TANK_LENGTH

    @property
    def amplitude_ratio(self) -> float:
        return self.amplitude / self.water_depth

    @property
    def dx(self) -> float:
        return self.water_depth / self.cells_per_depth

    def omega_theory(self) -> float:
        k = math.pi / self.tank_length
        return math.sqrt(GRAVITY * k * math.tanh(k * self.water_depth))


def _base(label: str, water_depth: float, **kw) -> Case:
    tank_height = kw.pop("tank_height", water_depth + 0.15)
    amplitude = kw.pop("amplitude", BASELINE_AMPLITUDE_RATIO * water_depth)
    return Case(label=label, water_depth=water_depth, tank_height=tank_height,
                amplitude=amplitude, **kw)


BASELINE = _base("base_cpd12", 0.25, tank_height=0.40, family="baseline",
                 note="reference point: L=0.60 m, H=0.25 m, cpd=12, cs/sqrt(gH)=80")

CASES = (
    BASELINE,
    # --- geometry sweep: H/L = 0.2 / 0.4 / 0.6 at the baseline resolution -----
    _base("geom_h020", 0.12, tank_height=0.25, family="geometry",
          note="H/L=0.20: tanh(kH)=0.557, shallow end of the dispersion curve"),
    _base("geom_h060", 0.36, tank_height=0.55, family="geometry",
          note="H/L=0.60: tanh(kH)=0.955, deep-water end"),
    # --- resolution ladder around the baseline ------------------------------
    _base("res_cpd08", 0.25, tank_height=0.40, cells_per_depth=8.0,
          family="resolution", note="coarsest grid"),
    _base("res_cpd16", 0.25, tank_height=0.40, cells_per_depth=16.0,
          family="resolution", note="2x baseline cells per depth"),
    _base("res_cpd20", 0.25, tank_height=0.40, cells_per_depth=20.0,
          family="resolution", note="finest grid"),
    # --- amplitude sensitivity (non-linearity) ------------------------------
    _base("amp_004", 0.25, tank_height=0.40,
          amplitude=0.04 * 0.25, family="amplitude", note="A/H=0.04"),
    _base("amp_016", 0.25, tank_height=0.40,
          amplitude=0.16 * 0.25, family="amplitude", note="A/H=0.16"),
    # --- weak-compressibility sensitivity -----------------------------------
    _base("cs_040", 0.25, tank_height=0.40, sound_speed_ratio=40.0,
          family="sound_speed", note="c_s/sqrt(gH)=40: 4x the Tait bias"),
    _base("cs_160", 0.25, tank_height=0.40, sound_speed_ratio=160.0,
          family="sound_speed", note="c_s/sqrt(gH)=160: 1/4 the Tait bias"),
)


def by_label(label: str) -> Case:
    for case in CASES:
        if case.label == label:
            return case
    known = ", ".join(c.label for c in CASES)
    raise KeyError(f"unknown case {label!r}; known: {known}")


def baseline() -> Case:
    return BASELINE