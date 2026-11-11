"""Weakly-compressible SPH verification chain (dam break + tank sloshing).

Package layout
--------------
``kernels``          quintic-spline and Gaussian kernels with analytic gradients
``solver``           explicit WCSPH: Tait EOS, symmetric momentum form, Monaghan
                     artificial viscosity, dynamic / mirror boundaries, KDK
                     leapfrog with a CFL limit
``cases``            particle pair, 2-D and 3-D dam break, 2-D forced sloshing
``linear_sloshing``  closed-form linear potential-flow reference for the tank
``chain``            the acceptance gates (pure functions of measured numbers)

The package is self-contained on purpose: it is the *verification* face of the
solver family, not a second production solver, and every numerical ingredient
the benchmark node asks for is individually reachable and individually tested.
"""
from . import chain, kernels
from .cases import (DamBreakConfig, DamBreakRun, PairRun, SloshingConfig,
                    SloshingRun, box_walls, dam_break, front_at,
                    front_positions, lattice_box, natural_frequency,
                    particle_pair, relative_front_error, sloshing)
from .linear_sloshing import LinearSloshing
from .solver import (Solver, State, WCSPHConfig, mirror_images, sound_speed_for,
                     stable_dt, tait_pressure)

__all__ = [
    "DamBreakConfig", "DamBreakRun", "LinearSloshing", "PairRun", "SloshingConfig",
    "SloshingRun", "Solver", "State", "WCSPHConfig", "box_walls", "chain",
    "dam_break", "front_at", "front_positions", "kernels", "lattice_box",
    "mirror_images", "natural_frequency", "particle_pair", "relative_front_error",
    "sloshing", "sound_speed_for", "stable_dt", "tait_pressure",
]