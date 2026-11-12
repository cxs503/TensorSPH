"""Compressible (ideal-gas) SPH: Euler equations with shocks.

Sub-package for genuinely compressible SPH formulations — continuity +
energy equation + ideal-gas equation of state — as opposed to the weakly
compressible (Tait) WCSPH machinery in :mod:`tensorsph.wcsph`, which is
inapplicable to gas-dynamic shocks (see docs/validation/wcsph_validation/REPORT.md:
64% L2 density error and negative convergence order for Sod).

Modules
-------
sod : Sod (1978) shock tube — SPH solver, exact Riemann reference, metrics.
"""
from .sod import (  # noqa: F401
    SodSPHConfig,
    run_sod_sph,
    exact_star_state,
    exact_wave_positions,
    evaluate_metrics,
)

__all__ = [
    "SodSPHConfig",
    "run_sod_sph",
    "exact_star_state",
    "exact_wave_positions",
    "evaluate_metrics",
]