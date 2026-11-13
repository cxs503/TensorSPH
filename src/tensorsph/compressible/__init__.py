"""Compressible (ideal-gas) SPH: Euler equations with shocks.

Sub-package for genuinely compressible SPH formulations — continuity +
energy equation + ideal-gas equation of state — as opposed to the weakly
compressible (Tait) WCSPH machinery in :mod:`tensorsph.wcsph`, which is
inapplicable to gas-dynamic shocks (see docs/validation/wcsph_validation/REPORT.md:
64% L2 density error and negative convergence order for Sod).

Modules
-------
sod          : Sod (1978) shock tube — SPH solver, exact Riemann reference, metrics.
fv_reference : independent HLLC finite-volume solution, used to cross-check the
               analytic profile that all of the case's error numbers are measured
               against (docs/audit/audit_verdict_compressible_sod.md, F5).
riemann_dissipation : Mach-limited Riemann dissipation for SPH — replaces
               Monaghan artificial viscosity with a genuine Riemann solver
               at particle interfaces, scaled by local Mach number.
"""
from .sod import (  # noqa: F401
    SodSPHConfig,
    run_sod_sph,
    exact_star_state,
    exact_wave_positions,
    evaluate_metrics,
    plateau_windows,
)
from . import fv_reference  # noqa: F401
from .riemann_dissipation import (  # noqa: F401
    RiemannDissipationConfig,
    run_sod_riemann,
    hllc_star_pressure,
    roe_average,
    evaluate_metrics as evaluate_metrics_riemann,
)

__all__ = [
    "SodSPHConfig",
    "run_sod_sph",
    "exact_star_state",
    "exact_wave_positions",
    "evaluate_metrics",
    "plateau_windows",
    "fv_reference",
    "RiemannDissipationConfig",
    "run_sod_riemann",
    "hllc_star_pressure",
    "roe_average",
    "evaluate_metrics_riemann",
]
