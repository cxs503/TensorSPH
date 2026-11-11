"""Free-surface verification surface: Ritter / solitary wave / sloshing / breaking.

Four classical free-surface problems, each against a closed-form solution, plus
the resolution ladder that says how the error behaves under refinement:

===============  ==========================================  ==================
case             analytic anchor                            module
===============  ==========================================  ==================
dry-bed dam      Ritter (1892) shallow-water fan            ``ritter_anchor``
break
solitary wave    first-order ``H sech^2(alpha(x-ct))``      ``solitary_anchor``
sloshing, mode 1 ``omega_1 = sqrt(pi g/L tanh(pi H/L))``    ``sloshing_mode``
breaking wave    McCowan (1894) ``H/L > 0.14`` threshold    ``breaking_wave``
===============  ==========================================  ==================

``profiles`` holds the measurement operators (binned depth/velocity, fronts,
norms, modal projection, decay fits) so all four cases are scored the same way;
``gates`` holds the acceptance thresholds; ``runner`` drives the study and draws
the figures. The numbers, including the velocities and wall-clock costs, are in
``docs/benchmarks/free-surface/report.json``.
"""
from __future__ import annotations

from . import (
    breaking_wave,
    gates,
    profiles,
    ritter_anchor,
    runner,
    sloshing_mode,
    solitary_anchor,
)

__all__ = [
    "breaking_wave",
    "gates",
    "profiles",
    "ritter_anchor",
    "runner",
    "sloshing_mode",
    "solitary_anchor",
]
