"""Free-surface verification surface: Ritter / solitary wave / sloshing.

Three classical free-surface problems, each against a closed-form solution, plus
the resolution ladder that says how the error behaves under refinement:

===============  ==========================================  ==================
case             analytic anchor                            module
===============  ==========================================  ==================
dry-bed dam      Ritter (1892) shallow-water fan            ``ritter_anchor``
break
solitary wave    first-order ``H sech^2(alpha(x-ct))``      ``solitary_anchor``
sloshing, mode 1 ``omega_1 = sqrt(pi g/L tanh(pi H/L))``    ``sloshing_mode``
===============  ==========================================  ==================

``profiles`` holds the measurement operators (binned depth/velocity, fronts,
norms, modal projection, decay fits) so all three cases are scored the same way;
``gates`` holds the acceptance thresholds; ``runner`` drives the study and draws
the figures. The numbers, including the velocities and wall-clock costs, are in
``docs/benchmarks/free-surface/report.json``.
"""
from __future__ import annotations

from . import gates, profiles, ritter_anchor, runner, sloshing_mode, solitary_anchor

__all__ = [
    "gates",
    "profiles",
    "ritter_anchor",
    "runner",
    "sloshing_mode",
    "solitary_anchor",
]
