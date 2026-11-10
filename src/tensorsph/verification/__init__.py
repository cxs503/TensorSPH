"""Machine-checkable verification face for free-surface SPH (campaign P482-N04).

Four measurements, each with a gate and a negative control, plus the evidence
store that makes every reported number traceable to a SQL query:

===========  ==================================================================
``dam_break_2d``   M1: front ``Z/a(tau)`` and column height against measured data
                   and against two exact height anchors; control ``g = 0``.
``hydrostatic``    M2: ``p(z) = rho g (H - z)`` on ``N >= 1000`` particles;
                   control ``rho0 = 0``.
``conservation``   M3: mass bookkeeping in a closed tank (0.1 %) and in an open
                   one (forgetting the outflow must lose > 1 %).
``convergence``    M4: particle-spacing convergence of the SPH interpolation
                   (order >= 1) and the truncated-support control.
===========  ==================================================================

The package is deliberately thin: it re-uses ``tensorsph.dam_break`` and
``tensorsph.vv.hydrostatic`` for the physics and adds only the parts the
campaign asks for -- the measurement definitions, the gates, the negative
controls and the provenance.
"""
from .evidence import EvidenceStore, SCHEMA_VERSION, sha256_of

__all__ = [
    "EvidenceStore",
    "SCHEMA_VERSION",
    "sha256_of",
    "dam_break_2d",
    "hydrostatic",
    "conservation",
    "convergence",
    "report",
]
