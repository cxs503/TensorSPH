"""Verification-and-validation ladder for TensorSPH (zero-to-one anchors).

Three rungs, each with a closed reference and a machine-checkable gate:

1. :mod:`tensorsph.vv.hydrostatic` -- hydrostatic pressure balance against
   ``p = rho g h``;
2. :mod:`tensorsph.vv.tgv` -- Taylor-Green vortex viscous decay against the
   exact unsteady Navier-Stokes solution;
3. :mod:`tensorsph.vv.dambreak` -- dam-break free surface against the Ritter
   analytic front, cross-checked (not gated) against Martin & Moyce (1952).

:mod:`tensorsph.vv.ladder` runs the three rungs plus the negative controls and
emits one JSON document, :mod:`tensorsph.vv.schema` pins that document.
"""
from .schema import LADDER_SCHEMA_VERSION, validate_ladder_json

__all__ = ["LADDER_SCHEMA_VERSION", "validate_ladder_json"]
