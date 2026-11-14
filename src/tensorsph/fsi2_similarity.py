"""Dimensionless-similarity audit for the Turek-Hron FSI2 benchmark case.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this module exists
----------------------
A comparison against the Turek-Hron FSI2 reference numbers (drag/lift
coefficients, beam-tip displacement amplitudes) is only meaningful if the case
that is run *is* the benchmark case, i.e. if it reproduces the benchmark's
dimensionless groups.  The benchmark itself is posed in SI units, so any
re-scaling of the problem (lower inlet velocity to keep the SPH Mach number
small, higher sound speed, a different particle spacing) silently changes the
physics unless the groups below are held fixed:

===================  ==================================================  ===========
group                definition                                          FSI2 target
===================  ==================================================  ===========
``Re``               rho_f U_mean D / mu                                 100
``rho_s/rho_f``      beam density / fluid density                        1
``Ae``               E / (rho_f U_mean^2)  ("normalised Young's modulus") 5.6e3
``nu_s``             solid Poisson ratio                                 0.4
``L/t``              beam length / beam thickness                        17.5
===================  ==================================================  ===========

``Ae`` is the one that is easy to lose: the beam is load-controlled by the
dynamic pressure ``q = 0.5 rho U^2``, so the small-deflection tip response
scales as ``delta ~ q L^4 / (E I) ~ 1/Ae`` and the bending frequency as
``f ~ sqrt(E) ~ sqrt(Ae)``.  Running at ``U_mean = 0.2`` m/s instead of the
benchmark's 1.0 m/s therefore requires ``E`` to be scaled by ``(0.2/1.0)^2``,
not kept at the dimensional SI value.

Provenance of the target values
-------------------------------
``Turek & Hron (2006)``, LNCSE 53, 371-385; the FSI2 parameter set is the one
already quoted in :mod:`tensorsph.fsi2_turek_hron` (``src/tensorsph/fsi2_turek_hron.py``,
"Parameters (Turek-Hron FSI2)" table).  The values were *not* re-verified against
the paper PDF in this round (no access); they are treated as secondary evidence
and are the quantities the repository has been claiming to match.

The SPHinXsys case constants below were read from the upstream files archived in
``benchmarks/xval_fsi2/reference/oss/README.md`` (``fsi2.h`` / ``fsi2.cpp`` line
numbers are given there).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = [
    "BENCHMARK_FSI2",
    "OSS_FSI2_CASE",
    "SimilarityGroup",
    "dimensionless_groups",
    "similarity_report",
    "amplitude_scaling_consequence",
    "similarity_corrected_case",
]


#: Turek-Hron FSI2 targets (secondary provenance: quoted in
#: ``fsi2_turek_hron.FSI2Case`` docstring; see module docstring).
BENCHMARK_FSI2: Dict[str, float] = {
    "Re": 100.0,
    "Ae": 5.6e3,
    "rho_s_over_rho_f": 1.0,
    "poisson_s": 0.4,
    "slenderness": 17.5,
}

#: The SPHinXsys ``test_2d_fsi2`` set-up, in its own scaled units
#: (``tests/2d_examples/test_2d_fsi2/fsi2.h``).
OSS_FSI2_CASE: Dict[str, float] = {
    "rho0_f": 1.0,
    "rho0_s": 10.0,
    "U_f": 1.0,
    "Re": 100.0,
    "Ae": 1.4e3,
    "poisson_s": 0.4,
    "beam_thickness": 0.2,
    "beam_length": 3.5,
    "length_scale": 10.0,
}

#: Relative tolerance used by :func:`similarity_report` for "same case" (a
#: discretisation is allowed to be run at a different resolution, not at a
#: different point in parameter space).
DEFAULT_TOL = 0.02

#: Groups that the discrete elastic-rod solid of this repository does not
#: consume (``beam_nu`` never enters ``EI`` or ``EA``).  They are reported but
#: do not make a case "different": for the rod model they are informational.
NOT_CONSUMED = ("poisson_s",)


@dataclass
class SimilarityGroup:
    """One dimensionless group: value, target, relative error, pass flag."""

    name: str
    value: float
    target: float
    rel_error: float
    passed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "target": self.target,
            "rel_error": self.rel_error,
            "passed": self.passed,
        }


def dimensionless_groups(case: Any) -> Dict[str, float]:
    """Dimensionless groups of a :class:`tensorsph.fsi2_turek_hron.FSI2Case`.

    Duck-typed on purpose: any object exposing ``rho0``, ``viscosity``,
    ``inlet_u_max``, ``cylinder_d``, ``beam_E``, ``beam_rho``, ``beam_nu``,
    ``beam_length`` and ``beam_thickness`` works (no torch import needed).
    """
    u_mean = 2.0 / 3.0 * float(case.inlet_u_max)
    return {
        "Re": float(case.rho0) * u_mean * float(case.cylinder_d) / float(case.viscosity),
        "Ae": float(case.beam_E) / (float(case.rho0) * u_mean ** 2),
        "rho_s_over_rho_f": float(case.beam_rho) / float(case.rho0),
        "poisson_s": float(case.beam_nu),
        "slenderness": float(case.beam_length) / float(case.beam_thickness),
        "mach": u_mean / float(getattr(case, "sound_speed", 1.0)),
    }


def similarity_report(
    case: Any,
    target: Optional[Dict[str, float]] = None,
    tol: float = DEFAULT_TOL,
) -> Dict[str, Any]:
    """Compare ``case`` with the FSI2 targets, group by group.

    Returns a dict with ``groups`` (list of :class:`SimilarityGroup` dicts),
    ``passed`` (all groups inside ``tol``) and ``failed`` (names outside).
    ``mach`` is reported but not gated -- it is a numerical-control parameter
    of the SPH model, not a property of the benchmark.
    """
    target = BENCHMARK_FSI2 if target is None else target
    groups = dimensionless_groups(case)
    out = []
    for name, ref in target.items():
        val = groups[name]
        rel = (val - ref) / ref
        out.append(SimilarityGroup(name, val, ref, rel, abs(rel) <= tol).as_dict())
    failed = [g["name"] for g in out if not g["passed"] and g["name"] not in NOT_CONSUMED]
    return {
        "groups": out,
        "passed": not failed,
        "failed": failed,
        "tol": tol,
        "mach": groups["mach"],
        "informational": [g["name"] for g in out if not g["passed"] and g["name"] in NOT_CONSUMED],
    }


def amplitude_scaling_consequence(
    case: Any, target: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """First-order consequence of a wrong ``Ae`` on the measured quantities.

    Small-deflection, load-controlled cantilever:

    * tip deflection ``delta ~ q L^4 / (E I)``  ->  ``delta/delta_ref = Ae_ref/Ae``
    * first bending frequency ``f ~ sqrt(E I / (rho_s A))``  ->
      ``f/f_ref = sqrt(Ae/Ae_ref)``

    These are the *expected* factors on a coupled comparison.  They are first
    order: VIV is not quasi-static and added mass is not in the ratio, so the
    numbers are a scale, not a prediction (the measured 9x discrepancy of the
    OSS record -- see ``docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md`` --
    mixes this effect with a 10x density-ratio change).
    """
    target = BENCHMARK_FSI2 if target is None else target
    groups = dimensionless_groups(case)
    ae = groups["Ae"]
    delta_ratio = target["Ae"] / ae
    freq_ratio = (ae / target["Ae"]) ** 0.5
    return {
        "Ae": ae,
        "Ae_target": target["Ae"],
        "tip_amplitude_ratio": delta_ratio,
        "bending_frequency_ratio": freq_ratio,
        "tip_amplitude_rel_error": delta_ratio - 1.0,
    }


def similarity_corrected_case(
    U_mean: float = 0.2,
    Re: float = 100.0,
    rho0: float = 1000.0,
    Ae: Optional[float] = None,
    **kwargs: Any,
):
    """Build an :class:`~tensorsph.fsi2_turek_hron.FSI2Case` at a chosen velocity.

    The geometry and the density ratios stay at the benchmark values; the
    viscosity and the beam Young's modulus are derived from ``Re`` and ``Ae`` so
    the case is similarity-equivalent to Turek-Hron FSI2 at an arbitrary (lower,
    SPH-friendly) inlet velocity.  ``kwargs`` are forwarded to ``FSI2Case``.

    Imported lazily so this module stays torch-free.
    """
    from .fsi2_turek_hron import FSI2Case

    target = BENCHMARK_FSI2
    ae = target["Ae"] if Ae is None else float(Ae)
    case = FSI2Case(**kwargs)
    # inlet_u_max = 1.5 U_mean for the benchmark's parabola u = 4 U_max y (H-y)/H^2
    case.inlet_u_max = 1.5 * U_mean
    case.rho0 = rho0
    case.viscosity = rho0 * U_mean * case.cylinder_d / float(Re)
    case.beam_E = ae * rho0 * U_mean ** 2
    case.beam_rho = rho0 * target["rho_s_over_rho_f"]
    case.beam_nu = target["poisson_s"]
    return case