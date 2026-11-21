"""Evidence-gated scheme selection for TensorSPH.

Why this module exists
----------------------
The repository ships several *different* SPH schemes (weakly compressible with
``delta``-SPH density diffusion, compressible/Riemann, incompressible
projection, CSF surface tension, thermal conduction, transport-velocity
correction, compression-based regularisation ...).  Picking the wrong one is a
silent accuracy failure: e.g. the WCSPH branch reproduces the wave-flume target
to ~3 % but misses the smooth inviscid Gresho vortex velocity field by ~37 %
(``benchmarks/gresho_vortex/results/verdict.json``).

Historically the choice was made in prose, per benchmark.  This module turns the
choice into a *pure function of a case profile* whose output is gated on the
repository's own verification records: a scheme is only ever ``SELECTED`` when a
machine-readable benchmark record exists for it and its measured error is inside
the owner precision standard (<15 %).  Everything else comes back as
``WITHHELD`` (no record) or ``FAILED`` (record exists, error too large) together
with the reason -- the selector is not allowed to nominate a method it cannot
back with evidence.

This is deliberately small and dependency-free (stdlib only) so it can be
imported by solvers, by ``python -m tensorsph.method_selection`` and by the
tests without pulling torch in.

The registry consumed here is produced programmatically by
``benchmarks/method_selection/build_registry.py`` from committed benchmark
artefacts (each record carries the source path and its sha256); see
``docs/reports/method_selection_registry.json``.

Typical use
-----------
>>> from tensorsph.method_selection import CaseProfile, select_method
>>> sel = select_method(CaseProfile(acoustic_mach=0.02, has_free_surface=True,
...                                 gravity_waves=True))
>>> sel.primary.method
'wcsph_delta_sph'
>>> sel.primary.status
'SELECTED'
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

__all__ = [
    "OWNER_TOLERANCE_PCT",
    "EVIDENCE_ACCURACY",
    "EVIDENCE_KINDS",
    "EVIDENCE_ROBUSTNESS",
    "CaseProfile",
    "Decision",
    "Evidence",
    "MethodSpec",
    "METHODS",
    "MODIFIERS",
    "RULES",
    "MODIFIER_RULES",
    "Selection",
    "default_registry_path",
    "load_registry",
    "evidence_for",
    "select_method",
    "render_decision",
    "main",
]

#: Owner precision standard: a benchmark error above this is not acceptable.
OWNER_TOLERANCE_PCT = 15.0

#: A record that measures an error against an analytic/reference answer.
EVIDENCE_ACCURACY = "accuracy"
#: A record that measures the *conditioning* of a reading (how far the number
#: would move under a modelling choice) rather than its distance to a reference.
#: Such a record is evidence about robustness; it may never be subtracted from,
#: nor silently added to, the accuracy claim.
EVIDENCE_ROBUSTNESS = "robustness"
EVIDENCE_KINDS = (EVIDENCE_ACCURACY, EVIDENCE_ROBUSTNESS)


# --------------------------------------------------------------------------- #
# Method catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MethodSpec:
    """A scheme TensorSPH actually implements (module paths are importable)."""

    key: str
    family: str
    modules: tuple[str, ...]
    summary: str
    citation: str


METHODS: dict[str, MethodSpec] = {
    "sun_delta_plus": MethodSpec(
        key="sun_delta_plus",
        family="weakly compressible SPH",
        modules=("tensorsph.delta_plus",),
        summary="2-D single-resolution δ⁺-SPH with evolved density, corrected diffusion "
                "and interface-aware positional shifting",
        citation="Sun et al. (2017), CMAME 315, 25–49, doi:10.1016/j.cma.2016.10.028",
    ),
    "wcsph_delta_sph": MethodSpec(
        key="wcsph_delta_sph",
        family="weakly compressible SPH",
        modules=("tensorsph.wcsph.solver", "tensorsph.density_diffusion"),
        summary="Tait EOS + delta-SPH density diffusion + XSPH, free-surface flows "
                "driven by gravity/inertia at low acoustic Mach number",
        citation="Antuono, Colagrossi & Marrone (2010) Comput. Phys. Commun. 181:532; "
                 "Monaghan (1992) J. Comput. Phys. 110:399 (XSPH)",
    ),
    "compressible_riemann": MethodSpec(
        key="compressible_riemann",
        family="compressible SPH",
        modules=("tensorsph.compressible.sod", "tensorsph.compressible.riemann_dissipation"),
        summary="explicit compressible SPH with Riemann-based artificial dissipation, "
                "shock/contact/rarefaction capturing",
        citation="Sod (1978) J. Comput. Phys. 27:1 (reference); Zhang, Hu & Adams "
                 "(2017) J. Comput. Phys. 338:210 (Riemann WCSPH dissipation)",
    ),
    "isph_projection": MethodSpec(
        key="isph_projection",
        family="incompressible projection SPH",
        modules=("tensorsph.isph_channel", "tensorsph.laplace_isph",
                 "tensorsph.projection_adjoint"),
        summary="pressure projection (Poisson solve) so that density stays exactly "
                "constant; closed/periodic low-Mach flows without free surface",
        citation="Cummins & Rudman (1999) J. Comput. Phys. 152:584",
    ),
    "wcsph_wall_no_slip": MethodSpec(
        key="wcsph_wall_no_slip",
        family="weakly compressible SPH",
        modules=("tensorsph.wall_shear", "tensorsph.wcsph.solver"),
        summary="WCSPH with layered no-slip walls; wall-bounded viscous channel "
                "flows (Couette/Poiseuille class)",
        citation="Adami, Hu & Adams (2012) J. Comput. Phys. 231:705 (wall boundary "
                 "condition); Morris, Fox & Zhu (1997) J. Comput. Phys. 136:214",
    ),
    "csf_surface_tension": MethodSpec(
        key="csf_surface_tension",
        family="continuum surface force",
        modules=("tensorsph.csf_curvature", "tensorsph.csf_consistent",
                 "tensorsph.csf_mixture"),
        summary="CSF surface tension with delta-correction / consistent-operator "
                "curvature; capillary-dominated drops and interfaces",
        citation="Brackbill, Kothe & Zemach (1992) J. Comput. Phys. 100:335; "
                 "Adami, He, Xiong, Hu (2012) J. Comput. Phys. 231:705 (delta-correction)",
    ),
    "heat_conduction_operator": MethodSpec(
        key="heat_conduction_operator",
        family="diffusion operator",
        modules=("tensorsph.heat_conduction",),
        summary="SPH Laplacian for thermal conduction, flux-normalised stencil",
        citation="Cleary & Monaghan (1999) J. Comput. Phys. 148:227",
    ),
}

#: Add-on terms that change *what the chosen solver does*, kept separate from the
#: primary scheme because several can be active at once.
MODIFIERS: dict[str, MethodSpec] = {
    "density_diffusion": MethodSpec(
        key="density_diffusion",
        family="delta-SPH density regularisation",
        modules=("tensorsph.density_diffusion",),
        summary="Molteni-Colagrossi delta-term keeps the free-surface density field "
                "from drifting; the wave-flume records use delta = 0.1",
        citation="Antuono, Colagrossi & Marrone (2010) Comput. Phys. Commun. 181:532",
    ),
    "transport_velocity_correction": MethodSpec(
        key="transport_velocity_correction",
        family="particle regularisation",
        modules=("tensorsph.transport_velocity_correction",),
        summary="limiter-damped transport-velocity shift removes tensile instability "
                "under strong stretching (breaking waves, impact)",
        citation="Zhang, Hu & Adams (2017) J. Comput. Phys. 338:210; Litvinov et al. "
                 "(2015) J. Comput. Phys. 300:327",
    ),
    "particle_shifting": MethodSpec(
        key="particle_shifting",
        family="particle regularisation",
        modules=("tensorsph.particle_shifting",),
        summary="Shifting scheme (non-uniform) to keep particle distribution regular "
                "in stretched/compressed regions",
        citation="Xu, Yu & Tian (2009) J. Comput. Phys. 228:5915 (Lind et al. 2012)",
    ),
    "compression_regularization": MethodSpec(
        key="compression_regularization",
        family="density regularisation",
        modules=("tensorsph.compression_regularization",),
        summary="SPHinXsys-style compression-sum / average-compression density reset",
        citation="Zhang et al. (SPHinXsys) compression-based density regularisation; "
                 "cross-validated in docs/report_xval_method_selection_m17501.md",
    ),
}


# --------------------------------------------------------------------------- #
# Case profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseProfile:
    """Dimensionless/physical descriptors of the case the user wants to solve.

    The fields are what the routing rules are allowed to look at; anything not
    described here cannot influence the decision (no hidden knobs).
    """

    #: acoustic Mach number U/c0 (<=0.1 keeps density error small for Tait EOS).
    acoustic_mach: float = 0.01
    #: free surface present?
    has_free_surface: bool = False
    #: boundary is driven by gravity waves (wave flume / sloshing class)?
    gravity_waves: bool = False
    #: shock / strong pressure jump that must be captured (M >= 0.3 or p2/p1 >> 1)
    shock_advection: bool = False
    #: capillary effects dominate (Weber <= 1 or length ~ capillary length)
    capillary_dominated: bool = False
    #: heavy/light density ratio across an interface (1.0 = single phase)
    density_ratio: float = 1.0
    #: bulk Reynolds number (informational for the wall-bounded branch)
    reynolds: float = 100.0
    #: energy transport by conduction is part of the question
    thermal_transport: bool = False
    #: flow stretches particles (breaking wave, impact, strong shear) -> tensile
    #: instability risk
    tensile_stretching: bool = False
    #: smooth inviscid vortical flow (Gresho/Taylor-Green class) without walls
    smooth_inviscid_vortex: bool = False
    #: "closed" | "periodic" | "open" -- closed/periodic allow a projection solve
    domain: str = "closed"

    def __post_init__(self) -> None:
        if self.domain not in ("closed", "periodic", "open"):
            raise ValueError(f"domain must be closed/periodic/open, got {self.domain!r}")
        if self.acoustic_mach < 0.0:
            raise ValueError("acoustic_mach must be >= 0")
        if self.density_ratio < 1.0:
            raise ValueError("density_ratio must be >= 1 (heavier/lightest)")
        if self.reynolds < 0.0:
            raise ValueError("reynolds must be >= 0")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "CaseProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"unknown CaseProfile field(s): {sorted(unknown)}")
        return cls(**dict(mapping))  # type: ignore[arg-type]

    def replace(self, **changes: object) -> "CaseProfile":
        return replace(self, **changes)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Evidence registry
# --------------------------------------------------------------------------- #
#: Record states.  ``banked`` records are the evidence a scheme may claim;
#: ``refuted`` records stay in the registry (deleting evidence is how a bad
#: reading gets forgotten) but may never enter an accuracy claim.
RECORD_BANKED = "banked"
RECORD_REFUTED = "refuted"
RECORD_STATES = (RECORD_BANKED, RECORD_REFUTED)


@dataclass(frozen=True)
class Evidence:
    """One benchmark record for one method, extracted from a committed artefact."""

    method: str
    case: str
    quantity: str
    measured_pct: float
    gate_pct: float
    source: str
    source_sha256: str = ""
    status: str = RECORD_BANKED
    precondition: str = ""
    #: ``accuracy`` (default) or ``robustness`` -- see ``EVIDENCE_ROBUSTNESS``.
    kind: str = EVIDENCE_ACCURACY

    def __post_init__(self) -> None:
        if self.status not in RECORD_STATES:
            raise ValueError(f"evidence status must be one of {RECORD_STATES}, "
                             f"got {self.status!r}")
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f"evidence kind must be one of {EVIDENCE_KINDS}, "
                             f"got {self.kind!r}")

    @property
    def own_gate_passed(self) -> bool:
        return self.measured_pct <= self.gate_pct

    @property
    def within_owner_tolerance(self) -> bool:
        return self.measured_pct <= OWNER_TOLERANCE_PCT

    @property
    def is_refuted(self) -> bool:
        """True when a later audit invalidated the measurement itself."""
        return self.status == RECORD_REFUTED

    @property
    def is_accuracy(self) -> bool:
        """True when the record is an error against a reference (not conditioning)."""
        return self.kind == EVIDENCE_ACCURACY

    def as_dict(self) -> dict:
        out = asdict(self)
        out["own_gate_passed"] = self.own_gate_passed
        out["within_owner_tolerance"] = self.within_owner_tolerance
        out["is_refuted"] = self.is_refuted
        return out


def default_registry_path() -> Path:
    """``docs/reports/method_selection_registry.json`` relative to the repo root."""
    root = Path(__file__).resolve().parents[2]
    return root / "docs" / "reports" / "method_selection_registry.json"


def load_registry(path: str | Path | None = None) -> dict[str, list[Evidence]]:
    """Load the evidence registry, indexed by method key.

    A missing registry is an error, not a silent "everything is fine": without
    evidence the selector must not select anything.
    """
    target = Path(path) if path is not None else default_registry_path()
    doc = json.loads(target.read_text())
    if not isinstance(doc, dict) or "records" not in doc:
        raise ValueError(f"{target}: registry must be an object with a 'records' list")
    registry: dict[str, list[Evidence]] = {}
    for raw in doc["records"]:
        ev = Evidence(
            method=str(raw["method"]),
            case=str(raw["case"]),
            quantity=str(raw["quantity"]),
            measured_pct=float(raw["measured_pct"]),
            gate_pct=float(raw["gate_pct"]),
            source=str(raw["source"]),
            source_sha256=str(raw.get("source_sha256", "")),
            status=str(raw.get("status", RECORD_BANKED)),
            precondition=str(raw.get("precondition", "")),
            kind=str(raw.get("kind", EVIDENCE_ACCURACY)),
        )
        registry.setdefault(ev.method, []).append(ev)
    return registry


def evidence_for(method: str, registry: Mapping[str, Sequence[Evidence]]) -> list[Evidence]:
    return list(registry.get(method, ()))


def case_family(case: str) -> str:
    """Coarse provenance key of an evidence record.

    ``"CASE-LAPLACE-SPH (consistent CSF)"`` and ``"CASE-LAPLACE-SPH"`` are two
    variants of *one* benchmark, not two independent measurements: a bound that
    is only reproduced on variants of the same case has not been reproduced.
    """
    return re.split(r"[()\[\]]", case.strip(), maxsplit=1)[0].strip().rstrip(",;").lower()


# --------------------------------------------------------------------------- #
# Decision objects
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    method: str
    status: str  # SELECTED | WITHHELD | FAILED
    role: str  # "primary" | "modifier"
    rationale: str
    citation: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def worst_measured_pct(self) -> float | None:
        """Accuracy bound claimed for this scheme.

        The worst error among the records that are *inside* the owner standard;
        if nothing passes, the worst error overall (so a FAILED decision shows
        how far off it is).
        """
        if not self.evidence:
            return None
        banked = [ev for ev in self.evidence if not ev.is_refuted]
        if not banked:
            return None
        passing = [ev.measured_pct for ev in banked if ev.within_owner_tolerance]
        if passing:
            return max(passing)
        return max(ev.measured_pct for ev in banked)

    @property
    def worst_banked_pct(self) -> float | None:
        """Worst error over *every* banked record, failing ones included.

        ``worst_measured_pct`` deliberately quotes only records inside the owner
        standard; that makes it a *claim*, not a bound on the scheme's worst
        measured behaviour.  This property is the honest companion: if the two
        differ, the decision card must say so (mission 20432, evidence-coverage
        audit in ``docs/reports/method_selection_coverage_audit_m20432.md``).
        """
        if not self.evidence:
            return None
        return max(ev.measured_pct for ev in self.evidence)

    @property
    def worst_banked_record(self) -> Evidence | None:
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda ev: ev.measured_pct)

    @property
    def worst_accuracy_pct(self) -> float | None:
        """Worst *accuracy* record of this scheme, refuted ones excluded.

        This is the number a selection may stand on: ``worst_measured_pct`` is
        a maximum over records that happen to be inside the owner standard, so
        it is an existential claim ("some benchmark passed"), not a bound on the
        scheme's worst measured error.  Conditioning records
        (``kind=robustness``, e.g. the mobility of a reading across a sweep) are
        reported separately by ``worst_robustness_pct``: adding them to an error
        bound would compare two different quantities.
        """
        values = [ev.measured_pct for ev in self.evidence
                  if ev.is_accuracy and not ev.is_refuted]
        return max(values) if values else None

    @property
    def worst_accuracy_record(self) -> Evidence | None:
        live = [ev for ev in self.evidence if ev.is_accuracy and not ev.is_refuted]
        return max(live, key=lambda ev: ev.measured_pct) if live else None

    @property
    def worst_robustness_pct(self) -> float | None:
        values = [ev.measured_pct for ev in self.evidence
                  if not ev.is_accuracy and not ev.is_refuted]
        return max(values) if values else None

    @property
    def accuracy_within_owner_standard(self) -> bool | None:
        """``None`` when the scheme has no live accuracy record at all."""
        worst = self.worst_accuracy_pct
        if worst is None:
            return None
        return worst <= OWNER_TOLERANCE_PCT

    @property
    def refuted_evidence(self) -> list[Evidence]:
        """Records kept for the audit trail but excluded from the claim."""
        return [ev for ev in self.evidence if ev.is_refuted]

    @property
    def live_accuracy_records(self) -> list[Evidence]:
        """Accuracy records still standing (refuted and conditioning records out).

        These are the records the claim above actually rests on; ``refuted`` ones
        were invalidated by a later audit and ``kind=robustness`` ones measure
        conditioning, not distance to a reference.
        """
        return [ev for ev in self.evidence if ev.is_accuracy and not ev.is_refuted]

    @property
    def evidence_depth(self) -> dict:
        """How much independent measurement stands behind this decision.

        A number measured once is a data point, not a bound.  This dict makes the
        multiplicity explicit -- record count, distinct benchmark *families*,
        and the spread between the best and the worst live accuracy record -- so
        a card cannot present a one-record figure as if it had been reproduced.
        ``single_source`` is True when every live accuracy record comes from one
        benchmark family (or there is at most one of them).
        """
        live = self.live_accuracy_records
        families = {case_family(ev.case) for ev in live}
        values = [ev.measured_pct for ev in live]
        return {
            "n_records": len(self.evidence),
            "n_live_accuracy": len(live),
            "n_benchmark_families": len(families),
            "benchmark_families": sorted(families),
            "single_source": 0 < len(families) <= 1,
            "worst_pct": max(values) if values else None,
            "best_pct": min(values) if values else None,
            "spread_pct": (max(values) - min(values)) if len(values) > 1 else None,
        }

    @property
    def is_single_source(self) -> bool:
        """True when the scheme's evidence comes from a single benchmark family."""
        return bool(self.live_accuracy_records) and self.evidence_depth["single_source"]

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "status": self.status,
            "role": self.role,
            "rationale": self.rationale,
            "citation": self.citation,
            "worst_measured_pct": self.worst_measured_pct,
            "worst_banked_pct": self.worst_banked_pct,
            "worst_accuracy_pct": self.worst_accuracy_pct,
            "worst_robustness_pct": self.worst_robustness_pct,
            "accuracy_within_owner_standard": self.accuracy_within_owner_standard,
            "n_refuted": len(self.refuted_evidence),
            "evidence_depth": self.evidence_depth,
            "evidence": [ev.as_dict() for ev in self.evidence],
            "caveats": list(self.caveats),
        }


@dataclass
class Selection:
    profile: CaseProfile
    primary: Decision | None
    modifiers: list[Decision]
    candidates: list[Decision]
    notes: list[str] = field(default_factory=list)
    #: True when the recommendation was built with ``strict_bound=True``: only
    #: schemes whose *worst* live accuracy record is inside the owner standard
    #: may be put forward (see ``select_method``).
    strict_bound: bool = False
    #: Schemes the rules would have put forward but that strict mode refused:
    #: kept here (never dropped silently) with the record that disqualified them.
    strict_excluded: list[Decision] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        """True when a rule produced an evidence-backed scheme (status gate only).

        Note what this does *not* say: it does not say the scheme's worst
        accuracy record is inside the owner standard.  Use
        ``within_owner_standard`` for that.
        """
        return self.primary is not None and self.primary.status == "SELECTED"

    @property
    def endorsed(self) -> list[Decision]:
        """The decisions this selection actually puts forward (primary+modifiers)."""
        return ([self.primary] if self.primary is not None else []) + list(self.modifiers)

    @property
    def headline_bound_pct(self) -> float | None:
        """Worst live *accuracy* record among the schemes put forward here.

        One number for the whole recommendation, so that a card cannot advertise
        the primary's favourable record while a modifier carries a 17 % one.
        ``None`` when nothing put forward has an accuracy record.
        """
        values = [d.worst_accuracy_pct for d in self.endorsed
                  if d.worst_accuracy_pct is not None]
        return max(values) if values else None

    @property
    def headline_record(self) -> tuple[str, Evidence] | None:
        """(method, record) that sets ``headline_bound_pct``."""
        best: tuple[str, Evidence] | None = None
        for decision in self.endorsed:
            record = decision.worst_accuracy_record
            if record is None:
                continue
            if best is None or record.measured_pct > best[1].measured_pct:
                best = (decision.method, record)
        return best

    @property
    def single_sourced(self) -> list[str]:
        """Schemes put forward whose bound rests on one benchmark family.

        Not a defect by itself -- but the consumer of a card is entitled to know
        which of its numbers have never been reproduced on a second benchmark.
        """
        return [d.method for d in self.endorsed if d.is_single_source]

    @property
    def weakest_link(self) -> dict | None:
        """The put-forward scheme with the fewest independent measurements."""
        endorsed = [d for d in self.endorsed if d.live_accuracy_records]
        if not endorsed:
            return None
        worst = min(endorsed, key=lambda d: (d.evidence_depth["n_benchmark_families"],
                                             d.evidence_depth["n_live_accuracy"]))
        info = dict(worst.evidence_depth)
        info["method"] = worst.method
        return info

    @property
    def within_owner_standard(self) -> bool:
        """The whole recommendation, not one favourable record, is inside 15 %."""
        if not self.is_verified:
            return False
        bound = self.headline_bound_pct
        return bound is not None and bound <= OWNER_TOLERANCE_PCT

    def as_dict(self) -> dict:
        headline = self.headline_record
        return {
            "profile": asdict(self.profile),
            "strict_bound": self.strict_bound,
            "strict_excluded": [
                {"method": d.method, "role": d.role, "status": d.status,
                 "worst_accuracy_pct": d.worst_accuracy_pct,
                 "worst_accuracy_case": (
                     None if d.worst_accuracy_record is None
                     else d.worst_accuracy_record.case),
                 "rationale": d.rationale, "caveats": list(d.caveats)}
                for d in self.strict_excluded
            ],
            "primary": self.primary.as_dict() if self.primary else None,
            "modifiers": [d.as_dict() for d in self.modifiers],
            "candidates": [d.as_dict() for d in self.candidates],
            "notes": list(self.notes),
            "verified": self.is_verified,
            "headline_bound_pct": self.headline_bound_pct,
            "headline_quantity": (
                None if headline is None
                else f"{headline[0]}: {headline[1].case} -- {headline[1].quantity}"
            ),
            "within_owner_standard": self.within_owner_standard,
            "single_sourced": self.single_sourced,
            "weakest_link": self.weakest_link,
            "endorsed_depth": {d.method: d.evidence_depth for d in self.endorsed},
        }


# --------------------------------------------------------------------------- #
# Routing rules
# --------------------------------------------------------------------------- #
Predicate = Callable[[CaseProfile], bool]
Resolver = Callable[[CaseProfile], tuple[tuple[str, ...], str]]


@dataclass(frozen=True)
class Rule:
    key: str
    statement: str
    fires: Predicate
    resolve: Resolver  # -> (ordered candidate method keys, rationale)


def _shock(profile: CaseProfile) -> bool:
    return profile.shock_advection or profile.acoustic_mach >= 0.3


def _capillary(profile: CaseProfile) -> bool:
    return profile.capillary_dominated


def _free_surface_flow(profile: CaseProfile) -> bool:
    return profile.has_free_surface and not _shock(profile)


def _vortex_class_case(case: str) -> bool:
    """True when a benchmark's case name is a smooth inviscid vortex family."""
    lowered = case.lower()
    return "vortex" in lowered or "taylor-green" in lowered or "taylor_green" in lowered


def _wall_bounded(profile: CaseProfile) -> bool:
    return (not profile.has_free_surface) and profile.domain in ("closed", "periodic")


RULES: tuple[Rule, ...] = (
    Rule(
        key="shock",
        statement="shock/pressure jump (M>=0.3) -> compressible Riemann SPH; "
                  "Tait-based WCSPH cannot capture a discontinuity",
        fires=_shock,
        resolve=lambda p: (
            ("compressible_riemann",),
            f"acoustic Mach {p.acoustic_mach:g}"
            + (" and explicit shock flag" if p.shock_advection else "")
            + " >= 0.3: weak compressibility assumption breaks down",
        ),
    ),
    Rule(
        key="thermal",
        statement="conduction is part of the question -> heat-conduction operator "
                  "on top of the flow solver",
        fires=lambda p: p.thermal_transport,
        resolve=lambda p: (
            ("heat_conduction_operator",),
            "thermal_transport=True: the SPH Laplacian with flux-normalised "
            "stencil is the validated conduction operator",
        ),
    ),
    Rule(
        key="capillary",
        statement="capillary-dominated interface -> CSF surface tension",
        fires=_capillary,
        resolve=lambda p: (
            ("csf_surface_tension",),
            "capillary_dominated=True: surface-tension force is a leading-order "
            "term, CSF with delta-correction is the implemented model",
        ),
    ),
    Rule(
        key="free_surface",
        statement="free surface + low Mach -> WCSPH with delta-SPH density diffusion",
        fires=_free_surface_flow,
        resolve=lambda p: (
            ("wcsph_delta_sph",),
            f"free surface with acoustic Mach {p.acoustic_mach:g} < 0.1: the "
            "weakly compressible branch is the implemented free-surface solver",
        ),
    ),
    Rule(
        key="wall_bounded",
        statement="closed/periodic incompressible domain -> projection (ISPH) first, "
                  "else WCSPH with no-slip wall layers",
        fires=_wall_bounded,
        resolve=lambda p: (
            (
                ("isph_projection", "wcsph_wall_no_slip")
                if not p.smooth_inviscid_vortex
                else ("isph_projection",)
            ),
            f"domain={p.domain!r} with Re={p.reynolds:g}: a divergence-free "
            "projection solve is the preferred branch"
            + (
                ", WCSPH no-slip walls the fallback"
                if not p.smooth_inviscid_vortex
                else "; the dissipative WCSPH fallback is NOT allowed here -- the "
                "Gresho vortex record shows it loses ~37 % of the velocity field"
            ),
        ),
    ),
)


#: Rules that identify physics a *later* rule's scheme cannot represent.  If one
#: of them fires and none of its candidates can be put forward (no evidence, or
#: refuted evidence), the walk must stop: continuing would replace, say, a
#: compressible Riemann solver for an M=0.9 shock tube with an incompressible
#: projection solver that has no shock physics at all.  Found by the
#: leave-one-record-out audit in mission 21797 -- the substitution only appears
#: once a record is missing, which is exactly when nobody looks.
EXCLUSIVE_RULES = frozenset({"shock", "capillary"})

@dataclass(frozen=True)
class ModifierRule:
    key: str
    statement: str
    fires: Predicate
    methods: tuple[str, ...]


MODIFIER_RULES: tuple[ModifierRule, ...] = (
    ModifierRule(
        key="free_surface_density",
        statement="free-surface flow keeps delta-SPH density diffusion (delta=0.1)",
        fires=lambda p: p.has_free_surface and not _shock(p),
        methods=("density_diffusion",),
    ),
    ModifierRule(
        key="tensile_stretching",
        statement="strong stretching -> transport-velocity correction and/or "
                  "particle shifting to avoid tensile instability",
        fires=lambda p: p.tensile_stretching,
        methods=("transport_velocity_correction", "particle_shifting"),
    ),
    ModifierRule(
        key="density_reset",
        statement="long-running nearly-incompressible free-surface runs may add "
                  "compression-based density regularisation instead of delta-SPH",
        fires=lambda p: p.has_free_surface and p.acoustic_mach <= 0.05,
        methods=("compression_regularization",),
    ),
)


# --------------------------------------------------------------------------- #
# Selector
# --------------------------------------------------------------------------- #
def _classify(method_key: str, role: str, rationale: str,
              registry: Mapping[str, Sequence[Evidence]],
              catalogue: Mapping[str, MethodSpec]) -> Decision:
    """Turn evidence records into a SELECTED/WITHHELD/FAILED decision."""
    spec = catalogue[method_key]
    records = evidence_for(method_key, registry)
    caveats: list[str] = []
    if not records:
        return Decision(
            method=method_key, status="WITHHELD", role=role,
            rationale=rationale + " -- however no verification record exists",
            citation=spec.citation, evidence=[],
            caveats=["no benchmark record in the method-selection registry: "
                     "the rule fires but the scheme stays unselectable "
                     "(evidence gate)"],
        )
    banked = [ev for ev in records if not ev.is_refuted]
    refuted = [ev for ev in records if ev.is_refuted]
    if not banked:
        return Decision(
            method=method_key, status="WITHHELD", role=role,
            rationale=rationale + " -- every record for this scheme was refuted "
                                   "by a later audit, so it has no live evidence",
            citation=spec.citation, evidence=records,
            caveats=[
                f"refuted: {ev.case}: {ev.quantity} = {ev.measured_pct:.3g}% -- "
                f"{ev.precondition or 'invalidated by a later audit'} ({ev.source})"
                for ev in refuted
            ],
        )
    accuracy_banked = [ev for ev in banked if ev.is_accuracy]
    if not accuracy_banked:
        return Decision(
            method=method_key, status="WITHHELD", role=role,
            rationale=rationale + " -- its records measure the conditioning of a "
                                   "reading, not an error against a reference, so "
                                   "there is no accuracy evidence to stand on",
            citation=spec.citation, evidence=records,
            caveats=[
                f"{ev.case}: {ev.quantity} = {ev.measured_pct:.3g}% "
                f"(kind={ev.kind}, {ev.source})"
                for ev in banked
            ],
        )
    # Only accuracy records may satisfy the owner standard: a conditioning
    # record that happens to read 0 % (e.g. a sweep with zero mobility) must not
    # rescue a scheme whose real error records fail (m21115).
    passing = [ev for ev in accuracy_banked if ev.within_owner_tolerance]
    for ev in banked:
        if not ev.is_accuracy:
            caveats.append(
                f"{ev.case}: {ev.quantity} = {ev.measured_pct:.3g}% -- a conditioning "
                f"record (kind={ev.kind}), not an error against a reference: reported "
                f"for transparency, excluded from the accuracy bound ({ev.source})"
            )
        elif not ev.within_owner_tolerance:
            caveats.append(
                f"{ev.case}: {ev.quantity} = {ev.measured_pct:.3g}% exceeds the "
                f"{OWNER_TOLERANCE_PCT:g}% owner standard ({ev.source})"
            )
        elif not ev.own_gate_passed:
            caveats.append(
                f"{ev.case}: {ev.quantity} = {ev.measured_pct:.3g}% misses its own "
                f"{ev.gate_pct:g}% gate but is inside {OWNER_TOLERANCE_PCT:g}% "
                f"({ev.source})"
            )
    for ev in refuted:
        caveats.append(
            f"refuted record, excluded from the accuracy claim: {ev.case}: "
            f"{ev.quantity} = {ev.measured_pct:.3g}% -- "
            f"{ev.precondition or 'invalidated by a later audit'} ({ev.source})"
        )
    if not passing:
        worst = max(banked, key=lambda ev: ev.measured_pct)
        return Decision(
            method=method_key, status="FAILED", role=role,
            rationale=rationale + " -- but its own benchmarks are outside tolerance",
            citation=spec.citation, evidence=records,
            caveats=caveats or [
                f"worst record {worst.case}: {worst.measured_pct:.3g}%"
            ],
        )
    return Decision(
        method=method_key, status="SELECTED", role=role,
        rationale=rationale, citation=spec.citation, evidence=records,
        caveats=caveats,
    )


def _strict_eligible(decision: Decision) -> bool:
    """Strict bound gate: the scheme's *worst* live accuracy record must be inside
    the owner standard.

    ``Decision.status`` answers an existential question ("some benchmark of this
    scheme passed"); a recommendation needs a bound ("no benchmark of it is
    outside 15 %").  m21115 showed the two differ for three schemes, so the
    difference is made explicit here instead of being buried in a card note.
    """
    return decision.accuracy_within_owner_standard is True


def _strict_reason(decision: Decision) -> str:
    record = decision.worst_accuracy_record
    worst = decision.worst_accuracy_pct
    if worst is None:
        return "no live accuracy record, so no bound can be claimed"
    where = "" if record is None else f" ({record.case}: {record.quantity})"
    return (f"worst live accuracy record {worst:.3g}%{where} exceeds the "
            f"{OWNER_TOLERANCE_PCT:g}% owner standard")


def select_method(profile: CaseProfile | Mapping[str, object],
                  registry: Mapping[str, Sequence[Evidence]] | None = None,
                  registry_path: str | Path | None = None,
                  strict_bound: bool = False) -> Selection:
    """Route a case profile to a scheme, gated on the evidence registry.

    ``strict_bound=True`` additionally requires every scheme put forward
    (primary and modifiers) to have a *worst* live accuracy record inside the
    owner standard; schemes the rules would have added are then listed in
    ``Selection.strict_excluded`` with the record that disqualified them.  The
    default (``False``) keeps the historical route matrix unchanged, which is
    what the pinned XVAL matrix checks.
    """
    if isinstance(profile, Mapping):
        profile = CaseProfile.from_mapping(profile)
    if registry is None:
        registry = load_registry(registry_path)

    notes: list[str] = []
    candidates: list[Decision] = []
    strict_excluded: list[Decision] = []
    primary: Decision | None = None

    for rule in RULES:
        if not rule.fires(profile):
            continue
        keys, rationale = rule.resolve(profile)
        scoped = registry
        if rule.key == "wall_bounded" and profile.smooth_inviscid_vortex:
            # A wall-bounded channel record does not validate a periodic smooth
            # vortex: only vortex/Taylor-Green-class evidence may claim it.
            scoped = {
                k: [ev for ev in v if _vortex_class_case(ev.case)]
                for k, v in registry.items()
            }
        decisions = [
            _classify(key, "primary", f"[{rule.key}] {rationale}", scoped, METHODS)
            for key in keys
        ]
        candidates.extend(decisions)
        selectable = [d for d in decisions if d.status == "SELECTED"
                      and (not strict_bound or _strict_eligible(d))]
        for d in decisions:
            if d.status == "SELECTED" and strict_bound and not _strict_eligible(d):
                strict_excluded.append(d)
                notes.append(f"strict bound: {d.method} is evidence-backed "
                             f"({d.status}) but " + _strict_reason(d))
        chosen = selectable[0] if selectable else None
        if chosen is not None:
            primary = chosen
            blocked = [d for d in decisions if d is not chosen]
            for d in blocked:
                # every caveat, not just the first: a fallback whose *worst*
                # quantity fails the owner standard must say so here (m20432).
                detail = "; ".join(d.caveats) if d.caveats else ""
                notes.append(
                    f"{d.method} preferred by rule {rule.key!r} but {d.status}: "
                    f"{detail}".rstrip(": ")
                )
            if chosen is not decisions[0]:
                notes.append(
                    f"rule {rule.key!r}: fell back from {decisions[0].method} to "
                    f"{chosen.method} on evidence grounds"
                )
            break
        notes.append(
            f"rule {rule.key!r} fired but no candidate is selectable "
            f"({', '.join(f'{d.method}={d.status}' for d in decisions)})"
        )
        if rule.key in EXCLUSIVE_RULES:
            notes.append(
                f"regime locked: rule {rule.key!r} identifies the physics of this "
                "case but none of its candidates can be put forward "
                f"({', '.join(f'{d.method}={d.status}' for d in decisions)}); "
                "no later rule is substituted, because its schemes cannot represent "
                "this regime -- the honest answer is WITHHELD"
            )
            break
        if strict_bound and any(d.status == "SELECTED" for d in decisions):
            # The rule that *covers* this regime has no compliant scheme.  Walking
            # on to a later rule would substitute a scheme chosen for a different
            # regime (e.g. surface-tension-free WCSPH for a capillary problem),
            # which is worse than withholding: stop here and say why.
            notes.append(
                f"strict bound leaves rule {rule.key!r} without a compliant scheme; "
                "no later rule is substituted, because it does not cover this regime"
            )
            break

    if primary is None:
        notes.append(
            "no rule produced an evidence-backed scheme: profile outside the "
            "validated regime of this repository, do not run silently"
        )

    modifiers: list[Decision] = []
    if primary is None and any(rule.fires(profile) for rule in MODIFIER_RULES):
        # An add-on term changes *what the chosen solver does*; with no primary
        # there is nothing to add it to, so listing one would be a promise the
        # decision card cannot keep (m21537).
        notes.append("modifiers withheld: no primary scheme was selected, so "
                     "no add-on term can be applied")
    for rule in (MODIFIER_RULES if primary is not None else ()):
        if not rule.fires(profile):
            continue
        for key in rule.methods:
            decision = _classify(key, "modifier", f"[{rule.key}] {rule.statement}",
                                 registry, MODIFIERS)
            if strict_bound and not _strict_eligible(decision):
                strict_excluded.append(decision)
                notes.append(f"strict bound: modifier {decision.method} withheld -- "
                             + _strict_reason(decision))
                continue
            modifiers.append(decision)

    return Selection(profile=profile, primary=primary, modifiers=modifiers,
                     candidates=candidates, notes=notes, strict_bound=strict_bound,
                     strict_excluded=strict_excluded)


def render_decision(selection: Selection) -> str:
    """Human-readable decision card (used by the CLI and the report)."""
    lines = ["# method-selection decision", "", "## profile"]
    for key, value in asdict(selection.profile).items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## verdict"]
    bound = selection.headline_bound_pct
    headline = selection.headline_record
    if bound is None:
        lines.append(f"- accuracy bound: unavailable (no accuracy record among the "
                     f"schemes put forward; owner standard {OWNER_TOLERANCE_PCT:g}%)")
    else:
        where = "" if headline is None else f" -- set by {headline[0]} ({headline[1].case})"
        lines.append(
            f"- accuracy bound: {bound:.3g}% over every live accuracy record of the "
            f"schemes put forward (owner standard {OWNER_TOLERANCE_PCT:g}%){where}"
        )
    lines.append(
        "- within owner standard: " + ("YES" if selection.within_owner_standard else "NO")
        + ("" if selection.is_verified else " (nothing was selected)")
    )

    lines += ["", "## primary scheme"]
    if selection.primary is None:
        lines.append("- none (WITHHELD: no evidence-backed rule fired)")
    else:
        d = selection.primary
        worst = d.worst_measured_pct
        lines.append(f"- {d.method}: {d.status}")
        lines.append(f"- why: {d.rationale}")
        if worst is not None:
            lines.append(
                f"- claimed bound: {worst:.3g}% (owner standard "
                f"{OWNER_TOLERANCE_PCT:g}%; passing records only)"
            )
            accuracy = d.worst_accuracy_pct
            if accuracy is not None and accuracy > worst + 1e-12:
                record = d.worst_accuracy_record
                case = f" ({record.case})" if record else ""
                lines.append(
                    f"- accuracy bound: {accuracy:.3g}%{case} -- the worst live "
                    "accuracy record of this scheme; above it the claim above is "
                    "not reachable"
                )
            robustness = d.worst_robustness_pct
            if robustness is not None:
                lines.append(
                    f"- conditioning record (not an error vs a reference): "
                    f"{robustness:.3g}%"
                )
            banked = d.worst_banked_pct
            if banked is not None and banked > worst + 1e-12:
                record = d.worst_banked_record
                case = f" ({record.case}: {record.quantity})" if record else ""
                lines.append(
                    f"- worst banked record: {banked:.3g}%{case} -- outside the "
                    "owner standard, so the claim above is NOT a bound on this "
                    "scheme's worst measured error"
                )
        if d.citation:
            lines.append(f"- reference: {d.citation}")
        for caveat in d.caveats:
            lines.append(f"- caveat: {caveat}")
    lines += ["", "## modifiers"]
    if not selection.modifiers:
        lines.append("- none")
    for d in selection.modifiers:
        lines.append(f"- {d.method}: {d.status} -- {d.rationale.split('] ', 1)[-1]}")
        for caveat in d.caveats:
            lines.append(f"    - {caveat}")
    if selection.strict_bound:
        lines += ["", f"## strict bound (worst live accuracy record <= "
                      f"{OWNER_TOLERANCE_PCT:g}%)"]
        if not selection.strict_excluded:
            lines.append("- nothing excluded: every scheme put forward is inside "
                         "the owner standard")
        for d in selection.strict_excluded:
            lines.append(f"- excluded {d.method} [{d.role}]: " + _strict_reason(d))
    depth_rows = [d for d in selection.endorsed if d.live_accuracy_records]
    if depth_rows:
        lines += ["", "## evidence depth"]
        for d in depth_rows:
            info = d.evidence_depth
            if info["single_source"]:
                tag = ("SINGLE SOURCE (1 benchmark family)" if info["n_live_accuracy"] <= 1
                       else f"single source ({info['n_live_accuracy']} records, "
                            "1 benchmark family)")
            else:
                tag = (f"{info['n_benchmark_families']} benchmark families, "
                       f"{info['n_live_accuracy']} records")
            spread = info["spread_pct"]
            extra = "" if spread is None else f"; spread {spread:.3g} pp over live records"
            lines.append(f"- {d.method}: {tag}{extra}")
        if selection.single_sourced:
            lines.append(
                "- caveat: the bound of "
                + ", ".join(selection.single_sourced)
                + " has not been reproduced on a second benchmark family -- "
                "treat it as a measured data point, not an independently confirmed bound"
            )
    if selection.notes:
        lines += ["", "## notes"]
        lines += [f"- {note}" for note in selection.notes]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _profiles_from_cli(argv: Sequence[str]) -> tuple[CaseProfile, str | None]:
    parser = argparse.ArgumentParser(
        prog="python -m tensorsph.method_selection",
        description="Evidence-gated SPH scheme selection for TensorSPH.",
    )
    parser.add_argument("--json", help="path to a JSON object of CaseProfile fields")
    parser.add_argument("--registry", help="override registry JSON path")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict-bound", action="store_true",
                        help="only put forward schemes whose worst live accuracy "
                             "record is inside the owner standard")
    args = parser.parse_args(argv)
    if args.json:
        profile = CaseProfile.from_mapping(json.loads(Path(args.json).read_text()))
    else:
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    return profile, args.registry, bool(args.strict_bound)


def main(argv: Sequence[str] | None = None) -> int:
    profile, registry_path, strict_bound = _profiles_from_cli(
        sys.argv[1:] if argv is None else argv)
    selection = select_method(profile, registry_path=registry_path,
                              strict_bound=strict_bound)
    sys.stdout.write(render_decision(selection))
    return 0 if selection.is_verified else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
