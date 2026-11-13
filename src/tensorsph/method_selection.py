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
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

__all__ = [
    "OWNER_TOLERANCE_PCT",
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

    @property
    def own_gate_passed(self) -> bool:
        return self.measured_pct <= self.gate_pct

    @property
    def within_owner_tolerance(self) -> bool:
        return self.measured_pct <= OWNER_TOLERANCE_PCT

    def as_dict(self) -> dict:
        out = asdict(self)
        out["own_gate_passed"] = self.own_gate_passed
        out["within_owner_tolerance"] = self.within_owner_tolerance
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
        )
        registry.setdefault(ev.method, []).append(ev)
    return registry


def evidence_for(method: str, registry: Mapping[str, Sequence[Evidence]]) -> list[Evidence]:
    return list(registry.get(method, ()))


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
        passing = [ev.measured_pct for ev in self.evidence if ev.within_owner_tolerance]
        if passing:
            return max(passing)
        return max(ev.measured_pct for ev in self.evidence)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "status": self.status,
            "role": self.role,
            "rationale": self.rationale,
            "citation": self.citation,
            "worst_measured_pct": self.worst_measured_pct,
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

    @property
    def is_verified(self) -> bool:
        return self.primary is not None and self.primary.status == "SELECTED"

    def as_dict(self) -> dict:
        return {
            "profile": asdict(self.profile),
            "primary": self.primary.as_dict() if self.primary else None,
            "modifiers": [d.as_dict() for d in self.modifiers],
            "candidates": [d.as_dict() for d in self.candidates],
            "notes": list(self.notes),
            "verified": self.is_verified,
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
    passing = [ev for ev in records if ev.within_owner_tolerance]
    for ev in records:
        if not ev.within_owner_tolerance:
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
    if not passing:
        worst = max(records, key=lambda ev: ev.measured_pct)
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


def select_method(profile: CaseProfile | Mapping[str, object],
                  registry: Mapping[str, Sequence[Evidence]] | None = None,
                  registry_path: str | Path | None = None) -> Selection:
    """Route a case profile to a scheme, gated on the evidence registry."""
    if isinstance(profile, Mapping):
        profile = CaseProfile.from_mapping(profile)
    if registry is None:
        registry = load_registry(registry_path)

    notes: list[str] = []
    candidates: list[Decision] = []
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
        chosen = next((d for d in decisions if d.status == "SELECTED"), None)
        if chosen is not None:
            primary = chosen
            blocked = [d for d in decisions if d is not chosen]
            for d in blocked:
                notes.append(
                    f"{d.method} preferred by rule {rule.key!r} but {d.status}: "
                    f"{d.caveats[0] if d.caveats else ''}".rstrip(": ")
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

    if primary is None:
        notes.append(
            "no rule produced an evidence-backed scheme: profile outside the "
            "validated regime of this repository, do not run silently"
        )

    modifiers: list[Decision] = []
    for rule in MODIFIER_RULES:
        if not rule.fires(profile):
            continue
        for key in rule.methods:
            modifiers.append(
                _classify(key, "modifier", f"[{rule.key}] {rule.statement}",
                          registry, MODIFIERS)
            )

    return Selection(profile=profile, primary=primary, modifiers=modifiers,
                     candidates=candidates, notes=notes)


def render_decision(selection: Selection) -> str:
    """Human-readable decision card (used by the CLI and the report)."""
    lines = ["# method-selection decision", "", "## profile"]
    for key, value in asdict(selection.profile).items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## primary scheme"]
    if selection.primary is None:
        lines.append("- none (WITHHELD: no evidence-backed rule fired)")
    else:
        d = selection.primary
        worst = d.worst_measured_pct
        lines.append(f"- {d.method}: {d.status}")
        lines.append(f"- why: {d.rationale}")
        if worst is not None:
            lines.append(f"- worst measured error: {worst:.3g}% "
                         f"(owner standard {OWNER_TOLERANCE_PCT:g}%)")
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
    args = parser.parse_args(argv)
    if args.json:
        profile = CaseProfile.from_mapping(json.loads(Path(args.json).read_text()))
    else:
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    return profile, args.registry


def main(argv: Sequence[str] | None = None) -> int:
    profile, registry_path = _profiles_from_cli(sys.argv[1:] if argv is None else argv)
    selection = select_method(profile, registry_path=registry_path)
    sys.stdout.write(render_decision(selection))
    return 0 if selection.is_verified else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
