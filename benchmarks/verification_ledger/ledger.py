"""Cross-family verification ledger for the Laplace pressure jump.

Why this file exists
--------------------
Between 2026-10 and 2026-11 the *same* physical quantity (the pressure jump
across a curved interface, ``dp = sigma/R``) was quoted in this repository as
``-76 %``, ``-107 %``, ``-87.6 %``, ``+5.6 %`` and ``-34 %``.  Those numbers
mostly disagree because of the **measurement convention** (which two pressure
operands are subtracted, and whether the reading is a single time sample or a
time average), not because the solver changed.  Nothing in the repo stopped a
later agent from re-quoting a superseded convention as if it were current
precision -- ``backlog_update_m2971.md`` (P2.1) asked for a machine guard.

This module *is* that guard.  It reads every committed validation artefact
under ``benchmarks/`` that reports a Laplace pressure jump (plus the two
auxiliary quantities of the same verification chain: the half-domain cut
force), and normalises them into one ledger with **one record per reading**.
Design rules:

* numbers are always **read from the artefacts** -- never hand copied, and
  never re-typed into the generated document;
* every record carries the convention it was produced under, and every
  convention carries an explicit ``sanctioned`` flag plus the *reason* why
  it is or is not allowed to be quoted as current precision;
* ``tests/test_verification_ledger.py`` fails if the ledger, the artefacts
  and ``docs/verification_ledger.md`` ever drift apart.

The canonical convention (``backlog_update_m2971.md`` P0/P2.1) is
**absolute pressure + plateau time average**; everything else is registered
as historical, with its measured value preserved.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# Convention registry: single source of truth for "how was dp measured".
#   operand         : which two pressures are subtracted
#   time            : 'instant' | 'time_average' | 'relaxed'
#   free_surface    : does the reading cross a one-sided free surface?
#   sanctioned      : may this convention be quoted as current precision?
# --------------------------------------------------------------------------
CONVENTIONS: Dict[str, Dict[str, Any]] = {
    "abs_tail_avg": dict(
        operand="p_core - 0 (absolute)",
        time="time_average",
        free_surface=True,
        sanctioned=True,
        rationale="m2971 canonical: absolute pressure, plateau-segment time average",
    ),
    "abs_plateau_avg": dict(
        operand="p_plateau - p_ambient (absolute)",
        time="time_average",
        free_surface=True,
        sanctioned=True,
        rationale="m2971 canonical: plateau mean over a time-averaged window",
    ),
    "abs_instant": dict(
        operand="p_core - 0 (absolute)",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="single time sample; the initial spurious pressure has not decayed",
    ),
    "bandA_instant": dict(
        operand="p_core - p(0.80-0.95 R)",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="upper band sits in the surface density-deficit layer: biases dp low",
    ),
    "bandB_instant": dict(
        operand="p_core - p(0.60-0.95 R)",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="wider version of bandA; same density-deficit contamination",
    ),
    "two_phase_abs_instant": dict(
        operand="p_in - p_out (two-phase, smoothed colour)",
        time="instant",
        free_surface=False,
        sanctioned=False,
        rationale="both sides wetted, so no free-surface closure is exercised; still a single instant",
    ),
    "two_phase_sharp_instant": dict(
        operand="p_in - p_out (two-phase, sharp colour step)",
        time="instant",
        free_surface=False,
        sanctioned=False,
        rationale="as two_phase_abs_instant, but with an under-resolved colour step",
    ),
    "legacy_gradient_instant": dict(
        operand="p_in - p_out (legacy gradient colour field)",
        time="instant",
        free_surface=False,
        sanctioned=False,
        rationale="pre-CSF colour gradient; kept only as a regression floor",
    ),
    "one_sided_free_surface": dict(
        operand="p_core - p_ambient (one-sided kernel)",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="one-sided kernel truncation at the free surface",
    ),
    "shell_band_relax": dict(
        operand="p_core - p_shell (dS-weighted shell)",
        time="relaxed",
        free_surface=True,
        sanctioned=False,
        rationale="m277/m2174 convention; the shell band sits inside the interface",
    ),
    "center_minus_band_instant": dict(
        operand="p_center - p_band(interface)",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="band pressure can be exactly 0 (degenerate), so the reading silently becomes absolute",
    ),
    "center_minus_band_relax": dict(
        operand="p_center - p_band(interface)",
        time="relaxed",
        free_surface=True,
        sanctioned=False,
        rationale="relaxed form of center_minus_band_instant; band location drifts with dx",
    ),
    "band_ensemble_instant": dict(
        operand="p_core - p_band, 5 lattice phases",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="ensemble over lattice phase at fixed t = 1 t_sigma; still an instant",
    ),
    "legacy_status_quo": dict(
        operand="legacy static-probe dp",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="pre-CSF status-quo probe, superseded by the consistent scheme",
    ),
    "cut_force_half": dict(
        operand="F_cut / (2 sigma L)",
        time="instant",
        free_surface=True,
        sanctioned=False,
        rationale="half-domain cut force, not a pressure difference (auxiliary quantity)",
    ),
}

QUANTITIES = {
    "laplace_dp": "Laplace pressure jump dp = sigma/R",
    "cut_force_ratio": "half-domain cut force ratio F_cut/(2 sigma L)",
}

# Artefacts that are part of the chain but deliberately out of ledger scope.
OUT_OF_SCOPE = [
    ("benchmarks/oscillating_droplet/validation_report.json",
     "droplet oscillation frequency, not a pressure jump (quantity = omega)"),
    ("benchmarks/dam_break/*.json",
     "free-surface flow validation, no Laplace reference value"),
    ("benchmarks/csf_consistent/validation_report.json:instant",
     "curvature and cut force at t=0 only via a different operator chain; "
     "its relaxed dp entry is in scope, the instant curvature rows are not"),
]


# --------------------------------------------------------------------------
# Record model
# --------------------------------------------------------------------------
@dataclass
class Claim:
    id: str
    family: str
    quantity: str
    source: str
    convention: str
    measured: float
    theory: float
    dx_m: Optional[float] = None
    n_particles: Optional[int] = None
    reported_rel_error_pct: Optional[float] = None
    note: str = ""

    @property
    def convention_meta(self) -> Dict[str, Any]:
        return CONVENTIONS[self.convention]

    @property
    def rel_error_pct(self) -> float:
        return (self.measured - self.theory) / self.theory * 100.0

    @property
    def sanctioned(self) -> bool:
        return bool(self.convention_meta["sanctioned"])

    @property
    def free_surface(self) -> bool:
        return bool(self.convention_meta["free_surface"])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["convention_operand"] = self.convention_meta["operand"]
        d["convention_time"] = self.convention_meta["time"]
        d["sanctioned"] = self.sanctioned
        d["free_surface"] = self.free_surface
        d["rel_error_pct"] = round(self.rel_error_pct, 6)
        d["reported_sign_matches"] = (
            None if self.reported_rel_error_pct is None
            else bool(abs(self.reported_rel_error_pct - self.rel_error_pct)
                      < abs(self.reported_rel_error_pct + self.rel_error_pct))
        )
        d["within_15pct"] = bool(abs(self.rel_error_pct) < 15.0)
        d["within_3pct"] = bool(abs(self.rel_error_pct) < 3.0)
        return d


def _load(root: Path, rel: str) -> Dict[str, Any]:
    with open(root / rel, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _theory_from_reported(measured: float, reported: Optional[float]) -> Optional[float]:
    """Recover dp_theory when an artefact stored only measured + relative error."""
    if reported is None or reported == -100.0:
        return None
    return measured / (1.0 + reported / 100.0)


# --------------------------------------------------------------------------
# Adapters: one per artefact family.  Each yields raw claim dicts.
# --------------------------------------------------------------------------
def _adapter_floor(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/laplace_pressure_floor/report.json"
    rep = _load(root, rel)
    theory = float(rep["reference"]["delta_p_Pa"])
    defs = [
        ("definition_absolute", "abs_instant"),
        ("definition_bandA", "bandA_instant"),
        ("definition_bandB", "bandB_instant"),
        ("absolute_time_average_tail", "abs_tail_avg"),
    ]
    for i, run in enumerate(rep["runs"]):
        for key, conv in defs:
            block = run.get(key) or {}
            measured = _num(block.get("delta_p_Pa"))
            if measured is None:
                continue
            reported = _num(block.get("rel_error_pct"))
            yield dict(
                id=f"floor#run{i}#{key}",
                family="laplace_pressure_floor",
                quantity="laplace_dp",
                source=rel,
                convention=conv,
                measured=measured,
                theory=theory,
                dx_m=_num(run.get("dx_m")),
                n_particles=run.get("n_particles"),
                reported_rel_error_pct=reported,
                note=f"c0={_num(run.get('sound_speed'))}",
            )


def _adapter_plateau(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/laplace_plateau/validation_report.json"
    rep = _load(root, rel)
    ref = rep.get("reference") or {}
    theory = _num(ref.get("delta_p_Pa")) or _num(ref.get("laplace_dp_Pa"))
    for i, row in enumerate(rep.get("ladder", [])):
        lap = row.get("laplace") or {}
        measured = _num(lap.get("delta_p_Pa"))
        reported = _num(lap.get("rel_error_pct"))
        if measured is None:
            continue
        t = theory if theory is not None else _theory_from_reported(measured, reported)
        yield dict(
            id=f"plateau#ladder{i}",
            family="laplace_plateau",
            quantity="laplace_dp",
            source=rel,
            convention="abs_plateau_avg",
            measured=measured,
            theory=t,
            dx_m=_num(row.get("dx_m")),
            n_particles=row.get("n_particles"),
            reported_rel_error_pct=reported,
            note="absolute pressure, plateau time average",
        )


def _adapter_surface_tension_ladder(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/surface_tension/laplace_ladder.json"
    rep = _load(root, rel)
    for i, lvl in enumerate(rep.get("levels", [])):
        measured = _num(lvl.get("jump_measured_pa"))
        theory = _num(lvl.get("jump_theory_pa"))
        reported = _num(lvl.get("jump_rel_error"))
        if measured is None or theory is None:
            continue
        yield dict(
            id=f"ladder#level{i}",
            family="surface_tension_ladder",
            quantity="laplace_dp",
            source=rel,
            convention="legacy_status_quo",
            measured=measured,
            theory=theory,
            dx_m=_num(lvl.get("dx_m")),
            n_particles=lvl.get("n_particles"),
            reported_rel_error_pct=(reported * 100.0) if reported is not None else None,
            note="legacy lattice ladder (m2174)",
        )


def _adapter_surface_tension_laplace(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/surface_tension_laplace/validation_report.json"
    rep = _load(root, rel)
    theory = _num((rep.get("theory") or {}).get("laplace_dp_Pa"))
    groups = [
        ("two_phase_sweep_smoothed_color", "two_phase_abs_instant"),
        ("two_phase_sweep_sharp_step_color", "two_phase_sharp_instant"),
        ("one_sided_free_surface", "one_sided_free_surface"),
    ]
    for key, conv in groups:
        block = rep.get(key) or {}
        rows = block.get("cases") if isinstance(block, dict) else None
        if rows is None and isinstance(block, list):
            rows = block
        for i, row in enumerate(rows or []):
            measured = _num(row.get("dp_Pa"))
            reported = _num(row.get("rel_error"))
            if measured is None or theory is None:
                continue
            yield dict(
                id=f"stlaplace#{key}#{i}",
                family="surface_tension_laplace",
                quantity="laplace_dp",
                source=rel,
                convention=conv,
                measured=measured,
                theory=theory,
                dx_m=_num(row.get("dx_m")),
                n_particles=row.get("n_particles"),
                reported_rel_error_pct=(reported * 100.0) if reported is not None else None,
                note=key,
            )
    legacy = (rep.get("legacy_gradient_comparison_at_dx_0.005") or {})
    if _num(legacy.get("dp_Pa")) is not None and theory is not None:
        rep_err = _num(legacy.get("rel_error"))
        yield dict(
            id="stlaplace#legacy_gradient#0",
            family="surface_tension_laplace",
            quantity="laplace_dp",
            source=rel,
            convention="legacy_gradient_instant",
            measured=_num(legacy["dp_Pa"]),
            theory=theory,
            dx_m=0.005,
            n_particles=legacy.get("n_particles"),
            reported_rel_error_pct=(rep_err * 100.0) if rep_err is not None else None,
            note="pre-CSF colour field, regression floor",
        )


def _adapter_static_droplet(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/static_droplet/laplace_validation.json"
    rep = _load(root, rel)
    for i, row in enumerate(rep.get("rows", [])):
        measured = _num(row.get("dp_measured_Pa"))
        theory = _num(row.get("dp_theory_Pa"))
        reported = _num(row.get("dp_rel_error"))
        if measured is None or theory is None:
            continue
        yield dict(
            id=f"static#row{i}",
            family="static_droplet",
            quantity="laplace_dp",
            source=rel,
            convention="shell_band_relax",
            measured=measured,
            theory=theory,
            dx_m=_num(row.get("dx_m")),
            n_particles=row.get("n_particles"),
            reported_rel_error_pct=(reported * 100.0) if reported is not None else None,
            note=f"R={_num(row.get('radius_m'))}",
        )


def _adapter_laplace_law(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/laplace_law/validation_report.json"
    rep = _load(root, rel)
    lap = rep.get("laplace") or {}
    measured = _num(lap.get("delta_p_measured_Pa"))
    theory = _num(lap.get("delta_p_theory_Pa"))
    reported = _num(lap.get("rel_error_pct"))
    if measured is not None and theory is not None:
        yield dict(
            id="laplace_law#relaxed#0",
            family="laplace_law",
            quantity="laplace_dp",
            source=rel,
            convention="center_minus_band_instant",
            measured=measured,
            theory=theory,
            dx_m=_num((rep.get("setup") or {}).get("dx_m")),
            n_particles=(rep.get("setup") or {}).get("n_particles"),
            reported_rel_error_pct=reported,
            note="p_interface_band is 0.0 here, so the reading degenerates to absolute",
        )


def _adapter_csf_consistent(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/csf_consistent/validation_report.json"
    rep = _load(root, rel)
    lap = (rep.get("relax") or {}).get("laplace") or {}
    measured = _num(lap.get("delta_p_measured_Pa"))
    reported = _num(lap.get("rel_error_pct"))
    if measured is None:
        return
    theory = _num((rep.get("reference") or {}).get("delta_p_Pa")) or _theory_from_reported(
        measured, reported
    )
    yield dict(
        id="csf_consistent#relaxed#0",
        family="csf_consistent",
        quantity="laplace_dp",
        source=rel,
        convention="center_minus_band_relax",
        measured=measured,
        theory=theory,
        dx_m=_num(((rep.get("relax") or {}).get("setup") or {}).get("dx_m")),
        n_particles=None,
        reported_rel_error_pct=reported,
        note="consistent scheme, relaxed state",
    )


def _adapter_csf_curvature_fit(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/csf_curvature_fit/validation_report.json"
    rep = _load(root, rel)
    for i, row in enumerate(rep.get("relax", [])):
        lap = row.get("laplace") or {}
        measured = _num(lap.get("delta_p_measured_Pa"))
        reported = _num(lap.get("rel_error_pct"))
        if measured is None:
            continue
        theory = _num((row.get("reference") or {}).get("delta_p_Pa")) or _theory_from_reported(
            measured, reported
        )
        yield dict(
            id=f"csf_curv_fit#relaxed#{i}",
            family="csf_curvature_fit",
            quantity="laplace_dp",
            source=rel,
            convention="center_minus_band_relax",
            measured=measured,
            theory=theory,
            dx_m=_num((row.get("setup") or {}).get("dx_m")),
            n_particles=None,
            reported_rel_error_pct=reported,
            note=f"scheme={row.get('csf_scheme')}, curvature={row.get('curvature')}",
        )


def _adapter_paper(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/paper_laplace_uncertainty/out/results.json"
    rep = _load(root, rel)
    for i, row in enumerate(rep.get("rows", [])):
        measured = _num(row.get("delta_p_measured_mean_Pa"))
        theory = _num(row.get("delta_p_theory_Pa"))
        reported = _num(row.get("rel_error_pct"))
        if measured is None or theory is None:
            continue
        ci = _num(row.get("rel_error_ci95_pct"))
        yield dict(
            id=f"paper#row{i}",
            family="paper_laplace_uncertainty",
            quantity="laplace_dp",
            source=rel,
            convention="band_ensemble_instant",
            measured=measured,
            theory=theory,
            dx_m=_num(row.get("dx_m")),
            n_particles=row.get("n_particles"),
            reported_rel_error_pct=reported,
            note=f"5-phase ensemble, ci95={ci:.2f} pp" if ci is not None else "5-phase ensemble",
        )


def _adapter_legacy_laplace(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/laplace/report.json"
    rep = _load(root, rel)
    setup = rep.get("setup") or {}
    radius = _num(setup.get("radius_m"))
    for key, block in (rep.get("checks") or {}).items():
        if not key.endswith(":laplace_pressure"):
            continue
        match = re.search(r"R/dx=(\d+(?:\.\d+)?)", key)
        ratio = float(match.group(1)) if match else None
        lap = block.get("laplace_pressure") or block
        measured = _num(lap.get("measured"))
        theory = _num(lap.get("theory"))
        if measured is None or theory is None:
            continue
        yield dict(
            id=f"legacy#{key}",
            family="laplace_status_quo",
            quantity="laplace_dp",
            source=rel,
            convention="legacy_status_quo",
            measured=measured,
            theory=theory,
            dx_m=(radius / ratio if radius and ratio else None),
            n_particles=None,
            reported_rel_error_pct=None,
            note=key,
        )


def _adapter_cut_force(root: Path) -> Iterable[Dict[str, Any]]:
    rel = "benchmarks/laplace_pressure/validation_report.json"
    rep = _load(root, rel)
    for i, row in enumerate(rep.get("cases", [])):
        ratio = _num(row.get("sigma_eff_over_sigma"))
        if ratio is None:
            continue
        yield dict(
            id=f"cut_force#case{i}",
            family="laplace_cut_force",
            quantity="cut_force_ratio",
            source=rel,
            convention="cut_force_half",
            measured=ratio,
            theory=1.0,
            dx_m=_num(row.get("dx_m")),
            n_particles=row.get("particles"),
            reported_rel_error_pct=None,
            note=f"R={_num(row.get('radius_m'))}",
        )


ADAPTERS: List[Tuple[str, Callable[[Path], Iterable[Dict[str, Any]]]]] = [
    ("laplace_pressure_floor", _adapter_floor),
    ("laplace_plateau", _adapter_plateau),
    ("surface_tension_ladder", _adapter_surface_tension_ladder),
    ("surface_tension_laplace", _adapter_surface_tension_laplace),
    ("static_droplet", _adapter_static_droplet),
    ("laplace_law", _adapter_laplace_law),
    ("csf_consistent", _adapter_csf_consistent),
    ("csf_curvature_fit", _adapter_csf_curvature_fit),
    ("paper_laplace_uncertainty", _adapter_paper),
    ("laplace_status_quo", _adapter_legacy_laplace),
    ("laplace_cut_force", _adapter_cut_force),
]


def collect(root: Path) -> List[Claim]:
    """Read every registered artefact and return the normalised ledger."""
    claims: List[Claim] = []
    seen = set()
    for family, adapter in ADAPTERS:
        for raw in adapter(root):
            claim = Claim(**raw)
            if claim.convention not in CONVENTIONS:
                raise KeyError(f"{claim.id}: unknown convention {claim.convention!r}")
            if claim.id in seen:
                raise ValueError(f"duplicate claim id {claim.id}")
            seen.add(claim.id)
            claims.append(claim)
    claims.sort(key=lambda c: (c.quantity, c.family, c.id))
    return claims


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
CASE_KEYS = {
    "laplace_pressure_floor": ("dx_m", "note"),  # note carries c0
}


def _case_key(claim: Claim) -> Tuple[Any, ...]:
    keys = CASE_KEYS.get(claim.family, ("dx_m", "note"))
    return (claim.family,) + tuple(getattr(claim, k) for k in keys)


def convention_sensitivity(claims: List[Claim]) -> List[Dict[str, Any]]:
    """For each physical case measured under >=2 conventions, the spread in pp."""
    groups: Dict[Tuple[Any, ...], List[Claim]] = {}
    for claim in claims:
        if claim.quantity != "laplace_dp":
            continue
        groups.setdefault(_case_key(claim), []).append(claim)
    rows = []
    for key, bucket in groups.items():
        convs = sorted({c.convention for c in bucket})
        if len(convs) < 2:
            continue
        errs = [c.rel_error_pct for c in bucket]
        rows.append(
            dict(
                case=" / ".join(str(k) for k in key if k is not None),
                conventions=convs,
                n_readings=len(bucket),
                min_pct=round(min(errs), 3),
                max_pct=round(max(errs), 3),
                spread_pp=round(max(errs) - min(errs), 3),
            )
        )
    rows.sort(key=lambda r: -r["spread_pp"])
    return rows


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    pos = (len(vs) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (pos - lo)


def summarise(claims: List[Claim]) -> Dict[str, Any]:
    """Aggregate every reading into the verdict block of the ledger."""
    out: Dict[str, Any] = {"quantities": {}}
    for quantity, label in QUANTITIES.items():
        subset = [c for c in claims if c.quantity == quantity]
        if not subset:
            continue
        errs = [c.rel_error_pct for c in subset]
        sanctioned = [c for c in subset if c.sanctioned]
        s_errs = [c.rel_error_pct for c in sanctioned]
        fs = [c for c in subset if c.free_surface]
        nofs = [c for c in subset if not c.free_surface]
        out["quantities"][quantity] = dict(
            label=label,
            n_readings=len(subset),
            n_families=len({c.family for c in subset}),
            families=sorted({c.family for c in subset}),
            n_within_15pct=sum(1 for e in errs if abs(e) < 15.0),
            n_within_3pct=sum(1 for e in errs if abs(e) < 3.0),
            median_pct=round(_percentile(errs, 0.5), 3),
            worst_abs_pct=round(max(abs(e) for e in errs), 3),
            n_sanctioned=len(sanctioned),
            sanctioned_min_abs_pct=round(min(abs(e) for e in s_errs), 3) if s_errs else None,
            sanctioned_median_pct=round(_percentile(s_errs, 0.5), 3) if s_errs else None,
            sanctioned_max_abs_pct=round(max(abs(e) for e in s_errs), 3) if s_errs else None,
            sanctioned_all_within_15pct=all(abs(e) < 15.0 for e in s_errs) if s_errs else None,
            n_free_surface=len(fs),
            free_surface_within_15pct=sum(
                1 for c in fs if abs(c.rel_error_pct) < 15.0
            ),
            free_surface_min_abs_pct=round(min(abs(c.rel_error_pct) for c in fs), 3) if fs else None,
            free_surface_all_fail_15pct=all(abs(c.rel_error_pct) >= 15.0 for c in fs) if fs else None,
            n_no_free_surface=len(nofs),
            no_free_surface_min_abs_pct=(
                round(min(abs(c.rel_error_pct) for c in nofs), 3) if nofs else None
            ),
            no_free_surface_within_15pct=sum(
                1 for c in nofs if abs(c.rel_error_pct) < 15.0
            ),
        )
    out["n_reported_as_magnitude"] = sum(
        1 for c in claims
        if c.reported_rel_error_pct is not None
        and c.reported_rel_error_pct > 0.0 > c.rel_error_pct
    )
    out["convention_sensitivity"] = convention_sensitivity(claims)
    best = [c for c in claims if c.quantity == "laplace_dp" and c.sanctioned]
    if best:
        winner = min(best, key=lambda c: abs(c.rel_error_pct))
        out["best_sanctioned_reading"] = dict(
            id=winner.id,
            family=winner.family,
            convention=winner.convention,
            measured_Pa=round(winner.measured, 3),
            theory_Pa=round(winner.theory, 3),
            rel_error_pct=round(winner.rel_error_pct, 3),
        )
    return out


CONTROL_PATH = "benchmarks/verification_ledger/provenance_control.json"


def load_control(root: Path) -> Optional[Dict[str, Any]]:
    path = root / CONTROL_PATH
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def provenance_findings(claims: List[Claim], control: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn the three control questions into machine-derived statements.

    Only the *derived* statements are stored here; the raw readings stay in
    ``provenance_control.json`` so the document and the tests never disagree.
    """
    if not control:
        return {}
    cases = control.get("cases") or {}
    out: Dict[str, Any] = {"control_path": CONTROL_PATH, "cases": cases}

    committed = [c for c in claims if c.id == "laplace_law#relaxed#0"]
    recomputed = cases.get("sigma7_rdx5") or {}
    if committed and recomputed:
        out["q1_reproducible"] = dict(
            committed_rel_error_pct=round(committed[0].reported_rel_error_pct or 0.0, 4),
            recomputed_rel_error_pct=recomputed["rel_error_pct"],
            delta_pp=round(abs((committed[0].reported_rel_error_pct or 0.0)
                               - recomputed["rel_error_pct"]), 4),
            committed_measured_Pa=round(committed[0].measured, 4),
            recomputed_measured_Pa=recomputed["dp_measured_Pa"],
            reproducible=False,
        )

    s0 = cases.get("sigma0_rdx5") or {}
    s7 = cases.get("sigma7_rdx5") or {}
    if s0 and s7:
        trace0 = s0.get("p_core_trace_Pa") or []
        out["q2_sigma_control"] = dict(
            sigma_reference_N_per_m=float(control.get("sigma_reference_N_per_m", 7.2)),
            sigma_control_N_per_m=s0["sigma_N_per_m"],
            dp_sigma_reference_Pa=s7["dp_measured_Pa"],
            dp_sigma_control_Pa=s0["dp_measured_Pa"],
            p_core_t_start_Pa=trace0[0] if trace0 else None,
            p_core_t_after_50_steps_Pa=trace0[1] if len(trace0) > 1 else None,
            generated_by_surface_tension=bool(s7["dp_measured_Pa"] != 0.0
                                              and s0["dp_measured_Pa"] == 0.0),
        )

    r5 = cases.get("sigma7_rdx5") or {}
    r10 = cases.get("sigma7_rdx10") or {}
    if r5 and r10:
        out["q3_resolution_stability"] = dict(
            coarse_R_over_dx=r5["R_over_dx"],
            coarse_rel_error_pct=r5["rel_error_pct"],
            coarse_p_band_Pa=r5["p_band_Pa"],
            coarse_within_15pct=r5["within_15pct"],
            fine_R_over_dx=r10["R_over_dx"],
            fine_rel_error_pct=r10["rel_error_pct"],
            fine_p_band_Pa=r10["p_band_Pa"],
            fine_within_15pct=r10["within_15pct"],
            resolution_stable=bool(abs(r10["rel_error_pct"] - r5["rel_error_pct"]) < 15.0),
            band_clamped_only_coarse=bool(r5["p_band_Pa"] == 0.0 and r10["p_band_Pa"] != 0.0),
        )
    return out


def free_surface_passes(claims: List[Claim], control: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every free-surface reading that clears 15 %, plus its resolution verdict.

    A single reading below the gate is *not* evidence of a working free-surface
    closure unless it is also resolution-stable.  For each such reading we look
    for a refinement ladder inside its own family and record whether the error
    is monotone decreasing (would be credible) or has an interior minimum
    (a zero crossing of a diverging curve -- not credible).
    """
    out: List[Dict[str, Any]] = []
    for claim in claims:
        if claim.quantity != "laplace_dp" or not claim.free_surface:
            continue
        if abs(claim.rel_error_pct) >= 15.0:
            continue
        siblings = [
            c for c in claims
            if c.family == claim.family and c.quantity == "laplace_dp"
            and c.dx_m is not None and c.convention == claim.convention
        ]
        siblings.sort(key=lambda c: c.dx_m)
        ladder = [
            dict(dx_m=c.dx_m, rel_error_pct=round(c.rel_error_pct, 3))
            for c in siblings
        ]
        verdict = "no_ladder"
        if len(ladder) >= 3:
            errs = [abs(row["rel_error_pct"]) for row in ladder]
            i_min = errs.index(min(errs))
            if 0 < i_min < len(errs) - 1:
                verdict = "zero_crossing"
            elif i_min == len(errs) - 1:
                verdict = "monotone_converging"
            else:
                verdict = "monotone_diverging"
        entry = dict(
            id=claim.id,
            family=claim.family,
            convention=claim.convention,
            rel_error_pct=round(claim.rel_error_pct, 3),
            dx_m=claim.dx_m,
            source=claim.source,
            ladder=ladder,
            resolution_verdict=verdict,
            resolution_stable=(verdict == "monotone_converging"),
        )
        out.append(entry)
    return out


def build(root: Path) -> Dict[str, Any]:
    claims = collect(root)
    sources = sorted({c.source for c in claims})
    control = load_control(root)
    return dict(
        benchmark="verification_ledger",
        purpose=(
            "single machine-checked register of every Laplace-jump reading committed "
            "in benchmarks/, tagged by measurement convention"
        ),
        generated_by="benchmarks/verification_ledger/build.py",
        note="numbers are extracted from the listed artefacts at build time; never hand copied",
        n_readings=len(claims),
        n_families=len({c.family for c in claims}),
        sources=sources,
        conventions={
            k: {kk: vv for kk, vv in v.items()} for k, v in CONVENTIONS.items()
        },
        out_of_scope=[dict(artefact=a, reason=r) for a, r in OUT_OF_SCOPE],
        readings=[c.to_dict() for c in claims],
        verdict=summarise(claims),
        provenance=provenance_findings(claims, control),
        free_surface_passes=free_surface_passes(claims, control),
    )


__all__ = [
    "CONVENTIONS",
    "QUANTITIES",
    "OUT_OF_SCOPE",
    "Claim",
    "collect",
    "free_surface_passes",
    "load_control",
    "provenance_findings",
    "summarise",
    "convention_sensitivity",
    "build",
]
