#!/usr/bin/env python3
"""TensorSPH verification claim ledger — generator.

Why this exists
---------------
TensorSPH now carries ~15 machine-readable verification artifacts
(``benchmarks/**/validation_report.json`` and friends).  Each was written by a
different mission, with its own definition of the measured quantity, its own
tolerance and its own status vocabulary (``pass`` / ``FAIL`` / ``WITHHELD`` /
``WITHDRAWN``).  The same physical quantity (the Young-Laplace jump at
sigma = 7.2 N/m, R = 20 mm) therefore appears five times with numbers between
-87.6 % and +5.6 %, and nothing in the repository states which reading is
current, which is superseded, and which one is disqualified by its own gauge.

This generator reconciles them into a single table:

  * one entry per *claim* (artifact + JSON path + reading),
  * the reference value, the measured value, the signed relative error,
  * a grade derived by one repo-wide rule (3 % / 15 % ladder, see ``grade``),
  * **findings**: machine-detectable defects in the evidence itself
    (degenerate gauge, self-declared PASS above tolerance, several readings of
    one quantity, unregistered artifacts, stale numbers quoted in prose).

Nothing is hand typed: every number is read from the artifact at generation
time, and ``tests/test_verification_ledger.py`` regenerates the ledger and
fails when the committed copy drifts.

Usage
-----
    python benchmarks/verification_ledger/build_ledger.py            # write
    python benchmarks/verification_ledger/build_ledger.py --figure   # + png
    python benchmarks/verification_ledger/build_ledger.py --check    # diff only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "benchmarks" / "verification_ledger"
LEDGER_JSON = OUT_DIR / "ledger.json"
WAIVERS = OUT_DIR / "waivers.json"
FIGURE = OUT_DIR / "figures" / "claim_error_landscape.png"
DOC_MD = REPO / "docs" / "verification_ledger.md"

#: Platform acceptance ladder: > 15 % is not acceptable, <= 3 % is paper grade.
CRITERION_PCT = 15.0
PAPER_PCT = 3.0

GRADES = ("PASS", "MARGINAL", "WITHHELD", "FAIL", "WITHDRAWN")

#: Sub-directories of a benchmark hold *fragments* (per-case dumps, resolution
#: ladders, raw traces): the parent report carries the registered claim, so the
#: fragments inherit its provenance.
FRAGMENT_DIRS = ("cases", "raw")
#: A file inside ``out/`` is a fragment too, unless it carries a primary report
#: name -- some benchmarks commit *only* ``out/results.json``, and that file is
#: their evidence, not a fragment.
PRIMARY_NAMES = re.compile(r"(results|report|summary|validation_report)\.json$")
LEDGER_DIR = "benchmarks/verification_ledger"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load(rel: str) -> dict:
    path = REPO / rel
    if not path.is_file():
        raise FileNotFoundError(f"ledger artifact missing: {rel}")
    with path.open() as fh:
        return json.load(fh)


def _get(obj, dotted: str):
    """``a.b[0].c`` style lookup; raises on schema drift (on purpose)."""
    cur = obj
    for token in dotted.split("."):
        name, *idx = token.split("[")
        if name:
            cur = cur[name]
        for i in idx:
            cur = cur[int(i.rstrip("]"))]
    return cur


def _r(x, nd=6):
    return None if x is None else round(float(x), nd)


def _rel(measured: float, reference: float) -> float:
    return (float(measured) - float(reference)) / float(reference) * 100.0


def _claim_validity():
    """Load the claim-validity gate (``benchmarks/claim_validity/validity.py``).

    The gate decides whether a reading was taken from a live simulation before any
    tolerance is applied; see ``docs/verification_claim_validity.md``.
    """
    import importlib.util

    path = REPO / "benchmarks" / "claim_validity" / "validity.py"
    spec = importlib.util.spec_from_file_location("claim_validity_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(
    *,
    claim_id: str,
    quantity_id: str,
    quantity: str,
    artifact: str,
    json_path: str,
    reference: tuple[float, str, str],
    measured: tuple[float, str, str],
    guarding_test: str,
    artifact_status: str | None,
    backlog_ref: str,
    notes: str = "",
    rel_error_pct: float | None = None,
    gauge_degenerate: bool = False,
    superseded_by: str | None = None,
    claim_type: str = "relative_error",
    gate_result: str | None = None,
    gate_limit: float | None = None,
    run_validity: str = "UNPROBED",
    admissibility: dict | None = None,
) -> dict:
    ref_v, ref_u, ref_src = reference
    mea_v, mea_u, mea_def = measured
    if rel_error_pct is None:
        rel_error_pct = _rel(mea_v, ref_v)
    return {
        "claim_id": claim_id,
        "quantity_id": quantity_id,
        "quantity": quantity,
        "claim_type": claim_type,
        "artifact": artifact,
        "json_path": json_path,
        "reference": {"value": _r(ref_v), "unit": ref_u, "source": ref_src},
        "measured": {"value": _r(mea_v), "unit": mea_u, "definition": mea_def},
        "rel_error_pct": _r(rel_error_pct),
        "criterion_pct": CRITERION_PCT,
        "paper_criterion_pct": PAPER_PCT,
        "gauge_degenerate": bool(gauge_degenerate),
        "run_validity": run_validity,
        "artifact_status": artifact_status,
        "gate_result": gate_result,
        "gate_limit": _r(gate_limit) if gate_limit is not None else None,
        "admissibility": admissibility,
        "superseded_by": superseded_by,
        "guarding_test": guarding_test,
        "backlog_ref": backlog_ref,
        "notes": notes,
    }



def resolve(obj, path: str):
    """Resolve ``a.b[0].c`` and also tolerate keys that contain dots themselves
    (e.g. ``criteria.error_at_dx_0.0025_lt_15pct``)."""
    toks = path.split(".")

    def rec(cur, toks):
        if not toks:
            return cur
        for k in range(len(toks), 0, -1):
            key = ".".join(toks[:k])
            name, *idx = key.split("[")
            target, ok = cur, True
            if name:
                if isinstance(target, dict) and name in target:
                    target = target[name]
                else:
                    ok = False
            if ok:
                try:
                    for i in idx:
                        target = target[int(i.rstrip("]"))]
                except (KeyError, IndexError, TypeError, ValueError):
                    ok = False
            if ok:
                return rec(target, toks[k:])
        raise KeyError(path)

    return rec(obj, toks)


# ---------------------------------------------------------------------------
# claim registry
# ---------------------------------------------------------------------------
def young_laplace_claims() -> list[dict]:
    """Young-Laplace jump, sigma = 7.2 N/m, R = 20 mm -> sigma/R = 360 Pa."""
    qid = "young_laplace_jump_sigma7.2_R0.02"
    qty = "dp = sigma/R (2-D), sigma=7.2 N/m, R=0.020 m -> 360 Pa"
    out: list[dict] = []

    # 1/5 — pressure floor: best absolute, time-averaged tail reading
    rel = "benchmarks/laplace_pressure_floor/report.json"
    floor = _load(rel)
    runs = floor["runs"]
    tails = [(r["absolute_time_average_tail"]["rel_error_pct"], i) for i, r in enumerate(runs)]
    best_err, best_i = max(tails)  # all readings are low, closest to zero wins
    best = runs[best_i]
    out.append(
        _entry(
            claim_id="young_laplace.absolute_tail.best_of_6",
            quantity_id=qid,
            quantity=qty,
            artifact=rel,
            json_path=f"runs[{best_i}].absolute_time_average_tail",
            reference=(floor["reference"]["delta_p_Pa"], "Pa", "Young-Laplace sigma/R"),
            measured=(
                best["absolute_time_average_tail"]["delta_p_Pa"],
                "Pa",
                f"p_core - p_ambient (absolute pressure, no vacuum gauge), tail time "
                f"average; dx={best['dx_m']} m, c0={best['sound_speed']} m/s",
            ),
            rel_error_pct=best_err,
            guarding_test="tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct",
            artifact_status="WITHHELD",
            backlog_ref="backlog_update_m2971.md#P0",
            notes=(
                "Best of 6 runs (c0 20/5 x dx 0.002/0.0015 plus controls). Reproduces "
                "backlog P0: the residual is the free-surface pressure closure, not the "
                "curvature operator."
            ),
        )
    )

    # 2/5 — consistent CSF scheme, relaxed, dx = 2 mm
    rel = "benchmarks/csf_consistent/validation_report.json"
    d = _load(rel)
    lap = _get(d, "relax.laplace")
    out.append(
        _entry(
            claim_id="young_laplace.csf_consistent.dx2mm",
            quantity_id=qid,
            quantity=qty,
            artifact=rel,
            json_path="relax.laplace",
            reference=(_get(d, "relax.reference.delta_p_Pa"), "Pa", "Young-Laplace sigma/R"),
            measured=(
                lap["delta_p_measured_Pa"],
                "Pa",
                "p_center - p_interface_band (two-gauge readout, band at 0.80-0.95 R)",
            ),
            rel_error_pct=lap["rel_error_pct"],
            guarding_test="tests/test_csf_consistent.py",
            artifact_status="PASS" if lap["within_15pct"] else "WITHHELD",
            backlog_ref="backlog_update_m2971.md#P0",
            notes=(
                "This artifact's historic value (-75.6 %) is the figure quoted in "
                "docs/verification_surface_tension_consistent.md; it has been regenerated "
                "since (see SUPERSEDED_NUMBERS in the generator)."
            ),
        )
    )

    # 3/5 — curvature-fit family, MLS operator, dx = 2 mm
    rel = "benchmarks/csf_curvature_fit/validation_report.json"
    d = _load(rel)
    lap = _get(d, "relax[0].laplace")
    out.append(
        _entry(
            claim_id="young_laplace.csf_curvature_fit.dx2mm",
            quantity_id=qid,
            quantity=qty,
            artifact=rel,
            json_path="relax[0].laplace",
            reference=(_get(d, "relax[0].reference.delta_p_Pa"), "Pa", "Young-Laplace sigma/R"),
            measured=(
                lap["delta_p_measured_Pa"],
                "Pa",
                "p_center - p_interface_band, MLS curvature operator",
            ),
            rel_error_pct=lap["rel_error_pct"],
            guarding_test="tests/test_csf_curvature_fit.py",
            artifact_status="PASS" if lap["within_15pct"] else "WITHHELD",
            backlog_ref="backlog_update_m2971.md#P0",
            notes="Same pressure-closure deficit through a different curvature operator.",
        )
    )

    # 4/5 — laplace_law: declares PASS using a vacuum ambient gauge
    rel = "benchmarks/laplace_law/validation_report.json"
    d = _load(rel)
    lap = _get(d, "laplace")
    band = lap["p_interface_band_Pa"]
    out.append(
        _entry(
            claim_id="young_laplace.laplace_law.vacuum_gauge",
            quantity_id=qid,
            quantity=qty,
            artifact=rel,
            json_path="laplace",
            reference=(lap["delta_p_theory_Pa"], "Pa", "Young-Laplace sigma/R"),
            measured=(
                lap["delta_p_measured_Pa"],
                "Pa",
                f"p_center - p_interface_band with p_interface_band = {band} Pa "
                "(i.e. p_center - vacuum, not p_core - p_ambient)",
            ),
            rel_error_pct=lap["rel_error_pct"],
            gauge_degenerate=abs(band) == 0.0,
            guarding_test="tests/test_laplace_law.py",
            artifact_status="PASS" if lap["within_15pct"] else "WITHHELD",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Declares within_15pct=True, but the readout subtracts a zero ambient "
                "pressure: the core pressure of a wcSPH droplet carries a ~rho c0^2 "
                "additive offset, so the +5.6 % is not comparable with the -37.9 % "
                "absolute reading above and must not be cited as a pass."
            ),
        )
    )

    # 5/5 — plateau definition, worst of the resolution ladder
    rel = "benchmarks/laplace_plateau/validation_report.json"
    d = _load(rel)
    vd = d["verdict"]
    static = d["static"][0]
    ref_pa = d["reference"]["delta_p_Pa"]
    out.append(
        _entry(
            claim_id="young_laplace.plateau_definition",
            quantity_id=qid,
            quantity=qty,
            artifact=rel,
            json_path="verdict",
            reference=(ref_pa, "Pa", "Young-Laplace sigma/R"),
            measured=(
                ref_pa * (1.0 + vd["worst_rel_error_pct"] / 100.0),
                "Pa",
                "p_plateau(r<0.6R) - p_ambient(r>1.0R), worst of the ladder",
            ),
            rel_error_pct=vd["worst_rel_error_pct"],
            guarding_test="tests/test_laplace_plateau.py",
            artifact_status=vd["evidence_grade"],
            backlog_ref="backlog_update_m2971.md#P0",
            notes=(
                "Plateau settles at ~0.55 sigma/R at every resolution -> systematic, not "
                "resolution limited. Kernel-sum bias of the static lattice: "
                f"{static['kernel_bias_pct']:.3f} % (P1 in backlog_update_m2971.md)."
            ),
        )
    )
    return out


def force_path_claims() -> list[dict]:
    """CSF force magnitude — bypasses the pressure field entirely."""
    rel = "benchmarks/laplace_pressure/validation_report.json"
    d = _load(rel)
    case = d["cases"][0]
    return [
        _entry(
            claim_id="csf_force.sigma_eff_over_sigma.dx5mm",
            quantity_id="csf_force_sigma_eff",
            quantity=(
                "effective surface tension from the half-droplet force integral, "
                "sigma_eff/sigma"
            ),
            artifact=rel,
            json_path="cases[0]",
            reference=(1.0, "-", "sigma_eff = sigma (force integral / 2 sigma)"),
            measured=(
                case["sigma_eff_over_sigma"],
                "-",
                f"|F_half| / (2 sigma) at R/h = {case['radius_over_h']:.2f}",
            ),
            rel_error_pct=_rel(case["sigma_eff_over_sigma"], 1.0),
            guarding_test="tests/test_laplace_pressure.py",
            artifact_status="FAIL",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Force path is 30 % low while being mirror/rotation symmetric to 1e-14 "
                "(gates pass): the deficit lives in the colour-surface delta, not in the "
                "discrete operators."
            ),
        )
    ]


def operator_level_claims() -> list[dict]:
    """Claims that isolate the discrete operator from the free surface.

    These are the ones that tell *where* the Laplace deficit lives: on a closed
    two-phase interface (periodic, no free surface) the same consistent-CSF
    chain reproduces sigma/R to 4-5 %; the one-sided free-surface cases above
    lose 34-88 %.  The colour-surface normalisation (integral |grad c| dr) is
    ~0.74-0.75 instead of 1.0, which is the missing 25 % of the force.
    """
    out: list[dict] = []

    rel = "benchmarks/surface_tension_laplace/validation_report.json"
    d = _load(rel)
    key = "two_phase_laplace_error_at_dx_0.0025_lt_15pct"
    c = d["criteria"][key]
    out.append(
        _entry(
            claim_id="two_phase_laplace.smoothed_colour.dx2.5mm",
            quantity_id="two_phase_laplace_error",
            quantity="Laplace error on a closed two-phase interface (periodic, no free surface)",
            artifact=rel,
            json_path="criteria.two_phase_laplace_error_at_dx_0.0025_lt_15pct",
            reference=(0.0, "fraction", "exact Laplace balance on the closed interface"),
            measured=(
                c["measured"],
                "fraction",
                "|dp_measured/dp_theory - 1| at dx = 2.5 mm, smoothed colour field",
            ),
            rel_error_pct=c["measured"] * 100.0,
            guarding_test="tests/test_surface_tension_laplace.py",
            artifact_status=d["validation"]["status"],
            backlog_ref="backlog_update_m2971.md#P0",
            claim_type="gate",
            gate_result=c["result"],
            gate_limit=c["limit"],
            notes=(
                "Decisive control: with a resolved closed interface the very same CSF "
                "chain is 4-5 % accurate at both dx = 2.5 and 5 mm and the error shrinks "
                "under refinement, so the operator is not the defect. The one-sided "
                "free-surface readings above are."
            ),
        )
    )

    rel = "benchmarks/csf_consistent/curvature_audit.json"
    d = _load(rel)
    row = d["rows"][0]
    var = row["variants"][0]
    out.append(
        _entry(
            claim_id="curvature_operator.delta_s_weighted.dx3mm",
            quantity_id="csf_curvature_error",
            quantity="delta_s-weighted curvature kappa vs analytic 1/R (R = 20 mm)",
            artifact=rel,
            json_path="rows[0].variants[0]",
            reference=(
                1.0 / row["radius_m"] if "radius_m" in row else 50.0,
                "1/m",
                "analytic curvature of a circle",
            ),
            measured=(
                var["kappa_delta_s_weighted_1_per_m"],
                "1/m",
                f"delta_s-weighted kappa, {var['label']}",
            ),
            rel_error_pct=var["kappa_rel_error_pct"],
            guarding_test="tests/test_csf_curvature_audit.py",
            artifact_status="WITHHELD" if not var["kappa_within_15pct"] else "PASS",
            backlog_ref="backlog_update_m2971.md#P1",
            notes=(
                "Shipped operator: 21.5-28.3 % high over the ladder and non-monotone "
                "(monotone_convergence_of_shipped="
                f"{d['verdicts']['monotone_convergence_of_shipped']}); "
                "the exact-normal control is also outside 15 % at the coarsest levels, so part "
                "of the error is the delta_s weighting itself, not the fit."
            ),
        )
    )

    out.append(
        _entry(
            claim_id="colour_surface.delta_s_integral",
            quantity_id="csf_delta_s_normalisation",
            quantity="colour-surface normalisation, integral |grad c| dr (exact: 1)",
            artifact=rel,
            json_path="verdicts.delta_s_integral_dn",
            reference=(1.0, "-", "CSF requires the surface delta to integrate to one"),
            measured=(
                d["verdicts"]["delta_s_integral_dn"][0],
                "-",
                "coarsest level of the audit ladder (dx = 3 mm)",
            ),
            rel_error_pct=_rel(d["verdicts"]["delta_s_integral_dn"][0], 1.0),
            guarding_test="tests/test_csf_consistent.py",
            artifact_status="WITHHELD",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Root cause of the force-path deficit: the delta function carries only "
                f"{d['verdicts']['delta_s_integral_dn'][0]:.3f} of unit mass, and the "
                "section that does exist is ~28 % overweight, so the cut force comes out "
                "~0.70 sigma. This is the quantity to fix, not the pressure assembly."
            ),
        )
    )
    return out


def model_level_claims() -> list[dict]:
    """Claims about the merged model as a whole (ladders, verdicts)."""
    out: list[dict] = []

    rel = "benchmarks/laplace/report.json"
    d = _load(rel)
    chk = _get(d, "checks.R/dx=10:laplace_pressure")
    out.append(
        _entry(
            claim_id="laplace_report.laplace_pressure.Rdx10",
            quantity_id="laplace_pressure_codeunits_Rdx10",
            quantity="pressure balance of a static droplet at R/dx = 10 (code units)",
            artifact=rel,
            json_path="checks.R/dx=10:laplace_pressure",
            reference=(chk["theory"], "-", "sigma/R in code units"),
            measured=(chk["measured"], "-", "measured dp in code units"),
            rel_error_pct=_rel(chk["measured"], chk["theory"]),
            guarding_test="tests/test_laplace_pressure.py",
            artifact_status=d["validation"]["status"],
            backlog_ref="backlog_update_m2971.md#P0",
            notes="Platform-level verdict at generation time: 9 of 11 criteria failed.",
        )
    )

    rel = "benchmarks/laplace_droplet/results.json"
    d = _load(rel)
    vd = d["verdict"]
    out.append(
        _entry(
            claim_id="laplace_droplet.merged_csf.8cases",
            quantity_id="young_laplace_jump_sigma0.072_R0.1",
            quantity="dp = sigma/R (2-D), sigma=0.072 N/m, R=0.1 m -> 0.72 Pa",
            artifact=rel,
            json_path="verdict",
            reference=(1.0, "-", "dp_model/dp_theory = 1"),
            measured=(
                1.0 + vd["max_relative_error"],
                "-",
                "worst of 8 lattice/jitter cases (0 of 8 within 15 %)",
            ),
            rel_error_pct=vd["max_relative_error"] * 100.0,
            guarding_test="tests/test_laplace_droplet.py::test_laplace_jump_current_state",
            artifact_status=vd["status"],
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Legacy verdict of the merged CSF model: no case within tolerance and no "
                "convergence under refinement. NOTE the fixture-level tests in "
                "tests/test_laplace_droplet.py now record 0.70-0.95 of sigma/R on the "
                "same geometry, i.e. this artifact predates the consistent-scheme refactor "
                "even though it has no superseded_by pointer."
            ),
        )
    )

    rel = "benchmarks/surface_tension/laplace_ladder.json"
    d = _load(rel)
    g = d["gates"]["laplace_error_finest"]
    out.append(
        _entry(
            claim_id="surface_tension_ladder.laplace_error_finest",
            quantity_id="young_laplace_jump_sigma0.072_R0.1",
            quantity="Laplace error of the surface-tension-only ladder, finest level",
            artifact=rel,
            json_path="gates.laplace_error_finest",
            reference=(g["limit"], "fraction", "acceptance limit on |dp_model/dp_theory - 1|"),
            measured=(g["measured"], "fraction", "finest-level relative error"),
            rel_error_pct=g["measured"] * 100.0,
            guarding_test="tests/test_surface_tension_laplace.py",
            artifact_status=d["verdict"],
            backlog_ref="docs/verification_ledger.md",
            notes=(
                f"Observed convergence order {d['laplace_error_order']:.3f} (negative: the "
                "error grows under refinement) -> this ladder's operator is not convergent."
            ),
        )
    )

    rel = "benchmarks/paper_laplace_uncertainty/out/results.json"
    d = _load(rel)
    row = d["rows"][0]
    band = row["metrics"]["band"]
    out.append(
        _entry(
            claim_id="paper_uncertainty.band.dx3mm",
            quantity_id="young_laplace_jump_sigma7.2_R0.02",
            quantity="Young-Laplace jump (2-D), sigma=7.2 N/m, R=20 mm -> 360 Pa",
            artifact=rel,
            json_path="rows[0].metrics.band",
            reference=(band["delta_p_theory_Pa"], "Pa", "Young-Laplace sigma/R"),
            measured=(
                band["delta_p_measured_mean_Pa"],
                "Pa",
                f"band definition, mean of {row['n_realisations']} realisations",
            ),
            rel_error_pct=band["rel_error_pct"],
            guarding_test="tests/test_paper_laplace_uncertainty.py",
            artifact_status="WITHHELD",
            backlog_ref="backlog_update_m2971.md#P0",
            notes=(
                f"Largest R/dx = {row['R_over_dx']:.1f} of the sweep; 95 % CI "
                f"+/-{band['rel_error_ci95_pct']:.2f} pt."
            ),
        )
    )
    return out


def threed_claims() -> list[dict]:
    """3-D operator/decay/hydrostatic claims (benchmarks/threed_*).

    The 3-D branch is new: the shipped kernel is 2-D normalised and ``rhs``
    rejects any cloud whose second axis is not 2, so nothing here could be
    measured before.  Three readings are registered -- the decay rate of the exact
    unsteady-Stokes shear wave, the same rate with the *unrenormalised* viscous
    operator (kept as the A/B partner), and the hydrostatic bottom pressure -- plus
    the Tait-only wall control run, which must stay visible as a failure so nobody
    cites it as a 3-D boundary result.
    """
    qid_decay = "threed_shear_decay_rate_2nuk2"
    qty_decay = (
        "decay exponent of u = A sin(ky) sin(kz) exp(-2 nu k^2 t), periodic 3-D cube, "
        "nu = 0.05 m^2/s, k = 2 pi / m")
    out: list[dict] = []

    rel = "benchmarks/threed_shear/report.json"
    d = _load(rel)
    ladder = d["resolution_ladder"]
    best_i = min(range(len(ladder)), key=lambda i: ladder[i]["kh"])
    best = ladder[best_i]
    out.append(
        _entry(
            claim_id="threed_shear.decay_rate.finest",
            quantity_id=qid_decay,
            quantity=qty_decay,
            artifact=rel,
            json_path=f"resolution_ladder[{best_i}].decay",
            reference=(best["decay"]["analytic_rate"], "1/s", "2 nu k^2 (exact)"),
            measured=(
                best["decay"]["fit_rate"], "1/s",
                f"least-squares slope of ln a(t), a = 4 <u_x sin(ky) sin(kz)>; "
                f"n = {best['n']}^3 particles, k h = {best['kh']:.3f}, "
                f"second-moment renormalised viscous operator",
            ),
            guarding_test="tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria",
            artifact_status=d["verdict"].upper(),
            backlog_ref="docs/verification_threed.md",
            notes=(
                "Renormalising the Laplacian by its discrete second moment "
                "(K = 2.347 instead of the continuum 2 on a cubic lattice with "
                "h = 1.3 dx) is what brings this inside the 3 % band."
            ),
        )
    )

    raw = d.get("raw_vs_renormalised", {}).get("raw")
    if raw:
        out.append(
            _entry(
                claim_id="threed_shear.decay_rate.raw_laplacian",
                quantity_id=qid_decay,
                quantity=qty_decay,
                artifact=rel,
                json_path="raw_vs_renormalised.raw.decay",
                reference=(raw["decay"]["analytic_rate"], "1/s", "2 nu k^2 (exact)"),
                measured=(
                    raw["decay"]["fit_rate"], "1/s",
                    f"same run with the raw second-derivative Laplacian "
                    f"(n = {raw['n']}^3, k h = {raw['kh']:.3f})",
                ),
                guarding_test="tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria",
                artifact_status=d["verdict"].upper(),
                backlog_ref="docs/verification_threed.md",
                notes=(
                    "The raw operator is 17 % too strong for a locally quadratic "
                    "field on a cubic lattice; the shear wave converts that into a "
                    "decay-rate bias of the same order. Kept as the A/B partner."
                ),
            )
        )

    rel = "benchmarks/threed_hydrostatic/report.json"
    h = _load(rel)
    runs = h["runs"]
    adami = [i for i, r in enumerate(runs) if r["wall_model"] == "adami"]
    best_i = min(adami, key=lambda i: abs(runs[i]["pressure"]["bottom_rel_error"]))
    run = runs[best_i]
    qid_p = "threed_hydrostatic_bottom_pressure"
    qty_p = "bottom pressure of a 3-D column at rest, p = rho0 g H"
    out.append(
        _entry(
            claim_id="threed_hydrostatic.bottom_pressure.adami_best",
            quantity_id=qid_p,
            quantity=qty_p,
            artifact=rel,
            json_path=f"runs[{best_i}].pressure",
            reference=(run["pressure"]["bottom_analytic"], "Pa", "rho0 g H, H = nz dx"),
            measured=(
                run["pressure"]["bottom_measured"], "Pa",
                f"mean p of the deepest fluid layer; {run['case']}, "
                f"dx = {run['dx']} m, {run['steps']} steps",
            ),
            guarding_test="tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria",
            artifact_status=h["verdict"].upper(),
            backlog_ref="docs/verification_threed.md",
            notes=(
                "Per-layer relative errors alternate between ~-6 % and up to -43 % "
                "toward the free surface: a particle lattice supports an odd-even "
                "mode in the pressure field. The bottom layer, which is what holds "
                "the load, is quoted here; the odd-even amplitude is recorded in the "
                "artifact so the two are never conflated."
            ),
        )
    )

    controls = [i for i, r in enumerate(runs) if r["wall_model"] == "tait"]
    if controls:
        ci = controls[0]
        control = runs[ci]
        out.append(
            _entry(
                claim_id="threed_hydrostatic.bottom_pressure.tait_wall_control",
                quantity_id=qid_p,
                quantity=qty_p,
                artifact=rel,
                json_path=f"runs[{ci}].pressure",
                reference=(control["pressure"]["bottom_analytic"], "Pa",
                           "rho0 g H, H = nz dx"),
                measured=(
                    control["pressure"]["bottom_measured"], "Pa",
                    f"same case with the wall pressure taken from the wall's own "
                    f"summation density instead of the Adami extrapolation "
                    f"({control['case']})",
                ),
                guarding_test="tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria",
                artifact_status="WITHHELD",
                backlog_ref="docs/verification_threed.md",
                notes=(
                    "Control run, expected to fail: without the Adami extrapolation "
                    "the wall under-pressurises, the column keeps sinking and the "
                    "settling-window drift reaches tens of percent. Recorded so the "
                    "wall model is never assumed to be optional decoration."
                ),
            )
        )

    # -- the unfiltered readings: what the SAME artifacts say when nothing is
    #    selected after the fact ------------------------------------------------
    #  The two claims above are best-of-N selections (finest ladder rung, best of
    #  six hydrostatic runs).  Selection is legitimate only if the spread it hides
    #  is registered next to it, otherwise "PASS" describes the choice, not the
    #  method.  These four rows are the complement: the worst rung of the same
    #  ladder, the worst admissible hydrostatic run, and the interior pressure
    #  profile that the bottom-layer reading does not represent.
    worst_i = max(range(len(ladder)), key=lambda i: abs(ladder[i]["decay"]["rel_error"]))
    worst = ladder[worst_i]
    out.append(
        _entry(
            claim_id="threed_shear.decay_rate.worst_of_ladder",
            quantity_id=qid_decay,
            quantity=qty_decay,
            artifact="benchmarks/threed_shear/report.json",
            json_path=f"resolution_ladder[{worst_i}].decay",
            reference=(worst["decay"]["analytic_rate"], "1/s", "2 nu k^2 (exact)"),
            measured=(
                worst["decay"]["fit_rate"], "1/s",
                f"coarsest rung of the same ladder, n = {worst['n']}^3 particles, "
                f"k h = {worst['kh']:.3f}, second-moment renormalised viscous operator",
            ),
            guarding_test="tests/test_threed_artifacts.py::test_shear_report_meets_its_criteria",
            artifact_status=d["verdict"].upper(),
            backlog_ref="docs/verification_threed.md",
            notes=(
                "Complement of `threed_shear.decay_rate.finest`: the ladder runs "
                f"{worst['decay']['rel_error'] * 1e2:+.2f} % (n = {worst['n']}) to "
                f"{best['decay']['rel_error'] * 1e2:+.2f} % (n = {best['n']}). Quoting only "
                "the finest rung hides that the coarsest one is outside the paper band, "
                "and the artifact's own h/dx sweep is incomplete (see missing_cases)."
            ),
        )
    )

    adami_worst_i = max(adami, key=lambda i: abs(runs[i]["pressure"]["bottom_rel_error"]))
    worst_run = runs[adami_worst_i]
    out.append(
        _entry(
            claim_id="threed_hydrostatic.bottom_pressure.adami_worst",
            quantity_id=qid_p,
            quantity=qty_p,
            artifact=rel,
            json_path=f"runs[{adami_worst_i}].pressure",
            reference=(worst_run["pressure"]["bottom_analytic"], "Pa", "rho0 g H, H = nz dx"),
            measured=(
                worst_run["pressure"]["bottom_measured"], "Pa",
                f"worst of the {len(adami)} Adami runs on the bottom layer; "
                f"{worst_run['case']}, dx = {worst_run['dx']} m",
            ),
            guarding_test="tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria",
            artifact_status=h["verdict"].upper(),
            backlog_ref="docs/verification_threed.md",
            notes=(
                "Complement of `threed_hydrostatic.bottom_pressure.adami_best`: the "
                f"{len(adami)} Adami runs span "
                f"{worst_run['pressure']['bottom_rel_error'] * 1e2:+.2f} % to "
                f"{runs[best_i]['pressure']['bottom_rel_error'] * 1e2:+.2f} % "
                "on the same collapse quantity, so the single best value is a selection, "
                "not a property of the wall model alone (it tracks c0 and therefore the "
                "density error, as Adami et al. 2012 predict)."
            ),
        )
    )

    out.append(
        _entry(
            claim_id="threed_hydrostatic.profile.mean_rel_error_core.adami_best",
            quantity_id="threed_hydrostatic_profile_mean_error",
            quantity=(
                "mean relative error of the interior (non-bottom, non-surface) pressure "
                "profile of a 3-D column at rest"),
            artifact=rel,
            json_path=f"runs[{best_i}].pressure",
            reference=(0.0, "%", "exact profile p = rho0 g (H - z): target error 0 %"),
            measured=(
                run["pressure"]["mean_rel_error_core"] * 1e2, "%",
                f"mean of the per-layer relative errors over the {run['pressure']['core_layers']} "
                f"core layers; {run['case']}",
            ),
            rel_error_pct=run["pressure"]["mean_rel_error_core"] * 1e2,
            guarding_test="tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria",
            artifact_status=h["verdict"].upper(),
            backlog_ref="docs/verification_threed.md",
            notes=(
                "The bottom-layer reading above is one point of a profile that is NOT "
                "within the band everywhere: the interior is systematically low, which is "
                "why the artifact's own criterion `interior_pressure_within_15pct_best` "
                "is false. Registered so the PASS on the bottom layer cannot be read as "
                "'the 3-D hydrostatic profile is verified'."
            ),
        )
    )

    out.append(
        _entry(
            claim_id="threed_hydrostatic.profile.max_rel_error_core.adami_best",
            quantity_id="threed_hydrostatic_profile_max_error",
            quantity=(
                "worst per-layer relative error over the interior pressure profile of a "
                "3-D column at rest"),
            artifact=rel,
            json_path=f"runs[{best_i}].pressure",
            reference=(0.0, "%", "exact profile p = rho0 g (H - z): target error 0 %"),
            measured=(
                run["pressure"]["max_rel_error_core"] * 1e2, "%",
                f"max over the {run['pressure']['core_layers']} core layers; {run['case']}",
            ),
            rel_error_pct=run["pressure"]["max_rel_error_core"] * 1e2,
            guarding_test="tests/test_threed_artifacts.py::test_hydrostatic_report_meets_its_criteria",
            artifact_status="FAIL",
            backlog_ref="docs/verification_threed.md",
            notes=(
                "The odd-even pressure mode of the particle lattice: adjacent layers "
                "alternate between ~-6 % and up to -43 %, so any single-layer reading of "
                "this benchmark is a sample of a mode, not a converged profile."
            ),
        )
    )
    return out


def _oscillation_note(V, omega: float, reference: float) -> str:
    """Prose that states the coupling arithmetic instead of asserting it."""
    cp = V.coupling()
    probe = V.evaluate("oscillating_droplet.frequency.lamb")
    checks = {c["id"]: c for c in probe["checks"]}
    return (
        "The coupling to the force path does NOT hold for this number: with "
        f"sigma_eff/sigma = {cp['static_sigma_eff_over_sigma']:.4f} the capillary scaling "
        f"omega ~ sqrt(sigma_eff) predicts {cp['predicted_rel_error_pct']:+.2f} %, while the "
        f"quoted reading is {cp['quoted_rel_error_pct']:+.2f} % "
        f"(residual {cp['note_claim']['residual_pt']:.2f} pt > "
        f"{cp['note_claim']['criterion_pt']:.0f} pt). "
        "The reason is upstream: the recorded run is dead — rms radius "
        f"{checks['containment']['value']:.0f} R against its own "
        f"{checks['containment']['limit']:g} R gate, "
        f"peak speed Mach {checks['mach_number']['value']:.0f} — so the reading is graded "
        "WITHHELD (evidence insufficient), not FAIL (measured defect). The one sibling rung that "
        "stays coherent (dx = 2 mm) does reproduce the prediction, see "
        "docs/verification_claim_validity.md."
    )


def kernel_claims() -> list[dict]:
    """3-D kernel normalisation: the only exact (machine-precision) reference in the repo.

    The 3-D cubic-spline kernel must integrate to one over its support and its
    Laplacian's second moment must be six; both are analytic identities, so a
    tolerance of 15 % would be meaningless — this entry exists because the file
    was otherwise an unregistered evidence artifact (``unregistered_artifact``
    finding) and the ledger's coverage rule is only meaningful if every
    machine-readable reference measurement is registered.
    """
    rel = "benchmarks/threed_kernel/report.json"
    d = _load(rel)
    q = d["quadrature"]
    return [
        _entry(
            claim_id="threed_kernel.integral_of_W",
            quantity_id="threed_kernel_normalisation",
            quantity="volume integral of the 3-D cubic-spline kernel over its support",
            artifact=rel,
            json_path="quadrature.kernel_integral",
            reference=(1.0, "-", "int W dV = 1 (cubic-spline normalisation)"),
            measured=(
                q["kernel_integral"],
                "-",
                f"quadrature of W over the support at h = {q['h']}",
            ),
            guarding_test="tests/test_threed_kernel.py::test_kernel_integrates_to_one",
            artifact_status="PASS" if d["verdict"] == "pass" else "FAIL",
            backlog_ref="docs/verification_threed.md",
            notes=(
                f"Second moment of laplacian W = {q['second_moment']:.9f} against the "
                "analytic 6, and laplacian W(0) matches its analytic limit to 1e-12 "
                "relative: this is the one quantity in the repo that is verified at "
                "round-off level rather than on the 3 % ladder."
            ),
        )
    ]


def other_physics_claims() -> list[dict]:
    """Non-Laplace validation claims + evidence the repo has withdrawn."""
    out: list[dict] = []
    V = _claim_validity()

    rel = "benchmarks/oscillating_droplet/validation_report.json"
    d = _load(rel)
    row = d["verdict"]["rows"][0]
    out.append(
        _entry(
            claim_id="oscillating_droplet.frequency.lamb",
            quantity_id="oscillating_droplet_omega",
            quantity="fundamental frequency of an inviscid 2-D droplet (Lamb omega_2)",
            artifact=rel,
            json_path="verdict.rows[0]",
            reference=(
                row["omega_reference"],
                "rad/s",
                "Lamb (1932) omega^2 = n(n^2-1) sigma/(rho R^3), n = 2",
            ),
            measured=(
                row["omega_measured"],
                "rad/s",
                "FFT peak of the semi-axis oscillation, dx = 2 mm",
            ),
            rel_error_pct=row["rel_error_pct"],
            guarding_test="tests/test_oscillating_droplet.py",
            artifact_status="FAIL" if not d["verdict"]["worst_within_15pct"] else "PASS",
            backlog_ref="backlog_update_m2971.md#P0",
            notes=_oscillation_note(V, row["omega_measured"], row["omega_reference"]),
        )
    )

    rel = "benchmarks/dam_break/resolution_ladder.json"
    d = _load(rel)
    g = d["gates"]["experimental_comparison"]
    out.append(
        _entry(
            claim_id="dam_break.martin_moyce.mae",
            quantity_id="dam_break_front_position",
            quantity="dam-break front position vs Martin & Moyce (1952), mean abs error in Z/a",
            artifact=rel,
            json_path="gates.experimental_comparison",
            reference=(g["limit"], "Z/a", "acceptance limit on the mean absolute error"),
            measured=(
                g["measured"],
                "Z/a",
                f"mean absolute error over {g['details']['points_compared']} digitised points",
            ),
            rel_error_pct=(g["measured"] / g["limit"] - 1.0) * 100.0,
            guarding_test="tests/test_dam_break_experimental.py",
            artifact_status="PASS" if g["result"] == "pass" else "FAIL",
            backlog_ref="docs/verification_ledger.md",
            claim_type="gate",
            gate_result=g["result"],
            gate_limit=g["limit"],
            notes=(
                f"RMSE {g['details']['rmse']:.3f} Z/a, "
                f"max {g['details']['max_absolute_error']:.3f} Z/a; "
                f"fraction inside experimental uncertainty "
                f"{g['details']['within_uncertainty_fraction']:.2f}. Wave breaking is a "
                "3-D/viscous/non-hydrostatic problem: this is not a surface-tension defect."
            ),
        )
    )

    rel = "benchmarks/static_droplet/validation_report.json"
    d = _load(rel)
    out.append(
        _entry(
            claim_id="static_droplet.force_direction.WITHDRAWN",
            quantity_id="csf_force_direction",
            quantity="sign/shape of the CSF force on a static droplet (not a physical gate)",
            artifact=rel,
            json_path="validation",
            reference=(1.0, "-", "no physical reference: the criterion was self-referential"),
            measured=(
                d["measured"]["force_direction_check"]["avg_dot_product"],
                "-",
                "average radial dot product of the surface-particle force",
            ),
            rel_error_pct=_rel(d["measured"]["force_direction_check"]["avg_dot_product"], 1.0),
            guarding_test="tests/test_laplace_validation.py",
            artifact_status=d["validation"]["status"],
            superseded_by=d["superseded_by"],
            backlog_ref="backlog_update_m2971.md#P2",
            notes=(
                "Kept on purpose: this artifact used to PASS with a criterion no experiment "
                "could falsify, and was withdrawn after audit. Recording it stops the old "
                "PASS from being re-cited."
            ),
        )
    )
    return out


CLOSURE_BENCH = "benchmarks/laplace_pressure_closure"
#: Adami, Ihmsen & Adams (2012) spurious-current criterion for a static drop:
#: the residual velocity of a drop at rest must stay below 10 % of the reference.
ADMISSIBILITY_LIMIT = 0.1


def quiescence_claims() -> list[dict]:
    """Resting-pool / flume noise floor — the defect register entry ``F1``.

    ``benchmarks/startup_quiescence/report.json`` was added by m6281 and never
    registered, which left it as an ``unregistered_artifact`` finding the moment the
    ledger was regenerated.  It carries a real gate (residual velocity of a flume at
    rest must stay below 0.06 m/s), so it is registered as a gate claim instead of
    being waived: the shipped configuration misses it, and the ablation that does meet
    it is recorded next to the failure.
    """
    rel = "benchmarks/startup_quiescence/report.json"
    d = _load(rel)
    i = next(i for i, f in enumerate(d["findings"]) if f["id"] == "F1_flume_resting_noise")
    f1 = d["findings"][i]
    limit = float(f1["limit"])
    measured = float(f1["measured"])
    calmest = f1["ablation"]["calmest"]
    return [
        _entry(
            claim_id="quiescence.flume_resting_noise.v_late_rms",
            quantity_id="quiescence_flume_resting_noise",
            quantity=(
                "residual velocity of a flume at rest: RMS over the last quarter of "
                "max_i |v_i| on interior particles (m/s)"
            ),
            claim_type="gate",
            artifact=rel,
            json_path=f"findings[{i}]",
            reference=(limit, "m/s", "gate_parameters.v_floor_m_s (resting-noise ceiling)"),
            measured=(
                measured,
                "m/s",
                "shipped flume resting column, paddle frozen at rest "
                f"({f1['ablation']['shipped']:.4f} m/s interior RMS)",
            ),
            rel_error_pct=_rel(measured, limit),
            gate_result=f1["status"].lower(),
            gate_limit=limit,
            guarding_test="tests/test_startup_quiescence.py",
            artifact_status="FAIL" if f1["status"].upper() == "OPEN" else "PASS",
            backlog_ref="docs/verification_startup_quiescence.md",
            notes=(
                f"OPEN defect register entry F1: the shipped resting flume reads "
                f"{measured:.4f} m/s against a {limit:g} m/s ceiling, i.e. "
                f"{f1['ratio_to_orbital_velocity'] * 100:.0f} % of the wave orbital velocity "
                "(signal-to-noise ~"
                f"{1.0 / f1['ratio_to_orbital_velocity']:.1f}:1). The ablation "
                f"'{calmest['label']}' reaches {calmest['interior_v_late_rms']:.4f} m/s "
                f"({calmest['interior_v_late_rms'] / measured:.2f}x calmer), so the noise is "
                "controlled by density diffusion + viscosity, not by the boundary model — "
                "see docs/verification_startup_quiescence.md §5."
            ),
        )
    ]


def orphan_evidence_claims() -> list[dict]:
    """Committed evidence that the pre-m6453 coverage rule could not see.

    ``benchmarks/density_diffusion_decay``, ``benchmarks/poiseuille`` and
    ``benchmarks/effective_sigma`` all commit a real analytic comparison, and all
    three sat outside the ledger: the first because its artifact is called
    ``decay_rate.json`` (the rule matched file *names*), the other two because
    their only committed evidence lives in ``out/`` (the rule skipped that
    sub-directory wholesale).  Nothing about the physics changed when they were
    registered; what changed is that the numbers are now graded, and that an
    honest FAIL is visible instead of absent.
    """
    out: list[dict] = []

    # --- Antuono/Colagrossi/Marrone density diffusion: gamma = D k^2 -----------
    rel = "benchmarks/density_diffusion_decay/decay_rate.json"
    d = _load(rel)
    lim = float(d["gates"]["rate_relative_error"])
    out.append(
        _entry(
            claim_id="density_diffusion.decay_rate.worst_of_ladder",
            quantity_id="density_diffusion_decay_rate",
            quantity=(
                "decay rate of a plane density wave vs the analytic gamma = D k^2, "
                "D = delta c0 h (Antuono, Colagrossi & Marrone 2012)"
            ),
            claim_type="gate",
            artifact=rel,
            json_path="max_rate_relative_error",
            reference=(lim, "fraction", "gates.rate_relative_error on |gamma_measured/gamma_theory - 1|"),
            measured=(
                float(d["max_rate_relative_error"]),
                "fraction",
                "worst |gamma_measured/gamma_theory - 1| over the resolution / wavenumber / "
                "smoothing-length ladders",
            ),
            rel_error_pct=100.0 * float(d["max_rate_relative_error"]),
            gate_result="pass" if str(d["verdict"]).lower() == "pass" else "fail",
            gate_limit=lim,
            artifact_status="PASS" if str(d["verdict"]).lower() == "pass" else "FAIL",
            guarding_test=(
                "tests/test_density_diffusion_decay.py::"
                "test_the_diffusive_term_decays_a_plane_wave_at_the_analytic_rate"
            ),
            backlog_ref="docs/verification_density_diffusion_decay.md",
            notes=(
                "D = delta c0 h is not fitted: delta, c0 and h = h_factor dx are solver inputs, "
                "so a sign flip, a factor of two or a dx/h confusion would miss by far more than "
                "the gate. Before this registration the term had only conservation and "
                "smoothing tests, either of which such a defect would pass."
            ),
        )
    )
    ft = d["finite_time"]
    lim2 = float(d["gates"]["finite_time_relative_error"])
    out.append(
        _entry(
            claim_id="density_diffusion.finite_time.amplitude",
            quantity_id="density_diffusion_finite_time_amplitude",
            quantity="mode amplitude after a finite integration time vs rho0 A exp(-gamma t)",
            claim_type="gate",
            artifact=rel,
            json_path="finite_time",
            reference=(lim2, "fraction", "gates.finite_time_relative_error"),
            measured=(
                float(ft["relative_error"]),
                "fraction",
                "|rho(x,t)/rho0 - 1 - A exp(-gamma t)| amplitude error after the run",
            ),
            rel_error_pct=100.0 * float(ft["relative_error"]),
            gate_result="pass" if ft.get("passed") else "fail",
            gate_limit=lim2,
            artifact_status="PASS" if ft.get("passed") else "FAIL",
            guarding_test=(
                "tests/test_density_diffusion_decay.py::"
                "test_the_mode_amplitude_follows_the_exponential_over_a_finite_time"
            ),
            backlog_ref="docs/verification_density_diffusion_decay.md",
        )
    )

    # --- plane Poiseuille channel: u(y) = g (y-a)(b-y) / (2 nu) ----------------
    rel = "benchmarks/poiseuille/out/results.json"
    d = _load(rel)
    rows = d["results"]
    row = max(rows, key=lambda r: r["resolution_H_over_dx"])
    i = rows.index(row)
    u_ref = float(row["fit"]["u_centre_analytic_nominal"])
    out.append(
        _entry(
            claim_id="poiseuille.centre_velocity.finest",
            quantity_id="poiseuille_centre_velocity",
            quantity="centre velocity of a body-force-driven plane channel at steady state",
            artifact=rel,
            json_path=f"results[{i}].fit.u_centre_measured",
            reference=(u_ref, "m/s", "u_max = g H^2 / (8 nu), H = 0.1 m, nu = 0.01 m^2/s, g = 9.81 m/s^2"),
            measured=(
                float(row["fit"]["u_centre_measured"]),
                "m/s",
                f"parabola fitted to the time-averaged profile at H/dx = {row['resolution_H_over_dx']}",
            ),
            guarding_test="tests/test_poiseuille.py::test_steady_channel_flow_develops_the_poiseuille_parabola",
            artifact_status=None,
            backlog_ref="docs/verification_ledger_coverage_m6453.md",
        )
    )
    out.append(
        _entry(
            claim_id="poiseuille.wall_shear_ratio.finest",
            quantity_id="poiseuille_wall_shear_ratio",
            quantity="no-slip wall shear stress of the same channel, relative to rho g H / 2",
            artifact=rel,
            json_path=f"results[{i}].wall_shear_ratio",
            reference=(1.0, "-", "tau_wall = rho nu du/dy|_wall = rho g H / 2 = 490.5 Pa"),
            measured=(
                float(row["wall_shear_ratio"]),
                "-",
                f"measured wall friction divided by the analytic value at H/dx = {row['resolution_H_over_dx']}",
            ),
            guarding_test="tests/test_poiseuille.py::test_steady_channel_flow_develops_the_poiseuille_parabola",
            artifact_status=None,
            backlog_ref="docs/verification_ledger_coverage_m6453.md",
            notes=(
                "WITHHELD rather than FAIL: the two readings inside one artifact disagree. The "
                f"measured centre velocity implies tau = 4 rho nu u_max / H = "
                f"{4 * 1000.0 * 0.01 * float(row['fit']['u_centre_measured']) / 0.1:.0f} Pa, while the "
                f"wall-band readout says {float(row['wall_shear_analytic']) * float(row['wall_shear_ratio']):.0f} Pa, "
                "so the wall-band estimator (not necessarily the flow) carries the deficit and the "
                "number may not be quoted as a physical gate."
            ),
        )
    )

    # --- effective sigma from the half-droplet force integral ------------------
    rel = "benchmarks/effective_sigma/out/report.json"
    d = _load(rel)
    row = next(r for r in d["rows"] if r["dx"] == 0.005 and r["curvature"] == "fit_quadratic")
    i = d["rows"].index(row)
    out.append(
        _entry(
            claim_id="effective_sigma.sigma_eff_over_sigma.dx5mm",
            quantity_id="csf_force_sigma_eff",
            quantity=(
                "effective surface tension from the half-droplet force integral, sigma_eff/sigma "
                "(R = 40 mm, curvature estimator: quadratic interface fit)"
            ),
            artifact=rel,
            json_path=f"rows[{i}].ratio",
            reference=(1.0, "-", "sigma_eff = sigma (|F_half| / (2 sigma))"),
            measured=(
                float(row["ratio"]),
                "-",
                f"|F_half| / (2 sigma) at R/h = {row['R_over_h']:.2f}",
            ),
            guarding_test="tests/test_effective_sigma.py::test_current_default_delivers_the_documented_ratio",
            artifact_status="FAIL",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Independent replication of claim csf_force.sigma_eff_over_sigma.dx5mm at twice the "
                "radius with a different curvature estimator: the two agree to ~1 pt, so the ~30 % "
                "force deficit is not a radius or operator artefact. Same root cause as "
                "colour_surface.delta_s_integral (the colour-surface delta carries ~0.75 of unit mass)."
            ),
        )
    )
    return out


def closure_claims() -> list[dict]:
    """Free-surface pressure closure = the P0 fix of ``backlog_update_m2971.md``.

    ``benchmarks/laplace_pressure_closure`` is where the fix was measured: with the
    Adami/Ihmsen/Adams shell-pressure extrapolation switched on, the absolute
    Young-Laplace read-out of the same drop moves from -38 % to -3.4 %.  Until this
    registry existed those numbers lived only in that benchmark's own document, so
    the canonical claim surface still described the platform as "-34 %, closure is
    the open defect" (``young_laplace.absolute_tail.best_of_6``) long after the fix
    had landed in ``tensorsph.droplet_impact.rhs``.

    Two rules are enforced here instead of in prose:

    * every row is read from ``raw/relax_ladder*.json`` / ``report_force_balance.json``
      at generation time (no number is typed by hand),
    * the *admissibility* of the run travels with the reading.  Two of the four
      closure points keep a residual spurious current above the 10 % limit, so their
      in-band errors stay WITHHELD: the closure is PROVISIONAL (it works at
      dx = 1.5 mm / c0 = 20 and dx = 2 mm / c0 = 10, not yet everywhere).
    """
    qid = "young_laplace_jump_sigma7.2_R0.02"
    qty = "Young-Laplace pressure jump dp = sigma/R, sigma = 7.2 N/m, R = 0.020 m"
    out: list[dict] = []

    # 1) closure on/off relaxation ladder, one row per recorded variant
    for rel in sorted(
        str(p.relative_to(REPO)) for p in (REPO / CLOSURE_BENCH / "raw").glob("relax_ladder*.json")
    ):
        d = _load(rel)
        dx_mm = d["dx_m"] * 1e3
        c0 = d["sound_speed"]
        ref = d["analytic_dp_Pa"]
        by_case = {c["case"]: c for c in d["cases"]}
        off = by_case.get("closure_off")
        for idx, case in enumerate(d["cases"]):
            name = case["case"]
            if name == "closure_off":
                # the closure-off control *is* young_laplace.absolute_tail.* below;
                # re-registering it would double-count one run as two claims
                continue
            err = case["tail_rel_error_pct"]
            vratio = case["vmax_over_vref"]
            adm = {
                "gate": "v_max/v_ref < 0.1 (Adami, Ihmsen & Adams 2012)",
                "value": _r(vratio),
                "limit": ADMISSIBILITY_LIMIT,
                "pass": bool(vratio < ADMISSIBILITY_LIMIT),
            }
            in_band = abs(err) <= CRITERION_PCT
            status = "WITHHELD" if not adm["pass"] else ("PASS" if in_band else "FAIL")
            if off is not None:
                gain = abs(off["tail_rel_error_pct"]) - abs(err)
                gain_txt = (
                    f" The closure-off control of the same lattice reads "
                    f"{off['tail_rel_error_pct']:+.2f} %, i.e. this variant recovers "
                    f"{gain:+.2f} points of the deficit."
                )
            else:
                gain_txt = ""
            if not adm["pass"]:
                adm_txt = (
                    f" INADMISSIBLE: residual spurious current v_max/v_ref = {vratio:.3f} "
                    f">= {ADMISSIBILITY_LIMIT:g}, so the drop was not at rest when the "
                    "reading was taken; the number is recorded but not credited."
                )
            elif in_band:
                adm_txt = (
                    f" Admissible (v_max/v_ref = {vratio:.3f} < {ADMISSIBILITY_LIMIT:g}) and "
                    f"inside the {CRITERION_PCT:.0f} % band, but outside the "
                    f"{PAPER_PCT:.0f} % paper band."
                )
            else:
                adm_txt = f" Out of band at v_max/v_ref = {vratio:.3f}."
            claim_id = f"young_laplace.closure.{name}.dx{dx_mm:g}mm_c{c0:g}"
            twin = next(
                (c["claim_id"] for c in out
                 if c["artifact"] == rel
                 and abs(c["measured"]["value"] - case["tail_mean_Pa"]) < 1e-3),
                None,
            )
            out.append(
                _entry(
                    claim_id=claim_id,
                    quantity_id=qid,
                    quantity=qty,
                    artifact=rel,
                    json_path=f"cases[{idx}]",
                    reference=(ref, "Pa", "Young-Laplace sigma/R"),
                    measured=(
                        case["tail_mean_Pa"],
                        "Pa",
                        f"absolute <p_core> over the capillary tail average, {name}, "
                        f"dx = {dx_mm:g} mm, c0 = {c0:g} m/s, {case['n_particles']} particles, "
                        f"{case['steps']} steps",
                    ),
                    rel_error_pct=err,
                    guarding_test=(
                        "tests/test_young_laplace_closure_ledger.py"
                        "::test_closure_claim_tracks_the_raw_ladder"
                    ),
                    artifact_status=status,
                    admissibility=adm,
                    backlog_ref="backlog_update_m2971.md#P0",
                    notes=(
                        "Free-surface pressure closure (Adami/Ihmsen/Adams 2012 §3.3) wired "
                        "into ``droplet_impact.rhs``." + gain_txt + adm_txt +
                        " Overall the closure is PROVISIONAL: it is admissible and in band "
                        "at two of the four tested (dx, c0) points, not at all of them."
                    ),
                )
            )
            if twin is not None:
                # e.g. layers=4 repeats layers=2 to 1e-9: beyond the shell that many
                # layers do not change the donor set, so it is provenance, not a
                # second piece of evidence.
                out[-1]["superseded_by"] = twin
                out[-1]["notes"] += (
                    f" Bit-identical to `{twin}`: extra donor layers do not change the "
                    "donor set at this resolution, so this row is kept as provenance only "
                    "and does not count as independent evidence."
                )

    # 2) static force balance: the same jump from the surface-tension force, no closure
    rel = f"{CLOSURE_BENCH}/report_force_balance.json"
    d = _load(rel)
    ref = d["reference"]["analytic_dp_Pa"]
    for idx, case in enumerate(d["cases"]):
        fs = case["force_side"]
        dx_mm = case["dx_m"] * 1e3
        err = fs["deficit_pct"]
        out.append(
            _entry(
                claim_id=f"young_laplace.force_balance.flux_integral.dx{dx_mm:g}mm",
                quantity_id=qid,
                quantity=qty,
                artifact=rel,
                json_path=f"cases[{idx}].force_side",
                reference=(ref, "Pa", "Young-Laplace sigma/R"),
                measured=(
                    fs["delta_p_force_Pa"],
                    "Pa",
                    "radial flux integral -int_0^R rho a_r dr of a single CSF evaluation on "
                    "the calibrated lattice (no time stepping)",
                ),
                rel_error_pct=err,
                guarding_test=(
                    "tests/test_young_laplace_closure_ledger.py"
                    "::test_force_balance_integral_is_registered_and_paper_grade"
                ),
                artifact_status="PASS" if abs(err) <= CRITERION_PCT else "FAIL",
                backlog_ref="backlog_update_m2971.md#P0",
                notes=(
                    f"Bypasses the pressure field entirely, so it separates the two halves of "
                    f"the Young-Laplace deficit: the surface-tension force already delivers "
                    f"sigma/R to {err:+.2f} %, hence the historical -38 % lived in the free "
                    "surface pressure assembly (the closure), not in the force model. "
                    "Bins 400/800 agree to 0.01 % (``bin_sensitivity``), so this is not a "
                    "quadrature artefact."
                ),
            )
        )
    return out


#: numbers that no longer describe the current code -> explanation that any
#: document quoting them has to carry next to the number.
SUPERSEDED_NUMBERS = {
    "-75.6%": (
        "csf_consistent relaxed readout was regenerated after the consistent-scheme "
        "refactor (the artifact now reports -78.6 %); current value in "
        "docs/verification_ledger.md"
    ),
    "-98.5%": "legacy gradient CSF scheme, kept only as a before/after contrast",
    "-106.7%": "legacy gradient CSF scheme, kept only as a before/after contrast",
}
MARKERS = ("历史", "historical", "superseded", "旧代码", "legacy", "verification_ledger", "作废")


def ice_floe_added_mass_claims() -> list[dict]:
    """Added-mass correction of the reduced floe launch model.

    ``benchmarks/ice_floe_added_mass`` used to sit outside the ledger entirely
    (``uncovered_benchmark`` finding): its numbers lived only in
    ``docs/verification_ice_floe_added_mass.md``, injected from an uncommitted
    ``artifacts/`` report.  The report is now committed and its readings are
    registered here: the potential-flow anchor against the Lamb closed form
    (raw finest rung + Richardson extrapolation), the plane-limit recovery
    gate, and the launch speed ratio against its algebraic identity.
    """
    rel = "benchmarks/ice_floe_added_mass/report.json"
    d = _load(rel)
    ref_c = d["reference"]["closed_form"]
    src = d["reference"]["source"]
    status = "PASS" if d["verdict"] == "pass" else "FAIL"
    out: list[dict] = []

    ladder = d["anchor_ladder"]
    finest = ladder[-1]
    out.append(
        _entry(
            claim_id="ice_floe_added_mass.anchor.c_m_unbounded",
            quantity_id="ice_floe_added_mass_c_m_unbounded",
            quantity="added-mass coefficient of a 2-D circular section in an unbounded fluid",
            artifact=rel,
            json_path=f"anchor_ladder[{len(ladder) - 1}]",
            reference=(ref_c, "-", src),
            measured=(
                finest["coefficient"],
                "-",
                f"potential-flow SOR, zero initial, {finest['cells_per_radius']} "
                f"cells/radius, box {d['config']['box_radii']:g}R, "
                f"tol {d['config']['tolerance']:g}",
            ),
            rel_error_pct=finest["error_vs_lamb_pct"],
            guarding_test="tests/test_ice_floe_added_mass.py",
            artifact_status=status,
            backlog_ref="docs/verification_ice_floe_added_mass.md",
            notes=(
                "First-order staircase body boundary: the raw finest rung is a "
                "systematic lower bound; the Richardson reading is the converged "
                "estimate. Box sensitivity < 0.5 % and x/y isotropy at machine "
                "precision are recorded in the same artifact."
            ),
        )
    )

    rich = d["richardson_extrapolation"]
    out.append(
        _entry(
            claim_id="ice_floe_added_mass.anchor.c_m_richardson",
            quantity_id="ice_floe_added_mass_c_m_unbounded",
            quantity="added-mass coefficient of a 2-D circular section in an unbounded fluid",
            artifact=rel,
            json_path="richardson_extrapolation",
            reference=(ref_c, "-", src),
            measured=(
                rich["coefficient"],
                "-",
                f"first-order Richardson over {rich['cells_per_radius_pair'][0]} -> "
                f"{rich['cells_per_radius_pair'][1]} cells/radius",
            ),
            rel_error_pct=rich["error_vs_lamb_pct"],
            guarding_test="tests/test_ice_floe_added_mass.py",
            artifact_status=status,
            backlog_ref="docs/verification_ice_floe_added_mass.md",
            notes="Extrapolation assumes first-order convergence of the staircase boundary.",
        )
    )

    planes = d["plane_limits"]
    limit = CRITERION_PCT
    dev = planes["max_ratio_deviation_pct"]
    out.append(
        _entry(
            claim_id="ice_floe_added_mass.plane_limits.max_deviation",
            quantity_id="ice_floe_added_mass_plane_limit_recovery",
            quantity=(
                "plane-limit recovery: C_m(gap)/C_m(infinity) for a rigid wall and a "
                "linearised free surface, gap >= 2 radii"
            ),
            claim_type="gate",
            artifact=rel,
            json_path="plane_limits",
            reference=(limit, "%", "platform acceptance line (CRITERION_PCT) on the max ratio deviation"),
            measured=(
                dev,
                "%",
                "max |C_m(gap)/C_m(infinity) - 1| over 2 boundaries x 2 axes x 2 gaps",
            ),
            rel_error_pct=dev,
            gate_result="pass" if dev <= limit else "fail",
            gate_limit=limit,
            guarding_test="tests/test_ice_floe_added_mass.py",
            artifact_status=status,
            backlog_ref="docs/verification_ice_floe_added_mass.md",
            notes=(
                "gap < 2 radii (the real floe-on-deck configuration) is unresolved on "
                "the Cartesian ladder and is deliberately not quoted (doc section 5)."
            ),
        )
    )

    lb = d["launch_bias"]
    out.append(
        _entry(
            claim_id="ice_floe_added_mass.launch.speed_ratio",
            quantity_id="ice_floe_added_mass_launch_speed_ratio",
            quantity=(
                "floe launch speed ratio v_wet/v_dry vs the algebraic identity "
                "rho_ice/(rho_ice + C_m rho_w)"
            ),
            artifact=rel,
            json_path="launch_bias.speed_ratio_mean",
            reference=(
                lb["ratio_closed_form"],
                "-",
                "speed_ratio_with_added_mass(rho_ice, rho_w, C_m = 1)",
            ),
            measured=(
                lb["speed_ratio_mean"],
                "-",
                "mean over 5 floes of the SPH impulse run (dx = 0.0125 m, t = 0.46 s)",
            ),
            rel_error_pct=100.0 * (lb["speed_ratio_mean"] / lb["ratio_closed_form"] - 1.0),
            guarding_test="tests/test_ice_floe_added_mass.py",
            artifact_status=status,
            backlog_ref="docs/verification_ice_floe_added_mass.md",
            notes=(
                "The correction is an algebraic identity applied to the SPH impulse; "
                "the SPH run supplies the impulse only, so agreement is expected at "
                "machine precision."
            ),
        )
    )
    return out


def grade(e: dict) -> str:
    """One repo-wide grading rule, applied uniformly to every entry."""
    if e["superseded_by"]:
        return "WITHDRAWN"
    if e.get("run_validity") == "INVALID":
        # The trajectory was no longer a simulation when the reading was taken, so
        # "measured defect" (FAIL) would be a false statement about the model: the
        # honest grade is evidence insufficient (WITHHELD).
        return "WITHHELD"
    if e["claim_type"] == "gate":
        if e["gate_result"] != "pass":
            return "FAIL"
        err = abs(e["rel_error_pct"]) if e["rel_error_pct"] is not None else 0.0
        if err <= PAPER_PCT:
            return "PASS"
        return "MARGINAL" if err <= CRITERION_PCT else "FAIL"
    if e["gauge_degenerate"]:
        return "WITHHELD"
    adm = e.get("admissibility")
    if adm and not adm.get("pass", False):
        # A run the admissibility gate rejects cannot be credited however close its
        # number lands: the reading was taken from a drop that was not quiescent, so
        # the agreement (or the miss) is a property of the spurious current.
        return "WITHHELD"
    err = abs(e["rel_error_pct"])
    if err <= PAPER_PCT:
        return "PASS"
    if err <= CRITERION_PCT:
        return "MARGINAL"
    if (e["artifact_status"] or "").upper() == "FAIL":
        return "FAIL"
    return "WITHHELD"


def _is_fragment(rel: str) -> bool:
    """True for per-case dumps / ladders that inherit their parent's provenance."""
    parts = Path(rel).parts
    if len(parts) <= 3:                      # benchmarks/<dir>/<file>.json
        return False
    sub = parts[2]
    if sub in FRAGMENT_DIRS:
        return True
    if sub == "out":
        return not PRIMARY_NAMES.search(parts[-1])
    return True


def _harness_dirs(root: Path) -> list[Path]:
    """Benchmark directories that ship a runnable harness (``run*.py``)."""
    base = root / "benchmarks"
    if not base.is_dir():
        return []
    return [
        d
        for d in sorted(base.iterdir())
        if d.is_dir() and any(p.name.startswith("run") and p.suffix == ".py" for p in d.iterdir())
    ]


def _is_covered(rel: str, registered: set[str], waived: set[str]) -> bool:
    """A directory is covered by any claim artifact under it, or by a waiver."""
    rel = rel.rstrip("/")
    if rel in waived or rel + "/" in waived:
        return True
    if any(w.startswith(rel + "/") for w in waived if w.endswith("/")):
        return True
    return any(a.startswith(rel + "/") for a in registered)


def coverage_findings(
    registered: set[str], waived: set[str], root: Path | None = None
) -> list[dict]:
    """Evidence coverage: no benchmark may sit outside the ledger unnoticed.

    Before m6453 this rule matched artifact *names* (``report|results|ladder|
    audit.json``) and skipped everything under ``out/``.  Two things therefore
    escaped it silently: an artifact with any other name (``decay_rate.json``,
    ``verdict_matrix.json``) and -- worse -- a whole harness that commits no
    artifact at all, whose printed numbers can never be recomputed from the
    repository.  Coverage is now decided by *directory*, not by file name.
    """
    root = REPO if root is None else root
    out: list[dict] = []

    # (c1) a harness with no claim and no waiver: nothing it prints is graded
    for d in _harness_dirs(root):
        rel = str(d.relative_to(root))
        if _is_covered(rel, registered, waived):
            continue
        runs = sorted(p.name for p in d.iterdir() if p.name.startswith("run") and p.suffix == ".py")
        out.append(
            {
                "rule": "uncovered_benchmark",
                "severity": "medium",
                "claim_id": rel,
                "detail": (
                    f"benchmark harness ({', '.join(runs)}) has no ledger claim and no "
                    "waiver: nothing it produces is graded, so a regression there is "
                    "invisible and its published numbers are not reproducible from the repo"
                ),
            }
        )

    # (c2) a primary evidence file that no claim points at
    for path in sorted((root / "benchmarks").rglob("*.json")):
        rel = str(path.relative_to(root))
        if rel.startswith(LEDGER_DIR) or _is_fragment(rel):
            continue
        if rel in registered or rel in waived:
            continue
        out.append(
            {
                "rule": "unregistered_artifact",
                "severity": "medium",
                "claim_id": rel,
                "detail": "machine-readable evidence file not referenced by any ledger claim",
            }
        )
    return out


def findings(entries: list[dict], root: Path | None = None) -> list[dict]:
    """Machine-detectable defects in the evidence base itself."""
    out: list[dict] = []

    # (a) an artifact that declares PASS but whose reading we refuse to credit
    for e in entries:
        g = grade(e)
        declared = (e["artifact_status"] or "").upper()
        if declared == "PASS" and g in {"WITHHELD", "FAIL"}:
            why = (
                "vacuum-gauge readout (additive |p| offset, not a physical dp)"
                if e["gauge_degenerate"]
                else f"|rel error| = {abs(e['rel_error_pct']):.2f} % > {CRITERION_PCT:.0f} %"
            )
            out.append(
                {
                    "rule": "declared_vs_derived",
                    "severity": "high",
                    "claim_id": e["claim_id"],
                    "detail": f"artifact declares PASS, ledger derives {g}: {why}",
                }
            )

    # (a2) a criterion that passes while the physics behind it is outside the band
    for e in entries:
        if e["claim_type"] == "gate" and e["gate_result"] == "pass" and grade(e) == "FAIL":
            out.append({
                "rule": "gate_vs_band",
                "severity": "medium",
                "claim_id": e["claim_id"],
                "detail": (
                    f"gate passes at limit {e['gate_limit']:g} but the observed error "
                    f"({e['rel_error_pct']:+.2f} %) is outside the {CRITERION_PCT:.0f} % band"
                ),
            })

    # (a3) a reading whose source run was dead when the number was taken
    for e in entries:
        if e.get("run_validity") == "INVALID":
            out.append({
                "rule": "run_validity",
                "severity": "high",
                "claim_id": e["claim_id"],
                "detail": (
                    "validity gate says INVALID (containment / Mach / estimator / "
                    "fit / null-control): the run had stopped being a liquid before "
                    "the reading was taken, so the reading is graded WITHHELD instead "
                    "of FAIL — see docs/verification_claim_validity.md"
                ),
            })

    # (a4) a reading inside the band whose run the admissibility gate rejects
    for e in entries:
        adm = e.get("admissibility")
        if not adm or adm.get("pass", True):
            continue
        out.append(
            {
                "rule": "inadmissible_run",
                "severity": "high",
                "claim_id": e["claim_id"],
                "detail": (
                    f"{e['rel_error_pct']:+.2f} % is inside the {CRITERION_PCT:.0f} % band, "
                    f"but the run violates the admissibility gate "
                    f"({adm['gate']}: {adm['value']:g} >= {adm['limit']:g}): the residual "
                    "spurious current, not the physics, sets the agreement, so the reading "
                    "is graded WITHHELD and may not be quoted as a pass"
                ),
            }
        )

    # (a5) a credit whose own artifact declares a failure (best-of-N / partial ladder)
    out.extend(credit_findings(entries))

    # (b) one physical quantity, several irreconcilable readings
    by_q: dict[str, list[dict]] = {}
    for e in entries:
        if e["claim_type"] == "relative_error" and not e["superseded_by"]:
            by_q.setdefault(e["quantity_id"], []).append(e)
    for qid, group in sorted(by_q.items()):
        if len(group) < 2:
            continue
        errs = [e["rel_error_pct"] for e in group]
        spread = max(errs) - min(errs)
        if spread > CRITERION_PCT:
            out.append(
                {
                    "rule": "quantity_spread",
                    "severity": "high",
                    "claim_id": qid,
                    "detail": (
                        f"{len(group)} active readings of one quantity span "
                        f"{min(errs):+.1f} % … {max(errs):+.1f} % (spread {spread:.1f} pt): "
                        + ", ".join(f"{e['claim_id']}={e['rel_error_pct']:+.1f} %" for e in group)
                        + " — definitions differ (gauge / scheme / resolution), so no single "
                        "number may be quoted without naming its definition."
                    ),
                }
            )

    # (c) artifacts / harnesses present in the repo but absent from the ledger
    registered = {e["artifact"] for e in entries}
    waived = _load_waivers()
    out.extend(coverage_findings(registered, waived, root=root))

    # (d) stale numbers still quoted in prose without a marker
    for path in sorted((REPO / "docs").rglob("*.md")):
        rel = str(path.relative_to(REPO))
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for token, why in SUPERSEDED_NUMBERS.items():
                if token in line and not any(m in line for m in MARKERS):
                    out.append(
                        {
                            "rule": "unmarked_superseded_number",
                            "severity": "medium",
                            "claim_id": f"{rel}:{lineno}",
                            "detail": f"quotes {token} without a historical marker ({why})",
                        }
                    )
    return out


def _load_waivers() -> set[str]:
    if not WAIVERS.is_file():
        return set()
    with WAIVERS.open() as fh:
        return set(json.load(fh)["unregistered_ok"])


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# declared verdicts: what the artifact itself says about the run it came from
# ---------------------------------------------------------------------------
VERDICT_KEYS = ("verdict", "status", "evidence_grade")
_FAIL_TOKENS = {"fail", "failed", "false", "withheld", "invalid", "incomplete"}
_PASS_TOKENS = {"pass", "passed", "ok", "true", "valid"}


def normalise_verdict(value) -> str | None:
    """Map a declared status of any shape onto ``pass`` / ``fail`` / ``None``."""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, str):
        tok = value.strip().lower()
        if tok in _FAIL_TOKENS:
            return "fail"
        if tok in _PASS_TOKENS:
            return "pass"
        return None
    if isinstance(value, dict):
        for key in VERDICT_KEYS:
            if key in value:
                norm = normalise_verdict(value[key])
                if norm:
                    return norm
        if "worst_within_15pct" in value:
            return "pass" if value["worst_within_15pct"] else "fail"
    return None


def declared_verdict(artifact: str, _cache: dict = {}) -> dict | None:
    """The artifact's own top-level verdict, with the machine-checkable reasons.

    A ledger entry records *one reading*; the artifact that carries it declares
    a verdict about the *whole run set*.  Those two are not the same statement:
    a best-of-N or a finest-rung selection can read inside the band while the
    artifact's own verdict is ``fail`` (missing ladder case, failing interior
    criterion).  This helper makes that declaration travel with the reading so
    it cannot be silently dropped.
    """
    if artifact in _cache:
        return _cache[artifact]
    try:
        d = _load(artifact)
    except FileNotFoundError:
        _cache[artifact] = None
        return None
    found = None
    if isinstance(d, dict):
        for key in VERDICT_KEYS:
            if key not in d:
                continue
            norm = normalise_verdict(d[key])
            if norm:
                criteria = d.get("criteria") if isinstance(d.get("criteria"), dict) else {}
                missing = d.get("missing_cases") or []
                found = {
                    "value": norm,
                    "json_path": key,
                    "failing_criteria": sorted(k for k, v in criteria.items() if v is False),
                    "missing_cases": [str(m) for m in missing],
                }
                break
    _cache[artifact] = found
    return found


def credit_findings(entries: list[dict]) -> list[dict]:
    """Readings credited as PASS/MARGINAL whose own artifact declares a failure."""
    out = []
    for e in entries:
        if e["grade"] not in {"PASS", "MARGINAL"}:
            continue
        dv = e.get("declared_verdict")
        if not dv or dv["value"] != "fail":
            continue
        why = []
        if dv["failing_criteria"]:
            why.append("failing criteria: " + ", ".join(dv["failing_criteria"]))
        if dv["missing_cases"]:
            why.append("missing cases: " + ", ".join(dv["missing_cases"]))
        reason = f" ({'; '.join(why)})" if why else ""
        out.append(
            {
                "rule": "credit_from_failing_artifact",
                "severity": "high",
                "claim_id": e["claim_id"],
                "detail": (
                    f"{e['rel_error_pct']:+.2f} % is credited {e['grade']}, but the "
                    f"artifact `{e['artifact']}` declares `{dv['json_path']} = fail`"
                    f"{reason}: this is a best-of-N / finest-rung selection out of a run "
                    "set that does not pass, so it may not be quoted as verification of "
                    "the capability -- see docs/verification_ledger_overcredit_m6510.md"
                ),
            }
        )
    return out


def ice_thermal_claims() -> list[dict]:
    """Ice-ocean thermal-saline-mechanical multi-physics face (IN13W2-N07).

    ``benchmarks/ice_thermal_multiphysics`` came in with the campaign-node
    delivery and sat outside the ledger (``uncovered_benchmark`` finding): its
    anchors were only quoted in the campaign report.  The three readings
    registered here are the ones the node's acceptance rests on, and each is
    compared with an *independent* reference -- the Neumann similarity solution
    of the Stefan problem, the exact Horton-Rogers-Lapwood onset ``4 pi^2``, and
    the closed-form irreducible-brine floor (``v_irr`` is an input, not a fit).
    The artifact is the run report published next to the harness
    (``validation_report.json``, byte-identical to the campaign run in
    ``runs/IN13W2-N07``: the numbers are in the repository, not in a scratch
    dir, and ``evidence.py`` checks the two copies against each other).
    """
    rel = "benchmarks/ice_thermal_multiphysics/validation_report.json"
    d = _load(rel)
    out: list[dict] = []
    test = "tests/test_ice_thermal_integration.py"
    stefan = d["stefan"][-1]
    out.append(
        _entry(
            claim_id="ice_thermal.stefan_front.finest",
            quantity_id="ice_thermal_stefan_front_x",
            quantity="1-D Stefan freezing front X(t_end) vs the Neumann similarity solution",
            artifact=rel,
            json_path=f"stefan[{len(d['stefan']) - 1}]",
            reference=(stefan["front_analytic_m"], "m",
                       "Neumann solution X = 2 lambda sqrt(alpha t), lambda from "
                       "sqrt(pi) lambda exp(lambda^2) erf(lambda) = Ste"),
            measured=(stefan["front_final_m"], "m",
                      f"SPH enthalpy particle front, dx = {stefan['dx_m']:g} m, "
                      f"t_end = 4e4 s, {stefan['steps']} steps"),
            rel_error_pct=stefan["front_rel_error_final"] * 100.0,
            guarding_test=test,
            artifact_status="PASS",
            backlog_ref="docs/verification_ice_thermal_multiphysics.md",
            notes=("Resolution ladder dx = 0.02/0.0125/0.008 m is in the same artifact; "
                   "the front error is 3.7 % at the first sample of the coarsest grid "
                   "(start-up transient) and <= 2 % at t_end."),
        )
    )
    darcy = d["darcy"]
    out.append(
        _entry(
            claim_id="ice_thermal.darcy_benard.ra_c",
            quantity_id="ice_thermal_darcy_ra_c",
            quantity="critical Rayleigh number of Horton-Rogers-Lapwood onset",
            artifact=rel,
            json_path="darcy",
            reference=(darcy["ra_critical_exact"], "-", "exact 4 pi^2"),
            measured=(darcy["ra_critical_numeric"], "-",
                      f"finite-difference eigenvalue solve, {darcy['fd_nodes']} nodes, "
                      f"{len(darcy['curve']['a'])} wavenumbers"),
            rel_error_pct=darcy["ra_relative_error"] * 100.0,
            guarding_test=test,
            artifact_status="PASS",
            backlog_ref="docs/verification_ice_thermal_multiphysics.md",
        )
    )
    des = d["desalination"]
    out.append(
        _entry(
            claim_id="ice_thermal.brine_floor.min_margin",
            quantity_id="ice_thermal_brine_floor_margin",
            quantity="drainage must never remove salt below the irreducible-brine floor",
            artifact=rel,
            json_path="desalination.floor_min_margin_psu",
            reference=(0.0, "psu", "one-sided bound: S(z) >= S_eq(T(z)), S_eq from v_b = v_irr"),
            measured=(min(0.0, des["floor_min_margin_psu"]), "psu",
                      "min over the column of S - S_eq after 60 d of gravity drainage"),
            rel_error_pct=0.0 if des["floor_min_margin_psu"] >= -1e-6 else 100.0,
            claim_type="gate",
            gate_result="pass" if des["floor_min_margin_psu"] >= -1e-6 else "fail",
            gate_limit=1e-6,
            guarding_test=test,
            artifact_status="PASS",
            backlog_ref="docs/verification_ice_thermal_multiphysics.md",
            notes=("One-sided by construction: drainage is rate limited, so the 60-day "
                   f"column stays above the floor (final bulk {des['final_bulk_salinity_psu']:.3f} "
                   f"psu vs the asymptote {des['floor_predicted_bulk_psu']:.3f} psu). "
                   "Crossing the floor would be a physics defect, not a tolerance miss."),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Coverage closure (m14729)
# ---------------------------------------------------------------------------
#: Readings that lived in committed artifacts the ledger could not see, plus the
#: harnesses that have no artifact at all.  The point of registering them is that
#: a regression in any of these harnesses used to be invisible: nothing graded
#: them, so nothing could fail.
def coverage_closure_claims() -> list[dict]:
    out: list[dict] = []

    # ---- aeration benchmark (bubble rise / air entrainment / hydraulic jump) ----
    rel = "benchmarks/aeration/report.json"
    d = _load(rel)
    crit = d["criteria"]
    verdict = "PASS" if d["all_criteria_passed"] else "FAIL"
    out.append(
        _entry(
            claim_id="aeration.void_fraction.erf_anchor_l2",
            quantity_id="aeration_void_fraction_profile",
            quantity="steady void-fraction profile of the aerated hydraulic jump vs the erf anchor",
            artifact=rel,
            json_path="criteria.C2.measured",
            reference=(0.0, "-", "exact erf anchor profile (zero residual)"),
            measured=(
                crit["C2"]["measured"],
                "-",
                "relative L2 difference between the measured and the anchor profile",
            ),
            # The metric *is* a relative error against the anchor, so it is entered
            # directly instead of being normalised by its own acceptance limit.
            rel_error_pct=crit["C2"]["measured"] * 100.0,
            claim_type="gate",
            gate_result="pass" if crit["C2"]["passed"] else "fail",
            gate_limit=crit["C2"]["threshold"],
            guarding_test="tests/test_aeration.py",
            artifact_status=verdict,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Registered because the harness was invisible to the ledger "
                "(uncovered_benchmark). 9.5 % against a 15 % limit is MARGINAL, not a "
                "pass: the anchor is an erf fit of the void-fraction profile, not "
                "experimental data."
            ),
        )
    )
    out.append(
        _entry(
            claim_id="aeration.air_volume_drift",
            quantity_id="aeration_air_mass_conservation",
            quantity="air volume drift per characteristic time of the aerated jump",
            artifact=rel,
            json_path="criteria.C5.measured",
            reference=(0.0, "-", "exact conservation of entrained air volume"),
            measured=(
                crit["C5"]["measured"],
                "-",
                "relative air-volume drift per characteristic time",
            ),
            rel_error_pct=crit["C5"]["measured"] * 100.0,
            guarding_test="tests/test_aeration.py",
            artifact_status=verdict,
            backlog_ref="docs/verification_ledger.md",
            notes="0.37 % drift over the characteristic time; conservation reading only.",
        )
    )
    out.append(
        _entry(
            claim_id="aeration.inception_constant.self_referential",
            quantity_id="aeration_inception_froude",
            quantity="inception Froude constant of air entrainment vs Chanson",
            artifact=rel,
            json_path="inception.constant_rel_error",
            reference=(
                d["inception"]["empirical_constant"],
                "-",
                "Chanson empirical criterion constant (input to the criterion)",
            ),
            measured=(
                d["inception"]["derived_constant"],
                "-",
                "constant the harness derives from its own onset sweep",
            ),
            rel_error_pct=d["inception"]["constant_rel_error"] * 100.0,
            gauge_degenerate=True,
            guarding_test="tests/test_aeration.py",
            artifact_status=verdict,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Exactly 0.0 % because the derived constant is the empirical constant "
                "rearranged: the criterion is checked against itself, so the reading "
                "is WITHHELD regardless of the artifact's all_criteria_passed = true. "
                "The same degeneracy hits C4 (bubble rise velocity vs Mendelson), which "
                "reports 0.0 % because the module *implements* the Mendelson formula; "
                "that reading is deliberately not registered as a claim."
            ),
        )
    )

    # ---- Rayleigh-Plateau ladder (analytic dispersion + 3-D column run) ----
    rel = "benchmarks/rayleigh_plateau/report.json"
    d = _load(rel)
    an = d["analytic"]
    out.append(
        _entry(
            claim_id="rayleigh_plateau.analytic.kR_star",
            quantity_id="rayleigh_plateau_most_unstable_kR",
            quantity="most unstable wavenumber of the Rayleigh-Plateau instability (kR)*",
            artifact=rel,
            json_path="analytic.most_unstable_kR",
            reference=(
                an["most_unstable_kR_literature"],
                "-",
                "Rayleigh (1879) / literature value (kR)* = 0.697",
            ),
            measured=(
                an["most_unstable_kR"],
                "-",
                "root of the implemented dispersion relation with the Bessel term",
            ),
            rel_error_pct=an["most_unstable_kR_error"] * 100.0,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "This validates the *dispersion solver* (the Bessel term matters: the "
                "naive kR = 1/sqrt(3) = 0.5774 is -17.2 % off), not a simulation. The "
                "dynamic 3-D column run in the same artifact does not deliver a growth "
                "rate (see rayleigh_plateau.dynamic.volume_drift)."
            ),
        )
    )
    out.append(
        _entry(
            claim_id="rayleigh_plateau.analytic.lambda_over_R",
            quantity_id="rayleigh_plateau_most_unstable_wavelength",
            quantity="most unstable wavelength of the Rayleigh-Plateau instability, lambda/R",
            artifact=rel,
            json_path="analytic.most_unstable_lambda_over_R",
            reference=(an["most_unstable_lambda_over_R_literature"], "-", "literature lambda/R = 9.02"),
            measured=(
                an["most_unstable_lambda_over_R"],
                "-",
                "2 pi / (kR)* from the implemented dispersion relation",
            ),
            rel_error_pct=an["most_unstable_lambda_over_R_error"] * 100.0,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes="Same analytic solver as kR*; -0.06 % is a rounding-level agreement.",
        )
    )
    cur = next(
        row
        for row in d["curvature"]
        if row["estimator"] == "profile" and row["radius_over_dx"] == 3
    )
    mls = next(
        row for row in d["curvature"] if row["estimator"] == "mls" and row["radius_over_dx"] == 3
    )
    axisym = next(
        row for row in d["curvature"] if row["estimator"] == "axisym" and row["radius_over_dx"] == 3
    )
    out.append(
        _entry(
            claim_id="rayleigh_plateau.curvature.profile_self_referential",
            quantity_id="csf_curvature_3d_cylinder",
            quantity="3-D CSF curvature on a cylindrical column, R/dx = 3",
            artifact=rel,
            json_path="curvature[0].relative_error",
            reference=(0.0, "-", "exact curvature 1/R of the constructed cylinder"),
            measured=(cur["relative_error"], "-", "'profile' estimator: fits the radius of the surface profile"),
            rel_error_pct=cur["relative_error"] * 100.0,
            gauge_degenerate=True,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "2.2e-16 is machine zero, i.e. the estimator recovers the radius it was "
                "built from: self-referential. The independent estimators on the same "
                f"cloud are mls {mls['relative_error'] * 100:+.1f} % and axisym "
                f"{axisym['relative_error'] * 100:+.1f} % -- that spread is the real "
                "curvature evidence, and it is not paper grade."
            ),
        )
    )
    run = d["column_run"]
    out.append(
        _entry(
            claim_id="rayleigh_plateau.dynamic.volume_drift",
            quantity_id="rayleigh_plateau_volume_drift",
            quantity="volume drift of the 3-D Rayleigh-Plateau column run (80 steps, 864 particles)",
            artifact=rel,
            json_path="column_run.final_volume_error",
            reference=(0.0, "-", "exact volume conservation of the incompressible column"),
            measured=(run["final_volume_error"], "-", "final relative volume error of the run"),
            rel_error_pct=run["final_volume_error"] * 100.0,
            guarding_test="tests/test_rayleigh_plateau.py",
            artifact_status=None,
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "The growth-rate fit of the same run is NaN "
                f"(omega_measured = NaN, max density error {run['max_density_error'] * 100:.1f} %), "
                "so the dynamic Rayleigh-Plateau capability is NOT validated: this entry "
                "records only the volume bookkeeping of a run that could not be reduced "
                "to a growth rate. Closing it needs a longer/steadier run."
            ),
        )
    )

    # ---- CSF dispersion ladder (capillary wave + Lamb modes) ----
    # The measurement moved into ``tensorsph.surface_tension_dispersion`` +
    # ``scripts/run_surface_tension_dispersion.py`` (m13895); the committed artifact
    # is ``validation_report.json``, next to this harness.  m14729 wired these two
    # claims to the pre-move path and the pre-move schema, so ``build()`` raised
    # FileNotFoundError at HEAD and took every other claim down with it.  The paths
    # below are the ones the current runner actually writes (m14849).
    rel = "runs/surface_tension_dispersion/report.json"
    d = _load(rel)
    g = d["gates"]
    c1, c2, c3 = g["C1_capillary_wave"], g["C2_lamb_modes"], g["C3_convergence"]
    _best = d["option_screen_best"]
    out.append(
        _entry(
            claim_id="csf_dispersion.capillary_wave.omega_error.local_range",
            quantity_id="capillary_wave_omega",
            quantity="capillary-wave frequency error of the local-range CSF operator, k = 1..3",
            artifact=rel,
            json_path="gates.C1_capillary_wave.max_abs_error_pct",
            reference=(
                c1["tolerance_pct"] / 100.0,
                "-",
                "harness criterion C1: within 5 % of omega^2 = sigma k^3 / rho "
                "(Lamb 1932, Art. 267)",
            ),
            measured=(
                c1["max_abs_error_pct"] / 100.0,
                "-",
                "max relative omega error over k = 1..3 at the finest grid (ppw = 48), "
                f"delta normalisation = {d['setup']['delta_normalization']}, "
                f"curvature = {d['setup']['curvature_scheme']}",
            ),
            claim_type="gate",
            gate_result="pass" if c1["passed"] else "fail",
            gate_limit=0.05,
            guarding_test="tests/test_csf_dispersion.py",
            artifact_status="PASS" if c1["passed"] else "FAIL",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                "Registered because the harness was uncovered; the reading is the "
                "harness's own failed gate, not an averaged score. It misses the 5 % "
                f"criterion by {c1['max_abs_error_pct'] / c1['tolerance_pct']:.1f}x "
                f"(|error| = {c1['max_abs_error_pct']:.1f} %). The option screen over "
                "delta normalisation x curvature scheme says no shipped combination "
                "passes it (best |omega| error "
                f"{_best['wave_min_abs_error_pct']:.1f} % over "
                f"{_best['screened']} combinations), and the C3 refinement orders of "
                f"the same run are {c3['min_order']:+.2f} (required {c3['required']}), "
                "so the error does not shrink under refinement: an operator-consistency "
                "defect, not a resolution problem."
            ),
        )
    )
    out.append(
        _entry(
            claim_id="csf_dispersion.lamb_mode.omega_error.local_range",
            quantity_id="lamb_mode_omega",
            quantity="Lamb-mode frequency error of the local-range CSF operator, n = 2..4",
            artifact=rel,
            json_path="gates.C2_lamb_modes.max_abs_error_pct",
            reference=(
                c2["tolerance_pct"] / 100.0,
                "-",
                "harness criterion C2: within 3 % of omega^2 = n(n^2-1) sigma/(rho R^3)",
            ),
            measured=(
                c2["max_abs_error_pct"] / 100.0,
                "-",
                "max relative omega error over n = 2..4 at the finest grid (R/dx = 32)",
            ),
            claim_type="gate",
            gate_result="pass" if c2["passed"] else "fail",
            gate_limit=0.03,
            guarding_test="tests/test_csf_dispersion.py",
            artifact_status="PASS" if c2["passed"] else "FAIL",
            backlog_ref="docs/verification_ledger.md",
            notes=(
                f"|error| = {c2['max_abs_error_pct']:.2f} % against a 3 % gate: better "
                "than C1 by two orders of magnitude but still a FAIL, so it is not "
                "averaged into a PASS. The best screened combination reaches "
                f"{_best['lamb_min_abs_error_pct']:.2f} % and still misses the gate "
                f"(any_pass = {_best['any_lamb_pass']}), i.e. the drop family is close to "
                "acceptable while the wave family is not."
            ),
        )
    )

    # ---- fluctuating hydrodynamics (MD thermostat ladder) ----
    rel = "benchmarks/fluctuating_hydro/report.json"
    d = _load(rel)
    row = d["case"]["runs"][0]
    out.append(
        _entry(
            claim_id="fluctuating_hydro.thermostat.mean_temperature.n128",
            quantity_id="fluctuating_hydro_temperature",
            quantity="mean temperature of the LJ fluid under the Langevin (BAOAB) thermostat, N = 128",
            artifact=rel,
            json_path="case.runs[0].mean_temperature",
            reference=(1.0, "T*", "target temperature T* = 1.0 (reduced LJ units)"),
            measured=(row["mean_temperature"], "T*", f"mean over {row['n_steps']} steps"),
            guarding_test="tests/test_fluctuating_hydro.py",
            artifact_status="PASS" if d["verdicts"]["L4_thermostat_holds_target_temperature"] else "FAIL",
            backlog_ref="docs/verified/fluctuating_hydrodynamics_verification.md",
            notes=(
                "Supersedes the legacy reading T = 0.0152 (error -98.5 %) quoted in "
                "docs/verified/fluctuating_hydrodynamics_verification.md: the legacy "
                "lattice overlapped particles (up to 81 overlapping pairs at N = 256), "
                "which is what collapsed the temperature. Registered here so the old "
                "number cannot be quoted as current."
            ),
        )
    )

    # ---- same quantity, steady state, uncertainty-quantified (m16346 audit) ----
    audit_rel = "benchmarks/fluctuating_hydro/thermostat_audit.json"
    audit_path = REPO / audit_rel if "REPO" in globals() else None
    try:
        audit = json.loads(pathlib.Path(audit_rel).read_text())
    except (OSError, ValueError):
        audit = None
    if audit:
        ens = audit["ensemble_summary"]
        replay = audit["replay"]
        out.append(
            _entry(
                claim_id="fluctuating_hydro.thermostat.steady_state.n128.uq",
                quantity_id="fluctuating_hydro_temperature",
                quantity=(
                    "steady-state temperature of the LJ fluid under the Langevin "
                    "(BAOAB) thermostat, N = 128, uncertainty quantified"
                ),
                artifact=audit_rel,
                json_path="ensemble_summary.inverse_variance_combined",
                reference=(1.0, "T*", "target temperature T* = 1.0 (reduced LJ units)"),
                measured=(ens["inverse_variance_combined"], "T*",
                          f"{ens['n_seeds']} independent seeds, "
                          f"+/- {ens['inverse_variance_se']:.4f} (1 SE)"),
                guarding_test="tests/test_fluctuating_hydro_thermostat.py",
                artifact_status="FAIL",
                backlog_ref="docs/verification_fluctuating_hydro_thermostat.md",
                notes=(
                    f"Second reading of the same quantity, taken at a converged window "
                    f"(10 relaxation times discarded, block-averaged SE, SE plateau "
                    f"checked over 4-12 blocks). The window ladder is flat, so this is "
                    f"a systematic offset of {ens['dev_over_se']:.1f} sigma, not sample "
                    f"noise. The archived entry above reads {replay['mean_temperature']:.5f} T* "
                    f"(+8.81 %) from an 800-step run whose discarded window is 1.6 "
                    f"relaxation times: that reading is the startup transient, it points "
                    f"the opposite way, and it must not be quoted as the thermostat's "
                    f"steady state."
                ),
            )
        )
    return out


def build() -> dict:
    entries = (
        young_laplace_claims()
        + force_path_claims()
        + operator_level_claims()
        + model_level_claims()
        + other_physics_claims()
        + kernel_claims()
        + threed_claims()
        + closure_claims()
        + quiescence_claims()
        + orphan_evidence_claims()
        + ice_floe_added_mass_claims()
        + ice_thermal_claims()
        + coverage_closure_claims()
    )
    V = _claim_validity()
    for e in entries:
        e["run_validity"] = V.evaluate(e["claim_id"])["status"]
        e["grade"] = grade(e)
        e["declared_verdict"] = declared_verdict(e["artifact"])
        # A credit is *qualified* when the artifact that carries it does not
        # contradict it.  ``None`` = the artifact declares nothing at all, which is
        # weaker than a pass but is not the same defect as a declared failure.
        if e["grade"] in {"PASS", "MARGINAL"}:
            dv = e["declared_verdict"]
            e["credit_qualified"] = (dv is None) or dv["value"] == "pass"
        else:
            e["credit_qualified"] = None
        # schema-drift guard: every reading must be reachable from its artifact
        if not (REPO / e["artifact"]).is_file():
            raise FileNotFoundError(f"{e['claim_id']}: artifact gone: {e['artifact']}")
        resolve(_load(e["artifact"]), e["json_path"])
    entries.sort(key=lambda e: e["claim_id"])
    found = findings(entries)
    summary = {
        "n_claims": len(entries),
        "n_quantities": len({e["quantity_id"] for e in entries}),
        "by_grade": {g: sum(1 for e in entries if e["grade"] == g) for g in GRADES},
        "n_findings": len(found),
        "findings_by_rule": {
            r: sum(1 for f in found if f["rule"] == r)
            for r in sorted({f["rule"] for f in found})
        },
        "criterion_pct": CRITERION_PCT,
        "paper_criterion_pct": PAPER_PCT,
        "n_credits_unqualified": sum(
            1 for e in entries
            if e["grade"] in {"PASS", "MARGINAL"} and e["credit_qualified"] is False),
        "n_credits_without_declared_verdict": sum(
            1 for e in entries
            if e["grade"] in {"PASS", "MARGINAL"}
            and e.get("declared_verdict") is None),
        "n_artifact_verdicts_fail": sum(
            1 for e in entries
            if (e.get("declared_verdict") or {}).get("value") == "fail"),
    }
    return {"schema_version": 1, "summary": summary, "entries": entries, "findings": found}


def _fmt_err(e: dict) -> str:
    return "—" if e["claim_type"] == "gate" else f"{e['rel_error_pct']:+.2f} %"


def render_markdown(ledger: dict) -> str:
    s = ledger["summary"]
    L = [
        "# TensorSPH 验证断言账本（generated — 勿手改）",
        "",
        "> 由 `benchmarks/verification_ledger/build_ledger.py` 从 `benchmarks/**/*.json` "
        "程序化生成；`tests/test_verification_ledger.py` 在重生成结果与提交副本不一致时失败。"
        "表内每个数字都能按「工件 → JSON 路径」溯源，零手抄。",
        "",
        "## 1. 为什么需要这本账",
        "",
        "同一个物理量（σ=7.2 N/m、R=20 mm 的 Young–Laplace 压差，解析值 360 Pa）在仓里"
        "有 5 份互相矛盾的机器可读证据，口径/容差/状态词汇各不相同：有的用 "
        "`p_core − p_band`（旧双标尺）、有的用绝对压强时间平均、有的拿 `p_center` 对真空当压差、"
        "有的用 `p_plateau − p_ambient`。这些数字被 `docs/` 跨文引用，却没有任何一处说明"
        "**哪个是当前值、哪个已作废、哪个被自己的量具 disqualify**。本账本把这件事机器化：",
        "",
        "1. 每条 claim =（工件 + JSON 路径 + 读数 + 参考值 + 守护测试 + 欠账指针）；",
        "2. 分级规则全仓唯一（§2），**不由各工件自述决定**；",
        "3. §5 的 findings 记录证据本身的可检测缺陷：退化量具、自述 PASS 高于容差、"
        "同一物理量多读数、未登记工件、散文里未标注的作废数字。",
        "",
        "## 2. 分级规则（机裁）",
        "",
        "| 等级 | 判据 | 含义 |",
        "|---|---|---|",
        f"| PASS | \\|rel err\\| ≤ {PAPER_PCT:.0f}% | 论文级 |",
        f"| MARGINAL | {PAPER_PCT:.0f}% < \\|rel err\\| ≤ {CRITERION_PCT:.0f}% "
        "| 过平台线、不够论文级 |",
        f"| WITHHELD | \\|rel err\\| > {CRITERION_PCT:.0f}%，工件未给出收敛性证伪 "
        "| 测得出但不可采信 |",
        f"| FAIL | \\|rel err\\| > {CRITERION_PCT:.0f}% 且工件自证不收敛/不满足 | 已证伪 |",
        "| WITHDRAWN | 工件被仓自身撤销（`superseded_by` / `withdrawn`） | 不得再引用 |",
        "",
        "量具退化（`gauge_degenerate`：压差用真空当环境压）**一律** WITHHELD，"
        "无论其 rel error 多小。",
        "",
        "## 3. 汇总",
        "",
        f"- claims **{s['n_claims']}** ／ 物理量 **{s['n_quantities']}** "
        f"／ findings **{s['n_findings']}**",
        "- 分级分布：" + "，".join(f"{k} {v}" for k, v in s["by_grade"].items() if v),
        "- findings 分类：" + "，".join(f"{k} {v}" for k, v in s["findings_by_rule"].items()),
        f"- **credit 资格**：{s['by_grade'].get('PASS', 0)} 条 PASS / "
        f"{s['by_grade'].get('MARGINAL', 0)} 条 MARGINAL，其中 "
        f"**{s['n_credits_unqualified']}** 条取自「工件自述 verdict = fail」的运行集"
        "（best-of-N / 阶梯不全，不可作为能力验证引用）；另有 "
        f"{s['n_credits_without_declared_verdict']} 条 credit 的工件根本没有声明 verdict"
        "（等级只能被读成「该读数自身合规」，不是「该能力被验证」）。",
        "",
        "## 4. 断言表",
        "",
        "| claim | 物理量 / 口径 | 参考 | 实测 | rel err | 等级 | 工件 → JSON 路径 | 守护测试 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in ledger["entries"]:
        if e["claim_type"] == "gate":
            ref = f"limit {e['gate_limit']:g}"
            mea = f"{e['measured']['value']:g}"
        else:
            ref = f"{e['reference']['value']:g} {e['reference']['unit']}"
            mea = f"{e['measured']['value']:g} {e['measured']['unit']}"
        L.append(
            f"| `{e['claim_id']}` | {e['quantity']} | {ref} | {mea} | {_fmt_err(e)} | "
            f"**{e['grade']}** | `{e['artifact']}` → `{e['json_path']}` | `{e['guarding_test']}` |"
        )

    L += ["", "## 5. findings（证据缺陷清单）", ""]
    if not ledger["findings"]:
        L.append("（无）")
    for f in ledger["findings"]:
        L += [
            f"- **{f['rule']}**（`{f['severity']}`）`{f['claim_id']}` — {f['detail']}",
        ]

    L += ["", "## 6. 读数明细（口径决定数字）", ""]
    for e in ledger["entries"]:
        sup = f"；被 `{e['superseded_by']}` 取代" if e["superseded_by"] else ""
        L += [
            f"### `{e['claim_id']}` — {e['grade']}",
            "",
            f"- 物理量：{e['quantity']}",
            f"- 参考：{e['reference']['value']:g} {e['reference']['unit']}"
            f"（{e['reference']['source']}）",
            f"- 实测：{e['measured']['value']:g} {e['measured']['unit']}",
            f"  - 口径：{e['measured']['definition']}",
            f"- 相对误差：{_fmt_err(e)}（平台线 ±{e['criterion_pct']:.0f}%，"
            f"论文级 ±{e['paper_criterion_pct']:.0f}%）",
            f"- 工件自述状态：`{e['artifact_status']}`；本账等级：**{e['grade']}**",
            (
                "- 工件自身 verdict：**"
                f"{e['declared_verdict']['value']}**（`{e['declared_verdict']['json_path']}`"
                + (
                    "；未通过项：" + "、".join(e["declared_verdict"]["failing_criteria"])
                    if e["declared_verdict"]["failing_criteria"] else ""
                )
                + (
                    "；缺失用例：" + "、".join(e["declared_verdict"]["missing_cases"])
                    if e["declared_verdict"]["missing_cases"] else ""
                )
                + "）"
                if e.get("declared_verdict") else "- 工件自身 verdict：未声明"
            ),
            f"- 溯源：`{e['artifact']}` → `{e['json_path']}`",
            f"  - 守护测试：`{e['guarding_test']}`",
            f"- 欠账指针：`{e['backlog_ref']}`" + sup,
        ]
        if e["notes"]:
            L.append(f"- 注：{e['notes']}")
        L.append("")

    L += [
        "## 7. 怎么用 / 怎么维护",
        "",
        "```bash",
        "# 新增或刷新任意 benchmark 工件后，重生成账本与图",
        "python benchmarks/verification_ledger/build_ledger.py --figure",
        "# 只检查是否漂移（CI 语义，退出码 1 = 有漂移）",
        "python benchmarks/verification_ledger/build_ledger.py --check",
        "```",
        "",
        "纪律：",
        "",
        "1. **不要手改本文件**，改生成器；重生成即得。",
        "2. 新增 benchmark 工具必须登记 claim（否则 `unregistered_artifact` finding），或在 "
        "`benchmarks/verification_ledger/waivers.json` 里显式豁免并写理由。",
        "3. 引用任何验证数字必须写明口径（本表 §6 的 `definition` 字段）；"
        "`WITHHELD`/`FAIL`/`WITHDRAWN` 的数字不得作为当前精度引用。",
        "4. 散文里出现 `SUPERSEDED_NUMBERS` 中的作废数字必须带历史标记，否则 finding。",
        "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def render_figure(ledger: dict, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [e for e in ledger["entries"] if e["claim_type"] == "relative_error"]
    rows.sort(key=lambda e: e["rel_error_pct"])
    labels = [e["claim_id"] for e in rows]
    vals = [e["rel_error_pct"] for e in rows]
    colors = [
        {"PASS": "#2a9d8f", "MARGINAL": "#e9c46a", "WITHHELD": "#f4a261", "FAIL": "#e76f51",
         "WITHDRAWN": "#8d99ae"}[e["grade"]]
        for e in rows
    ]
    fig, ax = plt.subplots(figsize=(9.6, 0.36 * len(rows) + 2.0))
    y = range(len(rows))
    ax.barh(list(y), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8, family="monospace")
    for limit, style in ((15.0, "--"), (-15.0, "--")):
        ax.axvline(limit, color="#c1121f", linestyle=style, linewidth=1.2,
                   label="±15 % platform limit" if limit > 0 else None)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("signed relative error of the measured quantity [%]")
    ax.set_title(
        f"TensorSPH verification ledger — {len(rows)} claims, "
        f"{ledger['summary']['n_findings']} findings (bar colour = derived grade)"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_all(with_figure: bool = False) -> dict:
    ledger = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_JSON.write_text(_dump(ledger))
    DOC_MD.write_text(render_markdown(ledger))
    if with_figure:
        render_figure(ledger)
    return ledger


def check() -> int:
    ledger = build()
    problems: list[str] = []
    if not LEDGER_JSON.is_file() or LEDGER_JSON.read_text() != _dump(ledger):
        problems.append("ledger.json is stale (run build_ledger.py)")
    md = render_markdown(ledger)
    if not DOC_MD.is_file() or DOC_MD.read_text() != md:
        problems.append("docs/verification_ledger.md is stale (run build_ledger.py)")
    for p in problems:
        print(f"DRIFT: {p}")
    if not problems:
        print(
            f"ledger up to date: {ledger['summary']['n_claims']} claims, "
            f"{ledger['summary']['n_findings']} findings"
        )
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    ap.add_argument("--figure", action="store_true", help="also regenerate the png")
    ap.add_argument("--list-findings", action="store_true", help="print findings and exit")
    args = ap.parse_args(argv)

    if args.check:
        return check()
    ledger = write_all(with_figure=args.figure)
    print(
        f"wrote {LEDGER_JSON.relative_to(REPO)} and {DOC_MD.relative_to(REPO)}: "
        f"{ledger['summary']['n_claims']} claims, {ledger['summary']['n_findings']} findings"
    )
    if args.list_findings:
        for f in ledger["findings"]:
            print(f"[{f['severity']}] {f['rule']}: {f['claim_id']} — {f['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
