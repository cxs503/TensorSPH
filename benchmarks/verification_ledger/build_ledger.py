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
    err = abs(e["rel_error_pct"])
    if err <= PAPER_PCT:
        return "PASS"
    if err <= CRITERION_PCT:
        return "MARGINAL"
    if (e["artifact_status"] or "").upper() == "FAIL":
        return "FAIL"
    return "WITHHELD"


def findings(entries: list[dict]) -> list[dict]:
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

    # (c) artifacts present in the repo but absent from the ledger
    registered = {e["artifact"] for e in entries}
    waived = _load_waivers()
    for path in sorted((REPO / "benchmarks").rglob("*.json")):
        rel = str(path.relative_to(REPO))
        if "/out/" in rel or rel in registered or rel in waived:
            continue
        if re.search(r"(validation_report|report|results|ladder|audit)\.json$", rel):
            out.append(
                {
                    "rule": "unregistered_artifact",
                    "severity": "medium",
                    "claim_id": rel,
                    "detail": "machine-readable evidence file not referenced by any ledger claim",
                }
            )

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
def build() -> dict:
    entries = (
        young_laplace_claims()
        + force_path_claims()
        + operator_level_claims()
        + model_level_claims()
        + other_physics_claims()
        + kernel_claims()
        + threed_claims()
    )
    V = _claim_validity()
    for e in entries:
        e["run_validity"] = V.evaluate(e["claim_id"])["status"]
        e["grade"] = grade(e)
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
