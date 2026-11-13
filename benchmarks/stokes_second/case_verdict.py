#!/usr/bin/env python3
"""Case-level verdict for ``CASE-SPH-STOKES-2ND-PROBLEM`` (six predeclared gates).

Why this file exists
--------------------
``benchmarks/stokes_second/DESIGN.md`` predeclares six gates before the ladder
was run: G1 amplitude L2, G2 worst row, G3 decay rate, G4 finite-depth L2,
G5 free-slip negative control, G6 resolution, plus the requirement that the
Stokes layer contain at least five particle rows.  The published artefacts
grade them in **different places and at different levels**:

* G1-G4 and the row count are per-rung fields of ``out/results_ladder.json``;
* G5 lives in ``out/results_sensitivity.json`` (the ``free_slip`` run);
* G6 is the ``convergence`` block of the ladder.

``out/benchmark.json`` calls itself "one row per judged quantity" (DESIGN §7)
but carries only the finest rung's G1-G4-ish rows: nothing in the machine-
readable bundle says whether the negative control held or whether the ladder
was complete.  A reader who trusts ``benchmark.json`` therefore cannot tell a
qualified case from one whose control was never run -- and the first delivery of
this case shipped exactly that failure (a one-rung "ladder" with
``G6_not_growing: true``).

This module produces the missing artefact: **one verdict object** that

* recomputes G1-G4 from each rung's own persisted rows (not from the stored
  ``metrics`` fields, which it cross-checks) so a hand-edited metric cannot
  certify itself;
* re-derives G6 from the persisted L2 series *and* cross-checks the stored
  ``complete``/``G6_not_growing`` flags (a stored ``true`` that the rows do not
  support is a FAIL, not a pass);
* grades G5 only when a free-slip control run exists **and** its
  ``provenance.case`` fingerprint equals the ladder's -- otherwise the gate is
  ``WITHHELD`` (fail-closed: no evidence is not a pass);
* reports an ``overall`` status that is ``PASS`` only when every predeclared
  gate passed on evidence from the same case configuration.

Usage
-----
    PYTHONPATH=.:src python benchmarks/stokes_second/case_verdict.py            # print
    PYTHONPATH=.:src python benchmarks/stokes_second/case_verdict.py --write    # + JSON/MD
    PYTHONPATH=.:src python benchmarks/stokes_second/case_verdict.py --check    # exit code gate

Exit codes: 0 = PASS, 1 = FAIL, 2 = WITHHELD (missing or mismatched evidence).

The thresholds below are the predeclared ones.  They are duplicated here on
purpose (this module must be importable without pulling in torch through
``run.py``) and ``tests/test_stokes_second_case_verdict.py`` asserts that they
equal ``run.GATES`` and the numbers written in ``DESIGN.md``, so drift is a test
failure rather than a silent re-grading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
VERDICT_JSON = HERE / "case_verdict.json"
VERDICT_DOC = REPO / "docs" / "stokes_second_case_verdict.md"

ANCHOR = "CASE-SPH-STOKES-2ND-PROBLEM"
CASE = "stokes_second_oscillating_plate"

#: Predeclared gates (DESIGN.md §4).  ``key`` is the field name ``run.gates()``
#: uses for the per-run gates; kept identical so the two implementations can be
#: compared mechanically.
PREDECLARED = {
    "G1": {"metric": "amplitude_l2_half_space_pct",
           "threshold_pct": 5.0,
           "statement": "amplitude relative L2 over rows with y <= delta, "
                        "vs U0 exp(-y/delta)"},
    "G2": {"metric": "amplitude_worst_row_half_space_pct",
           "threshold_pct": 10.0,
           "statement": "worst single row in the same set"},
    "G3": {"metric": "decay_rate_error_pct",
           "threshold_pct": 5.0,
           "statement": "fitted decay rate vs 1/delta "
                        "(delta_measured/delta - 1)"},
    "G4": {"metric": "amplitude_l2_finite_depth_pct",
           "threshold_pct": 5.0,
           "statement": "same L2 vs the finite-depth exact solution "
                        "(discretisation only)"},
    "G5": {"metric": "amplitude_max_over_U0",
           "threshold": 0.05, "direction": "le",
           "statement": "negative control: free-slip oscillating wall drives "
                        "no flow (A_max/U0 < 5 %)"},
    "G6": {"metric": "l2_by_resolution",
           "threshold_pct": 0.5,
           "statement": "resolution: L2 does not grow from the coarsest to the "
                        "finest mesh (finest <= coarsest + 0.5 pp) and the "
                        "predeclared ladder is complete"},
}

#: The case-pool requirement that the Stokes layer must be resolved by >= 5 rows.
LAYER_MIN_ROWS = 5
REQUIRED_RESOLUTIONS = (8, 12, 18)

#: ``provenance.case`` keys that must agree between the ladder and the control
#: block before either can be used to grade the other.  ``ladder`` is included
#: because a control run at a different ``h/dx`` or ``c0`` is a different case.
FINGERPRINT_KEYS = (
    "nu", "omega", "U0", "c0", "periods", "measure_periods", "H_over_delta",
    "columns", "rho0", "gamma", "h_factor", "wall_friction",
)

#: The persisted ``rows`` are rounded to 6 significant digits, so a relative
#: error of ~1e-4 (the graded quantity near the finest rung) reproduces only to
#: ~1e-6 absolute; the cross-check below is therefore done in the metric's own
#: units (percentage points) rather than as a relative comparison of a number
#: that is itself a relative error.
TOL_PCT = 1e-3          # absolute, percentage points
TOL_REL = 1e-5          # relative, for dimensionless ratios (delta_fit/delta)


# --------------------------------------------------------------------------- #
# Loading and hashing
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifacts(out_dir: Path | str = OUT):
    """Return ``(ladder, sensitivity, provenance)``; sensitivity may be ``None``."""
    out_dir = Path(out_dir)
    ladder_path = out_dir / "results_ladder.json"
    if not ladder_path.is_file():
        raise FileNotFoundError(f"no ladder artefact at {ladder_path}")
    ladder = json.loads(ladder_path.read_text())
    sens_path = out_dir / "results_sensitivity.json"
    sensitivity = json.loads(sens_path.read_text()) if sens_path.is_file() else None
    provenance = {
        "results_ladder.json": {"sha256": _sha256(ladder_path),
                                "bytes": ladder_path.stat().st_size},
    }
    if sensitivity is not None:
        provenance["results_sensitivity.json"] = {
            "sha256": _sha256(sens_path), "bytes": sens_path.stat().st_size}
    return ladder, sensitivity, provenance


def _fingerprint(payload) -> dict:
    case = ((payload or {}).get("provenance") or {}).get("case") or {}
    return {k: case.get(k) for k in FINGERPRINT_KEYS}


# --------------------------------------------------------------------------- #
# Independent re-derivation of a rung's graded metrics
# --------------------------------------------------------------------------- #

def recompute_rung(run: dict) -> dict:
    """Recompute a rung's G1-G4 ingredients from its own persisted ``rows``.

    Deliberately independent of ``run["metrics"]``: the relative error is
    rebuilt from ``amplitude`` and the persisted analytic column, the decay rate
    from a least-squares fit of ``ln A`` against ``y`` over ``y <= delta`` with
    ``A > 0`` -- the definitions in ``run.evaluate``.
    """
    rows = run["rows"]
    delta = run["delta"]
    layer = [r for r in rows if r["y"] <= delta]
    if not layer:
        raise ValueError(f"rung {run.get('resolution_delta_over_dx')} has no "
                         "rows inside the Stokes layer")

    rel_half, rel_fd = [], []
    for r in layer:
        ref_half = r["amplitude_half_space"]
        ref_fd = r["amplitude_finite_depth"]
        rel_half.append((r["amplitude"] - ref_half) / ref_half)
        rel_fd.append((r["amplitude"] - ref_fd) / ref_fd)

    def l2(vals):
        return math.sqrt(sum(v * v for v in vals) / len(vals))

    fit = [r for r in rows if r["y"] <= delta and r["amplitude"] > 0.0]
    if len(fit) < 3:
        raise ValueError("not enough rows for the decay-rate fit")
    slope = float(np.polyfit([r["y"] for r in fit],
                             np.log([r["amplitude"] for r in fit]), 1)[0])
    return {
        "resolution_delta_over_dx": run["resolution_delta_over_dx"],
        "n_rows_stokes_layer": len(layer),
        "amplitude_l2_half_space_pct": 100.0 * l2(rel_half),
        "amplitude_worst_row_half_space_pct": 100.0 * max(abs(v) for v in rel_half),
        "amplitude_l2_finite_depth_pct": 100.0 * l2(rel_fd),
        "decay_rate_measured": slope,
        "delta_ratio": (-1.0 / slope) / delta,
        "decay_rate_error_pct": 100.0 * (slope * delta + 1.0),
    }


def _close(a: float, b: float, *, key: str = "") -> bool:
    if a is None or b is None:
        return False
    try:
        if math.isnan(a) or math.isnan(b):
            return False
    except TypeError:
        return False
    if key.endswith("_pct"):
        return abs(a - b) <= TOL_PCT
    return abs(a - b) <= TOL_REL * max(abs(a), abs(b), 1.0)


def _cross_check(run: dict, recomputed: dict) -> list[str]:
    """Stored ``metrics`` fields that the persisted rows do not support."""
    stored = run["metrics"]
    bad = []
    for key, value in recomputed.items():
        if key in ("resolution_delta_over_dx", "n_rows_stokes_layer"):
            continue
        if not _close(stored.get(key), value, key=key):
            bad.append(f"rung delta/dx={run['resolution_delta_over_dx']}: stored "
                       f"{key}={stored.get(key)!r} vs rows {value!r}")
    if stored.get("n_rows_stokes_layer") != recomputed["n_rows_stokes_layer"]:
        bad.append(f"rung delta/dx={run['resolution_delta_over_dx']}: stored "
                   f"n_rows_stokes_layer={stored.get('n_rows_stokes_layer')!r} vs "
                   f"rows {recomputed['n_rows_stokes_layer']}")
    return bad


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #

def _gate(gid, status, measured, evidence, statement=None, threshold=None,
          unit=None, direction=None, **extra) -> dict:
    spec = PREDECLARED.get(gid, {})
    out = {"id": gid,
           "statement": statement or spec.get("statement", gid),
           "threshold": (threshold if threshold is not None
                         else spec.get("threshold", spec.get("threshold_pct"))),
           "direction": (direction if direction is not None
                         else spec.get("direction", "le")),
           "unit": unit or ("fraction_of_U0" if gid == "G5" else "%"),
           "status": status, "measured": measured, "evidence": evidence}
    out.update(extra)
    return out


def case_verdict(ladder: dict, sensitivity: dict | None, *,
                 provenance: dict | None = None,
                 required=REQUIRED_RESOLUTIONS) -> dict:
    """Aggregate the six predeclared gates into one fail-closed verdict."""
    runs = sorted(ladder["runs"], key=lambda r: r["resolution_delta_over_dx"])
    if not runs:
        raise ValueError("ladder artefact contains no runs")
    rec = [recompute_rung(r) for r in runs]
    consistency: list[str] = []
    for run, r in zip(runs, rec):
        consistency.extend(_cross_check(run, r))

    gates: list[dict] = []
    gate_evidence = "out/results_ladder.json:runs[*].rows"

    # ---- G1..G4: every rung must pass; the reported value is the worst rung --
    for gid in ("G1", "G2", "G3", "G4"):
        metric = PREDECLARED[gid]["metric"]
        thr = PREDECLARED[gid]["threshold_pct"]
        # Published value (what results_ladder.json certifies) is the gate input;
        # the row-derived value is reported next to it and cross-checked above.
        per_rung = {f"{run['resolution_delta_over_dx']}": run["metrics"][metric]
                    for run in runs}
        from_rows = {f"{r['resolution_delta_over_dx']}": r[metric] for r in rec}
        worst = max(per_rung.values(), key=abs)
        status = "PASS" if all(abs(v) <= thr for v in per_rung.values()) else "FAIL"
        gates.append(_gate(gid, status, worst, gate_evidence,
                           per_rung=per_rung, per_rung_from_rows=from_rows,
                           worst_rung=max(per_rung, key=lambda k: abs(per_rung[k]))))

    # ---- layer resolution (case-pool requirement, not a numeric gate) -------
    rows_per_rung = {f"{r['resolution_delta_over_dx']}": r["n_rows_stokes_layer"]
                     for r in rec}
    gates.append(_gate("rows_in_layer",
                       "PASS" if min(rows_per_rung.values()) >= LAYER_MIN_ROWS
                       else "FAIL",
                       min(rows_per_rung.values()),
                       gate_evidence, per_rung=rows_per_rung,
                       statement="the Stokes layer is resolved by at least "
                                 f"{LAYER_MIN_ROWS} particle rows (case-pool "
                                 "requirement)",
                       threshold=LAYER_MIN_ROWS, unit="rows", direction="ge"))

    # ---- G6: recomputed from the persisted L2 series, stored flags checked ---
    res = [r["resolution_delta_over_dx"] for r in rec]
    l2s = [abs(r["amplitude_l2_half_space_pct"]) for r in rec]
    conv = ladder.get("convergence") or {}
    complete = res == [int(x) for x in required]
    growing = l2s[-1] <= l2s[0] + PREDECLARED["G6"]["threshold_pct"]
    stored_complete = bool(conv.get("complete"))
    stored_not_growing = bool(conv.get("G6_not_growing"))
    stored_l2 = [abs(float(x)) for x in conv.get("l2_by_resolution", [])]
    if len(stored_l2) == len(l2s):
        for s_val, r_val, rung in zip(stored_l2, l2s, res):
            if not _close(s_val, r_val, key="l2_by_resolution_pct"):
                consistency.append(f"rung delta/dx={rung}: stored convergence "
                                   f"L2={s_val!r} vs rows {r_val!r}")
    else:
        consistency.append("stored convergence block does not list one L2 per rung")
    g6_measured = {"resolutions": res, "required_resolutions": [int(x) for x in required],
                   "l2_pct": l2s, "l2_pct_stored": stored_l2,
                   "complete": complete, "not_growing": growing,
                   "stored_complete": stored_complete,
                   "stored_not_growing": stored_not_growing}
    if not complete:
        g6 = _gate("G6", "FAIL", g6_measured, "out/results_ladder.json:convergence",
                   reason="predeclared ladder is not the set of rungs that were run")
    elif not growing:
        g6 = _gate("G6", "FAIL", g6_measured, "out/results_ladder.json:convergence",
                   reason="L2 grows from the coarsest to the finest mesh")
    elif not (stored_complete and stored_not_growing):
        g6 = _gate("G6", "FAIL", g6_measured, "out/results_ladder.json:convergence",
                   reason="stored convergence flags contradict the persisted L2 series")
    else:
        g6 = _gate("G6", "PASS", g6_measured, "out/results_ladder.json:convergence")
    gates.append(g6)

    # ---- G5: negative control, fail-closed on missing / mismatched evidence --
    fp_ladder = _fingerprint(ladder)
    fp_sens = _fingerprint(sensitivity)
    mismatches = [f"{k}: ladder={fp_ladder[k]!r} control={fp_sens[k]!r}"
                  for k in FINGERPRINT_KEYS if fp_ladder.get(k) != fp_sens.get(k)]
    if sensitivity is None:
        gates.append(_gate("G5", "WITHHELD", None,
                           "out/results_sensitivity.json (missing)",
                           reason="no sensitivity artefact: the negative control "
                                  "was not run, so it cannot have held"))
    else:
        controls = [r for r in sensitivity.get("sensitivity", [])
                    if r.get("slip") == "free_slip"]
        if mismatches:
            gates.append(_gate("G5", "WITHHELD", None,
                               "out/results_sensitivity.json:free_slip",
                               reason="control run does not share the ladder's "
                                      "case configuration: " + "; ".join(mismatches)))
        elif len(controls) != 1:
            gates.append(_gate("G5", "WITHHELD", None,
                               "out/results_sensitivity.json:free_slip",
                               reason=f"expected exactly one free-slip control, "
                                      f"found {len(controls)}"))
        else:
            a_max = controls[0]["metrics"]["amplitude_max_over_U0"]
            gates.append(_gate(
                "G5", "PASS" if a_max <= PREDECLARED["G5"]["threshold"] else "FAIL",
                a_max, "out/results_sensitivity.json:sensitivity[slip=free_slip]",
                label=controls[0]["label"]))

    counts = {s.lower(): sum(1 for g in gates if g["status"] == s)
              for s in ("PASS", "FAIL", "WITHHELD")}
    if consistency:
        overall = "FAIL"
    elif counts["fail"]:
        overall = "FAIL"
    elif counts["withheld"]:
        overall = "WITHHELD"
    else:
        overall = "PASS"

    # ---- informational controls: measured, but not predeclared as gates -----
    informational = []
    for r in (sensitivity or {}).get("sensitivity", []):
        if r.get("slip") == "free_slip":
            continue
        informational.append({
            "label": r.get("label"),
            "amplitude_l2_half_space_pct": r["metrics"]["amplitude_l2_half_space_pct"],
            "amplitude_max_over_U0": r["metrics"]["amplitude_max_over_U0"],
            "role": _control_role(r.get("label", "")),
        })

    finest = rec[-1]
    return {
        "anchor": ANCHOR,
        "case": CASE,
        "design": "benchmarks/stokes_second/DESIGN.md",
        "graded_artifact": "out/results_ladder.json",
        "artifacts": provenance or {},
        "overall": overall,
        "counts": counts,
        "all_predeclared_gates_pass": overall == "PASS",
        "gates": gates,
        "recomputed_per_rung": rec,
        "consistency": {"ok": not consistency, "problems": consistency},
        "fingerprint": {"ladder": fp_ladder, "sensitivity": fp_sens,
                        "match": not mismatches, "mismatches": mismatches},
        "informational_controls": informational,
        "graded_run": {
            "resolution_delta_over_dx": finest["resolution_delta_over_dx"],
            "primary_metric": PREDECLARED["G1"]["metric"],
            "primary_value_pct": finest["amplitude_l2_half_space_pct"],
        },
        "repro_cmd": ladder.get("repro_cmd"),
        "interpretation": (
            "PASS qualifies the viscous operator and the moving no-slip mirror "
            "wall for smooth wall-driven solenoidal shear at Ma ~ 2e-3 on a "
            "regular lattice with delta/dx >= 8, h/dx in [1.3, 1.9], H/delta = 6. "
            "It does not qualify free-surface, high-Ma, turbulent or "
            "pressure-gradient boundary-layer flow."),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

CONTROL_ROLES = {
    "sensitivity_U0_half":
        "linearity: halving U0 must leave A/U0 unchanged (measured, not a "
        "predeclared gate)",
    "sensitivity_h_factor_1.9":
        "smoothing sensitivity: h/dx = 1.9 must stay inside the gates "
        "(measured, not a predeclared gate)",
    "sensitivity_static_wall_control":
        "still-wall control: a non-oscillating wall must not drive flow "
        "(measured, not a predeclared gate)",
}


def _control_role(label: str) -> str:
    return CONTROL_ROLES.get(label, "control run (not a predeclared gate)")


def _fmt(value, nd=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.{nd}g}"
    return str(value)


def render_markdown(v: dict) -> str:
    L = []
    A = L.append
    A("# Stokes' second problem — case verdict (`CASE-SPH-STOKES-2ND-PROBLEM`)\n")
    A("*Generated by `benchmarks/stokes_second/case_verdict.py` from the published "
      "artefacts. Do not hand-edit. The predeclared gates are in "
      "`benchmarks/stokes_second/DESIGN.md`, written before the ladder was run; the "
      "gate inputs are the published metrics, each of which is cross-checked "
      "against an independent re-derivation from that rung's persisted rows.*\n")
    A(f"**Overall: {v['overall']}** — {v['counts']['pass']} pass, "
      f"{v['counts']['fail']} fail, {v['counts']['withheld']} withheld "
      "(a withheld gate is missing evidence, not a pass).\n")
    A("| gate | statement | threshold | measured | status | evidence |")
    A("|---|---|---|---|---|---|")
    for g in v["gates"]:
        m = g["measured"]
        if g["id"] == "G6" and isinstance(m, dict):
            m = (f"L2 = {['%.4f' % x for x in m['l2_pct']]} % on "
                 f"{m['resolutions']} (complete: {_fmt(m['complete'])})")
        elif g["id"] == "G5":
            m = (f"{m:.3g} of $U_0$ — exactly zero at the persisted 6 significant "
                 "digits" if m == 0 else f"{m:.3g} of $U_0$")
        else:
            m = f"{_fmt(m)} {g['unit']}"
        thr, unit = g["threshold"], g["unit"]
        thr_s = f"{'<=' if g['direction'] == 'le' else '>='} {thr:g} {unit}"
        A(f"| {g['id']} | {g['statement']} | {thr_s} | {m} | **{g['status']}** | "
          f"`{g['evidence']}` |")
    A("")
    A("## Per-rung re-derivation (from each rung's own `rows`)\n")
    A("| $\\delta/dx$ | rows in layer | G1 L2 % | G2 worst row % | G3 decay err % | "
      "$\\delta_{fit}/\\delta$ | G4 finite-depth L2 % |")
    A("|---|---|---|---|---|---|---|")
    for r in v["recomputed_per_rung"]:
        A(f"| {r['resolution_delta_over_dx']} | {r['n_rows_stokes_layer']} | "
          f"{r['amplitude_l2_half_space_pct']:.4f} | "
          f"{r['amplitude_worst_row_half_space_pct']:.4f} | "
          f"{r['decay_rate_error_pct']:+.4f} | {r['delta_ratio']:.5f} | "
          f"{r['amplitude_l2_finite_depth_pct']:.4f} |")
    A("")
    A(f"Stored `metrics` cross-check: "
      f"{'consistent with the rows' if v['consistency']['ok'] else '**INCONSISTENT**'}.")
    if v["consistency"]["problems"]:
        for p in v["consistency"]["problems"]:
            A(f"* {p}")
    A("")
    A("## Evidence and controls\n")
    for name, meta in v["artifacts"].items():
        A(f"* `out/{name}` — sha256 `{meta['sha256'][:16]}…`, {meta['bytes']} bytes")
    fp = v["fingerprint"]
    A(f"* case fingerprint (ladder vs control block): "
      f"{'match on all ' + str(len(fp['ladder'])) + ' keys' if fp['match'] else 'MISMATCH'}")
    if fp["mismatches"]:
        for m in fp["mismatches"]:
            A(f"  * {m}")
    for c in v["informational_controls"]:
        A(f"* `{c['label']}`: L2 = {c['amplitude_l2_half_space_pct']:.4f} %, "
          f"$A_{{max}}/U_0$ = {c['amplitude_max_over_U0']:.3g} — {c['role']}")
    A("")
    A("## Interpretation limits\n")
    A(v["interpretation"])
    A("")
    A("Reproduction: `" + (v.get("repro_cmd") or "n/a") + "`")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--json", type=Path, default=VERDICT_JSON)
    ap.add_argument("--doc", type=Path, default=VERDICT_DOC)
    ap.add_argument("--write", action="store_true",
                    help="write the JSON verdict and the generated markdown page")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 0 only when the verdict is PASS")
    args = ap.parse_args(argv)

    ladder, sensitivity, provenance = load_artifacts(args.out_dir)
    verdict = case_verdict(ladder, sensitivity, provenance=provenance)

    for g in verdict["gates"]:
        print(f"{g['id']:4s} {g['status']:8s} {_fmt(g['measured'])}"
              + (f"  [{g['reason']}]" if g.get("reason") else ""))
    print(f"overall: {verdict['overall']}  ({verdict['counts']})")
    if not verdict["consistency"]["ok"]:
        for p in verdict["consistency"]["problems"]:
            print(f"consistency: {p}", file=sys.stderr)

    if args.write:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=1, default=_jsonable) + "\n")
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(render_markdown(verdict))
        print(f"wrote {args.json}")
        print(f"wrote {args.doc}")

    if args.check:
        return {"PASS": 0, "FAIL": 1, "WITHHELD": 2}[verdict["overall"]]
    return 0


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())
