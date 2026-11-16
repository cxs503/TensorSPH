"""CASE-POISEUILLE-SPH -- the run-length axis at the *certified* Mach and dt.

Why this study exists
---------------------
``study_ma_envelope.py`` ran seven points to decide whether the anchor's <5 %
certificate carries a Mach envelope.  Its verdict came out MIXED_MA_TIME and
left one item explicitly open (``out/ma_envelope.json:verdict.open_item``):

    "Two candidates survive and are not separated by the present sweep:
     (i) Ma-dependent growth of particle disorder, (ii) the doubled dt that
     comes with the lowered c_s (P3/P4 run at dt = 3.25e-4 vs 1.63e-4 for P0).
     The discriminating run is c_s = 20 with t_end = 2.0 ..."

That run was never executed, and the ``psi_index`` the study reported as a
post-hoc correlation over its own seven points was never tested out of sample.
This module does two things about it:

* ``existing_evidence()`` -- settles, **from the committed seven-point artifact
  alone**, everything the existing data can settle: the two surviving
  candidates are *not* symmetric.  Candidate (ii) is "the doubled dt"; but P3
  and P4 were run at the *same* ``dt = 3.25e-4`` and differ only in run length,
  and one passes while the other fails.  A dt change therefore cannot be a
  sufficient cause of the failure.  The numbers are recomputed here from
  ``out/ma_envelope.json`` -- nothing is typed in.

* the run-length sweep ``POINTS`` below -- the *missing* axis: hold the
  certified operating point (``g = 9.81``, ``c_s = 20``, hence the certified Ma
  and the certified ``dt``) and vary only ``t_end``.  The script is shardable
  and keeps one JSON part per point, so the three runs can be executed
  independently (local 120 s cap / scheduler queue) and merged with
  ``--collect`` without any shared write.

Geometry / parameters (identical to ``run.py`` and to the Ma envelope study)
---------------------------------------------------------------------------
H = 0.1 m, nu = 0.01 m^2/s, rho0 = 1000 kg/m^3, c_s = 20 m/s, g = 9.81 m/s^2,
12 periodic columns, h = 1.3 dx, kernel support 2h, H/dx = 12
(dx = 8.3333e-3 m), averaging window ``t_average`` = 0.2 s at the end of each
run.  Analytic reference: u(y) = g (H-y) y / (2 nu), u_max = g H^2/(8 nu) =
1.22625 m/s, tau_w = rho g H/2 (White, *Viscous Fluid Flow* 3rd ed. §3-2.2;
Batchelor §4.2).

Predeclared points (certified ``g`` and ``c_s``: Ma and dt are P0's)
--------------------------------------------------------------------
=====================  ======  =======  =================  =================
point                  t_end   steps    Psi (analytic)     role
=====================  ======  =======  =================  =================
T1_runlen_2s           2.0      12308    18.04 (pass zone)  the run the Ma study named
T2_runlen_4s           4.0      24616    36.09 (fail zone)  out-of-sample Psi point
T3_runlen_8s           8.0      49231    72.18 (far fail)   Psi extrapolation
=====================  ======  =======  =================  =================

Predeclared hypotheses and decision rule (written before any run)
-----------------------------------------------------------------
H_time    The band breaks with accumulated run length alone at the certified
          Ma and dt  ->  T1 FAIL.
H_psi     The boundary is the accumulated index Psi (in-sample band: passing
          at Psi <= 20.3, failing at Psi >= 32.6)  ->  T1 PASS, T2 FAIL.
H_ma_dt   Failure needs Ma (or dt) above the certified values  ->  T1 PASS,
          T2 PASS.

Decision:
  T1 FAIL                        -> H_TIME
  T1 PASS, T2 FAIL               -> H_PSI
  T1 PASS, T2 PASS, T3 FAIL      -> H_MA_DT (bounded: valid for t_end <= 4 s)
  T1 PASS, T2 PASS, T3 PASS      -> H_MA_DT (strong, up to t_end = 8 s)
  T1 PASS, T2 PASS, T3 not run   -> H_MA_DT (established for t_end <= 4 s)
  anything else                  -> UNEXPLAINED / NOT_RUN

No threshold is moved after the fact: the 5 % band is the case-pool value, the
Psi in-sample band is read out of the committed ``ma_envelope.json``, and the
predictions above are functions of ``t_end`` only.

Metrics (identical to the Ma study, so the two are comparable)
--------------------------------------------------------------
m1 u_max error vs g H^2/(8 nu)                              [%]
m2 row-RMS profile error, interior rows 0.1 <= y/H <= 0.9    [%]
m3 worst wall-shear error vs rho g H / 2                     [%]
m4 steady_drift (relative u_max change over the last 0.25 s)
m5 max_row_std_over_umax
m6 max_mach reached

Usage
-----
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_time_axis.py \
        --points T1_runlen_2s                     # one point per call (shardable)
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_time_axis.py \
        --collect                                 # merge parts, write verdict
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_time_axis.py \
        --predictions                             # print the declared table only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.poiseuille_mirror.study_ma_envelope import (  # noqa: E402
    HEIGHT, RESOLUTION, TOL_PCT, _backfill, run_point)

OUT = REPO / "benchmarks" / "poiseuille_mirror" / "out"
PARTS = OUT / "time_axis"
MA_ARTIFACT = OUT / "ma_envelope.json"

VISCOSITY = 0.01
SOUND_SPEED = 20.0
BODY_FORCE = 9.81
DX = HEIGHT / RESOLUTION
U_MAX_ANALYTIC = BODY_FORCE * HEIGHT ** 2 / (8.0 * VISCOSITY)
PSI_PER_SECOND = U_MAX_ANALYTIC ** 2 / (SOUND_SPEED * DX)
# the certified `dt` of the committed ladder (P0: 6154 steps for t_end = 1 s)
CERTIFIED_DT = 1.625e-4

POINTS = [
    {"id": "T1_runlen_2s", "g": BODY_FORCE, "cs": SOUND_SPEED, "t_end": 2.0,
     "role": "the run named by the Ma-envelope open item: run length doubled "
             "at the certified Ma and dt"},
    {"id": "T2_runlen_4s", "g": BODY_FORCE, "cs": SOUND_SPEED, "t_end": 4.0,
     "role": "out-of-sample Psi point: predicted fail if the accumulated "
             "index, and not Ma, sets the boundary"},
    {"id": "T3_runlen_8s", "g": BODY_FORCE, "cs": SOUND_SPEED, "t_end": 8.0,
     "role": "Psi extrapolation, 2.2x beyond the in-sample failing point"},
]


def psi_predicted(t_end: float) -> float:
    """Psi from the *analytic* u_max, i.e. the prediction made before running."""
    return U_MAX_ANALYTIC ** 2 * t_end / (SOUND_SPEED * DX)


def by_id(pid: str) -> dict:
    for p in POINTS:
        if p["id"] == pid:
            return p
    raise SystemExit(f"unknown point {pid}; known: {[p['id'] for p in POINTS]}")


def predictions() -> dict:
    """The declared prediction table (a function of t_end only)."""
    return {p["id"]: {"t_end": p["t_end"], "psi_analytic": psi_predicted(p["t_end"])}
            for p in POINTS}


def _ma_doc() -> dict:
    if not MA_ARTIFACT.exists():
        return {}
    return json.loads(MA_ARTIFACT.read_text())


def in_sample_band() -> dict:
    """Read the Psi band of the seven-point Ma sweep out of its committed artifact."""
    doc = _ma_doc()
    out = {"artifact": str(MA_ARTIFACT.relative_to(REPO)), "available": bool(doc)}
    if not doc:
        return out
    v = doc.get("verdict", {})
    out.update({
        "psi_passing_max": v.get("psi_passing_max"),
        "psi_failing_min": v.get("psi_failing_min"),
        "n_in_sample": len(doc.get("points", [])),
        "status": v.get("status"),
        "open_item": v.get("open_item"),
    })
    return out


def existing_evidence() -> dict:
    """What the committed seven-point sweep *already* settles.

    The open item lists two surviving causes for the P4 failure: (i) Ma-driven
    particle disorder and (ii) the doubled ``dt`` that comes with ``c_s = 10``.
    Cause (ii) is testable inside the existing data without any new run: P3 and
    P4 were both run at ``c_s = 10`` (same ``dt``) and differ only in ``t_end``
    -- if a doubled ``dt`` were sufficient, both would have to fail.

    Every number below is recomputed from ``out/ma_envelope.json``.
    """
    doc = _ma_doc()
    pts = {p["id"]: _backfill(p) for p in doc.get("points", [])}
    need = ("P0_baseline", "P3_cert_re_high_ma", "P4_cert_re_high_ma_long")
    if not all(k in pts for k in need):
        return {"available": False,
                "reason": "committed ma_envelope.json lacks P0/P3/P4"}
    p0, p3, p4 = (pts[k] for k in need)
    dt3, dt4 = float(p3["dt"]), float(p4["dt"])
    same_dt = abs(dt4 / dt3 - 1.0) < 1e-9
    evidence = {
        "available": True,
        "artifact": str(MA_ARTIFACT.relative_to(REPO)),
        "dt_p3": dt3, "dt_p4": dt4, "dts_equal": same_dt,
        "certified_dt": CERTIFIED_DT,
        "p3": {k: p3[k] for k in ("t_end", "dt", "psi_index", "max_mach",
                                  "reynolds_number", "profile_rms_interior_pct",
                                  "wall_shear_worst_abs_pct", "u_max_error_pct",
                                  "steady_drift", "passes_all_gates")},
        "p4": {k: p4[k] for k in ("t_end", "dt", "psi_index", "max_mach",
                                  "reynolds_number", "profile_rms_interior_pct",
                                  "wall_shear_worst_abs_pct", "u_max_error_pct",
                                  "steady_drift", "passes_all_gates")},
        "p0": {k: p0[k] for k in ("t_end", "dt", "psi_index", "max_mach",
                                  "reynolds_number", "profile_rms_interior_pct",
                                  "wall_shear_worst_abs_pct", "u_max_error_pct",
                                  "passes_all_gates")},
        "run_length_ratio_p4_over_p3": p4["t_end"] / p3["t_end"],
        "psi_ratio_p4_over_p3": p4["psi_index"] / p3["psi_index"],
        "rms_amplification_p4_over_p3": (p4["profile_rms_interior_pct"]
                                         / p3["profile_rms_interior_pct"]),
        "shear_amplification_p4_over_p3": (p4["wall_shear_worst_abs_pct"]
                                           / p3["wall_shear_worst_abs_pct"]),
        "dt_over_certified": dt3 / CERTIFIED_DT,
    }
    if same_dt and p3["passes_all_gates"] and not p4["passes_all_gates"]:
        evidence["dt_candidate"] = "ELIMINATED_AS_SUFFICIENT_CAUSE"
        evidence["dt_candidate_statement"] = (
            "P3 and P4 share dt = %.3e s (%.2fx the certified dt) and differ only "
            "in run length (%g s vs %g s); P3 passes every gate and P4 fails "
            "profile RMS and wall shear. A doubled dt is therefore not sufficient "
            "to break the band: within this pair the discriminating variable is "
            "the run length (Psi %.1f -> %.1f)."
            % (dt3, dt3 / CERTIFIED_DT, p3["t_end"], p4["t_end"],
               p3["psi_index"], p4["psi_index"]))
    else:  # pragma: no cover - would mean the committed data changed
        evidence["dt_candidate"] = "NOT_SEPARABLE"
        evidence["dt_candidate_statement"] = (
            "The committed P3/P4 pair does not separate dt from run length.")
    evidence["still_open"] = (
        "The same run length has never been run at the *certified* Ma and dt, so "
        "the failure cannot yet be attributed to accumulated time alone rather "
        "than to the higher Ma; that is exactly point T1 (t_end = 2.0 s, "
        "c_s = 20), which this module declares and grades.")
    return evidence


def load_parts() -> list[dict]:
    parts = {}
    for f in sorted(PARTS.glob("*.json")):
        row = json.loads(f.read_text())
        parts[row["id"]] = _backfill(row)
    return [parts[p["id"]] for p in POINTS if p["id"] in parts]


def _detail(row: dict) -> dict:
    return {
        "id": row["id"], "t_end": row["t_end"], "steps": row["steps"],
        "dt": row["dt"], "n_fluid": row["n_fluid"],
        "psi_index": row["psi_index"],
        "psi_analytic_prediction": psi_predicted(row["t_end"]),
        "max_mach": row["max_mach"], "reynolds_number": row["reynolds_number"],
        "u_max_measured": row["u_max_measured"],
        "u_max_error_pct": row["u_max_error_pct"],
        "profile_rms_interior_pct": row["profile_rms_interior_pct"],
        "profile_rms_all_rows_pct": row["profile_rms_all_rows_pct"],
        "wall_shear_worst_abs_pct": row["wall_shear_worst_abs_pct"],
        "wall_slip_intercept_over_umax": row["wall_slip_intercept_over_umax"],
        "steady_drift": row["steady_drift"],
        "max_row_std_over_umax": row["max_row_std_over_umax"],
        "gates": row["gates"],
        "passes_all_gates": row["passes_all_gates"],
        "wall_clock_s": row["wall_clock_s"],
    }


def verdict(rows: list[dict]) -> dict:
    """Apply the predeclared rule.  Reads stored rows only; never re-tunes."""
    have = {r["id"]: r for r in rows}
    band = in_sample_band()
    out = {
        "tolerance_pct": TOL_PCT,
        "resolution_H_over_dx": RESOLUTION,
        "rule_version": "v1",
        "operating_point": {
            "g": BODY_FORCE, "nu": VISCOSITY, "cs": SOUND_SPEED,
            "H": HEIGHT, "dx": DX, "u_max_analytic": U_MAX_ANALYTIC,
            "psi_per_second": PSI_PER_SECOND,
            "certified_dt": CERTIFIED_DT},
        "predictions": predictions(),
        "in_sample_band": band,
        "declared_points": [p["id"] for p in POINTS],
        "available_points": sorted(have),
        "points": {r["id"]: _detail(r) for r in rows},
    }
    if not rows:
        out["status"] = "NOT_RUN"
        out["statement"] = (
            "None of the declared run-length points has an artifact on disk, so "
            "this study makes no claim about the run-length axis. What the "
            "committed seven-point sweep already settles is recorded separately "
            "in existing_evidence(): the doubled-dt candidate is eliminated as a "
            "sufficient cause, and the certified-Ma long run (T1) remains the "
            "missing discriminator.")
        return out

    def passed(pid):
        return have[pid]["passes_all_gates"] if pid in have else None

    t1, t2, t3 = passed("T1_runlen_2s"), passed("T2_runlen_4s"), passed("T3_runlen_8s")

    lo = band.get("psi_passing_max")
    hi = band.get("psi_failing_min")
    checks = {}
    if lo is not None and hi is not None:
        for r in rows:
            psi = r["psi_index"]
            pred = ("pass" if psi <= lo else "fail" if psi >= hi else "between")
            checks[r["id"]] = {"psi": psi, "predicted": pred,
                               "observed": "pass" if r["passes_all_gates"] else "fail",
                               "agrees": (pred == "between" or
                                          pred == ("pass" if r["passes_all_gates"] else "fail"))}
        out["psi_oos_checks"] = checks

    if t1 is None:
        out["status"] = "INCOMPLETE"
        out["statement"] = (
            "T1 (the discriminating run named by the Ma study) has not been "
            "executed; the run-length axis is not decided.")
        return out
    if t1 is False:
        out["status"] = "H_TIME_SUPPORTED"
        out["statement"] = (
            "At the certified Ma and dt the band already breaks once the run "
            "length reaches P4's t_end, so the earlier 'Ma envelope' reading was "
            "a run-length effect. The Psi index is NOT validated by this outcome.")
        return out
    if t2 is None:
        out["status"] = "H_MA_DT_SUPPORTED_PARTIAL"
        out["statement"] = (
            "T1 passes, so a doubled dt is not sufficient (consistent with "
            "existing_evidence()) and the run length alone does not break the "
            "band by t_end = 2.0 s at the certified Ma. T2 (t_end = 4 s, "
            "Psi ~ 36, out of sample) has not been executed, so the statement is "
            "bounded to t_end <= 2 s and Psi is untested out of sample.")
        return out
    if t2 is False:
        out["status"] = "H_PSI_SUPPORTED"
        out["statement"] = (
            "T1 (Psi ~ 18) passes and T2 (Psi ~ 36) fails at identical Ma, Re "
            "and dt, i.e. the boundary follows the accumulated index Psi and not "
            "the Mach number. This is a two-point bracketing, not a fitted law.")
        return out
    if t3 is True:
        out["status"] = "H_MA_DT_SUPPORTED_STRONG"
        out["statement"] = (
            "Every run length at the certified Ma and dt stays inside the band "
            "(t_end up to 8 s, Psi up to ~72). The P4 failure must therefore be "
            "attributed to c_s = 10 (Ma 0.123 / dt 3.25e-4). The Psi index is "
            "refuted as a stability criterion.")
        return out
    if t3 is False:
        out["status"] = "H_MA_DT_SUPPORTED_BOUNDED"
        out["statement"] = (
            "The band survives t_end = 4 s (Psi ~ 36) at the certified Ma and dt "
            "but breaks by t_end = 8 s (Psi ~ 72). P4's failure at t_end = 2 s is "
            "still attributable to c_s, but the run-length envelope is finite: "
            "the certificate is valid for t_end <= 4 s at Ma 0.061.")
        return out
    out["status"] = "H_MA_DT_SUPPORTED_PARTIAL"
    out["statement"] = (
        "T1 and T2 pass at the certified Ma and dt; T3 was not executed, so the "
        "statement is bounded to t_end <= 4 s.")
    return out


def collect() -> dict:
    rows = load_parts()
    doc = {
        "case": "plane_poiseuille_channel_mirror_wall",
        "anchor": "CASE-POISEUILLE-SPH",
        "study": "run-length axis at the certified Ma and dt (closes the "
                 "ma_envelope open item; out-of-sample test of Psi)",
        "resolution_H_over_dx": RESOLUTION,
        "tolerance_pct": TOL_PCT,
        "hypotheses": "H_TIME / H_PSI / H_MA_DT (see module docstring)",
        "declared_points": [dict(p) for p in POINTS],
        "points": [_detail(r) for r in rows],
        "raw_rows": rows,
        "evidence": existing_evidence(),
        "verdict": verdict(rows),
    }
    artifact = OUT / "time_axis.json"
    artifact.write_text(json.dumps(doc, indent=1))
    print(f"[time-axis] wrote {artifact}")
    print(f"[time-axis] verdict {doc['verdict']['status']}")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", default=[],
                    help="point ids to run (one JSON part each)")
    ap.add_argument("--collect", action="store_true",
                    help="merge the parts into out/time_axis.json and grade")
    ap.add_argument("--predictions", action="store_true",
                    help="print the declared prediction table and exit")
    ap.add_argument("--evidence", action="store_true",
                    help="print the evidence that the committed sweep settles")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    if args.predictions:
        print(json.dumps(predictions(), indent=1))
        return 0
    if args.evidence:
        print(json.dumps(existing_evidence(), indent=1))
        return 0

    PARTS.mkdir(parents=True, exist_ok=True)
    for pid in args.points:
        point = by_id(pid)
        t0 = time.time()
        row = run_point(point, viscosity=VISCOSITY, threads=args.threads,
                        device=args.device)
        row["study"] = "time_axis"
        (PARTS / f"{pid}.json").write_text(json.dumps(row, indent=1))
        print(f"[time-axis] {pid}: t_end={point['t_end']} "
              f"steps={row['steps']} psi={row['psi_index']:.2f} "
              f"rms={row['profile_rms_interior_pct']:.3f}% "
              f"{'PASS' if row['passes_all_gates'] else 'FAIL'} "
              f"({time.time() - t0:.0f}s)", flush=True)

    if args.collect or not args.points:
        collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())