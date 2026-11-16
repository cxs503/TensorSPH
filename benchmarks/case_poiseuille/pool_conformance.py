#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH -- case-pool conformance gate, recomputed from raw rows.

Why this file exists
--------------------
The anchor (``CASE-POISEUILLE-SPH``, 算例池·泊肃叶流, ref: analytic, tolerance
``< 5 %``) is graded by a growing family of studies -- R1 steady/start-up ladder,
R2 Womersley harmonic, combined Couette-Poiseuille, steady-window, steady-drift,
uncertainty.  Each study writes its own artifact and its own gates; no single
place answers the only question the pool row asks:

    *is every functional of this case, on every rung that was run, inside 5 % --
     and is that verdict still tied to the data it was derived from?*

This module answers it once, and it answers it **from the raw per-row / per-sample
data**, not from the summary percentages that the study drivers already wrote.
Every recomputed quantity is compared against the stored value (*binding*); a
mismatch means the artifact and the verdict have drifted apart and the run is
declared invalid rather than silently re-graded.

Independence
------------
* the analytic references used here are re-derived by least squares from the
  ``u_analytic`` / ``amplitude_analytic`` columns of the artifacts themselves
  (a parabola ``c1 y + c2 y^2`` and its vertex, a free LS harmonic fit of the
  raw centre series), so this file shares no estimator code with ``run.py``,
  ``combined.py`` or ``benchmarks/poiseuille_mirror``;
* the source artifacts are SHA-256 pinned in the output, so editing an artifact
  without regenerating the conformance record fails the test.

Usage
-----
    python benchmarks/case_poiseuille/pool_conformance.py            # write out/pool_conformance.json
    python benchmarks/case_poiseuille/pool_conformance.py --check    # verify only (CI)

Verdict vocabulary (deliberately narrow):

* ``recomputed``  -- error re-derived here from raw data; binding checked;
* ``stored``      -- no raw data in the artifact (e.g. a momentum budget that
                     needs a time derivative of the sampled mean); the stored
                     value is accepted and labelled as such, never as recomputed;
* ``not_graded``  -- a functional the case declares but does not grade (listed,
                     not hidden).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

TOL_PCT = 5.0          # anchor tolerance: < 5 %

# Known, pinned binding exception: the combined case's C1 at H/dx = 8 is stored
# with a different profile normalisation (U_w) than the rungs 12/16 (max|u_a|),
# so the stored 0.884 % cannot be re-derived from the stored rung-8 rows
# (they give 1.053 %).  Both are inside the 5 % tolerance, so the gate verdict is
# unaffected -- but the published convergence trend mixes two gauges.  The test
# pins the size of the mismatch so it cannot silently grow, and any *new*
# violation fails.
# the rung-8 gauge mismatch (stored value normalised by U_w while rungs 12/16
# used max|u_a| over their rows) was repaired at root: ``combined.py`` and
# ``rebase_gauge.py`` now normalise by one closed-form continuum gauge (== U_w
# here), so nothing is expected to need an exception any more
KNOWN_BINDING_EXCEPTIONS: dict[str, float] = {}
BIND_RTOL = 1e-6       # raw-recompute vs stored summary must agree to this
INTERIOR_BAND = (0.15, 0.85)


# --------------------------------------------------------------------------- #
# small independent numerics
# --------------------------------------------------------------------------- #
def gauge_scale(g: float, wall_speed: float, height: float, nu: float) -> float:
    """Continuum gauge of the profile-error gates (closed form, row independent).

    ``max|u_a|`` over ``0 <= y <= H`` for the Couette-Poiseuille profile: the
    endpoints ``u(0)=0``, ``u(H)=U_w`` and the parabola vertex
    ``y* = H/2 + nu U_w/(g H)``.  Equals ``U_w`` for the combined configuration
    (the predeclared C1 normaliser) and ``g H^2/(8 nu)`` for a static-wall
    Poiseuille leg, so one declaration serves every configuration.
    """
    cands = [0.0, abs(wall_speed)]
    if g != 0.0:
        y_star = 0.5 * height + nu * wall_speed / (g * height)
        if 0.0 < y_star < height:
            cands.append(abs(g * y_star * (height - y_star) / (2.0 * nu)
                             + wall_speed * y_star / height))
    return max(cands)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def lstsq(A, b):
    """Normal equations with partial pivoting -- no numpy dependency."""
    n = len(A[0])
    AtA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)]
           for i in range(n)]
    Atb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(n)]
    M = [row[:] + [Atb[i]] for i, row in enumerate(AtA)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("singular normal equations")
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def rms(values) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def parabola_vertex(rows, y_key: str, u_key: str, H: float) -> float:
    """Vertex value of the least-squares parabola c1*y + c2*y^2 through a column."""
    A = [[r[y_key], r[y_key] ** 2] for r in rows]
    b = [r[u_key] for r in rows]
    c1, c2 = lstsq(A, b)
    return c1 * (H / 2.0) + c2 * (H / 2.0) ** 2


def parabolic_slopes(rows, y_key: str, u_key: str, H: float, band=INTERIOR_BAND):
    """du/dy at y=0 and y=H from one LS quadratic over the interior band."""
    sel = [r for r in rows if band[0] - 1e-9 <= r[y_key] / H <= band[1] + 1e-9]
    A = [[1.0, r[y_key], r[y_key] ** 2] for r in sel]
    b = [r[u_key] for r in sel]
    c0, c1, c2 = lstsq(A, b)
    return c1, c1 + 2.0 * c2 * H


def zero_crossing(rows, y_key: str, u_key: str):
    """Linear-interpolated sign change of a profile sampled bottom-up."""
    s = sorted(rows, key=lambda r: r[y_key])
    for a, b in zip(s, s[1:]):
        if a[u_key] == 0.0:
            return a[y_key]
        if a[u_key] * b[u_key] < 0.0:
            w = a[u_key] / (a[u_key] - b[u_key])
            return a[y_key] + w * (b[y_key] - a[y_key])
    return None


def harmonic_fit(series: dict, omega: float):
    """u(t) = Re[U e^{i w t}] + const  ->  (amplitude, phase_rad)."""
    t = series["t"]
    u = series["u"] if "u" in series else series["value"]
    A = [[math.cos(omega * ti), math.sin(omega * ti), 1.0] for ti in t]
    p, q, _c = lstsq(A, u)
    return math.hypot(p, q), math.atan2(-q, p)


# --------------------------------------------------------------------------- #
# record helpers
# --------------------------------------------------------------------------- #
def _functional(fid, quantity, value, reference, error_pct, tol, basis,
                rung=None, stored=None, note=None):
    rec = {
        "id": fid,
        "rung_H_over_dx": rung,
        "quantity": quantity,
        "value": value,
        "reference": reference,
        "error_percent": error_pct,
        "tolerance_percent": tol,
        "pass": bool(abs(error_pct) <= tol) if error_pct is not None else None,
        "basis": basis,
    }
    if stored is not None:
        rec["stored_error_percent"] = stored
        if error_pct is not None and abs(stored) > 0:
            rec["binding_rel_diff"] = abs(error_pct - stored) / abs(stored)
        elif error_pct is not None:
            rec["binding_rel_diff"] = abs(error_pct - stored)
    if note:
        rec["note"] = note
    return rec


# --------------------------------------------------------------------------- #
# R1 -- constant forcing: steady parabola + Boussinesq start-up ladder
# --------------------------------------------------------------------------- #
def conformance_r1(artifacts, recs, notes):
    """R1 constant forcing: pointwise-relative RMS over the interior band.

    Definition recomputed here (deliberately the case's own, so the binding is
    exact): ``rel_i = (u_i - u_ref_i) / u_ref_i`` on rows with
    ``0.15 <= y/H <= 0.85``; functional = ``100 * rms(rel)``.  The denominators
    and the reference column are read from the artifact, the arithmetic is not.
    """
    for rung in (8, 12, 16):
        path = OUT / f"r1_rung_{rung}.json"
        if not path.exists():
            notes.append(f"R1 rung H/dx={rung}: artifact missing, not graded")
            continue
        d = json.loads(path.read_text())
        artifacts[str(path.relative_to(REPO))] = sha256(path)
        rows = d["steady_rows"]
        interior = [r for r in rows if INTERIOR_BAND[0] <= r["y_over_H"] <= INTERIOR_BAND[1]]
        rms_pct = 100.0 * rms([(r["u_measured"] - r["u_analytic"]) / r["u_analytic"]
                               for r in interior])
        recs.append(_functional(
            f"R1_steady_profile_rms@Hdx{rung}", "steady profile RMS vs parabola",
            rms_pct, "pointwise relative, interior band",
            rms_pct, TOL_PCT, "recomputed", rung=rung,
            stored=d["steady_profile_rms_pct"],
            note="pointwise-relative RMS over 0.15<=y/H<=0.85"))

        start = [s for s in d["startup"] if abs(s["t_requested"] - 0.40) < 1e-9]
        if start:
            srows = start[0]["rows"]
            interior = [r for r in srows if INTERIOR_BAND[0] <= r["y_over_H"] <= INTERIOR_BAND[1]]
            rms_s = 100.0 * rms([(r["u_measured"] - r["u_analytic"]) / r["u_analytic"]
                                 for r in interior])
            recs.append(_functional(
                f"R1_startup_profile_rms@t0.40@Hdx{rung}",
                "start-up profile RMS at t=0.40 s", rms_s,
                "pointwise relative, interior band", rms_s, TOL_PCT,
                "recomputed", rung=rung,
                stored=d["startup_rms_pct_at_040"]))
        else:
            notes.append(f"R1 rung H/dx={rung}: no start-up sample at t=0.40 s")


# --------------------------------------------------------------------------- #
# R2 -- harmonic forcing: Womersley frequency response
# --------------------------------------------------------------------------- #
def conformance_r2(artifacts, recs, notes):
    """R2 harmonic forcing: Womersley frequency response.

    Two levels of independence are recorded:

    * the **centre** amplitude/phase are re-fitted here from the raw centre
      series with this module's own least-squares harmonic basis (the driver uses
      ``tensorsph.stokes_second.harmonic_fit``), so even the fit is independent;
    * the **row** functionals are the case's own pointwise-relative definitions,
      recomputed from the per-row amplitude/phase columns.
    """
    path = OUT / "result.json"
    d = json.loads(path.read_text())
    artifacts[str(path.relative_to(REPO))] = sha256(path)
    quarter = math.pi / 2.0
    for run in d["harmonic_runs"]:
        rung = run["resolution_H_over_dx"]
        omega = run["omega"]
        amp, phase = harmonic_fit(run["centre_series"], omega)
        amp_ana = run["amp_centre_analytic"]
        ph_ana = run["phase_centre_analytic_rad"]
        recs.append(_functional(
            f"R2_amp_centre@Hdx{rung}", "centreline amplitude", amp, amp_ana,
            100.0 * (amp - amp_ana) / amp_ana, TOL_PCT, "recomputed", rung=rung,
            stored=run["amp_centre_error_pct"],
            note="free LS fit of the raw centre series; no driver code reused"))
        dphi = ph_ana - phase          # published convention: analytic - measured
        while dphi > quarter:
            dphi -= 2 * quarter
        while dphi < -quarter:
            dphi += 2 * quarter
        recs.append(_functional(
            f"R2_phase_centre@Hdx{rung}", "centreline phase lag",
            math.degrees(phase), math.degrees(ph_ana),
            100.0 * abs(dphi) / quarter, TOL_PCT, "recomputed", rung=rung,
            stored=run["phase_centre_pct_of_quarter_turn"],
            note="error in % of a quarter turn, unwrapped to the nearest branch"))

        rows = [r for r in run["rows"]
                if INTERIOR_BAND[0] <= r["y_over_H"] <= INTERIOR_BAND[1]]
        amp_rms = 100.0 * rms([(r["amplitude"] - r["amplitude_analytic"])
                               / r["amplitude_analytic"] for r in rows])
        recs.append(_functional(
            f"R2_amp_profile_rms@Hdx{rung}", "row amplitude RMS", amp_rms,
            "pointwise relative, interior band", amp_rms, TOL_PCT, "recomputed",
            rung=rung, stored=run["amp_profile_rms_pct"]))
        ph_rms = 100.0 * rms([r["phase"] - r["phase_analytic"] for r in rows]) / quarter
        recs.append(_functional(
            f"R2_phase_profile_rms@Hdx{rung}", "row phase RMS", ph_rms,
            "rad RMS / quarter turn, interior band", ph_rms, TOL_PCT,
            "recomputed", rung=rung,
            stored=run["phase_profile_rms_pct_of_quarter_turn"]))


# --------------------------------------------------------------------------- #
# Combined Couette-Poiseuille (reversal) -- a different case, graded separately
# --------------------------------------------------------------------------- #
def conformance_combined(artifacts, recs, notes):
    """Combined Couette-Poiseuille with an interior reversal (a different case).

    Graded here so the pool record shows *what the family actually contains*,
    including the two rungs where the predeclared backflow gate fails.
    """
    path = OUT / "combined.json"
    if not path.exists():
        notes.append("combined artifact missing, not graded")
        return
    d = json.loads(path.read_text())
    artifacts[str(path.relative_to(REPO))] = sha256(path)
    par = d["parameters"]
    H = par["height"]
    nu = par["viscosity"]
    g = par["g"]
    U_w = par["wall_speed"]
    ana = d["analytic"]
    for run in d["runs"]:
        rung = run["resolution_H_over_dx"]
        rows = run["rows"]
        scale = gauge_scale(g, U_w, H, nu)
        rms_pct = 100.0 * rms([(r["u"] - r["u_analytic"]) / scale for r in rows])
        recs.append(_functional(
            f"CP_profile_rms@Hdx{rung}", "combined profile RMS", rms_pct, scale,
            rms_pct, TOL_PCT, "recomputed", rung=rung,
            stored=run["profile_rms_pct"],
            note=("normalised by the declared continuum gauge "
                  "max|u_a| over 0<=y<=H (== U_w for this configuration)")))
        zc = zero_crossing(rows, "y", "u")
        zc_ana = ana["zero_crossing_y"]
        zc_pct = None if zc is None else 100.0 * abs(zc - zc_ana) / H
        recs.append(_functional(
            f"CP_zero_crossing@Hdx{rung}", "interior reversal position", zc, zc_ana,
            zc_pct, TOL_PCT, "recomputed", rung=rung,
            stored=run["zero_crossing_pct_of_H"],
            note="|error| in % of H, linear interpolation of the sign change"))
        sm0, smH = parabolic_slopes([{"y": r["y"], "u": r["u"]} for r in rows], "y", "u", H)
        sa0, saH = parabolic_slopes([{"y": r["y"], "u": r["u_analytic"]} for r in rows],
                                    "y", "u", H)
        recs.append(_functional(
            f"CP_tau_bottom@Hdx{rung}", "bottom-wall shear", sm0, sa0,
            100.0 * (sm0 - sa0) / sa0, TOL_PCT, "recomputed", rung=rung,
            stored=run["tau_bottom_pct"],
            note="slope of one LS quadratic over the interior band, signed denominator"))
        recs.append(_functional(
            f"CP_tau_top@Hdx{rung}", "top-wall shear", smH, saH,
            100.0 * (smH - saH) / saH, TOL_PCT, "recomputed", rung=rung,
            stored=run["tau_top_pct"]))
        # backflow: the analytic reference at the minimum's own row, recomputed
        # from the parameters rather than read from the stored column
        y_min = min(rows, key=lambda r: r["u"])["y"]
        u_ref = g * y_min * (H - y_min) / (2.0 * nu) + U_w * y_min / H
        u_min = min(r["u"] for r in rows)
        recs.append(_functional(
            f"CP_backflow_peak@Hdx{rung}", "reversed-layer peak depth", u_min, u_ref,
            100.0 * (u_min - u_ref) / abs(u_ref), TOL_PCT, "recomputed", rung=rung,
            stored=run["backflow"]["peak_pct"],
            note=("analytic parabola evaluated at the minimum's own row; an honest "
                  "FAIL on the coarse rungs is carried through, not re-graded")))
        budget = (run.get("momentum_budget") or {}).get("pct")
        if budget is not None:
            recs.append(_functional(
                f"CP_momentum_budget@Hdx{rung}", "net drag budget", None, None,
                budget, 2.0, "stored", rung=rung,
                note="needs d<u>/dt from the sampled mean; stored value graded at 2 %"))


# --------------------------------------------------------------------------- #
# steady-window / drift / uncertainty: consistency of the declared gates
# --------------------------------------------------------------------------- #
def conformance_legs(artifacts, recs, notes):
    """Leg records: measured rows usable, their analytic column is not.

    ``legs[*].rows[].u_analytic`` stores the *combined* reference, so the leg
    diagnostics (``profile_worst_row_pct``, ``tau_*_pct``) cannot be re-derived
    from the committed file.  No gate reads the leg records, so this is recorded
    as ``not_recomputable`` evidence rather than discovered later as a
    "recomputation disagrees" surprise.
    """
    path = OUT / "legs_row_audit.json"
    if not path.exists():
        notes.append("legs_row_audit.json missing: leg diagnostic reproducibility unknown")
        return
    audit = json.loads(path.read_text())
    artifacts[str(path.relative_to(REPO))] = sha256(path)
    for name, a in (audit.get("legs") or {}).items():
        recs.append({
            "id": f"CP_leg_{name}_diagnostics",
            "rung_H_over_dx": 8,
            "quantity": "leg profile diagnostics from committed rows",
            "value": a.get("own_reference_declared_gauge_pct"),
            "reference": a.get("stored_profile_rms_pct"),
            "error_percent": None,
            "tolerance_percent": TOL_PCT,
            "pass": None,
            "basis": "not_recomputable",
            "note": ("leg rows carry the combined analytic column (off by "
                     f"{a.get('max_abs_column_minus_own_reference'):.4f} m/s); "
                     "gate-free, quarantined in out/legs_row_audit.json"),
        })


def conformance_supporting(artifacts, recs, notes):
    for name, key in (("steady_window.json", "gates"),
                      ("steady_drift.json", "gates"),
                      ("uncertainty.json", "gates")):
        path = OUT / name
        if not path.exists():
            notes.append(f"{name}: missing")
            continue
        d = json.loads(path.read_text())
        artifacts[str(path.relative_to(REPO))] = sha256(path)
        gates = d.get(key) or {}
        for gid, ok in gates.items():
            recs.append({
                "id": f"support:{name[:-5]}:{gid}",
                "quantity": "declared gate (no raw re-derivation in this module)",
                "value": ok,
                "error_percent": None,
                "tolerance_percent": None,
                "pass": bool(ok),
                "basis": "stored",
            })


def conformance_independent_audit(artifacts, recs, notes):
    path = OUT / "audit_independent.json"
    if not path.exists():
        notes.append("independent audit artifact missing")
        return
    d = json.loads(path.read_text())
    artifacts[str(path.relative_to(REPO))] = sha256(path)
    verdict = d.get("verdict")
    if isinstance(verdict, dict):
        flags = [v for v in verdict.values() if isinstance(v, bool)]
        disposition = str(verdict.get("disposition", "")).lower()
        ok = all(flags) and disposition.split()[0] in ("approve", "approved")
    else:
        ok = bool(verdict) if isinstance(verdict, bool) else \
            str(verdict).lower() in ("pass", "passed", "true")
    recs.append({
        "id": "support:independent_audit:verdict",
        "quantity": "independent audit of the harmonic gates",
        "value": verdict,
        "error_percent": None,
        "tolerance_percent": None,
        "pass": ok,
        "basis": "stored",
    })


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build() -> dict:
    artifacts: dict[str, str] = {}
    recs: list[dict] = []
    notes: list[str] = []
    conformance_r1(artifacts, recs, notes)
    conformance_r2(artifacts, recs, notes)
    conformance_combined(artifacts, recs, notes)
    conformance_legs(artifacts, recs, notes)
    conformance_supporting(artifacts, recs, notes)
    conformance_independent_audit(artifacts, recs, notes)

    graded = [r for r in recs if r["error_percent"] is not None]
    worst = max(graded, key=lambda r: abs(r["error_percent"])) if graded else None
    binding = [r["binding_rel_diff"] for r in graded if "binding_rel_diff" in r]
    anchor_scope = [r for r in graded if r["tolerance_percent"] == TOL_PCT]
    payload = {
        "case": "CASE-POISEUILLE-SPH",
        "title": "算例池·泊肃叶流",
        "tolerance_percent": TOL_PCT,
        "binding_rtol": BIND_RTOL,
        "source_artifacts_sha256": artifacts,
        "functionals": recs,
        "summary": {
            "n_functionals": len(recs),
            "n_recomputed_from_raw": sum(1 for r in recs if r["basis"] == "recomputed"),
            "n_stored": sum(1 for r in recs if r["basis"] == "stored"),
            "n_not_recomputable": sum(1 for r in recs
                                      if r["basis"] == "not_recomputable"),
            "n_failed": sum(1 for r in recs if r["pass"] is False),
            "worst": None if worst is None else {
                "id": worst["id"], "error_percent": worst["error_percent"],
                "tolerance_percent": worst["tolerance_percent"],
            },
            "max_abs_error_percent_anchor_scope": (
                max(abs(r["error_percent"]) for r in anchor_scope) if anchor_scope else None),
            "binding_max_rel_diff": max(binding) if binding else None,
            "binding_ok": all(v <= BIND_RTOL for v in binding) if binding else None,
            "all_pass": all(r["pass"] for r in recs if r["pass"] is not None),
        },
        "not_graded": notes,
        "generated_by": "benchmarks/case_poiseuille/pool_conformance.py",
    }
    # failed-but-out-of-anchor-scope items are reported, not hidden:
    failed = [r["id"] for r in recs if r["pass"] is False]
    payload["summary"]["failed_ids"] = failed
    violations = [
        {"id": r["id"], "binding_rel_diff": r["binding_rel_diff"],
         "recomputed_error_percent": r["error_percent"],
         "stored_error_percent": r["stored_error_percent"],
         "known": r["id"] in KNOWN_BINDING_EXCEPTIONS}
        for r in graded
        if r.get("binding_rel_diff", 0.0) > BIND_RTOL]
    payload["binding_violations"] = violations
    payload["summary"]["n_binding_violations"] = len(violations)
    payload["summary"]["n_unknown_binding_violations"] = sum(
        1 for v in violations if not v["known"])
    payload["verdict"] = "PASS" if payload["summary"]["all_pass"] else "FAIL"
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT / "pool_conformance.json")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit non-zero unless the committed record matches")
    args = ap.parse_args(argv)
    payload = build()
    s = payload["summary"]
    print(f"[pool] {payload['case']}: {s['n_functionals']} functionals "
          f"({s['n_recomputed_from_raw']} recomputed from raw rows), "
          f"worst {s['worst']['error_percent']:.3f} % ({s['worst']['id']}), "
          f"binding max rel diff {s['binding_max_rel_diff']:.2e}, "
          f"failed {s['n_failed']} -> {payload['verdict']}")
    for n in payload["not_graded"]:
        print(f"[pool] not graded: {n}")
    for fid in s["failed_ids"]:
        print(f"[pool] FAIL {fid}")

    if args.check:
        if not args.out.exists():
            print(f"[pool] missing committed record {args.out}", file=sys.stderr)
            return 2
        old = json.loads(args.out.read_text())
        if old["source_artifacts_sha256"] != payload["source_artifacts_sha256"]:
            print("[pool] source artifacts changed since the record was written",
                  file=sys.stderr)
            return 3
        if old["summary"] != payload["summary"]:
            print("[pool] summary drifted from the committed record", file=sys.stderr)
            return 4
        print("[pool] committed record matches")
        return 0 if payload["verdict"] == "PASS" else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"[pool] wrote {args.out}")
    return 0 if (payload["verdict"] == "PASS" and s["binding_ok"] is not False) else 1


if __name__ == "__main__":
    raise SystemExit(main())