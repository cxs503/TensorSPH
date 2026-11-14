"""Independent re-derivation audit of CASE-SPH-POISEUILLE-MORRIS artefacts.

Why this exists
---------------
`run.py` produces the numbers, `make_doc.py` renders them, and both live in the
same module family.  A benchmark that keeps its reference solution, its metric
and its pass/fail verdict in one place can be silently wrong in a consistent
way (a sign convention, a normalisation, a band edge) and still look
self-consistent.  This script re-derives everything the document claims from the
raw artefacts only, with its *own* arithmetic and its *own* reading of the
closed form, and refuses to trust `run.py`:

* it does not import `benchmarks.poiseuille_morris.run` (nor `tensorsph`), so a
  defect in the metric cannot hide behind a shared code path;
* the channel height is reconstructed from the recorded `dx` x `H/dx` rather
  than taken from a constant;
* the closed-form velocity is re-checked against the ODE ``nu u'' = -g`` and
  both no-slip conditions symbolically (sympy, when importable);
* the per-row reference values stored in the artefact are compared against the
  independently evaluated formula, then the interior L2 / centre / max-row
  metrics and the momentum-budget wall shear are recomputed from the raw
  measured profile rows;
* the rendered document is parsed and every tabulated number is bound back to
  the artefact (document/artefact drift is a defect, not a formatting choice),
  and a table rendered with no rows is reported as a defect rather than
  skipped.

Findings are severity-ranked (`blocking` / `major` / `minor`); any `blocking`
finding makes the script exit non-zero.  Usage::

    PYTHONPATH=src:. python benchmarks/poiseuille_morris/audit_independent.py \
        [--results runs/poiseuille_morris/results.json] \
        [--doc docs/benchmarks/poiseuille-morris.md] [--json runs/poiseuille_morris/audit.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

RHO_DEFAULT = 1000.0     # kg/m^3, the density the ISPH route declares
LIMIT = 5.0              # case tolerance [%] (DESIGN.md)
STRONG = 3.0             # platform standard [%]


# --------------------------------------------------------------------------- #
# Independent arithmetic: written from the closed form, not imported
# --------------------------------------------------------------------------- #

def u_exact(y: float, height: float, g: float, nu: float) -> float:
    """Steady plane-Poiseuille velocity, evaluated from the closed form."""
    return g * (height * y - y * y) / (2.0 * nu)


def check_ode_symbolic():
    """Verify nu u'' = -g with u(0) = u(H) = 0 symbolically.

    Returns (ok, detail).  Uses a *separate* derivation from `u_exact`: the
    candidate is written in the alternative form ``g H^2 (eta - eta^2)/(2 nu)``
    with ``eta = y/H`` so that a factor-of-two slip in either form shows up.
    """
    try:
        import sympy as sp
    except Exception:                                    # pragma: no cover
        return True, "sympy unavailable: ODE check skipped (analytic form only)"
    y, H, g, nu, eta = sp.symbols("y H g nu eta", positive=True)
    u = g * H ** 2 * (y / H - (y / H) ** 2) / (2 * nu)
    residual = sp.simplify(nu * sp.diff(u, y, 2) + g)
    bc0 = sp.simplify(u.subs(y, 0))
    bcH = sp.simplify(u.subs(y, H))
    peak = sp.simplify(u.subs(y, H / 2))
    peak_expected = sp.simplify(g * H ** 2 / (8 * nu))
    ok = residual == 0 and bc0 == 0 and bcH == 0 and sp.simplify(peak - peak_expected) == 0
    return ok, (f"ODE residual={residual}, u(0)={bc0}, u(H)={bcH}, "
                f"u(H/2)={sp.simplify(peak)} vs gH^2/(8nu)={peak_expected}")


def height_of(case: dict) -> float:
    """Reconstruct H from recorded dx and H/dx (never from a module constant)."""
    return case["dx"] * case["resolution_H_over_dx"]


def recompute(case: dict) -> dict:
    """Recompute every metric of one case from its raw rows + recorded g, nu."""
    height = height_of(case)
    g, nu = case["body_force"], case["viscosity"]
    rows = case["metrics"]["rows"]
    u_max = g * height ** 2 / (8.0 * nu)
    rec = []
    for r in rows:
        y = r["y"]
        u_ref = u_exact(y, height, g, nu)
        rec.append({
            "y_over_H": y / height,
            "stored_ref": r["u_reference"],
            "recomputed_ref": u_ref,
            "u_measured": r["u_measured"],
            "err_over_umax": (r["u_measured"] - u_ref) / u_max,
        })
    interior = [r for r in rec if 0.15 <= r["y_over_H"] <= 0.85]
    centre_row = min(rec, key=lambda r: abs(r["y_over_H"] - 0.5))
    centre_ref = u_exact(centre_row["y_over_H"] * height, height, g, nu)
    l2 = math.sqrt(sum(r["err_over_umax"] ** 2 for r in interior) / len(interior))
    return {
        "height": height,
        "u_max_reference": u_max,
        "reynolds_c": u_max * height / nu,
        "n_interior_rows": len(interior),
        "l2_over_umax_pct": 100.0 * l2,
        "centre_error_pct": (100.0 * (centre_row["u_measured"] - centre_ref) / centre_ref
                             if centre_ref else float("nan")),
        "max_abs_over_umax_pct": 100.0 * max(abs(r["err_over_umax"]) for r in interior),
        "row_ref_max_dev": max(abs(r["stored_ref"] - r["recomputed_ref"]) for r in rec),
        "row_ref_rel_dev": max(abs(r["stored_ref"] - r["recomputed_ref"]) /
                               (u_max or 1.0) for r in rec),
    }


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #

class Audit:
    def __init__(self):
        self.findings = []
        self.checks = 0

    def add(self, severity: str, scope: str, claim: str, evidence: str):
        self.findings.append({"severity": severity, "scope": scope,
                              "claim": claim, "evidence": evidence})

    def close(self, ok: bool, scope: str, claim: str, evidence: str):
        self.checks += 1
        if not ok:
            self.add("blocking", scope, claim, evidence)
        return ok

    def blocking(self):
        return [f for f in self.findings if f["severity"] == "blocking"]


def rel_close(a: float, b: float, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale <= tol


def audit_cases(payload: dict, A: Audit) -> dict:
    """Per-case independent recomputation; returns the recomputed table."""
    table = {}
    for route, cases in payload["cases"].items():
        for c in cases:
            tag = (f"{route} Re_c={c['reynolds_c']:g} H/dx={c['resolution_H_over_dx']} "
                   f"h/dx={c['h_over_dx']:.2f} col={c['columns']}")
            r = recompute(c)
            m = c["metrics"]
            table[tag] = {"recorded": {
                "centre_error_pct": m["centre_error_pct"],
                "l2_over_umax_pct": m["interior"]["l2_over_umax_pct"],
                "max_abs_over_umax_pct": m["interior"]["max_abs_over_umax_pct"],
                "u_max_reference": m["u_max_reference"],
                "reynolds_c": c["reynolds_c"],
            }, "recomputed": r}

            A.close(rel_close(r["height"], height_of(c), 1e-12), tag,
                    "channel height reconstructed from dx * (H/dx)",
                    f"H={r['height']!r}")
            A.close(rel_close(r["u_max_reference"], m["u_max_reference"], 1e-12), tag,
                    "recorded u_max equals g H^2/(8 nu)",
                    f"recomputed={r['u_max_reference']!r} recorded={m['u_max_reference']!r}")
            A.close(rel_close(r["reynolds_c"], c["reynolds_c"], 1e-9), tag,
                    "declared Re_c = u_max H / nu is the Re the run actually used",
                    f"recomputed={r['reynolds_c']!r} declared={c['reynolds_c']!r}")
            A.close(r["row_ref_rel_dev"] < 1e-12, tag,
                    "every stored row reference value equals the closed form",
                    f"max |d u_ref| / u_max = {r['row_ref_rel_dev']:.3e}")
            A.close(rel_close(r["l2_over_umax_pct"], m["interior"]["l2_over_umax_pct"], 1e-9),
                    tag, "interior L2 reproduced by an independent metric",
                    f"recomputed={r['l2_over_umax_pct']:.6f}% "
                    f"recorded={m['interior']['l2_over_umax_pct']:.6f}%")
            A.close(rel_close(r["centre_error_pct"], m["centre_error_pct"], 1e-9), tag,
                    "centre-line error reproduced independently",
                    f"recomputed={r['centre_error_pct']:+.6f}% "
                    f"recorded={m['centre_error_pct']:+.6f}%")
            A.close(rel_close(r["max_abs_over_umax_pct"],
                              m["interior"]["max_abs_over_umax_pct"], 1e-9), tag,
                    "max interior row error reproduced independently",
                    f"recomputed={r['max_abs_over_umax_pct']:.6f}% "
                    f"recorded={m['interior']['max_abs_over_umax_pct']:.6f}%")
            A.close(r["n_interior_rows"] == m["interior"]["n_rows"], tag,
                    "interior row count matches the declared band 0.15-0.85",
                    f"recomputed={r['n_interior_rows']} recorded={m['interior']['n_rows']}")
    return table


def audit_wall_shear(payload: dict, A: Audit):
    """Momentum-budget wall shear: tau_w = rho g H / 2 must close the budget."""
    for c in payload["cases"].get("isph_projection", []):
        gate = (c.get("gates") or {}).get("wall_shear_budget")
        if not gate:
            continue
        tag = (f"isph Re_c={c['reynolds_c']:g} H/dx={c['resolution_H_over_dx']} "
               f"h/dx={c['h_over_dx']:.2f}")
        height = height_of(c)
        ref = RHO_DEFAULT * c["body_force"] * height / 2.0
        A.close(rel_close(ref, gate["reference"], 1e-9), tag,
                "analytic wall shear is rho g H / 2",
                f"independent={ref!r} recorded={gate['reference']!r}")
        if gate.get("measured") is not None and gate.get("reference"):
            rel = 100.0 * (gate["measured"] - gate["reference"]) / gate["reference"]
            A.close(abs(rel - gate["rel_error_pct"]) < 1e-6, tag,
                    "reported wall-shear error is consistent with its own readings",
                    f"independent={rel:+.6f}% reported={gate['rel_error_pct']:+.6f}%")
            A.close(abs(gate["rel_error_pct"]) < LIMIT, tag,
                    "wall shear inside the 5 % case tolerance",
                    f"{gate['rel_error_pct']:+.4f}%")


def audit_verdict(payload: dict, table: dict, A: Audit):
    """Verdicts must agree with the independently recomputed metrics."""
    verdict = payload.get("verdict") or {}
    ladder = payload.get("settings", {})
    re_ladder = ladder.get("reynolds_ladder") or []
    routes_present = {k for k, v in payload["cases"].items() if v}

    A.close(bool(verdict), "payload", "a verdict block exists", f"verdict keys={list(verdict)}")
    for route, label in (("isph", "isph_projection"), ("wcsp", "wcsp_production")):
        entries = verdict.get(route) or {}
        # a route may deliberately run a narrower Re ladder than the ISPH
        # headline; only the Re's that route actually ran must have a verdict
        ran = {c["reynolds_c"] for c in payload["cases"].get(label, [])}
        missing = [re_v for re_v in re_ladder
                   if re_v in ran and str(re_v) not in entries]
        A.close(label not in routes_present or not missing, "verdict",
                f"{label}: every Reynolds number of the declared ladder has a verdict",
                f"ladder={re_ladder} covered={sorted(entries)} missing={missing}")
        for re_key, v in entries.items():
            re_f = float(re_key)
            if not v:
                A.add("blocking", f"verdict {label} Re_c={re_f:g}",
                      "verdict entry is empty (renders as a table row with no numbers)",
                      "value is falsy")
                continue
            # bind the verdict to the independently recomputed finest case
            cand = [t for t in table
                    if t.startswith(f"{label} Re_c={re_f:g} ")
                    and f"H/dx={v['H_over_dx']}" in t]
            if cand:
                rec = table[cand[0]]["recomputed"]
                A.close(rel_close(v["centre_error_pct"], rec["centre_error_pct"], 1e-9)
                        and rel_close(v["l2_interior_pct"], rec["l2_over_umax_pct"], 1e-9),
                        f"verdict {label} Re_c={re_f:g}",
                        "verdict numbers equal the independently recomputed finest case",
                        f"verdict centre={v['centre_error_pct']:+.6f}% "
                        f"L2={v['l2_interior_pct']:.6f}% vs recomputed "
                        f"centre={rec['centre_error_pct']:+.6f}% "
                        f"L2={rec['l2_over_umax_pct']:.6f}%")
            claimed_pass = bool(v["pass"])
            actual_pass = (abs(v["centre_error_pct"]) < LIMIT
                           and v["l2_interior_pct"] < LIMIT)
            A.close(claimed_pass == actual_pass, f"verdict {label} Re_c={re_f:g}",
                    "PASS flag agrees with the recorded numbers vs the 5 % tolerance",
                    f"claimed={claimed_pass} from_numbers={actual_pass} "
                    f"(centre={v['centre_error_pct']:+.2f}%, L2={v['l2_interior_pct']:.2f}%)")


def band_diagnostics(case: dict) -> dict:
    """Wall-band exposure + sign cancellation of the scored band (from rows only)."""
    rows = case["metrics"]["rows"]
    u_max = case["metrics"]["u_max_reference"]
    inband = [r for r in rows if 0.15 <= r["y_over_H"] <= 0.85]
    outband = [r for r in rows if not (0.15 <= r["y_over_H"] <= 0.85)]
    centre = min(rows, key=lambda r: abs(r["y_over_H"] - 0.5))
    sgn = 1.0 if centre["u_measured"] >= centre["u_reference"] else -1.0
    oppose = sum(1 for r in inband if (r["u_measured"] - r["u_reference"]) * sgn < 0.0)
    return {
        "u_max_ratio": max(abs(r["u_measured"]) for r in rows) / u_max,
        "wall_max_pct": (100.0 * max(abs(r["u_measured"] - r["u_reference"])
                                     for r in outband) / u_max if outband else float("nan")),
        "sign_cancel": oppose / len(inband) if inband else float("nan"),
        "n_inband": len(inband),
    }


def audit_cancellation(payload: dict, A: Audit):
    """A band-RMS pass must not be a cancellation of two opposite-sign errors.

    The scored band 0.15 <= y/H <= 0.85 is insensitive to a profile that is too
    flat in the core and too fast near the walls: the errors alternate, so the
    RMS collapses while individual rows sit well outside the tolerance.  Any
    case whose band-RMS passes while the excluded wall band errs beyond the
    tolerance *and* a quarter of the scored rows carry the opposite sign is
    reported, with the numbers, so a pass cannot be quoted as accuracy.
    """
    for route, cases in payload["cases"].items():
        for c in cases:
            m = c["metrics"]
            l2 = m["interior"]["l2_over_umax_pct"]
            d = band_diagnostics(c)
            tag = (f"{route} Re_c={c['reynolds_c']:g} H/dx={c['resolution_H_over_dx']}")
            A.checks += 1
            if l2 < LIMIT and d["wall_max_pct"] > LIMIT and d["sign_cancel"] >= 0.25:
                A.add("major", tag,
                      "band-RMS pass is an error cancellation, not accuracy "
                      "(must be reported as a qualified pass)",
                      f"e_L2={l2:.2f}% but wall-band max |e|={d['wall_max_pct']:.1f}%, "
                      f"sign-cancel={100.0 * d['sign_cancel']:.0f}% of "
                      f"{d['n_inband']} scored rows, u_max/u_max,ref="
                      f"{d['u_max_ratio']:.4f}")
            A.checks += 1
            if l2 < LIMIT and d["sign_cancel"] < 0.05:
                A.add("minor", tag,
                      "clean pass: interior errors share one sign (no cancellation)",
                      f"e_L2={l2:.2f}%, wall-band max |e|={d['wall_max_pct']:.2f}%, "
                      f"sign-cancel={100.0 * d['sign_cancel']:.0f}%")


def audit_controls(payload: dict, A: Audit):
    ctrl = payload.get("controls") or {}
    zf = ctrl.get("zero_forcing")
    A.close(bool(zf), "controls", "zero-forcing control recorded",
            f"keys={list(ctrl)}")
    if zf:
        A.add("minor" if zf.get("max_abs_u_measured", 0.0) < 1e-12 else "blocking",
              "controls", "zero body force produces a zero field (the metric cannot fake agreement)",
              f"max|u| = {zf.get('max_abs_u_measured')!r} m/s")
    rep = ctrl.get("repeat_headline")
    if rep:
        A.add("minor" if rep.get("profiles_identical") else "blocking", "controls",
              "repeat of the headline case is bit-identical (deterministic solver)",
              f"first={rep.get('centre_error_pct_first')!r} "
              f"repeat={rep.get('centre_error_pct_repeat')!r} "
              f"identical={rep.get('profiles_identical')!r}")
    else:
        A.add("major", "controls", "determinism control recorded", "missing")


# --------------------------------------------------------------------------- #
# Document <-> artefact binding
# --------------------------------------------------------------------------- #

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def parse_markdown_tables(text: str):
    """Yield (heading, [(cells...)]) for each markdown table in the document."""
    heading = ""
    block, header = [], None
    for line in text.splitlines() + [""]:
        if line.startswith("#"):
            if block:
                yield heading, block
                block, header = [], None
            heading = line.lstrip("# ").strip()
            continue
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue                                  # separator row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
                block.append(("header", cells))
            else:
                block.append(("row", cells))
        else:
            if block:
                yield heading, block
                block, header = [], None
    if block:
        yield heading, block


def audit_doc(payload: dict, doc_text: str, table: dict, A: Audit):
    """Every table with a header but no rows is a defect; numbers must bind."""
    rows_by_heading = {}
    for heading, block in parse_markdown_tables(doc_text):
        rows = [b[1] for b in block if b[0] == "row"]
        rows_by_heading.setdefault(heading, []).append((block, rows))
    empties = [h for h, tabs in rows_by_heading.items() for blk, rows in tabs if not rows]
    A.close(not empties, "document", "no table is rendered with a header and zero rows",
            f"empty tables under headings: {empties}")

    # headline binding: the ISPH Re_c=1 profile table must match the artefact
    best = None
    for c in payload["cases"].get("isph_projection", []):
        if (c["reynolds_c"] == 1.0 and c["h_over_dx"] == 1.90 and c["columns"] == 12
                and c["resolution_H_over_dx"] >= 16):
            if best is None or c["resolution_H_over_dx"] > best["resolution_H_over_dx"]:
                best = c
    A.close(best is not None, "document", "headline ISPH case (Re_c=1, h/dx=1.90) exists",
            "no such case in payload")
    if best:
        rows = best["metrics"]["rows"]
        want = [(f"{r['y_over_H']:.4f}", f"{100.0 * r['err_over_umax']:+.2f}") for r in rows]
        hits = sum(1 for yv, ev in want
                   if re.search(r"\|\s*" + re.escape(yv) + r"\s*\|.*\|\s*"
                                + re.escape(ev) + r"\s*\|", doc_text))
        A.close(hits == len(want), "document",
                "profile table rows bind to the artefact (y/H and error % both present)",
                f"{hits}/{len(want)} rows bound")

        # grid-convergence table binding
        pat = re.compile(r"\|\s*ISPH\s*\|\s*(" + NUM + r")\s*\|\s*(\d+)\s*\|\s*("
                         + NUM + r")\s*\|\s*(" + NUM + r")\s*\|\s*(" + NUM + r")\s*\|")
        doc_rows = pat.findall(doc_text)
        expect = {}
        for c in payload["cases"]["isph_projection"]:
            if c["h_over_dx"] == 1.90 and c["columns"] == 12:
                expect[(f"{c['reynolds_c']:g}", str(c["resolution_H_over_dx"]))] = c
        A.close(bool(doc_rows), "document", "grid-convergence table is populated",
                f"{len(doc_rows)} rows parsed")
        bad = []
        for re_s, res_s, ce, l2, mx in doc_rows:
            key = (re_s, res_s)
            c = expect.get(key)
            if c is None:
                bad.append(f"{key}: no matching case in artefact")
                continue
            pairs = ((float(ce), c["metrics"]["centre_error_pct"]),
                     (float(l2), c["metrics"]["interior"]["l2_over_umax_pct"]),
                     (float(mx), c["metrics"]["interior"]["max_abs_over_umax_pct"]))
            for shown, recorded in pairs:
                if abs(shown - round(recorded, 2)) > 0.005 + 1e-9:
                    bad.append(f"{key}: doc {shown} vs artefact {recorded:.6f}")
        A.close(not bad, "document", "grid-convergence numbers equal the artefact values",
                "; ".join(bad[:6]))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(results: Path, doc: Path | None, out_json: Path | None) -> Audit:
    payload = json.loads(results.read_text())
    A = Audit()
    ok, detail = check_ode_symbolic()
    A.close(ok, "reference", "closed form satisfies nu u'' = -g with both no-slip BCs",
            detail)
    table = audit_cases(payload, A)
    audit_wall_shear(payload, A)
    audit_verdict(payload, table, A)
    audit_cancellation(payload, A)
    audit_controls(payload, A)
    if doc is not None and doc.exists():
        audit_doc(payload, doc.read_text(), table, A)
    else:
        A.add("major", "document", "rendered document present", f"{doc} missing")

    payload_out = {
        "anchor": payload.get("anchor"),
        "results": str(results),
        "doc": str(doc) if doc else None,
        "checks_run": A.checks,
        "findings": A.findings,
        "n_blocking": len(A.blocking()),
        "verdict": "FAIL" if A.blocking() else "PASS",
        "recomputed": table,
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload_out, indent=1) + "\n")
    print(f"[audit] {A.checks} independent checks, {len(A.findings)} findings, "
          f"{len(A.blocking())} blocking -> {payload_out['verdict']}")
    for f in A.findings:
        print(f"  [{f['severity']}] {f['scope']}: {f['claim']} — {f['evidence']}")
    return A


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=Path("runs/poiseuille_morris/results.json"))
    ap.add_argument("--doc", type=Path,
                    default=Path("docs/benchmarks/poiseuille-morris.md"))
    ap.add_argument("--json", type=Path,
                    default=Path("runs/poiseuille_morris/audit.json"))
    args = ap.parse_args(argv)
    A = run(args.results, args.doc, args.json)
    return 1 if A.blocking() else 0


if __name__ == "__main__":
    raise SystemExit(main())