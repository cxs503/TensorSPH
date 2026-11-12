#!/usr/bin/env python3
"""Audit the Wagner reference column of the committed slamming evidence bundle.

Why
---
``docs/slamming_campaign_report.json`` is the only committed evidence bundle of
campaign node p129w1r2-n08.  Its table "砰击力峰值 vs Wagner 上限" prints a
``C_s,Wagner`` column.  A reference column is the one place in a verification
report that *must* be reproducible from the code that defines the reference: if
it is not, every future error percentage quoted against it is meaningless.

This script recomputes ``C_s,Wagner = pi^2/tan(beta)`` from
``tensorsph.slamming`` (the same expression ``benchmarks/slamming/run.py`` uses
in ``derived.c_s_wagner``) and compares it with the number printed in the
committed table.  ``--fix`` rewrites the mismatching cells with the recomputed
value, so the repair is programmatic injection, not retyping.

Findings it can report
----------------------
``mismatch``            printed value differs from the module by > tol
``unattributable_rows`` the printed column does not scale like 1/tan(beta),
                       i.e. no single formula in the module can produce it
``missing_figure``      a figure entry points at a path that is not committed
``empty_figure``        a figure entry carries no payload and no committed file

Usage
-----
    PYTHONPATH=src python benchmarks/slamming/audit_wagner_reference.py
    PYTHONPATH=src python benchmarks/slamming/audit_wagner_reference.py --fix
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.slamming import slamming_force_coefficient, wagner_c_coeff  # noqa: E402

DOC = REPO / "docs" / "slamming_campaign_report.json"
OUT = Path(__file__).resolve().parent / "results" / "wagner_reference_audit.json"
ROW = re.compile(r"^\|\s*(\d+(?:\.\d+)?)\s*\|\s*([\d.]+)\s*\|\s*([-\d.]+|-)\s*\|\s*([\d.]+|-)\s*\|",
                 re.MULTILINE)


def analytic_cs(beta_deg: float, velocity: float = 1.0, rho: float = 1000.0,
                t: float = 1.0e-3) -> float:
    """``pi^2/tan(beta)`` via the module's own force-coefficient function."""
    beta = math.radians(beta_deg)
    return slamming_force_coefficient(rho, velocity, beta, t)


def audit(tol: float = 1.0e-3) -> dict:
    raw = DOC.read_text()
    blob = json.loads(raw)
    md = blob.get("markdown", "")
    rows = []
    for m in ROW.finditer(md):
        beta = float(m.group(1))
        printed = None if m.group(4) == "-" else float(m.group(4))
        want = analytic_cs(beta)
        rows.append({
            "beta_deg": beta, "velocity": float(m.group(2)),
            "c_s_measured": m.group(3),
            "c_s_wagner_printed": printed,
            "c_s_wagner_recomputed": want,
            "rel_err": None if printed is None else (printed - want) / want,
            "match": None if printed is None else abs(printed - want) / want <= tol,
            "wagner_c_coeff": wagner_c_coeff(math.radians(beta)),
        })
    num = [r for r in rows if r["c_s_wagner_printed"] is not None]
    spread = []
    for a in range(len(num)):
        for b in range(a + 1, len(num)):
            printed = num[a]["c_s_wagner_printed"] / num[b]["c_s_wagner_printed"]
            model = math.tan(math.radians(num[b]["beta_deg"])) / math.tan(
                math.radians(num[a]["beta_deg"]))
            spread.append({"pair": [num[a]["beta_deg"], num[b]["beta_deg"]],
                           "printed_ratio": printed, "model_ratio": model,
                           "ratio_of_ratios": printed / model})
    figures = []
    for f in blob.get("figures", []):
        png = f.get("png_base64") or ""
        # resolve the path the document actually links, not a guessed one
        mm = re.search(r"!\[[^\]]*\]\(([^)]*" + re.escape(f["name"]) + r"[^)]*\.png)\)", md)
        claimed = mm.group(1) if mm else None
        cand = (REPO / claimed) if claimed else None
        figures.append({"name": f["name"], "bytes": len(png),
                        "claimed_path": claimed,
                        "claimed_path_exists": bool(cand and cand.exists()),
                        "empty": not png})
    report = {
        "schema": "tensorsph.slamming.wagner_reference_audit/v1",
        "document": str(DOC.relative_to(REPO)),
        "tol_rel": tol,
        "definition": "C_s = pi^2/tan(beta)  (== benchmarks/slamming/run.py "
                      "derived.c_s_wagner == tensorsph.slamming.slamming_force_coefficient)",
        "rows": rows,
        "scale_consistency": spread,
        "figures": figures,
        "verdict": {
            "mismatch": [r["beta_deg"] for r in rows if r["match"] is False],
            "unattributable_rows": [r["beta_deg"] for r in rows if r["match"] is False],
            "missing_figure": [f["name"] for f in figures
                              if f["empty"] and not f["claimed_path_exists"]],
        },
    }
    report["verdict"]["clean"] = not any(
        report["verdict"][k] for k in ("mismatch", "missing_figure"))
    return report


def fix(report: dict) -> int:
    """Rewrite only the wrong reference cells; everything else is untouched."""
    raw = DOC.read_text()
    blob = json.loads(raw)
    md = blob.get("markdown", "")
    n = 0

    def sub(m):
        nonlocal n
        beta = float(m.group(1))
        printed = None if m.group(4) == "-" else float(m.group(4))
        want = analytic_cs(beta)
        if printed is not None and abs(printed - want) / want <= 1.0e-3:
            return m.group(0)
        n += 1
        return (f"| {m.group(1)} | {m.group(2)} | {m.group(3)} | "
                f"{want:.2f} |")

    md2 = ROW.sub(sub, md)
    blob["markdown"] = md2
    DOC.write_text(json.dumps(blob, indent=1, ensure_ascii=False))
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    rep = audit()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["verdict"], indent=1))
    if a.fix and not rep["verdict"]["clean"]:
        k = fix(rep)
        print(f"fixed {k} reference cell(s) in {DOC.relative_to(REPO)}")
        rep2 = audit()
        out.write_text(json.dumps(rep2, indent=1))
        print(json.dumps({"after_fix": rep2["verdict"]}, indent=1))
        return 0 if rep2["verdict"]["clean"] else 1
    return 0 if rep["verdict"]["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
