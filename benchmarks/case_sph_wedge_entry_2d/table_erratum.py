"""Can ``docs/verification_wedge_entry.md`` Table 3.2 be reproduced from committed evidence?

Why this exists
---------------
``tests/test_case_sph_wedge_entry_2d_acceptance.py`` used to grade the mission
acceptance (peak slamming ``Cp`` within 15 % of the reference at ``beta = 30 deg``)
on a hand-copied ladder

    dx = 0.5 mm -> peak-Cp error 1.1 %  ->  "< 15 %"  ->  PASS

whose stated source was Table 3.2 of ``docs/verification_wedge_entry.md``.  Two
things were wrong with that and neither had been checked:

1. the numbers in the test did not match the table it cited (e.g. it lists
   ``c/(V t) = 2.783`` at ``dx = 2 mm`` while the table says 2.6124), and
2. no artefact of that campaign was ever committed, so nobody could tell which
   measurement operator produced ``p_max = 4.245 kPa`` at ``dx = 0.5 mm``.

This script settles (2) *mechanically*: it parses Table 3.2 out of the document,
converts its ``p_max`` column into the dimensionless ``Cp = p_max / q`` the case
grades on, and puts the committed estimators at the same ``dx`` next to it.  If
no committed estimator lands on the table value, the table's peak column is
unreproducible evidence and must not be used as an acceptance verdict -- which is
exactly the claim the rewritten acceptance test binds itself to.

Estimators available in committed artefacts (``docs/artifacts/...``)
-------------------------------------------------------------------
``resolved_band``  61-frame self-similar ensemble, sample-gated binned profile,
                   resolution-consistent bin band (``ensemble_peak.py``)
``footprint_max``  maximum of the per-frame footprint peak over the 61 frames
                   (the literal "peak anywhere on the wet surface" reading; it is
                   a single-particle statistic and is known to be noise dominated)
``footprint_med``  median of the same per-frame values

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/table_erratum.py
    python benchmarks/case_sph_wedge_entry_2d/table_erratum.py --doc <md> \
        --artifacts <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DEFAULT_DOC = ROOT / "docs" / "verification_wedge_entry.md"
DEFAULT_ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
DEFAULT_OUT = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_table_erratum"

GATE = 0.15          # the case tolerance the table's peak column was read against
RHO = 1000.0
VELOCITY = 1.0
SECTION = "### 3.2"


def _num(cell: str):
    """First float in a markdown cell ('3.5 %' -> 3.5, '1' -> 1.0)."""
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cell.replace("", ""))
    return float(m.group()) if m else None


def parse_table_3_2(doc: Path) -> list[dict]:
    """Rows of the grid-ladder table, keyed by the header cells (never by index)."""
    text = doc.read_text().splitlines()
    try:
        start = next(i for i, ln in enumerate(text) if ln.strip().startswith(SECTION))
    except StopIteration:                              # pragma: no cover
        raise SystemExit(f"{SECTION} not found in {doc}")
    header = None
    rows: list[dict] = []
    for ln in text[start + 1:]:
        ln = ln.strip()
        if ln.startswith("### "):
            break
        if not ln.startswith("|"):
            if header and rows:
                break
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):   # separator row
            continue
        row = dict(zip(header, cells))
        dx_cell = next((k for k in row if k.lower().startswith("dx")), None)
        if dx_cell is None:
            continue
        dx_mm = _num(row[dx_cell])
        if dx_mm is None:
            continue
        pmax_cell = next((k for k in row if k.startswith("p_max SPH")), None)
        c_cell = next((k for k in row if k.startswith("c/(V t)")), None)
        step_cell = next((k for k in row if k.lower().startswith("steps")), None)
        rows.append({
            "dx": dx_mm * 1e-3,
            "row": row,
            "table_pmax_kpa": _num(row[pmax_cell]) if pmax_cell else None,
            "table_pmax_err": _num(row[next(k for k in row
                                            if k.startswith("p_max err"))])
            if any(k.startswith("p_max err") for k in row) else None,
            "table_c_over_Vt": _num(row[c_cell]) if c_cell else None,
            "table_steps": int(_num(row[step_cell])) if step_cell else None,
        })
    return rows


def doc_config(doc: Path) -> dict:
    """Solver settings the document itself declares for its tables."""
    text = doc.read_text()
    out = {"alpha_visc": None, "c0_over_V": None, "gamma": None, "c_end_m": None}
    m = re.search(r"alpha\s*=\s*([0-9.]+)", text)
    if m:
        out["alpha_visc"] = float(m.group(1))
    m = re.search(r"c0\s*=\s*([0-9.]+)\s*V", text)
    if m:
        out["c0_over_V"] = float(m.group(1))
    m = re.search(r"gamma\s*=\s*([0-9.]+)", text)
    if m:
        out["gamma"] = float(m.group(1))
    m = re.search(r"c_end\s*=\s*([0-9.]+)\s*mm", text)
    if m:
        out["c_end_m"] = float(m.group(1)) * 1e-3
    return out


def wagner_cp_max(beta_deg: float = 30.0) -> float:
    """Wagner peak ``Cp`` of the same idealisation (closed form, not typed in)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tensorsph.wedge_entry import wagner_peak_over_q  # noqa: WPS433

    return wagner_peak_over_q(math.radians(beta_deg))


def committed_estimators(artifacts: Path) -> list[dict]:
    out = []
    for path in sorted(Path(artifacts).glob("ensemble_*.json")):
        art = json.loads(path.read_text())
        dx = art["config"]["dx"]
        tag = f"dx{dx * 1e3:g}mm"
        frames = art["by_frames"].get("61", {})
        rb = frames.get("resolution_band", {})
        scan = art.get("frame_scan", [])
        foot = [r["cp_footprint_max"] for r in scan
                if isinstance(r.get("cp_footprint_max"), (int, float))]
        out.append({
            "tag": tag,
            "path": str(path.relative_to(ROOT)),
            "dx": dx,
            "q": art.get("q"),
            "steps": art.get("steps"),
            "frames": art["config"].get("frames"),
            "c0_factor": art["config"].get("c0_factor"),
            "alpha_visc": art["config"].get("alpha_visc"),
            "resolved_band": rb.get("peak_median"),
            "resolved_band_spread": rb.get("spread_rel"),
            "footprint_max": max(foot) if foot else None,
            "footprint_median": sorted(foot)[len(foot) // 2] if foot else None,
            "n_frames_footprint": len(foot),
        })
    return out


def rel(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b - 1.0


def build(doc: Path, artifacts: Path) -> dict:
    rows = parse_table_3_2(doc)
    est = committed_estimators(artifacts)
    cp_wagner = wagner_cp_max()
    q = 0.5 * RHO * VELOCITY ** 2
    est_by_dx = {round(e["dx"], 9): e for e in est}

    comparisons = []
    for r in rows:
        e = est_by_dx.get(round(r["dx"], 9))
        table_cp = (r["table_pmax_kpa"] * 1000.0 / q) if r["table_pmax_kpa"] else None
        rec = {
            "dx": r["dx"],
            "table_steps": r["table_steps"],
            "table_c_over_Vt": r["table_c_over_Vt"],
            "table_pmax_kpa": r["table_pmax_kpa"],
            "table_pmax_err_printed": r["table_pmax_err"],
            "table_cp": table_cp,
            "table_cp_err_recomputed": rel(table_cp, cp_wagner),
            "committed": None,
            "match": "no committed artefact at this dx",
        }
        if e is not None:
            cand = {k: e[k] for k in ("resolved_band", "footprint_max",
                                      "footprint_median")}
            rec["committed"] = {
                "artifact": e["path"],
                "steps": e["steps"],
                "frames": e["frames"],
                "c0_factor": e["c0_factor"],
                "alpha_visc": e["alpha_visc"],
                "cp": cand,
                "cp_err_vs_wagner": {k: rel(v, cp_wagner) for k, v in cand.items()},
                "cp_err_vs_table": {k: rel(v, table_cp) for k, v in cand.items()}
                if table_cp else None,
            }
            closest = min((abs(rel(v, table_cp)), k) for k, v in cand.items()
                          if v is not None and table_cp) if table_cp else None
            rec["closest_estimator"] = closest[1] if closest else None
            rec["closest_rel_gap_to_table"] = closest[0] if closest else None
            # A table value counts as reproduced when *some* committed estimator
            # at the same dx lands inside the case tolerance of it.
            rec["match"] = ("reproduced" if closest and closest[0] <= GATE
                            else "unreproducible")
        comparisons.append(rec)

    reproduced = [c for c in comparisons if c["match"] == "reproduced"]
    unreproducible = [c for c in comparisons if c["match"] == "unreproducible"]
    no_artefact = [c for c in comparisons if c["committed"] is None]

    dc = doc_config(doc)
    art_cfg = {c["tag"]: {"steps": c["steps"], "c0_factor": c["c0_factor"],
                          "alpha_visc": c["alpha_visc"]} for c in est}
    cfg_diff = {
        "doc_declared": dc,
        "committed_runs": art_cfg,
        "alpha_mismatch": (dc["alpha_visc"] is not None and bool(art_cfg)
                           and all(abs(c["alpha_visc"] - dc["alpha_visc"]) > 1e-12
                                   for c in art_cfg.values())),
        "c0_mismatch": (dc["c0_over_V"] is not None and bool(art_cfg)
                        and any(abs(c["c0_factor"] - dc["c0_over_V"]) > 1e-12
                                for c in art_cfg.values())),
    }
    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "doc_config_vs_committed": cfg_diff,
        "script": "benchmarks/case_sph_wedge_entry_2d/table_erratum.py",
        "doc": str(doc.relative_to(ROOT)) if doc.is_relative_to(ROOT) else str(doc),
        "section": SECTION,
        "design": {"gate": GATE, "q_pa": q, "rho": RHO, "velocity": VELOCITY,
                   "wagner_cp_max": cp_wagner,
                   "estimators": ["resolved_band (binned, 61 frames)",
                                  "footprint_max (max over frames)",
                                  "footprint_median (median over frames)"]},
        "comparisons": comparisons,
        "summary": {
            "rows_in_table": len(comparisons),
            "rows_with_committed_artifact": len(comparisons) - len(no_artefact),
            "reproduced": [c["dx"] for c in reproduced],
            "unreproducible": [c["dx"] for c in unreproducible],
            "no_committed_artifact": [c["dx"] for c in no_artefact],
            "verdict": _verdict(comparisons),
        },
    }


def _verdict(comparisons: list[dict]) -> dict:
    """Is the table's peak column usable as an acceptance verdict?"""
    if not comparisons:
        return {"usable_as_acceptance": False, "reason": "table not parsed"}
    unrepro = [c for c in comparisons if c["match"] == "unreproducible"]
    if not unrepro:
        return {"usable_as_acceptance": True,
                "reason": "every table row is reproduced by a committed estimator"}
    worst = max(unrepro, key=lambda c: c["closest_rel_gap_to_table"] or 0.0)
    return {
        "usable_as_acceptance": False,
        "reason": (f"{len(unrepro)}/{len(comparisons)} table rows are not "
                   f"reproduced by any committed estimator within {GATE:.0%}; "
                   f"worst dx={worst['dx'] * 1e3:g} mm, closest estimator "
                   f"{worst['closest_estimator']} off by "
                   f"{(worst['closest_rel_gap_to_table'] or float('nan')):.1%}"),
        "worst_row": worst["dx"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    ap.add_argument("--artifacts", type=Path, default=DEFAULT_ART)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    result = build(args.doc, args.artifacts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "erratum.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result["summary"], indent=1))
    for c in result["comparisons"]:
        print(f"dx={c['dx'] * 1e3:g}mm table_Cp={c['table_cp']} "
              f"match={c['match']} closest={c.get('closest_estimator')} "
              f"gap={c.get('closest_rel_gap_to_table')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())