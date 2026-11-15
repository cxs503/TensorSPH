"""Footprint-average peak-pressure ledger and case verdict (m21677).

The graded quantity
-------------------
``Cp_Phi = max over sliding windows of the mean p/q inside a footprint of fixed
physical width ``Phi`` -- the quantity a pressure transducer reports (see
``footprint_peak.py``).  ``Phi = 8 mm`` is declared once, before any run, as the
modelling choice for "the sensor": it is >= 2 cells on the coarsest grid of the
ladder (dx = 4 mm) and 16 cells on the finest (dx = 0.5 mm), so the same
functional can be evaluated on every grid and the same operator is applied to
the reference.

Why not the point maximum
-------------------------
A point maximum of a square-root-singular field is not measurable, and the
binned-maximum estimator used by the committed chain is noise-dominated and
aliased: its observed reading at dx = 4 mm runs 8.00 ... 12.87 over 2.5 ... 0.68
cells per bin while the analytic field through the same operator is monotone
(``bandwidth_matched.json``).  The footprint average divides the single-sample
spike by the number of samples in the window and is insensitive to where the bin
edges fall.

Rule (predeclared, identical to ``ladder_closure.py``)
------------------------------------------------------
* residual: ``r_g = obs_g / synth_g - 1`` with ``synth_g`` = Wagner field
  (the analytic solution of the same idealisation) through the *same* footprint
  operator on the same grid;
* uncertainty: ``U_g`` = frame-bootstrap half-width (5..95 %) of ``obs_g``;
* judgeable: ``U_g <= 15 %``;
* verdict: two finest judgeable grids, same sign of ``r`` and ``|r| <= 15 %``
  -> PASS; same sign and exceeding -> FAIL_above / FAIL_below; otherwise
  NOT_DECIDABLE.

Usage::

    python benchmarks/case_sph_wedge_entry_2d/footprint_ledger.py \
        --artifacts docs/artifacts/case_sph_wedge_entry_2d_footprint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import estimator_bias as eb  # noqa: E402
import footprint_peak as fpm  # noqa: E402
import literature_refs as lr  # noqa: E402

GATE = 0.15
PHI_M = 8.0e-3                 # declared transducer footprint (8 mm)
GRID_TAGS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")
BETA_DEG = 30.0


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and v == v and math.isinf(v) is False


def _row(rows, cells):
    for r in rows:
        if abs(r["footprint_cells"] - cells) < 1e-9:
            return r
    return None


def grid_record(tag: str, art_dir: Path, point: float) -> dict:
    art = json.loads((Path(art_dir) / f"ensemble_{tag}_c0v20.json").read_text())
    dx = art["config"]["dx"]
    t_end = art["t_end"]
    C = art["C"]
    vel = art["config"]["velocity"]
    c_end = C * vel * t_end
    cells = PHI_M / dx                      # footprint in cells for this grid
    fp = art.get("footprint_peak")
    if not fp:
        raise SystemExit(f"{tag}: artifact has no footprint_peak block -- rerun "
                         "run_ladder_shard.py (m21677)")
    row = _row(fp["rows"], round(cells, 6)) or _row(fp["rows"], cells)
    if row is None:
        raise SystemExit(f"{tag}: artifact has no footprint row for {cells} cells")
    boot = fp["bootstrap"].get(str(row["footprint_cells"]), {})

    # reference side: the same operator on the analytic field, same grid
    chain = eb.SynthChain(dx, eb.FRAMES_TOTAL, t_end, True)
    taus: list[float] = []
    cps: list[float] = []
    for tau, cp in chain.frames:
        taus.extend(float(v) for v in tau)
        cps.extend(float(v) for v in cp)
    synth_rows = fpm.footprint_peaks(taus, cps, dx, c_end, (row["footprint_cells"],))
    synth = synth_rows[0]["cp_peak"]

    obs = row["cp_peak"]
    u = boot.get("rel_half_width", float("nan"))
    residual = (obs - synth) / synth if _finite(obs) and _finite(synth) and synth else float("nan")
    return {
        "tag": tag, "dx_m": dx, "phi_m": PHI_M, "footprint_cells": row["footprint_cells"],
        "delta_tau": row["delta_tau"], "obs_cp": obs, "synth_cp": synth,
        "bootstrap": boot, "uncertainty": u, "residual": residual,
        "judgeable": bool(_finite(u) and u <= GATE and _finite(residual)),
        "obs_vs_wagner_point": obs / point - 1.0 if _finite(obs) else float("nan"),
        "obs_vs_designated": (obs / lr.designated_cp_max(BETA_DEG) - 1.0
                              if _finite(obs) else float("nan")),
        "all_footprints": fp["rows"],
    }


def verdict(rows: list[dict], gate: float = GATE) -> dict:
    judgeable = [r for r in rows if r["judgeable"]]
    if len(judgeable) < 2:
        return {"verdict": "NOT_DECIDABLE",
                "reason": f"only {len(judgeable)} judgeable grid(s) (U > {gate:.0%})",
                "grids_judgeable": [r["tag"] for r in judgeable]}
    fine, coarse = judgeable[-1], judgeable[-2]
    if (fine["residual"] < 0.0) != (coarse["residual"] < 0.0):
        return {"verdict": "NOT_DECIDABLE",
                "reason": f"sign flip: {coarse['tag']} {coarse['residual']:+.1%} vs "
                          f"{fine['tag']} {fine['residual']:+.1%}",
                "grids_judgeable": [r["tag"] for r in judgeable]}
    worst = max(judgeable, key=lambda r: abs(r["residual"]))
    ok = all(abs(r["residual"]) <= gate for r in judgeable)
    return {"verdict": "PASS" if ok else ("FAIL_above" if worst["residual"] > 0 else "FAIL_below"),
            "reason": f"two finest judgeable grids {coarse['tag']}, {fine['tag']} share sign; "
                      f"worst |r| = {abs(worst['residual']):.1%} at {worst['tag']}",
            "grids_judgeable": [r["tag"] for r in judgeable],
            "paired": [coarse["tag"], fine["tag"]]}


def build(art_dir: Path) -> dict:
    point = eb.wagner_peak_over_q(math.radians(BETA_DEG))
    rows = [grid_record(tag, art_dir, point) for tag in GRID_TAGS
            if (Path(art_dir) / f"ensemble_{tag}_c0v20.json").exists()]
    missing = [t for t in GRID_TAGS
               if not (Path(art_dir) / f"ensemble_{t}_c0v20.json").exists()]
    rows.sort(key=lambda r: -r["dx_m"])
    designated = lr.designated_cp_max(BETA_DEG)
    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/footprint_ledger.py",
        "design": {"gate": GATE, "phi_m": PHI_M,
                   "metric": "Cp_Phi = max mean(p/q) over sliding windows of width Phi",
                   "reference": "Wagner field through the same footprint operator",
                   "residual": "obs/synth - 1", "uncertainty": "frame bootstrap 5..95 %",
                   "verdict_rule": "ladder_closure.py rule, unchanged"},
        "wagner_cp_max_point": point,
        "designated": {"model": lr.DESIGNATED_MODEL, "cp_q": designated},
        "exact": {"model": lr.EXACT_MODEL, "cp_q": lr.cp_max_model(lr.EXACT_MODEL, BETA_DEG)},
        "reference_half_spread_rel": lr.band(BETA_DEG)["rel_half_spread"],
        "grids": rows,
        "missing_grids": missing,
        "verdict": verdict(rows),
    }


def figures(rep: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    g = rep["grids"]
    dx = [r["dx_m"] * 1e3 for r in g]

    # fig1: footprint reading vs grid, with reference levels and bootstrap bars
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    obs = [r["obs_cp"] for r in g]
    lo = [r["obs_cp"] - (r["bootstrap"].get("p95", r["obs_cp"]) - r["bootstrap"].get("p05", r["obs_cp"])) / 2 for r in g]
    hi = [r["obs_cp"] + (r["bootstrap"].get("p95", r["obs_cp"]) - r["bootstrap"].get("p05", r["obs_cp"])) / 2 for r in g]
    ax.errorbar(dx, obs, yerr=[[o - l for o, l in zip(obs, lo)], [h - o for h, o in zip(hi, obs)]],
                marker="o", label=r"SPH, $\Phi$=8 mm footprint")
    ax.plot(dx, [r["synth_cp"] for r in g], marker="s", ls="--", label="Wagner through same operator")
    ax.axhline(rep["wagner_cp_max_point"], color="grey", ls=":", label="Wagner point max %.2f" % rep["wagner_cp_max_point"])
    ax.axhline(rep["designated"]["cp_q"], color="k", ls="-.", label="Zhao&Faltinsen COM %.2f" % rep["designated"]["cp_q"])
    ax.axhline(rep["exact"]["cp_q"], color="tab:green", ls="--", lw=1, label="exact similarity %.2f" % rep["exact"]["cp_q"])
    ax.set_xscale("log"); ax.set_xlabel("dx [mm]"); ax.set_ylabel(r"$C_p$")
    ax.set_title("Footprint-averaged peak pressure, $\\beta=30^\\circ$")
    ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout()
    p = fig_dir / "fig1_footprint_ladder.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # fig2: residuals vs gate
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(dx, [r["residual"] for r in g], marker="o")
    ax.axhspan(-GATE, GATE, color="tab:green", alpha=.15, label="15 % gate")
    ax.axhline(0, color="k", lw=.8)
    for r in g:
        ax.annotate(f"{r['residual']:+.1%}\nU={r['uncertainty']:.0%}", (r["dx_m"] * 1e3, r["residual"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("dx [mm]"); ax.set_ylabel("same-operator residual")
    ax.set_title("Residual against the same-operator reference"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "fig2_footprint_residual.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # fig3: footprint sensitivity per grid (all footprints)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for r in g:
        ax.plot([q["footprint_cells"] for q in r["all_footprints"]],
                [q["cp_peak"] for q in r["all_footprints"]], marker=".", label=r["tag"])
    ax.axhline(rep["designated"]["cp_q"], color="k", ls="-.", lw=1, label="Z&F COM")
    ax.set_xlabel("footprint [cells]"); ax.set_ylabel(r"$C_p$")
    ax.set_title("Footprint sensitivity (variance falls with footprint)")
    ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout()
    p = fig_dir / "fig3_footprint_sensitivity.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(ROOT / "docs/artifacts/case_sph_wedge_entry_2d_footprint"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out_dir = Path(args.out) if args.out else Path(args.artifacts)
    rep = build(Path(args.artifacts))
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = figures(rep, out_dir)
    rep["figures"] = [str(p.relative_to(out_dir)) for p in figs]
    (out_dir / "footprint_ledger.json").write_text(json.dumps(rep, indent=1))
    print("[footprint] Phi = %.1f mm" % (PHI_M * 1e3))
    for r in rep["grids"]:
        print(f"  {r['tag']:8s} cells={r['footprint_cells']:5.1f} obs={r['obs_cp']:.3f} "
              f"synth={r['synth_cp']:.3f} r={r['residual']:+.3f} U={r['uncertainty']:.3f} "
              f"judgeable={r['judgeable']}")
    print(f"  verdict: {rep['verdict']['verdict']} -- {rep['verdict']['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
