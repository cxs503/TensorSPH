"""Is the frame ensemble behind the peak ``Cp`` of CASE-SPH-WEDGE-ENTRY-2D stationary?

Why this module exists
----------------------
``ensemble_peak.py`` (m17899) grades the peak wall pressure of the wedge-entry
case by *pooling* 61 frames into one self-similar profile ``Cp(tau)``,
``tau = |x| / (C V t)``, and reading the peak of that pooled profile.  Pooling is
only legitimate if the frames are members of one ensemble.  That assumption has
never been tested, and two committed artefacts contradict each other about it:

* ``ensemble_dx2mm_c0v20.json:by_frames.61.judgeable`` is **True**
  (``spread_rel = 0.140 < 0.15``), i.e. "this grid can be graded";
* the *same* record carries ``jackknife_halves = {early 8.208, late 5.742}``,
  ``jackknife_rel = 0.4295`` -- splitting the window in two moves the reading by
  43 %.  ``ladder_closure.py`` (m20161) reads ``max(band, jackknife)`` and
  therefore calls the *same* grid NOT judgeable.

Both modules are in the same directory and disagree about one number.  The
looser one (``judgeable = spread over the bin grid < 15 %``) is the one that
writes the artefacts, so a false positive can flow into a report.  This module
re-derives the question from the per-frame data that the artefacts already
carry (``frame_scan``), with no solver run and no new threshold, and answers
three questions:

Q1 (mechanism).  Is the point peak a physical feature or an extreme-value
    statistic of a noisy field?  Test: does the per-frame peak grow with the
    number of wetted nodes ``n_wet`` (more samples -> larger maximum), and does
    the location of the peak jump between the stagnation region and the spray
    root from frame to frame?
Q2 (attribution).  Is the 43 % early/late split at ``dx = 2 mm`` a property of
    the flow, or of the fact that the early half simply has fewer wetted nodes?
    Test: re-split the window after a declared wet-node gate.
Q3 (grading).  What uncertainty and judgeability follow when the half-split is
    counted, and does that change the case verdict?

Predeclared rules (fixed before looking at any number)
------------------------------------------------------
* **Wet-node gate**: a frame may enter the ensemble only if ``n_wet >= 16``.
  Justification is internal to the estimator, not chosen to pass: the pooled
  profile is read at bin counts up to ``BIN_GRID``'s 16 (and the resolution band
  starts at 8), and ``MIN_BIN_SAMPLES`` already forbids a bin from setting the
  peak on fewer than its own minimum; a *frame* contributing fewer than 16 wall
  nodes cannot populate the coarsest admissible binning at all.
* **Uncertainty** of a grid reading = ``max(resolution_band.spread_rel,
  jackknife_rel)`` -- the same rule ``ladder_closure.py`` already declares, so
  this module never introduces a new discriminant, only applies the stricter of
  the two committed ones.
* **Judgeable** = uncertainty <= 0.15 (the acceptance gate of the case).
* Nothing here changes the solver, the case definition, the reference value or
  the gate.  It only audits the estimator that is already committed.

Outputs: ``docs/artifacts/case_sph_wedge_entry_2d_stationarity/stationarity.json``
and ``docs/figures/case_sph_wedge_entry_2d_stationarity.png``.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/frame_stationarity.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

GATE = 0.15                     # acceptance tolerance of the case
WET_NODE_GATE = 16              # predeclared, see module docstring
GRIDS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")
FRAME_KEY = "61"                # the frame count all committed readings use
#: n_wet strata for the extreme-value signature (upper bound exclusive)
EV_EDGES = (1, 6, 16, 31, 61, 10 ** 9)


# --------------------------------------------------------------------------- #
# loading / shaping
# --------------------------------------------------------------------------- #
def artifact_path(tag: str, root: Path | str) -> Path:
    return Path(root) / f"ensemble_{tag}_c0v20.json"


def load_artifact(tag: str, root: Path | str) -> dict:
    p = artifact_path(tag, root)
    art = json.loads(p.read_text())
    art.setdefault("_tag", tag)
    art["_tag"] = tag
    return art


def frame_table(art: dict) -> list[dict]:
    """Frames with a finite peak (the ``t = 0`` frame has no wetted node).

    ``cp_peak_single_frame`` is ``nan`` exactly when ``n_wet == 0``; the artefact's
    own ``argmax_frame`` block picks that frame (``nan`` compares False against
    everything), so the committed ``argmax_frame`` is meaningless and is not used
    here.  Dropping it is not a filter of convenience: a frame with no wetted node
    has no wall pressure to contribute.
    """
    out = []
    for f in art["frame_scan"]:
        cp = f.get("cp_peak_single_frame")
        if cp is None or not (cp == cp):
            continue
        out.append({"t": f["t"], "n_wet": int(f["n_wet"]), "cp": float(cp),
                    "tau_at_peak": f.get("tau_at_peak"),
                    "cp_centre": f.get("cp_centre"),
                    "max_rho_ratio": f.get("max_rho_ratio")})
    return out


def median(values) -> float:
    vals = [v for v in values if v == v]
    return statistics.median(vals) if vals else float("nan")


def gated(frames: list[dict], wet_gate: int = 0) -> list[dict]:
    return [f for f in frames if f["n_wet"] >= wet_gate]


# --------------------------------------------------------------------------- #
# Q1: extreme-value signature
# --------------------------------------------------------------------------- #
def ev_strata(frames: list[dict], edges=EV_EDGES) -> list[dict]:
    """Median per-frame peak per ``n_wet`` stratum (sample-count dependence)."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [f for f in frames if lo <= f["n_wet"] < hi]
        vals = [f["cp"] for f in sel]
        rows.append({"n_wet_lo": lo, "n_wet_hi": hi, "n": len(vals),
                     "median_n_wet": median([f["n_wet"] for f in sel]),
                     "median_cp": median(vals),
                     "max_cp": max(vals) if vals else float("nan")})
    return rows


def ev_slope(rows: list[dict]) -> float:
    """Least-squares slope of ``log(median cp)`` on ``log(median n_wet)``.

    A positive slope is the signature of an extreme-value statistic: adding
    samples raises the maximum.  A physical peak would be scale-free here.
    """
    pts = []
    for r in rows:
        if r["n"] < 3 or not (r["median_cp"] == r["median_cp"]) or r["median_cp"] <= 0:
            continue
        # representative sample count of the stratum: the *observed* median, not
        # the declared edge (the top stratum's upper edge is unbounded)
        mw = r.get("median_n_wet", 0.5 * (r["n_wet_lo"] + min(r["n_wet_hi"], 120)))
        pts.append((math.log(max(2.0, mw)), math.log(r["median_cp"])))
    if len(pts) < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return float("nan")
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx


def peak_location_flip(frames: list[dict], lo: float = 0.30, hi: float = 0.70) -> dict:
    """Fraction of frames whose peak sits near the stagnation line vs spray root.

    The analytic wall profile rises monotonically towards ``tau -> 1``, so a
    genuine point maximum lives at the spray root.  Frames whose maximum sits at
    ``tau < 0.3`` are reading noise on the wetted face, not the physical peak.
    """
    taus = [f["tau_at_peak"] for f in frames if f["tau_at_peak"] == f["tau_at_peak"]]
    if not taus:
        return {"n": 0, "stagnation": float("nan"), "spray": float("nan")}
    n = len(taus)
    return {"n": n,
            "stagnation": sum(1 for t in taus if t < lo) / n,
            "spray": sum(1 for t in taus if t > hi) / n}


def dispersion(frames: list[dict]) -> dict:
    cps = [f["cp"] for f in frames]
    med = median(cps)
    return {"n": len(cps), "median_cp": med,
            "cov": (statistics.pstdev(cps) / med) if med else float("nan"),
            "iqi_ratio": (median([c for c in cps if c > med]) /
                          median([c for c in cps if c < med]))
            if len(cps) > 3 and med else float("nan")}


# --------------------------------------------------------------------------- #
# Q2: half-split attribution
# --------------------------------------------------------------------------- #
def half_split(frames: list[dict], wet_gate: int = 0) -> dict:
    """Early/late half medians of the per-frame peak, optionally wet-node gated.

    Deliberately a *different route* from ``ensemble_peak.jackknife_halves``
    (which pools the profile and reads a gated bin peak): if the two routes agree
    the split is not an artefact of one estimator.
    """
    sel = gated(frames, wet_gate)
    h = len(sel) // 2
    early, late = sel[:h], sel[h:]
    e, l = median(f["cp"] for f in early), median(f["cp"] for f in late)
    rel = abs(l - e) / max(abs(l), 1e-12) if l == l and e == e else float("nan")
    return {"n_used": len(sel), "n_early": len(early), "n_late": len(late),
            "early_median": e, "late_median": l, "rel_split": rel,
            "early_mean_n_wet": statistics.mean([f["n_wet"] for f in early]) if early else float("nan"),
            "late_mean_n_wet": statistics.mean([f["n_wet"] for f in late]) if late else float("nan")}


def gate_sweep(frames: list[dict], gates=(0, 8, 16, 30, 60)) -> list[dict]:
    return [dict(wet_gate=g, **half_split(frames, g)) for g in gates]


# --------------------------------------------------------------------------- #
# Q3: grading ledger
# --------------------------------------------------------------------------- #
def judgeability_ledger(art: dict, frame_key: str = FRAME_KEY) -> dict:
    """Three committed/declared readings of the same grid, side by side."""
    rec = art["by_frames"][frame_key]
    rb = rec["resolution_band"]["spread_rel"]
    jk = rec["jackknife_rel"]
    full = rec["spread_rel"]
    unc = max(rb, jk) if (rb == rb and jk == jk) else float("nan")
    return {
        "artifact_judgeable": bool(rec["judgeable"]),
        "artifact_judgeable_rule": "spread over full bin grid < 15 %",
        "artifact_basis_spread_rel": full,
        "resolution_band_peak_median": rec["resolution_band"]["peak_median"],
        "resolution_band_spread_rel": rb,
        "jackknife_rel": jk,
        "declared_uncertainty": unc,
        "stationarity_aware_judgeable": bool(unc == unc and unc <= GATE),
        "contradiction": bool(rec["judgeable"]) != bool(unc == unc and unc <= GATE),
    }


def audit(root: Path | str, grids=GRIDS) -> dict:
    out = {"gate": GATE, "wet_node_gate": WET_NODE_GATE, "grids": {}}
    for tag in grids:
        p = artifact_path(tag, root)
        if not p.exists():
            out["grids"][tag] = {"missing": str(p)}
            continue
        art = load_artifact(tag, root)
        frames = frame_table(art)
        strata = ev_strata(frames)
        gated_frames = gated(frames, WET_NODE_GATE)
        out["grids"][tag] = {
            "dx": art["config"]["dx"],
            "n_frames": len(art["frame_scan"]),
            "n_frames_used": len(frames),
            "n_frames_gated": len(gated_frames),
            "ev_strata": strata,
            "ev_slope": ev_slope(strata),
            "peak_location": peak_location_flip(frames),
            "dispersion": dispersion(frames),
            "half_split_raw": half_split(frames, 0),
            "half_split_gated": half_split(frames, WET_NODE_GATE),
            "gate_sweep": gate_sweep(frames),
            "ledger": judgeability_ledger(art),
        }
    out["summary"] = summarise(out)
    return out


def summarise(audit_result: dict) -> dict:
    per = {}
    for tag, g in audit_result["grids"].items():
        if "ledger" not in g:
            continue
        raw = g["half_split_raw"]["rel_split"]
        gt = g["half_split_gated"]["rel_split"]
        per[tag] = {
            "ev_slope": g["ev_slope"],
            "half_split_raw": raw,
            "half_split_gated": gt,
            "split_explained_by_wet_nodes": bool(raw == raw and gt == raw - raw),
            "uncertainty": g["ledger"]["declared_uncertainty"],
            "stationarity_aware_judgeable": g["ledger"]["stationarity_aware_judgeable"],
            "artifact_judgeable": g["ledger"]["artifact_judgeable"],
        }
    judgeable = [t for t, v in per.items() if v["stationarity_aware_judgeable"]]
    return {"per_grid": per, "grids_judgeable": judgeable,
            "n_judgeable": len(judgeable),
            "contradictions": [t for t, v in per.items()
                               if v["artifact_judgeable"] != v["stationarity_aware_judgeable"]]}


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(audit_result: dict, out_png: Path | str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = [t for t in audit_result["grids"] if "ledger" in audit_result["grids"][t]]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for tag in grids:
        g = audit_result["grids"][tag]
        rows = [r for r in g["ev_strata"] if r["n"] >= 3]
        if not rows:
            continue
        x = [0.5 * (r["n_wet_lo"] + min(r["n_wet_hi"], 120)) for r in rows]
        y = [r["median_cp"] for r in rows]
        ax.plot(x, y, "o-", label=f"{tag} (slope {g['ev_slope']:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("wetted wall nodes in the frame  $n_{wet}$")
    ax.set_ylabel("median per-frame peak $C_p$")
    ax.set_title("Q1  peak grows with sample count\n(extreme-value signature)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    w = 0.35
    xs = range(len(grids))
    raw = [audit_result["grids"][t]["half_split_raw"]["rel_split"] for t in grids]
    gt = [audit_result["grids"][t]["half_split_gated"]["rel_split"] for t in grids]
    ax.bar([x - w / 2 for x in xs], raw, w, label="all frames")
    ax.bar([x + w / 2 for x in xs], gt, w,
           label=f"$n_{{wet}}\\geq${audit_result['wet_node_gate']} gate")
    ax.axhline(audit_result["gate"], color="crimson", ls="--",
               label=f"{audit_result['gate']:.0%} gate")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(grids)
    ax.set_ylabel("early/late half split (rel.)")
    ax.set_title("Q2  is the split caused by wet-node count?")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    peak = [audit_result["grids"][t]["ledger"]["resolution_band_peak_median"] for t in grids]
    err = [audit_result["grids"][t]["ledger"]["declared_uncertainty"] for t in grids]
    ax.errorbar([audit_result["grids"][t]["dx"] * 1000 for t in grids], peak,
                yerr=[p * e for p, e in zip(peak, err)], fmt="o-", capsize=4)
    for t, p, e in zip(grids, peak, err):
        ax.annotate(f"{t}\n$U$={e:.0%}", (audit_result["grids"][t]["dx"] * 1000, p),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("$dx$ (mm)")
    ax.set_ylabel("resolution-band peak $C_p$")
    ax.set_title("Q3  ladder with stationarity-aware uncertainty")
    ax.grid(alpha=0.3)

    fig.suptitle("CASE-SPH-WEDGE-ENTRY-2D — frame-ensemble stationarity audit "
                 "(committed artefacts, no solver run)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return str(out_png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="docs/artifacts/case_sph_wedge_entry_2d_ensemble")
    ap.add_argument("--out", default="docs/artifacts/case_sph_wedge_entry_2d_stationarity")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    res = audit(args.artifacts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stationarity.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res["summary"], indent=1))
    for tag, g in res["grids"].items():
        if "ledger" not in g:
            print(tag, "MISSING")
            continue
        print(f"{tag:8s} ev_slope={g['ev_slope']:+.2f} "
              f"raw_split={g['half_split_raw']['rel_split']:.3f} "
              f"gated_split={g['half_split_gated']['rel_split']:.3f} "
              f"U={g['ledger']['declared_uncertainty']:.3f} "
              f"artifact_judgeable={g['ledger']['artifact_judgeable']} "
              f"judgeable={g['ledger']['stationarity_aware_judgeable']}")
    if not args.no_figure:
        png = make_figure(res, "docs/figures/case_sph_wedge_entry_2d_stationarity.png")
        print("figure:", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())