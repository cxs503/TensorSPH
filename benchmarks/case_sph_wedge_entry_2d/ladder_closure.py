"""Does the peak slamming pressure become *judgeable* when the ladder is closed?

Context (measured, not assumed)
-------------------------------
CASE-SPH-WEDGE-ENTRY-2D (beta = 30 deg, V = 1 m/s, self-similar) had two
predeclared routes left open by the earlier rounds:

* m17548 -- the binned wall-profile peak at ``dx = 2 mm`` swings over a 3x range
  when the bin count changes (bin narrower than a particle spacing = a single
  particle's noise); route (a) more frames / gated binning, route (b) ``dx <= 1 mm``;
* m17899 -- route (a) was taken: the 61-frame self-similar ensemble with a
  sample-gated, resolution-consistent bin band gives 6.07 at ``dx = 2 mm`` and
  10.36 at ``dx = 4 mm``, whose *same-operator* residuals to Wagner (-21.9 % /
  +50.8 %, m18382) still change sign between grids, so the peak was declared
  **not judgeable at dx >= 2 mm** and route (b) was left as the open item.

This script closes route (b) for the judgement only: the SPH runs at
``dx = 1 mm`` and ``dx = 0.5 mm`` are produced by ``ensemble_peak.py`` (same
solver, same case, same estimator, same 61 self-similar frames); nothing here
re-runs the solver, changes the case, or relaxes a threshold.  What is new is
(a) the two missing rungs and (b) a grid-convergence reading of the *same
operator* residual, which is the quantity that actually competes with the 15 %
gate.

Predeclared decision rule (fixed before the dx <= 1 mm artefacts existed)
------------------------------------------------------------------------
Per grid ``g`` with spacing ``dx_g``:

    obs_g   = ensemble artefact, ``by_frames["61"].resolution_band.peak_median``
              (the resolution-consistent bin band of the gated estimator; the
              same statistic m17899 reported, not a new metric choice)
    U_g     = max(resolution_band.spread_rel, jackknife_rel)   (estimator spread)
    truth_g = bias report, ``synth_smoothed.peak_median`` at 61 frames
              (Wagner's field pushed through the *same* measurement operator)
    r_g     = obs_g / truth_g - 1                              (same-operator residual)

1. **Judgeable grid**: ``U_g <= 0.15`` (the case tolerance; unchanged).
2. **Case verdict**, read off the two *finest* judgeable grids ``f`` and ``f'``:

   * ``PASS``            : both judgeable, same sign of ``r``, ``|r_f| <= 0.15``;
   * ``FAIL_below`` / ``FAIL_above``: both judgeable, same sign of ``r``,
     ``|r_f| > 0.15`` (the sign says on which side of Wagner the solution sits);
   * ``NOT_DECIDABLE``   : sign flip between ``f`` and ``f'``, or ``f'`` not
     judgeable, or ``U_f > 0.15``.  A single grid inside +/-15 % is *not*
     accepted as PASS -- that is exactly the trap m18382 documented.

3. **Grid convergence** (reported, never used to overrule 2): observed order
   from the residuals of the finest same-sign judgeable triple,

       p = ln(|r_coarse| / |r_fine|) / ln(dx_coarse / dx_fine)

   and a one-term Richardson estimate of the asymptotic same-operator residual
   ``r_inf = r_fine / (2^p - 1)``; the extrapolated peak is then
   ``obs_inf = obs_fine - r_inf * truth_fine``.  Both are flagged
   ``trustworthy = False`` when ``p`` leaves ``(0, 3]`` or when the per-pair
   orders disagree by more than 30 %, because a Richardson estimate without a
   monotone observed order is numerology.

The Zhao & Faltinsen (1993) table value at beta = 30 deg stays **WITHHELD**
(closed access, no registered copy in the ecosystem): this script does not
paste a remembered or secondary number into a cell that claims to be the
paper's.  It only makes the *Wagner same-operator* comparison judgeable or
honestly not.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/ladder_closure.py \
        --artifacts docs/artifacts/case_sph_wedge_entry_2d_ensemble \
        --bias docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json \
        --out docs/artifacts/case_sph_wedge_entry_2d_ladder
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# --- predeclared design (do not tune after seeing the fine-grid numbers) ---- #
GATE = 0.15                    # case tolerance on the peak Cp (unchanged)
FRAME_COUNT = "61"             # the self-similar ensemble size used throughout
GRID_TAGS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")
ORDER_RANGE = (0.0, 3.0)       # plausible observed orders for this scheme
ORDER_CONSISTENCY = 0.30       # max relative disagreement between pair orders


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and v == v and math.isinf(v) is False


def load_grids(artifacts: Path, bias: dict) -> list[dict]:
    """One record per grid, sorted coarse -> fine; absent grids are skipped."""
    rows = []
    for tag in GRID_TAGS:
        path = Path(artifacts) / f"ensemble_{tag}_c0v20.json"
        if not path.exists():
            continue
        art = json.loads(path.read_text())
        dx = art["config"]["dx"]
        frames = art["by_frames"].get(FRAME_COUNT)
        if frames is None:
            continue
        rb = frames["resolution_band"]
        obs = rb["peak_median"]
        u_spread = rb["spread_rel"]
        u_jk = frames.get("jackknife_rel", float("nan"))
        u = max(v for v in (u_spread, u_jk) if _finite(v)) if any(
            _finite(v) for v in (u_spread, u_jk)) else float("nan")
        grid_bias = (bias.get("grids", {}).get(tag) or {}).get("by_frames", {}).get(
            FRAME_COUNT, {})
        synth = grid_bias.get("synth_smoothed", {})
        truth = synth.get("peak_median", float("nan"))
        truth_spread = synth.get("spread_rel", float("nan"))
        residual = ((obs - truth) / truth
                    if _finite(obs) and _finite(truth) and truth else float("nan"))
        rows.append({
            "tag": tag,
            "dx": dx,
            "obs_peak": obs,
            "obs_spread_rel": u_spread,
            "obs_jackknife_rel": u_jk,
            "uncertainty": u,
            "synth_peak": truth,
            "synth_spread_rel": truth_spread,
            "residual": residual,
            "err_vs_wagner_point": frames.get("err_vs_wagner_median", float("nan")),
            # judgeable needs both a bounded estimator spread and a truth to
            # compare against; a grid with no same-operator truth cannot be judged.
            "judgeable": bool(_finite(u) and u <= GATE and _finite(residual)),
            "n_bins_used": rb.get("n_bins_used", []),
            "pooled_samples": frames.get("pooled_samples"),
            "mean_wet_nodes": frames.get("mean_wet_nodes"),
            "wall_seconds": art.get("wall_seconds"),
            "steps": art.get("steps"),
        })
    rows.sort(key=lambda r: -r["dx"])
    return rows


def verdict(rows: list[dict], gate: float = GATE) -> dict:
    """Apply the predeclared decision rule to the two finest judgeable grids."""
    judgeable = [r for r in rows if r["judgeable"]]
    if len(judgeable) < 2:
        return {"verdict": "NOT_DECIDABLE",
                "reason": f"only {len(judgeable)} judgeable grid(s) "
                          f"(estimator spread > {gate:.0%})",
                "grids_judgeable": [r["tag"] for r in judgeable]}
    fine, coarse = judgeable[-1], judgeable[-2]
    same_sign = (fine["residual"] < 0.0) == (coarse["residual"] < 0.0)
    if not same_sign:
        return {"verdict": "NOT_DECIDABLE",
                "reason": (f"same-operator residual changes sign between "
                           f"{coarse['tag']} ({coarse['residual']*100:+.1f} %) and "
                           f"{fine['tag']} ({fine['residual']*100:+.1f} %)"),
                "grids_judgeable": [r["tag"] for r in judgeable]}
    if abs(fine["residual"]) <= gate:
        return {"verdict": "PASS",
                "reason": (f"|residual| = {abs(fine['residual'])*100:.1f} % <= "
                           f"{gate:.0%} at {fine['tag']}, sign stable vs {coarse['tag']}"),
                "grids_judgeable": [r["tag"] for r in judgeable]}
    side = "below" if fine["residual"] < 0.0 else "above"
    return {"verdict": f"FAIL_{side}",
            "reason": (f"|residual| = {abs(fine['residual'])*100:.1f} % > {gate:.0%} "
                       f"at {fine['tag']}, sign stable vs {coarse['tag']}"),
            "grids_judgeable": [r["tag"] for r in judgeable]}


def observed_order(rows: list[dict], gate: float = GATE) -> dict:
    """Observed order + one-term Richardson on the finest same-sign triple."""
    judgeable = [r for r in rows if r["judgeable"]]
    # keep the longest same-sign suffix of the residual sequence
    if not judgeable:
        return {"available": False, "reason": "no judgeable grid"}
    sign = judgeable[-1]["residual"] < 0.0
    suffix = []
    for r in reversed(judgeable):
        if (r["residual"] < 0.0) == sign:
            suffix.append(r)
        else:
            break
    suffix.reverse()
    if len(suffix) < 3:
        return {"available": False,
                "reason": f"need 3 same-sign judgeable grids, have {len(suffix)}",
                "grids": [r["tag"] for r in suffix]}
    triple = suffix[-3:]
    pairs = []
    for coarse, fine in zip(triple[:-1], triple[1:]):
        ratio_obs = abs(coarse["residual"]) / abs(fine["residual"])
        ratio_dx = coarse["dx"] / fine["dx"]
        p = (math.log(ratio_obs) / math.log(ratio_dx)
             if ratio_obs > 0.0 and ratio_dx != 1.0 else float("nan"))
        pairs.append({"coarse": coarse["tag"], "fine": fine["tag"],
                      "residual_ratio": ratio_obs, "dx_ratio": ratio_dx,
                      "order": p})
    orders = [p["order"] for p in pairs if _finite(p["order"])]
    if not orders:
        return {"available": False, "reason": "order undefined (zero residual)",
                "pairs": pairs}
    p_mean = statistics.mean(orders)
    lo, hi = ORDER_RANGE
    consistent = (len(orders) == 1 or
                  all(abs(o - p_mean) <= ORDER_CONSISTENCY * abs(p_mean)
                      for o in orders))
    trustworthy = bool(lo < p_mean <= hi and consistent and p_mean > 0)
    fine = triple[-1]
    r_inf = float("nan")
    obs_inf = float("nan")
    if trustworthy:
        r_inf = fine["residual"] / (2.0 ** p_mean - 1.0)
        obs_inf = fine["obs_peak"] - r_inf * fine["synth_peak"]
    return {"available": True, "pairs": pairs, "order_mean": p_mean,
            "consistent": consistent, "trustworthy": trustworthy,
            "residual_asymptotic": r_inf, "obs_peak_asymptotic": obs_inf,
            "grids": [r["tag"] for r in triple]}


def build(artifacts: Path, bias_path: Path) -> dict:
    bias = json.loads(Path(bias_path).read_text()) if Path(bias_path).exists() else {}
    rows = load_grids(artifacts, bias)
    res = verdict(rows)
    order = observed_order(rows)
    wagner_point = bias.get("wagner_cp_max_point", float("nan"))
    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/ladder_closure.py",
        "design": {"gate": GATE, "frame_count": FRAME_COUNT,
                   "grid_tags": list(GRID_TAGS),
                   "order_range": list(ORDER_RANGE),
                   "order_consistency": ORDER_CONSISTENCY,
                   "metric": "by_frames[61].resolution_band.peak_median",
                   "truth": "bias_report synth_smoothed.peak_median (same operator)"},
        "wagner_cp_max_point": wagner_point,
        "grids": rows,
        "missing_grids": [t for t in GRID_TAGS
                          if t not in {r["tag"] for r in rows}],
        "verdict": res,
        "grid_convergence": order,
        "zhao_faltinsen_1993_beta30": {
            "status": "WITHHELD",
            "reason": ("closed access, no registered copy in the ecosystem; "
                       "no remembered/secondary number is substituted")},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path,
                    default=ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ensemble")
    ap.add_argument("--bias", type=Path,
                    default=ROOT / "docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ladder")
    args = ap.parse_args(argv)
    report = build(args.artifacts, args.bias)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ladder_closure.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"[ladder] grids: {[r['tag'] for r in report['grids']]} -> {path}")
    for r in report["grids"]:
        print(f"  {r['tag']:>8}  dx={r['dx']*1e3:4.2f} mm  obs={r['obs_peak']:.3f}"
              f"  U={r['uncertainty']*100:5.1f}%  truth={r['synth_peak']:.3f}"
              f"  residual={r['residual']*100:+7.1f}%"
              f"  judgeable={r['judgeable']}")
    print(f"  verdict: {report['verdict']['verdict']} -- {report['verdict']['reason']}")
    gc = report["grid_convergence"]
    if gc.get("available"):
        print(f"  observed order p={gc['order_mean']:.2f} trustworthy={gc['trustworthy']}"
              f"  residual_inf={gc['residual_asymptotic']*100:+.1f}%"
              f"  obs_inf={gc['obs_peak_asymptotic']:.3f}")
    else:
        print(f"  observed order: unavailable ({gc.get('reason')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())