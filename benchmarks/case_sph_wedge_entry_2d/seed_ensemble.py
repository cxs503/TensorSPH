"""Independent realizations of the wedge-entry case: is the peak *judgeable*?

Context (measured, not assumed)
-------------------------------
``docs/case_sph_wedge_entry_2d_noise_floor_m22798.md`` closed the estimator
question for CASE-SPH-WEDGE-ENTRY-2D: the frame-level peak reading of the
committed case has a residual against the *same-operator* Wagner reference that
is positive at every grid and decays as ``dx^{+0.64}``, i.e. like the upward
bias of a maximum taken on a noisy field (``dx^{+1/2}``) and not like a
second-order discretisation error.  Its conclusion lists the cheapest way to
actually decide the case:

    "the cheapest road to a verdict is **independent realizations**, not a finer
    grid: 4 independent implementations cut the standard error of the
    fine-grid reading (~ +16.6 %) to ~8.3 %, i.e. inside the 15 % gate, at
    ~2.6 core-hours instead of the ~8x particle count a ``dx = 0.38 mm``
    trajectory would need.  The change is small but belongs to the solver:
    ``WedgeConfig`` gains a predeclared initial-state perturbation (e.g. a
    lattice phase jitter of amplitude ``<= dx/50``), written into the case
    definition."

This script is that round.  It adds nothing to the physics: the *solver* change
is two sub-cell discrete-phase offsets (``WedgeConfig.phase_x_dx`` and
``gap_jitter_dx``, drawn from a seed by
``tensorsph.wedge_entry.realisation_phases``), both default-off so every
committed artefact still reproduces bit for bit; this driver only runs one
realization per call and aggregates them.

Predeclared decision rule (fixed before the realization runs existed)
--------------------------------------------------------------------
Realizations ``i = 1..N`` of grid ``g`` (same case, same 61 self-similar frames,
same estimator, only the discrete phase differs):

    obs_i    = artefact, by_frames["61"].resolution_band.peak_median
               (the statatistic ladder_closure.py judges; not a new metric)
    truth_g  = bias report, grids[tag].by_frames["61"].synth_smoothed.peak_median
               (Wagner's field through the *same* measurement operator;
               seed-independent, so it is not re-derived here)
    r_i      = obs_i / truth_g - 1                       (same-operator residual)
    U_real   = max( t_{0.95,N-1} * sd(r_i) / sqrt(N),     (realization CI halfwidth)
                    median_i(band spread_i) )             (within-run band spread)
    judgeable_g  <=>  U_real <= 0.15                     (the case tolerance)
    verdict      = ladder_closure.verdict(rows)          (unchanged rule)

``t_{0.95,N-1}`` (two-sided, 95 %) rather than 1.96 because ``N`` is 4-6.

Two auxiliary questions, also predeclared, because the whole plan rests on them:

* **A1 -- are the realizations actually independent?**  Pairwise Pearson
  correlation of the per-frame ``cp_centre`` series, and the ratio
  ``sd_between(r_i) / sd_within(frame peaks)``.  A knob that does not decorrelate
  the trajectory gives ``rho -> 1`` and then the realization CI is meaningless
  (it would be a *lower* bound on the uncertainty, which is the dangerous
  direction).  Reported whatever the answer is.
* **A2 -- does the answer depend on the perturbation size?**  Same grid at
  ``scale = 0`` (bit-identical to the committed run: the determinism negative
  control), ``0.25`` and ``1``.  If the spread grows with scale the knob is not
  sampling a phase, it is changing the flow.

Usage
-----
::

    # one realization (one cluster shard)
    python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py \
        --dx 0.001 --seed 3 --threads 4 --out docs/artifacts/<dir>

    # aggregate every realization found in the directory
    python benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py \
        --analyse --out docs/artifacts/<dir>
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import ensemble_peak as ep  # noqa: E402
import ladder_closure as lc  # noqa: E402
from run_ladder_shard import BIN_BAND  # noqa: E402
from tensorsph.wedge_entry import (  # noqa: E402
    GAP_JITTER_MAX_DX,
    PHASE_X_MAX_DX,
    WedgeConfig,
    realisation_phases,
)

FRAME_COUNT = lc.FRAME_COUNT
GATE = lc.GATE
SCALE_CHECK = (0.0, 0.25, 1.0)

# Two-sided 95 % Student-t quantiles, df = 1..30 (no scipy in the image).
T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
       2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
       2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def t95(df: int) -> float:
    if df < 1:
        return float("nan")
    return T95[min(df, 30) - 1]


# --------------------------------------------------------------------------- #
# One realization
# --------------------------------------------------------------------------- #

def run_one(dx: float, seed: int, scale: float, frames: int, c0_factor: float,
            threads: int, velocity: float = 1.0) -> dict:
    """Run the case once at discrete phase ``seed`` and return its artefact."""
    t0 = time.time()
    case = ep.collect(dx, velocity, frames, c0_factor, threads,
                      seed=seed, scale=scale)
    # Primary reading: the committed bin grid, because the same-operator truth in
    # the bias report was built on it, so r_i stays the ladder_closure residual
    # and not a mixture of band change and realization change.
    result = ep.analyse(case, ep.BIN_GRID)
    # Robustness reading on the band that m22695 fixed (candidates not truncated
    # at 30); no truth exists on these bin counts, so it is reported, never judged.
    fixed = ep.analyse(case, BIN_BAND)["by_frames"][str(FRAME_COUNT)]["resolution_band"]
    series = result["frame_scan"]
    payload = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py",
        "realisation": {"seed": int(seed), "scale": float(scale),
                        "phase_x_dx": case["cfg"].get("phase_x_dx"),
                        "gap_jitter_dx": case["cfg"].get("gap_jitter_dx"),
                        "design": "sub-cell discrete phase; see WedgeConfig"},
        "design": {"frame_counts": list(ep.FRAME_COUNTS),
                   "bin_grid": list(BIN_BAND),
                   "min_bin_samples": ep.MIN_BIN_SAMPLES,
                   "spread_tolerance": ep.SPREAD_TOL,
                   "tau_max": ep.TAU_MAX},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "threads": threads},
        **result,
        "config": case["cfg"], "steps": case["steps"],
        "wall_seconds": round(case["wall_seconds"], 1),
        "realisation_wall_seconds": round(time.time() - t0, 1),
        "c0": case["c0"], "q": case["q"], "C": case["C"],
        "band_fixed_m22695": {"peak_median": fixed["peak_median"],
                              "spread_rel": fixed["spread_rel"],
                              "n_bins_used": fixed["n_bins_used"]},
        "series": {"t": [r["t"] for r in series],
                   "cp_centre": [r["cp_centre"] for r in series],
                   "cp_footprint_max": [r["cp_footprint_max"] for r in series],
                   "max_rho_ratio": [r["max_rho_ratio"] for r in series],
                   "n_wet": [r["n_wet"] for r in series]},
    }
    return payload


def artefact_name(dx: float, seed: int, scale: float) -> str:
    tag = f"dx{dx * 1e3:g}mm_c0v20_seed{int(seed)}"
    if scale != 1.0:
        tag += f"_scale{scale:g}"
    return f"real_{tag}.json"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    if any(v != v for v in a) or any(v != v for v in b):
        pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
        if len(pairs) < 3:
            return float("nan")
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def aggregate(out_dir: Path, bias_path: Path) -> dict:
    """Read every realization artefact in ``out_dir`` and apply the rule."""
    bias = json.loads(Path(bias_path).read_text())
    arts = sorted(Path(out_dir).glob("real_dx*.json"))
    by_grid: dict[float, list[dict]] = {}
    for path in arts:
        art = json.loads(path.read_text())
        art["_path"] = str(path)
        by_grid.setdefault(float(art["config"]["dx"]), []).append(art)

    grids = []
    for dx in sorted(by_grid, reverse=True):
        items = by_grid[dx]
        tag = f"dx{dx * 1e3:g}mm"
        truth_row = (bias.get("grids", {}).get(tag) or {}).get(
            "by_frames", {}).get(FRAME_COUNT, {}).get("synth_smoothed", {})
        truth = truth_row.get("peak_median", float("nan"))
        rows = []
        for art in items:
            frames = art["by_frames"].get(FRAME_COUNT)
            if frames is None:
                continue
            rb = frames["resolution_band"]
            obs = rb["peak_median"]
            resid = (obs / truth - 1.0) if truth and truth == truth else float("nan")
            rows.append({
                "seed": art["realisation"]["seed"],
                "scale": art["realisation"]["scale"],
                "phase_x_dx": art["realisation"]["phase_x_dx"],
                "gap_jitter_dx": art["realisation"]["gap_jitter_dx"],
                "obs_peak": obs,
                "obs_peak_band_fixed": (art.get("band_fixed_m22695") or {}).get(
                    "peak_median"),
                "band_spread_rel": rb["spread_rel"],
                "jackknife_rel": frames.get("jackknife_rel"),
                "residual": resid,
                "cp_centre": art["series"]["cp_centre"],
                "max_rho_ratio": max(
                    (v for v in art["series"]["max_rho_ratio"] if v == v),
                    default=float("nan")),
                "wall_seconds": art.get("wall_seconds"),
                "steps": art.get("steps"),
            })
        rows.sort(key=lambda r: (r["scale"], r["seed"]))
        # realization ensemble = the nominal scale only (scale variants are A2)
        nom = [r for r in rows if r["scale"] == 1.0]
        res = [r["residual"] for r in nom if r["residual"] == r["residual"]]
        u_ci = float("nan")
        sd = float("nan")
        if len(res) >= 2:
            sd = statistics.stdev(res)
            u_ci = t95(len(res) - 1) * sd / math.sqrt(len(res))
        u_band = (statistics.median([r["band_spread_rel"] for r in nom])
                  if nom else float("nan"))
        u_real = max(v for v in (u_ci, u_band) if v == v) if (
            any(v == v for v in (u_ci, u_band))) else float("nan")
        # A1: pairwise independence of the trajectories
        corr = [_pearson(a["cp_centre"], b["cp_centre"])
                for i, a in enumerate(nom) for b in nom[i + 1:]]
        corr = [c for c in corr if c == c]
        within_sd = (statistics.median([
            statistics.stdev([v for v in r["cp_centre"] if v == v])
            for r in nom if len(r["cp_centre"]) > 2]) if nom else float("nan"))
        grids.append({
            "tag": tag,
            "dx": dx,
            "truth_same_operator": truth,
            "n_realisations": len(nom),
            "realisations": [{k: v for k, v in r.items() if k != "cp_centre"}
                             for r in rows],
            "obs_median": (statistics.median([r["obs_peak"] for r in nom])
                           if nom else float("nan")),
            "residual_median": (statistics.median(res) if res else float("nan")),
            "residual_min": min(res) if res else float("nan"),
            "residual_max": max(res) if res else float("nan"),
            "residual_sd": sd,
            "U_realisation_ci": u_ci,
            "U_band": u_band,
            "uncertainty": u_real,
            "judgeable": bool(u_real == u_real and u_real <= GATE
                              and res != []),
            "independence": {
                "pairwise_cp_centre_pearson": corr,
                "pairwise_median": statistics.median(corr) if corr else float("nan"),
                "pairwise_min": min(corr) if corr else float("nan"),
                "sd_between_realisations": sd,
                "sd_within_trajectory_cp_centre": within_sd,
                "ratio_between_over_within": (sd / within_sd
                                              if within_sd and within_sd == within_sd
                                              else float("nan")),
            },
            "scale_check": _scale_check(rows),
        })

    rows_for_rule = [{"tag": g["tag"], "dx": g["dx"], "residual": g["residual_median"],
                      "uncertainty": g["uncertainty"], "judgeable": g["judgeable"],
                      "obs_peak": g["obs_median"],
                      "synth_peak": g["truth_same_operator"]}
                     for g in grids]
    rows_for_rule.sort(key=lambda r: -r["dx"])
    verdict = lc.verdict(rows_for_rule, gate=GATE)
    return {"case": "CASE-SPH-WEDGE-ENTRY-2D",
            "script": "benchmarks/case_sph_wedge_entry_2d/seed_ensemble.py",
            "rule": {
                "statistic": 'by_frames["61"].resolution_band.peak_median',
                "truth": "bias report, same-operator Wagner (seed independent)",
                "U_real": "max(t95(N-1)*sd/sqrt(N), median band spread)",
                "judgeable": f"U_real <= {GATE}",
                "verdict": "ladder_closure.verdict on the realization medians",
                "phase_bounds": {"phase_x_dx": PHASE_X_MAX_DX,
                                 "gap_jitter_dx": GAP_JITTER_MAX_DX},
                "scale_check": list(SCALE_CHECK),
            },
            "grids": grids,
            "rule_inputs": rows_for_rule,
            "verdict": verdict}


def _scale_check(rows: list[dict]) -> dict:
    """A2: does the reading depend on the perturbation *size*?"""
    out = {}
    for scale in SCALE_CHECK:
        sel = [r for r in rows if r["scale"] == scale]
        if not sel:
            continue
        out[f"{scale:g}"] = {
            "n": len(sel),
            "obs_peak_median": statistics.median([r["obs_peak"] for r in sel]),
            "residual_median": statistics.median([r["residual"] for r in sel]),
            "residual_sd": (statistics.stdev([r["residual"] for r in sel])
                            if len(sel) > 1 else float("nan")),
            "seeds": [r["seed"] for r in sel],
        }
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=1.0e-3)
    ap.add_argument("--velocity", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--c0-factor", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analyse", action="store_true",
                    help="aggregate every realization artefact in --out")
    ap.add_argument("--bias", type=Path,
                    default=ROOT / "docs" / "artifacts"
                    / "case_sph_wedge_entry_2d_bias" / "bias_report.json")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.analyse:
        agg = aggregate(out_dir, args.bias)
        path = out_dir / "seed_ensemble.json"
        path.write_text(json.dumps(agg, indent=1))
        for g in agg["grids"]:
            ind = g["independence"]
            print(f"[seed] {g['tag']}: N={g['n_realisations']}"
                  f" obs_med={g['obs_median']:.3f}"
                  f" r_med={g['residual_median']*100:+.1f}%"
                  f" [{g['residual_min']*100:+.1f}, {g['residual_max']*100:+.1f}]"
                  f" U_ci={g['U_realisation_ci']*100:.1f}%"
                  f" U_band={g['U_band']*100:.1f}%"
                  f" U={g['uncertainty']*100:.1f}%"
                  f" judgeable={g['judgeable']}"
                  f" rho_med={ind['pairwise_median']:.3f}")
        print(f"[seed] verdict: {agg['verdict']['verdict']} --"
              f" {agg['verdict']['reason']}")
        print(f"[seed] -> {path}")
        return 0

    payload = run_one(args.dx, args.seed, args.scale, args.frames,
                      args.c0_factor, args.threads, args.velocity)
    name = artefact_name(args.dx, args.seed, args.scale)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=1))
    rb = payload["by_frames"][str(FRAME_COUNT)]["resolution_band"]
    print(f"[seed] {name}: steps={payload['steps']}"
          f" wall={payload['wall_seconds']:.1f}s"
          f" phase_x_dx={payload['realisation']['phase_x_dx']:+.4f}"
          f" gap_jitter_dx={payload['realisation']['gap_jitter_dx']:+.4f}"
          f" obs={rb['peak_median']:.3f} band_spread={rb['spread_rel']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())