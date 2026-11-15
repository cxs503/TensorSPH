"""Why does the pooled-peak estimator split 43 % between the two halves of one window?

The question, and why it decides the case
----------------------------------------
``ensemble_peak.py`` reports a *frame half-split* (``jackknife_rel``) next to every
grid, and ``ladder_closure.py`` uses ``max(resolution-band spread, half-split)`` as
the uncertainty of the reading.  At ``dx = 2 mm`` that half-split is **43.0 %**
(``early 8.208 / late 5.742``), which is what keeps the only other judgeable grid
out of the closure rule -- while the *per-frame* median of the same 61 frames
splits by only **2.7 %** (``frame_stationarity.py``).  A 16x disagreement between
two routes to the same question cannot be left standing: one of them is not
measuring what it claims.

The two routes are not the same estimator, and that is the point:

* ``jackknife_halves`` pools each half's ``(tau, Cp)`` cloud and reads the peak of
  the pooled, sample-gated profile **separately for each half**, taking the median
  over the bin counts ``(10, 12, 16)``;
* ``frame_stationarity.half_split`` takes the median of each half's *per-frame*
  point peak.

If the split is a property of the flow (frames not members of one ensemble) both
routes must see it.  If it is a property of the estimator -- different bin counts
being admissible in the two halves, so the "same" reading is taken through
different operators -- only the pooled route sees it.

What this script measures
-------------------------
For one grid, from one fresh trajectory that must reproduce the committed artefact
bit-for-bit in the graded numbers (that check is asserted, not assumed):

1. **Determinism / provenance** -- recomputed ``peak_median`` and half values vs
   the committed artefact.
2. **Admissible-bin inventory per half** -- which bin counts clear
   ``MIN_BIN_SAMPLES`` in each half, and the peak each bin count yields.  If the
   halves admit different bin sets, the pooled half-split compares unlike
   operators.
3. **Common-bin split** -- the half-split recomputed at *one* bin count admitted
   by both halves (the finest such count, i.e. the closest to the resolution band),
   and at the band median over the *common* bin set.
4. **Frame bootstrap** -- resample the frames with replacement (``--boot`` draws,
   fixed seed) and take the resolution-band reading each time: this is a direct
   sampling uncertainty for the pooled estimator and does not depend on splitting
   the window in two.
5. **Wet-node stratification** -- the same reading on the sub-window where every
   frame has ``n_wet >= gate``.

Nothing here changes the solver, the case, the reference or the 15 % gate.  The
script prints a JSON block and writes it when ``--out`` is given, so a report can
quote it without transcription.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/half_split_decomposition.py --dx 0.002
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import ensemble_peak as ep  # noqa: E402


def rel(a: float, b: float) -> float:
    if not (a == a and b == b):
        return float("nan")
    return abs(b - a) / max(abs(b), 1e-12)


def half_reading(shots: list[dict], n_bins: int) -> dict:
    """Pooled, sample-gated peak of one shot list at a fixed bin count."""
    taus, cps = ep.pooled(shots)
    centres, means, cnts = ep.binned(taus, cps, n_bins)
    peak, tau_at, n_at = ep.gated_peak(centres, means, cnts)
    return {"n_bins": n_bins, "peak": peak, "tau_at_peak": tau_at,
            "samples_at_peak": n_at,
            "n_admissible_bins": sum(1 for m, n, c in zip(means, cnts, centres)
                                     if m == m and n >= ep.MIN_BIN_SAMPLES and c <= 1.0)}


def band_reading(shots: list[dict], n_bins_list) -> dict:
    vals = [half_reading(shots, nb)["peak"] for nb in n_bins_list]
    lo, hi, med, spread_rel = ep.spread(vals)
    return {"n_bins_used": list(n_bins_list), "peaks": vals,
            "peak_min": lo, "peak_max": hi, "peak_median": med,
            "spread_rel": spread_rel}


def decompose(shots: list[dict], dx: float) -> dict:
    half = len(shots) // 2
    early, late = shots[:half], shots[half:]
    c_last = shots[-1]["c_sph"]
    _lo, _hi, band = ep.resolution_bin_band(c_last, dx)

    out: dict = {"dx": dx, "n_frames": len(shots),
                 "resolution_band_bins": list(band),
                 "resolution_band_peak_median": band_reading(shots, band)["peak_median"],
                 "resolution_band_spread_rel": band_reading(shots, band)["spread_rel"]}

    # 2. admissible-bin inventory per half
    inv = {}
    for tag, sub in (("early", early), ("late", late)):
        inv[tag] = [half_reading(sub, nb) for nb in ep.BIN_GRID]
    out["admissible_bins"] = {
        tag: {"bins": [r["n_bins"] for r in rows
                       if r["samples_at_peak"] >= ep.MIN_BIN_SAMPLES],
              "readings": {str(r["n_bins"]): r["peak"] for r in rows}}
        for tag, rows in inv.items()}
    common = sorted(set(out["admissible_bins"]["early"]["bins"]) &
                    set(out["admissible_bins"]["late"]["bins"]))
    out["common_bins"] = common

    # legacy route, reproduced (median over the fixed triple 10/12/16)
    legacy = {tag: ep.spread([r["peak"] for r in inv[tag] if r["n_bins"] in (10, 12, 16)])[2]
              for tag in ("early", "late")}
    out["legacy_half_readings"] = legacy
    out["legacy_half_split"] = rel(legacy["early"], legacy["late"])

    # 3. common-bin splits
    splits = {}
    for nb in common:
        e = [r["peak"] for r in inv["early"] if r["n_bins"] == nb][0]
        l = [r["peak"] for r in inv["late"] if r["n_bins"] == nb][0]
        splits[str(nb)] = {"early": e, "late": l, "rel_split": rel(e, l)}
    out["common_bin_splits"] = splits
    if common:
        finest = str(max(common))
        out["common_bin_finest_split"] = splits[finest]["rel_split"]
        band_common = [nb for nb in band if nb in common] or common
        be = band_reading(early, band_common)["peak_median"]
        bl = band_reading(late, band_common)["peak_median"]
        out["common_band"] = {"n_bins_used": list(band_common),
                              "early": be, "late": bl, "rel_split": rel(be, bl)}
    else:
        out["common_bin_finest_split"] = float("nan")
        out["common_band"] = {"n_bins_used": [], "early": float("nan"),
                              "late": float("nan"), "rel_split": float("nan")}

    # 4. bootstrap over frames
    out["bootstrap"] = bootstrap(shots, band)
    # 5. wet-node stratification
    out["wet_gate"] = {}
    for g in (16, 30):
        sub = [s for s in shots if s["n_wet"] >= g]
        if len(sub) >= 8:
            br = band_reading(sub, band)
            out["wet_gate"][str(g)] = {"n_frames": len(sub),
                                       "peak_median": br["peak_median"],
                                       "spread_rel": br["spread_rel"]}
    return out


def bootstrap(shots: list[dict], band, draws: int = 200, seed: int = 20261115) -> dict:
    """Sampling uncertainty of the pooled reading: resample frames, fixed operator."""
    rng = random.Random(seed)
    vals = []
    n = len(shots)
    for _ in range(draws):
        sub = [shots[rng.randrange(n)] for _ in range(n)]
        vals.append(band_reading(sub, band)["peak_median"])
    lo, hi, med, _rel = ep.spread(vals)
    sd = statistics.pstdev(vals)
    return {"draws": draws, "seed": seed, "peak_median": med,
            "p05": sorted(vals)[max(0, int(0.05 * draws))],
            "p95": sorted(vals)[min(draws - 1, int(0.95 * draws))],
            "sd": sd, "rel_sd": sd / med if med else float("nan"),
            "rel_range": (hi - lo) / med if med else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--c0-factor", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--brief", action="store_true",
                    help="one-line JSON summary (cluster logs are tail-truncated)")
    args = ap.parse_args(argv)

    case = ep.collect(args.dx, 1.0, args.frames, args.c0_factor, args.threads)
    res = decompose(case["shots"], args.dx)
    res["bootstrap"] = bootstrap(case["shots"], res["resolution_band_bins"],
                                 draws=args.boot)
    res["provenance"] = {"wall_seconds": case["wall_seconds"], "steps": case["steps"],
                         "t_end": case["t_end"], "c0": case["c0"],
                         "config": case["cfg"], "threads": args.threads}
    if args.brief:
        print("DECOMP " + json.dumps({
            "dx": res["dx"],
            "resolution_band_bins": res["resolution_band_bins"],
            "resolution_band_peak_median": res["resolution_band_peak_median"],
            "resolution_band_spread_rel": res["resolution_band_spread_rel"],
            "legacy_half_readings": res["legacy_half_readings"],
            "legacy_half_split": res["legacy_half_split"],
            "common_bins": res["common_bins"],
            "common_bin_splits": {k: v["rel_split"]
                                  for k, v in res["common_bin_splits"].items()},
            "common_band": res["common_band"],
            "bootstrap": res["bootstrap"],
            "wet_gate": res["wet_gate"],
            "wall_seconds": res["provenance"]["wall_seconds"],
        }, sort_keys=True))
    else:
        print(json.dumps(res, indent=1))
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1, sort_keys=True))
        print("wrote", p, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())