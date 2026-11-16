"""Re-derive the CASE-SPH-WEDGE-ENTRY-2D peak estimator from dumped frame samples.

Why this exists (m23280)
------------------------
``docs/case_sph_wedge_entry_2d_band_coverage_m22695.md`` §5 recorded the reason the
resolution-band defect could not be re-measured: *"re-measuring the band at
40..240 bins requires re-running the ensembles (a solver run, not a re-binning of
committed data -- the per-bin peaks are not stored)"*.  The per-*frame* wall
samples are the sufficient statistic of that estimator (every bin peak is a
function of the pooled ``(tau, cp)`` list only), so storing them once makes every
future bin-count question a CPU-second question instead of a grid-run question.

This tool therefore:

1. reads a committed ensemble artefact (``ensemble_peak.py`` output) **and** its
   ``--dump-frames`` companion,
2. rebuilds the per-frame shots from the dump,
3. re-runs the *same* estimator (``ensemble_peak.analyse``) on those shots, and
4. reports the deviation against the numbers stored in the artefact.

A deviation of zero is the interesting result: it proves the committed reading is
derivable from the committed samples, i.e. the uncertainty model is auditable
after the fact.  ``--bin-list`` additionally evaluates bin counts that were never
run through the solver, which is how a truncated band gets measured.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/rebuild_from_frames.py \
        --ensemble docs/artifacts/case_sph_wedge_entry_2d_ensemble/ensemble_dx1mm_c0v20.json \
        --frames   docs/artifacts/case_sph_wedge_entry_2d_ensemble/frames_dx1mm_c0v20.json \
        --json     docs/artifacts/case_sph_wedge_entry_2d_rebin/dx1mm.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ensemble_peak as ep  # noqa: E402


def shots_from_dump(dump: dict) -> list[dict]:
    """Rebuild ``ensemble_peak.collect``'s shot list from a frame dump.

    ``c_sph`` (the SPH wetted-edge coefficient used to set the band) is not part
    of the dump; the band candidate list is therefore taken from the artefact
    itself (``resolution_band.n_bins_used``), which is exactly the list that was
    used when the artefact was written.
    """
    shots = []
    for fr in dump["frames"]:
        shots.append({
            "t": float(fr["t"]),
            "tau": [float(v) for v in fr["tau"]],
            "cp": [float(v) for v in fr["cp"]],
            "n_wet": int(fr["n_wet"]),
            "max_rho_ratio": float(fr.get("max_rho_ratio", float("nan"))),
            "c_sph": float("nan"),
        })
    return shots


def _rel(a, b) -> float:
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return float("nan")
    if a != a or b != b:
        return 0.0 if (a != a and b != b) else float("nan")
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def rebuild(ensemble: dict, dump: dict, bin_list=None) -> dict:
    """Re-run the estimator on the dumped samples; compare with the artefact.

    The rebuilt statistics are computed here from ``ensemble_peak``'s own
    helpers (``pooled`` / ``binned`` / ``gated_peak`` / ``spread``) -- the same
    functions the solver run used.  ``analyse`` is not called because it needs
    ``c_sph`` (the SPH wetted edge of the last frame) to derive the band, and the
    band candidate list is committed in the artefact anyway; using it keeps the
    comparison apples-to-apples.
    """
    shots = shots_from_dump(dump)
    cfg = ensemble.get("config", {})

    rows = []
    worst = 0.0
    for n_frames, stored in ensemble["by_frames"].items():
        n = min(int(n_frames), len(shots))
        sel = shots[len(shots) - n:] if n < len(shots) else shots
        taus, cps = ep.pooled(sel)
        rb_s = stored["resolution_band"]
        band_bins = list(rb_s["n_bins_used"])

        band_rows = [{"n_bins": nb,
                      "cp_peak_gated": ep.gated_peak(*ep.binned(taus, cps, nb))[0]}
                     for nb in band_bins]
        peaks = [r["cp_peak_gated"] for r in band_rows]
        lo_v, hi_v, med_v, rel_v = ep.spread(peaks)
        if len(band_bins) < 2:
            rel_v = float("nan")          # a one-member band has no spread (m22695)

        # independent cross-check: the full BIN_GRID median (historical H1 field)
        grid_peaks = [ep.gated_peak(*ep.binned(taus, cps, nb))[0]
                      for nb in ep.BIN_GRID]
        grid_med = ep.spread(grid_peaks)[2]

        stored_at = {r["n_bins"]: r["cp_peak_gated"]
                     for r in rb_s.get("band_peaks", [])}
        devs = [_rel(r["cp_peak_gated"], stored_at.get(r["n_bins"], float("nan")))
                for r in band_rows]
        dev = max(devs) if devs else float("nan")
        if dev == dev:
            worst = max(worst, dev)
        rows.append({
            "n_frames": int(n_frames),
            "band_bins": band_bins,
            "band_members": len(band_bins),
            "band_peaks_rebuilt": band_rows,
            "peak_median_stored": rb_s["peak_median"],
            "peak_median_rebuilt": med_v,
            "spread_stored": rb_s["spread_rel"],
            "spread_rebuilt": rel_v,
            "peak_median_rel_dev": _rel(med_v, rb_s["peak_median"]),
            "spread_abs_dev": (abs(rel_v - rb_s["spread_rel"])
                               if rel_v == rel_v else float("nan")),
            "grid_peak_median_stored": stored["peak_median"],
            "grid_peak_median_rebuilt": grid_med,
            "grid_peak_median_rel_dev": _rel(grid_med, stored["peak_median"]),
            "max_band_peak_rel_dev": dev,
        })

    out = {
        "script": "benchmarks/case_sph_wedge_entry_2d/rebuild_from_frames.py",
        "ensemble_artefact": ensemble.get("script"),
        "tag": dump.get("tag"),
        "dx": cfg.get("dx"),
        "frames_in_dump": len(shots),
        "worst_band_peak_rel_dev": worst,
        "by_frames": rows,
    }
    if bin_list:
        taus, cps = ep.pooled(shots)
        out["extra_bin_list"] = [
            {"n_bins": int(nb),
             "cp_peak_gated": ep.gated_peak(*ep.binned(taus, cps, int(nb)))[0]}
            for nb in bin_list]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ensemble", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--bin-list", default=None,
                    help="comma separated extra bin counts to evaluate from the dump")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    ensemble = json.loads(Path(args.ensemble).read_text())
    dump = json.loads(Path(args.frames).read_text())
    bins = ([int(v) for v in args.bin_list.split(",") if v.strip()]
            if args.bin_list else None)
    res = rebuild(ensemble, dump, bins)
    text = json.dumps(res, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text)
        print(f"[rebin] -> {args.json}")
    print(f"[rebin] {res['tag']}: {res['frames_in_dump']} frames, "
          f"worst band-peak deviation vs artefact = "
          f"{res['worst_band_peak_rel_dev']:.3e}")
    print(f"  {'frames':>6} {'members':>7} {'peak(stored)':>13} "
          f"{'peak(rebuilt)':>14} {'spread(stored)':>14} {'spread(rebuilt)':>15}")
    for r in res["by_frames"]:
        print(f"  {r['n_frames']:>6} {r['band_members']:>7} "
              f"{r['peak_median_stored']:>13.4f} {r['peak_median_rebuilt']:>14.4f} "
              f"{r['spread_stored']:>14.4f} {r['spread_rebuilt']:>15.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())