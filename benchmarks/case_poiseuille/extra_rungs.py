"""CASE-POISEUILLE-SPH / C6 attribution: finer rungs + a long-time control.

Two questions the committed ladder (``H/dx = 8, 12, 16``) cannot answer:

1. **Does the level bias keep converging?**  Three rungs give a two-interval
   order estimate; a fourth and fifth rung turn "order ~2.4" into a trend that
   can be extrapolated and checked.  The rungs are added with the *same* driver
   and the *same* estimator as ``combined.py`` (imported, not re-implemented),
   so the new points live on the published ladder rather than beside it.
2. **Is the level bias a transient that the 0.8-1.0 s window is too early for,
   or a steady bias of the discrete system?**  These are very different verdicts
   (a too-early window is a measurement bug; a steady bias is a property of the
   scheme) and the committed artifact cannot distinguish them, because it stops
   at ``t = 1.0``.  The control re-runs the coarsest rung to ``t = 4.0`` -- four
   times the integration time -- and re-measures the same functional on the
   *last* window.  If the bias is unsteady it must decay; if it is steady it must
   not move.  The momentum budget gate (C5) closes to <0.06 % at every rung, so
   a stationary drift is the prior, but this tests it directly.

Run::

    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/extra_rungs.py
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/extra_rungs.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import combined as C  # noqa: E402
from benchmarks.case_poiseuille import backflow_level as BL  # noqa: E402

OUT = HERE / "out"
TARGET = OUT / "backflow_level_target.json"

DEFAULT_RUNGS = (24, 32, 48)
LONGTIME_RESOLUTION = 8
LONGTIME_ENDS = (2.0, 4.0)


def _window(t_end: float, n: int = 11) -> list[float]:
    """Steady window of length 0.2 s ending at ``t_end`` (same shape as STEADY_TIMES)."""
    t0 = t_end - 0.20
    return [round(t0 + 0.02 * i, 10) for i in range(n)]


def run_ladder(resolutions, device) -> list[dict]:
    runs = []
    for r in resolutions:
        t0 = time.time()
        case = C.run_channel(int(r), device)
        graded = C.grade_run(case, f"extra_rung_{r}")
        graded["t_end"] = C.T_END
        graded["wall_clock_s"] = time.time() - t0
        runs.append(graded)
        print(f"[extra] H/dx={r:3d}  C6={graded['backflow']['peak_pct']:+7.3f} %  "
              f"({graded['wall_clock_s']:.1f} s, n={graded['n_fluid']})", flush=True)
    return runs


def run_longtime(device, resolution: int = LONGTIME_RESOLUTION,
                 ends=LONGTIME_ENDS) -> list[dict]:
    """Same coarsest rung, much longer: does the level bias decay or stay?"""
    recs = []
    for t_end in ends:
        times = _window(t_end)
        t0 = time.time()
        case = C.run_channel(int(resolution), device, t_end=t_end, samples=times)
        graded = C.grade_run(case, f"longtime_{resolution}_t{t_end:g}")
        graded["t_end"] = float(t_end)
        graded["window"] = [times[0], times[-1]]
        graded["wall_clock_s"] = time.time() - t0
        recs.append(graded)
        print(f"[extra] longtime H/dx={resolution} t_end={t_end:g}  "
              f"C6={graded['backflow']['peak_pct']:+7.3f} %  ({time.time() - t0:.1f} s)",
              flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rungs", type=int, nargs="+", default=list(DEFAULT_RUNGS))
    ap.add_argument("--quick", action="store_true",
                    help="one added rung (24) and no long-time control")
    ap.add_argument("--out", type=Path, default=TARGET)
    args = ap.parse_args(argv)
    rungs = [24] if args.quick else list(args.rungs)
    device = torch.device("cpu")
    t0 = time.time()

    runs = run_ladder(rungs, device)
    longtime = [] if args.quick else run_longtime(device)

    # attribution of every new point, with the published estimator
    par = {"height": C.HEIGHT, "viscosity": C.VISCOSITY, "g": C.G_ADVERSE,
           "wall_speed": C.WALL_SPEED}
    extra = []
    for rec in runs:
        d = BL.decompose(rec["rows"], par["height"], par["viscosity"], par["g"],
                         par["wall_speed"], rec["dx"])
        d["stored_peak_pct"] = rec["backflow"]["peak_pct"]
        d["t_end"] = rec["t_end"]
        extra.append(d)
    long_recs = []
    for rec in longtime:
        d = BL.decompose(rec["rows"], par["height"], par["viscosity"], par["g"],
                         par["wall_speed"], rec["dx"])
        d["stored_peak_pct"] = rec["backflow"]["peak_pct"]
        d["t_end"] = rec["t_end"]
        d["window"] = rec["window"]
        long_recs.append(d)

    payload = {
        "case": C.CASE_KEY,
        "anchor": "CASE-POISEUILLE-SPH",
        "study": BL.STUDY,
        "purpose": ("finer rungs on the published ladder; long-time control testing "
                    "whether the level bias is a transient or a steady discrete bias"),
        "parameters": {**par, "sound_speed": C.SOUND_SPEED,
                       "columns": C.COLUMNS, "t_end": C.T_END},
        "runs": runs,
        "attribution": extra,
        "longtime": long_recs,
        "longtime_runs_meta": [{"t_end": r["t_end"], "window": r["window"],
                               "steps": r["steps"], "dt": r["dt"],
                               "n_fluid": r["n_fluid"],
                               "wall_clock_s": r["wall_clock_s"]} for r in longtime],
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "wall_clock_s": time.time() - t0,
        "generated_by": "benchmarks/case_poiseuille/extra_rungs.py",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"[extra] wrote {args.out}  ({payload['wall_clock_s']:.1f} s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())