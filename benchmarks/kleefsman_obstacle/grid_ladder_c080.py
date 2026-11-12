"""Third grid level of the Kleefsman (2005) obstacle case at c0 = 80 m/s.

Why this exists
---------------
``docs/verification_kleefsman_c0_sensitivity.md`` (mission m16758) settled the
*compressibility* half of the debt: raising the artificial sound speed pushes
the impact-region density compression from 12-14 % down to ~3.4 %, i.e. below
the 5 % controllability line.  It left the *resolution* half open:

* at c0 = 20 m/s the probe-height peak was **not** a monotone function of dx
  (0.425 m @ dx=0.04, 0.518 m @ dx=0.02), so no convergence order could be
  quoted;
* at c0 = 80 m/s the peak is 0.513 m @ dx=0.04 and 0.455 m @ dx=0.02 -- still
  non-monotone in the old direction, and the cross-grid gap only shrank from
  21.6 % to 11.2 %.  Two levels cannot separate "converging to 0.455 m" from
  "oscillating".

This driver adds the missing **third level (dx = 0.01 m) at the fixed sound
speed c0 = 80 m/s** and decides, with criteria fixed in code *before* the run:

* ``monotone_converging`` -- the two successive differences have the same sign
  and the fine one is smaller; an observed order ``p`` and a GCI follow;
* ``oscillatory`` -- signs differ; the peak is not resolved at this c0 and the
  honest verdict is "no convergence", not a number;
* ``monotone_diverging`` -- same sign but growing: the case is grid-diverging.

The first two levels are **read from the committed c0-scan artefacts** (no
recomputation, no re-typing); only dx = 0.01 m is simulated.  Nothing in
``src/tensorsph`` is touched: the runs go through
``c0_sensitivity.build_config`` / ``run_one`` exactly as in the scan.

Usage
-----
    python benchmarks/kleefsman_obstacle/grid_ladder_c080.py            # run + analyse
    python benchmarks/kleefsman_obstacle/grid_ladder_c080.py --analyse  # analysis only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c0_sensitivity as scan  # noqa: E402

#: Ladder, coarse -> fine.  Only the last entry is simulated by this driver.
DXS = (0.04, 0.02, 0.01)
SOUND_SPEED = 80.0
T_END = 1.2
#: Pre-declared analysis constants (Roache 1994, fixed before the run).
SAFETY_FACTOR = 1.25
#: Pre-declared controllability line on density compression (unchanged from m16758).
COMPRESSION_MAX = 0.05


def ladder_metrics(dxs: tuple[float, ...], values: tuple[float, ...],
                   safety: float = SAFETY_FACTOR) -> dict:
    """Observed order / GCI / verdict of a three-level refinement ladder.

    Pure arithmetic on the numbers handed in, so the decision logic is testable
    without running the solver.  ``dxs`` and ``values`` run coarse -> fine.
    """
    if len(dxs) != 3 or len(values) != 3:
        raise ValueError("a three-level ladder is required")
    f1, f2, f3 = values
    d12, d23 = f1 - f2, f2 - f3
    r = dxs[0] / dxs[1]
    same_sign = d12 * d23 > 0.0
    out: dict = {
        "dx": list(dxs), "peak_height_m": list(values),
        "coarse_fine_rel_change": [abs(d12) / abs(f2), abs(d23) / abs(f3)],
        "refinement_ratio": r,
        "differences": [d12, d23],
        "signs": [math.copysign(1.0, d12), math.copysign(1.0, d23)],
    }
    if not same_sign:
        out.update(verdict="oscillatory", observed_order=None,
                   gci_fine=None,
                   note="successive refinement flips the sign of the change: "
                        "no convergent sequence, no order may be quoted")
        return out
    if abs(d23) >= abs(d12):
        out.update(verdict="monotone_diverging", observed_order=None,
                   gci_fine=None,
                   note="change grows under refinement")
        return out
    if d23 == 0.0:
        out.update(verdict="monotone_converging", observed_order=None,
                   gci_fine=0.0, note="bit-identical on the two finest grids")
        return out
    p = math.log(abs(d12 / d23)) / math.log(r)
    gci = safety * abs(d23 / f3) / (r ** p - 1.0)
    out.update(verdict="monotone_converging", observed_order=p, gci_fine=gci,
               note="monotone contraction; order and GCI are formal consequences")
    return out


def collect(dxs=DXS, c0=SOUND_SPEED, outdir: Path | None = None) -> list[dict]:
    outdir = outdir or (HERE / "results" / "c0_scan")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dx in dxs:
        p = outdir / f"{scan.case_tag(dx, c0)}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
            print(f"[reuse] {p.name}", flush=True)
    return rows


def analyse(rows: list[dict], t_end: float = T_END) -> dict:
    by_dx = {r["dx"]: r for r in rows}
    have = [dx for dx in DXS if dx in by_dx]
    result: dict = {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "sound_speed": SOUND_SPEED, "t_end": t_end,
        "ladder_dx": list(DXS), "levels_available": have,
        "compression_limit": COMPRESSION_MAX,
        "safety_factor": SAFETY_FACTOR,
    }
    if len(have) == 3:
        peaks = tuple(by_dx[dx]["peak_height_m"] for dx in DXS)
        times = tuple(by_dx[dx]["peak_time_s"] for dx in DXS)
        result["peak_ladder"] = ladder_metrics(DXS, peaks)
        result["peak_time_ladder"] = ladder_metrics(DXS, times)
    fine = by_dx.get(DXS[-1])
    if fine is None:
        result["status"] = "incomplete"
        return result
    comp = fine["max_density_compression"]
    result["compression"] = {
        "fine_dx": DXS[-1], "max_density_compression": comp,
        "within_limit": comp <= COMPRESSION_MAX,
        "per_level": {f"{dx:g}": by_dx[dx]["max_density_compression"] for dx in have},
    }
    result["wall_clock_s"] = {f"{dx:g}": by_dx[dx].get("wall_clock_s") for dx in have}
    result["steps"] = {f"{dx:g}": by_dx[dx].get("steps") for dx in have}
    result["status"] = "complete" if len(have) == 3 else "incomplete"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true", help="skip the run, analyse what exists")
    ap.add_argument("--t-end", type=float, default=T_END)
    args = ap.parse_args()

    outdir = HERE / "results" / "c0_scan"
    if not args.analyse:
        scan.run_one(DXS[-1], SOUND_SPEED, args.t_end, outdir,
                     chunk_budget=scan.CHUNK_BUDGET)
    rows = collect(outdir=outdir)
    res = analyse(rows, t_end=args.t_end)
    (outdir / "grid_ladder_c080.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("status", "levels_available")}, indent=2))
    if "peak_ladder" in res:
        print(json.dumps(res["peak_ladder"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())