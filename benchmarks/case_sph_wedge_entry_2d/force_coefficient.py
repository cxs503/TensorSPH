"""CASE-SPH-WEDGE-ENTRY-2D -- the slamming force coefficient as a robust metric.

Why this round exists
---------------------
``docs/case_sph_wedge_entry_2d_phase_ladder_consistency_m23321.md`` closed the
refinement and realisation routes for the *peak* metric ``Cp_max`` and left a
falsifiable metric note (§7-4): ``Cp_max`` is an *extremal* reading of a noisy
field, its intrinsic scatter is of the same order as the 15 % gate, and the
anchor should be scored on an *integral* quantity that is insensitive to the
spray-root singularity -- the slamming force coefficient ``C_s``.

This module is that note, made executable.  It adds nothing to the physics:

* the reference is Wagner's added-mass force coefficient
  ``C_s = pi^2 / tan(beta)`` (time independent; re-exposed as
  :func:`tensorsph.wedge_entry.wagner_force_coefficient` and pinned to
  ``tensorsph.slamming.slamming_force_coefficient`` at machine precision);
* the SPH reading is ``C_s = 2 ∫_0^1 cp(tau) dtau`` (the
  ``cs_profile`` diagnostic of :meth:`WedgeImpact.diagnostics`), an *integral*
  reading in which the square-root cusp of the Wagner profile is averaged over
  the last bin instead of being sampled.

Predeclared decision rule (fixed before the runs)
-------------------------------------------------
For each grid ``g`` the self-similar window is the same 61 frames as every
committed artefact.  The force coefficient is *time independent* in Wagner's
theory, so the reading is the median of ``cs_profile`` over the frames where it
is defined (the first frames are before first contact and are ``nan``).  The
residual is ``r_g = Cs_sph,g / Cs_wagner - 1`` and the case is *decidable on
the force metric* iff ``|r_g| <= 0.15`` on at least two grids of the ladder.

Nothing here changes the solver, the case definition or the acceptance gate:
the peak metric ``Cp_max`` and its 15 % gate are untouched; this is a
*supplementary* reading of the same committed runs.

Usage
-----
::

    # one grid (local, < 10 min for dx >= 2 mm)
    PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/force_coefficient.py \
        --dx 0.002 --threads 4 --out artifacts/case_sph_wedge_entry_2d_force

    # aggregate every grid found in the directory
    PYTHONPATH=src python benchmarks/case_sph_wedge_entry_2d/force_coefficient.py \
        --analyse --out artifacts/case_sph_wedge_entry_2d_force
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorsph.wedge_entry import (  # noqa: E402
    WedgeConfig,
    WedgeImpact,
    force_coefficient_from_profile,
    wagner_c_coeff,
    wagner_force_coefficient,
)

GATE = 0.15


def run_grid(dx: float, threads: int, frames: int = 61, beta_deg: float = 30.0,
             c0_factor: float = 20.0, out_dir: Path | None = None) -> dict:
    """Run the case once (unperturbed lattice) and read the force coefficient."""
    import torch
    torch.set_num_threads(max(1, int(threads)))
    cfg = WedgeConfig(beta_deg=beta_deg, velocity=1.0, dx=dx, frames=frames,
                      alpha_visc=0.03, sound_speed_factor=c0_factor,
                      max_courant=0.35)
    cfg.validate()
    t0 = time.time()
    sim = WedgeImpact(cfg)
    t_end = cfg.resolved_t_end()
    times = [t_end * k / (frames - 1) for k in range(frames)]
    cs_series = []
    cp_series = []
    target = 0
    steps = 0
    while sim.time < t_end - 1.0e-15:
        dt = min(sim.dt(), t_end - sim.time)
        if target < len(times):
            dt = min(dt, max(times[target] - sim.time, 1.0e-18))
        sim.advance(dt)
        steps += 1
        if target < len(times) and sim.time >= times[target] - 1.0e-15:
            diag = sim.diagnostics()
            cs_series.append(diag.get("cs_profile", float("nan")))
            cp_series.append(diag["p_max_footprint"] / sim.q)
            target += 1
    wall = time.time() - t0
    cs = [v for v in cs_series if v == v and math.isfinite(v)]
    cp = [v for v in cp_series if v == v and math.isfinite(v)]
    cs_wagner = wagner_force_coefficient(cfg.beta)
    cs_sph = st.median(cs) if cs else float("nan")
    out = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/force_coefficient.py",
        "cfg": {"beta_deg": 30.0, "velocity": 1.0, "dx": dx, "frames": frames,
                "c0_factor": 20.0, "courant": 0.35, "alpha_visc": 0.03},
        "tag": _tag(dx),
        "steps": steps,
        "wall_seconds": wall,
        "cs_wagner": cs_wagner,
        "cs_sph_median": cs_sph,
        "cs_sph_series": cs_series,
        "cp_peak_series": [v for v in cp_series if v == v],
        "residual_same_operator": (cs_sph / cs_wagner - 1.0) if cs_sph == cs_sph
        else float("nan"),
        "gate": GATE,
        "judgeable": bool(cs_sph == cs_sph and abs(cs_sph / cs_wagner - 1.0) <= GATE),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"force_{_tag(dx)}_c0v20.json").write_text(
            json.dumps(out, indent=1))
    return out


def _tag(dx: float) -> str:
    key = f"{dx:.4f}".rstrip("0").rstrip(".")
    return {"0.004": "dx4mm", "0.002": "dx2mm", "0.001": "dx1mm",
            "0.0005": "dx0.5mm"}.get(key, f"dx{dx}")


def analyse(out_dir: Path) -> dict:
    rows = []
    for path in sorted(out_dir.glob("force_dx*_c0v20.json")):
        d = json.loads(path.read_text())
        rows.append({
            "tag": d["tag"],
            "dx_m": d["cfg"]["dx"],
            "cs_wagner": d["cs_wagner"],
            "cs_sph": d["cs_sph_median"],
            "residual": d["residual_same_operator"],
            "judgeable": d["judgeable"],
            "wall_seconds": d["wall_seconds"],
        })
    rows.sort(key=lambda r: -r["dx_m"])
    judgeable = [r for r in rows if r["judgeable"]]
    verdict = ("DECIDABLE_ON_FORCE_METRIC" if len(judgeable) >= 2
               else "NOT_DECIDABLE_ON_FORCE_METRIC_AT_MEASURED_GRIDS")
    return {"case": "CASE-SPH-WEDGE-ENTRY-2D",
            "gate": GATE,
            "rows": rows,
            "n_judgeable": len(judgeable),
            "verdict": verdict}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args(argv)
    if args.analyse:
        out = args.out or (REPO / "artifacts/case_sph_wedge_entry_2d_force")
        res = analyse(out)
        print(json.dumps(res, indent=1))
        return 0
    if args.dx is None:
        ap.error("--dx is required unless --analyse")
    out = args.out or (REPO / "artifacts/case_sph_wedge_entry_2d_force")
    res = run_grid(args.dx, args.threads, args.frames, out_dir=out)
    print(json.dumps({k: res[k] for k in
                      ("tag", "cs_wagner", "cs_sph_median",
                       "residual_same_operator", "judgeable", "wall_seconds")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
