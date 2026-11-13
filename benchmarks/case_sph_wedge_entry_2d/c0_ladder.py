"""Sound-speed (``c0/V``) ladder for CASE-SPH-WEDGE-ENTRY-2D.

Anchor: ``CASE-SPH-WEDGE-ENTRY-2D`` (open).  The committed grid ladder of this
case fails the mission tolerance at every affordable spacing
(``cp_band_peak`` +234 % at ``dx = 2 mm``), and the case document already shows
why the failure is suspicious: the *compressibility error budget*

    Cp_budget = 2 (c0/V)^2 / gamma * [ (max rho/rho0)^gamma - 1 ]

which is what the measured maximum density excursion alone produces, was 42-43
against a Wagner reference of 8.40.  A quantity whose noise floor is five times
the reference cannot be graded against that reference.  This script tests
whether the floor is what the failures are made of.

Predeclared hypothesis (frozen before the first run)
---------------------------------------------------
**H1.** At fixed ``dx``, the measured Cp error of this case is bounded below by
the density excursion: raising ``c0/V`` lowers ``max|rho/rho0 - 1|`` roughly as
``(c0/V)^-2`` (the acoustic scaling), and the measured Cp errors of *all* the
graded quantities fall with it, until the residual discretisation error (the
smoothed spray-root cusp) takes over.

**Credit rule (written before the runs).**  H1 is credited only if, at the
finest ``dx`` tested, going from the smallest to the largest ``c0/V``

1. lowers ``max|rho/rho0 - 1|`` by at least 4x (the acoustic prediction for a
   4x rise in ``c0``), **and**
2. lowers the ``cp_band_peak`` relative error by at least 5 percentage points,
   **and**
3. leaves the mass gate intact (the run stays weakly compressible and finite).

Otherwise the verdict is ``NOT_CREDITED`` and the negative result is recorded
as such - the tolerance is not moved and the metric is not redefined.

**What this cannot do.**  The mission-designated reference (Zhao & Faltinsen
1993) is closed access and the repository carries no registered copy of its
tables, so the Cp numbers here are graded against the in-repo Wagner (1932)
closed form only; ``refs.json`` records that substitution and the case above
``beta = 30 deg`` limits where Wagner is the right idealisation.  No Z&F value
appears anywhere in this file or in its outputs.

Usage
-----
    python benchmarks/case_sph_wedge_entry_2d/c0_ladder.py --stage run --shard 0/3
    python benchmarks/case_sph_wedge_entry_2d/c0_ladder.py --stage accept
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

# Load the case driver by path: a bare ``import run`` resolves to whichever
# ``run.py`` an earlier import cached in ``sys.modules`` (m17109).
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("case_sph_wedge_entry_2d_run", HERE / "run.py")
case = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(case)

ART = case.ROOT / "artifacts" / "case_sph_wedge_entry_2d" / "c0_ladder"

# --------------------------------------------------------------------------- #
# Predeclared design (frozen before the first run)
# --------------------------------------------------------------------------- #
# Fixed spacing, three sound speeds: 20 V is the case's shipped default, 40 V
# and 80 V are the two doublings that the acoustic scaling needs to be visible
# over a factor 4.  dx = 2 mm is the spacing at which the committed ladder's
# documented numbers exist (cp_band_peak 28.05), so the c0 = 20 V member is a
# reproducibility point for that table as well as the control of this ladder.
DX_LADDER = (2.0e-3, 1.0e-3)
C0_FACTORS = (20.0, 40.0, 80.0)

MIN_DENSITY_DROP = 4.0        # credit rule 1: >= 4x fall in max|rho/rho0 - 1|
MIN_CP_ERROR_IMPROVEMENT = 0.05   # credit rule 2: >= 5 percentage points
MASS_GATE = 0.05              # credit rule 3: max|rho/rho0 - 1| <= 5 %


def plan() -> list[dict]:
    """The predeclared case list: ``dx`` x ``c0/V``, ids carry both."""
    specs = []
    for dx in DX_LADDER:
        for c0 in C0_FACTORS:
            specs.append({
                "id": f"dx{dx * 1e3:g}mm_c0v{c0:g}",
                "beta_deg": case.BETA_DEG,
                "velocity": 1.0,
                "dx": dx,
                "sound_speed_factor": c0,
                # ``kind`` is deliberately not "ladder": the sweep varies the
                # sound speed, not the resolution, so the case's GCI branch
                # (which divides by the spacing ratio) must not run on it.
                "kind": "c0_sweep",
            })
    return specs


def run_stage(out_dir: Path, shard: str | None = None) -> list[dict]:
    specs = plan()
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        specs = [s for k, s in enumerate(specs) if k % n == i]
    done = []
    for spec in specs:
        print(f"[c0] {spec['id']} dx={spec['dx']} c0/V={spec['sound_speed_factor']}",
              flush=True)
        art = case.run_spec(spec, out_dir)
        lf = case.last_frame(art)
        print(f"[c0] {spec['id']} steps={art['steps']} "
              f"wall={art['wall_seconds']:.1f}s "
              f"max_rho={max(r['max_rho_ratio'] for r in art['series']):.5f} "
              f"cp_centre={lf['cp_centre']:.4f}", flush=True)
        done.append({"case_id": art["case_id"],
                     "steps": art["steps"],
                     "wall_seconds": round(art["wall_seconds"], 1)})
    return done


def _load(out_dir: Path) -> dict:
    arts = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in ("summary.json", "refs_derived.json"):
            continue
        a = json.loads(path.read_text())
        if "series" in a and a.get("kind") == "c0_sweep":
            arts[a["case_id"]] = a
    return arts


def _loglog_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log(y) vs log(x); nan unless all values are > 0."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
           if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return sxy / sxx if sxx else float("nan")


def accept(out_dir: Path) -> dict:
    arts = _load(out_dir)
    ref = case.refs_derived()
    graded = case.grade(arts, ref) if arts else {"rows": []}
    rows = graded.get("rows", [])

    # Per-spacing grouping, coarse spacing first.
    by_dx: dict[float, list[dict]] = {}
    for r in rows:
        by_dx.setdefault(round(r["dx"], 12), []).append(r)
    for group in by_dx.values():
        group.sort(key=lambda r: r["c0_over_V"])

    ladders = {}
    for dx, group in sorted(by_dx.items(), reverse=True):
        eps = [r["max_rho_ratio"] - 1.0 for r in group]
        c0 = [r["c0_over_V"] for r in group]
        ladders[f"{dx * 1e3:g}mm"] = {
            "dx": dx,
            "c0_over_V": c0,
            "max_rho_ratio": [r["max_rho_ratio"] for r in group],
            "density_excursion": eps,
            "density_scaling_exponent": _loglog_slope(c0, eps),
            "cp_band_peak": [r["cp_band_peak"] for r in group],
            "err_band_peak": [r["err_band_peak"] for r in group],
            "err_at_peak_tau": [r["err_at_peak_tau"] for r in group],
            "err_centre_mean": [r["err_centre_mean"] for r in group],
            "cp_density_budget": [r["cp_density_budget"] for r in group],
            "verdict_band_peak": [r["verdict_band_peak"] for r in group],
            "verdict_centre": [r["verdict_centre"] for r in group],
            "verdict_mass": [r["verdict_mass"] for r in group],
        }

    # ---- credit rule, evaluated exactly as predeclared -------------------- #
    finest_key = None
    if ladders:
        finest_key = min(ladders, key=lambda k: ladders[k]["dx"])
    credit = {"rule": {
        "min_density_drop": MIN_DENSITY_DROP,
        "min_cp_error_improvement": MIN_CP_ERROR_IMPROVEMENT,
        "mass_gate": MASS_GATE,
    }, "evaluated_at": finest_key}
    if finest_key:
        lad = ladders[finest_key]
        eps = lad["density_excursion"]
        e_peak = lad["err_band_peak"]
        cond1 = (len(eps) >= 2 and eps[0] > 0 and eps[-1] > 0
                 and (eps[0] / eps[-1]) >= MIN_DENSITY_DROP)
        cond2 = (len(e_peak) >= 2 and abs(e_peak[0]) - abs(e_peak[-1])
                 >= MIN_CP_ERROR_IMPROVEMENT)
        cond3 = all(abs(v - 1.0) <= MASS_GATE for v in lad["max_rho_ratio"])
        credit.update({
            "density_drop_factor": (eps[0] / eps[-1]
                                    if eps and eps[-1] > 0 else float("nan")),
            "cp_band_peak_error_improvement": (abs(e_peak[0]) - abs(e_peak[-1])
                                              if len(e_peak) >= 2 else float("nan")),
            "cond1_density_falls": bool(cond1),
            "cond2_cp_error_improves": bool(cond2),
            "cond3_mass_gate_intact": bool(cond3),
            "verdict": "CREDITED" if (cond1 and cond2 and cond3) else "NOT_CREDITED",
        })
    else:
        credit["verdict"] = "NO_DATA"

    # Reproducibility point: the shipped c0 = 20 V, dx = 2 mm member should
    # reproduce the case document's cp_band_peak = 28.05 (committed table).
    repro = None
    for r in rows:
        if abs(r["dx"] - 2.0e-3) < 1e-12 and abs(r["c0_over_V"] - 20.0) < 1e-12:
            repro = {"case_id": r["case_id"],
                     "cp_band_peak": r["cp_band_peak"],
                     "documented_cp_band_peak": 28.05,
                     "rel_diff": (r["cp_band_peak"] - 28.05) / 28.05}
    summary = {
        "anchor": "CASE-SPH-WEDGE-ENTRY-2D",
        "hypothesis": "H1: at fixed dx the Cp error of this case is bounded "
                      "below by the density excursion and falls with c0/V",
        "plan": {"dx": list(DX_LADDER), "c0_over_V": list(C0_FACTORS)},
        "reference": {"wagner_cp_max": ref["wagner"]["cp_max"],
                      "wagner_cp_centre": ref["wagner"]["cp_centre"],
                      "zhao_faltinsen_1993": ref["zhao_faltinsen_1993"]["status"]},
        "cases": [{k: r[k] for k in
                   ("case_id", "dx", "c0_over_V", "steps", "wall_seconds",
                    "cp_band_peak", "tau_at_band_peak", "cp_at_peak_tau",
                    "cp_centre_mean", "cp_interior_max", "cp_footprint_max",
                    "max_rho_ratio", "cp_density_budget", "err_band_peak",
                    "err_at_peak_tau", "err_centre_mean",
                    "verdict_band_peak", "verdict_centre", "verdict_mass")}
                  for r in rows],
        "ladders": ladders,
        "credit": credit,
        "reproducibility": repro,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1,
                                                     sort_keys=True))
    (out_dir / "refs_derived.json").write_text(json.dumps(ref, indent=1,
                                                          sort_keys=True))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="accept", choices=["run", "accept"])
    ap.add_argument("--shard", default=None, help="i/n, i counted from 0")
    ap.add_argument("--out-dir", default=str(ART))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.stage == "run":
        run_stage(out_dir, args.shard)
        return 0
    summary = accept(out_dir)
    print(json.dumps(summary["credit"], indent=1, sort_keys=True))
    for r in summary["cases"]:
        print(f"{r['case_id']:22s} dx={r['dx'] * 1e3:>4g}mm c0/V={r['c0_over_V']:>5g} "
              f"maxrho-1={r['max_rho_ratio'] - 1.0:+.5f} "
              f"Cp_band={r['cp_band_peak']:9.3f} err={r['err_band_peak']:+.3%} "
              f"Cp_c={r['cp_centre_mean']:8.3f} err={r['err_centre_mean']:+.3%} "
              f"floor={r['cp_density_budget']:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())