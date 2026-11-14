"""CASE-SPH-WEDGE-ENTRY-2D, compressibility closure: is the +234 % Cp excess a
weakly-compressible (acoustic) noise floor, or the solution?

Why this study exists
---------------------
``run.py`` graded the beta = 30 deg wedge against the Wagner anchor at
``c0/V = 20`` and found ``Cp_band_peak = 28.0`` against ``Cp_max = 8.402``, i.e.
+234 %.  The report (``docs/case_sph_wedge_entry_2d.md``, sec. 5.3) *diagnosed*
the cause without ever testing it: WCSPH synthesises pressure from the density
error, ``p = B[(rho/rho0)^gamma - 1]`` with ``B = rho0 c0^2/gamma``, so at
``c0/V = 20`` even a 1 % density excursion is worth
``Cp = 2 (c0/V)^2/gamma [(1.01)^7 - 1] = 8.2`` -- the same order as the
reference peak itself.  A diagnosis that is never put at risk is a guess, so
this driver turns it into a falsifiable hypothesis with one free parameter.

Hypothesis (H)
--------------
The excess is dominated by the acoustic noise floor, therefore raising
``c0/V`` at fixed grid must drive ``Cp_band_peak`` towards ``Cp_max,Wagner``:
the physically identical flow (same beta, V, dx, t_end) is computed with a
stiffer equation of state, i.e. a smaller Mach number ``V/c0``.

Falsifier (F)
-------------
If ``Cp_band_peak`` does not fall as ``c0/V`` rises (predeclared tolerance: a
level may rise by at most ``MONOTONE_TOL = 2 %`` relative to the coarser-sound
level before it), then the excess is *not* compressibility-driven and the
report must say so and move the lever to resolution / the pressure estimator.
The verdict is emitted by :func:`classify`, not by hand.

Predeclared design (frozen before the first run of this study)
-------------------------------------------------------------
* beta = 30 deg, V = 1 m/s, dx = 4 mm, c0/V in {20, 40, 80, 160} (Mach
  V/c0 = 0.05, 0.025, 0.0125, 0.00625).  dx, beta, V, t_end, frames, alpha,
  Courant number identical across the four: ``c0/V`` is the only knob.
* One resolution confirmation at the stiffest sound speed: dx = 2 mm,
  c0/V = 160.  This separates "compressibility was the whole story" from
  "compressibility was the dominant story and the rest is grid".
* Tolerance for the case-pool acceptance stays 15 % (``CP_TOL``) exactly as in
  ``run.py``; nothing here relaxes it.
* Aggregate metric unchanged: ``cp_band_peak`` = maximum of the tau-binned
  ensemble mean wall-pressure profile over the last ``ENSEMBLE_FRAMES`` frames
  restricted to ``tau <= 1`` (Wagner's own peak abscissa is tau = 0.93).
  Same definition as ``run.py`` -- see ``run.grade``.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/compressibility.py --stage sweep
    python benchmarks/case_sph_wedge_entry_2d/compressibility.py --stage accept
    python benchmarks/case_sph_wedge_entry_2d/make_compressibility_figures.py

Both stages are deterministic: no RNG is used anywhere in the solver path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import run as base  # noqa: E402  (the frozen case driver, reused not copied)

OUT = base.ART / "compressibility"

# --------------------------------------------------------------------------- #
# Predeclared design.  Changing any value below invalidates the verdict of the
# already-recorded run, so the literals are pinned by a test.
# --------------------------------------------------------------------------- #
FROZEN = {
    "beta_deg": 30.0,
    "velocity": 1.0,
    "dx": 4.0e-3,
    "sound_speed_factor_levels": (20.0, 40.0, 80.0, 160.0),
    "confirm_dx": 2.0e-3,
    "confirm_sound_speed_factor": 160.0,
    "frames": 13,
    "model": "constant-velocity Wagner, beta = 30 deg (TensorSPH analytic)",
}

# --------------------------------------------------------------------------- #
# Stage 2 -- noise-source identification, declared AFTER stage 1 was graded
# (sequential/adaptive design, disclosed in docs/case_sph_wedge_entry_2d_
#  compressibility.md sec. 2).  Stage 1 showed the Cp excess is *not* monotone in
# c0/V, i.e. it is not the acoustic floor per se, so the next question is what
# the noise is made of.  Stage 2 keeps dx, beta, V, c0/V and the metric fixed and
# varies only the delta-SPH diffusive density term (Molteni form), which acts on
# exactly the density field that stage 1 pointed at.
# --------------------------------------------------------------------------- #
DD_FROZEN = {
    "beta_deg": 30.0,
    "velocity": 1.0,
    "dx": 4.0e-3,
    "sound_speed_factor": 20.0,          # stage-1 reference level, unchanged
    "delta_levels": (0.0, 0.05, 0.10, 0.20),
    "form": "molteni",
    "confirm_dx": 2.0e-3,                # best delta re-run at dx/2
    "frames": 13,
}
# Rule for picking the delta that the resolution confirmation uses: fixed here,
# before the delta ladder is complete, so the confirmation cannot cherry-pick.
DD_PICK_RULE = "argmin |err_band_peak| over delta_levels at dx = 4 mm"

CP_TOL = base.THRESHOLDS["cp_peak_rel_err"]        # 0.15, same as the case sheet
MASS_TOL = base.THRESHOLDS["mass_conservation_max_rho"]  # 0.05
MONOTONE_TOL = 0.02          # a level may rise 2 % (noise) before "not falling"
EXPLAINED_DOMINANT = 0.50    # fraction of the excess that c0 must account for


def case_list() -> list[dict]:
    """The frozen case matrix: the c0 ladder plus one resolution confirmation."""
    cases = [{
        "id": f"b{FROZEN['beta_deg']:g}_V{FROZEN['velocity']:g}"
              f"_dx{FROZEN['dx'] * 1e3:g}mm_c0{int(c0):03d}",
        "beta_deg": FROZEN["beta_deg"],
        "velocity": FROZEN["velocity"],
        "dx": FROZEN["dx"],
        "sound_speed_factor": c0,
        "frames": FROZEN["frames"],
        "kind": "compressibility",
        "role": "sound_ladder",
    } for c0 in FROZEN["sound_speed_factor_levels"]]
    cases.append({
        "id": f"b{FROZEN['beta_deg']:g}_V{FROZEN['velocity']:g}"
              f"_dx{FROZEN['confirm_dx'] * 1e3:g}mm"
              f"_c0{int(FROZEN['confirm_sound_speed_factor']):03d}",
        "beta_deg": FROZEN["beta_deg"],
        "velocity": FROZEN["velocity"],
        "dx": FROZEN["confirm_dx"],
        "sound_speed_factor": FROZEN["confirm_sound_speed_factor"],
        "frames": FROZEN["frames"],
        "kind": "compressibility",
        "role": "resolution_confirmation",
    })
    return cases


# --------------------------------------------------------------------------- #
# Acoustic noise floor (closed form, computed - never typed)
# --------------------------------------------------------------------------- #

def cp_density_floor(c0_over_V: float, rho_ratio: float, gamma: float = 7.0) -> float:
    """Cp that a density excursion ``rho/rho0`` alone produces.

    ``p = B[(rho/rho0)^gamma - 1]``, ``B = rho0 c0^2/gamma``, ``q = rho0 V^2/2``
    => ``Cp = p/q = 2 (c0/V)^2 / gamma * [(rho/rho0)^gamma - 1]``.
    Independent of the solution being right; it is the floor under any
    pressure reading taken from a WCSPH density error.
    """
    return (2.0 * c0_over_V ** 2 / gamma
            * (rho_ratio ** gamma - 1.0))


# --------------------------------------------------------------------------- #
# Stage: sweep
# --------------------------------------------------------------------------- #

def sweep(out_dir: Path = OUT, shard: str | None = None) -> list[dict]:
    cases = case_list()
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        cases = [c for k, c in enumerate(cases) if k % n == i]
    done = []
    for spec in cases:
        print(f"[compress] {spec['id']} c0/V={spec['sound_speed_factor']:g} "
              f"dx={spec['dx'] * 1e3:g}mm", flush=True)
        art = base.run_spec(spec, out_dir)
        lf = base.last_frame(art)
        done.append({"case_id": art["case_id"], "steps": art["steps"],
                     "wall_seconds": art["wall_seconds"],
                     "cp_footprint_last": lf["cp_max_footprint"],
                     "cp_centre_last": lf["cp_centre"],
                     "max_rho_ratio": max(r["max_rho_ratio"] for r in art["series"])})
        d = done[-1]
        print(f"[compress] {d['case_id']} steps={d['steps']} "
              f"wall={d['wall_seconds']:.1f}s cp_foot={d['cp_footprint_last']:.3f} "
              f"cp_c={d['cp_centre_last']:.3f} maxrho={d['max_rho_ratio']:.4f}",
              flush=True)
    return done


# --------------------------------------------------------------------------- #
# Stage: accept
# --------------------------------------------------------------------------- #

def load(out_dir: Path = OUT) -> dict:
    arts = {}
    for p in sorted(out_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if "case_id" in d and "series" in d:
            arts[d["case_id"]] = d
    return arts


def metrics(arts: dict) -> list[dict]:
    """Per-case metrics, taken from the frozen grader of ``run.py`` (not re-implemented)."""
    ref = base.refs_derived()
    graded = base.grade(arts, ref)
    by_id = {k: v for k, v in arts.items()}
    rows = []
    for spec in case_list():
        cid = spec["id"]
        if cid not in by_id:
            continue
        row = next(r for r in graded["rows"] if r["case_id"] == cid)
        taus, cps, n = base.ensemble_profile(by_id[cid])
        row.update({
            "role": spec["role"],
            "c0_over_V": spec["sound_speed_factor"],
            "cp_density_floor": cp_density_floor(
                spec["sound_speed_factor"], row["max_rho_ratio"]),
            "ensemble_frames": n,
            "profile_tau": taus,
            "profile_cp": cps,
        })
        rows.append(row)
    return rows


def classify(ladder: list[dict], ref_cp_max: float) -> dict:
    """Machine verdict on hypothesis H (see module docstring).  Pure function.

    ``ladder`` is ordered coarse-sound -> stiff-sound (c0/V increasing).
    """
    if len(ladder) < 2:
        return {"verdict": "WITHHELD", "reason": "fewer than two sound levels"}
    vals = [r["cp_band_peak"] for r in ladder]
    if any(not (isinstance(v, (int, float)) and math.isfinite(v)) for v in vals):
        return {"verdict": "WITHHELD", "reason": "non-finite cp_band_peak"}
    non_increasing = all(
        vals[i + 1] <= vals[i] * (1.0 + MONOTONE_TOL) for i in range(len(vals) - 1))
    # "Non-increasing within noise" is not the same as "falls": a ladder that
    # merely wobbles (or closes a fraction of a percent) must not be promoted to
    # a partial confirmation, so a real drop at the stiff end is required too.
    falling = vals[-1] < vals[0]
    excess0 = vals[0] - ref_cp_max
    drop = vals[0] - vals[-1]
    explained = drop / excess0 if excess0 > 0 else float("nan")
    err_final = abs(vals[-1] - ref_cp_max) / ref_cp_max
    out = {
        "cp_first": vals[0], "cp_last": vals[-1],
        "c0_first": ladder[0]["c0_over_V"], "c0_last": ladder[-1]["c0_over_V"],
        "excess_at_first": excess0, "drop": drop,
        "explained_fraction": explained,
        "rel_err_last": err_final,
        "non_increasing": non_increasing,
        "falls_at_stiff_end": falling,
        "cp_tol": CP_TOL, "monotone_tol": MONOTONE_TOL,
    }
    if not (non_increasing and falling):
        out["verdict"] = "FALSIFIED_NO_TREND"
        out["reason"] = ("raising c0/V does not lower Cp_band_peak: the excess is not "
                         "the acoustic noise floor; move the lever to resolution or "
                         "the pressure estimator")
    elif err_final <= CP_TOL:
        out["verdict"] = "SUPPORTED_FULLY"
        out["reason"] = ("Cp_band_peak converges onto the Wagner anchor within the "
                         "case-pool tolerance once the Mach number is small enough: "
                         "the earlier excess was compressibility noise")
    elif explained >= EXPLAINED_DOMINANT:
        out["verdict"] = "PARTIALLY_SUPPORTED"
        out["reason"] = ("compressibility explains the dominant part of the excess "
                         "but the stiffest affordable level still misses the anchor; "
                         "the residual is a second mechanism (grid or estimator)")
    else:
        out["verdict"] = "REFUTED_DOMINANT"
        out["reason"] = ("Cp_band_peak falls with c0/V but accounts for less than "
                         f"{EXPLAINED_DOMINANT:.0%} of the excess: compressibility is "
                         "not the dominant error at this configuration")
    return out


def accept(out_dir: Path = OUT) -> dict:
    arts = load(out_dir)
    ref = base.refs_derived()
    rows = metrics(arts)
    ladder = sorted([r for r in rows if r["role"] == "sound_ladder"],
                    key=lambda r: r["c0_over_V"])
    confirm = [r for r in rows if r["role"] == "resolution_confirmation"]
    verdict = classify(ladder, ref["wagner"]["cp_max"])
    for r in rows:
        r["verdict_band_peak"] = (
            "PASS" if abs(r["err_band_peak"]) <= CP_TOL else "FAIL")
        r["verdict_mass"] = "PASS" if r["max_rho_ratio"] - 1.0 <= MASS_TOL else "FAIL"
        r["noise_to_signal"] = (r["cp_density_floor"] / r["cp_band_peak"]
                                if r["cp_band_peak"] else float("nan"))
    # How much of the observed Cp the floor accounts for: the floor is the Cp a
    # density excursion of the observed size would produce on its own.
    summary = {
        "study": "CASE-SPH-WEDGE-ENTRY-2D/compressibility-closure",
        "frozen": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in FROZEN.items()},
        "reference": ref,
        "hypothesis": ("the +Cp excess at c0/V = 20 is the weakly-compressible "
                       "acoustic noise floor, therefore curing with stiffer EOS"),
        "falsifier": ("Cp_band_peak does not fall with c0/V (tolerance "
                      f"{MONOTONE_TOL:.0%} per level)"),
        "verdict": verdict,
        "rows": rows,
        "missing": [c["id"] for c in case_list() if c["id"] not in arts],
        "resolution_confirmation": confirm,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    write_csv(rows, out_dir / "table_compressibility.csv")
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["case_id", "role", "c0_over_V", "dx", "cp_band_peak", "tau_at_band_peak",
            "cp_at_peak_tau", "cp_centre_mean", "cp_footprint_max",
            "cp_max_reference", "cp_centre_reference", "err_band_peak",
            "cp_density_floor", "max_rho_ratio",
            "verdict_band_peak", "verdict_mass"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c)
                cells.append(f"{v:.6g}" if isinstance(v, float) else str(v))
            fh.write(",".join(cells) + "\n")


def dd_case_list() -> list[dict]:
    """Stage-2 frozen matrix: delta ladder at dx = 4 mm + the confirmation case."""
    cases = []
    for delta in DD_FROZEN["delta_levels"]:
        cases.append({
            "id": (f"b{DD_FROZEN['beta_deg']:g}_V{DD_FROZEN['velocity']:g}"
                   f"_dx{DD_FROZEN['dx'] * 1e3:g}mm"
                   f"_c0{int(DD_FROZEN['sound_speed_factor']):03d}"
                   f"_dd{delta:g}"),
            "beta_deg": DD_FROZEN["beta_deg"],
            "velocity": DD_FROZEN["velocity"],
            "dx": DD_FROZEN["dx"],
            "sound_speed_factor": DD_FROZEN["sound_speed_factor"],
            "density_diffusion": delta,
            "density_diffusion_form": DD_FROZEN["form"],
            "frames": DD_FROZEN["frames"],
            "kind": "density_diffusion",
            "role": "delta_ladder",
        })
    cases.append({
        "id": (f"b{DD_FROZEN['beta_deg']:g}_V{DD_FROZEN['velocity']:g}"
               f"_dx{DD_FROZEN['confirm_dx'] * 1e3:g}mm"
               f"_c0{int(DD_FROZEN['sound_speed_factor']):03d}_ddstar"),
        "beta_deg": DD_FROZEN["beta_deg"],
        "velocity": DD_FROZEN["velocity"],
        "dx": DD_FROZEN["confirm_dx"],
        "sound_speed_factor": DD_FROZEN["sound_speed_factor"],
        "density_diffusion": None,      # filled by the frozen pick rule at accept
        "density_diffusion_form": DD_FROZEN["form"],
        "frames": DD_FROZEN["frames"],
        "kind": "density_diffusion",
        "role": "resolution_confirmation",
    })
    return cases


def dd_sweep(out_dir: Path = OUT, shard: str | None = None, delta_star=None) -> list[dict]:
    cases = dd_case_list()
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        cases = [c for k, c in enumerate(cases) if k % n == i]
    done = []
    for spec in cases:
        spec = dict(spec)
        if spec["density_diffusion"] is None:
            if delta_star is None:
                print(f"[dd] skipping {spec['id']}: pass --delta-star (see DD_PICK_RULE)",
                      flush=True)
                continue
            spec["density_diffusion"] = float(delta_star)
        print(f"[dd] {spec['id']} delta={spec['density_diffusion']:g} "
              f"dx={spec['dx'] * 1e3:g}mm", flush=True)
        art = base.run_spec(spec, out_dir)
        lf = base.last_frame(art)
        done.append({"case_id": art["case_id"], "steps": art["steps"],
                     "wall_seconds": art["wall_seconds"],
                     "max_rho_ratio": max(r["max_rho_ratio"] for r in art["series"])})
        print(f"[dd] {done[-1]['case_id']} steps={done[-1]['steps']} "
              f"wall={done[-1]['wall_seconds']:.1f}s "
              f"cp_foot={lf['cp_max_footprint']:.3f} maxrho={done[-1]['max_rho_ratio']:.4f}",
              flush=True)
    return done


def dd_accept(out_dir: Path = OUT) -> dict:
    """Grade the delta ladder; pick delta* by the frozen rule; report confirmation."""
    arts = {}
    for p in sorted(out_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if "case_id" in d and "series" in d and "density_diffusion" in d["config"]:
            arts[d["case_id"]] = d
    ref = base.refs_derived()
    graded = base.grade(arts, ref)
    by_id = {k: v for k, v in arts.items()}
    ladder, confirm = [], []
    for spec in dd_case_list():
        cid = spec["id"]
        match = None
        if cid in by_id:
            match = by_id[cid]
        else:  # the confirmation case id depends on delta*, match by config
            for k, v in by_id.items():
                c = v["config"]
                if (spec["role"] == "resolution_confirmation"
                        and abs(c["dx"] - spec["dx"]) < 1e-12
                        and abs(c["sound_speed_factor"] - spec["sound_speed_factor"]) < 1e-12
                        and c.get("density_diffusion", 0.0) > 0.0):
                    match = v
                    break
        if match is None:
            continue
        row = next((r for r in graded["rows"] if r["case_id"] == match["case_id"]), None)
        if row is None:
            continue
        row = dict(row)
        taus, cps, n = base.ensemble_profile(match)
        row.update({"role": spec["role"],
                    "delta": match["config"]["density_diffusion"],
                    "c0_over_V": match["config"]["sound_speed_factor"],
                    "cp_density_floor": cp_density_floor(
                        match["config"]["sound_speed_factor"], row["max_rho_ratio"]),
                    "ensemble_frames": n,
                    "profile_tau": taus, "profile_cp": cps})
        (ladder if spec["role"] == "delta_ladder" else confirm).append(row)
    ladder.sort(key=lambda r: r["delta"])
    base_cp = None
    for r in ladder:
        if r["delta"] == 0.0:
            base_cp = r["cp_band_peak"]
    excess0 = (base_cp - ref["wagner"]["cp_max"]) if base_cp is not None else float("nan")
    for r in ladder:
        r["err_band_peak"] = abs(r["cp_band_peak"] - ref["wagner"]["cp_max"]) / ref["wagner"]["cp_max"]
        r["excess_reduction"] = ((excess0 - (r["cp_band_peak"] - ref["wagner"]["cp_max"]))
                                / excess0 if excess0 else float("nan"))
        r["verdict_band_peak"] = "PASS" if r["err_band_peak"] <= CP_TOL else "FAIL"
        r["verdict_mass"] = "PASS" if r["max_rho_ratio"] - 1.0 <= MASS_TOL else "FAIL"
    best = min(ladder, key=lambda r: r["err_band_peak"]) if ladder else None
    verdict = {
        "pick_rule": DD_PICK_RULE,
        "delta_star": best["delta"] if best else None,
        "delta_star_err_band_peak": best["err_band_peak"] if best else None,
        "pass_at_dx4mm": bool(best and best["err_band_peak"] <= CP_TOL),
        "base_excess": excess0,
        "max_excess_reduction": (max(r["excess_reduction"] for r in ladder)
                                 if ladder else float("nan")),
    }
    if ladder and all(r["err_band_peak"] > CP_TOL for r in ladder):
        verdict["verdict"] = ("PARTIAL_NOISE_IS_DENSITY" if verdict["max_excess_reduction"] > 0.5
                              else "FALSIFIED_NOISE_NOT_DENSITY")
    elif ladder:
        verdict["verdict"] = "PASS_WITHIN_TOLERANCE"
    else:
        verdict["verdict"] = "WITHHELD"
    summary = {
        "study": "CASE-SPH-WEDGE-ENTRY-2D/density-diffusion-closure",
        "frozen": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in DD_FROZEN.items()},
        "pick_rule": DD_PICK_RULE,
        "reference": ref,
        "hypothesis": ("the Cp excess is carried by the density field, so the "
                       "delta-SPH diffusive density term reduces it at fixed dx"),
        "falsifier": ("no excess reduction > 50 % over the delta ladder"),
        "verdict": verdict,
        "rows": ladder,
        "resolution_confirmation": confirm,
    }
    (out_dir / "summary_dd.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    write_csv(ladder + confirm, out_dir / "table_density_diffusion.csv")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="accept",
                    choices=["sweep", "accept", "dd-sweep", "dd-accept"])
    ap.add_argument("--shard", default=None, help="i/n, i counted from 0")
    ap.add_argument("--delta-star", type=float, default=None,
                    help="delta for the resolution confirmation (see DD_PICK_RULE)")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.stage == "sweep":
        sweep(out_dir, args.shard)
        return 0
    if args.stage == "dd-sweep":
        dd_sweep(out_dir, args.shard, args.delta_star)
        return 0
    if args.stage == "dd-accept":
        summary = dd_accept(out_dir)
        print(json.dumps(summary["verdict"], indent=1))
        return 0
    summary = accept(out_dir)
    print(json.dumps({"verdict": summary["verdict"],
                      "missing": summary["missing"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())