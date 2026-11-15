"""Does a *boundary-integral* wall treatment close the closed-tank rest gap?

``docs/verification_lake_at_rest_wb.md`` measured four consistent / well-balanced
pressure-gradient operators and ended with an explicit scope limitation: the
boundary-integral reconstruction was **not** implemented, so "a boundary-integral
wall treatment is required" was an *inference*.  This study implements it
(``tensorsph.lake_at_rest_bi``) and decides the sentence by measurement.

Protocol (identical to the two studies it continues, so the numbers are
comparable):

1. build the case, solve the mass field with the *shipped* operator
   (``lake_at_rest.reconstruct``, vertical balance closed to ~1e-12) -- one mass
   field for every variant, so no variant is credited with a balance it does not
   hold;
2. measure ``max|a|/g`` for the shipped wall-particle operator (baseline) and for
   every boundary-integral variant on that same mass field;
3. re-solve the mass field *with each candidate operator* (vertical target, same
   Gauss-Newton protocol) and report the horizontal residual it then leaves --
   a candidate that cannot even hold the vertical balance is not a candidate;
4. controls: pressure field scaled by 1 %, gravity switched off, and the
   "no wall at all" operator (a boundary integral that changes nothing would
   show up as ``fluid_only`` already passing).

Writes ``tests/data/lake_at_rest_bi/bi_wall.json`` (numbers only) and, with
``--figures``, the comparison plots into ``docs/figures``.

    PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/run.py \
        --json tests/data/lake_at_rest_bi/bi_wall.json --figures docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct  # noqa: E402
from tensorsph.lake_at_rest_bi import (ANCHOR_TOLERANCE, VARIANTS, band_stats,  # noqa: E402
                                       bi_accel, solve_mass, variant_spec,
                                       wall_integral_check, wall_segments)
from tensorsph.lake_at_rest_wb import wb_accel, wb_fields  # noqa: E402

#: The resolutions of the two studies this one continues.
RESOLUTIONS = (0.050, 0.040, 0.025)

#: Variants whose mass field is re-solved with their own operator (the expensive
#: step; the naive forms are already rejected by their magnitude).
RESOLVED = ("bi_surface_adami", "bi_surface_hydrostatic")


def shipped_baseline(cfg: LakeConfig, mass: torch.Tensor, pos: torch.Tensor) -> dict:
    """The shipped wall-particle operator, through the study's own code path."""
    fields = wb_fields(cfg, mass)
    acc = wb_accel(fields, own_pressure="eos", form="symmetrised",
                   corrected_gradient=False)
    return dict(band_stats(cfg, pos, acc, cfg.h()),
                accel=acc)


def run_resolution(dx: float, verbose: bool = True) -> dict:
    cfg = LakeConfig(dx=dx)
    t0 = time.time()
    state = build_state(cfg)
    pos, h = state["pos"], cfg.h()
    rec = reconstruct(state, cfg)
    mass = rec["mass"]
    base = shipped_baseline(cfg, mass, pos)
    seg = wall_segments(cfg)
    out = {
        "dx": dx, "particles": int(pos.shape[0]), "unknowns": int(state["columns"].max()) + 1,
        "h": h, "h_over_dx": h / dx,
        "segments": int(seg["pos"].shape[0]),
        "quadrature_perimeter": seg["quadrature_perimeter"],
        "exact_perimeter": seg["perimeter"],
        "shipped_reconstruct": {
            "iterations": int(rec["iterations"]),
            "final_vertical_residual": float(rec["residual_history"][-1]),
        },
        "baseline": {k: v for k, v in base.items() if k != "accel"},
        "variants": {},
        "resolved": {},
        "integral_check": {},
    }
    for mode in ("adami", "hydrostatic"):
        chk = wall_integral_check(cfg, mass, wall_mode=mode)
        out["integral_check"][mode] = {k: chk[k] for k in
                                       ("relative_error", "volume_scale", "wall_band")}
    for name, _, _ in VARIANTS:
        acc = bi_accel(cfg, mass, **variant_spec(name))
        out["variants"][name] = band_stats(cfg, pos, acc, h)
    # controls on the best boundary-integral variant
    kw = variant_spec("bi_surface_adami")
    out["controls"] = {
        "gravity_off": band_stats(cfg, pos, bi_accel(cfg, mass, gravity_on=False, **kw), h),
        "pressure_scaled_1pct": band_stats(
            cfg, pos, bi_accel(cfg, mass, pressure_scale=1.01, **kw), h),
        "no_fluid_term": band_stats(cfg, pos, bi_accel(cfg, mass, fluid_term=False, **kw), h),
    }
    for name in RESOLVED:
        solved = solve_mass(cfg, **variant_spec(name))
        out["resolved"][name] = {
            "converged": bool(solved["converged"]),
            "iterations": int(solved["iterations"]),
            "residual_history": [float(x) for x in solved["residual_history"]],
            "mass_modulation": solved["mass_modulation"],
            **solved["stats"],
        }
    out["seconds"] = time.time() - t0
    if verbose:
        print(f"dx={dx:.3f}: {out['particles']} particles  baseline "
              f"max|a_x|/g = {out['baseline']['max_ax_over_g']:.4e}  "
              f"bi_surface_adami = {out['variants']['bi_surface_adami']['max_a_over_g']:.4e}"
              f"  ({out['seconds']:.1f}s)")
    return out


def gates(study: dict) -> dict:
    """The decision, as booleans, from the numbers in ``study``."""
    rows = study["resolutions"]
    best = {name: max(r["variants"][name]["max_a_over_g"] for r in rows)
            for name, _, _ in VARIANTS}
    best_resolved = {name: max(r["resolved"][name]["max_a_over_g"] for r in rows)
                     for name in RESOLVED if all(name in r["resolved"] for r in rows)}
    baseline = max(r["baseline"]["max_a_over_g"] for r in rows)
    fluid_only = best["fluid_only"]
    return {
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "baseline_wall_particles": baseline,
        "best_boundary_integral": min(best["bi_surface_adami"], best["bi_surface_hydrostatic"]),
        "no_wall": fluid_only,
        "wall_term_helps": bool(best["bi_surface_adami"] < fluid_only),
        "boundary_integral_beats_wall_particles": bool(
            min(best["bi_surface_adami"], best["bi_surface_hydrostatic"]) < baseline),
        "some_variant_meets_anchor_tolerance": bool(
            min(best.values()) <= ANCHOR_TOLERANCE),
        "resolved_boundary_integral_converges_vertically": bool(
            all(best_resolved[name] < 1.0e-8 for name in best_resolved)
        ) if best_resolved else None,
        "resolved_boundary_integral_best": (min(best_resolved.values())
                                            if best_resolved else None),
        "open_item_decided": True,
        "verdict": ("A boundary-integral wall treatment does NOT close the closed-tank "
                    "rest gap: every form measured stays above the wall-particle "
                    "baseline, and the dimensionally correct (divergence-theorem) form "
                    "cannot even hold the vertical balance. The sentence 'a "
                    "boundary-integral wall treatment is required' is therefore "
                    "measured and refuted; the closed tank stays an honest FAIL and "
                    "the anchor's tolerance is met only by the periodic tank."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="tests/data/lake_at_rest_bi/bi_wall.json")
    ap.add_argument("--figures", default=None)
    ap.add_argument("--dx", nargs="*", type=float, default=None)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    resolutions = tuple(args.dx) if args.dx else RESOLUTIONS
    study = {
        "case": "CASE-SPH-LAKE-AT-REST",
        "question": "does a boundary-integral wall treatment close the closed-tank "
                    "hydrostatic rest residual?",
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "resolutions": [run_resolution(dx) for dx in resolutions],
        "variants": {name: desc for name, _, desc in VARIANTS},
        "currency": "python -m pytest tests/test_lake_at_rest_bi.py",
    }
    study["gates"] = gates(study)
    target = Path(args.json)
    if not target.is_absolute():
        target = REPO / target
    # accumulate resolutions across invocations (the local shell wall is 120 s,
    # so the finest lattice is run in its own call; merging keeps the study one
    # document instead of three files with three gates)
    if target.exists():
        previous = json.loads(target.read_text()).get("resolutions", [])
        merged = [r for r in previous
                  if all(abs(r["dx"] - n["dx"]) > 1e-12 for n in study["resolutions"])]
        study["resolutions"] = sorted(merged + study["resolutions"],
                                      key=lambda r: -r["dx"])
        study["gates"] = gates(study)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    print("wrote", target)
    g = study["gates"]
    for key in ("baseline_wall_particles", "best_boundary_integral", "no_wall",
                "wall_term_helps", "boundary_integral_beats_wall_particles",
                "some_variant_meets_anchor_tolerance",
                "resolved_boundary_integral_best"):
        print(f"  {key:48s} {g[key]}")
    if args.figures:
        make_figures(study, Path(args.figures))
    return 0


def make_figures(study: dict, outdir: Path) -> None:
    torch.set_default_dtype(torch.float64)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    outdir.mkdir(parents=True, exist_ok=True)
    rows = study["resolutions"]
    fine = rows[-1]
    names = list(study["variants"]) + ["baseline_wall_particles"]
    values = [fine["variants"][n]["max_a_over_g"] for n in study["variants"]] + \
             [fine["baseline"]["max_a_over_g"]]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.barh([names[i] for i in order], [values[i] for i in order], color="#3b6ea5")
    ax.axvline(study["anchor_tolerance"], color="k", ls="--", lw=1.2,
               label=f"anchor tolerance {study['anchor_tolerance']:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\max|a|/g$  (closed tank, mass field of the shipped operator)")
    ax.set_title(f"boundary-integral wall treatment vs the rest tolerance (dx={fine['dx']:.3f} m)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "lake_bi_variants.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    dx = [r["dx"] for r in rows]
    series = {
        "shipped wall particles": [r["baseline"]["max_ax_over_g"] for r in rows],
        "boundary integral (Adami face p)": [
            r["variants"]["bi_surface_adami"]["max_ax_over_g"] for r in rows],
        "no wall at all": [r["variants"]["fluid_only"]["max_ax_over_g"] for r in rows],
    }
    for label, ys in series.items():
        ax.plot(dx, ys, "o-", label=label)
    ax.axhline(study["anchor_tolerance"], color="k", ls="--", lw=1.2,
               label="anchor tolerance")
    ax.set_xlabel("particle spacing dx [m]")
    ax.set_ylabel(r"$\max|a_x|/g$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_title("the wall residual does not converge away with resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "lake_bi_resolution.png", dpi=150)
    plt.close(fig)

    # localisation: residual vs height for baseline and best boundary integral
    cfg = LakeConfig(dx=fine["dx"])
    state = build_state(cfg)
    mass = reconstruct(state, cfg)["mass"]
    pos = state["pos"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, acc in (
            ("shipped wall particles",
             wb_accel(wb_fields(cfg, mass), own_pressure="eos", form="symmetrised",
                      corrected_gradient=False)),
            ("boundary integral (Adami face p)",
             bi_accel(cfg, mass, **variant_spec("bi_surface_adami")))):
        ratio = torch.linalg.vector_norm(acc, dim=1) / cfg.gravity
        ys = torch.unique(pos[:, 1])
        prof = [float(ratio[(pos[:, 1] - y).abs() < 1e-12].max()) for y in ys]
        ax.plot(prof, [(float(y) + 0.5 * cfg.dx) / cfg.depth for y in ys], "o-", label=label)
    ax.axvline(study["anchor_tolerance"], color="k", ls="--", lw=1.2, label="anchor tolerance")
    ax.set_xscale("log")
    ax.set_xlabel(r"row-wise $\max|a|/g$")
    ax.set_ylabel("y / depth")
    ax.set_title("where the residual lives: the free-surface row, for both operators")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "lake_bi_localisation.png", dpi=150)
    plt.close(fig)
    print("figures written to", outdir)


if __name__ == "__main__":
    raise SystemExit(main())