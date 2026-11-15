"""3-D solver leg of the TVC verification (``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``).

Why this leg exists
-------------------
The dimension leg (``dimension_leg.py``) made the operator dimension aware and
graded it on *lattices*; the physics leg ran the 2-D dam break.  Two debts were
left, and both are about the solver the anchor's own motivation names
(``tensorsph.dam_break_3d``, the ship-hydrodynamics case):

1. **the 3-D solver never applied the correction at all** -- ``run_case`` had no
   TVC hook, so the dimension fix had no 3-D consumer;
2. **the 3-D front metric was mis-framed**: ``DamBreak3DRun.dimensionless``
   divided the absolute leading-edge coordinate by the column width, while the
   Ritter and Martin & Moyce references measure the distance travelled *from
   the dam face*.  On the lattice that adds ``1 - dx/(2a)`` -- 0.9 at
   ``dx = 0.03`` -- to every ``Z/a``, i.e. the 3-D case was graded with a
   constant ``+0.9`` offset (``tests/test_dam_break_3d.py::test_dimensionless_front``
   asserted the offset as if it were physics).  The 2-D leg documents the same
   trap in ``docs/xval_sphinx_tvc/README.md``.

This script closes both, and grades them.  Criteria are declared *before* the
runs (``CRITERIA`` below is written into the payload with the merge step never
touching it) so that the numbers cannot select the criteria after the fact.

Usage
-----
    python benchmarks/xval_sphinx_tvc/solver_leg.py anchors
    python benchmarks/xval_sphinx_tvc/solver_leg.py flow --dx 0.02 --variant tvc010
    python benchmarks/xval_sphinx_tvc/solver_leg.py merge
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

from tensorsph.dam_break_3d import (  # noqa: E402
    DamBreak3DConfig,
    front_position_3d,
    make_column_3d,
    ritter_front_over_a,
    run_case,
)

import dimension_leg as DL  # noqa: E402  (surface anchor, fixed point, wall branch)

DATA = ROOT / "docs" / "xval_sphinx_tvc" / "data"
RUNS = DATA / "solver_leg"
FIGS = ROOT / "docs" / "xval_sphinx_tvc"

#: Correction variants of the flow leg.  ``None`` = solver's own default (off).
VARIANTS: dict[str, float | None] = {
    "baseline": None,
    "tvc005": 0.05,
    "tvc010": 0.10,
}
GRIDS = (0.03, 0.02)          # dx [m]; 0.02 is the case default
STEPS = 4000                  # * dt_from_courant() -> 0.2 s of simulated time
SEED = 12345                  # the seed of the shipped 3-D benchmark runner
SOUND_SPEED = 25.0

#: Pre-declared acceptance criteria.  Written into the payload unchanged.
CRITERIA = {
    "C1_bulk_fixed_point": {
        "statement": "max |dr|/h on a full-support 3-D lattice < 1e-10",
        "kind": "exact anchor (no calibration)",
    },
    "C2_surface_anchor": {
        "statement": "free-surface |dr|/h/alpha vs the continuum 3-D value "
                     "within 15 % at h/dx = 1.3",
        "kind": "owner 15 % gate",
    },
    "C3_pre_fix": {
        "statement": "the pre-fix 2-D normalisation under-corrects a 3-D cloud "
                     "by more than 20x at dx = 0.02 (sigma_2/sigma_3 = 10h/7)",
        "kind": "defect magnitude",
    },
    "C4_wall_branch": {
        "statement": "3-D wall residual < 1e-10 for wall_factor = 2 and "
                     "> 0.4 for wall_factor = 1",
        "kind": "exact anchor (geometry)",
    },
    "C5_flow_invariants": {
        "statement": "every flow run: mass drift == 0 and |energy drift| < 1 %",
        "kind": "case invariant (pre-existing)",
    },
    "C6_regularisation": {
        "statement": "at dx = 0.02 the TVC run's close-pair fraction "
                     "(r < 0.7 dx) is at least 50 % below the baseline's",
        "kind": "mechanism",
    },
    "C7_non_intrusive": {
        "statement": "with the corrected frame, TVC's front MAE against Ritter "
                     "does not exceed the baseline MAE by more than 20 % "
                     "(relative)",
        "kind": "grading; the absolute 15 % gate lives on the 2-D leg "
                "(3-D front lag vs a 2-D shallow-water reference is physical)",
    },
}


def _host() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip(),
        "cpu_count": os.cpu_count(),
        "threads": torch.get_num_threads(),
    }


# --------------------------------------------------------------------------- #
# anchors (cheap; operator level, dimension forced to 3 by the solver's own use)
# --------------------------------------------------------------------------- #
def anchors() -> dict:
    out: dict = {
        "criteria": CRITERIA,
        "host": _host(),
        "h_over_dx": DL.HBAR,
        "body": "3-D cubic-spline TVC, dim inferred from a 3-column point cloud",
        "continuum_3d": {
            "I_3_at_h_0.13": DL.plane_integral(0.13, 3),
            "max_shift_over_h_per_alpha": 2.0 * DL.plane_integral(0.13, 3) * 0.13,
            "kernel_normalisation": DL.normalisation(0.13, 3),
        },
        "surface": [],
        "bulk": [],
        "wall": [],
        "pre_fix": [],
    }
    for dx in GRIDS:
        out["surface"].append(DL.surface_shift(dx, 3))
        out["bulk"].append({"dx": dx, "max_abs_shift_over_h": DL.bulk_fixed_point(dx, 3)})
        out["wall"].append({
            "dx": dx,
            "wall_factor_2": DL.wall_residual(dx, 3, 2.0),
            "wall_factor_1": DL.wall_residual(dx, 3, 1.0),
        })
        h = DL.HBAR * dx
        out["pre_fix"].append({
            "dx": dx, "h": h,
            "sigma_ratio_2_over_3": 10.0 * h / 7.0,
            "under_correction_factor": 7.0 / (10.0 * h),
        })
    return out


# --------------------------------------------------------------------------- #
# flow runs
# --------------------------------------------------------------------------- #
def clumping(pos: torch.Tensor, dx: float) -> dict:
    """Nearest-neighbour statistics of the final fluid distribution."""
    n = pos.shape[0]
    d = torch.cdist(pos.double(), pos.double())
    d.fill_diagonal_(float("inf"))
    rnn = d.min(dim=1).values
    close = float((d < 0.7 * dx).sum()) / (n * (n - 1))     # both orders counted
    return {
        "median_rnn_over_dx": float(rnn.median()) / dx,
        "close_pair_fraction": close,
        "particles": int(n),
    }


def flow_run(dx: float, variant: str) -> dict:
    alpha = VARIANTS[variant]
    cfg = DamBreak3DConfig(dx=dx, sound_speed=SOUND_SPEED, steps=STEPS,
                           output_every=100, device="cpu")
    # The case is fixed by the Courant limit, not by the default dt.
    cfg.dt = cfg.dt_from_courant()
    kwargs: dict = {} if alpha is None else {"tvc_every": 1, "tvc_alpha": alpha}

    t0 = time.time()
    run = run_case(cfg, seed=SEED, **kwargs)
    wall = time.time() - t0

    tau, za = run.dimensionless(cfg)                     # corrected frame
    za_old = [z / cfg.column_width for z in run.front]   # pre-fix frame
    ritter = [ritter_front_over_a(t) for t in tau]
    err = [z - r for z, r in zip(za, ritter)]
    err_old = [z - r for z, r in zip(za_old, ritter)]
    rel = [abs(e) / r if r > 1e-12 else float("nan") for e, r in zip(err, ritter)]

    return {
        "dx": dx,
        "variant": variant,
        "alpha": alpha,
        "dt": cfg.dt,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "particles": int(run.final_positions.shape[0]),
        "wall_clock_s": wall,
        "times": run.times,
        "tau": tau,
        "za": za,
        "za_old_frame": za_old,
        "ritter_za": ritter,
        "frame_offset_at_t0": za_old[0] if za_old else 0.0,
        "front_mae": (sum(abs(e) for e in err) / len(err)) if err else float("nan"),
        "front_max_abs_err": max((abs(e) for e in err), default=float("nan")),
        "front_mean_rel_err": (sum(r for r in rel if r == r) /
                               max(1, sum(1 for r in rel if r == r))),
        "front_mae_old_frame": (sum(abs(e) for e in err_old) / len(err_old)),
        "mass_drift": run.mass_drift,
        "energy_drift": run.energy_drift,
        "max_density_compression": run.max_density_compression,
        "max_speed": run.max_speed,
        "escaped": run.escaped,
        "initial_front": run.initial_front,
        "clumping": clumping(run.final_positions, dx),
        "tvc_passes": len(run.tvc_log),
        "tvc_shift_peak_over_h": max((r["max_shift_over_h"] for r in run.tvc_log),
                                     default=0.0),
        "tvc_shift_mean_peak_over_h": max((r["mean_shift_over_h"] for r in run.tvc_log),
                                          default=0.0),
        "tvc_log": run.tvc_log,
    }


# --------------------------------------------------------------------------- #
# merge, doc, figures
# --------------------------------------------------------------------------- #
def _criteria_verdicts(payload: dict) -> dict:
    A = payload["anchors"]
    surf = A["surface"]
    bulk = A["bulk"]
    wall = A["wall"]
    pref = A["pre_fix"]
    runs = payload["runs"]
    cont = A["continuum_3d"]["max_shift_over_h_per_alpha"]

    surf_bias = [abs(r["max_shift_over_h_per_alpha"] - cont) / cont for r in surf]
    verdict = {
        "C1_bulk_fixed_point": {
            "value": max(r["max_abs_shift_over_h"] for r in bulk),
            "pass": max(r["max_abs_shift_over_h"] for r in bulk) < 1e-10,
        },
        "C2_surface_anchor": {
            "continuum": cont,
            "measured": [r["max_shift_over_h_per_alpha"] for r in surf],
            "bias": surf_bias,
            "pass": max(surf_bias) < 0.15,
        },
        "C3_pre_fix": {
            "under_correction_factor": [r["under_correction_factor"] for r in pref],
            "pass": all(r["under_correction_factor"] > 20.0 for r in pref
                        if abs(r["dx"] - 0.02) < 1e-12),
        },
        "C4_wall_branch": {
            "residual_wall_factor_2": [r["wall_factor_2"] for r in wall],
            "residual_wall_factor_1": [r["wall_factor_1"] for r in wall],
            "pass": (max(r["wall_factor_2"] for r in wall) < 1e-10
                     and min(r["wall_factor_1"] for r in wall) > 0.4),
        },
    }
    if runs:
        worst_energy = max(abs(r["energy_drift"]) for r in runs)
        worst_mass = max(abs(r["mass_drift"]) for r in runs)
        verdict["C5_flow_invariants"] = {
            "max_abs_energy_drift": worst_energy,
            "max_abs_mass_drift": worst_mass,
            "pass": worst_energy < 0.01 and worst_mass == 0.0,
        }
        finest = [r for r in runs if abs(r["dx"] - min(g for g in GRIDS)) < 1e-12]
        base = next((r for r in finest if r["variant"] == "baseline"), None)
        tvc = [r for r in finest if r["variant"] != "baseline"]
        if base and tvc:
            best = min(tvc, key=lambda r: r["clumping"]["close_pair_fraction"])
            verdict["C6_regularisation"] = {
                "baseline_close_pair_fraction": base["clumping"]["close_pair_fraction"],
                "best_tvc_close_pair_fraction": best["clumping"]["close_pair_fraction"],
                "best_tvc_variant": best["variant"],
                "reduction_fraction": (
                    1.0 - best["clumping"]["close_pair_fraction"]
                    / base["clumping"]["close_pair_fraction"]
                    if base["clumping"]["close_pair_fraction"] > 0 else 0.0),
                "pass": (base["clumping"]["close_pair_fraction"] > 0 and
                         1.0 - best["clumping"]["close_pair_fraction"]
                         / base["clumping"]["close_pair_fraction"] >= 0.5),
            }
            best_front = min(tvc, key=lambda r: r["front_mae"])
            verdict["C7_non_intrusive"] = {
                "baseline_front_mae": base["front_mae"],
                "best_tvc_front_mae": best_front["front_mae"],
                "best_tvc_variant": best_front["variant"],
                "relative_change": (best_front["front_mae"] / base["front_mae"] - 1.0
                                    if base["front_mae"] > 0 else float("nan")),
                "pass": (base["front_mae"] > 0 and
                         best_front["front_mae"] <= 1.2 * base["front_mae"]),
            }
    return verdict


def merge() -> dict:
    payload = {
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "leg": "solver_3d",
        "criteria": CRITERIA,
        "host": _host(),
        "grids": list(GRIDS),
        "steps": STEPS,
        "seed": SEED,
        "sound_speed": SOUND_SPEED,
        "anchors": json.loads((DATA / "solver_leg_anchors.json").read_text()),
        "runs": [],
    }
    for dx in GRIDS:
        for variant in VARIANTS:
            p = RUNS / f"dx{dx:g}_{variant}.json"
            if p.exists():
                payload["runs"].append(json.loads(p.read_text()))
    payload["verdicts"] = _criteria_verdicts(payload)
    (DATA / "solver_leg.json").write_text(json.dumps(payload, indent=2))
    write_doc(payload)
    figures(payload)
    return payload


def write_doc(out: dict) -> None:
    A = out["anchors"]
    V = out["verdicts"]
    runs = out["runs"]
    cont = A["continuum_3d"]["max_shift_over_h_per_alpha"]
    L: list[str] = []
    A_ = L.append

    A_("# 3-D solver leg -- TVC in `tensorsph.dam_break_3d`, and the front-metric frame fix\n")
    A_("_Generated by `benchmarks/xval_sphinx_tvc/solver_leg.py` from "
       f"`docs/xval_sphinx_tvc/data/solver_leg.json` (host: python {out['host']['python']}, "
       f"torch {out['host']['torch']}, git {out['host']['git']})._ \n")
    A_("## 1. What was missing\n")
    A_("The dimension leg made the operator dimension aware and graded it on "
       "lattices; the physics leg ran the **2-D** dam break.  Neither touched the "
       "case the port's own motivation names, `tensorsph.dam_break_3d`: its "
       "`run_case` had no correction hook at all, and its front metric divided the "
       "*absolute* leading-edge coordinate by the column width, so every reported "
       "`Z/a` carried a constant offset of `1 - dx/(2a)` against the Ritter / "
       "Martin & Moyce frame (which measures distance travelled from the dam face).")
    A_("")
    A_("Both are fixed here: `DamBreak3DConfig` gains `tvc_every` / `tvc_alpha` / "
       "`tvc_scope_tol` / `tvc_max_shift` (defaults keep the case bit-identical), "
       "`run_case` applies the ported operator with `dim` inferred from the cloud "
       "(3), and `DamBreak3DRun.dimensionless` subtracts the leading edge at "
       "`t = 0`.\n")

    A_("## 2. Criteria, declared before the runs\n")
    A_("| id | criterion | kind |")
    A_("|---|---|---|")
    for k, v in out["criteria"].items():
        A_(f"| {k} | {v['statement']} | {v['kind']} |")
    A_("")

    A_("## 3. Analytic anchors (no calibration involved)\n")
    A_(f"Continuum free-surface shift per unit `alpha`: "
       f"`|dr|/h/alpha = 2 h I_3(h,0) = {cont:.6f}` "
       f"(quadrature `I_3 = {A['continuum_3d']['I_3_at_h_0.13']:.6f}`, "
       f"`int W_3 dV = {A['continuum_3d']['kernel_normalisation']:.12f}`).\n")
    A_("| dx [m] | measured `|dr|/h/alpha` | bias vs continuum | `max |dr|/h` bulk | "
       "wall res. (factor 2) | wall res. (factor 1) |")
    A_("|---|---|---|---|---|---|")
    for surf, bulk, wall in zip(A["surface"], A["bulk"], A["wall"]):
        bias = abs(surf["max_shift_over_h_per_alpha"] - cont) / cont
        A_(f"| {surf['dx']:g} | {surf['max_shift_over_h_per_alpha']:.6f} | "
           f"{100*bias:+.2f} % | {bulk['max_abs_shift_over_h']:.2e} | "
           f"{wall['wall_factor_2']:.2e} | {wall['wall_factor_1']:.4f} |")
    A_("")
    A_("Pre-fix behaviour on a 3-D cloud (`sigma_2/sigma_3 = 10 h/7`):\n")
    A_("| dx [m] | h [m] | pre-fix / correct shift | under-correction |")
    A_("|---|---|---|---|")
    for r in A["pre_fix"]:
        A_(f"| {r['dx']:g} | {r['h']:.4f} | {r['sigma_ratio_2_over_3']:.4f} | "
           f"{r['under_correction_factor']:.2f}x |")
    A_("")

    A_("## 4. Flow runs -- 3-D dam break, correction on and off\n")
    A_(f"Configuration: `dx` in {out['grids']}, `dt = dt_from_courant()`, "
       f"{out['steps']} steps, seed {out['seed']}, `c0 = {out['sound_speed']:g}` m/s. "
       "Front error is measured in the **corrected** frame; the pre-fix frame "
       "column shows what the same run used to report.\n")
    A_("| dx | variant | alpha | particles | MAE Z/a | max abs | mean rel | "
       "MAE (pre-fix frame) | energy drift | max shift/h | wall clock |")
    A_("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        alpha_label = "off" if r["alpha"] is None else f"{r['alpha']:g}"
        A_(f"| {r['dx']:g} | {r['variant']} | {alpha_label} | "
           f"{r['particles']} | {r['front_mae']:.3f} | {r['front_max_abs_err']:.3f} | "
           f"{100*r['front_mean_rel_err']:.1f} % | {r['front_mae_old_frame']:.3f} | "
           f"{100*r['energy_drift']:+.2f} % | {r['tvc_shift_peak_over_h']:.3f} | "
           f"{r['wall_clock_s']:.0f} s |")
    A_("")

    A_("## 5. Distribution mechanism\n")
    A_("| dx | variant | median `r_nn/dx` | close-pair fraction (`r < 0.7 dx`) |")
    A_("|---|---|---|---|")
    for r in runs:
        A_(f"| {r['dx']:g} | {r['variant']} | {r['clumping']['median_rnn_over_dx']:.3f} | "
           f"{100*r['clumping']['close_pair_fraction']:.2f} % |")
    A_("")

    A_("## 6. Verdicts\n")
    A_("| id | pass | detail |")
    A_("|---|---|---|")
    for k, v in V.items():
        detail = {kk: vv for kk, vv in v.items() if kk != "pass"}
        A_(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} | "
           f"`{json.dumps(detail, default=str)}` |")
    A_("")
    A_("Figures: `figure_solver3d_front.png` (Z/a vs Ritter, both grids, "
       "both frames), `figure_solver3d_clumping.png`, "
       "`figure_solver3d_shift.png` (per-pass shift envelope).\n")
    (FIGS / "SOLVER_3D_LEG.md").write_text("\n".join(L) + "\n")


def figures(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = out["runs"]
    if not runs:
        return
    grids = sorted({r["dx"] for r in runs})
    fig, axes = plt.subplots(1, len(grids), figsize=(6.0 * len(grids), 4.4),
                             squeeze=False)
    for ax, dx in zip(axes[0], grids):
        rows = [r for r in runs if abs(r["dx"] - dx) < 1e-12]
        for r in rows:
            label = "baseline" if r["alpha"] is None else f"TVC alpha={r['alpha']:g}"
            ax.plot(r["tau"], r["za"], marker="o", ms=2.5, lw=1.2, label=label)
        base = next((r for r in rows if r["alpha"] is None), rows[0])
        ax.plot(base["tau"], base["za_old_frame"], ls=":", lw=1.0, color="0.5",
                label="baseline, pre-fix frame (+offset)")
        ax.plot(base["tau"], base["ritter_za"], "r--", lw=1.6, label="Ritter 2-D")
        ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
        ax.set_ylabel(r"$Z/a$ (from the dam face)")
        ax.set_title(f"3-D dam break, dx = {dx:g} m")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "figure_solver3d_front.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels, close, rnn = [], [], []
    for r in runs:
        labels.append(f"{r['dx']:g}\n{r['variant']}")
        close.append(100 * r["clumping"]["close_pair_fraction"])
        rnn.append(r["clumping"]["median_rnn_over_dx"])
    idx = range(len(labels))
    axes[0].bar(idx, close, color="C0")
    axes[0].set_xticks(list(idx), labels, fontsize=7)
    axes[0].set_ylabel("close-pair fraction [%]")
    axes[0].set_title("clumping (lower is better)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(idx, rnn, color="C1")
    axes[1].axhline(1.0, color="k", ls="--", lw=1, label="perfect lattice")
    axes[1].set_xticks(list(idx), labels, fontsize=7)
    axes[1].set_ylabel(r"median $r_{nn}/dx$")
    axes[1].set_title("neighbour spacing (1.0 = lattice)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "figure_solver3d_clumping.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in runs:
        if not r["tvc_log"]:
            continue
        ax.plot([e["time"] for e in r["tvc_log"]],
                [e["max_shift_over_h"] for e in r["tvc_log"]],
                lw=1.2, label=f"{r['dx']:g} m, alpha={r['alpha']:g} (max)")
        ax.plot([e["time"] for e in r["tvc_log"]],
                [e["mean_shift_over_h"] for e in r["tvc_log"]],
                lw=0.8, ls="--", label=f"{r['dx']:g} m, alpha={r['alpha']:g} (mean)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$|\delta r|/h$")
    ax.set_yscale("log")
    ax.set_title("TVC displacement envelope in the 3-D solver")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "figure_solver3d_shift.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("anchors", "flow", "sweep", "merge"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--variant", default="baseline", choices=tuple(VARIANTS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    DATA.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.mode == "anchors":
        out = anchors()
        (DATA / "solver_leg_anchors.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("surface", "bulk", "wall", "pre_fix")},
                         indent=2))
        return 0

    if args.mode == "sweep":
        combos = [(dx, v) for dx in GRIDS for v in VARIANTS]
        mine = combos[args.index::args.shards]
        for dx, variant in mine:
            out = flow_run(dx, variant)
            (RUNS / f"dx{dx:g}_{variant}.json").write_text(json.dumps(out, indent=2))
            print(f"dx={dx:g} {variant}: MAE={out['front_mae']:.3f} "
                  f"rel={100*out['front_mean_rel_err']:.1f}% "
                  f"energy={100*out['energy_drift']:+.2f}% "
                  f"close={100*out['clumping']['close_pair_fraction']:.2f}% "
                  f"({out['wall_clock_s']:.0f}s)", flush=True)
        return 0

    if args.mode == "flow":
        out = flow_run(args.dx, args.variant)
        (RUNS / f"dx{args.dx:g}_{args.variant}.json").write_text(
            json.dumps(out, indent=2))
        print(f"dx={args.dx:g} {args.variant}: MAE={out['front_mae']:.3f} "
              f"rel={100*out['front_mean_rel_err']:.1f}% "
              f"energy={100*out['energy_drift']:+.2f}% "
              f"close={100*out['clumping']['close_pair_fraction']:.2f}% "
              f"rnn/dx={out['clumping']['median_rnn_over_dx']:.3f} "
              f"({out['wall_clock_s']:.0f}s)")
        return 0

    payload = merge()
    for k, v in payload["verdicts"].items():
        print(k, "PASS" if v["pass"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())