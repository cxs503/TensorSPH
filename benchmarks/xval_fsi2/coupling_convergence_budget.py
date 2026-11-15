"""Interface sub-iteration budget of the coupled FSI2 driver.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Decision
--------
The long FSI2 production run (needed to reach the periodic regime and compare
``C_d``/``C_l``/beam-tip amplitudes) is blocked by the interface sub-iteration:
``docs/xval_fsi2/TRANSIENT_VERDICT_AND_TOLERANCE.md`` recorded that 92 % of the
steps saturate the 20-iteration budget without reaching ``tol = 1e-2`` and that
the measured reduction floor is ``~5.6e-2``.  Two competing explanations:

* **H1 (accelerator defect)** -- the reduction floor is produced by the
  interface accelerator's bookkeeping, so a corrected accelerator restores
  fast convergence and the long run becomes affordable.
* **H2 (map property)** -- the floor is a property of the fluid map
  ``x -> S(F(x))`` (interface response not smooth enough at the working
  resolution), so no accelerator can drive ``||r||/||r_0||`` below it and the
  gate must be set from the measured floor instead.

This script separates them by running *the same* map with accelerators that have
nothing in common except the map itself, and then measures what the floor costs
in the reported quantities.

Arms (all on the identical snapshot set, identical ``dt``)
---------------------------------------------------------
``iqn_shipped``          IQN-ILS with history spanning the step boundary
                         (pre-2026-11 behaviour, ``span_steps=True``)
``iqn_restart``          IQN-ILS restarted at each step boundary (new default)
``iqn_restart_cold05``   as above, first iteration under-relaxed by 0.5
``aitken_restart``       Irons-Tuck dynamic relaxation, restarted per step
``fp_05``                plain fixed point with constant relaxation 0.5

Predeclared metrics
-------------------
* ``floor`` -- ``min_k ||r_k||/||r_0||`` reached within ``max_iter`` per step,
  plus the fraction of steps with ``floor > tol`` (budget saturation).
* ``n_solves`` -- fluid solves consumed per step (the cost of a policy).
* QoI sensitivity (pass B) -- secant-free trajectory comparison: for
  ``max_iter in {2, 3, 5, 8, 20, 40}`` at fixed ``tol = 1e-2``, the maximum over
  time of ``|q_m(t) - q_40(t)|`` relative to the half-range of ``q_40``, for
  ``q = tip_y`` [m] and ``q = C_d = drag / (0.5 rho U_mean^2 D)``.

Predeclared thresholds (fixed before the run, not retuned afterwards)
--------------------------------------------------------------------
* interface resolution: ``tol = 1e-2`` is **reachable** iff the median per-step
  floor ``<= 1e-2``;
* a truncated budget is **harmless** iff the QoI deviation is ``< 3 %``
  (reporting target) and **acceptable** iff ``< 15 %`` (owner standard).

Stopping rule: no threshold is changed after the results are known; every arm is
reported pass or fail.

Environment
-----------
torch CPU float64, ``dx`` from the CLI (default 0.05 m, the coarse probe
resolution), similarity-corrected FSI2 case at ``U_mean = 0.2 m/s``, ``Re = 100``,
``Ae = 5600``, ``rho_s/rho_f = 1`` (``tensorsph.fsi2_similarity``).  Pass A takes
its snapshots from a 20-step reference trajectory produced by the new default
accelerator; pass B starts from ``initial_state``.  Both passes cover the
startup transient (``t <= 100 dt``); the periodic regime (``t >~ 2.5 s``) is out
of the budget of this probe and is stated as unmeasured.

Output: ``artifacts/xval_fsi2/coupling_convergence_budget.json`` -- every number
in the report is read from this file, nothing is hand-typed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.fsi2_similarity import (  # noqa: E402
    dimensionless_groups, similarity_corrected_case,
)
from tensorsph.fsi2_strong_coupling import (  # noqa: E402
    AitkenRelaxation, IQNILS, step_fsi2_strong,
)
from tensorsph.fsi2_turek_hron import initial_state  # noqa: E402

RHO = 1000.0


class FixedRelaxation:
    """Plain fixed point ``x <- x + omega r`` (no secant, no history)."""

    def __init__(self, omega: float = 0.5) -> None:
        self.omega = float(omega)

    def start_step(self) -> None:      # protocol, nothing to reset
        return None

    def next(self, x, r):
        return x + self.omega * r, self.omega


#: reduction targets whose cost is compared across arms
REDUCTION_TARGETS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-6)


def _iters_to(rows, target: float) -> dict:
    """Median/percentile iterations to first reach ``||r_k||/||r_0|| <= target``.

    A step that never reaches the target counts as ``max_iter`` (it would have
    consumed the whole budget); ``reached`` is the fraction that does reach it.
    """
    vals, reached = [], 0
    for r in rows:
        hit = None
        for k, red in enumerate(r["residual_history"], start=1):
            if red <= target:
                hit = k
                break
        if hit is None:
            hit = r["iterations"]
        else:
            reached += 1
        vals.append(hit)
    if not vals:
        return {"median": float("inf"), "reached": 0.0}
    return {"median": float(np.median(vals)), "p90": float(np.percentile(vals, 90)),
            "reached": reached / len(rows)}


ARMS = {
    "iqn_shipped": lambda: IQNILS(span_steps=True),
    "iqn_restart": lambda: IQNILS(),
    "iqn_restart_cold05": lambda: IQNILS(cold_omega=0.5),
    "aitken_restart": lambda: AitkenRelaxation(),
    "fp_05": lambda: FixedRelaxation(0.5),
}


def make_case(dx: float, u_mean: float = 0.2, c0: float = 5.0) -> "FSI2Case":
    case = similarity_corrected_case(U_mean=u_mean)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = RHO
    case.n_beam_nodes = max(15, int(case.beam_length / dx))
    case.max_rod_substeps = 2000
    case.rod_stability_target = 0.2
    return case


def _finite(row) -> bool:
    return all(math.isfinite(row[k]) for k in ("tip_x", "tip_y", "drag",
                                               "max_speed"))


# --------------------------------------------------------------------------- #
# Pass A: per-step reduction curves on a fixed snapshot set
# --------------------------------------------------------------------------- #
def pass_a(dx: float, n_snap: int, every: int, max_iter: int) -> dict:
    case = make_case(dx)
    dt = case.dt()
    accel = IQNILS()
    state = initial_state(case)
    snaps = []
    t0 = time.time()
    for k in range(n_snap * every):
        row, _ = step_fsi2_strong(state, case, dt, accelerator=accel,
                                  max_iter=20, tol=1e-2)
        if not _finite(row):
            break
        if k % every == every - 1:
            snaps.append((copy.deepcopy(state), float(row["t"])))
    ref_wall = time.time() - t0

    out = {
        "dx": dx, "dt": dt, "beam_length": float(case.beam_length),
        "n_snapshots": len(snaps),
        "snapshot_step_interval": every, "max_iter": max_iter,
        "reference_trajectory_wall_s": round(ref_wall, 2),
        "arms": {},
    }
    for name, factory in ARMS.items():
        acc = factory()
        rows = []
        t0 = time.time()
        for snap, t in snaps:
            st = copy.deepcopy(snap)
            row, stats = step_fsi2_strong(st, case, dt, accelerator=acc,
                                          max_iter=max_iter, tol=0.0)
            hist = list(stats.residual_history)
            rows.append({
                "t": t, "iterations": int(stats.iterations),
                "floor": float(min(hist)) if hist else float("inf"),
                "residual_history": hist,
                "accepted": bool(stats.converged),
                "aborted": bool(stats.aborted),
                "finite": bool(math.isfinite(stats.residual_rel)),
                # quantity of interest of the *accepted* iterate: two
                # accelerators that reach the same fixed point must agree here
                # even though their iteration paths are unrelated.
                "tip_y": float(row["tip_y"]), "tip_x": float(row["tip_x"]),
                "drag": float(row["drag"]), "lift": float(row["lift"]),
                "load_norm": float(st. load.norm()) if st.load is not None else 0.0,
            })
        floors = np.array([r["floor"] for r in rows])
        out["arms"][name] = {
            "steps": rows,
            "median_floor": float(np.median(floors)),
            "max_floor": float(floors.max()),
            "frac_floor_gt_1e-2": float(np.mean(floors > 1e-2)),
            "frac_floor_gt_1e-1": float(np.mean(floors > 1e-1)),
            "frac_aborted": float(np.mean([r["aborted"] for r in rows])),
            "n_solves_per_step": int(np.mean([r["iterations"] for r in rows])),
            # cost that is comparable across arms: iterations needed to reach a
            # predeclared reduction.  ``n_solves_per_step`` above is measured at
            # ``tol = 0`` and therefore only shows which arms hit the machine
            # floor, not what a production gate would cost.
            "iters_to": {str(tgt): _iters_to(rows, tgt)
                         for tgt in REDUCTION_TARGETS},
            "wall_clock_s": round(time.time() - t0, 2),
        }
        if not np.all(np.isfinite(floors)):
            out["arms"][name]["median_floor"] = float("inf")
    return out


# --------------------------------------------------------------------------- #
# Pass B: does a truncated budget bias the reported quantities?
# --------------------------------------------------------------------------- #
def _drag_coefficient(drag: np.ndarray, case) -> np.ndarray:
    """``C_d = F_x / (0.5 rho U_mean^2 D)`` with the case's own ``U_mean``."""
    q = 0.5 * float(case.rho0) * case.inlet_u_mean() ** 2 * case.cylinder_d
    return drag / q


def _deviation(q: np.ndarray, ref: np.ndarray) -> dict:
    n = min(len(q), len(ref))
    if n == 0:
        return {"max_abs_dev": float("inf"), "rel_to_half_range": float("inf")}
    a, b = q[:n], ref[:n]
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return {"max_abs_dev": float("inf"), "rel_to_half_range": float("inf")}
    scale = 0.5 * float(np.max(b) - np.min(b))
    dev = float(np.max(np.abs(a - b)))
    return {"max_abs_dev": dev,
            "rel_to_half_range": dev / scale if scale > 0 else float("inf"),
            "n_compared": int(n)}


def pass_b(dx: float, n_steps: int, budgets, tol: float,
           arms=("iqn_restart", "aitken_restart")) -> dict:
    case = make_case(dx)
    dt = case.dt()
    traj = {}
    t0 = time.time()
    for name in arms:
        traj[name] = {}
        for m in budgets:
            # a fresh accelerator *and* a fresh state per budget: the arms are
            # compared on identical initial conditions, only the truncation of
            # the sub-iteration differs.
            acc = ARMS[name]()
            state = initial_state(case)
            tip_y, drag, its, reds, finite = [], [], [], [], True
            for _ in range(n_steps):
                row, stats = step_fsi2_strong(state, case, dt, accelerator=acc,
                                              max_iter=m, tol=tol)
                if not _finite(row):
                    finite = False
                    break
                tip_y.append(row["tip_y"])
                drag.append(row["drag"])
                its.append(stats.iterations)
                reds.append(stats.residual_rel)
            traj[name][m] = {"tip_y": tip_y, "drag": drag, "iterations": its,
                             "residual_rel": reds, "finite": finite,
                             "n_steps": len(tip_y)}
    # pass B is reported per arm: the recommended arm must be the one whose
    # truncation bias is measured.
    return {
        "dx": dx, "dt": dt, "n_steps": n_steps, "tol": tol,
        "u_mean": float(case.inlet_u_mean()),
        "dynamics": dimensionless_groups(case),
        "arms": {name: _pass_b_arm(name, traj, case) for name in arms},
        "wall_clock_s": round(time.time() - t0, 2),
        "trajectories": {name: {str(k): {
            "t": [dt * (i + 1) for i in range(len(v["tip_y"]))],
            "tip_y": v["tip_y"], "drag": v["drag"]}
            for k, v in arm.items()} for name, arm in traj.items()},
    }


def _pass_b_arm(name: str, traj: dict, case) -> dict:
    """Truncation bias of one arm against its own ``max_iter=40`` reference."""
    budgets = sorted(traj[name])
    ref_key = max(budgets)
    ref = traj[name][ref_key]
    ref_cd = _drag_coefficient(np.array(ref["drag"]), case)
    ref_ty = np.array(ref["tip_y"])

    rows = []
    for m in budgets:
        cd = _drag_coefficient(np.array(traj[name][m]["drag"]), case)
        ty = np.array(traj[name][m]["tip_y"])
        rows.append({
            "max_iter": int(m),
            "finite": bool(traj[name][m]["finite"]),
            "n_steps": int(traj[name][m]["n_steps"]),
            "mean_iterations": float(np.mean(traj[name][m]["iterations"]))
            if traj[name][m]["iterations"] else float("inf"),
            "frac_saturated": float(np.mean([i >= m for i in
                                             traj[name][m]["iterations"]]))
            if traj[name][m]["iterations"] else 1.0,
            "total_solves": int(np.sum(traj[name][m]["iterations"])),
            "tip_y_dev": _deviation(ty, ref_ty),
            "cd_dev": _deviation(cd, ref_cd),
            "tip_y_last": float(ty[-1]) if len(ty) else float("nan"),
            "cd_last": float(cd[-1]) if len(cd) else float("nan"),
        })
    return {"arm": name, "reference_budget": int(ref_key), "rows": rows,
            "min_budget_harmless_3pct": _min_budget(rows, 0.03),
            "min_budget_acceptable_15pct": _min_budget(rows, 0.15)}


def _min_budget(rows, thresh: float):
    """Smallest ``max_iter`` whose QoI deviation is below ``thresh``."""
    ok = [r["max_iter"] for r in rows
          if math.isfinite(r["tip_y_dev"]["rel_to_half_range"])
          and math.isfinite(r["cd_dev"]["rel_to_half_range"])
          and r["tip_y_dev"]["rel_to_half_range"] < thresh
          and r["cd_dev"]["rel_to_half_range"] < thresh]
    return int(min(ok)) if ok else None


def verdict(a: dict, b: dict) -> dict:
    """Apply the predeclared thresholds; no post-hoc retuning."""
    arms = a["arms"]
    best = min((kv for kv in arms.items()
                if math.isfinite(kv[1]["median_floor"])),
               key=lambda kv: (kv[1]["median_floor"],
                               kv[1]["iters_to"]["0.01"]["median"]),
               default=(None, {"median_floor": float("inf")}))
    shipped = arms.get("iqn_shipped", {})
    restarted_iqn = arms.get("iqn_restart", {})
    h1 = bool(math.isfinite(best[1]["median_floor"])
              and best[1]["median_floor"] <= 1.0e-2)

    # cross-arm fixed-point agreement: every arm solves the same interface
    # equation, so the accepted QoI must agree.  The arm with the smallest
    # median floor is the reference; the disagreement of the others is the
    # error of *their* interface resolution, not of the fluid map.
    agree = {}
    ref_arm = best[0]
    beam_length = float(a.get("beam_length", 0.35)) or 0.35
    if ref_arm is not None:
        ref_steps = arms[ref_arm]["steps"]
        for name, arm in arms.items():
            if name == ref_arm:
                continue
            dev, n = 0.0, 0
            for a_row, b_row in zip(arm["steps"], ref_steps):
                ty_a, ty_b = a_row["tip_y"], b_row["tip_y"]
                if not (math.isfinite(ty_a) and math.isfinite(ty_b)):
                    continue
                dev = max(dev, abs(ty_a - ty_b))
                n += 1
            agree[name] = {"max_abs_tip_y_dev": dev, "n_snapshots": n,
                           "rel_to_beam_length": dev / beam_length}

    # pass B: per-arm truncation bias of the reported quantities
    bias = {}
    for name, arm in b["arms"].items():
        ref_key = arm["reference_budget"]
        worst, worst_at = 0.0, None
        for r in arm["rows"]:
            if r["max_iter"] == ref_key:
                continue
            for key in ("tip_y_dev", "cd_dev"):
                v = r[key]["rel_to_half_range"]
                if math.isfinite(v) and v > worst:
                    worst, worst_at = v, r["max_iter"]
        bias[name] = {
            "worst_rel_dev": worst,
            "worst_at_budget": worst_at,
            "min_budget_harmless_3pct": arm["min_budget_harmless_3pct"],
            "min_budget_acceptable_15pct": arm["min_budget_acceptable_15pct"],
            "solves_at_harmless_budget": _solves_at(arm, arm["min_budget_harmless_3pct"]),
        }
    recommended = best[0]
    rec_bias = bias.get(recommended, {})
    return {
        "hypothesis_supported": "H1" if h1 else "H2",
        "best_arm": recommended,
        "best_median_floor": best[1]["median_floor"],
        "best_iters_to_1e-2": best[1]["iters_to"]["0.01"],
        "shipped_median_floor": shipped.get("median_floor"),
        "shipped_iters_to_1e-2": shipped.get("iters_to", {}).get("0.01"),
        "restart_only_median_floor": restarted_iqn.get("median_floor"),
        "tol_1e-2_reachable_by_best_arm": h1,
        "tol_1e-2_reachable_by_restart_only": bool(
            restarted_iqn.get("median_floor", float("inf")) <= 1.0e-2),
        "budget_bias": bias,
        "recommended_policy": {
            "interface_solver": recommended,
            "tol": b["tol"],
            "max_iter": rec_bias.get("min_budget_harmless_3pct"),
            "note_empty_max_iter_means_no_budget_passes_3pct": None,
            "budget_harmless_3pct": bool(
                rec_bias.get("min_budget_harmless_3pct") is not None),
            "budget_acceptable_15pct": bool(
                rec_bias.get("min_budget_acceptable_15pct") is not None),
        },
        "cross_arm_agreement": agree,
        "pass_a_arms": list(arms),
        "pass_b_arms": list(b["arms"]),
    }


def _solves_at(arm: dict, budget) -> int:
    if budget is None:
        return -1
    for r in arm["rows"]:
        if r["max_iter"] == budget:
            return int(r["total_solves"])
    return -1


def make_figures(a: dict, b: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    # Fig 1: reduction curves of the arms on the hardest snapshot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name, arm in a["arms"].items():
        steps = arm["steps"]
        if not steps:
            continue
        worst = max(steps, key=lambda r: r["floor"])
        hist = worst["residual_history"]
        ax.semilogy(range(1, len(hist) + 1), np.maximum(hist, 1e-16),
                    marker=".", label=name)
    ax.axhline(1e-2, color="k", ls="--", lw=1, label="tol = 1e-2")
    ax.set_xlabel("sub-iteration k")
    ax.set_ylabel(r"$\|r_k\|/\|r_0\|$")
    ax.set_title("worst snapshot per arm")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = list(a["arms"])
    fl = [a["arms"][n]["median_floor"] for n in names]
    sv = [a["arms"][n]["n_solves_per_step"] for n in names]
    ax2 = ax.twinx()
    x = np.arange(len(names))
    ax.bar(x - 0.2, np.maximum(fl, 1e-16), width=0.4, label="median floor")
    ax2.bar(x + 0.2, sv, width=0.4, color="tab:orange",
            label="solves / step")
    ax.set_yscale("log")
    ax.axhline(1e-2, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("median reduction floor")
    ax2.set_ylabel("fluid solves / step")
    ax.set_title("floor and cost per arm")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = outdir / "coupling_budget_pass_a.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Left: interface residual reduction per "
                            "sub-iteration on the worst snapshot of each arm. "
                            "Right: median reduction floor and fluid solves per "
                            "step per arm (log scale; dashed line = tol 1e-2)."})

    # Fig 2: QoI trajectories vs budget, one panel per pass-B arm
    arm_names = list(b["trajectories"])
    fig, axes = plt.subplots(len(arm_names), 2,
                            figsize=(11, 3.6 * len(arm_names)), squeeze=False)
    for i, arm_name in enumerate(arm_names):
        for j, (key, lab) in enumerate((("tip_y", "beam tip $y$ [m]"),
                                        ("drag", "beam drag [N/m]"))):
            ax = axes[i][j]
            for mb, tr in b["trajectories"][arm_name].items():
                ax.plot(tr["t"], tr[key], lw=1.0, label=f"max_iter={mb}")
            ax.set_xlabel("t [s]")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
            if j == 0:
                ax.set_title(f"{arm_name}: QoI vs sub-iteration budget")
    fig.tight_layout()
    p = outdir / "coupling_budget_pass_b.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Beam-tip displacement (left) and beam drag (right) "
                            "time histories for different sub-iteration budgets "
                            "at fixed tol = 1e-2: the truncation bias of the "
                            "reported quantities."})
    return figs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--n-snap", type=int, default=12)
    ap.add_argument("--snap-every", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=80)
    ap.add_argument("--budgets", default="2,3,5,8,20,40")
    ap.add_argument("--passb-arms", default="iqn_restart,aitken_restart")
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "xval_fsi2"
                                         / "coupling_convergence_budget.json"))
    args = ap.parse_args()

    budgets = [int(v) for v in args.budgets.split(",")]
    print(f"pass A: dx={args.dx} snapshots={args.n_snap}x{args.snap_every} "
          f"max_iter={args.max_iter}", flush=True)
    a = pass_a(args.dx, args.n_snap, args.snap_every, args.max_iter)
    for name, arm in a["arms"].items():
        it2 = arm["iters_to"]["0.01"]
        print(f"  {name:22s} median_floor={arm['median_floor']:.3e} "
              f"max_floor={arm['max_floor']:.3e} "
              f"sat>1e-2={arm['frac_floor_gt_1e-2']*100:.0f}% "
              f"iters_to_1e-2={it2['median']:.1f}"
              f"(reached {it2['reached']*100:.0f}%) "
              f"wall={arm['wall_clock_s']}s", flush=True)

    print(f"pass B: dx={args.dx} n_steps={args.n_steps} budgets={budgets}",
          flush=True)
    b = pass_b(args.dx, args.n_steps, budgets, args.tol,
               arms=tuple(args.passb_arms.split(",")))
    for name, arm in b["arms"].items():
        print(f"  [{name}]", flush=True)
        for r in arm["rows"]:
            print(f"    max_iter={r['max_iter']:3d} "
                  f"mean_it={r['mean_iterations']:.2f} "
                  f"saturated={r['frac_saturated']*100:.0f}% "
                  f"tipy_dev={r['tip_y_dev']['rel_to_half_range']*100:.2f}% "
                  f"cd_dev={r['cd_dev']['rel_to_half_range']*100:.2f}%",
                  flush=True)

    v = verdict(a, b)
    print("verdict:", json.dumps(v, indent=2), flush=True)

    payload = {
        "decision": ("Is the shipped interface sub-iteration gate (tol=1e-2, "
                     "max_iter=20) reachable, and does failing it bias the "
                     "reported FSI2 quantities?"),
        "thresholds_predeclared": {
            "tol_reachable_if_median_floor_le": 1e-2,
            "budget_harmless_if_rel_dev_lt": 0.03,
            "budget_acceptable_if_rel_dev_lt": 0.15,
        },
        "environment": {
            "case": "similarity-corrected FSI2 (U_mean=0.2 m/s, Re=100, "
                    "Ae=5600, rho_s/rho_f=1)",
            "dx": args.dx, "tol_pass_b": args.tol,
            "max_iter_pass_a": args.max_iter,
        },
        "pass_a": a, "pass_b": b, "verdict": v,
        "figures": [],
    }
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    if args.figures:
        try:
            figs = make_figures(a, b, outdir)
            payload["figures"] = [f["name"] for f in figs]
            (outdir / "coupling_budget_figures.json").write_text(
                json.dumps(figs, indent=2))
        except Exception as exc:            # pragma: no cover - optional
            payload["figure_error"] = repr(exc)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())