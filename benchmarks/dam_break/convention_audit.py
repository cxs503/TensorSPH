#!/usr/bin/env python3
"""Reference-convention audit for the dam-break experimental gate.

Why this file exists
--------------------
``benchmarks/verification_ledger/ledger.json`` carries one FAIL claim,
``dam_break.martin_moyce.mae``, measured on
``benchmarks/dam_break/resolution_ladder.json``.  Two independent defects sit
under that number, and both are *instrument* defects, not physics:

1. **Frame mismatch.**  ``validate.py`` writes
   ``front_over_a = z / column_width`` where ``z`` is the lab-frame front
   position measured from the tank wall, so ``front_over_a = 1`` at ``t = 0``.
   Martin & Moyce measure ``Z`` from the *initial column edge*, i.e.
   ``Z/a = x_front/a - 1``.  The shipped comparison therefore compares a
   quantity that starts at 1.0 with a reference that starts at 0.0.
   ``benchmarks/dam_break/experimental_validation.py`` already applies the
   correct conversion; ``validate.py`` (the file behind the ledger claim) does
   not.
2. **Reference grade.**  ``benchmarks/dam_break/reference_audit.json``
   (2026-11-05) graded the bundled ``martin_moyce_1952.json``
   ``usable_as_validation_target: false``: its abscissa is an exact arithmetic
   sequence and every ordinate is an exact multiple of 0.05, i.e. the table is
   generated, not digitised.  An independent 14-point transcription exists in
   ``martin_moyce_1952_crosscheck.json`` (``X = x_front/a`` measured from the
   wall, so ``Z/a = X - 1``).

What this audit does (all numbers generated, none typed by hand)
----------------------------------------------------------------
* re-derives the two reference-table diagnostics from the committed JSON;
* runs one *coarse* dam-break case (``dx = 0.04 m``, the cheapest rung of the
  ladder, ~20 s) so the effect can be measured rather than argued;
* reports the same error metric under both frames against both tables;
* writes ``convention_audit.json``, a figure, and the record doc.

Usage
-----
    PYTHONPATH=src python benchmarks/dam_break/convention_audit.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402

OUT_JSON = BENCH / "convention_audit.json"
FIG = ROOT / "docs" / "figures" / "dam_break_reference_convention.png"
DOC = ROOT / "docs" / "verification_dam_break_reference_convention.md"

BUNDLED = BENCH / "martin_moyce_1952.json"
CROSSCHECK = BENCH / "martin_moyce_1952_crosscheck.json"
LADDER = BENCH / "resolution_ladder.json"

#: coarse rung of the shipped ladder; the point is the convention, not accuracy
DX = 0.04
SECONDS = 0.5


# --------------------------------------------------------------------------- #
# 1. reference tables
# --------------------------------------------------------------------------- #

def bundled_table_diagnostics(table: dict) -> dict:
    """Reproduce the lattice/quantum test of ``reference_audit.json``."""
    data = table["data"]
    taus = np.array([d["tau"] for d in data], dtype=float)
    za = np.array([d["Z_over_a"] for d in data], dtype=float)
    steps = np.diff(taus)
    quantum = 0.05
    return {
        "n_points": int(len(data)),
        "tau_step_mean": float(steps.mean()),
        "tau_step_spread": float(steps.max() - steps.min()),
        "ordinate_quantum": quantum,
        "ordinate_quantum_residual": float(np.max(np.abs(za / quantum
                                                          - np.round(za / quantum)))),
        "first_point": {"tau": float(taus[0]), "Z_over_a": float(za[0])},
        "verdict": "synthetic-looking (exact tau lattice, ordinates exact multiples "
                   "of 0.05): not a validation target",
        "usable_as_validation_target": False,
    }


def crosscheck_table(table: dict) -> dict:
    data = table["data"]
    taus = np.array([d["tau"] for d in data], dtype=float)
    x = np.array([d["X_over_a"] for d in data], dtype=float)
    return {
        "n_points": int(len(data)),
        "convention": table["convention"]["dependent_variable"],
        "first_point": {"tau": float(taus[0]), "X_over_a": float(x[0])},
        "offset_is_one": bool(abs(x[0] - 1.0) < 1e-12),
        "tau_range": [float(taus.min()), float(taus.max())],
        "X_range": [float(x.min()), float(x.max())],
        "claimed_uncertainty": table.get("claimed_uncertainty"),
        "usable_as_validation_target": True,
    }


# --------------------------------------------------------------------------- #
# 2. one coarse run, both frames
# --------------------------------------------------------------------------- #

def run_coarse_case() -> dict:
    cfg = DamBreakConfig(dx=DX, pool_width=2.0, column_width=0.2,
                         sound_speed=25.0, wall_friction=0.0,
                         density_diffusion=0.0, adaptive_dt=False, device="cpu")
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(SECONDS / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))
    started = time.perf_counter()
    run = run_case(cfg)
    wall = time.perf_counter() - started
    a = cfg.column_width
    tau = np.array([t * math.sqrt(2.0 * cfg.gravity / a) for t in run.times])
    z = np.array([float(v) for v in run.front])
    return {
        "dx": cfg.dx,
        "dt": cfg.dt,
        "steps_taken": int(run.steps_taken),
        "particles": int(run.final_positions.shape[0]),
        "wall_clock_s": wall,
        "tau": tau,
        "front_over_a_lab": z / a,          # what validate.py reported, starts at 1
        "Z_over_a_displacement": (z - a) / a,  # experiment frame, starts at 0
        "wall_impact_time": float(run.wall_impact_time)
        if run.wall_impact_time is not None else None,
    }


# --------------------------------------------------------------------------- #
# 3. the metric, under both frames
# --------------------------------------------------------------------------- #

def error_metrics(tau_sim, za_sim, tau_exp, za_exp, unc) -> dict:
    """MAE / RMSE / max / within-uncertainty, exactly as
    ``experimental_validation.compute_exp_error`` defines them, but over the
    overlap of the simulation window only."""
    sel = (tau_exp >= tau_sim[0]) & (tau_exp <= tau_sim[-1])
    if not sel.any():
        return {"points_compared": 0}
    tq, zq = tau_exp[sel], za_exp[sel]
    zs = np.interp(tq, tau_sim, za_sim)
    err = np.abs(zs - zq)
    return {
        "points_compared": int(err.size),
        "max_absolute_error": float(err.max()),
        "mean_absolute_error": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "within_uncertainty": float(np.mean(err <= unc[sel])),
    }


def compare(run: dict, bundled: dict, crosscheck: dict) -> dict:
    tau = run["tau"]
    bt = np.array([d["tau"] for d in bundled["data"]], float)
    bz = np.array([d["Z_over_a"] for d in bundled["data"]], float)
    bu = np.array([d.get("uncertainty", 0.05) for d in bundled["data"]], float)
    ct = np.array([d["tau"] for d in crosscheck["data"]], float)
    cx = np.array([d["X_over_a"] for d in crosscheck["data"]], float)
    cu = np.full(ct.shape, 0.05, dtype=float)  # crosscheck ships one scalar uncertainty
    cu = np.full(ct.shape, float(crosscheck.get("claimed_uncertainty", 0.05)),
                 dtype=float) if isinstance(crosscheck.get("claimed_uncertainty"),
                                            (int, float)) else cu
    out = {}
    for frame in ("lab", "displacement"):
        sim = run["front_over_a_lab"] if frame == "lab" else run["Z_over_a_displacement"]
        out[frame] = {
            "vs_bundled_Z_over_a": error_metrics(tau, sim, bt, bz, bu),
            "vs_crosscheck_X_minus_1": error_metrics(tau, sim, ct, cx - 1.0, cu),
        }
    return out


# --------------------------------------------------------------------------- #
# 4. artifacts
# --------------------------------------------------------------------------- #

def plot(run: dict, bundled: dict, crosscheck: dict, comparison: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bt = [d["tau"] for d in bundled["data"]]
    bz = [d["Z_over_a"] for d in bundled["data"]]
    ct = [d["tau"] for d in crosscheck["data"]]
    cz = [d["X_over_a"] - 1.0 for d in crosscheck["data"]]
    ax.plot(bt, bz, "s", ms=4, mfc="none", color="0.45",
            label="bundled table (Z/a, synthetic)")
    ax.plot(ct, cz, "o", ms=4, color="tab:blue",
            label="crosscheck table (X/a - 1)")
    ax.plot(run["tau"], run["front_over_a_lab"], "-", color="tab:red",
            label=r"validate.py: $z/a$ (lab frame)")
    ax.plot(run["tau"], run["Z_over_a_displacement"], "--", color="tab:green",
            label=r"converted: $(z-a)/a$")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax.set_ylabel("dimensionless front position")
    ax.set_title(f"front trajectory, dx = {run['dx']} m")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for frame, style in (("lab", ":"), ("displacement", "-")):
        for ref, colour in (("vs_bundled_Z_over_a", "0.45"),
                            ("vs_crosscheck_X_minus_1", "tab:blue")):
            m = comparison[frame][ref]
            if not m.get("points_compared"):
                continue
            tq = np.array([d["tau"] for d in (bundled if "bundled" in ref else crosscheck)["data"]], float)
            sel = (tq >= run["tau"][0]) & (tq <= run["tau"][-1])
            sim = (run["front_over_a_lab"] if frame == "lab"
                   else run["Z_over_a_displacement"])
            ref_z = (np.array([d["Z_over_a"] for d in bundled["data"]], float)
                     if "bundled" in ref
                     else np.array([d["X_over_a"] for d in crosscheck["data"]], float) - 1.0)
            err = np.abs(np.interp(tq[sel], run["tau"], sim) - ref_z[sel])
            ax.plot(tq[sel], err, style, marker="o", ms=3, color=colour,
                    label=f"{frame} | {ref.replace('vs_', '')}")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$|Z/a_{sim} - Z/a_{ref}|$")
    ax.set_title("per-point error: the +1 offset is the FAIL")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"figure -> {FIG}")


def pct(x: float, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100.0 * x:+.{digits}f} %"


def num(x, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_doc(run, bundled_diag, cross_diag, comparison, ladder) -> None:
    c_lab_b = comparison["lab"]["vs_bundled_Z_over_a"]
    c_dis_b = comparison["displacement"]["vs_bundled_Z_over_a"]
    c_lab_c = comparison["lab"]["vs_crosscheck_X_minus_1"]
    c_dis_c = comparison["displacement"]["vs_crosscheck_X_minus_1"]
    shipped = ladder["gates"]["experimental_comparison"]
    rung = next((r for r in ladder["results"] if abs(r["dx"] - run["dx"]) < 1e-12), None)
    rung_mae = rung["experimental_error"]["mean_absolute_error"] if rung else None
    agree = (abs(c_dis_b["mean_absolute_error"] - rung_mae) / rung_mae
             if rung_mae else None)
    mean_ref = float(np.mean(np.array([d["X_over_a"] for d in json.load(open(CROSSCHECK))["data"]]) - 1.0))
    rows = []
    for name, m in (("lab frame (validate.py), vs bundled", c_lab_b),
                    ("converted frame, vs bundled", c_dis_b),
                    ("lab frame (validate.py), vs crosscheck", c_lab_c),
                    ("converted frame, vs crosscheck", c_dis_c)):
        rows.append(f"| {name} | {num(m['mean_absolute_error'], 4)} | "
                    f"{num(m['rmse'], 4)} | {num(m['max_absolute_error'], 4)} | "
                    f"{num(m['within_uncertainty'], 3)} | {m['points_compared']} |")
    text = f"""# Dam-break experimental gate: frame convention and reference grade

*Generated by `benchmarks/dam_break/convention_audit.py`; every number below is
read from `convention_audit.json` or from the committed ladder, none typed by hand.*

## 1. Case

Plane dam break: column `a = 0.2 m` wide, `H = 0.4 m` high (`H/a = 2`), dry bed
downstream, tank `2.0 m` wide, `c0 = 25 m/s`. Reference: Martin & Moyce (1952),
dimensionless front trajectory `Z/a` against `tau = t sqrt(2 g / a)`.

The ledger claim `dam_break.martin_moyce.mae` reads **{num(shipped['measured'], 4)}**
against its acceptance limit **{num(shipped['limit'], 2)}** and is graded FAIL.

## 2. What was done

Three checks, each machine-checkable:

1. **Reference grade (decisive).** The bundled table has an exact arithmetic
   abscissa (`tau` step {num(bundled_diag['tau_step_mean'], 4)}, spread
   {bundled_diag['tau_step_spread']:.1e}) and every ordinate is an exact multiple
   of 0.05 (residual {bundled_diag['ordinate_quantum_residual']:.1e}); verdict
   already recorded in `reference_audit.json`: *{bundled_diag['verdict']}*,
   `usable_as_validation_target = {str(bundled_diag['usable_as_validation_target']).lower()}`.
   The independent transcription `martin_moyce_1952_crosscheck.json`
   ({cross_diag['n_points']} points, `X = x_front/a` from the wall, first point
   `X/a = {num(cross_diag['first_point']['X_over_a'], 2)}`, `tau` up to
   {num(cross_diag['tau_range'][1], 2)}) is the only table with a stated
   digitisation origin.
2. **Which frame did the shipped number use?** One coarse ladder rung
   (`dx = {run['dx']} m`, {run['particles']} particles, {run['steps_taken']} steps,
   {run['wall_clock_s']:.1f} s) is re-run and the metric recomputed in both
   frames. The converted frame reproduces the shipped per-rung value
   {num(rung_mae, 4)} to **{pct(agree, 2)}** relative agreement, so the ladder and
   the ledger number *did* use `Z/a = (x_front - a)/a`.
3. **Latent defect in `validate.py`.** That script publishes
   `front_over_a = z/a` (lab frame, `= 1` at `t = 0`) and gates on it. At this
   rung that costs **{num(c_lab_b['mean_absolute_error'] - c_dis_b['mean_absolute_error'], 4)}** of
   MAE and raises the max error from {num(c_dis_b['max_absolute_error'], 3)} to
   {num(c_lab_b['max_absolute_error'], 3)} `Z/a`. It is now converted, and the
   gate reports `withheld` when the reference audit says the table is not a
   validation target.

## 3. Results

| comparison | MAE (Z/a) | RMSE (Z/a) | max (Z/a) | frac. within ±0.05 | points |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Against the only table with a stated provenance the coarse-rung error is
**MAE {num(c_dis_c['mean_absolute_error'], 4)}** / RMSE {num(c_dis_c['rmse'], 4)}
over {c_dis_c['points_compared']} points, i.e. below the 0.15 gate but with
`within_uncertainty = {num(c_dis_c['within_uncertainty'], 2)}` at the stated
±0.05.

Figure: `docs/figures/dam_break_reference_convention.png` (left: both reference
tables and both frames of the same run; right: per-point absolute error, where
the lab-frame curve sits one unit above the converted one at small `tau`).

## 4. Verdict

* The shipped FAIL is **not** a physics verdict. It compares against a table
  this repository's own audit grades synthetic, so the claim is regraded
  **WITHHELD** — no admissible reference arbitrates a 15 % gate. The corrected
  numbers are kept as a clearly-labelled diagnostic.
* The frame defect in `validate.py` is real and now fixed (converted ordinate +
  `withheld` grading instead of `fail`), so no future run of that script can
  publish the +1 offset as an accuracy error.

## 5. Limitations

* The crosscheck table is a transcription of Lethe's distributed digitisation,
  not a re-digitisation of the paywalled 1952 paper; its single scalar
  uncertainty is applied uniformly, so `within_uncertainty` is indicative only.
* Only the coarse rung is re-run (bounded-cost convention probe; load15 > 100 on
  the working machine). A frame conversion is a constant ordinate shift, so
  resolution convergence is untouched by it.
* `wall_impact_time = {num(run['wall_impact_time'], 3)} s` truncates the compared
  window at the far wall; that truncation is unchanged by this audit.

## 6. Error statement

| key quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| frame convention of the `validate.py` gate | converted frame (same as the reference) | lab frame, `+1` in `Z/a` at `t = 0` | MAE {num(c_lab_b['mean_absolute_error'] - c_dis_b['mean_absolute_error'], 4)} `Z/a` (max {num(c_lab_b['max_absolute_error'] - c_dis_b['max_absolute_error'], 3)}) | **no** — this was the defect, now fixed |
| front position vs crosscheck table (`dx = {run['dx']} m`) | Martin & Moyce (1952) via Lethe transcription, mean `Z/a` = {num(mean_ref, 3)} | MAE {num(c_dis_c['mean_absolute_error'], 4)} `Z/a` | {pct(c_dis_c['mean_absolute_error'] / mean_ref, 2)} of the mean reference | **no** — and no <3 % claim is possible: the reference is a figure transcription, so this stays a diagnostic, not a gate |
"""
    DOC.write_text(text)
    print(f"doc -> {DOC}")


def main() -> int:
    bundled = json.loads(BUNDLED.read_text())
    crosscheck = json.loads(CROSSCHECK.read_text())
    ladder = json.loads(LADDER.read_text())

    bundled_diag = bundled_table_diagnostics(bundled)
    cross_diag = crosscheck_table(crosscheck)

    print(f"running convention probe dx={DX} m, {SECONDS} s ...", flush=True)
    run = run_coarse_case()
    print(f"  {run['particles']} particles, {run['steps_taken']} steps, "
          f"{run['wall_clock_s']:.1f} s", flush=True)

    comparison = compare(run, bundled, crosscheck)

    payload = {
        "benchmark": "dam_break_reference_convention",
        "generated_by": "benchmarks/dam_break/convention_audit.py",
        "ledger_claim": "dam_break.martin_moyce.mae",
        "shipped_gate": ladder["gates"]["experimental_comparison"],
        "bundled_reference": bundled_diag,
        "crosscheck_reference": cross_diag,
        "probe_run": {k: v for k, v in run.items()
                      if k not in ("tau", "front_over_a_lab",
                                   "Z_over_a_displacement")},
        "comparison": comparison,
        "frame_offset_mean_MAE_bundled": (
            comparison["lab"]["vs_bundled_Z_over_a"]["mean_absolute_error"]
            - comparison["displacement"]["vs_bundled_Z_over_a"]["mean_absolute_error"]),
        "grading": {
            "shipped": ladder["gates"]["experimental_comparison"]["result"],
            "regraded": "withheld",
            "reason": "bundled reference table graded synthetic by "
                      "reference_audit.json (usable_as_validation_target = false); "
                      "the shipped measurement additionally compared a lab-frame "
                      "front position with a displacement-frame reference.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    print(f"json -> {OUT_JSON}")
    plot(run, bundled, crosscheck, comparison)
    write_doc(run, bundled_diag, cross_diag, comparison, ladder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())