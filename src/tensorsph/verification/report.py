"""Figures and the report document, built from the evidence store.

Two rules:

* **no hand-typed numbers**: every value that appears in the report is fetched
  from the store through :meth:`EvidenceStore.digest`, which returns the value
  together with the SQL that produced it.  The provenance block of the report is
  that list.
* **no hand-made figures**: every figure is drawn from a ``series`` payload that
  was written by a run, so a figure cannot exist without the data it plots.

Figures are written as PNG next to the report; the report document itself is
generated markdown, and the PNGs are collected into ``figures/``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .evidence import EvidenceStore

CASE_M1 = "dam_break_2d"
CASE_M1_CONTROL = "dam_break_2d_gravity_off"
CASE_M2 = "hydrostatic_pressure_profile"
CASE_M2_CONTROL = "hydrostatic_zero_density_control"
CASE_M3 = "mass_conservation_dam_break"
CASE_M3_OPEN = "mass_conservation_dam_break_open"
CASE_M4 = "kernel_interpolation_convergence"
CASE_M4_CONTROL = "hydrostatic_truncated_support"

#: Metric names that the report quotes, per case.  The provenance block is built
#: from this list, so a metric that is not listed is not reported.
REPORTED_METRICS = {
    CASE_M1: [
        "front_max_rel_error", "front_rms_rel_error", "front_n_points",
        "height_residual_max_rel_error", "height_ballistic_max_rel_error",
        "ritter_ratio_mean", "mass_drift_rel", "max_mach",
        "max_density_compression", "particles", "steps",
        "gate_front_pass", "gate_height_pass",
    ],
    CASE_M1_CONTROL: ["front_drift_over_dx", "height_drift_rel",
                      "max_particle_displacement_over_dx", "gate_pass"],
    CASE_M2: ["max_band_error_pct", "mean_band_error_pct", "n_particles",
              "gate_pass"],
    CASE_M2_CONTROL: ["eos_at_zero_density", "gate_pass"],
    CASE_M3: ["relative_loss_accounted", "relative_loss_at_horizon",
              "steps_run", "gate_pass"],
    CASE_M3_OPEN: ["relative_loss_accounted", "relative_loss_unaccounted",
                   "particles_removed", "gate_pass"],
    CASE_M4: ["min_order_normalized", "mean_order_normalized",
              "defect_ratio", "gate_pass"],
    CASE_M4_CONTROL: ["max_band_error_pct", "diverged", "gate_pass"],
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _latest_run(store: EvidenceStore, case: str) -> int:
    rows = store.query("SELECT run_id FROM runs WHERE case_name=? ORDER BY run_id DESC"
                       " LIMIT 1", (case,))
    if not rows:
        raise KeyError(f"no run recorded for case {case!r}")
    return int(rows[0]["run_id"])


def collect(store: EvidenceStore) -> dict:
    """Every reported number, with its SQL."""
    out = {"cases": {}, "provenance": []}
    for case, names in REPORTED_METRICS.items():
        run_id = _latest_run(store, case)
        numbers = {}
        for name in names:
            try:
                d = store.digest(run_id, name)
            except KeyError:
                continue
            numbers[name] = d["value"]
            out["provenance"].append({"case": case, "metric": name,
                                      "run_id": run_id, "sql": d["sql"],
                                      "params": d["params"], "value": d["value"],
                                      "verdict": d["verdict"]})
        out["cases"][case] = {"run_id": run_id, "metrics": numbers}
    out["gates"] = {case: _case_gates(store, case) for case in REPORTED_METRICS}
    return out


def _case_gates(store: EvidenceStore, case: str) -> dict:
    run_id = _latest_run(store, case)
    rows = store.query(
        "SELECT name, value, gate_limit, verdict FROM metrics WHERE run_id=?"
        " AND gate_limit IS NOT NULL ORDER BY name", (run_id,))
    return {r["name"]: {"value": r["value"], "limit": r["gate_limit"],
                        "verdict": r["verdict"]} for r in rows}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figure_dam_break(store: EvidenceStore, run_id: int, out_png: Path,
                     table: dict) -> Path:
    plt = _mpl()
    series = json.loads(store.query(
        "SELECT payload FROM series WHERE run_id=? AND name='front'",
        (run_id,))[0]["payload"])
    profiles = json.loads(store.query(
        "SELECT payload FROM series WHERE run_id=? AND name='profiles'",
        (run_id,))[0]["payload"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    taus = [p["tau"] for p in table["data"]]
    za = [p["Z_over_a"] for p in table["data"]]
    unc = [p.get("uncertainty") or 0.0 for p in table["data"]]
    ax.errorbar(taus, za, yerr=unc, fmt="ko", ms=4, capsize=2,
                label="Martin & Moyce (1952) [grade WITHHELD]")
    ax.plot(series["tau"], series["ritter"], "r--", lw=1.2,
            label=r"Ritter shallow water, $Z/a=2\tau$")
    for curve in series["fronts"]:
        ax.plot(curve["tau"], curve["Z_over_a"], "-", lw=1.8,
                label=rf"TensorSPH $\Delta x$={curve['dx']:g} m (N={curve['particles']})")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax.set_ylabel(r"$Z/a$")
    ax.set_title("Dam-break front trajectory")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for prof in profiles:
        xs = [(i + 0.5) * prof["bin"] / prof["a"] for i in range(len(prof["h"]))]
        ys = [1.0 if v is None else v / prof["a"] for v in prof["h"]]
        ax.plot(xs, ys, lw=1.4, label=rf"$\tau$={prof['tau']:.1f}")
    ax.set_xlabel(r"$x/a$")
    ax.set_ylabel(r"$h/a$")
    ax.set_title("Free-surface profiles (SPH)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_hydrostatic(store: EvidenceStore, run_id: int, out_png: Path) -> Path:
    plt = _mpl()
    bands = json.loads(store.query(
        "SELECT payload FROM series WHERE run_id=? AND name='bands'",
        (run_id,))[0]["payload"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    y = [b["y"] for b in bands]
    p = [b["pressure"] for b in bands]
    pr = [b["pressure_analytic"] for b in bands]
    ax.plot(p, y, "o-", label="SPH (time mean)")
    ax.plot(pr, y, "k--", label=r"$\rho g (H-z)$")
    ax.set_xlabel("pressure [Pa]")
    ax.set_ylabel("z [m]")
    ax.set_title("Hydrostatic pressure profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    err = [b["rel_error_pct"] for b in bands]
    colours = ["tab:red" if b["free_surface"] else "tab:blue" for b in bands]
    ax.barh(y, err, color=colours, height=0.04)
    ax.axvline(2.0, color="k", ls=":", label="2 % gate")
    ax.set_xlabel("relative error [%]")
    ax.set_ylabel("z [m]")
    ax.set_title("Band error (red = free-surface band, not gated)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_convergence(store: EvidenceStore, run_id: int, out_png: Path,
                       control_run_id: int | None = None) -> Path:
    plt = _mpl()
    ladder = json.loads(store.query(
        "SELECT payload FROM series WHERE run_id=? AND name='ladder'",
        (run_id,))[0]["payload"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    h = [row["h"] for row in ladder]
    for key, style, label in (
            ("l2_error_normalized", "o-", "Shepard-normalised (gated)"),
            ("l2_error_raw", "s--", "raw summation"),
            ("l2_error_truncated_support", "^:", "truncated support $R=1.5h$")):
        ax.loglog(h, [row[key] for row in ladder], style, label=label)
    ref = ladder[0]["l2_error_normalized"]
    ax.loglog([h[0], h[-1]],
              [ref, ref * (h[-1] / h[0]) ** 2], "k:", lw=1.0, label=r"slope 2")
    ax.set_xlabel("$h$ [m]")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_title("SPH interpolation convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_mass(store: EvidenceStore, closed_id: int, open_id: int,
                out_png: Path) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for run_id, style, label in ((closed_id, "o-", "closed tank (accounted)"),
                                 (open_id, "s-", "open tank, outflow accounted"),
                                 (open_id, "^--", "open tank, outflow forgotten")):
        hist = json.loads(store.query(
            "SELECT payload FROM series WHERE run_id=? AND name='mass_history'",
            (run_id,))[0]["payload"])
        steps = [r["step"] for r in hist]
        if label.endswith("forgotten"):
            mass = [r["mass_inside"] for r in hist]
        elif "open" in label:
            mass = [r["mass_accounted"] for r in hist]
        else:
            mass = [r["mass_accounted"] for r in hist]
        m0 = mass[0]
        ax.plot(steps, [m / m0 for m in mass], style, label=label, ms=3)
    ax.set_xlabel("step")
    ax.set_ylabel("$M(t)/M(0)$")
    ax.set_title("Mass bookkeeping")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #

def table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def summary_table(collected: dict) -> str:
    rows = []
    for case, entry in collected["cases"].items():
        for name, value in entry["metrics"].items():
            gate = collected["gates"].get(case, {}).get(name)
            rows.append([case, name, value,
                         None if gate is None else gate["limit"],
                         None if gate is None else gate["verdict"]])
    return table(rows, ["case", "metric", "value", "limit", "verdict"])


def provenance_block(collected: dict) -> str:
    rows = [[p["case"], p["metric"], p["run_id"], p["value"], p["sql"]]
            for p in collected["provenance"]]
    return table(rows, ["case", "metric", "run", "value", "SQL"])


def write_json(collected: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collected, indent=1, sort_keys=True))
    return path
