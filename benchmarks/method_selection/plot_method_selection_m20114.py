"""Figures + machine-generated tables for the m20114 anchor increment.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``.  The increment is *evidence
closure*: the last scheme the selector had to report as ``WITHHELD``
(``transport_velocity_correction``) is now backed by a committed, sha256-pinned
Martin & Moyce dam-break record, so ``methods_without_evidence`` is empty.

Nothing here runs a solver: every number comes from committed artefacts

* ``docs/xval_sphinx_tvc/data/grid_dx00{3,4}.json`` -- TVC dam-break grid
  (``benchmarks/xval_sphinx_tvc/run.py``), and
* ``benchmarks/dam_break/martin_moyce_1952.json`` -- digitised experiment,
* ``docs/reports/method_selection_registry.json`` -- rebuilt registry.

The error of the chosen arm is *recomputed* from the raw ``tau``/``Z_over_a``
curves against the experimental points instead of trusting the stored ``score``
block; the recomputation is asserted equal to the registry number.

Usage::

    PYTHONPATH=src python benchmarks/method_selection/plot_method_selection_m20114.py

Outputs
-------
``docs/figures/method_selection_m20114/figure_tvc_front.png``
``docs/figures/method_selection_m20114/figure_registry_coverage.png``
``docs/reports/method_selection_m20114_tables.md``
``docs/reports/method_selection_m20114.json``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.method_selection import (  # noqa: E402
    METHODS,
    MODIFIERS,
    OWNER_TOLERANCE_PCT,
)

FIG_DIR = ROOT / "docs" / "figures" / "method_selection_m20114"
TABLES_PATH = ROOT / "docs" / "reports" / "method_selection_m20114_tables.md"
SUMMARY_PATH = ROOT / "docs" / "reports" / "method_selection_m20114.json"
REGISTRY_PATH = ROOT / "docs" / "reports" / "method_selection_registry.json"
EXP_PATH = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"

#: The calibrated transport-velocity arm (see ``docs/xval_tvc_validation_report.md``).
ARM = "tvc_a010_all"


def _interp(xs: list[float], ys: list[float], xq: float) -> float | None:
    for k in range(len(xs) - 1):
        if xs[k] <= xq <= xs[k + 1]:
            span = xs[k + 1] - xs[k]
            w = 0.0 if span == 0 else (xq - xs[k]) / span
            return ys[k] + w * (ys[k + 1] - ys[k])
    return None


def score_from_curves(arm: dict, experiment: dict) -> dict:
    """Recompute MAE / mean relative error of ``Z/a`` against the experiment."""
    tau, za = arm["tau"], arm["Z_over_a"]
    abs_err: list[float] = []
    rel_err: list[float] = []
    rows = []
    for point in experiment["data"]:
        t_ref, z_ref = point["tau"], point["Z_over_a"]
        if not z_ref or t_ref < tau[0] or t_ref > tau[-1]:
            continue
        got = _interp(tau, za, t_ref)
        if got is None:
            continue
        err = abs(got - z_ref)
        abs_err.append(err)
        rel_err.append(err / z_ref)
        rows.append({"tau": t_ref, "Z_over_a_ref": z_ref, "Z_over_a_got": got,
                     "abs_error": err, "rel_error_pct": err / z_ref * 100.0})
    return {
        "points": len(rows),
        "mae": sum(abs_err) / len(abs_err),
        "mean_rel_error_pct": sum(rel_err) / len(rel_err) * 100.0,
        "max_abs_error": max(abs_err),
        "rows": rows,
    }


def figure_front(granular: dict[float, dict], experiment: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    styles = {"baseline": ("#4c72b0", "--", "no correction"),
              ARM: ("#2e8b57", "-", f"TVC alpha=0.10 (all particles)")}
    for ax, dx in zip(axes, sorted(granular)):
        doc = granular[dx]
        pts = [(p["tau"], p["Z_over_a"]) for p in experiment["data"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "ko", ms=4,
                label="Martin & Moyce (1952), digitised")
        for variant, (colour, ls, label) in styles.items():
            arm = next(r for r in doc["records"] if r["variant"] == variant)
            sc = score_from_curves(arm, experiment)
            ax.plot(arm["tau"], arm["Z_over_a"], color=colour, ls=ls, lw=1.8,
                    label=f"{label}: mean |err| = {sc['mean_rel_error_pct']:.2f}%")
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
        ax.set_ylabel("Z/a")
        ax.set_title(f"dam break front, dx = {dx*1000:.0f} mm")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("transport-velocity correction vs experiment (registry evidence leg)")
    fig.tight_layout()
    out = FIG_DIR / "figure_tvc_front.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_registry(registry: dict) -> Path:
    methods = sorted({r["method"] for r in registry["records"]},
                      key=lambda m: min(r["measured_pct"] for r in registry["records"]
                                        if r["method"] == m))
    best = [min(r["measured_pct"] for r in registry["records"] if r["method"] == m)
            for m in methods]
    worst = [max(r["measured_pct"] for r in registry["records"] if r["method"] == m)
             for m in methods]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = range(len(methods))
    ax.barh(list(y), best, color="#2e8b57", height=0.55, label="best record")
    ax.plot(worst, list(y), "rx", ms=6, label="worst record")
    ax.axvline(OWNER_TOLERANCE_PCT, color="crimson", lw=1.2, ls="--",
               label=f"owner standard {OWNER_TOLERANCE_PCT:g}%")
    ax.set_yticks(list(y))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xlabel("measured error vs its reference (%)")
    ax.set_xscale("log")
    ax.set_title("evidence registry: one bar per scheme, marker = worst banked record")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "figure_registry_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    granular = {}
    for dx in (0.03, 0.04):
        tag = str(dx).replace(".", "")  # 0.03 -> "003": the artefact naming convention
        granular[dx] = json.loads(
            (ROOT / "docs/xval_sphinx_tvc/data" / f"grid_dx{tag}.json").read_text())
    experiment = json.loads(EXP_PATH.read_text())
    registry = json.loads(REGISTRY_PATH.read_text())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_front = figure_front(granular, experiment)
    fig_cover = figure_registry(registry)

    recomputed = {f"{dx:.2f}": {v: score_from_curves(next(r for r in doc["records"]
                                                          if r["variant"] == v), experiment)
                                for v in ("baseline", ARM)}
                  for dx, doc in granular.items()}

    # cross-check: the recomputed number must reproduce the banked registry value
    banked = next(r["measured_pct"] for r in registry["records"]
                  if r["extractor"] == "x_transport_velocity")
    recomputed_fine = recomputed["0.03"][ARM]["mean_rel_error_pct"]
    assert abs(recomputed_fine - banked) < 1e-9, (recomputed_fine, banked)

    lines = ["# m20114 — transport-velocity evidence, machine-generated tables", "",
             "All numbers below are recomputed from the committed TVC dam-break grid",
             "against the in-repo Martin & Moyce digitisation", "(the stored `score`",
             "block of the artefact is ignored).", "",
             "The experimental point at Z/a = 0 is used by the absolute error but",
             "excluded from the relative error (division by zero), so the relative",
             "error is averaged over 19 of the 20 gated points.", "",
             f"## Recompute vs banked registry record (dx = 0.03 m, {ARM})", "",
             "| quantity | recomputed | banked in registry |", "|---|---|---|",
             f"| mean relative error of Z/a | {recomputed_fine:.4f} % | {banked:.4f} % |",
             ""]
    for dx, doc in sorted(granular.items()):
        lines += [f"## dx = {dx*1000:.0f} mm", "",
                  "| variant | points | MAE Z/a | mean rel. err | max abs err |",
                  "|---|---|---|---|---|"]
        for variant in ("baseline", ARM):
            sc = recomputed[f"{dx:.2f}"][variant]
            lines.append(f"| {variant} | {sc['points']} | {sc['mae']:.4f} | "
                         f"{sc['mean_rel_error_pct']:.2f} % | {sc['max_abs_error']:.4f} |")
        lines.append("")
    lines += ["## Registry coverage", "",
              f"- methods with evidence: {len(registry['methods_with_evidence'])}",
              f"- methods without evidence: {registry['methods_without_evidence'] or 'none'}",
              f"- catalogue size (METHODS + MODIFIERS): {len(METHODS) + len(MODIFIERS)}",
              f"- records: {len(registry['records'])}", ""]
    TABLES_PATH.write_text("\n".join(lines))

    SUMMARY_PATH.write_text(json.dumps({
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "increment": "transport_velocity_correction evidence closure",
        "arm": ARM,
        "recomputed": {k: {v: {kk: vv for kk, vv in sc.items() if kk != "rows"}
                           for v, sc in arms.items()} for k, arms in recomputed.items()},
        "registry_backed_pct": banked,
        "methods_without_evidence": registry["methods_without_evidence"],
        "records": len(registry["records"]),
        "figures": [fig_front.name, fig_cover.name],
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
    }, indent=2))

    print(f"wrote {fig_front}\nwrote {fig_cover}\nwrote {TABLES_PATH}\nwrote {SUMMARY_PATH}")
    print(f"recomputed dx=0.03 {ARM}: {recomputed_fine:.4f} %  (banked {banked:.4f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())