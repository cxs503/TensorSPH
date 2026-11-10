"""Driver for the free-surface verification surface.

The study is split into *parts* so that the expensive pieces can be run as
separate cluster jobs and merged afterwards; ``assemble`` merges the parts,
scores the gates (:mod:`tensorsph.freesurface.gates`) and draws the figures.
Every part writes a self-describing JSON file, so a missing part shows up as an
``UNMEASURED`` gate instead of a silent gap::

    python scripts/run_freesurface_verification.py --part ritter --part sloshing \\
        --out docs/benchmarks/free-surface
    python scripts/run_freesurface_verification.py --part assemble --out <same>
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path


from . import gates, profiles
from .ritter_anchor import RitterCase, run_ritter
from .sloshing_mode import SloshingModeCase, run_sloshing_mode
from .solitary_anchor import SolitaryCase, run_solitary

__all__ = [
    "assemble",
    "ladder_part",
    "ritter_part",
    "sloshing_part",
    "solitary_part",
]


def _write(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"part_{name}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    return path


def ritter_part(out_dir: str | Path, cells_per_depth: float = 10.0,
                viscosity: float = 1.0e-4, label: str = "base",
                device: str = "cpu") -> dict:
    """One dry-bed dam-break run against the Ritter solution."""
    case = RitterCase(cells_per_depth=float(cells_per_depth),
                      viscosity=float(viscosity), label=label, device=device)
    result = run_ritter(case)
    summary = result.summary()
    payload = {"kind": "ritter", "summary": summary,
               "case": {k: v for k, v in asdict(case).items()}}
    _write(Path(out_dir), f"ritter_{label}", payload)
    return payload


def ladder_part(out_dir: str | Path, cells=(6.0, 10.0, 20.0),
                viscosity: float = 1.0e-4, device: str = "cpu") -> dict:
    """Resolution ladder: depth L2 error against particle spacing."""
    rows = []
    for count in cells:
        case = RitterCase(cells_per_depth=float(count), viscosity=float(viscosity),
                          label=f"ladder{int(count)}", device=device)
        result = run_ritter(case)
        summary = result.summary()
        rows.append(summary)
    spacings = [r["dx"] for r in rows]
    errors = [r["depth_l2"] for r in rows]
    slope = profiles.loglog_slope(spacings, errors)
    resolved_errors = [r["depth_l2_resolved"] for r in rows]
    slope_resolved = (profiles.loglog_slope(spacings, resolved_errors)
                      if all(math.isfinite(v) for v in resolved_errors) else float("nan"))
    # the halving pair, if the ladder contains one, as a direct slope
    halving = float("nan")
    for coarse, fine in zip(rows, rows[1:]):
        ratio = coarse["dx"] / fine["dx"]
        if abs(ratio - 2.0) < 0.05:
            halving = math.log(fine["depth_l2"] / coarse["depth_l2"]) / math.log(0.5)
    payload = {
        "kind": "ladder",
        "viscosity": float(viscosity),
        "cells_per_depth": [float(c) for c in cells],
        "spacings": spacings,
        "depth_l2": errors,
        "depth_l2_resolved": resolved_errors,
        "slope": slope,
        "slope_resolved": slope_resolved,
        "slope_halving_pair": halving,
        "runs": rows,
    }
    _write(Path(out_dir), "ladder", payload)
    return payload


def sloshing_part(out_dir: str | Path, cells_per_depth: float = 20.0,
                  viscosity: float = 1.0e-3, periods: float = 3.0,
                  device: str = "cpu", label: str = "base") -> dict:
    """Released first-mode sloshing: frequency and numerical damping."""
    case = SloshingModeCase(cells_per_depth=float(cells_per_depth),
                            viscosity=float(viscosity), periods=float(periods),
                            device=device, label=label)
    result = run_sloshing_mode(case)
    summary = result.summary()
    analytic_times = summary["times"]
    summary["modal_linear"] = [
        case.amplitude * math.cos(case.omega1 * t) for t in analytic_times]
    payload = {"kind": "sloshing", "summary": summary,
               "case": {k: v for k, v in asdict(case).items()},
               "omega_analytic": case.omega1}
    _write(Path(out_dir), f"sloshing_{label}", payload)
    return payload


def solitary_part(out_dir: str | Path, cells_per_depth: float = 8.0,
                  viscosity: float = 1.0e-4, xsph: float = 0.2,
                  device: str = "cpu", label: str = "base",
                  depth: float | None = None, height: float | None = None,
                  flume_length: float | None = None, gauges=None,
                  duration: float | None = None,
                  density_diffusion: float | None = None) -> dict:
    """Goring solitary wave: amplitude, celerity and shape at three gauges."""
    overrides = {}
    for key, value in (("depth", depth), ("height", height),
                       ("flume_length", flume_length), ("duration", duration),
                       ("density_diffusion", density_diffusion)):
        if value is not None:
            overrides[key] = float(value)
    if gauges:
        overrides["gauge_positions"] = tuple(float(g) for g in gauges)
    case = SolitaryCase(cells_per_depth=float(cells_per_depth),
                        viscosity=float(viscosity), xsph=float(xsph),
                        device=device, label=label, **overrides)
    result = run_solitary(case)
    summary = result.summary()
    payload = {"kind": "solitary", "summary": summary,
               "case": {k: v for k, v in asdict(case).items()}}
    _write(Path(out_dir), f"solitary_{label}", payload)
    return payload


# --------------------------------------------------------------------------- #
# Assembly: gates + figures + report
# --------------------------------------------------------------------------- #

def _load_parts(out_dir: Path) -> dict:
    parts: dict = {}
    for path in sorted(out_dir.glob("part_*.json")):
        payload = json.loads(path.read_text())
        parts.setdefault(payload["kind"], []).append(payload)
    return parts


def _pick(parts: dict, kind: str):
    entries = parts.get(kind, [])
    if not entries:
        return None
    for entry in entries:
        summary = entry.get("summary", {})
        if summary.get("label") == "base":
            return entry
    return entries[0]


def assemble(out_dir: str | Path, figures: bool = True,
             notes: dict | None = None) -> dict:
    """Merge the parts, score the gates, draw the figures, write report.json."""
    out_dir = Path(out_dir)
    parts = _load_parts(out_dir)
    ritter = _pick(parts, "ritter")
    ladder = _pick(parts, "ladder")
    sloshing = _pick(parts, "sloshing")
    solitary = _pick(parts, "solitary")

    # C5 is mass conservation proper: the mass arrays of all three cases, not
    # the released-volume accuracy (that one is a physics number and is
    # reported as part of the Ritter accuracy block instead).
    drifts = {}
    if ritter:
        drifts["ritter"] = ritter["summary"].get("mass_drift_relative", float("nan"))
    if sloshing:
        drifts["sloshing"] = sloshing["summary"].get("mass_drift", float("nan"))
    if solitary:
        drifts["solitary"] = solitary["summary"].get("mass_drift_relative",
                                                     float("nan"))
    finite_drifts = [abs(v) for v in drifts.values() if v == v]
    measured = {
        "C1": ritter["summary"]["front_error_max"] if ritter else float("nan"),
        "C2": (solitary["summary"]["decay_10_lambda_extrapolated"]
               if solitary else float("nan")),
        "C3": sloshing["summary"]["frequency_error"] if sloshing else float("nan"),
        "C4": ladder["slope_resolved"] if ladder else float("nan"),
        "C5": max(finite_drifts) if finite_drifts else float("nan"),
    }
    notes = dict(notes or {})
    notes.setdefault("C5", "relative mass drift per case: "
                    + ", ".join(f"{k}={v:.3e}" for k, v in drifts.items())
                    + "; escaped particles: ritter="
                    + str(ritter["summary"].get("particles_escaped") if ritter else "n/a"))
    report = {
        "study": "free-surface verification surface (Ritter / solitary / sloshing)",
        "measured": measured,
        "parts": {kind: len(entries) for kind, entries in parts.items()},
        "ritter": ritter["summary"] if ritter else None,
        "ladder": ({k: v for k, v in ladder.items() if k != "runs"} | {
            "runs": [{key: run[key] for key in
                      ("label", "dx", "depth_l2", "depth_l2_resolved", "front_error_max",
                       "velocity_l2", "released_volume_error", "steps", "particles",
                       "wall_clock")}
                     for run in ladder["runs"]]} if ladder else None),
        "sloshing": ({k: v for k, v in sloshing["summary"].items()
                      if k not in ("times", "modal")} if sloshing else None),
        "solitary": ({k: v for k, v in solitary["summary"].items()
                      if k not in ("time_series", "shape")} if solitary else None),
    }
    scored = gates.evaluate(measured, notes=notes)
    report.update(scored)
    report["maturity"] = gates.maturity(scored)
    report["mass_drift_by_case"] = drifts
    report["convergence"] = ({"slope_full_fan": ladder["slope"],
                              "slope_resolved": ladder["slope_resolved"]}
                             if ladder else None)
    if figures:
        report["figures"] = draw_figures(out_dir, parts, measured)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    return report


def draw_figures(out_dir: Path, parts: dict, measured: dict | None = None) -> list[str]:
    """Six figures: three cases, the convergence ladder and a delta view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    ritter = _pick(parts, "ritter")
    sloshing = _pick(parts, "sloshing")
    solitary = _pick(parts, "solitary")
    ladder = _pick(parts, "ladder")

    if ritter:
        rows = ritter["summary"]["snapshot_rows"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for row, colour in zip(rows, ("tab:blue", "tab:orange", "tab:green")):
            centres = row["centres"]
            axes[0].plot(centres, row["depth"], color=colour, lw=1.6,
                         label=f"SPH t={row['time']:.3f}s")
            axes[0].plot(centres, row["depth_exact"], color=colour, lw=1.0, ls="--")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("h [m]")
        axes[0].set_title("Ritter dry-bed: depth (solid=SPH, dashed=exact)")
        axes[0].legend(fontsize=8)
        axes[0].set_xlim(0.0, ritter["summary"]["column_width"]
                         + 2.4 * ritter["summary"]["validity_window"]
                         * math.sqrt(9.81 * ritter["summary"]["depth0"]))
        axes[1].plot([r["time"] for r in rows], [r["depth_l2"] for r in rows],
                     "o-", label="depth L2")
        axes[1].plot([r["time"] for r in rows], [r["velocity_l2"] for r in rows],
                     "s-", label="velocity L2")
        axes[1].plot([r["time"] for r in rows], [r["front_error"] for r in rows],
                     "^-", label="front error")
        axes[1].axhline(0.10, color="k", ls=":", label="C1 gate 10%")
        axes[1].set_xlabel("t [s]")
        axes[1].set_ylabel("relative error")
        axes[1].set_title("Ritter: measured errors")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        path = fig_dir / "ritter_profiles.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path.relative_to(out_dir)))

    if ladder:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.loglog(ladder["spacings"], ladder["depth_l2"], "o-", label="depth L2")
        slope = ladder["slope"]
        reference = [e * (ladder["spacings"][0] / h) ** 1.0
                     for h, e in zip(ladder["spacings"], ladder["depth_l2"])]
        ax.loglog(ladder["spacings"], reference, "k--", label="slope 1 reference")
        ax.set_xlabel("particle spacing dx [m]")
        ax.set_ylabel("relative depth L2")
        ax.set_title(f"Ritter convergence: slope {slope:.2f}")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        path = fig_dir / "ritter_convergence.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path.relative_to(out_dir)))

    if sloshing:
        summary = sloshing["summary"]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.plot(summary["times"], summary["modal"], lw=1.4, label="SPH mode 1 amplitude")
        ax.plot(summary["times"], summary["modal_linear"], "k--", lw=1.0,
                label=r"linear $A\cos(\omega_1 t)$")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$a_1$ [m]")
        ax.set_title(f"Sloshing free decay: omega={summary['omega_measured']:.2f} "
                     f"vs {summary['omega_analytic']:.2f} rad/s")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = fig_dir / "sloshing_mode.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path.relative_to(out_dir)))

    if solitary:
        summary = solitary["summary"]
        times = summary["time_series"]["times"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for series, gauge in zip(summary["time_series"]["elevations"],
                                 summary["gauges"]):
            axes[0].plot(times, series, lw=1.2, label=f"x={gauge['x']:.1f} m "
                                                      f"(H={gauge['crest_height']*1e3:.1f} mm)")
        axes[0].set_xlabel("t [s]")
        axes[0].set_ylabel("elevation [m]")
        axes[0].set_title(f"Solitary wave, H={summary['height']*1e3:.1f} mm, "
                          f"h={summary['depth']*1e3:.0f} mm")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        records = summary.get("shape_records") or []
        if records:
            record = records[min(1, len(records) - 1)]
            axes[1].plot(record["t"], record["eta_measured"], lw=1.5, label="SPH gauge")
            axes[1].plot(record["t"], record["eta_analytic"], "k--", lw=1.2,
                         label=r"$H\,\mathrm{sech}^2$ (measured $H$)")
            axes[1].set_xlabel("t [s]")
            axes[1].set_ylabel("elevation [m]")
            axes[1].set_title(f"Crest record at x={record['gauge_x']:.2f} m, "
                              f"shape L2={record['shape_l2']:.3f}")
            axes[1].legend(fontsize=9)
            axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = fig_dir / "solitary_wave.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path.relative_to(out_dir)))

    if measured:
        # ratio = measured / gate; 1.0 is the gate. C4 is a band criterion and
        # is annotated separately instead of being squeezed onto this axis.
        labels, ratios, colours = [], [], []
        for criterion in gates.CRITERIA:
            if criterion.kind != "max":
                continue
            value = measured.get(criterion.identifier, float("nan"))
            if value != value:
                continue
            ratio = value / criterion.threshold
            labels.append(f"{criterion.identifier}\n{criterion.description[:28]}")
            ratios.append(ratio)
            colours.append("tab:green" if ratio <= 1.0 else "tab:red")
        fig, ax = plt.subplots(figsize=(8.2, 4.0))
        bars = ax.bar(labels, ratios, color=colours)
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", ls=":", label="gate")
        for bar, ratio in zip(bars, ratios):
            ax.annotate(f"{ratio:.3g}x", (bar.get_x() + bar.get_width() / 2, ratio),
                        ha="center", va="bottom", fontsize=9)
        slope = measured.get("C4", float("nan"))
        ax.set_title("Three free-surface cases vs their criteria "
                     f"(bar = measured/gate, red = FAIL); C4 slope = {slope:.2f}")
        ax.set_ylabel("measured / gate")
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = fig_dir / "summary_gates.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path.relative_to(out_dir)))

    return written
