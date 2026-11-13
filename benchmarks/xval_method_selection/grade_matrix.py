"""XVAL-SPH-METHOD-SELECTION-2026-11-12: grade a matrix of wave-flume records.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` (mission 17501) cross-validates the
method selection recorded on 2026-11-12 (``oss_solutions`` id=408) against the
analytic references it names, and decides which mature open-source (SPHinXsys)
practices are worth porting.

This driver is deliberately dumb: every number in the report is read out of the
JSON records written by ``benchmarks/wave_flume/run.py`` through the shipped
grader ``tensorsph.wave_flume_grade.grade_regular_record`` -- nothing is typed in
by hand.  Records that are missing (a cluster job that never dispatched) are
reported as gaps instead of being silently dropped.

Environment variables ``XVAL_RUN_DIR`` / ``XVAL_OUT`` / ``XVAL_FIG_DIR`` override
the defaults (``runs/local``, ``docs/reports/xval_method_selection_m17501.json``,
``docs/figures/xval_method_selection_m17501``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.wave_flume_grade import (  # noqa: E402
    grade_regular_record,
    laminar_boundary_layer_damping,
)

# Rows of the published matrix: (label, record, variant group, changing knob).
# Paths are repository-relative, so the row a reader sees is the row that exists.
MATRIX = [
    ("D20_h025_A", "runs/xval/cell_0.json", "ladder",
     "method A, H/dx = 3.20, t_end 5.5 s"),
    ("D13_h025_A", "runs/xval/cell_1.json", "ladder",
     "method A, H/dx = 2.08, t_end 5.5 s"),
    ("D20_h025_xsph", "runs/xval/cell_2.json", "mechanism",
     "method A + XSPH 0.05 in the propagation phase (post-hoc probe)"),
    ("D20_h025_comp", "runs/xval/cell_3.json", "mechanism",
     "method A + SPHinXsys compression regularization (post-hoc probe)"),
    ("D20_h020_archive", "docs/reports/artifacts_wave_flume/wf_D20_unstable.json",
     "archive", "m17501 archive record (h = 0.20 m, t_end 3.7 s)"),
]

# The seven cells the pre-registered protocol asked for.  They are *not* re-run
# blindly: a cell whose record is absent is listed as an explicit gap (the repair
# run answers the same questions with three decision-focused cells), and a cell
# whose record exists is graded like any other row.
PREDECLARED = [
    ("D13_hn016_d010_settle", "wf_D13_settle_d010.json", "ladder", "ref A, H/dx=2.08"),
    ("D16_hn016_d010_settle", "wf_D16_settle_d010.json", "ladder", "ref A, H/dx=2.56"),
    ("D20_hn016_d010_settle", "wf_D20_settle_d010.json", "ladder", "ref A, H/dx=3.20"),
    ("D13_hn016_d000_settle", "wf_D13_settle_d000.json", "delta", "delta-SPH = 0.0"),
    ("D13_hn016_d020_settle", "wf_D13_settle_d020.json", "delta", "delta-SPH = 0.2"),
    ("D13_hn010_d010_settle", "wf_D13_hr010_settle_d010.json", "height", "H/h = 0.10"),
    ("D13_hn016_d010_nosettle", "wf_D13_nosettle_d010.json", "settle", "no settle window"),
]

PREDECLARED_DIR = Path("runs/local")


def locate(run_dir: Path, name: str) -> Path:
    """Resolve a matrix record: bare names live in ``run_dir``, paths stand alone."""
    path = Path(name)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return run_dir / path

WINDOWS = (4.0, 5.0)
TOLERANCE = 0.10      # the case-pool gate (CASE-WAVE-FLUME)
OWNER_TOL = 0.15      # owner precision standard for the XVAL anchor

# Stability gate.  Both thresholds are copied from the pre-registered protocol
# (docs/xval_method_selection_m17501/protocol.md §3 "稳定性": ``max|v| < c0`` and
# density compression < 10 %), which was written before any record existed, so
# they are *not* tuned to the data.  ``escaped`` is recorded as a diagnostic but
# is not a gate: the protocol only asks for the number (there is no physical
# reason a wave flume cannot lose a few particles into the sponge).
SPEED_RATIO_MAX = 1.0     # max|v| / c0
COMPRESSION_MAX = 0.10    # max density compression


def stability(diag: dict, sound_speed) -> dict:
    """Judge a record's *numerical stability* from its own diagnostics.

    A diverged run still writes finite gauge samples for a while, so the wave
    height it yields can look plausible while the field is exploding.  The audit
    of mission 17501 found exactly this: a record with ``max|v|/c0 = 3.6`` and
    45 % compression was scored as a normal row.  The gate below is the fix --
    an unstable record can never be counted as a pass, whatever its gauge error.
    """
    speed = diag.get("max_speed")
    comp = diag.get("max_density_compression")
    ratio = None
    if speed is not None and sound_speed:
        ratio = float(speed) / float(sound_speed)
    violations = []
    if ratio is None or not math.isfinite(ratio):
        violations.append("acoustic Mach number unavailable")
    elif ratio >= SPEED_RATIO_MAX:
        violations.append(f"max|v|/c0 = {ratio:.3f} >= {SPEED_RATIO_MAX:g}")
    if comp is None or not math.isfinite(float(comp)):
        violations.append("density compression unavailable")
    elif float(comp) >= COMPRESSION_MAX:
        violations.append(f"compression {100 * float(comp):.2f} % >= "
                          f"{100 * COMPRESSION_MAX:g} %")
    return {
        "sound_speed": sound_speed,
        "speed_ratio": ratio,
        "speed_ratio_max": SPEED_RATIO_MAX,
        "max_density_compression": (float(comp) if comp is not None
                                    and math.isfinite(float(comp)) else None),
        "compression_max": COMPRESSION_MAX,
        "escaped": diag.get("escaped"),
        "stable": not violations,
        "violations": violations,
    }


def verdict(stable: bool, within_tolerance: bool) -> str:
    """One word for the row: the gate is *stability AND accuracy*."""
    if not stable and not within_tolerance:
        return "fail_stability+tolerance"
    if not stable:
        return "fail_stability"
    if not within_tolerance:
        return "fail_tolerance"
    return "pass"


def _sanitize(obj):
    """Replace NaN/inf by None so the report JSON stays strictly valid."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def grade_one(path: Path, start: float) -> dict:
    return grade_regular_record(load(path), start=start, tolerance=TOLERANCE)


def summarise(label: str, knob: str, group: str, path: Path) -> dict:
    """Grade one record in both analysis windows and pack the comparison row."""
    row = {"label": label, "group": group, "knob": knob, "record": str(path)}
    if not path.exists():
        row.update({"present": False, "note": "record missing (job not dispatched)"})
        return row
    record = load(path)
    if record.get("failed"):
        row.update({"present": True, "failed": True, "error": record.get("error"),
                    "passed": False})
        return row
    graded = {f"start_{w:g}": grade_one(path, w) for w in WINDOWS}
    primary = graded[f"start_{WINDOWS[0]:g}"]
    cfg = record.get("config", {})
    diag = record.get("diagnostics", {})
    if primary.get("failed"):
        # The grader refused the series (the analysis window holds no usable
        # samples, e.g. the record stopped before 4 s).  That is a *gap in the
        # evidence*, never a measurement of zero: keep it out of every count
        # that could be mistaken for a score.
        row.update({
            "present": True,
            "failed": False,
            "gradeable": False,
            "error": primary.get("error"),
            "gauge_errors": [gg.get("error") for gg in primary.get("gauges", [])],
            "gauges": [{"x": gg.get("x"), "error": gg.get("error"),
                        "dry_fraction": gg.get("dry_fraction")}
                       for gg in primary.get("gauges", [])],
            "target_height": primary.get("target_height"),
            "stability": stability(diag, cfg.get("sound_speed")),
            "within_owner_tol": False,
            "within_case_tol": False,
            "passed": False,
            "verdict": "ungradeable",
            "windows": graded,
        })
        return row
    gauges = [
        {"x": g["x"], "height": g.get("height"),
         "error_vs_target": g.get("error_vs_target"),
         "height_crossings": g.get("height_crossings"),
         "h_rms": g.get("h_rms"), "dry_fraction": g.get("dry_fraction")}
        for g in primary.get("gauges", [])
    ]
    worst = primary.get("worst_error_vs_target")
    stable_block = stability(diag, cfg.get("sound_speed"))
    within_owner = bool(worst is not None and abs(worst) <= OWNER_TOL)
    row.update({
        "present": True,
        "failed": False,
        "gradeable": True,
        "stability": stable_block,
        "passed": bool(stable_block["stable"] and within_owner),
        "verdict": verdict(stable_block["stable"], within_owner),
        "depth_cells": cfg.get("depth_cells"),
        "height_cells": cfg.get("height_cells"),
        "dx": cfg.get("dx"),
        "particles": cfg.get("particle_count"),
        "steps": diag.get("steps_taken"),
        "dt": diag.get("dt"),
        "simulated_time": diag.get("simulated_time"),
        "density_diffusion": cfg.get("density_diffusion"),
        "settle_time": cfg.get("settle_time"),
        "viscosity_settle": cfg.get("viscosity_settle"),
        "xsph_settle": cfg.get("xsph_settle"),
        "sound_speed": cfg.get("sound_speed"),
        "target_height": primary.get("target_height"),
        "gauges": gauges,
        "worst_error_vs_target": worst,
        "within_owner_tol": bool(worst is not None and abs(worst) <= OWNER_TOL),
        "within_case_tol": bool(worst is not None and abs(worst) <= TOLERANCE),
        "attenuation": primary.get("attenuation"),
        "theory": primary.get("theory"),
        "excess_loss_over_span": primary.get("excess_loss_over_span"),
        "dispersion": primary.get("dispersion"),
        "reflection": primary.get("reflection"),
        "max_speed": diag.get("max_speed"),
        "speed_ratio": (diag.get("max_speed") / cfg.get("sound_speed")
                        if diag.get("max_speed") is not None and cfg.get("sound_speed")
                        else None),
        "max_density_compression": diag.get("max_density_compression"),
        "escaped": diag.get("escaped"),
        "window_sensitivity": {
            k: (v.get("worst_error_vs_target") if not v.get("failed") else None)
            for k, v in graded.items()
        },
        "windows": graded,
    })
    return row


def measurable(row: dict) -> bool:
    """True only for a row that is a *measurement*: present, graded and stable.

    Figures and error statistics must never be built on a missing record, on a
    record the grader refused, or on a diverged field.
    """
    return bool(row.get("present") and not row.get("failed")
                and row.get("gradeable", True) and row["stability"]["stable"])


def fig_ladder(rows, path: Path) -> bool:
    pts = [r for r in rows if r["group"] == "ladder" and measurable(r)]
    if not pts:
        return False
    pts.sort(key=lambda r: r["height_cells"])
    x = [r["height_cells"] for r in pts]
    y = [100.0 * r["worst_error_vs_target"] for r in pts]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(x, y, "o-", label="worst gauge $|H_{meas}-H_{target}|/H_{target}$")
    for r, xi, yi in zip(pts, x, y):
        ax.annotate(f"{yi:+.1f}%", (xi, yi), textcoords="offset points", xytext=(6, 4),
                    fontsize=8)
    ax.axhline(OWNER_TOL * 100, color="C3", ls="--", lw=1.2, label="15% owner gate")
    ax.axhline(-OWNER_TOL * 100, color="C3", ls="--", lw=1.2)
    ax.axhline(TOLERANCE * 100, color="C2", ls=":", lw=1.2, label="10% case gate")
    ax.axhline(-TOLERANCE * 100, color="C2", ls=":", lw=1.2)
    ax.set_xlabel("$H/\\Delta x$ (particles per wave height)")
    ax.set_ylabel("wave-height error vs piston transfer [%]")
    ax.set_title("CASE-WAVE-FLUME resolution ladder (method A)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_series(rows, run_dir: Path, path: Path) -> bool:
    """Gauge record + harmonic fit at the gauge of the finest ladder point."""
    pts = [r for r in rows if r["group"] == "ladder" and measurable(r)]
    if not pts:
        return False
    best = max(pts, key=lambda r: r["height_cells"])
    record = load(Path(best["record"]))
    spec = record["spec"]
    times = record["times"]
    series = record["probe_eta"]
    omega = float(spec["omega"])
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    ax = axes[0]
    start = WINDOWS[0]
    for idx, (x, eta) in enumerate(zip(record["probe_x"], series)):
        ax.plot(times, eta, lw=1.0, label=f"gauge {idx} x={x:.2f} m")
    ax.axvline(start, color="k", ls="--", lw=1.0, label=f"analysis start {start:g} s")
    ax.axvline(float(record["config"].get("settle_time", 0.0)), color="C7", ls=":",
               lw=1.0, label="settle end")
    ax.set_ylabel("free-surface elevation $\\eta$ [m]")
    ax.set_title(f"gauge records, {best['label']} (H/\\\\Delta x = {best['height_cells']:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    target_a = float(best["target_height"]) / 2.0
    for x, eta in zip(record["probe_x"], series):
        ax.plot(times, eta, lw=0.8, alpha=0.45)
        ax.annotate(f"x={x:.2f} m", (times[-1], eta[-1]), fontsize=7,
                    textcoords="offset points", xytext=(2, 0))
    ax.plot(times, [target_a * math.sin(omega * (tt - times[0])) for tt in times],
            "k--", lw=1.0, label=f"piston-transfer analogue A=H/2={target_a * 1e3:.1f} mm")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$\\eta$ [m]")
    ax.set_title("all gauges vs the piston-transfer amplitude (phase-free analogue)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_attenuation(rows, path: Path) -> bool:
    pts = [r for r in rows if measurable(r)]
    if not pts:
        return False
    best = max((r for r in pts if r["group"] == "ladder"),
               key=lambda r: r["height_cells"], default=None)
    if best is None:
        return False
    if best is None:
        return False
    xs = [g["x"] for g in best["gauges"]]
    hs = [g["height"] for g in best["gauges"]]
    if any(h is None or not math.isfinite(h) or h <= 0 for h in hs):
        return False
    mu = best["attenuation"]["mu_per_m"]
    mu_theory = best["theory"]["mu_per_m"]
    h0 = hs[0] * math.exp(mu * xs[0])
    grid = [min(xs) + i * (max(xs) - min(xs)) / 50 for i in range(51)]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(xs, hs, "o", label="harmonic-fit wave height")
    ax.plot(grid, [h0 * math.exp(-mu * x) for x in grid], "-",
            label=f"fit $\\mu$ = {mu:.3e} 1/m")
    ax.plot(grid, [h0 * math.exp(-mu_theory * x) for x in grid], "--",
            label=f"laminar BL $\\mu_{{theory}}$ = {mu_theory:.3e} 1/m")
    ax.set_xlabel("downstream distance $x$ [m]")
    ax.set_ylabel("wave height $H$ [m]")
    ax.set_title(f"attenuation attribution, {best['label']}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_sensitivity(rows, path: Path) -> bool:
    groups = {}
    for r in rows:
        if not measurable(r):
            continue
        groups.setdefault(r["group"], []).append(r)
    if not groups:
        return False
    labels, values, colours = [], [], []
    for group, members in sorted(groups.items()):
        for r in members:
            labels.append(f"{group}\n{r['knob']}")
            values.append(100.0 * r["worst_error_vs_target"])
            if abs(r["worst_error_vs_target"]) <= OWNER_TOL:
                colours.append("C0")
            else:
                colours.append("C3")
    fig, ax = plt.subplots(figsize=(max(6.0, 1.05 * len(labels)), 4.2))
    ax.bar(range(len(labels)), values, color=colours)
    for i, v in enumerate(values):
        ax.annotate(f"{v:+.1f}%", (i, v), textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -12), ha="center", fontsize=8)
    ax.axhline(OWNER_TOL * 100, color="C3", ls="--", lw=1.2)
    ax.axhline(-OWNER_TOL * 100, color="C3", ls="--", lw=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("worst wave-height error [%]")
    ax.set_title("sensitivity / ablation matrix (red = outside the 15% gate)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=os.environ.get("XVAL_RUN_DIR", "runs/local"))
    parser.add_argument("--out", default=os.environ.get(
        "XVAL_OUT", "docs/reports/xval_method_selection_m17501.json"))
    parser.add_argument("--figures", default=os.environ.get(
        "XVAL_FIG_DIR", "docs/figures/xval_method_selection_m17501"))
    parser.add_argument("--matrix", default=None,
                        help="JSON file with an alternative [[label,file,group,knob],...]")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if args.matrix:
        matrix = [tuple(row) for row in json.loads(Path(args.matrix).read_text())]
        rows = [summarise(label, knob, group, locate(run_dir, name))
                for label, name, group, knob in matrix]
        unexecuted = []
    else:
        rows = [summarise(label, knob, group, locate(run_dir, name))
                for label, name, group, knob in MATRIX]
        unexecuted = []
        for label, name, group, knob in PREDECLARED:
            path = PREDECLARED_DIR / name
            if path.exists():
                rows.append(summarise(label, knob, group, path))
            else:
                unexecuted.append({
                    "label": label, "record": str(path), "group": group,
                    "knob": knob, "status": "not_executed",
                    "reason": "record absent: the 7-cell grid was not re-run; the "
                              "repair run answered the same questions with three "
                              "decision-focused cells (group 'ladder'/'mechanism')",
                })

    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for name, fn in (("ladder.png", lambda p: fig_ladder(rows, p)),
                     ("series.png", lambda p: fig_series(rows, run_dir, p)),
                     ("attenuation.png", lambda p: fig_attenuation(rows, p)),
                     ("sensitivity.png", lambda p: fig_sensitivity(rows, p))):
        target = fig_dir / name
        made = fn(target)
        figures.append({"name": name, "path": str(target), "made": made})

    present = [r for r in rows if r.get("present")]
    graded = [r for r in present if not r.get("failed")]
    measured = [r for r in rows if measurable(r)]
    theory = measured[0]["theory"] if measured else {}
    payload = {
        "mission": 17906,
        "mission_history": [17501, 17906],
        "repair_of": {"mission": 17501,
                      "audit": "verdict_m17501.md (reopened: stale report, no "
                               "stability gate, deliverables uncommitted)"},
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "reference": "oss_solutions id=408 (TensorSPH CASE-WAVE-FLUME method-selection-2026-11-12)",
        "oss_counterparts": {
            "no_delta_sph_compression_regularization": "oss_solutions id=356 (Xiangyu-Hu/SPHinXsys)",
            "mach_limited_riemann_dissipation": "oss_solutions id=355 (Xiangyu-Hu/SPHinXsys)",
            "transport_velocity_correction": "oss_solutions id=358 (Xiangyu-Hu/SPHinXsys)",
        },
        "tolerance_owner": OWNER_TOL,
        "tolerance_case": TOLERANCE,
        "windows": list(WINDOWS),
        "physics_reference": theory,
        "rows": rows,
        "predeclared_unexecuted": unexecuted,
        "figures": figures,
        "verdict_rule": (
            "a row passes only when its own diagnostics satisfy the pre-registered "
            "stability gate (max|v|/c0 < 1, max density compression < 10 %) AND its "
            "worst gauge wave-height error is within the owner 15 % gate; "
            "'ungradeable' means the record exists but the analysis window holds no "
            "usable samples, and is reported as a gap, never as a zero"),
        "counts": {
            "matrix": len(rows),
            "present": len(present),
            "missing": len(rows) - len(present),
            "failed": len(present) - len(graded),
            "ungradeable": sum(1 for r in graded if not r.get("gradeable", True)),
            "unstable": sum(1 for r in graded
                            if r.get("gradeable", True)
                            and not r["stability"]["stable"]),
            "within_owner_tol": sum(1 for r in graded if r["within_owner_tol"]),
            "passed": sum(1 for r in graded if r.get("passed")),
        },
    }
    # A diverged record can carry NaN/inf: JSON cannot, so write them as null.
    text = json.dumps(_sanitize(payload), indent=2, default=str, allow_nan=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}: {len(present)}/{len(rows)} records present, "
          f"{payload['counts']['within_owner_tol']} within {OWNER_TOL:.0%}")
    for r in rows:
        if not r.get("present"):
            print(f"  MISSING  {r['label']}: {r['note']}")
        elif r.get("failed"):
            print(f"  FAILED   {r['label']}: {r['error']}")
        elif not r.get("gradeable", True):
            print(f"  NOGRADE  {r['label']}: {r['error']}  "
                  f"(stable={r['stability']['stable']})")
        else:
            print(f"  {r['label']:26s} worst {100 * r['worst_error_vs_target']:+7.2f}%  "
                  f"max|v|/c0 {r['speed_ratio']:.3f}  comp {r['max_density_compression']:.2e}"
                  f"  -> {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())