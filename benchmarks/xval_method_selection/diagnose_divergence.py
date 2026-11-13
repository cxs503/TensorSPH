"""Diagnose *why* the wave-flume record diverges, and write a paper-grade report.

The repaired matrix driver (``grade_matrix.py``) can only say "this record is
unstable".  This script says *how*: it reads the record's own gauge series and
diagnostics and reports, without running the solver,

* the growth envelope of the fundamental harmonic at each gauge,
* a two-probe incident/reflected decomposition (so "the sponge does not absorb"
  can be separated from "the wavemaker over-generates"),
* the onset time of the acoustic Mach number passing one, and the compression
  history -- the two quantities the pre-registered stability gate watches.

Everything it prints is also written to
``docs/reports/xval_method_selection_diagnostics.json`` and the narrative report
``docs/report_xval_method_selection_m17906.md`` is assembled from those same
values, so no digit in the report is typed by hand.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/diagnose_divergence.py
"""
from __future__ import annotations

import cmath
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.wave_flume_grade import wave_height_target  # noqa: E402

OUT_JSON = REPO / "docs" / "reports" / "xval_method_selection_diagnostics.json"
FIG_DIR = REPO / "docs" / "figures" / "xval_method_selection_m17501"
OUT_MD = REPO / "docs" / "report_xval_method_selection_m17906.md"
ARCHIVE = REPO / "docs" / "reports" / "artifacts_wave_flume" / "wf_D20_unstable.json"
CELL_DIR = REPO / "runs" / "xval"
CELLS = {0: "D20_h025_A", 1: "D13_h025_A", 2: "D20_h025_xsph"}


def records():
    """Every record that can be diagnosed today (archive first, then repair cells)."""
    found = [("archive_h020", ARCHIVE)]
    found += [(label, CELL_DIR / f"cell_{i}.json")
              for i, label in sorted(CELLS.items())]
    return [(label, path) for label, path in found if path.exists()]


def envelope(record, label):
    """Fundamental-harmonic amplitude and phase per gauge, in sliding windows.

    One wave period per window, half a period apart, so the estimate is a moving
    average of a signal whose amplitude is *not* steady -- exactly the case here.
    """
    t = np.asarray(record["times"], dtype=float)
    eta = np.asarray(record["probe_eta"], dtype=float)
    xs = [float(x) for x in record["probe_x"]]
    omega = float(record["spec"]["omega"])
    k = float(record["spec"]["k"])
    period = 2.0 * math.pi / omega
    windows = []
    start = 0.0
    while start + period <= t[-1] + 1e-12:
        stop = start + period
        amp, phase, resid = [], [], []
        for series in eta:
            mask = (t >= start) & (t <= stop) & np.isfinite(series)
            if mask.sum() < 10:
                amp.append(float("nan"))
                phase.append(float("nan"))
                resid.append(float("nan"))
                continue
            tt, ee = t[mask], series[mask]
            design = np.column_stack([np.cos(omega * tt), np.sin(omega * tt),
                                      np.ones_like(tt)])
            coef, *_ = np.linalg.lstsq(design, ee, rcond=None)
            amp.append(float(math.hypot(coef[0], coef[1])))
            phase.append(float(math.atan2(-coef[1], coef[0])))
            resid.append(float(np.sqrt(np.mean((design @ coef - ee) ** 2))))
        # two-probe separation on both gauge pairs: A(x) = I e^{ikx} + R e^{-ikx}
        pairs = {}
        for a, b in ((0, 1), (2, 3)):
            if not (math.isfinite(amp[a]) and math.isfinite(amp[b])):
                continue
            A = [amp[i] * cmath.exp(1j * phase[i]) for i in (a, b)]
            matrix = np.array([[cmath.exp(1j * k * xs[a]), cmath.exp(-1j * k * xs[a])],
                               [cmath.exp(1j * k * xs[b]), cmath.exp(-1j * k * xs[b])]])
            incident, reflected = np.linalg.solve(matrix, np.array(A))
            pairs[f"{a}{b}"] = {
                "incident_amp_mm": abs(incident) * 1e3,
                "reflected_amp_mm": abs(reflected) * 1e3,
                "reflection_ratio": abs(reflected) / abs(incident) if incident else None,
            }
        windows.append({
            "t_start": start, "t_stop": stop,
            "amplitude_mm": [None if not math.isfinite(a) else a * 1e3 for a in amp],
            "phase_rad": [None if not math.isfinite(p) else p for p in phase],
            "fit_residual_mm": [None if not math.isfinite(r) else r * 1e3 for r in resid],
            "separation": pairs,
        })
        start += 0.5 * period
    return {"label": label, "period": period, "wave_number": k, "windows": windows}


def growth_rate(series_of_windows):
    """Least-squares e-folding time of the amplitude envelope (per gauge)."""
    out = []
    n_gauge = len(series_of_windows[0]["amplitude_mm"])
    for g in range(n_gauge):
        pts = [(w["t_start"] + 0.5 * w["t_stop"] - 0.5 * w["t_start"], w["amplitude_mm"][g])
               for w in series_of_windows]
        pts = [(t, a) for t, a in pts if a and a > 0]
        if len(pts) < 3:
            out.append(None)
            continue
        t = np.array([p[0] for p in pts])
        a = np.log(np.array([p[1] for p in pts]))
        slope, _ = np.polyfit(t, a, 1)
        out.append({"slope_per_s": float(slope),
                    "e_folding_s": float(1.0 / slope) if slope > 0 else None})
    return out


def divergence_onset(record):
    """First time the acoustic Mach number passes one, plus the peak values."""
    t = record["times"]
    speed = record.get("speed_history") or []
    comp = record.get("compression_history") or []
    c0 = float(record["config"]["sound_speed"])
    onset = None
    for tt, s in zip(t, speed):
        if s / c0 >= 1.0:
            onset = float(tt)
            break
    return {
        "sound_speed": c0,
        "mach_gt_one_at_s": onset,
        "max_speed": max(speed) if speed else None,
        "max_mach": (max(speed) / c0) if speed else None,
        "max_compression": max(comp) if comp else None,
        "escaped": record.get("diagnostics", {}).get("escaped"),
        "particles": record.get("config", {}).get("particle_count"),
        "simulated_time": record.get("diagnostics", {}).get("simulated_time"),
    }


def figures(diag):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    first = diag["records"][0]
    env = first["envelope"]
    mid = [0.5 * (w["t_start"] + w["t_stop"]) for w in env["windows"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for g in range(len(env["windows"][0]["amplitude_mm"])):
        series = [w["amplitude_mm"][g] for w in env["windows"]]
        ax.plot(mid, series, "o-", ms=3, lw=1.0, label=f"gauge {g}")
    target = 0.5 * diag["target_height"] * 1e3
    ax.axhline(target, color="k", ls="--", lw=1.0,
               label=f"piston transfer $H/2$ = {target:.1f} mm")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fundamental-harmonic amplitude [mm]")
    ax.set_title(f"growth envelope, {env['label']}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for key, colour in (("01", "C0"), ("23", "C3")):
        inc = [w["separation"].get(key, {}).get("incident_amp_mm") for w in env["windows"]]
        ref = [w["separation"].get(key, {}).get("reflected_amp_mm") for w in env["windows"]]
        ax.plot(mid, inc, "o-", color=colour, ms=3, lw=1.0,
                label=f"incident, pair {key}")
        ax.plot(mid, ref, "s--", color=colour, ms=3, lw=1.0,
                label=f"reflected, pair {key}")
    ax.axhline(target, color="k", ls=":", lw=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("two-probe amplitude [mm]")
    ax.set_title("incident / reflected decomposition")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG_DIR / "growth.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append({"name": "growth.png", "path": str(path.relative_to(REPO))})

    record = json.loads(ARCHIVE.read_text())
    t = record["times"]
    c0 = float(record["config"]["sound_speed"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(t, [s / c0 for s in record["speed_history"]], lw=1.2,
            label="max |v| / $c_0$")
    ax.plot(t, [c / 0.10 for c in record["compression_history"]], lw=1.2,
            label="density compression / 10 %")
    ax.axhline(1.0, color="C3", ls="--", lw=1.0, label="stability gate")
    onset = diag["records"][0]["onset"]["mach_gt_one_at_s"]
    if onset is not None:
        ax.axvline(onset, color="C7", ls=":", lw=1.0,
                   label=f"Mach 1 at t = {onset:.2f} s")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("gate-normalised diagnostic")
    ax.set_yscale("log")
    ax.set_title("diagnostics of the archived record vs the pre-registered gate")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG_DIR / "divergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append({"name": "divergence.png", "path": str(path.relative_to(REPO))})
    return made


def markdown(diag, fig_list):
    rows = []
    for rec in diag["records"]:
        env = rec["envelope"]
        onset = rec["onset"]
        rows.append(f"\n### {rec['label']}（`{rec['record']}`）\n")
        rows.append("| 量 | 值 |\n|---|---|")
        rows.append(f"| 目标波高 H（活塞一阶传递） | {rec['target_height'] * 1e3:.2f} mm |")
        rows.append(f"| 模拟时长 / 粒子数 | {onset['simulated_time']:.2f} s / "
                    f"{onset['particles']} |")
        rows.append(f"| max\\|v\\| / c0 | {onset['max_mach']:.3f}（门 1.0）|")
        rows.append(f"| max 密度压缩 | {100 * onset['max_compression']:.1f} %（门 10 %）|")
        rows.append(f"| 逃逸粒子 | {onset['escaped']} |")
        if onset["mach_gt_one_at_s"] is not None:
            rows.append(f"| Mach 1 时刻 | {onset['mach_gt_one_at_s']:.2f} s |")
        rows.append("\n| 窗口 [s] | " + " | ".join(
            f"A{g} [mm]" for g in range(len(env["windows"][0]["amplitude_mm"])))
            + " | R01 | R23 |")
        rows.append("|" + "---|" * (len(env["windows"][0]["amplitude_mm"]) + 3))
        for w in env["windows"]:
            amps = " | ".join("—" if a is None else f"{a:.2f}"
                              for a in w["amplitude_mm"])
            r01 = w["separation"].get("01", {}).get("reflection_ratio")
            r23 = w["separation"].get("23", {}).get("reflection_ratio")
            fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
            rows.append(f"| {w['t_start']:.2f}–{w['t_stop']:.2f} | {amps} | "
                        f"{fmt(r01)} | {fmt(r23)} |")
        growth = rec["growth"]
        efolds = [g["e_folding_s"] for g in growth if g and g["e_folding_s"]]
        if efolds:
            rows.append(f"\n振幅包络的 e 倍时间（最小二乘）："
                        f"{min(efolds):.2f}–{max(efolds):.2f} s。")
        else:
            rows.append("\n振幅包络不足以拟合增长指数（点数不足）。")
    figures_md = "\n".join(
        f"![{f['name']}]({f['path']})\n" for f in fig_list)
    return "\n".join(rows) + "\n\n" + figures_md


def main() -> int:
    all_records = records()
    if not all_records:
        print("no records to diagnose")
        return 1
    diag = {"anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12", "mission": 17906,
            "target_height": None, "records": [], "figures": []}
    for label, path in all_records:
        record = json.loads(path.read_text())
        diag["target_height"] = wave_height_target(record["spec"])
        env = envelope(record, label)
        diag["records"].append({
            "label": label,
            "record": str(path.relative_to(REPO)),
            "target_height": diag["target_height"],
            "half_height_mm": 0.5 * diag["target_height"] * 1e3,
            "onset": divergence_onset(record),
            "envelope": env,
            "growth": growth_rate(env["windows"]),
        })
    diag["figures"] = figures(diag)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(diag, indent=2, allow_nan=False))
    OUT_MD.write_text(markdown(diag, diag["figures"]))
    print(f"wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)}")
    for rec in diag["records"]:
        o = rec["onset"]
        print(f"  {rec['label']:16s} M_max {o['max_mach']:.2f}  "
              f"comp {100 * o['max_compression']:.1f}%  "
              f"Mach1@{o['mach_gt_one_at_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())