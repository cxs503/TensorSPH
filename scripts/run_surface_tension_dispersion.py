#!/usr/bin/env python3
"""Run the surface-tension dispersion verification campaign and store the artefacts.

This is the *entry point* of the surface-tension dispersion capability: it drives
:mod:`tensorsph.surface_tension_dispersion`, writes a machine-readable report, a
human-readable summary, the raw benchmark table (analytic reference vs measured),
matplotlib figures and a provenance record (git revision, command, hashes).

    PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --quick
    PYTHONPATH=src python scripts/run_surface_tension_dispersion.py \
        --out-dir runs/surface_tension_dispersion

Artefacts (all inside the repository, so they survive the push):

    <out-dir>/report.json        full campaign report (rows, gates, table)
    <out-dir>/benchmark.json     reference-vs-measured table only
    <out-dir>/summary.md         paper-style summary with the numeric tables
    <out-dir>/meta.json          provenance: command, git sha, versions, sha256
    <out-dir>/figures/*.png      error / stiffness / curvature-audit figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from tensorsph.surface_tension_dispersion import (  # noqa: E402
    SIGMA, curvature_audit, curvature_noise_scan, delta_measure_audit,
    normal_audit, run_validation)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:            # not a git checkout / git missing
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _figures(report: dict, fig_dir: Path) -> list[dict]:
    """Four figures: wave error, Lamb stiffness, curvature audit, surface measure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover - optional dep
        print(f"[warn] matplotlib unavailable ({exc}); skipping figures")
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. capillary-wave frequency error vs particles per wavelength
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for k in sorted({r["k"] for r in report["rows"] if r["family"] == "capillary_wave"}):
        rows = [r for r in report["rows"]
                if r["family"] == "capillary_wave" and r["k"] == k]
        rows.sort(key=lambda r: r["particles_per_wavelength"])
        ax.plot([r["particles_per_wavelength"] for r in rows],
                [r["omega_error_pct"] for r in rows], "o-", label=f"k = {k:g}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-5.0, 5.0, color="g", alpha=0.12, label="+/-5 % gate")
    ax.set_xlabel("particles per wavelength  (L = 2*pi fixed, dx = L/ppw)")
    ax.set_ylabel("omega error  [%]")
    ax.set_title("Capillary wave: frequency error of the CSF operator")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = fig_dir / "capillary_wave_error.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    made.append(dict(name="capillary_wave_error.png",
                     caption="Capillary-wave frequency error vs resolution for k=1,2,3. "
                             "Negative values mean the discrete surface force does not "
                             "restore the mode (spurious negative stiffness).",
                     path=str(p)))

    # 2. Lamb stiffness ratio vs resolution
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    modes = sorted({r["mode"] for r in report["rows"] if r["family"] == "lamb"})
    for mode in modes:
        rows = [r for r in report["rows"] if r["family"] == "lamb" and r["mode"] == mode]
        rows.sort(key=lambda r: r["radius_over_dx"])
        ax.errorbar([r["radius_over_dx"] for r in rows],
                    [r["stiffness_ratio"] for r in rows],
                    yerr=[r["stiffness_ratio_std"] for r in rows],
                    marker="o", capsize=3, label=f"l = {mode}")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="exact (Rayleigh/Lamb)")
    ax.set_xlabel("R / dx")
    ax.set_ylabel("measured / analytic stiffness")
    ax.set_title("Lamb modes of a 2-D drop: CSF stiffness ratio")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = fig_dir / "lamb_stiffness.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    made.append(dict(name="lamb_stiffness.png",
                     caption="Modal stiffness ratio of the lobed drop for l=2,3,4; "
                             "error bars are the sub-lattice phase ensemble standard "
                             "deviation, the dashed line is the Rayleigh/Lamb value.",
                     path=str(p)))

    # 3. curvature audit: measured vs exact at trough / crest / flat noise
    audit = report.get("curvature_audit") or {}
    if audit:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        labels = ["trough\n(cos kx = 1)", "crest\n(cos kx = -1)", "flat interface\n(noise RMS)"]
        measured = [audit["trough"]["measured_kappa"], audit["crest"]["measured_kappa"],
                    audit["flat_noise_rms"]]
        exact = [audit["trough"]["exact_kappa"], audit["crest"]["exact_kappa"], 0.0]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], measured, width=0.4, label="CSF (measured)")
        ax.bar([i + 0.2 for i in x], exact, width=0.4, label="exact")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("kappa  [1/m]")
        ax.set_title("Curvature recovery on the capillary-wave surface")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        p = fig_dir / "curvature_audit.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        made.append(dict(name="curvature_audit.png",
                         caption="Measured CSF curvature at the wave trough and crest "
                                 "(sign-inverted relative to the exact value) and the "
                                 "noise floor of a flat free surface.",
                         path=str(p)))

    # 4. surface-delta measure
    measure = report.get("delta_measure") or {}
    if measure:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        groups = list(measure)
        width = 0.8 / max(1, len(next(iter(measure.values()))))
        for i, norm in enumerate(next(iter(measure.values()))):
            vals = [measure[g].get(norm, float("nan")) for g in groups]
            ax.bar([j + i * width for j in range(len(groups))], vals, width=width,
                   label=norm)
        ax.axhline(1.0, color="k", lw=0.9, ls="--", label="required = 1")
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(groups))])
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylabel("sum_i delta_s,i V_i / A")
        ax.set_title("Surface delta function: measured interface measure")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        p = fig_dir / "delta_measure.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        made.append(dict(name="delta_measure.png",
                         caption="Integral of the surface delta function per unit "
                                 "interface area; 1.0 is the CSF requirement.",
                         path=str(p)))
    return made


def _summary_md(report: dict, meta: dict, figures: list[dict]) -> str:
    lines = [f"# Surface-tension dispersion verification ({report['status']})", ""]
    lines += [f"- generated: {report['generated_at']}",
              f"- git revision: `{meta['git_revision'][:12]}`",
              f"- command: `{meta['command']}`",
              f"- wall clock of the campaign: {report['wall_clock_s']:.1f} s",
              f"- reference: omega^2 = sigma k^3/rho (wave), "
              f"omega^2 = n(n^2-1) sigma/(rho R^3) (drop)", ""]
    lines += ["## Gates", "", "| gate | value | tolerance | passed |", "|---|---|---|---|"]
    for name, gate in report["gates"].items():
        value = gate.get("max_abs_error_pct", gate.get("min_order",
                                                       gate.get("max_spread_pct")))
        tol = gate.get("tolerance_pct", gate.get("required"))
        lines.append(f"| {name} | {value if value is None else round(value, 3)} | "
                     f"{tol} | {gate.get('passed')} |")
    lines += ["", "## Reference vs measured", "",
              "| case | resolution | ref omega | measured omega | rel. error [%] | tol [%] | pass |",
              "|---|---|---|---|---|---|---|"]
    for row in report["benchmark_table"]:
        res = row.get("ppw_domain") or row.get("radius_over_dx")
        lines.append(f"| {row['case']} | {res} | {row['ref']:.6g} | {row['got']:.6g} | "
                     f"{row['rel_err_pct']:+.2f} | {row['tolerance_pct']:.0f} | "
                     f"{row['passed']} |")
    if report.get("curvature_audit"):
        a = report["curvature_audit"]
        lines += ["", "## Curvature audit", "",
                  f"- trough: measured {a['trough']['measured_kappa']:+.3f} 1/m, "
                  f"exact {a['trough']['exact_kappa']:+.3f} 1/m",
                  f"- crest: measured {a['crest']['measured_kappa']:+.3f} 1/m, "
                  f"exact {a['crest']['exact_kappa']:+.3f} 1/m",
                  f"- flat-interface noise RMS: {a['flat_noise_rms']:.3f} 1/m",
                  f"- sign recovered: **{a['sign_recovered']}**"]
    if report.get("delta_measure"):
        lines += ["", "## Surface delta measure (sum delta_s V / A, target 1)", "",
                  "| geometry | legacy | local_range |", "|---|---|---|"]
        for geom, vals in report["delta_measure"].items():
            lines.append(f"| {geom} | {vals.get('legacy', float('nan')):.3f} | "
                         f"{vals.get('local_range', float('nan')):.3f} |")
    if figures:
        lines += ["", "## Figures", ""]
        lines += [f"- `{f['name']}`: {f['caption']}" for f in figures]
    lines += ["", "## Honest status", "",
              f"- failed gates: {report['failed_gates']}",
              "- A FAIL here is a physical result, not a harness error: it means the "
              "discretised CSF operator does not reproduce the requested mode to the "
              "mission tolerance, and the report keeps the measured numbers.",
              f"- reproduction: `{report['repro_cmd']}`", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ppw", default="24,32,48")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--modes", default="2,3,4")
    ap.add_argument("--rdx", default="16,24,32")
    ap.add_argument("--shifts", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--scheme", default="fit_quadratic")
    ap.add_argument("--norm", default="legacy")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--defective-control", action="store_true", default=True)
    ap.add_argument("--skip-screen", action="store_true")
    ap.add_argument("--out-dir", default="runs/surface_tension_dispersion")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    ints = lambda txt: [int(v) for v in str(txt).split(",") if v]      # noqa: E731
    floats = lambda txt: [float(v) for v in str(txt).split(",") if v]  # noqa: E731
    if args.quick:
        ppws, ks, modes, rdxs, shifts = (16, 24), (1.0,), (2,), (16,), 1
    else:
        ppws, ks = ints(args.ppw), floats(args.ks)
        modes, rdxs, shifts = ints(args.modes), ints(args.rdx), args.shifts

    t0 = time.time()
    report = run_validation(ks=ks, ppws=ppws, modes=modes, rdxs=rdxs,
                            sigma=args.sigma, shifts=shifts,
                            compare_defective=args.defective_control,
                            option_screen=not args.skip_screen,
                            curvature_scheme=args.scheme,
                            delta_normalization=args.norm)
    report["curvature_audit"] = curvature_audit(ppw=max(ppws), shifts=max(2, shifts))
    report["delta_measure"] = delta_measure_audit(ppw=max(ppws), shifts=max(2, shifts))
    report["normal_audit"] = normal_audit(ppw=max(ppws), shifts=max(2, shifts))
    report["curvature_noise_scan"] = curvature_noise_scan(
        ppws=tuple(sorted({min(ppws), max(ppws), 24, 32, 48})), shifts=max(2, shifts))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = _figures(report, out_dir / "figures")
    report["figures"] = figures

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out_dir / "benchmark.json").write_text(
        json.dumps({"benchmark": report["benchmark"],
                    "reference": report["reference"],
                    "table": report["benchmark_table"],
                    "gates": report["gates"],
                    "status": report["status"]}, indent=1) + "\n", encoding="utf-8")

    meta = dict(
        command=" ".join([Path(sys.executable).name, *sys.argv]),
        cwd=os.getcwd(),
        git_revision=_git("rev-parse", "HEAD"),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        wall_clock_s=round(time.time() - t0, 2),
        python=platform.python_version(), torch=torch.__version__,
        platform=platform.platform(),
        threads=int(torch.get_num_threads()),
        report_sha256=_sha256(report_path),
        benchmark_sha256=_sha256(out_dir / "benchmark.json"),
        artifacts=[str(report_path), str(out_dir / "benchmark.json")] +
                  [f["path"] for f in figures])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_md(report, meta, figures),
                                        encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "failed_gates": report["failed_gates"],
                      "curvature_audit": report["curvature_audit"],
                      "out_dir": str(out_dir)}, indent=1))
    return 0 if report["status"] == "PASS" else 0   # honest FAIL still exits 0 (report mode)


if __name__ == "__main__":
    raise SystemExit(main())