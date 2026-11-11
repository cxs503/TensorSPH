#!/usr/bin/env python
"""Render ``docs/benchmarks/wcsph-chain.md`` and the figures from the artefact.

Same contract as the rest of the repository: the document is a rendering of
``artifacts/wcsph_chain/report.json`` and every number in it is injected from
that file, so a stale number cannot survive in the text.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "wcsph_chain"
FIG = ART / "figures"
DOC = ROOT / "docs" / "benchmarks" / "wcsph-chain.md"


def _load() -> dict:
    path = ART / "report.json"
    if not path.exists():
        raise SystemExit(f"missing artefact {path}; run benchmarks/wcsph_chain/run.py assemble")
    return json.loads(path.read_text())


def fig_front(report: dict, path: Path) -> bool:
    front = report.get("front_2d")
    if not front:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    curve = front["curve"]
    ax.plot([c["tau"] for c in curve], [c["ref"] for c in curve], "ko",
            ms=5, label="Martin & Moyce (1952), Lethe transcription")
    for rung in front["ladder"]:
        dx = rung["settings"]["dx"]
        taus = [None] * 0
        del taus
        scale = math.sqrt(2.0 * 9.81 / 0.2)
        taus = [t * scale for t in rung["times"]]
        z = [(x - 0.2) / 0.2 for x in rung["front_toe"]]
        ax.plot(taus, z, "-", lw=1.2, label=f"SPH dx = {dx:g} m")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax.set_ylabel(r"$Z/a$")
    ax.set_title("Dam-break leading edge vs experiment")
    ax.set_xlim(0, 5.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_convergence(report: dict, path: Path) -> bool:
    conv = report.get("convergence", {}).get("pairs") or []
    if not conv:
        return False
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labels, values = [], []
    for row in conv:
        labels.append(row["label"])
        values.append(row["order"])
    ax.bar(labels, values, color="#3b6ea5")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="required order 1.0")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("observed order $p$ ($L_2 \\sim \\Delta x^p$)")
    ax.set_title("Depth-profile self-convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_sloshing(report: dict, path: Path) -> bool:
    lines = report.get("sloshing", {}).get("lines") or {}
    if not lines:
        return False
    n = len(lines)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (freq, series) in zip(axes, sorted(lines.items())):
        ax.plot(series["times"], series["p_wall_probe"], lw=1.2,
                label="SPH (dynamic boundary particles)")
        ax.plot(series["times"], series["linear_p_wall_probe"], "--", lw=1.0,
                label="linear potential flow")
        ax.set_ylabel("p at SWL [Pa]")
        ax.set_title(f"end-wall pressure, f = {freq} Hz")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_profile(report: dict, path: Path) -> bool:
    front = report.get("front_2d")
    if not front:
        return False
    ladder = front["ladder"]
    if not ladder:
        return False
    # the archived profiles live in the per-rung artefacts, not in report.json
    raw = sorted(ART.glob("dam2d_dx*.json"))
    if not raw:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for path_rung in raw:
        payload = json.loads(path_rung.read_text())
        key = sorted(payload.get("profiles", {}))[-1]
        prof = payload["profiles"][key]
        edges = payload["profile_edges"]
        xs = [edges[0] + (edges[1] - edges[0]) * (i + 0.5) / len(prof)
              for i in range(len(prof))]
        ax.plot(xs, prof, "-", lw=1.2,
                label=f"dx = {payload['settings']['dx']:g} m, t = {key} s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth [m]")
    ax.set_title("Free-surface profile at the last archived instant")
    ax.set_xlim(0, 1.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_pair(report: dict, path: Path) -> bool:
    raw = ART / "pair.json"
    if not raw.exists():
        return False
    payload = json.loads(raw.read_text())
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    e0 = payload["energy"][0]
    ax.plot(payload["times"], [(e - e0) / abs(e0) * 100 for e in payload["energy"]],
            lw=1.2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("(E - E0)/E0 [%]")
    ax.set_title("Particle-pair energy error (alpha_av = 0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def render(report: dict) -> str:
    prov = report.get("provenance", {})
    lines = [
        "# WCSPH dam-break + tank-sloshing verification chain",
        "",
        "**Generated file** -- every number below is injected from "
        "`artifacts/wcsph_chain/report.json` by `benchmarks/wcsph_chain/make_doc.py`.",
        "Do not edit by hand; rerun the generator instead.",
        "",
        f"* provenance: `{prov.get('git_branch')}` @ `{prov.get('git_sha')}`, "
        f"torch {prov.get('torch')}, python {prov.get('python')}, "
        f"{prov.get('dtype')}, generated {prov.get('generated')}",
        "",
        "## 1. What this chain measures",
        "",
        "A new weakly-compressible SPH solver (`src/tensorsph/wcsph/`) is verified "
        "bottom-up: a two-particle collision (the discrete invariants), a 2-D "
        "dam break against the Martin & Moyce (1952) experiment, the same column "
        "in 3-D, a forced rectangular tank against the linear potential-flow "
        "solution, and a resolution ladder for the observed convergence order.",
        "",
        "## 2. Settings",
        "",
        "| rung | setting | value |",
        "|---|---|---|",
    ]
    pair = report.get("pair")
    if pair:
        s = pair["settings"]
        lines += [f"| pair | v0 / c0 / alpha | {s['v0']} m/s / {s['c0']} m/s / "
                  f"{s['alpha_av']} |",
                  f"| pair | steps / dx | {s['steps']} / {s['dx']} m |"]
    front = report.get("front_2d")
    if front:
        for rung in front["ladder"]:
            s = rung["settings"]
            lines.append(f"| dam 2-D dx={s['dx']:g} | c0 / alpha / t_end | "
                         f"{s['c0']} m/s / {s['alpha_av']} / {s['t_end']} s |")
    slosh = report.get("sloshing")
    if slosh:
        for row in slosh["rows"]:
            lines.append(f"| sloshing f={row['frequency']:.1f} Hz | "
                         f"peak p / reference | {row['sph_peak_pa']:.0f} Pa / "
                         f"{row['linear_peak_pa']:.0f} Pa |")
    lines += ["", "## 3. Results", ""]
    gates = report.get("gates", [])
    lines += ["| criterion | measured | limit | verdict |", "|---|---|---|---|"]
    for g in gates:
        measured = g["measured"]
        shown = "n/a" if measured is None else (
            f"{measured:.3e}" if isinstance(measured, float) and abs(measured) < 1e-2
            else f"{measured:.4g}" if isinstance(measured, float) else str(measured))
        lines.append(f"| {g['criterion']} | {shown} | {g['limit']} "
                     f"({g['rule']}) | **{g['verdict']}** |")
    summary = report.get("summary", {})
    lines += ["", f"Overall: **{summary.get('verdict')}** "
                  f"({summary.get('pass', 0)} pass / {summary.get('fail', 0)} fail / "
                  f"{summary.get('withheld', 0)} withheld).", ""]
    if front:
        lines += ["### 3.1 Dam-break front vs experiment", "",
                  "| t [s] | τ | Z/a (SPH) | Z/a (exp) | rel. error |", "|---|---|---|---|---|"]
        for row in front["rows"]:
            rel = row["relative_error"]
            lines.append(f"| {row['t']} | {row['tau']:.3f} | "
                         f"{row['sim_Z_over_a']:.3f} | {row['ref_Z_over_a']:.3f} | "
                         f"{'n/a' if rel is None else f'{rel:+.1%}'} |")
        lines += ["", f"Whole-curve mean absolute error in `Z/a`: "
                      f"**{front['mae_over_curve']:.3f}**.", ""]
    conv = report.get("convergence", {}).get("pairs") or []
    if conv:
        lines += ["### 3.2 Convergence", "",
                  "Triple-grid order from the archived depth profiles "
                  "(`p = log(|f_h - f_h/2| / |f_h/2 - f_h/4|)/log 2`):", "",
                  "| rungs | t [s] | order |", "|---|---|---|"]
        for row in conv:
            order = "n/a" if row["order"] is None else f"{row['order']:.2f}"
            lines.append(f"| {row['label']} | {row['t']} | {order} |")
        lines.append("")
    if slosh:
        lines += ["### 3.3 Sloshing wall pressure", "",
                  "| f [Hz] | SPH peak [Pa] | linear peak [Pa] | rel. error | "
                  "SPH η amp [m] | linear η amp [m] |", "|---|---|---|---|---|---|"]
        for row in slosh["rows"]:
            rel = row["relative_error"]
            lines.append(f"| {row['frequency']:.1f} | {row['sph_peak_pa']:.0f} | "
                         f"{row['linear_peak_pa']:.0f} | "
                         f"{'n/a' if rel is None else f'{rel:+.1%}'} | "
                         f"{row['sph_eta_amplitude_m']:.4f} | "
                         f"{row['linear_eta_amplitude_m']:.4f} |")
        lines.append("")
    lines += ["## 4. Figures", ""]
    for name, caption in (
            ("front_trajectory.png", "Leading edge in the experiment's non-dimensional form."),
            ("convergence.png", "Depth-profile self-convergence order."),
            ("sloshing_pressure.png", "End-wall pressure: SPH vs linear theory."),
            ("dam_break_profile.png", "Free-surface profile at the last archived instant."),
            ("pair_conservation.png", "Particle-pair energy error with alpha = 0.")):
        if (FIG / name).exists():
            lines += [f"![{caption}](figures/{name})", "",
                      f"*{caption}*", ""]
    lines += ["## 5. Limitations", "",
              "* The sloshing reference is linear potential flow, not an experiment: "
              "no digitised tank-pressure experiment is bundled in the repository, "
              "so the experimental gate is reported as withheld, never as a pass.",
              "* The dam-break reference is a transcription of the 1952 figure via "
              "Lethe; its own uncertainty is a few percent of `a`, and the "
              "transcription has not been checked against the paywalled original.",
              "* Near the first sloshing resonance the linear reference is only "
              "formally valid (zero damping, unbounded growth); the comparison is "
              "quoted only with that caveat.",
              ""]
    return "\n".join(lines)


def main() -> int:
    report = _load()
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in (("front_trajectory.png", fig_front),
                     ("convergence.png", fig_convergence),
                     ("sloshing_pressure.png", fig_sloshing),
                     ("dam_break_profile.png", fig_profile),
                     ("pair_conservation.png", fig_pair)):
        try:
            if fn(report, FIG / name):
                made.append(name)
        except Exception as exc:  # pragma: no cover - figure best effort
            print(f"figure {name} failed: {exc}")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render(report))
    print(f"wrote {DOC.relative_to(ROOT)} and {len(made)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())