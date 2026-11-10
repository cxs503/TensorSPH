"""Figures for the verification ladder (matplotlib, Agg backend).

Every figure is produced from the ladder JSON, so a figure can never show a
number that the document does not carry. All four are written as PNG into the
artefact directory and are embedded in the published report as base64.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _anchor(doc: dict, anchor_id: str) -> dict:
    for anchor in doc["anchors"]:
        if anchor["id"] == anchor_id:
            return anchor
    raise KeyError(anchor_id)


def fig_hydrostatic(doc: dict, out: Path) -> Path:
    anchor = _anchor(doc, "C1_hydrostatic")
    bands = anchor["measured"]["bands"]
    ys = [b["y"] for b in bands]
    p = [b["pressure"] for b in bands]
    p_ref = [b["pressure_analytic"] for b in bands]
    free = [b["free_surface"] for b in bands]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax.plot(p_ref, ys, "k-", label="analytic $p=\\rho g h$")
    ax.plot([v for v, f in zip(p, free) if not f],
            [v for v, f in zip(ys, free) if not f], "o", ms=6,
            label="SPH (interior bands, gated)")
    ax.plot([v for v, f in zip(p, free) if f],
            [v for v, f in zip(ys, free) if f], "s", mfc="none", ms=8,
            label="free-surface band (excluded)")
    ax.set_xlabel("pressure [Pa]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Rung C1 hydrostatic: max interior error "
                 f"{anchor['error_pct']:.3f} %")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    err = [b["rel_error_pct"] for b in bands]
    labels = [f"{b['y']:.2f}" for b in bands]
    colors = ["tab:red" if b["free_surface"] else "tab:blue" for b in bands]
    ax2.bar(range(len(bands)), err, color=colors)
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xticks(range(len(bands)))
    ax2.set_xticklabels(labels, rotation=60, fontsize=7)
    ax2.set_ylabel("relative error [%]")
    ax2.set_xlabel("band centre y [m] (red = free-surface band)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_tgv(doc: dict, out: Path) -> Path:
    anchor = _anchor(doc, "C2_tgv")
    series = anchor["series"]
    t = series["time"]
    e = series["energy"]
    ea = series["energy_analytic"]
    e0 = e[0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax.semilogy(t, [v / e0 for v in e], "o", ms=3, label="SPH")
    ax.semilogy(t, [v / e0 for v in ea], "k-", lw=1.6,
                label=f"exact $e^{{-4\\nu k^2 t}}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$E(t)/E(0)$")
    rate = anchor["measured"]["energy_decay_rate"]
    rate_a = anchor["reference"]["energy_decay_rate"]
    ax.set_title(f"Rung C2 TGV: rate {rate:.4f} vs {rate_a:.4f} 1/s "
                 f"({anchor['error_pct']:.2f} %)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    lo, hi = anchor["measured"]["fit_window"]
    ax2.axvspan(lo, hi, color="0.9", label="fit window")
    ax2.plot(t, [math.log(max(v, 1e-300) / e0) for v in e], "o", ms=3,
             label="ln E (simulation)")
    ax2.plot(t, [math.log(max(v, 1e-30) / e0) for v in ea], "k-", lw=1.4,
             label="analytic slope")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("ln(E/E0)")
    ax2.set_title("log-linear fit quality "
                  f"$R^2$={anchor['measured']['energy_decay_rate_r2']:.5f}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_dambreak(doc: dict, out: Path) -> Path:
    from .dambreak import load_martin_moyce
    anchor = _anchor(doc, "C3_dambreak")
    series = anchor["series"]
    tau = series["tau"]
    za = series["Z_over_a"]
    table = load_martin_moyce()["data"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax.plot([d["tau"] for d in table], [d["Z_over_a"] for d in table], "ks-",
            ms=4, lw=1.0, label="Martin & Moyce (1952), transcribed")
    ax.plot(tau, za, "o-", ms=3, lw=1.2, label="TensorSPH")
    ritter = [(r["tau"], r["Z_over_a_ritter"]) for r in series["ritter"]]
    if ritter:
        ax.plot([r[0] for r in ritter], [r[1] for r in ritter], "r--", lw=1.0,
                label="Ritter (1892) shallow-water front")
        t_valid = anchor["config"]["ritter_validity_window"]
        ax.axvline(t_valid * math.sqrt(2.0 * 9.81 / anchor["config"]["column_width"]),
                   color="0.4", ls=":", lw=1.0, label="Ritter validity limit")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax.set_ylabel("$Z/a$")
    ax.set_title(f"Rung C3: Ritter gate {anchor['error_pct']:.2f} %, "
                 f"experiment MAE "
                 f"{anchor['measured']['experimental_mae_Z_over_a']:.3f} $Z/a$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ratio = [(r["tau"], r["ratio"]) for r in series["ritter"]]
    ax2.plot([r[0] for r in ratio], [r[1] for r in ratio], "o-", ms=3)
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel("$Z_{SPH}/Z_{Ritter}$")
    ax2.set_title("Front ratio to the shallow-water front")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_controls(doc: dict, out: Path) -> Path:
    controls = {c["id"]: c for c in doc["negative_controls"]}
    labels, values = [], []
    if "C4a_gravity_off" in controls:
        labels.append("gravity off\n(hydrostatic err %)")
        values.append(controls["C4a_gravity_off"]["measured"]["error_pct"])
    if "C4b_viscosity_off" in controls:
        labels.append("viscosity off\n(rate / analytic %)")
        values.append(controls["C4b_viscosity_off"]["measured"]["ratio"] * 100.0)
    if "C4c_surface_tension" in controls:
        labels.append("sigma on vs off\n(spread response %)")
        values.append(controls["C4c_surface_tension"]["measured"]["spread_response_pct"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = ["tab:red" if abs(v) >= 50 else "tab:orange" for v in values]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("response [%]")
    ax.set_title("Negative controls: mechanism switched off")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_all(doc: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "C1_hydrostatic": fig_hydrostatic(doc, out_dir / "fig1_hydrostatic_profile.png"),
        "C2_tgv": fig_tgv(doc, out_dir / "fig2_tgv_energy_decay.png"),
        "C3_dambreak": fig_dambreak(doc, out_dir / "fig3_dambreak_front.png"),
        "controls": fig_controls(doc, out_dir / "fig4_negative_controls.png"),
    }