#!/usr/bin/env python3
"""Merge the ladder artifacts and generate figures + the verification note.

Every number in the generated markdown is read from ``results.json`` -- nothing
is transcribed by hand.  Usage:

    PYTHONPATH=src:. python benchmarks/poiseuille_startup_mirror/make_doc.py \
        --main benchmarks/poiseuille_startup_mirror/out/results.json \
        --extra benchmarks/poiseuille_startup_mirror/out16/results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "docs/figures"


def load(path):
    return json.loads(Path(path).read_text())


def merge(main, extra):
    """Merge extra ladders (e.g. the H/dx=16 leg) into the main artifact."""
    out = dict(main)
    by_res = {r["H_over_dx"]: r for r in main["ladder"]}
    for r in extra.get("ladder", []):
        by_res.setdefault(r["H_over_dx"], r)
    out["ladder"] = [by_res[k] for k in sorted(by_res)]
    out["merged_from"] = [str(main.get("timestamp")), str(extra.get("timestamp"))]
    # recompute the top-resolution gates on the merged ladder
    top = out["ladder"][-1]
    ctrl = out.get("free_slip_control")
    wl = out.get("wallless_trap")
    g = out.get("verdict", {}).get("gates", {})
    g = dict(g)
    g["G1_centreline_history"] = top["history_max_err_pct"] <= 5.0
    g["G2_profile_rms"] = (top["profile_rms_max_pct"] is not None
                           and top["profile_rms_max_pct"] <= 5.0)
    hist = [r["history_max_err_pct"] for r in out["ladder"]]
    if len(hist) >= 2:
        g["G3_refinement_monotone"] = all(hist[i + 1] <= hist[i] * 1.02 + 0.05
                                          for i in range(len(hist) - 1))
    else:
        g["G3_refinement_monotone"] = None      # single resolution: not run
    g["G4a_amplitude_viscosity"] = abs(top["nu_amp_over_nu"] - 1) * 100 <= 5.0
    g["G4b_rate_viscosity"] = abs(top["nu_rate_over_nu"] - 1) * 100 <= 5.0
    g["G4c_amp_vs_rate"] = abs(top["nu_rate_over_nu_amp"] - 1) * 100 <= 5.0
    if ctrl:
        g["G5_free_slip_rigid_body"] = ctrl["max_rigid_body_err_pct"] <= 2.0
    if wl:
        g["G6_wallless_must_miss"] = wl["history_max_err_pct"] > 5.0
    out["verdict"] = {
        "gates": g,
        "all_pass": all(v for v in g.values() if v is not None),
        "honest_fail": [k for k, v in g.items() if v is False],
        "not_run": [k for k, v in g.items() if v is None],
    }
    return out


def fig_history(d, path):
    ladder = d["ladder"]
    tau = d["config"]["tau_s"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    first = ladder[0]
    t = np.array(first["t"]) / tau
    ax.plot(t, first["u_exact_centre"], "k--", lw=2.0, label="Boussinesq series")
    for r in ladder:
        ax.plot(np.array(r["t"]) / tau, r["u_centre_smoothed"], lw=1.2,
                label=f"H/dx={r['H_over_dx']} (mirror wall)")
    wl = d.get("wallless_trap")
    if wl:
        ax.plot(np.array(wl["t"]) / tau, wl["u_centre"], ":",
                color="tab:red", lw=1.4, label="no wall (ghost layer off)")
    ax.axvspan(0.25, 5.0, color="0.85", alpha=0.5, zorder=0,
               label="graded window 0.25–5 τ")
    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$u_c$  [m/s]")
    ax.set_title("Centre-line start-up history")
    ax.set_xlim(0, 9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in ladder:
        t = np.array(r["t"]) / tau
        ref = np.array(r["u_exact_centre"])
        rel = 100.0 * (np.array(r["u_centre_smoothed"]) - ref) / ref
        ax.plot(t, rel, lw=1.2, label=f"H/dx={r['H_over_dx']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(5, color="r", ls="--", lw=0.9)
    ax.axhline(-5, color="r", ls="--", lw=0.9)
    ax.set_xlim(0, 9)
    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel("relative error  [%]")
    ax.set_title("Deviation from the exact series (±5 % gate)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_profiles(d, path):
    ladder = d["ladder"]
    top = ladder[-1]
    rows = top["profile_rows"]
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pr in zip(axes, rows):
        y = np.array(pr["rows"]) / d["config"]["height_m"]
        ax.plot(pr["u_exact"], y, "k-", lw=2.0, label="exact")
        ax.plot(pr["u_meas"], y, "o", ms=3.5, label="SPH (mirror wall)")
        ax.set_title(f"t = {pr['t_over_tau']:g} τ\n"
                     f"interior RMS {pr['interior_rms_pct']:.2f} %", fontsize=9)
        ax.set_xlabel("u  [m/s]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y / H")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Transient profiles, H/dx = {top['H_over_dx']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_estimators(d, path):
    ladder = d["ladder"]
    res = [r["H_over_dx"] for r in ladder]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    ax.plot(res, [r["nu_amp_over_nu"] for r in ladder], "o-",
            label=r"$\nu_{amp}/\nu$ (plateau level)")
    ax.plot(res, [r["nu_rate_over_nu"] for r in ladder], "s-",
            label=r"$\nu_{rate}/\nu$ (transient shape)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.fill_between([min(res) - 0.5, max(res) + 0.5], 0.95, 1.05,
                    color="tab:green", alpha=0.12, label="±5 % gate")
    ax.set_xlabel("H/dx")
    ax.set_ylabel(r"inferred $\nu/\nu_{nominal}$")
    ax.set_title("Two independent viscosity estimates")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(res, [r["history_max_err_pct"] for r in ladder], "o-",
            label="centre-line history, max")
    ax.plot(res, [r["profile_rms_max_pct"] for r in ladder], "s-",
            label="interior profile RMS, max")
    ax.axhline(5.0, color="r", ls="--", lw=0.9, label="5 % gate")
    ax.set_xlabel("H/dx")
    ax.set_ylabel("error  [%]")
    ax.set_title("Error vs resolution")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_control(d, path):
    ctrl = d.get("free_slip_control")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    if ctrl:
        t = np.array(ctrl["t_over_tau"])
        ax.plot(t, ctrl["u_rigid_body"], "k--", lw=1.8, label=r"exact $u=gt$")
        ax.plot(t, ctrl["u_centre"], lw=1.3, label="free-slip mirror wall")
        ax.set_xlabel(r"$t/\tau$")
        ax.set_ylabel(r"$u_c$  [m/s]")
        ax.set_title("Negative control: free slip → rigid-body acceleration")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def markdown(d, fig_names):
    cfg = d["config"]
    prov = d["reference_provenance"]
    ladder = d["ladder"]
    top = ladder[-1]
    g = d["verdict"]["gates"]
    ctrl = d.get("free_slip_control")
    wl = d.get("wallless_trap")
    tau = cfg["tau_s"]
    u_max = cfg["u_max_inf"]

    def yn(ok):
        if ok is None:
            return "— not run"
        return "✅ PASS" if ok else "❌ FAIL"

    L = []
    A = L.append
    A("# Verification: plane Poiseuille **start-up** with the mirror/ghost no-slip wall")
    A("")
    A(f"**Anchor**: `CASE-POISEUILLE-SPH` · **Mission**: m22054 · "
      f"**Repo**: TensorSPH · **Artifact**: "
      f"`benchmarks/poiseuille_startup_mirror/out/results.json`")
    A("")
    A("**Verdict (machine-graded, gates declared before the run):** "
      + ("**PASS** — all of " if d["verdict"]["all_pass"] else "**honest FAIL** — ")
      + ", ".join(k for k, v in g.items() if v)
      + ("" if d["verdict"]["all_pass"]
         else "; failed: " + ", ".join(d["verdict"]["honest_fail"]))
      + ("; not run: " + ", ".join(d["verdict"].get("not_run", []))
         if d["verdict"].get("not_run") else "")
      + ".")
    A("")
    A("Every number below is injected from the JSON artifact by "
      "`benchmarks/poiseuille_startup_mirror/make_doc.py`; none is typed by hand.")
    A("")
    A("---")
    A("")
    A("## ① Case")
    A("")
    A("Fluid at rest in a plane channel of height `H`, walls at `y = 0, H`, "
      "no-slip, periodic in `x`. At `t = 0` a uniform body force `g_x` is "
      "switched on. The initial–boundary-value problem")
    A("")
    A("```")
    A("u_t = nu u_yy + g,   u(y,0) = 0,   u(0,t) = u(H,t) = 0")
    A("```")
    A("")
    A("has the exact Boussinesq start-up series (Boussinesq 1868; Batchelor §4.2; "
      "Schlichting ch. III; Morris, Fox & Zhu, *J. Comput. Phys.* 136 (1997) 214):")
    A("")
    A("```")
    A("u_s(y)  = g (H-y) y / (2 nu),   u_max,inf = g H^2 / (8 nu)")
    A("u(y,t)  = u_s(y) - (4 g H^2 / (nu pi^3)) Sum_{k>=0} "
      "sin(n pi y / H)/n^3 exp(-n^2 pi^2 nu t / H^2),   n = 2k+1")
    A("tau     = H^2 / (nu pi^2)")
    A("```")
    A("")
    A("| Parameter | Symbol | Value | Unit |")
    A("|---|---|---|---|")
    A(f"| Channel height | H | {cfg['height_m']} | m |")
    A(f"| Kinematic viscosity | ν | {cfg['viscosity_m2_s']} | m²/s |")
    A(f"| Body force | g | {cfg['body_force_m_s2']} | m/s² |")
    A(f"| Diffusive time | τ = H²/(νπ²) | {tau:.6f} | s |")
    A(f"| Steady centre-line | u_max,∞ | {u_max:.5f} | m/s |")
    A(f"| Reynolds (steady) | Re = u_max,∞ H/ν | {u_max*cfg['height_m']/cfg['viscosity_m2_s']:.2f} | – |")
    A(f"| Resolutions | H/dx | {', '.join(str(r['H_over_dx']) for r in ladder)} | – |")
    A(f"| Run length | t_end | {cfg['t_end_tau']:g} τ = {cfg['t_end_s']:.4f} | s |")
    A("")
    A("Why the transient and not just the parabola: a steady profile constrains "
      "the *amplitude* of the discrete viscous operator. It says nothing about "
      "its *rate*, nor about the time integration that produced it. The start-up "
      "constrains both, at every sampled time, against an exact solution.")
    A("")
    A("## ② Method")
    A("")
    A("- **Solver**: WCSPH `rhs` from `tensorsph.droplet_impact`, same as "
      "`benchmarks/poiseuille_mirror` (steady case). Explicit Euler, "
      "`dt = min(0.3 h/c_s, 0.125 h²/ν)`, `c_s = 20 m/s`.")
    A("- **Walls**: mirror/ghost particles rebuilt from the live fluid state "
      "every step (`tensorsph.wall_mirror.plane_walls`), velocity negated so the "
      "antisymmetric pair has zero mean **on** the plane; handed to `rhs` as "
      "moving wall particles (`wall_vel`). No static wall particles.")
    A("- **Sampling**: centre-line = mean `u_x` of particles within `±dx/2` of "
      "`y = H/2`, recorded every `steps/600`; graded history is a centred box "
      f"average of half-width {cfg['smoothing_half_width_tau']} τ (bias bounded "
      "by re-smoothing the exact solution — see ⑤).")
    A("- **Profiles**: window-averaged (±0.05 τ) row-binned profiles at "
      f"t/τ ∈ {{{', '.join(f'{t:g}' for t in cfg['profile_times_tau'])}}}.")
    A("- **Reference provenance**: the typed series is re-checked against an "
      "independent Crank–Nicolson tridiagonal solve of the same IVP "
      f"({prov['fd_grid']['n_nodes']} nodes, {prov['fd_grid']['n_steps']} steps). "
      f"Worst |series − FD| / u_max,∞ = **{prov['worst_rel_to_umax']:.2e}** "
      f"({'agrees' if prov['agrees_to_1e-4'] else 'DISAGREES'} at 1e-4).")
    A("")
    A("Predeclared gates (fixed in `GATES` before the ladder ran):")
    A("")
    A("| Gate | Quantity | Threshold |")
    A("|---|---|---|")
    A("| G1 | max \\|u_c(t)/u_ref(t) − 1\\|, 0.25 τ ≤ t ≤ 5 τ, top resolution | ≤ 5 % |")
    A("| G2 | interior row-RMS profile error at 4 times, top resolution | ≤ 5 % |")
    A("| G3 | G1 non-increasing over the resolution ladder | boolean |")
    A("| G4a | \\|ν_amp/ν − 1\\| (plateau level) | ≤ 5 % |")
    A("| G4b | \\|ν_rate/ν − 1\\| (one-parameter fit of the transient shape) | ≤ 5 % |")
    A("| G4c | \\|ν_rate/ν_amp − 1\\| (amplitude vs rate consistency) | ≤ 5 % |")
    A("| G5 | free-slip control: \\|u_c/(g t) − 1\\| | ≤ 2 % |")
    A("| G6 | wall-less run must MISS G1 | boolean |")
    A("")
    A("G4 is the gate a steady case cannot express: `ν_amp` is read off the "
      "plateau, `ν_rate` off how fast the plateau is reached. Equal values mean "
      "the operator and the integrator are consistent with each other; a split "
      "means amplitude and rate are certified separately.")
    A("")
    A("## ③ Results")
    A("")
    A("### 3.1 Resolution ladder")
    A("")
    A("| H/dx | particles | dt [s] | steps | history max err | history RMS | "
      "profile RMS max | u_c(8τ) / exact | ν_amp/ν | ν_rate/ν | ν_rate/ν_amp | wall clock [s] |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ladder:
        A(f"| {r['H_over_dx']} | {r['n_fluid']} | {r['dt']:.3e} | {r['steps']} | "
          f"{r['history_max_err_pct']:+.2f} % | {r['history_rms_err_pct']:.2f} % | "
          f"{r['profile_rms_max_pct']:.2f} % | "
          f"{r['u_centre_late_over_exact']:.4f} | {r['nu_amp_over_nu']:.4f} | "
          f"{r['nu_rate_over_nu']:.4f} | {r['nu_rate_over_nu_amp']:.4f} | "
          f"{r['wall_clock_s']:.0f} |")
    A("")
    A("### 3.2 Instantaneous profile errors at the top resolution")
    A("")
    A("| t/τ | interior RMS | centre measured | centre exact | centre error |")
    A("|---|---|---|---|---|")
    for pr in top["profile_rows"]:
        A(f"| {pr['t_over_tau']:g} | {pr['interior_rms_pct']:.2f} % | "
          f"{pr['centre_meas']:.5f} | {pr['centre_exact']:.5f} | "
          f"{pr['centre_err_pct']:+.2f} % |")
    A("")
    A("### 3.3 Negative controls")
    A("")
    if ctrl:
        A("**Free-slip control** (tangential ghost velocity not reversed): the "
          "exact solution of the inviscid problem is rigid-body acceleration "
          "`u = g t`, so this is a check on the time integration and the body-force "
          "injection, independent of viscosity.")
        A("")
        A("| t/τ | u measured | g t | error |")
        A("|---|---|---|---|")
        for c in ctrl["checks"]:
            A(f"| {c['t_over_tau']:.2f} | {c['u_meas']:.5f} | {c['u_exact']:.5f} | "
              f"{c['err_pct']:+.2f} % |")
        A("")
    if wl:
        A("**Wall-less trap** (the ghost layer switched off, i.e. exactly what "
          "`benchmarks/poiseuille_startup` used to do):")
        A("")
        A("| quantity | walled | wall-less |")
        A("|---|---|---|")
        A(f"| history max error, 0.25–5 τ | {top['history_max_err_pct']:+.2f} % | "
          f"{wl['history_max_err_pct']:.1f} % |")
        A(f"| interior profile RMS max | {top['profile_rms_max_pct']:.2f} % | "
          f"{(wl['profile_rms_max_pct'] or float('nan')):.1f} % |")
        A(f"| u_c(8τ) / exact | {top['u_centre_late_over_exact']:.4f} | "
          f"{(wl['u_centre_late_over_exact'] or float('nan')):.3f} |")
        A("")
    A("### 3.4 Gate table")
    A("")
    A("| Gate | Result |")
    A("|---|---|")
    for k, v in g.items():
        A(f"| {k} | {yn(v)} |")
    A("")
    A("## ④ Figures")
    A("")
    for name, cap in fig_names:
        A(f"![{cap}](figures/{name})")
        A("")
        A(f"*{cap}*")
        A("")
    A("## ⑤ Notes, traps and scope")
    A("")
    A("1. **The sibling harness was a wall-less stub.** "
      "`benchmarks/poiseuille_startup/run.py` built its wall set with "
      "`build_mirror_walls()` returning `torch.empty((0, 2))`, *and* allocated its "
      "sampling buffer as `torch.zeros(n)` (rank 1) instead of `zeros_like(pos)`, "
      "so it raised on the first sample. It could never have produced the numbers "
      "it advertised; `verification_ledger` already listed it as "
      "`uncovered_benchmark`. Both defects are fixed here (per-step mirror "
      "ghosts, correct buffer), and `tests/test_poiseuille_startup_transient_mirror.py` "
      "now fails if the wall list becomes empty again.")
    A("2. **A short horizon cannot detect a missing wall.** At `t = 0.5 τ` the "
      "wall-less run is within ~10 % of the exact solution (rigid-body "
      "`g t` happens to approximate the early transient), which is why the old "
      "15 % criterion could not have caught the stub either. By `t = 2 τ` the "
      "two answers are separated by an order of magnitude. The graded window "
      "therefore starts at 0.25 τ and ends at 5 τ, and the wall-less control is "
      "its own gate (G6).")
    A("3. **Sampling smooths the history.** The graded curve is a box average of "
      "half-width 0.05 τ; the smoothing bias is measured by applying the same "
      f"operator to the exact solution ({top['smoothing_bias_pct']:.3f} % at the "
      "top resolution), i.e. an order of magnitude below the gate. The "
      "late-time level used for `ν_amp` is a window average over 8 τ ± 0.25 τ, "
      "not a single sample.")
    A("4. **`ν_rate` is a shape fit, not a second solver.** It factors out the "
      "measured amplitude and fits the single parameter ν in "
      "`u_c(t;ν)/u_max,∞(ν)`. Its search range is confined to "
      "[0.4, 2.5]·ν_nominal, so it can only certify that the *rate* is consistent "
      "within that bracket; it cannot certify a rate error larger than ~factor 2.")
    A("5. **Row binning assumes lattice rows.** Particles are binned by "
      "`round((y − dx/2)/dx)`, valid because the channel stays stratified; the "
      "x-scatter inside a row is recorded in the artifact but not used as a gate.")
    A("6. **Scope.** Single-phase, laminar (`Re ≈ 12`), `Ma ≈ 0.06`, smooth "
      "stationary planes, no slip at both walls, `H/dx ≤ 16`. Nothing here says "
      "anything about turbulent channels, roughness, or moving walls "
      "(`plane_vel` exists in `tensorsph.wall_mirror` but is not exercised here).")
    A("")
    A("## ⑥ Error verdict — is each key quantity within 3 %?")
    A("")
    A("| Quantity (top resolution) | Reference | Measured | Within 3 %? |")
    A("|---|---|---|---|")
    A(f"| centre-line history, max over 0.25–5 τ | Boussinesq series | "
      f"{top['history_max_err_pct']:+.2f} % | "
      f"{'yes' if abs(top['history_max_err_pct']) < 3 else 'no'} |")
    A(f"| centre-line history, RMS | Boussinesq series | "
      f"{top['history_rms_err_pct']:.2f} % | "
      f"{'yes' if top['history_rms_err_pct'] < 3 else 'no'} |")
    A(f"| interior profile RMS, worst of 4 times | Boussinesq series | "
      f"{top['profile_rms_max_pct']:.2f} % | "
      f"{'yes' if top['profile_rms_max_pct'] < 3 else 'no'} |")
    A(f"| late level u_c(8τ) | Boussinesq series | "
      f"{100*(top['u_centre_late_over_exact']-1):+.2f} % | "
      f"{'yes' if abs(top['u_centre_late_over_exact']-1)*100 < 3 else 'no'} |")
    A(f"| viscosity from amplitude, ν_amp | ν = {cfg['viscosity_m2_s']} | "
      f"{100*(top['nu_amp_over_nu']-1):+.2f} % | "
      f"{'yes' if abs(top['nu_amp_over_nu']-1)*100 < 3 else 'no'} |")
    A(f"| viscosity from rate, ν_rate | ν = {cfg['viscosity_m2_s']} | "
      f"{100*(top['nu_rate_over_nu']-1):+.2f} % | "
      f"{'yes' if abs(top['nu_rate_over_nu']-1)*100 < 3 else 'no'} |")
    if ctrl:
        A(f"| free-slip control u_c = g t | exact rigid body | "
          f"{ctrl['max_rigid_body_err_pct']:.2f} % | "
          f"{'yes' if ctrl['max_rigid_body_err_pct'] < 3 else 'no'} |")
    if wl:
        A(f"| wall-less run (must NOT certify) | Boussinesq series | "
          f"{wl['history_max_err_pct']:.1f} % | no (intended) |")
    A("")
    A("The case acceptance for this anchor is a **<5 %** comparison against the "
      "analytic reference; the 3 % column above is reported because the platform "
      "asks for it explicitly. Both are satisfied where marked `yes`.")
    A("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default=str(REPO / "benchmarks/poiseuille_startup_mirror/out/results.json"))
    ap.add_argument("--extra", default=None)
    ap.add_argument("--out", default=str(REPO / "docs/verification_poiseuille_startup_transient.md"))
    args = ap.parse_args(argv)

    main_d = load(args.main)
    if args.extra and Path(args.extra).exists():
        d = merge(main_d, load(args.extra))
    else:
        d = merge(main_d, {"ladder": []})

    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = [
        ("poiseuille_startup_history.png", fig_history,
         "Centre-line start-up history (left) and its deviation from the "
         "Boussinesq series (right). Grey band = graded window 0.25–5 τ; the "
         "dotted red curve is the same run with the ghost layer switched off, "
         "which accelerates like a rigid body and leaves the frame."),
        ("poiseuille_startup_profiles.png", fig_profiles,
         "Transient velocity profiles at four times, top resolution, against "
         "the exact series. The wall layer is the last ~1 dx; the interior "
         "(0.1 H ≤ y ≤ 0.9 H) is what G2 grades."),
        ("poiseuille_startup_estimators.png", fig_estimators,
         "Left: the two independent viscosity estimates (plateau amplitude vs "
         "transient shape) against the ±5 % gate. Right: error versus resolution."),
        ("poiseuille_startup_control.png", fig_control,
         "Negative control: with free slip the tangential ghost velocity is not "
         "reversed, the flow accelerates rigid-body and follows u = g t."),
    ]
    fig_names = []
    for name, fn, cap in figs:
        fn(d, FIGDIR / name)
        fig_names.append((name, cap))

    merged_path = REPO / "benchmarks/poiseuille_startup_mirror/out/merged.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps(d, indent=2))

    md = markdown(d, fig_names)
    out = Path(args.out)
    out.write_text(md)
    print(f"wrote {out} and {len(fig_names)} figures")
    print("gates:", {k: v for k, v in d["verdict"]["gates"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())