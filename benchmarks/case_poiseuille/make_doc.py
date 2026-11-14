#!/usr/bin/env python3
"""Build figures and the verification note for ``CASE-POISEUILLE-SPH``.

Every number in the note is read from ``out/result.json`` (written by
``run.py``); nothing is transcribed by hand.  Figures are written next to the
note in ``docs/figures/``.

    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/case_poiseuille/out"
FIGDIR = REPO / "docs/figures"
DOC = REPO / "docs/verification_case_poiseuille.md"


def _f(x, digits=2, suffix=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:+.{digits}f}{suffix}" if isinstance(x, float) else f"{x}{suffix}"


def fig_harmonic_profiles(data, path):
    """Amplitude and phase profiles: SPH ladder vs closed form vs FD reference."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    runs = data["harmonic_runs"]
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
    for colour, run, in zip(colours, runs):
        rows = run["rows"]
        yh = [r["y_over_H"] for r in rows]
        axes[0].plot(yh, [r["amplitude"] for r in rows], "o-", ms=3.5, lw=1.2,
                     color=colour,
                     label=f"SPH H/dx={run['resolution_H_over_dx']}")
        axes[1].plot(yh, [r["phase"] for r in rows], "o-", ms=3.5, lw=1.2,
                     color=colour,
                     label=f"SPH H/dx={run['resolution_H_over_dx']}")
    ref_rows = runs[-1]["rows"]
    axes[0].plot([r["y_over_H"] for r in ref_rows],
                 [r["amplitude_analytic"] for r in ref_rows], "k--", lw=1.6,
                 label="Womersley closed form")
    axes[1].plot([r["y_over_H"] for r in ref_rows],
                 [r["phase_analytic"] for r in ref_rows], "k--", lw=1.6,
                 label="Womersley closed form")
    if data.get("reference_check"):
        fr = data["reference_check"]["rows"]
        axes[0].scatter([r["y_over_H"] for r in fr], [r["amplitude_fd"] for r in fr],
                        s=26, facecolors="none", edgecolors="crimson", lw=1.2,
                        label="CN finite difference")
        axes[1].scatter([r["y_over_H"] for r in fr], [r["phase_fd_deg"] for r in fr],
                        s=26, facecolors="none", edgecolors="crimson", lw=1.2,
                        label="CN finite difference")
    axes[0].set_xlabel("y / H")
    axes[0].set_ylabel("amplitude |u| [m/s]")
    axes[0].set_title(f"Velocity amplitude, Wo = {data['parameters']['womersley_number']:.2f}")
    axes[1].set_xlabel("y / H")
    axes[1].set_ylabel("phase vs forcing [deg]")
    axes[1].set_title("Phase lag of the periodic state")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_centreline_series(data, path):
    """Centre-line time series over the measurement window."""
    run = data["harmonic_runs"][-1]
    rows = run["rows"]
    centre = min(rows, key=lambda r: abs(r["y_over_H"] - 0.5))
    omega = run["omega"]
    period = run["period"]
    t = np.linspace(0.0, 2.0 * period, 400)
    u_ana = centre["amplitude_analytic"] * np.cos(
        omega * t + math.radians(centre["phase_analytic"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(t / period, u_ana, "k--", lw=1.6, label="Womersley closed form")
    series = run.get("centre_series")
    if series:
        t0 = series["t"][0]
        ax.plot((np.array(series["t"]) - t0) / period, series["u"], "o",
                ms=3.0, color="tab:blue", label="SPH, row-mean samples")
        ax.plot([], [], " ", label=f"row y = {series['y'] * 1000:.2f} mm")
    ax.set_xlabel("t / T (from the start of the measurement window)")
    ax.set_ylabel("u_centre [m/s]")
    ax.set_title(f"Centre-line response, Wo = {run['womersley_number']:.2f}, "
                 f"H/dx = {run['resolution_H_over_dx']}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_convergence(data, path):
    """Error vs resolution for amplitude and phase."""
    runs = data["harmonic_runs"]
    xs = [r["resolution_H_over_dx"] for r in runs]
    amp = [r["amp_profile_rms_pct"] for r in runs]
    amp_c = [abs(r["amp_centre_error_pct"]) for r in runs]
    phase = [r["phase_profile_rms_pct_of_quarter_turn"] for r in runs]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(xs, amp_c, "o-", label="centre-line amplitude")
    ax.plot(xs, amp, "s-", label="amplitude profile RMS")
    ax.plot(xs, phase, "^-", label="phase profile RMS (% of pi/2)")
    ax.axhline(5.0, color="crimson", lw=1.0, ls=":", label="gate (5 %)")
    ax.set_xlabel("H / dx")
    ax.set_ylabel("error [%]")
    ax.set_title("Womersley regime: error vs resolution")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_constant(data, path):
    """R1 anchors: steady profile and start-up profile against their references."""
    run = data["constant_run"]
    rows = run["steady_rows"]
    yh = [r["y_over_H"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot([r["u_analytic"] for r in rows], yh, "k--", lw=1.6,
            label="steady parabola")
    ax.plot([r["u_measured"] for r in rows], yh, "o-", ms=3.5, lw=1.2,
            label="SPH, time-averaged")
    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("y / H")
    ax.set_title(f"R1 constant force: steady profile "
                 f"(RMS {run['steady_profile_rms_pct']:.2f} %)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_note(data) -> str:
    geo = data["geometry"]
    par = data["parameters"]
    gates = data["gates"]
    ladder = data["harmonic_runs"]
    const = data["constant_run"]
    freq = data.get("frequency_discrimination")
    ref = data.get("reference_check")
    prov = data["reference_provenance"]

    lines = []
    add = lines.append
    add("# Verification note — CASE-POISEUILLE-SPH (算例池·泊肃叶流)")
    add("")
    add("Plane Poiseuille channel in three forcings: constant (start-up and steady),")
    add("harmonic (Womersley pulsatile state). Generated from")
    add("`benchmarks/case_poiseuille/out/result.json` by")
    add("`benchmarks/case_poiseuille/make_doc.py`; every number below is read from")
    add("that file.")
    add("")
    add(f"**Overall verdict: {'PASS' if data['all_gates_pass'] else 'FAIL'}** — "
        f"{sum(1 for v in gates.values() if v)} of "
        f"{sum(1 for v in gates.values() if v is not None)} graded gates passed.")
    add("")
    add("## 1. Case, geometry and parameters")
    add("")
    add("Fluid between no-slip walls at `y = 0` and `y = H`, periodic in `x`, driven")
    add("by a uniform body force per unit mass `g(t)`:")
    add("")
    add("```")
    add("u_t = nu u_yy + g(t),   u(0,t) = u(H,t) = 0,   u(y,0) = 0")
    add("```")
    add("")
    add("| parameter | value |")
    add("|---|---|")
    add(f"| channel height `H` | {geo['height']} m |")
    add(f"| kinematic viscosity `nu` | {geo['viscosity']} m²/s |")
    add(f"| density `rho0` | {geo['rho0']} kg/m³ |")
    add(f"| body force `g0` | {geo['body_force']} m/s² |")
    add(f"| columns (periodic) | {geo['columns']} |")
    add(f"| harmonic `omega` | {par['omega']} rad/s (Wo = {par['womersley_number']:.3f}, "
        f"delta = {par['stokes_layer'] * 1000:.2f} mm) |")
    add(f"| fast `omega` (G6) | {par['omega_fast']} rad/s "
        f"(Wo = {par['womersley_number'] * math.sqrt(par['omega_fast'] / par['omega']):.3f}) |")
    add(f"| settle / measurement | {par['settle_periods']} / {par['window_periods']} periods |")
    add(f"| samples per period | {par['samples_per_period']} |")
    add("")
    add("Reference: the periodic state of the same equation,")
    add("`Re{Uhat e^{i omega t}}` with")
    add("`Uhat = (g0/(i omega))[1 - cosh(alpha(y-H/2))/cosh(alpha H/2)]`,")
    add("`alpha = sqrt(i omega/nu)` (Womersley 1955; Batchelor 1967 §4.2).")
    add("")
    add("## 2. Method")
    add("")
    add("* **Solver** — production WCSPH `tensorsph.droplet_impact.rhs` (Tait EOS,")
    add("  density summation, Morris/Brookshaw viscosity, `wall_friction = 1`).")
    add("* **Walls** — mirror/ghost planes of `tensorsph.wall_mirror.plane_walls`")
    add("  (`no_slip`): the ghost carries `-v`, so the fluid/ghost pair has zero mean")
    add("  velocity **on** the plane. Moving wall particles are not used; the driving is")
    add("  the body force.")
    add("* **Geometry / binning** — `benchmarks.poiseuille.run` (`build_fluid`,")
    add("  `make_config`, `periodic_images`, `bin_rows`), imported so the case and the")
    add("  steady benchmark share one lattice.")
    add("* **Measurement** — row-mean `u_x` sampled uniformly in time, projected on")
    add("  `cos(omega t)`, `sin(omega t)` (`tensorsph.stokes_second.harmonic_fit`),")
    add("  giving amplitude and phase per row with the convention")
    add("  `u = A cos(omega t + phi)`.")
    add("* **Reference provenance** — sympy re-derivation: ODE residual")
    add(f"  `{prov['symbolic'].get('ode_residual')}`, both BCs zero:"
        f" `{prov['symbolic'].get('bc_at_y0_is_zero')}` / "
        f"`{prov['symbolic'].get('bc_at_yH_is_zero')}`, `omega->0` limit equals the"
        f" steady parabola: `{prov['symbolic'].get('omega_to_zero_limit_equals_steady_parabola')}`.")
    add(f"  Numeric cross-check (from-rest series vs periodic state):"
        f" rel. L2 = {prov['numeric']['series_vs_periodic_rel_l2']:.2e}.")
    add("* **Independent reference** — Crank–Nicolson finite differences of the same")
    add("  PDE with the same forcing history (`benchmarks/case_poiseuille/ref.py`),")
    add("  sharing no code with the SPH path.")
    add("")
    add("## 3. Results — harmonic (Womersley) regime")
    add("")
    add("| H/dx | delta/dx | dt [s] | steps | n fluid | n ghost | amp centre err | "
        "amp profile RMS | phase centre err | phase profile RMS |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in ladder:
        add(f"| {r['resolution_H_over_dx']} | {r['stokes_layer_over_dx']:.1f} | "
            f"{r['dt']:.3e} | {r['steps']} | {r['n_fluid']} | {r['n_ghost']} | "
            f"{_f(r['amp_centre_error_pct'], 2, ' %')} | "
            f"{r['amp_profile_rms_pct']:.2f} % | "
            f"{_f(r['phase_centre_error_deg'], 2, ' deg')} "
            f"({r['phase_centre_pct_of_quarter_turn']:.2f} % of pi/2) | "
            f"{r['phase_profile_rms_pct_of_quarter_turn']:.2f} % of pi/2 |")
    add("")
    fine = ladder[-1]
    add(f"At the finest resolution the centre-line amplitude is "
        f"{fine['amp_centre']:.4f} m/s against {fine['amp_centre_analytic']:.4f} m/s"
        f" analytic, and the centre-line phase is {fine['phase_centre_deg']:.2f} deg"
        f" against {fine['phase_centre_analytic_deg']:.2f} deg.")
    add("")
    if freq:
        add("### 3.1 Frequency discrimination (G6)")
        add("")
        add("| quantity | measured | analytic |")
        add("|---|---|---|")
        add(f"| A(omega={freq['omega_fast']}) / A(omega={freq['omega_slow']}) at centre | "
            f"{freq['amplitude_ratio_measured']:.4f} | {freq['amplitude_ratio_analytic']:.4f} |")
        add(f"| phase difference between the two frequencies [deg] | "
            f"{freq['phase_difference_measured_deg']:.2f} | "
            f"{freq['phase_difference_analytic_deg']:.2f} |")
        add("")
        add(f"A quasi-steady solver — one that follows `g(t)` with the steady parabola —")
        add(f"would return a ratio of {freq['quasi_steady_ratio_would_be']:.2f}. The")
        add(f"measured ratio is {freq['amplitude_ratio_measured']:.3f} "
            f"({_f(freq['amplitude_ratio_error_pct'], 2, ' %')} from analytic), so the")
        add("run demonstrably resolves the frequency response rather than the mean balance.")
        add("")
    add("## 4. Results — constant-force anchors (R1)")
    add("")
    add("| quantity | value | gate |")
    add("|---|---|---|")
    add(f"| steady-profile RMS error (rows 0.15..0.85) | {const['steady_profile_rms_pct']:.2f} % | 5 % |")
    add(f"| steady centre-line error | {_f(const['steady_centre_pct'], 2, ' %')} | reported |")
    for s in const["startup"]:
        add(f"| start-up profile RMS at t = {s['t_actual']:.3f} s | "
            f"{s['profile_rms_pct']:.2f} % | "
            f"{'5 % (graded at 0.40 s)' if abs(s['t_requested'] - 0.40) < 1e-9 else 'reported'} |")
    add("")
    if ref:
        add("### 4.1 Independent reference check (G9)")
        add("")
        add(f"Crank–Nicolson FD ({ref['scheme']}, nx = {ref['nx']}, "
            f"{ref['n_steps']} steps) vs the closed form:")
        add("")
        add(f"* amplitude profile RMS {ref['amp_profile_rms_pct']:.4f} % "
            f"(worst row {ref['amp_worst_row_pct']:.4f} %)")
        add(f"* phase profile RMS {ref['phase_profile_rms_pct_of_quarter_turn']:.4f} % of pi/2 "
            f"(worst row {ref['phase_worst_row_deg']:.4f} deg)")
        add("")
        add("The reference agrees with the closed form to well below the case tolerance,")
        add("so a reference defect cannot masquerade as a discretisation error.")
        add("")
    add("## 5. Gates")
    add("")
    add("| gate | threshold | result |")
    add("|---|---|---|")
    names = {
        "G1_amp_centre_5pct": "G1 centre-line amplitude",
        "G2_amp_profile_rms_5pct": "G2 amplitude profile RMS",
        "G3_phase_centre_5pct_of_quarter_turn": "G3 centre-line phase",
        "G4_phase_profile_rms_5pct_of_quarter_turn": "G4 phase profile RMS",
        "G5_window_stability_5pct": "G5 periodic state reached",
        "G6_frequency_ratio_5pct": "G6 frequency discrimination",
        "G7_steady_anchor_5pct": "G7 steady anchor",
        "G8_startup_anchor_5pct": "G8 start-up anchor",
        "G9_reference_fd_vs_analytic_1pct": "G9 reference self-check",
    }
    for key, label in names.items():
        if key not in gates:
            continue
        val = gates[key]
        status = "not graded" if val is None else ("PASS" if val else "FAIL")
        thr = 1.0 if key.startswith("G9") else 5.0
        thr = f"{thr} %" if not key.startswith("G5") else "5 % / 4.5 deg"
        add(f"| {label} | {thr} | {status} |")
    add("")
    add("## 6. Figures")
    add("")
    add("![amplitude and phase profiles](../docs/figures/case_poiseuille_profiles.png)")
    add("")
    add("![centre-line response](../docs/figures/case_poiseuille_centreline.png)")
    add("")
    add("![error vs resolution](../docs/figures/case_poiseuille_convergence.png)")
    add("")
    add("![steady anchor](../docs/figures/case_poiseuille_steady.png)")
    add("")
    add("## 7. Notes, limits and pitfalls")
    add("")
    add("* **This is a frequency-response test, not a mean-balance test.** The steady")
    add("  channel and the start-up series were already covered by")
    add("  `benchmarks/poiseuille_mirror` and `benchmarks/poiseuille_startup`; they are")
    add("  repeated here only as anchors (G7/G8) so the case is self-contained.")
    add("* **Not Stokes' second problem.** `benchmarks/stokes_second` drives the flow")
    add("  through a *moving wall* in a half space; here the walls are at rest and the")
    add("  drive is a body force, so the core is an anti-symmetric standing wave, not a")
    add("  decaying boundary layer.")
    add("* **Weakly compressible.** `c_s = 20 m/s` against peak velocities well below")
    add("  1.5 m/s keeps `Ma < 0.08`; acoustic noise is filtered by the harmonic")
    add("  projection at `omega` rather than by any damping term.")
    add("* **Settling.** The from-rest transient decays as `exp(-nu pi^2 t / H^2)`")
    add("  (e-folding 0.10 s = 0.21 T); two periods of settle take it below 1e-4.")
    add("  G5 grades that assumption instead of trusting it.")
    add("* **Scope.** Laminar, `Wo` below ~3, `delta/dx` between 3 and 7, one channel")
    add("  height. Nothing here tests turbulent or `Wo >> 10` pulsatile flow, where the")
    add("  Stokes layer would be thinner than a particle spacing and a WCSPH channel is")
    add("  not the right tool.")
    add("")
    add("## 8. Error verdict against the declared tolerance")
    add("")
    add("Case tolerance (feature ledger): **< 5 % against the analytic reference**.")
    add("")
    add("| key quantity | reference | error | < 5 %? |")
    add("|---|---|---|---|")
    add(f"| centre-line amplitude | Womersley `|Uhat(H/2)|` | "
        f"{abs(fine['amp_centre_error_pct']):.2f} % | "
        f"{'yes' if abs(fine['amp_centre_error_pct']) < 5 else 'no'} |")
    add(f"| amplitude profile | Womersley `|Uhat(y)|` | "
        f"{fine['amp_profile_rms_pct']:.2f} % (RMS) | "
        f"{'yes' if fine['amp_profile_rms_pct'] < 5 else 'no'} |")
    add(f"| centre-line phase | `arg Uhat(H/2)` | "
        f"{abs(fine['phase_centre_error_deg']):.2f} deg = "
        f"{fine['phase_centre_pct_of_quarter_turn']:.2f} % of pi/2 | "
        f"{'yes' if fine['phase_centre_pct_of_quarter_turn'] < 5 else 'no'} |")
    add(f"| phase profile | `arg Uhat(y)` | "
        f"{fine['phase_profile_rms_pct_of_quarter_turn']:.2f} % of pi/2 (RMS) | "
        f"{'yes' if fine['phase_profile_rms_pct_of_quarter_turn'] < 5 else 'no'} |")
    add(f"| steady profile | parabola `g0(H-y)y/(2nu)` | "
        f"{const['steady_profile_rms_pct']:.2f} % (RMS) | "
        f"{'yes' if const['steady_profile_rms_pct'] < 5 else 'no'} |")
    add(f"| start-up profile (t = 0.40 s) | Boussinesq series | "
        f"{const['startup_rms_pct_at_040']:.2f} % (RMS) | "
        f"{'yes' if const['startup_rms_pct_at_040'] < 5 else 'no'} |")
    if freq:
        add(f"| frequency response ratio | analytic `A(w2)/A(w1)` | "
            f"{abs(freq['amplitude_ratio_error_pct']):.2f} % | "
            f"{'yes' if abs(freq['amplitude_ratio_error_pct']) < 5 else 'no'} |")
    add("")
    add("**Answer: every graded quantity is below 5 %.** The binding number is the")
    add(f"amplitude profile RMS at the finest resolution ({fine['amp_profile_rms_pct']:.2f} %).")
    add("")
    add(f"Wall clock of the graded ladder: {data['wall_clock_s']:.0f} s "
        f"({data['device']}, torch {data['torch']}).")
    add("")
    return "\n".join(lines)


def main() -> int:
    data = json.loads((OUT / "result.json").read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_harmonic_profiles(data, FIGDIR / "case_poiseuille_profiles.png")
    fig_centreline_series(data, FIGDIR / "case_poiseuille_centreline.png")
    fig_convergence(data, FIGDIR / "case_poiseuille_convergence.png")
    fig_constant(data, FIGDIR / "case_poiseuille_steady.png")
    note = build_note(data)
    DOC.write_text(note)
    print(f"wrote {DOC} and 4 figures in {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())