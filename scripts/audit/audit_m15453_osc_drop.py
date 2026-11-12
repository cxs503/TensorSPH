#!/usr/bin/env python3
"""Board audit of mission m15328 (CASE-OSC-DROP, TensorSPH) — CASE-OSC-DROP 交差核验.

This script is the *machine-checkable* part of the audit.  It does not re-do the
work; it re-derives the adjudication from raw artefacts:

1. compares the mission's committed campaign artefacts with an independent
   re-run of the recorded ``repro_cmd`` on this worktree (reproduction test),
2. recomputes the three benchmark gates *from the rows* instead of trusting the
   ``gates`` block of the committed report,
3. re-derives the analytic reference used by the campaign (2-D Rayleigh/Lamb
   branch ``omega^2 = n(n^2-1) sigma/(rho R^3)``) from the geometry, i.e. fits the
   exact curvature of ``r = R(1 + eps cos n theta)`` and checks the modal
   coefficient against ``(n^2-1) eps / R``; the 3-D branch
   ``n(n-1)(n+2) sigma/(rho R^3)`` is shown alongside because the report cites
   Rayleigh 1879 / Lamb Art. 275, which are the 3-D results,
4. probes the degenerate gate (C4) and the delta-function normalisation,
5. writes ``audit_summary.json`` + figures.

Usage::

    PYTHONPATH=src python scripts/audit/audit_m15453_osc_drop.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "runs/m15453_audit"          # this auditor's re-run
SOURCE = AUDIT / "_source_m15328"            # m15328 committed artefacts (copied)
FIGDIR = REPO / "docs/board_audit/m15453"

SIGMA = 0.072        # N/m, module constant SIGMA
RHO = 1000.0         # kg/m^3
RADIUS = 0.02        # m, lamb_case(radius=0.02)
EPS = 0.10
TOL_LAMB = 3.0       # tensorsph.surface_tension_dispersion.TOL_LAMB_PCT
TOL_WAVE = 5.0       # TOL_WAVE_PCT
MIN_ORDER = 2.0      # MIN_ORDER


# --------------------------------------------------------------------------- #
# 1. analytic reference, re-derived
# --------------------------------------------------------------------------- #
def exact_curvature_mode(n: int, eps: float = EPS, radius: float = RADIUS,
                         samples: int = 20001) -> dict:
    """Mode-n coefficient of the exact curvature of ``r = R (1 + eps cos n th)``.

    ``kappa = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^{3/2}`` in polar coordinates.
    Linearising ``kappa = (1/R) + (n^2 - 1) eps cos(n th) / R`` (the denominator
    contributes the ``-3 eps cos`` term); the fit below recovers ``n^2 - 1`` from
    the *geometric* curvature, so the reference used by ``drop_modal_stiffness``
    is verified against the geometry the code builds rather than against a
    remembered formula.  The pressure balance of the same perturbation,
    ``-i w rho A R^n = sigma (n^2-1) eta / R^2`` with ``eta = n A R^{n-1}/(i w)``,
    gives the 2-D branch ``omega^2 = n (n^2-1) sigma/(rho R^3)`` used by the code.
    """
    th = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    r = radius * (1.0 + eps * np.cos(n * th))
    rp = -radius * eps * n * np.sin(n * th)
    rpp = -radius * eps * n * n * np.cos(n * th)
    kappa = (r * r + 2.0 * rp * rp - r * rpp) / (r * r + rp * rp) ** 1.5
    # orthogonal projection onto {1, cos(n th), sin(n th)}
    basis = np.stack([np.ones_like(th), np.cos(n * th), np.sin(n * th)])
    coef, *_ = np.linalg.lstsq(basis.T, kappa, rcond=None)
    return dict(n=n, kappa_mean=float(coef[0]), kappa_mode_cos=float(coef[1]),
                kappa_prefactor=float(coef[1] / eps * radius),  # -> +(n^2-1)
                analytic_prefactor=n * n - 1.0,
                rel_err_pct=float(100.0 * (coef[1] / eps * radius / (n * n - 1.0) - 1.0)))


def omega_2d(n: int, sigma: float = SIGMA, radius: float = RADIUS) -> float:
    """2-D (cylindrical) Rayleigh/Lamb branch: omega^2 = n (n^2-1) sigma/(rho R^3)."""
    return math.sqrt(n * (n * n - 1.0) * sigma / (RHO * radius ** 3))


def omega_3d(n: int, sigma: float = SIGMA, radius: float = RADIUS) -> float:
    """3-D Rayleigh branch: omega^2 = n (n-1)(n+2) sigma/(rho R^3)."""
    return math.sqrt(n * (n - 1.0) * (n + 2.0) * sigma / (RHO * radius ** 3))


# --------------------------------------------------------------------------- #
# 2. gates recomputed from the rows
# --------------------------------------------------------------------------- #
def recompute_gates(rows: list[dict]) -> dict:
    wave = [abs(r["omega_error_pct"]) for r in rows if r["family"] == "capillary_wave"]
    lamb = [abs(r["omega_error_pct"]) for r in rows if r["family"] == "lamb"]
    c2 = max(lamb) if lamb else None
    return dict(
        C1_capillary_wave=dict(max_abs_error_pct=max(wave) if wave else None,
                               tolerance_pct=TOL_WAVE,
                               passed=bool(wave and max(wave) <= TOL_WAVE)),
        C2_lamb_modes=dict(max_abs_error_pct=c2, tolerance_pct=TOL_LAMB,
                           passed=bool(c2 is not None and c2 <= TOL_LAMB)),
        n_wave_rows=len(wave), n_lamb_rows=len(lamb),
    )


def order_of(rows: list[dict], key: str) -> float | None:
    """Least-squares slope of log|err| vs log(dx); dx = R/(R/dx)."""
    pts = [(r["radius_over_dx"], abs(r["omega_error_pct"])) for r in rows
           if r["case"].startswith("lamb") and r["omega_error_pct"] != 0.0]
    if len(pts) < 2:
        return None
    x = np.log([RADIUS / p[0] for p in pts])
    y = np.log([max(p[1], 1e-12) for p in pts])
    return float(np.polyfit(x, y, 1)[0])


# --------------------------------------------------------------------------- #
# 3. main
# --------------------------------------------------------------------------- #
def main() -> dict:
    src = json.loads((SOURCE / "report.json").read_text())
    aud = json.loads((AUDIT / "report.json").read_text())
    meta_src = json.loads((SOURCE / "meta.json").read_text())
    meta_aud = json.loads((AUDIT / "meta.json").read_text())

    # --- reproduction: same rows? ------------------------------------------
    key = lambda r: (r["case"], r.get("radius_over_dx"), r["family"])
    a = {key(r): r["omega_error_pct"] for r in aud["rows"]}
    b = {key(r): r["omega_error_pct"] for r in src["rows"]}
    common = set(a) & set(b)
    repro = dict(
        rows_auditor=len(a), rows_mission=len(b), rows_common=len(common),
        max_abs_diff_pct_points=float(max((abs(a[k] - b[k]) for k in common), default=float("nan"))),
        bitwise_identical=bool(common and all(a[k] == b[k] for k in common)),
        auditor_git=meta_aud.get("git_revision"),
        mission_git=meta_src.get("git_revision"),
        auditor_platform=meta_aud.get("platform"),
        mission_platform=meta_src.get("platform"),
    )

    # --- reference: 2-D (as coded) vs 3-D (as cited) ------------------------
    ref_check = [exact_curvature_mode(n) for n in (2, 3, 4)]
    lamb_rows = [r for r in src["rows"] if r["family"] == "lamb"]
    table = []
    for r in lamb_rows:
        n = r["mode"]
        w = r["omega_measured"]
        table.append(dict(
            mode=n, radius_over_dx=r["radius_over_dx"],
            omega_measured=w,
            omega_ref_2d=omega_2d(n), err_2d_pct=100.0 * (w / omega_2d(n) - 1.0),
            omega_ref_3d=omega_3d(n), err_3d_pct=100.0 * (w / omega_3d(n) - 1.0),
            tol_pct=TOL_LAMB,
            stiffness_ratio=r["stiffness_ratio"],
            stiffness_ratio_std=r.get("stiffness_ratio_std"),
            surface_measure_ratio=r.get("surface_measure_ratio"),
            kappa_mode_ratio=r.get("kappa_mode_ratio"),
        ))

    gates = recompute_gates(src["rows"])
    gates_committed = src["gates"]
    orders = {c: order_of([r for r in src["rows"] if r["case"] == c], "omega_error_pct")
              for c in sorted({r["case"] for r in src["rows"]})}

    # --- degenerate gate probe: C4 is exactly zero by construction ---------
    sig_ref = src.get("sigma_sensitivity", {})
    c4 = dict(committed=src["gates"]["C4_sigma_sensitivity"],
              spread_is_identically_zero=bool(src["gates"]["C4_sigma_sensitivity"]["max_spread_pct"] == 0.0),
              rows=[{k: r.get(k) for k in ("sigma", "omega_measured", "omega_theory",
                                           "omega_error_pct", "ratio_to_nominal")}
                    for r in (sig_ref.get("rows") or [])][:6],
              why="the CSF traction is linear in sigma and the measured omega is taken "
                  "as sqrt(stiffness_ratio)*omega_theory, so omega/sqrt(sigma) is "
                  "invariant: a 0.0 % spread is an algebraic identity, not evidence")

    # --- delta-function normalisation (Brackbill unit mass) -----------------
    dm = src.get("delta_measure", {})
    delta_rows = [dict(**r) for r in (dm.get("rows") or [])]
    lamb_measure = [r["surface_measure_ratio"] for r in lamb_rows]
    delta = dict(committed=dm, lamb_surface_measure_ratio_min=min(lamb_measure),
                 lamb_surface_measure_ratio_max=max(lamb_measure),
                 target=1.0,
                 deficit_pct=100.0 * (float(np.mean(lamb_measure)) - 1.0),
                 note="legacy normalisation leaves ~28 % of the interface mass "
                      "missing; the local_range variant overshoots (1.65) on the wave case")

    summary = dict(
        audit_of="m15328 CASE-OSC-DROP (TensorSPH)",
        auditor_mission=15453,
        audited_commit="b5280f37b50eab41096511e15fcdc2bcc1ff7d27",
        reproduced_commit=meta_aud.get("git_revision"),
        reproduces_on_main=False,
        reproduced_commit_note="b5280f3 is not an ancestor of the audited branch tip "
                               "(git merge-base --is-ancestor -> no); the audit re-ran the "
                               "same command on the current trunk worktree and got identical rows",
        repro=repro,
        reference_curvature_check=ref_check,
        lamb_table=table,
        gates_recomputed=gates,
        gates_committed={k: {kk: vv for kk, vv in v.items() if kk in
                             ("max_abs_error_pct", "min_order", "tolerance_pct", "passed")}
                         for k, v in gates_committed.items()},
        gates_agree_with_committed=bool(
            abs(gates["C1_capillary_wave"]["max_abs_error_pct"]
                - gates_committed["C1_capillary_wave"]["max_abs_error_pct"]) < 1e-9
            and abs(gates["C2_lamb_modes"]["max_abs_error_pct"]
                    - gates_committed["C2_lamb_modes"]["max_abs_error_pct"]) < 1e-9),
        convergence_orders_recomputed=orders,
        convergence_order_committed=src["gates"]["C3_convergence"]["orders"],
        convergence_required=MIN_ORDER,
        option_screen=src.get("option_screen_best"),
        degenerate_gate_c4=c4,
        delta_measure=delta,
        status_committed=src["status"],
    )

    FIGDIR.mkdir(parents=True, exist_ok=True)
    (AUDIT / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    figures = _figures(summary, FIGDIR)
    summary["figures"] = [str(p.relative_to(REPO)) for p in figures]
    (AUDIT / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _figures(s: dict, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    # fig 1: measured vs both analytic branches
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    rows = s["lamb_table"]
    modes = sorted({r["mode"] for r in rows})
    rdx = sorted({r["radius_over_dx"] for r in rows})
    colors = {rdx[0]: "tab:blue", rdx[len(rdx) // 2]: "tab:orange", rdx[-1]: "tab:green"}
    for r in rows:
        ax.plot(r["mode"], r["omega_measured"], "o", color=colors[r["radius_over_dx"]],
                alpha=0.85, label=f"SPH  R/dx={r['radius_over_dx']}")
    ax.plot(modes, [omega_2d(n) for n in modes], "k-", lw=2, label="2-D Rayleigh/Lamb (as coded)")
    ax.plot(modes, [omega_3d(n) for n in modes], "k--", lw=1.4, label="3-D Rayleigh/Lamb (as cited)")
    ax.set_xlabel("azimuthal mode  $n$")
    ax.set_ylabel(r"$\omega$  [rad/s]")
    ax.set_title("CASE-OSC-DROP: measured modal frequency vs analytic branches")
    ax.set_xticks(modes)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig1_omega_vs_mode.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # fig 2: relative error vs resolution with the 3 % tolerance band
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for n in modes:
        sel = [r for r in rows if r["mode"] == n]
        ax.plot([r["radius_over_dx"] for r in sel], [r["err_2d_pct"] for r in sel],
                "o-", label=f"n={n}  vs 2-D ref")
        ax.plot([r["radius_over_dx"] for r in sel], [r["err_3d_pct"] for r in sel],
                "s:", alpha=0.6, label=f"n={n}  vs 3-D ref")
    ax.axhspan(-TOL_LAMB, TOL_LAMB, color="tab:green", alpha=0.12,
               label=f"gate tolerance ±{TOL_LAMB:.0f} %")
    ax.set_xlabel(r"resolution  $R/\Delta x$")
    ax.set_ylabel("frequency error  [%]")
    ax.set_title("CASE-OSC-DROP: no resolution/mode inside the 3 % gate")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig2_error_vs_resolution.png"
    fig.savefig(p); plt.close(fig); paths.append(p)

    # fig 3: the two diagnostics that carry the diagnosis
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=140)
    ax = axes[0]
    for r in rows:
        ax.plot(r["mode"], 100.0 * r["stiffness_ratio"], "o",
                color=colors[r["radius_over_dx"]], alpha=0.85)
    ax.axhline(100.0, color="k", ls="--", lw=1.2, label="ratio = 1 (exact)")
    ax.axhline(97.0, color="tab:green", ls=":", lw=1.0)
    ax.axhline(103.0, color="tab:green", ls=":", lw=1.0, label="±3 % band")
    ax.set_xlabel("mode $n$"); ax.set_ylabel("modal stiffness  [% of analytic]")
    ax.set_title("modal stiffness deficit (0.65–0.99)"); ax.set_xticks(modes)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(modes, [100.0 * float(np.mean([r["surface_measure_ratio"] for r in rows
                                           if r["mode"] == n])) for n in modes], "o-")
    ax.axhline(100.0, color="k", ls="--", lw=1.2, label="Brackbill unit mass")
    ax.set_xlabel("mode $n$"); ax.set_ylabel(r"$\int\delta_s\,dV /A$  [%]")
    ax.set_title("surface delta mass (target 100 %)"); ax.set_xticks(modes)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig3_diagnostics.png"
    fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: out[k] for k in ("repro", "gates_recomputed",
                                          "gates_agree_with_committed",
                                          "convergence_required")}, indent=1))
    for row in out["lamb_table"]:
        print(f"n={row['mode']} R/dx={row['radius_over_dx']:>2} got={row['omega_measured']:.5f} "
              f"ref2d={row['omega_ref_2d']:.5f} err2d={row['err_2d_pct']:+.2f}% "
              f"ref3d={row['omega_ref_3d']:.5f} err3d={row['err_3d_pct']:+.2f}%")
    print("figures:", out["figures"])