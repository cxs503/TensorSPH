#!/usr/bin/env python3
"""CASE-SPH-CONDUCTION-1D -- 1-D transient heat conduction with the SPH operator vs erfc.

The case
--------
Two semi-infinite plates (``T_hot`` for ``x < 0``, ``T_cold`` for ``x > 0``) are
put in perfect thermal contact at ``t = 0``.  The field is self-similar::

    T(x,t) = T_cold + (T_hot - T_cold)/2 * erfc(eta),   eta = x / (2 sqrt(alpha t))

the classical error-function solution for two semi-infinite media in contact
(Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed. 1959, Ch. II, §2.6).
The SPH side is the repository's existing Cleary & Monaghan (1999, J. Comput.
Phys. **148**:227-264, doi:10.1006/jcph.1998.6118) conduction operator
(:mod:`tensorsph.ice_thermal.sph_thermal`, reused unchanged), driven here by
:mod:`tensorsph.heat_conduction`.

Non-dimensionalisation
----------------------
``l = 1 m``, ``alpha = 1 m^2/s`` (``rho = cp = 1`` so ``k = alpha``); the case
time is the Fourier/similarity time ``t* = alpha t / l^2 = 0.5``.

What this harness measures
--------------------------
1. **Main gate** -- max ``|T_sph - T_erfc| / dT`` over ``|x| <= 4 m`` (2.8
   similarity lengths).  Gate: 3 %.
2. **Resolution ladder** in two calibration modes -- the repository's
   ``lattice_diffusion_factor`` (mean of the three longest periodic modes) and
   its ``k -> 0`` limit -- with measured orders.  This is where the interesting
   number of the case lives: the 3-mode mean sits 4.1e-4 below the ``k -> 0``
   symbol, and that gap shows up as a *residual diffusivity* (an error plateau
   independent of ``dx``) rather than as a truncation error.
3. **Parameter sensitivities** -- ``h/dx``, the Courant factor of the explicit
   step, and a domain-doubling check that the mirror-ghost walls are not the
   error term.
4. **Operator equivalence** -- the ``O(N)`` lattice stencil used in the time loop
   against the full ``O(N^2)`` ``conduction_rhs`` pair sum (round-off), plus one
   short run on the dense path.
5. **Independent reference** -- a Crank-Nicolson finite-difference solution of
   the same PDE (``scipy.linalg.solve_banded``) sharing no code with the SPH
   path: it is what proves the erfc profile is the right target.

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_conduction_1d/run.py [--figure] [--quick]

Writes ``result.json`` (machine-readable run evidence; every number in the
report comes from it) and, with ``--figure``, ``profile.png`` /
``convergence.png``.  Exit status is non-zero when a gate fails.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)   # tiny tensors: thread launch overhead dominates otherwise

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.heat_conduction import (  # noqa: E402
    ContactConfig,
    contact_temperature,
    fd_contact_reference,
    lattice_stencil,
    lattice_symbol_ratio,
    run_contact_sph,
    stencil_vs_dense,
)
from tensorsph.ice_thermal.sph_thermal import lattice_diffusion_factor  # noqa: E402

CASE = "CASE-SPH-CONDUCTION-1D"
GATE = 0.03                    # 3 % on the normalised max temperature error
X_CUT = 4.0                    # m, 2.8 similarity lengths at t* = 0.5
ALPHA = 1.0
T_HOT, T_COLD = 1.0, 0.0
FOURIER = 0.5
DOMAIN = 12.0                  # m, x in [-6, 6]
DX_LADDER = (0.08, 0.04, 0.02, 0.01)
DX_FINEST_K0 = 0.005

CITATION = {
    "primary": ("Carslaw, H.S. & Jaeger, J.C. (1959) Conduction of Heat in Solids, "
                "2nd ed., Oxford University Press, Ch. II (linear flow of heat in the "
                "infinite and semi-infinite solid), §2.6: two semi-infinite media "
                "initially at different uniform temperatures, put in contact -- the "
                "erf/erfc similarity solution used here as the reference."),
    "secondary": ("Cleary, P.W. & Monaghan, J.J. (1999) Conduction Modelling Using "
                  "Smoothed Particle Hydrodynamics, J. Comput. Phys. 148:227-264, "
                  "doi:10.1006/jcph.1998.6118 -- source of the SPH conduction operator."),
    "provenance": ("Reference values are *computed* from the closed form by code "
                   "(tensorsph.heat_conduction.contact_temperature), not transcribed "
                   "from any table, so no number is hand-copied. The secondary "
                   "citation's bibliographic fields (journal, volume, pages, DOI) were "
                   "verified programmatically against Crossref (see crossref_verified). "
                   "The book's equation numbering was not machine-verified (no library "
                   "access from the sandbox): the formula itself is validated here by a "
                   "Crank-Nicolson finite-difference solver, by its limits and by "
                   "self-similarity."),
}


# --------------------------------------------------------------------- helpers
def n_for(dx, domain=DOMAIN):
    return int(round(domain / dx))


def one(dx, calibration="lattice", domain=DOMAIN, courante=0.2, h_factor=1.3):
    res = run_contact_sph(ContactConfig(n_particles=n_for(dx, domain), dx=dx,
                                        courante=courante, h_factor=h_factor,
                                        calibration=calibration))
    mx, rms = res.error(X_CUT)
    return {"dx": dx, "n": res.n_particles, "steps": res.steps,
            "calibration": res.calibration,
            "diffusion_factor": res.diffusion_factor,
            "max": mx, "rms": rms,
            "interface_T": res.interface_temperature()}, res


def order_of(rows):
    out = []
    for a, b in zip(rows, rows[1:]):
        if b["max"] <= 0.0:
            continue
        out.append({"dx_pair": [a["dx"], b["dx"]],
                    "observed_order": math.log(a["max"] / b["max"]) / math.log(a["dx"] / b["dx"])})
    return out


def symbol_table(dx=1.0, h_factor=1.3):
    """``alpha_eff(k)/alpha`` of the lattice operator vs ``k dx`` (dispersion)."""
    h = h_factor * dx
    out = {"k_dx": [], "ratio": []}
    for kdx in (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4):
        out["k_dx"].append(kdx)
        out["ratio"].append(lattice_symbol_ratio(kdx / dx, dx, h))
    modes = [2.0 * math.pi * m / 256.0 for m in (1, 2, 3)]      # the repo's calibration modes
    repo_factor = lattice_diffusion_factor(h_factor=h_factor)[0]
    k0 = lattice_symbol_ratio(1.0e-3, dx, h)
    return {"rows": out,
            "repo_three_mode_factor": repo_factor,
            "mode_k_dx": modes,
            "mode_ratios": [lattice_symbol_ratio(k / dx, dx, h) for k in modes],
            "k0_limit": k0,
            "three_mode_vs_k0_gap": (k0 - repo_factor) / k0}


def residual_diffusivity(res):
    """Equivalent ``d(alpha)/alpha`` of a run's profile error, via ``e ~ dalpha * t T''``.

    For the linear heat equation ``dT/dalpha = t d2T/dx2`` exactly, so
    regressing the profile error on the analytic second derivative isolates the
    part of the error that is a *residual diffusivity* (a plateau) from the part
    that is a truncation ripple.
    """
    x = torch.tensor(res.x, dtype=torch.float64)
    a = torch.tensor(res.temperature, dtype=torch.float64)
    b = torch.tensor(res.analytic, dtype=torch.float64)
    e = a - b
    t2 = torch.autograd.functional.jacobian
    # analytic second derivative of the erfc profile, in closed form
    eta = x / (2.0 * math.sqrt(ALPHA * res.time))
    tp = 0.5 * (T_COLD - T_HOT) * (2.0 / math.sqrt(math.pi)) * torch.exp(-eta ** 2) \
        / (2.0 * math.sqrt(ALPHA * res.time))
    tpp = -2.0 * eta * tp / (2.0 * math.sqrt(ALPHA * res.time))
    keep = x.abs() <= X_CUT
    lhs = e[keep]
    rhs = (res.time * tpp[keep])
    d_alpha = float((lhs * rhs).sum() / (rhs * rhs).sum())
    resid = float((lhs - d_alpha * rhs).abs().max())
    return {"d_alpha_over_alpha": d_alpha, "max_error": float(lhs.abs().max()),
            "residual_after_removing_residual_diffusivity": resid,
            "d_alpha_abs": d_alpha * ALPHA}


def h_factor_ladder(dx=0.02, h_factors=(1.1, 1.3, 1.5, 1.7)):
    rows = []
    for hf in h_factors:
        out = {"h_factor": hf,
               "stencil_coeffs": [[m, a] for m, a in lattice_stencil(dx, hf * dx)]}
        for mode in ("none", "lattice", "k0"):
            row, res = one(dx, calibration=mode, h_factor=hf)
            out[mode] = {"max": row["max"], "rms": row["rms"],
                         "diffusion_factor": res.diffusion_factor}
        out["metric_ratio_lattice_over_none"] = out["lattice"]["max"] / out["none"]["max"]
        rows.append(out)
    return rows


def dt_ladder(dx=0.02, courantes=(0.05, 0.1, 0.2, 0.3)):
    rows = []
    for c in courantes:
        row, _ = one(dx, calibration="k0", courante=c)
        row["courante"] = c
        rows.append(row)
    return rows


def domain_check(dx=0.02, calibration="k0"):
    """Same dx, twice the domain: the mirror-ghost walls must not matter."""
    a, _ = one(dx, calibration=calibration)
    b, _ = one(dx, calibration=calibration, domain=2.0 * DOMAIN)
    return {"dx": dx, "n": a["n"], "n_wide": b["n"],
            "max_narrow": a["max"], "max_wide": b["max"],
            "delta": abs(a["max"] - b["max"])}


def operator_checks():
    """Stencil vs dense pair sum, and a short run on the dense path."""
    eq = stencil_vs_dense()
    fast, _ = one(0.1, calibration="k0")
    cfg_dense = ContactConfig(n_particles=120, dx=0.1, courante=0.4,
                              calibration="k0", dense=True)
    dense = run_contact_sph(cfg_dense)
    cfg_fast = ContactConfig(n_particles=120, dx=0.1, courante=0.4, calibration="k0")
    fast_run = run_contact_sph(cfg_fast)
    return {"stencil_vs_dense_max_abs": eq,
            "dense_run": {"n": dense.n_particles, "dx": dense.dx, "steps": dense.steps,
                          "max": dense.error(X_CUT)[0]},
            "fast_run_same_cfg": {"max": fast_run.error(X_CUT)[0]},
            "dense_vs_fast_max_abs": float(
                (torch.tensor(dense.temperature)
                 - torch.tensor(fast_run.temperature)).abs().max())}


def fd_ladder(ns=(601, 1201, 2401)):
    """Crank-Nicolson vs the erfc formula -- the reference checks itself."""
    rows = []
    for n in ns:
        x, T = fd_contact_reference(n=n)
        exact = contact_temperature(x, FOURIER, ALPHA, T_HOT, T_COLD)
        e = (T - exact).abs()
        far = x.abs() >= 0.2
        rows.append({"n": n, "dx": 2.0 * 6.0 / (n - 1),
                     "max_all": float(e.max()),
                     "max_away_from_interface": float(e[far].max())})
    return rows


def self_similarity(dx=0.02, calibration="k0"):
    """Error at several times, in the similarity variable (self-similar collapse)."""
    cfg = ContactConfig(n_particles=n_for(dx), dx=dx, calibration=calibration,
                        sample_every=1000)
    res = run_contact_sph(cfg)
    out = []
    x = torch.tensor(res.x)
    for t, prof in zip(res.profile_time, res.profiles):
        if t <= 0.0:
            continue
        exact = contact_temperature(x, t, ALPHA, T_HOT, T_COLD)
        e = (torch.tensor(prof) - exact).abs()
        keep = x.abs() <= X_CUT
        out.append({"t": t, "fourier": t, "max": float(e[keep].max()),
                    "rms": float(torch.sqrt((e[keep] ** 2).mean()))})
    return out


# --------------------------------------------------------------------- figures
def make_figures(res, ladder, h_rows, sym, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array(res.x)
    t_sph = np.array(res.temperature)
    t_ref = np.array(res.analytic)
    eta = x / (2.0 * math.sqrt(ALPHA * res.time))

    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.0))
    ax[0].plot(x, t_ref, "-", lw=2.5, label=r"erfc reference ($t^*=0.5$)")
    ax[0].plot(x[::20], t_sph[::20], "o", ms=3.5, label="SPH, dx=%.3f m" % res.dx)
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("T [K]")
    ax[0].set_title("contact profile (k0-calibrated)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(eta, t_ref, "-", lw=2.5)
    ax[1].plot(eta[::20], t_sph[::20], "o", ms=3.5)
    ax[1].set_xlabel(r"similarity variable $\eta = x/(2\sqrt{\alpha t})$")
    ax[1].set_ylabel("T [K]"); ax[1].set_title("self-similar collapse")
    ax[1].grid(alpha=0.3)

    ax[2].semilogy(eta, np.abs(t_sph - t_ref), "o", ms=3.5)
    ax[2].axhline(GATE, color="r", ls="--", label="3 % gate")
    ax[2].set_xlabel(r"$\eta$"); ax[2].set_ylabel(r"$|T_{SPH}-T_{erfc}|$ [K]")
    ax[2].set_title("pointwise error"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "profile.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.0))
    for mode, style in (("lattice", "s-"), ("k0", "o-")):
        rows = ladder[mode]
        dx = np.array([r["dx"] for r in rows])
        mx = np.array([r["max"] for r in rows])
        ax[0].loglog(dx, mx, style, label=f"SPH, {mode} calibration")
    dx0 = ladder["k0"][0]["dx"]
    ax[0].loglog([dx0, ladder["k0"][-1]["dx"]],
                 [ladder["k0"][0]["max"] * (ladder["k0"][0]["dx"] / dx0) ** 2,
                  ladder["k0"][0]["max"] * (ladder["k0"][-1]["dx"] / dx0) ** 2],
                 "k--", lw=1.0, label=r"$\propto dx^2$")
    ax[0].axhline(GATE, color="r", ls="--", label="3 % gate")
    ax[0].set_xlabel("dx [m]"); ax[0].set_ylabel(r"max $|T_{SPH}-T_{erfc}|/\Delta T$")
    ax[0].set_title("resolution ladder"); ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    hf = np.array([r["h_factor"] for r in h_rows])
    for mode, style in (("none", "^:"), ("lattice", "s-"), ("k0", "o-")):
        ax[1].semilogy(hf, [max(r[mode]["max"], 1e-12) for r in h_rows], style,
                       label=mode)
    ax[1].axhline(GATE, color="r", ls="--")
    ax[1].set_xlabel("h / dx"); ax[1].set_ylabel("max normalised error")
    ax[1].set_title("kernel support (dx = 0.02 m)"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    ax[2].semilogx(sym["rows"]["k_dx"], sym["rows"]["ratio"], "o-")
    ax[2].axhline(sym["k0_limit"], color="k", ls=":", lw=1.0,
                  label="k -> 0 limit = %.6f" % sym["k0_limit"])
    ax[2].axhline(sym["repo_three_mode_factor"], color="m", ls="--", lw=1.0,
                  label="repo 3-mode mean = %.6f" % sym["repo_three_mode_factor"])
    ax[2].set_xlabel(r"$k\,dx$"); ax[2].set_ylabel(r"$\alpha_{eff}(k)/\alpha$")
    ax[2].set_title("lattice symbol: where the plateau comes from")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(outdir / "convergence.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    outdir = Path(__file__).resolve().parent
    t0 = time.time()
    ladder = {"lattice": [], "k0": []}
    dxs = DX_LADDER[:3] if args.quick else DX_LADDER
    for dx in dxs:
        for mode in ("lattice", "k0"):
            row, _ = one(dx, calibration=mode)
            ladder[mode].append(row)
    if not args.quick:
        row, _ = one(DX_FINEST_K0, calibration="k0")
        ladder["k0"].append(row)
    orders = {m: order_of(rows) for m, rows in ladder.items()}

    h_rows = h_factor_ladder()
    dt_rows = dt_ladder()
    dom = domain_check()
    ops = operator_checks()
    fd = fd_ladder()
    sym = symbol_table()
    ss = self_similarity() if not args.quick else []

    main_row, main_res = one(0.01, calibration="k0")
    main_max, main_rms = main_res.error(X_CUT)
    res_lat = residual_diffusivity(main_res)
    _, lat_res = one(0.01, calibration="lattice")
    res_lat_row = residual_diffusivity(lat_res)
    raw_row, raw_res = one(0.01, calibration="none")

    fine_orders = [o["observed_order"] for o in orders["k0"][-2:]]
    gates = [
        {"name": "sph_vs_erfc_max_normalised_error",
         "value": main_max, "tol": GATE, "op": "<=", "pass": bool(main_max <= GATE)},
        {"name": "every_ladder_rung_inside_gate",
         "value": max(r["max"] for m in ladder for r in ladder[m]), "tol": GATE,
         "op": "<=", "pass": bool(all(r["max"] <= GATE for m in ladder for r in ladder[m]))},
        {"name": "interface_temperature_vs_mean",
         "value": abs(main_res.interface_temperature() - 0.5 * (T_HOT + T_COLD)),
         "tol": 0.01, "op": "<=",
         "pass": bool(abs(main_res.interface_temperature() - 0.5 * (T_HOT + T_COLD)) <= 0.01)},
        {"name": "stencil_matches_dense_operator",
         "value": ops["stencil_vs_dense_max_abs"], "tol": 1e-6, "op": "<=",
         "pass": bool(ops["stencil_vs_dense_max_abs"] <= 1e-6)},
        {"name": "k0_calibrated_second_order_trend",
         "value": min(fine_orders), "tol": [1.5, 2.5], "op": "in",
         "pass": bool(all(1.5 <= o <= 2.5 for o in fine_orders))},
        {"name": "wall_independence_domain_doubling",
         "value": dom["delta"], "tol": 1e-3, "op": "<=",
         "pass": bool(dom["delta"] <= 1e-3)},
        {"name": "fd_reference_agrees_with_erfc",
         "value": fd[-1]["max_all"], "tol": GATE, "op": "<=",
         "pass": bool(fd[-1]["max_all"] <= GATE)},
    ]
    passed = all(g["pass"] for g in gates)

    payload = {
        "case": CASE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 1),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "machine": platform.machine(),
                        "dtype": "torch.float64",
                        "crossref_verified": {
                            "title": "Conduction Modelling Using Smoothed Particle Hydrodynamics",
                            "journal": "Journal of Computational Physics", "volume": 148,
                            "pages": "227-264", "year": 1999,
                            "authors": ["Cleary", "Monaghan"],
                            "doi": "10.1006/jcph.1998.6118"}},
        "reference": {
            "formula": "T(x,t) = T_cold + (T_hot - T_cold)/2 * erfc(x / (2 sqrt(alpha t)))",
            **CITATION},
        "parameters": {
            "alpha_m2_s": ALPHA, "rho": 1.0, "cp": 1.0, "k_W_mK": ALPHA,
            "T_hot_K": T_HOT, "T_cold_K": T_COLD, "delta_T_K": T_HOT - T_COLD,
            "fourier_t_star": FOURIER, "l_ref_m": 1.0, "t_end_s": FOURIER / ALPHA,
            "domain_m": [-0.5 * DOMAIN, 0.5 * DOMAIN],
            "penetration_2sqrt_alpha_t_m": 2.0 * math.sqrt(FOURIER),
            "eta_at_wall": 0.5 * DOMAIN / (2.0 * math.sqrt(FOURIER)),
            "metric_region_m": [-X_CUT, X_CUT],
            "kernel": "1-D cubic spline (M4), support 2h, h = 1.3 dx, eta = 0.01 h",
            "stencil_offsets_dx": [m for m, _ in lattice_stencil(0.02, 0.026)],
        },
        "main_run": {"n_particles": main_res.n_particles, "dx_m": main_res.dx,
                     "dt_s": main_res.dt, "steps": main_res.steps, "t_s": main_res.time,
                     "fourier": main_res.fourier, "calibration": "k0",
                     "diffusion_factor": main_res.diffusion_factor,
                     "max_normalised_error": main_max, "rms_normalised_error": main_rms,
                     "interface_temperature_K": main_res.interface_temperature(),
                     "residual_diffusivity": res_lat,
                     "lattice_calibration_run": {"max": lat_res.error(X_CUT)[0],
                                                 **res_lat_row},
                     "none_calibration_run": {"max": raw_res.error(X_CUT)[0],
                                              "rms": raw_res.error(X_CUT)[1]}},
        "resolution_ladder": ladder,
        "convergence_order": orders,
        "h_factor_sensitivity": h_rows,
        "dt_sensitivity": dt_rows,
        "domain_doubling": dom,
        "operator_equivalence": ops,
        "fd_reference_ladder": fd,
        "lattice_symbol": sym,
        "self_similarity": ss,
        "gates": gates,
        "passed": passed,
        "profile": {"x": main_res.x, "T_sph": main_res.temperature,
                    "T_erfc": main_res.analytic},
    }
    (outdir / "result.json").write_text(json.dumps(payload, indent=1))
    if args.figure:
        make_figures(main_res, ladder, h_rows, sym, outdir)

    print(f"{CASE}: max normalised error {main_max:.3e} "
          f"({'PASS' if main_max <= GATE else 'FAIL'} vs {GATE:.0%} gate)")
    for mode in ladder:
        for r in ladder[mode]:
            print(f"  [{mode:7s}] dx={r['dx']:.3f} n={r['n']:5d} steps={r['steps']:6d} "
                  f"factor={r['diffusion_factor']:.7f} max={r['max']:.3e} rms={r['rms']:.3e}")
        print(f"  [{mode:7s}] order:", ", ".join(
            f"{o['dx_pair']} -> {o['observed_order']:.2f}" for o in orders[mode]))
    print(f"  symbol: 3-mode mean {sym['repo_three_mode_factor']:.7f} vs k->0 "
          f"{sym['k0_limit']:.7f} (gap {sym['three_mode_vs_k0_gap']:.2e})")
    print(f"  residual d_alpha/alpha (lattice cal): {res_lat_row['d_alpha_over_alpha']:.3e}"
          f"; (k0 cal): {res_lat['d_alpha_over_alpha']:.3e}")
    for g in gates:
        print(f"  gate {g['name']}: {g['value']} ({'ok' if g['pass'] else 'FAIL'})")
    print(f"  wall {payload['wall_seconds']} s -> {outdir / 'result.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())