#!/usr/bin/env python3
"""CASE-SPH-CONDUCTION-2D -- second benchmark family for the SPH conduction operator.

The case
--------
A periodic square ``[0, L] x [0, L]`` is initialised with one Fourier mode,

    T(x, y, 0) = cos(kx x) cos(ky y),      kx = 2 pi px / L,  ky = 2 pi py / L,

which is an exact solution of the linear heat equation ``dT/dt = alpha lap(T)``
decaying as ``exp(-alpha (kx^2 + ky^2) t)`` (substitute; it is the periodic
solution of the linear heat equation, Carslaw & Jaeger 1959, *Conduction of Heat
in Solids*, 2nd ed., Ch. II -- same chapter as the 1-D ``erfc`` target, different
family: a discrete *symbol* question rather than a similarity profile).

The operator is the repository's SPH conduction pair sum
(:mod:`tensorsph.ice_thermal.sph_thermal`, Cleary & Monaghan 1999), lifted to 2-D
by :mod:`tensorsph.heat_conduction_2d`:

    dT_i/dt = (2/rho) sum_j V_j (2 k_i k_j/(k_i + k_j)) F_ij (T_i - T_j).

Why this family exists
----------------------
The 1-D ``erfc`` case is the only accuracy evidence for ``heat_conduction_operator``
in the method-selection registry.  It cannot see two things this case can:
whether the ``2/rho sum_j V_j ...`` form is *dimensionally* consistent, and
whether the 1-D lattice calibration transfers.  It does not:

* raw operator in 2-D: ``alpha_eff/alpha ~ 1.9 dx`` (O(dx) too small);
* the shipped 1-D calibration factor (1.0218) applied in 2-D therefore diffuses
  ~``1/dx`` too slowly -- the mode-decay error is ~770-870 % (banked as a
  ``refuted`` registry record: a documented misuse, not a method error);
* redoing in 2-D what the 1-D solver already does in 1-D (measure the lattice
  symbol, divide it out) restores consistency: error 6.4e-3 -> 1.6e-3 -> 3.9e-4
  at n = 32/64/128, i.e. second order and 0.039 % at the finest rung.

That 0.039 % is the second-family record banked for ``heat_conduction_operator``.

Also measured
-------------
* lattice isotropy of the discrete symbol over a mode table;
* spurious-mode coupling (the product mode is *not* an eigenfunction of the 2-D
  stencil once diagonal offsets are in the support: ``cos cos`` feeds ``sin sin``,
  so the "modal" arm is worse than the symbol-calibrated arm and the leaked
  energy fraction is reported);
* dense pair-sum vs the ``torch.roll`` fast path (round-off) and the degenerate
  ``ny = 1`` case against the shipped 1-D ``conduction_rhs`` (exact);
* an independent Crank-Nicolson (periodic, 5-point) solve of the same initial
  value problem, which validates the closed form as a target.

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_conduction_2d/run.py [--quick] [--figure]

Writes ``result.json`` (every number in the report is read from it) and, with
``--figure``, ``mode_decay.png``.  Exit status is non-zero when the main gate
fails.
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

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.heat_conduction_2d import (  # noqa: E402
    ModeDecayConfig,
    degenerate_1d_agreement,
    fd_mode_decay,
    lattice_diffusion_factor_2d,
    mode_amplitude,
    mode_exact,
    mode_field,
    run_mode_decay,
    stencil_vs_dense_2d,
    symbol_rate,
)

CASE = "CASE-SPH-CONDUCTION-2D"
GATE = 0.03                # 3 % on the normalised field error (same as the 1-D case)
LENGTH = 1.0               # m
ALPHA = 1.0                # m^2/s
PERIODS = (1, 1)           # mode (2 pi / L, 2 pi / L)
AMPLITUDE_END = 0.1        # exact amplitude at t_end, relative to t = 0
MAIN_N = 128               # lattice of the banked record
MAIN_CALIBRATION = "k0_2d"
MAIN_INTEGRATOR = "rk4"
LADDER_N = (32, 64, 128)
#: (calibration, integrator) arms.  RK4 arms isolate the *spatial* reading; the
#: Euler arms are the shipped conduction solver's stepping and show what the
#: first-order time error does to the same operator (at n = 32 it hides the
#: spatial error by cancelling it).
ARMS = (("k0_2d", "rk4"), ("k0_2d", "euler"), ("mean_2d", "rk4"),
        ("mode_2d", "rk4"), ("lattice_1d", "euler"), ("none", "euler"))
N_QUICK = (32, 64)

CITATION = {
    "primary": (
        "Carslaw, H.S. & Jaeger, J.C. (1959) Conduction of Heat in Solids, 2nd ed., "
        "Oxford University Press, Ch. II (linear flow of heat; periodic solutions): "
        "the decaying Fourier mode T = exp(-alpha (kx^2+ky^2) t) cos(kx x) cos(ky y) "
        "used here as the reference."
    ),
    "secondary": (
        "Cleary, P.W. & Monaghan, J.J. (1999) Conduction Modelling Using Smoothed "
        "Particle Hydrodynamics, J. Comput. Phys. 148:227-264, "
        "doi:10.1006/jcph.1998.6118 -- source of the SPH conduction operator "
        "reused unchanged by tensorsph.heat_conduction_2d."
    ),
    "provenance": (
        "Reference values are computed from the closed form by code "
        "(tensorsph.heat_conduction_2d.mode_exact), not transcribed; the closed form "
        "is itself checked here by an independent Crank-Nicolson finite-difference "
        "solve that shares no code, kernel or discretisation with the SPH path. The "
        "book's chapter/section numbering is not machine-verified from the sandbox; "
        "the formula is verified by substitution into the PDE."
    ),
}


def mode_energy_fractions(field: torch.Tensor, n: int, dx: float,
                          kx: float, ky: float) -> dict:
    """Ledger of where the field's energy sits after the run.

    ``cos cos`` is the initialised mode, ``sin sin`` is the mode the diagonal
    stencil offsets feed it into (the product mode is not an eigenfunction of the
    2-D operator), and ``other`` is everything else (round-off plus further
    leakage).  Fractions of ``sum T^2`` after subtracting the modal projections.
    """
    cc = mode_field(n, dx, kx, ky)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    xg, yg = torch.meshgrid(x, x, indexing="ij")
    ss = torch.sin(kx * xg) * torch.sin(ky * yg)
    amp_cc = float((field * cc).sum() / (cc * cc).sum())
    amp_ss = float((field * ss).sum() / (ss * ss).sum())
    resid = field - amp_cc * cc - amp_ss * ss
    total = float((field * field).sum())
    return {
        "amplitude_cos_cos": amp_cc,
        "amplitude_sin_sin": amp_ss,
        "fraction_cos_cos": float((amp_cc ** 2) * (cc * cc).sum() / total),
        "fraction_sin_sin": float((amp_ss ** 2) * (ss * ss).sum() / total),
        "fraction_other": float((resid * resid).sum() / total),
    }


def isotropy_table(n: int) -> dict:
    """``alpha_eff/alpha`` of the raw operator over a table of lattice modes.

    Same |k| along axes vs diagonals is not representable on a square lattice, so
    the table simply lists modes and the spread of the ratio; the spread is the
    quantitative statement of symbol anisotropy.
    """
    dx = LENGTH / n
    h = 1.3 * dx
    rows = []
    for px, py in ((1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3)):
        kx = 2.0 * math.pi * px / LENGTH
        ky = 2.0 * math.pi * py / LENGTH
        ratio = symbol_rate(dx, h, kx, ky) / (kx * kx + ky * ky)
        rows.append({"periods": [px, py], "alpha_eff_over_alpha": ratio,
                     "alpha_eff_over_alpha_dx": ratio / dx})
    vals = [r["alpha_eff_over_alpha"] for r in rows]
    return {"n": n, "dx": dx, "rows": rows,
            "spread_over_mean": (max(vals) - min(vals)) / (sum(vals) / len(vals))}


def convergence_order(pairs: list[tuple[float, float]]) -> float:
    """Least-squares slope of log(error) vs log(dx)."""
    if len(pairs) < 2:
        return float("nan")
    xs = [math.log(p[0]) for p in pairs]
    ys = [math.log(p[1]) for p in pairs]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    return num / den


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse ladder only")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    ladder_n = N_QUICK if args.quick else LADDER_N

    ladder = []
    for n in ladder_n:
        for cal, integ in ARMS:
            res = run_mode_decay(ModeDecayConfig(
                n=n, length=LENGTH, alpha=ALPHA, periods=PERIODS,
                amplitude_end=AMPLITUDE_END, calibration=cal, integrator=integ))
            ladder.append({
                "n": n, "dx": res.dx, "steps": res.steps, "dt": res.dt,
                "calibration": cal, "integrator": integ, "factor": res.factor,
                "max_normalised_error": res.max_normalised_error,
                "max_error_over_initial_amplitude": res.max_error_over_initial,
                "decay_rate_error": res.decay_rate_error,
                "raw_symbol_rate": res.raw_symbol_rate,
                "analytic_rate": ALPHA * (res.kx ** 2 + res.ky ** 2),
            })
            print(f"n={n:4d} {cal:11s} {integ:5s} factor={res.factor:10.6g} "
                  f"err={res.max_normalised_error:9.3e} "
                  f"decay_err={res.decay_rate_error:9.3e}", flush=True)

    main_run = run_mode_decay(ModeDecayConfig(
        n=MAIN_N, length=LENGTH, alpha=ALPHA, periods=PERIODS,
        amplitude_end=AMPLITUDE_END, calibration=MAIN_CALIBRATION,
        integrator=MAIN_INTEGRATOR, sample_every=25))
    dx_main = LENGTH / MAIN_N
    ledger = mode_energy_fractions(
        torch.tensor(main_run.field_values, dtype=torch.float64).reshape(MAIN_N, MAIN_N),
        MAIN_N, dx_main, main_run.kx, main_run.ky)
    analytic_field = mode_exact(MAIN_N, dx_main, main_run.kx, main_run.ky,
                                main_run.time, ALPHA)
    analytic_amp = mode_amplitude(analytic_field, MAIN_N, dx_main,
                                 main_run.kx, main_run.ky)

    raw_arms = [row for row in ladder if row["calibration"] == "none"]
    one_d_arms = [row for row in ladder if row["calibration"] == "lattice_1d"]
    cal_arms = [row for row in ladder if row["calibration"] == "k0_2d"
                and row["integrator"] == "rk4"]
    euler_arms = [row for row in ladder if row["calibration"] == "k0_2d"
                  and row["integrator"] == "euler"]
    order = convergence_order([(row["dx"], row["max_normalised_error"])
                               for row in cal_arms])

    dimensional = [{"n": row["n"], "dx": row["dx"],
                    "alpha_eff_over_alpha": row["factor"],
                    "alpha_eff_over_alpha_dx": row["factor"] / row["dx"]}
                   for row in cal_arms]

    result = {
        "case": CASE,
        "gate": GATE,
        "gate_passed": main_run.max_normalised_error <= GATE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 1),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "dtype": "torch.float64",
            "numpy_scipy_reference": "Crank-Nicolson, periodic 5-point",
        },
        "reference": {
            "formula": "T(x,y,t) = exp(-alpha (kx^2+ky^2) t) cos(kx x) cos(ky y)",
            "family": "decay of a Fourier mode on a periodic square (2-D)",
            **CITATION,
        },
        "parameters": {
            "length_m": LENGTH, "alpha_m2_s": ALPHA, "rho": 1.0, "cp": 1.0,
            "k_W_mK": 1.0, "periods": list(PERIODS),
            "kx": 2.0 * math.pi * PERIODS[0] / LENGTH,
            "ky": 2.0 * math.pi * PERIODS[1] / LENGTH,
            "amplitude_end_over_start": AMPLITUDE_END,
            "courante": 0.2, "h_over_dx": 1.3, "eta_factor": 0.01,
            "main_n": MAIN_N, "main_calibration": MAIN_CALIBRATION,
            "main_integrator": MAIN_INTEGRATOR,
            "ladder_n": list(ladder_n), "arms": list(ARMS),
        },
        "main_run": {
            "n": main_run.n, "dx": main_run.dx, "steps": main_run.steps,
            "dt": main_run.dt, "time": main_run.time,
            "calibration": main_run.calibration,
            "integrator": main_run.integrator, "factor": main_run.factor,
            "max_normalised_error": main_run.max_normalised_error,
            "max_error_over_initial_amplitude": main_run.max_error_over_initial,
            "decay_rate_error": main_run.decay_rate_error,
            "analytic_amplitude": analytic_amp,
            "sph_amplitude": main_run.amplitude_end,
            "energy_ledger": ledger,
            "normalisation": "max|T_sph - T_exact| / max|T_exact| at t_end "
                             "(the error is quoted against the amplitude that is "
                             "left, i.e. 10x stricter than against the initial one)",
        },
        "ladder": ladder,
        "error_budget": [
            {"n": row["n"], "dx": row["dx"],
             "spatial_rel_error": row["max_normalised_error"],
             "euler_rel_error": next(e["max_normalised_error"] for e in euler_arms
                                     if e["n"] == row["n"])}
            for row in cal_arms
        ],
        "convergence": {
            "calibration": "k0_2d",
            "integrator": "rk4",
            "order_log_log": order,
            "pairs": [[row["dx"], row["max_normalised_error"]] for row in cal_arms],
        },
        "dimensional_scaling": {
            "claim": "raw operator's effective diffusivity is O(dx) in 2-D, so the "
                     "1-D lattice calibration (1.0218) does not transfer",
            "rows": dimensional,
            "one_d_factor": lattice_diffusion_factor_2d(1.3, 1.0e-3, 0.01, "k0"),
        },
        "arms_summary": {
            "none_worst_error": max(row["max_normalised_error"] for row in raw_arms),
            "lattice_1d_worst_error": max(row["max_normalised_error"] for row in one_d_arms),
            "k0_2d_worst_error": max(row["max_normalised_error"] for row in cal_arms),
            "note": "the 1-D-calibrated and uncalibrated arms are documented traps; "
                    "the banked record reads the 2-D-calibrated arm (main_run)",
        },
        "isotropy": [isotropy_table(n) for n in ladder_n],
        "equivalence": {
            "stencil_vs_dense_max_abs": stencil_vs_dense_2d(),
            "degenerate_1d_vs_shipped_conduction_rhs_max_abs": degenerate_1d_agreement(),
        },
        "fd_reference": [fd_mode_decay(n=n, length=LENGTH, alpha=ALPHA,
                                       periods=PERIODS, amplitude_end=AMPLITUDE_END)
                         for n in ladder_n],
        "samples": {
            "time": main_run.sampled_time,
            "amplitude_over_start": main_run.sampled_ratio,
            "exact_over_start": main_run.exact_ratio,
        },
    }

    out = Path(__file__).resolve().parent / "result.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"\nmain run: n={main_run.n} err={main_run.max_normalised_error:.4e} "
          f"(gate {GATE:.0%}) -> {'PASS' if result['gate_passed'] else 'FAIL'}")
    print(f"order(k0_2d) = {order:.2f}; wrote {out}")

    if args.figure:
        plot(result, Path(__file__).resolve().parent / "mode_decay.png")
    return 0 if result["gate_passed"] else 1


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogy(result["samples"]["time"], result["samples"]["exact_over_start"],
                "k-", label="exact exp(-$\\alpha(k_x^2+k_y^2)t$)")
    ax.semilogy(result["samples"]["time"], result["samples"]["amplitude_over_start"],
                "o", ms=3, label="SPH amplitude (2-D calibrated)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("amplitude / amplitude(0)")
    ax.set_title(f"mode decay, n={result['main_run']['n']}, "
                 f"err={result['main_run']['max_normalised_error']:.2e}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for cal, marker in (("k0_2d", "o-"), ("mean_2d", "s-"), ("mode_2d", "^-"),
                        ("lattice_1d", "v-"), ("none", "x-")):
        rows = [r for r in result["ladder"] if r["calibration"] == cal
                and r["integrator"] == "rk4"]
        if not rows:
            continue
        ax.loglog([r["dx"] for r in rows], [r["max_normalised_error"] for r in rows],
                  marker, label=cal)
    ax.axhline(result["gate"], color="r", ls="--", lw=1, label="gate 3 %")
    ax.axhline(15.0, color="k", ls=":", lw=1, label="owner 15 %")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("max normalised error")
    ax.set_title("calibration arms vs resolution")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())