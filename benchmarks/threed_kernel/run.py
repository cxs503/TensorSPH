#!/usr/bin/env python3
"""3-D kernel and operator verification: quadrature, lattice bias, convergence.

Three questions, each with an exact answer:

1. **Is the 3-D cubic-spline kernel normalised?**  ``int W dV`` over the ``2h``
   ball must be 1 and ``int r^2 laplacian W dV`` must be 6 (the second moment
   that makes the Brookshaw Laplacian consistent).  Both are evaluated by
   spherical quadrature on a refined radial grid.
2. **What is the lattice zeroth-moment bias ``C - 1`` and what does it cost?**
   Swept over ``h / dx`` in ``[1.0, 2.0]`` together with the spurious background
   pressure ``B[(1 + b)^gamma - 1]`` it produces, and with the calibrated mass
   residual that removes it.
3. **Do the SPH gradient and Laplacian reproduce an analytic field, and at what
   order?**  ``f = sin(kx) sin(ky) sin(kz)`` on a triply periodic lattice, with
   ``laplacian f = -3 k^2 f``.  The Laplacian is measured twice: raw, and with the
   discrete second moment divided out (:func:`laplacian_second_moment`), because
   on a cubic lattice the raw operator is 17 % too strong and no mesh refinement
   fixes that -- ``h / dx`` is fixed, so ``K`` is a fixed lattice number.

Usage::

    PYTHONPATH=src python benchmarks/threed_kernel/run.py            # all cases
    PYTHONPATH=src python benchmarks/threed_kernel/run.py --only ladder-n24 --out /tmp/x.json
    PYTHONPATH=src python benchmarks/threed_kernel/run.py --merge /tmp/*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.threed import (  # noqa: E402
    analytic_laplacian_of_mode,
    calibrated_particle_mass,
    cubic_spline_kernel,
    cubic_spline_laplacian,
    gradient_of_field,
    kernel_normalisation,
    laplacian_of_field,
    laplacian_second_moment,
    lattice_kernel_sum,
    periodic_pairs,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "threed_kernel"
RHO0, C0, GAMMA = 1000.0, 20.0, 7.0
WAVENUMBER = 2.0 * math.pi

#: ``h / dx`` sweep for the lattice bias, and the analytic Laplace ladder.
H_OVER_DX = (1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0)
LADDER = (12, 16, 24, 32)


def quadrature_check(h: float = 0.013) -> dict:
    """``int W dV = 1`` and ``int r^2 laplacian W dV = 6`` by Gauss-like quadrature."""
    integral, _ = kernel_normalisation(h, n_r=4000)
    r = torch.linspace(0.0, 2.0 * h, 400001, dtype=torch.float64)
    delta = torch.stack((r, torch.zeros_like(r), torch.zeros_like(r)), dim=-1)
    lap = cubic_spline_laplacian(delta, h)
    second = float((4.0 * math.pi * lap * r.pow(4)).sum() * (r[1] - r[0]))
    peak = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
    return {
        "h": h,
        "kernel_integral": integral,
        "kernel_integral_error": integral - 1.0,
        "second_moment": second,
        "second_moment_error": second - 6.0,
        "laplacian_at_zero": float(cubic_spline_laplacian(
            torch.zeros(1, 3, dtype=torch.float64), h)),
        "laplacian_at_zero_analytic": -9.0 * peak / (h * h),
        "note": (
            "The Tait law clamps a negative bias to p = 0, so the tension-side "
            "entries of the lattice sweep cost nothing in pressure but shift the "
            "density by the same amount."),
    }


def lattice_case(h_over_dx: float, dx: float = 0.01) -> dict:
    """Zeroth-moment bias ``C - 1``, its spurious pressure, and the calibrated mass."""
    h = h_over_dx * dx
    raw = lattice_kernel_sum(dx, h)
    bias = raw - 1.0
    stiffness = RHO0 * C0 ** 2 / GAMMA
    p_spurious = stiffness * ((1.0 + bias) ** GAMMA - 1.0)
    m_raw = RHO0 * dx ** 3
    m_cal = calibrated_particle_mass(dx, h, RHO0)
    # Residual of the calibrated mass, measured on an infinite lattice sum.
    residual = m_cal * raw / (RHO0 * dx ** 3) - 1.0
    return {
        "h_over_dx": h_over_dx,
        "bias_sign": "tension-side (clamped by the Tait law)" if bias < 0 else
                     "compression-side (costs a spurious pressure)",
        "dx": dx,
        "h": h,
        "C": raw,
        "bias": bias,
        "spurious_pressure_Pa": p_spurious,
        "spurious_ratio_c0_20": p_spurious / stiffness,
        "mass_ratio_calibrated": m_cal / m_raw,
        "calibrated_residual": residual,
    }


def ladder_case(n: int, h_over_dx: float = 1.3) -> dict:
    """Operator accuracy on ``sin sin sin`` at ``n`` points per wavelength."""
    box = 1.0
    dx = box / n
    h = h_over_dx * dx
    axes = [torch.arange(n, dtype=torch.float64) * dx] * 3
    pos = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    pairs = periodic_pairs(pos, box, 2.0 * h)
    volume = torch.full((pos.shape[0],), dx ** 3, dtype=torch.float64)
    field, analytic = analytic_laplacian_of_mode(pos, WAVENUMBER)
    grad_analytic = torch.stack([
        WAVENUMBER * torch.cos(WAVENUMBER * pos[:, 0]) * torch.sin(WAVENUMBER * pos[:, 1])
        * torch.sin(WAVENUMBER * pos[:, 2]),
        WAVENUMBER * torch.sin(WAVENUMBER * pos[:, 0]) * torch.cos(WAVENUMBER * pos[:, 1])
        * torch.sin(WAVENUMBER * pos[:, 2]),
        WAVENUMBER * torch.sin(WAVENUMBER * pos[:, 0]) * torch.sin(WAVENUMBER * pos[:, 1])
        * torch.cos(WAVENUMBER * pos[:, 2]),
    ], dim=-1)
    grad = gradient_of_field(pos, field, volume, h, pairs, box)
    lap_raw = laplacian_of_field(pos, field, volume, h, pairs, box, renormalise=False)
    lap_cor = laplacian_of_field(pos, field, volume, h, pairs, box, renormalise=True)
    moment = laplacian_second_moment(pos, volume, h, pairs, box)
    return {
        "n": n,
        "particles": n ** 3,
        "kh": WAVENUMBER * h,
        "h_over_dx": h_over_dx,
        "pairs": int(pairs.shape[0]),
        "gradient_rel_l2": float((grad - grad_analytic).norm() / grad_analytic.norm()),
        "laplacian_raw_rel_l2": float((lap_raw - analytic).norm() / analytic.norm()),
        "laplacian_corrected_rel_l2": float((lap_cor - analytic).norm() / analytic.norm()),
        "second_moment_K": float(moment.mean()),
        "second_moment_K_spread": float(moment.std()),
        "second_moment_K_continuum": 2.0,
    }


def lattice_case_dx_independence() -> dict:
    """``C`` must not depend on ``dx``: it is a lattice sum in units of ``h``."""
    rows = []
    for dx in (0.02, 0.01, 0.005, 0.0025):
        rows.append({"dx": dx, "C": lattice_kernel_sum(dx, 1.3 * dx)})
    spread = max(r["C"] for r in rows) - min(r["C"] for r in rows)
    return {"rows": rows, "spread": spread}


CASES = {
    **{f"lattice-h{h:g}": (lattice_case, (h,)) for h in H_OVER_DX},
    **{f"ladder-n{n}": (ladder_case, (n,)) for n in LADDER},
    "quadrature": (lambda: quadrature_check(), ()),
    "dx-independence": (lattice_case_dx_independence, ()),
}


def run_case(name: str) -> dict:
    function, args = CASES[name]
    return {"case": name, "result": function(*args)}


def merge(results: list[dict]) -> dict:
    """Assemble the case results into one report with the criteria evaluated."""
    by_name = {r["case"]: r["result"] for r in results}
    missing = sorted(set(CASES) - set(by_name))
    lattice = [by_name[f"lattice-h{h:g}"] for h in H_OVER_DX if f"lattice-h{h:g}" in by_name]
    ladder = [by_name[f"ladder-n{n}"] for n in LADDER if f"ladder-n{n}" in by_name]
    best = min(ladder, key=lambda r: r["kh"]) if ladder else None
    quadrature = by_name.get("quadrature")
    criteria = {
        "kernel_normalised_to_1e-6": bool(
            quadrature and abs(quadrature["kernel_integral_error"]) < 1e-6),
        "second_moment_6_to_1e-3": bool(
            quadrature and abs(quadrature["second_moment_error"]) < 1e-3),
        "calibrated_mass_leaves_residual_below_1e-12": bool(
            lattice and all(abs(r["calibrated_residual"]) < 1e-12 for r in lattice)),
        "laplacian_corrected_error_below_5pct_at_finest": bool(
            best and best["laplacian_corrected_rel_l2"] < 0.05),
        "laplacian_correction_improves_at_finest": bool(
            best and best["laplacian_corrected_rel_l2"]
            < best["laplacian_raw_rel_l2"]),
        "second_moment_is_lattice_constant": bool(
            ladder and all(abs(r["second_moment_K_spread"]) < 1e-12 for r in ladder)),
    }
    return {
        "benchmark": "threed_kernel",
        "reference": (
            "3-D cubic spline: int W dV = 1, int r^2 laplacian W dV = 6, "
            "laplacian(sin sin sin) = -3 k^2 f"
        ),
        "params": {"rho0": RHO0, "sound_speed": C0, "gamma": GAMMA,
                   "wavenumber": WAVENUMBER},
        "quadrature": quadrature,
        "lattice_sweep": lattice,
        "dx_independence": by_name.get("dx-independence"),
        "operator_ladder": ladder,
        "criteria": criteria,
        "missing_cases": missing,
        "verdict": "pass" if all(criteria.values()) and not missing else "fail",
    }


def figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = report["operator_ladder"]
    lattice = report["lattice_sweep"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    kh = [r["kh"] for r in ladder]
    axes[0].loglog(kh, [r["laplacian_raw_rel_l2"] for r in ladder], "o-",
                   label="raw Laplacian")
    axes[0].loglog(kh, [r["laplacian_corrected_rel_l2"] for r in ladder], "s-",
                   label="second-moment corrected")
    axes[0].loglog(kh, [r["gradient_rel_l2"] for r in ladder], "^-",
                   label="gradient")
    axes[0].axhline(0.03, color="k", ls=":", lw=1, label="3 % paper line")
    axes[0].set_xlabel(r"$k h$")
    axes[0].set_ylabel("relative $L_2$ error")
    axes[0].set_title("SPH operators on $\\sin kx\\sin ky\\sin kz$")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy([r["h_over_dx"] for r in lattice],
                     [abs(r["bias"]) for r in lattice], "o-")
    axes[1].set_xlabel("$h/\\Delta x$")
    axes[1].set_ylabel("$|C-1|$ (zeroth moment bias)")
    axes[1].set_title("3-D lattice kernel-sum bias")
    axes[1].grid(alpha=0.3, which="both")

    axes[2].semilogy([r["h_over_dx"] for r in lattice],
                     [r["spurious_pressure_Pa"] for r in lattice], "s-",
                     label="uncalibrated")
    axes[2].axhline(0.072 / 0.025, color="r", ls="--",
                    label=r"$\sigma/R$ of a 25 mm 3-D drop")
    axes[2].set_xlabel("$h/\\Delta x$")
    axes[2].set_ylabel("spurious background pressure (Pa)")
    axes[2].set_title("cost of the bias at $c_0 = 20$ m/s")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=None,
                        help="run these cases only (repeatable)")
    parser.add_argument("--out", default=None, help="write one case's JSON here")
    parser.add_argument("--merge", nargs="*", default=None,
                        help="merge per-case JSON files into report.json")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()

    if args.merge is not None:
        pattern = args.merge or [str(OUT / "cases" / "*.json")]
        files = [Path(p) for pat in pattern for p in glob.glob(pat)]
        results = [json.loads(f.read_text()) for f in files]
        report = merge(results)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.figure:
            figure(report, OUT / "figures" / "threed_kernel.png")
        print(json.dumps({"report": str(OUT / "report.json"),
                          "verdict": report["verdict"],
                          "criteria": report["criteria"],
                          "missing": report["missing_cases"]}, indent=2))
        return 0 if report["verdict"] == "pass" else 1

    torch.set_default_dtype(torch.float64)
    names = args.only or list(CASES)
    out_path = Path(args.out) if args.out else None
    if out_path and len(names) != 1:
        raise SystemExit("--out needs exactly one --only case")
    for name in names:
        result = run_case(name)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
