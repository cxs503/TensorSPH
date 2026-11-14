"""Lattice-orientation ensemble for the axisymmetric drop-frequency measurement.

Motivation
----------
``round2.py`` and ``interface_reconstruction.py`` both show that a single hex
lattice cut by a sphere gives a mode-``l = 2`` surface measure whose *scatter*
between discretisations is of the same order as the seeded perturbation
``eps = 0.02``.  That scatter is not physics: the problem is rotationally
symmetric, so the answer may not depend on the *orientation* of the lattice
relative to the drop axis.  This script therefore measures the same drop on
``n_psi`` rotations of the lattice (the drop axis stays on ``+y``, which the
axisymmetric reduction requires) and reports

* the **ensemble mean** of the frequency error -- the only component that
  survives averaging and therefore the one that has to be inside the gate,
* the **orientation scatter** (sample standard deviation over ``psi``) -- the
  discretisation noise that a single-lattice run reports as a fake answer, and
* the **bias at ``eps -> 0``** -- the ``P_l`` amplitude the operator invents out
  of a perfectly spherical drop, i.e. the noise floor of the estimator itself.

Three operators are scored on the same ensembles:

``delta``   the shipped CSF force ``-sigma kappa delta_s`` with the round-1
            curvature (``fit_quadratic`` meridional + CSF azimuthal) and the
            colour-gradient ``delta_s`` renormalised so that the discrete
            axisymmetric coarea identity holds exactly (``2 sum delta_s dV /
            (4 pi R^2) = 1``) -- without that normalisation the force is 38 %
            short by construction (``diagnose.py``).
``delta_exact``  the same measure with the *exact* curvature: isolates the
            measure/quadrature error.
``surface`` the interface-reconstruction operator: every shell particle is
            mapped to the ``c = 1/2`` level set with its own local gradient,
            a Legendre series is fitted through those sub-particle interface
            points, and the traction is integrated analytically along the
            reconstructed curve (:mod:`interface_reconstruction`).

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/ensemble.py [--quick]
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(_HERE))

from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_curvature,
    exact_modal_stress,
    legendre_p,
    rayleigh_frequency_3d,
)
from tensorsph.surface_tension_dispersion import csf_state  # noqa: E402

from interface_reconstruction import (  # noqa: E402
    legendre_basis,
    modal_stress_reconstructed,
    reconstructed_fields,
)

OUT = _HERE
GATE_PCT = 6.0
R_DEFAULT, SIGMA_DEFAULT, RHO_DEFAULT = 0.02, 7.2, 1000.0


# --------------------------------------------------------------------------- #
# geometry with a rotated lattice
# --------------------------------------------------------------------------- #

def axisym_geometry_rotated(radius: float, dx: float, *, mode: int, eps: float,
                            rho0: float, psi: float) -> dict:
    """Same construction as :func:`axisym_drop_geometry`, lattice rotated by psi.

    The drop axis stays on ``+y`` (the axisymmetric reduction needs it there);
    only the hex lattice is rotated, which is the degree of freedom the physics
    must not depend on.
    """
    h = 1.3 * dx
    dy = dx * math.sqrt(3.0) / 2.0
    extent = radius * (1.0 + abs(eps) + 1.5)
    ny = int(math.ceil(extent / dy)) + 2
    n = int(math.ceil(extent / dx)) + 2
    rows = []
    for j in range(-ny, ny + 1):
        shift = 0.5 * dx if (j % 2) else 0.0
        xs = torch.arange(-n, n + 1, dtype=torch.float64) * dx + shift
        rows.append(torch.stack((xs, torch.full_like(xs, j * dy)), dim=1))
    pts = torch.cat(rows, dim=0)
    c, s = math.cos(psi), math.sin(psi)
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    pts = pts @ rot.T
    r = torch.linalg.vector_norm(pts, dim=1)
    pts = pts[r <= radius]
    r = torch.linalg.vector_norm(pts, dim=1)
    cos_ref = pts[:, 1] / r.clamp_min(1e-30)
    scale = 1.0 + eps * legendre_p(mode, cos_ref)
    pts = pts * scale.unsqueeze(-1)
    mass2d = torch.full((pts.shape[0],), rho0 * dx * dy, dtype=pts.dtype)
    rho = pts[:, 0].abs()
    vol = math.pi * rho * (mass2d / rho0)
    rad = torch.linalg.vector_norm(pts, dim=1)
    return {"pos": pts.contiguous(), "mass": mass2d, "dx": dx, "h": h, "rho0": rho0,
            "radius": radius, "mode": mode, "eps": eps, "rho": rho,
            "rho_safe": rho.clamp_min(0.25 * h), "dvol": vol,
            "cos_theta": pts[:, 1] / rad.clamp_min(1e-30),
            "n_particles": int(pts.shape[0])}


# --------------------------------------------------------------------------- #
# level-set reconstruction of the interface (sub-particle, all shell particles)
# --------------------------------------------------------------------------- #

def fit_surface_levelset(geom: dict, state, *, kmax: int = 6,
                         cutoff_frac: float = 0.05) -> dict:
    """Map every shell particle onto the ``c = 1/2`` level set, then fit ``r(theta)``.

    ``r_int = r + (c - 1/2)/|grad c|`` uses the local colour gradient, so each
    shell particle contributes one sub-particle interface point instead of the
    one-per-sector extrapolation of
    :func:`interface_reconstruction.fit_interface_color`.
    """
    ds = state.delta_s
    keep = ds > cutoff_frac * float(ds.max())
    pos = geom["pos"][keep]
    col = state.color[keep]
    grad = torch.linalg.vector_norm(state.grad_c[keep], dim=1).clamp_min(1e-30)
    rad = torch.linalg.vector_norm(pos, dim=1)
    # outward is the direction of decreasing colour, so r_int = r + (c-1/2)/|grad c|
    r_int = rad + (col - 0.5) / grad
    c = pos[:, 1] / rad.clamp_min(1e-30)
    basis = legendre_basis(kmax, c)
    w = ds[keep]
    sw = w / w.sum()
    ata = (basis * sw[:, None]).T @ basis
    atb = (basis * sw[:, None]).T @ r_int
    coef = torch.linalg.solve(ata + 1e-12 * torch.eye(kmax + 1, dtype=ata.dtype), atb)
    resid = r_int - basis @ coef
    return {"coef": coef, "kmax": kmax, "n_surface": int(keep.sum()),
            "rms_residual_over_R": float((resid * sw).pow(2).sum().sqrt())
            / geom["radius"]}


# --------------------------------------------------------------------------- #
# the two operators, on one geometry
# --------------------------------------------------------------------------- #

def measure_delta(geom, sigma, *, mode, eps, curvature, normalize_area=True) -> dict:
    state, _, _ = csf_state(geom, sigma, periodic_x=False)
    kappa = (axisym_curvature(state, geom) if curvature == "estimator"
             else (2.0 / geom["radius"]
                   + (mode - 1.0) * (mode + 2.0) * eps
                   * legendre_p(mode, geom["cos_theta"]) / geom["radius"]))
    dvol = geom["dvol"]
    area = 4.0 * math.pi * geom["radius"] ** 2
    coarea = float(2.0 * (state.delta_s * dvol).sum() / area)
    scale = 1.0 / coarea if (normalize_area and coarea > 0.0) else 1.0
    share = legendre_p(mode, geom["cos_theta"])
    dp = float((2.0 * mode + 1.0) / area
               * (-sigma * kappa * state.delta_s * dvol * share).sum()) * scale
    return {"dp_measured": dp, "coarea_ratio": coarea,
            "area_normalization": scale}


def measure_surface(geom, sigma, *, mode, eps, kmax=6) -> dict:
    state, _, _ = csf_state(geom, sigma, periodic_x=False)
    fit = fit_surface_levelset(geom, state, kmax=kmax)
    res = modal_stress_reconstructed(geom, sigma, fit, mode=mode, eps=eps)
    return {"dp_measured": res["dp_measured"], "area_ratio": res["area_ratio"],
            "kappaR_mean": res["kappaR_mean"],
            "mode_l_amplitude": res["mode_l_amplitude"],
            "rms_residual_over_R": res["rms_residual_over_R"],
            "n_surface": res["n_surface"]}


OPERATORS = ("delta", "delta_exact", "surface")


def measure(geom, sigma, *, operator, mode, eps, kmax=6) -> dict:
    if operator == "delta":
        return measure_delta(geom, sigma, mode=mode, eps=eps, curvature="estimator")
    if operator == "delta_exact":
        return measure_delta(geom, sigma, mode=mode, eps=eps, curvature="exact")
    return measure_surface(geom, sigma, mode=mode, eps=eps, kmax=kmax)


# --------------------------------------------------------------------------- #
# ensemble driver
# --------------------------------------------------------------------------- #

def ensemble(radius, sigma, rho, *, mode, eps, rdx, psis, operators=OPERATORS,
             kmax: int = 6) -> dict:
    dp = {op: [] for op in operators}
    extra = {op: [] for op in operators}
    for psi in psis:
        geom = axisym_geometry_rotated(radius, radius / rdx, mode=mode, eps=eps,
                                       rho0=rho, psi=psi)
        for op in operators:
            r = measure(geom, sigma, operator=op, mode=mode, eps=eps, kmax=kmax)
            dp[op].append(r["dp_measured"])
            extra[op].append(r)
    dp_exact = exact_modal_stress(sigma, radius, mode, eps)
    omega_exact = rayleigh_frequency_3d(sigma, rho, radius, mode)
    out = {"mode": mode, "eps": eps, "radius_over_dx": rdx, "n_psi": len(psis),
           "omega_analytic": omega_exact, "operators": {}}
    for op in operators:
        t = torch.tensor(dp[op], dtype=torch.float64)
        mean = float(t.mean())
        std = float(t.std(unbiased=True)) if t.numel() > 1 else 0.0
        ratio = (mean / dp_exact) if dp_exact != 0.0 else float("nan")
        if math.isfinite(ratio) and ratio != 0.0:
            omega_ratio = math.copysign(1.0, ratio) * math.sqrt(abs(ratio))
            werr = (omega_ratio - 1.0) * 100.0
        else:
            omega_ratio, werr = float("nan"), float("nan")
        # scatter of the *frequency* implied by the scatter of the stiffness
        w_scatter = (0.5 * std / abs(dp_exact) * 100.0) if dp_exact != 0.0 else float("nan")
        out["operators"][op] = {
            "dp_mean": mean, "dp_std": std, "dp_exact": dp_exact,
            "omega_ratio": omega_ratio, "omega_error_pct": werr,
            "omega_scatter_pct": w_scatter,
            "dp_per_psi": dp[op],
            "extra": extra[op][0],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ladder = [12.0, 16.0] if args.quick else [12.0, 16.0, 20.0, 30.0]
    eps_list = [0.05] if args.quick else [0.02, 0.05]
    n_psi = 6 if args.quick else 24
    psis = [math.pi * k / n_psi for k in range(n_psi)]
    t0 = time.time()
    rows = []
    for eps in eps_list + [0.0]:
        for rdx in ladder:
            r = ensemble(R_DEFAULT, SIGMA_DEFAULT, RHO_DEFAULT, mode=2, eps=eps,
                         rdx=rdx, psis=psis)
            rows.append(r)
            line = f"eps={eps:<6g} R/dx={rdx:4.0f} "
            for op in OPERATORS:
                o = r["operators"][op]
                line += (f"| {op:11s} werr={o['omega_error_pct']:+8.2f}%"
                         f" noise={o['omega_scatter_pct']:6.2f}% ")
            print(line, flush=True)
    payload = {
        "case": "SPH-OSC-DROP-FREQ", "round": 2, "study": "lattice-orientation ensemble",
        "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71; Lamb 1932 Art. 275",
        "gate_pct": GATE_PCT,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "parameters": {"radius_m": R_DEFAULT, "sigma_N_m": SIGMA_DEFAULT,
                       "rho_kg_m3": RHO_DEFAULT, "n_psi": n_psi,
                       "legendre_kmax": 6, "wall_seconds": time.time() - t0},
        "rows": rows, "verdict": verdict(rows),
    }
    (OUT / "ensemble.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["verdict"], indent=2))
    return 0


def verdict(rows) -> dict:
    scored = [r for r in rows if r["eps"] > 0.0]
    out = {"gate_pct": GATE_PCT, "operators": {}}
    for op in OPERATORS:
        passing = [r for r in scored
                   if abs(r["operators"][op]["omega_error_pct"]) < GATE_PCT]
        noise = [r for r in rows if r["eps"] == 0.0]
        out["operators"][op] = {
            "n_scored": len(scored), "n_passing": len(passing),
            "passing": [{"eps": r["eps"], "radius_over_dx": r["radius_over_dx"],
                         "omega_error_pct": r["operators"][op]["omega_error_pct"],
                         "omega_scatter_pct": r["operators"][op]["omega_scatter_pct"]}
                        for r in passing],
            "worst_abs_error_pct": max(abs(r["operators"][op]["omega_error_pct"])
                                       for r in scored),
            "noise_floor_eps0": [
                {"radius_over_dx": r["radius_over_dx"],
                 "dp_mean_over_dp_eps002": r["operators"][op]["dp_mean"]
                 / exact_modal_stress(SIGMA_DEFAULT, R_DEFAULT, 2, 0.02)}
                for r in noise],
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())