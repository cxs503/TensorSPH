"""Where does the CSF shell force lose its amplitude?

Why this harness exists
-----------------------
``csf_consistent`` replaced the legacy colour function and made the *curvature*
and the *delta-function normalisation* individually testable
(``tests/test_csf_consistent.py``).  What was never scored is the quantity the
pressure field actually sees: the **radial integral of the CSF force across the
interface**, which continuum-wise *is* the Laplace jump,

    dp = int_{-inf}^{+inf} f_r dr = sigma * kappa = sigma / R      (2-D circle)

with ``f = sigma kappa delta_s n`` the force per unit volume.  Field reports
quote this amplitude as ``~0.56 sigma/R``, i.e. the force carried by the shell
is 44 % short, and it is not known which of the three ingredients is
responsible.  This harness measures the decomposition rather than the total:

    A  = sum_i (-a_{r,i}) rho V_i / (sigma kappa_exact L),          L = 2 pi R
       = f_kappa * f_delta * f_n                                   (exact)

    f_kappa = sum w_i kappa_i / (kappa_exact sum w_i)      curvature bias
    f_delta = sum w_i V_i / L                              delta normalisation
    f_n     = sum w_i n_{r,i} / sum w_i                    normal coherence
    w_i     = delta_{s,i} V_i                              interface weight

The product identity holds to round-off for any field (it is a test), so a
measured ``A`` can be attributed *exactly* to the three factors instead of by
argument.  ``f_delta`` and ``f_n`` are pure geometry/truncation diagnostics;
``f_kappa`` isolates the MLS curvature.  A fourth diagnostic, ``s_i`` (the
kernel-sum deficit of a shell particle relative to a bulk particle), records
the truncation mechanism behind whichever factor turns out to dominate.

Nothing here integrates the equations of motion: every number is a one-shot
static evaluation of the SPH operators, so the harness costs seconds and does
not depend on a time step, a CFL limit or a pressure closure.  That is the
point -- the force amplitude is scored *independently of the pressure field*,
which is where the remaining Laplace gap has been blamed.

Usage::

    PYTHONPATH=src python benchmarks/csf_force_amplitude/run.py
    PYTHONPATH=src python benchmarks/csf_force_amplitude/run.py --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import (
    CSFState,
    csf_fields,
    surface_tension_acceleration,
)
from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

RADIUS = 0.02          # m, the drop radius used by tests/test_csf_consistent.py
RHO0 = 1000.0          # kg/m^3
SIGMA = 7.2            # N/m
H_FACTOR = 1.3


def _load_sibling(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "benchmarks/laplace_law/run.py")


# --------------------------------------------------------------------------- #
# Fourier decomposition of the shell force
# --------------------------------------------------------------------------- #
def shell_decomposition(state: CSFState, acc: torch.Tensor, pos: torch.Tensor,
                        mass: torch.Tensor, rho0: float, radius: float,
                        kappa_exact: float | None = None):
    """Split the CSF force amplitude into its three exact factors.

    Returns a dict with ``amplitude`` ``A`` and the factors
    ``f_kappa * f_delta * f_n == A`` (to round-off), plus the raw shell sums so
    a reader can re-derive every number.
    """
    kappa_exact = 1.0 / radius if kappa_exact is None else float(kappa_exact)
    r = torch.linalg.norm(pos, dim=-1)
    r_hat = pos / r[:, None]
    volume = mass / rho0
    w = state.delta_s * volume                       # interface weight (per particle)
    length = 2.0 * math.pi * radius                  # analytic interface length

    # signed radial acceleration: negative = inward (compressing the drop)
    a_r = (acc * r_hat).sum(dim=-1)
    radial_force = float((-a_r * rho0 * volume).sum())     # N per unit depth
    A = radial_force / (SIGMA * kappa_exact * length)

    sum_w = float(w.sum())
    f_kappa = float((w * state.kappa).sum()) / (kappa_exact * sum_w)
    f_delta = sum_w / length
    f_n = float((w * (state.normal * r_hat).sum(dim=-1)).sum()) / sum_w

    return {
        "amplitude_A": A,
        "f_kappa": f_kappa,
        "f_delta": f_delta,
        "f_normal": f_n,
        "product_residual": A - f_kappa * f_delta * f_n,
        "radial_force_N_per_m": radial_force,
        "sigma_over_R_Pa": SIGMA * kappa_exact,
        "sum_w_A": sum_w,
        "interface_length_m": length,
        "n_particles": int(pos.shape[0]),
        "n_shell": int((state.delta_s > 0.05 * float(state.delta_s.max())).sum()),
    }


def support_deficit(state: CSFState, pos: torch.Tensor, radius: float,
                    bulk_fraction: float = 0.5):
    """Kernel-sum deficit of shell particles relative to the bulk.

    ``state.color`` *is* ``sum_j V_j W_ij`` (the smoothed indicator).  A
    particle with a full kernel has ``color ~ 1``; a shell particle is missing
    the void half of its support, so ``color < 1``.  The ratio over the
    interface shell is the truncation measure behind ``f_delta`` / ``f_kappa``.
    """
    r = torch.linalg.norm(pos, dim=-1)
    shell = state.delta_s > 0.05 * float(state.delta_s.max())
    bulk = r < bulk_fraction * radius
    bulk_value = float(state.color[bulk].mean())
    return {
        "color_shell_mean": float(state.color[shell].mean()),
        "color_bulk_mean": bulk_value,
        "support_ratio": float(state.color[shell].mean()) / bulk_value,
    }


def radial_force_profile(state: CSFState, acc: torch.Tensor, pos: torch.Tensor,
                         mass: torch.Tensor, rho0: float, radius: float,
                         n_bins: int = 24):
    """Radial force density per unit depth, binned in ``r/R``.

    The continuum target is ``sigma * kappa * delta_s`` concentrated on the
    interface; with ``int delta_s dr = 1`` the integral over ``r`` must return
    ``sigma kappa`` regardless of how the shell is resolved.
    """
    r = torch.linalg.norm(pos, dim=-1) / radius
    r_hat = pos / torch.linalg.norm(pos, dim=-1)[:, None]
    a_r = (acc * r_hat).sum(dim=-1)
    bins = torch.linspace(0.0, 1.1, n_bins + 1, dtype=pos.dtype)
    edges = bins.tolist()
    profile = []
    for k in range(n_bins):
        sel = (r >= edges[k]) & (r < edges[k + 1])
        if not bool(sel.any()):
            continue
        # volume per unit radial length in the annulus: V / dr
        dr = (edges[k + 1] - edges[k]) * radius
        f_r = float((-a_r[sel] * rho0 * mass[sel] / rho0).sum()) / dr
        profile.append({"r_over_R": 0.5 * (edges[k] + edges[k + 1]),
                        "dF_dr_N_per_m2": f_r})
    return profile


def angle_resolved(state: CSFState, acc: torch.Tensor, pos: torch.Tensor,
                   mass: torch.Tensor, rho0: float, radius: float,
                   n_bins: int = 24):
    """``A(theta)``: radial force per unit polar angle, normalised by sigma*kappa*R.

    ``d(int f_r dA)/dtheta = sigma kappa R`` for a circular interface, so the
    mean of ``A(theta)`` over uniform bins equals the global ``A``.  The spread
    is a lattice-anisotropy measure: a coarse hexagonal packing cannot make the
    shell force axisymmetric, and that anisotropy is what makes the *x*-projected
    cut-force identity and the *radial* integral disagree.
    """
    r = torch.linalg.norm(pos, dim=-1)
    r_hat = pos / r[:, None]
    a_r = (acc * r_hat).sum(dim=-1)
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    volume = mass / rho0
    edges = torch.linspace(-math.pi, math.pi, n_bins + 1, dtype=pos.dtype)
    dtheta = float(edges[1] - edges[0])
    norm = SIGMA * (1.0 / radius) * radius * dtheta
    out = []
    for k in range(n_bins):
        sel = (theta >= edges[k]) & (theta < edges[k + 1])
        if not bool(sel.any()):
            continue
        out.append({
            "theta_mid_deg": math.degrees(0.5 * float(edges[k] + edges[k + 1])),
            "A_local": float((-a_r[sel] * rho0 * volume[sel]).sum()) / norm,
        })
    vals = [o["A_local"] for o in out]
    mean = sum(vals) / len(vals)
    return out, {"A_theta_mean": mean,
                 "A_theta_min": min(vals), "A_theta_max": max(vals),
                 "A_theta_spread_rel": (max(vals) - min(vals)) / mean}


def cut_force_ratio(acc: torch.Tensor, pos: torch.Tensor, mass: torch.Tensor,
                    sigma: float):
    """The repo's existing amplitude anchor: ``sum_{x>0} m a_x / (-2 sigma)``.

    It is the *vector* (x-projected) form of the line-tension identity, i.e. the
    continuum value is exactly 1.  Quoted here so the two integral measures of
    the same force field sit in one artifact.
    """
    right = pos[:, 0] > 0
    f_cut = float((mass[right, None] * acc[right]).sum(dim=0)[0])
    return f_cut / (-2.0 * sigma)


def delta_s_integral_local(state: CSFState, pos: torch.Tensor, mass: torch.Tensor,
                           rho0: float, radius: float, lo: float = 0.6,
                           hi: float = 1.05):
    """The repo's local-circumference normalisation, for reconciliation.

    ``int delta_s dA = 2 pi R int delta_s dn`` is recovered by dividing the shell
    sum by the *local* circumference ``2 pi r`` instead of ``2 pi R``.  It
    differs from ``f_delta`` by where the shell weight sits in ``r``.
    """
    r = torch.linalg.norm(pos, dim=-1)
    m = (r > lo * radius) & (r < hi * radius)
    volume = mass / rho0
    contrib = (state.delta_s[m] * volume[m] / (2.0 * math.pi * r[m].clamp_min(1e-30))).sum()
    return float(contrib)


# --------------------------------------------------------------------------- #
# one resolution
# --------------------------------------------------------------------------- #
def run_resolution(dx: float, radius: float = RADIUS, sigma: float = SIGMA,
                   rho0: float = RHO0, h_factor: float = H_FACTOR):
    cfg = Config(dx=dx, h_factor=h_factor, rho0=rho0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01, surface_tension_coefficient=sigma)
    h = cfg.h_factor * cfg.dx
    pos, mass = _law.hexagonal_droplet(radius, cfg.dx, rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    state = csf_fields(pos, mass, pairs, h, rho0)
    rho = torch.full_like(mass, rho0)
    acc, _ = surface_tension_acceleration(pos, rho, mass, pairs, h, sigma, rho0)

    out = shell_decomposition(state, acc, pos, mass, rho0, radius)
    out.update(support_deficit(state, pos, radius))
    angles, angle_stats = angle_resolved(state, acc, pos, mass, rho0, radius)
    out["cut_force_ratio"] = cut_force_ratio(acc, pos, mass, sigma)
    out["delta_s_integral_local"] = delta_s_integral_local(state, pos, mass, rho0, radius)
    out["projection_disagreement"] = abs(out["cut_force_ratio"] - out["amplitude_A"])
    out.update(angle_stats)
    out["angle_profile"] = angles
    out["dx"] = float(dx)
    out["R_over_dx"] = float(radius / dx)
    out["h_over_dx"] = float(h_factor)
    out["profile"] = radial_force_profile(state, acc, pos, mass, rho0, radius)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="single coarse resolution")
    ap.add_argument("--out", default=str(HERE / "report.json"))
    args = ap.parse_args(argv)

    ladder = [0.004, 0.002, 0.00125]
    if args.quick:
        ladder = [0.004]

    rows = []
    for dx in ladder:
        row = run_resolution(dx)
        rows.append(row)
        print(f"dx={dx:<8g} R/dx={row['R_over_dx']:5.1f}  "
              f"A={row['amplitude_A']:.4f}  "
              f"f_kappa={row['f_kappa']:.4f}  f_delta={row['f_delta']:.4f}  "
              f"f_n={row['f_normal']:.4f}  support={row['support_ratio']:.4f}  "
              f"|res|={abs(row['product_residual']):.2e}  "
              f"cut={row['cut_force_ratio']:.4f}  "
              f"d_s_local={row['delta_s_integral_local']:.4f}  "
              f"A(theta)spread={row['A_theta_spread_rel']:.3f}")

    report = {
        "benchmark": "csf_force_amplitude",
        "question": "does the CSF shell force integrate to sigma*kappa, and if "
                    "not, which of curvature / delta-normalisation / normal "
                    "coherence is short",
        "geometry": {"radius_m": RADIUS, "rho0": RHO0, "sigma": SIGMA,
                     "h_factor": H_FACTOR, "lattice": "hexagonal",
                     "analytic_target": "int f_r dr = sigma * kappa = sigma/R"},
        "reference": {
            "sigma_over_R_Pa": SIGMA / RADIUS,
            "source": "Young-Laplace for a 2-D circular interface (Brackbill "
                      "et al. 1992, eq. 2; the CSF force is the discrete "
                      "gradient of that jump)",
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main() else 0)