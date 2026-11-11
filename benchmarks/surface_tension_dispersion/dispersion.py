"""Discrete dispersion relation of TensorSPH's CSF surface-tension operator.

What this benchmark measures
----------------------------
Two classical free-surface normal modes with a closed-form surface-tension
stiffness are used:

**capillary wave** (deep, inviscid, 2-D), ``eta(x) = -a cos(k x)``
    ``omega^2 = sigma k^3 / rho``  (Lamb 1932, Art. 267)

**oscillating 2-D drop**, ``r(theta) = R (1 + eps cos n theta)``
    ``omega^2 = n (n^2 - 1) sigma / (rho R^3)``  (Rayleigh 1879; Lamb Art. 275)

In the linear regime the surface-tension **normal stress** is the whole physics
of the restoring force:

* capillary wave: ``Delta p(x) = sigma kappa(x) = sigma k^2 a cos(k x)``
* drop:           ``Delta p(theta) = sigma (n^2 - 1) eps cos(n theta) / R^2``

and the inertia is the exact potential-flow inertia (``rho/k`` per unit area for
the wave, ``rho R/n`` for the drop).  Therefore

    omega_measured^2 / omega_theory^2 = Delta p_mode / Delta p_mode,theory ,

i.e. the frequency ratio is the *modal* normal-stress ratio of the discretised
CSF operator.  This is what the benchmark reports.  It is a **linear response
(dispersion) measurement of the discrete operator**, not a time integration:
:mod:`docs/verification_capillary_wave_open_surface.md` (m13812) measured that an
actual time integration of the same operator is inadmissible at HEAD (spurious
acceleration 23x the wave's own ``omega^2 A``), so a frequency fitted from a
trajectory would measure the spurious force, not the dispersion relation.

Why an ensemble over sub-lattice shifts
--------------------------------------
The interface cuts the particle lattice at an arbitrary phase; the CSF fields are
not invariant under that phase (staircase).  The physical (smooth) part of
``Delta p`` is, so the estimator is averaged over a ``shifts x shifts`` ensemble
of sub-lattice translations.  The ensemble standard deviation is reported: it is
the lattice noise that a single realisation would carry.

Usage::

    PYTHONPATH=src python benchmarks/surface_tension_dispersion/dispersion.py --quick
    PYTHONPATH=src python benchmarks/surface_tension_dispersion/dispersion.py \
        --out benchmarks/surface_tension_dispersion/validation_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import math
import time
from pathlib import Path

import torch

from tensorsph.csf_consistent import surface_tension_acceleration
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
torch.set_num_threads(int(os.environ.get("TSPH_THREADS", "2")))
DTYPE = torch.float64
RHO0 = 1000.0
SIGMA = 0.072          # water/air at 20 C [N/m]
H_FACTOR = 1.3
AMPLITUDE_FRACTION = 0.02   # a / lambda : linear regime, resolvable curvature
DROP_EPS = 0.10


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hex_lattice(dx: float, x0: float, x1: float, y0: float, y1: float,
                sx: float = 0.0, sy: float = 0.0) -> torch.Tensor:
    """Hexagonal lattice with a sub-lattice offset ``(sx, sy) dx``."""
    dy = dx * math.sqrt(3.0) / 2.0
    xs = torch.arange(x0 + 0.5 * dx + sx * dx, x1, dx, dtype=DTYPE)
    ys = torch.arange(y0 + 0.5 * dy + sy * dy, y1, dy, dtype=DTYPE)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    row = torch.round((pts[:, 1] - y0 - sy * dy) / dy).long()
    pts[:, 0] += torch.where(row % 2 == 1, 0.5 * dx, 0.0)
    return pts


def wave_geometry(k: float, amplitude: float, dx: float, periods: int = 2,
                  depth_factor: float = 0.8, sx: float = 0.0, sy: float = 0.0):
    """Deep-water capillary wave: fluid below ``eta(x) = -a cos(kx)``."""
    lam = 2.0 * math.pi / k
    length = periods * lam
    depth = depth_factor * lam
    cell = dx * dx * math.sqrt(3.0) / 2.0
    pts = hex_lattice(dx, 0.0, length, -depth, 0.0, sx, sy)
    eta = -amplitude * torch.cos(k * pts[:, 0])
    pos = pts[pts[:, 1] < eta]
    mass = torch.full((pos.shape[0],), RHO0 * cell, dtype=DTYPE)
    # exact outward normal of the interface graph y = eta(x)
    slope = amplitude * k * torch.sin(k * pos[:, 0])
    normal = torch.stack((-slope, torch.ones_like(slope)), dim=1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    return dict(pos=pos, mass=mass, normal=normal, lam=lam, length=length,
                depth=depth, cell=cell, amplitude=amplitude, dx=dx)


def drop_geometry(mode: int, eps: float, R: float, dx: float,
                  sx: float = 0.0, sy: float = 0.0):
    """Lobed drop ``r(theta) = R (1 + eps cos n theta)`` centred at the origin."""
    cell = dx * dx * math.sqrt(3.0) / 2.0
    reach = 1.4 * R
    pts = hex_lattice(dx, -reach, reach, -reach, reach, sx, sy)
    r = torch.linalg.vector_norm(pts, dim=1)
    theta = torch.atan2(pts[:, 1], pts[:, 0])
    r_surf = R * (1.0 + eps * torch.cos(mode * theta))
    pos = pts[r < r_surf]
    mass = torch.full((pos.shape[0],), RHO0 * cell, dtype=DTYPE)
    # exact outward normal of the polar curve r(theta)
    x, y = pos[:, 0], pos[:, 1]
    rs = torch.linalg.vector_norm(pos, dim=1)
    cos_t, sin_t = x / rs, y / rs
    r_prime = -R * eps * mode * torch.sin(mode * torch.atan2(y, x))
    # outward normal of (x, y) = r(theta)(cos theta, sin theta):  (T_y, -T_x)
    nx = r_prime * sin_t + rs * cos_t
    ny = rs * sin_t - r_prime * cos_t
    normal = torch.stack((nx, ny), dim=1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    return dict(pos=pos, mass=mass, normal=normal, radius=R, cell=cell,
                eps=eps, mode=mode, dx=dx)


# --------------------------------------------------------------------------- #
# CSF evaluation + modal analysis
# --------------------------------------------------------------------------- #
def csf_acceleration(geom, sigma: float, scheme: str, delta_norm: str,
                     stencil: float = 4.0):
    pos, mass = geom["pos"], geom["mass"]
    h = H_FACTOR * geom["dx"]
    density = torch.full((pos.shape[0],), RHO0, dtype=DTYPE)
    acc, state = surface_tension_acceleration(
        pos, density, mass, cell_linked_pairs(pos, 2.0 * h), h, sigma,
        rho0=RHO0, curvature_scheme=scheme, curvature_stencil=stencil,
        delta_normalization=delta_norm)
    return acc, state, h


def exact_wave_stress_coeff(k: float, amplitude: float, n: int = 720) -> float:
    """Mode-``k`` coefficient of the exact normal stress ``sigma*kappa*ds/dx``.

    ``eta(x) = -a cos(kx)``; ``kappa = -eta'' (1 + eta'^2)^{-3/2}``; the
    arclength factor ``ds/dx`` is kept, because the modal projection in
    :func:`wave_modal` is a volume (area) integral.  Keeping the exact geometric
    curvature instead of the linear ``a k^2`` means the reference stays valid at
    the (finite) amplitudes where the lattice noise is resolvable.
    """
    x = torch.linspace(0.0, 2.0 * math.pi / k, n + 1, dtype=DTYPE)[:-1]
    eta1 = amplitude * k * torch.sin(k * x)
    eta2 = amplitude * k * k * torch.cos(k * x)
    # Delta p = -sigma kappa with kappa = -eta''/(1+eta'^2)^{3/2}, i.e. the
    # cos-mode coefficient returned here is +a k^2 (times sigma in the caller).
    kappa = -eta2 / (1.0 + eta1 ** 2) ** 1.5
    weight = torch.sqrt(1.0 + eta1 ** 2)         # ds/dx
    return float(-2.0 / n * (kappa * weight * torch.cos(k * x)).sum())


def _with_dx(geom, dx):
    geom = dict(geom)
    geom["dx"] = dx
    return geom


def wave_modal(geom, sigma: float, k: float, scheme: str, delta_norm: str):
    """Modal normal-stress coefficient of the capillary wave (per unit length)."""
    acc, state, h = csf_acceleration(geom, sigma, scheme, delta_norm)
    pos, mass, normal = geom["pos"], geom["mass"], geom["normal"]
    V = mass / RHO0
    force = (-(acc * normal).sum(dim=1) * RHO0)          # f.n = rho a.n  [N/m^3]
    w = force * V                                        # [N/m] per particle
    dp_k = 2.0 / geom["length"] * (w * torch.cos(k * pos[:, 0])).sum()
    exact = sigma * exact_wave_stress_coeff(k, geom["amplitude"])
    measure = (state.delta_s * V).sum() / geom["length"]
    # curvature diagnostic: mode-k coefficient of the CSF curvature field itself
    return dict(dp_k=float(dp_k), exact=float(exact), ratio=float(dp_k / exact),
                surface_measure=float(measure))


def drop_modal(geom, sigma: float, scheme: str, delta_norm: str):
    """Modal normal-stress coefficient of Lamb mode ``n``."""
    acc, state, h = csf_acceleration(geom, sigma, scheme, delta_norm)
    pos, mass, normal = geom["pos"], geom["mass"], geom["normal"]
    V = mass / RHO0
    force = (-(acc * normal).sum(dim=1) * RHO0)
    w = force * V
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    n = geom["mode"]
    dp_n = (w * torch.cos(n * theta)).sum() / (math.pi * geom["radius"])
    exact = sigma * (n * n - 1) * geom["eps"] / geom["radius"]
    measure = (state.delta_s * V).sum() / (2.0 * math.pi * geom["radius"])
    return dict(dp_n=float(dp_n), exact=float(exact), ratio=float(dp_n / exact),
                surface_measure=float(measure))


def ensemble(builder, modal, sigma, scheme, delta_norm, shifts=3):
    """Average a modal estimator over sub-lattice shifts; report the spread."""
    vals = []
    for a in range(shifts):
        for b in range(shifts):
            geom = builder(a / shifts, b / shifts)
            vals.append(modal(geom))
    keys = [k for k in vals[0] if k != "ratio"]
    mean = {k: sum(v[k] for v in vals) / len(vals) for k in keys}
    std = {k: float(torch.tensor([v[k] for v in vals]).std()) for k in keys}
    ratio = mean["dp_k"] / mean["exact"] if "dp_k" in mean else mean["dp_n"] / mean["exact"]
    ratio_std = float(torch.tensor([v["dp_k"] / v["exact"] if "dp_k" in v
                                    else v["dp_n"] / v["exact"] for v in vals]).std())
    mean["ratio"] = ratio
    return mean, std, ratio_std


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
def capillary_wave_case(k: float, ppw: int, sigma: float, scheme: str,
                        delta_norm: str, shifts: int = 3):
    lam = 2.0 * math.pi / k
    dx = lam / ppw
    amplitude = AMPLITUDE_FRACTION * lam

    def builder(sx, sy):
        g = wave_geometry(k, amplitude, dx, sx=sx, sy=sy)
        g["dx"] = dx
        return g

    mean, std, rstd = ensemble(builder,
                               lambda g: wave_modal(g, sigma, k, scheme, delta_norm),
                               sigma, scheme, delta_norm, shifts)
    omega_ratio = math.sqrt(abs(mean["ratio"]))
    return dict(case=f"capillary_wave_k{k}", k=k, wavelength=lam, dx=dx,
                particles_per_wavelength=ppw, n_particles=None,
                dp_mode=mean["dp_k"], dp_exact=mean["exact"],
                stiffness_ratio=mean["ratio"], stiffness_ratio_std=rstd,
                omega_ratio=omega_ratio, omega_error_pct=(omega_ratio - 1.0) * 100.0,
                surface_measure_ratio=mean["surface_measure"],
                # curvature mode implied by force/measure: kappa_n/kappa_exact
                kappa_mode_ratio=(mean["ratio"] / mean["surface_measure"]))


def lamb_case(mode: int, R_over_dx: float, sigma: float, scheme: str,
              delta_norm: str, shifts: int = 3):
    R = 0.02
    dx = R / R_over_dx

    def builder(sx, sy):
        g = drop_geometry(mode, DROP_EPS, R, dx, sx=sx, sy=sy)
        g["dx"] = dx
        return g

    mean, std, rstd = ensemble(builder,
                               lambda g: drop_modal(g, sigma, scheme, delta_norm),
                               sigma, scheme, delta_norm, shifts)
    omega_ratio = math.sqrt(abs(mean["ratio"]))
    return dict(case=f"lamb_mode{mode}", mode=mode, radius=R, dx=dx,
                radius_over_dx=R_over_dx, eps=DROP_EPS,
                dp_mode=mean["dp_n"], dp_exact=mean["exact"],
                stiffness_ratio=mean["ratio"], stiffness_ratio_std=rstd,
                omega_ratio=omega_ratio, omega_error_pct=(omega_ratio - 1.0) * 100.0,
                surface_measure_ratio=mean["surface_measure"],
                # curvature mode implied by force/measure: kappa_n/kappa_exact
                kappa_mode_ratio=(mean["ratio"] / mean["surface_measure"]))


# --------------------------------------------------------------------------- #
# convergence
# --------------------------------------------------------------------------- #
def fit_order(rows, key="omega_error_pct"):
    """Least-squares slope of ``log|error|`` vs ``log h`` (two bins minimum)."""
    xs, ys = [], []
    for r in rows:
        err = abs(r[key])
        if err <= 0:
            continue
        xs.append(math.log(r["dx"]))
        ys.append(math.log(err))
    if len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def surface_measure_table(shifts=2):
    """Interface-area factor  sum_i delta_s,i V_i / A  on the two geometries.

    This is the direct measurement of the CSF surface-delta defect: the model's
    force per interface area is ``sigma kappa`` only if the delta function
    integrates to one interface area.
    """
    out = {}
    TARGET = 1.0
    for norm_name in ("legacy", "local_range"):
        rows = {}
        for ppw, rdx in ((32, 16), (32, 32)):
            k = 1
            lam = 2.0 * math.pi / k
            dx = lam / ppw
            amplitude = AMPLITUDE_FRACTION * lam
            vals = []
            for a in range(shifts):
                for b in range(shifts):
                    g = wave_geometry(k, amplitude, dx, sx=a / shifts, sy=b / shifts)
                    _, state, _ = csf_acceleration(g, SIGMA, "fit_quadratic", norm_name)
                    V = g["mass"] / RHO0
                    vals.append(float((state.delta_s * V).sum() / g["length"]))
            rows[f"capillary_wave_ppw{ppw}"] = sum(vals) / len(vals)
            R = 0.02
            dxd = R / rdx
            vals = []
            for a in range(shifts):
                for b in range(shifts):
                    g = drop_geometry(2, DROP_EPS, R, dxd, sx=a / shifts, sy=b / shifts)
                    _, state, _ = csf_acceleration(g, SIGMA, "fit_quadratic", norm_name)
                    V = g["mass"] / RHO0
                    vals.append(float((state.delta_s * V).sum() / (2 * math.pi * R)))
            rows[f"drop_R_over_dx{rdx}"] = sum(vals) / len(vals)
        rows["geometry_spread"] = (max(rows[k] for k in rows if isinstance(rows[k], float))
                                   / min(rows[k] for k in rows if isinstance(rows[k], float)))
        rows["target"] = TARGET
        out[norm_name] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shifts", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--scheme", default="fit_quadratic")
    ap.add_argument("--ppw", default="24,32,48")
    ap.add_argument("--rdx", default="16,24,32,48")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--modes", default="2,3,4")
    ap.add_argument("--norms", default="legacy,local_range")
    ap.add_argument("--skip-sigma", action="store_true")
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="merge this chunk into an existing --out file")
    ap.add_argument("--out", default=str(HERE / "validation_report.json"))
    args = ap.parse_args(argv)

    t0 = time.time()
    ints = lambda txt: [int(v) for v in str(txt).split(",") if v]  # noqa: E731
    if args.quick:
        ppw_list, rdx_list, ks, modes, shifts = [24, 32], [12, 16], [1, 2], [2, 3], 2
        norms = ("legacy", "local_range")
    else:
        ppw_list, rdx_list = ints(args.ppw), ints(args.rdx)
        ks, modes = ints(args.ks), ints(args.modes)
        shifts = args.shifts
        norms = tuple(v for v in args.norms.split(",") if v)

    if args.merge and Path(args.out).exists():
        report = json.loads(Path(args.out).read_text())
        for norm in norms:
            report["rows"].setdefault(norm, [])
            report["convergence"].setdefault(norm, {})
        report.setdefault("sigma_sensitivity", {})
        report.setdefault("gates", {})
        report.setdefault("surface_measure", {})
    else:
      report = dict(
        benchmark="csf_dispersion",
        reference=dict(
            capillary_wave="omega^2 = sigma k^3 / rho (Lamb 1932, Art. 267)",
            drop="omega^2 = n(n^2-1) sigma / (rho R^3) (Rayleigh 1879; Lamb Art. 275)",
            measurement=("omega ratio = sqrt(modal normal-stress ratio of the "
                         "discretised CSF operator): the inertia is the exact "
                         "potential-flow value, so the score isolates the "
                         "surface-force operator (linear-response / discrete "
                         "dispersion analysis, not a time integration)")),
        setup=dict(amplitude_fraction=AMPLITUDE_FRACTION, drop_eps=DROP_EPS,
                   h_factor=H_FACTOR, rho0=RHO0, sigma=args.sigma,
                   curvature_scheme=args.scheme, shifts=shifts),
        rows={}, convergence={}, sigma_sensitivity={}, gates={})

    for norm in norms:
        report["rows"][norm] = []
        report["convergence"][norm] = {}
        for k in ks:
            for ppw in ppw_list:
                row = capillary_wave_case(k, ppw, args.sigma, args.scheme, norm, shifts)
                report["rows"][norm].append(row)
                print(f"[{norm:11s}] wave k={k} ppw={ppw:3d} "
                      f"stiff={row['stiffness_ratio']:7.4f} "
                      f"omega_err={row['omega_error_pct']:+7.2f}% "
                      f"S/A={row['surface_measure_ratio']:.3f}", flush=True)
        for mode in modes:
            for rdx in rdx_list:
                row = lamb_case(mode, rdx, args.sigma, args.scheme, norm, shifts)
                report["rows"][norm].append(row)
                print(f"[{norm:11s}] drop n={mode} R/dx={rdx:3d} "
                      f"stiff={row['stiffness_ratio']:7.4f} "
                      f"omega_err={row['omega_error_pct']:+7.2f}% "
                      f"S/A={row['surface_measure_ratio']:.3f}", flush=True)
        for k in ks:
            rows = [r for r in report["rows"][norm] if r.get("k") == k]
            report["convergence"][norm][f"capillary_wave_k{k}"] = fit_order(rows)
        for mode in modes:
            rows = [r for r in report["rows"][norm] if r.get("mode") == mode]
            report["convergence"][norm][f"lamb_mode{mode}"] = fit_order(rows)

    if not args.skip_sigma:
      report["sigma_sensitivity"] = {}
      for norm in norms:
        rows = {}
        for factor in (0.5, 1.0, 2.0):
            row = lamb_case(2, rdx_list[-1], args.sigma * factor, args.scheme,
                            norm, shifts)
            rows[f"sigma_x{factor}"] = row["stiffness_ratio"]
        base = rows["sigma_x1.0"]
        rows["max_rel_spread"] = max(abs(v - base) / abs(base) for v in rows.values())
        report["sigma_sensitivity"][norm] = rows

    report["gates"] = {}
    for norm in norms:
        wave = [r for r in report["rows"][norm]
                if r["case"].startswith("capillary_wave") and
                r["particles_per_wavelength"] == ppw_list[-1]]
        drop = [r for r in report["rows"][norm]
                if r["case"].startswith("lamb") and r["radius_over_dx"] == rdx_list[-1]]
        orders = [v for v in report["convergence"].get(norm, {}).values()
                  if v is not None]
        c1 = max(abs(r["omega_error_pct"]) for r in wave) if wave else None
        c2 = max(abs(r["omega_error_pct"]) for r in drop) if drop else None
        report["gates"][norm] = dict(
            C1_capillary_wave_max_error_pct=c1,
            C1_within_5pct=bool(c1 is not None and c1 < 5.0),
            C2_lamb_modes_max_error_pct=c2,
            C2_within_3pct=bool(c2 is not None and c2 < 3.0),
            C3_orders=report["convergence"].get(norm, {}),
            C3_order_ge_2=bool(orders and min(orders) >= 2.0),
            C4_max_sigma_spread=report.get("sigma_sensitivity", {}).get(
                norm, {}).get("max_rel_spread"),
            C4_within_5pct=bool(report.get("sigma_sensitivity", {}).get(
                norm, {}).get("max_rel_spread", 1.0) < 0.05))
    if not args.skip_measure:
        report["surface_measure"] = surface_measure_table(shifts=max(2, shifts - 1))
    report["wall_clock_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["gates"], indent=1))
    return report


if __name__ == "__main__":
    main()