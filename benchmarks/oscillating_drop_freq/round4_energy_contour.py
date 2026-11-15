"""Round 4 of ``CASE-SPH-OSC-DROP-FREQ``: an *independent* estimator.

Why a fourth round
------------------
Round 3 reported the anchor as met (``R/dx >= 24`` keeps both modes inside the
6 % Rayleigh gate) but listed, as its own caveat #1, that the estimator is
**self-referential**: the meridian curve is fitted with a 7-term Legendre series
that can represent the very mode ``P_l`` being measured, so the reported
"amplitude recovery" could in principle be an artefact of the fitting basis
rather than a property of the colour field.  A claim that is not reproduced by
an estimator sharing no basis, no parameterisation and no differencing scheme
with the original one is not yet verified.  This round supplies that estimator.

The independent estimator (shares only the colour field, which *is* the input)
-----------------------------------------------------------------------------
1. **Interface extraction**: the ``c = 1/2`` level of the same colour field
   ``c(x) = sum_j V_j W(|x - x_j|, h)`` is followed by **marching squares on a
   Cartesian ``(rho, y)`` grid** (``matplotlib.contour``, an independently
   implemented contouring algorithm).  Round 3 cast 240 *rays* from the axis and
   interpolated in radius; the two schemes fail differently (ray casting has a
   non-uniform quadrature in arc length, marching squares trips on lattice-noise
   islands), so agreement between them is evidence about the field, not about
   the sampler.
2. **Mode readout, route A (modal projection)**: the raw contour points are used
   *directly*, with no fitting basis at all:

       a_l = (2l+1)/2 * int (r(theta) - <r>) P_l(cos theta) sin theta dtheta

   ``P_l`` is the *definition* of the mode here, not a basis the data are fitted
   onto: no coefficient is estimated by least squares, so the mode cannot be
   absorbed into the basis that is supposed to reveal it -- the specific defect
   round 3 flagged.
3. **Mode readout, route B (surface energy)**: for the volume-preserving family
   ``r(theta) = R (1 + g eps P_l)`` the surface area is even in ``eps``,
   ``A(eps) = A0 (1 + K g^2 eps^2)``, hence

       g = sqrt( [d2A/deps2]_measured / [d2A/deps2]_analytic )

   The same transfer number from a *first-order* quantity of the shape (area):
   no differencing of ``r(theta)``, no curvature, no mode projection.  Two routes
   sharing nothing but the contour either agree or the number is not a property
   of the contour.
4. **Control on the identical grid**: the whole pipeline is run on an
   *analytic* level set whose ``c = 1/2`` level is exactly
   ``r = R(1 + eps P_l)``.  That control is the estimator's own bias (grid
   resolution, contour interpolation, pole truncation) and is divided out.

Frequency statement
-------------------
Under round 3's audited projection convention (subtract the base curvature,
project on the reference measure) the modal traction is linear in the
reconstructed amplitude, so ``omega_meas/omega_exact = sqrt(dp_meas/dp_exact)
~ a_l/eps``.  The amplitude transfer is therefore the frequency error, and the
pre-registered gate ``|g - 1| < 6 %`` is the anchor's tolerance.

Part D splits the shipped CSF operator's failure (round 3 carried ``-223.8 %``
forward unretested) into *algorithm* vs *sampling*: the shipped
``fit_curvature`` is fed the **exact** outward normals, the exact particle
positions and an exact shell indicator of the deformed drop.  If the deformation
response is right under perfect inputs, the defect lives in the measured
normal/measure of the colour field, not in the fit.

Output: ``round4.json`` -- every number in ``docs/case_sph_osc_drop_freq_round4.md``
is rendered from it.

Run: ``PYTHONPATH=src python benchmarks/oscillating_drop_freq/round4_energy_contour.py``
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tensorsph.csf_curvature import fit_curvature  # noqa: E402
from tensorsph.interface_level_set import color_at_points  # noqa: E402
from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_drop_geometry, exact_modal_stress, legendre_p,
    rayleigh_frequency_3d,
)

OUT = HERE / "round4.json"
R_DEFAULT, SIGMA_DEFAULT, RHO_DEFAULT = 0.02, 7.2, 1000.0
GATE_PCT = 6.0
LEVEL = 0.5
ALL_MODES = (2, 3, 4, 5, 6)


# --------------------------------------------------------------------------- #
# exact shape family and its analytic area
# --------------------------------------------------------------------------- #
def shape_radius(theta: torch.Tensor, radius: float, mode: int,
                 eps: float) -> torch.Tensor:
    """The family the lattice is built from: ``r = R (1 + eps P_l(cos theta))``."""
    return radius * (1.0 + eps * legendre_p(mode, torch.cos(theta)))


def legendre_p_deriv(l: int, c: torch.Tensor) -> torch.Tensor:
    """``P_l'(c)`` by ``(1 - c^2) P_l' = l (P_{l-1} - c P_l)``."""
    pl = legendre_p(l, c)
    pl1 = legendre_p(l - 1, c)
    return l * (pl1 - c * pl) / (1.0 - c ** 2).clamp_min(1e-12)


def analytic_area(radius: float, mode: int, eps: float,
                  *, n_theta: int = 20001) -> float:
    """``2 pi int r sin(theta) sqrt(r^2 + r'^2) dtheta``, high resolution."""
    theta = torch.linspace(1e-9, math.pi - 1e-9, n_theta, dtype=torch.float64)
    r = shape_radius(theta, radius, mode, eps)
    # d/dtheta P_l(cos theta) = -sin(theta) P_l'(cos theta)
    dr = -radius * eps * torch.sin(theta) * legendre_p_deriv(mode, torch.cos(theta))
    integ = r * torch.sin(theta) * torch.sqrt(r ** 2 + dr ** 2)
    dth = float(theta[1] - theta[0])
    return float(2.0 * math.pi * integ.sum() * dth)


# --------------------------------------------------------------------------- #
# marching-squares contour, Cartesian grid
# --------------------------------------------------------------------------- #
def contour_points(rho: torch.Tensor, y: torch.Tensor, col: torch.Tensor,
                   *, level: float = LEVEL) -> dict:
    """Largest closed contour of ``col == level`` on the ``(rho, y)`` grid.

    ``matplotlib.contour`` is used as an independent marching-squares
    implementation.  Lattice noise makes small spurious islands, so the polyline
    with the largest extent is kept: the drop's meridian arc is the outer
    boundary and any island is by construction much shorter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # matplotlib wants Z[i, j] = f(X[j], Y[i]), i.e. (n_y, n_rho); the field is
    # held internally as (n_rho, n_y) to match the meshgrid ordering.
    z = col.numpy()
    if z.shape == (rho.numel(), y.numel()):
        z = z.T
    cs = ax.contour(rho.numpy(), y.numpy(), z, levels=[level])
    segs = [p.vertices for p in cs.get_paths() if p.vertices.shape[0] >= 8]
    plt.close(fig)
    if not segs:
        raise RuntimeError(f"no contour at level {level}")

    def extent(v):
        return float((v[:, 0].max() - v[:, 0].min())
                     + (v[:, 1].max() - v[:, 1].min()))

    pts = torch.as_tensor(max(segs, key=extent), dtype=torch.float64)
    r = torch.linalg.vector_norm(pts, dim=1)
    theta = torch.atan2(pts[:, 0].abs(), pts[:, 1])   # polar angle from +y
    return {"pts": pts, "r": r, "theta": theta,
            "n_segments": len(segs), "n_points": int(pts.shape[0])}


def contour_geometry(pts: torch.Tensor) -> dict:
    """Area, enclosed volume and extent of a meridional arc (raw polyline).

    ``A = 2 pi int rho ds`` (surface of revolution); ``V = int pi rho^2 dy``
    (Green's theorem, region on the left of the traversal).
    """
    rho = pts[:, 0].abs()
    y = pts[:, 1]
    d = torch.linalg.vector_norm(pts[1:] - pts[:-1], dim=1)
    ds = d
    rho_mid = 0.5 * (rho[1:] + rho[:-1])
    area = float(2.0 * math.pi * (rho_mid * ds).sum())
    vol = float((math.pi * rho[:-1] ** 2 * (y[1:] - y[:-1])).sum())
    return {"area": area, "volume": abs(vol), "arc_length": float(ds.sum()),
            "rho_max": float(rho.max()), "y_min": float(y.min()),
            "y_max": float(y.max()), "n_points": int(pts.shape[0])}


def _grid(geom: dict, n_grid_per_dx: float, pad: float):
    radius, dx = float(geom["radius"]), float(geom["dx"])
    step = dx / n_grid_per_dx
    n_rho = int(math.ceil(pad * radius / step)) + 1
    n_y = 2 * n_rho + 1
    rho = torch.linspace(0.0, pad * radius, n_rho, dtype=torch.float64)
    y = torch.linspace(-pad * radius, pad * radius, n_y, dtype=torch.float64)
    rr, yy = torch.meshgrid(rho, y, indexing="ij")
    return rho, y, rr.reshape(-1), yy.reshape(-1), step


def sph_contour(geom: dict, *, n_grid_per_dx: float = 6.0,
                pad: float = 1.35) -> dict:
    """Marching-squares contour of the SPH colour field."""
    rho, y, rr, yy, step = _grid(geom, n_grid_per_dx, pad)
    xq = torch.stack((rr, yy), dim=1)
    col = color_at_points(geom["pos"], geom["mass"], xq, geom["h"],
                          geom["rho0"]).reshape(rho.numel(), y.numel())
    out = contour_points(rho, y, col)
    out.update({"grid_step": step, "n_shell_particles": int(geom["pos"].shape[0])})
    return out


def analytic_contour(geom: dict, *, n_grid_per_dx: float = 6.0,
                     pad: float = 1.35) -> dict:
    """Control: contour of an analytic level set of the same exact shape.

    ``c = 1/2 (1 - tanh((r - r_shape)/h))`` crosses ``1/2`` exactly on
    ``r = r_shape``, so what is measured here is the estimator's own bias only.
    """
    rho, y, rr, yy, step = _grid(geom, n_grid_per_dx, pad)
    r = torch.sqrt(rr ** 2 + yy ** 2).clamp_min(1e-30)
    theta = torch.atan2(rr, yy)
    r_shape = shape_radius(theta, float(geom["radius"]), geom["mode"], geom["eps"])
    col = 0.5 * (1.0 - torch.tanh((r - r_shape) / float(geom["h"])))
    out = contour_points(rho, y, col.reshape(rho.numel(), y.numel()))
    out.update({"grid_step": step})
    return out


# --------------------------------------------------------------------------- #
# modal readout on a raw contour
# --------------------------------------------------------------------------- #
def uniform_theta(theta: torch.Tensor, r: torch.Tensor, *, n: int = 4001) -> dict:
    """Resample ``r(theta)`` on a uniform grid by linear interpolation.

    Interpolation *on the contour*, not a model fit: no basis is introduced and
    no coefficient is estimated.
    """
    order = torch.argsort(theta)
    th, rr = theta[order], r[order]
    keep = torch.ones_like(th, dtype=torch.bool)
    keep[1:] = (th[1:] - th[:-1]).abs() > 1e-12
    th, rr = th[keep], rr[keep]
    grid = torch.linspace(float(th[0]), float(th[-1]), n, dtype=torch.float64)
    idx = torch.searchsorted(th, grid).clamp(1, th.numel() - 1)
    t0, t1 = th[idx - 1], th[idx]
    r0, r1 = rr[idx - 1], rr[idx]
    w = ((grid - t0) / (t1 - t0).clamp_min(1e-30)).clamp(0.0, 1.0)
    return {"theta": grid, "r": r0 + w * (r1 - r0),
            "theta_lo": float(th[0]), "theta_hi": float(th[-1])}


def modal_amplitudes(theta: torch.Tensor, r: torch.Tensor,
                     *, modes=ALL_MODES) -> dict:
    """``a_l = (2l+1)/2 int (r - <r>) P_l(cos theta) sin theta dtheta``.

    Trapezoid quadrature on the uniform grid; the mean radius is removed so the
    pole truncation of the contour (both ends sit on the axis, where the
    ``sin(theta)`` weight vanishes) cannot leak into a mode.
    """
    sin = torch.sin(theta)
    mean_r = float((r * sin).sum() / sin.sum())
    out = {"mean_radius": mean_r}
    for l in modes:
        share = legendre_p(l, torch.cos(theta))
        out[f"a{l}"] = float((2.0 * l + 1.0) / 2.0
                             * torch.trapezoid((r - mean_r) * share * sin, theta))
    return out


def energy_transfer(areas, eps_grid) -> dict:
    """``g`` from the even fit ``A(eps) = A0 + c2 eps^2`` (no differencing of r)."""
    ep = torch.tensor(eps_grid, dtype=torch.float64)
    a = torch.tensor(areas, dtype=torch.float64)
    m = torch.stack([torch.ones_like(ep), ep ** 2], dim=1)
    sol = torch.linalg.lstsq(m, a[:, None]).solution.squeeze(-1)
    resid = (m @ sol - a).abs().max()
    return {"A0": float(sol[0]), "c2": float(sol[1]),
            "rel_residual_max": float(resid / a.abs().max())}


# --------------------------------------------------------------------------- #
# A. estimator control
# --------------------------------------------------------------------------- #
def estimator_control(radius: float, rho: float, *, modes, ladder, eps: float,
                      n_grid_per_dx: float) -> list:
    rows = []
    for mode in modes:
        for rdx in ladder:
            geom = axisym_drop_geometry(radius, radius / rdx, mode=mode,
                                        eps=eps, rho0=rho)
            ac = analytic_contour(geom, n_grid_per_dx=n_grid_per_dx)
            u = uniform_theta(ac["theta"], ac["r"])
            amps = modal_amplitudes(u["theta"], u["r"], modes=(mode,))
            # a_l has units of length; the seed is r = R (1 + eps P_l), so the
            # dimensionless transfer is a_l / (eps R).
            g = amps[f"a{mode}"] / (eps * radius)
            geo = contour_geometry(ac["pts"])
            a_ex = analytic_area(radius, mode, eps)
            rows.append({"mode": mode, "radius_over_dx": rdx, "eps": eps,
                         "control_transfer": g,
                         "control_error_pct": 100.0 * (g - 1.0),
                         "area_ratio": geo["area"] / a_ex,
                         "mean_radius": amps["mean_radius"],
                         "theta_span": [u["theta_lo"], u["theta_hi"]]})
            print(f"  [A] control l={mode} rdx={rdx:4.0f} g={g:.6f} "
                  f"({100*(g-1):+.3f}%) A/Aex={geo['area']/a_ex:.5f}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# B. transfer ladder on the SPH colour field (two independent routes)
# --------------------------------------------------------------------------- #
def transfer_ladder(radius: float, sigma: float, rho: float, *, modes, ladder,
                    eps_values, n_grid_per_dx: float,
                    energy_ladder=None) -> dict:
    rows = []
    for mode in modes:
        for rdx in ladder:
            # eps = 0 first: it is the *null input* of the estimator, so the
            # differential readout below is available for every seeded row.
            for eps in (0.0,) + tuple(e for e in eps_values if e != 0.0):
                geom = axisym_drop_geometry(radius, radius / rdx, mode=mode,
                                            eps=eps, rho0=rho)
                sc = sph_contour(geom, n_grid_per_dx=n_grid_per_dx)
                u = uniform_theta(sc["theta"], sc["r"])
                amps = modal_amplitudes(u["theta"], u["r"])
                geo = contour_geometry(sc["pts"])
                ac = analytic_contour(geom, n_grid_per_dx=n_grid_per_dx)
                ua = uniform_theta(ac["theta"], ac["r"])
                amps_c = modal_amplitudes(ua["theta"], ua["r"])
                a_ex = analytic_area(radius, mode, eps)
                scale = eps * radius if eps else float("nan")
                leak = {f"a{k}": amps[f"a{k}"] / scale
                        for k in ALL_MODES if k != mode}
                leak_norm = (math.sqrt(sum(v * v for v in leak.values()))
                             if eps else float("nan"))
                # a_l has units of length; the seed is r = R (1 + eps P_l), so
                # the dimensionless transfer is a_l / (eps R).
                g = amps[f"a{mode}"] / (eps * radius) if eps else float("nan")
                gc = (amps_c[f"a{mode}"] / (eps * radius)
                      if eps else float("nan"))
                rows.append({
                    "mode": mode, "radius_over_dx": rdx, "eps": eps,
                    "amplitude": amps[f"a{mode}"],
                    "transfer_sph": g,
                    "transfer_control": gc,
                    "transfer_corrected": g / gc if eps else float("nan"),
                    "spurious_amplitude": (amps[f"a{mode}"] / radius
                                           if eps == 0.0 else None),
                    # the modal traction (stiffness) is linear in the
                    # reconstructed amplitude, so omega ~ sqrt(amplitude):
                    # omega error = sqrt(g) - 1, NOT g - 1.
                    "omega_error_pct": (100.0 * (math.sqrt(g) - 1.0)
                                        if eps else float("nan")),
                    "leakage_modes": leak,
                    "leakage_rel": (leak_norm / max(abs(amps[f"a{mode}"]), 1e-30)
                                    if eps else None),
                    "area_ratio": geo["area"] / a_ex,
                    "volume_ratio": geo["volume"]
                                    / (4.0 / 3.0 * math.pi * radius ** 3),
                    "level_offset_over_dx":
                        (amps["mean_radius"] - radius) / float(geom["dx"]),
                    "n_particles": int(geom["pos"].shape[0]),
                    "dp_exact": exact_modal_stress(sigma, radius, mode, eps),
                    "omega_rayleigh": rayleigh_frequency_3d(sigma, rho, radius, mode),
                })
                r = rows[-1]
                print(f"  [B] l={mode} rdx={rdx:4.0f} eps={eps:.2f} "
                      f"g={g:+.4f} g_corr={r['transfer_corrected']:+.4f} "
                      f"A/Aex={r['area_ratio']:.4f} V/V0={r['volume_ratio']:.4f} "
                      f"leak={r['leakage_rel'] if r['leakage_rel'] is None else round(r['leakage_rel'],3)} "
                      f"off={r['level_offset_over_dx']:+.2f}dx", flush=True)

    # differential readout: subtract the eps = 0 (null) row of the same
    # (mode, rdx) cell.  Every static reconstruction bias -- the inward
    # level-set offset, the lattice asymmetry between the +x and -x halves, a
    # truncated pole cap -- is a property of the undeformed shape and cancels
    # here, leaving the operator's *response* to the mode.
    for row in rows:
        if row["eps"] == 0.0:
            continue
        null = [q for q in rows if q["mode"] == row["mode"]
                and q["radius_over_dx"] == row["radius_over_dx"]
                and q["eps"] == 0.0]
        if not null:
            continue
        a0 = null[0]["amplitude"]
        row["differential_transfer"] = ((row["amplitude"] - a0)
                                        / (row["eps"] * radius))

    # route B: second derivative of the area w.r.t. eps
    eps_energy = [0.0, 0.02, -0.02, 0.04, -0.04]
    energy = []
    for mode in modes:
        for rdx in (ladder if energy_ladder is None else energy_ladder):
            a_sph, a_ctl, a_ex = [], [], []
            for eps in eps_energy:
                geom = axisym_drop_geometry(radius, radius / rdx, mode=mode,
                                            eps=eps, rho0=rho)
                a_sph.append(contour_geometry(
                    sph_contour(geom, n_grid_per_dx=n_grid_per_dx)["pts"])["area"])
                a_ctl.append(contour_geometry(
                    analytic_contour(geom, n_grid_per_dx=n_grid_per_dx)["pts"])["area"])
                a_ex.append(analytic_area(radius, mode, eps))
            e_sph = energy_transfer(a_sph, eps_energy)
            e_ctl = energy_transfer(a_ctl, eps_energy)
            e_ex = energy_transfer(a_ex, eps_energy)
            g_energy = math.sqrt(max(e_sph["c2"], 0.0) / e_ex["c2"]) if e_ex["c2"] else float("nan")
            g_energy_corr = (math.sqrt(max(e_sph["c2"], 0.0) / max(e_ctl["c2"], 1e-30))
                             if e_ctl["c2"] else float("nan"))
            energy.append({"mode": mode, "radius_over_dx": rdx,
                           "areas_sph": a_sph, "areas_analytic_quadrature": a_ex,
                           "c2_sph": e_sph["c2"], "c2_control": e_ctl["c2"],
                           "c2_analytic": e_ex["c2"],
                           "relative_fit_residual": e_sph["rel_residual_max"],
                           "g_energy": g_energy,
                           "g_energy_corrected": g_energy_corr,
                           "omega_error_pct": 100.0 * (math.sqrt(g_energy_corr) - 1.0)})
            print(f"  [B2] l={mode} rdx={rdx:4.0f} g_energy="
                  f"{g_energy:+.4f} corr={g_energy_corr:+.4f} "
                  f"(fit resid {e_sph['rel_residual_max']:.2e})", flush=True)
    return {"rows": rows, "energy": energy, "eps_energy": eps_energy}


# --------------------------------------------------------------------------- #
# C. transfer law  g = 1 - c / (R/dx)
# --------------------------------------------------------------------------- #
def transfer_law(rows: list) -> list:
    """Least-squares ``1.91``-style coefficient, per mode, per route."""
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        for eps in sorted({r["eps"] for r in rows}):
            sel = [r for r in rows if r["mode"] == mode and r["eps"] == eps]
            x = torch.tensor([1.0 / r["radius_over_dx"] for r in sel],
                             dtype=torch.float64)
            y = torch.tensor([r["transfer_corrected"] for r in sel],
                             dtype=torch.float64)
            m = torch.stack([torch.ones_like(x), -x], dim=1)
            sol = torch.linalg.lstsq(m, y[:, None]).solution.squeeze(-1)
            resid = (m @ sol - y).abs().max()
            out.append({"mode": mode, "eps": eps,
                        "intercept": float(sol[0]), "coefficient": float(sol[1]),
                        "max_abs_residual": float(resid),
                        "n_points": len(sel),
                        "threshold_rdx_at_gate":
                            (float(sol[1]) / (1.0 - (1.0 - GATE_PCT / 100.0) ** 2)
                             if sol[1] > 0 else None)})
            print(f"  [C] l={mode} eps={eps:.2f}: g = {sol[0]:.3f} - "
                  f"{sol[1]:.3f}/(R/dx)  resid={resid:.4f}  "
                  f"R/dx(6%)={out[-1]['threshold_rdx_at_gate']}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# D. shipped CSF operator with exact inputs: algorithm or sampling?
# --------------------------------------------------------------------------- #
def brute_pairs(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """All ``i < j`` pairs within ``cutoff`` (brute force; fine for these n)."""
    d = torch.cdist(pos, pos)
    iu = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    keep = d[iu[0], iu[1]] <= cutoff
    return torch.stack((iu[0][keep], iu[1][keep]), dim=1)


def shipped_with_exact_inputs(radius: float, sigma: float, rho: float, *,
                              mode: int, eps: float, rdx: float,
                              shell_dx: float = 1.2) -> dict:
    """``fit_curvature`` fed exact positions, exact normals, exact shell mask.

    Everything the measured CSF pipeline gets wrong on a lattice (normals from
    ``grad c``, the surface delta) is replaced by its exact value, so whatever
    error is left belongs to the *fit operator's geometry response*.
    """
    geom = axisym_drop_geometry(radius, radius / rdx, mode=mode, eps=eps, rho0=rho)
    pos, dx, h = geom["pos"], float(geom["dx"]), float(geom["h"])
    r = torch.linalg.vector_norm(pos, dim=1)
    normal = pos / r.clamp_min(1e-30).unsqueeze(-1)
    theta = torch.atan2(pos[:, 0].abs(), pos[:, 1])
    r_shape = shape_radius(theta, radius, mode, eps)
    delta_s = ((r_shape - r) <= shell_dx * dx).to(pos.dtype)
    pairs = brute_pairs(pos, 4.0 * h)
    kappa_m = fit_curvature(pos, normal, delta_s, pairs, h,
                            method="fit_quadratic", stencil=4.0)
    sign = torch.where(pos[:, 0] >= 0.0, torch.ones_like(geom["rho"]),
                       -torch.ones_like(geom["rho"]))
    kappa = kappa_m + normal[:, 0] * sign / geom["rho_safe"]
    share = legendre_p(mode, geom["cos_theta"])
    area = 4.0 * math.pi * radius ** 2
    dp = float((2.0 * mode + 1.0) / area
               * ((-sigma * kappa) * delta_s * geom["dvol"] * share).sum())
    dp_ex = exact_modal_stress(sigma, radius, mode, eps)
    kappa_R = float((kappa * delta_s * geom["dvol"]).sum()
                    / (delta_s * geom["dvol"]).sum()) * radius
    return {"mode": mode, "eps": eps, "radius_over_dx": rdx,
            "n_particles": int(pos.shape[0]), "n_pairs": int(pairs.shape[0]),
            "n_shell": int(delta_s.sum()),
            "dp_measured": dp, "dp_exact": dp_ex,
            # eps = 0 has no mode to restore, so the ratio is undefined there;
            # that row exists only to read the static curvature (kappaR).
            "omega_ratio": (math.sqrt(dp / dp_ex)
                            if dp_ex != 0.0 and dp * dp_ex > 0 else float("nan")),
            "omega_error_pct": (100.0 * (math.sqrt(abs(dp / dp_ex)) - 1.0)
                                * (1.0 if dp * dp_ex > 0 else -1.0))
                               if dp_ex != 0.0 else float("nan"),
            "kappaR_mean_exact_inputs": kappa_R}


# --------------------------------------------------------------------------- #
# E. verdict
# --------------------------------------------------------------------------- #
def verdict(rows: list, energy: list, laws: list) -> dict:
    rdx_ok = {}
    for rdx in sorted({r["radius_over_dx"] for r in rows}):
        sel = [r for r in rows
               if r["radius_over_dx"] == rdx and r["eps"] != 0.0]
        # omega error = sqrt(transfer) - 1; the transfer is the stiffness gain.
        worst = max(abs(math.sqrt(r["transfer_sph"]) - 1.0) for r in sel)
        worst_d = max(abs(math.sqrt(r["differential_transfer"]) - 1.0)
                      for r in sel)
        rdx_ok[rdx] = {"worst_abs_omega_error_pct": 100.0 * worst,
                       "worst_abs_omega_error_pct_differential": 100.0 * worst_d,
                       "passes_gate_static": 100.0 * worst < GATE_PCT,
                       "passes_gate_differential": 100.0 * worst_d < GATE_PCT,
                       "n_rows": len(sel)}
    worst_energy = (max(abs(e["omega_error_pct"]) for e in energy
                        if e["omega_error_pct"] == e["omega_error_pct"])
                    if energy else float("nan"))
    return {"gate_pct": GATE_PCT,
            "per_resolution": rdx_ok,
            "energy_route_worst_pct": worst_energy,
            "transfer_law_coefficients": [l["coefficient"] for l in laws],
            "transfer_law_round3_reference": 1.91}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=R_DEFAULT)
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--grid-per-dx", type=float, default=6.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="reduced ladder for a local (<2 min) run")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    t0 = time.time()

    modes = (2, 3)
    ladder = (12.0, 16.0, 24.0) if args.quick else (12.0, 16.0, 20.0, 24.0, 32.0)
    eps_values = (0.02, 0.05)

    print("[A] estimator control (analytic level set, same grid)", flush=True)
    control = estimator_control(args.radius, args.rho, modes=modes, ladder=ladder,
                                eps=0.02, n_grid_per_dx=args.grid_per_dx)
    print("[B] SPH colour-field transfer ladder", flush=True)
    lad = transfer_ladder(args.radius, args.sigma, args.rho, modes=modes,
                          ladder=ladder, eps_values=eps_values,
                          n_grid_per_dx=args.grid_per_dx,
                          energy_ladder=(16.0, 24.0) if args.fast else None)
    print("[C] transfer law per mode", flush=True)
    laws = transfer_law(lad["rows"])
    print("[D] shipped fit_curvature with exact inputs", flush=True)
    shipped = []
    for mode in (modes[:1] if args.fast else modes):
        for rdx in ((12.0,) if args.fast else (12.0, 16.0)):
            for eps in eps_values:
                row = shipped_with_exact_inputs(args.radius, args.sigma, args.rho,
                                                mode=mode, eps=eps, rdx=rdx)
                shipped.append(row)
                print(f"  [D] l={mode} rdx={rdx:4.0f} eps={eps:.2f} "
                      f"kappaR={row['kappaR_mean_exact_inputs']:.3f} "
                      f"werr={row['omega_error_pct']:+8.2f}%", flush=True)

    out = {"case": "SPH-OSC-DROP-FREQ", "round": 4,
           "study": "independent estimator: Cartesian marching-squares contour, "
                    "basis-free mode readout, surface-energy route",
           "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71; Lamb 1932 Art. 275",
           "gate_pct": GATE_PCT,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "environment": {"python": platform.python_version(),
                           "torch": torch.__version__,
                           "device": "cpu", "threads": 4},
           "parameters": {"radius_m": args.radius, "sigma_N_m": args.sigma,
                          "rho_kg_m3": args.rho,
                          "grid_points_per_dx": args.grid_per_dx,
                          "ladder_radius_over_dx": list(ladder),
                          "energy_ladder_radius_over_dx":
                              [16.0, 24.0] if args.fast else list(ladder),
                          "fast_mode": bool(args.fast),
                          "eps_values": list(eps_values),
                          "eps_energy": lad["eps_energy"],
                          "wall_seconds": round(time.time() - t0, 1)},
           "estimator_control": control,
           "transfer_rows": lad["rows"],
           "energy_rows": lad["energy"],
           "transfer_law": laws,
           "shipped_exact_inputs": shipped,
           "verdict": verdict(lad["rows"], lad["energy"], laws)}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} in {out['parameters']['wall_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())