"""Round 5 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE: a *non-constant* curvature interface.

Why this round exists
---------------------
Rounds 1-4 (``docs/xval_projection_laplace_oss.md``, ``..._robustness.md``,
``..._density_ratio.md``) measured the jump ``dp = sigma/R`` on a **circular**
droplet, and a circle has one number for its curvature.  Every gate they declare
therefore tests only the *global* form of the law, ``<dp> = sigma <kappa>``: a
curvature estimator with a station-dependent error cancels in the integral, and a
constant-relative-bias estimator is indistinguishable from a calibration.  The
form the CSF model actually discretises is the *differential* one,

    dp(s) = sigma kappa(s)      at every interface station s,

and a circle cannot test it, because ``kappa`` is constant there by construction.

This round replaces the circle with an **ellipse** of the same area
(``pi a b = pi R_eq**2``), whose curvature varies along the interface by
``(a/b)**3`` (``kappa_min = b/a**2`` at ``t = 0, pi``; ``kappa_max = a/b**2`` at
``t = pi/2``).  ``R_eq`` is held at the round-1 value, so ``R_eq/dx`` stays
comparable with the previous rounds.

Instruments (all station-resolved, all numbers from the JSON)
------------------------------------------------------------
1. ``dp_flux(t)`` = ``-sum_k f_k . n V_k / L`` over the interface shell in an
   angular window around station ``t`` -- the traction line integral.  It never
   touches the pressure solve, so it measures the CSF construction alone.
2. ``dp_pressure(t)`` = ``<p>_inner(t) - <p>_outer(t)`` from the round-4 rho-blind
   force-density Poisson solve -- the *projection* path, the number a solver
   reports.
3. ``kappa_model(t)`` = ``delta_s``-weighted mean of the module curvature in the
   same window, i.e. the curvature field the force is built from.
4. Whitney-Graustein gauge ``(1/2 pi) oint kappa dl = 1``: exact for *any* smooth
   closed convex curve at *any* aspect ratio and resolution.  A scalar test of
   the curvature field with no Poisson solve and no pressure reading.
5. Net traction ``|oint sigma kappa n dl|``: zero for a closed interface, and on
   an ellipse the summands are not symmetric, so a curvature field that is right
   on average but wrong in distribution leaves a residual.

The reference of an instrument that averages over a window of width ``2*Delta``
is the *arc-length-weighted* mean of the exact curvature over the same window,
``kappa_ref = oint_window kappa dl / L`` -- not ``kappa(t_center)``: at
``Delta = 30 deg`` on an ellipse the two differ by tens of percent, and comparing
against the centre value would report the curvature variation as an error of the
method.

Ablation ``variant="kappa_exact"``: the analytic ellipse curvature per particle,
with the model ``delta_s``/normal/operators.  Whatever error remains is *not* the
curvature estimator -- it is the accuracy floor of the instrument and of the
``delta_s`` normalisation, and it is what the model arm's station noise has to be
judged against.

GATES are declared in ``GATES`` below, before the sweep is run, are copied
verbatim into the JSON, and are **not** relaxed afterwards.  They are scoped by a
resolvability precondition fixed a priori: an interface point is *resolved* when
its radius of curvature spans at least ``R_CURV_MIN_DX`` cells (below that the
curvature is not representable by the kernel and no accuracy claim is made).  A
gate that the measurement breaks is recorded as failed and, in the pytest module,
as a strict ``xfail``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.csf_consistent import mls_divergence  # noqa: E402
from tensorsph.laplace_isph import (  # noqa: E402
    csf_state_two_phase, hex_block, laplace_isph_case, pressure_poisson,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# declared parameters and gates (fixed before the sweep is run)
# --------------------------------------------------------------------------- #
SIGMA = 0.072          # N/m, water/air at 20 C
RHO0 = 1000.0          # kg/m^3, both phases (static verification carrier)
R_EQ = 0.02            # m, area-equivalent radius (= round 1's R)
OUTER_DX = 12.0        # Dirichlet block thickness, in dx (round-2 baseline)
N_STATION = 24         # stations around the ellipse
R_CURV_MIN_DX = 5.0    # a priori resolvability precondition

GATES = {
    # owner standard: an error > 15 % on a key quantity is not acceptable;
    # 3 % is the "publishable" band used by rounds 2-4 for the validated carrier.
    "local_bias_pct_circle_R_over_dx_ge_20": 3.0,
    "local_bias_pct_aspect_le_1p5_resolved": 3.0,
    "local_bias_pct_all_aspects_R_over_dx_40": 15.0,
    # pointwise (station-scale) verification: the field itself, not its mean.
    # Declared at the owner's 15 % because the round-1..4 claim is a *jump*
    # claim; if the pointwise field cannot meet it, that is the finding.
    "station_rms_pct_aspect_le_2_resolved": 15.0,
    # curvature field alone (Whitney-Graustein, exact for every closed convex curve)
    "whitney_graustein_err_pct_resolved": 1.0,
    # the number a projection solver actually reports: the pressure readout.  Same
    # owner gate, but the readout is a *different* instrument and it is allowed to
    # be the one that fails.
    "pressure_bias_pct_all_aspects_R_over_dx_40": 15.0,
    # closed interface: the traction must not push the drop anywhere (model arm --
    # the scheme under test; the analytic-curvature ablation is reported only,
    # because an exact curvature field is *less* force-balanced against the
    # shipped smoothed normals, which is itself a finding)
    "net_traction_residual_model_arm": 1.0e-2,
    # two independent instruments must agree on the *bias* (round-2 convention:
    # a readout spread of 5 % is the gate)
    "pressure_vs_flux_bias_diff_pct_aspect_le_1p5": 5.0,
    # sigma = 0 => no traction at all
    "zero_sigma_abs_dp_Pa": 1.0e-9,
    # the a=b control must reproduce the shipped radius-based case
    "circle_vs_module_dp_diff_pct": 2.0,
    # window-scale ladder: averaging over a longer interface arc must bring the
    # station scatter into the 15 % band and must beat the kernel-scale scatter by
    # at least a factor 2
    "window_ladder_rms_pct_at_30deg_aspect_1p5": 15.0,
    "window_ladder_noise_halving": 0.5,
    # mechanism: with *both* the analytic curvature and the per-station delta_s
    # normalisation the local law must close to the instrument floor
    "both_exact_bias_pct_aspect_le_2_resolved": 1.5,
}

ASPECTS = (1.0, 1.5, 2.0, 3.0)
R_OVER_DX = (10.0, 20.0, 40.0)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def ellipse_axes(aspect: float, r_eq: float = R_EQ):
    """Semi-axes with the round-1 area: ``pi a b = pi r_eq**2``."""
    return r_eq * math.sqrt(aspect), r_eq / math.sqrt(aspect)


def ellipse_lattice(a: float, b: float, dx: float, outer: float,
                    rho0: float = RHO0):
    """Triangular lattice block, phases given by ``(x/a)^2 + (y/b)^2 < 1``.

    The block comes from the shipped ``hex_block`` at the *major* semi-axis (so
    the whole ellipse plus the vapour ring is inside) and only the phase label is
    replaced.  The cell volume is ``dx*dy`` everywhere, exactly as in rounds 1-4,
    so the colour function is the only object that sees the shape.
    """
    pos, mass, _circle, half = hex_block(a, dx, outer, rho0)
    liquid = (pos[:, 0] / a) ** 2 + (pos[:, 1] / b) ** 2 < 1.0
    return pos, mass, liquid, half


def ellipse_geometry(a: float, b: float, n: int):
    """Stations: parameter, position, outward normal, curvature, ``|dX/dt|``."""
    t = 2.0 * math.pi * torch.arange(n, dtype=torch.float64) / n
    x, y = a * torch.cos(t), b * torch.sin(t)
    nx, ny = torch.cos(t) / a, torch.sin(t) / b
    nn = torch.sqrt(nx * nx + ny * ny)
    s2 = (a * torch.sin(t)) ** 2 + (b * torch.cos(t)) ** 2
    kappa = a * b / s2 ** 1.5
    return (t, torch.stack([x, y], dim=1),
            torch.stack([nx / nn, ny / nn], dim=1), kappa, torch.sqrt(s2))


def kappa_exact_at(pos, a: float, b: float):
    """Analytic ellipse curvature at the station parameter of each point."""
    t = torch.atan2(pos[:, 1] / b, pos[:, 0] / a)
    s2 = (a * torch.sin(t)) ** 2 + (b * torch.cos(t)) ** 2
    return a * b / s2 ** 1.5, t


def kappa_window_mean(a: float, b: float, t0: float, half_window: float,
                      n_quad: int = 257):
    """``oint_window kappa dl / L`` -- the reference of a windowed average."""
    tt = torch.linspace(t0 - half_window, t0 + half_window, n_quad,
                        dtype=torch.float64)
    s2 = (a * torch.sin(tt)) ** 2 + (b * torch.cos(tt)) ** 2
    kap = a * b / s2 ** 1.5
    spd = torch.sqrt(s2)
    dl = spd * (tt[1] - tt[0])
    return float((kap * dl).sum() / dl.sum())


def interface_distance(pos, a: float, b: float, n_newton: int = 4):
    """Signed distance to the ellipse (positive outside) and the projection parameter.

    The shell band (``|d| <= 3 dx``) and the pressure probes must be selected by
    the *distance to the interface*, not by the projection on a station's plane
    normal: on a curved interface the plane slab of a wide angular window
    truncates --- at ``R = 40 dx`` a point 30 deg along the interface is already
    ``R (1 - cos 30) = 5.4 dx`` away *along the normal* while sitting exactly on
    the interface.  That would make the measured ``oint delta_s dn`` (and hence
    the traction integral) depend on the window width, which is an instrument
    artefact, not physics.

    Newton on the parametric projection ``t -> (a cos t, b sin t)`` (a few
    iterations from the ray parameter), then the signed normal offset.
    """
    t = torch.atan2(pos[:, 1] / b, pos[:, 0] / a)
    for _ in range(n_newton):
        c, s_ = torch.cos(t), torch.sin(t)
        dxp = pos[:, 0] - a * c
        dyp = pos[:, 1] - b * s_
        vx, vy = -a * s_, b * c                       # dX/dt
        sp2 = (vx * vx + vy * vy).clamp_min(1e-300)
        t = t + (dxp * vx + dyp * vy) / sp2
    c, s_ = torch.cos(t), torch.sin(t)
    dxp = pos[:, 0] - a * c
    dyp = pos[:, 1] - b * s_
    nx, ny = c / a, s_ / b
    nn = torch.sqrt(nx * nx + ny * ny).clamp_min(1e-300)
    d = (dxp * nx + dyp * ny) / nn                    # outward positive
    return d, t, torch.stack([nx / nn, ny / nn], dim=1)


def station_map(t_i, t_st, half_window: float):
    """Window index, signed normal distance and parametric angle difference.

    Two properties are load-bearing here:

    * the angular window is the ellipse *parameter* ``t``
      (``x = a cos t``, ``y = b sin t``), not the polar angle ``atan2(y, x)``.
      On a circle the two coincide; on an ellipse they differ by up to
      ``atan(a/b) - 45 deg``, so a polar-angle window would probe a different
      piece of the interface (and a different exact curvature) than its station.
    * the windows **tile in ``t``** (``n_station * 2 * half_window = 2 pi``), so
      every interface particle belongs to exactly one window and the window
      arclength used to normalise the traction integral is the arclength the
      window actually covers.  A nearest-station (Voronoi) assignment instead
      loses part of the arc as soon as the windows are wide: at
      ``half_window = 30 deg`` the Voronoi cell of a station on a 1.5:1 ellipse
      holds only 67 % of its arc, and the *instrument* then reports a -33 %
      deficit that has nothing to do with the physics.
    """
    n = len(t_st)
    t_i = t_i % (2.0 * math.pi)
    # bins are *centred* on the stations (t_k = 2 pi k / n, width 2 half_window),
    # so each station is the midpoint of its own window and the windows tile
    shifted = (t_i + half_window) % (2.0 * math.pi)
    k = torch.clamp((shifted * n / (2.0 * math.pi)).floor().long(), 0, n - 1)
    dt = t_i - t_st[k]
    dt = (dt + math.pi) % (2.0 * math.pi) - math.pi
    return k, dt


# --------------------------------------------------------------------------- #
# one equilibrium
# --------------------------------------------------------------------------- #
def ellipse_case(aspect: float = 1.5, r_eq: float = R_EQ, dx: float | None = None,
                 sigma: float = SIGMA, rho0: float = RHO0, h_factor: float = 1.3,
                 outer_dx: float = OUTER_DX, variant: str = "model",
                 n_station: int = N_STATION, half_window=None,
                 shell_dx: float = 3.0, probe_in_dx=(2.0, 4.0),
                 probe_out_dx=(3.0, 6.0)):
    """Static equilibrium of the CSF traction on an elliptical two-phase carrier.

    ``variant``
        ``model``          shipped curvature + shipped ``delta_s = |grad c|``
        ``kappa_exact``    analytic ellipse curvature per particle, shipped ``delta_s``
        ``delta_local``    shipped curvature, ``delta_s`` renormalised *per station*
                           so that ``oint delta_s dn = 1`` locally (diagnostic: the
                           correction uses the exact arclength, so it is not a
                           usable scheme -- it measures what the deficit costs)
        ``both_exact``     the two ablations together: everything that is left is
                           the operator/instrument floor
    """
    if dx is None:
        dx = r_eq / 20.0
    if half_window is None:
        half_window = math.pi / N_STATION
    a, b = ellipse_axes(aspect, r_eq)
    h = h_factor * dx
    pos, mass, liquid, half = ellipse_lattice(a, b, dx, outer_dx * dx, rho0)
    cell = dx * dx * math.sqrt(3.0) / 2.0
    rho_phase = mass / cell
    st = csf_state_two_phase(pos, mass, liquid, h, rho0, sigma,
                             curvature_method="mls", rho_phase=rho_phase)
    vol, pairs, delta_s, normal = st["vol"], st["pairs"], st["delta_s"], st["normal"]
    active = st["active"]

    t_st, X, Nst, k_center, speed = ellipse_geometry(a, b, n_station)
    k_exact_pt, _ = kappa_exact_at(pos, a, b)
    kappa = (torch.where(active, k_exact_pt, torch.zeros_like(k_exact_pt))
             if variant in ("kappa_exact", "both_exact") else st["kappa"])

    # ---- station-resolved geometry (needed by the delta_s renormalisation) --- #
    d_if, t_if, _n_if = interface_distance(pos, a, b)
    k_near, dt_near = station_map(t_if, t_st, half_window)
    s_near = d_if
    # traction shell: the delta_s band used by the line integral
    shell = s_near.abs() <= shell_dx * dx
    # probe band: the pressure readout needs points *outside* the traction shell
    band = s_near.abs() <= max(shell_dx, probe_out_dx[1] + 2.0) * dx
    win_all = band & (dt_near.abs() <= half_window)
    in_win = shell & (dt_near.abs() <= half_window)
    l_win = [float(speed[kk]) * 2.0 * half_window for kk in range(n_station)]
    w_all = delta_s * vol
    # per-station normal integral of delta_s: (volume integral over the window
    # shell) / (arclength of the window) == oint delta_s dn for that patch
    ds_num = torch.zeros(n_station, dtype=pos.dtype)
    ds_num = ds_num.index_add(0, k_near[in_win], w_all[in_win])
    ds_line = [float(ds_num[kk]) / l_win[kk] if l_win[kk] > 0 else float("nan")
               for kk in range(n_station)]

    if variant in ("delta_local", "both_exact"):
        scale = torch.ones_like(delta_s)
        for kk in range(n_station):
            ok = (k_near == kk) & in_win
            if bool(ok.any()) and math.isfinite(ds_line[kk]) and ds_line[kk] > 1e-6:
                scale = torch.where(k_near == kk, 1.0 / ds_line[kk], scale)
        delta_use = delta_s * scale
    else:
        delta_use = delta_s

    f = (-sigma * kappa * delta_use)[:, None] * normal          # force density
    div_f = mls_divergence(pos, f, vol, pairs, h)
    source = -div_f

    edge = 1.6 * dx
    dirichlet = ((pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge))
    p, n_cg = pressure_poisson(pos, source, pairs, vol, h, dirichlet, rel_tol=1e-10)

    rows = []
    for kk in range(n_station):
        win = in_win & (k_near == kk)
        wprobe = win_all & (k_near == kk)
        if not bool(win.any()) or l_win[kk] <= 0.0:
            rows.append(None)
            continue
        l_k = l_win[kk]
        k_ref = kappa_window_mean(a, b, float(t_st[kk]), half_window)
        # inward traction per unit arclength, with the *particle's own* normal:
        # f is along -n_i, so -f.n_i = |f|.  (Projecting on the station normal
        # instead would multiply the window edges by cos(Delta), a pure
        # instrument error that grows with the window width.)
        flux = -float((f[win] * normal[win]).sum(dim=1).mul(vol[win]).sum()) / l_k
        w = delta_use[win] * vol[win]
        k_model = float((kappa[win] * w).sum() / w.sum().clamp_min(1e-300))
        inner = wprobe & (s_near < -probe_in_dx[0] * dx) & (s_near > -probe_in_dx[1] * dx)
        outer = wprobe & (s_near > probe_out_dx[0] * dx) & (s_near < probe_out_dx[1] * dx)
        dp_p = (float(p[inner].mean()) - float(p[outer].mean())
                if bool(inner.any()) and bool(outer.any()) else float("nan"))
        ref = sigma * k_ref
        rows.append({
            "station": kk,
            "t_deg": float(t_st[kk]) * 180.0 / math.pi,
            "x_m": float(X[kk, 0]), "y_m": float(X[kk, 1]),
            "kappa_exact_center_per_m": float(k_center[kk]),
            "kappa_exact_window_per_m": k_ref,
            "kappa_model_per_m": k_model,
            "kappa_ratio": k_model / k_ref,
            "kappa_err_pct": (k_model / k_ref - 1.0) * 100.0,
            "dp_theory_Pa": ref,
            "dp_flux_Pa": flux,
            "dp_flux_err_pct": (flux / ref - 1.0) * 100.0 if sigma else float("nan"),
            "dp_pressure_Pa": dp_p,
            "dp_pressure_err_pct": ((dp_p / ref - 1.0) * 100.0
                                    if math.isfinite(dp_p) and sigma else float("nan")),
            "delta_s_normal_integral": ds_line[kk],
            "n_inner": int(inner.sum()), "n_outer": int(outer.sum()),
        })

    good = [r for r in rows if r is not None]
    err_f = [r["dp_flux_err_pct"] for r in good]
    err_p = [r["dp_pressure_err_pct"] for r in good
             if math.isfinite(r["dp_pressure_err_pct"])]
    err_k = [r["kappa_err_pct"] for r in good]
    err_d = [(r["delta_s_normal_integral"] - 1.0) * 100.0 for r in good
             if math.isfinite(r["delta_s_normal_integral"])]
    f_abs = float((torch.linalg.vector_norm(f, dim=1) * vol).sum())
    net = (f * vol[:, None]).sum(dim=0)

    def stat(xs):
        if not xs:
            return {"mean": float("nan"), "rms": float("nan"), "max_abs": float("nan")}
        m = sum(xs) / len(xs)
        return {"mean": m, "rms": math.sqrt(sum(x * x for x in xs) / len(xs)),
                "max_abs": max(abs(x) for x in xs)}

    r_curv_min = float(1.0 / k_center.max())
    return {
        "aspect_ratio": aspect, "variant": variant,
        "a_m": a, "b_m": b, "r_eq_m": r_eq, "dx_m": dx, "h_m": h,
        "r_eq_over_dx": r_eq / dx,
        "r_curv_min_m": r_curv_min,
        "r_curv_min_over_dx": r_curv_min / dx,
        "resolved": bool(r_curv_min / dx >= R_CURV_MIN_DX),
        "kappa_min_per_m": float(k_center.min()),
        "kappa_max_per_m": float(k_center.max()),
        "kappa_ratio_max_over_min": float(k_center.max() / k_center.min()),
        "n_particles": int(pos.shape[0]), "n_liquid": int(liquid.sum()),
        "sigma_N_per_m": sigma, "h_factor": h_factor,
        "n_station": n_station,
        "half_window_deg": half_window * 180.0 / math.pi,
        "flux_err": stat(err_f), "pressure_err": stat(err_p), "kappa_err": stat(err_k),
        "delta_s_norm_err": stat(err_d),
        "delta_s_norm_mean": (1.0 + (sum(err_d) / len(err_d)) / 100.0) if err_d else float("nan"),
        "whitney_graustein_kappa_line": float((kappa * w_all).sum()),
        "whitney_graustein_err_pct": (float((kappa * w_all).sum()) / (2.0 * math.pi) - 1.0) * 100.0,
        "net_traction_residual": float(torch.linalg.vector_norm(net)) / max(f_abs, 1e-300),
        "poisson_source_charge_rel": (float(div_f.mul(vol).sum())
                                      / max(float(div_f.abs().mul(vol).sum()), 1e-300)),
        "cg_iterations": int(n_cg),
        "mean_dp_flux_Pa": (sum(r["dp_flux_Pa"] for r in good) / len(good)) if good else float("nan"),
        "mean_dp_pressure_Pa": (sum(r["dp_pressure_Pa"] for r in good
                                    if math.isfinite(r["dp_pressure_Pa"]))
                                / max(len([r for r in good
                                            if math.isfinite(r["dp_pressure_Pa"])]), 1)),
        "mean_kappa_exact_per_m": sum(r["kappa_exact_window_per_m"] for r in good) / len(good),
        "mean_kappa_model_per_m": sum(r["kappa_model_per_m"] for r in good) / len(good),
        "stations": rows,
    }


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def sweep():
    configs = []
    for aspect in ASPECTS:
        for rdx in R_OVER_DX:
            print(f"[curv-variation] aspect={aspect:g} R_eq/dx={rdx:g}", flush=True)
            cfg = ellipse_case(aspect=aspect, dx=R_EQ / rdx)
            for v in ("kappa_exact", "delta_local", "both_exact"):
                abl = ellipse_case(aspect=aspect, dx=R_EQ / rdx, variant=v)
                cfg[f"ablation_{v}"] = {
                    "flux_err": abl["flux_err"], "kappa_err": abl["kappa_err"],
                    "pressure_err": abl["pressure_err"],
                    "delta_s_norm_mean": abl["delta_s_norm_mean"],
                    "whitney_graustein_err_pct": abl["whitney_graustein_err_pct"],
                    "net_traction_residual": abl["net_traction_residual"],
                    "mean_dp_flux_Pa": abl["mean_dp_flux_Pa"],
                }
            cfg["bias_decomposition"] = {
                "total_pct": cfg["flux_err"]["mean"],
                "delta_s_norm_pct": cfg["delta_s_norm_err"]["mean"],
                "kappa_pct": cfg["kappa_err"]["mean"],
                "residual_pct": (cfg["flux_err"]["mean"]
                                 - cfg["delta_s_norm_err"]["mean"]
                                 - cfg["kappa_err"]["mean"]),
            }
            configs.append(cfg)

    # window-scale ladder: how long an interface arc must be averaged over before
    # the pointwise curvature noise is beaten down to the 15 % band
    print("[curv-variation] window-scale ladder", flush=True)
    ladder = []
    for aspect, rdx in ((1.0, 40.0), (1.5, 40.0)):
        for hw_deg, n_st in ((7.5, 24), (15.0, 12), (22.5, 8), (30.0, 6)):
            r = ellipse_case(aspect=aspect, dx=R_EQ / rdx, half_window=math.radians(hw_deg),
                             n_station=n_st)
            ladder.append({"aspect_ratio": aspect, "r_eq_over_dx": rdx,
                           "half_window_deg": hw_deg, "n_station": n_st,
                           "flux_err": r["flux_err"], "kappa_err": r["kappa_err"],
                           "pressure_err": r["pressure_err"],
                           "whitney_graustein_err_pct": r["whitney_graustein_err_pct"]})

    # controls
    print("[curv-variation] controls", flush=True)
    zero = ellipse_case(aspect=1.5, dx=R_EQ / 20.0, sigma=0.0)
    mod = laplace_isph_case(radius=R_EQ, dx=R_EQ / 20.0, sigma=SIGMA, rho0=RHO0,
                            outer=OUTER_DX / 20.0 * R_EQ,
                            poisson_weighting="force_density")
    circle20 = next(c for c in configs
                    if c["aspect_ratio"] == 1.0 and c["r_eq_over_dx"] == 20.0)
    zmax = max((abs(r["dp_flux_Pa"]) for r in zero["stations"] if r), default=0.0)

    # ---- gate verdicts ---------------------------------------------------- #
    def scoped(**kw):
        sel = [c for c in configs if all(
            (c[k] >= v[0] if isinstance(v, tuple) else c[k] == v) for k, v in kw.items())]
        return sel

    def worst_bias(sel):
        return max((abs(c["flux_err"]["mean"]) for c in sel), default=float("nan"))

    def worst_rms(sel):
        return max((c["flux_err"]["rms"] for c in sel), default=float("nan"))

    resolved = [c for c in configs if c["resolved"]]
    v = {
        "local_bias_pct_circle_R_over_dx_ge_20": (
            worst_bias(scoped(aspect_ratio=1.0, r_eq_over_dx=(20.0, 1e9))),
            GATES["local_bias_pct_circle_R_over_dx_ge_20"], "<="),
        "local_bias_pct_aspect_le_1p5_resolved": (
            worst_bias([c for c in resolved
                        if c["aspect_ratio"] <= 1.5 and c["r_eq_over_dx"] >= 20.0]),
            GATES["local_bias_pct_aspect_le_1p5_resolved"], "<="),
        "local_bias_pct_all_aspects_R_over_dx_40": (
            worst_bias([c for c in configs if c["r_eq_over_dx"] == 40.0]),
            GATES["local_bias_pct_all_aspects_R_over_dx_40"], "<="),
        "station_rms_pct_aspect_le_2_resolved": (
            worst_rms([c for c in resolved if c["aspect_ratio"] <= 2.0]),
            GATES["station_rms_pct_aspect_le_2_resolved"], "<="),
        "whitney_graustein_err_pct_resolved": (
            max(abs(c["whitney_graustein_err_pct"]) for c in resolved),
            GATES["whitney_graustein_err_pct_resolved"], "<="),
        "pressure_bias_pct_all_aspects_R_over_dx_40": (
            max(abs(c["pressure_err"]["mean"]) for c in configs
                if c["r_eq_over_dx"] == 40.0
                and math.isfinite(c["pressure_err"]["mean"])),
            GATES["pressure_bias_pct_all_aspects_R_over_dx_40"], "<="),
        "net_traction_residual_model_arm": (
            max(c["net_traction_residual"] for c in configs),
            GATES["net_traction_residual_model_arm"], "<="),
        "pressure_vs_flux_bias_diff_pct_aspect_le_1p5": (
            max(abs(c["pressure_err"]["mean"] - c["flux_err"]["mean"])
                for c in resolved
                if c["aspect_ratio"] <= 1.5 and c["r_eq_over_dx"] >= 20.0),
            GATES["pressure_vs_flux_bias_diff_pct_aspect_le_1p5"], "<="),
        "zero_sigma_abs_dp_Pa": (zmax, GATES["zero_sigma_abs_dp_Pa"], "<="),
        "circle_vs_module_dp_diff_pct": (
            abs(circle20["mean_dp_flux_Pa"] / mod["dp_measured_Pa"] - 1.0) * 100.0,
            GATES["circle_vs_module_dp_diff_pct"], "<="),
        "both_exact_bias_pct_aspect_le_2_resolved": (
            max((abs(c["ablation_both_exact"]["flux_err"]["mean"])
                 for c in resolved if c["aspect_ratio"] <= 2.0), default=float("nan")),
            GATES["both_exact_bias_pct_aspect_le_2_resolved"], "<="),
        "window_ladder_rms_pct_at_30deg_aspect_1p5": (
            next(l["flux_err"]["rms"] for l in ladder
                 if l["aspect_ratio"] == 1.5 and l["half_window_deg"] == 30.0),
            GATES["window_ladder_rms_pct_at_30deg_aspect_1p5"], "<="),
    }
    verdicts = {k: {"value": float(val), "gate": g, "mode": mode,
                    "passed": bool(math.isfinite(val) and abs(val) <= g)}
                for k, (val, g, mode) in v.items()}
    # noise halving: a longer arc must beat the kernel-scale scatter by >= 2x
    lad = sorted([l for l in ladder], key=lambda l: (l["aspect_ratio"], l["half_window_deg"]))
    ratios = {}
    for asp in (1.0, 1.5):
        sel = [l for l in lad if l["aspect_ratio"] == asp]
        r = sel[-1]["flux_err"]["rms"] / sel[0]["flux_err"]["rms"]
        ratios[f"aspect_{asp}"] = round(r, 4)
        verdicts[f"window_ladder_noise_halving_aspect_{asp}"] = {
            "value": r, "gate": GATES["window_ladder_noise_halving"], "mode": "<=",
            "passed": bool(r <= GATES["window_ladder_noise_halving"]),
        }
    verdicts["window_ladder_rms_half_window_deg"] = {
        "value": [l["half_window_deg"] for l in lad if l["aspect_ratio"] == 1.5],
        "gate": "informational", "passed": True,
    }

    result = {
        "anchor": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE",
        "round": 5,
        "case": "Young-Laplace at a non-constant curvature interface (ellipse)",
        "sigma_N_per_m": SIGMA, "rho0_kg_per_m3": RHO0, "r_eq_m": R_EQ,
        "outer_dx": OUTER_DX, "n_station": N_STATION,
        "r_curv_min_dx_precondition": R_CURV_MIN_DX,
        "gates_declared": GATES,
        "module_circle": {
            "radius_m": R_EQ, "r_over_dx": 20.0,
            "dp_measured_Pa": mod["dp_measured_Pa"],
            "dp_theory_Pa": mod["dp_theory_Pa"],
            "dp_error_pct": mod["dp_error_pct"],
            "dp_flux_error_pct": (mod["dp_flux_Pa"] / (SIGMA / R_EQ) - 1.0) * 100.0,
        },
        "configs": configs,
        "window_ladder": ladder,
        "controls": {"zero_sigma_max_abs_dp_flux_Pa": zmax,
                     "zero_sigma_net_traction_residual": zero["net_traction_residual"]},
        "gate_verdicts": verdicts,
        "all_gates_passed": all(g["passed"] for g in verdicts.values()),
    }
    (OUT / "curvature_variation.json").write_text(json.dumps(result, indent=1))
    print("[curv-variation] wrote", OUT / "curvature_variation.json")
    for k, g in verdicts.items():
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {k}: {g['value']}"
              f" (gate {g['gate']})")
    return result


if __name__ == "__main__":
    torch.set_num_threads(4)
    sweep()