"""Oscillating-droplet benchmark: the 2-D Rayleigh-Lamb frequency.

This is the *dynamical* half of gate B3.  ``benchmarks/csf_consistent`` scores
the CSF surface-tension force against three **static** analytic references
(curvature, cut-force identity, Young-Laplace pressure jump) and found the force
amplitude and direction correct to 7-11 % while the end-to-end pressure jump is
still blocked by the equation-of-state resolution floor of summation-density
WCSPH.  The frequency of a free capillary oscillation does not use the pressure
jump at all: it is set by the restoring force, i.e. by the effective surface
tension the CSF model actually applies.  So it is the independent, *dynamical*
probe of the same model.

It has been named as the open half of B3 in
``docs/benchmarks/laplace-droplet.md``, ``docs/benchmarks/surface-tension-laplace.md``,
``docs/benchmarks/surface-tension-verification.md`` and
``docs/algorithm-gap-analysis.md`` since the surface-tension model merged.

Analytic reference
------------------
For an inviscid 2-D drop of radius ``R``, density ``rho`` and surface tension
``sigma``, a normal-mode perturbation ``r(theta, t) = R (1 + eps cos(n theta))``
oscillates with (Rayleigh 1879; Lamb, *Hydrodynamics* 1932, art. 273)

    omega_n^2 = n (n^2 - 1) sigma / (rho R^3)

Derivation (linearised; interior potential flow ``phi = A r^n cos n theta``,
unsteady Bernoulli ``p = -rho dphi/dt``, Young-Laplace
``kappa = (1 + (n^2-1) eps cos n theta)/R``) -- implemented as
:func:`rayleigh_lamb_omega`, with the 3-D coefficient pinned separately in the
tests so the wrong formula cannot be substituted silently.  ``n = 1`` is a
translation with zero frequency, so the lowest genuine shape mode is ``n = 2``:

    omega_2^2 = 6 sigma / (rho R^3)

Initial condition
-----------------
A standing mode-2 wave has ``r(theta, t) = R (1 + eps cos(omega t) cos 2theta)``
and therefore **zero** radial velocity at maximum deformation, so a static
area-preserving ellipse with semi-axes ``R(1 -/+ eps)`` and zero velocity is the
pure single-mode initial condition of linear theory.  No velocity field has to
be manufactured.

Observable
----------
Second moments of the particle distribution.  For a uniform ellipse ``Sxx =
M a^2/4``, ``Syy = M b^2/4``, so

    delta(t) = (Sxx - Syy)/(Sxx + Syy) = (a^2-b^2)/(a^2+b^2) ~ 2 eps cos(omega t)

is an integral quantity: smooth, insensitive to particle count and far less
noisy than tracking the extreme particle.  The extreme-particle estimator is
kept as an independent cross-check.

Stability
---------
WCSPH with summation density has no built-in dissipation, and a free capillary
oscillation is an undamped resonator: lattice noise at the interface is fed back
through the pressure field and grows (the first run of this benchmark reached
Mach 9270 with a fixed step and ``viscosity = 0``).  The driver therefore
(a) re-applies the solver's own acoustic CFL limit every step with the current
``|v|`` (:func:`tensorsph.droplet_impact.next_dt`), (b) defaults to a small
physical viscosity whose viscous damping time is far longer than the oscillation
period, and (c) aborts and *reports* a divergence instead of fitting a blow-up.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_droplet/run.py --quick
    PYTHONPATH=src python benchmarks/oscillating_droplet/run.py \
        --dx 0.002,0.0015,0.001 --periods 3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorsph.droplet_impact import (
    Config,
    next_dt,
    rhs,
    stable_dt,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


# --------------------------------------------------------------------------- #
# analytic reference
# --------------------------------------------------------------------------- #
def rayleigh_lamb_omega(n: int, sigma: float, rho: float, radius: float) -> float:
    """Angular frequency of the ``n``-th 2-D Rayleigh-Lamb mode.

    ``omega^2 = n (n^2 - 1) sigma / (rho R^3)``.  ``n = 1`` is a translation
    (``omega = 0``) and is rejected: it carries no restoring force.
    """
    if n < 2:
        raise ValueError("mode n=1 is a translation, not a shape oscillation")
    return math.sqrt(n * (n * n - 1) * sigma / (rho * radius ** 3))


def scaled_viscous_rate(nu: float, radius: float) -> float:
    """Viscous decay rate of the mode-2 shape oscillation (1/s).

    2-D Lamb: ``gamma = 2 n (n-1) nu / R^2`` (n=2), the vorticity-free limit of
    the viscous damping of a capillary mode.  Used only to report how far the
    run is from the inviscid assumption, never to correct the measurement.
    """
    return 2.0 * 2 * (2 - 1) * nu / radius ** 2


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def elliptical_droplet(semi_major: float, semi_minor: float, dx: float,
                       rho0: float):
    """Hexagonally packed ellipse with a uniform particle mass.

    Hex packing (6 neighbours instead of 4) halves the noise of the SPH gradient
    operators; the lattice constant is chosen so ``mass * number_density =
    rho0``.  A uniform mass with an elliptically cut boundary means the mean
    density inside is ``rho0`` and the total mass is ``rho0 * pi a b``.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    mass = torch.tensor(rho0 * dx * dy, dtype=torch.float64)
    pts = []
    j = 0
    y = -semi_minor
    while y <= semi_minor:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -semi_major + offset
        while x <= semi_major:
            if (x / semi_major) ** 2 + (y / semi_minor) ** 2 <= 1.0:
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


def shape_observables(pos: torch.Tensor, mass: torch.Tensor) -> dict:
    """Dimensionless mode-2 amplitude from second moments (+ extreme cross-check)."""
    sxx = float((mass * pos[:, 0] ** 2).sum())
    syy = float((mass * pos[:, 1] ** 2).sum())
    delta = (sxx - syy) / (sxx + syy)

    R = math.sqrt(float(pos[:, 0].abs().max() * pos[:, 1].abs().max()))
    band_y = pos[:, 1].abs() <= 0.35 * R
    band_x = pos[:, 0].abs() <= 0.35 * R
    a_ext = float(pos[band_y, 0].abs().max())
    b_ext = float(pos[band_x, 1].abs().max())
    return {
        "delta": delta,
        "delta_ext": (a_ext - b_ext) / (a_ext + b_ext),
        "a_ext": a_ext,
        "b_ext": b_ext,
    }


# --------------------------------------------------------------------------- #
# frequency extraction
# --------------------------------------------------------------------------- #
def zero_crossing_frequency(t: np.ndarray, y: np.ndarray) -> dict:
    """Frequency from the mean spacing of linear-interpolated zero crossings.

    Successive crossings are half a period apart, so the period is twice their
    mean spacing.  ``period_spread_pct`` reports the width of that spacing
    distribution: a clean single mode gives a small number, a multi-mode or
    diverging signal a large one, which is what makes this a usable guard.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y = y - np.median(y)
    idx = np.nonzero(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    if idx.size < 2:
        return {"omega": float("nan"), "period": float("nan"),
                "n_crossings": int(idx.size), "period_spread_pct": float("nan")}
    tc = []
    for k in idx:
        y0, y1 = y[k], y[k + 1]
        frac = abs(y0) / (abs(y0) + abs(y1))
        tc.append(t[k] + frac * (t[k + 1] - t[k]))
    tc = np.asarray(tc)
    half = np.diff(tc)
    period = 2.0 * float(np.mean(half))
    return {
        "omega": 2.0 * math.pi / period,
        "period": period,
        "n_crossings": int(tc.size),
        "period_spread_pct": float(np.std(half) / np.mean(half) * 100.0),
    }


def fft_guess_omega(t: np.ndarray, y: np.ndarray) -> float:
    """Coarse frequency from the peak of a Hann-windowed, zero-padded FFT."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y = (y - y.mean()) * np.hanning(y.size)
    dt = float(np.mean(np.diff(t)))
    n = 1 << int(math.ceil(math.log2(max(y.size, 2) * 32)))
    spec = np.abs(np.fft.rfft(y, n=n))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0
    return 2.0 * math.pi * float(freqs[int(np.argmax(spec))])


def damped_cosine_fit(t: np.ndarray, y: np.ndarray, omega_lo: float,
                      omega_hi: float, n_omega: int = 200,
                      n_lambda: int = 40) -> dict:
    """Least-squares fit of ``y = C + exp(-lam t)(p cos wt + q sin wt)``.

    For a fixed ``(omega, lam)`` the model is *linear* in ``(p, q, C)``, so the
    fit is a two-dimensional grid search with an exact linear solve at every
    node - no optimiser and no sensitivity to a starting guess.  Two passes: a
    coarse grid over the caller's band, then a refinement around the best node.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    T = t[-1] - t[0]

    def sweep(w_lo, w_hi, n_w, l_lo, l_hi, n_l):
        best = None
        for w in np.linspace(w_lo, w_hi, n_w):
            cw, sw = np.cos(w * t), np.sin(w * t)
            for lam in np.linspace(l_lo, l_hi, n_l):
                e = np.exp(-lam * t)
                A = np.stack([e * cw, e * sw, np.ones_like(t)], axis=1)
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                res = float(((A @ coef - y) ** 2).sum())
                if best is None or res < best[0]:
                    best = (res, float(w), float(lam), coef)
        return best

    coarse = sweep(omega_lo, omega_hi, n_omega, 0.0, 12.0 / T, n_lambda)
    _r, w0, lam0, _c = coarse
    dw = (omega_hi - omega_lo) / max(n_omega - 1, 1)
    dl = (12.0 / T) / max(n_lambda - 1, 1)
    res, w, lam, coef = sweep(w0 - dw, w0 + dw, 60,
                              max(lam0 - dl, 0.0), lam0 + dl, 20)
    return {
        "omega": float(w),
        "period": 2.0 * math.pi / float(w),
        "decay_rate_per_s": float(lam),
        "amplitude": float(math.hypot(coef[0], coef[1])),
        "rms_residual": float(math.sqrt(res / max(t.size, 1))),
    }


# --------------------------------------------------------------------------- #
# the simulation
# --------------------------------------------------------------------------- #
def _analyse(hist: dict, setup: dict, reference: dict, scheme: str,
             samples_per_period: int) -> dict:
    """Turn a recorded history into a frequency measurement (pure post-processing)."""
    omega_theory = reference["omega_rad_per_s"]
    ti = np.asarray(hist["t"])
    di = np.asarray(hist["delta"])
    dei = np.asarray(hist["delta_ext"])
    start = int(0.5 * samples_per_period)          # drop the startup transient
    if ti.size - start < 32:
        return {"status": "TOO_SHORT", "note": "not enough samples for a fit"}
    # Adaptive dt makes the record non-uniform; resample onto a uniform grid so
    # the FFT and the least-squares fit see an evenly spaced signal.
    tu = np.linspace(ti[start], ti[-1], ti.size - start)
    y_fit = np.interp(tu, ti, di)
    y_ext = np.interp(tu, ti, dei)

    w0 = fft_guess_omega(tu, y_fit)
    fit = damped_cosine_fit(tu, y_fit, 0.75 * w0, 1.25 * w0)
    zc = zero_crossing_frequency(tu, y_fit)
    zc_ext = zero_crossing_frequency(tu, y_ext)

    def score(omega_meas):
        if not math.isfinite(omega_meas):
            return {"omega_meas": float("nan"), "period_meas_s": float("nan"),
                    "rel_error_pct": float("nan"),
                    "within_3pct": False, "within_15pct": False}
        err = (omega_meas - omega_theory) / omega_theory * 100.0
        return {"omega_meas": float(omega_meas),
                "period_meas_s": float(2.0 * math.pi / omega_meas),
                "rel_error_pct": float(err),
                "within_3pct": bool(abs(err) < 3.0),
                "within_15pct": bool(abs(err) < 15.0)}

    ke = np.asarray(hist["ke"])
    amp_first = float(np.abs(di[:max(len(di) // 2, 1)]).max())
    amp_last = float(np.abs(di[-max(len(di) // 4, 1):]).max())
    return {
        "status": "OK",
        "measured": {
            "primary_damped_cosine": score(fit["omega"]),
            "cross_check_zero_crossing": score(zc["omega"]),
            "cross_check_extreme_particle": score(zc_ext["omega"]),
            "fit": fit,
            "zero_crossing": zc,
            "fft_initial_guess_omega": w0,
            "estimator_agreement_pct": (abs(fit["omega"] - zc["omega"]) * 100.0
                                        / omega_theory
                                        if math.isfinite(zc["omega"]) else float("nan")),
        },
        "dissipation": {
            "decay_rate_fit_per_s": fit["decay_rate_per_s"],
            "viscous_decay_rate_theory_per_s": setup["viscous_decay_rate_per_s"],
            "viscous_Q_estimate": setup["viscous_Q_estimate"],
            "amplitude_first_half": amp_first,
            "amplitude_last_quarter": amp_last,
            "amplitude_ratio": amp_last / amp_first if amp_first else float("nan"),
            "ke_peak_J": float(np.abs(ke).max()),
            "max_speed_m_per_s": float(np.abs(np.asarray(hist["vmax"])).max()),
            "mach_max": float(np.abs(np.asarray(hist["vmax"])).max()
                              / setup["sound_speed"]),
        },
    }


def run_oscillation(radius: float = 0.02, dx: float = 0.002, sigma: float = 7.2,
                    rho0: float = 1000.0, sound_speed: float = 20.0,
                    viscosity: float = 3.0e-3, eps: float = 0.08,
                    periods: float = 2.5, scheme: str = "consistent",
                    density_diffusion: float = 0.0,
                    samples_per_period: int = 120, n: int = 2,
                    blowup_mach: float = 0.5,
                    max_steps: int = 400_000,
                    state_path: Path | None = None,
                    max_wall: float | None = None) -> dict:
    """Launch a mode-``n`` standing wave on a 2-D drop and measure its period.

    ``state_path`` + ``max_wall`` make the run resumable: the integrator state and
    the accumulated history are written every ``max_wall`` seconds and reloaded on
    the next call, so a simulation longer than one shell time limit can still be
    completed without shrinking the problem.  ``status`` is then ``"PAUSED"`` and
    the caller simply invokes it again.
    """
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 density_diffusion=density_diffusion,
                 surface_tension_coefficient=sigma, csf_scheme=scheme)
    h = cfg.h_factor * cfg.dx
    dt_cap = float(stable_dt(cfg))

    omega_theory = rayleigh_lamb_omega(n, sigma, rho0, radius)
    T_theory = 2.0 * math.pi / omega_theory
    tau_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    sample_dt = T_theory / samples_per_period
    t_target = periods * T_theory

    a = radius * (1.0 + eps)
    b = radius * (1.0 - eps)

    resumed = False
    if state_path is not None and Path(state_path).exists():
        st = torch.load(Path(state_path), weights_only=False)
        pos, vel, mass = st["pos"], st["vel"], st["mass"]
        t, step, dt = st["t"], st["step"], st["dt"]
        hist = st["hist"]
        next_sample = st["next_sample"]
        resumed = True
    else:
        pos, mass = elliptical_droplet(a, b, dx, rho0)
        vel = torch.zeros_like(pos)
        t, step, dt, next_sample = 0.0, 0, dt_cap, 0.0
        hist = {"t": [], "delta": [], "delta_ext": [], "ke": [], "vmax": [],
                "dt": []}

    setup = {
        "radius_m": radius, "dx_m": dx, "h_m": h, "R_over_h": radius / h,
        "n_particles": int(pos.shape[0]),
        "sigma_N_per_m": sigma, "rho0": rho0, "sound_speed": sound_speed,
        "viscosity_m2_per_s": viscosity, "mode_n": n,
        "density_diffusion": density_diffusion,
        "initial_amplitude_eps": eps,
        "semi_axes_m": [a, b],
        "area_deficit_pct": eps ** 2 * 100.0,
        "dt_initial_s": dt_cap,
        "dt_limiter": "acoustic CFL, re-applied each step with current |v|",
        "periods_target": periods,
        "tau_sigma_s": tau_sigma,
        "viscous_decay_rate_per_s": scaled_viscous_rate(viscosity, radius),
        "viscous_Q_estimate": (omega_theory / scaled_viscous_rate(viscosity, radius)
                               if viscosity > 0 else float("inf")),
    }
    reference = {
        "omega_rad_per_s": omega_theory,
        "period_s": T_theory,
        "formula": "omega^2 = n(n^2-1) sigma/(rho R^3), n=2",
    }

    wall_start = time.time()
    diverged = False
    while t < t_target and step < max_steps:
        if max_wall is not None and time.time() - wall_start > max_wall:
            break
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        t += dt
        step += 1

        vmax = float(torch.linalg.vector_norm(vel, dim=1).max())
        if not math.isfinite(vmax) or vmax > blowup_mach * sound_speed:
            diverged = True
            break
        dt = next_dt(cfg, dt, vmax)      # solver's own CFL limit, re-applied

        if t >= next_sample:
            obs = shape_observables(pos, mass)
            hist["t"].append(t)
            hist["delta"].append(obs["delta"])
            hist["delta_ext"].append(obs["delta_ext"])
            hist["ke"].append(0.5 * float((mass * (vel ** 2).sum(dim=1)).sum()))
            hist["vmax"].append(vmax)
            hist["dt"].append(dt)
            next_sample = t + sample_dt

    setup.update({"dt_final_s": float(dt), "steps": step, "t_end_s": t,
                  "periods_simulated": t / T_theory, "resumed": resumed})

    base = {"scheme": scheme, "setup": setup, "reference": reference,
            "history": hist}

    if diverged:
        return base | {"status": "DIVERGED", "measured": None, "dissipation": None,
                       "note": (f"aborted: max|v| exceeded {blowup_mach:g} c0 "
                                f"at t={t:.5f} s")}

    if t < t_target:
        if state_path is not None:
            torch.save({"pos": pos, "vel": vel, "mass": mass, "t": t,
                        "step": step, "dt": dt, "hist": hist,
                        "next_sample": next_sample}, Path(state_path))
        return base | {"status": "PAUSED", "measured": None, "dissipation": None,
                       "note": f"wall-clock budget reached at t={t:.5f} s "
                               f"({t / T_theory:.2f} periods); state saved"}

    if state_path is not None and Path(state_path).exists():
        Path(state_path).unlink()       # finished: do not resume into a stale run
    return base | _analyse(hist, setup, reference, scheme, samples_per_period)


def make_figures(runs: list[dict], out_dir: Path = HERE) -> Path:
    """Two panels: the oscillation traces, and the frequency error ladder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in runs if r.get("status") == "OK"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    ax = axes[0]
    for run in ok:
        t = np.asarray(run["history"]["t"])
        d = np.asarray(run["history"]["delta"])
        T = run["reference"]["period_s"]
        ax.plot(t / T, d, lw=1.2,
                label=f"SPH dx={run['setup']['dx_m']*1e3:.1f} mm "
                      f"(R/h={run['setup']['R_over_h']:.1f})")
    if ok:
        r0 = ok[0]
        T = r0["reference"]["period_s"]
        eps = r0["setup"]["initial_amplitude_eps"]
        tmax = max(np.asarray(r["history"]["t"]).max() / T for r in ok)
        tt = np.linspace(0.0, tmax, 600)
        ax.plot(tt, 2 * eps * np.cos(2 * math.pi * tt), "k--", lw=1.1,
                label=r"linear theory $2\varepsilon\cos(\omega t)$")
    ax.set_xlabel("t / T$_{theory}$")
    ax.set_ylabel(r"$\delta=(S_{xx}-S_{yy})/(S_{xx}+S_{yy})$")
    ax.set_title("(a) mode-2 oscillation of the drop shape")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if ok:
        dxs = [r["setup"]["dx_m"] * 1e3 for r in ok]
        errs = [r["measured"]["primary_damped_cosine"]["rel_error_pct"] for r in ok]
        errs_x = [r["measured"]["cross_check_zero_crossing"]["rel_error_pct"] for r in ok]
        ax.axhspan(-15, 15, color="tab:green", alpha=0.10, label="platform limit 15 %")
        ax.axhline(0, color="k", lw=0.8)
        ax.plot(dxs, errs, "o-", label="second-moment damped-cosine fit")
        ax.plot(dxs, errs_x, "s--", label="zero-crossing cross-check")
        for x, e in zip(dxs, errs):
            ax.annotate(f"{e:+.1f}%", (x, e), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, ha="center")
        ax.set_xlabel("dx [mm]")
        ax.set_ylabel(r"$\omega$ error vs $\sqrt{6\sigma/\rho R^3}$ [%]")
        ax.set_title("(b) frequency error vs resolution")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Oscillating droplet (2-D Rayleigh-Lamb mode n=2)")
    fig.tight_layout()
    out = out_dir / "oscillating_droplet.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=str, default="0.002",
                    help="comma separated resolution list [m]")
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--periods", type=float, default=2.5)
    ap.add_argument("--max-wall", type=float, default=None,
                    help="seconds of integration per invocation before saving "
                         "state and returning status=PAUSED (resumable runs)")
    ap.add_argument("--state-dir", type=str, default="/tmp/osc_state")
    ap.add_argument("--nu-list", type=str, default=None,
                    help="comma separated kinematic viscosity list [m^2/s]; "
                         "a sweep over (dx x nu) is run when given")
    ap.add_argument("--viscosity", type=float, default=None,
                    help="single viscosity (overrides --nu-list)")
    ap.add_argument("--density-diffusion", type=float, default=0.0)
    ap.add_argument("--scheme", type=str, default="consistent",
                    choices=["consistent", "legacy"])
    ap.add_argument("--shard", type=str, default=None,
                    help="i/n: run only the i-th of n slices of the (dx x nu) "
                         "grid, so the cluster can spread the sweep")
    ap.add_argument("--quick", action="store_true",
                    help="one coarse resolution, 1.5 periods")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    dxs = [float(x) for x in a.dx.split(",")]
    nu_s = [a.viscosity] if a.viscosity is not None else (
        [float(x) for x in a.nu_list.split(",")] if a.nu_list else [1.0e-4])
    periods = a.periods
    if a.quick:
        dxs, periods = [0.002], 1.5

    grid = [(dx, nu) for dx in dxs for nu in nu_s]
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        grid = [cfg for k, cfg in enumerate(grid) if k % n == i % n]
        print(f"shard {i}/{n}: {len(grid)} of the grid", flush=True)

    runs = []
    for dx, nu in grid:
        run = run_oscillation(radius=a.radius, dx=dx, sigma=a.sigma,
                              eps=a.eps, periods=periods,
                              viscosity=nu,
                              density_diffusion=a.density_diffusion,
                              scheme=a.scheme,
                              max_wall=a.max_wall,
                              state_path=(Path(a.state_dir)
                                          / f"osc_dx{dx*1e3:g}mm_nu{nu:g}.pt"))
        tag = f"{run['scheme']}_dx{dx*1e3:g}mm_nu{nu:g}"
        if run.get("status") == "OK":
            m = run["measured"]
            print(f"dx={dx*1e3:.2f} mm nu={nu:g} N={run['setup']['n_particles']:5d} "
                  f"R/h={run['setup']['R_over_h']:4.1f} "
                  f"T={m['primary_damped_cosine']['period_meas_s']:.5f} s "
                  f"(theory {run['reference']['period_s']:.5f}) "
                  f"err={m['primary_damped_cosine']['rel_error_pct']:+.2f}% "
                  f"zc={m['cross_check_zero_crossing']['rel_error_pct']:+.2f}% "
                  f"mach={run['dissipation']['mach_max']:.3f} "
                  f"decay={run['dissipation']['decay_rate_fit_per_s']:.1f}/s",
                  flush=True)
        else:
            print(f"dx={dx*1e3:.2f} mm nu={nu:g} status={run.get('status')} "
                  f"note={run.get('note')}", flush=True)
            tag += "_" + str(run.get("status", "FAIL")).lower()
        runs.append(run)
        (HERE / f"osc_{tag}.json").write_text(json.dumps(run, indent=2))
        summary = {
            "reference": {"formula": "omega^2 = n(n^2-1) sigma/(rho R^3), n=2",
                          "value": "omega_2 = sqrt(6 sigma/(rho R^3))"},
            "scheme": a.scheme,
            "amplitude_eps": a.eps,
            "density_diffusion": a.density_diffusion,
            "runs": [{k: r[k] for k in ("setup", "reference", "status")}
                     | ({"measured": r["measured"], "dissipation": r["dissipation"]}
                        if r.get("status") == "OK" else {"note": r.get("note")})
                     for r in runs],
        }
        name = ("oscillation_summary.json" if not a.shard
                else f"oscillation_summary_shard{a.shard.replace('/', '_')}.json")
        (HERE / name).write_text(json.dumps(summary, indent=2))
    if not a.no_plot:
        try:
            print("figure ->", make_figures(runs))
        except Exception as exc:            # a partial sweep may have nothing to plot
            print("figure skipped:", exc)


if __name__ == "__main__":
    main()
