"""Capillary-gravity wave benchmark: dynamic verification of the CSF surface tension.

A standing wave on the free surface of an intermediate-depth pool oscillates at

    omega^2 = (g k + sigma k^3 / rho) tanh(k H)        (Lamb 1932, Art. 267)

with ``k = 2 pi / lambda`` and ``H`` the fluid depth.  The wavelength is placed
at the capillary length ``2 pi sqrt(sigma/(rho g))`` (~17 mm for water) so that
gravity and surface tension each supply about half of ``omega^2``: a model that
gets the capillary term wrong by X % moves the measured period by about X/2 %,
which is what makes the period a sharp test of the surface-tension operator.
The static counterpart, the Young-Laplace pressure jump, is verified in
``benchmarks/csf_consistent`` and ``benchmarks/laplace_law``; a model can pass
those and still miss the dispersion relation, because the curvature of a moving
interface, the momentum coupling and the free-surface treatment are different
operators.

Why not pure capillary waves (g = 0)
------------------------------------
Without gravity a finite body of fluid has no flat equilibrium: surface tension
rounds the pool up.  Measured on this platform: with g = 0, sigma = 0.072 N/m
and a 190 mm pool the mean interface level drifted 45 mm (6 amplitudes) and the
wave mode was destroyed.  Gravity is what makes "flat interface + sinusoidal
perturbation" an equilibrium, so the run keeps real water surface tension and
real gravity at the capillary length scale, where the two restoring forces are
comparable.

Setup
-----
* Rigid, free-slip boundaries (``boundary="clamp"``); the mode
  ``eta = y0 + A cos(2 pi x / lambda)`` has antinodes at both side walls, so the
  zero-normal-velocity wall condition is satisfied by the mode itself
  (``pool_width = lambda/2``).
* Fluid depth ``H = 0.16 lambda`` (``k H = 1``): intermediate depth.  The
  ``tanh(kH) = 0.762`` factor is kept in the analytical reference.  A shallower
  layer keeps the particle count affordable at ``dx = A/3``; a deep-water
  (``kH = 4``) pool of the same ``A/dx`` costs eight times as much.
* No viscosity, no density diffusion: the only dissipation is numerical.
* Initial condition: the hydrostatic state of the displaced interface,
  ``p = rho0 g (eta(x) - y)`` (``hydrostatic_density`` with the local surface
  height).  With a flat-surface hydrostatic density the run starts with a
  pressure step ``rho0 g A`` across the interface and radiates an acoustic
  transient.

Observable
----------
The interface height is a poor observable at ``A ~ 3 dx``: the topmost particle
in a column quantises the surface to ``dx`` and the residual is particle noise.
The benchmark therefore measures the **mass-weighted modal velocity** of the
vertical motion,

    a_v(t) = sum_i m_i v_y,i cos(k x_i) / sum_i m_i cos^2(k x_i)

which averages over every particle in the pool (the linear standing wave has
``u_y ~ cos(k x)`` at every depth) and is far cleaner than the interface height.
For ``eta = A cos(kx) cos(omega t)`` this is ``a_v = -A omega sin(omega t)``,
so its zero crossings are spaced ``T/2`` apart and the period is a
least-squares fit of ``t_n = t_0 + n T/2`` (robust for a damped signal, unlike
differencing neighbouring crossings).  The interface mode is still recorded and
plotted for the physics picture, and its period is reported as a cross-check.

Usage
-----
    python benchmarks/capillary_wave/run.py --quick
    python benchmarks/capillary_wave/run.py --dx-list 0.000135,0.00009 \
        --json-out benchmarks/capillary_wave/validation_report.json \
        --png-out benchmarks/capillary_wave/capillary_wave.png

Exit status is non-zero when the measured period is off by more than
``--max-freq-error`` (default 15 %).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    hydrostatic_density,
    next_dt,
    resolve_device,
    rhs,
    stable_dt,
)
from tensorsph.neighbor_search import PairListCache

# Free board above the mean interface as a fraction of the fluid depth: room
# for the wave crest and for a few ballistic surface particles.
POOL_HEADROOM = 1.6


# --------------------------------------------------------------------------- #
# Analytical reference
# --------------------------------------------------------------------------- #

def analytical_frequency(sigma: float, k: float, rho: float,
                         gravity: float = 0.0, depth: float | None = None) -> float:
    """``omega^2 = (g k + sigma k^3/rho) tanh(k H)``.

    ``gravity = 0`` and ``depth = None`` reduce to the pure capillary branch
    ``omega = sqrt(sigma k^3 / rho)`` (Lamb 1932, Art. 267).
    """
    omega_sq = sigma * k**3 / rho
    if gravity:
        omega_sq += gravity * k
    if depth is not None:
        omega_sq *= math.tanh(k * depth)
    return math.sqrt(omega_sq)


def analytical_period(sigma: float, k: float, rho: float,
                      gravity: float = 0.0, depth: float | None = None) -> float:
    return 2.0 * math.pi / analytical_frequency(sigma, k, rho, gravity, depth)


def gravity_share(sigma: float, k: float, rho: float, gravity: float) -> float:
    """Fraction of the inviscid ``omega^2`` supplied by gravity."""
    cap = sigma * k**3 / rho
    grav = gravity * k
    return grav / (grav + cap) if (grav + cap) > 0 else 0.0


def capillary_length(sigma: float, rho: float, gravity: float) -> float:
    """``sqrt(sigma / (rho g))``, ~2.7 mm for water/air."""
    return math.sqrt(sigma / (rho * gravity))


# --------------------------------------------------------------------------- #
# Case generation
# --------------------------------------------------------------------------- #

def interface_level(cfg: Config) -> float:
    """Mean interface height: the pool is ``pool_depth / POOL_HEADROOM`` deep."""
    return cfg.pool_depth / POOL_HEADROOM


def make_capillary_wave(cfg: Config, wavelength: float, amplitude: float,
                        device: torch.device):
    """Standing mode ``eta(x) = y0 + A cos(2 pi x / wavelength)``.

    Domain ``[0, pool_width] x [0, pool_depth]``; particles below the interface
    are fluid.  Masses come from the hydrostatic density of the local surface
    height, so the state carries ``p = rho0 g (eta(x) - y)``.
    """
    dx = cfg.dx
    y0 = interface_level(cfg)
    xs = torch.arange(dx * 0.5, cfg.pool_width, dx, device=device)
    ys = torch.arange(dx * 0.5, cfg.pool_depth, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)

    k = 2.0 * math.pi / wavelength
    surface = y0 + amplitude * torch.cos(k * positions[:, 0])
    fluid = positions[:, 1] < surface
    positions = positions[fluid]
    surface = surface[fluid]

    rho = hydrostatic_density(positions[:, 1], cfg, surface=surface)
    mass = rho * dx * dx
    mass *= lattice_mass_correction(positions, mass, cfg)
    return positions, mass


def lattice_mass_correction(positions: torch.Tensor, mass: torch.Tensor,
                            cfg: Config) -> float:
    """Factor that makes the kernel-sum density of the lattice match the target.

    The discrete kernel sum over a square lattice is not exactly one (0.992 for
    ``h = 1.3 dx`` on this kernel), so masses taken from the analytical
    hydrostatic density leave the SPH density ~0.8 % low.  That is small in
    absolute terms but it is *large* compared with the hydrostatic compression
    ``p_bottom / B`` (2 % here), so the initial pressure collapses to the
    ``p = max(p, 0)`` clamp and the pool starts in free fall (measured:
    -33 m/s^2 at the bottom of the pool instead of ~0).  A single global factor,
    measured in the interior band 0.3-0.7 H, restores the balance without
    touching the physical shape of the density profile.
    """
    from tensorsph.droplet_impact import density_from_neighbor_pairs

    h = cfg.h_factor * cfg.dx
    pairs, _ = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                             fluid_count=positions.shape[0]).get(positions)
    rho_est = density_from_neighbor_pairs(positions, mass, h, pairs=pairs)
    target = mass / (cfg.dx * cfg.dx)
    y = positions[:, 1]
    H = interface_level(cfg)
    interior = (y > 0.3 * H) & (y < 0.7 * H)
    if not bool(interior.any()):
        return 1.0
    ratio = float((rho_est[interior] / target[interior]).mean())
    return 1.0 / ratio if ratio > 0.0 else 1.0


# --------------------------------------------------------------------------- #
# Interface measurement (physics picture)
# --------------------------------------------------------------------------- #

def interface_profile(positions: torch.Tensor, cfg: Config,
                      n_columns: int | None = None):
    """Interface height per x column, ``(x_centre, height)``.

    The topmost particle in a column sits half a spacing below the surface, so
    ``max(y) + dx/2`` is used.  Columns without particles are dropped.
    """
    dx = cfg.dx
    n_columns = n_columns or max(1, int(round(cfg.pool_width / dx)))
    x = positions[:, 0]
    y = positions[:, 1]
    idx = torch.clamp((x / dx).long(), 0, n_columns - 1)

    height = torch.full((n_columns,), float("-inf"), dtype=y.dtype)
    height.scatter_reduce_(0, idx, y, reduce="amax", include_self=True)
    valid = torch.isfinite(height)
    centres = (torch.arange(n_columns, dtype=y.dtype, device=y.device) + 0.5) * dx
    return centres[valid], height[valid] + 0.5 * dx


def modal_amplitude(positions: torch.Tensor, cfg: Config, wavelength: float,
                    n_columns: int | None = None) -> tuple[float, float]:
    """Signed ``cos(kx)`` amplitude of the interface height and the mean level."""
    centres, heights = interface_profile(positions, cfg, n_columns)
    if centres.numel() < 4:
        return 0.0, 0.0
    k = 2.0 * math.pi / wavelength
    mode = torch.cos(k * centres)
    deviation = heights - heights.mean()
    denom = (mode * mode).sum()
    if float(denom) <= 0.0:
        return 0.0, float(heights.mean())
    return float((deviation * mode).sum() / denom), float(heights.mean())


def velocity_mode(velocity: torch.Tensor, mass: torch.Tensor,
                  positions: torch.Tensor, wavelength: float) -> float:
    """Mass-weighted amplitude of the ``cos(kx)`` mode of the vertical velocity.

    ``a_v = <m v_y cos(kx)> / <m cos^2(kx)>``.  For the standing wave
    ``eta = A cos(kx) cos(omega t)`` this is ``-A omega sin(omega t)``: a smooth
    signal whose zero crossings are exactly ``T/2`` apart.
    """
    k = 2.0 * math.pi / wavelength
    mode = torch.cos(k * positions[:, 0])
    denom = (mass * mode * mode).sum()
    if float(denom) <= 0.0:
        return 0.0
    return float((mass * velocity[:, 1] * mode).sum() / denom)


# --------------------------------------------------------------------------- #
# Analysis of the time series
# --------------------------------------------------------------------------- #

def zero_crossings(times: list[float], values: list[float]) -> list[float]:
    """Times at which ``values`` changes sign (linear interpolation)."""
    crossings: list[float] = []
    for i in range(1, len(values)):
        a0, a1 = values[i - 1], values[i]
        if a0 * a1 < 0.0:
            t0, t1 = times[i - 1], times[i]
            crossings.append(t0 - a0 * (t1 - t0) / (a1 - a0))
    return crossings


def fit_period(times: list[float], values: list[float]) -> dict | None:
    """Period from a least-squares fit of the zero-crossing times.

    Consecutive crossings are half a period apart, so ``t_n = t_0 + n T/2``.
    Fitting all crossings instead of differencing neighbours averages out the
    interpolation error and is unbiased when the amplitude decays (a decaying
    signal crowds its crossings towards the turning points).
    """
    crossings = zero_crossings(times, values)
    if len(crossings) < 3:
        return None
    n = torch.arange(len(crossings), dtype=torch.float64)
    t = torch.tensor(crossings, dtype=torch.float64)
    slope = ((n - n.mean()) * (t - t.mean())).sum() / ((n - n.mean()) ** 2).sum()
    intercept = float(t.mean() - slope * n.mean())
    residuals = t - (intercept + slope * n)
    return {
        "period_s": float(2.0 * slope),
        "n_crossings": len(crossings),
        "fit_residual_rms_s": float(torch.sqrt((residuals ** 2).mean())),
        "first_crossing_s": float(intercept),
    }


def envelope_decay(times: list[float], values: list[float]) -> dict:
    """Amplitude of the first and last extremum of the signal."""
    extrema = []
    for i in range(1, len(values) - 1):
        if (values[i] - values[i - 1]) * (values[i + 1] - values[i]) < 0.0:
            extrema.append((times[i], abs(values[i])))
    if not extrema:
        return {"n_extrema": 0, "decay_ratio": None}
    a0, a1 = extrema[0][1], extrema[-1][1]
    return {
        "n_extrema": len(extrema),
        "first_extremum": {"t": extrema[0][0], "a": a0},
        "last_extremum": {"t": extrema[-1][0], "a": a1},
        "decay_ratio": (a1 / a0) if a0 > 0.0 else None,
    }


# --------------------------------------------------------------------------- #
# Simulation driver
# --------------------------------------------------------------------------- #

def build_config(dx: float, wavelength: float, sigma: float, *,
                 amplitude: float, sound_speed: float = 3.0,
                 gravity: float = 9.81,
                 depth_per_wavelength: float = 0.16,
                 csf_scheme: str = "consistent",
                 csf_curvature: str = "mls",
                 device: str = "cpu") -> Config:
    """Intermediate-depth pool (``k H = 1``) under gravity, wave at the surface."""
    H = depth_per_wavelength * wavelength
    cfg = Config(
        dx=dx,
        pool_width=0.5 * wavelength,   # half a wavelength between free-slip walls
        pool_depth=round(POOL_HEADROOM * H, 9),
        sound_speed=sound_speed,
        gamma=7.0,
        gravity=gravity,
        viscosity=0.0,
        density_diffusion=0.0,
        hydrostatic_init=True,
        surface_tension_coefficient=sigma,
        device=device,
        boundary="clamp",
        adaptive_dt=False,
        max_courant=0.25,
        csf_scheme=csf_scheme,
        csf_curvature=csf_curvature,
    )
    cfg.dt = stable_dt(cfg, max_speed=0.0)
    return cfg


def run_case(cfg: Config, wavelength: float, amplitude: float,
             seconds: float, output_every: int | None = None) -> dict:
    """Advance the standing wave; return both modal time series and the period."""
    device = resolve_device(cfg.device)
    positions, mass = make_capillary_wave(cfg, wavelength, amplitude, device)
    velocity = torch.zeros_like(positions)

    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=positions.shape[0])

    steps = int(round(seconds / cfg.dt))
    output_every = output_every or max(1, steps // 400)
    n_columns = max(1, int(round(cfg.pool_width / cfg.dx)))

    times: list[float] = []
    amps: list[float] = []
    vmodes: list[float] = []
    levels: list[float] = []
    dt = cfg.dt
    elapsed = 0.0
    max_dev = 0.0
    min_dev = 0.0

    for step in range(steps + 1):
        pairs, _ = cache.get(positions)
        density, _, acc = rhs(positions, velocity, mass, cfg, pairs=pairs,
                              wall=None, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        rel = (density - cfg.rho0) / cfg.rho0
        max_dev = max(max_dev, float(rel.max()))
        min_dev = min(min_dev, float(rel.min()))

        if step % output_every == 0 or step == steps:
            a, level = modal_amplitude(positions, cfg, wavelength, n_columns)
            times.append(elapsed)
            amps.append(a)
            vmodes.append(velocity_mode(velocity, mass, positions, wavelength))
            levels.append(level)

        velocity = velocity + dt * acc
        positions = positions + dt * velocity
        apply_boundary(positions, velocity, cfg, top=cfg.pool_depth)
        elapsed += dt
        if cfg.adaptive_dt:
            max_speed = float(torch.linalg.vector_norm(velocity, dim=-1).max())
            dt = next_dt(cfg, dt, max_speed)

    H = interface_level(cfg)
    k = 2.0 * math.pi / wavelength
    period_analytical = analytical_period(cfg.surface_tension_coefficient, k,
                                          cfg.rho0, gravity=cfg.gravity, depth=H)
    fit_v = fit_period(times, vmodes)
    fit_eta = fit_period(times, amps)
    period = None if fit_v is None else fit_v["period_s"]
    error = None if period is None else abs(period - period_analytical) / period_analytical
    return {
        "dx": cfg.dx,
        "particles": int(positions.shape[0]),
        "steps": steps,
        "dt": dt if cfg.adaptive_dt else cfg.dt,
        "seconds_simulated": seconds,
        "interface_level_initial_m": H,
        "time_series": {"t": times, "a_eta": amps, "a_v": vmodes,
                        "mean_level": levels},
        "period_measured_s": period,
        "period_analytical_s": period_analytical,
        "frequency_error": error,
        "period_fit_velocity": fit_v,
        "period_fit_interface": fit_eta,
        "envelope_velocity": envelope_decay(times, vmodes),
        "max_density_deviation": max_dev,
        "min_density_deviation": min_dev,
        "mean_level_drift_m": (levels[-1] - levels[0]) if levels else 0.0,
        "n_columns": n_columns,
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def make_figure(report: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = report["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for case in cases:
        ts = case["time_series"]
        ax.plot(ts["t"], torch.tensor(ts["a_v"]) * 1e3, lw=1.2,
                label=f"dx = {case['dx']*1e6:.1f} um")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"modal velocity $a_v(t)$ (mm/s)")
    ax.set_title("Standing capillary-gravity wave: vertical-velocity mode")
    ax.legend(fontsize=8)

    ax = axes[1]
    measured = [c for c in cases if c["frequency_error"] is not None]
    if measured:
        labels = [f"{c['dx']*1e6:.0f}" for c in measured]
        ax.bar(labels, [100 * c["frequency_error"] for c in measured],
               color="#3b6ea5")
        ax.axhline(15.0, color="r", ls="--", lw=1, label="platform bar 15 %")
        ax.axhline(3.0, color="orange", ls=":", lw=1, label="3 %")
        ax.set_ylabel(r"period error vs $\omega^2=(gk+\sigma k^3/\rho)\tanh kH$ (%)")
        ax.set_xlabel("dx (um)")
        ax.set_title("Dispersion error")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no period measured", ha="center", va="center")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=0.072,
                        help="surface tension coefficient (N/m), water/air")
    parser.add_argument("--wavelength", type=float, default=0.017,
                        help="physical wavelength (m); default = 2 pi capillary "
                             "length of water, where gravity and capillary terms "
                             "are equal")
    parser.add_argument("--amplitude", type=float, default=0.0004,
                        help="initial interface amplitude (m); a*k ~ 0.15")
    parser.add_argument("--dx-list", default="0.000135",
                        help="comma separated particle spacings for the ladder")
    parser.add_argument("--seconds", type=float, default=None,
                        help="simulated seconds per case (default 2.4 periods)")
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--depth-per-wavelength", type=float, default=0.16,
                        help="fluid depth / wavelength (default is k H = 1)")
    parser.add_argument("--sound-speed", type=float, default=3.0,
                        help="artificial speed of sound (m/s); needs B >> rho g H")
    parser.add_argument("--csf-scheme", default="consistent",
                        choices=("consistent", "legacy"))
    parser.add_argument("--csf-curvature", default="mls",
                        choices=("mls", "fit_quadratic", "fit_circle"))
    parser.add_argument("--max-freq-error", type=float, default=0.15,
                        help="fail when |dT|/T exceeds this fraction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true",
                        help="one coarse case, short run (CI smoke test)")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--png-out", default=None)
    args = parser.parse_args()

    dx_list = sorted([float(x) for x in args.dx_list.split(",")], reverse=True)
    k = 2.0 * math.pi / args.wavelength
    H = args.depth_per_wavelength * args.wavelength
    period_analytic = analytical_period(args.sigma, k, 1000.0,
                                       gravity=args.gravity, depth=H)
    if args.quick:
        dx_list = [max(dx_list)]
        args.seconds = 0.5 * period_analytic
    elif args.seconds is None:
        args.seconds = 2.4 * period_analytic

    g_share = gravity_share(args.sigma, k, 1000.0, args.gravity)
    print(f"Analytical: k = {k:.3f} 1/m (lambda = {args.wavelength*1e3:.2f} mm), "
          f"kH = {k*H:.3f}, omega = {2*math.pi/period_analytic:.3f} rad/s, "
          f"T = {period_analytic:.5f} s, gravity share of omega^2 = "
          f"{100*g_share:.1f} %")

    configs = [
        build_config(dx, args.wavelength, args.sigma,
                     amplitude=args.amplitude,
                     sound_speed=args.sound_speed,
                     gravity=args.gravity,
                     depth_per_wavelength=args.depth_per_wavelength,
                     csf_scheme=args.csf_scheme,
                     csf_curvature=args.csf_curvature,
                     device=args.device)
        for dx in dx_list
    ]
    first = configs[0]

    cases = []
    for cfg in configs:
        print(f"--- dx = {cfg.dx*1e6:.1f} um: {cfg.pool_width*1e3:.3f} x "
              f"{cfg.pool_depth*1e3:.3f} mm, interface at "
              f"{interface_level(cfg)*1e3:.3f} mm, dt = {cfg.dt:.3e} s, "
              f"steps = {int(round(args.seconds/cfg.dt))}")
        started = time.perf_counter()
        case = run_case(cfg, args.wavelength, args.amplitude, args.seconds)
        case["wall_clock_s"] = time.perf_counter() - started
        err = case["frequency_error"]
        print(f"    particles = {case['particles']}, T_meas = "
              f"{case['period_measured_s']}, error = "
              f"{None if err is None else f'{100*err:.1f} %'}, "
              f"T_eta = {None if case['period_fit_interface'] is None else round(case['period_fit_interface']['period_s'], 6)}, "
              f"decay = {case['envelope_velocity']['decay_ratio']}, "
              f"density dev [{case['min_density_deviation']:+.2e}, "
              f"{case['max_density_deviation']:+.2e}], "
              f"level drift = {case['mean_level_drift_m']:+.2e} m, "
              f"wall = {case['wall_clock_s']:.1f} s")
        cases.append(case)

    measured = [c["frequency_error"] for c in cases
                if c["frequency_error"] is not None]
    report = {
        "benchmark": "capillary_wave",
        "scheme": {"csf_scheme": args.csf_scheme,
                   "csf_curvature": args.csf_curvature},
        "setup": {
            "sigma": args.sigma,
            "gravity": args.gravity,
            "wavelength": args.wavelength,
            "k": k,
            "amplitude": args.amplitude,
            "ak": k * args.amplitude,
            "rho0": first.rho0,
            "sound_speed": args.sound_speed,
            "seconds": args.seconds,
            "fluid_depth_m": H,
            "capillary_length_m": capillary_length(args.sigma, first.rho0,
                                                  args.gravity),
            "gravity_share_of_omega_sq": g_share,
            "mode": "cos(pi x / W) between free-slip walls, lambda = 2 W",
            "initial_condition": "displaced interface + local hydrostatic density",
            "observable": "mass-weighted cos(kx) mode of v_y",
        },
        "analytical": {"omega_rad_s": 2 * math.pi / period_analytic,
                       "period_s": period_analytic,
                       "formula": "omega^2 = (g k + sigma k^3/rho) tanh(k H)",
                       "reference": "Lamb (1932) Art. 267, capillary-gravity branch"},
        "cases": cases,
        "verdict": {
            "max_frequency_error": max(measured) if measured else None,
            "threshold": args.max_freq_error,
            "pass": bool(measured) and max(measured) < args.max_freq_error,
        },
    }

    if args.png_out:
        make_figure(report, Path(args.png_out))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.json_out}")

    if args.quick:
        # CI smoke test: only the machinery is exercised, no accuracy claim.
        report["verdict"]["smoke_only"] = True
        report["verdict"]["pass"] = True
    if not report["verdict"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()