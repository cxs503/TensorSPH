"""Verification of the shipped viscous-transport operator against an exact solution.

What is verified
----------------
``tensorsph.droplet_impact.rhs`` carries the laminar viscous term
(``VISCOUS_LAPLACIAN_FACTOR * nu * sum_j (m_j / rho_j) * ReLU(-(r_ij . grad W_ij))
/ (|r_ij|^2 + 0.01 h^2)``, the regularised Morris form) and, with
``Config.wall_friction = 1``, a no-slip condition that lets the same term act on
fluid-wall pairs whose velocity is zero.

Two things were known before this script: the operator returns the requested
viscosity on *one* frozen field at *one* resolution
(``tests/test_pair_based_rhs.py::test_viscosity_is_calibrated``), and one flow
reproduced the Stokes shear-decay rate to a ratio of 0.84 at dx = 0.02
(``tests/test_boundary.py``), a 16 % error that is above the 15 % acceptance
band and was never checked under refinement. So the question this script
answers is: *does a flow integrated with this operator decay at the rate the
Navier-Stokes equations demand, and how does that error behave as dx falls?*

The exact solution used here
----------------------------
A slab between two parallel plates, no slip at both, no gravity, no body force,
initialised with the first viscous eigenmode:

    u_x(y, t) = U0 sin(k y) exp(-nu k^2 t),        k = pi / H,

which satisfies u = 0 at y = 0 and at y = H.  Its kinetic energy is

    E_k(t) = 1/2 sum m u^2 = E_k(0) exp(-2 nu k^2 t),

and its modal amplitude, ``A(t) = sum m u_x sin(k y) / sum m sin^2(k y)``, is
``U0 exp(-nu k^2 t)``.  Both are evaluated in closed form here and the two
independent estimators of the effective viscosity are

* ``nu_modal`` -- least squares on ``ln A(t)``; insensitive to energy that leaks
  into motion the exact mode does not have;
* ``nu_E``     -- least squares on ``ln E_k(t)``; the estimator the older test
  used, sensitive to every other motion that may be present.

The reference is the nominal ``nu`` the case was configured with, so the
reported quantity is ``nu_measured / nu_nominal``; 1.0 is exact.

Why the domain is what it is (measured, not assumed)
----------------------------------------------------
The obvious first choice -- a free-slip lid and only a floor, as the earlier
test used -- was tried and rejected: at dx = 0.01 the top surface rose by 32 % of
the channel height and the vertical velocity reached 2x the initial shear
amplitude, i.e. the "flat, purely horizontal" idealisation of the exact
solution was violated, and the measured viscosity drifted to 1.5x nominal at
dx = 0.005.  Adding a rigid lid (a wall lattice above y = H) removes the free
surface and the departure collapses (``peak_vertical_velocity`` below).  The
lateral ends have to stay open: an x-independent velocity field cannot satisfy
no-slip at x = 0 and x = L (a closed box measured ``nu_modal/nu = 17``), so the
ends are an artificial open boundary instead, and every metric is evaluated in
a central x-band away from them.

The consequence of leaving the ends open is measured too: at dx = 0.005 the
open boundary develops a disturbance that the band metrics still reject
(``|u_y|`` in the band stays at 2e-3 U0) but which raises the global compression
to 4 % and makes the two estimators disagree.  Such a run is flagged
``idealisation_ok = False`` and is excluded from the convergence fit instead of
being quietly dropped.

Controls
--------
* **viscosity halving** -- at dx = 0.02 the same case at nu/2 must decay at half
  the rate: proves the estimator tracks the configured viscosity rather than
  reproducing one number at one setting.
* **time-step halving** -- at dx = 0.02 the same case at dt/2 must give the same
  rate: the ladder refines dt with dx, so without this the ladder could not tell
  spatial from temporal error.
* **momentum** -- no gravity, no body force: every fluid-fluid pressure and
  viscous pair is antisymmetric in its two members, so ``sum m u_x`` of the
  fluid may only change through the wall reactions, which the static wall
  particles do not receive back.  The wall-free run isolates the operators.

Artifacts
---------
``results.json``  every measured number, plus the exact-solution samples, so the
                  document and the tests read one file;
``viscous_transport.png``  decay curves, profile comparison, error ladder.

Run:  python benchmarks/viscous_transport/run.py [--resolutions 0.04,0.02,0.01,0.005]
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorsph.droplet_impact import (  # noqa: E402
    Config,
    PairListCache,
    apply_boundary,
    rhs,
)

HEIGHT = 0.2      # channel height H (m)
WIDTH = 0.6       # slab width (m); the ends are an artificial open boundary
U0 = 0.1          # initial shear amplitude (m/s)
NU = 0.05         # nominal kinematic viscosity (m^2/s)
COURANT = 0.29    # acoustic Courant, held fixed across the ladder
T_END = 0.25      # physical end time (s)
BAND = (0.35, 0.65)   # x-band, as a fraction of WIDTH, where metrics are taken

# A run is only used for the convergence claim if the idealisation of the exact
# solution holds on it.  The thresholds are not tuned: they are the levels at
# which the departure from the exact solution is far below the error being
# measured.
IDEALISATION = {
    "peak_vertical_velocity": 0.02 * U0,   # flow must stay horizontal
    "peak_density_compression": 0.01,      # flow must stay incompressible
    "peak_mach": 0.10,                     # flow must stay subsonic
    "estimator_agreement": 0.10,           # the two estimators must agree
}


def wave_number() -> float:
    return math.pi / HEIGHT


def exact_velocity(y, t: float, nu: float = NU, u0: float = U0):
    k = wave_number()
    return u0 * math.sin(k * float(y)) * math.exp(-nu * k * k * t)


def exact_energy_fraction(t: float, nu: float = NU) -> float:
    return math.exp(-2.0 * nu * wave_number() ** 2 * t)


def exact_modal_amplitude(t: float, nu: float = NU) -> float:
    return U0 * math.exp(-nu * wave_number() ** 2 * t)


def channel_wall(dx: float, width: float, height: float, layers: int = 3,
                 top: bool = True, sides: bool = False):
    """Static wall particles above and below the slab, three layers each.

    The shipped ``wall_lattice`` builds a floor and two side columns; this case
    needs a *lid* and no sides (see the module docstring), so the lattice is
    built here in the same style: same spacing, half a spacing outside the fluid
    domain, extended around the corners by ``(layers - 0.5) dx``.  The wall mass
    and the Adami et al. (2012) pressure extrapolation are the solver's own.
    """
    xs = torch.arange(-(layers - 0.5) * dx, width + (layers - 0.5) * dx, dx)
    parts = []
    for layer in range(1, layers + 1):
        levels = [dx * 0.5 - layer * dx]
        if top:
            levels.append(height + dx * (layer - 0.5))
        for y in levels:
            gx, gy = torch.meshgrid(xs, torch.tensor([y]), indexing="xy")
            parts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1))
    if sides:
        ys = torch.arange(dx * 0.5, height, dx)
        for layer in range(1, layers + 1):
            for x in (-(layer - 0.5) * dx, width + (layer - 0.5) * dx):
                gx, gy = torch.meshgrid(torch.tensor([x]), ys, indexing="xy")
                parts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1))
    return torch.cat(parts, dim=0)


def build_case(dx: float, viscosity: float = NU, friction: float = 1.0,
               dt_scale: float = 1.0, with_wall: bool = True,
               wall_mode: str = "lid"):
    cfg = Config(dx=dx, pool_width=WIDTH, pool_depth=HEIGHT, drop_radius=2 * dx,
                 drop_center_y=HEIGHT + 0.5, gravity=0.0, viscosity=viscosity,
                 wall_friction=friction, device="cpu", output_every=10**9)
    cfg.dt = dt_scale * COURANT * cfg.h_factor * dx / cfg.sound_speed
    xs = torch.arange(dx * 0.5, WIDTH, dx)
    ys = torch.arange(dx * 0.5, HEIGHT, dx)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx)
    if not with_wall:
        wall = None
    elif wall_mode == "lid":
        wall = channel_wall(dx, WIDTH, HEIGHT)
    elif wall_mode == "floor":
        wall = channel_wall(dx, WIDTH, HEIGHT, top=False)
    elif wall_mode == "box":
        wall = channel_wall(dx, WIDTH, HEIGHT, sides=True)
    else:
        raise ValueError(f"unknown wall_mode {wall_mode!r}")
    vel = torch.zeros_like(pos)
    vel[:, 0] = U0 * torch.sin(wave_number() * pos[:, 1])
    return cfg, pos, mass, wall, vel


def _least_squares_slope(ts, values):
    mean_t = sum(ts) / len(ts)
    mean_v = sum(values) / len(values)
    numerator = sum((a - mean_t) * (b - mean_v) for a, b in zip(ts, values))
    denominator = sum((a - mean_t) ** 2 for a in ts)
    return numerator / denominator


def run_case(dx: float, viscosity: float = NU, friction: float = 1.0,
             dt_scale: float = 1.0, t_end: float = T_END, sample_every: int = 5,
             keep_profiles: bool = False, with_wall: bool = True,
             wall_mode: str = "lid"):
    """Integrate the shipped operator with the shipped explicit Euler step.

    The loop mirrors ``run()``: ``vel += dt * acc`` then ``pos += dt * vel``,
    ``apply_boundary`` after the drift, and a Verlet pair cache.
    """
    cfg, pos, mass, wall, vel = build_case(dx, viscosity, friction, dt_scale,
                                          with_wall, wall_mode)
    steps = int(round(t_end / cfg.dt))
    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=pos.shape[0])
    all_pos = torch.cat((pos, wall), dim=0) if wall is not None else pos

    k = wave_number()
    band = ((pos[:, 0] > BAND[0] * WIDTH) & (pos[:, 0] < BAND[1] * WIDTH))
    band_mass = mass * band
    sin_k = torch.sin(k * pos[:, 1])
    modal_norm = float((band_mass * sin_k.square()).sum())
    row_index = torch.round((pos[:, 1] - cfg.dx * 0.5) / cfg.dx).long()
    rows = sorted({int(r) for r in row_index[band]})

    samples = []
    profiles = {}
    initial_energy = float(0.5 * (band_mass[:, None] * vel.square()).sum())
    initial_momentum = float((mass[:, None] * vel).sum(dim=0)[0])
    peak_mach = 0.0
    peak_compression = 0.0
    peak_u_y_band = 0.0
    peak_u_y_global = 0.0
    peak_rise = 0.0
    boundary_events = 0

    for step in range(steps + 1):
        if wall is not None:
            all_pos[: pos.shape[0]] = pos
        pairs, _ = cache.get(all_pos)
        rho, _pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall,
                                  dt=cfg.dt)
        if not bool(torch.isfinite(vel).all() and torch.isfinite(acc).all()):
            raise SystemExit(f"non-finite state at step {step} (dx={dx:g})")

        speed = torch.linalg.vector_norm(vel, dim=-1)
        if step % sample_every == 0 or step == steps:
            t = step * cfg.dt
            energy = 0.5 * float((band_mass[:, None] * vel.square()).sum())
            amplitude = float((band_mass * vel[:, 0] * sin_k).sum()) / modal_norm
            samples.append({
                "step": step,
                "time": t,
                "kinetic_energy": energy,
                "kinetic_energy_ratio": energy / initial_energy,
                "exact_energy_ratio": exact_energy_fraction(t, viscosity),
                "modal_amplitude": amplitude,
                "exact_modal_amplitude": exact_modal_amplitude(t, viscosity),
                "momentum_x": float((mass[:, None] * vel).sum(dim=0)[0]),
            })
            if keep_profiles:
                profiles[f"{t:.6f}"] = _profile(vel, row_index, rows, band, mass, cfg)
        peak_compression = max(peak_compression,
                               float(rho.max()) / cfg.rho0 - 1.0)
        peak_mach = max(peak_mach, float(speed.max()) / cfg.sound_speed)
        peak_u_y_band = max(peak_u_y_band, float(vel[band, 1].abs().max()))
        peak_u_y_global = max(peak_u_y_global, float(vel[:, 1].abs().max()))
        if wall is None:
            peak_rise = 0.0
        else:
            peak_rise = max(peak_rise,
                            float(pos[:, 1].max()) - (HEIGHT - cfg.dx * 0.5))
        boundary_events += apply_boundary(pos, vel, cfg)
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel

    nu_E, nu_modal = fit_nu_energy(samples)
    final_momentum = float((mass[:, None] * vel).sum(dim=0)[0])
    result = {
        "dx": dx,
        "viscosity": viscosity,
        "wall_friction": friction,
        "with_wall": with_wall,
        "wall_mode": wall_mode,
        "dt": cfg.dt,
        "dt_scale": dt_scale,
        "steps": steps,
        "t_end": t_end,
        "h_over_dx": cfg.h_factor,
        "rows_across_channel": len(rows),
        "particle_count": int(pos.shape[0]),
        "wall_particle_count": 0 if wall is None else int(wall.shape[0]),
        "acoustic_courant": cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx),
        "peak_mach": peak_mach,
        "peak_density_compression": peak_compression,
        "peak_vertical_velocity": peak_u_y_band,
        "peak_vertical_velocity_global": peak_u_y_global,
        "peak_surface_rise": peak_rise,
        "boundary_events": boundary_events,
        "nu_from_energy": nu_E,
        "nu_from_modal": nu_modal,
        "nu_energy_ratio": (nu_E / viscosity) if viscosity else None,
        "nu_modal_ratio": (nu_modal / viscosity) if viscosity else None,
        "momentum_x_initial": initial_momentum,
        "momentum_x_final": final_momentum,
        "momentum_x_relative_drift": (
            abs(final_momentum - initial_momentum)
            / max(abs(initial_momentum), 1e-30)),
        "profile_error": None,
        "profile": None,
        "samples": samples,
    }
    result["nu_modal_ratio_err"] = (abs(result["nu_modal_ratio"] - 1.0)
                                    if result["nu_modal_ratio"] is not None else None)
    result["nu_energy_ratio_err"] = (abs(result["nu_energy_ratio"] - 1.0)
                                     if result["nu_energy_ratio"] is not None else None)
    result["idealisation"] = idealisation_report(result)
    result["idealisation_ok"] = all(result["idealisation"].values())
    if keep_profiles:
        result["profile"] = profiles
        result["profile_error"] = _profile_errors(profiles, viscosity)
    return result


def idealisation_report(result: dict) -> dict:
    """Which assumptions of the exact solution the run actually satisfies."""
    return {
        "flow_is_horizontal": result["peak_vertical_velocity"]
        <= IDEALISATION["peak_vertical_velocity"],
        "flow_is_incompressible": result["peak_density_compression"]
        <= IDEALISATION["peak_density_compression"],
        "flow_is_subsonic": result["peak_mach"] <= IDEALISATION["peak_mach"],
        "estimators_agree": (result["nu_modal_ratio"] is not None
                             and abs(result["nu_modal_ratio"]
                                     - result["nu_energy_ratio"])
                             <= IDEALISATION["estimator_agreement"]),
    }


def _profile(vel, row_index, rows, band, mass, cfg):
    """Mass-weighted mean u_x per lattice row, central band only."""
    out = {"y": [], "u": []}
    u = vel[:, 0]
    for row in rows:
        mask = band & (row_index == row)
        if int(mask.sum()) == 0:
            continue
        w = mass[mask]
        out["y"].append((float(row) + 0.5) * cfg.dx)
        out["u"].append(float((w * u[mask]).sum() / w.sum()))
    return out


def _profile_errors(profiles, viscosity):
    errors = {}
    for key, prof in profiles.items():
        t = float(key)
        if not prof["y"]:
            continue
        exact = [exact_velocity(y, t, viscosity) for y in prof["y"]]
        diff = [a - b for a, b in zip(prof["u"], exact)]
        norm = math.sqrt(sum(b * b for b in exact)) or 1.0
        errors[key] = {
            "time": t,
            "l2_relative": math.sqrt(sum(d * d for d in diff)) / norm,
            "linf_fraction_of_u0": max(abs(d) for d in diff) / U0,
        }
    return errors


def fit_nu_energy(samples, k=None):
    """Fit nu from the modal amplitude and from the kinetic energy.

    The fit windows are stated rather than optimised: the modal amplitude is
    used while it is between 35 % and 98 % of U0, the energy while it is between
    50 % and 98 % of its initial value.  Outside those windows the residual is
    at the level of the float32 round-off of the sum.
    """
    k = wave_number() if k is None else k
    energy_window = [s for s in samples if 0.5 <= s["kinetic_energy_ratio"] <= 0.98]
    modal_window = [s for s in samples
                    if 0.35 <= s["modal_amplitude"] / U0 <= 0.98]
    nu_E = float("nan")
    nu_modal = float("nan")
    if len(energy_window) >= 3:
        slope = _least_squares_slope([s["time"] for s in energy_window],
                                     [math.log(s["kinetic_energy"])
                                      for s in energy_window])
        nu_E = -slope / (2.0 * k * k)
    if len(modal_window) >= 3:
        slope = _least_squares_slope(
            [s["time"] for s in modal_window],
            [math.log(max(s["modal_amplitude"], 1e-30)) for s in modal_window])
        nu_modal = -slope / (k * k)
    return nu_E, nu_modal


def frozen_field_viscosity(dx: float, viscosity: float = NU,
                           with_wall: bool = True):
    """Decay rate the operator imposes on the exact mode *before any motion*.

    One right-hand-side evaluation at t = 0.  With ``u = U0 sin(k y)`` the exact
    equation gives ``du/dt = -nu k^2 u``, so

        nu_inst = -(sum m a_x sin(k y) / sum m sin^2(k y)) / (k^2 U0)

    is the viscosity the discrete operator actually applies to that mode.  It
    isolates the operator's transfer function from the wall closure, the
    pressure response and the integrator: if it reproduces the flow-level
    result, the deficit is the regularised Laplacian under-resolving the mode,
    and no amount of tuning elsewhere will move it.
    """
    cfg, pos, mass, wall, vel = build_case(dx, viscosity, 1.0, 1.0, with_wall)
    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=pos.shape[0])
    all_pos = torch.cat((pos, wall), dim=0) if wall is not None else pos
    pairs, _ = cache.get(all_pos)
    _rho, _p, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall, dt=cfg.dt)
    band = (pos[:, 0] > BAND[0] * WIDTH) & (pos[:, 0] < BAND[1] * WIDTH)
    band_mass = mass * band
    sin_k = torch.sin(wave_number() * pos[:, 1])
    norm = float((band_mass * sin_k.square()).sum())
    acc_modal = float((band_mass * acc[:, 0] * sin_k).sum()) / norm
    k = wave_number()
    nu_inst = -acc_modal / (k * k * U0)
    return {
        "dx": dx,
        "with_wall": with_wall,
        "nu_inst": nu_inst,
        "nu_ratio": nu_inst / viscosity,
        "acceleration_modal": acc_modal,
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def plot(results, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = sorted(results["ladder"], key=lambda r: r["dx"])
    usable = [r for r in ladder if r["idealisation_ok"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    ax = axes[0][0]
    for run in ladder:
        style = "-" if run["idealisation_ok"] else ":"
        ax.semilogy([s["time"] for s in run["samples"]],
                    [s["kinetic_energy_ratio"] for s in run["samples"]],
                    style, label=f"dx = {run['dx']:g} m"
                                 + ("" if run["idealisation_ok"] else " (excluded)"))
    reference = usable[0] if usable else ladder[0]
    ax.semilogy([s["time"] for s in reference["samples"]],
                [s["exact_energy_ratio"] for s in reference["samples"]],
                "k--", lw=1.0, label="exact $e^{-2\\nu k^2 t}$")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("$E_k(t)/E_k(0)$ (central band)")
    ax.set_title("Kinetic energy of the decaying shear mode")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    finest = min(usable, key=lambda r: r["dx"]) if usable else ladder[0]
    profiles = finest["profile"] or {}
    times = sorted(profiles, key=float)
    for key in ([times[0], times[len(times) // 2], times[-1]] if times else []):
        t = float(key)
        prof = profiles[key]
        line, = ax.plot(prof["u"], prof["y"], "o", ms=3,
                        label=f"SPH t = {t:g} s (dx = {finest['dx']:g})")
        ys = [(i / 40.0) * HEIGHT for i in range(41)]
        ax.plot([exact_velocity(y, t) for y in ys], ys, "-", lw=1.0,
                color=line.get_color(), label=f"exact t = {t:g} s")
    ax.set_xlabel("$u_x$ (m/s)")
    ax.set_ylabel("y (m)")
    ax.set_title("Profile vs exact mode, finest usable resolution")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    if usable:
        dxs = [r["dx"] for r in usable]
        ax.loglog(dxs, [r["nu_modal_ratio_err"] for r in usable], "o-",
                  label="|$\\nu_{modal}/\\nu-1$|")
        ax.loglog(dxs, [r["nu_energy_ratio_err"] for r in usable], "s-",
                  label="|$\\nu_E/\\nu-1$|")
        slope = results["convergence"]["modal_slope"]
        if slope == slope:  # not NaN
            ax.loglog(dxs, [dyx(d, dxs[0], slope) for d in dxs], "k--", lw=1.0,
                      label=f"observed order {slope:.2f}")
    excluded = [r for r in ladder if not r["idealisation_ok"]]
    if excluded:
        ax.loglog([r["dx"] for r in excluded],
                  [r["nu_modal_ratio_err"] for r in excluded], "x",
                  color="grey", ms=9, label="excluded (idealisation)")
    ax.axhline(0.15, color="r", ls=":", lw=1.0, label="15 % acceptance")
    ax.axhline(0.03, color="orange", ls=":", lw=1.0, label="3 %")
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("relative viscosity error")
    ax.set_title("Effective viscosity error under refinement")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][1]
    labels, values, colors = [], [], []
    for run in ladder:
        if not run["idealisation_ok"]:
            continue
        labels += [f"dx={run['dx']:g}", f"dx={run['dx']:g}"]
        values += [run["nu_modal_ratio"], run["nu_energy_ratio"]]
        colors += ["C0", "C1"]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(1.0, color="k", lw=1.0)
    ax.axhspan(0.85, 1.15, color="red", alpha=0.08)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("$\\nu_{measured}/\\nu_{nominal}$")
    ax.set_title("Measured vs configured viscosity (blue modal, orange energy)")
    fig.suptitle("TensorSPH: viscous transport verified against the exact Stokes decay",
                 fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125)
    print(f"figure -> {out_png}")


def dyx(d: float, d0: float, slope: float) -> float:
    """Reference line for an error decaying as d**slope."""
    return (d / d0) ** slope


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="0.04,0.02,0.01,0.005")
    parser.add_argument("--viscosity", type=float, default=NU)
    parser.add_argument("--t-end", type=float, default=T_END)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resolutions = [float(x) for x in args.resolutions.split(",") if x.strip()]

    started = time.time()
    ladder = []
    for dx in resolutions:
        t0 = time.time()
        run = run_case(dx, viscosity=args.viscosity, t_end=args.t_end,
                       sample_every=args.sample_every, keep_profiles=True)
        run["wall_clock"] = time.time() - t0
        ladder.append(run)
        print(f"dx={dx:g}  rows={run['rows_across_channel']:2d}  "
              f"N={run['particle_count']:5d}  "
              f"nu_modal/nu={run['nu_modal_ratio']:.4f}  "
              f"nu_E/nu={run['nu_energy_ratio']:.4f}  "
              f"|u_y|/U0={run['peak_vertical_velocity'] / U0:.2e}  "
              f"comp={run['peak_density_compression'] * 100:.2f}%  "
              f"ok={run['idealisation_ok']}  ({run['wall_clock']:.1f} s)")

    base = next((r for r in ladder if abs(r["dx"] - 0.02) < 1e-12), None)
    if base is None:
        base = run_case(0.02, viscosity=args.viscosity, t_end=args.t_end,
                        sample_every=args.sample_every)
        base["idealisation_ok"] = all(idealisation_report(base).values())

    half_nu = run_case(0.02, viscosity=args.viscosity / 2.0, t_end=args.t_end,
                       sample_every=args.sample_every)
    viscosity_control = {
        "description": "half the viscosity must halve the decay rate",
        "nu": args.viscosity / 2.0,
        "nu_from_modal": half_nu["nu_from_modal"],
        "ratio_to_base": half_nu["nu_from_modal"] / base["nu_from_modal"],
        "expected_ratio": 0.5,
    }
    print(f"viscosity control: nu/2 -> {viscosity_control['ratio_to_base']:.4f} "
          f"of the base rate (expected 0.5)")

    half_dt = run_case(0.02, viscosity=args.viscosity, t_end=args.t_end,
                       sample_every=args.sample_every, dt_scale=0.5)
    dt_control = {
        "description": "half the time step must not change the rate",
        "dt": half_dt["dt"],
        "nu_from_modal": half_dt["nu_from_modal"],
        "ratio_to_base": half_dt["nu_from_modal"] / base["nu_from_modal"],
        "expected_ratio": 1.0,
    }
    print(f"dt control: dt/2 -> {dt_control['ratio_to_base']:.4f} of the base rate")

    inviscid = run_case(0.02, viscosity=0.0, t_end=args.t_end,
                        sample_every=args.sample_every, with_wall=False)
    viscous = run_case(0.02, viscosity=args.viscosity, t_end=args.t_end,
                       sample_every=args.sample_every, with_wall=False)
    momentum_invariant = {
        "inviscid_without_walls": {
            "description": "no viscosity, no walls: the pressure term alone is "
                           "pairwise antisymmetric, so sum m u_x is invariant",
            "relative_drift": inviscid["momentum_x_relative_drift"],
        },
        "viscous_without_walls": {
            "description": "viscosity on, no walls: the m_j/rho_j weighting is "
                           "antisymmetric only when rho_i = rho_j, so this is the "
                           "momentum the viscous form does not conserve",
            "relative_drift": viscous["momentum_x_relative_drift"],
            "peak_density_compression": viscous["peak_density_compression"],
        },
        "with_walls": {
            "description": "with rigid wall particles the reaction is discarded "
                           "by design (the walls are static), so this drift is a "
                           "boundary reaction, not a defect",
            "relative_drift": base["momentum_x_relative_drift"],
        },
    }
    print(f"momentum drift: inviscid/no walls {inviscid['momentum_x_relative_drift']:.2e}, "
          f"viscous/no walls {viscous['momentum_x_relative_drift']:.2e}")

    rejected = []
    for mode in ("floor", "box"):
        variant = run_case(0.02, viscosity=args.viscosity, t_end=args.t_end,
                           sample_every=args.sample_every, wall_mode=mode)
        rejected.append(variant)
        print(f"rejected variant {mode}: nu_modal/nu="
              f"{variant['nu_modal_ratio']:.3f}  "
              f"|u_y|/U0={variant['peak_vertical_velocity'] / U0:.1e}  "
              f"rise={variant['peak_surface_rise']:.2e}  "
              f"comp={100 * variant['peak_density_compression']:.2f}%")

    frozen = []
    for dx in resolutions:
        frozen.append(frozen_field_viscosity(dx, args.viscosity, with_wall=False))
        frozen.append(frozen_field_viscosity(dx, args.viscosity, with_wall=True))
    print("frozen field: " + ", ".join(
        f"dx={f['dx']:g} wall={f['with_wall']} -> {f['nu_ratio']:.3f}"
        for f in frozen))

    usable = sorted([r for r in ladder if r["idealisation_ok"]], key=lambda r: r["dx"])
    convergence = {"points": [], "usable_resolutions": [r["dx"] for r in usable]}
    if len(usable) >= 3:
        tail = usable[:3]
        for run in tail:
            convergence["points"].append({
                "dx": run["dx"],
                "nu_modal_ratio_err": run["nu_modal_ratio_err"],
                "nu_energy_ratio_err": run["nu_energy_ratio_err"],
            })
        for key, label in (("nu_modal_ratio_err", "modal_slope"),
                           ("nu_energy_ratio_err", "energy_slope")):
            pairs = [(math.log(p["dx"]), math.log(p[key])) for p in convergence["points"]]
            convergence[label] = _least_squares_slope([p[0] for p in pairs],
                                                      [p[1] for p in pairs])
    else:
        convergence["modal_slope"] = float("nan")
        convergence["energy_slope"] = float("nan")

    results = {
        "case": {
            "height": HEIGHT, "width": WIDTH, "u0": U0,
            "viscosity": args.viscosity, "t_end": args.t_end,
            "courant": COURANT, "wave_number": wave_number(),
            "band": list(BAND),
            "exact_velocity": "u_x(y,t) = U0 sin(k y) exp(-nu k^2 t), k = pi/H",
            "exact_energy": "E_k(t) = E_k(0) exp(-2 nu k^2 t)",
            "exact_modal": "A(t) = U0 exp(-nu k^2 t)",
            "initial_condition": "u_x = U0 sin(pi y / H), u_y = 0, no body force",
            "boundaries": "three layers of wall particles above and below the "
                          "slab, wall_friction = 1 (no slip at both plates); the "
                          "lateral ends are an open boundary and no metric is "
                          "taken near them",
            "integrator": "explicit Euler, vel += dt*acc; pos += dt*vel, "
                          "apply_boundary after each drift, Verlet pair cache",
        },
        "idealisation_thresholds": IDEALISATION,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "ladder": ladder,
        "viscosity_control": viscosity_control,
        "dt_control": dt_control,
        "momentum_invariant": momentum_invariant,
        "convergence": convergence,
        "frozen_field": frozen,
        "rejected_variants": rejected,
        "wall_clock_total": time.time() - started,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"results -> {out / 'results.json'}  "
          f"({results['wall_clock_total']:.1f} s)")
    if not args.no_plot:
        plot(results, out / "viscous_transport.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())