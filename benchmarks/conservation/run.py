"""Conservation, invariance and time-integration audit of the merged solver.

Why this benchmark exists
-------------------------
``docs/algorithm-gap-analysis.md`` closes its status section with a list of
things a green CI does *not* establish: "It does **not** establish conservation,
convergence, or agreement with experiment."  Convergence got a ladder
(``tests/test_convergence.py``), agreement with a reference got B1-B3, but
**conservation was never measured at all**, and the time integrator was never
scored.  The solver step is

    v_{n+1} = v_n + dt * a(x_n, v_n)          (semi-implicit Euler)
    x_{n+1} = x_n + dt * v_{n+1}

which is first order in the trajectory and symplectic-ish in energy, but no
committed number backs either claim.

What is measured here (all references are analytic, no fitting)
---------------------------------------------------------------
G0  mass.  Lagrangian particles with fixed mass: ``sum(m)`` is constant by
    construction.  Recorded so the claim is explicit rather than implied.

G1  per-step force balance.  Every interaction in ``rhs`` is evaluated on both
    members of each pair with opposite sign (pressure gradient, and the
    viscous term through ``+-dv``), so the net force of the internal terms must
    vanish and

        sum_i m_i a_i = M g                                        (G1)

    must hold to roundoff.  Measured as

        leak = |sum_i m_i a_i - M g| / (sum_i |m_i a_i| + M |g|)

    which is dimensionless and directly comparable between cases.  Any
    configuration whose leak is far above roundoff has a *non-symmetric*
    operator; the two candidates in this solver are the viscous term (it
    weights the pair by ``m_j/rho_j`` for ``i`` and ``m_i/rho_i`` for ``j``, so
    the two halves do not cancel unless ``rho_i = rho_j``) and the CSF
    surface-tension force (computed per particle from a curvature field, not
    from a pair sum).

G2  momentum.  Without gravity, (G1) integrates to ``P(t) = P(0)``.  A free
    droplet in vacuum is run for ~0.5 capillary times and the drift

        drift = |P(t) - P(0)| / max_t sum_i |m_i v_i|

    is reported per configuration.

G3  Galilean invariance.  A uniform translation ``u0`` is not a force, so two
    runs that differ only by ``u0`` must produce the same internal motion.
    The shifted velocity and position fields are compared element-wise.

G4  time-integration order.  For semi-implicit Euler the per-step trajectory
    offset is ``+(1/2) a dt^2`` (``x_n - x_{n-1} = v_n dt = v_{n-1} dt +
    a dt^2`` against the exact ``v_{n-1} dt + (1/2) a dt^2``), so the position
    lags the exact ballistic solution ``x0 + v0 t + (1/2) g t^2`` by

        lag(t) = (1/2) g dt t                                        (G4)

    *in closed form*, with no free parameter.  A dt ladder at fixed physical
    time scores both the coefficient and the order: ``lag / (1/2 g dt T)``
    must be 1 and ``lag`` must halve when dt halves.

Geometry and parameters: a hexagonally packed 2-D droplet of radius
``R = 0.02 m`` at ``dx = 0.002 m`` (R/dx = 10), ``rho0 = 1000``, ``c0 = 20``,
``gamma = 7``, ``h = 1.3 dx``; surface tension (when used) ``sigma = 7.2 N/m``,
i.e. ten times water so the capillary time fits inside the run.  Wall
particles are *off* in every case: a static wall absorbs momentum by design,
which would confound G1/G2.

Usage::

    PYTHONPATH=src python benchmarks/conservation/run.py            # full
    PYTHONPATH=src python benchmarks/conservation/run.py --quick    # coarse
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    VISCOUS_LAPLACIAN_FACTOR,
    Config,
    pair_kernels,
    rhs,
    stable_dt,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent

RADIUS = 0.02          # m
DX = 0.002             # m
RHO0 = 1000.0          # kg/m^3
SOUND_SPEED = 20.0     # m/s
GAMMA = 7.0
GRAVITY = 9.81         # m/s^2
VISCOSITY = 0.01       # m^2/s (module default)
SIGMA = 7.2            # N/m


# --------------------------------------------------------------------------- #
# Case construction
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius: float, dx: float, rho0: float, center=(0.0, 0.0)):
    """Hexagonally packed circular drop, mass set so ``mass * number_density = rho0``.

    Same construction (and the same lattice constant) as
    ``benchmarks/laplace_law/run.py``.  Hex packing is used because the
    4-neighbour square lattice biases every SPH gradient.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((center[0] + x, center[1] + y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), mass, dtype=torch.float64)


def make_config(**kw) -> Config:
    base = dict(dx=DX, h_factor=1.3, rho0=RHO0, sound_speed=SOUND_SPEED,
                gamma=GAMMA, gravity=0.0, viscosity=VISCOSITY)
    base.update(kw)
    return Config(**base)


def _net_leak(mass, acc, gravity, m_total):
    """(G1) as a dimensionless number."""
    force = mass[:, None] * acc
    g_vec = torch.tensor([0.0, -gravity], dtype=acc.dtype)
    net = force.sum(dim=0) - m_total * g_vec
    scale = force.abs().sum() + m_total * abs(gravity)
    if float(scale) == 0.0:
        # Exactly zero internal force (e.g. a uniform-pressure lattice with
        # viscosity and gravity off): the leak is 0, not 0/0.
        return 0.0
    return float(net.norm() / scale)


def integrate(pos, vel, mass, cfg, dt, steps, every=10, track_st=False):
    """Semi-implicit Euler, identical to the loop in ``droplet_impact.run``.

    Returns a trace dict plus the run-level scalars.  ``track_st`` additionally
    scores the momentum balance of the CSF term on its own, by evaluating it
    separately from the pressure/viscous/gravity part.
    """
    h = cfg.h_factor * cfg.dx
    m_total = float(mass.sum())
    gravity = cfg.gravity
    p0 = (mass[:, None] * vel).sum(dim=0)
    p_scale = 0.0
    leaks: list[float] = []
    st_leaks: list[float] = []
    tr = {k: [] for k in ("t", "px", "py", "com_x", "com_y", "vmax", "ek",
                          "leak", "st_leak")}
    finite = True
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, _pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                  dt=dt)
        if not torch.isfinite(acc).all():
            finite = False
            break
        leaks.append(_net_leak(mass, acc, gravity, m_total))
        if track_st and cfg.surface_tension_coefficient > 0.0:
            acc_st = compute_surface_tension_force(pos, rho, mass, pairs, cfg,
                                                   wall=None)
            f_st = mass[:, None] * acc_st
            st_leaks.append(float(f_st.sum(dim=0).norm()
                                  / f_st.abs().sum().clamp_min(1e-300)))

        vel = vel + dt * acc
        pos = pos + dt * vel

        if step % every == 0 or step == steps - 1:
            p = (mass[:, None] * vel).sum(dim=0)
            com = (mass[:, None] * pos).sum(dim=0) / m_total
            speed = torch.linalg.vector_norm(vel, dim=1)
            p_scale = max(p_scale, float((mass * speed).sum()))
            tr["t"].append((step + 1) * dt)
            tr["px"].append(float(p[0]))
            tr["py"].append(float(p[1]))
            tr["com_x"].append(float(com[0]))
            tr["com_y"].append(float(com[1]))
            tr["vmax"].append(float(speed.max()))
            tr["ek"].append(float(0.5 * (mass @ (vel * vel).sum(dim=1))))
            tr["leak"].append(leaks[-1])
            tr["st_leak"].append(st_leaks[-1] if st_leaks else float("nan"))

    p_end = (mass[:, None] * vel).sum(dim=0)
    drift = float((p_end - p0).norm()) / max(p_scale, 1e-300)
    return {
        "_state": (pos, vel),
        "trace": tr,
        "finite": finite,
        "steps_run": len(leaks),
        "dt_s": dt,
        "physical_time_s": len(leaks) * dt,
        "n_particles": int(pos.shape[0]),
        "mass_total_kg": float(m_total),
        "mass_drift_kg": float(abs(float(mass.sum()) - m_total)),
        "leak_median": (float(torch.tensor(leaks).nanmedian())
                        if leaks else float("nan")),
        "leak_nan_steps": int(sum(1 for v in leaks if v != v)),
        "leak_max": max(leaks) if leaks else float("nan"),
        "st_leak_median": (float(torch.tensor(st_leaks).median())
                           if st_leaks else None),
        "st_leak_max": max(st_leaks) if st_leaks else None,
        "momentum_initial": [float(p0[0]), float(p0[1])],
        "momentum_final": [float(p_end[0]), float(p_end[1])],
        "momentum_drift_rel": drift,
        "momentum_scale_kg_m_per_s": p_scale,
    }


def jsonable(obj):
    """Drop the private (tensor-carrying) keys before serialising a report."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# G1/G2 — momentum balance of each operator combination
# --------------------------------------------------------------------------- #
MOMENTUM_CASES = {
    # name: (viscosity, sigma, note)
    "pressure_only": (0.0, 0.0, "pressure gradient only: the symmetric reference"),
    "viscous": (VISCOSITY, 0.0, "adds the m_j/rho_j-weighted viscous Laplacian"),
    "csf_only": (0.0, SIGMA, "adds the per-particle CSF force"),
    "default": (VISCOSITY, SIGMA, "module defaults with surface tension on"),
}


def run_momentum(steps: int) -> dict:
    out = {}
    for name, (visc, sigma, note) in MOMENTUM_CASES.items():
        cfg = make_config(viscosity=visc, surface_tension_coefficient=sigma)
        dt = float(stable_dt(cfg))
        pos, mass = hexagonal_droplet(RADIUS, DX, RHO0)
        vel = torch.zeros_like(pos)
        rep = integrate(pos, vel, mass, cfg, dt, steps,
                        track_st=(sigma > 0.0))
        rep["note"] = note
        rep["viscosity_m2_per_s"] = visc
        rep["sigma_N_per_m"] = sigma
        out[name] = rep
    return out


# --------------------------------------------------------------------------- #
# G3 — Galilean invariance
# --------------------------------------------------------------------------- #
def run_galilean(steps: int, u0: float = 0.5) -> dict:
    cfg = make_config()
    pos0, mass = hexagonal_droplet(RADIUS, DX, RHO0)
    rest = integrate(pos0.clone(), torch.zeros_like(pos0), mass, cfg,
                     float(stable_dt(cfg)), steps)
    cfg_m = make_config()
    dt = float(stable_dt(cfg_m, max_speed=u0))
    vel_m = torch.zeros_like(pos0)
    vel_m[:, 0] = u0
    moving = integrate(pos0.clone(), vel_m, mass, cfg_m, dt, steps)
    return {
        "u0_m_per_s": u0,
        "steps": steps,
        "dt_s": dt,
        "physical_time_s": steps * dt,
        "rest_vmax_m_per_s": max(rest["trace"]["vmax"]),
        "moving_vmax_m_per_s": max(moving["trace"]["vmax"]),
        "momentum_drift_rel_rest": rest["momentum_drift_rel"],
        "momentum_drift_rel_moving": moving["momentum_drift_rel"],
        "com_drift_m": [
            moving["trace"]["com_x"][-1] - rest["trace"]["com_x"][-1]
            - u0 * moving["trace"]["t"][-1],
            moving["trace"]["com_y"][-1] - rest["trace"]["com_y"][-1],
        ],
        "note": "a uniform translation must not change the internal motion; the "
                "residual is reported as an absolute COM offset in metres",
        "_traces": (rest["trace"], moving["trace"]),
    }


def run_galilean_field(steps: int, u0: float = 0.5) -> dict:
    """Field-level Galilean check: compare the velocity *fields*, not the COM."""
    cfg = make_config()
    pos_a, mass = hexagonal_droplet(RADIUS, DX, RHO0)
    vel_a = torch.zeros_like(pos_a)
    cfg_m = make_config()
    dt = float(stable_dt(cfg_m, max_speed=u0))
    pos_b = pos_a.clone()
    vel_b = torch.zeros_like(pos_a)
    vel_b[:, 0] = u0
    h = cfg.h_factor * cfg.dx
    for _ in range(steps):
        pairs_a = cell_linked_pairs(pos_a, 2.0 * h)
        _, _, acc_a = rhs(pos_a, vel_a, mass, cfg, pairs=pairs_a, wall=None,
                          dt=dt)
        pairs_b = cell_linked_pairs(pos_b, 2.0 * h)
        _, _, acc_b = rhs(pos_b, vel_b, mass, cfg_m, pairs=pairs_b, wall=None,
                          dt=dt)
        vel_a = vel_a + dt * acc_a
        pos_a = pos_a + dt * vel_a
        vel_b = vel_b + dt * acc_b
        pos_b = pos_b + dt * vel_b
    shift = torch.zeros_like(vel_b)
    shift[:, 0] = u0
    dv = torch.linalg.vector_norm(vel_b - shift - vel_a, dim=1).max()
    t = steps * dt
    dx_pos = (pos_b - shift * t - pos_a).abs().max()
    v_scale = torch.linalg.vector_norm(vel_a, dim=1).max().clamp_min(1e-30)
    return {
        "u0_m_per_s": u0,
        "steps": steps,
        "dt_s": dt,
        "velocity_field_mismatch_m_per_s": float(dv),
        "velocity_field_mismatch_rel": float(dv / v_scale),
        "position_field_mismatch_m": float(dx_pos),
        "position_scale_m": float(RADIUS),
    }


# --------------------------------------------------------------------------- #
# G4 — closed-form first-order trajectory offset
# --------------------------------------------------------------------------- #
def ballistic_row(steps: int, total_time: float, viscosity: float) -> dict:
    """One dt of the free-fall ladder: COM_y against x0 + v0 t + (1/2) g t^2."""
    dt = total_time / steps
    cfg = make_config(gravity=GRAVITY, viscosity=viscosity,
                      surface_tension_coefficient=0.0)
    pos, mass = hexagonal_droplet(RADIUS, DX, RHO0)
    vel = torch.zeros_like(pos)
    m_total = float(mass.sum())
    com0 = (mass[:, None] * pos).sum(dim=0) / m_total
    rep = integrate(pos, vel, mass, cfg, dt, steps, every=max(1, steps // 20))
    pos_f, vel_f = rep["_state"]
    com1 = (mass[:, None] * pos_f).sum(dim=0) / m_total
    t = rep["physical_time_s"]
    exact_y = float(com0[1]) - 0.5 * GRAVITY * t * t
    lag = exact_y - float(com1[1])
    dv_com = float((mass * vel_f[:, 1]).sum()) / m_total
    return {
        "steps": steps,
        "dt_s": dt,
        "physical_time_s": t,
        "viscosity_m2_per_s": viscosity,
        "com_y_initial_m": float(com0[1]),
        "com_y_final_m": float(com1[1]),
        "exact_y_final_m": exact_y,
        "fall_distance_m": 0.5 * GRAVITY * t * t,
        "lag_measured_m": lag,
        "lag_predicted_m": 0.5 * GRAVITY * dt * t,
        "lag_predicted_from_dv_com_m": -0.5 * dv_com * dt,
        "lag_ratio": lag / (0.5 * GRAVITY * dt * t),
        "lag_over_fall_distance": lag / (0.5 * GRAVITY * t * t),
        "dt_over_t": dt / t,
        "com_velocity_final_m_per_s": dv_com,
        "internal_com_acceleration_m_per_s2": abs(dv_com / t - GRAVITY),
        "finite": rep["finite"],
        "vmax_m_per_s": max(rep["trace"]["vmax"]),
    }


def run_ballistic(ladder: list[int], total_time: float) -> dict:
    """Free-fall ladder with viscosity off, so no operator leaks momentum.

    The viscous term is switched off on purpose: its (measurable, see G1b)
    momentum leak gives the centre of mass a spurious acceleration that would
    be indistinguishable from the ballistic lag being scored here.  One row at
    the finest step repeats the run at the module default viscosity so the size
    of that contamination is on the record rather than assumed to be zero.
    """
    rows = [ballistic_row(n, total_time, 0.0) for n in ladder]
    control = ballistic_row(ladder[-1], total_time, VISCOSITY)

    lags = [r["lag_measured_m"] for r in rows]
    dts = [r["dt_s"] for r in rows]
    order = None
    if len(rows) > 1 and all(v > 0 for v in lags):
        order = math.log(lags[-1] / lags[0]) / math.log(dts[-1] / dts[0])
    return {
        "gravity_m_per_s2": GRAVITY,
        "total_time_s": total_time,
        "rows": rows,
        "observed_order": order,
        "default_viscosity_control": control,
        "internal_contamination_rel": max(abs(r["lag_ratio"] - 1.0)
                                          for r in rows),
        "note": "lag_predicted = (1/2) g dt t is derived, not fitted: "
                "semi-implicit Euler advances x with the already-updated v.",
    }


def run_viscous_pair_check(nu: float = VISCOSITY) -> dict:
    """G1b: closed form for the momentum leak of one viscous pair.

    ``rhs`` weights the viscous term of particle ``i`` by ``m_j/rho_j`` and that
    of ``j`` by ``m_i/rho_i``.  Multiplying by the masses, the pair does not
    cancel unless the two densities are equal:

        F_i + F_j = C nu m_i m_j (v_i - v_j) L (1/rho_i - 1/rho_j)

    with ``L`` the (kernel-dependent, but pair-common) Laplacian weight.  A
    two-particle system measures every quantity in that expression, so the
    mechanism is identified rather than inferred: the pressure part is removed
    by differencing two ``rhs`` calls at the same state.
    """
    h = 1.3 * DX
    pos = torch.tensor([[0.0, 0.0], [DX, 0.0]], dtype=torch.float64)
    mass = torch.tensor([3.0, 1.5], dtype=torch.float64) * RHO0 * DX * DX
    vel = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    cfg_nu = make_config(viscosity=nu)
    cfg_0 = make_config(viscosity=0.0)
    rho, _, acc_nu = rhs(pos, vel, mass, cfg_nu, pairs=pairs, wall=None,
                         dt=cfg_nu.dt)
    _, _, acc_0 = rhs(pos, vel, mass, cfg_0, pairs=pairs, wall=None,
                      dt=cfg_0.dt)
    a_visc = acc_nu - acc_0
    net = (mass[:, None] * a_visc).sum(dim=0)

    delta = pos[0] - pos[1]
    dist = torch.linalg.vector_norm(delta)
    _, grad_w = pair_kernels(delta.unsqueeze(0), h)
    lap = (torch.relu(-(delta * grad_w[0]).sum())
           / (dist * dist + 0.01 * h * h))
    c = VISCOUS_LAPLACIAN_FACTOR * nu
    dv = vel[0] - vel[1]
    predicted = (c * mass[0] * mass[1] * dv * lap
                 * (1.0 / rho[0] - 1.0 / rho[1]))
    # Same pair, equal densities: the leak must vanish identically.
    mass_eq = torch.full_like(mass, RHO0 * DX * DX)
    rho_eq, _, acc_eq = rhs(pos, vel, mass_eq, cfg_nu, pairs=pairs, wall=None,
                            dt=cfg_nu.dt)
    _, _, acc_eq0 = rhs(pos, vel, mass_eq, cfg_0, pairs=pairs, wall=None,
                        dt=cfg_0.dt)
    net_eq = (mass_eq[:, None] * (acc_eq - acc_eq0)).sum(dim=0)

    return {
        "separation_m": DX,
        "h_m": h,
        "masses_kg": [float(m) for m in mass],
        "densities_kg_m3": [float(rho[0]), float(rho[1])],
        "relative_density_contrast": float(abs(rho[0] - rho[1]) / rho[1]),
        "net_pair_force_measured_N": [float(net[0]), float(net[1])],
        "net_pair_force_closed_form_N": [float(predicted[0]),
                                         float(predicted[1])],
        "closed_form_ratio": (
            float(net.norm() / predicted.norm())
            if float(predicted.norm()) > 0 else None),
        "equal_density_net_pair_force_N": float(net_eq.norm()),
        "note": "equal densities are the only case in which the term cancels; "
                "a free surface always breaks that condition",
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    rows = report["ballistic"]["rows"]
    dts = [r["dt_s"] for r in rows]
    ax.loglog(dts, [r["lag_measured_m"] for r in rows], "o-",
              label="measured COM lag")
    ax.loglog(dts, [r["lag_predicted_m"] for r in rows], "s--",
              label=r"closed form $\frac{1}{2} g\,\Delta t\,T$")
    ax.set_xlabel(r"time step $\Delta t$ [s]")
    ax.set_ylabel("ballistic position lag [m]")
    _order = report["ballistic"]["observed_order"]
    _order_txt = "n/a" if _order is None else f"{_order:.2f}"
    ax.set_title(f"G4: first-order trajectory offset, observed order {_order_txt}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    names = list(report["momentum"].keys())
    def _pos(v):
        return v if (v == v and v > 1e-18) else 1e-18

    leaks = [_pos(report["momentum"][n]["leak_median"]) for n in names]
    drift = [_pos(report["momentum"][n]["momentum_drift_rel"]) for n in names]
    xs = range(len(names))
    ax.bar([x - 0.2 for x in xs], leaks, width=0.4,
           label="per-step force leak (G1)")
    ax.bar([x + 0.2 for x in xs], drift, width=0.4,
           label="momentum drift over run (G2)")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("dimensionless")
    ax.set_title("G1/G2: momentum balance by operator set")
    ax.axhline(2.2e-16, color="k", ls=":", lw=1, label="float64 roundoff")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="coarse ladder for a smoke run")
    ap.add_argument("--out", default=str(HERE / "report.json"))
    args = ap.parse_args()

    if args.quick:
        ladder, total_time, mom_steps = [60, 120], 0.004, 90
    else:
        ladder, total_time, mom_steps = [300, 600, 1200, 2400], 0.01, 400

    momentum = run_momentum(mom_steps)
    pair_check = run_viscous_pair_check()
    ballistic = run_ballistic(ladder, total_time)
    galilean = run_galilean(200 if not args.quick else 60)
    gal_field = run_galilean_field(200 if not args.quick else 60)
    galilean.pop("_traces", None)

    rows = ballistic["rows"]
    ratio_dev = max(abs(r["lag_ratio"] - 1.0) for r in rows)
    leak_ref = momentum["pressure_only"]["leak_median"]
    report = {
        "case": {
            "geometry": "hexagonal 2-D droplet in vacuum, no wall particles",
            "radius_m": RADIUS, "dx_m": DX, "h_over_dx": 1.3,
            "rho0_kg_m3": RHO0, "sound_speed_m_per_s": SOUND_SPEED,
            "gamma": GAMMA, "viscosity_m2_per_s": VISCOSITY,
            "sigma_N_per_m": SIGMA,
            "n_particles": momentum["pressure_only"]["n_particles"],
            "capillary_time_s": math.sqrt(RHO0 * RADIUS**3 / SIGMA),
        },
        "momentum": momentum,
        "viscous_pair": pair_check,
        "ballistic": ballistic,
        "galilean": galilean,
        "galilean_field": gal_field,
        "verdicts": {
            "mass_conservation_exact": all(
                v["mass_drift_kg"] == 0.0 for v in momentum.values()),
            "G1_pressure_gradient_symmetric": leak_ref < 1e-12,
            "G1_pressure_gradient_leak": leak_ref,
            "G4_lag_matches_closed_form_within_3pct": ratio_dev < 0.03,
            "G4_max_ratio_deviation": ratio_dev,
            "G4_observed_order": ballistic["observed_order"],
            "G4_order_is_first_order": (
                ballistic["observed_order"] is not None
                and 0.9 <= ballistic["observed_order"] <= 1.1),
            "G3_galilean_velocity_field_rel": gal_field[
                "velocity_field_mismatch_rel"],
            "G3_galilean_ok": gal_field["velocity_field_mismatch_rel"] < 1e-9,
            "G1b_viscous_pair_closed_form_ratio": pair_check["closed_form_ratio"],
            "G1b_viscous_pair_equal_density_leak_N":
                pair_check["equal_density_net_pair_force_N"],
            "G2_viscous_leak_median": momentum["viscous"]["leak_median"],
            "G2_csf_leak_median": momentum["csf_only"]["leak_median"],
        },
    }

    report = jsonable(report)
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"particles            {report['case']['n_particles']}")
    print("momentum (G1 leak median / G2 drift):")
    for name, rep in momentum.items():
        print(f"  {name:14s} leak={rep['leak_median']:.3e} "
              f"max={rep['leak_max']:.3e} drift={rep['momentum_drift_rel']:.3e}"
              f" st_leak={rep['st_leak_median']}")
    print("ballistic (G4):")
    for r in rows + [ballistic["default_viscosity_control"]]:
        print(f"  dt={r['dt_s']:.3e} lag={r['lag_measured_m']:.6e} "
              f"pred={r['lag_predicted_m']:.6e} ratio={r['lag_ratio']:.5f}")
    print(f"  observed order {ballistic['observed_order']}")
    print(f"  internal contamination {ballistic['internal_contamination_rel']:.3e}")
    print(f"viscous pair: measured {pair_check['net_pair_force_measured_N']} "
          f"closed form {pair_check['net_pair_force_closed_form_N']} "
          f"ratio {pair_check['closed_form_ratio']} "
          f"equal-density {pair_check['equal_density_net_pair_force_N']:.3e}")
    print(f"galilean COM offset {galilean['com_drift_m']}, "
          f"field rel {gal_field['velocity_field_mismatch_rel']:.3e}")
    make_figure(report, HERE / "conservation.png")
    print(f"wrote {out} and {HERE / 'conservation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())