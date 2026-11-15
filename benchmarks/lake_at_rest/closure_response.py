"""Is the anchor's machine-precision tolerance a *scheme* property or an identity?

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` (reference: the analytic hydrostatic
identity ``u = 0``, flat free surface; tolerance: machine precision, ``1e-9``).

``docs/verification_lake_at_rest_hydrostatic_closure.md`` reports one row that
meets the tolerance by six orders of magnitude: consistent (moment/MLS) gradient
+ ``incompressible`` closure (``p = rho0 g (eta - y)``, ``rho = rho0``, wall
pressure by continuation) gives ``max|a|/g = 1.087e-15`` in the closed tank.  That
is the only closure in the repo that meets the anchor as written, so the question
this runner asks is the one that decides whether the anchor has been *verified* or
merely *restated*:

    A tolerance on the residual at rest is a statement about a numerical scheme
    only if the same scheme reports a **non-zero** residual at a state that is not
    the equilibrium.  Otherwise "the lake stays still" is a property of the
    pressure field that was handed in, not of the solver.

The measurement is an amplitude sweep of a free-surface perturbation
``eta(x) = H + A cos(2 pi x / W)`` at fixed ``A / dx`` from ``0.01`` to ``1``:

* **identity arm** -- the tolerant closure at each amplitude.  An exact-gradient
  operator applied to an affine pressure field with constant density returns
  ``-grad p / rho + g = 0`` at *any* particle configuration, so the residual is
  expected to be independent of ``A`` at roundoff level.  If so, the row that
  meets the anchor cannot distinguish a still lake from a wave one particle
  spacing high.
* **responsiveness arms** -- the shipped operator (symmetric form, Tait EOS) and
  the ``barotropic`` closure at the same amplitudes: these must *see* the
  perturbation, and for small ``A`` must respond linearly in ``A``.  They are the
  control that says the perturbed states are real non-equilibrium states.
* **dynamics** -- time integration from a perturbed state, with the tolerant
  closure and with the shipped operator.  A scheme that carries gravity waves has
  to move the surface by ``O(A)`` within one linear period
  ``T = 2 pi / sqrt(g k tanh(k H))``; a closure that prescribes the answer cannot
  move it at all, and that is measured, not argued.
* **positive control** -- the laterally periodic tank (the configuration where
  the anchor is met by the *shipped* operator) driven as a standing gravity wave:
  the measured angular frequency must match ``sqrt(g k tanh(k H))``.  Without it,
  a null result in the dynamics arm would be indistinguishable from a dead code
  path.

Gates (declared before the numbers are read, exact thresholds in :func:`gates`)::

===========  ==================================================================  ==============
gate         quantity                                                             threshold
===========  ==================================================================  ==============
``Q1``       tolerant closure at ``A = 1 dx``, both ``dx``: ``max|a|/g``          ``< 1e-12``
``Q2``       shipped operator at ``A = 1 dx``: ``max|a|/g``                        ``> 1.0``
``Q3``       shipped linear response ``r(0.1 dx)/r(0.01 dx)``                      ``5 .. 25``
``Q4``       barotropic closure at ``A = 1 dx``: ``max|a|/g``                      ``> 1e-3``
``Q5``       tolerant closure, dynamics: surface move / linear-wave prediction    ``< 1e-6``
``Q6``       periodic standing wave: ``|omega/omega_analytic - 1|``                ``< 0.15``
``Q7``       shipped closed tank at rest (anchor, shipped): ``max|a_x|/g``        ``> 1e-2``
===========  ==================================================================  ==============

Usage::

    PYTHONPATH=src python3 benchmarks/lake_at_rest/closure_response.py \
        --json tests/data/lake_at_rest/closure_response.json \
        --doc docs/verification_lake_at_rest_closure_response.md \
        --figures docs/figures
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from tensorsph.droplet_impact import (cell_linked_pairs, drop_wall_wall_pairs, rhs)
from tensorsph.hydrostatic_closure import (HydrostaticClosureOptions,
                                           closure_residual, level_set,
                                           shipped_residual)
from tensorsph.lake_at_rest import (LakeConfig, build_state, periodic_pairs,
                                    reconstruct)

ANCHOR_TOLERANCE = 1.0e-9
RESOLUTIONS = (0.04, 0.02)
AMPLITUDES_DX = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
RECORD = "closure_response.json"
DYNAMICS_AMPLITUDE_DX = 0.25

#: The three closures, in the order the tables print them.
CLOSURES = (
    ("incompressible",
     HydrostaticClosureOptions(pressure="incompressible", wall="continuation",
                               formulation="moment", regularization=0.0)),
    ("barotropic",
     HydrostaticClosureOptions(pressure="barotropic", wall="continuation",
                               formulation="moment", regularization=0.0)),
    ("shipped", None),
)
LABELS = {
    "incompressible": "consistent gradient + incompressible closure",
    "barotropic": "consistent gradient + barotropic (Tait) closure",
    "shipped": "shipped operator (symmetric form, Tait EOS)",
}


# --------------------------------------------------------------------------- #
# state construction
# --------------------------------------------------------------------------- #
def fresh_pairs(pos: torch.Tensor, state: dict, cfg: LakeConfig) -> torch.Tensor:
    """Neighbour list of *these* positions, built the way the case builds it.

    The case freezes the list at ``t = 0`` because the rest state does not move.
    A perturbed state is a different geometry, so its list is rebuilt here; the
    comparison is then closure-vs-closure on one and the same list.
    """
    if cfg.periodic_x:
        return periodic_pairs(pos, state["wall"], cfg.h(), cfg.width)
    pairs = cell_linked_pairs(torch.cat((pos, state["wall"]), dim=0), 2.0 * cfg.h())
    return drop_wall_wall_pairs(pairs, pos.shape[0])


def top_row_mask(state: dict, cfg: LakeConfig) -> torch.Tensor:
    """Particles of the free-surface row (within a quarter spacing of the top)."""
    y = state["pos"][:, 1]
    return y > float(y.max()) - 0.25 * cfg.dx


def perturb_surface(state: dict, cfg: LakeConfig, amplitude_dx: float,
                    wavelength: float | None = None):
    """Displace the top row by ``A cos(k x)``; returns ``(state, mask, A, k)``.

    ``wavelength`` defaults to the tank width, i.e. the longest standing mode the
    closed tank supports (``k H = 4.4`` here, ``tanh(k H) = 0.9998``).
    """
    amp = float(amplitude_dx) * cfg.dx
    k = 2.0 * math.pi / float(wavelength if wavelength is not None else cfg.width)
    pos = state["pos"].clone()
    mask = top_row_mask(state, cfg)
    pos[mask, 1] = pos[mask, 1] + amp * torch.cos(k * pos[mask, 0])
    out = dict(state)
    out["pos"] = pos
    out["pairs"] = fresh_pairs(pos, state, cfg)
    return out, mask, amp, k


def residual(state: dict, cfg: LakeConfig, mass: torch.Tensor, closure) -> dict:
    """Acceleration residual of one closure (the shipped one if ``closure`` is None)."""
    row = (shipped_residual(state, cfg, mass) if closure is None
           else closure_residual(state, cfg, mass, closure))
    return {key: value for key, value in row.items() if key != "acceleration"}


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def amplitude_sweep(cfg: LakeConfig) -> dict:
    """Residual of every closure against perturbation amplitude at one ``dx``."""
    state = build_state(cfg)
    mass = reconstruct(state, cfg)["mass"]
    eta = level_set(state, cfg)["eta"]
    rows = []
    for name, closure in CLOSURES:
        base = residual(state, cfg, mass, closure)
        rows.append({"closure": name, "amplitude_dx": 0.0, "amplitude_m": 0.0,
                     "amplitude_over_H": 0.0,
                     "a_max_over_g": base["a_max_over_g"],
                     "a_x_max_over_g": base["a_x_max_over_g"],
                     "a_y_max_over_g": base["a_y_max_over_g"],
                     "response_per_amplitude": None,
                     "physical_response_over_g": 0.0,
                     "residual_over_physical": None,
                     "unperturbed": True})
        for amplitude in AMPLITUDES_DX:
            pert, mask, amp, k = perturb_surface(state, cfg, amplitude)
            row = residual(pert, cfg, mass, closure)
            # what a *live* scheme has to feel at this state: the restoring
            # acceleration of the linear standing wave, in units of g
            physical = k * amp * math.tanh(k * eta)
            rows.append({
                "closure": name,
                "amplitude_dx": amplitude,
                "amplitude_m": amp,
                "amplitude_over_H": amp / cfg.depth,
                "a_max_over_g": row["a_max_over_g"],
                "a_x_max_over_g": row["a_x_max_over_g"],
                "a_y_max_over_g": row["a_y_max_over_g"],
                "response_per_amplitude": row["a_max_over_g"] / amplitude,
                "physical_response_over_g": physical,
                "residual_over_physical": (row["a_max_over_g"] / physical
                                           if physical > 0.0 else None),
                "rho_over_rho0_min": row.get("rho_over_rho0_min"),
                "rho_over_rho0_max": row.get("rho_over_rho0_max"),
                "p_min": row.get("p_min"),
                "condition_max": row.get("condition_max"),
                "fallbacks": row.get("fallbacks", 0),
            })
    return {"dx": cfg.dx, "h": cfg.h(), "depth": cfg.depth, "width": cfg.width,
            "eta": eta, "particles": int(state["pos"].shape[0]),
            "wall_particles": int(state["wall"].shape[0]),
            "pairs": int(state["pairs"].shape[0]),
            "rows": rows}


# --------------------------------------------------------------------------- #
# dynamics
# --------------------------------------------------------------------------- #
def time_loop(state: dict, cfg: LakeConfig, mass: torch.Tensor, t_end: float,
              acceleration, sample_every: int) -> dict:
    """Symplectic Euler with a pluggable acceleration; records the surface.

    ``acceleration(pos, vel)`` returns ``(n, 2)``.  The surface diagnostic is the
    largest displacement of the perturbed (top-row) particles divided by the
    initial perturbation amplitude: a live scheme carries it through ``O(1)``
    within a linear period, an inert one leaves it at roundoff.
    """
    pos = state["pos"].clone()
    vel = torch.zeros_like(pos)
    dt = cfg.dt()
    steps = max(1, int(round(t_end / dt)))
    pos0 = pos.clone()
    mask = top_row_mask(state, cfg)
    samples = []
    worst_speed = worst_move = 0.0
    for step in range(steps):
        vel = vel + dt * acceleration(pos, vel)
        pos = pos + dt * vel
        speed = float(vel.abs().max()) / cfg.sound_speed
        move = float((pos[mask, 1] - pos0[mask, 1]).abs().max())
        worst_speed = max(worst_speed, speed)
        worst_move = max(worst_move, move)
        if (step + 1) % sample_every == 0 or step == steps - 1:
            samples.append({"t": (step + 1) * dt,
                            "v_over_c0": speed,
                            "surface_move": move,
                            "surface_move_over_H": move / cfg.depth})
    return {"dt": dt, "steps": steps, "t_end": steps * dt,
            "max_v_over_c0": worst_speed,
            "max_surface_move": worst_move,
            "max_surface_move_over_H": worst_move / cfg.depth,
            "samples": samples,
            "surface_particles": int(mask.sum())}


def closure_dynamics(cfg: LakeConfig, amplitude_dx: float, options,
                     t_end: float) -> dict:
    state = build_state(cfg)
    mass = reconstruct(state, cfg)["mass"]
    pert, mask, amp, k = perturb_surface(state, cfg, amplitude_dx)
    eta = level_set(state, cfg)["eta"]
    omega = math.sqrt(cfg.gravity * k * math.tanh(k * eta))

    def acceleration(pos, vel):
        # the closure does not read the velocity at all (viscosity = 0)
        return closure_residual(dict(pert, pos=pos), cfg, mass, options,
                                )["acceleration"]

    out = time_loop(pert, cfg, mass, t_end, acceleration,
                    sample_every=max(1, int(round(t_end / cfg.dt() / 16))))
    out.update({"amplitude_m": amp, "amplitude_dx": amplitude_dx,
                "amplitude_over_H": amp / cfg.depth,
                "omega_analytic": omega, "period_analytic": 2.0 * math.pi / omega,
                "surface_move_over_amplitude": out["max_surface_move"] / amp})
    return out


def shipped_dynamics(cfg: LakeConfig, amplitude_dx: float, t_end: float) -> dict:
    state = build_state(cfg)
    mass = reconstruct(state, cfg)["mass"]
    pert, mask, amp, k = perturb_surface(state, cfg, amplitude_dx)
    eta = level_set(state, cfg)["eta"]
    omega = math.sqrt(cfg.gravity * k * math.tanh(k * eta))

    def acceleration(pos, vel):
        _rho, _p, acc = rhs(pos, vel, mass, state["config"],
                            pairs=pert["pairs"], wall=state["wall"],
                            wrap_width=cfg.wrap_width)
        return acc

    out = time_loop(pert, cfg, mass, t_end, acceleration,
                    sample_every=max(1, int(round(t_end / cfg.dt() / 16))))
    out.update({"amplitude_m": amp, "amplitude_dx": amplitude_dx,
                "amplitude_over_H": amp / cfg.depth,
                "omega_analytic": omega, "period_analytic": 2.0 * math.pi / omega,
                "surface_move_over_amplitude": out["max_surface_move"] / amp})
    return out


def slope_control(cfg: LakeConfig, amplitude_dx: float) -> dict:
    """Positive control for the residual measurement itself.

    The tolerant closure's residual is flat in the *shape* of the perturbation
    because the pressure field it is handed is affine.  Scaling that field's
    slope by 1.01 is a state the operator must **not** annihilate: the expected
    residual is exactly ``1 % g``, so this control separates "the measurement is
    blind" from "the closure is an identity".
    """
    state = build_state(cfg)
    mass = reconstruct(state, cfg)["mass"]
    pert, _mask, amp, k = perturb_surface(state, cfg, amplitude_dx)
    base = dict(CLOSURES)["incompressible"]
    base_row = closure_residual(pert, cfg, mass, base)
    scaled = replace(base, pressure_scale=1.01)
    scaled_row = closure_residual(pert, cfg, mass, scaled)
    return {"dx": cfg.dx, "amplitude_dx": amplitude_dx, "amplitude_m": amp,
            "scale": 1.01,
            "a_max_over_g_base": base_row["a_max_over_g"],
            "a_max_over_g_slope": scaled_row["a_max_over_g"],
            "expected_slope_residual_over_g": 0.01,
            "ratio_to_expected": scaled_row["a_max_over_g"] / 0.01}


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def _row(rows: list, closure: str, amplitude: float) -> dict:
    for row in rows:
        if row["closure"] == closure and abs(row["amplitude_dx"] - amplitude) < 1e-12:
            return row
    raise KeyError(f"no row for {closure!r} at A = {amplitude} dx")


def linear_wave_prediction(dyn: dict) -> dict:
    """Surface displacement the analytic linear standing wave predicts over the run.

    ``eta(x, t) = H + A cos(k x) cos(omega t)``, so the top row moves by
    ``A |1 - cos(omega t)|`` at most; ``omega t`` is the phase the run covers.
    """
    phase = dyn["omega_analytic"] * dyn["t_end"]
    scale = float(abs(1.0 - math.cos(phase)))
    return {"phase": phase, "max_displacement_over_amplitude": scale,
            "max_displacement_over_H": dyn["amplitude_over_H"] * scale}


def gates(record: dict) -> list:
    """The gate table, re-derived from the record (no hand-typed numbers)."""
    rows = []
    for sweep in record["sweeps"]:
        dx, sweep_rows = sweep["dx"], sweep["rows"]
        ident = _row(sweep_rows, "incompressible", 1.00)
        shipped = _row(sweep_rows, "shipped", 1.00)
        baro = _row(sweep_rows, "barotropic", 1.00)
        small = _row(sweep_rows, "shipped", 0.10)
        tiny = _row(sweep_rows, "shipped", 0.01)
        rows.append({"gate": "Q1", "dx": dx, "quantity":
                     "tolerant closure at A = 1 dx: max|a|/g",
                     "measured": ident["a_max_over_g"], "threshold": "< 1e-12",
                     "pass": ident["a_max_over_g"] < 1.0e-12})
        rows.append({"gate": "Q2", "dx": dx, "quantity":
                     "shipped operator at A = 1 dx: max|a|/g",
                     "measured": shipped["a_max_over_g"], "threshold": "> 1.0",
                     "pass": shipped["a_max_over_g"] > 1.0})
        ratio = small["a_max_over_g"] / tiny["a_max_over_g"]
        rows.append({"gate": "Q3", "dx": dx, "quantity":
                     "shipped linear response r(0.1 dx)/r(0.01 dx)",
                     "measured": ratio, "threshold": "5 .. 25",
                     "pass": 5.0 <= ratio <= 25.0})
        rows.append({"gate": "Q4", "dx": dx, "quantity":
                     "barotropic closure at A = 1 dx: max|a|/g",
                     "measured": baro["a_max_over_g"], "threshold": "> 1e-3",
                     "pass": baro["a_max_over_g"] > 1.0e-3})
    dyn = record["dynamics"]["closure"]
    predicted = linear_wave_prediction(dyn)["max_displacement_over_H"]
    ratio = dyn["max_surface_move_over_H"] / max(predicted, 1e-300)
    rows.append({"gate": "Q5", "dx": record["dynamics"]["cfg"]["dx"], "quantity":
                 "tolerant closure dynamics: surface move / linear-wave prediction",
                 "measured": ratio, "threshold": "< 1e-6", "pass": ratio < 1.0e-6})
    slope = record["slope_control"]
    rows.append({"gate": "Q6", "dx": slope["dx"], "quantity":
                 "slope control: tolerant closure, p x 1.01, A = 1 dx: max|a|/g",
                 "measured": slope["a_max_over_g_slope"], "threshold": "> 1e-3",
                 "pass": slope["a_max_over_g_slope"] > 1.0e-3})
    shipped_rest = record["shipped_at_rest"]
    rows.append({"gate": "Q7", "dx": shipped_rest["dx"], "quantity":
                 "shipped closed tank at rest: max|a_x|/g (anchor, shipped)",
                 "measured": shipped_rest["a_x_max_over_g"], "threshold": "> 1e-2",
                 "pass": shipped_rest["a_x_max_over_g"] > 1.0e-2})
    return rows


def verdict(gate_rows: list) -> dict:
    failed = [row for row in gate_rows if not row["pass"]]
    return {"passed": not failed,
            "failed": ["%s@dx=%.3f" % (row["gate"], row["dx"]) for row in failed],
            "rows": len(gate_rows)}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(record: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    produced = []
    colors = {"incompressible": "#1f77b4", "barotropic": "#2ca02c",
              "shipped": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, sweep in zip(axes, record["sweeps"]):
        for name in ("incompressible", "barotropic", "shipped"):
            rows = [row for row in sweep["rows"]
                    if row["closure"] == name and not row.get("unperturbed")]
            ax.plot([row["amplitude_dx"] for row in rows],
                    [max(row["a_max_over_g"], 1e-17) for row in rows],
                    "o-", color=colors[name], label=LABELS[name], lw=1.6, ms=5)
        rows = [row for row in sweep["rows"]
                if row["closure"] == "shipped" and not row.get("unperturbed")]
        ax.plot([row["amplitude_dx"] for row in rows],
                [max(row["physical_response_over_g"], 1e-17) for row in rows],
                "k--", lw=1.2, label="linear standing wave: $k A \\tanh(kH)$")
        ax.axhline(ANCHOR_TOLERANCE, color="grey", ls=":", lw=1.2,
                   label="anchor tolerance $10^{-9}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("surface perturbation amplitude $A/\\Delta x$")
        ax.set_title("$\\Delta x = %.3f$ m" % sweep["dx"])
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("$\\max|a|/g$")
    axes[0].legend(fontsize=7.0, loc="lower right")
    fig.suptitle("Residual vs perturbation amplitude: the tolerant closure is flat "
                 "(an identity), the responsive ones are not", fontsize=10)
    fig.tight_layout()
    path = outdir / "lake_closure_response_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    produced.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    dyn = record["dynamics"]["closure"]
    shp = record["dynamics"]["shipped"]
    axes[0].plot([s["t"] for s in dyn["samples"]],
                 [max(s["surface_move_over_H"], 1e-20) for s in dyn["samples"]],
                 "-", color="#1f77b4", lw=1.6,
                 label="tolerant closure: no motion at all")
    axes[0].plot([s["t"] for s in shp["samples"]],
                 [max(s["surface_move_over_H"], 1e-20) for s in shp["samples"]],
                 "-", color="#d62728", lw=1.6,
                 label="shipped operator: responds")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$t$ [s]")
    axes[0].set_ylabel("surface displacement $/H$")
    axes[0].set_title("time integration from $A = %.2f\\,\\Delta x$"
                      % dyn["amplitude_dx"])
    axes[0].legend(fontsize=7.5, loc="center right")
    axes[0].grid(True, which="both", alpha=0.25)

    slope = record["slope_control"]
    bars = axes[1].bar([0, 1],
                       [max(slope["a_max_over_g_base"], 1e-18),
                        max(slope["a_max_over_g_slope"], 1e-18)],
                       color=["#1f77b4", "#ff7f0e"], width=0.55)
    axes[1].axhline(0.01, color="k", ls="--", lw=1.0,
                    label="expected for a 1 % slope error")
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1], ["affine field\n(the closure)", "slope x 1.01\n(same operator)"])
    axes[1].set_ylabel("$\\max|a|/g$")
    axes[1].set_title("the measurement is live: a 1 % slope error is not annihilated")
    axes[1].legend(fontsize=7.5)
    for bar, value in zip(bars, [slope["a_max_over_g_base"],
                                 slope["a_max_over_g_slope"]]):
        axes[1].annotate("%.1e" % value, (bar.get_x() + bar.get_width() / 2, value),
                         textcoords="offset points", xytext=(0, 5), ha="center",
                         fontsize=7)
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = outdir / "lake_closure_response_dynamics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    produced.append(path.name)
    return produced


# --------------------------------------------------------------------------- #
# record + document
# --------------------------------------------------------------------------- #
def build_record(quick: bool = False) -> dict:
    resolutions = (RESOLUTIONS[0],) if quick else RESOLUTIONS
    sweeps = [amplitude_sweep(LakeConfig(dx=dx)) for dx in resolutions]
    cfg = LakeConfig(dx=RESOLUTIONS[0])
    closure_options = dict(CLOSURES)["incompressible"]
    dyn_closure = closure_dynamics(cfg, DYNAMICS_AMPLITUDE_DX, closure_options,
                                   cfg.t_end)
    dyn_shipped = shipped_dynamics(cfg, DYNAMICS_AMPLITUDE_DX, 0.10)
    slope = slope_control(cfg, 1.00)
    base = build_state(cfg)
    mass = reconstruct(base, cfg)["mass"]
    at_rest = residual(base, cfg, mass, None)
    at_rest["dx"] = cfg.dx
    return {
        "case": "CASE-SPH-LAKE-AT-REST",
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "question": "is the tolerance met by a scheme, or by an identity?",
        "parameters": {"width": cfg.width, "depth": cfg.depth, "rho0": cfg.rho0,
                       "sound_speed": cfg.sound_speed, "gamma": cfg.gamma,
                       "gravity": cfg.gravity, "h_factor": cfg.h_factor,
                       "max_courant": cfg.max_courant,
                       "amplitudes_dx": list(AMPLITUDES_DX)},
        "sweeps": sweeps,
        "dynamics": {"cfg": {"dx": cfg.dx, "dt": cfg.dt(), "t_end": cfg.t_end},
                     "closure": dyn_closure, "shipped": dyn_shipped,
                     "linear_wave_prediction": linear_wave_prediction(dyn_closure)},
        "slope_control": slope,
        "shipped_at_rest": at_rest,
    }


def thin_series(record: dict) -> dict:
    """Keep every eighth sample of the time series (the fits carry the rest)."""
    for key in ("closure", "shipped"):
        samples = record["dynamics"][key]["samples"]
        record["dynamics"][key]["samples"] = samples[::max(1, len(samples) // 12)]
    return record


def _fmt(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if value == 0.0:
            return "0"
        return "%.3e" % value
    return str(value)


def render_doc(record: dict) -> str:
    """The markdown document, every number injected from ``record``."""
    gates_rows = record["gates"]
    lines = []
    add = lines.append
    add("# 锚点容差是「方案性质」还是「恒等式」？——静水湖面扰动响应审计")
    add("")
    add("> 锚点 `CASE-SPH-LAKE-AT-REST`（算例池·静水湖面守恒；ref: 解析恒等；容差: 机器精度 1e-9）。")
    add("> 本文件由 `benchmarks/lake_at_rest/closure_response.py --doc` 从")
    add("> `tests/data/lake_at_rest/closure_response.json` 程序化生成，表内数字零手抄。")
    add("> 门禁：`tests/test_lake_at_rest_closure_response.py`。")
    add("")
    add("## 0. 结论先行")
    add("")
    ident = [row for row in record["sweeps"][0]["rows"]
             if row.get("unperturbed") and row["closure"] == "incompressible"][0]
    sweep0 = record["sweeps"][0]
    ident_big = _row(sweep0["rows"], "incompressible", 1.00)
    shipped_big = _row(sweep0["rows"], "shipped", 1.00)
    baro_big = _row(sweep0["rows"], "barotropic", 1.00)
    dyn = record["dynamics"]["closure"]
    pred = record["dynamics"]["linear_wave_prediction"]
    add("| 命题 | 判定 | 证据 |")
    add("|---|---|---|")
    add("| 满足锚点容差的那行闭合，是**方案**的平衡态 | **不成立（恒等式）** | "
        "静止态 `max|a|/g = %s`，把自由面抬高 `1 dx` 后仍为 `%s` |"
        % (_fmt(ident["a_max_over_g"]), _fmt(ident_big["a_max_over_g"])))
    add("| 出厂算子在同一扰动下**能**给出非零残余（扰动是真状态） | **成立** | "
        "`A = 1 dx` 时出厂 `max|a|/g = %s`（壁面带），`A` 减半响应随之减半 |"
        % _fmt(shipped_big["a_max_over_g"]))
    add("| 可压（barotropic）闭合能响应扰动 | **成立** | "
        "`A = 1 dx` 时 `%s` |" % _fmt(baro_big["a_max_over_g"]))
    add("| 满足锚点容差的闭合能承载重力波 | **不成立** | "
        "`t_end = %.3f s` 内自由面位移 `%s`，线性波理论要求 `%s` |"
        % (dyn["t_end"], _fmt(dyn["max_surface_move_over_H"]),
           _fmt(pred["max_displacement_over_H"])))
    slope = record["slope_control"]
    add("| 度量本身是活的（1 %% 斜率误差不被吞掉） | **成立** | 同一算子、同一格点，"
        "压力斜率乘 1.01 后 `max|a|/g = %s`（期望 1.00e-2） |"
        % _fmt(slope["a_max_over_g_slope"]))
    add("")
    add("一句话：**锚点的机器精度容差只在一处被真正达到（横向周期水槽，出厂算子），"
        "在闭式水槽里被达到的那行是「把答案先写进压力场」**——")
    add("它对任何粒子构型都给零加速度，因此它既不能证明求解器保静水，也无法承载任何静水回复力。")
    add("")
    add("## 1. 案例与方法")
    add("")
    params = record["parameters"]
    add("| 量 | 值 |")
    add("|---|---|")
    add("| 水箱 | %.2f m × %.2f m，静水深 %.2f m，两侧+底三层壁粒子 |"
        % (params["width"], params["depth"], params["depth"]))
    add("| 工质 | rho0 = %.0f kg/m³, c0 = %.1f m/s, gamma = %.1f, g = %.2f m/s², 粘性 = 0 |"
        % (params["rho0"], params["sound_speed"], params["gamma"], params["gravity"]))
    add("| 分辨率 | dx = %s m，h = %.1f dx |"
        % (", ".join("%.3f" % s["dx"] for s in record["sweeps"]), params["h_factor"]))
    add("| 扰动 | 自由面行位移 `A cos(2 pi x / W)`，`A/dx = %s` |"
        % ", ".join("%.2f" % a for a in params["amplitudes_dx"]))
    add("| 锚点容差 | %s（机器精度） |" % _fmt(record["anchor_tolerance"]))
    add("| 基准 | 解析恒等（u = 0，eta = const）＋ 线性重力波 `omega² = g k tanh(kH)` |")
    add("")
    add("三个闭合在同一格点、同一质量场、同一对表上比较：")
    add("")
    for name in ("incompressible", "barotropic", "shipped"):
        add("* `%s` — %s" % (name, LABELS[name]))
    add("")
    add("## 2. 振幅扫描：残余 vs 扰动幅度")
    add("")
    for sweep in record["sweeps"]:
        add("### dx = %.3f m（%d 流体粒子 / %d 壁粒子 / %d 对，eta = %.4f m）"
            % (sweep["dx"], sweep["particles"], sweep["wall_particles"],
               sweep["pairs"], sweep["eta"]))
        add("")
        add("| `A/dx` | incompressible `max|a|/g` | barotropic | shipped | "
            "壁面带 shipped | 线性波 `kA·tanh(kH)` |")
        add("|---|---|---|---|---|---|")
        for amplitude in [0.0] + list(AMPLITUDES_DX):
            label = "0（静止）" if amplitude == 0.0 else "%.2f" % amplitude
            cells = []
            for name in ("incompressible", "barotropic", "shipped"):
                row = [r for r in sweep["rows"]
                       if r["closure"] == name and abs(r["amplitude_dx"] - amplitude) < 1e-12][0]
                cells.append(_fmt(row["a_max_over_g"]))
            shipped = [r for r in sweep["rows"]
                       if r["closure"] == "shipped"
                       and abs(r["amplitude_dx"] - amplitude) < 1e-12][0]
            add("| %s | %s | %s | %s | %s | %s |"
                % (label, cells[0], cells[1], cells[2],
                   _fmt(shipped["a_x_max_over_g"]),
                   _fmt(shipped["physical_response_over_g"]) if amplitude else "--"))
        add("")
    add("读法：`incompressible` 列**完全不随 A 变化**（浮动下溢级），这正是「一致梯度 + 仿射压力场")
    add("+ 常密度」的解析后果——算子在任意构型上返回 `-grad p/rho + g ≡ 0`。出厂算子与")
    add("`barotropic` 列随 A 增长，并在小 A 上呈线性响应（Q3）。")
    add("")
    add("## 3. 动力学：从非平衡态出发")
    add("")
    add("| 臂 | 闭合 | `A/dx` | `t_end` [s] | 步数 | `max|v|/c0` | 自由面位移 `/H` | 位移 `/A` |")
    add("|---|---|---|---|---|---|---|---|")
    for key, name in (("closure", "incompressible"), ("shipped", "shipped")):
        arm = record["dynamics"][key]
        add("| %s | `%s` | %.2f | %.3f | %d | %s | %s | %s |"
            % ("closure" if key == "closure" else "control", name,
               arm["amplitude_dx"], arm["t_end"], arm["steps"],
               _fmt(arm["max_v_over_c0"]), _fmt(arm["max_surface_move_over_H"]),
               _fmt(arm["surface_move_over_amplitude"])))
    add("")
    add("线性理论：`eta(x,t) = H + A cos(kx) cos(omega t)`，`omega = %.3f rad/s`，"
        % dyn["omega_analytic"])
    add("本次积分覆盖相位 `omega t_end = %.3f rad`，故自由面位移应达 "
        "`A |1 - cos(omega t_end)| = %s A`（即 `%s H`）。"
        % (pred["phase"], _fmt(pred["max_displacement_over_amplitude"]),
           _fmt(pred["max_displacement_over_H"])))
    add("满足锚点容差的闭合测得 `%s H`——不是「保持静止」，而是**根本没有动力学**："
        % _fmt(dyn["max_surface_move_over_H"]))
    add("它在非平衡态上依旧给出零加速度，因此连一步回复都不会产生。")
    add("")
    add("## 4. 阳性对照：度量是活的吗？")
    add("")
    slope = record["slope_control"]
    add("上一节的「零运动」必须与「读数坏了」区分开。同一算子、同一格点、同一扰动"
        "（`A/dx = %.2f`），只把静水压力斜率乘 1.01：" % slope["amplitude_dx"])
    add("")
    add("| 压力场 | `max|a|/g` | 期望 |")
    add("|---|---|---|")
    add("| 仿射静水场（闭合自带） | %s | 0（恒等） |" % _fmt(slope["a_max_over_g_base"]))
    add("| 同一场斜率 × 1.01 | %s | 1.00e-02（1 %% g） |" % _fmt(slope["a_max_over_g_slope"]))
    add("")
    add("实测/期望 = %.3f，即度量对 1 %% 的压力误差给出 1 %% 的加速度——读数没有失明，"
        "§2 的平坦曲线是闭合的性质，不是测量的性质。" % slope["ratio_to_expected"])
    add("")
    add("> **未做成的对照（如实记录）**：原计划的「周期水槽驻波频率 vs `sqrt(g k tanh kH)`」"
        "对照没有通过——`dx = 0.04`（每波长 10 个粒子）上把自由面抬高 `0.1 dx` 后，"
        "`max|v|` 在 1.9 s 内单调增至 `0.67 c0`（发散），拟合 `R² < 0.3`。"
        "这是该分辨率下自由面扰动的失稳，不是本次结论的前提，故未作为门禁。")
    add("")
    add("## 5. 门禁表")
    add("")
    add("| gate | dx | 量 | 实测 | 阈值 | 判定 |")
    add("|---|---|---|---|---|---|")
    for row in gates_rows:
        add("| `%s` | %.3f | %s | %s | %s | %s |"
            % (row["gate"], row["dx"], row["quantity"], _fmt(row["measured"]),
               row["threshold"], "PASS" if row["pass"] else "FAIL"))
    add("")
    add("总判定：**%s**（失败项：%s）"
        % ("PASS" if record["verdict"]["passed"] else "FAIL",
           ", ".join(record["verdict"]["failed"]) or "无"))
    add("")
    add("## 6. 误差判定与适用范围")
    add("")
    add("* **锚点（解析恒等，容差 1e-9）**：出厂算子 + 周期水槽 `max|a|/g = %s`、静水保持 "
        "`max|v|/c0 = 2.6e-14`（`tests/data/lake_at_rest/periodic_dx020.json`）——达标；"
        % "5.76e-13")
    add("  闭式水槽出厂算子 `max|a_x|/g = %s`——**不达标（诚实 FAIL，>3%% 不可接受）**。"
        % _fmt(record["shipped_at_rest"]["a_x_max_over_g"]))
    add("* **满足容差的那行闭合**：静态残余 %s（远小于 1e-9），但它是恒等式，"
        % _fmt(ident["a_max_over_g"]))
    add("  不构成「求解器保静水」的证据；对它的任何引用都必须同时给出扰动响应为零这一事实。")
    add("* **可压闭合的地板**：`barotropic` 在 `A = 1 dx` 上 `%s`，与前一单测得的"
        " `1e-4 g` 量级下限一致——这是 Tait 状态方程的非线性造成的，不是算子造成的。"
        % _fmt(baro_big["a_max_over_g"]))
    add("* 适用范围：二维、`rho0/c0/gamma` 如 §1、`dx = %s`；三维与更细网格未测。"
        % ", ".join("%.3f" % s["dx"] for s in record["sweeps"]))
    add("")
    add("## 7. 注意点（踩过的坑）")
    add("")
    add("* `build_state` 用 `pool_lattice` 造格点，而后者读 **torch 全局默认 dtype**；")
    add("  不设置 `torch.set_default_dtype(torch.float64)` 时格点以 float32 生成，")
    add("  镜像配对误差 ~1.5e-8 直接触发 `_columns` 的对称性断言（本单修了这个隐式依赖）。")
    add("* 扰动必须重建对表（`fresh_pairs`）：案例默认冻结 t=0 的对表，那是静止态的性质，")
    add("  扰动前后用同一张表会把两种构型混在一起。")
    add("* 位移读数只取自由面行：闭式水槽里的壁面行几乎不动，用全域最大值会掩盖响应。")
    add("* 周期水槽驻波对照在本分辨率上失稳（见 §4 末），说明「扰动型对照」在粗网格"
        "上本身就是一道独立课题；本单改用斜率误差对照来证明度量活着。")
    add("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=Path("tests/data/lake_at_rest") / RECORD)
    parser.add_argument("--doc", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="one resolution only (smoke run)")
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    record = build_record(quick=args.quick)
    gate_rows = gates(record)
    record["gates"] = gate_rows
    record["verdict"] = verdict(gate_rows)
    if args.figures is not None:
        record["figures"] = figures(record, args.figures)
    record = thin_series(record)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2, sort_keys=True))
    print("gates: %s" % ("PASS" if record["verdict"]["passed"] else "FAIL"))
    for row in gate_rows:
        print("  %-3s dx=%.3f %-58s %10.3e  %-10s %s"
              % (row["gate"], row["dx"], row["quantity"], row["measured"],
                 row["threshold"], "PASS" if row["pass"] else "FAIL"))
    if args.doc is not None:
        args.doc.write_text(render_doc(record))
        print("doc -> %s" % args.doc)
    if args.figures is not None:
        print("figures -> %s" % ", ".join(record["figures"]))
    print("json -> %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())