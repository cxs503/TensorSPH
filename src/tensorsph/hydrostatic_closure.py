"""Hydrostatic closure of the lake-at-rest residual: is the anchor tolerance reachable?

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` asks for a machine-precision statement:
with zero initial velocity and a flat free surface, nothing may move
(``ref``: the analytic hydrostatic identity, ``tol``: machine precision).  The
shipped configuration misses it by eleven orders and
``docs/verification_lake_at_rest_wall_gap.md`` / ``..._wellbalanced.md`` narrowed
the cause to the *pressure field*: the Tait law is fed by a truncated summation
density, ``max(p, 0)`` puts a kink at the free surface, and every linear-exact
gradient estimator amplifies it.

This module does not re-diagnose that.  It asks the question the diagnosis left
open, and answers it by measurement:

    Which closure -- (pressure law) x (wall pressure) x (gradient operator) --
    puts the closed tank at rest to the anchor tolerance, and what does that
    closure cost?

Three ingredients, each a named, standard choice, never a fitted knob:

``pressure``
    ``eos``            the shipped route: Tait law of the summation density,
                       floored at zero (this is what ``droplet_impact.rhs`` does);
    ``incompressible`` the well-balanced hydrostatic reconstruction: the still-water
                       level set ``eta`` is read off the *particle data*
                       (:func:`level_set`), and ``p = rho0 g (eta - y)``,
                       ``rho = rho0`` -- affine in ``y``, no clamp, no kink.
                       This is the closure a well-balanced finite-volume scheme
                       prescribes (Audusse et al. 2004; Xing & Shu 2005), read in
                       SPH terms; it replaces the EOS by a hydrostatic pressure.
    ``barotropic``     the exact *continuous* barotropic equilibrium of the Tait
                       EOS (``eos_hydrostatic_density``) with its own pressure --
                       the physically consistent compressible reference, and the
                       closure in which the anchor tolerance is *not* attainable
                       (see ``docs/verification_lake_at_rest_hydrostatic_closure.md``).

``wall``
    ``shipped_adami``  Adami's weighted extrapolation from the crossing pairs
                       (what ``wellbalanced.build_fields`` does, clamp included);
    ``continuation``   the hydrostatic continuation of the level-set pressure onto
                       the wall particles, ``p_w = rho0 g (eta - y_w)``, unclamped;
    ``continuation_clamped`` the same, floored at zero -- the control that
                       re-introduces the free-surface kink on the wall side.

``formulation``
    the three operators of :mod:`tensorsph.wellbalanced`: ``symmetric`` (the
    shipped pair form), ``difference`` and ``moment`` (first-order consistent).

Nothing here is case-specific except the geometry it is handed; the level set is
the only case-level input and it is measured, not injected.

Determinism: no random numbers, no fitted constants, no time stepping.
"""
from __future__ import annotations

import dataclasses

import torch

from tensorsph.droplet_impact import (tait_pressure)
from tensorsph.eos_hydrostatics import eos_hydrostatic_density
from tensorsph.lake_at_rest import _evaluate
from tensorsph.wellbalanced import (WellBalancedOptions, build_fields,
                                    pressure_acceleration)

PRESSURE_CLOSURES = ("eos", "incompressible", "barotropic")
WALL_CLOSURES = ("shipped_adami", "continuation", "continuation_clamped")
FORMULATIONS = ("symmetric", "difference", "moment")


@dataclasses.dataclass
class HydrostaticClosureOptions:
    """One point of the closure space.

    ``regularization`` is the Tikhonov term of the moment inverse.  It is a
    *measured* quantity here, not a tuning knob: the residual of the consistent
    operator is linear in it (``docs/verification_lake_at_rest_hydrostatic_closure.md``
    §5), so reporting the residual at a declared regularisation and the ladder
    that produced it keeps the number from being cherry-picked.
    """

    pressure: str = "incompressible"
    wall: str = "continuation"
    formulation: str = "moment"
    regularization: float = 0.0
    max_condition: float = 1.0e30
    #: control knobs, zero/one for every measurement of the sweep itself
    eta_shift: float = 0.0      # metres added to the reconstructed level set
    pressure_scale: float = 1.0  # factor on the hydrostatic slope

    def __post_init__(self) -> None:
        for name, value, allowed in (
                ("pressure", self.pressure, PRESSURE_CLOSURES),
                ("wall", self.wall, WALL_CLOSURES),
                ("formulation", self.formulation, FORMULATIONS)):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; "
                                 f"expected one of {allowed}")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")

    def wb_options(self) -> WellBalancedOptions:
        return WellBalancedOptions(formulation=self.formulation,
                                   regularization=self.regularization,
                                   max_condition=self.max_condition)


# --------------------------------------------------------------------------- #
# level set read off the particle data
# --------------------------------------------------------------------------- #
def level_set(state, cfg) -> dict:
    """Still-water level of the particle set, and how flat it is.

    ``eta`` is the top row of fluid particles plus half a spacing -- the surface
    the lattice actually represents.  It is *read from the data*: no depth is
    injected, and at a resolution where ``depth / dx`` is not an integer the
    lattice is shorter than the tank (``dx = 0.04`` gives ``eta = 0.28``), which
    is reported rather than repaired.  ``roughness`` is the spread of the top-row
    heights, i.e. the measured flatness of the initial surface.
    """
    pos = state["pos"]
    y = pos[:, 1]
    top = float(y.max())
    eta = top + 0.5 * float(cfg.dx)
    near_top = y > top - 0.25 * float(cfg.dx)
    roughness = float(y[near_top].max() - y[near_top].min())
    return {"eta": eta, "top_row_y": top, "roughness": roughness,
            "top_row_particles": int(near_top.sum()),
            "depth_reference": float(cfg.depth),
            "eta_over_depth": eta / float(cfg.depth)}


# --------------------------------------------------------------------------- #
# the closure: pressure and density fields
# --------------------------------------------------------------------------- #
def closure_fields(state, cfg, mass: torch.Tensor,
                   options: HydrostaticClosureOptions | None = None) -> dict:
    """Fields of :func:`wellbalanced.build_fields` with the pressure law replaced.

    Only the pressure/density arrays are touched; the pair table, the masses and
    the operator are the shipped ones, so any difference between two rows of the
    closure table is the closure and nothing else.
    """
    options = options or HydrostaticClosureOptions()
    f = build_fields(state, cfg, mass, options.wb_options())
    n, nw = f["n"], f["nw"]
    pos, wall = f["pos"], f["wall"]
    level = level_set(state, cfg)
    eta = level["eta"] + options.eta_shift
    scale = options.pressure_scale
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma

    if options.pressure == "eos":                     # shipped route, untouched
        p_fluid = f["pressure"][:n]
        rho_fluid = f["rho"][:n]
    elif options.pressure == "incompressible":        # well-balanced reconstruction
        p_fluid = scale * cfg.rho0 * cfg.gravity * (eta - pos[:, 1])
        rho_fluid = torch.full_like(p_fluid, cfg.rho0)
    elif options.pressure == "barotropic":            # exact continuous equilibrium
        rho_fluid = eos_hydrostatic_density(
            pos[:, 1], depth=eta, rho0=cfg.rho0, sound_speed=cfg.sound_speed,
            gamma=cfg.gamma, gravity=cfg.gravity)
        p_fluid = bulk * ((rho_fluid / cfg.rho0) ** cfg.gamma - 1.0)
    else:                                             # pragma: no cover
        raise AssertionError(options.pressure)

    if options.wall == "shipped_adami" and options.pressure == "eos":
        # the shipped closure verbatim (clamps included): taking it from
        # ``build_fields`` keeps the operator cross-check an *implementation*
        # check instead of a second derivation of the same formula.
        p_wall, rho_wall = f["p_wall"], f["rho_full"][n:]
    elif options.wall == "shipped_adami":
        # Adami's weighted extrapolation is a *closure*, so it is evaluated from
        # the fluid field this closure just built: pairing the barotropic fluid
        # pressure with the wall pressure of the EOS field would be a mixture of
        # two closures, not a variant.
        p_wall = adami_wall(f, cfg, p_fluid, rho_fluid)
        rho_wall = (cfg.rho0 * (p_wall / bulk + 1.0).pow(1.0 / cfg.gamma)
                    ).clamp_min(0.5 * cfg.rho0)
    elif options.wall == "continuation":
        p_wall = scale * cfg.rho0 * cfg.gravity * (eta - wall[:, 1])
        rho_wall = (cfg.rho0 * (p_wall / bulk + 1.0).pow(1.0 / cfg.gamma)).clamp_min(0.5 * cfg.rho0)
    elif options.wall == "continuation_clamped":
        p_wall = (scale * cfg.rho0 * cfg.gravity * (eta - wall[:, 1])).clamp_min(0.0)
        rho_wall = cfg.rho0 * (p_wall / bulk + 1.0).pow(1.0 / cfg.gamma)
    else:                                             # pragma: no cover
        raise AssertionError(options.wall)

    out = dict(f)
    out["pressure"] = p_fluid
    out["rho"] = rho_fluid
    out["p_wall"] = p_wall
    out["rho_wall"] = rho_wall
    out["p_full"] = torch.cat((p_fluid, p_wall))
    out["rho_full"] = torch.cat((rho_fluid, rho_wall))
    out["level_set"] = level
    out["options_closure"] = options
    return out


def adami_wall(f: dict, cfg, p_fluid: torch.Tensor,
               rho_fluid: torch.Tensor) -> torch.Tensor:
    """Adami's weighted wall-face pressure from the crossing pairs, unclamped.

    ``p_w = sum w (p_f + rho_f g (y_w - y_f)) / sum w`` over the fluid particles a
    wall particle interacts with -- the shipped extrapolation, with the fluid
    field handed in instead of read from the EOS.
    """
    n, nw = f["n"], f["nw"]
    i, j, wij = f["i"], f["j"], f["wij"]
    i_fluid = i < n
    crossing = i_fluid != (j < n)
    fluid_idx = torch.where(i_fluid, i, j)
    wall_idx = (torch.where(i_fluid, j, i) - n).clamp_(0, nw - 1)
    delta = f["all_pos"][i] - f["all_pos"][j]
    dy_wall = torch.where(i_fluid, -delta[:, 1], delta[:, 1])
    weight = torch.where(crossing, wij, torch.zeros_like(wij))
    source = p_fluid[fluid_idx] + rho_fluid[fluid_idx] * (-cfg.gravity * dy_wall)
    num = torch.zeros(nw, dtype=p_fluid.dtype, device=p_fluid.device).index_add(
        0, wall_idx, source * weight)
    den = torch.zeros(nw, dtype=p_fluid.dtype, device=p_fluid.device).index_add(
        0, wall_idx, weight)
    return num / den.clamp_min(1e-30)


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #
def _bands(cfg, pos):
    band = 2.0 * cfg.h()
    wall = (pos[:, 0] < band) | (pos[:, 0] > cfg.width - band)
    interior = (pos[:, 1] > band) & (pos[:, 1] < cfg.depth - band)
    return wall, interior


def closure_residual(state, cfg, mass: torch.Tensor,
                     options: HydrostaticClosureOptions | None = None,
                     fields: dict | None = None) -> dict:
    """Acceleration residual of one closure, split by component and by band.

    ``a_max_over_g`` is the quantity the anchor's tolerance is written in.  The
    remaining entries are the diagnostics that decide whether a row is
    *admissible*: an equilibrium bought by inverting a singular moment matrix, by
    a pressure field that is negative, or by a state that no longer matches its own
    summation density must say so.
    """
    options = options or HydrostaticClosureOptions()
    f = fields if fields is not None else closure_fields(state, cfg, mass, options)
    diag = pressure_acceleration(state, cfg, mass, options.wb_options(), fields=f)
    acc = diag["acceleration"]
    pos = state["pos"]
    mag = torch.linalg.vector_norm(acc, dim=1) / cfg.gravity
    vertical = acc[:, 1].abs() / cfg.gravity
    horizontal = acc[:, 0].abs() / cfg.gravity
    wall_band, interior = _bands(cfg, pos)
    weight = float(mass.sum() * cfg.gravity)
    p, rho = f["pressure"], f["rho"]
    return {
        "pressure": options.pressure, "wall": options.wall,
        "formulation": options.formulation,
        "regularization": float(options.regularization),
        "a_max_over_g": float(mag.max()),
        "a_median_over_g": float(mag.median()),
        "a_x_max_over_g": float(horizontal.max()),
        "a_y_max_over_g": float(vertical.max()),
        "a_max_over_g_wall_band": float(mag[wall_band].max()) if bool(wall_band.any()) else None,
        "a_max_over_g_interior": float(mag[interior].max()) if bool(interior.any()) else None,
        "net_horizontal_force_over_weight": float(
            (mass * acc[:, 0]).sum().abs() / max(weight, 1e-300)),
        "net_vertical_force_over_weight": float(
            (mass * acc[:, 1]).sum().abs() / max(weight, 1e-300)),
        "condition_max": diag.get("condition_max"),
        "fallbacks": int(diag.get("fallbacks", 0)),
        "particles": int(pos.shape[0]),
        "p_min": float(p.min()), "p_max": float(p.max()),
        "rho_over_rho0_min": float((rho / cfg.rho0).min()),
        "rho_over_rho0_max": float((rho / cfg.rho0).max()),
        "worst_particle": {"x": float(pos[int(mag.argmax()), 0]),
                           "y": float(pos[int(mag.argmax()), 1])},
        "acceleration": acc,
    }


def shipped_residual(state, cfg, mass: torch.Tensor) -> dict:
    """The shipped operator + shipped EOS closure: the control every row is read against."""
    _, pressure, acc = _evaluate(state, cfg, mass)
    pos = state["pos"]
    mag = torch.linalg.vector_norm(acc, dim=1) / cfg.gravity
    return {"label": "shipped_rhs",
            "pressure": "eos", "wall": "shipped_adami", "formulation": "symmetric",
            "a_max_over_g": float(mag.max()),
            "a_x_max_over_g": float((acc[:, 0].abs() / cfg.gravity).max()),
            "a_y_max_over_g": float((acc[:, 1].abs() / cfg.gravity).max()),
            "p_min": float(pressure.min()), "p_max": float(pressure.max()),
            "particles": int(pos.shape[0])}


# --------------------------------------------------------------------------- #
# the sweep the verdict is read from
# --------------------------------------------------------------------------- #
def closure_matrix(state, cfg, mass: torch.Tensor,
                   regularization: float = 0.0) -> list:
    """Every (pressure x wall x formulation) combination at one regularisation."""
    rows = []
    for pressure in PRESSURE_CLOSURES:
        for wall in WALL_CLOSURES:
            if pressure == "eos" and wall != "shipped_adami":
                continue                    # the EOS route defines its own wall field
            for formulation in FORMULATIONS:
                options = HydrostaticClosureOptions(
                    pressure=pressure, wall=wall, formulation=formulation,
                    regularization=regularization)
                rows.append(closure_residual(state, cfg, mass, options))
    return rows


def regularization_ladder(state, cfg, mass: torch.Tensor,
                          regularizations=(1e-6, 1e-9, 1e-12, 1e-15, 0.0)) -> list:
    """Residual of the consistent closure as a function of its Tikhonov term.

    The ladder is the proof that the reported floor is the regulariser and not a
    cherry-picked setting: the residual tracks ``regularization`` over six decades
    and stops at roundoff when the term is switched off.
    """
    rows = []
    for reg in regularizations:
        options = HydrostaticClosureOptions(
            pressure="incompressible", wall="continuation",
            formulation="moment", regularization=float(reg))
        rows.append(closure_residual(state, cfg, mass, options))
    return rows


def controls(state, cfg, mass: torch.Tensor) -> dict:
    """The controls that decide whether a closure row means anything.

    ``offset``       shifting the *whole* level set by 1 mm shifts ``p`` by one
                     constant on the fluid and on the wall particles alike, which
                     an affine-exact operator annihilates: the residual must be
                     the base residual to roundoff (a measured identity, not a
                     tolerance).  Shifting the level set is the only way to do it:
                     adding the constant to the fluid pressure alone is a
                     *different* state and the control is expected to fail.
    ``slope``        scaling the hydrostatic slope by 1.01 (fluid and wall) must
                     break the balance by ``O(1 % g)`` -- proof that the residual
                     sits at the physical equilibrium and has not been zeroed by
                     construction.
    ``clamped``      flooring the wall pressure at zero restores the kink.
    ``zero_pressure`` a pressure-less state must free-fall at ``1 g``.
    ``cross_check``  the re-implemented ``symmetric`` operator on the shipped
                     fields must reproduce ``droplet_impact.rhs``.
    """
    n = state["pos"].shape[0]
    base = HydrostaticClosureOptions(pressure="incompressible", wall="continuation")
    out = {"base": closure_residual(state, cfg, mass, base)["a_max_over_g"]}
    shifted = dataclasses.replace(base, eta_shift=1.0e-3)
    out["offset"] = closure_residual(state, cfg, mass, shifted)["a_max_over_g"]
    sloped = dataclasses.replace(base, pressure_scale=1.01)
    out["slope"] = closure_residual(state, cfg, mass, sloped)["a_max_over_g"]
    clamped = dataclasses.replace(base, wall="continuation_clamped")
    out["clamped"] = closure_residual(state, cfg, mass, clamped)["a_max_over_g"]
    m_unit = float(state["m_unit"])
    uniform = torch.full((n,), m_unit, dtype=state["pos"].dtype,
                         device=state["pos"].device)
    out["zero_pressure"] = closure_residual(
        state, cfg, uniform, base,
        fields=_zero_pressure_fields(state, cfg, uniform, base))["a_max_over_g"]
    f_sym = closure_fields(state, cfg, mass,
                           HydrostaticClosureOptions(pressure="eos",
                                                     wall="shipped_adami",
                                                     formulation="symmetric"))
    diag = pressure_acceleration(state, cfg, mass,
                                 WellBalancedOptions(formulation="symmetric"),
                                 fields=f_sym)
    _, _, acc = _evaluate(state, cfg, mass)
    out["cross_check"] = float((diag["acceleration"] - acc).abs().max())
    out["level_set"] = level_set(state, cfg)
    return out


def _zero_pressure_fields(state, cfg, mass, options) -> dict:
    """The same closure with the pressure field identically zero."""
    f = closure_fields(state, cfg, mass, options)
    n = f["n"]
    zero = torch.zeros_like(f["pressure"])
    out = dict(f)
    out["pressure"] = zero
    out["p_wall"] = torch.zeros_like(f["p_wall"])
    out["rho_wall"] = torch.full_like(f["rho_wall"], cfg.rho0)
    out["p_full"] = torch.cat((zero, out["p_wall"]))
    out["rho_full"] = torch.cat((torch.full((n,), cfg.rho0, dtype=zero.dtype,
                                            device=zero.device), out["rho_wall"]))
    return out


def tait_reference(state, cfg, mass: torch.Tensor) -> dict:
    """How far the incompressible closure sits from the compressible EOS state.

    The closure buys the anchor tolerance by prescribing the pressure; the price
    is stated here as a number: the depth-wise relative deviation between the
    incompressible affine pressure and the shipped Tait pressure, and between
    ``rho0`` and the EOS equilibrium density.  A reader who wants WCSPH's
    compressibility sees exactly what was traded away.
    """
    fields = closure_fields(state, cfg, mass,
                            HydrostaticClosureOptions(pressure="eos",
                                                      wall="shipped_adami"))
    n = fields["n"]
    p_eos = fields["pressure"][:n]
    y = state["pos"][:, 1]
    eta = level_set(state, cfg)["eta"]
    p_in = cfg.rho0 * cfg.gravity * (eta - y)
    rho_bt = eos_hydrostatic_density(y, depth=eta, rho0=cfg.rho0,
                                     sound_speed=cfg.sound_speed,
                                     gamma=cfg.gamma, gravity=cfg.gravity)
    scale = float(p_in.abs().max())
    return {
        "p_eos_min": float(p_eos.min()),
        "pressure_dev_max": float((p_eos - p_in).abs().max() / scale),
        "pressure_dev_rms": float(((p_eos - p_in) ** 2).mean().sqrt() / scale),
        "eos_density_dev_max": float((rho_bt / cfg.rho0 - 1.0).abs().max()),
        "eos_density_dev_at_bottom": float(abs(float(rho_bt[int(y.argmin())]) / cfg.rho0 - 1.0)),
        "sound_speed": float(cfg.sound_speed),
        "gravity_head_over_c0_squared": float(cfg.gravity * eta / cfg.sound_speed ** 2),
    }