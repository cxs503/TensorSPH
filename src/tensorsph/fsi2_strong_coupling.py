"""Strong (sub-iterated) coupling for the Turek-Hron FSI2 case.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this module exists
----------------------
``docs/xval_fsi2/COUPLED_RUN_DEFECTS.md`` established, with a controlled
experiment, that the shipped *loosely* coupled driver (one fluid solve + one
explicit rod step per ``dt``) diverges at the benchmark density ratio
``rho_s/rho_f = 1`` (beam speed 3.5e50 m/s inside 100 steps) for every variant
tested (bare loose, Riemann wall impedance, nodal-load under-relaxation).  That
is the classical added-mass instability of partitioned explicit FSI in the limit
``m_added ~ m_structural``: no load filter can cure it because there is no
coupling iteration to converge.

This module supplies the missing piece: a per-time-step **interface
sub-iteration** with two accelerators,

* ``aitken``   -- dynamic (Irons-Tuck / secant) relaxation of the interface
                 residual, the cheapest scheme that can produce the *negative*
                 relaxation factors an added-mass problem needs;
* ``iqn-ils``  -- interface quasi-Newton with least-squares (Degroote et al.,
                 2009): the interface secant increments are reused as a reduced
                 Jacobian, which is the standard cure for ``rho_s/rho_f ~ 1``.

The fluid solve itself is unchanged and re-entrant: one WCSPH step from a
snapshot of the step-start state, so re-solving for a new interface position is
exact (no state is accumulated inside the iteration).

Interface variable
------------------
The fixed-point map is ``x -> S(F(x))``: place the beam boundary particles at
the nodal positions ``x``, solve the fluid step, transfer the pressure load,
advance the explicit rod one ``dt`` (with its own sub-steps).  The residual is
``r = S(F(x)) - x`` in metres, i.e. a displacement error, and the convergence
gate is ``||r|| / L_beam < tol`` (default ``1e-3``).

Known limitation (stated, not hidden): the wall velocity handed to the fluid
solve is the step-start beam velocity (a first-order predictor), not
``(x_k - x_start)/dt``.  For the strong scheme this only affects the acoustic
(Riemann) wall-pressure term, which is off by default here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import droplet_impact as di
from .fsi2_turek_hron import (
    FSI2Case,
    FSI2State,
    _apply_open_boundaries,
    beam_advance,
    beam_boundary_geometry,
    body_force_from_fluid,
    boundary_force_from_pressure,
    boundary_pressure_from_fluid,
    initial_state,
)
from .neighbor_search import cell_linked_pairs

__all__ = [
    "AitkenRelaxation",
    "IQNILS",
    "CouplingStats",
    "step_fsi2_strong",
    "run_fsi2_strong",
]


# --------------------------------------------------------------------------- #
# Interface accelerators
# --------------------------------------------------------------------------- #

class AitkenRelaxation:
    """Dynamic (secant / Irons-Tuck) relaxation of the interface residual.

    ``omega_k = -omega_{k-1} (r_{k-1} . (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2``
    and ``x_{k+1} = x_k + omega_k r_k``.  The factor is allowed to be negative,
    which is what an added-mass-dominated interface requires; it is clipped to
    ``[omega_lo, omega_hi]`` so that a degenerate secant cannot blow the
    iteration up.
    """

    def __init__(self, omega0: float = 1.0, omega_lo: float = -3.0,
                 omega_hi: float = 1.0) -> None:
        self.omega0 = float(omega0)
        self.omega_lo = float(omega_lo)
        self.omega_hi = float(omega_hi)
        self.reset()

    def reset(self) -> None:
        self.k = 0
        self.omega = self.omega0
        self.r_prev: Optional[np.ndarray] = None

    def next(self, x: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return the next interface iterate and the relaxation factor used."""
        if self.k > 0 and self.r_prev is not None:
            dr = r - self.r_prev
            denom = float(dr @ dr)
            if denom > 1.0e-30:
                omega = -self.omega * float(self.r_prev @ dr) / denom
                omega = min(max(omega, self.omega_lo), self.omega_hi)
                if math.isfinite(omega) and abs(omega) > 1.0e-8:
                    self.omega = omega
        used = self.omega
        self.r_prev = r.copy()
        self.k += 1
        return x + used * r, used


class IQNILS:
    """Interface quasi-Newton with least-squares (IQN-ILS, Degroote 2009).

    Keeps the last ``max_history`` interface secant pairs ``(dx_i, dr_i)`` and
    solves ``min_q || r_k + V q ||`` (``V`` = columns of ``dr_i``); the same
    coefficients applied to ``W`` (columns of ``dx_i``) give the Newton step,
    ``x_{k+1} = x_k - omega W q`` where ``q = argmin ||r_k - V q||`` (the
    coefficients that express the current residual in the span of the secant
    residual increments, carried over to the iterate increments).  Columns are
    filter-reduced by a relative pivot test so that linearly dependent history does not pollute the
    least-squares solve.
    """

    def __init__(self, max_history: int = 15, omega: float = 1.0,
                 pivot_tol: float = 1.0e-8) -> None:
        self.max_history = int(max_history)
        self.omega = float(omega)
        self.pivot_tol = float(pivot_tol)
        self.reset()

    def reset(self) -> None:
        self.dR: List[np.ndarray] = []
        self.dX: List[np.ndarray] = []
        self.r_prev: Optional[np.ndarray] = None
        self.x_prev: Optional[np.ndarray] = None

    def next(self, x: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float]:
        # register the new secant pair (x, r) against the previous one
        if self.r_prev is not None and self.x_prev is not None:
            self.dR.append(r - self.r_prev)
            self.dX.append(x - self.x_prev)
            if len(self.dR) > self.max_history:
                self.dR.pop(0)
                self.dX.pop(0)
        self.r_prev = r.copy()
        self.x_prev = x.copy()

        step = r.copy()
        if self.dR:
            V = np.asarray(self.dR).T          # (n, m)
            W = np.asarray(self.dX).T          # (n, m)
            # column filter: keep independent columns (cheap Gram-Schmidt pivot)
            keep = self._independent_columns(V)
            if keep:
                Vk, Wk = V[:, keep], W[:, keep]
                try:
                    # express the current residual in the span of the secant
                    # residual increments, apply the same coefficients to the
                    # iterate increments: that is J_r^{-1} r, the Newton step
                    q, *_ = np.linalg.lstsq(Vk, r, rcond=None)
                    cand = Wk @ q
                    if np.all(np.isfinite(cand)):
                        step = cand
                except np.linalg.LinAlgError:      # pragma: no cover - defensive
                    pass
        used = self.omega
        return x - used * step, used

    def _independent_columns(self, V: np.ndarray) -> List[int]:
        keep: List[int] = []
        basis: List[np.ndarray] = []
        for j in range(V.shape[1]):
            v = V[:, j].copy()
            norm0 = float(np.linalg.norm(v))
            if norm0 <= 1.0e-30:
                continue
            for b in basis:
                v = v - (b @ v) * b
            norm = float(np.linalg.norm(v))
            if norm > self.pivot_tol * max(norm0, 1.0e-300) and norm > 1.0e-16:
                basis.append(v / norm)
                keep.append(j)
        return keep


def _make_accelerator(kind: str):
    kind = (kind or "none").lower()
    if kind in ("none", "fixed", ""):
        return None, None
    if kind == "aitken":
        acc = AitkenRelaxation()
        return acc, acc.reset
    if kind in ("iqn-ils", "iqnils", "iqn"):
        acc = IQNILS()
        return acc, acc.reset
    raise ValueError(f"unknown interface solver {kind!r}")


# --------------------------------------------------------------------------- #
# Coupled step
# --------------------------------------------------------------------------- #

@dataclass
class CouplingStats:
    """Per-step coupling diagnostics."""

    iterations: int = 0
    residual_rel: float = 0.0        # ||r_k|| / ||r_0|| (scale-free)
    residual_abs_rel: float = 0.0    # ||r_k|| / L_beam (physical scale)
    converged: bool = False
    aborted: bool = False
    relaxation: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)


def _fluid_load(state: FSI2State, case: FSI2Case, dt: float,
                nodes: torch.Tensor, beam_vel: torch.Tensor,
                fluid_pos: torch.Tensor, fluid_vel: torch.Tensor,
                fluid_mass: torch.Tensor,
                cfg, wall_vel_nodes: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
    """One WCSPH step from the given fluid snapshot with the beam at ``nodes``.

    Pure: nothing on ``state`` is mutated, so the step can be re-solved for a
    different interface position (the prerequisite for a coupling iteration).
    """
    beam_boundary, beam_normals, beam_interp, tributary = beam_boundary_geometry(
        nodes, case)
    wall_start = state.wall_static.shape[0] + state.cylinder_particles.shape[0]

    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_boundary), dim=0)
    m_unit = di.lattice_mass_unit(cfg)
    wall_mass = torch.cat((
        state.wall_mass_static,
        state.cylinder_mass,
        torch.full((beam_boundary.shape[0],), m_unit,
                   dtype=beam_boundary.dtype, device=beam_boundary.device),
    ))
    v_nodes = beam_vel if wall_vel_nodes is None else wall_vel_nodes
    beam_boundary_vel = beam_interp @ v_nodes
    wall_vel = torch.cat((
        torch.zeros_like(state.wall_static),
        torch.zeros_like(state.cylinder_particles),
        beam_boundary_vel,
    ))

    all_pos = torch.cat((fluid_pos, wall), dim=0)
    pairs = cell_linked_pairs(all_pos, 2.0 * case.h())
    rho, pressure, acc = di.rhs(fluid_pos, fluid_vel, fluid_mass, cfg,
                                pairs=pairs, wall=wall, dt=dt, wall_vel=wall_vel)

    p_wall, w_sum = boundary_pressure_from_fluid(
        fluid_pos, pressure, wall, wall_start, pairs, case.h(),
        normals=beam_normals if case.wall_impedance else None,
        vel=fluid_vel if case.wall_impedance else None,
        wall_vel=wall_vel if case.wall_impedance else None,
        rho0=case.rho0, sound_speed=case.sound_speed,
        impedance_sign=case.wall_impedance_sign)
    f_beam = boundary_force_from_pressure(p_wall, beam_normals, tributary)
    coverage = (float((w_sum > 1.0e-30).to(torch.float64).mean())
                if p_wall.numel() else 0.0)
    # The body (cylinder + beam) force-from-fluid arms: without them the
    # Turek-Hron drag/lift coefficients cannot be formed on this path, and this
    # path is the only one that survives ``rho_s/rho_f = 1`` (the loose driver
    # blows up).  Same operators as :func:`tensorsph.fsi2_turek_hron.step_fsi2`
    # -- one shared helper, so the two drivers cannot drift apart.
    body = body_force_from_fluid(
        fluid_pos, rho, pressure, wall, pairs, case,
        cyl_start=state.wall_static.shape[0],
        cyl_normals=state.cylinder_normals,
        cyl_tributary=state.cylinder_tributary,
        # the friction arm needs the fluid velocity whatever the interface
        # scheme is; the Robin arm is gated inside the helper
        vel=fluid_vel,
        wall_vel=wall_vel,
        fluid_mass=fluid_mass,
        beam_boundary=beam_boundary, beam_normals=beam_normals,
        beam_tributary=tributary, beam_boundary_vel=beam_boundary_vel)

    # Structural load = pressure traction + skin friction on the same faces
    # (shared fix with the loose driver in fsi2_turek_hron.step_fsi2).
    f_visc_beam = body["f_visc_beam"]
    f_beam_struct = (f_beam + f_visc_beam
                     if (case.beam_viscous_load and f_visc_beam.numel())
                     else f_beam)
    raw_load = (beam_interp.T @ f_beam_struct) * case.load_scale
    return {"raw_load": raw_load, "f_beam": f_beam,
            "f_beam_struct": f_beam_struct, "f_visc_beam": f_visc_beam,
            "acc": acc,
            "coverage": coverage,
            "f_cyl": body["f_cyl"], "f_cyl_fit": body["f_cyl_fit"],
            "f_visc_cyl": body["f_visc_cyl"], "f_visc_beam": body["f_visc_beam"],
            "cyl_coverage": body["cyl_coverage"]}


def _advance_solid(nodes0: torch.Tensor, vel0: torch.Tensor,
                   nodal_load: torch.Tensor, case: FSI2Case,
                   dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    n_sub = case.rod_substeps_effective()
    dt_rod = dt / n_sub
    nodes, vel = nodes0, vel0
    for _ in range(n_sub):
        nodes, vel = beam_advance(nodes, vel, nodal_load, case, dt_rod)
    return nodes, vel


def step_fsi2_strong(state: FSI2State, case: FSI2Case, dt: float,
                     interface_solver: str = "iqn-ils",
                     max_iter: int = 20, tol: float = 1.0e-3,
                     abs_tol: float = 1.0e-12,
                     wall_impedance: bool = False,
                     accelerator=None,
                     max_step_factor: float = 10.0,
                     wall_velocity_from_iterate: bool = False,
                     ) -> Tuple[Dict[str, float], CouplingStats]:
    """Advance the coupled FSI2 system by one step with interface sub-iterations.

    ``tol`` is the *relative residual reduction* ``||r_k|| / ||r_0||`` of the
    interface displacement residual ``r = S(F(x)) - x``; a step is accepted when
    it drops below ``tol`` (or when ``||r_k|| <= abs_tol * L_beam``, the
    machine-level floor).  The reduction form is essential: an absolute gate
    keyed to ``L_beam`` is satisfied by the first iterate whenever the response
    per step is small, and the iteration then silently degenerates into loose
    coupling (that degenerate gate is exactly how the added-mass blow-up got
    mistaken for "converged in 1 iteration" in the first probe of this round).
    """
    cfg = case.fluid_config()
    cfg.dt = dt
    case.wall_impedance = bool(wall_impedance)

    counters = {"injected": 0, "removed": 0}
    if case.open_boundaries:
        counters = _apply_open_boundaries(state, case)

    stats = CouplingStats()
    fluid_pos = state.fluid_pos.clone()
    fluid_vel0 = state.fluid_vel.clone()
    fluid_mass = state.fluid_mass.clone()
    nodes0 = state.beam_nodes.clone()
    vel0 = state.beam_vel.clone()

    if accelerator is not None:
        # reuse the interface Jacobian built in previous time steps: at the
        # first iteration of a step there is no in-step history, and for an
        # added-mass-dominated interface a cold start overshoots by the gain
        # factor (measured: 4.35x per step at rho_s/rho_f = 1)
        accel = accelerator
    else:
        accel, reset = _make_accelerator(interface_solver)
        if reset is not None:
            reset()

    n_nodes = nodes0.shape[0]
    x_k = nodes0.reshape(-1).to(torch.float64)
    l_ref = float(case.beam_length)
    nodal_load = state.load if state.load is not None else torch.zeros_like(nodes0)
    last = None
    nodes_new, vel_new = nodes0, vel0
    r0_norm = None

    for it in range(max_iter):
        nodes_k = x_k.to(nodes0.dtype).reshape(n_nodes, 2)
        if wall_velocity_from_iterate:
            # Robin (impedance) interface: the fluid sees the velocity implied
            # by the current interface iterate, not the step-start velocity.
            # ``nodes0`` is fixed, so this is the exact increment over the step.
            v_iter = (nodes_k - nodes0) / dt
            sol = _fluid_load(state, case, dt, nodes_k, vel0,
                              fluid_pos, fluid_vel0, fluid_mass, cfg,
                              wall_vel_nodes=v_iter)
        else:
            sol = _fluid_load(state, case, dt, nodes_k, vel0,
                              fluid_pos, fluid_vel0, fluid_mass, cfg)
        # load relaxation (kept for compatibility; 1.0 = none)
        alpha = float(case.coupling_relaxation)
        raw_load = sol["raw_load"]
        if alpha >= 1.0:
            nodal_load = raw_load
        else:
            nodal_load = alpha * raw_load + (1.0 - alpha) * nodal_load

        nodes_new, vel_new = _advance_solid(nodes0, vel0, nodal_load, case, dt)
        r_t = (nodes_new.reshape(-1).to(torch.float64) - x_k)
        r_norm = float(torch.linalg.vector_norm(r_t))
        if r0_norm is None:
            r0_norm = r_norm
        red = (r_norm / r0_norm) if r0_norm > 0.0 else 0.0
        stats.iterations = it + 1
        stats.residual_rel = red
        stats.residual_abs_rel = r_norm / l_ref
        stats.residual_history.append(red)
        last = sol

        # the secant pair (x_k, r_k) is registered even when the step is already
        # converged: the interface Jacobian is a multi-step object, and a step
        # that needs no iteration must still contribute its pair, otherwise the
        # history is empty on the first step that does need one
        if accel is None:
            x_next = x_k + r_t
            stats.relaxation.append(1.0)
        else:
            x_np, used = accel.next(x_k.numpy(), r_t.numpy())
            x_next = torch.as_tensor(x_np, dtype=x_k.dtype)
            # safeguard: never let the accelerator propose a step that is
            # absurdly larger than the residual it is trying to cancel
            step_norm = float(torch.linalg.vector_norm(x_next - x_k))
            cap = max_step_factor * max(r_norm, 1.0e-300)
            if step_norm > cap:
                x_next = x_k + (x_next - x_k) * (cap / step_norm)
            stats.relaxation.append(float(used))

        if red <= tol or r_norm <= abs_tol * l_ref:
            stats.converged = True
            break
        if not math.isfinite(r_norm) or red > 1.0e3:
            stats.aborted = True
            break
        x_k = x_next

    # ---- finalise the accepted state ------------------------------------- #
    state.beam_nodes = nodes_new.contiguous()
    state.beam_vel = vel_new.contiguous()
    state.load = nodal_load

    acc = last["acc"] if last is not None else torch.zeros_like(fluid_vel0)
    state.fluid_vel = fluid_vel0 + dt * acc
    state.fluid_pos = fluid_pos + dt * state.fluid_vel
    state.fluid_pos[:, 1] = state.fluid_pos[:, 1].clamp(0.0, case.channel_height)
    if not case.open_boundaries:
        state.fluid_pos[:, 0] = state.fluid_pos[:, 0].clamp(0.0, case.channel_length)
    state.time += dt

    f_beam = last["f_beam"] if last is not None else torch.zeros((0, 2))
    tip_x = float(state.beam_nodes[-1, 0]) - (case.cylinder_x + case.cylinder_d / 2
                                              + case.beam_length)
    tip_y = float(state.beam_nodes[-1, 1]) - case.cylinder_y
    def _sum(key):
        f = last.get(key) if last is not None else None
        if f is None or not f.numel():
            return torch.zeros(2, dtype=torch.float64)
        return f.sum(dim=0)

    f_cyl_c = _sum("f_cyl")
    f_cyl_fit_c = _sum("f_cyl_fit")
    visc_c = _sum("f_visc_cyl") + _sum("f_visc_beam")
    drag_beam = float(f_beam[:, 0].sum()) if f_beam.numel() else 0.0
    lift_beam = float(f_beam[:, 1].sum()) if f_beam.numel() else 0.0
    f_struct = last.get("f_beam_struct") if last is not None else None
    drag_struct = float(f_struct[:, 0].sum()) if (f_struct is not None
                                                  and f_struct.numel()) else 0.0
    lift_struct = float(f_struct[:, 1].sum()) if (f_struct is not None
                                                  and f_struct.numel()) else 0.0
    f_vb = last.get("f_visc_beam") if last is not None else None
    beam_friction = float(f_vb[:, 0].sum()) if (f_vb is not None
                                                and f_vb.numel()) else 0.0
    f_cyl_n = 0 if last is None or last.get("f_cyl") is None else int(last["f_cyl"].shape[0])
    row = {
        "t": state.time, "tip_x": tip_x, "tip_y": tip_y,
        "drag": drag_beam,
        "lift": lift_beam,
        "drag_struct": drag_struct, "lift_struct": lift_struct,
        "beam_drag_friction": beam_friction,
        "cyl_drag": float(f_cyl_c[0]), "cyl_lift": float(f_cyl_c[1]),
        "body_drag": float(f_cyl_c[0]) + drag_beam,
        "body_lift": float(f_cyl_c[1]) + lift_beam,
        "cyl_drag_fit": float(f_cyl_fit_c[0]), "cyl_lift_fit": float(f_cyl_fit_c[1]),
        "body_drag_fit": float(f_cyl_fit_c[0]) + drag_beam,
        "body_lift_fit": float(f_cyl_fit_c[1]) + lift_beam,
        "drag_viscous": float(visc_c[0]), "lift_viscous": float(visc_c[1]),
        "cyl_coverage": float(last.get("cyl_coverage", 0.0)) if last else 0.0,
        "n_cyl_particles_on_body": f_cyl_n,
        "max_speed": float(state.fluid_vel.norm(dim=1).max()),
        "n_fluid": int(state.fluid_pos.shape[0]),
        "injected": counters["injected"], "removed": counters["removed"],
        "force_coverage": last["coverage"] if last is not None else 0.0,
        "coupling_iterations": float(stats.iterations),
        "coupling_residual": float(stats.residual_rel),
        "coupling_residual_abs": float(stats.residual_abs_rel),
        "coupling_converged": float(1.0 if stats.converged else 0.0),
    }
    return row, stats


def run_fsi2_strong(case: FSI2Case, n_steps: int,
                    interface_solver: str = "iqn-ils", max_iter: int = 20,
                    tol: float = 1.0e-3, abs_tol: float = 1.0e-12,
                    wall_impedance: bool = False,
                    wall_velocity_from_iterate: bool = False,
                    state: Optional[FSI2State] = None,
                    stop_on_abort: bool = True):
    """Integrate ``n_steps`` with strong coupling; return state + time history."""
    if state is None:
        state = initial_state(case)
    dt = case.dt()
    hist: Dict[str, List[float]] = {
        "t": [], "tip_x": [], "tip_y": [], "drag": [], "lift": [],
        "drag_struct": [], "lift_struct": [], "beam_drag_friction": [],
        "max_speed": [], "n_fluid": [], "coupling_iterations": [],
        "coupling_residual": [], "coupling_converged": [],
    }
    aborted = False
    accel, _ = _make_accelerator(interface_solver)
    for k in range(n_steps):
        row, stats = step_fsi2_strong(
            state, case, dt, interface_solver=interface_solver,
            max_iter=max_iter, tol=tol, abs_tol=abs_tol,
            wall_impedance=wall_impedance, accelerator=accel,
            wall_velocity_from_iterate=wall_velocity_from_iterate)
        for key in hist:
            hist[key].append(float(row[key]))
        if stats.aborted:
            aborted = True
            if stop_on_abort:
                break
        if not math.isfinite(row["max_speed"]):
            aborted = True
            if stop_on_abort:
                break
    return state, {"history": hist, "aborted": aborted, "dt": dt,
                   "n_steps": len(hist["t"]), "dx": case.dx,
                   "n_fluid": int(state.fluid_pos.shape[0])}