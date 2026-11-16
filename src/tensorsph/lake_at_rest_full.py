"""Extended lake-at-rest reconstruction: per-particle mass field for full balance.

This module extends :mod:`tensorsph.lake_at_rest` by solving for a *per-particle*
mass field (no mirror-symmetry tying) that minimises **both** vertical and
horizontal acceleration of the shipped WCSPH operator.

Why this is needed
------------------
The original :func:`tensorsph.lake_at_rest.reconstruct` ties mirror pairs and
only solves for vertical balance.  Because the horizontal pressure force is odd
under mirror reflection, no symmetric mass field can cancel the side-wall
truncation residual.  By relaxing the symmetry constraint and solving for one
mass per particle, the horizontal residual can also be reduced.

Unlike the position-correction approach (which requires rebuilding the neighbour
list and breaks differentiability), this module keeps the lattice fixed and only
adjusts masses, so the shipped ``rhs`` remains fully differentiable through
autograd.

Usage
-----
>>> from tensorsph.lake_at_rest import LakeConfig, build_state, analytic_profile
>>> from tensorsph.lake_at_rest_full import reconstruct_full
>>> cfg = LakeConfig(dx=0.04, depth=0.28)
>>> state = build_state(cfg)
>>> prior = analytic_profile(state, cfg)
>>> sol = reconstruct_full(state, cfg, prior["mass"])
>>> sol.a_max_over_g  # both components reduced
>>> sol.a_max_over_g_horizontal
>>> sol.max_delta_r  # zero: no position corrections in this variant
0.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .droplet_impact import rhs
from .lake_at_rest import LakeConfig, _evaluate, residual_report

__all__ = ["reconstruct_full", "FullReconstructionResult"]


@dataclass
class FullReconstructionResult:
    """Result of the extended per-particle reconstruction."""

    mass: torch.Tensor
    """Reconstructed particle mass field (n_fluid,)."""

    pos: torch.Tensor
    """Particle positions (unchanged from the input state)."""

    delta_r: torch.Tensor
    """Position corrections (all zeros in this variant)."""

    density: torch.Tensor
    """Summation density at the reconstructed state (n_fluid,)."""

    pressure: torch.Tensor
    """Tait pressure at the reconstructed state (n_fluid,)."""

    a_max_over_g: float
    """Maximum |a|/g over all particles."""

    a_max_over_g_horizontal: float
    """Maximum |a_x|/g over all particles."""

    a_max_over_g_vertical: float
    """Maximum |a_y|/g over all particles."""

    iterations: int
    """Number of Gauss-Newton iterations."""

    converged: bool
    """Whether the solver converged to the tolerance."""

    residual_history: list[float]
    """Residual history (max|a|/g per iteration)."""

    max_delta_r: float
    """Maximum |delta_r| over all particles (always 0 here)."""

    p_max: float
    """Maximum pressure."""

    rho_over_rho0_min: float
    """Minimum rho/rho0."""

    rho_over_rho0_max: float
    """Maximum rho/rho0."""


def reconstruct_full(state: dict, cfg: LakeConfig,
                     prior: torch.Tensor | None = None,
                     verbose: bool = False) -> FullReconstructionResult:
    """Per-particle mass reconstruction for full (ax + ay) balance.

    Solves for one mass per particle (no mirror-symmetry tying) such that
    *both* vertical and horizontal acceleration are minimised.  The lattice
    and pair list are frozen (taken from the input state), so the shipped
    ``rhs`` is fully differentiable through autograd.

    Parameters
    ----------
    state : dict
        Output of :func:`tensorsph.lake_at_rest.build_state`.
    cfg : LakeConfig
        Case configuration.
    prior : torch.Tensor, optional
        Initial mass field guess (per particle). Defaults to uniform lattice mass.
    verbose : bool
        Print iteration diagnostics.

    Returns
    -------
    FullReconstructionResult
        The reconstructed state with diagnostics.
    """
    n = state["n_fluid"]
    m_unit = state["m_unit"]
    dtype = state["pos"].dtype
    device = state["pos"].device

    if prior is None:
        prior = torch.full((n,), m_unit, dtype=dtype, device=device)

    # u is per-particle (no symmetry tying)
    u = torch.log(prior / m_unit).clone()
    u_prior = u.clone()

    g_val = cfg.gravity
    tol = cfg.solver_tol
    max_iter = cfg.solver_max_iter
    prior_weight = cfg.prior_weight

    def residual(u_v: torch.Tensor) -> torch.Tensor:
        """Both acceleration components as residual."""
        mass = m_unit * torch.exp(u_v)
        _rho, _p, acc = _evaluate(state, cfg, mass)
        return torch.cat([acc[:, 0], acc[:, 1]])  # ax, ay

    history = []
    lam = prior_weight
    converged = False
    iteration = 0

    for iteration in range(max_iter):
        f = residual(u)
        ax = f[:n]
        ay = f[n:]
        a_mag = torch.sqrt(ax ** 2 + ay ** 2)
        worst = float(a_mag.max() / g_val)
        history.append(worst)
        if verbose:
            print(f"  iter {iteration}: max|a|/g = {worst:.3e} "
                  f"(ax={ax.abs().max()/g_val:.3e}, ay={ay.abs().max()/g_val:.3e})")

        if worst < tol:
            converged = True
            break

        # Jacobian via autograd
        jac = torch.autograd.functional.jacobian(residual, u)
        # jac shape: (2*n, n)

        # Solve [jac; lam*I] du = [-f; lam*(u_prior - u)]
        rows = [jac, lam * torch.eye(n, dtype=dtype, device=device)]
        rhs_vec = [-f, lam * (u_prior - u)]
        step = torch.linalg.lstsq(torch.cat(rows, dim=0),
                                  torch.cat(rhs_vec, dim=0)).solution

        # Backtracking line search
        def merit(u_v):
            f_v = residual(u_v)
            ax_v = f_v[:n]
            ay_v = f_v[n:]
            a_mag_v = torch.sqrt(ax_v ** 2 + ay_v ** 2)
            return float(a_mag_v.max() / g_val)

        alpha, base = 1.0, worst
        accepted = False
        for _ in range(30):
            candidate = u + alpha * step
            candidate_merit = merit(candidate)
            if candidate_merit < base:
                u = candidate
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    mass = m_unit * torch.exp(u)
    rho, pressure, acc = _evaluate(state, cfg, mass)
    ax, ay = acc[:, 0], acc[:, 1]
    a_mag = torch.sqrt(ax ** 2 + ay ** 2)

    return FullReconstructionResult(
        mass=mass,
        pos=state["pos"],
        delta_r=torch.zeros_like(state["pos"]),
        density=rho,
        pressure=pressure,
        a_max_over_g=float(a_mag.max() / g_val),
        a_max_over_g_horizontal=float(ax.abs().max() / g_val),
        a_max_over_g_vertical=float(ay.abs().max() / g_val),
        iterations=iteration + 1,
        converged=converged,
        residual_history=history,
        max_delta_r=0.0,
        p_max=float(pressure.max()),
        rho_over_rho0_min=float((rho / cfg.rho0).min()),
        rho_over_rho0_max=float((rho / cfg.rho0).max()),
    )
