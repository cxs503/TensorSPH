"""Lake at rest: a discrete equilibrium of the shipped WCSPH operator.

The case pool anchor ``CASE-SPH-LAKE-AT-REST`` asks for the textbook
well-balanced test: a body of water with a free surface, no forcing beyond
gravity, must stay **at rest** -- no spurious currents, no surface oscillation --
to machine precision.  "Zero" is the analytic identity; there is no
experimental uncertainty and no fitted constant anywhere in this module.

What is actually asserted, and why it is not a tautology
--------------------------------------------------------
A continuous hydrostatic state is *not* a discrete equilibrium of a particle
method: the kernel support is truncated at the bottom wall and at the free
surface, the lattice kernel sums are not the kernel integrals, and the Tait law
turns any density bias into a spurious pressure.  Initialising the shipped solver
with the analytic profile therefore leaves a finite residual acceleration --
measured below at ``max|a|/g = 1.00`` for the shipped uniform lattice (the
particles simply free-fall, because a uniform mass field sums to ``rho0``
*everywhere*, so the pressure is identically zero) and at ``O(1) g`` for the
analytic hydrostatic profile.

The test is therefore split into two claims that are verified separately:

1. **Discrete well-balancedness (machine precision).**
   :func:`reconstruct` solves for the *particle mass field* ``m_i`` such that the
   shipped :func:`tensorsph.droplet_impact.rhs` returns zero acceleration to
   roundoff.  Nothing in the operator is modified: the unknowns absorb the
   truncation instead of the physics.  With that state as the initial condition
   the velocity stays at rounding level for the whole run -- and that is a real
   statement, because the *same* operator on the *same* lattice with the
   analytic profile produces ``O(g)`` accelerations (negative controls
   ``uniform`` and ``analytic``).  The reconstruction is the discrete analogue of
   the hydrostatic reconstruction a well-balanced finite-volume scheme applies.
2. **Physical accuracy of that equilibrium (a measured error, not an identity).**
   The reconstructed mass/density/pressure profile is compared with

   * the incompressible reference ``p = rho0 g (H - y)``, and
   * the exact barotropic equilibrium of the Tait EOS
     (:func:`tensorsph.eos_hydrostatics.eos_hydrostatic_density`),

   on an interior band that excludes two smoothing lengths at the wall and at the
   free surface.  The difference is discretisation error and it is reported with
   its convergence order in ``dx`` -- it is *not* zero and must not be sold as
   such.  Separating it from claim 1 is what stops the machine-precision
   statement from being circular.

Non-uniqueness of the reconstruction
------------------------------------
The pair operator annihilates a constant pressure field on a complete support,
so the linearisation of the balance has a large near-null space: many mass
fields satisfy ``a = 0`` to roundoff and they differ by ``O(10 %)`` particle
mass near the free surface.  :func:`reconstruct` therefore minimises the
deviation from the analytic profile subject to the balance (a damped
Gauss-Newton step with a Tikhonov term towards the prior), and reports the
remaining scatter.  The scatter is a property of the rank-deficient balance
operator, not an error of the equilibrium: the residual is at rounding level for
the regularised solution as well.

Determinism
-----------
No random numbers, no fitted constants.  Every number this module reports can be
re-derived from ``dx``, ``H``, ``c0`` and the library's kernel.

Usage
-----
``python benchmarks/lake_at_rest/run.py --json out.json`` writes the artefact;
:func:`run_case` is what the tests call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from . import eos_hydrostatics as eos
from .droplet_impact import (Config, cubic_spline_kernel, drop_wall_wall_pairs,
                             lattice_mass_unit, pair_kernels, pool_lattice, rhs)
from .neighbor_search import cell_linked_pairs

__all__ = [
    "LakeConfig",
    "build_state",
    "analytic_profile",
    "reconstruct",
    "residual_report",
    "integrate_at_rest",
    "run_case",
]


@dataclass
class LakeConfig:
    """Geometry, fluid and solver parameters of the lake-at-rest case."""

    width: float = 0.40          # tank width (m); columns are mirror-symmetric
    depth: float = 0.30          # still-water depth (m)
    dx: float = 0.020            # particle spacing (m)
    h_factor: float = 1.3        # smoothing length / dx
    rho0: float = 1000.0
    sound_speed: float = 20.0
    gamma: float = 7.0
    gravity: float = 9.81
    viscosity: float = 0.0        # a resting lake has zero strain rate
    wall_layers: int = 3
    max_courant: float = 0.30    # dt = max_courant * h / c0
    t_end: float = 0.50          # simulated time (s)
    sample_every: int = 20
    solver_max_iter: int = 30
    solver_tol: float = 1.0e-11  # |a_y| / g at which Newton stops
    prior_weight: float = 1.0e-8  # Tikhonov pull towards the analytic profile
    surface_closure: bool = False  # shipped Adami closure: see the docstring
    dtype: str = "float64"
    device: str = "cpu"

    def h(self) -> float:
        return self.h_factor * self.dx

    def dt(self) -> float:
        return self.max_courant * self.h() / self.sound_speed

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]

    def shipped(self) -> Config:
        """The production ``Config`` this case drives (no case-private physics)."""
        return Config(pool_width=self.width, pool_depth=self.depth, dx=self.dx,
                      h_factor=self.h_factor, rho0=self.rho0,
                      sound_speed=self.sound_speed, gamma=self.gamma,
                      gravity=self.gravity, viscosity=self.viscosity,
                      wall_layers=self.wall_layers, boundary="wall",
                      density_calibration=True,
                      surface_pressure_closure=self.surface_closure)


def build_state(cfg: LakeConfig):
    """Lattice, wall particles and the (frozen) neighbour list.

    Positions never move in this case -- that is the point of it -- so the pair
    list is built once and reused; a changing pair list would only hide a
    drifting state.
    """
    shipped = cfg.shipped()
    device = torch.device(cfg.device)
    # Snap onto the exact lattice: the same column reached through two different
    # arithmetic paths differs in the last bits, and a lake that must be exactly
    # mirror-symmetric cannot afford two copies of "x = 0.45".
    pos = pool_lattice(shipped, device).to(cfg.torch_dtype).round(decimals=12)
    wall = wall_particles(cfg, device).to(cfg.torch_dtype).round(decimals=12)
    n, nw = pos.shape[0], wall.shape[0]
    pairs = cell_linked_pairs(torch.cat((pos, wall), dim=0), 2.0 * cfg.h())
    pairs = drop_wall_wall_pairs(pairs, n)
    m_unit = float(lattice_mass_unit(shipped))
    return {
        "config": shipped,
        "pos": pos,
        "wall": wall,
        "pairs": pairs,
        "m_unit": m_unit,
        "n_fluid": n,
        "n_wall": nw,
        "columns": _columns(pos, cfg),
    }


def wall_particles(cfg: LakeConfig, device) -> torch.Tensor:
    """Static wall particles, exactly mirror-symmetric about ``x = width / 2``.

    Same construction as :func:`tensorsph.droplet_impact.wall_lattice` -- three
    layers half a spacing outside the fluid box, extended around the corners --
    built as one lattice so that every column is the *same* float in the bottom
    row and in the side walls.  ``wall_lattice`` reaches the same columns through
    two arithmetic paths, and the halves can differ in the last bits; two
    almost-equal columns then break the mirror map (see
    :func:`wall_symmetry_report`).
    """
    dx, layers = cfg.dx, cfg.wall_layers
    columns = int(math.floor(cfg.width / dx + 1.0e-9))
    dtype = torch.get_default_dtype()
    k = torch.arange(-layers, columns + layers, dtype=dtype, device=device)
    xs = (k + 0.5) * dx
    rows = torch.arange(columns, dtype=dtype, device=device)
    ys_fluid = (rows + 0.5) * dx
    parts = []
    for layer in range(1, layers + 1):
        y = torch.full_like(xs, dx * 0.5 - layer * dx)
        parts.append(torch.stack((xs, y), dim=-1))
    for layer in range(1, layers + 1):
        for x in (dx * 0.5 - layer * dx, cfg.width - dx * 0.5 + layer * dx):
            parts.append(torch.stack((torch.full_like(ys_fluid, x), ys_fluid), dim=-1))
    return torch.cat(parts, dim=0)


def density_operator(state, cfg: LakeConfig):
    """The summation-density operator as a matrix: ``rho = A @ mass + b``.

    Density summation is *linear in the particle masses*, which is worth making
    explicit: it turns "find the mass field whose summation density is the
    analytic profile" from a divergent fixed point (the free-surface rows
    oscillate) into one direct solve.  ``b`` holds the (static) wall
    contribution.  The self term is on the diagonal.
    """
    pos, wall = state["pos"], state["wall"]
    pairs = state["pairs"]
    n = pos.shape[0]
    h = cfg.h()
    dtype, device = pos.dtype, pos.device
    i, j = pairs.unbind(dim=1)
    delta = torch.cat((pos, wall), dim=0)[i] - torch.cat((pos, wall), dim=0)[j]
    wij = pair_kernels(delta, h)[0]
    fluid = j < n
    A = torch.zeros((n, n), dtype=dtype, device=device)
    diagonal = cubic_spline_kernel(torch.zeros(1, dtype=dtype, device=device), h)[0]
    A[torch.arange(n, device=device), torch.arange(n, device=device)] = diagonal
    index = i[fluid] * n + j[fluid]
    A.view(-1).index_add_(0, index, wij[fluid])
    A.view(-1).index_add_(0, j[fluid] * n + i[fluid], wij[fluid])
    m_wall = torch.full((int((~fluid).sum()),), state["m_unit"], dtype=dtype,
                        device=device)
    b = torch.zeros(n, dtype=dtype, device=device)
    b.index_add_(0, i[~fluid], m_wall * wij[~fluid])
    return A, b


def _density_of(state, cfg: LakeConfig, A, b, mass: torch.Tensor) -> torch.Tensor:
    return A @ mass + b


def wall_symmetry_report(cfg: LakeConfig) -> dict:
    """Is the wall lattice mirror-symmetric about ``x = width / 2``?

    Measured, not assumed.  :func:`tensorsph.droplet_impact.wall_lattice` builds
    its bottom row from ``arange(-(layers-1/2)dx, width + (layers-1/2)dx, dx)``,
    which *is* symmetric in exact arithmetic, but the same column reached through
    two arithmetic paths (the bottom row and the side wall) can differ in the last
    bits: the unrounded wall then carries two almost-identical columns, the mirror
    map sees no partner for one of them, and the mass field loses its exact
    symmetry.  Rounding to ``1e-12`` (done in :func:`build_state`) removes it,
    which is why the count below is zero.  Reported so that a regression to the
    unrounded wall is visible instead of showing up as a mysterious horizontal
    residual.
    """
    from .droplet_impact import wall_lattice
    shipped = wall_lattice(cfg.shipped(), torch.device("cpu"), top=cfg.depth)
    xs = torch.unique(shipped[:, 0].round(decimals=12))
    mirror = cfg.width - xs
    paired = sum(1 for x in mirror.tolist()
                 if bool((xs - x).abs().min() < 1e-9 * cfg.width))
    case = wall_particles(cfg, torch.device("cpu"))
    case_xs = torch.unique(case[:, 0])
    case_paired = sum(1 for x in (cfg.width - case_xs).tolist()
                      if bool((case_xs - x).abs().min() < 1e-9 * cfg.width))
    return {"shipped_wall_columns": int(xs.numel()),
            "shipped_wall_columns_with_partner": int(paired),
            "unpaired": int(xs.numel() - paired),
            "case_wall_unpaired": int(case_xs.numel() - case_paired)}


def _columns(pos: torch.Tensor, cfg: LakeConfig):
    """Mirror-pair map: particle -> unique degree of freedom ``row * nc + col``.

    The tank is symmetric about ``x = width / 2``.  Tying the masses of mirror
    particles makes the mass field exactly symmetric, which makes the horizontal
    acceleration vanish *by symmetry* instead of leaving a small asymmetric
    residual that would slowly slide the lake sideways.  The degree of freedom is
    per (row, mirror column) pair -- grouping by column alone would average
    different depths together and destroy the profile.
    """
    xs = torch.unique(pos[:, 0].round(decimals=12))
    ys = torch.unique(pos[:, 1].round(decimals=12))
    count, rows = xs.numel(), ys.numel()
    tolerance = 1e-9 * cfg.width
    for k in range(count // 2):
        if abs(float(xs[k] + xs[count - 1 - k] - cfg.width)) > tolerance:
            raise ValueError(
                "the lattice is not mirror-symmetric about the tank centre: "
                f"column x={float(xs[k]):.6f} has no partner; check that "
                "width / dx is an even integer")
    unique_of_column = torch.tensor([min(k, count - 1 - k) for k in range(count)],
                                    dtype=torch.long, device=pos.device)
    x_rounded = pos[:, 0].round(decimals=12)
    y_rounded = pos[:, 1].round(decimals=12)
    distance_x = (x_rounded[:, None] - xs[None, :]).abs()
    column = distance_x.argmin(dim=1)
    if float(distance_x.gather(1, column[:, None]).max()) > tolerance:
        raise ValueError("particle columns do not match the lattice columns")
    distance_y = (y_rounded[:, None] - ys[None, :]).abs()
    row = distance_y.argmin(dim=1)
    if float(distance_y.gather(1, row[:, None]).max()) > tolerance:
        raise ValueError("particle rows do not match the lattice rows")
    return row * ((count + 1) // 2) + unique_of_column[column]


def analytic_profile(state, cfg: LakeConfig):
    """Analytic hydrostatic mass, density and pressure on the lattice.

    ``mass_i`` is the exact-EOS equilibrium density divided by the *local*
    lattice kernel-sum fraction, i.e. the profile that makes the summation
    density equal ``rho_eq(y)``.  This is the prior for the reconstruction and
    the second negative control: it is the best a continuum reference can do,
    and its residual is ``O(g)`` -- which is the whole reason a discrete
    reconstruction is needed.
    """
    pos = state["pos"]
    shipped = state["config"]
    m_unit = state["m_unit"]
    rho_uniform, _, _ = rhs(pos, torch.zeros_like(pos),
                            torch.full((pos.shape[0],), m_unit,
                                       dtype=pos.dtype, device=pos.device),
                            shipped, pairs=state["pairs"], wall=state["wall"])
    fraction = rho_uniform / cfg.rho0                      # support fraction
    rho_eq = eos.eos_hydrostatic_density(pos[:, 1], depth=cfg.depth,
                                         rho0=cfg.rho0,
                                         sound_speed=cfg.sound_speed,
                                         gamma=cfg.gamma, gravity=cfg.gravity)
    rho_ideal = cfg.rho0 * cfg.gravity * (cfg.depth - pos[:, 1]).clamp_min(0.0)
    A, b = density_operator(state, cfg)
    mass = torch.linalg.solve(A, rho_eq - b)
    return {"mass": mass, "density_fraction": fraction,
            "rho_reference": rho_eq, "p_reference": rho_ideal}


def _evaluate(state, cfg: LakeConfig, mass: torch.Tensor):
    pos = state["pos"]
    return rhs(pos, torch.zeros_like(pos), mass, state["config"],
               pairs=state["pairs"], wall=state["wall"])


def residual_report(state, cfg: LakeConfig, mass: torch.Tensor) -> dict:
    """Acceleration residual of a mass field, split by component and by band."""
    rho, pressure, acc = _evaluate(state, cfg, mass)
    pos = state["pos"]
    vertical = acc[:, 1].abs() / cfg.gravity
    horizontal = acc[:, 0].abs() / cfg.gravity
    band = _interior_mask(pos, cfg)
    wall_band = _wall_band(pos, cfg)
    scale = float(pressure.max()) if float(pressure.max()) > 0 else 1.0
    return {
        "a_max_over_g": float(vertical.max()),
        "a_max_over_g_horizontal": float(horizontal.max()),
        "a_max_over_g_horizontal_interior": (
            float(horizontal[~wall_band].max()) if bool((~wall_band).any()) else None),
        "a_max_over_g_horizontal_wall_band": (
            float(horizontal[wall_band].max()) if bool(wall_band.any()) else None),
        "a_max_over_g_interior": float(vertical[band].max()) if bool(band.any()) else None,
        "rho_over_rho0_min": float((rho / cfg.rho0).min()),
        "rho_over_rho0_max": float((rho / cfg.rho0).max()),
        "p_max": float(pressure.max()),
        "p_over_p_max": float((pressure / scale).abs().max()),
        "density": rho,
        "pressure": pressure,
        "acceleration": acc,
    }


def _wall_band(pos: torch.Tensor, cfg: LakeConfig) -> torch.Tensor:
    """Particles within two smoothing lengths of either side wall."""
    band = 2.0 * cfg.h()
    return (pos[:, 0] < band) | (pos[:, 0] > cfg.width - band)


def _interior_mask(pos: torch.Tensor, cfg: LakeConfig) -> torch.Tensor:
    """Particles two smoothing lengths away from the bottom wall and the surface."""
    band = 2.0 * cfg.h()
    return (pos[:, 1] > band) & (pos[:, 1] < cfg.depth - band)


def reconstruct(state, cfg: LakeConfig, prior=None, verbose: bool = False) -> dict:
    """Solve for the particle masses that put the shipped operator exactly at rest.

    Damped Gauss-Newton on ``u = log(m / m_unit)``: the residual is the vertical
    acceleration ``a_y`` of every fluid particle, the unknown is the mirror-pair
    reduced mass field, and the step solves

    .. math::  \\begin{bmatrix} J \\\\ \\lambda I \\end{bmatrix} \\delta u =
               \\begin{bmatrix} -a_y \\\\ \\lambda (u_{prior} - u) \\end{bmatrix}

    so that the near-null space of the balance operator is filled by the
    physically meaningful member (the analytic profile) instead of by whatever
    roundoff the least-squares solve happens to pick.  A backtracking line search
    keeps every accepted step a descent step.

    Returns the state together with the residual history and diagnostics.
    """
    pos = state["pos"]
    n = pos.shape[0]
    mapping = state["columns"]
    unique = int(mapping.max().item()) + 1
    m_unit = state["m_unit"]
    prior = analytic_profile(state, cfg)["mass"] if prior is None else prior
    u_prior = torch.log(prior / m_unit)

    # Start from the prior: the mean of each mirror pair.
    sums = torch.zeros(unique, dtype=pos.dtype, device=pos.device)
    sums = sums.index_add(0, mapping, u_prior)
    counts = torch.zeros(unique, dtype=pos.dtype, device=pos.device)
    counts = counts.index_add(0, mapping, torch.ones_like(u_prior))
    u_unique = sums / counts
    u_prior_unique = u_unique.clone()

    def residual(u_unique: torch.Tensor) -> torch.Tensor:
        mass = m_unit * torch.exp(u_unique[mapping])
        return _evaluate_as_float(state, cfg, mass)

    history = []
    lam = cfg.prior_weight
    for it in range(cfg.solver_max_iter):
        f = residual(u_unique)
        worst = float(f.abs().max())
        merit = float(torch.linalg.vector_norm(f)) / math.sqrt(max(f.numel(), 1))
        history.append(worst)
        if verbose:
            print(f"  iter {it:2d}  max|a_y|/g = {worst:.3e}   rms|a_y|/g = {merit:.3e}")
        if worst < cfg.solver_tol:
            break
        jac = torch.autograd.functional.jacobian(residual, u_unique)
        rows = [jac, lam * torch.eye(unique, dtype=pos.dtype, device=pos.device)]
        rhs_vec = [-f, lam * (u_prior_unique - u_unique)]
        step = torch.linalg.lstsq(torch.cat(rows, dim=0),
                                  torch.cat(rhs_vec, dim=0)).solution
        alpha, base = 1.0, merit
        accepted = False
        for _ in range(30):
            candidate = u_unique + alpha * step
            candidate_merit = float(torch.linalg.vector_norm(
                residual(candidate))) / math.sqrt(max(f.numel(), 1))
            if candidate_merit < base:
                u_unique = candidate
                accepted = True
                break
            alpha *= 0.5
        if not accepted:                          # no descent: stop honestly
            history.append(worst)
            break
    mass = m_unit * torch.exp(u_unique[mapping])
    report = residual_report(state, cfg, mass)
    report.update({
        "mass": mass,
        "u_unique": u_unique,
        "iterations": len(history),
        "residual_history": history,
        "converged": bool(history and history[-1] < cfg.solver_tol),
        "prior": prior,
    })
    return report


def _evaluate_as_float(state, cfg: LakeConfig, mass: torch.Tensor) -> torch.Tensor:
    """Vertical acceleration only (what the solver zeroes), kept differentiable."""
    _rho, _p, acc = _evaluate(state, cfg, mass)
    return acc[:, 1]


def integrate_at_rest(state, cfg: LakeConfig, mass: torch.Tensor,
                      t_end: float | None = None) -> dict:
    """Advance the reconstructed state and measure how far it leaves rest.

    Symplectic Euler with the acoustic Courant step.  Because the reconstructed
    acceleration is at rounding level the velocity has to stay there; a run that
    grows it is a run where the reconstruction failed.  One ``rhs`` call per step:
    the density of the current positions is the state of the step that is taken.
    """
    pos = state["pos"].clone()
    vel = torch.zeros_like(pos)
    dt = cfg.dt()
    t_end = cfg.t_end if t_end is None else float(t_end)
    steps = int(round(t_end / dt))
    y0 = pos[:, 1].clone()
    rho_ref = _evaluate(state, cfg, mass)[0]
    band = _wall_band(pos, cfg)
    samples = []
    worst_speed = worst_rise = worst_rho = 0.0
    worst_speed_in = worst_rise_in = worst_drho_in = 0.0
    for step in range(steps):
        rho, _p, acc = rhs(pos, vel, mass, state["config"],
                           pairs=state["pairs"], wall=state["wall"])
        vel = vel + dt * acc
        pos = pos + dt * vel
        speed = float(vel.abs().max()) / cfg.sound_speed
        rise = float((pos[:, 1] - y0).abs().max()) / cfg.depth
        drho = float(((rho - rho_ref) / cfg.rho0).abs().max())
        worst_speed = max(worst_speed, speed)
        worst_rise = max(worst_rise, rise)
        worst_rho = max(worst_rho, drho)
        if bool((~band).any()):
            worst_speed_in = max(worst_speed_in, float(
                vel[~band].abs().max()) / cfg.sound_speed)
            worst_rise_in = max(worst_rise_in, float(
                (pos[~band, 1] - y0[~band]).abs().max()) / cfg.depth)
            worst_drho_in = max(worst_drho_in, float(
                ((rho[~band] - rho_ref[~band]) / cfg.rho0).abs().max()))
        if (step + 1) % cfg.sample_every == 0 or step == steps - 1:
            samples.append({"t": (step + 1) * dt,
                            "v_over_c0": float(vel.abs().max()) / cfg.sound_speed,
                            "dy_over_H": float((pos[:, 1] - y0).abs().max()) / cfg.depth,
                            "drho_over_rho0": float(((rho - rho_ref)
                                                     / cfg.rho0).abs().max())})
    return {
        "dt": dt, "steps": steps, "t_end": steps * dt,
        "max_v_over_c0": worst_speed, "max_dy_over_H": worst_rise,
        "max_drho_over_rho0": worst_rho, "samples": samples,
        "max_v_over_c0_interior": worst_speed_in,
        "max_dy_over_H_interior": worst_rise_in,
        "max_drho_over_rho0_interior": worst_drho_in,
        "wall_band_particles": int(band.sum()),
        "interior_particles": int((~band).sum()),
        "total_mass": float(mass.sum()),
        "centre_of_mass_y": float((mass * pos[:, 1]).sum() / mass.sum()),
        "centre_of_mass_y_initial": float((mass * y0).sum() / mass.sum()),
    }


def profile_error(state, cfg: LakeConfig, rho: torch.Tensor,
                  pressure: torch.Tensor) -> dict:
    """Reconstructed profiles against the two analytic references (interior band)."""
    pos = state["pos"]
    band = _interior_mask(pos, cfg)
    y = pos[band, 1]
    rho_eq = eos.eos_hydrostatic_density(y, depth=cfg.depth, rho0=cfg.rho0,
                                         sound_speed=cfg.sound_speed,
                                         gamma=cfg.gamma, gravity=cfg.gravity)
    p_eos = eos.eos_hydrostatic_pressure(y, depth=cfg.depth, rho0=cfg.rho0,
                                         sound_speed=cfg.sound_speed,
                                         gamma=cfg.gamma, gravity=cfg.gravity)
    p_incompressible = cfg.rho0 * cfg.gravity * (cfg.depth - y).clamp_min(0.0)
    p_scale = float(p_incompressible.max())
    rho_measured = rho[band]
    p_measured = pressure[band]
    return {
        "band_count": int(band.sum()),
        "band_y_min": float(y.min()) if y.numel() else None,
        "band_y_max": float(y.max()) if y.numel() else None,
        "density_vs_eos_max_rel": float(((rho_measured - rho_eq).abs()
                                         / rho_eq).max()) if y.numel() else None,
        "pressure_vs_eos_max_rel": float(((p_measured - p_eos).abs()
                                          / p_scale).max()) if y.numel() else None,
        "pressure_vs_rho0gh_max_rel": float(((p_measured - p_incompressible).abs()
                                             / p_scale).max()) if y.numel() else None,
        "eos_pressure_bias_at_bottom": float(
            (p_eos - p_incompressible).abs().max() / p_scale) if y.numel() else None,
        "row_profiles": _row_profiles(pos, cfg, rho, pressure, mass=None),
    }


def _row_profiles(pos, cfg: LakeConfig, rho, pressure, mass=None):
    rows = []
    depth_index = int(round(cfg.depth / cfg.dx))
    for k in range(depth_index):
        y = (k + 0.5) * cfg.dx
        sel = (pos[:, 1] - y).abs() < 1e-9
        if not bool(sel.any()):
            continue
        entry = {"y": y, "count": int(sel.sum()),
                 "rho_over_rho0": float(rho[sel].mean() / cfg.rho0),
                 "pressure": float(pressure[sel].mean())}
        if mass is not None:
            entry["mass_over_unit"] = float(mass[sel].mean())
            entry["mass_spread"] = float(mass[sel].max() / mass[sel].min() - 1.0)
        rows.append(entry)
    return rows


def run_case(cfg: LakeConfig, verbose: bool = False) -> dict:
    """Full case: build, reconstruct, verify the rest, and cross-check the physics."""
    torch_dtype = cfg.torch_dtype
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)

        controls = {}
        uniform_mass = torch.full((state["n_fluid"],), state["m_unit"],
                                  dtype=state["pos"].dtype,
                                  device=state["pos"].device)
        controls["uniform_lattice_mass"] = _control(state, cfg, uniform_mass)
        controls["analytic_hydrostatic_profile"] = _control(state, cfg,
                                                            prior["mass"])
        solution = reconstruct(state, cfg, prior=prior["mass"], verbose=verbose)
        mass = solution["mass"]

        # Linear-response control: scaling the reconstruction must scale the
        # residual, otherwise the near-zero equilibrium residual would be
        # indistinguishable from a dead operator.
        perturb = {}
        for eps in (1.0e-2, 1.0e-3, 1.0e-4):
            scaled = mass * (1.0 + eps)
            perturb[eps] = residual_report(state, cfg, scaled)["a_max_over_g"]

        run = integrate_at_rest(state, cfg, mass)
        profiles = profile_error(state, cfg, solution["density"],
                                 solution["pressure"])
        profiles["row_profiles"] = _row_profiles(state["pos"], cfg,
                                                 solution["density"],
                                                 solution["pressure"], mass=mass)

        return {
            "case": "CASE-SPH-LAKE-AT-REST",
            "reference": "analytic hydrostatic identity (u = 0, eta = const)",
            "geometry": {
                "width": cfg.width, "depth": cfg.depth, "dx": cfg.dx,
                "h": cfg.h(), "h_factor": cfg.h_factor,
                "particles": state["n_fluid"], "wall_particles": state["n_wall"],
                "pairs": int(state["pairs"].shape[0]),
                "sound_speed": cfg.sound_speed, "gamma": cfg.gamma,
                "gravity": cfg.gravity,
                "acoustic_mach": cfg.sound_speed / math.sqrt(cfg.gravity * cfg.depth),
            },
            "equilibrium": {
                "a_max_over_g": solution["a_max_over_g"],
                "a_max_over_g_horizontal": solution["a_max_over_g_horizontal"],
                "a_max_over_g_horizontal_interior":
                    solution["a_max_over_g_horizontal_interior"],
                "a_max_over_g_horizontal_wall_band":
                    solution["a_max_over_g_horizontal_wall_band"],
                "a_max_over_g_interior": solution["a_max_over_g_interior"],
                "iterations": solution["iterations"],
                "converged": solution["converged"],
                "residual_history": solution["residual_history"],
                "p_max": solution["p_max"],
                "rho_over_rho0_min": solution["rho_over_rho0_min"],
                "rho_over_rho0_max": solution["rho_over_rho0_max"],
                "mass_over_unit_min": float((mass / state["m_unit"]).min()),
                "mass_over_unit_max": float((mass / state["m_unit"]).max()),
            },
            "wall_model": wall_symmetry_report(cfg),
            "controls": controls,
            "perturbation_response": perturb,
            "rest": run,
            "profiles": profiles,
        }
    finally:
        torch.set_default_dtype(previous)


def _control(state, cfg: LakeConfig, mass: torch.Tensor) -> dict:
    report = residual_report(state, cfg, mass)
    return {"a_max_over_g": report["a_max_over_g"],
            "a_max_over_g_horizontal": report["a_max_over_g_horizontal"],
            "a_max_over_g_interior": report["a_max_over_g_interior"],
            "p_max": report["p_max"],
            "rho_over_rho0_min": report["rho_over_rho0_min"],
            "rho_over_rho0_max": report["rho_over_rho0_max"]}