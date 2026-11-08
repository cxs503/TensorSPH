"""Two-dimensional wedge water entry: Wagner (1932) reference + WCSPH impact model.

Physical scope
--------------
A rigid symmetric wedge with deadrise angle ``beta`` (angle between each face and
the undisturbed free surface, so ``beta = 90 deg`` is a flat bottom) enters calm
water at constant vertical velocity ``V``.  Gravity is switched off, which is the
assumption Wagner's theory is derived under: the only length scale is ``V t`` and
the problem is self-similar.

Wagner (1932) reference -- what is implemented
----------------------------------------------
The inner (jet) region is unresolved; the outer flow is the Kirchhoff flow
around a flat plate whose half-width ``c(t)`` grows in time.  Its potential on
the plate is ``phi = V sqrt(c^2 - x^2)``, so the Wagner condition (the plate
solution must meet the free surface at the edge) gives

    c(t) = pi V t / (2 tan beta)                                    (1)

With ``C = pi / (2 tan beta)`` and ``s = 1 / sqrt(1 - (x/c)^2)`` the unsteady
Bernoulli pressure on the body is

    p(x, t) = (rho V^2 / 2) [ 2 C s - s^2 + 1 ]                     (2)

The ``-s^2`` term is the steady part ``-(1/2) rho |grad phi|^2`` of the Kirchhoff
flow, ``2 C s`` is ``-rho (dc/dt) dphi/dc`` (growth of the wetted area), and the
``+1`` carries the O(rho V^2) stagnation level of the same expansion.  ``p`` is
largest where ``s = C``, i.e. immediately behind the spray root:

    p_max = (rho V^2 / 2) [ C^2 + 1 ]      for C >= 1
          = rho V^2 C                      for C <  1                 (3)

The branch exists because ``s >= 1`` on the wetted surface: for blunt wedges
(``beta > 60 deg``) the maximum of (2) sits at the stagnation point ``x = 0``.

Consequences used as machine-checkable identities (see tests):

* ``p(0, t) = rho V^2 C = pi rho V^2 / (2 tan beta)``: proportional to
  ``V^2 / tan beta`` and *time independent*;
* ``p_max`` (3) is proportional to ``1 / tan^2 beta`` for small ``beta``;
* at ``beta = 45 deg`` (``C = pi/2``) eq. (3) returns
  ``(rho V^2 / 2)(pi^2/4 + 1)``, the flat-plate Wagner value quoted in the
  campaign sheet.  Note ``beta = 90 deg`` is degenerate in this family
  (``tan beta -> inf`` gives ``c -> 0``); see docs/verification_wedge_entry.md.

SPH side
--------
Weakly compressible SPH with

* Tait equation of state ``p = B[(rho/rho0)^gamma - 1]``, ``B = rho0 c0^2 / gamma``,
* Monaghan artificial viscosity (``alpha``),
* Adami, Hu & Adams (2012) static/moving wall particles for the wedge and the
  tank floor/sides: the wall pressure is extrapolated from the fluid,
  ``p_w = sum_f p_f W / sum_f W`` (gravity and wall acceleration are zero here),
* density summation every step, calibrated lattice mass so a particle with full
  support starts at ``rho = rho0`` exactly.

The wedge translates rigidly, so its particles live in a body frame and are
re-positioned every step; neighbour pairs are rebuilt every step (the body
motion invalidates a cached Verlet list).

Measured quantities (all defined so the analytic side can be evaluated with the
same definition):

``c(t)``      outermost ``|x|`` of a wedge surface particle whose extrapolated
              wall pressure exceeds ``threshold`` times the dynamic pressure
              ``q = rho V^2 / 2`` (a pressure footprint, insensitive to the
              square-root divergence of (2) which SPH smears over ~2 dx).
``p_max``     largest wall pressure over the footprint, either over the whole
              footprint or restricted to ``|x| <= 0.9 c`` (the interior part
              where Wagner (2) is smooth).
``p_centre``  wall pressure at the surface particle nearest ``x = 0``.
``x_jet``     spray-root abscissa from the *free surface*: outermost ``|x|`` of a
              fluid particle within ``3 dx`` of the body line
              ``y = |x| tan beta - V t``.  Wagner puts it at ``c(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from .droplet_impact import cubic_spline_kernel, lattice_kernel_sum, pair_kernels
from .neighbor_search import cell_linked_pairs

__all__ = [
    "WedgeConfig",
    "WetWidth",
    "wagner_c_coeff",
    "wet_half_width",
    "wagner_pressure",
    "wagner_validity_tau",
    "wagner_pmax",
    "wagner_jet_root",
    "lattice_mass_unit",
    "wedge_lattice",
    "WedgeImpact",
    "gci",
    "fit_exponent",
    "sig6",
    "round_floats",
]


# --------------------------------------------------------------------------- #
# Wagner (1932) reference solution
# --------------------------------------------------------------------------- #

def wagner_c_coeff(beta: float) -> float:
    """``C = pi / (2 tan beta)``, i.e. ``c(t) = C V t``.  ``beta`` in radians."""
    return math.pi / (2.0 * math.tan(beta))


def wet_half_width(V: float, beta: float, t: float) -> float:
    """Wagner wetted half-width (1).  ``beta`` in radians, ``t`` in seconds."""
    return wagner_c_coeff(beta) * V * t


def wagner_validity_tau(beta: float) -> float:
    """Largest ``|x|/c`` at which the outer solution (2) still gives ``p >= 0``.

    ``2 C s - s^2 + 1`` vanishes at ``s* = C + sqrt(C^2 + 1)`` and is negative
    beyond it: the linearised outer solution predicts suction in the last
    fraction of a percent of the footprint (``s* = 5.6`` for ``beta = 30 deg``),
    where the real flow separates into the jet.  Beyond ``tau*`` the reference is
    set to zero, which is the same clamp the SPH side applies to the Tait
    pressure.  Without it the reference would be negative and could not be
    compared with a positive-definite SPH pressure field.
    """
    C = wagner_c_coeff(beta)
    s_star = C + math.sqrt(C * C + 1.0)
    return math.sqrt(max(0.0, 1.0 - 1.0 / (s_star * s_star)))


def wagner_pressure(x, c: float, V: float, beta: float, rho: float,
                    edge_cut: float | None = None):
    """Wagner pressure (2) at ``x`` for wetted half-width ``c``.

    ``x`` may be a float or a tensor; ``|x| >= c`` returns 0 (the point is dry).
    The footprint is truncated at ``min(edge_cut, wagner_validity_tau(beta))``,
    where the outer solution stops predicting a positive pressure.
    ``edge_cut`` is an optional tighter cap (used by the tests to show the
    sensitivity of the root value).
    """
    q = 0.5 * rho * V * V
    C = wagner_c_coeff(beta)
    tau_max = wagner_validity_tau(beta)
    if edge_cut is not None:
        tau_max = min(tau_max, float(edge_cut))
    if isinstance(x, torch.Tensor):
        if c <= 0:
            return torch.full_like(x, float("nan"))
        tau = x.abs() / c
        inside = tau <= tau_max
        s = 1.0 / torch.sqrt((1.0 - tau.clamp(max=tau_max) ** 2).clamp_min(1.0e-12))
        val = q * (2.0 * C * s - s * s + 1.0)
        return torch.where(inside, val, torch.zeros_like(x))
    if c <= 0.0:
        return float("nan")
    tau = abs(x) / c
    if tau > tau_max:
        return 0.0
    s = 1.0 / math.sqrt(max(1.0 - tau * tau, 1.0e-12))
    return q * (2.0 * C * s - s * s + 1.0)


def wagner_pmax(V: float, beta: float, rho: float) -> float:
    """Wagner peak pressure (3)."""
    C = wagner_c_coeff(beta)
    q = 0.5 * rho * V * V
    return q * (C * C + 1.0) if C >= 1.0 else 2.0 * q * C


def wagner_p_centre(V: float, beta: float, rho: float) -> float:
    """Stagnation-line pressure ``p(0) = rho V^2 C`` (time independent)."""
    return rho * V * V * wagner_c_coeff(beta)


def wagner_jet_root(V: float, beta: float, t: float) -> float:
    """Spray-root abscissa.  Wagner identifies it with the wetted edge ``c(t)``."""
    return wet_half_width(V, beta, t)


def wagner_rmse_profile(x, t: float, V: float, beta: float, rho: float) -> float:
    """RMS of (2) over the wetted footprint, used as a scalar profile metric."""
    c = wet_half_width(V, beta, t)
    if c <= 0.0:
        return 0.0
    p = wagner_pressure(x, c, V, beta, rho)
    return float(torch.sqrt(torch.nan_to_num(p, nan=0.0).square().mean()))


@dataclass
class WetWidth:
    """A wetted-half-width estimate together with its threshold band."""

    value: float
    low: float
    high: float


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class WedgeConfig:
    """Case definition for the SPH side.

    Length scale of the campaign ladder: ``dx = L / n`` with ``L`` the tank
    width ``2 * half_width`` (0.4 m by default) and
    ``n in {200, 300, 400, 800}``, i.e. ``dx = 2.0 .. 0.5 mm``.  That is the
    literal reading of the campaign's ``dx = L/200 .. L/800``; the wetted
    half-width at the end of the window is ``c_end = 50 mm``, so the finest
    grid puts 200 particles across the wetted width and the coarsest 50.

    ``c_end`` sets the observation window through (1): ``t_end = 2 c_end
    tan(beta) / (pi V)``.  Fixing ``c_end`` (rather than ``t_end``) keeps every
    case at the same number of particles across the wetted width, and keeps the
    run affordable -- see docs/verification_wedge_entry.md for why the literal
    ``t = 0.01 .. 0.1 s`` window at ``beta = 10 deg, V = 5 m/s`` (c = 4.4 m)
    cannot be run at SPH resolution.
    """

    beta_deg: float = 30.0
    velocity: float = 1.0
    rho0: float = 1000.0
    gamma: float = 7.0
    dx: float = 1.0e-3
    h_factor: float = 1.3
    sound_speed_factor: float = 20.0
    alpha_visc: float = 0.1
    half_width: float = 0.20
    depth: float = 0.12
    headroom: float = 0.08
    wall_layers: int = 3
    c_end: float = 0.05
    t_end: float | None = None
    frames: int = 21
    max_courant: float = 0.3
    body_span_factor: float = 1.9
    tolerance_dx: float = 1.0
    # A wall particle counts as wetted when it has at least this fraction of the
    # half-space kernel support of a surface particle.  Without a support floor a
    # single distant neighbour at the edge of the support radius makes the
    # extrapolated wall pressure equal to *that* neighbour's pressure, which
    # spreads the footprint over the whole body (measured: c jumped to the body
    # span at every instant).
    contact_support: float = 0.15
    # A particle stands for a cell of edge ``dx``: a surface node at ``x`` is
    # wetted out to ``x + dx cos(beta)/2`` and the free surface is half a spacing
    # above the topmost particle centre.  Applying that convention to the
    # measured geometry removes a bias of order one lattice spacing that is a
    # property of the discretisation, not of the flow.  Set False to see the raw
    # (biased-low) numbers; the report quotes both.
    cell_offset: bool = True
    # The wedge starts this many particle spacings above the free surface and
    # the clock is shifted so that t = 0 is the instant the apex touches.  Without
    # the gap the apex wall particles sit inside the kernel support of the top
    # fluid row at t = 0 and purely geometrically compress it by 5.6 %, which at
    # c0 = 20 V is a spurious 26 kPa -- the same order as the Wagner stagnation
    # pressure at beta = 10 deg.
    initial_gap_dx: float = 2.0
    device: str = "cpu"

    # ---- derived ---------------------------------------------------------- #
    @property
    def beta(self) -> float:
        return math.radians(self.beta_deg)

    @property
    def tan_beta(self) -> float:
        return math.tan(self.beta)

    def h(self) -> float:
        return self.h_factor * self.dx

    def sound_speed(self) -> float:
        return self.sound_speed_factor * self.velocity

    def resolved_t_end(self) -> float:
        if self.t_end is not None:
            return float(self.t_end)
        return 2.0 * self.c_end * self.tan_beta / (math.pi * self.velocity)

    def c_coeff(self) -> float:
        return wagner_c_coeff(self.beta)

    def body_span(self) -> float:
        return self.body_span_factor * self.c_end

    def validate(self) -> None:
        if not 1.0 <= self.beta_deg < 90.0:
            raise ValueError("beta_deg must be in [1, 90)")
        if self.velocity <= 0 or self.rho0 <= 0 or self.gamma <= 0:
            raise ValueError("velocity, rho0 and gamma must be positive")
        if self.dx <= 0 or self.h_factor <= 0:
            raise ValueError("dx and h_factor must be positive")
        if not 0.0 <= self.alpha_visc <= 1.0:
            raise ValueError("alpha_visc must be in [0, 1]")
        if self.wall_layers < 1:
            raise ValueError("wall_layers must be positive")
        if self.half_width <= 0 or self.depth <= 0:
            raise ValueError("half_width and depth must be positive")
        if self.body_span() > self.half_width:
            raise ValueError("body_span exceeds the tank half-width")
        if self.resolved_t_end() <= 0:
            raise ValueError("t_end must be positive")
        courant = self.dt_hint() * (self.sound_speed() + self.velocity) / self.h()
        if courant > self.max_courant * (1.0 + 1.0e-9):
            raise ValueError(
                f"dt={self.dt_hint():.3e} gives acoustic Courant {courant:.3f} > "
                f"max_courant {self.max_courant:g}")

    def dt_hint(self) -> float:
        return self.max_courant * self.h() / (self.sound_speed() + self.velocity)


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #

def lattice_mass_unit(rho0: float, dx: float, h: float) -> float:
    """Particle mass for which the lattice kernel sum is exactly ``rho0``.

    ``m = rho0 dx^2 / sum_lattice W``.  Without the divisor a particle in full
    support starts at ``rho/rho0 = 1 + b`` with ``b ~ 6e-3``; the Tait law turns
    that bias into a spurious background pressure independent of the case.  The
    same unit is used for wall particles so there is no density step at the wall
    face (see docs/verification_lattice_mass_calibration.md upstream).
    """
    total = lattice_kernel_sum(h, dx)      # sum_j W_ij over a full lattice
    if total <= 0:
        raise ValueError("degenerate lattice kernel sum")
    return rho0 / total


def wedge_lattice(cfg: WedgeConfig):
    """Body-frame particles of the wedge, first surface layer flagged.

    Points are laid out on an orthogonal lattice in the face frame
    ``(s, n)``: ``s`` along the face from the apex, ``n`` into the body, with
    ``n = (j + 0.5) dx`` so the outermost row sits half a spacing inside the
    surface, which is what gives the neighbouring fluid full kernel support.
    Only the first ``wall_layers`` rows are generated: the fluid never reaches
    deeper, and the extra rows are what the Adami wall pressure needs.

    Returns ``(local_positions (N,2), is_first_layer (N,))`` in the body frame
    whose origin is the apex.
    """
    dx = cfg.dx
    beta = cfg.beta
    span = cfg.body_span()
    layers = cfg.wall_layers
    n_s = int(math.floor(span / (dx * math.cos(beta)) + 1.0e-9))
    if n_s < 1:
        raise ValueError("body_span is too small for dx")
    s = (torch.arange(n_s, dtype=torch.float64) + 0.5) * dx
    parts = []
    first = []
    for sign in (1.0, -1.0):
        for j in range(layers):
            n_in = (j + 0.5) * dx
            x = sign * (s * math.cos(beta) - n_in * math.sin(beta))
            y = s * math.sin(beta) + n_in * math.cos(beta)
            parts.append(torch.stack((x, y), dim=-1))
            first.append(torch.full((n_s,), j == 0))
    return torch.cat(parts, dim=0), torch.cat(first, dim=0)


def _tank_lattice(cfg: WedgeConfig):
    """Static floor and side-wall particles, laid outside the fluid box.

    They exist only so that the far-field fluid has a boundary; at
    ``half_width = 0.2 m`` and ``c_end = 0.05 m`` they are outside the support of
    any particle that matters during the window, which is checked in
    ``docs/verification_wedge_entry.md`` (tank-size independence).
    """
    dx = cfg.dx
    layers = cfg.wall_layers
    # The fluid lattice is cell-centred, so its outermost column sits half a
    # spacing inside the box; the first wall row therefore goes half a spacing
    # *outside* it, one spacing from that column, which is the convention the
    # repository's static-wall cases use.  Placing it a full spacing further out
    # (the first version of this function) left the bottom fluid row with 0.58 of
    # its support and a spurious vacuum layer above the floor.
    # columns exactly symmetric about x = 0: the case is symmetric, and an
    # off-centre floor row leaks a measurable asymmetry into the wall pressure
    k_max = int(math.floor((cfg.half_width + (layers - 0.5) * dx) / dx + 1.0e-9))
    xs = torch.arange(-k_max, k_max + 1, dtype=torch.float64) * dx
    ys = torch.arange(-cfg.depth - (layers - 0.5) * dx,
                      cfg.headroom + 0.5 * dx, dx, dtype=torch.float64)
    parts = []
    for layer in range(1, layers + 1):
        y = -cfg.depth - (layer - 0.5) * dx          # below the floor
        gx, _ = torch.meshgrid(xs, ys[:1], indexing="xy")
        parts.append(torch.stack((gx.reshape(-1), torch.full((gx.numel(),), y,
                                                             dtype=torch.float64)),
                                 dim=-1))
    # side columns start at the floor level: the bottom rows already cover the
    # corners, and a column that also runs below the floor puts two wall
    # particles at the same place, which over-densifies the corner fluid
    # particle by 1.8x and launches the run
    ys_side = torch.arange(-cfg.depth + 0.5 * dx, cfg.headroom + 0.5 * dx, dx,
                           dtype=torch.float64)
    for layer in range(1, layers + 1):
        for x in (-cfg.half_width - (layer - 0.5) * dx,
                  cfg.half_width + (layer - 0.5) * dx):
            parts.append(torch.stack((torch.full_like(ys_side, x), ys_side), dim=-1))
    return torch.cat(parts, dim=0)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

class WedgeImpact:
    """WCSPH model of a rigid wedge entering a rectangular tank at constant V."""

    def __init__(self, cfg: WedgeConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = torch.float64
        self.dx = float(cfg.dx)
        self.h = float(cfg.h())
        self.rho0 = float(cfg.rho0)
        self.V = float(cfg.velocity)
        self.beta = cfg.beta
        self.tan_beta = cfg.tan_beta
        self.c0 = cfg.sound_speed()
        self.B = self.rho0 * self.c0 ** 2 / cfg.gamma
        self.mass_unit = lattice_mass_unit(self.rho0, self.dx, self.h)
        self.support_ref = lattice_kernel_sum(self.h, self.dx)
        self.q = 0.5 * self.rho0 * self.V * self.V   # dynamic pressure scale

        self.fluid_pos, self.fluid_vel = self._initial_fluid()
        self.n_fluid = self.fluid_pos.shape[0]
        self.fluid_mass = torch.full((self.n_fluid,), self.mass_unit,
                                     dtype=self.dtype, device=self.device)

        local, first = wedge_lattice(cfg)
        self.wedge_local = local.to(self.device)
        self.is_first = first.to(self.device).bool()
        self.n_wedge = self.wedge_local.shape[0]
        tank = _tank_lattice(cfg).to(self.device)
        self.n_tank = tank.shape[0]
        # The tank is static: only the wedge translates.  Both live in one
        # concatenated array so that a single pair search covers them, but the
        # rigid-body offset must be applied to the wedge alone -- applying it to
        # the whole array (the first version) dragged the floor down at V with
        # the body, which starved the bottom rows of support and over-densified
        # the corner fluid by 1.75x.
        self.tank_local = tank
        self.wedge_world = self.wedge_local.clone()
        self.wall_local = torch.cat((self.wedge_local, tank), dim=0)
        self.wall_vel_body = torch.zeros_like(self.wall_local)
        self.wall_vel_body[: self.n_wedge, 1] = -self.V
        self.wall_mass = torch.full((self.wall_local.shape[0],), self.mass_unit,
                                    dtype=self.dtype, device=self.device)

        self.gap = cfg.initial_gap_dx * self.dx
        self.t_contact = self.gap / self.V
        self.time = -self.t_contact          # t = 0 is first contact
        self.body_offset = torch.tensor((0.0, self.gap), dtype=self.dtype,
                                        device=self.device)
        self.wedge_world = self.wedge_local + self.body_offset
        self.wall_pos = torch.cat((self.wedge_world, self.tank_local), dim=0)
        self.max_rho_ratio = 0.0
        self.max_speed = 0.0

    # ---- initial state ---------------------------------------------------- #
    def _initial_fluid(self):
        cfg = self.cfg
        dx = self.dx
        width = 2.0 * cfg.half_width
        nx = int(math.floor(width / dx + 1.0e-9))
        ny = int(math.floor(cfg.depth / dx + 1.0e-9))
        if nx < 4 or ny < 4:
            raise ValueError("domain disretization produced too few particles")
        # symmetric about x = 0: floor(width/dx) is usually even, so starting half
        # a spacing inside one wall leaves the lattice off-centre by up to half a
        # spacing, which biases the impact flow sideways (measured: 1 mm mean
        # drift before contact at dx = 3 mm)
        x = (torch.arange(nx, dtype=self.dtype, device=self.device)
             - 0.5 * (nx - 1)) * dx
        y = -cfg.depth + (torch.arange(ny, dtype=self.dtype, device=self.device) + 0.5) * dx
        gx, gy = torch.meshgrid(x, y, indexing="xy")
        pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
        vel = torch.zeros_like(pos)
        return pos, vel

    # ---- kernels / pairs -------------------------------------------------- #
    def all_positions(self):
        return torch.cat((self.fluid_pos, self.wall_pos), dim=0)

    def _pairs(self):
        pairs = cell_linked_pairs(self.all_positions(), 2.0 * self.h)
        if pairs.numel() == 0:
            return pairs
        i, j = pairs.unbind(dim=1)
        keep = i < self.n_fluid          # drop wall-wall pairs
        return pairs[keep]

    # ---- right-hand side -------------------------------------------------- #
    def _evaluate(self):
        """Density, pressure, acceleration and wall state for the current state."""
        cfg = self.cfg
        h = self.h
        n = self.n_fluid
        pos_all = self.all_positions()
        vel_all = torch.cat((self.fluid_vel, self.wall_vel_body), dim=0)
        self_kernel = float(cubic_spline_kernel(
            torch.zeros(1, dtype=self.dtype, device=self.device), h)[0])
        m = self.mass_unit
        pairs = self._pairs()
        i, j = pairs.unbind(dim=1)
        delta = pos_all[i] - pos_all[j]
        w, gw = pair_kernels(delta, h)

        rho_fluid = torch.full((n,), m * self_kernel, dtype=self.dtype,
                               device=self.device)
        ff = j < n
        rho_fluid = rho_fluid.index_add(0, i[ff], m * w[ff])
        rho_fluid = rho_fluid.index_add(0, j[ff], m * w[ff])
        fw = ~ff
        rho_fluid = rho_fluid.index_add(0, i[fw], m * w[fw])
        rho_fluid = rho_fluid.clamp_min(0.1 * self.rho0)

        pressure = self.B * ((rho_fluid / self.rho0) ** cfg.gamma - 1.0)
        pressure = pressure.clamp_min(0.0)

        # ---- wall pressure: Adami et al. (2012) eq. (11), g = a_w = 0 ---- #
        n_wall = self.wall_local.shape[0]
        num = torch.zeros(n_wall, dtype=self.dtype, device=self.device)
        den = torch.zeros(n_wall, dtype=self.dtype, device=self.device)
        wj = j[fw] - n
        num = num.index_add(0, wj, pressure[i[fw]] * w[fw])
        den = den.index_add(0, wj, w[fw])
        p_wall = torch.where(den > 1.0e-30, num / den.clamp_min(1.0e-30),
                             torch.zeros_like(num))
        rho_wall = self.rho0 * (p_wall / self.B + 1.0) ** (1.0 / cfg.gamma)

        # ---- momentum ------------------------------------------------------ #
        fluid_j = j.clamp_max(n - 1)
        wall_j = (j - n).clamp_min(0)
        rho_pair = torch.where(j < n, rho_fluid[fluid_j], rho_wall[wall_j])
        p_pair = torch.where(j < n, pressure[fluid_j], p_wall[wall_j])
        rho_i = rho_fluid[i]
        p_i = pressure[i]

        dv = vel_all[i] - vel_all[j]
        mu = h * (dv * delta).sum(dim=-1) / (delta.square().sum(dim=-1) + 0.01 * h * h)
        mu = torch.where(mu < 0.0, mu, torch.zeros_like(mu))
        rho_bar = 0.5 * (rho_i + rho_pair)
        pi_ij = -cfg.alpha_visc * 0.5 * (self.c0 + self.c0) * mu / rho_bar

        bracket = p_i / rho_i.square() + p_pair / rho_pair.square() + pi_ij
        contrib = (m * bracket).unsqueeze(-1) * gw
        acc = torch.zeros((n, 2), dtype=self.dtype, device=self.device)
        acc = acc.index_add(0, i[ff], -contrib[ff])
        acc = acc.index_add(0, j[ff], contrib[ff])
        acc = acc.index_add(0, i[fw], -contrib[fw])
        return {
            "rho": rho_fluid,
            "pressure": pressure,
            "acc": acc,
            "p_wall": p_wall,
            "den": den,
            "pairs": pairs,
        }

    # ---- time stepping ---------------------------------------------------- #
    def dt(self) -> float:
        speed = float(self.fluid_vel.abs().max()) if self.fluid_vel.numel() else 0.0
        return self.cfg.max_courant * self.h / (self.c0 + speed)

    def advance(self, dt: float) -> None:
        state = self._evaluate()
        self.fluid_vel = self.fluid_vel + dt * state["acc"]
        self.fluid_pos = self.fluid_pos + dt * self.fluid_vel
        self.time += dt
        self.body_offset = self.body_offset - torch.tensor(
            (0.0, dt * self.V), dtype=self.dtype, device=self.device)
        self.wedge_world = self.wedge_local + self.body_offset
        self._clamp()
        self.wall_pos = torch.cat((self.wedge_world, self.tank_local), dim=0)
        self.max_rho_ratio = max(
            self.max_rho_ratio, float((state["rho"] / self.rho0).max()))
        if self.fluid_vel.numel():
            self.max_speed = max(self.max_speed, float(self.fluid_vel.abs().max()))
        if not torch.isfinite(self.fluid_pos).all():
            raise FloatingPointError("wedge-entry run produced non-finite positions")

    def _clamp(self) -> None:
        """Safety net: keep particles in the box, kill the crossing normal velocity.

        With wall particles present a non-zero count means the wall model was
        bypassed.  The top of the box is open (headroom); particles reaching it
        are stopped rather than deleted, which is a documented artefact.
        """
        cfg = self.cfg
        tol = 0.5 * self.dx
        x_lo, x_hi = -cfg.half_width, cfg.half_width
        y_lo, y_hi = -cfg.depth, cfg.headroom
        p, v = self.fluid_pos, self.fluid_vel
        for dim, lo, hi in ((0, x_lo, x_hi), (1, y_lo, y_hi)):
            low = p[:, dim] < lo - tol
            high = p[:, dim] > hi + tol
            p[:, dim] = p[:, dim].clamp(lo, hi)
            v[:, dim] = torch.where(low | high, torch.zeros_like(v[:, dim]), v[:, dim])

    # ---- diagnostics ------------------------------------------------------ #
    def body_line(self, x: torch.Tensor) -> torch.Tensor:
        """World ``y`` of the body surface at abscissa ``x`` (apex at -V t)."""
        return x.abs() * self.tan_beta - self.V * self.time

    def diagnostics(self, state=None, thresholds=(0.005, 0.02, 0.10)) -> dict:
        if state is None:
            state = self._evaluate()
        cfg = self.cfg
        p_wall = state["p_wall"][: self.n_wedge]
        den = state["den"][: self.n_wedge]
        surf = self.is_first
        x = (self.wedge_local[:, 0] + self.body_offset[0])[surf]
        p = p_wall[surf]
        contact = den[surf] > cfg.contact_support * self.support_ref

        # ---- wetted half-width from the pressure footprint --------------- #
        thresholds = tuple(sorted(thresholds))
        widths = []
        for thr in thresholds:
            mask = contact & (p > thr * self.q)
            widths.append(float(x.abs()[mask].max()) if bool(mask.any()) else 0.0)
        c_mid = widths[len(widths) // 2]
        if cfg.cell_offset:
            dx_node = 0.5 * self.dx * math.cos(self.beta)
            widths = [w + dx_node if w > 0 else 0.0 for w in widths]
            c_mid += dx_node
        wet = WetWidth(value=c_mid, low=widths[0], high=widths[-1])

        # ---- peak pressure: whole footprint and its smooth interior ------ #
        foot = contact & (x.abs() <= wet.value * 1.001) & (wet.value > 0)
        p_foot = float(p[foot].max()) if bool(foot.any()) else 0.0
        inner = foot & (x.abs() <= 0.9 * wet.value)
        p_int = float(p[inner].max()) if bool(inner.any()) else 0.0
        near_axis = contact & (x.abs() <= 2.0 * self.dx)
        if bool(near_axis.any()):
            p_centre = float(p[near_axis].mean())
        else:
            p_centre = 0.0

        # ---- spray root from the free surface ---------------------------- #
        pos = self.fluid_pos
        line = self.body_line(pos[:, 0])
        band = cfg.tolerance_dx * self.dx
        near = (pos[:, 1] >= line - band) & (pos[:, 1] <= line + band)
        x_jet = float(pos[:, 0].abs()[near].max()) if bool(near.any()) else 0.0
        if cfg.cell_offset and x_jet > 0.0:
            x_jet += 0.5 * self.dx

        return {
            "t": self.time,
            "c": wet.value,
            "c_low": wet.low,
            "c_high": wet.high,
            "p_max_footprint": p_foot,
            "p_max_interior": p_int,
            "p_centre": p_centre,
            "x_jet": x_jet,
            "n_fluid": self.n_fluid,
            "n_wedge_surface": int(surf.sum().item()),
            "max_rho_ratio": float((state["rho"] / self.rho0).max()),
            "max_speed": float(self.fluid_vel.abs().max()) if self.fluid_vel.numel() else 0.0,
        }

    def free_surface_profile(self, bins: int = 80) -> dict:
        """Free-surface elevation ``eta(x)`` from the topmost fluid particle per bin."""
        cfg = self.cfg
        pos = self.fluid_pos
        edges = torch.linspace(-cfg.half_width, cfg.half_width, bins + 1,
                               dtype=self.dtype, device=self.device)
        idx = torch.bucketize(pos[:, 0], edges) - 1
        keep = (idx >= 0) & (idx < bins)
        eta = torch.full((bins,), -float("inf"), dtype=self.dtype, device=self.device)
        if bool(keep.any()):
            eta = eta.index_reduce(0, idx[keep], pos[keep, 1], "amax",
                                   include_self=True)
        if cfg.cell_offset:
            eta = eta + 0.5 * self.dx
        eta = torch.where(torch.isfinite(eta), eta,
                          torch.full_like(eta, float("nan")))
        centres = 0.5 * (edges[:-1] + edges[1:])
        return {"x": centres.cpu().tolist(), "eta": eta.cpu().tolist()}

    # ---- driver ----------------------------------------------------------- #
    def run(self, frames: int | None = None, t_end: float | None = None) -> list[dict]:
        cfg = self.cfg
        t_end = cfg.resolved_t_end() if t_end is None else float(t_end)
        frames = cfg.frames if frames is None else int(frames)
        times = torch.linspace(0.0, t_end, frames, dtype=self.dtype).tolist()
        steps = 0
        max_steps = 10_000_000
        while self.time < times[0] - 1.0e-15 and steps < max_steps:
            self.advance(min(self.dt(), times[0] - self.time))
            steps += 1
        out = [dict(self.diagnostics(), t=self.time)]
        target = 1
        while target < len(times) and steps < max_steps:
            dt = times[target] - self.time
            if dt <= 0:
                target += 1
                continue
            self.advance(min(self.dt(), dt))
            steps += 1
            if self.time >= times[target] - 1.0e-15:
                out.append(dict(self.diagnostics(), t=self.time))
                target += 1
        self.steps = steps
        return out


# --------------------------------------------------------------------------- #
# Numerics helpers used by the benchmark and the tests
# --------------------------------------------------------------------------- #

def gci(fine: float, coarse: float, ratio: float, order: float = 1.0) -> float:
    """Grid convergence index (Roache) of a fine/coarse pair, ``ratio = dx_c/dx_f``.

    ``GCI = Fs |e| / (r^p - 1)`` with the safety factor ``Fs = 1.25`` and the
    observed order ``p`` (default 1: first-order upwind-like behaviour is what
    the SPH pressure field shows).
    """
    if fine == 0.0:
        return float("inf")
    e = abs(coarse - fine) / abs(fine)
    return 1.25 * e / (ratio ** order - 1.0)


def fit_exponent(xs, ys) -> tuple[float, float]:
    """Least-squares fit of ``log y = a log x + b``; returns ``(a, r2)``."""
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
             if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    n = float(len(pairs))
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    den = n * sxx - sx * sx
    if abs(den) < 1.0e-18:
        return float("nan"), float("nan")
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    ss_tot = sum((p[1] - sy / n) ** 2 for p in pairs)
    ss_res = sum((p[1] - (a * p[0] + b)) ** 2 for p in pairs)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, r2


def sig6(value) -> float | None:
    """Round to 6 significant digits (the campaign's artefact format)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(v):
        return None
    if v == 0.0:
        return 0.0
    return float(f"{v:.6g}")


def round_floats(obj):
    """Recursively round every float in ``obj`` to 6 significant digits."""
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return sig6(obj)
    return obj


def write_json(path: str | Path, payload: dict) -> Path:
    """Write ``payload`` with every float rounded to 6 significant digits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_floats(payload), indent=2, sort_keys=False) + "\n")
    return path
