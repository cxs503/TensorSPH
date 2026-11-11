"""A small, explicit WCSPH solver used by the verification chain.

Why this exists next to ``tensorsph.droplet_impact``: the chain in this node is
required to exercise *specific* numerical ingredients that the production
solver does not carry -- a quintic-spline / Gaussian kernel pair, Monaghan
artificial viscosity with an explicit ``alpha``, an adaptive smoothing length
and a mirror-particle boundary -- and to do so in 2-D and 3-D with the same
code path.  Re-implementing the scheme on purpose keeps each ingredient
individually testable and keeps the production solver untouched; the two are
cross-checked on the dam-break front (see ``docs/benchmarks/wcsph-chain.md``).

Scheme (all standard; the citations are the ones implemented):

* **density** -- summation form ``rho_i = sum_j m_j W_ij + m_i W(0, h_i)``
  (Monaghan 1992, eq. 2.8), optional Shepard renormalisation of the truncation.
* **momentum** -- symmetric pressure form
  ``a_i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W_ij`` (Monaghan 1992,
  eq. 2.10).  The pair contribution is antisymmetric under ``i <-> j``, so the
  discrete total momentum is conserved to round-off by construction.
* **EOS** -- Tait: ``p = B[(rho/rho0)^gamma - 1]``, ``B = rho0 c0^2 / gamma``,
  ``gamma = 7`` (Monaghan 1994), clamped at zero on the free surface.
* **artificial viscosity** -- Monaghan & Gingold (1983) with the quadratic
  ``beta`` term:
  ``Pi_ij = (-alpha c_ij mu_ij + beta mu_ij^2) / rho_ij`` when
  ``v_ij . r_ij < 0``, ``mu_ij = h (v_ij . r_ij) / (r_ij^2 + eta^2)``,
  ``eta^2 = 0.01 h^2``.
* **boundary** -- either static *dynamic boundary particles* (Crespo, Gomez-Gesteira
  & Dalrymple 2007: wall particles carry mass, get a summation density, a Tait
  pressure clamped at zero, and are never integrated) or *mirror particles*
  (Colagrossi & Landrini 2003) about nearby flat walls.
* **time integration** -- kick-drift-kick leapfrog with one force evaluation
  per step, and a CFL step limit
  ``dt <= cfl * h / (c0 + max|v|)``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..neighbor_search import PairListCache
from . import kernels


@dataclass
class WCSPHConfig:
    """Configuration of the verification solver (SI units)."""

    dim: int = 2
    dx: float = 0.01
    h_factor: float = 1.3
    kernel: str = "quintic"
    rho0: float = 1000.0
    c0: float = 25.0
    gamma: float = 7.0
    alpha_av: float = 0.1
    beta_av: float = 0.0
    viscosity: float = 0.0
    gravity: float = 9.81
    cfl: float = 0.25
    integrator: str = "leapfrog"
    boundary: str = "dynamic"       # "dynamic" | "mirror" | "none"
    h_adaptive: bool = False
    density_renormalisation: bool = False
    #: scale the initial particle masses so that the summation density of the
    #: starting lattice is exactly ``rho0`` (2 fixed-point sweeps).  Without it
    #: the lattice defect plus the boundary neighbours start the run with a
    #: pressure step of several kPa, which then rings through the whole case.
    density_calibration: bool = True
    #: apparent horizontal acceleration used to force the sloshing tank in its
    #: own frame: ``a_x(t) = -amplitude * omega^2 * sin(omega t + phase)``.
    excitation_amplitude: float = 0.0
    excitation_omega: float = 0.0
    excitation_phase: float = 0.0
    device: str = "cpu"

    @property
    def h0(self) -> float:
        return self.h_factor * self.dx

    @property
    def support(self) -> float:
        return kernels.SUPPORT_FACTOR[self.kernel] * self.h0

    @property
    def bulk_modulus(self) -> float:
        return self.rho0 * self.c0**2 / self.gamma

    def validate(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if self.h_factor <= 0 or self.dx <= 0:
            raise ValueError("dx and h_factor must be positive")
        if self.c0 <= 0 or self.rho0 <= 0 or self.gamma <= 0:
            raise ValueError("c0, rho0 and gamma must be positive")
        if not 0.0 <= self.alpha_av <= 1.0:
            raise ValueError("alpha_av must be in [0, 1]")
        if self.beta_av < 0.0 or self.viscosity < 0.0:
            raise ValueError("beta_av and viscosity must be non-negative")
        if self.kernel not in kernels.SUPPORT_FACTOR:
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.boundary not in ("dynamic", "mirror", "none"):
            raise ValueError(f"unknown boundary {self.boundary!r}")
        if self.integrator not in ("leapfrog",):
            raise ValueError(f"unknown integrator {self.integrator!r}")
        if not 0.0 < self.cfl <= 0.5:
            raise ValueError("cfl must be in (0, 0.5]")
        if self.excitation_amplitude < 0 or self.excitation_omega < 0:
            raise ValueError("excitation amplitude and frequency must be non-negative")


@dataclass
class State:
    """Fluid particles, followed (in the pair list) by the wall particles."""

    pos: torch.Tensor
    vel: torch.Tensor
    mass: torch.Tensor
    rho: torch.Tensor
    pressure: torch.Tensor
    h: torch.Tensor
    wall_pos: torch.Tensor | None = None
    wall_mass: torch.Tensor | None = None
    wall_rho: torch.Tensor | None = None
    wall_pressure: torch.Tensor | None = None
    acc: torch.Tensor | None = None
    time: float = 0.0
    steps: int = 0
    max_speed: float = 0.0
    max_compression: float = 0.0
    max_courant: float = 0.0
    has_wall: bool = False

    @property
    def n_fluid(self) -> int:
        return int(self.pos.shape[0])

    @property
    def n_wall(self) -> int:
        return 0 if self.wall_pos is None else int(self.wall_pos.shape[0])


def tait_pressure(rho: torch.Tensor, cfg: WCSPHConfig) -> torch.Tensor:
    """``p = B[(rho/rho0)^gamma - 1]`` clamped at zero (free-surface clause)."""
    p = cfg.bulk_modulus * ((rho / cfg.rho0).clamp_min(0.0).pow(cfg.gamma) - 1.0)
    return p.clamp_min(0.0)


def sound_speed_for(max_speed: float, factor: float = 10.0) -> float:
    """Weak-compressibility requirement ``c0 >= factor * max|v|`` (with a floor)."""
    return max(factor * float(max_speed), 1.0)


def stable_dt(cfg: WCSPHConfig, max_speed: float) -> float:
    """CFL-limited step: acoustic, gravity and viscous branches."""
    h = cfg.h0
    dt = cfg.cfl * h / (cfg.c0 + float(max_speed))
    if cfg.gravity:
        dt = min(dt, 0.25 * math.sqrt(h / abs(cfg.gravity)))
    if cfg.viscosity > 0.0:
        dt = min(dt, 0.125 * h * h / cfg.viscosity)
    return dt


def _lattice_kernel_sum(cfg: WCSPHConfig) -> float:
    """``sum_j W`` over a complete lattice at spacing ``dx`` (renormalisation)."""
    rng = int(math.floor(kernels.SUPPORT_FACTOR[cfg.kernel] * cfg.h_factor)) + 1
    coords = torch.arange(-rng, rng + 1, dtype=torch.float64) * cfg.dx
    if cfg.dim == 2:
        gx, gy = torch.meshgrid(coords, coords, indexing="xy")
        pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), -1)
    else:
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="xy")
        pts = torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), -1)
    r = torch.linalg.vector_norm(pts, dim=1)
    return float(kernels.kernel_value(r, float(cfg.h0), cfg.dim, cfg.kernel).sum())


class Solver:
    """Owns the static wall, the pair-list cache and the integration loop."""

    def __init__(self, cfg: WCSPHConfig, pos: torch.Tensor, vel: torch.Tensor,
                 mass: torch.Tensor, wall_pos: torch.Tensor | None = None,
                 box: tuple[float, float, float, float] | None = None,
                 box3: tuple[float, float, float, float, float, float] | None = None,
                 mirror_planes: list[tuple[int, float]] | None = None,
                 dtype=torch.float64):
        cfg.validate()
        self.cfg = cfg
        self.dtype = dtype
        self.box = box
        self.box3 = box3
        self.state = State(
            pos=pos.to(dtype), vel=vel.to(dtype), mass=mass.to(dtype),
            rho=torch.full((pos.shape[0],), cfg.rho0, dtype=dtype),
            pressure=torch.zeros(pos.shape[0], dtype=dtype),
            h=torch.full((pos.shape[0],), float(cfg.h0), dtype=dtype),
        )
        self.mirror_planes = list(mirror_planes) if mirror_planes else None
        if cfg.boundary == "mirror" and not self.mirror_planes:
            raise ValueError("boundary='mirror' needs at least one wall plane")
        if box is not None and not (box[0] < box[1] and box[2] < box[3]):
            raise ValueError(f"box must be (xmin, xmax, ymin, ymax), got {box}")
        if box3 is not None and not (box3[0] < box3[1] and box3[2] < box3[3]
                                     and box3[4] < box3[5]):
            raise ValueError(f"box3 must be (xmin, xmax, ymin, ymax, zmin, zmax), "
                             f"got {box3}")
        if wall_pos is not None and wall_pos.numel():
            self.state.wall_pos = wall_pos.to(dtype)
            self.state.wall_mass = torch.full(
                (wall_pos.shape[0],), cfg.rho0 * cfg.dx ** cfg.dim, dtype=dtype)
            self.state.wall_rho = torch.full((wall_pos.shape[0],), cfg.rho0,
                                             dtype=dtype)
            self.state.wall_pressure = torch.zeros(wall_pos.shape[0], dtype=dtype)
            self.state.has_wall = True
        if cfg.density_calibration:
            self.calibrate_mass()
        cutoff = cfg.support * (1.25 if cfg.h_adaptive else 1.0)
        self.cache = PairListCache(cutoff, 0.25 * cfg.h0,
                                   fluid_count=self.state.n_fluid)
        self.state.acc = None

    # -- helpers ---------------------------------------------------------- #
    def calibrate_mass(self, sweeps: int = 2) -> None:
        """Rescale *both* the fluid and the wall mass to remove the lattice bias.

        Density is linear in mass for a fixed neighbour set, so a global factor
        is enough -- and a *global* factor matters: calibrating only the fluid
        would leave the wall particles carrying ``rho0 dx^d`` while the fluid
        carries ~0.5 % less, and the Tait law turns that 0.5 % step at the wall
        face into a kilopascal of spurious pressure.  The factor is taken from
        the median interior density, so the free-surface deficit (which is not
        a lattice defect) does not enter it.
        """
        for _ in range(sweeps):
            probe = PairListCache(self.cfg.support, 0.0,
                                  fluid_count=self.state.n_fluid)
            pos, _, _, _, _ = self._assemble()
            self._density_only(probe.get(pos)[0])
            rho = self.state.rho
            interior = rho > 0.95 * self.cfg.rho0
            if not bool(interior.any()):
                return
            scale = self.cfg.rho0 / float(rho[interior].median())
            self.state.mass = self.state.mass * scale
            if self.state.wall_mass is not None:
                self.state.wall_mass = self.state.wall_mass * scale
        self.state.acc = None

    def apply_hydrostatic_mass(self, surface: float) -> None:
        """Tait-consistent hydrostatic mass profile (``rho = rho0 (1 + p/B)^(1/gamma)``).

        A column that starts at uniform density starts with zero pressure
        everywhere and then has to fall into equilibrium, which launches an
        acoustic transient that rings for a long time in a closed tank.  Setting
        the mass from the analytic hydrostatic profile removes that transient
        and leaves the free surface where it belongs.
        """
        cfg = self.cfg
        if surface is None or not cfg.gravity:
            return
        depth = (surface - self.state.pos[:, 1]).clamp_min(0.0)
        p = cfg.rho0 * cfg.gravity * depth
        factor = (p / cfg.bulk_modulus + 1.0).pow(1.0 / cfg.gamma)
        self.state.mass = self.state.mass * factor
        self.state.acc = None

    def _density_only(self, pairs: torch.Tensor) -> None:
        """Summation density without touching the pressure/acceleration state."""
        cfg, st = self.cfg, self.state
        pos, _, mass, h, _ = self._assemble()
        n = st.n_fluid
        rho = mass * kernels.self_contribution(h, cfg.dim, cfg.kernel)
        if pairs.numel():
            i, j = pairs.unbind(dim=1)
            w, _ = kernels.kernel_pairs(pos[i] - pos[j], h[i], cfg.dim, cfg.kernel)
            rho = rho.index_add(0, i, mass[j] * w)
            rho = rho.index_add(0, j, mass[i] * w)
        st.rho = rho[:n].clamp_min(0.1 * cfg.rho0)


    def _assemble(self):
        """Fluid + boundary particles as one set, plus the ghost source index.

        ``dynamic`` boundary: static wall particles are appended (fixed mass,
        zero velocity).  ``mirror`` boundary: images of the fluid particles
        within one support of a wall are generated every step; the pair list is
        rebuilt each step because the image set changes identity.
        """
        st, cfg = self.state, self.cfg
        nw = st.n_wall
        if nw:
            pos = torch.cat((st.pos, st.wall_pos), 0)
            vel = torch.cat((st.vel, torch.zeros((nw, cfg.dim), dtype=self.dtype)), 0)
            mass = torch.cat((st.mass, st.wall_mass), 0)
            h = torch.cat((st.h, torch.full((nw,), float(cfg.h0), dtype=self.dtype)), 0)
            return pos, vel, mass, h, None
        if cfg.boundary == "mirror" and self.mirror_planes:
            gpos, gvel, gmass, src = mirror_images(
                st.pos, st.vel, st.mass, self.mirror_planes, cfg.support)
            if gpos.numel():
                pos = torch.cat((st.pos, gpos), 0)
                vel = torch.cat((st.vel, gvel), 0)
                mass = torch.cat((st.mass, gmass), 0)
                h = torch.cat((st.h, st.h[src]), 0)
                return pos, vel, mass, h, src
        return st.pos, st.vel, st.mass, st.h, None

    def pairs(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        if pos is None:
            pos, _, _, _, _ = self._assemble()
        if self.cfg.boundary == "mirror":
            probe = PairListCache(self.cfg.support * (1.25 if self.cfg.h_adaptive else 1.0),
                                  0.0, fluid_count=self.state.n_fluid)
            return probe.get(pos)[0]
        pairs, _ = self.cache.get(pos)
        return pairs

    # -- physics ---------------------------------------------------------- #
    def force(self, dt: float = 0.0) -> torch.Tensor:
        """Density, pressure and acceleration at the current positions.

        Wall pressure follows eq. (11) of Adami, Hu & Adams (2012): the wall
        particle carries the kernel-weighted extrapolation of the neighbouring
        fluid pressure plus the hydrostatic moment ``rho_f g . (r_w - r_f)``.
        A dynamic boundary particle whose density came from its own summation
        would sit below ``rho0`` (a one-sided support) and the Tait clamp would
        switch it off exactly where it is needed.
        """
        cfg = self.cfg
        st = self.state
        pos, vel, mass, h, ghost_src = self._assemble()
        pairs = self.pairs(pos)
        n = st.n_fluid
        nw = st.n_wall
        dim = cfg.dim

        # --- density (summation form) ---
        rho = mass * kernels.self_contribution(h, dim, cfg.kernel)
        w = grad = None
        if pairs.numel():
            i, j = pairs.unbind(dim=1)
            delta = pos[i] - pos[j]
            w, grad = kernels.kernel_pairs(delta, h[i], dim, cfg.kernel)
            rho = rho.index_add(0, i, mass[j] * w)
            rho = rho.index_add(0, j, mass[i] * w)
            if cfg.density_renormalisation:
                total = torch.zeros_like(rho)
                total = total.index_add(0, i, w)
                total = total.index_add(0, j, w)
                total = total + kernels.self_contribution(h, dim, cfg.kernel)
                rho = rho * (_lattice_kernel_sum(cfg) / total.clamp_min(1.0e-12))
        if ghost_src is not None and ghost_src.numel():
            rho = torch.cat((rho[:n], rho[ghost_src]))
        rho = rho.clamp_min(0.1 * cfg.rho0)
        pressure = tait_pressure(rho, cfg)

        # --- wall pressure (Adami extrapolation over fluid-wall pairs) ---
        if nw and pairs.numel():
            mask_fw = (i < n) & (j >= n)
            mask_wf = (i >= n) & (j < n)
            fidx = torch.cat((i[mask_fw], j[mask_wf]))
            widx = torch.cat((j[mask_fw], i[mask_wf]))
            wsub = torch.cat((w[mask_fw], w[mask_wf]))
            if fidx.numel():
                r_wf = pos[widx] - pos[fidx]
                gravity = torch.zeros(dim, dtype=self.dtype)
                gravity[1] = -cfg.gravity
                source = pressure[fidx] + rho[fidx] * (r_wf @ gravity)
                num = torch.zeros(nw, dtype=self.dtype).index_add(0, widx - n,
                                                                 source * wsub)
                den = torch.zeros(nw, dtype=self.dtype).index_add(0, widx - n, wsub)
                p_wall = (num / den.clamp_min(1.0e-300)).clamp_min(0.0)
                rho_wall = (cfg.rho0 * (p_wall / cfg.bulk_modulus + 1.0)
                            .pow(1.0 / cfg.gamma)).clamp_min(cfg.rho0)
                # replace the wall entries (the summation density of a one-sided
                # support is not the wall density) -- do not append a second copy
                pressure = torch.cat((pressure[:n], p_wall))
                rho = torch.cat((rho[:n], rho_wall))
            else:
                pressure = torch.cat((pressure[:n], torch.zeros(nw,
                                                                dtype=self.dtype)))
                rho = torch.cat((rho[:n], torch.full((nw,), cfg.rho0,
                                                     dtype=self.dtype)))
        elif nw:
            pressure = torch.cat((pressure[:n], torch.zeros(nw, dtype=self.dtype)))
            rho = torch.cat((rho[:n], torch.full((nw,), cfg.rho0, dtype=self.dtype)))
        if ghost_src is not None and ghost_src.numel():
            pressure = torch.cat((pressure[:n], pressure[ghost_src]))
            rho = torch.cat((rho[:n], rho[ghost_src]))

        st.rho = rho[:n]
        st.pressure = pressure[:n]
        if nw:
            st.wall_rho = rho[n:]
            st.wall_pressure = pressure[n:]

        # --- acceleration ---
        acc = torch.zeros_like(pos)
        if pairs.numel():
            eta2 = 0.01 * float(cfg.h0) ** 2
            r = torch.linalg.vector_norm(delta, dim=1)
            dv = vel[i] - vel[j]
            vdotr = (dv * delta).sum(dim=1)
            h_pair = 0.5 * (h[i] + h[j])
            mu = h_pair * vdotr / (r.square() + eta2)
            rho_ij = 0.5 * (rho[i] + rho[j])
            pi = torch.where(
                vdotr < 0.0,
                (-cfg.alpha_av * cfg.c0 * mu + cfg.beta_av * mu * mu) / rho_ij,
                torch.zeros_like(mu),
            )
            # Monaghan (1992) eq. 2.13: the artificial-viscosity term is *added*
            # to the pressure term.  With mu < 0 for approaching pairs,
            # Pi = -alpha c mu > 0, so the pair is pushed apart harder than by
            # pressure alone -- that is the dissipation.  Subtracting it instead
            # would make the pair attractive and the scheme anti-dissipative.
            coeff = (pressure[i] / rho[i].square()
                     + pressure[j] / rho[j].square()) + pi
            pair = coeff.reshape(-1, 1) * grad
            acc = acc.index_add(0, i, -mass[j].reshape(-1, 1) * pair)
            acc = acc.index_add(0, j, mass[i].reshape(-1, 1) * pair)
            if cfg.viscosity > 0.0:
                fluid_pair = (i < n) & (j < n)
                if bool(fluid_pair.any()):
                    ii, jj = i[fluid_pair], j[fluid_pair]
                    dd = delta[fluid_pair]
                    gg = grad[fluid_pair]
                    # (r . grad W) < 0 for a central kernel; keeping the sign
                    # makes the pair term decelerate the relative velocity
                    # instead of accelerating it
                    lap = (dd * gg).sum(dim=1) / (r[fluid_pair].square() + eta2)
                    common = ((dim + 2.0) * cfg.viscosity * lap).reshape(-1, 1) * (
                        vel[ii] - vel[jj])
                    acc = acc.index_add(0, ii, common
                                        * (mass[jj] / rho[jj]).reshape(-1, 1))
                    acc = acc.index_add(0, jj, -common
                                        * (mass[ii] / rho[ii]).reshape(-1, 1))

        fluid_acc = acc[:n].clone()
        if cfg.gravity:
            fluid_acc[:, 1] -= cfg.gravity
        if cfg.excitation_amplitude and cfg.excitation_omega:
            drive = -cfg.excitation_amplitude * cfg.excitation_omega ** 2 * math.sin(
                cfg.excitation_omega * st.time + cfg.excitation_phase)
            fluid_acc[:, 0] += drive

        st.max_compression = max(st.max_compression,
                                 float(st.rho.max()) / cfg.rho0 - 1.0)
        speed = float(torch.linalg.vector_norm(st.vel, dim=1).max()) if n else 0.0
        st.max_speed = max(st.max_speed, speed)
        if dt > 0.0:
            st.max_courant = max(st.max_courant, dt * (cfg.c0 + speed) / cfg.h0)
        return fluid_acc

    # -- time integration ------------------------------------------------- #
    def step(self, dt: float | None = None) -> float:
        cfg, st = self.cfg, self.state
        dt = stable_dt(cfg, st.max_speed) if dt is None else float(dt)
        if st.acc is None:
            st.acc = self.force(dt)
        st.vel = st.vel + 0.5 * dt * st.acc
        st.pos = st.pos + dt * st.vel
        self._apply_box()
        acc = self.force(dt)
        st.vel = st.vel + 0.5 * dt * acc
        st.acc = acc
        st.time += dt
        st.steps += 1
        if cfg.h_adaptive:
            st.h = kernels.adaptive_h(st.rho, cfg.rho0, cfg.h0, cfg.dim)
        return dt

    def _apply_box(self) -> None:
        """Clamp to the tank and zero the normal velocity (safety net only)."""
        st, cfg = self.state, self.cfg
        if cfg.dim == 2 and self.box is not None:
            x0, x1, y0, y1 = self.box
            lower = torch.tensor([x0, y0], dtype=self.dtype)
            upper = torch.tensor([x1, y1], dtype=self.dtype)
        elif cfg.dim == 3 and self.box3 is not None:
            x0, x1, y0, y1, z0, z1 = self.box3
            lower = torch.tensor([x0, y0, z0], dtype=self.dtype)
            upper = torch.tensor([x1, y1, z1], dtype=self.dtype)
        else:
            return
        tol = 0.25 * cfg.dx
        below = st.pos < lower - tol
        above = st.pos > upper + tol
        st.pos = torch.maximum(torch.minimum(st.pos, upper), lower)
        st.vel = torch.where(below | above, torch.zeros_like(st.vel), st.vel)

    # -- diagnostics ------------------------------------------------------ #
    def kinetic_energy(self) -> float:
        return float(0.5 * (self.state.mass
                            * self.state.vel.square().sum(dim=1)).sum())

    def potential_energy(self) -> float:
        if not self.cfg.gravity:
            return 0.0
        return float(self.cfg.gravity * (self.state.mass
                                         * self.state.pos[:, 1]).sum())

    def momentum(self) -> torch.Tensor:
        return (self.state.mass.reshape(-1, 1) * self.state.vel).sum(dim=0)

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()


def mirror_images(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                  planes: list[tuple[int, float]], support: float
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror particles about nearby flat walls (Colagrossi & Landrini 2003).

    ``planes`` is a list of ``(axis, coordinate)`` pairs.  Every fluid particle
    within ``support`` of a plane is mirrored once across its nearest plane
    (single-reflection approximation; corner particles therefore carry one
    image).  The normal velocity component is reflected, the tangential
    component is kept (slip).

    Returns ``(ghost_pos, ghost_vel, ghost_mass, source_index)``.
    """
    dim = pos.shape[1]
    empty = (pos.new_zeros((0, dim)), pos.new_zeros((0, dim)), pos.new_zeros((0,)),
             torch.zeros(0, dtype=torch.long))
    if not planes or pos.numel() == 0:
        return empty
    dists, axes, coords = [], [], []
    for axis, coord in planes:
        dists.append((pos[:, axis] - coord).abs())
        axes.append(axis)
        coords.append(coord)
    dist = torch.stack(dists, dim=1)
    nearest = dist.argmin(dim=1)
    dmin = dist.gather(1, nearest.reshape(-1, 1)).reshape(-1)
    idx = torch.nonzero(dmin <= support, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return empty
    pick = nearest[idx].tolist()
    axis = torch.tensor([axes[k] for k in pick], dtype=torch.long)
    coord = torch.tensor([coords[k] for k in pick], dtype=pos.dtype)
    src = pos[idx]
    ar = torch.arange(idx.numel())
    gpos = src.clone()
    gpos[ar, axis] = 2.0 * coord - src[ar, axis]
    gvel = vel[idx].clone()
    gvel[ar, axis] = -gvel[ar, axis]
    return gpos, gvel, mass[idx], idx