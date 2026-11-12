"""Time integration for the two 3-D verification scenarios.

Two cases, both with an exact reference solution:

``run_shear_wave``
    Unsteady Stokes problem on a triply periodic cube.  The velocity field
    ``u = (A sin(k y) sin(k z), 0, 0)`` is divergence free, its convective term
    vanishes identically (``u_x`` does not depend on ``x``), so the incompressible
    Navier-Stokes equations reduce to the heat equation and the exact solution is

        u_x(y, z, t) = A sin(k y) sin(k z) exp(-2 nu k^2 t).

    The benchmark measures the decay exponent against ``2 nu k^2``.  This is a
    genuinely three-dimensional field (it varies in two directions and lives in
    three), it exercises the 3-D kernel, the periodic neighbour search, the
    density summation and the viscous Laplacian, and it has no wall and no free
    surface, so linear momentum is conserved to round-off -- which is what makes
    the conservation gate meaningful.

``run_hydrostatic``
    A wall-bounded column at rest.  The exact pressure is ``p(z) = rho0 g (H - z)``
    with ``H`` the free-surface height.  This is the case that tests the *wall*
    side of the 3-D machinery: static wall particles, the Adami pressure
    extrapolation and the Tait-consistent wall density.  Momentum is not
    conserved here (gravity and the wall reaction are external), so the meters are
    the pressure-profile error and the centre-of-mass drift.

Both drivers recompute neighbour pairs only when a particle has moved far enough
that the Verlet skin can no longer cover it, which is what makes a few thousand
steps affordable at 4096 particles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..surface_tension_dispersion import capillary_dt_limit
from .conservation import diagnostics
from .kernel3 import calibrated_particle_mass
from .neighbors3 import box_pairs, minimum_image, periodic_pairs
from .operators3 import (
    accelerations,
    analytic_mode_decay,
    mode_amplitude,
    shear_wave_field,
)

__all__ = [
    "ShearWaveResult",
    "HydrostaticResult",
    "cubic_lattice",
    "hydrostatic_density",
    "run_shear_wave",
    "run_hydrostatic",
    "acoustic_dt",
]

#: Acoustic Courant number used to pick the default step of both drivers.
COURANT = 0.25


def acoustic_dt(dx: float, h: float, sound_speed: float, courant: float = COURANT,
                sigma: float = 0.0, rho0: float = 1000.0) -> float:
    """Stable step: acoustic limit at rest, capped by the capillary limit.

    ``dt = courant * h / c0`` is the acoustic bound.  When a surface-tension
    coefficient is supplied the explicit capillary bound of
    :func:`tensorsph.surface_tension_dispersion.capillary_dt_limit` is applied as
    a second limiter,

        ``dt_sigma = courant * sqrt(rho0 h^3 / (sigma pi^3))``,

    because a capillary force stiffer than the acoustic one is still a hard
    stability constraint on an explicit integrator.  ``sigma = 0`` (default)
    reproduces the historical behaviour exactly.
    """
    dt = courant * h / sound_speed
    return min(dt, capillary_dt_limit(h, sigma, rho0, cfl=courant))


def cubic_lattice(shape, dx: float, origin=(0.0, 0.0, 0.0),
                  dtype=torch.float64) -> torch.Tensor:
    """``shape = (nx, ny, nz)`` points at cell centres, row-major (x fastest)."""
    axes = [torch.arange(n, dtype=dtype) * dx + origin[axis]
            for axis, n in enumerate(shape)]
    grid = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, len(shape))


def wall_shell(fluid_shape, layers: int, dx: float,
               dtype=torch.float64) -> torch.Tensor:
    """Lattice points of a ``layers``-thick shell around a ``fluid_shape`` block.

    The shell and the fluid live on the *same* lattice, so the wall is exactly
    ``layers`` particles thick and the fluid block starts at
    ``(layers * dx, layers * dx, layers * dx)``.  Returned row-major like
    :func:`cubic_lattice`, wall particles only.
    """
    outer = tuple(int(n) + 2 * int(layers) for n in fluid_shape)
    axes = [torch.arange(n, dtype=dtype) * dx for n in outer]
    grid = torch.meshgrid(*axes, indexing="ij")
    index = torch.meshgrid(
        *[torch.arange(n) for n in outer], indexing="ij")
    keep = torch.zeros(outer, dtype=torch.bool)
    inner = []
    for axis, n in enumerate(fluid_shape):
        idx = index[axis]
        inner.append((idx >= layers) & (idx < layers + int(n)))
    inside = inner[0]
    for extra in inner[1:]:
        inside = inside & extra
    keep = ~inside
    return torch.stack(grid, dim=-1)[keep]


def hydrostatic_density(z: torch.Tensor, height: float, rho0: float,
                        sound_speed: float, gamma: float,
                        gravity: float) -> torch.Tensor:
    """``rho(z)`` for ``p = rho0 g (H - z)`` through the Tait law."""
    stiffness = rho0 * sound_speed ** 2 / gamma
    p = rho0 * gravity * (height - z).clamp_min(0.0)
    return rho0 * (1.0 + p / stiffness) ** (1.0 / gamma)


@dataclass
class ShearWaveResult:
    """Outcome of one shear-wave run."""

    n: int
    dx: float
    h: float
    nu: float
    wavenumber: float
    sound_speed: float
    amplitude0: float
    dt: float
    steps: int
    renormalise: bool
    viscous_order: int = 2
    times: list = field(default_factory=list)
    amplitudes: list = field(default_factory=list)
    fit_rate: float = float("nan")
    analytic_rate: float = float("nan")
    diagnostics: list = field(default_factory=list)
    pairs_last: int = 0
    pair_rebuilds: int = 0
    momentum_scale: float = 0.0

    @property
    def rate_error(self) -> float:
        return (self.fit_rate - self.analytic_rate) / self.analytic_rate

    def as_dict(self) -> dict:
        first = self.diagnostics[0]
        last = self.diagnostics[-1]
        return {
            "case": "shear_wave_3d",
            "n": self.n,
            "particles": self.n ** 3,
            "dx": self.dx,
            "h": self.h,
            "h_over_dx": self.h / self.dx,
            "kh": self.wavenumber * self.h,
            "nu": self.nu,
            "wavenumber": self.wavenumber,
            "sound_speed": self.sound_speed,
            "amplitude0": self.amplitude0,
            "mach": self.amplitude0 / self.sound_speed,
            "dt": self.dt,
            "steps": self.steps,
            "renormalise_viscous_operator": self.renormalise,
            "viscous_operator_order": self.viscous_order,
            "decay": {
                "fit_rate": self.fit_rate,
                "analytic_rate": self.analytic_rate,
                "rel_error": self.rate_error,
                "amplitude_final": self.amplitudes[-1] if self.amplitudes else None,
            },
            "conservation": {
                "momentum_scale_sum_m_abs_v": self.momentum_scale,
                "momentum_final_over_scale": (
                    last["momentum_magnitude"] / self.momentum_scale
                    if self.momentum_scale > 0.0 else None),
                "mass_relative_drift": (last["mass"] - first["mass"]) / first["mass"],
                "momentum_initial": first["momentum"],
                "momentum_final": last["momentum"],
                "momentum_magnitude_initial": first["momentum_magnitude"],
                "momentum_magnitude_final": last["momentum_magnitude"],
                "momentum_magnitude_peak": max(d["momentum_magnitude"]
                                               for d in self.diagnostics),
                "kinetic_energy_initial": first["kinetic_energy"],
                "kinetic_energy_final": last["kinetic_energy"],
                "total_energy_initial": first["total_energy"],
                "total_energy_final": last["total_energy"],
                "total_energy_relative_drift": (
                    (last["total_energy"] - first["total_energy"])
                    / first["total_energy"]),
                "angular_momentum_drift": [
                    b - a for a, b in zip(first["angular_momentum"],
                                          last["angular_momentum"])],
            },
            "pairs_last": self.pairs_last,
            "pair_rebuilds": self.pair_rebuilds,
            "series": {"times": self.times, "amplitudes": self.amplitudes},
        }


def run_shear_wave(*, n: int = 16, box: float = 1.0, h_factor: float = 1.3,
                   periods: int = 1, nu: float = 0.02, amplitude: float = 0.1,
                   sound_speed: float = 10.0, rho0: float = 1000.0,
                   gamma: float = 7.0, t_end: float | None = None,
                   dt: float | None = None, steps: int | None = None,
                   sample_every: int = 10, skin_factor: float = 0.25,
                   renormalise: bool = True, viscous_order: int = 2,
                   dtype=torch.float64) -> ShearWaveResult:
    """Integrate the 3-D double-shear field and fit its decay exponent.

    ``t_end`` defaults to two decay times, which leaves the amplitude at
    ``e^-2 = 0.14`` -- low enough that the fitted slope is not dominated by the
    initial transient, high enough that the tail is still far above round-off.
    """
    dx = box / n
    h = h_factor * dx
    wavenumber = 2.0 * math.pi * periods / box
    rate = analytic_mode_decay(nu, wavenumber)
    if t_end is None:
        t_end = 2.0 / rate
    if dt is None:
        dt = acoustic_dt(dx, h, sound_speed)
    if steps is None:
        steps = int(math.ceil(t_end / dt))
    viscous_limit = 0.125 * h * h / max(nu, 1.0e-30)
    if dt > viscous_limit:
        raise ValueError(
            f"dt = {dt:.3e} s exceeds the viscous limit 0.125 h^2/nu = "
            f"{viscous_limit:.3e} s")
    if wavenumber * h > 0.9:
        raise ValueError(
            f"k h = {wavenumber * h:.2f}: the kernel cannot represent this mode")

    pos = cubic_lattice((n, n, n), dx, dtype=dtype)
    mass = torch.full((pos.shape[0],),
                      calibrated_particle_mass(dx, h, rho0), dtype=dtype)
    vel = shear_wave_field(pos, amplitude, wavenumber)

    cutoff = 2.0 * h * (1.0 + skin_factor)
    skin = cutoff - 2.0 * h
    pairs = periodic_pairs(pos, box, cutoff)
    reference = pos.clone()
    rebuilds = 1

    result = ShearWaveResult(
        n=n, dx=dx, h=h, nu=nu, wavenumber=wavenumber, sound_speed=sound_speed,
        amplitude0=amplitude, dt=dt, steps=steps, renormalise=renormalise,
        viscous_order=viscous_order,
        pairs_last=int(pairs.shape[0]), pair_rebuilds=rebuilds)

    def rhs(p, v):
        acc, rho, _ = accelerations(p, v, mass, h, pairs, rho0=rho0,
                                    sound_speed=sound_speed, gamma=gamma,
                                    viscosity=nu, box=box,
                                    renormalise=renormalise,
                                    viscous_order=viscous_order)
        return acc, rho

    # Scale of the momentum the flow could carry.  Comparing the drift with the
    # *initial* momentum is meaningless here: it is zero by symmetry, so the ratio
    # would divide round-off by round-off.
    result.momentum_scale = float((mass.unsqueeze(-1) * vel.abs()).sum())

    acc, rho = rhs(pos, vel)
    times, amplitudes, rows = [], [], []
    for k in range(steps + 1):
        if k % sample_every == 0:
            times.append(k * dt)
            amplitudes.append(mode_amplitude(pos, vel, wavenumber))
            rows.append(diagnostics(pos, vel, mass, rho, rho0=rho0,
                                    sound_speed=sound_speed, gamma=gamma,
                                    periodic_box=box))
        if k == steps:
            break
        # Kick-drift-kick; positions are wrapped so the periodic search stays
        # inside the box.
        vel = vel + 0.5 * dt * acc
        pos = (pos + dt * vel).remainder(box)
        # ``pos`` is wrapped into the box, so the displacement has to be taken
        # modulo the period; otherwise every boundary crossing looks like a jump
        # of one box length and forces a rebuild on every step.
        moved = torch.linalg.vector_norm(
            minimum_image(pos - reference, box), dim=1).max()
        if 2.0 * float(moved) > skin:
            pairs = periodic_pairs(pos, box, cutoff)
            reference = pos.clone()
            rebuilds += 1
        acc, rho = rhs(pos, vel)
        vel = vel + 0.5 * dt * acc

    result.times = times
    result.amplitudes = amplitudes
    result.diagnostics = rows
    result.pairs_last = int(pairs.shape[0])
    result.pair_rebuilds = rebuilds
    result.analytic_rate = rate
    result.fit_rate = fit_decay_rate(times, amplitudes)
    return result


def fit_decay_rate(times, amplitudes) -> float:
    """Least-squares slope of ``ln a`` against ``t``.

    The first sample is dropped: at ``t = 0`` the discrete field is exactly the
    analytic one, so including it only biases the fit towards the initial
    condition.  The fit is over the whole remaining window, not a two-point
    estimate, so a single noisy sample cannot set the number.
    """
    t = torch.as_tensor(times[1:], dtype=torch.float64)
    a = torch.as_tensor([abs(float(x)) for x in amplitudes[1:]], dtype=torch.float64)
    keep = a > 0.0
    t, a = t[keep], a[keep]
    if t.numel() < 2:
        return float("nan")
    ta = t - t.mean()
    return float(-(ta * (torch.log(a) - torch.log(a).mean())).sum() / (ta * ta).sum())


@dataclass
class HydrostaticResult:
    """Outcome of one wall-bounded hydrostatic run."""

    dx: float
    h: float
    sound_speed: float
    gravity: float
    height: float
    dt: float
    steps: int
    fluid_particles: int
    wall_particles: int
    times: list = field(default_factory=list)
    bottom_pressure: list = field(default_factory=list)
    centre_of_mass_z: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)

    @property
    def analytic_bottom_pressure(self) -> float:
        return self.profile.get("rho0", 1000.0) * self.gravity * self.height


def run_hydrostatic(*, nx: int = 10, ny: int = 10, nz: int = 8, dx: float = 0.01,
                    h_factor: float = 1.3, wall_layers: int = 2,
                    sound_speed: float = 20.0, rho0: float = 1000.0,
                    gamma: float = 7.0, gravity: float = 9.81,
                    viscosity: float = 0.02, t_end: float | None = None,
                    dt: float | None = None, steps: int | None = None,
                    sample_every: int = 5, settle_fraction: float = 0.4,
                    wall_model: str = "adami", lid: bool = True,
                    density_diffusion: float = 0.0,
                    density_diffusion_form: str = "antuono",
                    dtype=torch.float64) -> HydrostaticResult:
    """Integrate a wall-bounded 3-D column and compare ``p(z)`` with ``rho0 g (H-z)``.

    The column is ``nx x ny x nz`` particles tall, surrounded by ``wall_layers``
    layers of static wall particles on the bottom and on all four sides.  Density
    is initialised from the analytic hydrostatic profile so the run starts in
    balance and only the discretisation error is left to measure; the reported
    profile is the time average over the last ``settle_fraction`` of the run.
    """
    height = nz * dx
    walls = wall_layers * dx
    if dt is None:
        dt = acoustic_dt(dx, h_factor * dx, sound_speed)
    if t_end is None:
        t_end = 40.0 * height / sound_speed + 0.05
    if steps is None:
        steps = int(math.ceil(t_end / dt))
    h = h_factor * dx

    fluid = cubic_lattice((nx, ny, nz), dx, origin=(walls, walls, walls), dtype=dtype)
    n_fluid = fluid.shape[0]
    # Lattice row of every fluid particle (0 = bottom row), taken from the
    # *initial* z.  The particles are created on the lattice planes themselves
    # (``z = walls + k dx``), which is why an Eulerian bin of width ``dx`` whose
    # edges sit on the same planes is a degenerate partition of the column: a
    # row lands in bin ``k`` or in bin ``k - 1`` depending on the last bit of
    # ``(z - walls) / dx`` and on the sub-cell settling of the run.  Keeping the
    # Lagrangian label lets the profile be built row by row, which is the only
    # binning-invariant way to compare a discrete profile with ``rho0 g (H-z)``.
    row_index = torch.unique(fluid[:, 2], sorted=True, return_inverse=True)[1]
    n_rows = int(row_index.max()) + 1 if n_fluid else 0
    mass_fluid = torch.full((n_fluid,), calibrated_particle_mass(dx, h, rho0),
                            dtype=dtype)

    # Wall shell: ``wall_layers`` layers of lattice points on the six faces of the
    # box that encloses the fluid block.  Selected by *index*, never by a
    # coordinate window: a window with a half-cell margin silently keeps the
    # first wall layer and then the wall overlaps the fluid, which makes every
    # pressure reading meaningless.
    wall = wall_shell((nx, ny, nz), wall_layers, dx, dtype=dtype)
    if not lid:
        # ``wall_shell`` closes *all six* faces, so the default configuration has
        # wall particles above the free surface -- a rigid lid one and two cells
        # above the topmost fluid row, i.e. inside the ``2 h = 2.6 dx`` support.
        # Those particles are not part of the problem this case poses (a column
        # with a free surface), and they inject a spurious force into the only
        # layer whose pressure should go to zero.  ``lid=False`` drops every wall
        # particle at or above ``z = H``; it is opt-in so the shipped artifact in
        # ``benchmarks/threed_hydrostatic/report.json`` keeps its meaning.
        wall = wall[wall[:, 2] <= walls + height + 1e-12]
    mass_wall = torch.full((wall.shape[0],), calibrated_particle_mass(dx, h, rho0),
                           dtype=dtype)

    pos = torch.cat((fluid, wall))
    mass = torch.cat((mass_fluid, mass_wall))
    vel = torch.zeros_like(pos)

    z = fluid[:, 2]
    rho0_field = hydrostatic_density(z, walls + height, rho0, sound_speed, gamma,
                                     gravity)
    pairs = box_pairs(pos, 2.0 * h)
    g = torch.tensor([0.0, 0.0, -gravity], dtype=dtype)

    result = HydrostaticResult(
        dx=dx, h=h, sound_speed=sound_speed, gravity=gravity, height=height,
        dt=dt, steps=steps, fluid_particles=n_fluid,
        wall_particles=int(wall.shape[0]))
    result.profile["wall_model"] = wall_model
    result.density_diffusion = density_diffusion
    result.density_diffusion_form = density_diffusion_form
    result.profile["rho0"] = rho0

    def rhs(p, v):
        acc, rho, pressure = accelerations(
            p, v, mass, h, pairs, rho0=rho0, sound_speed=sound_speed,
            gamma=gamma, viscosity=viscosity, gravity=g, n_fluid=n_fluid,
            wall_model=wall_model, density_diffusion=density_diffusion, dt=dt,
            density_diffusion_form=density_diffusion_form)
        return acc, rho, pressure

    acc, rho, pressure = rhs(pos, vel)
    wall_pos = pos[n_fluid:].clone()
    times, bottoms, coms = [], [], []
    sample_start = int(steps * (1.0 - settle_fraction))
    buckets: list[dict] = []
    wall_displacement = 0.0
    for k in range(steps + 1):
        if k >= sample_start and k % sample_every == 0:
            times.append(k * dt)
            bottoms.append(float(pressure[:n_fluid].max()))
            coms.append(float(pos[:n_fluid, 2].mean()))
            buckets.append({
                "z": pos[:n_fluid, 2].clone(),
                "p": pressure[:n_fluid].clone(),
                "rho": rho[:n_fluid].clone(),
            })
        if k == steps:
            break
        # Kick-drift-kick on the fluid; the wall rows are pinned to zero velocity
        # and to their lattice positions every step (they are a boundary
        # condition, not a material).
        vel = vel + 0.5 * dt * acc
        vel[n_fluid:] = 0.0
        pos = pos + dt * vel
        pos[n_fluid:] = wall_pos
        acc, rho, pressure = rhs(pos, vel)
        vel = vel + 0.5 * dt * acc
        vel[n_fluid:] = 0.0
    wall_displacement = float((pos[n_fluid:] - wall_pos).abs().max())

    result.times = times
    result.bottom_pressure = bottoms
    result.centre_of_mass_z = coms
    result.profile = build_pressure_profile(
        buckets, nz=nz, dx=dx, walls=walls, height=height, rho0=rho0,
        gravity=gravity, h=h, sound_speed=sound_speed, gamma=gamma,
        row_index=row_index)
    result.profile["wall_displacement"] = wall_displacement
    result.diagnostics = [diagnostics(fluid, vel[:n_fluid], mass_fluid,
                                      rho0_field, rho0=rho0,
                                      sound_speed=sound_speed, gamma=gamma)]
    return result


def build_pressure_profile(buckets, *, nz: int, dx: float, walls: float,
                           height: float, rho0: float, gravity: float, h: float,
                           sound_speed: float, gamma: float,
                           row_index=None) -> dict:
    """Time-averaged ``p(z)`` over layer bins, plus the analytic reference.

    Only particles whose support is complete in the vertical direction
    (``z`` at least ``2 h`` below the free surface) are used for the accuracy
    numbers: right at the surface the summation density under-counts and the Tait
    law clamps ``p`` to zero, so the relative error there says nothing about the
    operator.  The bins are one-particle-layer wide, which is the resolution at
    which a discrete profile can be compared at all.

    Two binning schemes are reported, because they answer different questions:

    ``rows``
        Eulerian bins ``floor((z - walls) / dx)`` labelled with the bin *centre*
        ``walls + (i + 1/2) dx``.  The column is created on the lattice planes
        ``walls + k dx``, i.e. exactly on the bin edges, so this partition is
        degenerate: a row lands in bin ``k`` or ``k - 1`` depending on the last
        bit of the coordinate and on the sub-cell settling, and the label is half
        a cell away from where the mass actually is.  The resulting per-layer
        error grows like ``(dx/2) / depth`` and therefore explodes towards the
        free surface even when the field itself is smooth.  Kept because the
        shipped artifacts and the verification ledger quote it.

    ``rows_by_lattice``
        Lagrangian rows from the particle's *initial* lattice index, compared with
        the analytic profile at the row's own (time-averaged) depth.  Invariant to
        the binning above, so this is the profile the accuracy numbers should be
        read from.  Pass ``row_index`` to build it.
    """
    z = torch.cat([b["z"] for b in buckets])
    p = torch.cat([b["p"] for b in buckets])
    rho = torch.cat([b["rho"] for b in buckets])
    surface = walls + height
    # Layer index: 0 is the bottom fluid layer.
    layer = torch.clamp(((z - walls) / dx).floor().to(torch.int64), 0, nz - 1)
    rows = []
    for index in range(nz):
        mask = layer == index
        if int(mask.sum()) == 0:
            continue
        z_centre = walls + (index + 0.5) * dx
        p_measured = float(p[mask].mean())
        p_analytic = rho0 * gravity * (surface - z_centre)
        rows.append({
            "layer": index,
            "z": z_centre,
            "depth_below_surface": surface - z_centre,
            "particles": int(mask.sum()),
            "p_measured": p_measured,
            "p_analytic": p_analytic,
            "p_rel_error": ((p_measured - p_analytic) / p_analytic
                            if p_analytic > 0.0 else None),
            "rho_measured": float(rho[mask].mean()),
            "rho_analytic": float(hydrostatic_density(
                torch.tensor(z_centre, dtype=torch.float64), surface, rho0,
                sound_speed, gamma, gravity)),
            "support_complete": bool(z_centre < surface - 3.0 * h),
        })
    core = [r for r in rows if r["support_complete"]]
    scale = rho0 * gravity * height
    lattice_rows: list[dict] = []
    if row_index is not None and z.numel():
        n_particles = int(row_index.numel())
        samples = z.numel() // n_particles
        label = torch.tile(row_index.to(torch.int64).reshape(-1), (samples,))
        for index in range(int(label.max()) + 1):
            mask = label == index
            if not bool(mask.any()):
                continue
            z_row = float(z[mask].mean())
            z_nominal = walls + index * dx
            p_measured = float(p[mask].mean())
            p_analytic = rho0 * gravity * (surface - z_row)
            lattice_rows.append({
                "row": index,
                "z": z_row,
                "z_nominal": z_nominal,
                "z_offset_over_dx": (z_row - z_nominal) / dx,
                "depth_below_surface": surface - z_row,
                "particles": int(mask.sum()) // samples,
                "p_measured": p_measured,
                "p_analytic": p_analytic,
                "p_rel_error": ((p_measured - p_analytic) / p_analytic
                                if p_analytic > 0.0 else None),
                "rho_measured": float(rho[mask].mean()),
                "support_complete": bool(z_nominal < surface - 3.0 * h),
            })
        for r in lattice_rows:
            if r["p_rel_error"] is None:
                r["support_complete"] = False
    core_lattice = [r for r in lattice_rows
                    if r["support_complete"] and r["p_rel_error"] is not None]
    errors = [r["p_rel_error"] for r in core]
    # Depth-integrated load: the quantity a wall actually feels.  It is immune to
    # the odd-even (checkerboard) mode that a particle lattice supports in the
    # pressure field, which is why it is reported next to the per-layer errors.
    integral_measured = sum(r["p_measured"] for r in rows) * dx
    integral_analytic = sum(r["p_analytic"] for r in rows) * dx
    odd_even = None
    if len(rows) >= 4:
        even = [r["p_measured"] for r in rows[0::2]]
        odd = [r["p_measured"] for r in rows[1::2]]
        odd_even = abs(sum(even) / len(even) - sum(odd) / len(odd)) / scale
    return {
        "rho0": rho0,
        "surface": surface,
        "rows": rows,
        "core_layers": len(core),
        "max_rel_error_core": max((abs(e) for e in errors), default=None),
        "mean_rel_error_core": (sum(errors) / len(errors)) if errors else None,
        "layers_within_15pct": sum(1 for e in errors if abs(e) < 0.15),
        "max_abs_error_over_scale": max(
            (abs(r["p_measured"] - r["p_analytic"]) / scale for r in core),
            default=None),
        "integral_p_measured": integral_measured,
        "integral_p_analytic": integral_analytic,
        "integral_rel_error": ((integral_measured - integral_analytic)
                               / integral_analytic if integral_analytic else None),
        "odd_even_amplitude_over_scale": odd_even,
        "bottom_pressure_measured": rows[0]["p_measured"] if rows else None,
        "bottom_pressure_analytic": rows[0]["p_analytic"] if rows else None,
        "binning": _binning_diagnostic(z, layer, row_index, dx, walls),
        "rows_by_lattice": lattice_rows,
        "core_layers_lattice": len(core_lattice),
        "max_rel_error_core_lattice": max(
            (abs(r["p_rel_error"]) for r in core_lattice), default=None),
        "mean_rel_error_core_lattice": (
            sum(r["p_rel_error"] for r in core_lattice) / len(core_lattice)
            if core_lattice else None),
        "layers_within_15pct_lattice": sum(
            1 for r in core_lattice if abs(r["p_rel_error"]) < 0.15),
        "max_abs_error_over_scale_lattice": max(
            (abs(r["p_measured"] - r["p_analytic"]) / scale
             for r in core_lattice), default=None),
        "odd_even_amplitude_over_scale_lattice": _odd_even(
            [r["p_measured"] for r in lattice_rows], scale),
    }

def _odd_even(values: list[float], scale: float):
    """Amplitude of the even/odd (checkerboard) mode of a layer sequence."""
    if len(values) < 4:
        return None
    even = values[0::2]
    odd = values[1::2]
    return abs(sum(even) / len(even) - sum(odd) / len(odd)) / scale


def _binning_diagnostic(z, layer, row_index, dx: float, walls: float) -> dict | None:
    """How far the Eulerian bins are from a clean one-row-per-bin partition.

    ``mean_offset_over_dx`` is the signed distance from the bin *centre* (the
    label the analytic profile is evaluated at) to the particles it contains; for
    a lattice-aligned column it is ``-0.5`` by construction.  ``misplaced_fraction``
    is the share of particle samples that land in a bin whose index differs from
    their lattice row, i.e. the share that the Eulerian estimator mixes with a
    neighbouring row.
    """
    if row_index is None or z.numel() == 0:
        return None
    n_particles = int(row_index.numel())
    samples = z.numel() // n_particles
    label = torch.tile(row_index.to(torch.int64).reshape(-1), (samples,))
    offset = z - (walls + (layer.to(z.dtype) + 0.5) * dx)
    misplaced = layer != label
    return {
        "mean_offset_over_dx": float(offset.mean()) / dx,
        "rms_offset_over_dx": float(torch.sqrt((offset * offset).mean())) / dx,
        "misplaced_fraction": float(misplaced.to(z.dtype).mean()),
        "rows": int(label.max()) + 1,
        "bins": int(layer.max()) + 1,
    }
