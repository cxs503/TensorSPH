"""Periodic liquid column: the Rayleigh-Plateau validation case in 3-D.

The setup is the classical one (Rayleigh 1878): a cylinder of radius ``R``,
periodic along ``z`` with period ``lambda``, perturbed by a single axisymmetric
mode ``r_s(z) = R (1 + eps cos k z)``.  For ``k R < 1`` the mode grows
exponentially at the rate predicted by
:func:`tensorsph.rayleigh_plateau.dispersion_rate`, the neck thins, and the
column pinches into drops spaced ``lambda`` apart.

Why the box is triply periodic
------------------------------
``neighbors3.periodic_pairs`` wraps all three axes.  The column is periodic in
``z`` by construction; in ``x``/``y`` the period is padded to ``2 R + margin``
with ``margin > 4 h`` so the free surface never sees its own image through the
kernel or through the curvature stencil, while a drop that drifts keeps the
exact periodic geometry.  The padding costs memory, not accuracy.

Time integration and stabilisation
----------------------------------
* WCSPH with the Tait EOS from ``tensorsph.threed.operators3`` (density by
  summation, pressure clipped at zero -- no tensile stress at a free surface).
* delta-SPH density diffusion (Antuono, Colagrossi & Marrone 2012), coefficient
  ``delta``, applied as a rate to the summation density exactly as the 2-D
  solver does.  Without it the pressure of a free-floating jet carries the
  summation noise and the spurious currents matter more than a 2 % mode.
* CSF surface tension from :mod:`tensorsph.threed.csf3`.
* Kick-drift-kick leapfrog with a fixed ``dt`` from the acoustic CFL (so a run
  is reproducible), pairs rebuilt every ``pair_rebuild`` steps.

Measuring the mode (and why three measures exist)
------------------------------------------------
The perturbation is a *radial* mode of an incompressible cylinder: uniform
density, perturbed surface.  Two tempting measures fail in a way worth recording
(both are measured in ``docs/verification_rayleigh_plateau.md``):

* cross-section area ``A(z) = sum_{slab} V_i / dz`` -- a radial map leaves the
  volume per slab invariant, so ``A`` has *zero* mode amplitude at ``t = 0``;
* second moment ``R_s = sqrt(2 <r^2>_slab)`` -- exact on the initial lattice but
  its noise floor after the lattice moves is ~5 % of ``R``, larger than a 2 %
  perturbation.

The measure used is the radius of the *interface shell*, ``delta_s = |grad c| >
shell_fraction * max``, averaged per axial slab with ``delta_s`` weights
(:func:`surface_radius_profile`).  It tracks the surface from the first sample,
has a constant (z-independent) offset from the shell thickness, is attenuated by
a constant factor from kernel smoothing, and both constants cancel in the
exponential *rate* the anchor scores.  A non-resonant mode is recorded alongside
as the noise floor, and the second harmonic tracks the onset of nonlinearity.

References
    Rayleigh, Lord (1878), Proc. London Math. Soc. 10:4.
    Antuono, Colagrossi & Marrone (2012), Comput. Phys. Commun. 183:2570.
    Adami, Ihmsen & Adams (2012), J. Comput. Phys. 231:7057.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..rayleigh_plateau import capillary_time, dispersion_rate
from .csf3 import CSFState3, csf_fields3, surface_tension_acceleration3
from .kernel3 import cubic_spline_kernel
from .neighbors3 import periodic_pairs
from .operators3 import (pair_geometry, pressure_acceleration, summation_density,
                         tait_pressure, viscous_acceleration)

__all__ = ["ColumnConfig", "build_column", "run_column", "ColumnResult",
           "axial_area_profile", "radial_profile", "surface_radius_profile",
           "shell_profile_fields", "mode_amplitude", "fit_growth_rate",
           "static_cylinder_curvature", "lattice_kernel_sum_3d"]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _axis_offset(pos: torch.Tensor, box) -> torch.Tensor:
    """Vector from the column axis to each particle, in the periodic box.

    ``run_column`` wraps positions into ``[0, L)``, so the axis sits at the box
    centre.  Measuring ``|x|`` about the origin instead turns the radius into a
    sawtooth of amplitude ``L/2`` -- measured: the mode amplitude collapses from
    2 % to 0.1 % within 25 steps, which is the symptom this helper removes.
    """
    centre = 0.5 * torch.as_tensor(box[:2], dtype=pos.dtype, device=pos.device)
    return pos[:, :2] - centre


def lattice_kernel_sum_3d(dx: float, h: float) -> float:
    """Discrete ``sum_j W_ij`` over a complete 3-D cubic lattice of spacing dx."""
    half = int(math.ceil(2.0 * h / dx)) + 1
    offsets = torch.arange(-half, half + 1, dtype=torch.float64) * dx
    gx, gy, gz = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
    r = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return float(cubic_spline_kernel(r, h).sum())


@dataclass
class ColumnConfig:
    """Geometry, fluid properties and discretisation of the column case."""

    radius: float = 0.006
    dx: float = 0.001
    h_factor: float = 1.3
    rho0: float = 1000.0
    sigma: float = 0.072
    nu: float = 1.0e-6          # kinematic viscosity (water at 20 C)
    sound_speed: float = 3.0    # Mach-limited: max |u| ~ 0.4 m/s -> Ma ~ 0.13
    gamma: float = 7.0
    epsilon: float = 0.02       # initial relative perturbation amplitude
    kR: float = 0.6971          # requested axial wavenumber times radius
    diffusion: float = 0.1      # delta-SPH coefficient (Antuono et al. 2012)
    courant: float = 0.25
    margin_dx: float = 6.0      # void gap per side, in dx (> 4 h / dx)
    pair_rebuild: int = 10
    #: CSF curvature estimator, see ``tensorsph.threed.csf3.CURVATURE_SCHEMES``.
    #: "profile" (default) = axisymmetric cross-section profile, spectrally
    #: differentiated; "axisym" = meridional interface fit + exact n_r/r;
    #: "mls" = raw 3-D div n.  All three are kept so the curvature-estimator
    #: bias can be measured instead of assumed away.
    curvature: str = "profile"
    profile_modes: int = 8
    #: Shepard renormalisation of the summation density, ``rho <- rho S_full /
    #: sum_j W``: removes the free-surface support deficit so that the Tait law
    #: responds to real compression instead of to truncation.  The 2-D solver
    #: exposes the same switch as ``Config.density_renormalization``, where it is
    #: what keeps a hydrostatic column from collapsing.
    renormalise_density: bool = True
    #: Adami free-surface pressure extrapolation (Adami et al. 2012 §3.3),
    #: ``p_i = sum_j p_j W_ij / sum_j W_ij`` for particles below
    #: ``surface_pressure_threshold * rho0``.  Without it the whole surface layer
    #: carries ``p = 0`` and nothing balances the interior Laplace pressure
    #: ``sigma/R`` -- the defect measured in
    #: ``docs/verification_laplace_pressure_floor.md`` for the 2-D solver.
    surface_pressure: bool = True
    surface_pressure_threshold: float = 0.99
    dtype: torch.dtype = torch.float64

    @property
    def h(self) -> float:
        return self.h_factor * self.dx

    @property
    def nz(self) -> int:
        """Axial cells of the periodic box: the mode must close on the period.

        A single-mode perturbation in a periodic box must have
        ``k = 2 pi m / Lz`` exactly, so the box length is rounded to an integer
        number of cells and the *realised* wavenumber is :attr:`kR_actual`.
        Requesting ``kR`` and ignoring the rounding leaves a phase jump across
        the periodic seam (measured: the mode projection collapses to zero at
        ``R/dx = 4``).
        """
        return max(4, int(round(2.0 * math.pi * self.radius / (self.kR * self.dx))))

    @property
    def wavelength(self) -> float:
        return self.nz * self.dx

    @property
    def kR_actual(self) -> float:
        return 2.0 * math.pi * self.radius / self.wavelength

    @property
    def box(self):
        pad = 2.0 * self.radius + 2.0 * self.margin_dx * self.dx
        return (pad, pad, self.wavelength)

    @property
    def cutoff(self) -> float:
        """Pair cutoff: kernel support, stretched to the fit stencil if used."""
        return 4.0 * self.h if self.curvature == "axisym" else 2.0 * self.h

    @property
    def dt(self) -> float:
        """Acoustic CFL step, capped by the surface-tension and viscous limits."""
        dt_acoustic = self.courant * self.h / self.sound_speed
        dt_capillary = (0.25 * math.sqrt(self.rho0 * self.dx ** 3
                                         / (2.0 * math.pi * self.sigma))
                        if self.sigma > 0.0 else float("inf"))
        dt_viscous = 0.125 * self.h ** 2 / max(self.nu, 1e-30)
        return min(dt_acoustic, dt_capillary, dt_viscous)

    @property
    def capillary_time(self) -> float:
        return capillary_time(self.sigma, self.rho0, self.radius)


def build_column(cfg: ColumnConfig):
    """Particle positions, masses and the periodic box of the perturbed column.

    The lattice is cubic, cut to ``r <= R``, and the perturbation is applied as a
    *radial map* ``r -> r (1 + eps cos k z)``.

    Mapping instead of masking is not cosmetic: a 2 % perturbation of a radius
    that is 4 cells wide is 0.08 dx, so a mask ``r <= R (1 + eps cos k z)``
    rounds the perturbation away completely -- the measured initial mode
    amplitude is *exactly zero* at ``R/dx = 4``.  The radial map represents the
    surface to machine precision, conserves each particle's mass (hence the
    volume, up to ``O(eps^2)``), keeps the lattice quasi-uniform and leaves a
    genuinely incompressible initial field.

    The mass is calibrated so that the summation density of a particle with
    complete support equals ``rho0`` (the 2-D solver's
    ``Config.calibrate_initial_mass``).  The material volume a particle
    represents is the lattice cell ``dx^3`` and is kept separate from that
    calibration -- see :func:`axial_area_profile`.
    """
    dx, h = cfg.dx, cfg.h
    lx, ly, lz = cfg.box
    dt = cfg.dtype
    xs = torch.arange(0.5 * dx, lx, dx, dtype=dt)
    ys = torch.arange(0.5 * dx, ly, dx, dtype=dt)
    zs = torch.arange(0.5 * dx, lz, dx, dtype=dt)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    pos = torch.stack((gx.reshape(-1) - 0.5 * lx, gy.reshape(-1) - 0.5 * ly,
                       gz.reshape(-1)), dim=-1)
    radius = torch.linalg.vector_norm(pos[:, :2], dim=1)
    pos = pos[radius <= cfg.radius].contiguous()

    k = 2.0 * math.pi / cfg.wavelength
    r = torch.linalg.vector_norm(pos[:, :2], dim=1)
    radial = pos[:, :2] / r.clamp_min(1e-30).unsqueeze(-1)
    scale = 1.0 + cfg.epsilon * torch.cos(k * pos[:, 2])
    pos[:, :2] = radial * (r * scale).unsqueeze(-1)

    # Density-neutral mass: the radial map compresses the cross-section at a neck
    # by (1 + eps cos)^2, so a *uniform* particle mass would raise the local
    # summation density by that factor and put a 200 Pa Tait pressure step
    # across the mode (measured: max p = 413 Pa, i.e. 5.7 sigma/R for this
    # radius) which radiates an acoustic transient strong enough to mask the
    # capillary mode.  Scaling the mass by the same factor keeps rho = rho0 to
    # the lattice bias and leaves a pressure-free initial state; the total mass
    # is renormalised so the body carries exactly rho0 * pi R^2 * lambda.
    mass_unit = cfg.rho0 / lattice_kernel_sum_3d(dx, h)
    shape = (1.0 + cfg.epsilon * torch.cos(k * pos[:, 2])) ** 2
    mass = mass_unit * shape
    target = cfg.rho0 * math.pi * cfg.radius ** 2 * cfg.wavelength
    mass = mass * (target / mass.sum())
    # Place the column on the box axis so that wrapping into ``[0, L)`` is a
    # no-op: with the cloud centred on the origin instead, the wrapped image
    # straddles the box corner and every radius measured about the centre is
    # wrong by up to ``L/2`` (measured: shell radius 0.011 m for R = 0.004 m).
    pos = pos.clone()
    pos[:, 0] = pos[:, 0] + 0.5 * lx
    pos[:, 1] = pos[:, 1] + 0.5 * ly
    return pos, mass, cfg.box


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def axial_area_profile(pos: torch.Tensor, mass: torch.Tensor, rho0: float,
                       box, nbin: int | None = None,
                       volume: torch.Tensor | None = None):
    """Cross-section area per unit length, ``A(z) = sum_{slab} V_i / dz``.

    ``V_i`` is the material volume of a particle: the lattice cell ``dx^3`` for a
    cubic lattice (pass ``volume``) or, by default, the rest volume ``m_i/rho0``.
    The two differ by the lattice kernel-sum bias (``sum_j W dx^3 = 0.964`` at
    ``h = 1.3 dx``), a 3.7 % *radius* bias if the wrong one is used for geometry:
    the mass calibration exists to make the summation density equal ``rho0``, not
    to define the cell volume.

    Note that ``A`` is a *volume* measure: it is invariant under a radial map, so
    it carries no Rayleigh-Plateau signal at ``t = 0`` (see the module docstring).
    It is kept for the volume-conservation diagnostic.
    """
    lz = float(box[2])
    if nbin is None:
        nbin = max(4, int(round(lz / 1.0)))
    dz = lz / nbin
    idx = torch.clamp((pos[:, 2] / lz * nbin).to(torch.int64), 0, nbin - 1)
    vol = mass / rho0 if volume is None else volume
    per_bin = torch.zeros(nbin, dtype=pos.dtype)
    per_bin = per_bin.index_add(0, idx, vol)
    return per_bin / dz, dz


def radial_profile(pos: torch.Tensor, box, nbin: int | None = None) -> dict:
    """Surface radius per slab from the second moment ``sqrt(2 <r^2>_slab)``.

    Exact on an undeformed lattice (``<r^2> = R^2/2`` for a uniform disc, and a
    radial map scales it by ``(1 + eps cos)^2``), but its noise floor once the
    particles have moved is several percent of ``R`` -- too large for a 2 %
    perturbation.  Kept as the *cross-check* on the initial condition and as the
    documented counter-example to "measure the radius directly".
    """
    lz = float(box[2])
    if nbin is None:
        nbin = max(4, int(round(lz / 1.0)))
    dz = lz / nbin
    idx = torch.clamp((pos[:, 2] / lz * nbin).to(torch.int64), 0, nbin - 1)
    r2 = _axis_offset(pos, box).square().sum(dim=1)
    count = torch.zeros(nbin, dtype=pos.dtype)
    count = count.index_add(0, idx, torch.ones_like(r2))
    total = torch.zeros(nbin, dtype=pos.dtype)
    total = total.index_add(0, idx, r2)
    r_max = torch.full((nbin,), -1.0, dtype=pos.dtype)
    r_max = r_max.scatter_reduce(0, idx, torch.sqrt(r2), reduce="amax",
                                 include_self=True)
    radius = torch.sqrt(2.0 * total / count.clamp_min(1.0))
    return {"radius": radius, "r_max": r_max, "count": count, "dz": dz,
            "nbin": nbin}


def colour_gradient(pos: torch.Tensor, mass: torch.Tensor, h: float,
                    pairs: torch.Tensor, *, rho0: float, box=None):
    """``c_i = sum_j V_j W_ij`` and ``grad c_i`` with the plain kernel gradient."""
    vol = mass / rho0
    i, j = pairs[:, 0], pairs[:, 1]
    delta, weight, grad, _ = pair_geometry(pos, pairs, h, box)
    self_weight = float(cubic_spline_kernel(
        torch.zeros(1, dtype=pos.dtype, device=pos.device), h)[0])
    color = torch.zeros_like(mass)
    color = color.index_add(0, i, vol[j] * weight)
    color = color.index_add(0, j, vol[i] * weight)
    color = color + vol * self_weight
    dc = color[j] - color[i]
    g = torch.zeros_like(pos)
    g = g.index_add(0, i, (vol[j] * dc).unsqueeze(-1) * grad)
    g = g.index_add(0, j, (vol[i] * dc).unsqueeze(-1) * grad)
    return color, g


def surface_radius_profile(pos: torch.Tensor, mass: torch.Tensor, h: float,
                           pairs: torch.Tensor, *, rho0: float, box, nbin: int,
                           shell_fraction: float = 0.3,
                           calibration: float = 1.0) -> dict:
    """Radius of the interface shell per axial slab, from the CSF interface weight.

    Particles with ``delta_s = |grad c| > shell_fraction * max`` form a shell of
    thickness ``~2 h`` around the surface; the ``delta_s``-weighted mean radius of
    that shell in a slab is the surface position plus a *constant* offset, scaled
    by ``calibration`` (fitted once on the initial cylinder, where the radius is
    known analytically).  A constant offset is invisible to the ``k``-mode
    projection, and kernel smoothing attenuates the mode by a constant factor
    (``exp(-k^2 (2h)^2/2) ~ 0.97`` at ``kR = 0.7``, ``R/dx = 8``); both cancel in
    the growth *rate*, which is what the anchor scores.
    """
    color, grad_c = colour_gradient(pos, mass, h, pairs, rho0=rho0, box=box)
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    shell = delta_s > shell_fraction * float(delta_s.max())
    r = torch.linalg.vector_norm(_axis_offset(pos, box), dim=1)
    lz = float(box[2])
    dz = lz / nbin
    idx = torch.clamp((pos[:, 2] / lz * nbin).to(torch.int64), 0, nbin - 1)
    weight = delta_s * shell
    num = torch.zeros(nbin, dtype=pos.dtype)
    num = num.index_add(0, idx, weight * r)
    den = torch.zeros(nbin, dtype=pos.dtype)
    den = den.index_add(0, idx, weight)
    radius = num / den.clamp_min(1e-30)
    radius = torch.where(den > 0.0, radius,
                         torch.full_like(radius, float(r.mean())))
    return {"radius": radius * calibration, "weight": den, "dz": dz,
            "nbin": nbin, "shell_particles": int(shell.sum()), "color": color,
            "grad_c": grad_c, "delta_s": delta_s}


def _spectral_derivatives(profile: torch.Tensor, dz: float, nmodes: int):
    n = profile.shape[0]
    spec = torch.fft.rfft(profile.to(torch.float64))
    m = torch.arange(spec.shape[0], dtype=torch.float64)
    k_m = 2.0 * math.pi * m / (n * dz)
    spec = torch.where(m <= nmodes, spec, torch.zeros_like(spec))
    d1 = torch.fft.irfft(1j * k_m * spec, n=n).to(profile.dtype)
    d2 = torch.fft.irfft(-(k_m ** 2) * spec, n=n).to(profile.dtype)
    return d1, d2


def shell_profile_fields(pos: torch.Tensor, mass: torch.Tensor, h: float,
                         pairs: torch.Tensor, *, rho0: float, box, nbin: int,
                         nmodes: int = 8, calibration: float = 1.0,
                         shell_fraction: float = 0.3) -> dict:
    """Curvature and normal of the axisymmetric surface from its profile.

    For a surface of revolution ``r = r_s(z)`` with no azimuthal structure
    (``m = 0``) the mean curvature is exact in terms of the profile::

        kappa(z) = 1 / (r_s sqrt(1 + r_s'^2)) - r_s'' / (1 + r_s'^2)^{3/2},

    which returns ``1/R`` for a cylinder and ``2/R`` for a sphere.  The profile is
    measured by :func:`surface_radius_profile`, differentiated spectrally in
    ``z`` -- legitimate because the box is periodic along ``z`` -- and
    band-limited to ``nmodes`` harmonics.  The band limit is a *modelling* choice,
    swept in the report: it removes lattice-scale interface noise from the
    curvature, which at ``R/dx = 6`` otherwise carries an ``O(10 %)`` bias
    (``docs/verification_rayleigh_plateau.md``, curvature-estimator study).
    """
    prof = surface_radius_profile(pos, mass, h, pairs, rho0=rho0, box=box,
                                  nbin=nbin, shell_fraction=shell_fraction,
                                  calibration=calibration)
    r_s = prof["radius"]
    d1, d2 = _spectral_derivatives(r_s, prof["dz"], nmodes)
    slope2 = 1.0 + d1 * d1
    root = torch.sqrt(slope2)
    kappa = 1.0 / (r_s * root) - d2 / slope2.pow(1.5)
    prof.update({"d1": d1, "d2": d2, "kappa": kappa, "n_r": 1.0 / root,
                 "n_z": -d1 / root, "mean_radius": float(r_s.mean())})
    return prof


def mode_amplitude(profile: torch.Tensor, dz: float, k: float):
    """Complex Fourier amplitude of ``profile - <profile>`` at wavenumber ``k``.

    Discretised as ``(2 / N) sum_s (f_s - <f>) exp(-i k z_s)`` so a pure cosine of
    amplitude ``a`` gives ``|amp| = a`` (the equal-time weight ``dz`` cancels).
    """
    n = profile.shape[0]
    z = (torch.arange(n, dtype=profile.dtype) + 0.5) * dz
    centred = profile - profile.mean()
    phase = torch.exp(-1j * k * z)
    return (2.0 / n) * torch.sum(centred.to(torch.complex128) * phase)


def fit_growth_rate(times, amplitudes, low: float, high: float) -> dict:
    """Least-squares slope of ``log(amplitude)`` inside ``[low, high] * a0``.

    The window is predeclared (surface tension grows exponentially only while the
    amplitude is small; the fit must not see the saturated/pinch-off part), and
    the returned ``r2`` shows whether the fitted interval really is exponential.
    """
    t = torch.as_tensor(times, dtype=torch.float64)
    a = torch.as_tensor(amplitudes, dtype=torch.float64)
    a0 = float(a[0])
    sel = (a >= low * a0) & (a <= high * a0)
    if int(sel.sum()) < 3:
        return {"omega": float("nan"), "r2": float("nan"),
                "n_points": int(sel.sum()), "window": [float("nan")] * 2}
    tt, yy = t[sel], torch.log(a[sel])
    slope = ((tt - tt.mean()) * (yy - yy.mean())).sum() / ((tt - tt.mean()) ** 2).sum()
    intercept = yy.mean() - slope * tt.mean()
    resid = yy - (slope * tt + intercept)
    ss_tot = ((yy - yy.mean()) ** 2).sum().clamp_min(1e-300)
    r2 = 1.0 - (resid ** 2).sum() / ss_tot
    return {"omega": float(slope), "r2": float(r2), "n_points": int(sel.sum()),
            "window": [float(a[sel][0]), float(a[sel][-1])]}


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
@dataclass
class ColumnResult:
    """Everything a run produces (raw series + the fitted growth rate)."""

    cfg: ColumnConfig
    times: list = field(default_factory=list)
    amplitudes: list = field(default_factory=list)       # relative radial mode
    null_amplitudes: list = field(default_factory=list)  # non-resonant control
    harmonic_amplitudes: list = field(default_factory=list)
    volume_error: list = field(default_factory=list)   # reconstructed volume
    density_error: list = field(default_factory=list)  # max |rho/rho0 - 1|
    max_velocity: list = field(default_factory=list)
    profiles: list = field(default_factory=list)
    positions: torch.Tensor | None = None
    velocities: torch.Tensor | None = None
    densities: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)

    def growth(self, low: float = 3.0, high: float = 25.0) -> dict:
        """Exponential growth rate of the relative radial mode amplitude."""
        return fit_growth_rate(self.times, self.amplitudes, low, high)

    def null_growth(self) -> dict:
        """Growth of a non-resonant mode: the noise floor of the measurement."""
        return fit_growth_rate(self.times, self.null_amplitudes, 1.0, 1.0e9)

    def to_dict(self) -> dict:
        cfg = self.cfg
        growth = self.growth()
        omega_ref = dispersion_rate(cfg.sigma, cfg.rho0, cfg.radius,
                                    cfg.kR_actual / cfg.radius)
        return {
            "kR": cfg.kR_actual,
            "kR_requested": cfg.kR,
            "radius": cfg.radius,
            "dx": cfg.dx,
            "radius_over_dx": cfg.radius / cfg.dx,
            "sigma": cfg.sigma,
            "wavelength": cfg.wavelength,
            "lambda_over_radius": cfg.wavelength / cfg.radius,
            "capillary_time": cfg.capillary_time,
            "dt": cfg.dt,
            "steps": len(self.times),
            "particles": int(self.positions.shape[0]) if self.positions is not None else None,
            "curvature": cfg.curvature,
            "omega_measured": growth["omega"],
            "omega_null": self.null_growth()["omega"],
            "omega_analytic": omega_ref,
            "omega_error": (growth["omega"] / omega_ref - 1.0) if omega_ref else float("nan"),
            "growth_fit_r2": growth["r2"],
            "growth_window": growth["window"],
            "max_volume_error": max(self.volume_error) if self.volume_error else None,
            "final_volume_error": self.volume_error[-1] if self.volume_error else None,
            "max_density_error": max(self.density_error) if self.density_error else None,
            "max_velocity": max(self.max_velocity) if self.max_velocity else None,
            "metadata": self.metadata,
        }


def _pinched(radius: torch.Tensor, threshold: float = 0.30) -> bool:
    """A neck has pinched when its slab radius falls below 30 % of the mean."""
    mean = float(radius.mean())
    return bool(torch.nan_to_num(radius, nan=mean).min() < threshold * mean)


def static_cylinder_curvature(radius: float, dx: float, h_factor: float = 1.3,
                              curvature: str = "profile",
                              profile_modes: int = 8,
                              dtype=torch.float64) -> dict:
    """CSF curvature of a static cylinder: the ``1/R`` anchor for the estimators.

    For an infinite cylinder ``div n = 1/R`` exactly, so this scores the
    curvature operator without any time integration.  Only interface particles
    are scored; the ``profile`` estimator is compared on the same cloud.
    """
    cfg = ColumnConfig(radius=radius, dx=dx, h_factor=h_factor,
                       kR=0.6, epsilon=0.0, curvature=curvature,
                       profile_modes=profile_modes, dtype=dtype)
    pos, mass, box = build_column(cfg)
    h = cfg.h
    pairs = periodic_pairs(pos, box, cfg.cutoff)
    if curvature == "profile":
        probe = surface_radius_profile(pos, mass, h, pairs, rho0=cfg.rho0, box=box,
                                       nbin=cfg.nz)
        calib = radius / float(probe["radius"].mean())
        prof = shell_profile_fields(pos, mass, h, pairs, rho0=cfg.rho0, box=box,
                                    nbin=cfg.nz, nmodes=profile_modes,
                                    calibration=calib)
        kappa = float(prof["kappa"].mean())
        return {"radius": radius, "radius_over_dx": radius / dx,
                "particles": int(pos.shape[0]), "estimator": "profile",
                "shell_radius_over_R": float(probe["radius"].mean()) / radius,
                "radius_calibration": calib,
                "kappa_mean": kappa, "kappa_analytic": 1.0 / radius,
                "relative_error_mean": kappa * radius - 1.0,
                "mean_radius_measured": prof["mean_radius"]}
    state = csf_fields3(pos, mass, h, pairs, rho0=cfg.rho0, box=box,
                        curvature=curvature)
    shell = state.surface_mask(0.3) & state.valid
    kappa = state.kappa[shell]
    r = torch.linalg.vector_norm(_axis_offset(pos, box)[shell], dim=1)
    return {
        "radius": radius,
        "radius_over_dx": radius / dx,
        "particles": int(pos.shape[0]),
        "estimator": curvature,
        "shell_particles": int(shell.sum()),
        "kappa_mean": float(kappa.mean()),
        "kappa_median": float(kappa.median()),
        "kappa_analytic": 1.0 / radius,
        "relative_error_mean": float(kappa.mean() * radius) - 1.0,
        "relative_error_median": float(kappa.median() * radius) - 1.0,
        "normal_radial_cos": float((state.normal[shell, :2]
                                    * (_axis_offset(pos, box)[shell] / r.unsqueeze(-1))
                                    ).sum(dim=1).mean()),
    }


def run_column(cfg: ColumnConfig, *, steps: int = 4000, sample_every: int = 20,
               stop_on_pinch: bool = False, progress=None) -> ColumnResult:
    """Integrate the perturbed column and record the axial mode amplitude."""
    pos, mass, box = build_column(cfg)
    pos = pos.remainder(torch.as_tensor(cfg.box, dtype=pos.dtype))
    h = cfg.h
    dt = cfg.dt
    vel = torch.zeros_like(pos)
    pairs = periodic_pairs(pos, box, cfg.cutoff)
    k = 2.0 * math.pi / cfg.wavelength
    volume0 = math.pi * cfg.radius ** 2 * cfg.wavelength

    # The interface-shell radius carries a constant geometric offset (the shell
    # is ~2 h thick, so on a thin cylinder its mean radius sits well inside the
    # surface).  Calibrate it once, on the initial cylinder whose radius is known
    # analytically, and use the same factor for the curvature and the diagnostic.
    calib = 1.0
    if cfg.curvature == "profile":
        probe = surface_radius_profile(pos, mass, h, pairs, rho0=cfg.rho0, box=box,
                                       nbin=cfg.nz)
        calib = cfg.radius / float(probe["radius"].mean())

    result = ColumnResult(cfg=cfg)
    result.metadata.update({
        "particles": int(pos.shape[0]), "box": list(box), "h": h, "dt": dt,
        "kR": cfg.kR_actual, "kR_requested": cfg.kR, "epsilon": cfg.epsilon,
        "diffusion": cfg.diffusion, "sound_speed": cfg.sound_speed, "nu": cfg.nu,
        "curvature": cfg.curvature, "profile_modes": cfg.profile_modes,
        "radius_calibration": calib, "dtype": str(cfg.dtype),
    })

    def kernel_totals(p):
        _, weight, _, _ = pair_geometry(p, pairs, h, box)
        i, j = pairs[:, 0], pairs[:, 1]
        self_weight = float(cubic_spline_kernel(
            torch.zeros(1, dtype=p.dtype, device=p.device), h)[0])
        total = torch.zeros(p.shape[0], dtype=p.dtype, device=p.device)
        total = total.index_add(0, i, weight)
        total = total.index_add(0, j, weight)
        total = total + self_weight
        return total, weight, i, j

    s_full = lattice_kernel_sum_3d(cfg.dx, h)

    def rhs(p, v, rho_prev=None):
        rho = summation_density(p, mass, h, pairs, box)
        if cfg.renormalise_density:
            total, _, _, _ = kernel_totals(p)
            rho = rho * (s_full / total)
        if cfg.diffusion > 0.0 and rho_prev is not None:
            # delta-SPH: D rho_i = delta c0 h sum_j V_j 2 (rho_i - rho_j)
            #             (x_i - x_j) . grad_i W / r^2
            # Signs and pair bookkeeping follow
            # ``droplet_impact.density_diffusion_rate`` exactly: with
            # L = -(delta . grad W)/r^2 > 0 the term is
            # -2 delta c0 h sum_j V_j (rho_i - rho_j) L, i.e. a density
            # *diffusion* whose volume-weighted mean vanishes.
            dvec, _, grad, _ = pair_geometry(p, pairs, h, box)
            r2 = dvec.square().sum(dim=1).clamp_min(1e-30)
            radial = -(dvec * grad).sum(dim=1)
            i, j = pairs[:, 0], pairs[:, 1]
            coupling = (rho[i] - rho[j]) * radial / r2
            scale = 2.0 * cfg.diffusion * cfg.sound_speed * h
            vol = mass / rho_prev
            drho = torch.zeros_like(rho)
            drho = drho.index_add(0, i, -scale * vol[j] * coupling)
            drho = drho.index_add(0, j, scale * vol[i] * coupling)
            rho = (rho + dt * drho).clamp_min(0.5 * cfg.rho0)
        pressure = tait_pressure(rho, cfg.rho0, cfg.sound_speed, cfg.gamma)
        if cfg.surface_pressure:
            total, weight, i, j = kernel_totals(p)
            donor = pressure > 0.0
            num = torch.zeros_like(pressure)
            num = num.index_add(0, i, weight * (pressure[j] * donor[j]))
            num = num.index_add(0, j, weight * (pressure[i] * donor[i]))
            extrapolated = num / total.clamp_min(1e-30)
            sparse = rho < cfg.surface_pressure_threshold * cfg.rho0
            pressure = torch.where(sparse, extrapolated.clamp_min(0.0), pressure)
        acc = pressure_acceleration(p, mass, rho, pressure, h, pairs, box)
        acc = acc + viscous_acceleration(p, v, mass, rho, cfg.nu, h, pairs, box)
        if cfg.sigma > 0.0:
            acc = acc + _surface_tension(p, mass, h, pairs, rho, box, cfg, calib)
        return acc, rho

    def sample(t):
        prof = surface_radius_profile(pos, mass, h, pairs, rho0=cfg.rho0, box=box,
                                      nbin=cfg.nz, calibration=calib)
        surf = prof["radius"]
        ref = float(surf.mean())
        result.times.append(t)
        result.amplitudes.append(
            float(abs(mode_amplitude(surf, prof["dz"], k))) / ref)
        # 3.5 k is far from every harmonic of the forced mode, so the projection
        # is a genuine noise floor and not spectral leakage from the fundamental.
        result.null_amplitudes.append(
            float(abs(mode_amplitude(surf, prof["dz"], 3.5 * k))) / ref)
        result.harmonic_amplitudes.append(
            float(abs(mode_amplitude(surf, prof["dz"], 2.0 * k))) / ref)
        # Volume from the reconstructed body of revolution, not from sum m/rho:
        # a free-surface particle has rho << rho0, so sum m/rho counts the void
        # and drifts by tens of percent on any free-surface run (measured 31 % on
        # a *static* column) -- it is a compression measure, not a volume one.
        vol = float((math.pi * surf.square() * prof["dz"]).sum())
        result.volume_error.append(vol / volume0 - 1.0)
        result.density_error.append(float((rho / cfg.rho0 - 1.0).abs().max()))
        result.max_velocity.append(float(torch.linalg.vector_norm(vel, dim=1).max()))
        result.profiles.append(surf.clone())
        return prof

    rho = summation_density(pos, mass, h, pairs, box)
    acc, rho = rhs(pos, vel, rho)
    sample(0.0)

    for step in range(1, steps + 1):
        if step % cfg.pair_rebuild == 0:
            pairs = periodic_pairs(pos, box, cfg.cutoff)
        vel = vel + 0.5 * dt * acc
        pos = pos + dt * vel
        pos = pos.remainder(torch.as_tensor(box, dtype=pos.dtype))
        acc, rho = rhs(pos, vel, rho)
        vel = vel + 0.5 * dt * acc

        if step % sample_every == 0 or step == steps:
            prof = sample(step * dt)
            if progress is not None:
                progress(step, result.times[-1], result.amplitudes[-1])
            if stop_on_pinch and _pinched(prof["radius"]):
                result.metadata["pinch_time"] = step * dt
                break

    result.positions = pos
    result.velocities = vel
    result.densities = rho
    return result


def _surface_tension(pos, mass, h, pairs, rho, box, cfg, calib) -> torch.Tensor:
    """CSF acceleration for the configured curvature estimator."""
    if cfg.curvature == "profile":
        prof = shell_profile_fields(pos, mass, h, pairs, rho0=cfg.rho0, box=box,
                                    nbin=cfg.nz, nmodes=cfg.profile_modes,
                                    calibration=calib)
        idx = torch.clamp((pos[:, 2] / float(box[2]) * prof["nbin"]).to(torch.int64),
                          0, prof["nbin"] - 1)
        offset = _axis_offset(pos, box)
        r = torch.linalg.vector_norm(offset, dim=1).clamp_min(1e-30)
        radial = offset / r.unsqueeze(-1)
        n_r, n_z = prof["n_r"][idx], prof["n_z"][idx]
        normal = torch.cat((radial * n_r.unsqueeze(-1), n_z.unsqueeze(-1)), dim=1)
        state = CSFState3(color=None, grad_c=prof["grad_c"], normal=normal,
                          kappa=prof["kappa"][idx],
                          delta_s=torch.linalg.vector_norm(prof["grad_c"], dim=1),
                          valid=torch.ones_like(r, dtype=torch.bool))
    else:
        state = csf_fields3(pos, mass, h, pairs, rho0=cfg.rho0, box=box,
                            curvature=cfg.curvature)
    return surface_tension_acceleration3(state, rho, cfg.sigma)
