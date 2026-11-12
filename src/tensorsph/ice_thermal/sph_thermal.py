"""SPH discretisation of the heat equation, coupled to the enthalpy method.

The energy equation solved on the particles is the standard SPH conduction
operator (Cleary & Monaghan 1999, "conduction" form), written directly in
enthalpy so that phase change needs no heat-capacity division::

    dH_i/dt = (2 / rho_i) * sum_j V_j * (2 k_i k_j / (k_i + k_j))
              * (T_i - T_j) * F_ij
    F_ij    = (r_ij . grad_i W_ij) / (r_ij^2 + eta^2)
    T_i     = e^{-1}(H_i, S_i)                        (enthalpy method)

``H`` in J/kg, ``V`` in m^3 (1-D: m), ``F`` in 1/m^3 so that ``V_j F_ij`` is
dimensionless.  The harmonic mean of the conductivities is the correct face
value for conduction in series across a particle pair; that is what makes an
ice/water interface (``k_i != k_j``) carry the right flux.  In the mushy band
the conductivity is blended with the liquid fraction rather than switched, so
the operator stays continuous while the front sweeps a particle.

Lattice consistency
-------------------
On a regular lattice the discrete operator has the symbol ``-alpha_eff k^2``
which is *not* exactly ``-alpha k^2``: the 1-D cubic-spline kernel at
``h = 1.3 dx`` under-resolves the second moment, so ``alpha_eff / alpha > 1``.
Rather than pretending otherwise the ratio is measured
(:func:`lattice_diffusion_factor`) and used as a documented calibration -- the
same practice as the existing :mod:`tensorsph.lattice_consistency`
zeroth-moment calibration.  The factor, its scatter over wavenumbers and its
``O(k^2 dx^2)`` trend are all reported in the benchmark JSON; the calibrated
value is what the Stefan anchor is run with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import properties as props
from .enthalpy import EnthalpyModel, neumann_lambda

__all__ = [
    "kernel_w",
    "kernel_dwdr",
    "conduction_rhs",
    "lattice_diffusion_factor",
    "StefanConfig",
    "StefanResult",
    "run_stefan_sph",
]

DTYPE = torch.float64


# --------------------------------------------------------------- 1-D kernel
def kernel_w(r, h):
    """1-D cubic-spline (M4) kernel, ``int W dr = 1``."""
    rr = torch.as_tensor(r, dtype=DTYPE) / float(h)
    q = rr.abs()
    norm = 2.0 / (3.0 * float(h))
    inner = 1.0 - 1.5 * q ** 2 + 0.75 * q ** 3
    outer = 0.25 * (2.0 - q) ** 3
    return norm * torch.where(q < 1.0, inner, torch.where(q < 2.0, outer,
                                                         torch.zeros_like(q)))


def kernel_dwdr(r, h):
    """Radial derivative ``dW/dr`` of the 1-D cubic-spline kernel (1/m^2)."""
    rr = torch.as_tensor(r, dtype=DTYPE)
    q = (rr / float(h)).abs()
    norm = 2.0 / (3.0 * float(h) ** 2)
    inner = -3.0 * q + 2.25 * q ** 2
    outer = -0.75 * (2.0 - q) ** 2
    val = norm * torch.where(q < 1.0, inner, torch.where(q < 2.0, outer,
                                                        torch.zeros_like(q)))
    return torch.sign(rr) * val


def pair_geometry(pos, volume, h, eta_factor=0.01):
    """Pre-factor ``V_j F_ij`` (1/m) -- geometry only, constant for fixed particles."""
    x = torch.as_tensor(pos, dtype=DTYPE)
    v = torch.as_tensor(volume, dtype=DTYPE)
    dr = x[:, None] - x[None, :]
    r = dr.abs()
    eta2 = (eta_factor * float(h)) ** 2
    f_ij = r * kernel_dwdr(r, h) / (r ** 2 + eta2)
    return f_ij * v[None, :]


def conduction_rhs(temperature, conductivity, volume, rho, pos, h, eta_factor=0.01,
                   fv=None):
    """``dH/dt`` (J/(kg s)) from the SPH conduction operator; shapes (N,).

    ``fv`` optionally carries the frozen pair matrix ``V_j F_ij`` (from
    :func:`pair_geometry`) when the geometry does not change between calls --
    a time loop that reuses it avoids rebuilding the ``O(N^2)`` kernel every
    step.  It must have been built from the *same* positions/volumes.
    """
    t = torch.as_tensor(temperature, dtype=DTYPE)
    k = torch.as_tensor(conductivity, dtype=DTYPE)
    v = torch.as_tensor(volume, dtype=DTYPE)
    rho_t = torch.as_tensor(rho, dtype=DTYPE)
    k, v, t = torch.broadcast_tensors(k, v, t)
    if fv is None:
        fv = pair_geometry(pos, v, h, eta_factor)
    k_ij = 2.0 * k[:, None] * k[None, :] / (k[:, None] + k[None, :])
    kernel = k_ij * fv
    return 2.0 / rho_t * (t * kernel.sum(dim=1) - kernel @ t)


def lattice_diffusion_factor(n=256, h_factor=1.3, eta_factor=0.01, kmodes=None):
    """Ratio ``alpha_eff / alpha`` of the SPH conduction operator on a periodic lattice.

    For each test wavenumber ``k`` the operator is applied to ``cos(k x)`` on a
    periodic 1-D lattice of ``n`` particles (spacing 1, ``rho cp = 1``,
    ``k_cond = 1`` so ``alpha = 1``); the discrete answer is
    ``-alpha_eff k^2 cos(kx)`` and the ratio follows from a least-squares
    projection.  Returns ``(factor, scatter)``: ``scatter`` is the spread over
    the modes, i.e. the measure of how legitimately a single factor can be
    used.  The default modes are the three longest resolvable ones
    (``k dx <= 0.074``), where a diffusion process actually lives.
    """
    dx = 1.0
    h = h_factor * dx
    pos = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    if kmodes is None:
        kmodes = [2.0 * math.pi * m / (n * dx) for m in (1, 2, 3)]
    ratios = []
    for k in kmodes:
        t = torch.cos(k * pos)
        pos_ext = torch.cat([pos - n * dx, pos, pos + n * dx])
        t_ext = torch.cat([t, t, t])
        k_ext = torch.ones(3 * n, dtype=DTYPE)
        v_ext = torch.full((3 * n,), dx, dtype=DTYPE)
        rhs = conduction_rhs(t_ext, k_ext, v_ext, 1.0, pos_ext, h, eta_factor)
        rhs = rhs[n:2 * n]
        ratios.append(float(-(rhs * t).sum() / ((t * t).sum() * k ** 2)))
    ratios_t = torch.tensor(ratios, dtype=DTYPE)
    return float(ratios_t.mean()), float(ratios_t.max() - ratios_t.min())


# ---------------------------------------------------------------- Stefan run
@dataclass
class StefanConfig:
    """One-dimensional SPH Stefan (freezing) case, per unit area."""

    n_particles: int = 120
    dx: float = 0.01          # m
    h_factor: float = 1.3
    wall_temperature: float = 263.15   # K, Dirichlet surface (-10 C)
    initial_temperature: float = props.T_MELT
    mushy_dt: float = 0.05    # K
    t_end: float = 4.0e4      # s
    courant: float = 0.35
    calibrate: bool = True
    pin_wall: bool = False    # True: also pin particle 0 to T_wall (adds a half cell)
    profile_every: int = 40
    seed: int = 0


@dataclass
class StefanResult:
    """Time series + diagnostics of the SPH Stefan run."""

    time: list = field(default_factory=list)
    front: list = field(default_factory=list)
    front_analytic: list = field(default_factory=list)
    lam: float = 0.0
    stefan_number: float = 0.0
    dt: float = 0.0
    steps: int = 0
    diffusion_factor: float = 1.0
    diffusion_scatter: float = 0.0
    profile_x: list = field(default_factory=list)
    profile_t: list = field(default_factory=list)
    profile_analytic: list = field(default_factory=list)
    cfg: StefanConfig = field(default_factory=StefanConfig)

    @property
    def final_front(self) -> float:
        return self.front[-1]

    def front_error(self, skip: int = 3):
        """Max/mean relative error of the SPH front vs the Neumann solution."""
        errs = []
        for t, xs, xa in zip(self.time, self.front, self.front_analytic):
            if t <= 0.0 or xa <= 1e-9 or xs <= 0.0:
                continue
            errs.append(abs(xs - xa) / xa)
        errs = errs[skip:]
        if not errs:
            return float("nan"), float("nan")
        return max(errs), sum(errs) / len(errs)

    def profile_error(self):
        """RMS temperature error in the solid vs the Neumann profile (K)."""
        if not self.profile_x:
            return float("nan")
        a = torch.tensor(self.profile_t, dtype=DTYPE)
        b = torch.tensor(self.profile_analytic, dtype=DTYPE)
        return float(torch.sqrt(((a - b) ** 2).mean()))


def _front_from_enthalpy(enthalpy, model: EnthalpyModel, dx, salinity=0.0):
    """Solidified thickness ``X = sum (1 - w_l) dx`` (latent-heat consistent).

    For the fresh-ice model the frozen fraction is read off the mushy band,
    ``f = clamp((T_melt - T)/mushy_dt, 0, 1)``; for sea ice it is ``1 - w_l``.
    Using the liquid fraction of the *fresh* model would be wrong (it is zero at
    ``S = 0``) and would count numerically cold liquid as solid.
    """
    t = model.temperature(enthalpy, salinity)
    if model.kind == "fresh":
        frozen = ((model.t_melt - t) / model.mushy_dt).clamp(0.0, 1.0)
    else:
        frozen = (1.0 - model.liquid_fraction(t, salinity)).clamp(0.0, 1.0)
    return float(frozen.sum() * dx)


def run_stefan_sph(cfg: StefanConfig | None = None, model: EnthalpyModel | None = None,
                   salinity: float = 0.0):
    """Freeze a column from a cold wall and compare the front to Neumann's solution.

    The wall at ``x = 0`` is a Dirichlet condition on the first particle plus a
    mirror ghost layer (``T_ghost = 2 T_wall - T_1``), so the face flux is
    second-order in ``dx``; the far end uses a zero-flux mirror.
    """
    cfg = cfg or StefanConfig()
    model = model or EnthalpyModel(kind="fresh", mushy_dt=cfg.mushy_dt)
    n, dx = cfg.n_particles, cfg.dx
    h = cfg.h_factor * dx
    pos = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    rho = props.RHO_ICE
    sal = torch.full((n,), float(salinity), dtype=DTYPE)

    factor, scatter = (1.0, 0.0)
    if cfg.calibrate:
        factor, scatter = lattice_diffusion_factor(h_factor=cfg.h_factor)
    alpha_corr = 1.0 / factor if cfg.calibrate else 1.0

    t_wall = float(cfg.wall_temperature)
    h_wall = float(model.enthalpy(torch.tensor([t_wall], dtype=DTYPE),
                                  torch.tensor([salinity], dtype=DTYPE))[0])
    h_field = model.enthalpy(torch.full((n,), float(cfg.initial_temperature), dtype=DTYPE),
                             sal)
    dt = cfg.courant * dx ** 2 / (2.0 * props.DIFF_ICE)
    steps = int(math.ceil(cfg.t_end / dt))
    ste = props.C_ICE * (props.T_MELT - t_wall) / model.latent
    lam = neumann_lambda(ste)

    # geometry is frozen: build the extended lattice once
    pos_ext = torch.cat([-pos.flip(0), pos, 2.0 * (n * dx) - pos.flip(0)])
    fv = pair_geometry(pos_ext, torch.full((3 * n,), dx, dtype=DTYPE), h)
    mirror = torch.cat([torch.arange(n - 1, -1, -1), torch.arange(n),
                        torch.arange(n - 1, -1, -1)])
    mirror_sign = torch.cat([torch.full((n,), -1.0, dtype=DTYPE),
                             torch.full((n,), 1.0, dtype=DTYPE),
                             torch.full((n,), 1.0, dtype=DTYPE)])
    copper = torch.cat([torch.full((n,), 2.0 * t_wall, dtype=DTYPE),
                        torch.zeros(2 * n, dtype=DTYPE)])

    res = StefanResult(lam=lam, stefan_number=ste, dt=dt, steps=steps,
                       diffusion_factor=factor, diffusion_scatter=scatter, cfg=cfg)

    temperature = model.temperature(h_field, sal)
    for step in range(steps + 1):
        time = step * dt
        if step % cfg.profile_every == 0 or step == steps:
            res.time.append(time)
            res.front.append(_front_from_enthalpy(h_field, model, dx, sal))
            res.front_analytic.append(2.0 * lam * math.sqrt(props.DIFF_ICE * time))
        if step == steps:
            break
        w_l = model.liquid_fraction(temperature, sal)
        k_p = props.K_ICE + w_l * (props.K_FRESH - props.K_ICE)
        k_ext = k_p[mirror]
        t_ext = copper + mirror_sign * temperature[mirror]
        k_ij = 2.0 * k_ext[:, None] * k_ext[None, :] / (k_ext[:, None] + k_ext[None, :])
        kernel = k_ij * fv
        rhs = 2.0 / rho * (t_ext * kernel.sum(dim=1) - kernel @ t_ext)
        h_field = h_field + dt * rhs[n:2 * n] * alpha_corr
        if cfg.pin_wall:
            h_field[0] = h_wall  # extra strong Dirichlet on the wall particle
        temperature = model.temperature(h_field, sal)
        if not torch.isfinite(h_field).all():
            raise FloatingPointError(f"enthalpy diverged at step {step}")

    x_front = res.front[-1]
    if x_front > 0.0 and res.time[-1] > 0.0:
        denom = math.erf(lam)
        xx, tt, an = [], [], []
        for i in range(n):
            x = float(pos[i])
            if x > x_front:
                break
            t_exact = t_wall + (props.T_MELT - t_wall) * math.erf(
                x / (2.0 * math.sqrt(props.DIFF_ICE * res.time[-1]))) / denom
            xx.append(x)
            tt.append(float(temperature[i]))
            an.append(t_exact)
        res.profile_x, res.profile_t, res.profile_analytic = xx, tt, an
    return res
