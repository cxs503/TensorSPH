"""1-D transient conduction anchor: two semi-infinite plates brought into contact.

CASE-SPH-CONDUCTION-1D.  Two semi-infinite solids at uniform temperatures
``T_hot`` (``x < 0``) and ``T_cold`` (``x > 0``) are put in perfect thermal
contact at ``t = 0``.  The temperature field is self-similar,

.. math::

    T(x,t) = T_{cold} + \\frac{T_{hot} - T_{cold}}{2} \\operatorname{erfc}(\\eta),
             \\eta = \\frac{x}{2\\sqrt{\\alpha t}} ,

with ``alpha = k / (rho cp)`` the thermal diffusivity and
``eta = x / (2 sqrt(alpha t))`` the similarity variable.  This is the classical
error-function solution for two semi-infinite media initially at different
uniform temperatures, put in contact (Carslaw & Jaeger, *Conduction of Heat in
Solids*, 2nd ed. 1959, Ch. II -- linear flow in the infinite/semi-infinite
solid, the two-media-in-contact section, §2.6).  ``docs/case-conduction-1d.md``
records exactly what is and is not verified about the citation itself.

Numerical side
--------------
The discretisation is the *existing* Cleary & Monaghan (1999, J. Comput. Phys.
148:227, doi:10.1006/jcph.1998.6118) SPH conduction operator of
:mod:`tensorsph.ice_thermal.sph_thermal`; nothing new is discretised here::

    dT_i/dt = (2 / rho_i) sum_j V_j (2 k_i k_j / (k_i + k_j)) (T_i - T_j) F_ij

``F_ij = (r_ij . grad_i W_ij) / (r_ij^2 + eta^2)``, so ``V_j F_ij`` is
dimensionless, the operator is conservative, and the pair sum is antisymmetric.

On the uniform lattice this operator collapses *exactly* (not approximately) to
a symmetric 5-point stencil, because the M4 kernel at ``h = 1.3 dx`` has support
``2h = 2.6 dx`` and the lattice spacing is ``dx``: only the offsets
``m = 1, 2`` have non-zero ``W'(r)``, and by translation invariance the pair
coefficient depends on ``m`` alone,

    dT_i/dt = (2 / rho) sum_{m=1,2} a_m (2 T_i - T_{i-m} - T_{i+m}) ,
    a_m     = V * r dW/dr (r) / (r^2 + eta^2) |_{r = m dx}   (< 0 for M4).

:func:`run_contact_sph(..., dense=False)` uses that stencil (``O(N)`` per step);
``dense=True`` calls :func:`conduction_rhs` on the full ``O(N^2)`` pair matrix.
The two agree to round-off -- :func:`stencil_vs_dense` measures it, and the
benchmark asserts it -- so the fast path is the same operator, not a
re-implementation of the physics.

Far field
---------
Both ends use the repo's mirror-ghost Dirichlet construction
(:func:`tensorsph.ice_thermal.sph_thermal.run_stefan_sph`): a fictitious layer
mirrors the first/last particle about the wall plane with
``T_ghost = 2 T_wall - T_mirror``, giving the ``O(dx^2)`` face value directly.
The walls sit 4.2 similarity lengths out, where the analytic deficit is ~1e-7 of
the jump, so the wall treatment is not what the error metric measures (checked
by the domain-doubling test in the benchmark).

Lattice calibration
-------------------
``alpha_eff / alpha = lattice_diffusion_factor(...)`` is measured on a periodic
lattice (the M4 kernel at ``h = 1.3 dx`` under-resolves the second moment) and
the operator is divided by it, exactly as ``run_stefan_sph`` does.  The
benchmark reports both raw and calibrated error so the size of that correction
stays on the record instead of being hidden inside the tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .ice_thermal.sph_thermal import (conduction_rhs, kernel_dwdr,
                                      lattice_diffusion_factor, pair_geometry)

__all__ = [
    "similarity_variable",
    "lattice_symbol_ratio",
    "contact_temperature",
    "lattice_stencil",
    "stencil_rhs",
    "stencil_vs_dense",
    "ContactConfig",
    "ContactResult",
    "run_contact_sph",
    "fd_contact_reference",
]

DTYPE = torch.float64


# ------------------------------------------------------------------ analytic
def similarity_variable(x, t, alpha):
    """``eta = x / (2 sqrt(alpha t))``; raises when ``t <= 0``."""
    t = float(t)
    if t <= 0.0:
        raise ValueError("the similarity variable needs t > 0")
    return torch.as_tensor(x, dtype=DTYPE) / (2.0 * math.sqrt(float(alpha) * t))


def contact_temperature(x, t, alpha, t_hot=1.0, t_cold=0.0):
    """Analytic ``T(x,t)`` of two semi-infinite plates in contact (erfc form).

    ``T = T_cold + (T_hot - T_cold)/2 erfc(eta)``, ``eta = x / (2 sqrt(alpha t))``.
    Limits: ``x -> -inf`` gives ``T_hot``, ``x -> +inf`` gives ``T_cold``,
    ``x = 0`` gives the arithmetic mean -- the three properties any correct
    implementation must reproduce (asserted in ``tests/``).  The same field
    written with ``erf`` is ``(T_hot+T_cold)/2 - (T_hot-T_cold)/2 erf(eta)``.
    """
    eta = similarity_variable(x, t, alpha)
    return t_cold + 0.5 * (t_hot - t_cold) * torch.erfc(eta)


# ------------------------------------------------------- lattice stencil (exact)
def lattice_stencil(dx, h, eta_factor=0.01, volume=None):
    """``[(m, a_m), ...]`` of the SPH Conduction operator on a uniform lattice.

    ``a_m = V r dW/dr (m dx) / ((m dx)^2 + eta^2)`` -- the *same* expression
    :func:`pair_geometry` uses, evaluated at the integer lattice offsets inside
    the kernel support ``2h``.  Coefficients are negative (M4) and symmetric in
    ``m``, so the stencil is a pure diffusion stencil.
    """
    vol = float(dx if volume is None else volume)
    eta2 = (eta_factor * float(h)) ** 2
    out = []
    mmax = int(math.floor(2.0 * h / float(dx) + 1e-12))
    for m in range(1, mmax + 1):
        r = m * float(dx)
        a = vol * float(r * kernel_dwdr(r, h)) / (r * r + eta2)
        if a != 0.0:
            out.append((m, a))
    return out


def stencil_rhs(temperature, stencil, rho=1.0, ends="periodic"):
    """``dT/dt`` by the lattice stencil; ``ends`` = ``"periodic"`` or ``"mirror"``.

    With ``"periodic"`` the field wraps around, so slicing ``[m : m + n]`` of the
    result of a mirror-extended state gives the interior rows unchanged (the
    stencil never reaches the wrap for those rows) -- that is how the contact
    solver applies its ghost layer.
    """
    t = torch.as_tensor(temperature, dtype=DTYPE)
    n = t.numel()
    acc = torch.zeros_like(t)
    for m, a in stencil:
        if ends == "periodic":
            lo, hi = torch.roll(t, m), torch.roll(t, -m)
        elif ends == "mirror":
            idx = torch.arange(n)
            i_lo = torch.where(idx - m < 0, m - idx, idx - m)
            i_hi = torch.where(idx + m > n - 1, 2 * (n - 1) - idx - m, idx + m)
            lo, hi = t[i_lo], t[i_hi]
        else:
            raise ValueError(f"unknown ends={ends!r}")
        acc = acc + a * (2.0 * t - lo - hi)
    return 2.0 / float(rho) * acc


def lattice_symbol_ratio(k, dx, h, eta_factor=0.01, volume=None):
    """``alpha_eff(k) / alpha`` of the lattice stencil -- its discrete dispersion.

    Applying the stencil to ``cos(k x)`` gives ``rhs = -[2 sum_m a_m
    (1 - cos(m k dx))] cos(k x)``, so the ratio of that factor to the exact
    ``alpha k^2`` is the wavenumber-dependent effective diffusivity.  ``k -> 0``
    is what :func:`lattice_diffusion_factor` measures (and the solver calibrates
    away); the deviation at larger ``k`` is the part a single factor cannot
    remove, and it is what puts a floor under the profile error.
    """
    stencil = lattice_stencil(dx, h, eta_factor, volume=volume)
    kk = float(k)
    acc = 0.0
    for m, a in stencil:
        theta = m * kk * float(dx)
        acc += a * 4.0 * math.sin(0.5 * theta) ** 2      # = 2 - 2 cos(theta), stable
    return -2.0 * acc / (kk * kk)


def stencil_vs_dense(dx=0.02, h_factor=1.3, eta_factor=0.01, n=40, seed=0):
    """Max ``|stencil - dense conduction_rhs|`` on a periodic lattice.

    The dense operator is applied on a 3x-repeated (hence periodic) copy of a
    random temperature field, exactly as :func:`lattice_diffusion_factor` does;
    only the middle copy's rows are compared, so neither path sees an end.
    Round-off sized agreement is what licenses the ``O(N)`` time loop.
    """
    h = h_factor * dx
    g = torch.Generator().manual_seed(seed)
    temp = torch.rand(n, dtype=DTYPE, generator=g) + 0.5
    pos = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    pos_ext = torch.cat([pos - n * dx, pos, pos + n * dx])
    t_ext = torch.cat([temp, temp, temp])
    vol_ext = torch.full((3 * n,), dx, dtype=DTYPE)
    k_ext = torch.ones(3 * n, dtype=DTYPE)
    dense = conduction_rhs(t_ext, k_ext, vol_ext, 1.0, pos_ext, h, eta_factor)[n:2 * n]
    fast = stencil_rhs(t_ext, lattice_stencil(dx, h, eta_factor),
                       ends="periodic")[n:2 * n]
    return float((dense - fast).abs().max())


# --------------------------------------------------------------------- setup
@dataclass
class ContactConfig:
    """Two half-spaces in contact; ``rho = cp = 1`` so the conductivity is alpha.

    Lengths in m, time in s.  The case is run in dimensionless form
    (``l = 1 m``, ``alpha = 1 m^2/s``) so that ``t* = Fourier = alpha t / l^2``.
    """

    n_particles: int = 600
    dx: float = 0.02            # m
    h_factor: float = 1.3
    alpha: float = 1.0          # m^2/s
    t_hot: float = 1.0          # K, x < 0
    t_cold: float = 0.0         # K, x > 0
    fourier: float = 0.5        # t* = alpha t / l^2, l = 1 m
    courante: float = 0.2       # dt = courante * dx^2 / alpha
    calibrate: bool = True
    calibration: str = "lattice"   # "lattice" (repo 3-mode mean) | "k0" | "none"
    eta_factor: float = 0.01
    dense: bool = False         # True: O(N^2) conduction_rhs every step (slow)
    sample_every: int = 0       # 0: final profile only


@dataclass
class ContactResult:
    """Profiles + metrics of one SPH run."""

    time: float = 0.0
    dt: float = 0.0
    steps: int = 0
    fourier: float = 0.0
    n_particles: int = 0
    dx: float = 0.0
    x: list = field(default_factory=list)
    temperature: list = field(default_factory=list)
    analytic: list = field(default_factory=list)
    profile_time: list = field(default_factory=list)
    profiles: list = field(default_factory=list)       # [(t, T(list)), ...]
    diffusion_factor: float = 1.0
    diffusion_scatter: float = 0.0
    calibration: str = "lattice"
    cfg: ContactConfig = field(default_factory=ContactConfig)

    # ---------------------------------------------------------------- metrics
    def error(self, x_cut: float | None = None):
        """``(max, rms)`` of ``|T_sph - T_exact|`` normalised by the imposed jump."""
        x = torch.tensor(self.x, dtype=DTYPE)
        a = torch.tensor(self.temperature, dtype=DTYPE)
        b = torch.tensor(self.analytic, dtype=DTYPE)
        d = abs(self.cfg.t_hot - self.cfg.t_cold) or 1.0
        if x_cut is not None:
            keep = x.abs() <= float(x_cut)
            a, b = a[keep], b[keep]
        e = (a - b).abs() / d
        return float(e.max()), float(torch.sqrt((e ** 2).mean()))

    def interface_temperature(self):
        """``T`` at ``x = 0`` by linear interpolation between the two neighbours."""
        x = torch.tensor(self.x, dtype=DTYPE)
        a = torch.tensor(self.temperature, dtype=DTYPE)
        right = int(torch.nonzero(x >= 0.0)[0])
        left = right - 1
        w = (0.0 - float(x[left])) / float(x[right] - x[left])
        return float((1.0 - w) * a[left] + w * a[right])


def _mirror_extend(values, t_left, t_right, m):
    """Mirror-ghost extension ``[ghost_left(m), values, ghost_right(m)]``.

    ``T_ghost = 2 T_wall - T_mirror`` (see the module docstring): the wall planes
    are ``pos[0] - dx/2`` and ``pos[-1] + dx/2``, so the mirrored partner of the
    first particle is the second particle, etc.
    """
    left = 2.0 * t_left - values[:m].flip(0)      # partners: 1, 2, ... (m deep)
    right = 2.0 * t_right - values[-m:].flip(0)
    return torch.cat([left, values, right])


# ----------------------------------------------------------------- the solver
def run_contact_sph(cfg: ContactConfig | None = None) -> ContactResult:
    """Run the 1-D SPH contact case and return profiles + metrics."""
    cfg = cfg or ContactConfig()
    n, dx = int(cfg.n_particles), float(cfg.dx)
    h = cfg.h_factor * dx
    alpha = float(cfg.alpha)
    length = n * dx
    pos = (torch.arange(n, dtype=DTYPE) + 0.5) * dx - 0.5 * length    # centred on x = 0
    vol = torch.full((n,), dx, dtype=DTYPE)

    t_end = float(cfg.fourier) * 1.0 ** 2 / alpha
    dt = cfg.courante * dx ** 2 / alpha
    steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / steps                                 # land exactly on t*

    factor, scatter = 1.0, 0.0
    mode = cfg.calibration if cfg.calibrate else "none"
    if mode == "lattice":
        # repository practice: mean of the three longest periodic modes
        factor, scatter = lattice_diffusion_factor(h_factor=cfg.h_factor)
    elif mode == "k0":
        # long-wavelength limit of the same symbol.  The three-mode mean sits
        # ~4e-4 below it (the symbol falls off like (k dx)^2), and that gap is
        # exactly the residual diffusivity this case exposes -- see
        # docs/case-conduction-1d.md.
        factor = lattice_symbol_ratio(1.0e-3 / dx, dx, h, cfg.eta_factor)
        scatter = 0.0
    elif mode != "none":
        raise ValueError(f"unknown calibration={cfg.calibration!r}")
    alpha_corr = 1.0 / factor if mode != "none" else 1.0

    stencil = lattice_stencil(dx, h, cfg.eta_factor)
    m = max(mm for mm, _ in stencil)

    temperature = torch.where(pos < 0.0,
                              torch.full((n,), float(cfg.t_hot), dtype=DTYPE),
                              torch.full((n,), float(cfg.t_cold), dtype=DTYPE))

    # dense path: frozen pair matrix on the mirror-extended lattice
    fv_dense = None
    if cfg.dense:
        x_l = float(pos[0]) - 0.5 * dx
        x_r = float(pos[-1]) + 0.5 * dx
        pos_ext = torch.cat([2.0 * x_l - pos.flip(0), pos, 2.0 * x_r - pos.flip(0)])
        vol_ext = torch.full((3 * n,), dx, dtype=DTYPE)
        fv_dense = pair_geometry(pos_ext, vol_ext, h, cfg.eta_factor)

    res = ContactResult(dt=dt, steps=steps, fourier=cfg.fourier, n_particles=n,
                        dx=dx, diffusion_factor=factor, calibration=mode,
                        diffusion_scatter=scatter, cfg=cfg)
    profile_time, profiles = [], []

    for step in range(steps + 1):
        if cfg.sample_every and step % cfg.sample_every == 0:
            profile_time.append(step * dt)
            profiles.append([float(v) for v in temperature])
        if step == steps:
            break
        if cfg.dense:
            # ghost layer mirrors the whole column (as run_stefan_sph does), so
            # the 3n x 3n pair matrix built once above stays valid
            t_ext = torch.cat([2.0 * float(cfg.t_hot) - temperature.flip(0),
                               temperature,
                               2.0 * float(cfg.t_cold) - temperature.flip(0)])
            k_ext = torch.full((3 * n,), alpha, dtype=DTYPE)
            rhs = conduction_rhs(t_ext, k_ext, torch.full((3 * n,), dx, dtype=DTYPE),
                                 1.0, None, h, cfg.eta_factor, fv=fv_dense)
            rate = rhs[n:2 * n]
        else:
            t_ext = _mirror_extend(temperature, float(cfg.t_hot),
                                   float(cfg.t_cold), m)
            rate = stencil_rhs(t_ext, stencil, rho=1.0)[m:m + n]
        temperature = temperature + dt * rate * alpha_corr
        if not torch.isfinite(temperature).all():
            raise FloatingPointError(f"conduction diverged at step {step}")

    res.time = steps * dt
    res.x = [float(v) for v in pos]
    res.temperature = [float(v) for v in temperature]
    res.analytic = [float(v) for v in
                    contact_temperature(pos, res.time, alpha, cfg.t_hot, cfg.t_cold)]
    res.profile_time = profile_time
    res.profiles = profiles
    return res


# ------------------------------------------------------- independent reference
def fd_contact_reference(fourier=0.5, alpha=1.0, half_length=6.0, n=1201,
                         r_target=0.4, t_hot=1.0, t_cold=0.0):
    """Crank-Nicolson finite-difference solution of the same PDE (independent check).

    Returns ``(x, T)`` on ``x in [-half_length, half_length]`` at ``t = fourier``.
    Ends are Dirichlet at the exact analytic values (they sit 4.2 similarity
    lengths out, where the analytic deficit is ~1e-7 of the jump); the
    tridiagonal system is solved with ``scipy.linalg.solve_banded`` at
    ``r = alpha dt / (2 dx^2) = 0.4``.  This shares no code, kernel or
    discretisation with the SPH path, so it independently checks both the
    analytic profile and the SPH field.
    """
    import numpy as np
    from scipy.linalg import solve_banded

    dx = 2.0 * float(half_length) / (n - 1)
    x = torch.linspace(-half_length, half_length, n, dtype=DTYPE)
    t_end = float(fourier)
    dt = float(r_target) * 2.0 * dx ** 2 / alpha
    steps = int(math.ceil(t_end / dt))
    dt = t_end / steps
    r = alpha * dt / (2.0 * dx ** 2)

    t = np.where(x.numpy() < 0.0, t_hot, t_cold).astype(np.float64)
    inner = n - 2
    ab = np.zeros((3, inner))
    ab[0, 1:] = -r
    ab[1, :] = 1.0 + 2.0 * r
    ab[2, :-1] = -r

    for _ in range(steps):
        b = t[1:-1] + r * (t[:-2] - 2.0 * t[1:-1] + t[2:])
        b[0] += r * t_hot
        b[-1] += r * t_cold
        t[1:-1] = solve_banded((1, 1), ab, b, overwrite_ab=False,
                               overwrite_b=True, check_finite=False)
    return x, torch.as_tensor(t.copy(), dtype=DTYPE)