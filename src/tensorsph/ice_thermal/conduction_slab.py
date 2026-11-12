"""Semi-infinite slab contact: the classical anchor for the conduction operator.

Why this module exists
----------------------
:mod:`.sph_thermal` discretises the heat equation and is exercised by the Stefan
problem (a *moving boundary*).  A Stefan case mixes three error sources --
the conduction operator, the enthalpy closure and the front bookkeeping -- so a
pass on it does not by itself say that ``dH/dt`` is right.  The anchor here
isolates the operator: **two semi-infinite slabs of the same material, initially
at ``T_hot`` (x < 0) and ``T_cold`` (x > 0), brought into perfect contact at
``t = 0``**.  The solution is a closed form (Carslaw & Jaeger 1959, *Conduction
of Heat in Solids*, 2nd ed., §2.5; the same textbook solution is used as the
reference by Cleary & Monaghan 1999, *J. Comput. Phys.* 148, 227)::

    T(x, t) = T_mean + (dT / 2) * erf(-x / (2 sqrt(alpha t)))
    T_mean  = (T_hot + T_cold) / 2 ,      dT = T_hot - T_cold

with ``alpha = k / (rho c)`` the *true* diffusivity of the material.  There is
no latent heat, no interface motion and no fitted constant in the reference, so
any disagreement is the operator's (or the time integrator's) fault.

Non-dimensionalisation
----------------------
Time is quoted as the Fourier number of the reference length::

    t* = alpha t / L_ref^2            (``cfg.fourier``)

and lengths in units of the **diffusion length** ``sigma = sqrt(alpha t)``:

* the numerical domain is ``+-cfg.domain_sigmas * sigma`` wide, so the ends sit
  in the untouched far field and are pinned (Dirichlet) at their initial value;
* the error is measured over ``|x| <= cfg.window_sigmas * sigma``.  Inside that
  window the analytic profile still spans ``erf(2.5) = 0.9996`` of the step, so
  the gate is not measuring an empty far field.

The operator is applied exactly as the library defines it (:func:`pair_geometry`
is the single source of truth for ``V_j F_ij``); on a uniform lattice that
product is Toeplitz, so the band coefficients are read once from the centre row
and the time loop is a banded update.  ``tests/test_conduction_slab.py`` pins
the equivalence of the banded update and
:func:`tensorsph.ice_thermal.sph_thermal.conduction_rhs` to machine precision,
so the fast path cannot silently drift from the library operator.

Calibration
-----------
``sph_thermal.lattice_diffusion_factor`` measures ``alpha_eff / alpha`` of the
operator on a periodic lattice (1.0218 at ``h = 1.3 dx``).  Because the anchor's
reference is the *true* ``alpha``, the same case is run twice: raw
(``calibrate=False``) and calibrated (``calibrate=True``, the operator is
multiplied by ``1 / factor`` -- exactly what :func:`run_stefan_sph` does).  Both
runs are reported, so the value of the calibration is measured rather than
assumed; it is not a free parameter of the comparison.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from . import properties as props
from .sph_thermal import lattice_diffusion_factor, pair_geometry

__all__ = [
    "SlabConfig",
    "SlabResult",
    "analytic_temperature",
    "band_coefficients",
    "band_apply",
    "run_slab_contact",
    "fit_effective_diffusivity",
    "diffusion_length",
]

DTYPE = torch.float64


# ------------------------------------------------------------------- config
@dataclass
class SlabConfig:
    """Resolution and time of the contact anchor (all inputs, no tuned number)."""

    n_particles: int = 240           # total particles over the symmetric domain
    fourier: float = 0.5             # t* = alpha t / L_ref^2 (0.5 = mid-range)
    reference_length: float = 0.1    # L_ref in m (defines t* only)
    domain_sigmas: float = 6.0       # half-domain length / diffusion length
    window_sigmas: float = 2.5       # comparison half-window / diffusion length
    h_factor: float = 1.3            # kernel support h / dx (library default)
    courant: float = 0.4             # dt / (dx^2 / (2 alpha_eff)); <=1 is stable
    calibrate: bool = True           # divide the operator by the lattice factor
    t_hot: float = 263.15            # K, slab on x < 0 (-10 C, stays ice)
    t_cold: float = 268.15           # K, slab on x > 0 (-5 C, stays ice)
    alpha: float = props.DIFF_ICE    # m^2/s, true diffusivity of ice
    negative_control: bool = False   # flip the operator sign (anti-diffusion)


def diffusion_length(alpha: float, time: float) -> float:
    """``sigma = sqrt(alpha t)`` (m) -- the length unit of the anchor."""
    return math.sqrt(alpha * float(time))


@dataclass
class SlabResult:
    """One run of the contact anchor."""

    cfg: SlabConfig
    time: float = 0.0
    dx: float = 0.0
    half_length: float = 0.0
    sigma: float = 0.0
    dt: float = 0.0
    steps: int = 0
    diffusion_factor: float = 1.0
    diffusion_scatter: float = 0.0
    x: list = field(default_factory=list)
    temperature: list = field(default_factory=list)
    analytic: list = field(default_factory=list)
    error: list = field(default_factory=list)
    in_window: list = field(default_factory=list)
    wall_time_s: float = 0.0
    # metrics
    rms_rel: float = float("nan")        # RMS error over dT (dimensionless)
    max_abs_k: float = float("nan")      # max |error| in K
    max_rel: float = float("nan")        # max |error| / dT
    fitted_alpha_ratio: float = float("nan")   # alpha_fit / alpha
    interface_k: float = float("nan")    # T at the particle nearest x = 0
    interface_analytic_k: float = float("nan")
    energy_drift: float = float("nan")   # |sum T(t) - sum T(0)| / (N dT)
    finite: bool = True

    @property
    def calibrated(self) -> bool:
        return self.cfg.calibrate


# ---------------------------------------------------------------- reference
def analytic_temperature(x, time, alpha, t_hot=None, t_cold=None):
    """Carslaw & Jaeger §2.5 two-slab contact profile (K) at ``time`` (s).

    ``T = Tm + (dT/2) erf(-x / (2 sqrt(alpha t)))``; for ``t -> 0`` the limit is
    the initial step, which is what the caller wants for ``time = 0``.
    """
    t_hot = 263.15 if t_hot is None else float(t_hot)
    t_cold = 268.15 if t_cold is None else float(t_cold)
    x = torch.as_tensor(x, dtype=DTYPE)
    t_mean = 0.5 * (t_hot + t_cold)
    half = 0.5 * (t_hot - t_cold)
    if time <= 0.0:
        return torch.where(x < 0, torch.full_like(x, t_hot),
                           torch.full_like(x, t_cold))
    width = 2.0 * diffusion_length(alpha, time)
    return t_mean + half * torch.erf(-x / width)


# ------------------------------------------------------------- discretisation
def band_coefficients(n_particles: int, dx: float, h_factor: float = 1.3,
                      eta_factor: float = 0.01):
    """Toeplitz band coefficients ``2 V_j F_ij`` and the half-bandwidth ``Q``.

    ``V_j F_ij`` is taken from :func:`sph_thermal.pair_geometry` (the library
    definition of the geometry factor), evaluated on the centre row of a uniform
    lattice -- valid for every interior particle because the lattice is uniform
    and :func:`pair_geometry` depends on ``x_i - x_j`` only.  Coefficients are
    **negative** (``dW/dr < 0``), i.e. the update
    ``T_i += dt * alpha * sum_q m_q (T_i - T_{i+q})`` is diffusive.
    """
    x = (torch.arange(n_particles, dtype=DTYPE) + 0.5) * float(dx)
    fv = pair_geometry(x, torch.full((n_particles,), float(dx), dtype=DTYPE),
                       float(h_factor) * float(dx), eta_factor)
    centre = n_particles // 2
    q_max = int(math.ceil(2.0 * float(h_factor)))  # support radius 2h
    coeffs = [0.0] * (q_max + 1)
    for q in range(1, q_max + 1):
        coeffs[q] = float(2.0 * fv[centre, centre + q])
    return coeffs, q_max


def band_apply(temperature, coeffs, q_max):
    """``sum_q m_q (T_i - T_{i+q})`` -- the discrete Laplacian action (1/m^2).

    Edges are handled by clamping the ghost value to the end particle; the end
    particles themselves are overwritten by the Dirichlet condition after every
    step, so only the interior of the band is ever used.
    """
    n = temperature.shape[0]
    flat = torch.cat([temperature[:1].expand(q_max), temperature,
                      temperature[-1:].expand(q_max)])
    acc = torch.zeros(n, dtype=DTYPE)
    for q in range(1, q_max + 1):
        m_q = coeffs[q]
        if m_q == 0.0:
            continue
        acc = acc + m_q * ((temperature - flat[q_max - q:q_max - q + n])
                           + (temperature - flat[q_max + q:q_max + q + n]))
    return acc


def _step(temperature, coeffs, q_max, dt, alpha_eff, left_value, right_value):
    """One explicit-Euler step of the banded conduction update (in place)."""
    temperature += dt * alpha_eff * band_apply(temperature, coeffs, q_max)
    # Dirichlet far field: the analytic solution is exponentially close to the
    # initial values there, so pinning the end particles is the correct BC.
    temperature[0] = left_value
    temperature[-1] = right_value
    return temperature


# ------------------------------------------------------------------- fitting
def fit_effective_diffusivity(x, temperature, time, t_hot, t_cold,
                              window=None, alpha_guess=None, tol=1e-9,
                              max_iter=200):
    """Diffusivity implied by the measured profile, ``alpha_fit`` (m^2/s).

    The profile shape is ``Tm + (dT/2) erf(-x / (2 sqrt(alpha t)))``; scale
    invariance means the only free parameter is ``sigma = sqrt(alpha t)``, so
    ``sigma`` is recovered by a golden-section search on the least-squares
    residual.  This turns the anchor into a *measurement of the operator's
    amplitude*: raw the symbol is ``-alpha_eff k^2`` with
    ``alpha_eff / alpha = lattice_diffusion_factor``, calibrated it is ``alpha``.
    """
    x = torch.as_tensor(x, dtype=DTYPE)
    t_num = torch.as_tensor(temperature, dtype=DTYPE)
    if window is None:
        window = torch.ones_like(x, dtype=torch.bool)
    window = torch.as_tensor(window, dtype=torch.bool)
    xw, tw = x[window], t_num[window]
    t_mean = 0.5 * (float(t_hot) + float(t_cold))
    half = 0.5 * (float(t_hot) - float(t_cold))
    alpha_guess = float(props.DIFF_ICE) if alpha_guess is None else float(alpha_guess)
    guess = math.sqrt(alpha_guess * float(time))

    def residual(sigma):
        model = t_mean + half * torch.erf(-xw / (2.0 * sigma))
        return float(((model - tw) ** 2).sum())

    lo, hi = 0.2 * guess, 5.0 * guess
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
    fc, fd = residual(c), residual(d)
    for _ in range(max_iter):
        if hi - lo <= tol * guess:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = residual(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = residual(d)
    sigma_star = 0.5 * (lo + hi)
    return sigma_star ** 2 / float(time)


# ---------------------------------------------------------------------- run
def run_slab_contact(cfg: SlabConfig | None = None) -> SlabResult:
    """Run the two-slab contact anchor and return the profile plus its metrics."""
    cfg = cfg or SlabConfig()
    t0 = time.time()
    alpha = float(cfg.alpha)
    t_end = cfg.fourier * cfg.reference_length ** 2 / alpha
    sigma = diffusion_length(alpha, t_end)
    half_length = cfg.domain_sigmas * sigma
    n = int(cfg.n_particles)
    dx = 2.0 * half_length / n
    h = cfg.h_factor * dx

    factor, scatter = lattice_diffusion_factor(h_factor=cfg.h_factor)
    alpha_eff = alpha / factor if cfg.calibrate else alpha
    if cfg.negative_control:  # harness self-check: this run MUST diverge
        alpha_eff = -alpha_eff

    x = (torch.arange(n, dtype=DTYPE) + 0.5) * dx - half_length
    coeffs, q_max = band_coefficients(n, dx, cfg.h_factor)

    t_hot, t_cold = float(cfg.t_hot), float(cfg.t_cold)
    temperature = torch.where(x < 0.0, torch.full_like(x, t_hot),
                              torch.full_like(x, t_cold))
    initial_sum = float(temperature.sum())

    dt = cfg.courant * dx ** 2 / (2.0 * alpha_eff)
    steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / steps

    res = SlabResult(cfg=cfg, time=t_end, dx=dx, half_length=half_length,
                     sigma=sigma, dt=dt, steps=steps, diffusion_factor=factor,
                     diffusion_scatter=scatter)
    for _ in range(steps):
        temperature = _step(temperature, coeffs, q_max, dt, alpha_eff, t_hot,
                            t_cold)

    finite = bool(torch.isfinite(temperature).all())
    analytic = analytic_temperature(x, t_end, alpha, t_hot, t_cold)
    error = temperature - analytic
    window = x.abs() <= cfg.window_sigmas * sigma
    err_w = error[window]
    d_temp = abs(t_hot - t_cold)

    res.finite = finite
    res.x = [float(v) for v in x]
    res.temperature = [float(v) for v in temperature]
    res.analytic = [float(v) for v in analytic]
    res.error = [float(v) for v in error]
    res.in_window = [bool(v) for v in window]
    if finite:
        res.rms_rel = float(torch.sqrt((err_w ** 2).mean()) / d_temp)
        res.max_abs_k = float(err_w.abs().max())
        res.max_rel = res.max_abs_k / d_temp
        nearest = int(torch.argmin(x.abs()))
        res.interface_k = float(temperature[nearest])
        res.interface_analytic_k = float(analytic[nearest])
        res.fitted_alpha_ratio = fit_effective_diffusivity(
            x, temperature, t_end, t_hot, t_cold, window,
            alpha_guess=alpha) / alpha
        res.energy_drift = abs(float(temperature.sum()) - initial_sum) / (n * d_temp)
    res.wall_time_s = round(time.time() - t0, 2)
    return res