"""Water-entry slamming: Wagner added-mass anchor, pressure-field noise control.

This module is the slamming load verification surface requested by campaign node
``p129w1r2-n08``.  It adds three things on top of :mod:`tensorsph.wedge_entry`
(which verifies the Wagner *kinematics* -- wetted width, spray root, peak
pressure of a single frame):

1. **Added-mass / force-coefficient anchor.**  The classical Wagner (1932)
   estimate of the vertical slamming load per unit length of a 2-D wedge,

       m_a(t) = (pi/2) rho c(t)^2 ,        c(t) = C V t ,   C = pi/(2 tan b)
       F(t)   = d/dt [ m_a(t) V ] = rho pi c dc/dt V
       C_s(t) = F / (0.5 rho V^2 c) = 2 pi C = pi^2 / tan(b)            (1)

   which is *time independent* at the self-similar stage.  ``b`` is the
   deadrise angle.  Note what is and is not ``sqrt(t)`` here: the wetted half
   width grows linearly in ``t`` and the added mass quadratically, so
   ``m_a ~ t^2`` for a straight-faced wedge; ``c ~ sqrt(t)`` -- and hence a
   ``sqrt``-type law for anything proportional to ``c`` -- is the signature of a
   *curved* (cylinder-like) section, ``c = sqrt(2 R V t)``.  Both statements are
   exposed here (``added_mass_law``, ``curved_section_width``) and pinned by
   tests, because the campaign sheet quotes "``m_a(t) ~ sqrt(t)``", which is not
   the 2-D straight-wedge law (see ``docs/verification_slamming.md`` 5.1).
   ``C_s`` in (1) is obtained from the added-mass route, *not* by integrating a
   pressure profile: the linearised Wagner pressure has a non-integrable
   ``1/sqrt(c-x)`` cusp at the spray root, so the integrated Wagner force
   diverges -- the SPH side has a finite smeared cusp and its force integral is
   well defined, which is exactly what makes the comparison informative.

2. **delta-SPH density diffusion and MLS pressure-gradient correction**, both as
   pure, independently testable functions plus an A/B switch on the solver:

   * ``delta_sph_density_rate`` -- Molteni & Colagrossi / Antuono et al. density
     diffusion ``D_i = 2 delta c0 h sum_j V_j (rho_i - rho_j) (x_ij.grad_i W_ij)
     / |x_ij|^2``.  Leading order in a smooth field is ``+ delta c0 h lap(rho)``
     (a *smoothing* term, see ``tests/test_slamming.py``), and it is exactly
     mass conserving in the discrete volume-weighted sense
     ``sum_i (m_i/rho_i) D_i = 0``.
   * ``mls_density`` -- zeroth-order moving-least-squares (Shepard)
     renormalisation of the summation density, ``rho_i / N_i`` with
     ``N_i = sum_j V_j W_ij``.  ``N_i`` is 1 in the bulk and < 1 wherever the
     kernel support is truncated, so this removes the free-surface density
     deficit that the Tait law turns into spurious pressure.  The *first-order*
     MLS pressure gradient (``mls_gradient`` / ``mls_pressure_acceleration``) is
     also implemented and is exact on a linear field, but it was measured to
     destabilise this summation-density solver at the free surface (max speed
     1.3e4 m/s within 50 steps at dx = 4 mm) and is therefore not wired in; it
     is kept as a tested library function and reported as a negative result.

3. **A slamming solver** (:class:`SlammingImpact`) that keeps every proven piece
   of :class:`~tensorsph.wedge_entry.WedgeImpact` (Tait EOS, density summation,
   Adami wall pressure, acoustic CFL, body-frame wedge) and adds the force
   coefficient, the wall pressure profile, mass bookkeeping and a pressure-noise
   metric.  With ``delta_sph = 0`` and ``mls_density = False`` it delegates to
   the parent RHS bit-for-bit (pinned by a test), so the A/B comparison isolates
   the two corrections.

Honest limitation (recorded in the report, not hidden): the *nonlinear*
Dobrovol'skaya (1969) self-similar solution is **not** implemented here.  Its
free surface wraps into an infinitely thin jet that runs along the body, which
is why Dobrovolskaya works in the hodograph plane; that is a research task of
its own and is declared WITHHELD rather than approximated by a formula invented
here.  The pressure-shape criterion of the node is therefore evaluated against
the linearised self-similar (Wagner) distribution, whose closed form is exact
and which the nonlinear solution is known to approach as ``b -> 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .droplet_impact import cubic_spline_kernel, pair_kernels
from .wedge_entry import (
    WedgeConfig,
    WedgeImpact,
    wet_half_width,
    wagner_c_coeff,
    wagner_p_centre,
    wagner_pmax,
)

__all__ = [
    "SlammingConfig",
    "added_mass_per_length",
    "added_mass_force",
    "slamming_force_coefficient",
    "added_mass_law",
    "curved_section_width",
    "wagner_shape",
    "pressure_shape_l2",
    "delta_sph_density_rate",
    "mls_gradient",
    "mls_pressure_acceleration",
    "SlammingImpact",
    "force_order",
]


# --------------------------------------------------------------------------- #
# 1. Wagner added-mass / force-coefficient anchor
# --------------------------------------------------------------------------- #

def added_mass_per_length(rho: float, V: float, beta: float, t: float) -> float:
    """Wagner added mass per unit length, ``(pi/2) rho c(t)^2`` [kg/m].

    ``beta`` is the deadrise angle in radians.  The factor ``pi/2`` is the 2-D
    flat-plate value evaluated at the Wagner wetted half width, which is the
    standard added-mass model behind every Wagner-type slamming load formula.
    """
    c = wet_half_width(V, beta, t)
    return 0.5 * math.pi * rho * c * c


def added_mass_force(rho: float, V: float, beta: float, t: float) -> float:
    """Vertical slamming force per unit length, ``d/dt[m_a(t) V]`` [N/m].

    Analytic derivative of ``(pi/2) rho c^2 V`` with ``c = C V t``:
    ``F = pi rho c (dc/dt) V = pi rho C^2 V^3 t``.  Written through ``c`` so the
    expression stays meaningful if a different wetting law is substituted.
    """
    C = wagner_c_coeff(beta)
    c = wet_half_width(V, beta, t)
    dc_dt = C * V
    return math.pi * rho * c * dc_dt * V


def slamming_force_coefficient(rho: float, V: float, beta: float, t: float) -> float:
    """``C_s(t) = F / (0.5 rho V^2 c)``; equals ``pi^2 / tan(beta)`` exactly."""
    c = wet_half_width(V, beta, t)
    if c <= 0.0:
        raise ValueError("wetted half width must be positive")
    return added_mass_force(rho, V, beta, t) / (0.5 * rho * V * V * c)


def added_mass_law(V: float = 1.0, beta: float = math.radians(20.0),
                   rho: float = 1000.0, t0: float = 1.0e-3,
                   ratio: float = 10.0) -> dict:
    """Measured exponents of ``c(t)`` and ``m_a(t)`` for the straight wedge.

    Returns ``{"c_exponent": 1.0, "added_mass_exponent": 2.0}`` up to round-off,
    by a two-point log-log difference at ``t0`` and ``ratio * t0``.  This is the
    machine-checkable statement that the 2-D straight wedge gives
    ``m_a ~ t^2``, not ``~ sqrt(t)``.
    """

    def expo(f):
        a, b = f(t0), f(ratio * t0)
        return math.log(b / a) / math.log(ratio)

    e_c = expo(lambda t: wet_half_width(V, beta, t))
    e_m = expo(lambda t: added_mass_per_length(rho, V, beta, t))
    return {"c_exponent": e_c, "added_mass_exponent": e_m}


def curved_section_width(V: float, radius: float, t: float) -> float:
    """Wagner wetted half width of a *circular* section: ``sqrt(2 R V t)``.

    Included only to make the ``sqrt(t)`` statement precise: it is the wetting
    law of a body with bottom curvature, not of a straight-faced wedge.
    """
    return math.sqrt(2.0 * radius * V * t)


def wagner_shape(x, c: float, beta: float):
    """Linearised self-similar pressure distribution normalised by its peak.

    ``s = 1/sqrt(1-(x/c)^2)``, ``p ~ 2 C s - s^2 + 1`` (the form implemented and
    audited in :func:`tensorsph.wedge_entry.wagner_pressure`), returned as
    ``p / p_max`` on ``|x| <= c``.  This is the *reference shape* the SPH wall
    pressure is compared with; the nonlinear Dobrovolskaya profile is not
    implemented (module docstring).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    C = wagner_c_coeff(beta)
    r = (x / c).clamp(-1.0 + 1.0e-12, 1.0 - 1.0e-12)
    s = 1.0 / torch.sqrt(1.0 - r * r)
    p = 2.0 * C * s - s * s + 1.0
    return p / p.max()


def pressure_shape_l2(x, p, c: float, beta: float) -> float:
    """Relative L2 distance between an SPH pressure profile and the Wagner shape.

    Both profiles are normalised by their own maximum on ``|x| <= 0.9 c`` (the
    interior, where the linearised solution is smooth); the error is
    ``||p_hat - r_hat|| / ||r_hat||``.  Inputs are 1-D tensors/sequences on the
    same abscissae.
    """
    xt = torch.as_tensor(x, dtype=torch.float64)
    pt = torch.as_tensor(p, dtype=torch.float64)
    keep = xt.abs() <= 0.9 * c
    if int(keep.sum()) < 3:
        return float("nan")
    xs, ps = xt[keep], pt[keep]
    order = torch.argsort(xs)
    xs, ps = xs[order], ps[order]
    ref = wagner_shape(xs, c, beta)
    pmax = float(ps.max())
    if pmax <= 0.0:
        return float("nan")
    ph = ps / pmax
    return float(torch.linalg.vector_norm(ph - ref) / torch.linalg.vector_norm(ref))


def force_order(levels, values) -> tuple[float, float]:
    """Observed convergence order of a peak value from a 2- or 3-level ladder.

    ``levels`` are the particle spacings (fine -> coarse or any order; they are
    sorted), ``values`` the measured peak.  The order is the least-squares slope
    of ``log|value - value_finest|`` against ``log(dx)`` -- the standard
    self-consistency estimate for a monotone ladder, and the fallback used when
    the Richardson triplet is not monotone.  Returns ``(order, r2)``.
    """
    pairs = sorted(zip(levels, values))
    if len(pairs) < 3:
        return float("nan"), float("nan")
    ref = pairs[0][1]
    xs, ys = [], []
    for dx, val in pairs[1:]:
        e = abs(val - ref)
        if e > 0.0 and dx > 0.0:
            xs.append(math.log(dx))
            ys.append(math.log(e))
    if len(xs) < 2:
        return float("nan"), float("nan")
    n = float(len(xs))
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1.0e-18:
        return float("nan"), float("nan")
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    ss_tot = sum((y - sy / n) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, r2


# --------------------------------------------------------------------------- #
# 2a. delta-SPH density diffusion (Molteni & Colagrossi 2013 / Antuono 2010)
# --------------------------------------------------------------------------- #

def delta_sph_density_rate(rho_all, i, j, delta, gw, volume_j, *, n_fluid: int,
                           c0: float, h: float, delta_coef: float):
    """delta-SPH density diffusion rate ``D_i`` on a pair list [kg/(m^3 s)].

    ``rho_all``  density of every particle (fluid ``[:n_fluid]`` then walls),
    ``i``, ``j`` pair indices into the combined array with ``i < n_fluid``,
    ``delta``    ``x_i - x_j``, ``gw`` the kernel gradient ``grad_i W_ij``,
    ``volume_j`` ``m_j / rho_j`` of the ``j`` particle (fluid or wall).

    Returns a length-``n_fluid`` tensor

        D_i = 2 delta c0 h sum_j V_j (rho_i - rho_j) (x_ij.grad_i W_ij)/|x_ij|^2

    Properties pinned by tests: (i) in a smooth field the leading order is
    ``+ delta c0 h lap(rho)`` (a *smoothing* operator); (ii) it is exactly
    conservative over fluid-fluid pairs, ``sum_i V_i D_i = 0``, because it is
    evaluated in the volume-symmetric form ``D_i += V_j w_ij``,
    ``D_j -= V_i w_ij`` (wall pairs only contribute to the fluid side, which is
    part of the boundary model, not of the fluid mass balance).
    """
    D = torch.zeros(int(n_fluid), dtype=rho_all.dtype, device=rho_all.device)
    if i.numel() == 0 or delta_coef <= 0.0:
        return D
    nf = int(n_fluid)
    ff = j < nf
    if not bool(ff.any()):
        return D
    ii = i[ff]
    dd = delta[ff]
    gg = gw[ff]
    jj = j[ff]
    r2 = dd.square().sum(dim=-1).clamp_min(1.0e-30)
    drift = 2.0 * delta_coef * c0 * h * (rho_all[ii] - rho_all[jj]) * (dd * gg).sum(dim=-1) / r2
    # Written in the *volume-symmetric* form D_i = sum_j V_j w_ij, D_j = -sum_i V_i w_ij,
    # so that sum_i V_i D_i vanishes identically for fluid-fluid pairs.  The plain
    # (rho_i - rho_j) form only has that property when the pair list is symmetric
    # (each pair listed twice), which is not what the cell-linked search returns.
    drift = drift.to(D.dtype)
    D = D.index_add(0, ii, volume_j[jj].to(D.dtype) * drift)
    ffj = jj
    fluid_j = ffj < nf
    if bool(fluid_j.any()):
        D = D.index_add(0, ffj[fluid_j],
                        -volume_j[ii[fluid_j]].to(D.dtype) * drift[fluid_j])
    return D


# --------------------------------------------------------------------------- #
# 2b. MLS (Shepard-renormalised) pressure gradient
# --------------------------------------------------------------------------- #

def shepard_density_correction(rho, i, j, w, volume_all, *, n_fluid: int,
                               floor: float = 0.5):
    """Zeroth-order MLS (Shepard) renormalisation of the summation density.

    The summation density ``rho_i = sum_j m_j W_ij`` is unbiased only where the
    kernel support is full: a particle at the free surface loses half its
    neighbours and comes out ~20 % light, so the pressure it feeds into the Tait
    law is a truncation artefact rather than a flow quantity.  The Shepard factor
    ``N_i = sum_j V_j W_ij`` (= 1 in the bulk, < 1 at a surface) measures that
    deficit and ``rho_i / N_i`` is the zeroth-order MLS estimate of the density.

    ``floor`` guards ``N_i`` for isolated particles (a particle with a handful of
    neighbours cannot be renormalised).

    Only the *density* is renormalised here.  The first-order variant applied to
    the pressure gradient (:func:`mls_gradient`) is exact on a linear field but
    was measured to destabilise this summation-density solver at the free
    surface (see the module notes and the report); it is deliberately not wired
    into :class:`SlammingImpact`.
    """
    nf = int(n_fluid)
    out = rho.clone()
    if i.numel() == 0:
        return out
    N = torch.zeros(rho.shape[0], dtype=rho.dtype, device=rho.device)
    N = N.index_add(0, i, volume_all[j] * w)
    N = N.clamp_min(floor)
    corrected = rho / N
    return torch.where(torch.arange(rho.shape[0], device=rho.device) < nf,
                       corrected, rho)


def mls_gradient(p, i, j, delta, gw, volume_j, *, n_fluid: int, ridge: float = 1.0e-8):
    """First-order consistent pressure gradient on a pair list.

    ``grad p_i = M_i^{-1} sum_j V_j (p_j - p_i) grad_i W_ij`` with the
    renormalisation matrix ``M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij``.

    ``p`` is the pressure of every particle (fluid ``[:n_fluid]`` then walls, so
    a wall pressure can be fed in the same way the Adami wall model does).
    Singular ``M_i`` (isolated particles, truncated support) is regularised with
    a ridge proportional to ``ridge * trace(M_i)`` so the operator degrades to a
    scaled identity instead of producing ``inf``; the returned ``cond`` flag
    marks those particles so a test can check the fallback is rare.

    Exactness on a linear field is the point: for ``p_j - p_i = (x_j - x_i).g``,
    ``grad p_i = M_i^{-1} M_i g = g`` to machine precision for any lattice.
    """
    dim = delta.shape[1]
    dtype = p.dtype if isinstance(p, torch.Tensor) else torch.float64
    device = delta.device
    nf = int(n_fluid)
    zero = torch.zeros(nf, dtype=dtype, device=device)
    if i.numel() == 0:
        return torch.zeros((nf, dim), dtype=dtype, device=device), torch.ones(
            nf, dtype=torch.bool, device=device)
    ff = j < nf
    ii, dd, gg = i[ff], delta[ff], gw[ff]
    vol = volume_j[j[ff]]
    dp = p[j[ff]] - p[ii]
    # rhs = sum_j V_j (p_j - p_i) grad_i W_ij
    rhs_x = zero.index_add(0, ii, vol * dp * gg[:, 0])
    rhs_y = zero.index_add(0, ii, vol * dp * gg[:, 1])
    # M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij = -sum_j V_j x_ij (x) grad_i W_ij
    mxx = zero.index_add(0, ii, vol * (-dd[:, 0]) * gg[:, 0])
    mxy = zero.index_add(0, ii, vol * (-dd[:, 0]) * gg[:, 1])
    myx = zero.index_add(0, ii, vol * (-dd[:, 1]) * gg[:, 0])
    myy = zero.index_add(0, ii, vol * (-dd[:, 1]) * gg[:, 1])
    # The continuum M is symmetric; the discrete one differs by the O(h^2)
    # lattice asymmetry, and symmetrising keeps the inverse well posed.
    mxy = 0.5 * (mxy + myx)

    # 2-D closed form with a trace-proportional ridge.
    tr = mxx + myy
    eps = float(ridge) * tr.abs().clamp_min(1.0e-30)
    a = mxx + eps
    d = myy + eps
    b = mxy
    det = a * d - b * b
    bad = det <= 1.0e-12 * tr.abs().square().clamp_min(1.0e-30)
    det = torch.where(bad, torch.ones_like(det), det)
    inv_xx = d / det
    inv_yy = a / det
    inv_xy = -b / det
    gx = inv_xx * rhs_x + inv_xy * rhs_y
    gy = inv_xy * rhs_x + inv_yy * rhs_y
    grad = torch.stack((gx, gy), dim=-1)
    grad = torch.where(bad.unsqueeze(-1), torch.zeros_like(grad), grad)
    return grad, bad


def _mls_matrix(i, j, delta, gw, volume_all, n_all, n_fluid, ridge):
    """Renormalisation matrices M_i = sum_j V_j (x_j-x_i) (x) grad_i W_ij."""
    dtype = delta.dtype
    device = delta.device
    zero = torch.zeros(n_all, dtype=dtype, device=device)
    ii = i
    jj = j
    vol = volume_all[jj]
    mxx = zero.index_add(0, ii, vol * (-delta[:, 0]) * gw[:, 0])
    mxx = mxx.index_add(0, jj, vol * (delta[:, 0]) * gw[:, 0])
    myy = zero.index_add(0, ii, vol * (-delta[:, 1]) * gw[:, 1])
    myy = myy.index_add(0, jj, vol * (delta[:, 1]) * gw[:, 1])
    mxy = zero.index_add(0, ii, vol * (-delta[:, 0]) * gw[:, 1])
    mxy = mxy.index_add(0, jj, vol * (delta[:, 0]) * gw[:, 1])
    myx = zero.index_add(0, ii, vol * (-delta[:, 1]) * gw[:, 0])
    myx = myx.index_add(0, jj, vol * (delta[:, 1]) * gw[:, 0])
    mxy = 0.5 * (mxy + myx)
    tr = mxx + myy
    eps = float(ridge) * tr.abs().clamp_min(1.0e-30)
    a = mxx + eps
    d = myy + eps
    b = mxy
    det = a * d - b * b
    bad = det <= 1.0e-8 * tr.abs().square().clamp_min(1.0e-30)
    det = torch.where(bad, torch.ones_like(det), det)
    inv = torch.stack((torch.stack((d / det, -b / det), dim=-1),
                       torch.stack((-b / det, a / det), dim=-1)), dim=-2)
    return inv, bad


def mls_pressure_acceleration(p_all, rho_all, i, j, delta, gw, mass, *,
                              n_fluid: int, ridge: float = 0.05):
    """Momentum-conserving, first-order consistent pressure-gradient force.

    ``a_i = -sum_j m_j (p_j - p_i) / (rho_i rho_j) * Mbar_ij^{-1} grad_i W_ij``
    with ``Mbar_ij = (M_i + M_j)/2``.  Averaging the renormalisation matrix makes
    the pair force equal and opposite, so the scheme conserves momentum exactly
    -- which the one-sided MLS gradient in :func:`mls_gradient` does not, and
    which is why this (not that) is what the solver uses.

    ``ridge`` regularises ``M`` where its support is truncated (free surface,
    wall layer); 5 % of the trace is enough to keep the free-surface rows from
    amplifying the gradient by an order of magnitude.
    """
    n_all = int(p_all.shape[0])
    if i.numel() == 0:
        return torch.zeros((n_fluid, delta.shape[1]), dtype=delta.dtype,
                           device=delta.device), 0
    inv, bad = _mls_matrix(i, j, delta, gw, mass / rho_all, n_all, n_fluid, ridge)
    g = torch.einsum("pab,pb->pa", inv[i] + inv[j], gw) * 0.5
    weight = (mass * (p_all[j] - p_all[i]) / (rho_all[i] * rho_all[j]))
    contrib = weight.unsqueeze(-1) * g
    acc = torch.zeros((n_fluid, delta.shape[1]), dtype=delta.dtype,
                      device=delta.device)
    acc = acc.index_add(0, i, -contrib)
    fluid_j = j < n_fluid
    if bool(fluid_j.any()):
        acc = acc.index_add(0, j[fluid_j], -contrib[fluid_j])
    return acc, int(bad[i].sum().item())


# --------------------------------------------------------------------------- #
# 3. Solver
# --------------------------------------------------------------------------- #

@dataclass
class SlammingConfig(WedgeConfig):
    """Wedge-entry case plus the two pressure-field switches of this node.

    ``delta_sph``      Molteni & Colagrossi diffusion coefficient (0 = off;
                       0.1 is the value used in the delta-SPH literature).
    ``mls_density``    Shepard (zeroth-order MLS) density renormalisation.
    ``force_threshold`` fraction of the dynamic pressure that defines the wetted
                       footprint for the force integral (same convention as the
                       parent's ``c(t)``, so ``F`` and ``c`` are consistent).
    """

    delta_sph: float = 0.0
    mls_density: bool = False
    force_threshold: float = 0.02

    def validate(self) -> None:
        super().validate()
        if self.delta_sph < 0.0:
            raise ValueError("delta_sph must be non-negative")
        if not 0.0 < self.force_threshold < 1.0:
            raise ValueError("force_threshold must be in (0, 1)")


class SlammingImpact(WedgeImpact):
    """Wedge entry with a slamming force history and optional noise controls."""

    def __init__(self, cfg: SlammingConfig):
        super().__init__(cfg)
        self.slam = cfg
        self.delta_coef = float(cfg.delta_sph)
        self.use_mls = bool(cfg.mls_density)
        self._dt_last = 0.0
        self.mass_initial = float(self.fluid_mass.sum())

    # ---- right-hand side -------------------------------------------------- #
    def _evaluate(self, dt: float | None = None):
        """Parent RHS when both switches are off; otherwise the extended one."""
        if self.delta_coef <= 0.0 and not self.use_mls:
            return super()._evaluate()
        dt = self._dt_last if dt is None else float(dt)
        cfg = self.cfg
        h = self.h
        n = self.n_fluid
        pos_all = self.all_positions()
        vel_all = torch.cat((self.fluid_vel, self.wall_vel_body), dim=0)
        self_kernel = float(cubic_spline_kernel(
            torch.zeros(1, dtype=self.dtype, device=self.device), h)[0])
        m = self.mass_unit
        pairs = self._pairs()
        empty = torch.zeros((0, 2), dtype=torch.long, device=self.device)
        if pairs.numel() == 0:
            pairs = empty
        i, j = pairs.unbind(dim=1)
        delta = pos_all[i] - pos_all[j]
        w, gw = pair_kernels(delta, h)

        # ---- summation density (+ delta-SPH diffusion) -------------------- #
        rho = torch.full((n,), m * self_kernel, dtype=self.dtype, device=self.device)
        ff = j < n
        rho = rho.index_add(0, i[ff], m * w[ff])
        rho = rho.index_add(0, j[ff], m * w[ff])
        fw = ~ff
        rho = rho.index_add(0, i[fw], m * w[fw])
        rho = rho.clamp_min(0.1 * self.rho0)
        if self.use_mls:
            vol_all = torch.cat(
                (m / rho, torch.full((self.wall_local.shape[0],), m / self.rho0,
                                     dtype=self.dtype, device=self.device)), dim=0)
            rho = shepard_density_correction(
                torch.cat((rho, torch.full((self.wall_local.shape[0],), self.rho0,
                                           dtype=self.dtype, device=self.device))),
                i, j, w, vol_all, n_fluid=n)[:n]
        if self.delta_coef > 0.0 and dt > 0.0:
            # Wall density is rho0 to O((p/B)^(1/gamma)) = 0.1 % at these impact
            # pressures, so the diffusion sees rho0 on the wall side rather than
            # closing a loop with the Tait law it is meant to precondition.
            rho_all0 = torch.cat(
                (rho, torch.full((self.wall_local.shape[0],), self.rho0,
                                 dtype=self.dtype, device=self.device)), dim=0)
            vol = m / rho_all0
            D = delta_sph_density_rate(rho_all0, i, j, delta, gw, vol,
                                       n_fluid=n, c0=self.c0, h=h,
                                       delta_coef=self.delta_coef)
            rho = (rho + dt * D).clamp_min(0.1 * self.rho0)

        pressure = (self.B * ((rho / self.rho0) ** cfg.gamma - 1.0)).clamp_min(0.0)

        # ---- Adami wall pressure ------------------------------------------ #
        n_wall = self.wall_local.shape[0]
        num = torch.zeros(n_wall, dtype=self.dtype, device=self.device)
        den = torch.zeros(n_wall, dtype=self.dtype, device=self.device)
        wj = j[fw] - n
        num = num.index_add(0, wj, pressure[i[fw]] * w[fw])
        den = den.index_add(0, wj, w[fw])
        p_wall = torch.where(den > 1.0e-30, num / den.clamp_min(1.0e-30),
                             torch.zeros_like(num))
        rho_wall = self.rho0 * (p_wall / self.B + 1.0) ** (1.0 / cfg.gamma)

        rho_all = torch.cat((rho, rho_wall), dim=0)
        p_all = torch.cat((pressure, p_wall), dim=0)

        # ---- momentum ------------------------------------------------------ #
        fluid_j = j.clamp_max(n - 1)
        wall_j = (j - n).clamp_min(0)
        rho_pair = torch.where(j < n, rho[fluid_j], rho_wall[wall_j])
        p_pair = torch.where(j < n, pressure[fluid_j], p_wall[wall_j])

        dv = vel_all[i] - vel_all[j]
        mu = h * (dv * delta).sum(dim=-1) / (delta.square().sum(dim=-1) + 0.01 * h * h)
        mu = torch.where(mu < 0.0, mu, torch.zeros_like(mu))
        rho_bar = 0.5 * (rho[i] + rho_pair)
        pi_ij = -cfg.alpha_visc * self.c0 * mu / rho_bar

        acc = torch.zeros((n, 2), dtype=self.dtype, device=self.device)
        bracket = pressure[i] / rho[i].square() + p_pair / rho_pair.square()
        contrib = (m * bracket).unsqueeze(-1) * gw
        acc = acc.index_add(0, i[ff], -contrib[ff])
        acc = acc.index_add(0, j[ff], contrib[ff])
        acc = acc.index_add(0, i[fw], -contrib[fw])
        # Monaghan viscosity is kept as its own antisymmetric term in both cases.
        visc = (m * pi_ij).unsqueeze(-1) * gw
        acc = acc.index_add(0, i[ff], -visc[ff])
        acc = acc.index_add(0, j[ff], visc[ff])
        acc = acc.index_add(0, i[fw], -visc[fw])

        return {
            "rho": rho,
            "pressure": pressure,
            "acc": acc,
            "p_wall": p_wall,
            "den": den,
            "pairs": pairs,
        }

    def advance(self, dt: float) -> None:      # noqa: D401 - mirrors the parent
        self._dt_last = float(dt)
        state = self._evaluate(dt)
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
            raise FloatingPointError("slamming run produced non-finite positions")

    # ---- slamming diagnostics --------------------------------------------- #
    def wall_profile(self, state=None, side: int = +1) -> dict:
        """Wall pressure on one face of the wedge, sorted by ``|x|``.

        Returns ``x`` (abscissa), ``p`` (extrapolated wall pressure) restricted
        to the wetted surface particles of one face (``x > 0`` for ``side=+1``),
        the wetted half width ``c`` and the force integral
        ``F = 2 dx sum_k p_k`` [N/m]: for the face at angle ``beta`` the pressure
        acts normally, ``dF_y = p cos(beta) ds`` with ``ds = dx/cos(beta)``, so
        the vertical component of a one-face strip is ``p dx``.
        """
        if state is None:
            state = self._evaluate(0.0)
        cfg = self.cfg
        p_wall = state["p_wall"][: self.n_wedge]
        den = state["den"][: self.n_wedge]
        surf = self.is_first
        x = (self.wedge_local[:, 0] + self.body_offset[0])[surf]
        p = p_wall[surf]
        contact = den[surf] > cfg.contact_support * self.support_ref
        wet = contact & (p > cfg.force_threshold * self.q)
        face = wet & (x > 0 if side > 0 else x < 0)
        if bool(face.any()):
            c = float(x.abs()[face].max())
        else:
            c = 0.0
        if cfg.cell_offset and c > 0.0:
            c += 0.5 * self.dx * math.cos(self.beta)
        xs = x[face].abs()
        ps = p[face]
        order = torch.argsort(xs)
        xs, ps = xs[order], ps[order]
        force = 2.0 * self.dx * float(ps.sum())
        return {"x": xs, "p": ps, "c": c, "force": force, "n_wet": int(face.sum())}

    def slamming_sample(self, state=None) -> dict:
        """One time sample of the slamming quantities used by the criteria."""
        prof = self.wall_profile(state)
        c = prof["c"]
        q = self.q
        F = prof["force"]
        cs = F / (0.5 * self.rho0 * self.V * self.V * c) if c > 0 else 0.0
        pmax = float(prof["p"].max()) if prof["p"].numel() else 0.0
        noise = pressure_noise(prof["x"], prof["p"], pmax, c) if c > 0 else float("nan")
        shape = (pressure_shape_l2(prof["x"], prof["p"], c, self.beta)
                 if c > 0 and prof["x"].numel() >= 4 else float("nan"))
        mass = float(self.fluid_mass.sum())
        return {
            "t": self.time,
            "c": c,
            "force": F,
            "c_s": cs,
            "c_s_wagner": math.pi ** 2 / math.tan(self.beta),
            "p_max": pmax,
            "p_max_wagner": float(wagner_pmax(self.V, self.beta, self.rho0)),
            "p_centre": float(wagner_p_centre(self.V, self.beta, self.rho0)),
            "noise": noise,
            "shape_l2": shape,
            "n_wet": prof["n_wet"],
            "mass": mass,
            "mass_drift": abs(mass - self.mass_initial) / self.mass_initial,
        }

    def run(self, frames: int | None = None, t_end: float | None = None):
        cfg = self.cfg
        t_end = cfg.resolved_t_end() if t_end is None else float(t_end)
        frames = cfg.frames if frames is None else int(frames)
        times = torch.linspace(0.0, t_end, frames, dtype=self.dtype).tolist()
        steps = 0
        max_steps = 10_000_000
        while self.time < times[0] - 1.0e-15 and steps < max_steps:
            self.advance(min(self.dt(), times[0] - self.time))
            steps += 1
        out = [self.slamming_sample()]
        target = 1
        while target < len(times) and steps < max_steps:
            step = times[target] - self.time
            if step <= 0:
                target += 1
                continue
            self.advance(min(self.dt(), step))
            steps += 1
            if self.time >= times[target] - 1.0e-15:
                out.append(self.slamming_sample())
                target += 1
        self.steps = steps
        return out


def pressure_noise(x, p, p_max: float, c: float, fraction: float = 0.8) -> float:
    """Spatial pressure roughness on the wetted face, detrended.

    ``sqrt(mean((p - <p>_3)^2)) / p_max`` over the interior ``|x| <= fraction*c``,
    where ``<p>_3`` is the 3-point moving average along the surface.  The moving
    average is what separates the physical variation (the Wagner profile has a
    square-root cusp at the spray root, whose second difference would otherwise
    dominate the metric) from the grid-scale content of the field, which is the
    numerical noise this node's delta-SPH criterion is about.  Profiles must be
    sorted by ``|x|`` (``SlammingImpact.wall_profile`` returns them that way).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    keep = (x <= fraction * c) & (x > 0.0)
    if int(keep.sum()) < 4 or p_max <= 0.0:
        return float("nan")
    ps = p[keep]
    kern = torch.ones(3, dtype=ps.dtype) / 3.0
    smooth = torch.nn.functional.conv1d(
        ps.view(1, 1, -1), kern.view(1, 1, -1)).view(-1)
    resid = ps[1:-1] - smooth
    return float(torch.sqrt((resid * resid).mean()) / p_max)