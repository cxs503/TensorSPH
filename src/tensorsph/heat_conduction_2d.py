"""Two-dimensional conduction operator: does the 1-D operator (and its lattice
calibration) transfer to two dimensions?

Why this module exists
----------------------
:mod:`tensorsph.heat_conduction` verifies the repository's SPH conduction
operator (Cleary & Monaghan 1999, *J. Comput. Phys.* **148**:227-264,
doi:10.1006/jcph.1998.6118) against the 1-D ``erfc`` similarity solution and
calibrates it with :func:`tensorsph.ice_thermal.sph_thermal.lattice_diffusion_factor`.
That single benchmark family is the whole accuracy evidence for the
``heat_conduction_operator`` entry of the method-selection registry
(``docs/reports/method_selection_registry.json``): the upper bound on the
method rests on *one* geometry, *one* reference family and *one* dimension.

This module supplies a second, independent family -- the decay of a Fourier mode
of the heat equation in a periodic *square* -- and, in doing so, answers a
question the 1-D case cannot ask: **is the pair formula dimensionally
consistent?**  It is not, naively.  The shipped formula

.. math::

    \frac{dT_i}{dt} = \frac{2}{\rho c_p}\sum_j V_j\,
                     \frac{2k_i k_j}{k_i + k_j}\,F_{ij}\,(T_i - T_j),
    \qquad F_{ij} = \frac{\boldsymbol r_{ij}\cdot\nabla_i W_{ij}}{r_{ij}^2+\eta^2}

carries the factor 2 and the volume ``V_j = dx`` of a *one-dimensional* layer
sum (:math:`\sum_j V_j r\,dW/dr \to \int W\,dx = 1` in 1-D).  Lifted to 2-D with
``V_j = dx^2`` the layer sum no longer reproduces that moment: the measured
effective diffusivity of the raw operator is

.. math::

    \frac{\alpha_{\rm eff}}{\alpha}\Big|_{\rm 2D} \approx 1.92\,dx
    \quad (O(dx), \text{ i.e. } \to 0 \text{ as } dx \to 0),

so a 2-D run that inherits the 1-D calibration factor (1.0218) diffuses ~``1/dx``
too slowly (measured below: a 60 % error at ``n = 32``).  The corrected
procedure -- measure the *2-D* symbol and divide the operator by it, exactly the
step the 1-D solver already performs in its own dimension -- restores a
consistent scheme: the mode-decay error falls 1.4e-3 -> 3.5e-4 -> 8.6e-5 for
``n = 32, 64, 128`` (second order), i.e. 0.0086 % at the finest rung.

So the honest content of this family is twofold, and both halves are banked:

1. **Accuracy record** -- with the 2-D calibration the operator reproduces the
   analytic mode decay to 0.0086 % (this is the second benchmark family of the
   registry record).
2. **Dimensional caveat** -- the 1-D calibration factor is *not* transferable:
   ``2/rho * sum V_j ...`` in 2-D is ``O(dx)`` too small without a
   dimension-specific normalisation.  Anchoring the record on the 2-D-calibrated
   arm without printing the uncalibrated and 1-D-calibrated arms would hide a
   trap, so all three arms are written to the artefact.

The reference is the exact separated solution of the heat equation on a periodic
square,

.. math::

    T(x,y,t) = e^{-\alpha(k_x^2+k_y^2)t}\cos(k_x x)\cos(k_y y),

obtained by substituting into :math:`\partial_t T = \alpha\nabla^2 T`; it is the
periodic (Fourier) solution of the linear heat equation (Carslaw & Jaeger 1959,
*Conduction of Heat in Solids*, 2nd ed., Ch. II, periodic solutions -- the same
chapter as the 1-D ``erfc`` target used by :mod:`tensorsph.heat_conduction`; the
book's section numbering is not machine-verified here, the formula is verified by
substitution and by an independent Crank-Nicolson solve).

What is *not* claimed
---------------------
* No claim about variable conductivity in 2-D: every check here uses
  ``k_i == k_j``, so the harmonic pair mean is exercised only trivially.
* No claim about walls/boundaries in 2-D: the domain is periodic, so no ghost
  layer is involved (the 1-D family already owns the wall treatment).
* No claim that this replaces the 1-D ``erfc`` case: the two families fail
  differently (similarity profile vs discrete symbol), which is the point of
  having both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .ice_thermal.sph_thermal import kernel_dwdr

__all__ = [
    "DTYPE",
    "LatticeOffsets",
    "lattice_offsets",
    "stencil_rhs_2d",
    "pair_geometry_2d",
    "conduction_rhs_2d",
    "symbol_rate",
    "lattice_diffusion_factor_2d",
    "mode_field",
    "mode_exact",
    "mode_amplitude",
    "stencil_vs_dense_2d",
    "degenerate_1d_agreement",
    "fd_mode_decay",
    "ModeDecayConfig",
    "ModeDecayResult",
    "run_mode_decay",
]

DTYPE = torch.float64

#: Lattice offsets ``[((m, n), a), ...]`` inside the kernel support, with
#: ``a = V r dW/dr / (r^2 + eta^2)`` and ``V = dx^2``.
LatticeOffsets = tuple[tuple[int, int], float]


# --------------------------------------------------------------------- lattice
def lattice_offsets(dx: float, h: float, eta_factor: float = 0.01,
                    thickness: float = 1.0) -> tuple[LatticeOffsets, ...]:
    """``[((m, n), a), ...]`` -- the 2-D pair operator on a uniform square lattice.

    ``r_{ij} = |(m, n)| dx`` is the *vector* distance, so the diagonal offsets
    the 1-D stencil never sees are included.  ``thickness`` is the out-of-plane
    thickness used for the particle volume (``V = dx^2 * thickness``); it is 1
    everywhere in this benchmark and exposed only so the dimensional statement in
    the module docstring can be checked.
    """
    dx = float(dx)
    h = float(h)
    eta2 = (float(eta_factor) * h) ** 2
    r_max = 2.0 * h
    m_max = int(math.floor(r_max / dx + 1e-12))
    vol = dx * dx * float(thickness)
    h_t = torch.tensor(h, dtype=DTYPE)
    out: list[tuple[tuple[int, int], float]] = []
    for m in range(-m_max, m_max + 1):
        for n in range(-m_max, m_max + 1):
            if m == 0 and n == 0:
                continue
            r = math.hypot(m, n) * dx
            if r > r_max + 1e-12:
                continue
            dwdr = float(kernel_dwdr(torch.tensor(r, dtype=DTYPE), h_t))
            a = vol * r * dwdr / (r * r + eta2)
            if a != 0.0:
                out.append(((m, n), a))
    return tuple(out)


def stencil_rhs_2d(temperature: torch.Tensor,
                   offsets: tuple[LatticeOffsets, ...],
                   rho: float = 1.0) -> torch.Tensor:
    """``dT/dt`` of the pair operator on a *periodic* 2-D grid (fast path).

    ``dT_i/dt = (2/rho) sum_offsets a (T_i - T_{i+offset})`` with the offsets
    wrapped, i.e. the same expression :func:`conduction_rhs_2d` evaluates pair by
    pair; :func:`stencil_vs_dense_2d` shows the two agree to round-off.
    """
    t = torch.as_tensor(temperature, dtype=DTYPE)
    acc = torch.zeros_like(t)
    for (m, n), a in offsets:
        acc = acc + a * (t - torch.roll(t, shifts=(-m, -n), dims=(0, 1)))
    return (2.0 / float(rho)) * acc


def pair_geometry_2d(pos: torch.Tensor, volume: torch.Tensor, h: float,
                     eta_factor: float = 0.01) -> torch.Tensor:
    """``V_j F_ij`` (m^2) for 2-D positions ``(N, 2)`` -- the dense path.

    Identical expression to :func:`tensorsph.ice_thermal.sph_thermal.pair_geometry`
    except that ``r = |r_vec|`` uses both components; ``eta`` regularises the
    ``r -> 0`` pair.
    """
    x = torch.as_tensor(pos, dtype=DTYPE)
    v = torch.as_tensor(volume, dtype=DTYPE)
    dr = x[:, None, :] - x[None, :, :]
    r = torch.linalg.norm(dr, dim=-1)
    eta2 = (float(eta_factor) * float(h)) ** 2
    dwdr = kernel_dwdr(r, float(h))
    f_ij = r * dwdr / (r * r + eta2)
    return f_ij * v[None, :]


def conduction_rhs_2d(temperature, conductivity, volume, rho, pos, h,
                      eta_factor: float = 0.01, fv=None) -> torch.Tensor:
    """``dT/dt`` (K/s) from the dense 2-D pair sum; shapes ``(N,)``.

    ``fv`` optionally carries the frozen pair matrix from
    :func:`pair_geometry_2d`.  The update is ``2/rho * sum_j k_ij V_j F_ij
    (T_i - T_j)`` with the harmonic pair mean ``k_ij = 2 k_i k_j / (k_i + k_j)``
    -- the same algebra as the 1-D ``conduction_rhs``.
    """
    t = torch.as_tensor(temperature, dtype=DTYPE)
    k = torch.as_tensor(conductivity, dtype=DTYPE)
    v = torch.as_tensor(volume, dtype=DTYPE)
    rho_t = torch.as_tensor(rho, dtype=DTYPE)
    k, v, t = torch.broadcast_tensors(k, v, t)
    if fv is None:
        fv = pair_geometry_2d(pos, v, h, eta_factor)
    k_ij = 2.0 * k[:, None] * k[None, :] / (k[:, None] + k[None, :])
    kernel = k_ij * fv
    return 2.0 / rho_t * (t * kernel.sum(dim=1) - kernel @ t)


# ---------------------------------------------------------------- mode algebra
def symbol_rate(dx: float, h: float, kx: float, ky: float,
                eta_factor: float = 0.01) -> float:
    """Rate ``lambda`` such that the raw operator gives ``dT/dt = -lambda T`` for
    ``T = cos(kx x) cos(ky y)`` on the lattice [1/s].

    Substituting the lattice mode into the pair sum: every offset contributes
    ``-a (1 - cos(m kx dx) cos(n ky dx))`` and the product mode is invariant
    under the shift (the four-dimensional span of ``cos cos``/``cos sin``/
    ``sin cos``/``sin sin`` is closed under translations), so a single scalar
    rate exists.  ``lambda / (kx^2 + ky^2) = alpha_eff / alpha``.
    """
    dx = float(dx)
    total = 0.0
    for (m, n), a in lattice_offsets(dx, h, eta_factor):
        total += a * (1.0 - math.cos(m * kx * dx) * math.cos(n * ky * dx))
    return -2.0 * total


def lattice_diffusion_factor_2d(h_factor: float = 1.3, dx: float = 1.0,
                                eta_factor: float = 0.01,
                                mode: str = "k0") -> float:
    """``alpha_eff / alpha`` of the raw 2-D operator on the unit lattice.

    ``mode="k0"`` takes the long-wavelength limit (``k dx = 1e-3``), the quantity
    a solver should divide by (the 1-D analogue is
    ``lattice_symbol_ratio(k -> 0)``); ``mode="mean"`` averages the symbol over
    the three longest periodic modes of an ``n = 256`` lattice, mirroring
    ``lattice_diffusion_factor``'s 3-mode mean.  Both are returned by the
    benchmark so the choice stays visible.
    """
    h = float(h_factor) * float(dx)
    if mode == "k0":
        k = 1.0e-3 / float(dx)
        return symbol_rate(dx, h, k, k, eta_factor) / (2.0 * k * k)
    if mode == "mean":
        n = 256
        kmodes = [2.0 * math.pi * m / (n * float(dx)) for m in (1, 2, 3)]
        ratios = [symbol_rate(dx, h, k, 0.0, eta_factor) / (k * k) for k in kmodes]
        return sum(ratios) / len(ratios)
    raise ValueError(f"unknown mode={mode!r}")


def mode_field(n: int, dx: float, kx: float, ky: float) -> torch.Tensor:
    """``cos(kx x) cos(ky y)`` at the cell centres of an ``n x n`` periodic grid."""
    x = (torch.arange(n, dtype=DTYPE) + 0.5) * float(dx)
    xg, yg = torch.meshgrid(x, x, indexing="ij")
    return torch.cos(float(kx) * xg) * torch.cos(float(ky) * yg)


def mode_exact(n: int, dx: float, kx: float, ky: float, t: float,
               alpha: float = 1.0) -> torch.Tensor:
    """Exact solution at time ``t``: the same mode times ``exp(-alpha (kx^2+ky^2) t)``."""
    decay = math.exp(-float(alpha) * (float(kx) ** 2 + float(ky) ** 2) * float(t))
    return decay * mode_field(n, dx, kx, ky)


def mode_amplitude(temperature: torch.Tensor, n: int, dx: float,
                   kx: float, ky: float) -> float:
    """Rayleigh quotient ``(T, mode) / (mode, mode)`` -- the amplitude of the mode."""
    m = mode_field(n, dx, kx, ky)
    return float((torch.as_tensor(temperature, dtype=DTYPE) * m).sum()
                 / (m * m).sum())


# ------------------------------------------------------------------ equivalence
def stencil_vs_dense_2d(n: int = 16, h_factor: float = 1.3,
                        eta_factor: float = 0.01, seed: int = 0) -> float:
    """Max ``|fast stencil - dense pair sum|`` on a periodic ``n x n`` lattice.

    The dense matrix is built from the same :func:`pair_geometry_2d` the 1-D
    solver's dense path uses, so a round-off-sized number licenses the
    ``torch.roll`` time loop as the *same* operator rather than a re-derivation.
    The dense lattice is the field tiled 3x3 (so it has no edge for the stencil
    to wrap over); only the middle tile is compared -- same construction as
    :func:`tensorsph.heat_conduction.stencil_vs_dense`.
    """
    dx = 1.0
    h = float(h_factor) * dx
    g = torch.Generator().manual_seed(seed)
    temp = torch.rand(n, n, dtype=DTYPE, generator=g) + 0.5
    big = temp.repeat(3, 3)
    m = 3 * n
    lin = (torch.arange(m, dtype=DTYPE) + 0.5) * dx
    yg, xg = torch.meshgrid(lin, lin, indexing="ij")
    pos = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=1)
    vol = torch.full((m * m,), dx * dx, dtype=DTYPE)
    dense = conduction_rhs_2d(big.reshape(-1), torch.ones(m * m, dtype=DTYPE),
                              vol, 1.0, pos, h, eta_factor).reshape(m, m)
    fast = stencil_rhs_2d(big, lattice_offsets(dx, h, eta_factor))
    middle = slice(n, 2 * n)
    return float((dense[middle, middle] - fast[middle, middle]).abs().max())


def degenerate_1d_agreement(n: int = 24, h_factor: float = 1.3,
                            eta_factor: float = 0.01, seed: int = 3) -> float:
    """Max ``|2-D code on an n x 1 grid - shipped 1-D conduction_rhs|``.

    Setting ``ky = 0`` (one row) must reproduce :func:`tensorsph.ice_thermal.
    sph_thermal.conduction_rhs` exactly -- if it did not, the 2-D module would be
    a different operator and could not claim to extend the ship.
    """
    from .ice_thermal.sph_thermal import conduction_rhs

    dx = 1.0
    h = h_factor * dx
    g = torch.Generator().manual_seed(seed)
    temp = torch.rand(n, 1, dtype=DTYPE, generator=g) + 0.5
    ys, xs = torch.meshgrid(torch.zeros(1, dtype=DTYPE),
                            (torch.arange(n, dtype=DTYPE) + 0.5) * dx,
                            indexing="ij")
    pos2 = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    vol = torch.full((n,), dx * dx, dtype=DTYPE)
    two_d = conduction_rhs_2d(temp.reshape(-1), torch.ones(n, dtype=DTYPE),
                              vol, 1.0, pos2, h, eta_factor)
    # 1-D ship: volume dx (a layer), same kernel, same 2/rho and harmonic mean
    pos1 = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    one_d = conduction_rhs(temp.reshape(-1), torch.ones(n, dtype=DTYPE),
                           torch.full((n,), dx, dtype=DTYPE), 1.0, pos1, h,
                           eta_factor)
    # the 2-D volume is dx^2, the 1-D volume is dx: same operator up to that
    return float((two_d / dx - one_d).abs().max())


# --------------------------------------------------------------------- solver
@dataclass
class ModeDecayConfig:
    """Decay of one Fourier mode on a periodic ``L x L`` square.

    ``fourier`` is the *amplitude* target: ``t_end`` is chosen so that the exact
    amplitude is ``amplitude_end`` times the initial one, i.e.
    ``t_end = -ln(amplitude_end) / (alpha (kx^2 + ky^2))``.  Lengths in m, time
    in s, ``alpha`` in m^2/s; ``rho = cp = 1`` so ``k = alpha``.
    """

    n: int = 64
    length: float = 1.0
    alpha: float = 1.0
    h_factor: float = 1.3
    eta_factor: float = 0.01
    periods: tuple[int, int] = (1, 1)     # mode = (2 pi px / L, 2 pi py / L)
    amplitude_end: float = 0.1
    courante: float = 0.2
    calibration: str = "k0_2d"            # "none" | "lattice_1d" | "k0_2d" | "mean_2d" | "mode_2d"
    integrator: str = "rk4"               # "rk4" (time error negligible) | "euler" (shipped style)
    sample_every: int = 0


@dataclass
class ModeDecayResult:
    n: int = 0
    dx: float = 0.0
    dt: float = 0.0
    steps: int = 0
    time: float = 0.0
    kx: float = 0.0
    ky: float = 0.0
    calibration: str = "none"
    integrator: str = "rk4"
    factor: float = 1.0
    raw_symbol_rate: float = 0.0     # lambda of the raw operator at the test mode
    amplitude_start: float = 0.0
    amplitude_end: float = 0.0
    amplitude_exact: float = 0.0
    max_normalised_error: float = 0.0    # max|dT| / max|T_exact| at t_end
    max_error_over_initial: float = 0.0  # max|dT| / amplitude at t = 0
    decay_rate_error: float = 0.0        # measured k_eff/ (alpha(kx^2+ky^2)) - 1
    sampled_time: list = field(default_factory=list)
    sampled_ratio: list = field(default_factory=list)   # amplitude(t)/amplitude(0)
    exact_ratio: list = field(default_factory=list)
    field_values: list = field(default_factory=list)
    analytic: list = field(default_factory=list)

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        return out


def run_mode_decay(cfg: ModeDecayConfig | None = None) -> ModeDecayResult:
    """Explicit-Euler time loop of the (calibrated) 2-D operator on one mode."""
    from .ice_thermal.sph_thermal import lattice_diffusion_factor

    cfg = cfg or ModeDecayConfig()
    n, length, alpha = int(cfg.n), float(cfg.length), float(cfg.alpha)
    dx = length / n
    h = cfg.h_factor * dx
    kx = 2.0 * math.pi * int(cfg.periods[0]) / length
    ky = 2.0 * math.pi * int(cfg.periods[1]) / length

    if cfg.calibration == "none":
        factor = 1.0
    elif cfg.calibration == "lattice_1d":
        factor = lattice_diffusion_factor(h_factor=cfg.h_factor)[0]
    elif cfg.calibration == "k0_2d":
        factor = lattice_diffusion_factor_2d(cfg.h_factor, dx, cfg.eta_factor, "k0")
    elif cfg.calibration == "mean_2d":
        factor = lattice_diffusion_factor_2d(cfg.h_factor, dx, cfg.eta_factor, "mean")
    elif cfg.calibration == "mode_2d":
        factor = symbol_rate(dx, h, kx, ky, cfg.eta_factor) / (kx * kx + ky * ky)
    else:
        raise ValueError(f"unknown calibration={cfg.calibration!r}")
    alpha_corr = 1.0 / factor if cfg.calibration != "none" else 1.0

    offsets = lattice_offsets(dx, h, cfg.eta_factor)
    temperature = mode_field(n, dx, kx, ky)

    t_end = -math.log(float(cfg.amplitude_end)) / (alpha * (kx * kx + ky * ky))
    dt = cfg.courante * dx * dx / alpha
    steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / steps

    decay0 = alpha * (kx * kx + ky * ky)
    res = ModeDecayResult(
        n=n, dx=dx, dt=dt, steps=steps, time=t_end, kx=kx, ky=ky,
        calibration=cfg.calibration, integrator=cfg.integrator, factor=factor,
        raw_symbol_rate=symbol_rate(dx, h, kx, ky, cfg.eta_factor),
        amplitude_start=mode_amplitude(temperature, n, dx, kx, ky),
    )
    times, ratios, exacts = [], [], []
    for step in range(steps + 1):
        t = step * dt
        if cfg.sample_every and step % cfg.sample_every == 0:
            a = mode_amplitude(temperature, n, dx, kx, ky)
            times.append(t)
            ratios.append(a / res.amplitude_start)
            exacts.append(math.exp(-decay0 * t))
        if step == steps:
            break
        if cfg.integrator == "euler":
            temperature = temperature + dt * alpha_corr * stencil_rhs_2d(
                temperature, offsets)
        elif cfg.integrator == "rk4":
            # classical RK4 on the *same* spatial operator; the time error becomes
            # negligible (O(dt^4)) so the reading is the spatial symbol error.  With
            # forward Euler the first-order time error at dt = 0.2 dx^2/alpha is
            # 1.8 % at n = 32 -- the same size as the spatial error and of the
            # opposite sign, i.e. it can masquerade as accuracy.
            def rhs(state):
                return alpha_corr * stencil_rhs_2d(state, offsets)
            k1 = rhs(temperature)
            k2 = rhs(temperature + 0.5 * dt * k1)
            k3 = rhs(temperature + 0.5 * dt * k2)
            k4 = rhs(temperature + dt * k3)
            temperature = temperature + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"unknown integrator={cfg.integrator!r}")
        if not torch.isfinite(temperature).all():
            raise FloatingPointError(f"2-D conduction diverged at step {step}")

    exact = mode_exact(n, dx, kx, ky, t_end, alpha)
    diff = (temperature - exact).abs()
    res.amplitude_end = mode_amplitude(temperature, n, dx, kx, ky)
    res.amplitude_exact = float(exact.abs().max())
    res.max_normalised_error = float(diff.max() / exact.abs().max())
    res.max_error_over_initial = float(diff.max() / res.amplitude_start)
    res.decay_rate_error = res.amplitude_end / (res.amplitude_start
                                                * math.exp(-decay0 * t_end)) - 1.0
    res.sampled_time, res.sampled_ratio, res.exact_ratio = times, ratios, exacts
    res.field_values = [float(v) for v in temperature.reshape(-1)]
    res.analytic = [float(v) for v in exact.reshape(-1)]
    return res


# --------------------------------------------------- independent reference (FD)
def fd_mode_decay(n: int = 64, length: float = 1.0, alpha: float = 1.0,
                  periods: tuple[int, int] = (1, 1), amplitude_end: float = 0.1,
                  r_target: float = 0.4) -> dict:
    """Crank-Nicolson (periodic, 5-point) reference, sharing no code with the SPH path.

    The initial field is a lattice eigenfunction, so this solver should agree
    with the closed form to its own truncation level -- that is what makes the
    closed form a *checked* target rather than an asserted one.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = int(n)
    dx = float(length) / n
    kx = 2.0 * math.pi * int(periods[0]) / float(length)
    ky = 2.0 * math.pi * int(periods[1]) / float(length)
    t_end = -math.log(float(amplitude_end)) / (alpha * (kx * kx + ky * ky))
    dt = float(r_target) * 2.0 * dx * dx / alpha
    steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / steps
    r = alpha * dt / (2.0 * dx * dx)

    idx = np.arange(n * n).reshape(n, n).reshape(-1)
    rows = [np.arange(n * n)]
    cols = [idx]
    vals = [np.full(n * n, -4.0)]
    for axis in (0, 1):
        for shift in (1, -1):
            rows.append(np.arange(n * n))
            cols.append(np.roll(np.arange(n * n).reshape(n, n), shift, axis=axis).reshape(-1))
            vals.append(np.ones(n * n))
    lap = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n * n, n * n))
    identity = sp.identity(n * n, format="csr")
    a_mat = (identity - r * lap).tocsc()
    b_mat = (identity + r * lap).tocsr()
    solve = spla.factorized(a_mat)

    x = (np.arange(n) + 0.5) * dx
    xg, yg = np.meshgrid(x, x, indexing="ij")
    temp = (np.cos(kx * xg) * np.cos(ky * yg)).reshape(-1)
    for _ in range(steps):
        temp = solve(b_mat @ temp)
    exact = (math.exp(-alpha * (kx * kx + ky * ky) * t_end)
             * np.cos(kx * xg) * np.cos(ky * yg)).reshape(-1)
    return {
        "n": n, "dx": dx, "dt": dt, "steps": steps, "t_end": t_end, "r": r,
        "max_normalised_error_vs_closed_form": float(
            np.abs(temp - exact).max() / np.abs(exact).max()),
    }
