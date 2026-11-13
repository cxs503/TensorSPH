"""Mixture-material surface tension: normalised phase-indicator colour.

Why this module exists
----------------------
The shipped colour function (:func:`tensorsph.csf_consistent.kernel_sum_color`)

    c_i = sum_j V_j W_ij ,        V_j = m_j / rho0   (a **scalar** rho0)

is a measure of *how much fluid is present* around particle ``i``. It is the
correct indicator for a **single-material** cloud with a free surface (``c``
falls 1 -> 1/2 at the outermost layer, the other half of the transition lives in
the empty void), which is why ``delta_s = 2 |grad c|`` is used there.

It is **not** a valid indicator for a **mixture of two materials**:

* the volume ``V_j = m_j / rho0`` needs one ``rho0`` for the whole cloud. If the
  two phases have rest densities ``rho_A != rho_B`` then one of them is assigned
  volumes that are wrong by ``rho_A / rho_B``. With ``rho_A = 1000``,
  ``rho_B = 1.2`` (water / air) and ``rho0 = 1000`` the light phase contributes
  ``1.2e-3 dx^2`` per particle, i.e. ``c`` is essentially a *heavy-phase*
  indicator; used the other way round (light phase inside) the colour is ~1 on
  both sides and the surface-tension force **vanishes**. The sign of the
  density contrast therefore changes the physics -- the defect is structural,
  not a discretisation error.
* ``int |grad c| dn`` is not 1 for a kernel sum: it is a presence measure, so
  the ``2 |grad c|`` special case tuned for the one-sided free surface does not
  generalise (measured ``int delta_s dl / (2 pi R) = 0.706`` on a complete
  two-phase interface, see ``benchmarks/two_phase_laplace/``).

What a mixture-material colour must satisfy
-------------------------------------------
The multi-material formulation used by SPHinXsys (Xiangyu-Hu/SPHinXsys,
``SurfaceTension``: the colour function is a kernel **ratio** of the target
material, i.e. the "C-correction" of Adami, Hu & Adams) is

    c^A_i = sum_{j in A} V_j W_ij  /  sum_j V_j W_ij        in [0, 1],

with ``V_j = m_j / rho0_phase(j)`` -- the **per-phase** volume.  Because the
denominator is the local kernel sum of the same volumes, ``c^A`` is
dimensionless and

* ``c^A = 1`` in the bulk of material A, ``c^A = 0`` in the bulk of any other
  material, whatever the density ratio (the normalisation cancels it);
* ``int |grad c^A| dn = c_A - c_B = 1`` across the interface, so the CSF delta
  function is ``delta_s = |grad c^A|`` -- **without** the factor 2 of the
  one-sided single-material rule;
* the interface gradient is resolved over the ~2h kernel shell for *both*
  phases, so the curvature band does not collapse when the light phase is the
  inner one.

The force is assembled as a **force density** ``f = -sigma kappa delta_s n``
(units N/m^3) and only afterwards divided by the particle density. This is the
mixture-consistent bookkeeping: in a two-material cloud the CSF force is a
volume force, so summing ``f_i V_i`` recovers the physical resultant on an arc
independently of which phase the particle belongs to. Dividing by ``rho``
before the summation (as a single-material acceleration form does) is what
makes the light phase appear to carry ``rho_A / rho_B`` times too much
momentum.

For a pair of materials ``A``, ``B`` the coefficient is ``sigma_AB`` (different
pairs may have different coefficients); the signed convention is the one of
:mod:`tensorsph.csf_consistent`: ``c`` decreases outward, ``n = -grad c / |grad c|``
is the outward normal, ``kappa = div n > 0`` on a convex drop of material A, and
``f`` points inward.

References
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335 (CSF).
    Adami, Hu & Adams (2010), J. Comput. Phys. 229:5011 (C-correction).
    Hu & Adams (2006), J. Comput. Phys. 213:844 (multi-phase SPH, continuum
        surface stress as a volume force).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .csf_consistent import mls_divergence, mls_gradient, shepard_smooth
from .csf_curvature import fit_curvature
from .droplet_impact import cubic_spline_kernel

__all__ = [
    "MixtureMaterial",
    "MixtureMaterialSet",
    "MixtureCSFState",
    "particle_volume",
    "mixture_color",
    "mixture_csf_fields",
    "mixture_surface_tension_force",
    "kernel_sum_color_single_rho0",
    "net_force",
    "laplace_force_ratio",
]


@dataclass(frozen=True)
class MixtureMaterial:
    """One material of a mixture.

    Attributes
    ----------
    name
        Identifier, used only for reporting.
    rho0
        Rest density [kg/m^3].  The particle mass of a material must be
        ``rho0 * V`` with the same lattice volume so that the mixture is
        volume-consistent (see :func:`particle_volume`).
    """

    name: str
    rho0: float


@dataclass(frozen=True)
class MixtureMaterialSet:
    """The materials of a simulation plus the pairwise surface-tension table.

    ``sigma(a, b)`` returns the coefficient for the interface between material
    index ``a`` and material index ``b``; it must be symmetric.
    """

    materials: tuple[MixtureMaterial, ...]
    sigma: dict[tuple[int, int], float] = field(default_factory=dict)

    def rho0_of(self, phase: torch.Tensor) -> torch.Tensor:
        """Per-particle rest density from an integer phase label array."""
        table = torch.tensor([m.rho0 for m in self.materials],
                             dtype=torch.float64)
        return table[phase.long()]

    def sigma_of(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        key = (min(a, b), max(a, b))
        if key not in self.sigma:
            raise KeyError(f"no surface-tension coefficient for pair {key}")
        return self.sigma[key]


def particle_volume(mass: torch.Tensor, phase: torch.Tensor,
                    materials: MixtureMaterialSet) -> torch.Tensor:
    """Per-phase volume ``V_i = m_i / rho0_phase(i)``.

    A volume-consistent lattice has ``m_i = rho0_phase * V_lattice``, so ``V_i``
    is the same for every particle; the formula is kept general so that
    unequal-resolution or unequal-volume clouds are still handled correctly.
    """
    return mass / materials.rho0_of(phase)


def _kernel_sum(pos: torch.Tensor, vol: torch.Tensor, pairs: torch.Tensor,
                h: float, weight: torch.Tensor) -> torch.Tensor:
    """``sum_j V_j weight_j W_ij`` (self contribution included)."""
    own = vol * cubic_spline_kernel(
        torch.zeros((), dtype=pos.dtype, device=pos.device), h)
    out = own * weight
    if pairs.numel() == 0:
        return out
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel(torch.linalg.vector_norm(pos[i] - pos[j], dim=1), h)
    out = out.index_add(0, i, w * vol[j] * weight[j])
    out = out.index_add(0, j, w * vol[i] * weight[i])
    return out


def mixture_color(pos: torch.Tensor, vol: torch.Tensor, phase: torch.Tensor,
                  pairs: torch.Tensor, h: float,
                  target_phase: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised phase-indicator colour of material ``target_phase``.

    Returns ``(c, den)`` with ``c = sum_{j in target} V_j W_ij / sum_j V_j W_ij``
    and ``den`` the kernel sum used as denominator (exposed so a caller can see
    where the cloud is too sparse for the ratio to be meaningful).
    """
    is_target = (phase == target_phase).to(pos.dtype)
    num = _kernel_sum(pos, vol, pairs, h, is_target)
    den = _kernel_sum(pos, vol, pairs, h, torch.ones_like(is_target))
    return num / den.clamp_min(1.0e-30), den


def kernel_sum_color_single_rho0(pos: torch.Tensor, mass: torch.Tensor,
                                 pairs: torch.Tensor, h: float,
                                 rho0: float) -> torch.Tensor:
    """The shipped single-material colour, verbatim, for the ablation arm.

    ``c_i = sum_j (m_j/rho0) W_ij`` clipped to [0, 1].  On a mixture with
    ``rho_A != rho_B`` this is the estimator that carries the density-ratio
    defect; :func:`mixture_color` is the normalised replacement.
    """
    vol = mass / rho0
    own = vol * cubic_spline_kernel(
        torch.zeros((), dtype=pos.dtype, device=pos.device), h)
    if pairs.numel() == 0:
        return own.clamp(0.0, 1.0)
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel(torch.linalg.vector_norm(pos[i] - pos[j], dim=1), h)
    c = own.index_add(0, i, w * vol[j])
    c = c.index_add(0, j, w * vol[i])
    return c.clamp(0.0, 1.0)


@dataclass
class MixtureCSFState:
    """Every field the mixture CSF force is assembled from (all scored)."""

    color: torch.Tensor        # (N,)  normalised indicator of the inner phase
    den: torch.Tensor          # (N,)  kernel sum (denominator of the ratio)
    grad_c: torch.Tensor       # (N,d) consistent gradient of the colour
    normal: torch.Tensor       # (N,d) outward unit normal
    kappa: torch.Tensor        # (N,)  curvature [1/m], > 0 on a convex drop
    delta_s: torch.Tensor      # (N,)  surface delta function [1/m]
    force_density: torch.Tensor  # (N,d) CSF force per unit volume [N/m^3]


def mixture_csf_fields(pos: torch.Tensor, vol: torch.Tensor,
                       phase: torch.Tensor, pairs: torch.Tensor, h: float,
                       sigma: float, inner_phase: int,
                       pairs_wide: torch.Tensor | None = None,
                       curvature_scheme: str = "mls",
                       curvature_stencil: float = 3.0,
                       normal_smoothing: int = 2,
                       delta_rule: str = "grad") -> MixtureCSFState:
    """Colour, normal, curvature, delta function and force density.

    Parameters
    ----------
    inner_phase
        Label of the material whose indicator defines ``c`` (``c -> 1`` inside
        that material).  The interface is shared, so the force is the same
        physical object whichever phase is named.
    curvature_scheme
        ``"mls"`` (``div n`` with the consistent divergence operator -- the
        default; measured -1.4 % on a circular interface at ``R/h = 12``) or
        ``"fit_circle"`` / ``"fit_quadratic"`` (local interface fit, see
        :mod:`tensorsph.csf_curvature`; on the mixture colour band both are far
        worse than ``div n``, see ``benchmarks/xval_mixture_laplace/``).
    delta_rule
        ``"grad"``: ``delta_s = |grad c|`` (the mixture rule -- the normalised
        colour completes the full 1 -> 0 transition, so no factor 2).
        ``"legacy2"``: ``delta_s = 2 |grad c|``, the single-material free
        surface rule, kept for the ablation.
    """
    color, den = mixture_color(pos, vol, phase, pairs, h, inner_phase)
    grad_c = mls_gradient(pos, color, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    tiny = 1.0e-12
    # c decreases outward (c = 1 in the inner material) => n = -grad c/|grad c|.
    normal = torch.where(mag > tiny, -grad_c / mag.clamp_min(tiny),
                         torch.zeros_like(grad_c))
    if normal_smoothing:
        smoothed = shepard_smooth(normal, pos, vol, pairs, h,
                                  passes=normal_smoothing)
        norm = torch.linalg.vector_norm(smoothed, dim=1, keepdim=True)
        normal = torch.where(mag > tiny, smoothed / norm.clamp_min(tiny), normal)

    mag_c = mag[:, 0]
    if delta_rule == "grad":
        delta_s = mag_c
    elif delta_rule == "legacy2":
        delta_s = 2.0 * mag_c
    else:
        raise ValueError(f"unknown delta_rule {delta_rule!r}")

    kappa = mls_divergence(pos, normal, vol, pairs, h)
    if curvature_scheme != "mls":
        wide = pairs_wide if pairs_wide is not None else pairs
        kappa = fit_curvature(pos, normal, delta_s, wide, h, fallback=kappa,
                             method=curvature_scheme, stencil=curvature_stencil)

    # Force density (volume force): f = -sigma kappa delta_s n .
    force_density = (-sigma * kappa * delta_s)[:, None] * normal
    return MixtureCSFState(color=color, den=den, grad_c=grad_c, normal=normal,
                           kappa=kappa, delta_s=delta_s,
                           force_density=force_density)


def mixture_surface_tension_force(pos: torch.Tensor, mass: torch.Tensor,
                                  phase: torch.Tensor, pairs: torch.Tensor,
                                  h: float, materials: MixtureMaterialSet,
                                  inner_phase: int,
                                  pairs_wide: torch.Tensor | None = None,
                                  **kwargs) -> tuple[torch.Tensor, MixtureCSFState]:
    """Convenience wrapper: build the state and return ``(acceleration, state)``.

    ``acceleration = force_density / rho`` with the **local** density, while the
    force density itself is the volume force -- the momentum of the pair is
    ``sum f_i V_i`` and is therefore independent of the density of the phase
    the particle belongs to.
    """
    vol = particle_volume(mass, phase, materials)
    sigma = materials.sigma_of(inner_phase, 1 - inner_phase) \
        if len(materials.materials) == 2 else max(materials.sigma.values())
    state = mixture_csf_fields(pos, vol, phase, pairs, h, sigma, inner_phase,
                               pairs_wide=pairs_wide, **kwargs)
    rho = materials.rho0_of(phase)
    return state.force_density / rho[:, None], state


def net_force(force_density: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    """``sum_i f_i V_i`` -- the resultant of the CSF force density [N] (2-D)."""
    return (force_density * vol[:, None]).sum(dim=0)


def laplace_force_ratio(force_density: torch.Tensor, vol: torch.Tensor,
                        pos: torch.Tensor, sigma: float,
                        axis: int = 0) -> dict:
    """``sigma_eff / sigma`` from the resultant on the ``+axis`` half.

    For a circular interface of radius ``R`` the exact resultant of
    ``f = -sigma kappa delta_s n`` over the half ``x > 0`` is

        int  sigma kappa delta_s n dV = sigma kappa * int_arc n dl = 2 sigma

    (the half-arc integrals to ``2 R n_hat`` and ``kappa = 1/R``), so the ratio
    of the measured resultant to ``2 sigma`` is the effective surface tension.
    """
    half = pos[:, axis] > 0
    f_half = (force_density[half] * vol[half, None]).sum(dim=0)
    resultant = float(torch.linalg.vector_norm(f_half))
    return {
        "resultant_half_N": resultant,
        "target_resultant_N": 2.0 * sigma,
        "sigma_eff_over_sigma": resultant / (2.0 * sigma),
    }