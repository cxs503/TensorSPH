"""Cross-validation port of SPHinXsys' surface-tension stress formulation.

What is ported, and from where
-------------------------------
This module reproduces the *momentum-conservative* surface-tension model of
SPHinXsys (Zhang & Hu) so that it can be run side by side with the CSF model
shipped in :mod:`tensorsph.surface_tension`.  Nothing below was transcribed from
memory: every formula is a line-for-line port of the upstream sources listed
here (fetched through the GitHub contents API; the raw bytes, their sha256 and
the fetch metadata are archived under
``docs/xval_sphinxys_surface_tension/raw/``)::

    src/shared/particle_dynamics/fluid_dynamics/surface_tension.cpp   (formulae)
    src/shared/particle_dynamics/fluid_dynamics/surface_tension.hpp   (data plumbing)
    src/shared/kernels/kernel_wendland_c2.cpp                         (kernel)
    src/shared/adaptations/adaptation.h                               (h/dp = 1.3)
    src/shared/particle_dynamics/fluid_dynamics/fluid_time_step.cpp   (capillary dt)
    tests/2d_examples/test_2d_square_droplet/2d_square_droplet.cpp    (the case)

The upstream model in three steps
---------------------------------
1. **Colour gradient from the *other* phase only** (``SurfaceTensionStress::
   interaction``), weighted by the mixture "contact fraction"
   ``f_k = rho0_i / (rho0_i + rho0_k)``::

       grad c_i = -2 f_k * sum_{j in other phase} V_j * grad_i W_ij
       n_i      = grad c_i / (|grad c_i| + Eps)
       S_i      = sigma * (I - n_i (x) n_i) * |grad c_i|          (stress, N/m^2)

   There is **no explicit curvature** anywhere in the model: the Laplace
   pressure is recovered from the divergence of the stress tensor (continuous
   surface stress, CSS), which is why the discretisation is momentum
   conservative:

       div S = sigma * ( d_t(|grad c|) t + kappa |grad c| n )

2. **Inner force** (``SurfaceStressForce<Inner<>>::interaction``), a symmetric
   two-point flux::

       a_i = (m_i / rho_i) sum_j V_j (S_i + S_j + H_ij) . grad_i W_ij

   with the zero-surface-energy ("hourglass") penalty
   ``H_ij = 4.5 sigma/2 (T_i + T_j) M_ij /(r_ij + Eps)``,
   ``T = I - n (x) n``, ``M_ij = -(g.e) g (x) e /(|g.e| |g|)``,
   ``g = (grad c_i + grad c_j)/2``.  ``H`` is antisymmetric under ``i <-> j``,
   the stress flux is symmetric, so the stress part conserves momentum exactly.

3. **Contact force** (``SurfaceStressForce<Contact<>>``) for the other phase::

       a_i += (m_i/rho_i) sum_{j in other} V_j (2(1-f) S_i + 2 f S_j + Hc_ij).grad_i W_ij

   ``2(1-f) S_i + 2 f S_j`` is the same combination from either side of the
   pair because ``f_j = 1 - f_i`` by construction.

Two things the port needs on top of the literal transcription, both switchable
and both documented in ``docs/xval_sphinxys_surface_tension/README.md``:

``delta="sphinxys"`` / ``delta="unit"``
    ``|grad c|`` is the surface delta function; its integral across the
    interface must be 1 for ``Delta p = sigma/R`` to hold.  Because
    ``grad c = -2 f_k grad(chi_other)`` with ``chi`` the smoothed phase
    indicator, the literal ``2 f_k`` prefactor integrates to ``2 f_k`` -- 1 for
    a 1:1 density pair and 2 for a 1000:1 pair.  ``"sphinxys"`` keeps the
    upstream factor; ``"unit"`` divides it out (``f_k -> 1/2``) so the delta is
    normalised at every density ratio.  The benchmark measures both.

``mode="two_phase"`` / ``mode="single_phase"``
    Upstream surface tension requires a second phase: a free surface in vacuum
    has no contact neighbours.  For a single-phase scene the missing half-space
    is recovered from the kernel-sum identity
    ``sum_{missing} V_j grad W_ij = -sum_{present} V_j grad W_ij`` (exact in the
    continuum for a complete kernel support), i.e. ``f_k -> 1`` and the sum runs
    over the *present* neighbours with a flipped sign.  The contact force then
    vanishes as ``rho_air/(rho_w + rho_air) -> 0`` and is dropped.

Units: SI, 2-D.  ``sigma`` [N/m], ``rho`` [kg/m^3], ``a`` [m/s^2],
``S`` [N/m^2] (a 2-D stress: force per unit length), ``f = rho a`` [N/m^3].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .droplet_impact import cubic_spline_gradient, wendland_c2_gradient

#: Upstream constants, read from the sources listed in the module docstring.
SPHINXSYS_H_SPACING_RATIO = 1.3      # adaptation.h: SPHAdaptation(.., h_spacing_ratio = 1.3)
SPHINXSYS_HOURGLASS_COEFF = 4.5      # surface_tension.h: hourglass_control_coeff = 4.5
SPHINXSYS_CAPILLARY_CFL = 0.6        # 2d_square_droplet.cpp: SurfaceTensionTimeStep(body, 0.6)
SPHINXSYS_EPS = 1.0e-12              # base_data_type.h Eps (double); kept for divisions

#: Kernel gradients with ``grad_i W(|x_i - x_j|)``, i.e. argument ``x_i - x_j``.
_GRADIENTS = {
    "wendland": wendland_c2_gradient,   # upstream default (KernelWendlandC2)
    "cubic": cubic_spline_gradient,     # TensorSPH default (2-D cubic spline)
}


def _grad(delta: torch.Tensor, h: float, kernel: str) -> torch.Tensor:
    try:
        return _GRADIENTS[kernel](delta, h)
    except KeyError:  # pragma: no cover - argument validation
        raise ValueError(f"unknown kernel {kernel!r}; expected {sorted(_GRADIENTS)}") from None


def contact_fraction(rho0_self: torch.Tensor, rho0_other: torch.Tensor) -> torch.Tensor:
    """``f_k = rho0_i / (rho0_i + rho0_k)`` (surface_tension.cpp line ~"contact_fraction_.push_back").

    The pair sums to one: ``f(rho_a, rho_b) + f(rho_b, rho_a) = 1``, which is what
    makes the two-sided contact flux of step 3 consistent.
    """
    return rho0_self / (rho0_self + rho0_other)


# --------------------------------------------------------------------------- #
# Step 1: colour gradient, normal, stress
# --------------------------------------------------------------------------- #
@dataclass
class StressFields:
    """Per-particle interface state (all exposed so the benchmark can audit it)."""

    color_gradient: torch.Tensor   # (N, 2) grad c
    normal: torch.Tensor           # (N, 2) n = grad c / (|grad c| + Eps)
    stress: torch.Tensor           # (N, 2, 2) S = sigma (I - n n^T) |grad c|
    delta_s: torch.Tensor          # (N,) |grad c|, the surface delta function


def contact_colour_gradient(
    pos: torch.Tensor,
    vol: torch.Tensor,
    pairs_cross: torch.Tensor,
    frac: torch.Tensor,
    h: float,
    kernel: str = "wendland",
    delta: str = "sphinxys",
) -> torch.Tensor:
    """``grad c_i = -2 f_i sum_{j in other phase} V_j grad_i W_ij``.

    Contributions are assembled from each unordered cross-phase pair once and
    added to *both* ends with the other end's volume, exactly as the upstream
    ``contact_configuration`` loops do for the two bodies.

    Args:
        pos: ``(N, 2)`` positions.
        vol: ``(N,)`` particle volumes ``V_j`` (= ``VolumetricMeasure``).
        pairs_cross: ``(P, 2)`` *unordered* pairs with the two ends in
            different phases; ``i < j`` is not assumed.
        frac: ``(N,)`` contact fraction ``f_k`` of each particle.
        delta: ``"sphinxys"`` keeps ``2 f_k``; ``"unit"`` replaces it by 1 so
            that ``int |grad c| dn = 1`` at any density ratio.
    """
    grad_c = torch.zeros_like(pos)
    if pairs_cross.numel() == 0:
        return grad_c
    i, j = pairs_cross.unbind(dim=1)
    gw = _grad(pos[i] - pos[j], h, kernel)

    if delta == "sphinxys":
        w_i = 2.0 * frac
        w_j = 2.0 * frac
    elif delta == "unit":
        w_i = torch.ones_like(frac)
        w_j = torch.ones_like(frac)
    else:  # pragma: no cover - argument validation
        raise ValueError(f"unknown delta {delta!r}; expected 'sphinxys' or 'unit'")

    grad_c = grad_c.index_add(0, i, -(w_i[i])[:, None] * vol[j][:, None] * gw)
    # grad_j W_ji = -grad_i W_ij, hence the sign flip for the partner particle.
    grad_c = grad_c.index_add(0, j, (w_j[j])[:, None] * vol[i][:, None] * gw)
    return grad_c


def single_phase_colour_gradient(
    pos: torch.Tensor,
    vol: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
    kernel: str = "wendland",
    delta: str = "sphinxys",
    scale: float = 1.0,
) -> torch.Tensor:
    """Vacuum reduction of step 1 for a single-phase scene.

    With no second phase the upstream sum runs over the *missing* neighbours,
    and ``sum_missing V grad W = -sum_present V grad W`` for a complete support,
    so ``grad c_i = +2 sum_{present} V_j grad_i W_ij`` (``f_k -> 1``).
    ``scale`` multiplies the result (the benchmark uses it to scan the delta
    normalisation without rebuilding pairs).
    """
    grad_c = torch.zeros_like(pos)
    if pairs.numel() == 0:
        return grad_c
    i, j = pairs.unbind(dim=1)
    gw = _grad(pos[i] - pos[j], h, kernel)
    factor = 2.0 if delta == "sphinxys" else 1.0
    factor *= scale
    grad_c = grad_c.index_add(0, i, factor * vol[j][:, None] * gw)
    # grad_j W_ji = -grad_i W_ij: the partner end carries the opposite sign.
    grad_c = grad_c.index_add(0, j, -factor * vol[i][:, None] * gw)
    return grad_c


def surface_stress(
    grad_c: torch.Tensor,
    sigma: float,
    eps: float = SPHINXSYS_EPS,
) -> StressFields:
    """``n = grad c/(|grad c| + Eps)``, ``S = sigma (I - n n^T) |grad c|``."""
    mag = torch.linalg.vector_norm(grad_c, dim=1)
    normal = grad_c / (mag + eps)[:, None]
    dim = grad_c.shape[1]
    eye = torch.eye(dim, dtype=grad_c.dtype, device=grad_c.device)
    tangent = eye - torch.einsum("na,nb->nab", normal, normal)
    stress = sigma * tangent * mag[:, None, None]
    return StressFields(color_gradient=grad_c, normal=normal, stress=stress, delta_s=mag)


# --------------------------------------------------------------------------- #
# Step 2: inner force (momentum-conservative stress divergence + hourglass)
# --------------------------------------------------------------------------- #
def inner_stress_force(
    pos: torch.Tensor,
    mass: torch.Tensor,
    rho: torch.Tensor,
    vol: torch.Tensor,
    pairs: torch.Tensor,
    fields: StressFields,
    sigma: float,
    h: float,
    kernel: str = "wendland",
    hourglass: bool = True,
    hourglass_coeff: float = SPHINXSYS_HOURGLASS_COEFF,
    eps: float = SPHINXSYS_EPS,
) -> torch.Tensor:
    """Acceleration from the same-phase stress divergence.

    ``a_i = (m_i/rho_i) sum_j V_j (S_i + S_j + H_ij) . grad_i W_ij``.

    The pair list is stored once, so the ``j`` end is assembled explicitly:
    ``H`` flips sign under ``i <-> j`` (``e_ji = -e_ij``) while ``S_i + S_j``
    does not, which is what makes the stress part conserve momentum.
    """
    acc = torch.zeros_like(pos)
    if pairs.numel() == 0:
        return acc
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    r = torch.linalg.vector_norm(delta, dim=1)
    e = delta / r.clamp_min(eps)[:, None]
    gw = _grad(delta, h, kernel)          # = W'(r) e_ij
    w_prime = (gw * e).sum(dim=1)         # W'(r) < 0

    flux_s = torch.einsum("pab,pb->pa", fields.stress[i] + fields.stress[j], gw)

    if hourglass and hourglass_coeff != 0.0:
        g_avg = 0.5 * (fields.color_gradient[i] + fields.color_gradient[j])
        ge = (g_avg * e).sum(dim=1)
        g_mag = torch.linalg.vector_norm(g_avg, dim=1)
        # H . gw = -sign(g.e) W' / |g| * (T_i + T_j) g,  T = I - n n^T.
        coef = -(w_prime * ge) / ((g_mag + eps) * (ge.abs() + eps))
        n_i, n_j = fields.normal[i], fields.normal[j]
        tg = (2.0 * g_avg
              - n_i * (n_i * g_avg).sum(dim=1)[:, None]
              - n_j * (n_j * g_avg).sum(dim=1)[:, None])
        hg = (0.5 * hourglass_coeff * sigma * coef / (r + eps))[:, None] * tg
    else:
        hg = torch.zeros_like(flux_s)

    acc = acc.index_add(0, i, (mass[i] / rho[i])[:, None] * vol[j][:, None] * (flux_s + hg))
    acc = acc.index_add(0, j, -(mass[j] / rho[j])[:, None] * vol[i][:, None] * (flux_s - hg))
    return acc


# --------------------------------------------------------------------------- #
# Step 3: contact force (two-phase only)
# --------------------------------------------------------------------------- #
def contact_stress_force(
    pos: torch.Tensor,
    mass: torch.Tensor,
    rho: torch.Tensor,
    vol: torch.Tensor,
    pairs_cross: torch.Tensor,
    fields: StressFields,
    frac: torch.Tensor,
    sigma: float,
    h: float,
    kernel: str = "wendland",
    hourglass: bool = True,
    hourglass_coeff: float = SPHINXSYS_HOURGLASS_COEFF,
    eps: float = SPHINXSYS_EPS,
) -> torch.Tensor:
    """Acceleration from the cross-phase stress flux.

    ``a_i += (m_i/rho_i) sum_j V_j (2(1-f_i) S_i + 2 f_i S_j + Hc_ij).grad_i W_ij``

    Upstream evaluates this in each body's own contact loop; because
    ``f_j = 1 - f_i`` the stress bracket is the *same* for both ends, so it is
    assembled once here and applied with opposite ``grad W`` signs.
    """
    acc = torch.zeros_like(pos)
    if pairs_cross.numel() == 0:
        return acc
    i, j = pairs_cross.unbind(dim=1)
    delta = pos[i] - pos[j]
    r = torch.linalg.vector_norm(delta, dim=1)
    e = delta / r.clamp_min(eps)[:, None]
    gw = _grad(delta, h, kernel)
    w_prime = (gw * e).sum(dim=1)

    f_i = frac[i]
    s_ij = torch.einsum("pab,pb->pa",
                        (2.0 * (1.0 - f_i))[:, None, None] * fields.stress[i]
                        + (2.0 * f_i)[:, None, None] * fields.stress[j], gw)

    if hourglass and hourglass_coeff != 0.0:
        g_avg = 0.5 * (fields.color_gradient[i] + fields.color_gradient[j])
        ge = (g_avg * e).sum(dim=1)
        g_mag = torch.linalg.vector_norm(g_avg, dim=1)
        n_i, n_j = fields.normal[i], fields.normal[j]
        # Hc . gw = -4 f(1-f) coeff sigma/2 / r * [(n_i n_i^T + n_j n_j^T) gw
        #                                      - sign(g.e) W'/|g| (n_i (n_i.g) + n_j (n_j.g))]
        pref = -4.0 * f_i * (1.0 - f_i) * 0.5 * hourglass_coeff * sigma / (r + eps)
        nn_gw = (n_i * (n_i * gw).sum(dim=1)[:, None]
                 + n_j * (n_j * gw).sum(dim=1)[:, None])
        coef = -(w_prime * ge) / ((g_mag + eps) * (ge.abs() + eps))
        nn_g = (n_i * (n_i * g_avg).sum(dim=1)[:, None]
                + n_j * (n_j * g_avg).sum(dim=1)[:, None])
        hg_ij = pref[:, None] * (nn_gw - coef[:, None] * nn_g)
        # partner end: e -> -e inside the ``I - sign(g.e) g e^T/|g|`` bracket
        hg_ji = pref[:, None] * (nn_gw + coef[:, None] * nn_g)
    else:
        hg_ij = torch.zeros_like(s_ij)
        hg_ji = hg_ij

    acc = acc.index_add(0, i, (mass[i] / rho[i])[:, None] * vol[j][:, None] * (s_ij + hg_ij))
    acc = acc.index_add(0, j, -(mass[j] / rho[j])[:, None] * vol[i][:, None] * (s_ij - hg_ji))
    return acc


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def surface_tension_acceleration(
    pos: torch.Tensor,
    mass: torch.Tensor,
    rho: torch.Tensor,
    vol: torch.Tensor,
    pairs_inner: torch.Tensor,
    pairs_cross: torch.Tensor | None,
    sigma: float,
    h: float,
    frac: torch.Tensor | None = None,
    kernel: str = "wendland",
    mode: str = "two_phase",
    delta: str = "sphinxys",
    hourglass: bool = True,
    hourglass_coeff: float = SPHINXSYS_HOURGLASS_COEFF,
    eps: float = SPHINXSYS_EPS,
) -> tuple[torch.Tensor, StressFields]:
    """Full SPHinXsys surface-tension acceleration + the fields it was built from.

    ``mode="two_phase"`` needs ``pairs_cross`` (each unordered cross-phase pair
    once) and ``frac`` (contact fraction per particle).  ``mode="single_phase"``
    ignores both and uses :func:`single_phase_colour_gradient`; the contact
    force is dropped because it scales with ``rho_other/(rho_self+rho_other)``.
    """
    if mode == "two_phase":
        if pairs_cross is None or frac is None:
            raise ValueError("two_phase mode needs pairs_cross and frac")
        grad_c = contact_colour_gradient(pos, vol, pairs_cross, frac, h,
                                         kernel=kernel, delta=delta)
    elif mode == "single_phase":
        grad_c = single_phase_colour_gradient(pos, vol, pairs_inner, h, kernel=kernel,
                                              delta=delta)
    else:  # pragma: no cover - argument validation
        raise ValueError(f"unknown mode {mode!r}; expected 'two_phase' or 'single_phase'")

    fields = surface_stress(grad_c, sigma, eps=eps)
    acc = inner_stress_force(pos, mass, rho, vol, pairs_inner, fields, sigma, h,
                             kernel=kernel, hourglass=hourglass,
                             hourglass_coeff=hourglass_coeff, eps=eps)
    if mode == "two_phase":
        acc = acc + contact_stress_force(pos, mass, rho, vol, pairs_cross, fields, frac,
                                         sigma, h, kernel=kernel, hourglass=hourglass,
                                         hourglass_coeff=hourglass_coeff, eps=eps)
    return acc, fields


def capillary_dt_limit(rho0: float, h: float, sigma: float,
                       cfl: float = SPHINXSYS_CAPILLARY_CFL) -> float:
    """Capillary time step of ``SurfaceTensionTimeStep::outputResult``.

    Upstream: ``dt = acousticCFL * h / max(c, sqrt(2 pi sigma / (rho0 h)))``;
    with the acoustic speed removed this is the capillary bound
    ``dt = cfl sqrt(rho0 h^3 / (2 pi sigma))``.  Compare
    :func:`tensorsph.surface_tension.capillary_dt_limit`, which uses
    ``cfl sqrt(rho0 h^3 / sigma)`` with ``cfl = 0.25`` -- the upstream form is
    ``sqrt(2 pi) = 2.51`` times looser per CFL unit.
    """
    if sigma <= 0.0:
        return float("inf")
    import math

    return cfl * math.sqrt(rho0 * h ** 3 / (2.0 * math.pi * sigma))