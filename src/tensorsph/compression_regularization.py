"""Compression-based density regularization -- a port of SPHinXsys.

Why this module exists
----------------------
The WCSPH path shipped here obtains density from a kernel summation and, when
asked, adds a *diffusive rate* on top of it (:mod:`tensorsph.density_diffusion`,
delta-SPH forms).  SPHinXsys obtains density by a different route: it evaluates a
**compression** -- the sum of the *reference* volumes in a particle's kernel
support -- and then *replaces* the density by ``rho0 * K`` every step.  The two
routes are not interchangeable (one integrates the continuity equation, the other
overwrites the field), so a fair cross-check has to port the second one as an
operator and not bolt it onto the first.

The port is a transcription of the reference implementation, not a paraphrase:
the arithmetic below is a line-by-line reading of

    Xiangyu-Hu/SPHinXsys @ master, tree sha 1612330e44c8947406f24b6756449e80dcd7158e
    src/shared/shared_ck/particle_dynamics/fluid_dynamics/density_regularization.{h,hpp,cpp}

and of the same file one month earlier (commit ``d5c793e08c``, 2026-06-19,
"using compression formulation for fluid"), before commit ``375ec50b42``
(2026-07-19, "only use compression sum larger than one to obtain average")
introduced the reference-mean normalisation.  Both formulations are supported
because they differ in one factor that turns out to matter a lot
(:data:`NORMALIZATIONS`).

The algorithm, in the order the C++ dynamics list runs it
---------------------------------------------------------
1. ``CompressionSummation<Inner>`` (+ ``<Contact>`` for wall particles)

       S_i = W(0) * Vref_i + (1/Kref) * sum_j W_ij * Vref_j

   with ``Vref`` the volumetric measure *at the reference state* (``m / rho0``;
   ``VolumetricMeasureRef`` is registered from ``VolumetricMeasure`` once and
   never updated) and ``Kref`` the single global scalar
   ``InverseReferenceCompression``.  The self term is **not** scaled by
   ``1/Kref`` in the reference implementation -- that asymmetry is reproduced
   here and listed in the cross-check report.

2. ``AverageCompression`` -- one reduction over the fluid body, restricted to
   particles whose sum exceeds one (``ReduceKernel::reduce``), giving

       Kref = 1 / ( mean of { S_i : S_i > 1 } ).

3. ``DensityRegularization<FlowType>`` -- per particle, through a flow-type
   "regularization" functor and then a plain density assignment:

       K_i    = rule(S_i)                (rule = internal / free_surface / ...)
       rho_i  = K_i * rho0_i

   ``internal`` returns ``S_i`` unchanged; ``free_surface`` returns
   ``max(S_i, 1)``, which is what stops a truncated support (a particle at the
   free surface has fewer neighbours than a reference-state particle) from
   reading as expansion.

The one-step lag
----------------
``CompressionSummation`` consumes the ``Kref`` that step 2 wrote during the
*previous* step; step 2 then overwrites it from the sums just computed, and step
3 uses those sums.  :class:`CompressionRegularization` keeps that state in an
attribute and reproduces the ordering, so a step-by-step replay matches the C++
dynamics list rather than a fixed point of it.

Scaling note (why the constant ``1`` matters)
---------------------------------------------
``S_i`` is dimensionless only because the reference volumes of a complete
lattice sum, kernel-weighted, to ``1``.  With the lattice volume
``V = dx**D`` the discrete sum is ``dx**D * sum_j W_ij = 1 + b``, i.e. it
carries the lattice offset ``b`` (5.88e-3 at ``h = 1.3 dx``, 2-D, measured by
``tests/test_lattice_mass_calibration.py``).  The port therefore expects the
volumes of the *reference* lattice, and the analytic checks compare against
``1 + b`` rather than 1 (``tests/test_compression_regularization.py``).
"""
from __future__ import annotations

import torch

#: Flow-type regularization rules of ``Regularization<FlowType>``.
RULES = ("internal", "free_surface", "free_stream")

#: ``"reference_mean"`` = master (neighbour sum scaled by ``1/Kref``);
#: ``"none"`` = the 2026-06-19 formulation (plain kernel sum of reference
#: volumes, ``Kref`` unused).
NORMALIZATIONS = ("reference_mean", "none")


def reference_volumes(mass: torch.Tensor, rho0: float) -> torch.Tensor:
    """``Vref = m / rho0`` -- the ``VolumetricMeasure`` at the reference state."""
    if not mass.is_floating_point():
        raise TypeError("mass must be floating-point")
    if not (rho0 > 0.0):
        raise ValueError("rho0 must be positive")
    return mass / rho0


def compression_summation(vol_ref: torch.Tensor, w_self: float,
                          pairs: torch.Tensor, wij: torch.Tensor,
                          fluid_count: int,
                          inverse_reference_compression: float = 1.0
                          ) -> torch.Tensor:
    """``CompressionSummation`` for the fluid particles.

    ``vol_ref`` holds the reference volumes of *all* particles with the fluid
    ones first (``vol_ref[:fluid_count]``), ``pairs`` is the unordered pair list
    over that same indexing (fluid-fluid and fluid-wall, no wall-wall pairs),
    ``wij = W(|x_i - x_j|, h)`` and ``w_self = W(0, h)``.

    Returns ``S`` of length ``fluid_count``::

        S_i = w_self * Vref_i + (1/Kref) * sum_j W_ij Vref_j

    The two ``CompressionSummation`` specialisations of the reference (``Inner``
    for fluid neighbours, ``Contact`` for wall neighbours) add the same term with
    the same ``1/Kref`` factor, so one combined pair list reproduces both.
    ``Kref`` enters the neighbour term only -- the self term is not scaled, as in
    the reference implementation.
    """
    if not (inverse_reference_compression > 0.0):
        raise ValueError("inverse_reference_compression must be positive")
    n = int(fluid_count)
    if vol_ref.shape[0] < n:
        raise ValueError("vol_ref must cover at least fluid_count particles")

    sums = w_self * vol_ref[:n]
    if pairs.numel() == 0:
        return sums

    i, j = pairs.unbind(dim=1)
    weighted = inverse_reference_compression * wij
    sums = sums.index_add(0, i, weighted * vol_ref[j])
    # The reversed direction exists only when the second end is a fluid
    # particle: a wall particle's own sum is never used, and index_add would
    # reject the out-of-range index.
    reverse = (j < n) | (i >= n)
    if bool(reverse.any()):
        ii = j[reverse]
        jj = i[reverse]
        sums = sums.index_add(0, ii, weighted[reverse] * vol_ref[jj])
    return sums


def average_inverse_reference_compression(
        compression_sum: torch.Tensor) -> tuple[float, int]:
    """``AverageCompression`` -- ``1 / mean{S_i : S_i > 1}``.

    Returns ``(inverse, counted)``.  ``counted`` is the number of particles that
    took part; the reference implementation divides two reduced sums, so an
    empty set is ``0/0``.  Here an empty set keeps ``Kref = 1`` (identity
    regularisation) instead of propagating ``nan``; the caller can see it from
    ``counted == 0``.
    """
    if compression_sum.numel() == 0:
        return 1.0, 0
    mask = compression_sum > 1.0
    counted = int(mask.sum())
    if counted == 0:
        return 1.0, 0
    total = float(compression_sum[mask].sum())
    if total <= 0.0:
        return 1.0, 0
    return counted / total, counted


def regularize(compression_sum: torch.Tensor, rule: str = "free_surface",
               indicator: torch.Tensor | None = None) -> torch.Tensor:
    """``Regularization<FlowType>::ComputingKernel::operator()``.

    * ``internal``     -- ``S`` unchanged.
    * ``free_surface`` -- ``max(S, 1)``.
    * ``free_stream``  -- ``1`` where ``indicator != 0``, else ``S`` (the
      reference uses a per-particle ``Indicator``; pass one of the same length).
    """
    if rule == "internal":
        return compression_sum
    if rule == "free_surface":
        return compression_sum.clamp_min(1.0)
    if rule == "free_stream":
        if indicator is None:
            raise ValueError("rule 'free_stream' needs an indicator array")
        if indicator.shape[0] != compression_sum.shape[0]:
            raise ValueError("indicator must be defined on the fluid particles")
        one = torch.ones_like(compression_sum)
        return torch.where(indicator != 0, one, compression_sum)
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def regularized_density(compression: torch.Tensor, rho0: float) -> torch.Tensor:
    """``rho_i = K_i * rho0_i`` (``eos_.getReferenceDensity(index_i)``)."""
    return compression * rho0


class CompressionRegularization:
    """The three-stage dynamics list of the reference, with its state.

    ``update`` runs the stages in the reference order and returns
    ``(rho, diagnostics)``.  Diagnostics carry the raw sums, the global
    inverse reference compression that was *used* and the one that was
    *written*, and how many particles took part in the reduction -- everything a
    cross-check or an ablation needs to attribute a difference.
    """

    def __init__(self, rule: str = "free_surface",
                 normalization: str = "reference_mean") -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization {normalization!r}; "
                             f"expected one of {NORMALIZATIONS}")
        self.rule = rule
        self.normalization = normalization
        # sv_compression_inv_ref_ is registered with Real(1).
        self.inverse_reference_compression = 1.0

    def update(self, vol_ref: torch.Tensor, w_self: float, pairs: torch.Tensor,
               wij: torch.Tensor, fluid_count: int, rho0: float,
               indicator: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, dict]:
        used = (self.inverse_reference_compression
                if self.normalization == "reference_mean" else 1.0)
        compression_sum = compression_summation(
            vol_ref, w_self, pairs, wij, fluid_count,
            inverse_reference_compression=used)
        written, counted = average_inverse_reference_compression(compression_sum)
        compression = regularize(compression_sum, self.rule, indicator)
        rho = regularized_density(compression, rho0)
        # Step 2 overwrites the scalar after step 1 has consumed it.
        self.inverse_reference_compression = written
        diagnostics = {
            "rule": self.rule,
            "normalization": self.normalization,
            "inverse_reference_compression_used": float(used),
            "inverse_reference_compression_written": float(written),
            "average_compression": (1.0 / written if written > 0.0 else 1.0),
            "counted_particles": counted,
            "compression_sum_min": float(compression_sum.min()) if compression_sum.numel() else None,
            "compression_sum_max": float(compression_sum.max()) if compression_sum.numel() else None,
            "compression_clipped_particles": int((compression_sum < 1.0).sum())
            if self.rule == "free_surface" else 0,
        }
        return rho, diagnostics


def compression_from_summation_density(rho: torch.Tensor, rho0: float
                                       ) -> torch.Tensor:
    """``K = rho / rho0`` -- lets a summation-density field be read as compression.

    Used by the cross-check report to state the *equivalence* of the ``"none"``
    normalisation with the shipped summation density: with
    ``rho_i = sum_j m_j W_ij`` and ``Vref_j = m_j / rho0`` the two are the same
    number up to the free-surface clip.
    """
    return rho / rho0