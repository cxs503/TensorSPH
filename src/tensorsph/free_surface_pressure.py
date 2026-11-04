"""Free-surface pressure closure (Adami, Ihmsen, Adams 2012, section 3.3).

Why this module exists
----------------------
``tait_pressure`` clamps the pressure at ``p >= 0`` to avoid the tensile
instability, and a WCSPH *sum-density* is deficient wherever the kernel support
is truncated.  On a free surface the two effects add up: every particle in the
outer shell carries a density deficit of tens of percent, the Tait law maps that
to ``p < 0``, the clamp maps it to ``p = 0``, and the shell therefore cannot
carry the pressure jump that a capillary surface must carry.

``benchmarks/laplace_pressure_floor`` measured the consequence: the interior of
a drop relaxes to ``0.57 +- 0.06`` of the compression that Young-Laplace
requires, independent of ``c0`` and ``dx``, i.e. the residual is a *closure*
error and not an EOS/noise floor.  This module is the closure: instead of the
clamped Tait value, a particle that sits in the deficient shell gets the
kernel-weighted extrapolation of its neighbours' pressure,

    p_i = max(0, sum_j (p_j + rho_j * g . (r_i - r_j)) W_ij) / sum_j W_ij ,

which is exactly eq. (11) of Adami et al. (2012) specialised to a static
particle (``a_i = 0``).  It is the same construction the solver already uses for
the static *wall* particles, which is why the wall rows of the hydrostatic
benchmark are accurate while the free surface was not.

Two variants are implemented, because they differ in one physical choice:

``"shell"``
    Extrapolate the pressure only.  The density stays the continuity solution,
    so the momentum term ``p_i / rho_i^2`` sees an extrapolated ``p`` next to a
    deficient ``rho``.  Cheap, but the surface layer is inconsistent.

``"shell_density"``
    Extrapolate the pressure *and* take the Tait-consistent density for the
    flagged particles, ``rho_i = rho0 * (1 + p_i / B)^(1/gamma)`` -- the same
    pair of lines the wall particles use.  The corrected density is used **only
    in the momentum assembly**: the continuity equation keeps the true
    sum-density, so mass is still conserved and the returned ``rho`` is
    unchanged.  This is the variant the acceptance criterion is stated for.

Nothing here touches the time integration: the function is a pure transform of
``(rho, pressure)`` at the current step, which keeps it testable in isolation
(``tests/test_free_surface_pressure.py``).
"""
from __future__ import annotations

import torch

MODES = ("shell", "shell_density")


def _body_force_term(cfg) -> bool:
    return float(getattr(cfg, "gravity", 0.0)) != 0.0


def extrapolated_pressure(pos, rho, pressure, mass, pairs, n, cfg,
                          threshold: float | None = None):
    """Return ``(pressure_eff, rho_eff)`` with the free surface closed.

    ``pairs`` holds *all* pairs the solver built (wall particles included).  Only
    fluid-fluid pairs are used, matching the wall treatment which never feeds
    the wall extrapolation back into the fluid.

    Particles are flagged as free surface when ``rho < threshold * rho0``
    (default ``cfg.free_surface_density_threshold``, 0.99 -- the deficit shell
    starts at about ``-12 %``, so 0.99 selects it and leaves the interior, whose
    lattice bias is ``+0.6 %``, alone).  A flagged particle with no fluid
    neighbour keeps ``p = 0``.
    """
    if getattr(cfg, "free_surface_pressure", "off") == "off":
        # The caller (rhs) already guards on the mode; the guard is repeated here
        # so the helper is safe to call unconditionally.
        return pressure, rho
    if pairs is None or pairs.numel() == 0:
        return pressure, rho
    thr = (cfg.free_surface_density_threshold if threshold is None
           else float(threshold))
    i, j = pairs.unbind(dim=1)
    # fluid-fluid pairs only, in both directions: the sums below are symmetric
    # kernel sums, so each pair contributes to both of its endpoints.
    fluid = (i < n) & (j < n)
    if not bool(fluid.any()):
        return pressure, rho
    i, j = i[fluid], j[fluid]
    delta = pos[i] - pos[j]
    from .droplet_impact import pair_kernels  # local import: avoid a cycle

    h = cfg.h_factor * cfg.dx
    wij, _ = pair_kernels(delta, h)

    source = pressure[j]
    if _body_force_term(cfg):
        # g = (0, -gravity); g . (r_i - r_j) = -gravity * dy
        source = source - cfg.gravity * delta[:, 1] * rho[j]
    zeros = torch.zeros(n, dtype=pressure.dtype, device=pressure.device)
    numerator = torch.zeros_like(zeros)
    denominator = torch.zeros_like(zeros)
    numerator = numerator.index_add(0, i, source * wij)
    numerator = numerator.index_add(0, j, (pressure[i]
                                           + (cfg.gravity * delta[:, 1] * rho[i]
                                              if _body_force_term(cfg) else 0.0))
                                   * wij)
    denominator = denominator.index_add(0, i, wij)
    denominator = denominator.index_add(0, j, wij)

    average = numerator / denominator.clamp_min(1.0e-30)
    free = rho < thr * cfg.rho0
    pressure_eff = torch.where(free, average.clamp_min(0.0), pressure)
    if cfg.free_surface_pressure == "shell_density":
        bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
        tait_density = cfg.rho0 * (pressure_eff / bulk + 1.0).pow(
            1.0 / cfg.gamma)
        rho_eff = torch.where(free, tait_density, rho)
    else:
        rho_eff = rho
    return pressure_eff, rho_eff


def flagged_fraction(rho, cfg, threshold: float | None = None) -> float:
    """Fraction of particles the closure treats as free surface (diagnostic)."""
    thr = (cfg.free_surface_density_threshold if threshold is None
           else float(threshold))
    if rho.numel() == 0:
        return 0.0
    return float((rho < thr * cfg.rho0).to(torch.float64).mean())