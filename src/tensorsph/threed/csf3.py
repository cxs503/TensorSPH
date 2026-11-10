"""Three-dimensional CSF surface tension for the periodic lattice solver.

Why this module exists
----------------------
The shipped CSF model (``tensorsph.csf_consistent``) is written for a 2-D
planar particle cloud: the MLS correction matrix is hard-coded ``2 x 2``,
``rhs`` refuses any cloud whose second axis is not 2, and the 3-D modules
(``tensorsph.threed``) carry no surface tension at all.  A Rayleigh-Plateau
validation -- the capillary breakup of a liquid *cylinder* -- cannot be posed in
the plane: a 2-D strip is linearly *stable* under capillary forces (its surface
energy ``2 sigma int sqrt(1 + eta'^2) dx`` is minimised by the flat strip and
there is no azimuthal curvature to drive the instability), which is exactly why
Rayleigh (1878) needed the ``1/R`` azimuthal term.  So the anchor needs a real
3-D implementation, and this module is it.

Formulation (Brackbill, Kothe & Zemach 1992, in D = 3)
-----------------------------------------------------
    c_i        = sum_j V_j W_ij                     colour, V_j = m_j / rho0
    grad c_i   = M_i^{-1} sum_j V_j (c_j - c_i) grad_i W_ij      (MLS, D x D)
    n_i        = -grad c_i / |grad c_i|             outward unit normal
    kappa_i    = div n_i = trace(S_i M_i^{-1}),
                 S_i = sum_j V_j (n_j - n_i) (x) grad_i W_ij
    delta_s_i  = |grad c_i|                         surface delta function
    f_i        = -sigma kappa_i delta_s_i n_i / rho_i

The colour is built from the *rest* volume ``m_j / rho0`` so the field is
normalised independently of the free-surface density deficit:
``c -> 1`` in the fluid, ``c -> 0`` in the void, and
``int |grad c| dn = c_in - c_out = 1`` across the interface shell, which is the
normalisation the CSF delta function requires.  The same construction works for
any kernel normalisation constant, so it composes with ``kernel3``'s spline
without a dimension-dependent rescaling.

The MLS matrix ``M_i`` repairs the truncated-support bias of a free surface --
the same operator, in 3-D, that the 2-D module documents in
``docs/verification_surface_tension_consistent.md``.  Degenerate stencils (an
interior particle with fewer than D+1 neighbours, which happens on lattice
defects and on the axis of a thin neck) fall back to the identity so the
operator never produces NaNs.

Periodicity: every operator takes the same optional ``box`` as
``tensorsph.threed.neighbors3.periodic_pairs``, so a liquid column that is
periodic along its axis (the Rayleigh-Plateau ansatz) uses minimum-image
separations and minimum-image kernels throughout.

References
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335.
    Bonet & Lok (1999), Comput. Methods Appl. Mech. Engrg. 180:97 (MLS).
    Popinet & Zaleski (1999), Int. J. Numer. Methods Fluids 30:775
        (CSF on a cylinder; the ``1/R`` azimuthal curvature term).
"""
from __future__ import annotations

import torch

from ..csf_curvature import fit_curvature
from .kernel3 import cubic_spline_kernel
from .operators3 import pair_geometry

__all__ = ["csf_fields3", "surface_tension_acceleration3", "CSFState3",
           "curvature_axisymmetric", "CURVATURE_SCHEMES"]

#: ``"mls"`` = 3-D ``div n`` (fully general); ``"axisym"`` = meridional fit plus
#: the exact ``n_r / r`` azimuthal term (valid for a surface of revolution, which
#: is the Rayleigh-Plateau setting).
CURVATURE_SCHEMES = ("mls", "axisym")


class CSFState3:
    """The fields a 3-D CSF force is assembled from (all exposed for scoring)."""

    __slots__ = ("color", "grad_c", "normal", "kappa", "delta_s", "valid")

    def __init__(self, color, grad_c, normal, kappa, delta_s, valid):
        self.color = color
        self.grad_c = grad_c
        self.normal = normal
        self.kappa = kappa
        self.delta_s = delta_s
        self.valid = valid

    def surface_mask(self, fraction: float = 0.05) -> torch.Tensor:
        """Particles carrying a non-negligible share of the interface."""
        return self.delta_s > fraction * float(self.delta_s.max())

    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "grad_c": self.grad_c,
            "normal": self.normal,
            "kappa": self.kappa,
            "delta_s": self.delta_s,
            "valid": self.valid,
        }


def _mls_matrix(pos, vol, delta, grad, i, j, dim: int):
    """``M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij`` (pair stored once)."""
    n = pos.shape[0]
    outer = -torch.einsum("pa,pb->pab", delta, grad)
    big_m = torch.zeros((n, dim, dim), dtype=pos.dtype, device=pos.device)
    big_m = big_m.index_add(0, i, vol[j, None, None] * outer)
    big_m = big_m.index_add(0, j, vol[i, None, None] * outer)

    det = torch.linalg.det(big_m)
    scale = det.abs().median().clamp_min(1e-30)
    ok = det.abs() > 1e-6 * scale
    eye = torch.eye(dim, dtype=pos.dtype, device=pos.device).expand_as(big_m)
    big_m = torch.where(ok[:, None, None], big_m, eye)
    return big_m, ok


def csf_fields3(pos: torch.Tensor, mass: torch.Tensor, h: float,
                pairs: torch.Tensor, *, rho0: float, box=None,
                volume: torch.Tensor | None = None,
                curvature: str = "mls") -> CSFState3:
    """Colour, normal, curvature and delta function of a 3-D particle cloud.

    ``volume`` overrides the rest volume ``m / rho0`` used to weight the kernel
    sums (the 2-D module offers the same knob through ``csf_fields``); the
    default is the rest volume, which is what keeps the colour field independent
    of the instantaneous density.
    """
    dim = pos.shape[1]
    vol = mass / rho0 if volume is None else volume
    i, j = pairs[:, 0], pairs[:, 1]
    delta, weight, grad, _ = pair_geometry(pos, pairs, h, box)

    self_weight = float(cubic_spline_kernel(
        torch.zeros(1, dtype=pos.dtype, device=pos.device), h)[0])

    color = torch.zeros_like(mass)
    color = color.index_add(0, i, vol[j] * weight)
    color = color.index_add(0, j, vol[i] * weight)
    color = color + vol * self_weight

    big_m, valid = _mls_matrix(pos, vol, delta, grad, i, j, dim)

    dc = (color[j] - color[i])
    num = torch.zeros((pos.shape[0], dim), dtype=pos.dtype, device=pos.device)
    num = num.index_add(0, i, (vol[j] * dc).unsqueeze(-1) * grad)
    num = num.index_add(0, j, (vol[i] * dc).unsqueeze(-1) * grad)
    grad_c = torch.linalg.solve(big_m, num.unsqueeze(-1)).squeeze(-1)

    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    normal = -grad_c / delta_s.clamp_min(1.0e-30).unsqueeze(-1)

    dn = normal[j] - normal[i]
    outer_n = torch.einsum("pa,pb->pab", dn, grad)
    s_mat = torch.zeros((pos.shape[0], dim, dim), dtype=pos.dtype,
                        device=pos.device)
    s_mat = s_mat.index_add(0, i, vol[j, None, None] * outer_n)
    s_mat = s_mat.index_add(0, j, vol[i, None, None] * outer_n)
    # kappa = trace(S M^{-1}) = sum_{a,b} S[a,b] M^{-1}[b,a]
    big_m_inv = torch.linalg.inv(big_m)
    kappa = (s_mat * big_m_inv.transpose(1, 2)).sum(dim=(1, 2))
    kappa = torch.where(valid, kappa, torch.zeros_like(kappa))

    if curvature == "axisym":
        kappa = curvature_axisymmetric(pos, normal, delta_s, pairs, h,
                                       fallback=kappa)
    elif curvature != "mls":
        raise ValueError(f"unknown curvature {curvature!r}; expected {CURVATURE_SCHEMES}")

    return CSFState3(color, grad_c, normal, kappa, delta_s, valid)


def curvature_axisymmetric(pos: torch.Tensor, normal: torch.Tensor,
                           delta_s: torch.Tensor, pairs: torch.Tensor, h: float,
                           *, fallback: torch.Tensor | None = None,
                           method: str = "fit_quadratic") -> torch.Tensor:
    """Curvature of a *surface of revolution* about the ``z`` axis.

    For an axisymmetric surface the divergence of the unit normal splits
    exactly into a meridional part and an azimuthal part::

        kappa = div n = dn_r/dr + dn_z/dz + n_r / r
                      = kappa_meridional + n_r / r

    The azimuthal term ``n_r / r`` is evaluated analytically (it carries the
    ``1/R`` that drives Rayleigh-Plateau), and only the meridional curvature is
    discretised -- by the same weighted quadratic interface fit that the 2-D
    module validated (``tensorsph.csf_curvature.fit_curvature``), applied to the
    projection ``(r, z)`` of the particles and of their normals.  This is the
    axisymmetric CSF of Brackbill et al. (1992, section 4) and Popinet & Zaleski
    (1999); it is exact for a cylinder (``kappa_meridional = 0``,
    ``kappa = 1/R``) and for a sphere (``kappa_meridional = 1/R``,
    ``kappa -> 2/R``), whereas the raw 3-D ``div n`` on a lattice staircase pays
    an ``O(1)`` truncation error at the resolutions a jet case can afford.

    ``pairs`` must reach ``4 h`` (the validated fit stencil), not the kernel
    support ``2 h``.  ``fallback`` (usually the 3-D ``div n``) is returned where
    the fit stencil is degenerate, e.g. on the axis ``r -> 0`` or at a pinch.

    **Known limitation (measured).**  The quadratic graph fit is designed for a
    one-dimensional interface *curve*.  Projecting a 3-D particle shell onto
    ``(r, z)`` does not produce a curve but a two-dimensional band (every
    azimuth, and the whole ~2 h shell thickness, collapse onto the same
    meridional footprint), and the fit then reports a spurious meridional
    curvature: on a static cylinder at ``R/dx = 4`` it returns
    ``kappa_meridional = -1.35/R`` against the analytic ``0``, i.e.
    ``kappa R = -0.35`` instead of ``+1`` (sign reversal).  The estimator is kept
    as the A/B arm that documents this, and the column case uses the
    profile-based ``"profile"`` scheme instead (``kappa R = 1.000`` on the same
    cylinder).  A correct implementation would fit the surface in 3-D, or fit
    the meridional *curve* extracted per azimuthal sector.
    """
    if fallback is None:
        fallback = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
    r = torch.linalg.vector_norm(pos[:, :2], dim=1)
    meridional = torch.stack((r, pos[:, 2]), dim=1)
    radial = pos[:, :2] / r.clamp_min(1e-30).unsqueeze(-1)
    n_r = (normal[:, :2] * radial).sum(dim=1)
    n_z = normal[:, 2]
    m_norm = torch.linalg.vector_norm(torch.stack((n_r, n_z), dim=1), dim=1)
    m_normal = torch.stack((n_r, n_z), dim=1) / m_norm.clamp_min(1e-30).unsqueeze(-1)
    k_meridional = fit_curvature(meridional, m_normal, delta_s, pairs, h,
                                 fallback=torch.zeros_like(r), method=method)
    azimuthal = n_r / r.clamp_min(0.25 * h)
    return k_meridional + azimuthal


def surface_tension_acceleration3(csf: CSFState3, rho: torch.Tensor,
                                  sigma: float,
                                  cutoff: float = 1.0e-3) -> torch.Tensor:
    """``a_i = -sigma kappa_i delta_s_i n_i / rho_i`` (force per unit mass).

    ``cutoff`` drops the interface weight of particles whose colour gradient is
    a lattice artefact (``|grad c| < cutoff * max|grad c|``), which keeps
    isolated spurious particles from carrying a singular normal.
    """
    if sigma == 0.0:
        return torch.zeros_like(csf.normal)
    weight = csf.delta_s
    limit = cutoff * float(weight.max()) if weight.numel() else 0.0
    active = (weight > limit).to(weight.dtype)
    scale = -(sigma * csf.kappa * weight * active / rho)
    return scale.unsqueeze(-1) * csf.normal
