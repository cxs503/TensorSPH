"""First-order-consistent CSF surface-tension model (kernel-sum colour).

Why this module exists
----------------------
``surface_tension.py`` (m277) implements Brackbill's CSF force

    f = -sigma * kappa * delta_s * n ,      delta_s = |grad c|

but it builds the colour function from a neighbour-count heuristic: ``c = 1``
in the interior and ``c = n_neigh / <n_neigh> ~ 0.7`` on the particles that are
labelled "surface".  The jump of ``c`` across the interface is therefore O(0.3)
instead of O(1) and ``|grad c|`` comes out 3-10x too small: the verification
round of m2174 measured ``delta_s / (1/2h) = 0.09 .. 0.29`` while a correct
delta function must satisfy ``int |grad c| dn = 1`` (i.e. ``|grad c| ~ 1/2h``
across the 2h-thick interface shell).  With delta_s ~ 5x too small the CSF force
is 5x too weak and the Young-Laplace jump ``sigma/R`` was missed by 77-118 %.

The three ingredients are replaced here by standard estimators:

colour function
    ``c_i = sum_j V_j W_ij`` with ``V_j = m_j / rho0``.  This is the smoothed
    indicator of Brackbill et al. (1992): it tends to 1 inside the fluid, to 0
    in the void, and varies monotonically over a shell of thickness ~2h.  Its
    gradient therefore integrates to ``c_in - c_out = 1`` across the interface
    -- exactly the normalisation the CSF delta function needs.  Using
    ``rho0`` (not the instantaneous density) in the volume keeps the colour
    field independent of the free-surface density deficit, which is a separate
    (hydrostatic) error source.

gradient / divergence
    first-order consistent (moving-least-squares) operators, Bonet & Lok (1997)
    / Randles & Libersky (1996):

        grad f_i = M_i^{-T} sum_j V_j (f_j - f_i) grad_i W_ij
        div n_i  = trace( S_i M_i^{-1} ),
        M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij
        S_i = sum_j V_j (n_j - n_i) (x) grad_i W_ij

    The correction matrix ``M_i`` repairs the truncated-kernel bias at a free
    surface, which is the root cause of both the underestimated ``|grad c|``
    and the 3-12x overestimated curvature of the legacy estimator.

curvature and force
    ``kappa = div n`` (outward normal ``n = -grad c/|grad c|``, so kappa > 0 on
    a convex drop) and ``a_i = -sigma kappa_i delta_s_i n_i / rho_i`` applied to
    every particle -- ``delta_s`` localises the force to the interface shell.

References
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335.
    Bonet & Lok (1999), Comput. Methods Appl. Mech. Engrg. 180:97.
    Randles & Libersky (1996), Comput. Methods Appl. Mech. Engrg. 139:375.
    Adami, Ihmsen & Adams (2012), J. Comput. Phys. 231:7057 (spurious currents).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .csf_curvature import fit_curvature
from .droplet_impact import cubic_spline_gradient, cubic_spline_kernel


@dataclass
class CSFState:
    """The fields the CSF force is assembled from (all exposed for scoring)."""

    color: torch.Tensor          # (N,) kernel-sum colour, 1 in fluid, 0 in void
    grad_c: torch.Tensor         # (N, 2) consistent grad(c)
    normal: torch.Tensor         # (N, 2) outward unit normal
    kappa: torch.Tensor          # (N,) curvature = div n
    delta_s: torch.Tensor        # (N,) surface delta function |grad c|

    def surface_mask(self, fraction: float = 0.05) -> torch.Tensor:
        """Particles carrying a non-negligible share of the interface."""
        cut = fraction * float(self.delta_s.max())
        return self.delta_s > cut


def _mls_operators(
    pos: torch.Tensor,
    vol: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MLS correction matrix ``M`` (N,2,2) and a per-particle validity mask.

    ``M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij``.  A pair ``(i, j)`` stored
    once contributes ``V_j (-delta) (x) gw`` to ``M_i`` and ``V_i (-delta) (x) gw``
    to ``M_j`` (``delta = x_i - x_j``, ``gw = grad_i W_ij``), because both the
    volume and the sign belong to the partner particle -- the same identity that
    makes ``compute_color_gradient`` reproduce a linear field.
    """
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    gw = cubic_spline_gradient(delta, h)
    outer = -torch.einsum("pa,pb->pab", delta, gw)
    big_m = torch.zeros((n, 2, 2), dtype=pos.dtype, device=pos.device)
    big_m = big_m.index_add(0, i, vol[j, None, None] * outer)
    big_m = big_m.index_add(0, j, vol[i, None, None] * outer)

    # Lattice defects (an interior particle with < 3 neighbours) make M
    # singular; those fall back to the uncorrected operator.
    det = big_m[:, 0, 0] * big_m[:, 1, 1] - big_m[:, 0, 1] * big_m[:, 1, 0]
    scale = det.abs().median().clamp_min(1e-30)
    ok = det.abs() > 1e-4 * scale
    eye = torch.eye(2, dtype=pos.dtype, device=pos.device)
    big_m = torch.where(ok[:, None, None], big_m, eye.expand_as(big_m))
    return big_m, ok


def mls_gradient(
    pos: torch.Tensor,
    field: torch.Tensor,
    vol: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """First-order consistent ``grad(field)`` with volume weights ``V_j``."""
    n = pos.shape[0]
    if pairs.numel() == 0:
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    gw = cubic_spline_gradient(delta, h)
    df = field[j] - field[i]
    contrib = df[:, None] * gw
    rhs = torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)
    rhs = rhs.index_add(0, i, vol[j, None] * contrib)
    rhs = rhs.index_add(0, j, vol[i, None] * contrib)

    big_m, _ = _mls_operators(pos, vol, pairs, h)
    return torch.linalg.solve(big_m.transpose(-1, -2), rhs.unsqueeze(-1)).squeeze(-1)


def mls_divergence(
    pos: torch.Tensor,
    field: torch.Tensor,
    vol: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """First-order consistent ``div(field)`` with volume weights ``V_j``."""
    n = pos.shape[0]
    if pairs.numel() == 0:
        return torch.zeros((n,), dtype=pos.dtype, device=pos.device)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    gw = cubic_spline_gradient(delta, h)
    df = field[j] - field[i]
    contrib = torch.einsum("pa,pb->pab", df, gw)
    s = torch.zeros((n, 2, 2), dtype=pos.dtype, device=pos.device)
    s = s.index_add(0, i, vol[j, None, None] * contrib)
    s = s.index_add(0, j, vol[i, None, None] * contrib)

    big_m, _ = _mls_operators(pos, vol, pairs, h)
    m_inv = torch.linalg.inv(big_m)
    return (s * m_inv.transpose(-1, -2)).sum(dim=(-2, -1))


def kernel_sum_color(
    pos: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
    rho0: float,
) -> torch.Tensor:
    """``c_i = sum_j V_j W_ij`` with ``V_j = m_j/rho0``, clipped to [0, 1]."""
    vol = mass / rho0
    # The particle's own volume is part of the discrete kernel sum; the pair
    # list never contains self-pairs, so it is added explicitly. On a hex
    # lattice with h = 1.3 dx this term is 0.233 of the total 1.006, i.e. the
    # colour function would saturate at 0.77 (and |grad c| would be 23% low)
    # without it.
    c = vol * cubic_spline_kernel(torch.zeros((), dtype=pos.dtype,
                                              device=pos.device), h)
    if pairs.numel() == 0:
        return c.clamp(0.0, 1.0)
    i, j = pairs.unbind(dim=1)
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)
    c = c.index_add(0, i, vol[j] * w)
    c = c.index_add(0, j, vol[i] * w)
    return c.clamp(0.0, 1.0)


def shepard_smooth(
    field: torch.Tensor,
    pos: torch.Tensor,
    vol: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
    passes: int = 1,
) -> torch.Tensor:
    """Kernel (Shepard) smoothing of a particle field, ``passes`` times.

    Used on the interface normal before the divergence: differentiating a noisy
    unit vector amplifies its angular noise, and the curvature came out 30-40 %
    too high without this.  For a smooth radial field the smoothing leaves the
    direction untouched (the kernel is isotropic), so it removes noise, not
    signal.
    """
    if passes <= 0 or pairs.numel() == 0:
        return field
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel(torch.linalg.vector_norm(pos[i] - pos[j], dim=1), h)
    out = field
    for _ in range(passes):
        num = out.new_zeros(out.shape)
        den = vol * cubic_spline_kernel(out.new_zeros(()), h)
        num = num.index_add(0, i, (vol[j] * w)[:, None] * out[j])
        den = den.index_add(0, i, vol[j] * w)
        num = num.index_add(0, j, (vol[i] * w)[:, None] * out[i])
        den = den.index_add(0, j, vol[i] * w)
        out = num / den.clamp_min(1e-30)[:, None]
    return out


def csf_fields(
    pos: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
    rho0: float,
    wall: torch.Tensor | None = None,
    contact_angle: float | None = None,
    normal_smoothing: int = 2,
    curvature_scheme: str = "mls",
    curvature_stencil: float = 4.0,
    curvature_color_smoothing: int = 0,
) -> CSFState:
    """Colour, its gradient, the outward normal, curvature and delta_s.

    ``contact_angle`` (radians) activates the wall BC of
    :func:`apply_contact_angle_boundary`; it needs ``wall`` particle positions.

    ``curvature_scheme`` selects the curvature operator:

    ``"mls"`` (default, historical)
        ``kappa = div n`` with MLS-consistent operators and a Shepard-smoothed
        normal.  Measured error on a circle: +25 %
        (``docs/verification_surface_tension_consistent.md``).
    ``"fit_quadratic"`` / ``"fit_circle"``
        weighted local interface fit, see :mod:`tensorsph.csf_curvature`; the
        MLS value is kept as the fallback where the stencil is too poor.

    ``curvature_color_smoothing`` (opt-in, 1) computes the curvature from a
    Shepard-smoothed copy of the colour field while the force keeps using the
    sharp field.  ``kappa = div n`` amplifies the angular discretisation error of
    the normal by ``~1/dx``, and on a one-sided free-surface support that error
    is concentrated in the inner shell, where ``c`` approaches 1 and the normal
    is barely defined: those particles carried 12-24 % of the force but a kappa
    of +95..+183 % (see ``benchmarks/csf_consistent/curvature_audit.py`` and
    ``docs/verification_csf_curvature_bias.md``).  One smoothing pass on ``c``
    removes that noise at the source and brings the delta_s-weighted curvature
    error from +22..+25 % to +7..+14 % for dx <= 2 mm.  The default 0 keeps
    the m2447 behaviour (with 1, the cut-force identity on the coarse
    verification lattice degrades below the m2447 gate); 1 enables the fix.
    """
    vol = mass / rho0
    color = kernel_sum_color(pos, mass, pairs, h, rho0)
    grad_c = mls_gradient(pos, color, vol, pairs, h)
    if wall is not None and contact_angle is not None:
        grad_c = apply_contact_angle_boundary(grad_c, pos, wall, h, contact_angle)

    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    # c decreases outward, so the outward normal is minus the gradient.
    tiny = 1e-12
    normal = torch.where(mag > tiny, -grad_c / mag.clamp_min(tiny),
                         torch.zeros_like(grad_c))
    if normal_smoothing:
        smoothed = shepard_smooth(normal, pos, vol, pairs, h, passes=normal_smoothing)
        norm = torch.linalg.vector_norm(smoothed, dim=1, keepdim=True)
        # Smoothing the interface only: interior particles keep their zero normal
        # (their colour field is flat, so they carry no surface information).
        normal = torch.where(mag > tiny, smoothed / norm.clamp_min(tiny), normal)

    if curvature_color_smoothing > 0:
        # Curvature from a smoothed colour field (force/delta_s stay sharp).
        c_s = shepard_smooth(color[:, None], pos, vol, pairs, h,
                             passes=curvature_color_smoothing)[:, 0].clamp(0.0, 1.0)
        grad_s = mls_gradient(pos, c_s, vol, pairs, h)
        if wall is not None and contact_angle is not None:
            grad_s = apply_contact_angle_boundary(grad_s, pos, wall, h, contact_angle)
        mag_s = torch.linalg.vector_norm(grad_s, dim=1, keepdim=True)
        normal_s = torch.where(mag_s > tiny, -grad_s / mag_s.clamp_min(tiny),
                               torch.zeros_like(grad_s))
        if normal_smoothing:
            smooth_s = shepard_smooth(normal_s, pos, vol, pairs, h,
                                      passes=normal_smoothing)
            normal_s = torch.where(
                mag_s > tiny,
                smooth_s / torch.linalg.vector_norm(smooth_s, dim=1, keepdim=True).clamp_min(tiny),
                normal_s)
        kappa = mls_divergence(pos, normal_s, vol, pairs, h)
    else:
        kappa = mls_divergence(pos, normal, vol, pairs, h)

    # Surface delta function.  The definition that makes the CSF force
    # mesh-consistent (Brackbill et al. 1992) is the one that satisfies
    #
    #     int_{interface} delta_s dn = 1 .
    #
    # In a single-phase cloud the colour function only exists inside the fluid:
    # it falls from c = 1 in the bulk to c = 1/2 on the outermost particle layer
    # (that layer sees half a kernel), and the remaining 1 -> 0 half of the
    # indicator lives in the void, where there are no particles to carry it.
    # Hence
    #
    #     int_fluid |grad c| dn = c_bulk - c_interface = 1 - 1/2 = 1/2 ,
    #
    # so the delta function of the two-phase model is twice the in-fluid
    # gradient magnitude.  Dropping this factor of 2 (or, as the legacy scheme
    # did, letting the colour jump be O(0.3) instead of O(1)) makes the CSF
    # force -- and therefore the Laplace pressure jump -- too weak by exactly
    # the missing factor.
    delta_s = 2.0 * torch.linalg.vector_norm(grad_c, dim=1)
    if curvature_scheme != "mls":
        kappa = fit_curvature(pos, normal, delta_s, pairs, h, fallback=kappa,
                              method=curvature_scheme, stencil=curvature_stencil)

    return CSFState(color=color, grad_c=grad_c, normal=normal, kappa=kappa,
                    delta_s=delta_s)


def surface_tension_acceleration(
    pos: torch.Tensor,
    density: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
    sigma: float,
    rho0: float | None = None,
    wall: torch.Tensor | None = None,
    contact_angle: float | None = None,
    curvature_scheme: str = "mls",
    curvature_stencil: float = 4.0,
    delta_scale: float = 1.0,
) -> tuple[torch.Tensor, CSFState | None]:
    """CSF acceleration ``a = -sigma kappa delta_s n / rho`` and its fields."""
    if sigma <= 0.0:
        return torch.zeros_like(pos), None
    rho0 = float(rho0 if rho0 is not None else density.mean())
    if curvature_scheme != "mls" and pairs.numel() > 0:
        # The interface fit needs an arc, not a kernel support: the force loop
        # otherwise hands us 2h pairs, which on a lattice is 3-4 staircase
        # points per particle.  Widening costs one extra neighbour search per
        # step and is only paid when the fit operator is selected.
        from .neighbor_search import cell_linked_pairs
        pairs = cell_linked_pairs(pos, curvature_stencil * h)
    state = csf_fields(pos, mass, pairs, h, rho0, wall=wall,
                       contact_angle=contact_angle,
                       curvature_scheme=curvature_scheme,
                       curvature_stencil=curvature_stencil)
    # Particles whose colour field is locally flat are not on the interface; the
    # clamped kernel sum makes their |grad c| numerically ~1e-12 rather than
    # exactly zero, so they are masked out. Bulk traction stays a bit-exact
    # no-op (tests/test_laplace_pressure.py::test_bulk_traction_is_exactly_zero).
    d_s = state.delta_s.clone()
    flat = d_s <= 1.0e-3 * d_s.max()
    d_s[flat] = 0.0
    state.delta_s = d_s
    force = (-sigma * delta_scale * state.kappa * state.delta_s)[:, None] * state.normal
    return force / density[:, None], state

# --------------------------------------------------------------------------- #
# wall contact-angle boundary condition
# --------------------------------------------------------------------------- #
def nearest_wall(
    pos: torch.Tensor,
    wall_pos: torch.Tensor,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance to the nearest wall particle and the wall normal (into fluid).

    The wall normal is approximated by the direction from the nearest wall
    particle to the fluid particle, which is exact for a lattice wall and is the
    usual choice for contact-line BCs on particle walls.
    """
    n = pos.shape[0]
    dmin = torch.full((n,), float("inf"), dtype=pos.dtype, device=pos.device)
    nearest = torch.zeros_like(pos)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(pos[start:stop], wall_pos)
        val, idx = d.min(dim=1)
        dmin[start:stop] = val
        nearest[start:stop] = wall_pos[idx]
    normal = pos - nearest
    mag = torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(1e-30)
    # A particle sitting on the wall has no well-defined direction; fall back to
    # the bulk wall orientation, estimated from every near-coincident particle.
    degenerate = mag.squeeze(-1) < 1e-12
    if bool(degenerate.any()):
        mean_dir = normal[~degenerate].mean(dim=0) if bool((~degenerate).any()) \
            else torch.zeros(pos.shape[1], dtype=pos.dtype, device=pos.device)
        fallback = mean_dir / mean_dir.norm().clamp_min(1e-30)
        normal = torch.where(degenerate[:, None], fallback.expand_as(normal), normal)
        mag = torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(1e-30)
    return dmin, normal / mag


def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall_pos: torch.Tensor,
    h: float,
    theta: float,
    reach: float = 2.0,
) -> torch.Tensor:
    """Enforce Young's contact angle on the colour gradient near a wall.

    For a fluid particle within ``reach * h`` of the wall, the interface normal
    ``n = -grad(c)/|grad(c)|`` is rotated so that it satisfies

        n . n_wall = cos(theta)

    with ``n_wall`` the wall normal pointing into the fluid (Adami et al. 2012;
    the "interface normal" formulation of the contact-angle BC).  The rotation
    preserves ``|grad(c)|`` -- the CSF force amplitude is ``sigma kappa |grad c|``,
    so only the direction may be touched here -- and is blended linearly with
    the distance to the wall (full strength at ``d = 0``, no change at
    ``d = reach * h``), which keeps the field continuous.

    Args:
        grad_c: (N, dim) colour gradient
        pos: (N, dim) fluid particle positions
        wall_pos: (M, dim) wall particle positions (may be empty)
        h: smoothing length
        theta: contact angle in radians measured inside the liquid
            (pi/2 = neutral, 0 = perfectly wetting, pi = perfectly non-wetting)
        reach: BC thickness in units of h

    Returns:
        The gradient with the near-wall normals rotated; untouched elsewhere.
    """
    out = grad_c.clone()
    if wall_pos is None or wall_pos.numel() == 0 or grad_c.numel() == 0:
        return out

    dim = pos.shape[1]
    d, n_wall = nearest_wall(pos, wall_pos)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    active = (d < reach * h) & (mag.squeeze(-1) > 0.0)
    if not bool(active.any()):
        return out

    n = -grad_c[active] / mag[active]
    w_n = n_wall[active]

    # tangential direction of the current normal (perpendicular to the wall)
    tangent = n - (n * w_n).sum(dim=1, keepdim=True) * w_n
    t_mag = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    if dim == 2:
        perp = torch.stack([-w_n[:, 1], w_n[:, 0]], dim=1)
        # keep the side of the wall the interface is currently on
        sgn = torch.where((tangent * perp).sum(dim=1, keepdim=True) < 0.0,
                          -torch.ones_like(t_mag), torch.ones_like(t_mag))
        tangent = perp * sgn
    else:
        tiny = t_mag < 1e-12
        fallback = torch.zeros_like(tangent)
        fallback[:, 0] = 1.0
        fallback = fallback - (fallback * w_n).sum(dim=1, keepdim=True) * w_n
        tangent = torch.where(tiny, fallback, tangent)
        t_mag = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
        tangent = tangent / t_mag.clamp_min(1e-30)

    target = torch.cos(torch.as_tensor(theta, dtype=pos.dtype,
                                       device=pos.device)) * w_n \
        + torch.sin(torch.as_tensor(theta, dtype=pos.dtype, device=pos.device)) * tangent

    blend = (1.0 - d[active] / (reach * h)).clamp(0.0, 1.0)[:, None]
    mixed = (1.0 - blend) * n + blend * target
    mixed = mixed / torch.linalg.vector_norm(mixed, dim=1, keepdim=True).clamp_min(1e-30)

    out[active] = -mag[active] * mixed
    return out
