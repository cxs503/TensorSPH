"""Continuum Surface Force (CSF) model for SPH surface tension.

Implements the CSF model of Brackbill, Kothe & Zemach (1992) adapted for
weakly-compressible SPH. The surface tension force is:

    f_st = sigma * kappa * n * delta_s

where:
    sigma: surface tension coefficient (N/m)
    kappa: curvature (1/m), computed as divergence of the normal field
    n: interface normal = grad(c) / |grad(c)|
    delta_s: surface delta function approximated by |grad(c)|

The color function c is taken as the density ratio rho / rho0, which provides
a smooth indicator of the interface in single-fluid SPH. For multi-phase
extensions, c would be a phase indicator.

References:
    Brackbill, Kothe & Zemach (1992): "A continuum surface force method"
    Monaghan (1992): "Smoothed particle hydrodynamics"
    Tartakovsky et al. (2005): "Modeling of physically adsorbing surfaces"
"""
from __future__ import annotations

import math

import torch

from .droplet_impact import Config, cubic_spline_gradient, cubic_spline_kernel
from .csf_consistent import (  # noqa: F401  (re-exported: public API of this module)
    CSFState,
    apply_contact_angle_boundary,
    csf_fields,
    kernel_sum_color,
    mls_divergence,
    mls_gradient,
    surface_tension_acceleration,
)


def compute_color_gradient(
    pos: torch.Tensor,
    color: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Compute gradient of color function using SPH.

    grad(c)_i = sum_j V_j * (c_j - c_i) * grad_i W_ij,   V_j = m_j / rho_j

    The pair list holds each unordered pair once, so particle ``j`` receives its
    own contribution from the same entry. Written out for ``j`` the term is

        grad(c)_j = V_i * (c_i - c_j) * grad_j W_ji
                  = V_i * (c_j - c_i) * grad_i W_ij

    because ``grad_j W_ji = -grad_i W_ij``; both the volume and the sign belong
    to the partner particle, which is what makes the discrete operator reproduce
    the gradient of a linear field (``tests/test_surface_tension.py``).

    Args:
        pos: (N, 2) particle positions
        color: (N,) color function values
        mass: (N,) particle masses
        density: (N,) particle densities
        pairs: (P, 2) neighbor pair indices
        h: smoothing length

    Returns:
        grad_c: (N, 2) gradient of color function
    """
    n = pos.shape[0]
    grad_c = torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    if pairs.numel() == 0:
        return grad_c

    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(delta, h)

    dc = color[j] - color[i]
    vol = mass / density

    grad_c = grad_c.index_add(0, i, vol[j, None] * dc[:, None] * grad_w)
    grad_c = grad_c.index_add(0, j, vol[i, None] * dc[:, None] * grad_w)

    return grad_c


def compute_normal(
    grad_c: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute interface normal from color gradient.

    n = grad(c) / |grad(c)|

    Args:
        grad_c: (N, 2) gradient of color function
        eps: small value to avoid division by zero

    Returns:
        normal: (N, 2) unit normal vectors (zero where |grad_c| is small)
    """
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    # Only normalize where gradient is significant (interface region)
    mask = mag.squeeze() > eps
    normal = torch.zeros_like(grad_c)
    normal[mask] = grad_c[mask] / mag[mask]
    return normal


def compute_curvature(
    pos: torch.Tensor,
    normal: torch.Tensor,
    mass: torch.Tensor,
    density: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Compute curvature as divergence of normal field.

    kappa = div(n) = sum_j m_j / rho_j * (n_j - n_i) . grad_W_ij

    Note: The sign convention is such that a convex droplet (with outward-pointing
    normal) has positive curvature (div(n) > 0), giving an inward-pointing force
    when combined with f = -sigma * kappa * n.

    Args:
        pos: (N, 2) particle positions
        normal: (N, 2) interface normals (pointing outward)
        mass: (N,) particle masses
        density: (N,) particle densities
        pairs: (P, 2) neighbor pair indices
        h: smoothing length

    Returns:
        kappa: (N,) curvature values (positive for convex surfaces)
    """
    n = pos.shape[0]
    kappa = torch.zeros(n, dtype=pos.dtype, device=pos.device)

    if pairs.numel() == 0:
        return kappa

    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(delta, h)

    # Divergence of normal: (n_j - n_i) . grad_W
    dn = normal[j] - normal[i]
    div_n = (dn * grad_w).sum(dim=1)
    vol_j = mass[j] / density[j]
    vol_i = mass[i] / density[i]

    # Symmetric contribution:
    # For particle i: V_j * (n_j - n_i) . grad_W_ij
    # For particle j: V_i * (n_j - n_i) . grad_W_ij (note: grad_W_ji = -grad_W_ij, but we use the same pair)
    kappa = kappa.index_add(0, i, vol_j * div_n)
    kappa = kappa.index_add(0, j, vol_i * div_n)

    return kappa


def _wall_normal_at(pos: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """Compute the wall normal (pointing into fluid) at each particle.

    For each particle, find the nearest wall particle and use the vector
    from that wall particle to the fluid particle as the wall normal direction.

    Args:
        pos: (N, 2) fluid particle positions
        wall: (M, 2) wall particle positions

    Returns:
        n_w: (N, 2) unit wall normals (zero where wall is empty)
    """
    n = pos.shape[0]
    m = wall.shape[0]
    device = pos.device

    if m == 0:
        return torch.zeros((n, 2), dtype=pos.dtype, device=device)

    # For each fluid particle, find nearest wall particle
    # Use batched distance computation: (N, M, 2) -> (N, M)
    # To avoid O(N*M) memory for large cases, process in chunks
    chunk_size = 4096
    n_w = torch.zeros((n, 2), dtype=pos.dtype, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pos_chunk = pos[start:end]  # (K, 2)

        # Distance from each fluid particle in chunk to each wall particle
        # pos_chunk: (K, 1, 2), wall: (1, M, 2)
        diff = pos_chunk.unsqueeze(1) - wall.unsqueeze(0)  # (K, M, 2)
        dist_sq = (diff ** 2).sum(dim=2)  # (K, M)

        # Find nearest wall particle for each fluid particle
        nearest_idx = dist_sq.argmin(dim=1)  # (K,)

        # Wall normal points from wall to fluid (into fluid)
        nearest_wall = wall[nearest_idx]  # (K, 2)
        normal_vec = pos_chunk - nearest_wall  # (K, 2)
        mag = torch.linalg.vector_norm(normal_vec, dim=1, keepdim=True)
        mask = mag.squeeze() > 1e-12
        n_w[start:end][mask] = normal_vec[mask] / mag[mask]

    return n_w


def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall: torch.Tensor,
    h: float,
    theta: float,
) -> torch.Tensor:
    """Modify the color gradient near walls to enforce a contact angle.

    At a solid wall, the fluid interface should meet the wall at the specified
    contact angle theta. This is achieved by rotating the color gradient so that
    the interface normal makes the correct angle with the wall.

    The procedure:
    1. Find the wall normal n_w (pointing into fluid) for each particle.
    2. For particles within distance h of the wall:
       a. Decompose grad_c into normal (n_w direction) and tangential components.
       b. The desired gradient direction makes angle (pi/2 - theta) with n_w.
          - For theta = pi/2 (neutral): gradient is purely tangential.
          - For theta < pi/2 (hydrophilic): gradient tilts toward the wall.
          - For theta > pi/2 (hydrophobic): gradient tilts away from the wall.
       c. Reconstruct grad_c with the new direction, preserving magnitude.

    Args:
        grad_c: (N, 2) color gradient (will be modified near walls)
        pos: (N, 2) particle positions
        wall: (M, 2) wall particle positions
        h: smoothing length (interaction range)
        theta: contact angle in radians

    Returns:
        grad_c_modified: (N, 2) modified color gradient
    """
    n = pos.shape[0]
    result = grad_c.clone()

    if wall.numel() == 0:
        return result

    # Compute wall normals
    n_w = _wall_normal_at(pos, wall)

    # Find particles within h of the wall
    # Distance to nearest wall particle
    chunk_size = 4096
    near_wall = torch.zeros(n, dtype=torch.bool, device=pos.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pos_chunk = pos[start:end]
        diff = pos_chunk.unsqueeze(1) - wall.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        min_dist = dist_sq.min(dim=1).values
        near_wall[start:end] = min_dist < h

    if not near_wall.any():
        return result

    # For particles near the wall, modify the gradient
    grad_near = result[near_wall]  # (K, 2)
    n_w_near = n_w[near_wall]  # (K, 2)

    # Decompose into normal and tangential components
    # Normal component: (grad_c . n_w) * n_w
    normal_comp = (grad_near * n_w_near).sum(dim=1, keepdim=True) * n_w_near
    # Tangential component: grad_c - normal_comp
    tangential_comp = grad_near - normal_comp

    # Desired direction: the gradient should make angle (pi/2 - theta) with n_w
    # Equivalently, the interface (perpendicular to grad_c) makes angle theta with wall
    #
    # Let alpha = pi/2 - theta be the angle between grad_c and n_w.
    # - alpha = 0 (theta = pi/2): grad_c is purely tangential
    # - alpha > 0 (theta < pi/2, hydrophilic): grad_c tilts toward wall
    # - alpha < 0 (theta > pi/2, hydrophobic): grad_c tilts away from wall
    #
    # The desired gradient direction is:
    #   g_desired = |grad_c| * (cos(alpha) * n_w + sin(alpha) * t_hat)
    # where t_hat is the unit tangential direction.
    #
    # But we want to preserve the tangential component magnitude and adjust
    # the normal component to achieve the correct angle.
    #
    # If |t| is the tangential magnitude and we want angle alpha from n_w:
    #   tan(alpha) = |t| / |n_desired|
    #   |n_desired| = |t| / tan(alpha) = |t| * cot(alpha)
    #
    # For alpha = 0 (theta = pi/2): |n_desired| = 0 (purely tangential)
    # For alpha -> pi/2 (theta -> 0): |n_desired| -> 0... wait, that's wrong.
    #
    # Let me reconsider. The contact angle theta is the angle between the
    # interface tangent and the wall. The gradient is perpendicular to the
    # interface, so the angle between grad_c and the wall normal n_w is
    # alpha = pi/2 - theta.
    #
    # For theta = pi/2 (neutral): alpha = 0, grad_c is parallel to wall (tangential)
    # For theta = 0 (perfect wetting): alpha = pi/2, grad_c is parallel to n_w
    # For theta = pi (perfect non-wetting): alpha = -pi/2, grad_c is anti-parallel to n_w
    #
    # So: grad_c_desired = |grad_c| * (cos(alpha) * t_hat + sin(alpha) * n_w_sign)
    # where t_hat is the unit tangential direction and n_w_sign is +/-n_w.
    #
    # Actually, the simplest approach: preserve tangential component, set normal
    # component to achieve the desired angle.
    #
    # |t| = |grad_c| * cos(alpha)  [tangential magnitude]
    # |n| = |grad_c| * sin(alpha)  [normal magnitude]
    #
    # We want to keep |t| the same and adjust |n|:
    #   |n_new| = |t| * tan(pi/2 - theta) = |t| * cot(theta)
    #
    # Wait, let me be more careful. The angle alpha between grad_c and n_w:
    #   cos(alpha) = (grad_c . n_w) / |grad_c|
    #   sin(alpha) = |tangential| / |grad_c|
    #
    # We want alpha = pi/2 - theta, so:
    #   cos(pi/2 - theta) = sin(theta) = (grad_c . n_w) / |grad_c|
    #   sin(pi/2 - theta) = cos(theta) = |tangential| / |grad_c|
    #
    # So the desired ratio is:
    #   |n_desired| / |t| = sin(theta) / cos(theta) = tan(theta)
    #   |n_desired| = |t| * tan(theta)
    #
    # For theta = pi/2: tan(pi/2) = inf -> |n| should be large... but we want
    # purely tangential. Hmm, the issue is that for theta = pi/2, the gradient
    # should be parallel to the wall, meaning the normal component should be zero.
    #
    # Let me reconsider the geometry. The contact angle theta is measured inside
    # the fluid, between the wall and the interface tangent.
    #
    # If the wall is horizontal (n_w pointing up), and theta = pi/2:
    #   The interface is vertical, so grad_c (perpendicular to interface) is horizontal.
    #   grad_c is purely tangential to the wall. Correct.
    #
    # If theta < pi/2 (hydrophilic, wetting):
    #   The interface curves toward the wall, so grad_c has a component toward the wall.
    #   grad_c has a positive normal component (in the n_w direction).
    #
    # If theta > pi/2 (hydrophobic):
    #   The interface curves away from the wall, so grad_c has a component away from wall.
    #   grad_c has a negative normal component (opposite to n_w).
    #
    # The relationship: the angle between grad_c and the wall tangent is theta.
    # The angle between grad_c and n_w is pi/2 - theta.
    #
    # So: grad_c . n_w = |grad_c| * cos(pi/2 - theta) = |grad_c| * sin(theta)
    # And: |tangential| = |grad_c| * sin(pi/2 - theta) = |grad_c| * cos(theta)
    #
    # We want to preserve |tangential| and set the normal component:
    #   |n_new| = |tangential| * tan(pi/2 - theta) = |tangential| / tan(theta)
    #           = |tangential| * cot(theta)
    #
    # For theta = pi/2: cot(pi/2) = 0 -> |n_new| = 0. Correct!
    # For theta -> 0: cot(0) -> inf -> |n_new| -> inf. This means grad_c aligns with n_w.
    # For theta -> pi: cot(pi) -> -inf -> |n_new| -> -inf. grad_c aligns with -n_w.
    #
    # But we also need to preserve the total magnitude |grad_c|.
    # |grad_c_new|^2 = |n_new|^2 + |t|^2 = |t|^2 * (cot^2(theta) + 1) = |t|^2 / sin^2(theta)
    # |grad_c_new| = |t| / sin(theta) = |grad_c| * cos(theta) / sin(theta) = |grad_c| * cot(theta)
    #
    # Hmm, that changes the magnitude. The tests say magnitude should be preserved.
    #
    # Let me re-read the tests:
    #   test_gradient_magnitude_preserved: "Contact angle enforcement should preserve
    #   gradient magnitude."
    #
    # So we need |grad_c_new| = |grad_c|. Given that:
    #   |grad_c_new|^2 = |n_new|^2 + |t|^2 = |grad_c|^2
    #   |n_new|^2 = |grad_c|^2 - |t|^2
    #
    # And we want the angle to be pi/2 - theta:
    #   cos(pi/2 - theta) = sin(theta) = |n_new| / |grad_c_new| = |n_new| / |grad_c|
    #   |n_new| = |grad_c| * sin(theta)
    #
    # Check: |n_new|^2 + |t|^2 = |grad_c|^2 * sin^2(theta) + |t|^2
    # For this to equal |grad_c|^2:
    #   |t|^2 = |grad_c|^2 * (1 - sin^2(theta)) = |grad_c|^2 * cos^2(theta)
    #   |t| = |grad_c| * |cos(theta)|
    #
    # But |t| is the original tangential magnitude, which may not equal |grad_c| * |cos(theta)|.
    #
    # So we can't simultaneously preserve magnitude AND set the exact angle unless
    # the original gradient already has the right tangential component.
    #
    # The practical approach used in SPH literature (e.g., Adami et al. 2010):
    #   grad_c_new = |grad_c| * (cos(theta) * t_hat + sin(theta) * n_w)
    # where t_hat = t / |t| is the unit tangential direction.
    #
    # This preserves magnitude and sets the correct angle. But it requires |t| > 0.
    # If |t| = 0 (gradient purely normal), we can't determine t_hat, so we leave
    # the gradient unchanged (or use a default tangential direction).

    # Compute magnitudes
    grad_mag = torch.linalg.vector_norm(grad_near, dim=1, keepdim=True)  # (K, 1)
    tang_mag = torch.linalg.vector_norm(tangential_comp, dim=1, keepdim=True)  # (K, 1)

    # Unit tangential direction (where tangential component is non-zero)
    t_hat = torch.zeros_like(tangential_comp)
    t_mask = tang_mag.squeeze() > 1e-12
    t_hat[t_mask] = tangential_comp[t_mask] / tang_mag[t_mask]

    # For particles with negligible tangential component, use a default
    # tangential direction (perpendicular to n_w in 2D)
    # In 2D: if n_w = (nx, ny), then t_default = (-ny, nx)
    if not t_mask.all():
        n_w_no_t = n_w_near[~t_mask]
        t_default = torch.stack([-n_w_no_t[:, 1], n_w_no_t[:, 0]], dim=1)
        t_hat[~t_mask] = t_default

    # Desired gradient: |grad_c| * (cos(theta) * t_hat + sin(theta) * n_w)
    # Note: the sign of the normal component depends on whether we want the
    # gradient to point toward or away from the wall.
    # For hydrophilic (theta < pi/2): gradient should point toward wall (positive n_w component)
    # For hydrophobic (theta > pi/2): gradient should point away from wall (negative n_w component)
    # sin(theta) is positive for theta in (0, pi), so we need to handle the sign.
    #
    # Actually, the standard convention: the contact angle theta is measured
    # inside the fluid. For theta < pi/2, the fluid wets the wall, and the
    # interface curves toward the wall. The color gradient (pointing from low c
    # to high c, i.e., from surface to interior) should have a component toward
    # the wall.
    #
    # The formula grad_c_new = |grad_c| * (cos(theta) * t_hat + sin(theta) * n_w)
    # gives a positive n_w component for all theta in (0, pi), which is correct
    # for the standard SPH color function (c = 1 in interior, c < 1 at surface).

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    grad_new = grad_mag * (cos_theta * t_hat + sin_theta * n_w_near)

    result[near_wall] = grad_new

    return result


def compute_surface_tension_force(
    pos: torch.Tensor,
    density: torch.Tensor,
    mass: torch.Tensor,
    pairs: torch.Tensor,
    cfg: Config,
    wall: torch.Tensor | None = None,
    scheme: str | None = None,
) -> torch.Tensor:
    """Compute surface tension acceleration using the CSF model.

    Two schemes are available through ``cfg.csf_scheme`` (or the ``scheme``
    argument):

    ``"consistent"`` (default)
        kernel-sum colour function + first-order (MLS) consistent gradient and
        divergence, see :mod:`tensorsph.csf_consistent`.  This is the model that
        reproduces the Young-Laplace jump (docs/verification_surface_tension_laplace.md).
    ``"legacy"``
        neighbour-count colour function with the uncorrected SPH operators, kept
        for the A/B comparison and for reproducing pre-m2188 results.

    ``wall`` (particle positions) together with ``cfg.contact_angle`` activates
    the wall contact-angle boundary condition.

    For single-fluid SPH with free surfaces (droplets/bubbles), we detect
    surface particles by counting neighbors. Surface particles have fewer
    neighbors than interior particles.

    The surface tension force is applied to surface particles only:
        f_st = sigma * kappa * n

    where n is the inward normal and kappa is the local curvature.

    Args:
        pos: (N, 2) particle positions
        density: (N,) particle densities
        mass: (N,) particle masses
        pairs: (P, 2) neighbor pair indices
        cfg: simulation configuration
        wall: (M, 2) optional wall particle positions for contact angle BC

    Returns:
        acc_st: (N, 2) surface tension acceleration
    """
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    sigma = cfg.surface_tension_coefficient
    scheme = scheme if scheme is not None else getattr(cfg, "csf_scheme", "consistent")

    if sigma <= 0.0 or pairs.numel() == 0:
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    if scheme not in ("consistent", "legacy"):
        raise ValueError(
            f"unknown csf_scheme {scheme!r}; expected 'consistent' or 'legacy'")

    if scheme == "consistent":
        acc, _ = surface_tension_acceleration(
            pos, density, mass, pairs, h, sigma, rho0=cfg.rho0, wall=wall,
            contact_angle=getattr(cfg, "contact_angle", None))
        return acc

    # Count neighbors for each particle
    i, j = pairs.unbind(dim=1)
    neighbor_count = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    neighbor_count = neighbor_count.index_add(0, i, torch.ones_like(i, dtype=pos.dtype))
    neighbor_count = neighbor_count.index_add(0, j, torch.ones_like(j, dtype=pos.dtype))

    # Detect surface particles: those with fewer neighbors than average
    # Interior particles typically have ~20-30 neighbors in 2D
    # Surface particles have significantly fewer
    avg_neighbors = neighbor_count.mean()
    surface_threshold = 0.7 * avg_neighbors  # 70% of average
    is_surface = neighbor_count < surface_threshold

    if not is_surface.any():
        # No surface particles detected
        return torch.zeros((n, 2), dtype=pos.dtype, device=pos.device)

    # For surface particles, compute a color function based on neighbor count
    # Interior particles: c = 1.0
    # Surface particles: c = neighbor_count / avg_neighbors (0 to 0.7)
    color = torch.ones(n, dtype=pos.dtype, device=pos.device)
    color[is_surface] = neighbor_count[is_surface] / avg_neighbors

    # Compute gradient of color function
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)

    # Apply contact angle boundary condition if wall is provided and contact_angle is set
    if wall is not None and wall.numel() > 0 and cfg.contact_angle is not None:
        grad_c = apply_contact_angle_boundary(grad_c, pos, wall, h, cfg.contact_angle)

    # Compute normal from color gradient
    # Note: color is high (1.0) in interior and low at surface, so grad(c) points inward.
    # For CSF model, we need normal pointing outward (from fluid to vacuum).
    # So we negate the gradient to get outward normal.
    normal = compute_normal(-grad_c)

    # Compute curvature
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)

    # Surface delta function: |grad(c)|
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)

    # CSF force per unit volume: f = sigma * kappa * n * delta_s
    # For a convex surface (droplet) with outward normal, kappa > 0 (divergence of outward normal)
    # The force should point inward (toward center) to contract the droplet.
    # Since n points outward and kappa > 0, f = sigma * kappa * n points outward.
    # But we want inward force! So we need to negate: f = -sigma * kappa * n * delta_s
    # OR: use inward normal (which we already have from grad(c)) and positive kappa.

    # Let's use the convention: force points inward for convex surfaces.
    # With outward normal and kappa = div(n) > 0 for convex surface:
    # f = -sigma * kappa * n points inward (correct!)
    force_mag = -sigma * kappa * delta_s
    acc_st = force_mag[:, None] * normal / density[:, None]

    return acc_st


def capillary_dt_limit(cfg: Config, safety: float = 0.25) -> float:
    """Time step limit from capillary waves.

    The capillary time step is:
        dt_cap = sqrt(rho * h^3 / sigma)

    This is analogous to the Courant limit but for surface tension waves.
    See Brackbill et al. (1992) or Lapinskas et al. (2022).

    Args:
        cfg: simulation configuration
        safety: safety factor (default 0.25)

    Returns:
        dt_cap: maximum stable time step from capillary limit
    """
    if cfg.surface_tension_coefficient <= 0.0:
        return float("inf")
    h = cfg.h_factor * cfg.dx
    rho = cfg.rho0
    sigma = cfg.surface_tension_coefficient
    # Capillary time scale: sqrt(rho * h^3 / sigma)
    dt_cap = safety * (rho * h**3 / sigma) ** 0.5
    return dt_cap
