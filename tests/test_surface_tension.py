"""Tests for surface tension (CSF model) implementation."""
import torch
import pytest
from tensorsph import (
    Config,
    compute_color_gradient,
    compute_curvature,
    compute_normal,
    compute_surface_tension_force,
    capillary_dt_limit,
    cubic_spline_gradient,
)
from tensorsph.neighbor_search import cell_linked_pairs


def test_capillary_dt_limit_zero_sigma():
    """Capillary time step should be infinite when sigma=0."""
    cfg = Config(surface_tension_coefficient=0.0)
    dt_cap = capillary_dt_limit(cfg)
    assert dt_cap == float("inf")


def test_capillary_dt_limit_positive_sigma():
    """Capillary time step should be finite and positive when sigma>0."""
    cfg = Config(surface_tension_coefficient=0.072)  # water-air
    dt_cap = capillary_dt_limit(cfg)
    assert dt_cap > 0
    assert dt_cap < float("inf")
    # Check formula: dt_cap = safety * sqrt(rho * h^3 / sigma)
    h = cfg.h_factor * cfg.dx
    expected = 0.25 * (cfg.rho0 * h**3 / cfg.surface_tension_coefficient) ** 0.5
    assert abs(dt_cap - expected) < 1e-10


def test_compute_color_gradient_uniform():
    """Gradient of uniform color field should be zero."""
    n = 100
    pos = torch.rand(n, 2, dtype=torch.float64) * 0.1
    color = torch.ones(n, dtype=torch.float64)
    mass = torch.ones(n, dtype=torch.float64) * 0.001
    density = torch.ones(n, dtype=torch.float64) * 1000.0
    
    h = 0.02
    pairs = cell_linked_pairs(pos, 2.0 * h)
    
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    
    # Gradient should be near zero for uniform field
    assert torch.allclose(grad_c, torch.zeros_like(grad_c), atol=1e-10)


def test_compute_normal_magnitude():
    """Normal vectors should be unit length where gradient is significant."""
    n = 50
    # Create a gradient field
    pos = torch.rand(n, 2, dtype=torch.float64) * 0.1
    grad_c = torch.randn(n, 2, dtype=torch.float64)
    # Make some gradients small (below threshold)
    grad_c[0] = 1e-10 * torch.ones(2)
    
    normal = compute_normal(grad_c)
    
    # Check unit length where gradient is significant
    mask = torch.linalg.vector_norm(grad_c, dim=1) > 1e-8
    if mask.any():
        magnitudes = torch.linalg.vector_norm(normal[mask], dim=1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-10)
    
    # Check zero where gradient is small
    assert torch.allclose(normal[~mask], torch.zeros_like(normal[~mask]))


def test_compute_curvature_flat_interface():
    """Curvature of flat interface should be near zero."""
    # Create a flat interface: particles in a line
    n = 20
    x = torch.linspace(0, 0.1, n, dtype=torch.float64)
    pos = torch.stack([x, torch.zeros_like(x)], dim=1)
    
    # Normal pointing up everywhere (flat interface)
    normal = torch.zeros(n, 2, dtype=torch.float64)
    normal[:, 1] = 1.0
    
    mass = torch.ones(n, dtype=torch.float64) * 0.001
    density = torch.ones(n, dtype=torch.float64) * 1000.0
    
    h = 0.02
    pairs = cell_linked_pairs(pos, 2.0 * h)
    
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)
    
    # Curvature should be near zero for flat interface
    # (except at boundaries where SPH is incomplete)
    interior = (n // 4, 3 * n // 4)
    assert torch.allclose(kappa[interior[0]:interior[1]], 
                         torch.zeros_like(kappa[interior[0]:interior[1]]), 
                         atol=1.0)


def test_surface_tension_force_zero_sigma():
    """Surface tension force should be zero when sigma=0."""
    n = 50
    pos = torch.rand(n, 2, dtype=torch.float64) * 0.1
    density = torch.ones(n, dtype=torch.float64) * 1000.0
    mass = torch.ones(n, dtype=torch.float64) * 0.001
    
    h = 0.02
    pairs = cell_linked_pairs(pos, 2.0 * h)
    
    cfg = Config(surface_tension_coefficient=0.0)
    acc_st = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    
    assert torch.allclose(acc_st, torch.zeros_like(acc_st))


def test_surface_tension_force_shape():
    """Surface tension force should have correct shape."""
    n = 50
    pos = torch.rand(n, 2, dtype=torch.float64) * 0.1
    density = torch.ones(n, dtype=torch.float64) * 1000.0
    mass = torch.ones(n, dtype=torch.float64) * 0.001
    
    h = 0.02
    pairs = cell_linked_pairs(pos, 2.0 * h)
    
    cfg = Config(surface_tension_coefficient=0.072)
    acc_st = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    
    assert acc_st.shape == (n, 2)
    assert acc_st.dtype == torch.float64
    assert acc_st.device == pos.device


def test_surface_tension_produces_force():
    """Surface tension should produce non-zero force when there's a surface."""
    # Create a droplet-like configuration
    n = 200
    pos = torch.rand(n, 2, dtype=torch.float64) * 0.3 + 0.05
    mass = torch.ones(n, dtype=torch.float64) * 0.001
    vel = torch.zeros(n, 2, dtype=torch.float64)
    
    cfg = Config(surface_tension_coefficient=0.072, gravity=0.0, dx=0.01)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    
    # Call rhs to get density
    from tensorsph import rhs
    rho, p, acc = rhs(pos, vel, mass, cfg, pairs=pairs)
    
    # Surface tension should produce some force (even if small)
    # because there are surface particles (boundary of the droplet)
    assert acc.shape == (n, 2)
    # At least some particles should have non-zero acceleration
    # (surface particles will have surface tension force)
    assert (acc.norm(dim=1) > 1e-10).any()


def test_config_has_surface_tension_coefficient():
    """Config should have surface_tension_coefficient field."""
    cfg = Config()
    assert hasattr(cfg, "surface_tension_coefficient")
    assert cfg.surface_tension_coefficient == 0.0  # default disabled
    
    cfg2 = Config(surface_tension_coefficient=0.072)
    assert cfg2.surface_tension_coefficient == 0.072


def test_stable_dt_includes_capillary():
    """stable_dt should consider capillary limit when sigma>0."""
    cfg_no_st = Config(surface_tension_coefficient=0.0)
    cfg_with_st = Config(surface_tension_coefficient=0.072)
    
    from tensorsph import stable_dt
    
    dt_no_st = stable_dt(cfg_no_st)
    dt_with_st = stable_dt(cfg_with_st)
    
    # With surface tension, dt should be smaller (more restrictive)
    assert dt_with_st <= dt_no_st


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --------------------------------------------------------------------------- #
# Analytic reference tests for the colour-gradient / normal / curvature chain
#
# The CSF force is sigma * kappa * n * |grad c| / rho, so each stage has an
# exact answer that does not depend on the surface-tension coefficient:
#   * grad c of a linear field is the constant (1, 0);
#   * the normal of a circular interface points outward;
#   * div(n_radial) on a disk of radius r is exactly 1/r.
# Before these existed the suite only checked that a uniform field gives zero
# gradient and that a flat line gives small curvature, neither of which can see
# a sign error in the pair contribution.
# --------------------------------------------------------------------------- #
def _lattice_disk(radius, dx, rho0=1000.0):
    """Solid disk carved out of a square lattice: a stencil symmetric about every
    interior particle, so the SPH gradient of a linear field has a mean value
    that can be compared with the analytic one without lattice noise."""
    ax = torch.arange(-radius, radius + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(ax, ax, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    pos = pos[torch.linalg.vector_norm(pos, dim=1) < radius]
    n = pos.shape[0]
    mass = torch.full((n,), rho0 * dx ** 2, dtype=torch.float64)
    density = torch.full((n,), rho0, dtype=torch.float64)
    return pos, mass, density


def test_color_gradient_reproduces_linear_field():
    """grad c must return the analytic gradient of a linear colour field.

    With c = x the answer is (1, 0) for every particle whose kernel support is
    fully inside the fluid. The tolerance is 2 %: the residual is the discrete
    zeroth-moment bias of the kernel (1.05 % here at two resolutions), not a
    spatial discretisation error.
    """
    radius, dx = 0.10, 0.01
    pos, mass, density = _lattice_disk(radius, dx)
    h = Config(dx=dx).h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rad = torch.linalg.vector_norm(pos, dim=1)
    interior = rad < radius - 3.0 * dx

    grad_c = compute_color_gradient(pos, pos[:, 0].clone(), mass, density, pairs, h)

    assert abs(float(grad_c[interior, 0].mean()) - 1.0) < 0.02
    assert abs(float(grad_c[interior, 1].mean())) < 0.02


def test_color_gradient_direction_is_from_low_to_high_colour():
    """The gradient must point towards increasing colour, not away from it.

    A linear field tilted along y is checked as well, so that a sign error in a
    single pair contribution cannot hide in a symmetric configuration.
    """
    radius, dx = 0.10, 0.01
    pos, mass, density = _lattice_disk(radius, dx)
    h = Config(dx=dx).h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rad = torch.linalg.vector_norm(pos, dim=1)
    interior = rad < radius - 3.0 * dx

    tilted = pos[:, 1] + 0.5 * pos[:, 0]
    grad_c = compute_color_gradient(pos, tilted, mass, density, pairs, h)

    assert abs(float(grad_c[interior, 1].mean()) - 1.0) < 0.02
    assert abs(float(grad_c[interior, 0].mean()) - 0.5) < 0.02


def test_surface_normal_points_outward_on_a_droplet():
    """For a droplet the colour is high inside, so the CSF normal must be
    outward. The code documents that convention, and the force direction and the
    sign of kappa both depend on it."""
    cfg = Config(dx=0.01)
    h = cfg.h_factor * cfg.dx
    radius, dx = 0.05, 0.01
    pos, mass, density = _lattice_disk(radius, dx, cfg.rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rad = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / rad.clamp_min(1e-30)[:, None]

    i, j = pairs.unbind(dim=1)
    count = torch.zeros(pos.shape[0], dtype=torch.float64)
    count = count.index_add(0, i, torch.ones_like(i, dtype=torch.float64))
    count = count.index_add(0, j, torch.ones_like(j, dtype=torch.float64))
    mean_count = count.mean()
    surface = count < 0.7 * mean_count
    colour = torch.ones(pos.shape[0], dtype=torch.float64)
    colour[surface] = count[surface] / mean_count

    grad_c = compute_color_gradient(pos, colour, mass, density, pairs, h)
    normal = compute_normal(-grad_c)

    assert surface.any()
    alignment = float((normal[surface] * rhat[surface]).sum(dim=1).mean())
    assert alignment > 0.9, f"normal is not outward on the rim (n.rhat = {alignment:.3f})"
    assert float((grad_c[surface] * rhat[surface]).sum(dim=1).mean()) < 0.0


def test_curvature_recovers_analytic_divergence_of_radial_normal():
    """div(r_hat) = 1/r, so a particle at radius r must carry kappa = 1/r.

    This is the only analytic reference available for the curvature stage, and
    it separates the estimator from the colour function: the normal is supplied.
    """
    cfg = Config(dx=0.01)
    h = cfg.h_factor * cfg.dx
    radius, dx = 0.10, 0.01
    pos, mass, density = _lattice_disk(radius, dx, cfg.rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rad = torch.linalg.vector_norm(pos, dim=1)
    normal = pos / rad.clamp_min(1e-30)[:, None]

    kappa = compute_curvature(pos, normal, mass, density, pairs, h)

    band = (rad > 0.03) & (rad < 0.07)
    exact = 1.0 / rad[band]
    rel = (kappa[band] - exact).abs() / exact
    assert float(rel.median()) < 0.05, f"median curvature error {float(rel.median()):.3f}"


def _ring_disk(radius, dx, rho0=1000.0):
    """Disk built from concentric rings: a rim that follows the circle instead of
    the lattice, so local curvature is not distorted by a stair-stepped edge."""
    pts, r = [], 0.0
    while r < radius:
        k = 1 if r == 0.0 else max(8, int(round(2.0 * torch.pi * r / dx)))
        th = torch.linspace(0, 2.0 * torch.pi, k + 1, dtype=torch.float64)[:k]
        pts.append(torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1))
        r += dx
    pos = torch.cat(pts)
    return pos, torch.full((pos.shape[0],), rho0 * dx ** 2, dtype=torch.float64)


def test_surface_tension_contracts_a_convex_interface():
    """Convention chain: outward normal -> positive kappa -> inward force.

    A convex droplet must carry positive curvature (divergence of the outward
    normal) and the resulting acceleration on the rim must point inward. The
    pre-fix operator produced kappa = -86 1/m (about -1/h) on this geometry, i.e.
    the sign was opposite to the documented convention and the magnitude was set
    by the bandwidth of the normal mask rather than by 1/R.
    """
    cfg = Config(dx=0.01, surface_tension_coefficient=0.072, gravity=0.0)
    h = cfg.h_factor * cfg.dx
    radius, dx = 0.05, 0.01
    pos, mass = _ring_disk(radius, dx, cfg.rho0)
    density = torch.full_like(mass, cfg.rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rad = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / rad.clamp_min(1e-30)[:, None]

    i, j = pairs.unbind(dim=1)
    count = torch.zeros(pos.shape[0], dtype=torch.float64)
    count = count.index_add(0, i, torch.ones_like(i, dtype=torch.float64))
    count = count.index_add(0, j, torch.ones_like(j, dtype=torch.float64))
    mean_count = count.mean()
    surface = count < 0.7 * mean_count
    colour = torch.ones(pos.shape[0], dtype=torch.float64)
    colour[surface] = count[surface] / mean_count

    grad_c = compute_color_gradient(pos, colour, mass, density, pairs, h)
    kappa = compute_curvature(pos, compute_normal(-grad_c), mass, density, pairs, h)
    kappa_band = float(kappa[surface].mean())
    assert kappa_band > 0.2 / radius, (
        f"kappa on a convex rim is {kappa_band:.2f} 1/m, expected about {1/radius:.1f} 1/m")

    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    radial = (acc * rhat).sum(dim=1)[surface]
    assert float(radial.sum()) < 0.0, "surface tension does not contract the rim"
