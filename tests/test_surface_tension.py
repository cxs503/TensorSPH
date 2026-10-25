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
