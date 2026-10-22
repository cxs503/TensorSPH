import torch

from tensorsph.droplet_impact import (
    Config, cubic_spline_gradient, cubic_spline_kernel, make_particles, rhs, run
)


def test_kernel_is_finite_nonnegative_and_compact():
    r = torch.tensor([0.0, 0.2, 0.9, 2.1], dtype=torch.float64)
    w = cubic_spline_kernel(r, 1.0)
    assert torch.isfinite(w).all()
    assert (w >= 0).all()
    assert w[-1].item() == 0.0
    assert w[0].item() > w[1].item()


def test_kernel_gradient_at_zero_is_finite():
    delta = torch.zeros((2, 2, 2), dtype=torch.float64)
    grad = cubic_spline_gradient(delta, 0.1)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) == 0


def test_initial_state_and_rhs_are_finite():
    cfg = Config(dx=0.05, steps=1, pool_width=0.4, pool_depth=0.15,
                 drop_radius=0.05, drop_center_y=0.25, device="cpu")
    pos, _, mass = make_particles(cfg, torch.device("cpu"))
    vel = torch.zeros_like(pos)
    rho, pressure, acc = rhs(pos, vel, mass, cfg)
    assert pos.ndim == 2 and pos.shape[1] == 2
    assert torch.isfinite(rho).all()
    assert torch.isfinite(pressure).all()
    assert torch.isfinite(acc).all()


def test_short_run_writes_auditable_artifacts(tmp_path):
    cfg = Config(dx=0.05, steps=2, output_every=1, pool_width=0.4,
                 pool_depth=0.15, drop_radius=0.05, drop_center_y=0.25,
                 device="cpu")
    rows = run(cfg, tmp_path)
    assert len(rows) == 3
    assert (tmp_path / "diagnostics.csv").exists()
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "particles_000002.pt").exists()
