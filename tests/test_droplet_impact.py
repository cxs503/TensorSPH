import json

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    acoustic_courant,
    evaluate_acceptance,
    cubic_spline_gradient,
    cubic_spline_kernel,
    make_particles,
    rhs,
    run,
    suggested_dt,
    tait_pressure,
)

SMALL = dict(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
             drop_center_y=0.25, device="cpu")


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


def test_tait_pressure_is_non_negative_and_monotonic():
    cfg = Config(**SMALL)
    rho = torch.linspace(0.4 * cfg.rho0, 1.1 * cfg.rho0, 64, dtype=torch.float64)
    pressure = tait_pressure(rho, cfg)
    assert torch.isfinite(pressure).all()
    assert (pressure >= 0).all(), "free-surface rarefaction must not create tensile pressure"
    assert torch.all(pressure[1:] >= pressure[:-1])
    compressed = torch.tensor([1.01 * cfg.rho0], dtype=torch.float64)
    expected = (cfg.rho0 * cfg.sound_speed**2 / cfg.gamma) * (1.01**cfg.gamma - 1.0)
    torch.testing.assert_close(tait_pressure(compressed, cfg),
                               torch.tensor([expected], dtype=torch.float64))


def test_pressure_field_has_no_tensile_values():
    cfg = Config(**SMALL)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    _, pressure, _ = rhs(pos, torch.zeros_like(pos), mass, cfg)
    assert torch.isfinite(pressure).all()
    assert (pressure >= 0).all()


def test_initial_state_and_rhs_are_finite():
    cfg = Config(**SMALL, steps=1)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    rho, pressure, acc = rhs(pos, torch.zeros_like(pos), mass, cfg)
    assert pos.ndim == 2 and pos.shape[1] == 2
    assert mass.shape == (pos.shape[0],)
    assert torch.isfinite(rho).all()
    assert torch.isfinite(pressure).all()
    assert torch.isfinite(acc).all()


def test_short_run_writes_auditable_artifacts(tmp_path):
    cfg = Config(**SMALL, steps=2, output_every=1)
    rows = run(cfg, tmp_path)
    assert len(rows) == 3
    for name in ("diagnostics.csv", "run_manifest.json", "particles_000002.pt"):
        assert (tmp_path / name).exists()


def test_run_rejects_invalid_time_step(tmp_path):
    cfg = Config(**SMALL, dt=0.0, steps=1)
    with pytest.raises(ValueError, match="dt"):
        run(cfg, tmp_path)


def test_run_rejects_time_step_above_courant_limit(tmp_path):
    cfg = Config(**SMALL, dt=5e-3, steps=1)
    with pytest.raises(ValueError, match="Courant"):
        run(cfg, tmp_path)


def test_suggested_dt_satisfies_the_limit():
    cfg = Config(**SMALL)
    dt = suggested_dt(cfg)
    assert acoustic_courant(dt, cfg) <= cfg.max_courant + 1e-12
    assert acoustic_courant(cfg.dt, cfg) <= cfg.max_courant


def test_diagnostics_include_stability_and_conservation_probes(tmp_path):
    cfg = Config(**SMALL, steps=1, output_every=1)
    rows = run(cfg, tmp_path)
    required = {
        "mass_total", "momentum_x", "momentum_y", "rho_max_relative_error",
        "max_speed", "max_acceleration", "courant_number", "mach_number",
    }
    assert required.issubset(rows[0])
    for row in rows:
        values = torch.tensor([row[key] for key in required])
        assert torch.isfinite(values).all()
        assert row["mass_total"] > 0


def test_mass_is_conserved_exactly(tmp_path):
    cfg = Config(**SMALL, steps=5, output_every=1)
    rows = run(cfg, tmp_path)
    masses = [row["mass_total"] for row in rows]
    assert max(masses) - min(masses) < 1e-9


def test_runs_are_deterministic(tmp_path):
    cfg = Config(**SMALL, steps=3, output_every=1)
    first = run(cfg, tmp_path / "a")
    second = run(cfg, tmp_path / "b")
    assert first == second


def test_manifest_records_statuses_and_acceptance(tmp_path):
    cfg = Config(**SMALL, steps=2, output_every=1)
    run(cfg, tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["solver"]["version"]
    assert set(manifest["statuses"]) >= {
        "software_execution", "numerical_verification", "physical_validation"}
    assert manifest["statuses"]["physical_validation"] == "not_evaluated"
    assert manifest["statuses"]["software_execution"] == "success"
    criteria = {item["id"]: item for item in manifest["acceptance"]["criteria"]}
    for key in ("finite_state", "density_compression", "mach_number",
                "acoustic_courant", "boundary_escape"):
        assert key in criteria
        assert criteria[key]["result"] in ("pass", "fail", "not_evaluated")
    assert "python" in manifest["environment"]
    assert "torch" in manifest["environment"]
    assert manifest["derived"]["suggested_dt"] > 0
    assert manifest["derived"]["viscous_dt_limit"] > 0
    assert manifest["derived"]["gravity_dt_limit"] > 0
    assert manifest["derived"]["diffusion_dt_limit"] is None  # no diffusion term
    assert manifest["derived"]["smoothing_length"] == pytest.approx(cfg.h_factor * cfg.dx)


def test_manifest_distinguishes_failed_numerical_gate(tmp_path):
    # A gravity far beyond the acoustic scaling drives |v| past max_mach within
    # a couple of steps: execution succeeds, numerical verification does not.
    cfg = Config(**SMALL, steps=2, output_every=1, gravity=5.0e3, fail_fast=False)
    run(cfg, tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    criteria = {item["id"]: item for item in manifest["acceptance"]["criteria"]}
    assert criteria["mach_number"]["result"] == "fail"
    assert manifest["statuses"]["numerical_verification"] == "fail"
    assert manifest["statuses"]["software_execution"] == "success"
    assert manifest["statuses"]["physical_validation"] == "not_evaluated"


def test_fail_fast_raises_but_still_writes_artifacts(tmp_path):
    from tensorsph.droplet_impact import SimulationUnstableError

    cfg = Config(**SMALL, steps=2, output_every=1, gravity=5.0e3)
    with pytest.raises(SimulationUnstableError, match="mach_number"):
        run(cfg, tmp_path)
    assert (tmp_path / "diagnostics.csv").exists()
    assert (tmp_path / "run_manifest.json").exists()


def test_boundary_escape_is_not_evaluated_for_clamp_mode(tmp_path):
    cfg = Config(**SMALL, steps=1, output_every=1, boundary="clamp")
    run(cfg, tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    criteria = {item["id"]: item for item in manifest["acceptance"]["criteria"]}
    assert criteria["boundary_escape"]["result"] == "not_evaluated"


def test_viscous_cfl_is_rejected_before_the_run(tmp_path):
    """The acoustic Courant alone does not bound the explicit viscous term.

    On the default case ``viscosity = 1`` is above the limit (1.3e-4 s against
    a 4e-4 s step) and used to be accepted, then diverged to 53 % density
    compression and Mach 1.2. Here the same relationship is reproduced on the
    smaller test geometry.
    """
    cfg = Config(**SMALL, viscosity=2.0, steps=1)
    with pytest.raises(ValueError, match="viscous limit"):
        run(cfg, tmp_path)


def test_gravity_cfl_is_rejected_before_the_run(tmp_path):
    cfg = Config(**SMALL, gravity=1.0e5, steps=1)
    with pytest.raises(ValueError, match="gravity limit"):
        run(cfg, tmp_path)


def test_rhs_rejects_non_two_dimensional_input():
    cfg = Config(**SMALL)
    pos = torch.zeros((4, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="2-D"):
        rhs(pos, torch.zeros_like(pos), torch.ones(4, dtype=torch.float64), cfg)


def test_acceptance_uses_all_step_extrema_not_only_sampled_rows():
    cfg = Config(**SMALL)
    rows = [{"rho_max": cfg.rho0 * 1.001, "max_speed": 0.1,
             "courant_number": 0.01}]
    sampled = evaluate_acceptance(rows, cfg, 0)
    extremes = {"peak_rho_compression": 0.4, "peak_speed": 9.0,
                "peak_courant": 0.9}
    measured_all = evaluate_acceptance(rows, cfg, 0, extremes=extremes)
    by_id = {c["id"]: c for c in measured_all}
    assert by_id["density_compression"]["result"] == "fail"
    assert by_id["mach_number"]["result"] == "fail"
    assert by_id["acoustic_courant"]["result"] == "fail"
    assert by_id["density_compression"]["population"] == "all_steps"
    # ... while the sampled rows alone would have passed every one of them.
    assert all(c["population"] == "sampled_rows" or c["result"] == "pass"
               for c in sampled)
    assert {c["id"]: c["result"] for c in sampled}["density_compression"] == "pass"


def test_manifest_records_the_extrema_population(tmp_path):
    cfg = Config(**SMALL, steps=4, output_every=2)
    rows = run(cfg, tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    sampling = manifest["sampling"]
    assert sampling["extrema_population"] == "all_steps"
    assert sampling["diagnostic_rows"] == len(rows)
    by_id = {c["id"]: c for c in manifest["acceptance"]["criteria"]}
    sampled_compression = max(row["rho_max"] / cfg.rho0 - 1.0 for row in rows)
    assert by_id["density_compression"]["measured"] >= sampled_compression
    assert manifest["sampling"]["extrema"]["peak_rho_compression"] >= 0.0


def test_plot_can_be_skipped_and_snapshots_are_written_anyway(tmp_path):
    cfg = Config(**SMALL, steps=2, output_every=1)
    run(cfg, tmp_path, plot=False)
    assert (tmp_path / "particles_000002.pt").exists()
    assert not (tmp_path / "particle_snapshots.png").exists()


def test_plot_reads_back_the_written_snapshots(tmp_path):
    pytest.importorskip("matplotlib")
    cfg = Config(**SMALL, steps=2, output_every=1)
    run(cfg, tmp_path, plot=True)
    assert (tmp_path / "particle_snapshots.png").stat().st_size > 0
