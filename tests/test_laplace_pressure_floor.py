"""Regression gates for the Young-Laplace error-source decomposition.

The physics runs are far too slow for a unit test suite (they advance the SPH
drop for 1.5 capillary times); what is pinned here is (a) the *initialisation*
facts that the decomposition rests on -- measured directly, they take
milliseconds -- and (b) the numbers recorded in
``benchmarks/laplace_pressure_floor/report.json``, so that a silent change in
the pressure closure, the EOS or the measurement definitions cannot slip
through unnoticed.

Two matrices are pinned, because they mean different things:

* ``report.json`` -- the *current* model, i.e. with the free-surface pressure
  closure ``Config.free_surface_pressure`` enabled (P0, see
  ``docs/verification_laplace_pressure_floor.md`` §8).  The Young-Laplace gate
  is asserted on this matrix.
* ``report_open_surface.json`` -- the *pre-closure* model, kept on purpose: the
  C1/C2/C3/C4 error decomposition below is a measurement of that model, and the
  historical failure must stay reproducible rather than be silently rewritten.
  Regenerate with ``run.py --open-surface``.

The former ``xfail(strict=True)`` 15 % gate is now a plain assertion: the
closure landed, so the gate is a regression gate.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
REPORT = BENCH / "report.json"


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "laplace_pressure_floor_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lpf = _load_run_module()


REPORT_OPEN = BENCH / "report_open_surface.json"


@pytest.fixture(scope="module")
def report():
    """Current model: free-surface pressure closure enabled."""
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py first")
    data = json.loads(REPORT.read_text())
    assert data["closure"]["free_surface_pressure"] is True
    return data


@pytest.fixture(scope="module")
def legacy():
    """Pre-closure model, pinned so the historical diagnosis stays reproducible."""
    if not REPORT_OPEN.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py --open-surface first")
    data = json.loads(REPORT_OPEN.read_text())
    # The archived matrix predates the closure (it has no "closure" block); a
    # regenerated one states it explicitly. Either way it must be the open model.
    assert data.get("closure", {}).get("free_surface_pressure", False) is False
    return data


# --------------------------------------------------------------------------- #
# initialisation facts (C2 / C3)
# --------------------------------------------------------------------------- #
def test_hex_lattice_sum_density_bias_is_systematic():
    """h = 1.3 dx makes the discrete kernel sum 1 + b; b ~ 5.9e-3, not noise."""
    st = lpf.density_structure(dx=0.002, sound_speeds=[20.0])
    assert st["lattice_bias_rel_density"] == pytest.approx(0.00588, abs=2e-4)
    # A perfect lattice has no scatter: the offset is a *bias*, not noise.
    assert st["lattice_bias_std"] < 1e-9


def test_mass_calibration_removes_the_bias():
    _, _, _, _, bias, after = lpf.calibrated_droplet(dx=0.002)
    assert bias > 1e-3
    assert after == pytest.approx(1.0, abs=1e-12)


def test_spurious_initial_pressure_grows_as_sound_speed_squared():
    """p_init = B[(1+b)^gamma-1] with B ~ c0^2, so p_init(20)/p_init(5) ~ 16."""
    st = lpf.density_structure(dx=0.002, sound_speeds=[20.0, 5.0])
    p20 = st["per_sound_speed"]["20"]["initial_spurious_pressure_Pa"]
    p5 = st["per_sound_speed"]["5"]["initial_spurious_pressure_Pa"]
    assert p20 > 2000.0                     # 6.6 x the 360 Pa Laplace signal
    assert p20 / p5 == pytest.approx(16.0, rel=0.05)


def test_free_surface_density_deficit_and_its_resolution_dependence():
    """The deficit is what band-A differencing mistakes for a pressure."""
    coarse = lpf.density_structure(dx=0.002, sound_speeds=[20.0])
    fine = lpf.density_structure(dx=0.0015, sound_speeds=[20.0])
    for st in (coarse, fine):
        assert st["bands"]["0.85-0.95"]["mean_rel_density"] < -0.03
        assert st["bands"]["0.95-1.05"]["mean_rel_density"] < -0.20
    # finer resolution -> thinner deficit shell
    assert (fine["bands"]["0.85-0.95"]["mean_rel_density"]
            > coarse["bands"]["0.85-0.95"]["mean_rel_density"])


# --------------------------------------------------------------------------- #
# recorded evidence (C1 / C3 / C4)
# --------------------------------------------------------------------------- #
def test_report_is_complete_and_finite(report):
    assert report["runs"], "no runs recorded"
    for run in report["runs"]:
        assert run["finite"], run
        for key in ("definition_bandA", "definition_bandB", "definition_absolute",
                    "absolute_time_average_tail"):
            assert math.isfinite(run[key]["rel_error_pct"])
        assert run["spurious_current"]["vmax_over_vref_final"] < 0.1


def test_band_differencing_costs_tens_of_percentage_points(legacy):
    """C1: the historical p_core - p(shell) read-out is contaminated by C3."""
    for run in legacy["runs"]:
        gap = (run["definition_bandA"]["rel_error_pct"]
               - run["definition_absolute"]["rel_error_pct"])
        assert gap < -30.0, (run["dx_m"], run["sound_speed"], gap)


def test_compression_ratio_is_c0_and_resolution_independent(legacy):
    """C3: the interior compresses to a fixed fraction of sigma/R -- closure, not noise.

    Pinned on the *open-surface* matrix: this is the defect the closure fixed.
    """
    ratios = []
    for run in legacy["runs"]:
        r = ((run["density"]["rho_core_over_rho0"] - 1.0)
             / run["density"]["compression_required"])
        ratios.append(r)
        assert 0.40 < r < 0.80, (run["dx_m"], run["sound_speed"], r)
    assert max(ratios) - min(ratios) < 0.25


def test_residual_is_not_explained_by_the_eos_noise_floor(legacy):
    """C4 is falsified: at low c0 the predicted pressure noise is tiny while the
    residual stays put.  Pinned on the open-surface matrix -- that is the model
    whose residual the closure then removed."""
    dp = legacy["reference"]["delta_p_Pa"]
    rho0 = legacy["reference"]["rho0"]
    worst = []
    for run in legacy["runs"]:
        if run["sound_speed"] > 5.5:
            continue
        predicted = (rho0 * run["sound_speed"] ** 2
                     * run["density"]["rho_core_rel_std"]) / dp
        residual = abs(run["absolute_time_average_tail"]["rel_error_pct"]) / 100.0
        assert predicted < 0.15, (run["dx_m"], predicted)
        assert residual > 0.25, (run["dx_m"], residual)
        worst.append(residual - predicted)
    assert worst and min(worst) > 0.15


def test_laplace_jump_within_15pct(report):
    """Regression gate for P0: with the closure on, every case is inside 15 %.

    The four-point grid was the pre-registered acceptance criterion for the
    free-surface pressure closure (docs/verification_laplace_pressure_floor.md
    §7/§8).  Before the closure the same grid sat at -34 % ... -38 %.
    """
    assert report["runs"], "no closure runs recorded"
    for run in report["runs"]:
        err = abs(run["absolute_time_average_tail"]["rel_error_pct"])
        assert err < 15.0, (run["dx_m"], run["sound_speed"], err)
    # At least half the grid must also clear the platform's 3 % bar, otherwise
    # the 15 % pass would be a gate that only just holds.
    tight = [r for r in report["runs"]
             if r["absolute_time_average_tail"]["within_3pct"]]
    assert len(tight) >= len(report["runs"]) / 2, len(tight)


def test_closure_improves_every_case(legacy, report):
    """A/B: the closure must be an improvement at every grid point, not a shuffle."""
    before = {(r["dx_m"], r["sound_speed"]):
              abs(r["absolute_time_average_tail"]["rel_error_pct"])
              for r in legacy["runs"]}
    for run in report["runs"]:
        key = (run["dx_m"], run["sound_speed"])
        after = abs(run["absolute_time_average_tail"]["rel_error_pct"])
        assert after < before[key], (key, before[key], after)


def test_closure_prolongation_converges(report):
    """The deficit shell is several particles thick: more passes must help.

    Pinned on the recorded pass ladder so the ``--passes 3`` default is
    evidence-based; a change that makes a single pass sufficient would also be
    a fine outcome, but it has to be recorded, not assumed.
    """
    ladder = report["passes_ladder"]
    assert [d["closure_passes"] for d in ladder] == [1, 2, 3]
    errors = [abs(d["tail_rel_error_pct"]) for d in ladder]
    assert errors[-1] < errors[0], errors
    assert errors[-1] < 15.0, errors


# --------------------------------------------------------------------------- #
# the closure operator itself (fast, no SPH time integration)
# --------------------------------------------------------------------------- #
def _lattice(n: int = 5, dx: float = 0.002, rho0: float = 1000.0):
    import torch

    ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    pos = torch.stack([xs.reshape(-1).float(), ys.reshape(-1).float()], dim=1) * dx
    mass = torch.full((pos.shape[0],), rho0 * dx * dx)
    return pos, mass


def _state(n: int = 5, dx: float = 0.002, rho0: float = 1000.0, c0: float = 20.0):
    import torch
    from tensorsph.droplet_impact import (Config, cubic_spline_kernel,
                                          tait_pressure)

    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=c0, gravity=0.0,
                 viscosity=0.0, free_surface_pressure=True)
    h = cfg.h_factor * cfg.dx
    pos, mass = _lattice(n, dx, rho0)
    delta = pos[:, None, :] - pos[None, :, :]
    r = torch.linalg.vector_norm(delta, dim=-1)
    w = cubic_spline_kernel(r, h)
    rho = (w * mass[None, :]).sum(dim=1)
    pressure = tait_pressure(rho, cfg)
    idx = torch.arange(pos.shape[0])
    ii, jj = torch.meshgrid(idx, idx, indexing="ij")
    keep = ii.reshape(-1) < jj.reshape(-1)
    pairs = torch.stack([ii.reshape(-1)[keep], jj.reshape(-1)[keep]], dim=1)
    return cfg, pos, mass, rho, pressure, pairs, h


def test_closure_is_off_by_default():
    """It changes every free-surface case, so it must not be silently on."""
    from tensorsph.droplet_impact import Config

    cfg = Config()
    assert cfg.free_surface_pressure is False
    assert cfg.free_surface_pressure_passes == 1
    assert cfg.free_surface_rho0_ratio == pytest.approx(0.99)


def test_closure_leaves_interior_pressure_untouched():
    """Only particles flagged as free surface may be modified."""
    import torch
    from tensorsph.droplet_impact import shell_pressure

    cfg, pos, _, rho, pressure, pairs, _ = _state()
    out = shell_pressure(pos, rho, pressure, pairs.unbind(dim=1), cfg)
    interior = rho >= cfg.free_surface_rho0_ratio * cfg.rho0
    assert bool(interior.any()) and bool((~interior).any())
    assert torch.equal(out[interior], pressure[interior])
    assert bool((out >= 0.0).all())


def test_closure_matches_an_independent_dense_average():
    """The pass-based implementation must equal the textbook kernel average."""
    import torch
    from tensorsph.droplet_impact import cubic_spline_kernel, shell_pressure

    cfg, pos, _, rho, pressure, pairs, h = _state()
    out = shell_pressure(pos, rho, pressure, pairs.unbind(dim=1), cfg)
    # Independently: dense full-neighbour average, no pair list, no scatter.
    r = torch.linalg.vector_norm(pos[:, None, :] - pos[None, :, :], dim=-1)
    w = cubic_spline_kernel(r, h)
    dense = (w * pressure[None, :]).sum(dim=1) / w.sum(dim=1).clamp_min(1e-30)
    free_surface = rho < cfg.free_surface_rho0_ratio * cfg.rho0
    assert torch.allclose(out[free_surface], dense[free_surface].clamp_min(0.0),
                          rtol=1e-5, atol=1e-8)


def test_closure_prolongation_is_monotone_on_a_lattice():
    """Extra passes may only raise the extrapolated pressure on a static field."""
    import torch
    from tensorsph.droplet_impact import shell_pressure

    cfg, pos, _, rho, pressure, pairs, _ = _state()
    free_surface = rho < cfg.free_surface_rho0_ratio * cfg.rho0
    values = []
    for passes in (1, 2, 3, 4):
        cfg.free_surface_pressure_passes = passes
        values.append(shell_pressure(pos, rho, pressure, pairs.unbind(dim=1), cfg))
    for lo, hi in zip(values, values[1:]):
        assert bool((hi[free_surface] >= lo[free_surface] - 1e-9).all())


def test_closure_gravity_term_enters_with_the_right_sign():
    """Adami eq. (12): a particle sitting *above* its neighbour gets the lower
    pressure, p_above = p_below - rho |g| dy (hydrostatic, pressure grows
    downward).  A sign flip here would invert the whole pressure field."""
    import torch
    from tensorsph.droplet_impact import Config, shell_pressure, tait_pressure

    cfg = Config(dx=0.002, h_factor=1.3, rho0=1000.0, sound_speed=20.0,
                 gravity=9.81, viscosity=0.0, free_surface_pressure=True,
                 free_surface_rho0_ratio=2.0)  # flag every particle
    pos = torch.tensor([[0.0, 0.0], [0.0, 0.0025], [0.01, 0.0]])
    rho = torch.full((3,), 1000.0)
    pressure = torch.tensor([100.0, 0.0, 0.0])
    # Pair (0, 1): particle 1 sits above 0; pair (0, 2): unrelated but in range.
    pairs = (torch.tensor([0]), torch.tensor([1]))
    out = shell_pressure(pos, rho, pressure, pairs, cfg)
    # Target 1 is 2.5 mm above target 0, so it must read 24.5 Pa *lower*.
    assert float(out[1]) < float(pressure[0])
    assert float(out[1]) == pytest.approx(float(pressure[0]) - 1000.0 * 9.81 * 0.0025, rel=1e-6)
    # Target 0 is below target 1, so its extrapolation must be *higher*, and the
    # difference must be exactly the hydrostatic increment rho |g| dy.
    assert float(out[0]) == pytest.approx(float(pressure[1]) + 1000.0 * 9.81 * 0.0025, rel=1e-6)
