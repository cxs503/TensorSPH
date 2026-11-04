"""Patch tests/test_laplace_pressure_floor.py: the historical diagnosis moves to
the open-surface matrix, the 15% gate becomes a real assertion, and the closure
itself gets unit tests against an independent dense implementation."""
from pathlib import Path

p = Path("tests/test_laplace_pressure_floor.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, "anchor not found: %r" % old[:70]
    s = s.replace(old, new)


# --- docstring: the module now covers both models -------------------------
sub(
    '''The last test is ``xfail(strict=True)``: it states the acceptance criterion for
the still-open P0 (free-surface pressure closure, see
``docs/verification_laplace_pressure_floor.md`` §7).  When that work lands the
test turns into an ``xpass`` and the suite fails on purpose, forcing the
verification record to be updated instead of drifting.
"""''',
    '''Two matrices are pinned, because they mean different things:

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
"""''',
)

# --- fixtures -------------------------------------------------------------
sub(
    '''@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py first")
    return json.loads(REPORT.read_text())''',
    '''REPORT_OPEN = BENCH / "report_open_surface.json"


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
    assert data["closure"]["free_surface_pressure"] is False
    return data''',
)

# --- diagnostic tests -> legacy matrix ------------------------------------
sub(
    '''def test_band_differencing_costs_tens_of_percentage_points(report):
    """C1: the historical p_core - p(shell) read-out is contaminated by C3."""
    for run in report["runs"]:''',
    '''def test_band_differencing_costs_tens_of_percentage_points(legacy):
    """C1: the historical p_core - p(shell) read-out is contaminated by C3."""
    for run in legacy["runs"]:''',
)
sub(
    '''def test_compression_ratio_is_c0_and_resolution_independent(report):
    """C3: the interior compresses to a fixed fraction of sigma/R -- closure, not noise."""
    ratios = []
    for run in report["runs"]:''',
    '''def test_compression_ratio_is_c0_and_resolution_independent(legacy):
    """C3: the interior compresses to a fixed fraction of sigma/R -- closure, not noise.

    Pinned on the *open-surface* matrix: this is the defect the closure fixed.
    """
    ratios = []
    for run in legacy["runs"]:''',
)
sub(
    '''def test_residual_is_not_explained_by_the_eos_noise_floor(report):
    """C4 is falsified: at low c0 the predicted pressure noise is tiny while the
    residual stays put."""
    worst = []
    for run in report["runs"]:
        if run["sound_speed"] > 5.5:
            continue
        dp = report["reference"]["delta_p_Pa"]''',
    '''def test_residual_is_not_explained_by_the_eos_noise_floor(legacy):
    """C4 is falsified: at low c0 the predicted pressure noise is tiny while the
    residual stays put (open-surface matrix -- the closure removed the residual)."""
    worst = []
    for run in legacy["runs"]:
        if run["sound_speed"] > 5.5:
            continue
        dp = legacy["reference"]["delta_p_Pa"]''',
)

# --- the former xfail gate becomes a real gate ----------------------------
sub(
    '''@pytest.mark.xfail(strict=True, reason="free-surface pressure closure (P0) not "
                                      "implemented: tait_pressure is clamped at "
                                      "p >= 0 across the deficit shell, so the "
                                      "interior settles at ~0.57 sigma/R")
def test_laplace_jump_within_15pct(report):
    """Acceptance criterion for the open P0: every recorded case inside 15 %."""
    for run in report["runs"]:
        err = abs(run["absolute_time_average_tail"]["rel_error_pct"])
        assert err < 15.0, (run["dx_m"], run["sound_speed"], err)''',
    '''def test_laplace_jump_within_15pct(report):
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
    assert errors[-1] < 15.0, errors''',
)

# --- unit tests for the closure itself ------------------------------------
s += '''

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
    """Adami eq. (12): the body-force increment is +rho |g| dy for a particle
    sitting above its neighbour, i.e. pressure grows downward."""
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
    # Target 1's source is p_0 + rho |g| (y_1 - y_0) > 100 Pa.
    assert out[1] > 100.0
    # And it must be exactly the weighted source, since the pair is the only one.
    assert out[1] == pytest.approx(pressure[0] + 1000.0 * 9.81 * 0.0025, rel=1e-6)
'''
p.write_text(s)
print("patched tests")