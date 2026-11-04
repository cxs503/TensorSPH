"""Steady-state verification of the viscous operator and the no-slip wall.

The case is plane Poiseuille flow in a 2-D channel: a constant body force is
balanced by wall shear, so the steady profile is ``u(y) = g (y-a)(b-y)/(2 nu)``
with peak ``g H^2/(8 nu)``.  See ``benchmarks/poiseuille/run.py`` and
``docs/verification_poiseuille_channel.md``.

Two kinds of test live here:

* exact, simulation-free checks of the fitting machinery that turns a measured
  profile into ``(nu_eff, wall offset)`` -- these would catch a silent sign or
  factor error in the analysis without needing a fluid;
* a coarse but real simulation that pinpoints the measured constants, so a
  change to the viscous term or to the wall treatment cannot pass unnoticed.
"""

import math

import pytest
import torch

from benchmarks.poiseuille.run import (
    analytic,
    bin_rows,
    build_fluid,
    build_walls,
    fit_profile,
    make_config,
    periodic_images,
    run_case,
)
from tensorsph.droplet_impact import resolve_device

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01


def _synthetic_profile(height, dx, nu, delta, force=BODY_FORCE, rows=None):
    """Sampled analytic parabola on lattice rows, as ``bin_rows`` produces."""
    n_rows = rows if rows is not None else int(round(height / dx))
    profile = []
    for k in range(n_rows):
        y = dx * 0.5 + k * dx
        profile.append({
            "row": k,
            "y": y,
            "y_over_H": y / height,
            "u_mean": analytic(y, height, force, nu, delta),
            "u_std_x": 0.0,
            "count": 1,
        })
    return profile


# --------------------------------------------------------------------------- #
# Analysis machinery
# --------------------------------------------------------------------------- #

def test_the_fit_recovers_a_nominal_parabola_exactly():
    dx = HEIGHT / 8
    fit = fit_profile(_synthetic_profile(HEIGHT, dx, VISCOSITY, 0.0),
                      HEIGHT, BODY_FORCE, VISCOSITY)
    assert fit["nu_eff_over_nu"] == pytest.approx(1.0, rel=1e-6)
    assert fit["delta"] == pytest.approx(0.0, abs=1e-9)
    assert fit["a_plus_b"] == pytest.approx(HEIGHT, rel=1e-6)


def test_the_fit_separates_viscosity_from_wall_offset():
    """A too-viscous fluid and an offset wall must not be confused with each other."""
    dx = HEIGHT / 8
    nu_true, delta_true = 1.25 * VISCOSITY, 0.75 * dx
    fit = fit_profile(_synthetic_profile(HEIGHT, dx, nu_true, delta_true),
                      HEIGHT, BODY_FORCE, VISCOSITY)
    assert fit["nu_eff_over_nu"] == pytest.approx(1.25, rel=1e-6)
    assert fit["delta"] / dx == pytest.approx(0.75, rel=1e-6)


def test_the_offset_sign_is_not_symmetric():
    """A wall outside the domain must come back positive, not flipped."""
    dx = HEIGHT / 12
    for delta_multiple in (0.25, 0.5, 1.0):
        fit = fit_profile(
            _synthetic_profile(HEIGHT, dx, VISCOSITY, delta_multiple * dx),
            HEIGHT, BODY_FORCE, VISCOSITY)
        assert fit["delta"] / dx == pytest.approx(delta_multiple, rel=1e-6)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def test_periodic_images_are_the_channel_shifted_in_x():
    device = torch.device("cpu")
    pos = torch.tensor([[0.01, 0.02], [0.19, 0.05]], dtype=torch.float64)
    images = periodic_images(pos, 0.2)
    assert images.shape == (6, 2)
    assert torch.allclose(images[:2], pos)
    assert torch.allclose(images[2:4], pos + torch.tensor([0.2, 0.0], dtype=torch.float64))
    assert torch.allclose(images[4:], pos - torch.tensor([0.2, 0.0], dtype=torch.float64))


def test_the_wall_bands_sit_outside_the_fluid_and_cover_the_period():
    device = torch.device("cpu")
    dx = HEIGHT / 8
    cfg = make_config(dx, VISCOSITY, 1.0, device)
    length = 12 * dx
    pos, _ = build_fluid(cfg, HEIGHT, length, device)
    wall = build_walls(cfg, HEIGHT, length, 3, device)
    fluid_lo, fluid_hi = float(pos[:, 1].min()), float(pos[:, 1].max())
    assert float(wall[:, 1].min()) < fluid_lo - 0.5 * dx + 1e-12
    assert float(wall[:, 1].max()) > fluid_hi + 0.5 * dx - 1e-12
    # Two bands of three layers, extended past both periodic ends.
    assert wall.shape[0] >= 6 * 12
    assert float(wall[:, 0].min()) < 0.0 and float(wall[:, 0].max()) > length


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #

def _simulate(resolution=8, friction=1.0, t_end=1.0, t_average=0.2):
    device = resolve_device("cpu")
    torch.set_num_threads(1)
    height = HEIGHT
    dx = height / resolution
    length = 12 * dx
    cfg = make_config(dx, VISCOSITY, friction, device)
    case = run_case(cfg, height, length, BODY_FORCE, t_end, t_average, device)
    case["dx"] = dx
    case["cfg"] = cfg
    return case, dx


@pytest.mark.slow
def test_steady_channel_flow_develops_the_poiseuille_parabola():
    """The body force must be balanced by viscous stress, not by the clamp."""
    case, dx = _simulate()
    fit = fit_profile(case["profile"], HEIGHT, BODY_FORCE, VISCOSITY)
    u_max = max(p["u_mean"] for p in case["profile"])

    # The run has to be statistically steady before the profile means anything.
    assert case["steady_drift"] < 0.03, case["steady_drift"]

    # The measured profile is a parabola to a fraction of a percent of u_max;
    # a broken viscous term shows up as curvature, i.e. as fit residual.
    assert fit["fit_residual_rms"] / u_max < 0.03, fit["fit_residual_rms"] / u_max

    # Effective viscosity of the discrete operator.  Measured 1.22 at H/dx = 8
    # (docs/verification_poiseuille_channel.md); the band leaves room for the
    # coarser/finer CPU variation without accepting a factor-of-two change.
    assert 1.05 <= fit["nu_eff_over_nu"] <= 1.45, fit["nu_eff_over_nu"]

    # The no-slip plane stands about one spacing outside the nominal boundary.
    # A sign flip here is exactly the bug this pins.
    assert 0.5 <= fit["delta"] / dx <= 1.5, fit["delta"] / dx

    # The bulk operator, read from the middle half only, is calibrated to well
    # under the wall defect it is attached to.
    bulk = fit_profile(case["profile"], HEIGHT, BODY_FORCE, VISCOSITY,
                       lo=0.25, hi=0.75)
    assert 0.85 <= bulk["nu_eff_over_nu"] <= 1.25, bulk["nu_eff_over_nu"]

    # Wall shear from the measured near-wall slope against the momentum-balance
    # value rho g H / 2.  Measured 0.58 at H/dx = 8: the blended no-slip
    # condition carries only about half the required stress, which is the
    # documented defect (docs/verification_poiseuille_channel.md).  A change to
    # the wall model must move this number, not only the peak velocity.
    rows = case["profile"]
    slope = (rows[1]["u_mean"] - rows[0]["u_mean"]) / dx
    tau = case["cfg"].rho0 * VISCOSITY * slope
    tau_analytic = case["cfg"].rho0 * BODY_FORCE * HEIGHT / 2.0
    assert 0.40 <= tau / tau_analytic <= 0.75, tau / tau_analytic

    # Symmetry about the centre line, to a few percent of u_max.
    first, last = case["profile"][0]["u_mean"], case["profile"][-1]["u_mean"]
    assert abs(first - last) / u_max < 0.05

    # The flow stays subsonic by construction (the run would not be trustworthy
    # otherwise) and inside the Navier-Stokes regime, Re = O(10).
    assert max(p["max_mach"] for p in case["trace"]) < 0.15


@pytest.mark.slow
def test_free_slip_leaves_the_body_force_unbalanced():
    """Negative control: without tangential wall stress there is no steady state.

    With free-slip walls the momentum equation has no term that can balance a
    constant body force, so the whole column accelerates rigidly: ``u -> g t``
    with a flat profile.  Measuring that is what makes the no-slip parabola a
    result rather than a tautology.
    """
    case, _ = _simulate(friction=0.0, t_end=0.3, t_average=0.05)
    profile = case["profile"]
    u_mean = sum(p["u_mean"] for p in profile) / len(profile)
    spread = max(p["u_mean"] for p in profile) - min(p["u_mean"] for p in profile)

    # Flat: the rows agree with each other to a few percent.
    assert spread / u_mean < 0.05, spread / u_mean

    # And the free-stream speed follows the rigid-body analytic u = g t to 20 %.
    expected = BODY_FORCE * 0.3
    assert 0.75 <= u_mean / expected <= 1.15, u_mean / expected

    # A steady state would show a vanishing drift; a free-slip column accelerates.
    assert case["steady_drift"] > 0.05, case["steady_drift"]