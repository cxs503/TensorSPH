"""Verification gates for the CSF surface-tension model, per scheme.

Two kinds of test live here.

*What the model does do* is pinned as a normal test: on an isolated droplet the
acceleration is predominantly radial and inward, the internal force field
carries almost no net momentum, the two schemes are measured with their own
fields, and at R/dx = 20 the acceleration field integrates to the Young-Laplace
jump ``dp = sigma / R`` within 15 %.

*What the model does not do* is pinned with ``xfail(strict=True)``.  Those pins
belong to the **legacy** scheme (neighbour-count colour function, uncorrected
SPH operators), which is still shipped as an A/B reference and still misses the
Laplace law badly.  When someone reworks it the ``strict`` xfail turns into an
XPASS failure and this file has to be re-pinned with new measured numbers.

History (m2681): every measurement below is produced by
``benchmarks/laplace/run.py::csf_diagnostics``, which now reports the fields of
*the scheme that generated the acceleration*.  Before m2681 that helper built a
neighbour-count colour field for **both** schemes, so the "consistent" force was
reported next to a "legacy" interfacial structure (``int |grad c| dr = 0.446``).
Those mixed numbers described neither model: on the same particles the shipped
scheme gives 1.226 for the normalised delta_s (criterion 1.0) and 0.613 for the
raw ``|grad c|`` (criterion 0.5, the missing half lives in the void -- see
``tensorsph.csf_consistent.csf_fields``).

Numbers are deterministic (fixed lattice, no randomness); they were measured at
R = 0.05 m, sigma = 0.072 N/m, rho0 = 1000 kg/m^3, h = 1.3 dx, square lattice,
i.e. ``sigma / R = 1.44 Pa``.  See docs/benchmarks/laplace-verification.md.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

RADIUS = 0.05
SIGMA = 0.072
RHO0 = 1000.0
LEVELS = 10                       # 318 particles: fast enough for the suite
HIGH_LEVELS = 20                  # 1262 particles: the resolution the gate needs


def _benchmark_module():
    path = (Path(__file__).resolve().parents[1]
            / "benchmarks" / "laplace" / "run.py")
    spec = importlib.util.spec_from_file_location("laplace_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accelerate(levels=LEVELS, scheme=None):
    """Lattice disc plus the CSF acceleration of ``scheme`` (default: solver's)."""
    bench = _benchmark_module()
    dx = RADIUS / levels
    cfg = Config(dx=dx, rho0=RHO0, surface_tension_coefficient=SIGMA,
                 gravity=0.0)
    pos, mass = bench.lattice_disc(RADIUS, dx, RHO0)
    rho = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg,
                                        scheme=scheme)
    return bench, cfg, pos, mass, rho, acc, pairs


@pytest.fixture(scope="module")
def droplet():
    """Lattice disc plus the CSF acceleration from the production path."""
    bench, cfg, pos, mass, rho, acc, pairs = _accelerate(LEVELS)
    diagnostics = bench.csf_diagnostics(pos, mass, rho, cfg, SIGMA)
    return bench, cfg, pos, mass, rho, acc, diagnostics


def test_the_solver_defaults_to_the_consistent_scheme(droplet):
    """The gate below only means something if the shipped default is measured."""
    _, cfg, *_ = droplet
    assert cfg.csf_scheme == "consistent"


def test_surface_tension_force_is_radial_and_inward(droplet):
    """The one property the merged benchmark did check, pinned as a test."""
    _, cfg, pos, mass, rho, acc, _diag = droplet
    r = torch.linalg.vector_norm(pos, dim=-1)
    outward = pos / r[:, None]
    tangential = torch.stack((-outward[:, 1], outward[:, 0]), dim=1)
    a_r = (acc * outward).sum(dim=1)
    a_t = (acc * tangential).sum(dim=1)
    band = (r > 0.9 * RADIUS) & (torch.linalg.vector_norm(acc, dim=-1) > 0.0)
    assert int(band.sum()) > 20
    radial_fraction = float((a_r[band].abs()
                             / (a_r[band].abs() + a_t[band].abs())).mean())
    inward_fraction = float((a_r[band] < 0).to(torch.float64).mean())
    # Measured at this resolution: 0.57 radial, 0.73 inward. The bounds leave
    # room for the lattice-orientation noise the model is known to have, while
    # still failing if the force stops pointing at the centre.
    assert radial_fraction > 0.45
    assert inward_fraction > 0.6


def test_surface_tension_net_momentum_is_near_zero(droplet):
    """CSF is a single-particle force, so conservation is approximate, not exact."""
    _, cfg, pos, mass, rho, acc, _diag = droplet
    momentum = (mass[:, None] * acc).sum(dim=0)
    scale = (mass * torch.linalg.vector_norm(acc, dim=-1)).sum()
    residual = float(torch.linalg.vector_norm(momentum) / scale)
    # Measured 8.6e-3 at R/dx = 10.
    assert residual < 0.05


def _discrete_laplace(bench, pos, mass, rho, cfg, acc, scheme=None):
    """Normalised delta_s integral, raw |grad c| integral, curvature and dp."""
    _, kappa, d_s, gmag, is_surface = bench.csf_diagnostics(
        pos, mass, rho, cfg, SIGMA, scheme=scheme)
    r = torch.linalg.vector_norm(pos, dim=-1)
    a_r = (acc * pos / r[:, None]).sum(dim=1)
    edges = torch.linspace(0.0, 1.2 * RADIUS, max(12, LEVELS))
    centre = 0.5 * (edges[1:] + edges[:-1])
    delta = edges[1:] - edges[:-1]
    ds_shell = bench.radial_bin(d_s, r, edges)
    gmag_shell = bench.radial_bin(gmag, r, edges)
    ar_shell = bench.radial_bin(a_r, r, edges)
    finite = torch.isfinite(ds_shell)
    delta_s_integral = float((ds_shell[finite] * delta[finite]).sum())
    finite_g = torch.isfinite(gmag_shell)
    grad_integral = float((gmag_shell[finite_g] * delta[finite_g]).sum())
    integrand = torch.where(torch.isfinite(ar_shell), -RHO0 * ar_shell,
                            torch.zeros_like(ar_shell))
    dp = float(torch.trapezoid(integrand, centre))
    band = is_surface & (r > 0.8 * RADIUS)
    kappa_band = float(kappa[band].mean()) if band.any() else float("nan")
    return {
        "delta_s_integral": delta_s_integral,
        "grad_integral": grad_integral,
        "kappa_times_R": kappa_band * RADIUS,
        "dp": dp,
        "dp_rel_error": dp / (SIGMA / RADIUS) - 1.0,
    }


# --------------------------------------------------------------------------- #
# the correction this mission is about: measure the scheme that made the force
# --------------------------------------------------------------------------- #
def test_diagnostics_report_the_fields_of_the_scheme_that_made_the_force(droplet):
    """Regression guard for the mixed-model artefact.

    Before m2681 ``csf_diagnostics`` always rebuilt the neighbour-count colour
    field, so this test could not have been written: both columns came out at
    the legacy value.  The two schemes must now differ by roughly the factor the
    normalisation is supposed to supply.
    """
    bench, cfg, pos, mass, rho, acc, _diag = droplet
    consistent = _discrete_laplace(bench, pos, mass, rho, cfg, acc)
    legacy_acc = compute_surface_tension_force(
        pos, rho, mass, cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx), cfg,
        scheme="legacy")
    legacy = _discrete_laplace(bench, pos, mass, rho, cfg, legacy_acc,
                               scheme="legacy")
    # Measured: consistent 1.226, legacy 0.446 at R/dx = 10.
    assert consistent["delta_s_integral"] > 0.9
    assert legacy["delta_s_integral"] < 0.7
    assert legacy["delta_s_integral"] == pytest.approx(legacy["grad_integral"])
    # Measured ratio 2.75; the naive "one scheme is just a bit better" story
    # would give a ratio of 1.
    assert consistent["delta_s_integral"] / legacy["delta_s_integral"] > 2.0


def test_raw_colour_gradient_integrates_to_one_half_in_a_one_phase_cloud(droplet):
    """The 1/2, not 1, is the correct target for a single-phase particle cloud.

    c runs from 1 in the bulk to ~1/2 on the outer particle layer (that layer
    sees half a kernel); the rest of the 1 -> 0 transition happens in the void,
    where there are no particles to carry it.  Comparing the one-phase
    ``int |grad c| dr`` with the two-phase criterion 1.0 is what made the old pin
    look like a 55 % error of the model rather than a wrong criterion.
    """
    bench, cfg, pos, mass, rho, acc, _diag = droplet
    measured = _discrete_laplace(bench, pos, mass, rho, cfg, acc)
    # Measured 0.613 at R/dx = 10 (theory 0.5, lattice +-10 %).
    assert 0.45 <= measured["grad_integral"] <= 0.75


def test_delta_s_integrates_to_unity_within_lattice_noise(droplet):
    """Measured 1.226 (criterion 1.0) at R/dx = 10; bound keeps the A/B meaning."""
    bench, cfg, pos, mass, rho, acc, _diag = droplet
    measured = _discrete_laplace(bench, pos, mass, rho, cfg, acc)
    assert 0.75 <= measured["delta_s_integral"] <= 1.45


def test_laplace_pressure_from_the_force_field_passes_at_r_over_dx_20():
    """The shipped scheme reaches the 15 % gate once the interface is resolved.

    Measured -10.8 % (R/dx = 20).  At R/dx = 10 the same operator gives +45.4 %,
    so resolution, not the force law, is what the low-resolution runs measured.
    """
    bench, cfg, pos, mass, rho, acc, _pairs = _accelerate(levels=HIGH_LEVELS)
    r = torch.linalg.vector_norm(pos, dim=-1)
    a_r = (acc * pos / r[:, None]).sum(dim=1)
    edges = torch.linspace(0.0, 1.2 * RADIUS, max(12, HIGH_LEVELS))
    centre = 0.5 * (edges[1:] + edges[:-1])
    ar_shell = bench.radial_bin(a_r, r, edges)
    integrand = torch.where(torch.isfinite(ar_shell), -RHO0 * ar_shell,
                            torch.zeros_like(ar_shell))
    dp = float(torch.trapezoid(integrand, centre))
    assert abs(dp / (SIGMA / RADIUS) - 1.0) <= 0.15


# --------------------------------------------------------------------------- #
# pins for the legacy scheme: still shipped, still wrong, measured with its own
# fields (this is what the pre-m2681 "mixed" numbers actually described)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "open gap, legacy scheme: delta_s = |grad c| with no normalisation, "
    "measured integral 0.446 instead of 1 (docs/benchmarks/"
    "laplace-verification.md)"))
def test_legacy_delta_s_integrates_to_unity(droplet):
    bench, cfg, pos, mass, rho, acc, _diag = droplet
    legacy_acc = compute_surface_tension_force(
        pos, rho, mass, cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx), cfg,
        scheme="legacy")
    measured = _discrete_laplace(bench, pos, mass, rho, cfg, legacy_acc,
                                 scheme="legacy")
    assert abs(measured["delta_s_integral"] - 1.0) <= 0.05


@pytest.mark.xfail(strict=True, reason=(
    "open gap, legacy scheme: measured kappa*R = 0.356 at R/dx = 10 "
    "(theory 1); the neighbour-count colour field has a 0.3 jump across a 1 "
    "particle layer (docs/benchmarks/laplace-verification.md)"))
def test_legacy_curvature_of_a_circle_is_one_over_R(droplet):
    bench, cfg, pos, mass, rho, acc, _diag = droplet
    legacy_acc = compute_surface_tension_force(
        pos, rho, mass, cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx), cfg,
        scheme="legacy")
    measured = _discrete_laplace(bench, pos, mass, rho, cfg, legacy_acc,
                                 scheme="legacy")
    assert abs(measured["kappa_times_R"] - 1.0) <= 0.15


@pytest.mark.xfail(strict=True, reason=(
    "open gap, legacy scheme: force-level dp is -64.2 % of sigma/R at "
    "R/dx = 10 (docs/benchmarks/laplace-verification.md)"))
def test_legacy_laplace_pressure_equals_sigma_over_R(droplet):
    bench, cfg, pos, mass, rho, acc, _diag = droplet
    legacy_acc = compute_surface_tension_force(
        pos, rho, mass, cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx), cfg,
        scheme="legacy")
    measured = _discrete_laplace(bench, pos, mass, rho, cfg, legacy_acc,
                                 scheme="legacy")
    assert abs(measured["dp_rel_error"]) <= 0.15