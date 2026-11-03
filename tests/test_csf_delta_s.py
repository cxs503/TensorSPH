"""delta_s normalisation: analytic-profile estimator and the operator bias behind
the 25 % deficit recorded in docs/verification_surface_tension_curvature.md §7.1.

The verification record scored ``int delta_s dn = 0.74-0.76`` and proposed three
field-side remedies.  ``benchmarks/csf_delta_s/run.py`` measures what is really
going on; these tests pin the findings so they cannot silently drift:

1. the tabulated mollified-indicator profile is exact (``int dc/dd dd = 1/2``);
2. the MLS lattice derivative under-reads ``|grad c|`` by ~24 % **at every
   resolution** -- a structural operator bias, not truncation error;
3. the profile estimator removes that bias (normalisation inside 15 %, and
   inside 3 % at the two finer resolutions) but *then* the cut-force identity
   overshoots by 36-56 %, i.e. the old deficit was cancelling an over-count in
   the force assembly -- so the estimator stays opt-in, never the default;
4. the colour *field* is not the culprit (RMS 0.03 against the exact indicator),
   and the three remedies proposed by the previous handoff all make the
   normalisation worse.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import csf_fields, kernel_sum_color
from tensorsph.csf_delta_s import kernel_profile, profile_delta_s
from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "csf_delta_s"


def _bench():
    spec = importlib.util.spec_from_file_location("csf_delta_s_run", BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench()


@pytest.fixture(scope="module")
def coarse(bench):
    """Static evaluation at the coarsest ladder point -- a couple of seconds."""
    gradient = bench.score(dx=0.003, delta_s_scheme="gradient", with_analytic=True)
    profile = bench.score(dx=0.003, delta_s_scheme="profile")
    return {"gradient": gradient, "profile": profile,
            "analytic": gradient["analytic"]}


# --------------------------------------------------------------------------- #
# the analytic indicator profile
# --------------------------------------------------------------------------- #
def test_indicator_profile_tabulation():
    """c(0) = 1/2 at the interface, c -> 1 at the support edge, integral = 1/2."""
    d, c, dc_dd = kernel_profile(1.3 * 0.002)
    assert float(c[0]) == pytest.approx(0.5, abs=1e-12)
    assert float(c[-1]) == pytest.approx(1.0, abs=1e-3)
    integrand = torch.trapezoid(dc_dd, d)
    assert float(integrand) == pytest.approx(0.5, abs=1e-3)
    # monotone: the inversion in profile_delta_s relies on it
    assert bool((c[1:] >= c[:-1]).all())
    # |grad c| peaks on the interface and decays to zero at the support edge
    assert float(dc_dd[0]) > 10.0 * float(dc_dd[-1])


def test_profile_delta_s_inverts_the_profile():
    """Feeding the tabulated c back in must return the tabulated gradient."""
    h = 1.3 * 0.002
    d, c, dc_dd = kernel_profile(h)
    idx = torch.tensor([1, 64, 128, 256, 384, 480])
    back = profile_delta_s(c[idx], h)
    assert torch.allclose(back, 2.0 * dc_dd[idx], rtol=0.08)


def test_profile_delta_s_bulk_is_zero():
    """A flat colour field carries no interface: delta_s must vanish there."""
    out = profile_delta_s(torch.ones(16, dtype=torch.float64), 1.3 * 0.002)
    assert float(out.max()) < 1.0e-2


# --------------------------------------------------------------------------- #
# the operator bias and the estimator that removes it
# --------------------------------------------------------------------------- #
def test_mls_gradient_underreads_the_analytic_gradient(coarse):
    """The bias is structural: ratio 0.758 / 0.760 / 0.764 at R/h = 5.1/7.7/10.3,
    i.e. it does not shrink with resolution."""
    ratio = coarse["analytic"]["gradient_ratio_mls_over_analytic"]
    assert 0.70 < ratio < 0.80


def test_analytic_gradient_through_the_same_quadrature_normalises(coarse):
    """Quadrature is not the culprit: with the exact |grad c| at the particles the
    shell integral is 1."""
    val = coarse["analytic"]["integral_analytic_gradient"]
    assert val == pytest.approx(1.0, abs=0.15)


def test_profile_estimator_normalises_the_delta_function(coarse):
    """delta_s normalisation: 0.755 (gradient) -> 0.986 (profile) at dx = 3 mm."""
    grad = coarse["gradient"]["delta_s"]["integral_dn"]
    prof = coarse["profile"]["delta_s"]["integral_dn"]
    assert grad < 0.85                     # the recorded debt is real
    assert prof == pytest.approx(1.0, abs=0.15)
    assert prof > grad + 0.15


def test_colour_field_is_not_the_culprit(coarse):
    """The kernel-sum colour differs from the exact indicator by ~3 % RMS."""
    rms = coarse["analytic"]["color_rms_kernelsum_vs_exact"]
    assert rms < 0.05


def test_profile_estimator_overshoots_the_cut_force(coarse):
    """Honest bookkeeping: with a correctly normalised delta_s the cut-force
    identity overshoots (1.56 at dx = 3 mm), so the previous 0.75 deficit was
    compensating an over-count in the force assembly.  The estimator therefore
    stays opt-in and this test records why."""
    ratio = coarse["profile"]["cut_force"]["ratio_measured_over_theory"]
    assert ratio > 1.2
    assert coarse["gradient"]["cut_force"]["within_15pct"] is True


# --------------------------------------------------------------------------- #
# remedies proposed by the previous handoff are refuted by measurement
# --------------------------------------------------------------------------- #
def test_field_side_remedies_do_not_help(bench):
    """Ghost colour, narrower colour kernel and Tikhonov MLS all score worse than
    the plain gradient estimator (0.755): 0.39 / 0.39-0.72 / 0.74-0.75."""
    base = bench.score(dx=0.003, delta_s_scheme="gradient")["delta_s"]["integral_dn"]
    narrow = bench.remedy_narrow_colour(0.02, 0.003)
    tik = bench.remedy_tikhonov(0.02, 0.003)
    ghost = bench.remedy_ghost(0.02, 0.003)["integral_dn"]
    assert max(narrow.values()) <= base + 0.05
    assert max(tik.values()) <= base + 0.05
    assert ghost < base - 0.2


# --------------------------------------------------------------------------- #
# backward compatibility of the physics path
# --------------------------------------------------------------------------- #
def test_default_stays_bit_exact():
    """csf_delta_s defaults to "gradient": adding the profile option must not
    move a single number on the historical path."""
    cfg = Config(dx=0.003, gravity=0.0, surface_tension_coefficient=7.2,
                 csf_scheme="consistent")
    assert cfg.csf_delta_s == "gradient"
    pos = torch.tensor([[0.0, 0.0], [0.003, 0.0], [0.0, 0.003], [0.003, 0.003]],
                       dtype=torch.float64)
    mass = torch.full((4,), 1000.0 * 0.003 * 0.003, dtype=torch.float64)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    a = csf_fields(pos, mass, pairs, h, 1000.0)
    b = csf_fields(pos, mass, pairs, h, 1000.0, delta_s_scheme="gradient")
    assert torch.equal(a.delta_s, b.delta_s)
    c = csf_fields(pos, mass, pairs, h, 1000.0, delta_s_scheme="profile")
    assert c.delta_s.shape == a.delta_s.shape


def test_invalid_scheme_is_rejected():
    cfg = Config(dx=0.003, csf_delta_s="bogus")
    with pytest.raises(ValueError, match="csf_delta_s"):
        cfg.validate() if hasattr(cfg, "validate") else _raise(cfg)


def _raise(cfg):
    from tensorsph.droplet_impact import validate_config
    validate_config(cfg)


# --------------------------------------------------------------------------- #
# committed artifact consistency (no compute)
# --------------------------------------------------------------------------- #
def test_committed_report_matches_the_record():
    rep = json.loads((BENCH / "validation_report.json").read_text())
    tab = rep["tabulation"]
    assert tab["c_at_interface"] == pytest.approx(0.5, abs=1e-12)
    assert tab["integral_delta_s_dn"] == pytest.approx(1.0, abs=2e-3)
    ladder = {(round(r["dx_m"], 5), r["delta_s_scheme"]): r
              for r in rep["resolution_ladder"]}
    for dx in (0.003, 0.002, 0.0015):
        g = ladder[(dx, "gradient")]
        p = ladder[(dx, "profile")]
        assert g["delta_s"]["integral_dn"] < 0.85, dx
        assert p["delta_s"]["integral_dn"] == pytest.approx(1.0, abs=0.05), dx
        # the honest verdict: normalisation fixed, force amplitude overshoots
        assert p["cut_force"]["ratio_measured_over_theory"] > 1.2, dx
        assert g["direction"]["cos_band_median"] > 0.99, dx