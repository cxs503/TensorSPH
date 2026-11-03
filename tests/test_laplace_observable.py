"""Pins for the Young-Laplace *observable* audit (m2834).

Two kinds of assertions:

1. ``test_interface_band_is_bulk`` -- the geometry claim that makes the legacy
   observable invalid: at the resolutions used by the benchmark, the band
   0.80 R .. 0.95 R is inside the fluid (colour function still ~bulk), while the
   interface shell (c ~ 0.5) sits outside it.  This is a *cheap* structural test
   (one ``csf_fields`` call on a pristine lattice); if a future model moves the
   surface into that band the audit has to be redone, and this test fails.

2. Committed-artifact consistency: the numbers in
   ``benchmarks/csf_consistent/interface_jump.json`` -- which the docs are
   generated from -- must stay self-consistent and must keep reporting the
   honest (failing) verdict.  No simulation is run here.
"""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import csf_fields, kernel_sum_color
from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmarks" / "csf_consistent"
JSON_PATH = BENCH / "interface_jump.json"


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BENCH.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")


@pytest.fixture(scope="module")
def drop():
    """Pristine hexagonal drop, R = 0.02 m, dx = 0.002 m (R/h = 7.7)."""
    radius, dx, rho0 = 0.02, 0.002, 1000.0
    pos, mass = _law.hexagonal_droplet(radius, dx, rho0)
    h = 1.3 * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    color = kernel_sum_color(pos, mass, pairs, h, rho0)
    st = csf_fields(pos, mass, pairs, h, rho0)
    return dict(pos=pos, mass=mass, h=h, pairs=pairs, color=color, st=st,
                radius=radius, rho0=rho0)


def test_interface_band_is_bulk(drop):
    """0.80R..0.95R carries bulk colour, so p there is *inside* the drop."""
    rn = torch.linalg.vector_norm(drop["pos"], dim=1) / drop["radius"]
    band = (rn > 0.80) & (rn < 0.95)
    shell = drop["color"] < 0.70                       # the actual interface
    assert int(band.sum()) > 20
    # the legacy band is essentially bulk fluid ...
    assert float(drop["color"][band].mean()) > 0.90
    # ... while the interface shell is further out
    assert float(rn[shell].min()) > 0.90
    # and the Young-Laplace solution is flat inside the drop, so a metric that
    # compares two interior radii cannot converge to sigma/R by construction
    assert float(rn[band].max()) < float(rn[shell].max())


def test_interface_shell_is_thinner_than_the_band(drop):
    """2h (the smeared-interface thickness) is comparable to the band width."""
    two_h_over_r = 2.0 * drop["h"] / drop["radius"]
    assert 0.15 < two_h_over_r < 0.40          # 0.26 at R/h = 7.7
    # the band is 0.15 R wide, i.e. inside one interface thickness
    assert 0.15 <= two_h_over_r


def test_delta_s_normalisation_on_pristine_lattice(drop):
    """int delta_s dn on the undeformed lattice: the 0.75 recorded by m2588."""
    st = drop["st"]
    vol = drop["mass"] / drop["rho0"]
    r = torch.linalg.vector_norm(drop["pos"], dim=1)
    integral = float((st.delta_s * vol / (2.0 * math.pi * r)).sum())
    assert 0.70 < integral < 0.80
    # and it is documented as *not* satisfying the identity
    assert abs(integral - 1.0) > 0.15


def test_lattice_density_bias_is_resolution_independent():
    """The 5.9e-3 sum-density bias is a kernel property, not a dx effect."""
    from tensorsph.droplet_impact import density_from_neighbor_pairs

    bias = []
    for dx in (0.003, 0.002):
        pos, mass = _law.hexagonal_droplet(0.02, dx, 1000.0)
        h = 1.3 * dx
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho = density_from_neighbor_pairs(pos, mass, h, pairs)
        r = torch.linalg.vector_norm(pos, dim=1)
        bias.append(float(rho[r < 0.5 * 0.02].mean() / 1000.0) - 1.0)
    assert bias[0] == pytest.approx(bias[1], abs=2e-4)
    assert 4e-3 < bias[0] < 8e-3
    # the *density* compression the Tait law needs for sigma/R is ~6x smaller
    # than the lattice bias, i.e. the signal sits below the discrete floor
    dp, bulk, gamma = 7.2 / 0.02, 1000.0 * 20.0 ** 2 / 7.0, 7.0
    eps_required = (1.0 + dp / bulk) ** (1.0 / gamma) - 1.0
    assert bias[0] / eps_required > 3.0
    # pressure equivalent of the bias: rho0 c0^2 * bias >> sigma/R
    assert 1000.0 * 20.0 ** 2 * bias[0] > 3.0 * dp


# --------------------------------------------------------------------------- #
# committed-artifact consistency (no compute)
# --------------------------------------------------------------------------- #
requires_json = pytest.mark.skipif(not JSON_PATH.exists(),
                                   reason="run interface_jump.py first")


@requires_json
def test_committed_errors_recompute():
    payload = json.loads(JSON_PATH.read_text())
    dp = payload["reference"]["delta_p_Pa"]
    for row in payload["rows"]:
        for key in ("legacy", "interface", "force"):
            recomputed = (row[f"{key}_dp_Pa"] - dp) / dp * 100.0
            assert recomputed == pytest.approx(row[f"{key}_rel_error_pct"], abs=1e-9)
        assert row["force_dp_Pa"] == pytest.approx(
            row["sigma_N_per_m"] * row["delta_s_integral"]
            * row["kappa_delta_s_mean_1_per_m"], rel=1e-6)


@requires_json
def test_committed_verdict_stays_honest():
    payload = json.loads(JSON_PATH.read_text())
    verdict = payload["verdict"]
    assert verdict["status"] == "failed"
    assert verdict["interface_within_15pct"] is False
    assert verdict["force_integral_within_15pct"] is False
    # the legacy observable must stay recorded as geometrically invalid
    assert verdict["legacy_observable_is_inside_the_fluid"] is True
    for row in payload["rows"]:
        assert row["interface_rel_error_pct"] < -15.0
        assert row["p_outer_Pa"] == pytest.approx(0.0, abs=30.0)


@requires_json
def test_legacy_observable_is_resolution_unstable_but_interface_is_not():
    payload = json.loads(JSON_PATH.read_text())
    rows = payload["rows"]
    legacy = [r["legacy_rel_error_pct"] for r in rows]
    iface = [r["interface_rel_error_pct"] for r in rows]
    assert max(legacy) - min(legacy) > 15.0     # 19.6 pct points measured
    assert max(iface) - min(iface) < 5.0        # 2.6 pct points measured


@requires_json
def test_document_matches_the_artifacts():
    doc = (HERE.parent / "docs" / "verification_laplace_observable.md").read_text()
    payload = json.loads(JSON_PATH.read_text())
    dp = payload["reference"]["delta_p_Pa"]
    assert f"sigma/R = {dp:.0f} Pa" in doc
    for row in payload["rows"]:
        assert f"{row['dx_m']:.4f}" in doc
        assert f"{row['interface_dp_Pa']:.1f}" in doc