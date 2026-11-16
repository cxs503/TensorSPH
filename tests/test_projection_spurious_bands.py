"""Gates for the radial band decomposition of the spurious-current reading.

The declared spurious-current gate of the dynamic ladder is a single number,
``max|u|/v_ref`` over all particles.  ``radial_band_stats`` splits that number
by radius so the reading can be attributed (liquid bulk vs interface shell vs
vapour carrier).  These tests pin the *instrument*, not the physics:

* the bands must partition the cloud (no particle counted twice, none dropped);
* the band edges must be inclusive on the interface shell, because the shell is
  the region the claim is about;
* exact rest must read zero and an undefined direction must read ``nan`` -- not
  a fabricated ``0.0``;
* the switch must be off by default, so the shipped ladder records keep their
  schema (a record whose keys silently changed is not reproducible).

No particle solve is performed here: every field is synthetic, so the test cost
is milliseconds and the assertions cannot depend on a solver's tolerances.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import torch

from tensorsph.projection_adjoint import (
    BAND_EDGES,
    radial_band_stats,
    relax_projection_consistent,
)

REPO = Path(__file__).resolve().parents[1]


def _cloud(ratios, speeds, radius=1.0, v_ref=1.0):
    """Axis-aligned particles at ``r = ratio * radius`` with the given speeds."""
    pos = torch.zeros((len(ratios), 2), dtype=torch.float64)
    u = torch.zeros_like(pos)
    for k, (ratio, speed) in enumerate(zip(ratios, speeds)):
        pos[k, 0] = float(ratio) * radius
        u[k, 0] = float(speed) * v_ref
    return pos, u


def test_bands_partition_the_cloud_and_edges_are_inclusive():
    lo, hi = BAND_EDGES
    ratios = [0.1, lo - 1e-9, lo, 0.99, hi, hi + 1e-9, 2.0]
    pos, u = _cloud(ratios, [0.01] * len(ratios), radius=1.0, v_ref=1.0)
    st = radial_band_stats(pos, u, 1.0, 1.0)

    total = sum(st[b]["n"] for b in
                ("liquid_core", "interface_shell", "vapour_carrier"))
    assert total == len(ratios), "a particle was dropped or double counted"
    # inclusive on both edges: the shell is what the claim is about
    assert st["liquid_core"]["n"] == 2          # 0.1 and lo-eps
    assert st["interface_shell"]["n"] == 3      # lo, 0.99, hi
    assert st["vapour_carrier"]["n"] == 2       # hi+eps, 2.0


def test_exact_rest_reads_zero_and_undefined_direction_reads_nan():
    pos, u = _cloud([0.3, 1.0, 1.5], [0.0, 0.0, 0.0], radius=1.0, v_ref=0.5)
    st = radial_band_stats(pos, u, 1.0, 0.5)
    for band in ("liquid_core", "interface_shell", "vapour_carrier"):
        assert st[band]["n"] >= 1
        assert st[band]["max_over_vref"] == 0.0
        assert st[band]["mean_over_vref"] == 0.0
    # no moving particle => no direction to attribute; must not be reported as 0
    assert math.isnan(st["umax_radial_fraction"])


def test_a_shell_peak_is_not_smeared_into_the_bulk_mean():
    """The reason the split exists: one number cannot distinguish the two cases."""
    # case A: interface-shell peak, bulk at rest
    pos_a, u_a = _cloud([0.5, 0.6, 0.9, 1.0, 1.1], [0.0, 0.0, 0.3, 0.3, 0.0],
                        radius=1.0, v_ref=1.0)
    a = radial_band_stats(pos_a, u_a, 1.0, 1.0)
    # case B: same global max, but the current sits in the liquid bulk
    pos_b, u_b = _cloud([0.5, 0.6, 0.9, 1.0, 1.1], [0.0, 0.3, 0.0, 0.0, 0.0],
                        radius=1.0, v_ref=1.0)
    b = radial_band_stats(pos_b, u_b, 1.0, 1.0)

    assert max(a[x]["max_over_vref"] for x in
               ("liquid_core", "interface_shell", "vapour_carrier")) == 0.3
    assert max(b[x]["max_over_vref"] for x in
               ("liquid_core", "interface_shell", "vapour_carrier")) == 0.3
    # identical global reading, opposite attribution
    assert a["liquid_core"]["max_over_vref"] == 0.0
    assert b["liquid_core"]["max_over_vref"] == 0.3
    assert a["interface_shell"]["max_over_vref"] == 0.3
    assert b["interface_shell"]["max_over_vref"] == 0.0


def test_radial_fraction_separates_breathing_from_a_tangential_current():
    # radial: velocity along the radius (a breathing mode, moves the area)
    pos_r = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    u_r = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
    assert radial_band_stats(pos_r, u_r, 1.0, 1.0)["umax_radial_fraction"] == 1.0
    # tangential: orthogonal to the radius (a lattice current, area-neutral)
    u_t = torch.tensor([[0.0, 0.2]], dtype=torch.float64)
    assert radial_band_stats(pos_r, u_t, 1.0, 1.0)["umax_radial_fraction"] == 0.0


def test_band_report_is_off_by_default_and_the_shipped_records_are_unchanged():
    sig = inspect.signature(relax_projection_consistent)
    assert sig.parameters["band_report"].default is False

    shipped = sorted((REPO / "benchmarks" / "laplace_sph" /
                      "results_dynamic_ladder").glob("dx*_nu*.json"))
    assert shipped, "no ladder record in the tree to check the schema against"
    for path in shipped:
        rec = json.loads(path.read_text())
        for sample in rec["history"]:
            assert "bands" not in sample, (
                "%s carries band data although the switch defaults off" % path.name)


def test_profile_covers_every_particle_exactly_once():
    ratios = [0.2, 0.5, 0.9, 1.0, 1.1, 1.6, 1.9]
    pos, u = _cloud(ratios, [0.1] * len(ratios), radius=1.0, v_ref=1.0)
    st = radial_band_stats(pos, u, 1.0, 1.0, profile=True, profile_bins=8)
    assert len(st["profile"]) == 8
    assert sum(row["n"] for row in st["profile"]) == len(ratios)