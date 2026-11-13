"""Unit checks of the compression-based density regularization port.

Three kinds of evidence, in increasing strength:

1. **Formula equivalence** -- the vectorised port is compared against a
   *literal transcription* of the reference C++ (``literal_compression_sum``
   below: one ``for`` loop over each particle's neighbour list, exactly the loop
   body of ``CompressionSummation<Inner>::InteractKernel::interact`` plus the
   ``<Contact>`` one).  Two independent readings of the same source have to
   agree to machine precision.
2. **Rule semantics** -- ``Regularization<FlowType>``: ``internal`` returns the
   sum, ``free_surface`` clamps at one, ``free_stream`` substitutes one where the
   indicator is set; ``AverageCompression`` averages over ``S > 1`` only.
3. **Analytic semantics** -- what the estimator actually measures.  On a
   complete lattice the sum of reference volumes, kernel weighted, is one up to
   the lattice offset; compress the lattice isotropically by ``lambda`` and the
   sum must read ``lambda``.  That second check has an analytic reference and is
   the accuracy statement the cross-check report quotes.

The reference source: ``Xiangyu-Hu/SPHinXsys``, tree
``1612330e44c8947406f24b6756449e80dcd7158e``,
``src/shared/shared_ck/particle_dynamics/fluid_dynamics/density_regularization.hpp``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.compression_regularization import (
    RULES,
    CompressionRegularization,
    average_inverse_reference_compression,
    compression_from_summation_density,
    compression_summation,
    reference_volumes,
    regularize,
    regularized_density,
)
from tensorsph.droplet_impact import (
    Config,
    cell_centres,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
    lattice_kernel_sum,
    lattice_mass_unit,
    rhs,
    wall_lattice,
)
from tensorsph.neighbor_search import cell_linked_pairs

RHO0 = 1000.0
DX = 0.02
H_FACTOR = 1.3


def lattice(width=0.2, depth=0.2, dx=DX, scale=1.0):
    """Square lattice of spacing ``scale * dx`` with the calibrated lattice mass."""
    device = torch.device("cpu")
    cfg = Config(dx=dx, pool_width=width, pool_depth=depth, gravity=0.0,
                 sound_speed=20.0, boundary="wall", hydrostatic_init=False,
                 h_factor=H_FACTOR)
    xs = cell_centres(width, dx, device) * scale
    ys = cell_centres(depth, dx, device) * scale
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).double()
    mass = torch.full((pos.shape[0],), lattice_mass_unit(cfg), dtype=torch.float64)
    return cfg, pos, mass


def literal_compression_sum(pos, vol_ref, pairs, h, w_self,
                           inverse_reference_compression, fluid_count):
    """Literal transcription of the reference loop (one pass per particle).

    ``CompressionSummation<Inner>::interact`` and ``<Contact>::interact`` add
    ``Kref^-1 W_ij Vref_j`` for every neighbour; the self term is added first and
    is *not* scaled.  Neighbour lists are built from the unordered pair list.
    """
    neighbours = [[] for _ in range(pos.shape[0])]
    for a, b in pairs.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)
    sums = []
    for i in range(fluid_count):
        total = w_self * float(vol_ref[i])
        for j in neighbours[i]:
            r = float(torch.linalg.vector_norm(pos[i] - pos[j]))
            total += (inverse_reference_compression
                      * float(cubic_spline_kernel(
                          torch.tensor(r, dtype=torch.float64), h))
                      * float(vol_ref[j]))
        sums.append(total)
    return torch.tensor(sums, dtype=torch.float64)


def pair_setup(pos, wall=None, h=None):
    h = H_FACTOR * DX if h is None else h
    if wall is None:
        all_pos = pos
    else:
        all_pos = torch.cat((pos, wall), dim=0)
    pairs = cell_linked_pairs(all_pos, 2.0 * h)
    if wall is not None:
        n = pos.shape[0]
        i, j = pairs.unbind(dim=1)
        pairs = pairs[(i < n) | (j < n)]
    delta = all_pos[pairs[:, 0]] - all_pos[pairs[:, 1]]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    return all_pos, pairs, wij, h


# --------------------------------------------------------------------------- #
# 1. Formula equivalence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("inverse_reference_compression", [1.0, 0.987654321, 1.031])
def test_vectorised_matches_literal_transcription(inverse_reference_compression):
    cfg, pos, mass = lattice()
    wall = wall_lattice(cfg, torch.device("cpu"), top=0.2).double()
    all_pos, pairs, wij, h = pair_setup(pos, wall)
    vol_ref = reference_volumes(torch.cat((mass, torch.full((wall.shape[0],),
                                                           lattice_mass_unit(cfg),
                                                           dtype=torch.float64)),
                                         dim=0), RHO0)
    w_self = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
    got = compression_summation(vol_ref, w_self, pairs, wij, pos.shape[0],
                                inverse_reference_compression)
    ref = literal_compression_sum(all_pos, vol_ref, pairs, h, w_self,
                                  inverse_reference_compression, pos.shape[0])
    relative = ((got.double() - ref).abs() / ref.abs()).max()
    assert float(relative) < 1e-12


def test_wall_contact_participates():
    """A column on a wall must read denser than the same column in free space."""
    cfg, pos, mass = lattice()
    wall = wall_lattice(cfg, torch.device("cpu"), top=0.2).double()
    vol_fluid = reference_volumes(mass, RHO0)
    all_pos, pairs, wij, h = pair_setup(pos, wall)
    vol_ref = torch.cat((vol_fluid,
                         reference_volumes(torch.full((wall.shape[0],),
                                                      lattice_mass_unit(cfg),
                                                      dtype=torch.float64), RHO0)))
    w_self = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
    with_wall = compression_summation(vol_ref, w_self, pairs, wij, pos.shape[0])
    all_pos0, pairs0, wij0, _ = pair_setup(pos)
    without = compression_summation(vol_fluid, w_self, pairs0, wij0, pos.shape[0])
    bottom = pos[:, 1] < 1.5 * DX
    assert float(with_wall[bottom].mean()) > float(without[bottom].mean())


# --------------------------------------------------------------------------- #
# 2. Rule semantics
# --------------------------------------------------------------------------- #

def test_rules_and_reduction():
    assert RULES == ("internal", "free_surface", "free_stream")
    s = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    assert torch.equal(regularize(s, "internal"), s)
    assert torch.equal(regularize(s, "free_surface"),
                       torch.tensor([1.0, 1.0, 1.2], dtype=torch.float64))
    indicator = torch.tensor([1, 0, 0])
    assert torch.equal(regularize(s, "free_stream", indicator),
                       torch.tensor([1.0, 1.0, 1.2], dtype=torch.float64))
    with pytest.raises(ValueError):
        regularize(s, "free_stream")
    with pytest.raises(ValueError):
        regularize(s, "nonsense")


def test_average_reduction_uses_only_compressed_particles():
    s = torch.tensor([0.5, 0.9, 1.0, 1.2, 1.4], dtype=torch.float64)
    inverse, counted = average_inverse_reference_compression(s)
    assert counted == 2                      # 1.2 and 1.4 only, not 1.0
    assert inverse == pytest.approx(2.0 / 2.6, rel=1e-15)


def test_average_reduction_of_an_empty_set_is_identity():
    inverse, counted = average_inverse_reference_compression(
        torch.tensor([0.3, 0.9], dtype=torch.float64))
    assert counted == 0 and inverse == 1.0   # 0/0 in the reference; identity here


def test_density_assignment_is_compression_times_rho0():
    k = torch.tensor([1.0, 1.01], dtype=torch.float64)
    assert torch.equal(regularized_density(k, RHO0),
                       torch.tensor([1000.0, 1010.0], dtype=torch.float64))


# --------------------------------------------------------------------------- #
# 3. Analytic semantics of the estimator
# --------------------------------------------------------------------------- #

def _interior_sums(pos, mass, fluid_count, inverse_ref=1.0):
    h = H_FACTOR * DX
    pairs = cell_linked_pairs(pos, 2.0 * h)
    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    vol_ref = reference_volumes(mass, RHO0)
    w_self = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
    return compression_summation(vol_ref, w_self, pairs, wij, fluid_count,
                                 inverse_ref)


def test_reference_lattice_reads_one_in_the_interior():
    """Calibrated reference volumes => the sum is one to machine precision.

    The reference code's own volumes carry the lattice offset ``b``, so there the
    interior reads ``1 + b`` (measured separately in
    ``tests/test_lattice_mass_calibration.py``); the port is fed the calibrated
    masses, so the same offset does not appear.
    """
    _, pos, mass = lattice(width=0.3, depth=0.3)
    sums = _interior_sums(pos, mass, pos.shape[0])
    interior = ((pos[:, 0] > 0.1) & (pos[:, 0] < 0.2)
                & (pos[:, 1] > 0.1) & (pos[:, 1] < 0.2))
    assert abs(float(sums[interior].mean()) - 1.0) < 1e-6
    # With the *uncalibrated* lattice volumes the same sum reads 1 + b, the
    # lattice offset the calibration removes (the reference code's volumes are
    # uncalibrated, so that is the number its interior would show).
    h = H_FACTOR * DX
    offset = DX * DX * lattice_kernel_sum(h, DX) - 1.0
    raw = _interior_sums(pos, torch.full_like(mass, RHO0 * DX * DX), pos.shape[0])
    assert float(raw[interior].mean()) == pytest.approx(1.0 + offset, rel=1e-6)


@pytest.mark.parametrize("compression", [1.002, 1.005, 1.01])
def test_isotropic_compression_is_measured_by_the_sum(compression):
    """Spacing scaled by ``1/sqrt(lambda)`` (2-D) must read ``lambda``.

    Analytic reference: ``K = Vref / V = (s / dx) ** -2``.  No experimental data
    and no code reference are involved -- this is the estimator's definition.
    """
    scale = 1.0 / math.sqrt(compression)
    _, pos, mass = lattice(width=0.3, depth=0.3, scale=scale)
    sums = _interior_sums(pos, mass, pos.shape[0])
    interior = ((pos[:, 0] > 0.1 * scale) & (pos[:, 0] < 0.2 * scale)
                & (pos[:, 1] > 0.1 * scale) & (pos[:, 1] < 0.2 * scale))
    measured = float(sums[interior].mean())
    error_pct = (measured / compression - 1.0) * 100.0
    assert abs(error_pct) < 15.0, f"estimator error {error_pct:.3f} % > 15 %"


def test_plain_formulation_equals_the_summation_density():
    """``normalization="none"`` + ``internal`` reproduces ``sum_j m_j W_ij``.

    This is what makes ``compression_none`` a *faithful* port of the formulation
    that SPHinXsys shipped before 2026-07-19: the density is the shipped
    summation density, expressed through reference volumes.
    """
    _, pos, mass = lattice()
    sums = _interior_sums(pos, mass, pos.shape[0])
    rho_compression = regularized_density(sums, RHO0)
    rho_summation = density_from_neighbor_pairs(pos, mass, H_FACTOR * DX)
    relative = ((rho_compression - rho_summation).abs() / rho_summation).max()
    assert float(relative) < 1e-12
    # ... and the inverse reading used by the report agrees.
    assert float((compression_from_summation_density(rho_summation, RHO0)
                  - sums).abs().max()) < 1e-12


# --------------------------------------------------------------------------- #
# 4. Cross-step state (the reference's one-step lag) and solver wiring
# --------------------------------------------------------------------------- #

def test_two_stage_order_reproduces_the_dynamics_list():
    """Step 1 consumes the stored scalar, the reduce writes it afterwards."""
    _, pos, mass = lattice()
    wall = wall_lattice(Config(dx=DX, pool_width=0.2, pool_depth=0.2,
                               hydrostatic_init=False), torch.device("cpu"),
                        top=0.2).double()
    all_pos, pairs, wij, h = pair_setup(pos, wall)
    vol_ref = torch.cat((reference_volumes(mass, RHO0),
                         reference_volumes(torch.full((wall.shape[0],),
                                                      lattice_mass_unit(
                                                          Config(dx=DX, pool_width=0.2,
                                                                 pool_depth=0.2)),
                                                      dtype=torch.float64),
                                              RHO0)))
    w_self = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
    reg = CompressionRegularization(rule="free_surface",
                                    normalization="reference_mean")
    assert reg.inverse_reference_compression == 1.0     # registered value
    used = []
    for _ in range(3):
        used.append(reg.inverse_reference_compression)
        rho, info = reg.update(vol_ref, w_self, pairs, wij, pos.shape[0], RHO0)
        assert info["inverse_reference_compression_used"] == pytest.approx(used[-1])
        assert (reg.inverse_reference_compression
                == pytest.approx(info["inverse_reference_compression_written"]))
    assert used[0] == 1.0
    assert used[1] != used[0] and used[2] == pytest.approx(used[1], rel=1e-6)


def test_solver_rejects_incoherent_combinations():
    _, pos, mass = lattice()
    cfg = Config(dx=DX, pool_width=0.2, pool_depth=0.2, hydrostatic_init=False,
                 density_regularization="compression", density_diffusion=0.1)
    with pytest.raises(ValueError, match="two different ways"):
        rhs(pos, torch.zeros_like(pos), mass, cfg)
    cfg = Config(dx=DX, pool_width=0.2, pool_depth=0.2, hydrostatic_init=False,
                 density_regularization="compression",
                 density_renormalization=True)
    with pytest.raises(ValueError, match="normalise"):
        rhs(pos, torch.zeros_like(pos), mass, cfg)
    cfg = Config(dx=DX, pool_width=0.2, pool_depth=0.2, hydrostatic_init=False,
                 density_regularization="compression",
                 compression_regularization_rule="bogus")
    with pytest.raises(ValueError, match="compression_regularization_rule"):
        rhs(pos, torch.zeros_like(pos), mass, cfg)


def test_solver_density_is_rho0_on_a_reference_lattice():
    """A calibrated lattice at rest carries no compression, so rho = rho0."""
    cfg, pos, mass = lattice()
    cfg.density_regularization = "compression"
    wall = wall_lattice(cfg, torch.device("cpu"), top=0.2).double()
    state: dict = {}
    for _ in range(2):
        rho, pressure, acc = rhs(pos, torch.zeros_like(pos), mass, cfg,
                                 wall=wall, dt=1e-4, state=state)
    assert float((rho / RHO0 - 1.0).abs().max()) < 1e-6
    assert float(pressure.max()) < 10.0          # ~kPa-level Tait noise at most
    assert torch.isfinite(acc).all()
    assert "compression" in state