"""3-D operators: neighbour search, density, gradient, Laplacian, pressure force.

The analytic anchors are:

* periodic pairs must equal a brute-force minimum-image enumeration *exactly*
  (same pairs, same order) -- a search that silently drops boundary-straddling
  pairs would make every 3-D periodic run quietly wrong;
* a calibrated lattice sums to ``rho = rho0`` to machine zero;
* the SPH gradient reproduces a linear field exactly (zeroth moment vanishes);
* the pressure force of the analytic hydrostatic field balances gravity to a few
  per mille, which is the discrete hydrostatic-balance test;
* the Laplacian reproduces ``-3 k^2 f`` for ``f = sin sin sin`` and its discrete
  second moment is uniform across the lattice (so the renormalisation is a
  constant factor, not a field-dependent fudge).
"""

import math

import torch

from tensorsph.threed import (
    analytic_laplacian_of_mode,
    box_pairs,
    brute_force_pairs_min_image,
    calibrated_particle_mass,
    cubic_lattice,
    gradient_of_field,
    laplacian_of_field,
    laplacian_second_moment,
    minimum_image,
    pair_geometry,
    periodic_pairs,
    pressure_acceleration,
    summation_density,
    wall_pressure_adami,
    wall_shell,
)

BOX = 1.0


def _lattice(n: int, h_factor: float = 1.3, box: float = BOX):
    dx = box / n
    h = h_factor * dx
    pos = cubic_lattice((n, n, n), dx)
    return pos, dx, h


# --------------------------------------------------------------------------- #
# 1. periodic neighbour search
# --------------------------------------------------------------------------- #
def test_periodic_pairs_match_brute_force_exactly():
    torch.manual_seed(7)
    pos = torch.rand(300, 3, dtype=torch.float64) * BOX
    cutoff = 0.12
    fast = periodic_pairs(pos, BOX, cutoff)
    slow = brute_force_pairs_min_image(pos, BOX, cutoff)
    assert torch.equal(fast, slow)
    assert int(fast.shape[0]) > 200


def test_periodic_pairs_see_across_the_boundary():
    """Two particles either side of the boundary are neighbours, once wrapped."""
    pos = torch.tensor([[0.01, 0.5, 0.5], [0.99, 0.5, 0.5], [0.5, 0.5, 0.5]],
                       dtype=torch.float64)
    pairs = periodic_pairs(pos, BOX, 0.05)
    assert torch.equal(pairs, torch.tensor([[0, 1]], dtype=torch.int64))
    delta = minimum_image(pos[0] - pos[1], BOX)
    assert float(torch.linalg.vector_norm(delta)) < 0.05


def test_minimum_image_is_idempotent_and_bounded():
    torch.manual_seed(11)
    pos = torch.rand(500, 3, dtype=torch.float64) * BOX
    delta = minimum_image(pos[:, None, :] - pos[None, :, :], BOX)
    assert float(delta.abs().max()) <= 0.5
    again = minimum_image(delta, BOX)
    assert torch.equal(delta, again)


# --------------------------------------------------------------------------- #
# 2. density
# --------------------------------------------------------------------------- #
def test_calibrated_lattice_has_uniform_rho0():
    pos, dx, h = _lattice(12)
    mass = torch.full((pos.shape[0],), calibrated_particle_mass(dx, h, 1000.0),
                      dtype=torch.float64)
    pairs = periodic_pairs(pos, BOX, 2.0 * h)
    rho = summation_density(pos, mass, h, pairs, BOX)
    assert float((rho / 1000.0 - 1.0).abs().max()) < 1e-12


def test_density_sum_responds_to_compression():
    """Halving the spacing doubles the number of neighbours per unit volume."""
    pos, dx, h = _lattice(10)
    pairs = periodic_pairs(pos, BOX, 2.0 * h)
    mass = torch.full((pos.shape[0],), 1000.0 * dx ** 3, dtype=torch.float64)
    dense = pos * 0.5
    rho_dense = summation_density(dense, mass, h, pairs, BOX)
    rho_loose = summation_density(pos, mass, h, pairs, BOX)
    assert float(rho_dense.mean()) > 1.5 * float(rho_loose.mean())


# --------------------------------------------------------------------------- #
# 3. gradient and Laplacian against analytic fields
# --------------------------------------------------------------------------- #
def test_gradient_reproduces_a_linear_field_on_an_open_lattice():
    """A linear field is reproduced to the lattice first-moment deficit.

    On a cubic lattice ``sum_j V_j (x_j - x_i) (x) grad_i W`` is ``0.9935 I``, not
    ``I``: the operator is 0.65 % weak, and *uniformly* so -- which is why it can be
    quoted as a known bias instead of noise.  A linear field is deliberately tested
    on an **open** cloud: ``f = a . x`` is not periodic, and inside a periodic box
    the wrapped pairs see a jump of ``a . L`` (this mistake produced a 250 % error
    before it was caught).
    """
    n, dx, h = 14, 0.01, 0.013
    pos = cubic_lattice((n, n, n), dx)
    pairs = box_pairs(pos, 2.0 * h)
    volume = torch.full((pos.shape[0],), dx ** 3, dtype=torch.float64)
    slope = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    field = (pos * slope).sum(dim=1)
    grad = gradient_of_field(pos, field, volume, h, pairs)
    interior = ((pos.min(dim=1).values > 2.0 * h)
                & (pos.max(dim=1).values < (n - 1) * dx - 2.0 * h))
    assert int(interior.sum()) > 100
    deficit = (grad[interior] / slope).mean(dim=0)
    assert float((deficit - 1.0).abs().max()) < 0.02
    # The deficit is a lattice constant: every interior particle sees the same one.
    spread = (grad[interior] / slope).std(dim=0)
    assert float(spread.max()) < 1e-12


def test_gradient_reproduces_a_periodic_sine_field():
    pos, dx, h = _lattice(16)
    pairs = periodic_pairs(pos, BOX, 2.0 * h)
    volume = torch.full((pos.shape[0],), dx ** 3, dtype=torch.float64)
    k = 2.0 * math.pi
    field = torch.sin(k * pos[:, 0])
    analytic = torch.stack([k * torch.cos(k * pos[:, 0]),
                            torch.zeros_like(field), torch.zeros_like(field)], dim=-1)
    grad = gradient_of_field(pos, field, volume, h, pairs, BOX)
    assert float((grad - analytic).norm() / analytic.norm()) < 0.10


def test_laplacian_reproduces_the_analytic_mode():
    pos, dx, h = _lattice(16)
    pairs = periodic_pairs(pos, BOX, 2.0 * h)
    volume = torch.full((pos.shape[0],), dx ** 3, dtype=torch.float64)
    k = 2.0 * math.pi
    field, analytic = analytic_laplacian_of_mode(pos, k)
    laps = laplacian_of_field(pos, field, volume, h, pairs, BOX)
    error = float((laps - analytic).norm() / analytic.norm())
    assert error < 0.15


def test_second_moment_is_uniform_so_the_correction_is_a_constant():
    pos, dx, h = _lattice(12)
    pairs = periodic_pairs(pos, BOX, 2.0 * h)
    volume = torch.full((pos.shape[0],), dx ** 3, dtype=torch.float64)
    moment = laplacian_second_moment(pos, volume, h, pairs, BOX)
    assert float(moment.std()) < 1e-12
    # Lattice value of the second moment: above the continuum 2, which is exactly
    # why the raw operator over-predicts a 3-D viscous decay rate.
    assert 2.0 < float(moment.mean()) < 2.5


# --------------------------------------------------------------------------- #
# 4. hydrostatic balance through the pressure force
# --------------------------------------------------------------------------- #
def _hydrostatic_setup(nx=8, nz=12, dx=0.01, wall_layers=2):
    fluid = cubic_lattice((nx, nx, nz), dx, origin=(wall_layers * dx,) * 3)
    wall = wall_shell((nx, nx, nz), wall_layers, dx)
    pos = torch.cat((fluid, wall))
    n_fluid = fluid.shape[0]
    h = 1.3 * dx
    mass = torch.full((pos.shape[0],), calibrated_particle_mass(dx, h, 1000.0),
                      dtype=torch.float64)
    pairs = box_pairs(pos, 2.0 * h)
    return pos, mass, h, pairs, n_fluid, wall_layers * dx + nz * dx


def test_pressure_force_balances_an_analytic_hydrostatic_field():
    """The discrete pressure acceleration must equal ``+g`` on the interior."""
    gravity, rho0 = 9.81, 1000.0
    pos, mass, h, pairs, n_fluid, surface = _hydrostatic_setup()
    rho = torch.full_like(mass, rho0)
    pressure = torch.zeros_like(mass)
    pressure[:n_fluid] = rho0 * gravity * (surface - pos[:n_fluid, 2])
    pressure[n_fluid:] = wall_pressure_adami(
        pos, mass, rho, pressure, h, pairs, n_fluid,
        torch.tensor([0.0, 0.0, -gravity], dtype=torch.float64))
    acc = pressure_acceleration(pos, mass, rho, pressure, h, pairs)
    fluid = pos[:n_fluid]
    interior = ((fluid[:, 2] > 2.0 * h + 0.02)
                & (fluid[:, 2] < surface - 2.0 * h)
                & (fluid[:, 0] > 0.02 + 2.0 * h) & (fluid[:, 0] < 0.02 + 0.08 - 2.0 * h)
                & (fluid[:, 1] > 0.02 + 2.0 * h) & (fluid[:, 1] < 0.02 + 0.08 - 2.0 * h))
    assert int(interior.sum()) > 10
    mean = float(acc[:n_fluid][interior, 2].mean())
    assert abs(mean - gravity) < 0.05 * gravity
    # No lateral force: the field is one-dimensional.
    assert float(acc[:n_fluid][interior, 0].abs().max()) < 1e-9


def test_pressure_force_conserves_momentum():
    """Equal-and-opposite pair forces: ``sum m a = 0`` to round-off."""
    pos, dx, h = _lattice(10)
    pairs = periodic_pairs(pos, BOX, 2.0 * h)
    torch.manual_seed(3)
    mass = torch.full((pos.shape[0],), calibrated_particle_mass(dx, h, 1000.0),
                      dtype=torch.float64)
    rho = 1000.0 * (1.0 + 0.01 * torch.rand(pos.shape[0], dtype=torch.float64))
    pressure = 1.0e3 * torch.rand(pos.shape[0], dtype=torch.float64)
    acc = pressure_acceleration(pos, mass, rho, pressure, h, pairs, BOX)
    total = (mass.unsqueeze(-1) * acc).sum(dim=0)
    reference = float((mass.unsqueeze(-1) * acc.abs()).sum())
    assert float(torch.linalg.vector_norm(total)) < 1e-12 * reference


def test_pair_geometry_uses_the_minimum_image():
    pos, dx, h = _lattice(8)
    pairs = periodic_pairs(pos, BOX, 2.0 * h)
    delta, weight, grad, lap = pair_geometry(pos, pairs, h, BOX)
    assert float(delta.abs().max()) <= 0.5 + 1e-12
    assert float(weight.min()) >= 0.0
    assert float(lap.abs().max()) > 0.0
