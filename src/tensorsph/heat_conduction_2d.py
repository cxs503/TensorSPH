"""Two-dimensional conduction operator: does the 1-D operator (and its lattice
calibration) transfer to two dimensions?

Why this module exists
----------------------
:mod:`tensorsph.heat_conduction` verifies the repository's SPH conduction
operator (Cleary & Monaghan 1999, *J. Comput. Phys.* **148**:227-264,
doi:10.1006/jcph.1998.6118) against the 1-D ``erfc`` similarity solution and
calibrates it with :func:`tensorsph.ice_thermal.sph_thermal.lattice_diffusion_factor`.
That single benchmark family is the whole accuracy evidence for the
``heat_conduction_operator`` entry of the method-selection registry
(``docs/reports/method_selection_registry.json``): the upper bound on the
method rests on *one* geometry, *one* reference family and *one* dimension.

This module supplies a second, independent family -- the decay of a Fourier mode
of the heat equation in a periodic *square* -- and, in doing so, answers a
question the 1-D case cannot ask: **is the pair formula dimensionally
consistent?**  It is not, naively.  The shipped formula

.. math::

    \frac{dT_i}{dt} = \frac{2}{\rho c_p}\sum_j V_j\,
                     \frac{2k_i k_j}{k_i + k_j}\,F_{ij}\,(T_i - T_j),
    \qquad F_{ij} = \frac{\boldsymbol r_{ij}\cdot\nabla_i W_{ij}}{r_{ij}^2+\eta^2}

carries the factor 2 and the volume ``V_j = dx`` of a *one-dimensional* layer
sum (:math:`\sum_j V_j r\,dW/dr \to \int W\,dx = 1` in 1-D).  Lifted to 2-D with
``V_j = dx^2`` the layer sum no longer reproduces that moment: the measured
effective diffusivity of the raw operator is

.. math::

    \frac{\alpha_{\rm eff}}{\alpha}\Big|_{\rm 2D} \approx 1.92\,dx
    \quad (O(dx), \text{ i.e. } \to 0 \text{ as } dx \to 0),

so a 2-D run that inherits the 1-D calibration factor (1.0218) diffuses ~``1/dx``
too slowly (measured below: a 60 % error at ``n = 32``).  The corrected
procedure -- measure the *2-D* symbol and divide the operator by it, exactly the
step the 1-D solver already performs in its own dimension -- restores a
consistent scheme: the mode-decay error falls 1.4e-3 -> 3.5e-4 -> 8.6e-5 for
``n = 32, 64, 128`` (second order), i.e. 0.0086 % at the finest rung.

So the honest content of this family is twofold, and both halves are banked:

1. **Accuracy record** -- with the 2-D calibration the operator reproduces the
   analytic mode decay to 0.0086 % (this is the second benchmark family of the
   registry record).
2. **Dimensional caveat** -- the 1-D calibration factor is *not* transferable:
   ``2/rho * sum V_j ...`` in 2-D is ``O(dx)`` too small without a
   dimension-specific normalisation.  Anchoring the record on the 2-D-calibrated
   arm without printing the uncalibrated and 1-D-calibrated arms would hide a
   trap, so all three arms are written to the artefact.

The reference is the exact separated solution of the heat equation on a periodic
square,

.. math::

    T(x,y,t) = e^{-\alpha(k_x^2+k_y^2)t}\cos(k_x x)\cos(k_y y),

obtained by substituting into :math:`\partial_t T = \alpha\nabla^2 T`; it is the
periodic (Fourier) solution of the linear heat equation (Carslaw & Jaeger 1959,
*Conduction of Heat in Solids*, 2nd ed., Ch. II, periodic solutions -- the same
chapter as the 1-D ``erfc`` target used by :mod:`tensorsph.heat_conduction`; the
book's section numbering is not machine-verified here, the formula is verified by
substitution and by an independent Crank-Nicolson solve).

What is *not* claimed
---------------------
* No claim about variable conductivity in 2-D: every check here uses
  ``k_i == k_j``, so the harmonic pair mean is exercised only trivially.
* No claim about walls/boundaries in 2-D: the domain is periodic, so no ghost
  layer is involved (the 1-D family already owns the wall treatment).
* No claim that this replaces the 1-D ``erfc`` case: the two families fail
  differently (similarity profile vs discrete symbol), which is the point of
  having both.
"""
from __future__ import annotations

import cmath
import math
import operator
from dataclasses import dataclass, field

import torch

from .ice_thermal.sph_thermal import _dense_pair_exchange, _pair_conductivity, kernel_dwdr

__all__ = [
    "DTYPE",
    "LatticeOffsets",
    "lattice_offsets",
    "stencil_rhs_2d",
    "solve_transient_layered_slab",
    "two_column_sheet_materials",
    "layered_slab_contact_temperatures",
    "layered_slab_modal_cut_flux",
    "extrapolate_layered_face_flux",
    "solve_low_frequency_layered_slab",
    "solve_low_frequency_layered_films",
    "pair_geometry_2d",
    "conduction_rhs_2d",
    "symbol_rate",
    "lattice_diffusion_factor_2d",
    "mode_field",
    "mode_exact",
    "mode_amplitude",
    "stencil_vs_dense_2d",
    "degenerate_1d_agreement",
    "fd_mode_decay",
    "ModeDecayConfig",
    "ModeDecayResult",
    "run_mode_decay",
]

DTYPE = torch.float64

#: Lattice offsets ``[((m, n), a), ...]`` inside the kernel support, with
#: ``a = V r dW/dr / (r^2 + eta^2)`` and ``V = dx^2``.
LatticeOffsets = tuple[tuple[int, int], float]


# --------------------------------------------------------------------- lattice
def lattice_offsets(dx: float, h: float, eta_factor: float = 0.01,
                    thickness: float = 1.0) -> tuple[LatticeOffsets, ...]:
    """``[((m, n), a), ...]`` -- the 2-D pair operator on a uniform square lattice.

    ``r_{ij} = |(m, n)| dx`` is the *vector* distance, so the diagonal offsets
    the 1-D stencil never sees are included.  ``thickness`` is the out-of-plane
    thickness used for the particle volume (``V = dx^2 * thickness``); it is 1
    everywhere in this benchmark and exposed only so the dimensional statement in
    the module docstring can be checked.
    """
    dx = float(dx)
    h = float(h)
    eta2 = (float(eta_factor) * h) ** 2
    r_max = 2.0 * h
    m_max = int(math.floor(r_max / dx + 1e-12))
    vol = dx * dx * float(thickness)
    h_t = torch.tensor(h, dtype=DTYPE)
    out: list[tuple[tuple[int, int], float]] = []
    for m in range(-m_max, m_max + 1):
        for n in range(-m_max, m_max + 1):
            if m == 0 and n == 0:
                continue
            r = math.hypot(m, n) * dx
            if r > r_max + 1e-12:
                continue
            dwdr = float(kernel_dwdr(torch.tensor(r, dtype=DTYPE), h_t))
            a = vol * r * dwdr / (r * r + eta2)
            if a != 0.0:
                out.append(((m, n), a))
    return tuple(out)


WallTemperature = float | complex | torch.Tensor
SlabWalls = tuple[WallTemperature | None, WallTemperature | None]


def _wall_column(value: WallTemperature, field: torch.Tensor) -> torch.Tensor:
    wall = torch.as_tensor(value, dtype=field.dtype, device=field.device)
    if wall.ndim == 0:
        return wall
    if wall.shape != (field.shape[0],):
        raise ValueError("wall temperature must be scalar or have one value per y row")
    return wall[:, None]


def stencil_rhs_2d(temperature: torch.Tensor,
                   offsets: tuple[LatticeOffsets, ...],
                   rho: float = 1.0, conductivity: torch.Tensor | None = None,
                   *, x_wall_temperatures: SlabWalls | None = None,
                   layered_x: bool = False, contact_resistance_x: torch.Tensor | None = None,
                   dx: float | None = None,
                   specific_heat: float | torch.Tensor | None = None) -> torch.Tensor:
    """``dT/dt`` of the pair operator on a *periodic* 2-D grid (fast path).

    ``dT_i/dt = (2/rho) sum_offsets a (T_i - T_{i+offset})`` with the offsets
    wrapped, i.e. the same expression :func:`conduction_rhs_2d` evaluates pair by
    pair; :func:`stencil_vs_dense_2d` checks the unit-conductivity case.
    Supply a positive scalar or grid of conductivities to weight each pair
    by the harmonic mean of its two material values. When x_wall_temperatures
    is supplied, x (the last axis) uses cell-centred Dirichlet walls with
    odd-reflected temperature about each wall value and even-reflected material.
    The first axis remains periodic. Each wall accepts a scalar or a vector
    of temperatures sampled at the y cell centres. Complex fields or walls
    return complex rates and powers, using real material coefficients. A None entry selects an
    insulated wall with even temperature reflection. With layered_x=True, supply
    a 1-D conductivity profile along x. Each pair then uses the integrated
    series resistance along its straight segment through the cell-centred layers.
    With layered_x and dx, contact_resistance_x supplies areal resistances
    (m² K/W) at faces between x columns, in order of increasing x. Supply
    nx-1 faces for fixed walls or nx faces (including the seam) for periodic x.
    specific_heat may be a positive scalar or broadcastable particle field in
    J/(kg K). Its default is unity.
    """
    rate, _ = _stencil_exchange(temperature, offsets, rho, conductivity, x_wall_temperatures, layered_x,
                                dx, contact_resistance_x, specific_heat)
    return rate


def stencil_rhs_with_wall_power_2d(
    temperature: torch.Tensor, offsets: tuple[LatticeOffsets, ...],
    dx: float, x_wall_temperatures: SlabWalls,
    rho: float = 1.0, conductivity: torch.Tensor | None = None, *, layered_x: bool = False,
    contact_resistance_x: torch.Tensor | None = None, specific_heat: float | torch.Tensor | None = None,
    power_by_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temperature rate and integrated inward power at the left and right walls.

    Assumes square cells and unit thickness. For a symmetric stencil, the two
    powers sum to dx**2 * sum(rho * specific_heat * rate). Any external
    diffusivity calibration must multiply the rate and powers equally.
    With power_by_row=True the powers have shape (2, ny), assigned to the
    reflected ghost's wall row. Sum over rows for the two total powers.
    """
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    if x_wall_temperatures is None:
        raise ValueError("wall powers require two x wall temperatures")
    return _stencil_exchange(temperature, offsets, rho, conductivity, x_wall_temperatures, layered_x,
                             dx, contact_resistance_x, specific_heat, collect_power=True, power_by_row=power_by_row)


def _heat_loss(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("heat_loss_coefficient must be finite and nonnegative")
    return float(value)


def solve_steady_slab_2d(heat_source: torch.Tensor, dx: float, x_wall_temperatures: SlabWalls,
                         *, conductivity: float = 1., h_factor: float = 1.3,
                         mean_temperature: float | None = None, heat_loss_coefficient: float = 0.,
                         x_wall_inward_flux=(0., 0.), return_x_flux: bool = False,
                         temperature_reference: float = 0., ambient_temperature=None, separate_mean: bool = False
                         ) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Steady temperature on a uniform lattice, periodic y and reflected x walls.

    Supply a (ny, nx) volumetric heat source in W/m³, a constant conductivity
    in W/(m K), and two wall conditions. None selects a flux wall. Two
    flux walls with zero heat loss require balanced input and a mean_temperature.
    A positive heat_loss_coefficient H (W/(m³ K)) exchanges H*(Ta-T) with
    a scalar or per-particle ambient_temperature Ta (default zero, K). Supply
    independent heat sources separately to retain small temperature differences.
    x_wall_inward_flux gives scalar or per-row heat fluxes into the domain in W/m² on
    sides with temperature None. The solution uses an analytical polynomial
    lift for prescribed flux and the calibrated SPH operator for the remainder.
    The returned field is T - temperature_reference in K. Set this reference
    near the operating temperature to retain small variations for gradient
    recovery. Wall inputs and mean_temperature remain absolute temperatures.
    With return_x_flux, also return the cross-section mean heat flux (W/m²)
    on all nx+1 faces, positive towards increasing x, including the walls.
    For two flux walls, separate_mean=True returns (T - mean(T),
    mean(T) - temperature_reference), inserting the flux before the mean
    when requested. This retains variations smaller than the absolute
    temperature's floating-point spacing.
    """
    if not math.isfinite(temperature_reference):
        raise ValueError("temperature_reference must be finite")
    return _solve_slab_modes(heat_source, dx, x_wall_temperatures, conductivity, h_factor,
                             mean_temperature, _heat_loss(heat_loss_coefficient), x_wall_inward_flux, return_x_flux, temperature_reference, ambient_temperature, separate_mean)


def solve_harmonic_slab_2d(heat_source: torch.Tensor, dx: float, x_wall_temperatures: SlabWalls,
                           angular_frequency: float, *, conductivity: float = 1.,
                           volumetric_heat_capacity: float = 1., h_factor: float = 1.3,
                           heat_loss_coefficient: float = 0., x_wall_inward_flux=(0., 0.), return_x_flux: bool = False,
                           ambient_temperature=None, separate_mean: bool = False
                         ) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Temperature phasors under exp(+iωt), for constant isotropic k and capacity.

    Use complex volumetric heating (W/m³) and scalar or row-wise wall phasors
    (K). On each wall with temperature None, x_wall_inward_flux specifies a
    scalar or per-row inward flux phasor (W/m²), default zero. Supply ω in
    rad/s and capacity in J/(m³ K). With return_x_flux, also return nx+1
    section-averaged complex heat fluxes (W/m²), positive towards increasing x.
    heat_loss_coefficient H adds H*(Ta-T) in W/m³. The ambient_temperature Ta
    is a scalar or per-particle temperature phasor (K), default zero.
    For two flux walls, separate_mean=True returns the zero-mean temperature
    phasor field and the scalar mean phasor, with the optional section flux
    between them. This resolves local oscillations even at very low frequency.
    """
    storage = angular_frequency*volumetric_heat_capacity
    if not all(math.isfinite(v) and v > 0 for v in (angular_frequency, volumetric_heat_capacity, storage)):
        raise ValueError("frequency and volumetric heat capacity must be positive and finite")
    return _solve_slab_modes(heat_source, dx, x_wall_temperatures, conductivity, h_factor,
                             None, _heat_loss(heat_loss_coefficient)+1j*storage, x_wall_inward_flux, return_x_flux,
                             ambient_temperature=ambient_temperature, separate_mean=separate_mean)


def _layered_response(dx, h_factor, conductivity, contact_resistance_x, transverse_wavenumber, split_interface_links=False):
    offsets = lattice_offsets(dx, h_factor*dx)
    factor = lattice_diffusion_factor_2d(h_factor, dx, mode="moment")
    if factor <= 0:
        raise ValueError("kernel support must include a conducting neighbour")
    if not math.isfinite(transverse_wavenumber):
        raise ValueError("transverse wavenumber must be finite")
    phase_step = transverse_wavenumber*dx
    def response(probes, walls):
        rate, _ = _stencil_exchange(probes, offsets, 1., conductivity, walls, True,
                                    dx, contact_resistance_x, None, transverse_phase=phase_step, split_interface_links=split_interface_links)
        return rate/factor

    return response, offsets


def solve_harmonic_layered_slab(heat_source, dx, x_wall_temperatures, angular_frequency,
                                *, conductivity, volumetric_heat_capacity,
                                contact_resistance_x=None, h_factor=1.3, transverse_wavenumber=0.,
                                backend="torch", heat_loss_coefficient=0., return_x_flux=False, ambient_temperature=None,
                                split_interface_links=False):
    """Return particle temperature phasors for one transverse Fourier mode of a layered slab.

    Use one source (W/m³) and conductivity (W/(m K)) per x column, a scalar
    or per-column heat capacity (J/(m³ K)), and optional interface resistances
    (m² K/W). heat_loss_coefficient is a scalar or per-column nonnegative
    bulk exchange coefficient H (W/(m³ K)), contributing H*(Ta-T). The ambient_temperature
    phasor Ta defaults to zero and may be scalar, per-column or sampled on the source grid.
    For a modal vector Ta uses the same transverse Fourier mode as the source.
    Each end has a scalar temperature phasor or None for insulation.
    The dense complex solve uses the calibrated 2-D particle-pair stencil with
    row dependence exp(i β y), with β=transverse_wavenumber in rad/m.
    With split_interface_links=True, resolve diagonal links crossing a
    conductivity change or contact resistance into normal exchange through
    the link and tangential diffusion within the particle's own material.
    Apply the same separation to links crossing prescribed-temperature walls.
    The inputs and output are modal amplitudes. Supply a periodic wavenumber
    compatible with the transverse period. Alternatively supply a (ny,nx)
    source and scalar or (ny,) end temperatures sampled on periodic rows.
    Then the output has the same grid shape and all Fourier modes are solved.
    Frequency is in rad/s under exp(+iωt). The default backend is dense torch.
    At zero frequency an insulated, lossless slab requires zero net source.
    Its capacity-weighted mean temperature is set to zero, the harmonic limit.
    backend="banded" uses SciPy on CPU and stores only the stencil diagonals.
    return_x_flux also returns nx+1 cross-section mean rightward heat flux
    phasors (W/m²) for a uniform transverse mode or a sampled periodic grid.
    For a sampled grid the mean integrates the full periodic section. For a
    vector with nonzero transverse_wavenumber it returns the local Fourier
    amplitudes on all cuts, including both walls, in the field's phase frame.
    For a sampled grid, return_x_flux="by_row" reconstructs those modal
    amplitudes at each periodic row, giving shape (ny, nx+1).
    """
    source = torch.as_tensor(heat_source, dtype=torch.complex128)
    if backend not in ("torch", "banded"):
        raise ValueError("backend must be torch or banded")
    if backend == "banded" and source.device.type != "cpu":
        raise ValueError("banded solves require CPU inputs")
    if isinstance(return_x_flux, str) and return_x_flux != "by_row":
        raise ValueError("return_x_flux must be a boolean or 'by_row'")
    if torch.is_complex(torch.as_tensor(heat_loss_coefficient)):
        raise ValueError("heat_loss_coefficient must be real")
    loss = torch.as_tensor(heat_loss_coefficient, dtype=DTYPE, device=source.device)
    if loss.shape not in ((),source.shape[-1:]) or not torch.all(torch.isfinite(loss) & (loss>=0)):
        raise ValueError("heat_loss_coefficient must be finite, nonnegative and scalar or per column")
    if ambient_temperature is not None:
        ambient = torch.as_tensor(ambient_temperature, dtype=source.dtype, device=source.device)
        if ambient.shape not in ((), source.shape[-1:], source.shape) or not torch.isfinite(ambient).all():
            raise ValueError("ambient temperature must be finite and scalar, per column or match the source grid")
        source = source + loss*ambient
    # A uniform offset is a null mode of lossless DC conduction. Remove it
    # before the transverse FFT, wall forcing and cut-current calculations.
    datum = 0.
    if angular_frequency == 0 and transverse_wavenumber == 0 and not torch.any(torch.as_tensor(heat_loss_coefficient) != 0):
        wall = next((w for w in x_wall_temperatures if w is not None), None)
        if wall is not None:
            values = torch.as_tensor(wall, dtype=source.dtype, device=source.device)
            if values.numel():
                datum = values.reshape(-1)[0]
                x_wall_temperatures = tuple(
                    None if w is None else torch.as_tensor(w, dtype=source.dtype, device=source.device)-datum
                    for w in x_wall_temperatures)
    def solution_with_flux(field,offsets):
        grid = field.reshape(-1,field.shape[-1])
        # With material properties uniform along each column, section averaging
        # commutes with the linear pair flux. Sum links at each cut directly
        # to retain weak currents far from the heated wall.
        walls = x_wall_temperatures
        if grid.shape[0] > 1:
            walls = tuple(None if w is None else _wall_column(w, grid).mean() for w in walls)
            grid = grid.mean(dim=0, keepdim=True)
        _, power = _stencil_exchange(grid, offsets, 1., conductivity, walls,
            True, dx, contact_resistance_x, None, modal_flux=True, transverse_phase=transverse_wavenumber*dx,
            split_interface_links=split_interface_links)
        return field+datum, power/(dx*lattice_diffusion_factor_2d(h_factor,dx,mode="moment"))

    if source.ndim == 2:
        if transverse_wavenumber != 0:
            raise ValueError("explicit wavenumbers apply only to modal vector inputs")
        if min(source.shape) == 0 or not math.isfinite(dx) or dx <= 0:
            raise ValueError("provide a nonempty periodic grid and positive finite dx")
        ny = source.shape[0]
        left, right = x_wall_temperatures

        def wall_modes(wall):
            if wall is None:
                return [None]*ny
            samples = _wall_column(wall, source).reshape(-1).expand(ny)
            return torch.fft.fft(samples.contiguous())

        modes = torch.fft.fft(source.contiguous(), dim=0)
        if angular_frequency==0 and left is None and right is None and not torch.any(torch.as_tensor(heat_loss_coefficient)!=0):
            # Check compatibility before Fourier cancellation obscures the load scale.
            if abs(source.sum())>64*torch.finfo(DTYPE).eps*source.abs().sum():
                raise ValueError("steady insulated slab requires zero net source")
            modes[0] -= modes[0].mean()
        waves = (2*math.pi)*torch.fft.fftfreq(ny, d=dx, dtype=DTYPE)
        rows = [solve_harmonic_layered_slab(load, dx, (west, east), angular_frequency,
                        conductivity=conductivity, volumetric_heat_capacity=volumetric_heat_capacity,
                        contact_resistance_x=contact_resistance_x, h_factor=h_factor,
                        transverse_wavenumber=float(wave), backend=backend, heat_loss_coefficient=heat_loss_coefficient,
                        return_x_flux=(return_x_flux == "by_row"), split_interface_links=split_interface_links)
                for load, west, east, wave in zip(modes, wall_modes(left), wall_modes(right), waves)]
        if return_x_flux == "by_row":
            fields, cuts = zip(*rows)
            return torch.fft.ifft(torch.stack(fields), dim=0)+datum, torch.fft.ifft(torch.stack(cuts), dim=0)
        field = torch.fft.ifft(torch.stack(rows), dim=0)
        return solution_with_flux(field,lattice_offsets(dx,h_factor*dx)) if return_x_flux else field+datum
    if source.ndim != 1 or source.numel() == 0 or not torch.isfinite(source).all():
        raise ValueError("provide a finite nonempty vector of column heat sources")
    if not all(math.isfinite(v) and v > 0 for v in (dx,h_factor)):
        raise ValueError("dx and h_factor must be positive and finite")
    if not math.isfinite(angular_frequency) or angular_frequency < 0:
        raise ValueError("frequency must be nonnegative and finite")
    capacity = torch.as_tensor(volumetric_heat_capacity, dtype=DTYPE, device=source.device)
    if capacity.shape not in ((), source.shape) or not torch.all(torch.isfinite(capacity) & (capacity > 0)):
        raise ValueError("provide a positive finite scalar or column heat capacities")
    left, right = x_wall_temperatures
    for wall in (left, right):
        if wall is not None and (torch.as_tensor(wall).ndim or not cmath.isfinite(wall)):
            raise ValueError("wall phasors must be finite scalars")
    homogeneous = tuple(None if wall is None else 0. for wall in (left, right))
    response, offsets = _layered_response(dx,h_factor,conductivity,contact_resistance_x,transverse_wavenumber,split_interface_links)
    load = response(torch.zeros_like(source)[None, :], (left, right))[0]
    closed_mode = False
    if angular_frequency==0 and left is None and right is None and not torch.any(loss>0):
        # A sampled nonconstant transverse mode supplies its own restoring term.
        phase = math.remainder(transverse_wavenumber*dx,math.tau)
        closed_mode = abs(phase)<=math.ulp(math.tau)
        if closed_mode and abs(source.sum())>64*torch.finfo(DTYPE).eps*source.abs().sum():
            raise ValueError("steady insulated slab requires zero net source")
    bulk_shift = loss+1j*angular_frequency*capacity.expand_as(source)
    size = source.numel()
    rhs = source+load
    if closed_mode:
        # Pin one row during the solve, then recover the energy-conserving mean.
        rhs[0] = 0.
    if backend == "banded":
        from scipy.linalg import solve_banded

        radius = min(size-1, max(abs(n) for (_, n), _ in offsets))
        stride = 2*radius+1
        columns = torch.arange(size)
        # Disjoint stencil supports let a single probe excite every stride-th column.
        classes = columns % stride
        probes = (torch.arange(stride)[:,None] == classes).to(DTYPE)
        rates = response(probes, homogeneous)
        bands = source.new_zeros((stride, size))
        for offset in range(-radius, radius+1):
            j = columns[max(0,-offset):min(size,size-offset)]
            bands[radius+offset,j] = -rates[classes[j],j+offset].to(source.dtype)
        bands[radius] += bulk_shift
        if closed_mode:
            j = columns[:radius+1]
            bands[radius-j,j] = 0.
            bands[radius,0] = 1.
        if radius == 0:
            field = rhs/bands[0]
        else:
            field = torch.from_numpy(solve_banded((radius,radius),bands.numpy(),rhs.numpy()))
    else:
        basis = torch.eye(size, dtype=DTYPE, device=source.device)
        operator = response(basis, homogeneous).T
        matrix = torch.diag(bulk_shift)-operator
        if closed_mode:
            matrix[0] = 0.
            matrix[0,0] = 1.
        field = torch.linalg.solve(matrix,rhs)
    if closed_mode:
        field = field-(capacity*field).mean()/capacity.mean()
    return solution_with_flux(field,offsets) if return_x_flux else field+datum


def solve_low_frequency_layered_slab(heat_source, dx, x_wall_temperatures, *, conductivity,
                                     volumetric_heat_capacity, contact_resistance_x=None, h_factor=1.3,
                                     transverse_wavenumber=0., backend="torch", heat_loss_coefficient=0.,
                                     return_x_flux=False, order=1):
    """Return coefficients of the low-frequency expansion in iω for fixed slab loads.

    Source, wall data and transverse representation follow solve_harmonic_layered_slab.
    The first-order field is in K s. A prescribed wall temperature has zero
    first-order coefficient, and insulated walls retain zero flux.
    With return_x_flux=True each coefficient is a (field, flux) pair. Fluxes
    have the same orientation and spatial sampling as the harmonic solve,
    in W/m² at steady state and W s/m² at first order. Use "by_row" for
    row-resolved grid fluxes. order=2 appends the coefficient of (iω)²,
    with units K s² and W s²/m². The default returns two coefficients.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    options = dict(conductivity=conductivity,volumetric_heat_capacity=volumetric_heat_capacity,
                   contact_resistance_x=contact_resistance_x,h_factor=h_factor,
                   transverse_wavenumber=transverse_wavenumber,backend=backend,
                   heat_loss_coefficient=heat_loss_coefficient,return_x_flux=return_x_flux)
    steady = solve_harmonic_layered_slab(heat_source,dx,x_wall_temperatures,0.,**options)
    field = steady[0] if return_x_flux else steady
    capacity = torch.as_tensor(volumetric_heat_capacity,dtype=DTYPE,device=field.device)
    lag_walls = tuple(None if wall is None else 0. for wall in x_wall_temperatures)
    coefficients = [steady]
    for _ in range(order):
        coefficient = solve_harmonic_layered_slab(-capacity*field,dx,lag_walls,0.,**options)
        coefficients.append(coefficient)
        field = coefficient[0] if return_x_flux else coefficient
    return tuple(coefficients)


def _unanchored_flux_mode(films, bulk_loss, transverse_wavenumber, dx):
    return (not torch.any(torch.as_tensor(films) != 0) and not torch.any(torch.as_tensor(bulk_loss) != 0)
            and abs(math.remainder(transverse_wavenumber*dx, math.tau)) <= math.ulp(math.tau))


def solve_harmonic_layered_films(heat_source, dx, ambient_temperatures, angular_frequency,
                                 *, film_coefficients, conductivity, volumetric_heat_capacity,
                                 contact_resistance_x=None, h_factor=1.3, backend="torch",
                                 wall_areal_heat_capacity=(0.,0.), heat_loss_coefficient=0.,
                                 wall_applied_flux=(0.,0.), mean_temperature=None,
                                 transverse_wavenumber=0., return_x_flux=False, separate_mean=False):
    """Solve a transverse Fourier mode of a layered slab with two convective walls.

    Return (particle temperatures in K, two wall temperatures in K, two inward
    heat fluxes in W/m²), all phasors under exp(+iωt). Film coefficients are
    real and nonnegative in W/(m² K). wall_applied_flux adds two inward
    applied heat flux phasors in W/m² independently of the films. Ambient
    phasors, applied fluxes and coefficients are ordered west, east. Source, material properties and contacts follow
    solve_harmonic_layered_slab's column-vector convention. Each wall can store
    heat with areal capacity in J/(m² K). Returned heat flux enters the slab.
    Ambient, particle and wall phasors multiply exp(i*transverse_wavenumber*y).
    For the steady constant transverse mode with two flux-only walls and no
    bulk loss, the wall inputs and integrated source must sum to zero. Supply
    mean_temperature (K) as the heat-capacity-weighted mean of the particle
    columns and both wall stores. Other boundary conditions set their own level.
    At positive frequency in this flux-only mode, separate_mean=True returns
    particle and wall temperatures relative to their capacity-weighted mean
    and appends that complex scalar (K) after the requested flux arrays.
    Set return_x_flux=True to append rightward heat fluxes on all nx+1 cuts.
    """
    source = torch.as_tensor(heat_source, dtype=torch.complex128)
    if source.ndim != 1:
        raise ValueError("film solve requires one heat source per x column")
    ambient = torch.as_tensor(ambient_temperatures, dtype=source.dtype, device=source.device)
    if torch.is_complex(torch.as_tensor(film_coefficients)):
        raise ValueError("film coefficients must be real")
    films = torch.as_tensor(film_coefficients, device=source.device, dtype=DTYPE)
    if ambient.shape != (2,) or not torch.isfinite(ambient).all():
        raise ValueError("provide two finite ambient temperature phasors")
    if films.shape != (2,) or not torch.all(torch.isfinite(films) & (films >= 0)):
        raise ValueError("provide two nonnegative finite film coefficients")

    applied = torch.as_tensor(wall_applied_flux, dtype=source.dtype, device=source.device)
    if applied.shape != (2,) or not torch.isfinite(applied).all():
        raise ValueError("provide two finite inward applied wall fluxes")
    wall_capacity = torch.as_tensor(wall_areal_heat_capacity, device=source.device, dtype=DTYPE)
    if wall_capacity.shape != (2,) or not torch.all(torch.isfinite(wall_capacity) & (wall_capacity >= 0)):
        raise ValueError("provide two finite nonnegative wall areal heat capacities")
    opts = dict(conductivity=conductivity, contact_resistance_x=contact_resistance_x)
    def response(load, walls):
        field, cuts = solve_harmonic_layered_slab(load, dx, walls, angular_frequency,
                      volumetric_heat_capacity=volumetric_heat_capacity, h_factor=h_factor,
                      backend=backend, heat_loss_coefficient=heat_loss_coefficient,
                      transverse_wavenumber=transverse_wavenumber, return_x_flux=True, **opts)
        return field, cuts

    closed = _unanchored_flux_mode(films, heat_loss_coefficient, transverse_wavenumber, dx)
    harmonic_closed = closed and angular_frequency > 0
    if separate_mean and not harmonic_closed:
        raise ValueError("separate_mean requires a positive-frequency lossless flux-only constant mode")
    if harmonic_closed:
        capacity = torch.as_tensor(volumetric_heat_capacity, dtype=DTYPE, device=source.device)
        if capacity.shape not in ((), source.shape) or not torch.all(torch.isfinite(capacity) & (capacity > 0)):
            raise ValueError("provide positive finite scalar or column heat capacities")
        weights = dx*capacity.expand_as(source)
        rate = (dx*source.sum()+applied.sum())/(weights.sum()+wall_capacity.sum())
        uniform = rate/(1j*angular_frequency)
        source = source-capacity*rate
        applied = applied-wall_capacity*rate

    base, base_cuts = response(source, (0., 0.))
    # Two unit-temperature solutions give the complete wall admittance,
    # including heat transmitted from one wall to the opposite wall.
    zero = torch.zeros_like(source)
    units = ((1.,0.), (0.,1.))
    fields, cuts = zip(*(response(zero, walls) for walls in units))
    cut_response = torch.stack(cuts, dim=1)
    admittance = torch.stack((cut_response[0], -cut_response[-1]))
    base_flux = torch.stack((base_cuts[0], -base_cuts[-1]))
    if angular_frequency == 0 and closed:
        if mean_temperature is None:
            raise ValueError("steady flux-only walls require a mean_temperature reference")
        target = torch.as_tensor(mean_temperature, dtype=source.dtype, device=source.device)
        if target.ndim or not torch.isfinite(target):
            raise ValueError("mean_temperature must be a finite scalar")
        net = applied.sum()+dx*source.sum()
        scale = applied.abs().sum()+dx*source.abs().sum()
        if net.abs() > 64*torch.finfo(DTYPE).eps*scale:
            raise ValueError("steady flux-only walls require balanced wall inputs and volumetric heating")
        # Fix the west temperature temporarily. The west flux determines the
        # relative east temperature. A uniform shift affects only temperatures.
        east = (applied[0]-base_flux[0])/admittance[0, 1]
        walls = torch.stack((torch.zeros_like(east), east))
        field = base+east*fields[1]
        weights = dx*torch.as_tensor(volumetric_heat_capacity, dtype=DTYPE, device=source.device).expand_as(source)
        mean = ((weights*field).sum()+(wall_capacity*walls).sum())/(weights.sum()+wall_capacity.sum())
        shift = target-mean
        result = (field+shift, walls+shift, base_flux+east*admittance[:, 1])
        return (*result, base_cuts+east*cuts[1]) if return_x_flux else result
    if mean_temperature is not None:
        raise ValueError("mean_temperature applies only to an unanchored steady flux-only mode")
    matrix = admittance + torch.diag(films+1j*angular_frequency*wall_capacity)
    rhs = films*ambient+applied-base_flux
    basis = torch.stack(fields)
    if harmonic_closed:
        # The summed wall equation degenerates at small frequency. Replace
        # it by the exact zero capacity mean of the remaining field.
        mean_row = (basis*weights).sum(dim=1)+wall_capacity
        matrix = torch.stack((matrix[0]-matrix[1], mean_row))
        rhs = torch.stack((rhs[0]-rhs[1], -(base*weights).sum()))
    walls = torch.linalg.solve(matrix, rhs)
    field = base + walls @ basis
    inward = base_flux + admittance @ walls
    result = (field, walls, inward)
    if return_x_flux:
        result += (base_cuts + cut_response @ walls,)
    if harmonic_closed:
        if separate_mean:
            return (*result, uniform)
        return (field+uniform, walls+uniform, *result[2:])
    return result


def solve_low_frequency_layered_films(heat_source, dx, ambient_temperatures, *,
                                      film_coefficients, conductivity, volumetric_heat_capacity,
                                      contact_resistance_x=None, h_factor=1.3, backend="torch",
                                      wall_areal_heat_capacity=(0.,0.), heat_loss_coefficient=0.,
                                      wall_applied_flux=(0.,0.),
                                      transverse_wavenumber=0., order=1, return_pole=False):
    """Return low-frequency coefficients of the exp(+iωt) film response.

    Each tuple contains particle temperatures, wall temperatures and inward
    fluxes. The default returns steady and linear coefficients of iω.
    order=2 appends the coefficient of (iω)², including both wall storage
    terms. Each power adds one factor of seconds to the coefficient units.
    The volumetric source, applied wall fluxes and ambient temperatures are frequency independent.
    A lossless constant mode with flux-only walls admits this Taylor series
    when total input is zero. Its finite-frequency energy balance selects zero
    capacity-weighted mean (including both wall stores) at every order.
    With return_pole=True return (residue, coefficients). The scalar residue
    (K/s) is net applied heat per area divided by total heat capacity per
    area for an unanchored mode, and zero for an anchored mode. Add
    residue/(iω) to all particle and wall temperatures. Heat flux has no
    pole. The returned regular coefficients include the wall storage rate
    associated with uniform heating.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    options = dict(film_coefficients=film_coefficients,conductivity=conductivity,
                   volumetric_heat_capacity=volumetric_heat_capacity,contact_resistance_x=contact_resistance_x,
                   h_factor=h_factor,backend=backend,wall_areal_heat_capacity=wall_areal_heat_capacity,
                   heat_loss_coefficient=heat_loss_coefficient,transverse_wavenumber=transverse_wavenumber)
    source = torch.as_tensor(heat_source, dtype=torch.complex128)
    residue = source.new_zeros(())
    if _unanchored_flux_mode(film_coefficients, heat_loss_coefficient, transverse_wavenumber, dx):
        options["mean_temperature"] = 0.
        if return_pole:
            applied = torch.as_tensor(wall_applied_flux, dtype=source.dtype, device=source.device)
            surface = torch.as_tensor(wall_areal_heat_capacity, dtype=DTYPE, device=source.device)
            capacity = torch.as_tensor(volumetric_heat_capacity, dtype=DTYPE, device=source.device)
            if source.ndim != 1 or not source.numel() or capacity.shape not in ((), source.shape):
                raise ValueError("provide a nonempty source vector and scalar or matching column capacities")
            if applied.shape != (2,) or surface.shape != (2,) or not torch.isfinite(applied).all():
                raise ValueError("provide two finite applied fluxes and two wall capacities")
            if (not math.isfinite(dx) or dx <= 0 or not torch.all(torch.isfinite(capacity) & (capacity > 0))
                    or not torch.all(torch.isfinite(surface) & (surface >= 0))):
                raise ValueError("require positive spacing and column capacities and nonnegative finite wall capacities")
            residue = (dx*source.sum()+applied.sum())/(dx*capacity.expand_as(source).sum()+surface.sum())
            heat_source = source-capacity*residue
            west = applied[0]-surface[0]*residue
            # Use global balance for the east load to avoid cancellation when
            # the uniform mode accounts for almost the entire forcing.
            wall_applied_flux = torch.stack((west, -west-dx*heat_source.sum()))
    steady = solve_harmonic_layered_films(heat_source,dx,ambient_temperatures,0.,wall_applied_flux=wall_applied_flux,**options)
    field,walls,_ = steady
    capacity = torch.as_tensor(volumetric_heat_capacity,dtype=DTYPE,device=field.device)
    surface = torch.as_tensor(wall_areal_heat_capacity,dtype=DTYPE,device=field.device)
    coefficients = [steady]
    for _ in range(order):
        # Storage at the preceding order supplies both the bulk and film loads.
        coefficient = solve_harmonic_layered_films(-capacity*field,dx,(0.,0.),0.,wall_applied_flux=-surface*walls,**options)
        coefficients.append(coefficient)
        field,walls,_ = coefficient
    return (residue, tuple(coefficients)) if return_pole else tuple(coefficients)


@dataclass
class _StoringSlabWall:
    side: int
    column: int
    index: int
    conductance: torch.Tensor
    film: float
    ambient: object
    applied_flux: object
    mode: torch.Tensor


def _polynomial_exponential_step(state, generator, capacity, load_start, load_end, dt, midpoint):
    """Integrate a degree-one or degree-two load and the interval mean state."""
    n = state.numel()
    extra = 3 if midpoint is None else 4
    block = state.new_zeros((n+extra,n+extra))
    block[:n,:n] = dt*generator
    slope = load_end-load_start
    if midpoint is not None:
        curvature = 2*(load_end+load_start-2*midpoint)
        block[:n,n] = 2*dt*(curvature/capacity)
        slope = slope-curvature
    block[:n,-3] = dt*(slope/capacity)
    block[:n,-2] = dt*(generator@state+load_start/capacity)
    # The polynomial chain integrates each coefficient on the unit interval.
    for i in range(n,n+extra-1):
        block[i,i+1] = 1.
    propagator = torch.linalg.matrix_exp(block)
    return state+propagator[:n,-2],state+propagator[:n,-1]


def layered_slab_contact_temperatures(temperature, x_flux, dx, conductivity, *, axial_flux_curvature=None):
    """Reconstruct temperatures on both material sides of all interior cuts.

    temperature holds particle-center temperatures (K) shaped (..., nx).
    x_flux contains rightward section heat fluxes (W/m²) shaped (..., nx+1).
    Alternatively, shape (..., nx, 2) supplies each particle column’s west
    and east material fluxes, both positive rightward. Use these one-sided
    values at a storing interface where the section flux is discontinuous.
    For interval means, supply the mean particle field and divide integrated
    section heat by the duration of each interval. Complex phasors are also
    supported. dx is the particle spacing; conductivity is positive and
    constant within each column (W/m/K).

    Return two arrays shaped (..., nx-1), ordered by interior cut. Each
    particle is extrapolated to its own side by integrating Fourier's law
    over half a column with linearly interpolated flux. Supply axial_flux_curvature
    (d²q/dx² in W/m⁴, broadcastable to the field) for a quadratic flux profile
    inside each column. In steady one-dimensional conduction this equals the
    volumetric source gradient. The temperature jump
    can thus be checked independently against resistance times section flux.
    """
    center = torch.as_tensor(temperature)
    if center.ndim<1 or center.shape[-1]==0 or not torch.isfinite(center).all():
        raise ValueError("provide a finite nonempty particle field")
    flux = torch.as_tensor(x_flux,device=center.device)
    if not torch.isfinite(flux).all():
        raise ValueError("section fluxes must be finite")
    if flux.shape == center.shape+(2,):
        west, east = flux.unbind(dim=-1)
    elif flux.shape == center.shape[:-1]+(center.shape[-1]+1,):
        west, east = flux[..., :-1], flux[..., 1:]
    else:
        raise ValueError("provide one flux per section or west/east fluxes per particle column")
    if not math.isfinite(dx) or dx<=0:
        raise ValueError("dx must be finite and positive")
    if torch.is_complex(torch.as_tensor(conductivity)):
        raise ValueError("conductivity must be real")
    k = torch.as_tensor(conductivity,dtype=DTYPE,device=center.device)
    if k.shape not in ((),center.shape[-1:]) or not torch.isfinite(k).all() or (k<=0).any():
        raise ValueError("conductivity must be a positive scalar or column profile")
    # Half-cell integration weights are 1/8 and 3/8 of the column width.
    scale = dx/(8*k)
    left_trace = center-(west+3*east)*scale
    right_trace = center+(3*west+east)*scale
    if axial_flux_curvature is not None:
        raw = torch.as_tensor(axial_flux_curvature, device=center.device)
        dtype = torch.promote_types(left_trace.dtype, raw.dtype)
        curvature = torch.as_tensor(axial_flux_curvature, dtype=dtype, device=center.device)
        try:
            curvature = torch.broadcast_to(curvature, center.shape)
        except RuntimeError as exc:
            raise ValueError("flux curvature must broadcast to the particle field") from exc
        if not torch.isfinite(curvature).all():
            raise ValueError("provide a finite flux curvature broadcastable to the particle field")
        # Integrate the quadratic correction, zero at both faces, over each half.
        offset = curvature*(dx**3/(24*k))
        left_trace = left_trace+offset
        right_trace = right_trace-offset
    return left_trace[..., :-1],right_trace[..., 1:]


def two_column_sheet_materials(conductivity, volumetric_heat_capacity, dx, face, *,
                               resistance, areal_heat_capacity, sheet_conductivity, contact_resistance_x=None):
    """Insert a storing sheet at interior face `face` of a uniform layered slab.

    The bulk conductivity vector (W/m/K) sets the number of original columns.
    Bulk volumetric capacity (J/m³/K) is scalar or columnwise. Face f is between
    original columns f-1 and f. Supply total sheet resistance R (m²K/W), areal
    capacity C (J/m²/K) and sheet conductivity k (W/m/K) with 2*dx/k <= R.

    contact_resistance_x optionally specifies one nonnegative areal resistance
    per original internal face. At f its value is divided equally between the
    two outer sheet faces, in addition to R. Other contacts keep their values.
    Omit it for perfect original contacts. The returned material keywords may
    be passed to solve_transient_layered_slab or expanded into another call to
    this builder (using the updated face indices). Insert two initial temperatures
    at f and distribute a sheet power per unit area P as P/(2*dx) on each of
    these columns. Their mean temperature times C gives stored energy per area.
    The heat ledger at cuts f and f+2 gives left and right transfers. This
    representation was verified for a transversely uniform slab at h/dx=1.3.
    """
    if not all(math.isfinite(x) and x > 0 for x in
               (dx, resistance, areal_heat_capacity, sheet_conductivity)):
        raise ValueError("spacing and sheet material parameters must be positive and finite")
    if any(torch.is_complex(torch.as_tensor(v)) for v in (conductivity, volumetric_heat_capacity)):
        raise ValueError("bulk material properties must be real")
    k = torch.as_tensor(conductivity, dtype=DTYPE)
    cv = torch.as_tensor(volumetric_heat_capacity, dtype=DTYPE, device=k.device)
    face = operator.index(face)
    if k.ndim != 1 or not 0 < face < k.numel():
        raise ValueError("supply a conductivity vector and an interior face index")
    if cv.shape not in ((), k.shape):
        raise ValueError("bulk capacity must be scalar or match the conductivity vector")
    if not all(torch.isfinite(v).all() and (v > 0).all() for v in (k, cv)):
        raise ValueError("bulk material properties must be positive and finite")
    cv = cv.expand_as(k)
    contact = resistance/2 - dx/sheet_conductivity
    if contact < 0:
        raise ValueError("sheet resistance must include the resistance of both inserted columns")
    if contact_resistance_x is None:
        bulk_contacts = k.new_zeros(k.numel()-1)
    else:
        if torch.is_complex(torch.as_tensor(contact_resistance_x)):
            raise ValueError("bulk contact resistances must be real")
        bulk_contacts = torch.as_tensor(contact_resistance_x, dtype=k.dtype, device=k.device)
        if bulk_contacts.shape != (k.numel()-1,):
            raise ValueError("supply one contact resistance per original internal face")
        if not torch.all(torch.isfinite(bulk_contacts) & (bulk_contacts >= 0)):
            raise ValueError("bulk contact resistances must be finite and nonnegative")
    outer_contact = contact + .5*bulk_contacts[face-1:face]
    contacts = torch.cat((bulk_contacts[:face-1], outer_contact, k.new_zeros(1),
                          outer_contact, bulk_contacts[face:]))
    return dict(conductivity=torch.cat((k[:face], k.new_full((2,), sheet_conductivity), k[face:])),
                volumetric_heat_capacity=torch.cat((cv[:face], cv.new_full((2,), areal_heat_capacity/(2*dx)), cv[face:])),
                contact_resistance_x=contacts)


def solve_transient_layered_slab(temperature, times, dx, x_wall_temperatures, *, conductivity,
                                 volumetric_heat_capacity, contact_resistance_x=None, heat_source=0.,
                                 heat_loss_coefficient=0., theta=.5, h_factor=1.3, load_jump_times=(),
                                 return_x_heat=False, x_wall_inward_flux=(0.,0.), rannacher=False,
                                 x_wall_films=(None,None), return_wall_temperatures=False, method="theta",
                                 bulk_ambient_temperature=0., return_bulk_exchange=False,
                                 wall_areal_heat_capacity=(0.,0.), initial_wall_temperatures=(None,None),
                                 return_external_heat=False, return_interval_temperature=False,
                                 return_interval_wall_temperatures=False, return_wall_heat_components=False,
                                 return_source_heat=False, return_directional_x_heat=False):
    """Integrate a transversely uniform layered slab with stationary material properties.

    Return temperatures (nt,nx) including the initial state. Specify one initial
    temperature and conductivity per column, and a scalar or column profile for
    heat capacity, volumetric source and bulk heat loss. Each end has a prescribed scalar temperature or a prescribed inward
    flux (temperature None). x_wall_inward_flux gives west and east fluxes in
    W/m², distributed over the reflected stencil. Zero gives insulation.
    For convection on an end with temperature None,
    supply (h, ambient_temperature) in x_wall_films. The positive constant
    film coefficient in W/(m² K) acts in series with the adjacent half column.
    A prescribed flux at the same surface adds to h*(Tamb-Twall).
    Ambient temperatures, wall temperatures, wall fluxes and heat_source
    can be functions of time. Supply their discontinuities in load_jump_times,
    with each callable returning the new value at the jump. Internal subdivisions
    preserve the requested output times. Contact resistances follow the
    harmonic layered solver. The dense implicit theta method accepts .5 to 1.
    method="sdirk2" selects a two-stage, second-order L-stable method. Loads
    and section heats use both internal stages. The same LU serves both solves.
    method="exponential" uses a dense matrix exponential for the endpoint
    and interval-mean temperatures. It is exact in time for affine loads on
    each segment and interpolates other histories linearly between endpoints.
    method="exponential_quadratic" also samples each segment midpoint and
    integrates the resulting quadratic load and its boundary exchanges.
    With rannacher=True the first Crank–Nicolson step and the first after
    each load jump use two backward Euler half steps.
    return_x_heat appends interval-integrated rightward section heat in J/m²,
    shaped (nt-1,nx+1), including any internal steps at load jumps.
    return_directional_x_heat appends a final dict of nonnegative rightward
    and leftward heats (J/m²) on the same cuts. Within each internal step
    fit an affine flux with the solver's interval mean and final flux (left
    limit at a load jump). Its signed integral equals the native net heat;
    directional integrals approximate the flux history within the step.
    Bulk exchange H*(Tb-T) uses heat_loss_coefficient and the scalar or column
    profile bulk_ambient_temperature in K, optionally callable in time.
    return_bulk_exchange appends one signed heat per output interval and column
    in J/m², after the optional surface temperatures.
    return_wall_temperatures appends instantaneous west and east surface
    temperatures (nt,2), after optional section heat. Flux and film walls use
    the half-column thermal resistance. At a listed jump the trace uses the
    new boundary data while the particle temperatures remain continuous.
    A flux or film wall with positive wall_areal_heat_capacity in J/(m² K)
    evolves its own temperature and accepts additive inward applied flux.
    initial_wall_temperatures defaults to the adjacent particle temperature
    for storing walls. return_external_heat appends inward external heat at
    both walls per output interval (J/m²), including any surface storage.
    return_interval_temperature appends mean particle temperatures (nt-1,nx)
    in K. The mean uses the same stage quadrature as the heat balance and
    includes all internal substeps within each requested interval.
    return_interval_wall_temperatures appends mean west and east surface
    temperatures (nt-1,2) in K, using the same quadrature.
    return_wall_heat_components appends a dictionary with applied and film
    inward heat arrays (nt-1,2) in J/m². These integrate prescribed flux and
    convective exchange separately. Both are zero on fixed-temperature ends;
    their thermostat input is available through return_external_heat.
    return_source_heat appends the signed integral of heat_source per column
    and unit slab area (nt-1,nx), in J/m². Positive heat adds energy. The
    integration follows the selected method and splits at each load jump.
    """
    if torch.is_complex(torch.as_tensor(temperature)):
        raise ValueError("initial column temperatures must be real")
    state = torch.as_tensor(temperature, dtype=DTYPE)
    if state.ndim!=1 or not state.numel() or not torch.isfinite(state).all():
        raise ValueError("provide a finite real vector of initial column temperatures")
    grid = [float(t) for t in times]
    if len(grid)<2 or not all(math.isfinite(t) for t in grid) or any(a>=b for a,b in zip(grid,grid[1:])):
        raise ValueError("times must be finite and strictly increasing")
    if not (.5<=theta<=1 and math.isfinite(dx) and dx>0 and math.isfinite(h_factor) and h_factor>0):
        raise ValueError("require theta in [.5,1] and positive finite dx and h_factor")
    if method not in ("theta","sdirk2","exponential","exponential_quadratic"):
        raise ValueError("method must be theta, sdirk2, exponential or exponential_quadratic")
    sdirk = method=="sdirk2"
    quadratic = method=="exponential_quadratic"
    exponential = method=="exponential" or quadratic
    if method!="theta" and (rannacher or theta!=.5):
        raise ValueError(f"{method} cannot be combined with Rannacher or a theta override")
    walls = tuple(x_wall_temperatures)
    if len(walls)!=2:
        raise ValueError("provide two wall temperatures or None")
    fluxes = tuple(x_wall_inward_flux)
    if len(fluxes)!=2:
        raise ValueError("provide two inward wall fluxes")
    films = tuple(x_wall_films)
    if len(films)!=2:
        raise ValueError("provide two film conditions or None")
    jumps = [float(t) for t in load_jump_times]
    events = set(jumps)
    if len(events)!=len(jumps) or not all(grid[0]<t<grid[-1] for t in jumps):
        raise ValueError("load_jump_times must contain distinct interior times")

    def wall_value(value,name="wall temperature"):
        if value is None or torch.as_tensor(value).ndim or torch.is_complex(torch.as_tensor(value)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
        return value

    particles = state
    nx = len(particles)

    def profile(value,name):
        if torch.is_complex(torch.as_tensor(value)):
            raise ValueError(f"{name} must be real")
        data = torch.as_tensor(value,dtype=DTYPE,device=state.device)
        if data.shape not in ((),particles.shape) or not torch.isfinite(data).all():
            raise ValueError(f"{name} must be a finite scalar or column profile")
        return data.expand_as(particles)

    capacity = profile(volumetric_heat_capacity,"volumetric_heat_capacity")
    loss = profile(heat_loss_coefficient,"heat_loss_coefficient")
    if (capacity<=0).any() or (loss<0).any():
        raise ValueError("capacity must be positive and heat loss nonnegative")
    if any(v is not None and torch.is_complex(torch.as_tensor(v)) for v in (conductivity,contact_resistance_x)):
        raise ValueError("conductivity and contact resistances must be real")
    homogeneous = tuple(None if wall is None else 0. for wall in walls)
    response, offsets = _layered_response(dx,h_factor,conductivity,contact_resistance_x,0.)
    zero = torch.zeros_like(state)[None,:]
    static_walls = tuple(0. if callable(w) else None if w is None else wall_value(w) for w in walls)
    static_load = response(zero,static_walls)[0]
    fixed_source = None if callable(heat_source) else profile(heat_source,"heat_source").clone()
    if fixed_source is not None:
        static_load = static_load+fixed_source
    fixed_reservoir = None if callable(bulk_ambient_temperature) else profile(bulk_ambient_temperature,"bulk ambient temperature")
    wall_modes = []
    for i,wall in enumerate(walls):
        if callable(wall):
            unit = list(homogeneous)
            unit[i] = 1.
            wall_modes.append((i,wall,response(zero,unit)[0]))

    flux_modes = []
    film_modes = []
    storing = []
    wall_initial = []
    if len(wall_areal_heat_capacity)!=2 or len(initial_wall_temperatures)!=2:
        raise ValueError("provide capacity and initial temperature entries for both walls")
    for i,flux in enumerate(fluxes):
        imposed = callable(flux) or wall_value(flux,"wall flux")!=0.
        film = films[i]
        cw = wall_value(wall_areal_heat_capacity[i],"wall capacity")
        if cw<0 or (cw and walls[i] is not None):
            raise ValueError("wall capacity must be nonnegative and prescribed temperatures cannot store heat")
        initial = initial_wall_temperatures[i]
        if initial is not None and not cw:
            raise ValueError("initial wall temperature requires positive wall capacity")
        if film is not None and walls[i] is not None:
            raise ValueError("a film requires temperature None")
        if not imposed and film is None and not cw:
            continue
        if walls[i] is not None:
            raise ValueError("prescribe heat flux only on a wall with temperature None")
        # A unit inward flux decreases temperature by the series resistance
        # from this wall. Reflect this exact profile to obtain the wall load
        # for every crossing link, including links spanning a material contact.
        thickness = dx/torch.as_tensor(conductivity,dtype=state.dtype,device=state.device)
        contacts = (torch.zeros_like(thickness[:-1]) if contact_resistance_x is None else
                    torch.as_tensor(contact_resistance_x,dtype=state.dtype,device=state.device))
        if i:
            thickness,contacts = thickness.flip((0,)),contacts.flip((0,))
        distance = thickness.cumsum(0)-.5*thickness+torch.cat((thickness.new_zeros(1),contacts.cumsum(0)))
        reference = -(distance.flip((0,)) if i else distance)[None,:]
        fixed = [None,None]
        fixed[i] = 0.
        mode = response(reference,fixed)[0]-response(reference,(None,None))[0]
        if cw:
            col = 0 if i==0 else nx-1
            h,ambient = (0.,0.) if film is None else film
            if film is not None and wall_value(h,"film coefficient")<=0:
                raise ValueError("film coefficient must be positive")
            wall_initial.append(particles[col] if initial is None else wall_value(initial))
            storing.append(_StoringSlabWall(i,col,nx+len(storing),2/thickness[0],h,ambient,flux,mode))
        elif film is None:
            flux_modes.append((i,flux,mode))
        else:
            h,ambient = film
            if wall_value(h,"film coefficient")<=0:
                raise ValueError("film coefficient must be positive")
            gain = 1/(1/h+.5*thickness[0])
            film_modes.append((i,ambient,gain,mode,gain/h,gain*.5*thickness[0]))

    def forcing(time):
        reservoir = (profile(bulk_ambient_temperature(time),"bulk ambient temperature")
                     if fixed_reservoir is None else fixed_reservoir)
        load = static_load+loss*reservoir
        source = fixed_source
        if source is None:
            source = profile(heat_source(time),"heat_source")
            if return_source_heat:
                source = source.clone()  # A callback can reuse its profile buffer.
            load = load+source
        sampled = list(static_walls)
        for i,wall,mode in wall_modes:
            sampled[i] = wall_value(wall(time))
            load = load+mode*sampled[i]
        inward = state.new_zeros(2)
        components = state.new_zeros((2,2)) if return_wall_heat_components else None
        for side,flux,mode in flux_modes:
            q = wall_value(flux(time) if callable(flux) else flux,"wall flux")
            if components is not None:
                components[0, side] = q
            inward[side] = q
            load = load+q*mode
        for side,ambient,gain,mode,heater_gain,film_fraction in film_modes:
            value = ambient(time) if callable(ambient) else ambient
            film_drive = gain*wall_value(value,"ambient temperature")
            flux = fluxes[side]
            applied = wall_value(flux(time) if callable(flux) else flux,"wall flux")
            # Eliminate the massless surface using Tw = Tp + Rhalf*q_in.
            inward[side] = film_drive + heater_gain*applied
            if components is not None:
                components[0, side] = applied
                components[1, side] = film_drive - film_fraction*applied
            load = load+inward[side]*mode
        if storing:
            wall_loads = []
            for wall in storing:
                ambient,flux = wall.ambient,wall.applied_flux
                fluid = wall_value(ambient(time) if callable(ambient) else ambient,"ambient temperature")
                applied = wall_value(flux(time) if callable(flux) else flux,"wall flux")
                if components is not None:
                    components[0, wall.side] = applied
                    components[1, wall.side] = wall.film*fluid
                inward[wall.side] = wall.film*fluid+applied
                wall_loads.append(inward[wall.side])
            load = torch.cat((load,torch.stack(wall_loads)))
        return load,sampled,inward,reservoir,components,source
    basis = torch.eye(len(state),device=state.device,dtype=state.dtype)
    operator = response(basis,homogeneous).T-torch.diag(loss)
    for side,_,gain,mode,_,_ in film_modes:
        operator[:,0 if side==0 else -1] -= gain*mode
    if storing:
        count = len(storing)
        coupled = state.new_zeros((nx+count,nx+count))
        coupled[:nx,:nx] = operator
        for wall in storing:
            col,j,g = wall.column,wall.index,wall.conductance
            # The half-column conducts g*(Tw-Tp) into the material. The
            # opposite power and the external film enter the wall ODE.
            coupled[:nx,col] -= g*wall.mode
            coupled[:nx,j] = g*wall.mode
            coupled[j,col] = g
            coupled[j,j] = -(wall.film+g)
        operator = coupled
        capacity = torch.cat((capacity,state.new_tensor([wall_areal_heat_capacity[w.side] for w in storing])))
        state = torch.cat((state,torch.stack([torch.as_tensor(v,device=state.device,dtype=state.dtype) for v in wall_initial])))
    mass = torch.diag(capacity)
    generator = operator/capacity[:,None] if exponential else None
    previous_shift = None
    fields = [state]
    if return_interval_temperature:
        integrated_temperature = state.new_zeros((len(grid)-1,nx))
    segments = sorted(set(grid)|events)
    damped = set()
    if rannacher:
        if theta!=.5:
            raise ValueError("Rannacher damping requires theta=.5")
        for start,stop in zip(segments,segments[1:]):
            if start==grid[0] or start in events:
                middle = start+(stop-start)/2
                if not start<middle<stop:
                    raise ValueError("damped interval has no representable midpoint")
                damped.update((start,middle))
        segments = sorted(set(segments)|damped)
    if return_external_heat:
        external = state.new_zeros((len(grid)-1,2))
    need_section_flux = return_x_heat or return_directional_x_heat or return_external_heat
    if return_directional_x_heat:
        rightward = state.new_zeros((len(grid)-1,nx+1))
        leftward = torch.zeros_like(rightward)
    if need_section_flux:
        if return_x_heat:
            heat = state.new_zeros((len(grid)-1,nx+1))
        calibration = lattice_diffusion_factor_2d(h_factor,dx,mode="moment")

    def section_flux(stage, stage_walls, inward):
        # Start at the directly tallied west wall power. Net pair heat
        # into each column is west minus east; integrate across all cuts.
        rate,powers = stencil_rhs_with_wall_power_2d(stage[None,:nx],offsets,dx,stage_walls,
                        conductivity=conductivity,layered_x=True,contact_resistance_x=contact_resistance_x)
        west = powers[:1]/(dx*calibration)
        flux = torch.cat((west,west-dx*rate[0].cumsum(dim=0)/calibration))
        wall_drive = None
        if flux_modes or film_modes or storing:
            inward = inward.clone()
            for side,_,gain,mode,_,_ in film_modes:
                col = 0 if side==0 else nx-1
                inward[side] -= gain*stage[col]
                flux[1:] -= dx*inward[side]*mode.cumsum(0)
            for side,_,mode in flux_modes:
                flux[1:] -= dx*inward[side]*mode.cumsum(0)
            wall_drive = inward.clone() if storing and return_external_heat else None
            for wall in storing:
                inward[wall.side] = wall.conductance*(stage[wall.index]-stage[wall.column])
                flux[1:] -= dx*inward[wall.side]*wall.mode.cumsum(0)
            flux += inward[0]
        return flux, wall_drive

    if return_source_heat:
        source_heat = state.new_zeros((len(grid)-1,nx))
    if return_bulk_exchange:
        exchange = state.new_zeros((len(grid)-1,nx))
    if return_wall_heat_components:
        component_heat = state.new_zeros((len(grid)-1,2,2))
    load_start,walls_start,flux_start,reservoir_start,components_start,source_start = forcing(grid[0])
    if return_interval_wall_temperatures:
        integrated_walls = state.new_zeros((len(grid)-1,2))
    if return_wall_temperatures or return_interval_wall_temperatures:
        end_indices = [0,-1]
        end_resistance = .5*dx/torch.as_tensor(conductivity,device=state.device,dtype=state.dtype)[end_indices]

        def wall_trace(particles,temperatures,inward):
            end_values = particles[:nx][end_indices]
            surface = end_values+inward*end_resistance
            for side,_,gain,_,_,_ in film_modes:
                surface[side] -= end_resistance[side]*gain*end_values[side]
            for side,value in enumerate(temperatures):
                if value is not None:
                    surface[side] = value
            for wall in storing:
                surface[wall.side] = particles[wall.index]
            return surface

    if return_wall_temperatures:
        wall_history = [wall_trace(state,walls_start,flux_start)]
    for start,stop in zip(segments,segments[1:]):
        dt = stop-start
        weight = 1. if start in damped else theta
        if sdirk:
            weight = 1-1/math.sqrt(2)
            stage_time = start+weight*dt
            if not start<stage_time<stop:
                raise ValueError("SDIRK interval has no representable stage time")
            load_start,walls_start,flux_start,reservoir_start,components_start,source_start = forcing(stage_time)
        at_jump = stop in events
        load_end,walls_end,flux_end,reservoir_end,components_end,source_end = forcing(math.nextafter(stop,start) if at_jump else stop)
        load = (1-weight)*load_start+weight*load_end
        shift = weight*dt
        if exponential:
            midpoint_load = None
            if quadratic:
                middle = start+.5*dt
                if not start<middle<stop:
                    raise ValueError("quadratic load interval has no representable midpoint")
                midpoint_load,middle_walls,middle_flux,middle_reservoir,middle_components,middle_source = forcing(middle)
            end,stage = _polynomial_exponential_step(state,generator,capacity,load_start,load_end,dt,midpoint_load)
        else:
            if shift!=previous_shift:
                factors = torch.linalg.lu_factor(mass-shift*operator)
                previous_shift = shift
            if sdirk:
                stored = capacity*state
                first = torch.linalg.lu_solve(*factors,(stored+shift*load_start)[:,None])[:,0]
                rhs = stored+dt*(load+(1-weight)*(operator@first))
                end = torch.linalg.lu_solve(*factors,rhs[:,None])[:,0]
            else:
                first = state
                delta = torch.linalg.lu_solve(*factors,(dt*(operator @ state+load))[:,None])[:,0]
                end = state+delta
            if need_section_flux or return_bulk_exchange or return_interval_temperature or return_interval_wall_temperatures or return_wall_heat_components:
                stage = (1-weight)*first+weight*end
        if return_wall_heat_components:
            drive = (1-weight)*components_start + weight*components_end
            if quadratic:
                drive = (drive+2*middle_components)/3
            for side, _, gain, _, _, _ in film_modes:
                drive[1, side] -= gain*stage[0 if side == 0 else nx-1]
            for wall in storing:
                drive[1, wall.side] -= wall.film*stage[wall.index]
            component_heat[len(fields)-1] += dt*drive
        if return_interval_temperature:
            integrated_temperature[len(fields)-1] += dt*stage[:nx]
        if return_source_heat:
            source_mean = (1-weight)*source_start + weight*source_end
            if quadratic:
                source_mean = (source_mean+2*middle_source)/3
            source_heat[len(fields)-1] += dt*dx*source_mean
        if return_bulk_exchange:
            reservoir = (1-weight)*reservoir_start+weight*reservoir_end
            if quadratic:
                reservoir = (reservoir+2*middle_reservoir)/3
            exchange[len(fields)-1] += dt*dx*loss*(reservoir-stage[:nx])
        if need_section_flux or return_interval_wall_temperatures:
            stage_walls = tuple(None if a is None else (1-weight)*a+weight*b for a,b in zip(walls_start,walls_end))
            if quadratic:
                stage_walls = tuple(None if a is None else (a+2*m)/3 for a,m in zip(stage_walls,middle_walls))
            inward = (1-weight)*flux_start+weight*flux_end
            if quadratic:
                inward = (inward+2*middle_flux)/3
            if return_interval_wall_temperatures:
                integrated_walls[len(fields)-1] += dt*wall_trace(stage,stage_walls,inward)
        if need_section_flux:
            flux, wall_drive = section_flux(stage, stage_walls, inward)
            if return_directional_x_heat:
                final_flux, _ = section_flux(end, walls_end, flux_end)
                initial_flux = flux+(flux-final_flux)
                crossing = ((initial_flux > 0) & (final_flux < 0)) | ((initial_flux < 0) & (final_flux > 0))
                small = torch.minimum(initial_flux.abs(), final_flux.abs())
                large = torch.maximum(initial_flux.abs(), final_flux.abs())
                ratio = small/torch.where(large > 0, large, 1.)
                minor = torch.where(crossing, .5*small*(ratio/(1+ratio)), 0.)
                rightward[len(fields)-1] += dt*(minor+flux.clamp_min(0))
                leftward[len(fields)-1] += dt*(minor+(-flux).clamp_min(0))
            if return_x_heat:
                heat[len(fields)-1] += dt*flux
            if return_external_heat:
                net = torch.stack((flux[0],-flux[-1]))
                for wall in storing:
                    net[wall.side] = wall_drive[wall.side]-wall.film*stage[wall.index]
                external[len(fields)-1] += dt*net
        state = end
        load_start,walls_start,flux_start,reservoir_start,components_start,source_start = forcing(stop) if at_jump else (load_end,walls_end,flux_end,reservoir_end,components_end,source_end)
        if stop==grid[len(fields)]:
            fields.append(state)
            if return_wall_temperatures:
                wall_history.append(wall_trace(state,walls_start,flux_start))
    history = torch.stack(fields)[:,:nx]
    outputs = [history]
    if return_x_heat:
        outputs.append(heat)
    if return_wall_temperatures:
        outputs.append(torch.stack(wall_history))
    if return_bulk_exchange:
        outputs.append(exchange)
    if return_external_heat:
        outputs.append(external)
    if return_interval_temperature or return_interval_wall_temperatures:
        widths = state.new_tensor([b-a for a,b in zip(grid,grid[1:])])
    if return_interval_temperature:
        outputs.append(integrated_temperature/widths[:,None])
    if return_interval_wall_temperatures:
        outputs.append(integrated_walls/widths[:,None])
    if return_wall_heat_components:
        outputs.append(dict(applied=component_heat[:,0], film=component_heat[:,1]))
    if return_source_heat:
        outputs.append(source_heat)
    if return_directional_x_heat:
        outputs.append(dict(rightward=rightward, leftward=leftward))
    return tuple(outputs) if len(outputs)>1 else history


def step_transient_slab_2d(temperature, dt, dx, x_wall_temperatures, *, heat_source=0.,
                           conductivity=1., volumetric_heat_capacity=1., theta=.5,
                           h_factor=1.3, x_wall_inward_flux=(0.,0.), return_x_flux=False, heat_loss_coefficient=0.):
    """Advance a homogeneous reflected slab one implicit theta step of duration dt (s).

    Supply real particle temperatures (ny,nx), periodic in y. Supply the
    theta-weighted start and end values of volumetric heating (W/m³), wall
    temperatures (K, scalar or per row), and scalar or per-row inward wall fluxes (W/m²)
    on sides whose wall temperature is None. theta=.5 is Crank–Nicolson and
    theta=1 is backward Euler. Optional section fluxes point right and are
    time-step averages, so dt times the flux difference gives heat per area.
    The bulk sink H*T uses the same theta stage as conduction; H is in W/(m³ K).
    """
    if not (.5 <= theta <= 1 and math.isfinite(dt) and dt > 0
            and math.isfinite(volumetric_heat_capacity) and volumetric_heat_capacity > 0):
        raise ValueError("require positive finite dt and capacity, and theta in [0.5,1]")
    field = torch.as_tensor(temperature)
    if field.ndim != 2 or min(field.shape)==0 or field.is_complex():
        raise ValueError("provide a real nonempty (ny,nx) temperature field")
    field = field.to(DTYPE)
    if torch.is_complex(torch.as_tensor(heat_source)) or any(
            wall is not None and torch.is_complex(torch.as_tensor(wall)) for wall in x_wall_temperatures):
        raise ValueError("transient wall temperatures and heating must be real")
    source = torch.as_tensor(heat_source, device=field.device, dtype=DTYPE)
    if source.shape not in ((),field.shape):
        raise ValueError("heat_source must be scalar or have the temperature field shape")
    shift = volumetric_heat_capacity/(theta*dt)
    total_shift = shift+_heat_loss(heat_loss_coefficient)
    if not math.isfinite(total_shift) or shift <= 0:
        raise ValueError("capacity divided by theta*dt must be finite and positive")
    result = _solve_slab_modes(shift*field+source, dx, x_wall_temperatures,
                               conductivity, h_factor, None, total_shift, x_wall_inward_flux, return_x_flux)
    stage, flux = result if return_x_flux else (result, None)
    updated = stage if theta==1. else field+(stage-field)/theta
    return (updated,flux) if return_x_flux else updated


def solve_transient_slab_2d(temperature, times, dx, x_wall_temperatures, *,
                            wall_jump_times=(), rannacher=False, heat_source=0.,
                            conductivity=1., volumetric_heat_capacity=1., theta=.5,
                            h_factor=1.3, x_wall_inward_flux=(0.,0.), return_x_heat=False, heat_loss_coefficient=0.,
                            method="theta", return_wall_heat_by_row=False, wall_gradient_fit="wall", temperature_reference=0.,
                            return_interval_temperatures=False, ambient_temperature=None, return_bulk_exchange=False,
                            return_source_heat=False):
    """Advance a reflected slab over requested times with prescribed wall histories.

    Each wall may be a fixed scalar, row profile, callable of time returning
    either, or None for prescribed flux. Volumetric heating (scalar or field)
    and each inward wall flux (scalar or row profile) may also depend on time.
    heat_loss_coefficient H adds volumetric exchange H*(Ta-T). The ambient_temperature
    Ta defaults to zero and may be a scalar, a particle field or a callable of time
    returning either. Supply other heating separately to retain weak sources.
    List discontinuities of any load in wall_jump_times and return the new
    value at the event time.
    Rannacher damping applies two backward Euler half steps at startup and
    immediately after each jump. The default integrator is Crank–Nicolson.
    method="sdirk2" samples all prescribed data at its two stages and integrates
    the section heats with the same second-order L-stable formula.
    Initial and wall temperatures are absolute. Return particle temperatures
    relative to temperature_reference in K, shaped (nt,ny,nx). With return_x_heat also return
    the interval-integrated rightward heat on each section (nt-1,nx+1), J/m².
    With return_wall_heat_by_row, append inward heat at each west and east
    wall row (nt-1,2,ny), J/m². At flux walls integrate the prescribed load.
    At temperature walls recover the gradient of a smooth field using at
    least five columns and match the conservative section mean. The default
    wall_gradient_fit="wall" uses the prescribed temperature and centers 2–4.
    Select "interior" to extrapolate from centers 2–5 for curved profiles
    driven by spatially varying wall temperatures. With return_interval_temperatures,
    append stage-weighted mean particle temperatures (nt-1,ny,nx) relative to
    temperature_reference, integrating every internal substep in each output interval.
    With return_bulk_exchange, append signed ambient heat absorbed per particle
    volume (nt-1,ny,nx), J/m³, integrated with the same stages as the field.
    With return_source_heat, append the integrated prescribed volumetric heat
    source, also in J/m³ per particle and output interval. Positive values
    add energy. This output follows all other requested arrays.
    """
    grid = [float(t) for t in times]
    jumps = [float(t) for t in wall_jump_times]
    if len(grid)<2 or not all(math.isfinite(t) for t in grid) or any(a>=b for a,b in zip(grid,grid[1:])):
        raise ValueError("times must be finite and strictly increasing")
    if len(set(jumps))!=len(jumps) or not all(grid[0]<t<grid[-1] for t in jumps):
        raise ValueError("wall_jump_times must be distinct interior times")
    if rannacher and theta!=.5:
        raise ValueError("Rannacher damping requires theta=.5")
    if method not in ("theta","sdirk2"):
        raise ValueError("method must be theta or sdirk2")
    if method=="sdirk2" and (rannacher or theta!=.5):
        raise ValueError("sdirk2 cannot be combined with Rannacher or a theta override")
    walls = tuple(x_wall_temperatures)
    if len(walls)!=2:
        raise ValueError("provide west and east wall temperatures")
    fluxes = tuple(x_wall_inward_flux)
    if len(fluxes)!=2:
        raise ValueError("provide west and east inward fluxes")
    def sample_load(load,start,end,weight):
        if not callable(load):
            return (load,)
        stop_value = load(end)
        if weight == 1.:
            return (stop_value,)
        # A load may update and return the same profile buffer at each time.
        if torch.is_tensor(stop_value):
            stop_value = stop_value.clone()
        elif not isinstance(stop_value, (int, float, complex)):
            import copy
            stop_value = copy.deepcopy(stop_value)
        return load(start), stop_value

    def stage_value(samples,weight,offset=None):
        def relative(value):
            if value is None or offset is None:
                return value
            value = torch.as_tensor(value, device=field.device,
                                    dtype=torch.complex128 if torch.is_complex(torch.as_tensor(value)) else field.dtype)
            return value-offset
        if len(samples) == 1:
            return relative(samples[0])
        start_value, stop_value = map(relative, samples)
        if any(torch.is_complex(torch.as_tensor(v)) for v in (start_value,stop_value)):
            raise ValueError("transient loads must be real")
        return ((1-weight)*torch.as_tensor(start_value,device=field.device,dtype=field.dtype)
                +weight*torch.as_tensor(stop_value,device=field.device,dtype=field.dtype))
    def stage_load(load,start,end,weight,offset=None):
        return stage_value(sample_load(load,start,end,weight),weight,offset)

    field = torch.as_tensor(temperature)
    if field.ndim!=2 or min(field.shape)==0 or field.is_complex():
        raise ValueError("provide a real nonempty (ny,nx) temperature field")
    field = field.to(DTYPE)
    if not math.isfinite(temperature_reference):
        raise ValueError("temperature_reference must be finite")
    datum = field[0, 0]
    field = field - datum
    loss = _heat_loss(heat_loss_coefficient)
    bulk_offset = loss * datum
    if wall_gradient_fit not in ("wall", "interior"):
        raise ValueError("wall_gradient_fit must be wall or interior")
    recover_gradient = return_wall_heat_by_row and any(w is not None for w in walls)
    if recover_gradient and field.shape[1] < 5:
        raise ValueError("rowwise heat at a temperature wall requires at least five columns")
    need_flux = return_x_heat or recover_gradient
    def advance(initial,dt,start,load_end,weight):
        stage = tuple(stage_load(wall,start,load_end,weight,datum) for wall in walls)
        stage_flux = tuple(stage_load(q,start,load_end,weight) for q in fluxes)
        samples = sample_load(heat_source,start,load_end,weight)
        ambient = -datum
        if ambient_temperature is None or loss == 0:
            source = stage_value(samples,weight,bulk_offset)
        else:
            ambient = stage_load(ambient_temperature,start,load_end,weight,datum)
            source = stage_value(samples,weight,0.)
            if any(value.shape not in ((),field.shape) for value in (ambient,source)):
                raise ValueError("ambient temperature and heat source must be scalar or match the field shape")
            source = source + loss*ambient
        result = step_transient_slab_2d(initial,dt,dx,stage,heat_source=source,
                     conductivity=conductivity,volumetric_heat_capacity=volumetric_heat_capacity,
                     theta=weight,h_factor=h_factor,x_wall_inward_flux=stage_flux,return_x_flux=need_flux,
                     heat_loss_coefficient=heat_loss_coefficient)
        updated, flux = result if need_flux else (result, None)
        wall_flux = None
        if return_wall_heat_by_row or return_bulk_exchange:
            stage_field = updated if weight == 1. else initial + weight*(updated-initial)
        if return_wall_heat_by_row:
            recovered = []
            for side, (wall, load) in enumerate(zip(stage, stage_flux)):
                if wall is None:
                    profile = _wall_column(load, stage_field).reshape(-1).expand(field.shape[0])
                else:
                    inward_field = stage_field if side == 0 else stage_field.flip((1,))
                    if wall_gradient_fit == "interior":
                        gradient_flux = _interior_wall_flux(inward_field[:, 1:5], conductivity, dx)
                    else:
                        samples = inward_field[:, 1:4] - _wall_column(wall, stage_field)
                        gradient_flux = _uniform_fixed_wall_flux(samples, conductivity, dx)
                    mean = flux[0] if side == 0 else -flux[-1]
                    profile = mean + (gradient_flux - gradient_flux.mean())
                recovered.append(profile)
            wall_flux = torch.stack(recovered)
        exchange = loss*(ambient-stage_field) if return_bulk_exchange else None
        source_power = stage_value(samples,weight,0.) if return_source_heat else None
        return updated, flux, wall_flux, exchange, source_power

    history = [field]
    heat = field.new_zeros((len(grid)-1,field.shape[1]+1)) if return_x_heat else None
    wall_heat = field.new_zeros((len(grid)-1, 2, field.shape[0])) if return_wall_heat_by_row else None
    averages = field.new_zeros((len(grid)-1, *field.shape)) if return_interval_temperatures else None
    bulk = field.new_zeros((len(grid)-1, *field.shape)) if return_bulk_exchange else None
    source_heat = field.new_zeros((len(grid)-1, *field.shape)) if return_source_heat else None
    events = set(jumps)
    segments = sorted(set(grid)|events)
    interval = 0
    for start,stop in zip(segments,segments[1:]):
        damp = rannacher and (start==grid[0] or start in events)
        ends = (start+(stop-start)/2,stop) if damp else (stop,)
        if damp and not start<ends[0]<stop:
            raise ValueError("damped interval has no representable midpoint")
        weight = 1. if damp else theta
        for end in ends:
            load_end = math.nextafter(end,start) if end in events else end
            dt = end-start
            if method=="sdirk2":
                gamma = 1-1/math.sqrt(2)
                stage_time = start+gamma*dt
                if not start<stage_time<end:
                    raise ValueError("SDIRK interval has no representable stage time")
                first,flux1,wall_flux1,exchange1,source1 = advance(field,gamma*dt,start,stage_time,1.)
                # Advance the explicit first-stage contribution before the
                # second implicit solve, using the same diagonal RK weight.
                base = field+((1-gamma)/gamma)*(first-field)
                field,flux,wall_flux,exchange,source_power = advance(base,gamma*dt,stage_time,load_end,1.)
                if return_source_heat:
                    source_power = (1-gamma)*source1 + gamma*source_power
                if return_bulk_exchange:
                    exchange = (1-gamma)*exchange1 + gamma*exchange
                if return_interval_temperatures:
                    stage_mean = (1-gamma)*first + gamma*field
                if return_x_heat:
                    flux = (1-gamma)*flux1+gamma*flux
                if return_wall_heat_by_row:
                    wall_flux = (1-gamma)*wall_flux1 + gamma*wall_flux
            else:
                previous = field
                field,flux,wall_flux,exchange,source_power = advance(previous,dt,start,load_end,weight)
                if return_interval_temperatures:
                    stage_mean = field if weight == 1. else previous + weight*(field-previous)
            if return_interval_temperatures:
                averages[interval] += dt*stage_mean
            if return_x_heat:
                heat[interval] += dt*flux
            if return_wall_heat_by_row:
                wall_heat[interval] += dt*wall_flux
            if return_bulk_exchange:
                bulk[interval] += dt*exchange
            if return_source_heat:
                source_heat[interval] += dt*source_power
            start = end
        if stop==grid[interval+1]:
            history.append(field)
            interval += 1
    fields = torch.stack(history) + (datum-temperature_reference)
    outputs = [fields]
    if return_x_heat:
        outputs.append(heat)
    if return_wall_heat_by_row:
        outputs.append(wall_heat)
    if return_interval_temperatures:
        durations = field.new_tensor([b-a for a,b in zip(grid,grid[1:])])
        outputs.append(averages/durations[:, None, None] + (datum-temperature_reference))
    if return_bulk_exchange:
        outputs.append(bulk)
    if return_source_heat:
        outputs.append(source_heat)
    return tuple(outputs) if len(outputs)>1 else fields


def _solve_slab_modes(heat_source, dx, x_wall_temperatures, conductivity, h_factor,
                      mean_temperature, shift, x_wall_inward_flux, return_x_flux, temperature_reference=0., ambient_temperature=None, separate_mean=False):
    source = torch.as_tensor(heat_source, dtype=torch.complex128 if isinstance(shift, complex) else DTYPE)
    if source.ndim != 2 or min(source.shape) == 0 or not torch.isfinite(source).all():
        raise ValueError("heat_source must be a finite nonempty (ny, nx) field")
    if not all(math.isfinite(v) and v > 0 for v in (dx, conductivity, h_factor)):
        raise ValueError("dx, conductivity and h_factor must be positive and finite")
    left, right = x_wall_temperatures
    closed = left is None and right is None
    if separate_mean and not closed:
        raise ValueError("separate_mean requires two flux walls")
    if not source.is_complex() and any(torch.is_complex(torch.as_tensor(q)) for q in x_wall_inward_flux):
        raise ValueError("real slab solves require real wall fluxes")
    west, east = (_wall_column(q, source) for q in x_wall_inward_flux)
    if not all(torch.isfinite(q).all() for q in (west, east)):
        raise ValueError("wall fluxes must be finite")
    west_loaded, east_loaded = bool((west != 0).any()), bool((east != 0).any())
    if (left is not None and west_loaded) or (right is not None and east_loaded):
        raise ValueError("prescribe heat flux only on a wall with temperature None")
    length = source.shape[1]*dx
    if closed and not shift:
        if mean_temperature is None or not math.isfinite(mean_temperature):
            raise ValueError("two flux walls require a finite mean_temperature")
        # Check physical input before the analytical lift cancels its forcing.
        net = source.mean()+(west.mean()+east.mean())/length
        scale = source.abs().mean()+(west.abs().mean()+east.abs().mean())/length
        if net.abs() > 64*torch.finfo(source.dtype).eps*scale:
            raise ValueError("net source plus inward wall heat must vanish (zero net heat input)")
    elif mean_temperature is not None:
        raise ValueError("mean_temperature applies only to two flux walls in a steady solve without bulk heat loss")
    lift = 0.
    if west_loaded or east_loaded:
        x = (torch.arange(source.shape[1], device=source.device, dtype=DTYPE)+0.5)*dx
        if closed:
            # -k T'(0)=west, +k T'(L)=east. Center the lift for the steady gauge.
            curvature_k = (west+east)/length
            lift = (0.5*curvature_k*x.square()-west*x)/conductivity
            lift = lift-lift.mean()
            source = source + curvature_k
        else:
            # The affine lift vanishes at the fixed-temperature wall.
            lift = (west*(length-x) if right is not None else east*x)/conductivity
        if lift.ndim == 2 and lift.shape[0] > 1:
            # The polynomial in x has periodic row-dependent coefficients.
            # Differentiate their trigonometric interpolant to supply k*T_yy.
            waves = 2*math.pi*torch.fft.fftfreq(lift.shape[0],d=dx,device=source.device,dtype=DTYPE)
            lateral = torch.fft.ifft(-waves[:,None].square()*torch.fft.fft(lift,dim=0),dim=0)
            source = source + conductivity*(lateral if source.is_complex() else lateral.real)
        if shift:
            source = source - shift*lift
    for wall in (left, right):
        if wall is not None and not torch.isfinite(_wall_column(wall, source)).all():
            raise ValueError("wall temperatures must be finite")
    offsets = lattice_offsets(dx, h_factor*dx)
    factor = -math.fsum(a*(n*dx)**2 for (_, n), a in offsets)
    if factor <= 0:
        raise ValueError("kernel support must include a conducting neighbour")
    # Lift the inhomogeneous wall forcing, leaving a homogeneous reflected operator.
    reference = (mean_temperature or 0.) if closed else _wall_column(left if left is not None else right, source).mean()
    ambient = None
    if ambient_temperature is not None:
        if not source.is_complex() and torch.is_complex(torch.as_tensor(ambient_temperature)):
            raise ValueError("steady ambient temperature must be real")
        ambient = torch.as_tensor(ambient_temperature, dtype=source.dtype, device=source.device)
        if ambient.shape not in ((), source.shape) or not torch.isfinite(ambient).all():
            raise ValueError("ambient temperature must be finite and scalar or match the field shape")
        if closed and shift and not source.is_complex():
            reference = ambient.reshape(-1)[0]
    shifted = tuple(None if wall is None else torch.as_tensor(wall, dtype=source.dtype,
                    device=source.device)-reference for wall in (left, right))
    wall_rate = stencil_rhs_2d(torch.zeros_like(source), offsets, x_wall_temperatures=shifted)
    rhs = source + (conductivity/factor)*wall_rate
    if shift:
        if ambient is None:
            rhs = rhs - shift*reference
        else:
            rhs = rhs + shift.real*(ambient-reference)
            if source.is_complex():
                rhs = rhs - 1j*shift.imag*reference
    right_image = rhs.flip((1,))
    rhs = torch.cat((rhs, right_image if right is None else -right_image), dim=1)
    if (left is None) != (right is None):
        # Odd at one wall and even at the other gives an antiperiod over 2*nx.
        rhs = torch.cat((rhs, -rhs), dim=1)
    ny, nx_ext = rhs.shape
    ky = torch.fft.fftfreq(ny, device=source.device, dtype=DTYPE)[:, None]
    xfreq = torch.fft.fftfreq if source.is_complex() else torch.fft.rfftfreq
    kx = xfreq(nx_ext, device=source.device, dtype=DTYPE)
    decay = torch.zeros((ny, len(kx)), dtype=DTYPE, device=source.device)
    for (m, n), a in offsets:
        decay += (-4*a*conductivity/factor)*torch.sin(math.pi*(m*ky+n*kx)).square()
    denominator = decay + shift
    if not shift:
        denominator[0, 0] = 1.
    modes = (torch.fft.fft2(rhs) if source.is_complex() else torch.fft.rfft2(rhs)) / denominator
    uniform_correction = 0.
    if closed and shift:
        # A weak bulk loss or low frequency amplifies the uniform temperature.
        # This mode carries no heat across a face.
        uniform_mode = modes[0, 0] if source.is_complex() else modes[0, 0].real
        uniform_correction = uniform_mode / rhs.numel()
        modes[0, 0] = 0.
    elif not shift:
        modes[0, 0] = 0.
    if source.is_complex():
        correction = torch.fft.ifft2(modes)[:, :source.shape[1]]
    else:
        correction = torch.fft.irfft2(modes, s=rhs.shape)[:, :source.shape[1]]
    temperature = lift + correction
    offset = uniform_correction + (reference - temperature_reference)
    if separate_mean:
        offset = torch.as_tensor(offset, dtype=source.dtype, device=source.device)
    else:
        temperature = temperature + offset
    if not return_x_flux:
        return (temperature, offset) if separate_mean else temperature

    # For constant k and periodic y, summing all crossing links over rows
    # commutes with the tangential shift in every diagonal pair.
    # Compute crossings relative to the same datum on particles and ghosts.
    # Adding it before differencing can erase a weak physical gradient.
    profile = correction.mean(dim=0)
    radius = max(abs(n) for (_, n), _ in offsets)
    low, high = profile[:radius].flip((0,)), profile[-radius:].flip((0,))
    if left is not None:
        low = 2*shifted[0].mean()-low
    if right is not None:
        high = 2*shifted[1].mean()-high
    extended = torch.cat((low, profile, high))
    face_count = profile.numel()+1
    flux = profile.new_zeros(face_count)
    for (_, n), a in offsets:
        if n > 0:
            jumps = extended[n:]-extended[:-n]
            crossings = jumps.unfold(0, n, 1)[radius-n:radius-n+face_count]
            flux += a*crossings.sum(dim=1)
    flux *= 2*conductivity*dx/factor
    if west_loaded or east_loaded:
        if closed:
            position = torch.arange(face_count, device=source.device, dtype=DTYPE)*dx
            flux += west.mean()-curvature_k.mean()*position
        else:
            flux += west.mean() if right is not None else -east.mean()
    return (temperature, flux, offset) if separate_mean else (temperature, flux)


def prepare_slab_film_rhs(template, offsets, dx, film_coefficients, *, rho=1.,
                          conductivity=None, layered_x=False, contact_resistance_x=None,
                          specific_heat=None, local_films=False):
    """Prepare the calibrated heat operator for two isothermal, massless walls.

    For fixed geometry and material data, return a callable (T, ambient_pair)
    yielding particle temperature rates, two inward powers and wall temperatures.
    Each wall has one temperature and exchanges total power h*A*(Tamb-Twall)
    with its ambient. A=ny*dx for unit thickness. h=0 insulates. The x second
    moment of the supplied symmetric lattice stencil normalizes conductivity.
    With local_films=True each wall row balances its own film and ghost-pair
    powers; h and ambient each accept two scalars or row profiles. Returned
    powers and wall temperatures then have shape (2, ny).
    """
    zero = torch.zeros_like(torch.as_tensor(template, dtype=DTYPE))
    if zero.ndim != 2 or min(zero.shape) == 0:
        raise ValueError("template must be a nonempty 2-D temperature grid")
    def wall_pair(values):
        if local_films:
            left, right = values
            return torch.stack([_wall_column(v, zero).reshape(-1).expand(zero.shape[0])
                                for v in (left, right)])
        pair = torch.as_tensor(values, dtype=zero.dtype, device=zero.device)
        if pair.shape != (2,):
            raise ValueError("supply one scalar per isothermal wall")
        return pair

    h = wall_pair(film_coefficients)
    if not torch.all(torch.isfinite(h) & (h >= 0)):
        raise ValueError("supply two nonnegative finite film coefficients")
    # Keep material data consistent with the precomputed wall responses even
    # if the caller later reuses or modifies its input arrays.
    def snapshot(value):
        return None if value is None else torch.as_tensor(value, dtype=zero.dtype, device=zero.device).clone()

    conductivity = snapshot(conductivity)
    specific_heat = snapshot(specific_heat)
    contact_resistance_x = snapshot(contact_resistance_x)
    rho = float(rho)
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    offsets = tuple(offsets)
    factor = -math.fsum(a * (n * dx)**2 for (_, n), a in offsets)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("the x diffusion moment must be finite and positive")
    film = h * (dx if local_films else zero.shape[0] * dx)

    def exchange(field, walls):
        rate, power = stencil_rhs_with_wall_power_2d(
            field, offsets, dx, walls, rho=rho, conductivity=conductivity,
            layered_x=layered_x, contact_resistance_x=contact_resistance_x,
            specific_heat=specific_heat, power_by_row=local_films)
        return rate / factor, power / factor

    left_rate, left_power = exchange(zero, (1., 0.))
    right_rate, right_power = exchange(zero, (0., 1.))
    conductance = torch.stack((left_power[0], right_power[1]))
    if not torch.all(conductance > 0):
        raise ValueError("both walls require positive wall-to-particle conductance")

    def evaluate(temperature, ambient):
        field = torch.as_tensor(temperature, dtype=zero.dtype, device=zero.device)
        ambient = wall_pair(ambient)
        if field.shape != zero.shape or not torch.isfinite(ambient).all():
            raise ValueError("supply the prepared field shape and two ambient temperatures")
        # Shift the origin to reduce cancellation at large absolute temperatures.
        origin = field.mean()
        base_rate, base_power = exchange(field - origin, (0., 0.))
        wall = (film * (ambient-origin) - base_power) / (film + conductance)
        if local_films:
            # Diagonal links couple a wall row to neighbouring particle rows.
            # Evaluate at the solved profiles to scatter that heat correctly.
            rate, power = exchange(field-origin, wall)
            return rate, power, wall + origin
        rate = base_rate + left_rate*wall[0] + right_rate*wall[1]
        power = base_power + conductance*wall
        return rate, power, wall + origin

    return evaluate


def _interior_wall_flux(samples, conductivity, dx):
    # Fit the cubic through centers 2–5. Differences cancel a common
    # temperature offset in the reflected solution near a curved boundary profile.
    weights = samples.new_tensor([333, -261, 71]) / 24
    return -(conductivity/dx) * ((samples[..., 1:] - samples[..., :1]) @ weights)


def _uniform_fixed_wall_flux(differences, conductivity, dx):
    # Derivative of the cubic through the wall and centers at 1.5, 2.5, 3.5 dx.
    weights = differences.new_tensor([35/12, -21/10, 15/28])
    return -(conductivity/dx) * (differences @ weights)


def recover_dirichlet_wall_flux_2d(temperature, dx, conductivity, wall_mean_flux, *,
                                   contact_resistance_x=None, x_wall_temperatures=None, side=None):
    """Recover rowwise inward flux (W/m²) at two fixed-temperature slab walls.

    For smooth fields on a uniform cell-centered (ny,nx) lattice, with a
    positive scalar or cellwise conductivity (one value per x column) and
    at least two columns with fixed wall samples or five without. Fit a cubic in conductive resistance through
    centers 2–5. Optional nonnegative contact resistances (one per internal
    face) add a temperature jump proportional to local flux. Supply the two
    inward wall means from the conservative stencil tally. When the two
    fixed wall temperatures are supplied, fit the wall and centers 2–4.
    On two- or three-column slabs with fixed samples use the available centers
    starting at center 2 for a linear or quadratic fit, respectively.
    Each temperature may be scalar or one value per periodic row.
    Set side="left" or "right" to recover only that fixed wall. In this
    mode supply a scalar inward mean, the opposite wall temperature may
    be None, and the result is a vector of length ny.
    """
    if side not in (None, "left", "right"):
        raise ValueError("side must be left or right when specified")
    sides = [0, 1] if side is None else [int(side == "right")]
    field = torch.as_tensor(temperature)
    minimum = 5 if x_wall_temperatures is None else 2
    if (field.ndim != 2 or field.shape[0] < 1 or field.shape[1] < minimum or
            not (field.is_floating_point() or field.is_complex()) or not torch.isfinite(field).all()):
        raise ValueError(f"provide a finite real or complex field with at least {minimum} columns")
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("spacing must be positive and finite")
    raw_k = torch.as_tensor(conductivity, device=field.device)
    if raw_k.is_complex() or raw_k.shape not in ((), (field.shape[1],)):
        raise ValueError("conductivity must be real and scalar or one value per column")
    k = torch.as_tensor(conductivity, dtype=field.real.dtype, device=field.device)
    if not torch.isfinite(k).all() or (k <= 0).any():
        raise ValueError("conductivity must be positive and finite")
    means = torch.as_tensor(wall_mean_flux, device=field.device)
    # Python sequences initially infer the default dtype. Convert their original
    # values directly when the field requires more precision.
    mean_dtype = torch.promote_types(field.dtype, means.dtype)
    if means.dtype != mean_dtype:
        means = torch.as_tensor(wall_mean_flux, dtype=mean_dtype, device=field.device)
    if means.shape != ((2,) if side is None else ()) or not torch.isfinite(means).all():
        raise ValueError("provide a finite inward mean per selected wall")
    means = means.reshape(-1)
    width = 5 if x_wall_temperatures is None else min(4, field.shape[1])
    samples = torch.stack((field[:,1:width],field[:,-width:-1].flip((1,))))[sides]
    if x_wall_temperatures is not None:
        if len(x_wall_temperatures) != 2 or any(x_wall_temperatures[i] is None for i in sides):
            raise ValueError("provide the selected fixed wall temperatures")
        walls = []
        for i in sides:
            value = x_wall_temperatures[i]
            if torch.as_tensor(value).is_complex() and not field.is_complex():
                raise ValueError("complex wall temperatures require a complex field")
            wall = _wall_column(value, field).expand(field.shape[0], 1)
            if not torch.isfinite(wall).all():
                raise ValueError("wall temperatures must be finite")
            walls.append(wall)
        differences = samples - torch.stack(walls)
    else:
        differences = samples[:,:,1:] - samples[:,:,:1]
    if width >= 4 and k.ndim == 0 and contact_resistance_x is None:
        if x_wall_temperatures is None:
            flux = _interior_wall_flux(samples, k, dx)
        else:
            flux = _uniform_fixed_wall_flux(differences, k, dx)
    else:
        k = k.expand(field.shape[1]).to(torch.float64)
        resistance = dx / torch.stack((k[:width], k[-width:].flip((0,))))[sides]
        edges = resistance.cumsum(dim=1)
        centers = (edges - resistance/2)[:,1:]
        scale = centers[:,-1:]
        nodes = centers / scale
        degree = min(3, width-1)
        powers = torch.arange(1, degree+1, device=field.device)
        basis = nodes[:,:,None] ** powers
        if contact_resistance_x is not None:
            raw = torch.as_tensor(contact_resistance_x, device=field.device)
            if raw.is_complex() or raw.shape != (field.shape[1]-1,):
                raise ValueError("provide one real contact resistance per internal face")
            faces = torch.as_tensor(contact_resistance_x, dtype=k.dtype, device=k.device)
            if not torch.isfinite(faces).all() or (faces < 0).any():
                raise ValueError("contact resistance must be finite and nonnegative")
            contacts = torch.stack((faces[:width-1], faces[-(width-1):].flip((0,))))[sides]
            # For each basis function p, add R*p'(r_face) to all downstream
            # temperatures. Conductive distance r excludes contact resistance.
            face_nodes = edges[:,:-1] / scale
            jumps = (contacts/scale)[:,:,None] * powers * face_nodes[:,:,None] ** (powers-1)
            basis = basis + jumps.cumsum(dim=1)
        # The constant is fixed by the wall or eliminated using center 2.
        matrix = basis[:,1:] - basis[:,:1] if x_wall_temperatures is None else basis
        target = matrix.new_zeros(len(sides), degree)
        target[:, 0] = 1.
        weights = torch.linalg.solve(matrix.transpose(1,2), target.unsqueeze(-1)).squeeze(-1)
        derivative = (weights / scale).to(field.real.dtype)
        flux = -(differences * derivative[:,None,:]).sum(dim=2)
    recovered = means[:,None] + (flux-flux.mean(dim=1,keepdim=True))
    return recovered if side is None else recovered[0]


def harmonic_wall_interval_heat(wall_flux, angular_frequency, start_time, end_time, *, steady_flux=0.):
    """Signed inward heat per wall row (J/m²) between two times in seconds.

    The instantaneous inward flux is steady_flux + Re(wall_flux * exp(iωt)).
    Supply peak phasors in W/m². Reversing the time limits reverses the sign.
    """
    if not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be positive and finite")
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError("integration times must be finite")
    if torch.is_complex(torch.as_tensor(steady_flux)):
        raise ValueError("steady_flux must be real")
    flux = torch.as_tensor(wall_flux, dtype=torch.complex128)
    bias = torch.as_tensor(steady_flux, dtype=torch.float64, device=flux.device)
    if not torch.isfinite(flux).all() or not torch.isfinite(bias).all():
        raise ValueError("wall fluxes must be finite")
    duration = end_time - start_time
    half_phase = angular_frequency * duration / 2
    midpoint_phase = angular_frequency * (start_time + duration/2)
    # Integrating about the midpoint avoids subtracting nearby phasors.
    average = math.sin(half_phase)/half_phase if half_phase else 1.
    midpoint_flux = flux.real*math.cos(midpoint_phase) - flux.imag*math.sin(midpoint_phase)
    mean = bias + average*midpoint_flux
    # Near a grazing zero, evaluate 1 - sinc(h)*cos(delta) as two
    # small positive terms. The squared chord gives 1 - cos(delta)
    # without subtracting nearly identical cosines.
    if abs(half_phase) < .05:
        h2 = half_phase**2
        sinc_loss = h2*(1/6 - h2*(1/120 - h2*(1/5040 - h2/362880)))
        amplitude = flux.abs()
        sign = torch.sign(bias)
        near_zero = (amplitude > 0) & (mean.abs() < .125*amplitude) & (bias != 0)
        scale = torch.where(amplitude > 0, amplitude, 1.)
        unit = flux / scale
        chord_real = unit.real + sign*math.cos(midpoint_phase)
        chord_imag = unit.imag - sign*math.sin(midpoint_phase)
        cosine_loss = (chord_real.square() + chord_imag.square()) / 2
        correction = amplitude*(sinc_loss + average*cosine_loss)
        # A small quadrature component can change the amplitude by less than
        # one ulp. Rationalize hypot(major, minor) - major in that case.
        major = torch.maximum(flux.real.abs(), flux.imag.abs())
        minor = torch.minimum(flux.real.abs(), flux.imag.abs())
        increment = minor*((minor/scale)/(1 + major/scale))
        gap = torch.where(minor < .125*major, (bias.abs()-major)-increment, bias.abs()-amplitude)
        mean = torch.where(near_zero, sign*(gap + correction), mean)
    return duration * mean


def _wall_cycle_geometry(wall_flux, angular_frequency, steady_flux):
    if not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be positive and finite")
    if torch.is_complex(torch.as_tensor(steady_flux)):
        raise ValueError("steady_flux must be real")
    flux = torch.as_tensor(wall_flux, dtype=torch.complex128)
    bias = torch.as_tensor(steady_flux, dtype=torch.float64, device=flux.device)
    amplitude = flux.abs()
    if not torch.isfinite(amplitude).all() or not torch.isfinite(bias).all():
        raise ValueError("wall fluxes must be finite")
    crossing = amplitude > bias.abs()
    safe_amplitude = torch.where(crossing, amplitude, 1.)
    gap = torch.where(crossing, (amplitude-bias.abs())/safe_amplitude, 0.)
    cosine = torch.where(crossing, bias.abs()/safe_amplitude, 1.)
    half_width = torch.atan2(torch.sqrt(gap*(1+cosine)), cosine)
    return flux, bias, amplitude, crossing, half_width


def harmonic_wall_reversal_times(wall_flux, angular_frequency, *, steady_flux=0.):
    """Return the onset times of absorption and release in one cycle (seconds).

    For inward flux steady_flux + Re(wall_flux * exp(iωt)), return two arrays
    broadcast over wall rows. Times are measured from phase time zero modulo
    the period. Both entries are NaN where the flux never changes sign,
    including a tangency or an identically zero flux.
    """
    flux, bias, _, crossing, half_width = _wall_cycle_geometry(wall_flux, angular_frequency, steady_flux)
    # Locate both roots relative to the extremum of the smaller lobe.
    center = torch.where(bias > 0, math.pi*torch.ones_like(bias), 0.) - flux.angle()
    offset = torch.where(bias > 0, half_width, -half_width)
    absorption = torch.remainder(center+offset, 2*math.pi)/angular_frequency
    release = torch.remainder(center-offset, 2*math.pi)/angular_frequency
    # Roundoff can map a tiny negative phase to the period itself.
    period = 2*math.pi/angular_frequency
    absorption = torch.where(absorption >= period, 0., absorption)
    release = torch.where(release >= period, 0., release)
    return torch.where(crossing, absorption, torch.nan), torch.where(crossing, release, torch.nan)


def directional_wall_cycle_duration(wall_flux, angular_frequency, *, steady_flux=0.):
    """Seconds of inward and outward flow per cycle for each harmonic wall flux.

    A tangency spends the full period in its prevailing direction. Identically
    zero flux spends zero time in either direction. Phase does not affect the
    durations, including for a window spanning the cycle boundary.
    """
    _, bias, amplitude, _, half_width = _wall_cycle_geometry(wall_flux, angular_frequency, steady_flux)
    minor = 2*half_width/angular_frequency
    major = torch.where((amplitude > 0) | (bias != 0), 2*math.pi/angular_frequency-minor, 0.)
    return torch.where(bias >= 0, major, minor), torch.where(bias >= 0, minor, major)


def directional_wall_interval_duration(wall_flux, angular_frequency, start_time, end_time, *, steady_flux=0.):
    """Seconds of strictly inward and outward flow during a forward interval.

    Use the same inward phasor, steady bias and phase convention as
    harmonic_wall_reversal_times. Zero flux contributes to neither duration.
    """
    if not math.isfinite(start_time) or not math.isfinite(end_time) or end_time < start_time:
        raise ValueError("provide finite, nondecreasing integration times")
    flux, bias, amplitude, _, half_width = _wall_cycle_geometry(wall_flux, angular_frequency, steady_flux)
    duration = end_time-start_time
    period = 2*math.pi/angular_frequency
    cycles, remainder = divmod(duration, period)
    extent = angular_frequency*remainder
    # Measure phase from the centre of the shorter directional window.
    phase = flux.angle() - torch.where(bias > 0, torch.full_like(bias, math.pi), 0.)
    start = math.remainder(start_time, period)*angular_frequency + phase
    start = torch.where(start.abs() <= math.pi, start, torch.remainder(start+math.pi, 2*math.pi)-math.pi)
    minor = cycles*(2*half_width/angular_frequency)
    for center in (0., 2*math.pi):
        offset = start-center
        # Intersection length of [offset, offset+extent] and [-half_width, half_width].
        overlap = torch.minimum(half_width-offset, offset+extent+half_width)
        overlap = torch.minimum(overlap, 2*half_width).clamp(0., extent)
        minor = minor + overlap/angular_frequency
    minor = minor.clamp_max(duration)
    major = torch.where((amplitude > 0) | (bias != 0), duration-minor, 0.)
    return torch.where(bias >= 0, major, minor), torch.where(bias >= 0, minor, major)


def directional_wall_cycle_heat(wall_flux, angular_frequency, *, steady_flux=0.):
    """Return absorbed and released heat magnitudes (J/m²) per wall row.

    Supply peak inward wall-flux phasors in W/m², e.g. from
    recover_dirichlet_wall_flux_2d, plus a real steady inward flux broadcastable
    to those rows. Integrate one sinusoidal period before spatial averaging.
    """
    _, bias, amplitude, crossing, angle = _wall_cycle_geometry(wall_flux, angular_frequency, steady_flux)
    square = angle.square()
    # The smaller lobe is proportional to sin(angle)-angle*cos(angle).
    series = angle*square*(1/3+square*(-1/30+square*(1/840-square/45360)))
    lobe = torch.where(angle < .05,series,torch.sin(angle)-angle*torch.cos(angle))
    minor = (2/angular_frequency)*torch.where(crossing,amplitude*lobe,0.)
    period = 2*math.pi/angular_frequency
    return minor+period*bias.clamp_min(0), minor+period*(-bias).clamp_min(0)


def directional_wall_interval_heat(wall_flux, angular_frequency, start_time, end_time, *, steady_flux=0.):
    """Absorbed and released heat magnitudes (J/m²) in a forward time interval.

    Supply peak inward flux phasors and real steady fluxes as for
    directional_wall_cycle_heat. The interval may include multiple cycles.
    """
    if not math.isfinite(start_time) or not math.isfinite(end_time) or end_time < start_time:
        raise ValueError("provide finite, nondecreasing integration times")
    cycle = directional_wall_cycle_heat(wall_flux, angular_frequency, steady_flux=steady_flux)
    flux = torch.as_tensor(wall_flux, dtype=torch.complex128)
    amplitude = flux.abs()
    bias = torch.as_tensor(steady_flux, dtype=amplitude.dtype, device=amplitude.device)
    period = 2*math.pi/angular_frequency
    cycles, remaining = divmod(end_time-start_time, period)
    extent = remaining*angular_frequency
    phase = math.remainder(start_time, period)*angular_frequency + flux.angle()

    def integrate(sign, cycle_heat):
        offset = sign*bias
        start = torch.remainder(phase + (math.pi if sign < 0 else 0.) + math.pi, 2*math.pi) - math.pi
        # Positive flux lies within +/-limit around each peak. Compute the
        # shorter angular distance first to resolve nearly grazing reversals.
        safe_amp = torch.where(amplitude > 0, amplitude, 1.)
        small = 2*torch.asin(torch.sqrt(((amplitude-offset.abs())/(2*safe_amp)).clamp(0, .5)))
        limit = torch.where(offset >= 0, math.pi-small, small)
        heat = cycles*cycle_heat
        for peak in (0., 2*math.pi):
            left = (peak-limit-start).clamp(0, extent)
            right = (peak+limit-start).clamp(0, extent)
            half = (right-left)/2
            mid = (start-peak) + (left+right)/2
            # cos(mid)*sinc(half)-cos(limit), without subtracting near-equal cosines.
            h2 = half.square()
            sinc_loss = torch.where(half < .05, h2*(1/6-h2*(1/120-h2/5040)),
                                    1-torch.sinc(half/math.pi))
            cosine_gap = 2*torch.sin((limit+mid)/2)*torch.sin((limit-mid)/2)
            mean = (offset-amplitude).clamp_min(0) + amplitude*(cosine_gap-mid.cos()*sinc_loss)
            heat = heat + (right-left)/angular_frequency * mean.clamp_min(0)
        return heat

    return integrate(1, cycle[0]), integrate(-1, cycle[1])


def harmonic_fixed_wall_cycle_heat_2d(temperature, dx, conductivity, wall_mean_flux, angular_frequency, *,
                                      contact_resistance_x=None, x_wall_temperatures=None, side=None):
    """Heat absorbed per unit area per row over one zero-mean sinusoidal cycle.

    Supply peak temperature phasors and inward mean flux phasors as for
    recover_dirichlet_wall_flux_2d. The returned positive heat (J/m²) equals
    the magnitude of the heat released per cycle. Average rows after computing
    these local integrals to retain exchange between out-of-phase rows.
    For a steady bias, pass the recovered fluxes to directional_wall_cycle_heat.
    """
    if not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be positive and finite")
    flux = recover_dirichlet_wall_flux_2d(temperature, dx, conductivity, wall_mean_flux,
        contact_resistance_x=contact_resistance_x, x_wall_temperatures=x_wall_temperatures, side=side)
    return (2/angular_frequency)*flux.abs()


def recover_layered_interior_flux(temperature, dx, conductivity, *, axial_flux_gradient=0.,
                                  contact_resistance_x=None):
    """Reconstruct rightward Fourier flux at internal faces from centre temperatures.

    For uniform columns with piecewise constant k and optional contact resistance
    R (m² K/W), integrate the two half-cell temperature drops and the jump R*q.
    Provide dq_x/dx in W/m³ at each centre to include the local curvature.
    In a harmonic transverse mode this is generation minus storage, lateral
    diffusion loss and any bulk heat loss. A constant gradient on each half
    cell gives exact reconstruction from point samples there. The result is
    a local Fourier estimate, distinct from the particle-pair energy tally.
    Accepts a modal vector or a (rows, columns) field. The gradient may be
    scalar, a column profile or the full field.
    """
    field = torch.as_tensor(temperature)
    if field.ndim not in (1, 2) or field.numel() == 0 or field.shape[-1] < 2:
        raise ValueError("provide at least two temperature columns")
    if not (field.is_floating_point() or field.is_complex()) or not torch.isfinite(field).all():
        raise ValueError("temperature must be finite and real or complex")
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("spacing must be positive and finite")
    columns = field.shape[-1]
    raw_k = torch.as_tensor(conductivity, device=field.device)
    if raw_k.is_complex() or raw_k.shape not in ((), (columns,)):
        raise ValueError("conductivity must be real and scalar or one value per column")
    k = torch.as_tensor(conductivity, dtype=field.real.dtype, device=field.device).expand(columns)
    if not torch.all(torch.isfinite(k) & (k > 0)):
        raise ValueError("conductivity must be positive and finite")
    resistance = (dx/2) * (1/k[:-1] + 1/k[1:])
    if contact_resistance_x is not None:
        raw = torch.as_tensor(contact_resistance_x)
        if raw.is_complex() or raw.shape != (columns-1,):
            raise ValueError("provide one real contact resistance per internal face")
        contact = torch.as_tensor(contact_resistance_x, dtype=k.dtype, device=k.device)
        if not torch.all(torch.isfinite(contact) & (contact >= 0)):
            raise ValueError("contact resistance must be finite and nonnegative")
        resistance = resistance + contact
    gradient = torch.as_tensor(axial_flux_gradient, device=field.device)
    dtype = torch.promote_types(field.dtype, gradient.dtype)
    gradient = torch.as_tensor(axial_flux_gradient, dtype=dtype, device=field.device)
    if gradient.shape not in ((), (columns,), field.shape) or not torch.isfinite(gradient).all():
        raise ValueError("axial flux gradient must be finite and match the columns or field")
    # q varies linearly from each centre to the common face. Integrating
    # in opposite directions gives opposite signs for the two corrections.
    return (-field.diff(dim=-1) - (dx**2/8)*(gradient/k).diff(dim=-1)) / resistance


def extrapolate_layered_face_flux(interior_flux, dx, face, inset_cells, *, axial_flux_gradient):
    """Estimate rightward flux at a cut independently from its left and right.

    Faces are numbered 0 through N. Supply the N-1 interior Fourier fluxes
    (W/m²) on a uniform grid and the net axial gradient dq/dx (W/m³) in
    each cell. The midpoint balance is exact for linear cellwise gradients.
    Choose the inset so each anchor lies inside its adjacent material. The
    interval must contain no sheet source or other prescribed flux jump.
    Returns (from_left, from_right), one pair per row for a batched field.
    """
    flux = torch.as_tensor(interior_flux)
    if flux.ndim not in (1, 2) or flux.numel() == 0 or not (
        flux.is_floating_point() or flux.is_complex()
    ) or not torch.isfinite(flux).all():
        raise ValueError("provide a finite real or complex interior face flux field")
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("spacing must be positive and finite")
    face, inset = operator.index(face), operator.index(inset_cells)
    lo, hi = face - inset, face + inset
    if inset < 1 or lo < 1 or hi > flux.shape[-1]:
        raise ValueError("both anchors must be internal faces with positive inset")
    shape = (*flux.shape[:-1], flux.shape[-1]+1)
    raw = torch.as_tensor(axial_flux_gradient, device=flux.device)
    dtype = torch.promote_types(flux.dtype, raw.dtype)
    gradient = torch.as_tensor(axial_flux_gradient, dtype=dtype, device=flux.device)
    if gradient.shape not in ((), (shape[-1],), shape) or not torch.isfinite(gradient).all():
        raise ValueError("axial flux gradient must be finite and match the columns or field")
    gradient = gradient.expand(shape)
    left = flux[..., lo-1] + dx*gradient[..., lo:face].sum(dim=-1)
    right = flux[..., hi-1] - dx*gradient[..., face:hi].sum(dim=-1)
    return left, right


def layered_slab_modal_cut_flux(temperature, dx, face, x_wall_temperatures, *, conductivity,
                                transverse_wavenumber=0., contact_resistance_x=None, h_factor=1.3,
                                split_interface_links=False):
    """Rightward flux phasor (W/m²) at a layered slab cut, including either wall.

    The vector contains particle amplitudes multiplying exp(i*beta*y).
    Wall temperatures are scalar amplitudes or None for insulation. The
    returned complex scalar uses the same transverse phase origin. Oblique
    links are assigned to the row centers bracketing their cut intersection,
    as in stencil_x_face_power_2d with power_by_row=True. Set
    split_interface_links to the same value used by solve_harmonic_layered_slab
    to recover its interface and wall fluxes from a saved modal field.
    """
    field = torch.as_tensor(temperature,dtype=torch.complex128)
    if field.ndim!=1 or not torch.isfinite(field).all():
        raise ValueError("provide a finite vector of particle phasors")
    if not field.numel() or not isinstance(face,int) or not 0<=face<=field.numel():
        raise ValueError("face must index a column boundary from 0 through the field length")
    if not (math.isfinite(dx) and dx>0 and math.isfinite(h_factor) and h_factor>0
            and math.isfinite(transverse_wavenumber)):
        raise ValueError("require positive finite spacing and h_factor and a finite wavenumber")
    if len(x_wall_temperatures)!=2 or any(v is not None and
            (torch.as_tensor(v).ndim or not cmath.isfinite(v)) for v in x_wall_temperatures):
        raise ValueError("wall phasors must be finite scalars or None")
    offsets = lattice_offsets(dx,h_factor*dx)
    _,power = _stencil_exchange(field[None,:],offsets,1.,conductivity,x_wall_temperatures,True,dx,
                                contact_resistance_x,None,modal_flux=True,
                                transverse_phase=transverse_wavenumber*dx, split_interface_links=split_interface_links)
    return power[face]/(dx*lattice_diffusion_factor_2d(h_factor,dx,mode="moment"))


def stencil_x_face_power_2d(
    temperature: torch.Tensor, offsets: tuple[LatticeOffsets, ...], dx: float,
    face: int, x_wall_temperatures: SlabWalls,
    conductivity: torch.Tensor | None = None, *, layered_x: bool = False,
    contact_resistance_x: torch.Tensor | None = None, power_by_row: bool = False,
) -> torch.Tensor:
    """Power crossing x=face*dx from left to right for a slab of unit thickness.

    face is the number of physical columns left of the interior cut. Count
    every crossing pair once, including diagonal and longer-range links.
    Use a symmetric pair stencil; divide by wall area and by the same
    diffusivity calibration used in the evolution to obtain calibrated flux.
    With power_by_row, distribute each link's power linearly to the two row
    centers bracketing its intersection with the cut, periodically in y.
    """
    if not isinstance(face, int) or not 0 < face < temperature.shape[-1]:
        raise ValueError("face must be an interior column boundary")
    if x_wall_temperatures is None or not math.isfinite(dx) or dx <= 0:
        raise ValueError("face power requires fixed walls and positive finite dx")
    _, power = _stencil_exchange(temperature, offsets, 1., conductivity, x_wall_temperatures,
                                 layered_x, dx, contact_resistance_x, None, cut=face, power_by_row=power_by_row)
    return power


def _contact_faces(values, field, dx, layered_x, fixed_walls, width):
    if not layered_x or dx is None or not math.isfinite(dx) or dx <= 0:
        raise ValueError("contact resistance requires layered_x and positive finite dx")
    faces = torch.as_tensor(values, device=field.device, dtype=field.dtype)
    if faces.shape != (field.shape[1] - int(fixed_walls),):
        raise ValueError("provide one contact resistance for each internal or periodic x face")
    if not torch.all(torch.isfinite(faces) & (faces >= 0)):
        raise ValueError("contact resistance must be finite and nonnegative")
    if fixed_walls:
        zero = faces.new_zeros(1)
        if width:
            # Mirror internal interfaces through both Dirichlet walls.
            count = width - 1
            faces = torch.cat((faces[:count].flip((0,)), zero, faces, zero,
                               faces[len(faces)-count:].flip((0,))))
        faces = torch.cat((faces, zero))  # unused outer seam
    return faces / dx


def _stencil_exchange(temperature, offsets, rho, conductivity, x_wall_temperatures,
                      layered_x, dx, contact_resistance_x, specific_heat, collect_power=False, cut=None, power_by_row=False, transverse_phase=None, modal_flux=False, split_interface_links=False):
    complex_field = torch.as_tensor(temperature).is_complex()
    if not complex_field and x_wall_temperatures is not None:
        complex_field = any(torch.as_tensor(wall).is_complex()
                            for wall in x_wall_temperatures if wall is not None)
    t = torch.as_tensor(temperature, dtype=torch.complex128 if complex_field else DTYPE)
    if specific_heat is not None:
        specific_heat = torch.as_tensor(specific_heat, dtype=DTYPE, device=t.device).expand_as(t)
        if not torch.all(torch.isfinite(specific_heat) & (specific_heat > 0)):
            raise ValueError("specific_heat must be finite and positive")
    # Material conductivities live at cell centres. The same pair mean
    # multiplies both directed transfers across each periodic interface.
    if layered_x and (conductivity is None or torch.as_tensor(conductivity).shape != (t.shape[-1],)):
        raise ValueError("layered_x requires one conductivity per x column")
    if conductivity is not None:
        k = torch.as_tensor(conductivity, device=t.device, dtype=DTYPE).expand_as(t)
        if not torch.all(torch.isfinite(k) & (k > 0)):
            raise ValueError("conductivity must be finite and positive")
    width = 0
    nx = t.shape[-1]
    faces = None
    if x_wall_temperatures is not None:
        left, right = x_wall_temperatures
        width = max((abs(n) for (_, n), _ in offsets), default=0)
        if width > nx:
            raise ValueError("x stencil radius exceeds slab width")
    if contact_resistance_x is not None:
        faces = _contact_faces(contact_resistance_x, t.real, dx, layered_x, x_wall_temperatures is not None, width)
    if width:
        left_ghost = t[:, :width].flip((1,))
        right_ghost = t[:, -width:].flip((1,))
        if left is not None:
            left_ghost = 2 * _wall_column(left, t) - left_ghost
        if right is not None:
            right_ghost = 2 * _wall_column(right, t) - right_ghost
        t = torch.cat((left_ghost, t, right_ghost), dim=1)
        if conductivity is not None:
            k = torch.cat((k[:, :width].flip((1,)), k, k[:, -width:].flip((1,))), dim=1)
    acc = torch.zeros_like(t)
    power = t.new_zeros((2, t.shape[0]) if power_by_row else 2) if collect_power else None
    if cut is not None:
        power = t.new_zeros(t.shape[0] if power_by_row else ())
    if modal_flux:
        power = t.new_zeros(nx+1)
    for (m, n), a in offsets:
        split = None
        kernel_weight = a
        if conductivity is not None:
            neighbor = torch.roll(k, shifts=(-m, -n), dims=(0, 1))
            if layered_x and (abs(n) > 1 or (n and faces is not None)):
                # Half a column at either end and full columns in between.
                resistance = 0.5 / k + 0.5 / neighbor
                direction = 1 if n > 0 else -1
                for j in range(direction, n, direction):
                    resistance = resistance + 1 / torch.roll(k, shifts=-j, dims=1)
                if faces is not None:
                    for j in range(min(n, 0), max(n, 0)):
                        resistance = resistance + torch.roll(faces, shifts=-j, dims=0)
                a = a * abs(n) / resistance
            else:
                a = a * _pair_conductivity(k, neighbor)
            if split_interface_links and m and n and transverse_phase:
                # Identify jumps along the whole link, even when its two
                # endpoints belong to the same material around a thin layer.
                split = torch.zeros_like(k[0], dtype=torch.bool)
                # A prescribed wall fixes temperature at the same transverse
                # position. Keep tangential exchange on the physical side.
                for boundary, wall in ((width, left), (width+nx, right)):
                    if wall is not None:
                        split[boundary-max(n, 0):boundary-min(n, 0)] = True
                for j in range(min(n, 0), max(n, 0)):
                    split |= torch.roll(k[0], -j) != torch.roll(k[0], -(j+1))
                    if faces is not None:
                        split |= torch.roll(faces, -j) != 0
        if transverse_phase is None:
            neighbour_t = torch.roll(t, shifts=(-m, -n), dims=(0, 1))
        elif modal_flux:
            # Cut attribution depends on the direction of the oblique link.
            neighbour_t = torch.roll(t,shifts=-n,dims=1)*cmath.exp(1j*m*transverse_phase)
        else:
            # Rows are independent probes. +/-m links share material and x
            # endpoints, so their complex phase factors average to cos(m β dx).
            neighbour_t = torch.roll(t, shifts=-n, dims=1)*math.cos(m*transverse_phase)
        transfer = a * (t - neighbour_t)
        if split is not None:
            normal = a * (t - torch.roll(t, -n, dims=1))
            transfer = torch.where(split, normal, transfer)
            lateral = 2 * math.sin(m*transverse_phase/2)**2
            acc += (kernel_weight * lateral) * k * t * split
        acc = acc + transfer
        if cut is not None and n > 0:
            # transfer is positive INTO the left partner; report outward power.
            stop = width + cut
            crossing = -transfer[:, stop-n:stop]
            if power_by_row:
                for j in range(n):
                    along = m*(n-j-.5)/n
                    lower = math.floor(along)
                    frac = along-lower
                    power += (1-frac)*torch.roll(crossing[:,j],lower)
                    power += frac*torch.roll(crossing[:,j],lower+1)
            else:
                power = power + crossing.sum()
        if modal_flux and n>0:
            # For a given link and intersection fraction, all interior cuts
            # have the same transverse phase shift and consecutive left partners.
            for j in range(n):
                along = m*(n-j-.5)/n
                lower = math.floor(along)
                frac = along-lower
                shift = ((1-frac)*cmath.exp(-1j*lower*transverse_phase)
                         +frac*cmath.exp(-1j*(lower+1)*transverse_phase))
                start = width+1-n+j
                if split is not None:
                    shift = torch.where(split[start:start+nx-1], 1., t.new_tensor(shift))
                power[1:-1] -= transfer[0,start:start+nx-1]*shift
        if (collect_power or modal_flux) and n and x_wall_temperatures[int(n > 0)] is not None:
            # Only physical particles whose neighbour is a ghost cross a wall.
            start = width if n < 0 else width + nx - n
            crossing = transfer[:, start:start + abs(n)]
            if modal_flux:
                # Wall power is assigned to the ghost row. Change inward to
                # positive-x orientation at the east wall.
                wall_phase = cmath.exp(-1j*m*transverse_phase)
                if split is not None:
                    wall_phase = torch.where(split[start:start+abs(n)], 1., t.new_tensor(wall_phase))
                value = (crossing * wall_phase).sum()
                if n<0:
                    power[0] += value
                else:
                    power[-1] -= value
            elif power_by_row:
                power[int(n > 0)] += torch.roll(crossing.sum(dim=1), shifts=m, dims=0)
            else:
                power[int(n > 0)] += crossing.sum()
    if power is not None:
        power = power * (2 * dx**2)
    rate = (2.0 / float(rho)) * acc[:, width:width + nx]
    if specific_heat is not None:
        rate = rate / specific_heat
    return rate, power


def pair_geometry_2d(pos: torch.Tensor, volume: torch.Tensor, h: float,
                     eta_factor: float = 0.01) -> torch.Tensor:
    """``V_j F_ij`` (m^2) for 2-D positions ``(N, 2)`` -- the dense path.

    Identical expression to :func:`tensorsph.ice_thermal.sph_thermal.pair_geometry`
    except that ``r = |r_vec|`` uses both components; ``eta`` regularises the
    ``r -> 0`` pair.
    """
    x = torch.as_tensor(pos, dtype=DTYPE)
    v = torch.as_tensor(volume, dtype=DTYPE)
    dr = x[:, None, :] - x[None, :, :]
    r = torch.linalg.norm(dr, dim=-1)
    eta2 = (float(eta_factor) * float(h)) ** 2
    dwdr = kernel_dwdr(r, float(h))
    f_ij = r * dwdr / (r * r + eta2)
    return f_ij * v[None, :]


def conduction_rhs_2d(temperature, conductivity, volume, rho, pos, h,
                      eta_factor: float = 0.01, fv=None) -> torch.Tensor:
    """``dT/dt`` (K/s) from the dense 2-D pair sum; shapes ``(N,)``.

    ``fv`` optionally carries the frozen pair matrix from
    :func:`pair_geometry_2d`.  The update is ``2/rho * sum_j k_ij V_j F_ij
    (T_i - T_j)`` with the harmonic pair mean ``k_ij = 2 k_i k_j / (k_i + k_j)``
    -- the same algebra as the 1-D ``conduction_rhs``.
    """
    t = torch.as_tensor(temperature, dtype=DTYPE)
    k = torch.as_tensor(conductivity, dtype=DTYPE)
    v = torch.as_tensor(volume, dtype=DTYPE)
    rho_t = torch.as_tensor(rho, dtype=DTYPE)
    k, v, t = torch.broadcast_tensors(k, v, t)
    if fv is None:
        fv = pair_geometry_2d(pos, v, h, eta_factor)
    return _dense_pair_exchange(t, k, fv, rho_t)


# ---------------------------------------------------------------- mode algebra
def symbol_rate(dx: float, h: float, kx: float, ky: float,
                eta_factor: float = 0.01) -> float:
    """Rate ``lambda`` such that the raw operator gives ``dT/dt = -lambda T`` for
    ``T = cos(kx x) cos(ky y)`` on the lattice [1/s].

    Substituting the lattice mode into the pair sum: every offset contributes
    ``-a (1 - cos(m kx dx) cos(n ky dx))`` and the product mode is invariant
    under the shift (the four-dimensional span of ``cos cos``/``cos sin``/
    ``sin cos``/``sin sin`` is closed under translations), so a single scalar
    rate exists.  ``lambda / (kx^2 + ky^2) = alpha_eff / alpha``.
    """
    dx = float(dx)
    total = 0.0
    for (m, n), a in lattice_offsets(dx, h, eta_factor):
        total += a * (1.0 - math.cos(m * kx * dx) * math.cos(n * ky * dx))
    return -2.0 * total


def lattice_diffusion_factor_2d(h_factor: float = 1.3, dx: float = 1.0,
                                eta_factor: float = 0.01,
                                mode: str = "k0") -> float:
    """``alpha_eff / alpha`` of the raw 2-D operator on the unit lattice.

    ``mode="k0"`` takes the long-wavelength limit (``k dx = 1e-3``), the quantity
    a solver should divide by (the 1-D analogue is
    ``lattice_symbol_ratio(k -> 0)``); ``mode="mean"`` averages the symbol over
    the three longest periodic modes of an ``n = 256`` lattice, mirroring
    ``lattice_diffusion_factor``'s 3-mode mean.  Both are returned by the
    benchmark so the choice stays visible. For steady wall fluxes, ``"moment"``
    gives the exact zero-wavenumber limit from the stencil second moment.
    """
    h = float(h_factor) * float(dx)
    if mode == "moment":
        # Along x, 2*(1-cos(k*n*dx))/k² tends to (n*dx)².
        return -math.fsum(a * (n * dx)**2 for (_, n), a in lattice_offsets(dx, h, eta_factor))
    if mode == "k0":
        k = 1.0e-3 / float(dx)
        return symbol_rate(dx, h, k, k, eta_factor) / (2.0 * k * k)
    if mode == "mean":
        n = 256
        kmodes = [2.0 * math.pi * m / (n * float(dx)) for m in (1, 2, 3)]
        ratios = [symbol_rate(dx, h, k, 0.0, eta_factor) / (k * k) for k in kmodes]
        return sum(ratios) / len(ratios)
    raise ValueError(f"unknown mode={mode!r}")


def mode_field(n: int, dx: float, kx: float, ky: float) -> torch.Tensor:
    """``cos(kx x) cos(ky y)`` at the cell centres of an ``n x n`` periodic grid."""
    x = (torch.arange(n, dtype=DTYPE) + 0.5) * float(dx)
    xg, yg = torch.meshgrid(x, x, indexing="ij")
    return torch.cos(float(kx) * xg) * torch.cos(float(ky) * yg)


def mode_exact(n: int, dx: float, kx: float, ky: float, t: float,
               alpha: float = 1.0) -> torch.Tensor:
    """Exact solution at time ``t``: the same mode times ``exp(-alpha (kx^2+ky^2) t)``."""
    decay = math.exp(-float(alpha) * (float(kx) ** 2 + float(ky) ** 2) * float(t))
    return decay * mode_field(n, dx, kx, ky)


def mode_amplitude(temperature: torch.Tensor, n: int, dx: float,
                   kx: float, ky: float) -> float:
    """Rayleigh quotient ``(T, mode) / (mode, mode)`` -- the amplitude of the mode."""
    m = mode_field(n, dx, kx, ky)
    return float((torch.as_tensor(temperature, dtype=DTYPE) * m).sum()
                 / (m * m).sum())


# ------------------------------------------------------------------ equivalence
def stencil_vs_dense_2d(n: int = 16, h_factor: float = 1.3,
                        eta_factor: float = 0.01, seed: int = 0) -> float:
    """Max ``|fast stencil - dense pair sum|`` on a periodic ``n x n`` lattice.

    The dense matrix is built from the same :func:`pair_geometry_2d` the 1-D
    solver's dense path uses, so a round-off-sized number licenses the
    ``torch.roll`` time loop as the *same* operator rather than a re-derivation.
    The dense lattice is the field tiled 3x3 (so it has no edge for the stencil
    to wrap over); only the middle tile is compared -- same construction as
    :func:`tensorsph.heat_conduction.stencil_vs_dense`.
    """
    dx = 1.0
    h = float(h_factor) * dx
    g = torch.Generator().manual_seed(seed)
    temp = torch.rand(n, n, dtype=DTYPE, generator=g) + 0.5
    big = temp.repeat(3, 3)
    m = 3 * n
    lin = (torch.arange(m, dtype=DTYPE) + 0.5) * dx
    yg, xg = torch.meshgrid(lin, lin, indexing="ij")
    pos = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=1)
    vol = torch.full((m * m,), dx * dx, dtype=DTYPE)
    dense = conduction_rhs_2d(big.reshape(-1), torch.ones(m * m, dtype=DTYPE),
                              vol, 1.0, pos, h, eta_factor).reshape(m, m)
    fast = stencil_rhs_2d(big, lattice_offsets(dx, h, eta_factor))
    middle = slice(n, 2 * n)
    return float((dense[middle, middle] - fast[middle, middle]).abs().max())


def degenerate_1d_agreement(n: int = 24, h_factor: float = 1.3,
                            eta_factor: float = 0.01, seed: int = 3) -> float:
    """Max ``|2-D code on an n x 1 grid - shipped 1-D conduction_rhs|``.

    Setting ``ky = 0`` (one row) must reproduce :func:`tensorsph.ice_thermal.
    sph_thermal.conduction_rhs` exactly -- if it did not, the 2-D module would be
    a different operator and could not claim to extend the ship.
    """
    from .ice_thermal.sph_thermal import conduction_rhs

    dx = 1.0
    h = h_factor * dx
    g = torch.Generator().manual_seed(seed)
    temp = torch.rand(n, 1, dtype=DTYPE, generator=g) + 0.5
    ys, xs = torch.meshgrid(torch.zeros(1, dtype=DTYPE),
                            (torch.arange(n, dtype=DTYPE) + 0.5) * dx,
                            indexing="ij")
    pos2 = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    vol = torch.full((n,), dx * dx, dtype=DTYPE)
    two_d = conduction_rhs_2d(temp.reshape(-1), torch.ones(n, dtype=DTYPE),
                              vol, 1.0, pos2, h, eta_factor)
    # 1-D ship: volume dx (a layer), same kernel, same 2/rho and harmonic mean
    pos1 = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    one_d = conduction_rhs(temp.reshape(-1), torch.ones(n, dtype=DTYPE),
                           torch.full((n,), dx, dtype=DTYPE), 1.0, pos1, h,
                           eta_factor)
    # the 2-D volume is dx^2, the 1-D volume is dx: same operator up to that
    return float((two_d / dx - one_d).abs().max())


# --------------------------------------------------------------------- solver
@dataclass
class ModeDecayConfig:
    """Decay of one Fourier mode on a periodic ``L x L`` square.

    ``fourier`` is the *amplitude* target: ``t_end`` is chosen so that the exact
    amplitude is ``amplitude_end`` times the initial one, i.e.
    ``t_end = -ln(amplitude_end) / (alpha (kx^2 + ky^2))``.  Lengths in m, time
    in s, ``alpha`` in m^2/s; ``rho = cp = 1`` so ``k = alpha``.
    """

    n: int = 64
    length: float = 1.0
    alpha: float = 1.0
    h_factor: float = 1.3
    eta_factor: float = 0.01
    periods: tuple[int, int] = (1, 1)     # mode = (2 pi px / L, 2 pi py / L)
    amplitude_end: float = 0.1
    courante: float = 0.2
    calibration: str = "k0_2d"            # "none" | "lattice_1d" | "k0_2d" | "mean_2d" | "mode_2d"
    integrator: str = "rk4"               # "rk4" (time error negligible) | "euler" (shipped style)
    sample_every: int = 0


@dataclass
class ModeDecayResult:
    n: int = 0
    dx: float = 0.0
    dt: float = 0.0
    steps: int = 0
    time: float = 0.0
    kx: float = 0.0
    ky: float = 0.0
    calibration: str = "none"
    integrator: str = "rk4"
    factor: float = 1.0
    raw_symbol_rate: float = 0.0     # lambda of the raw operator at the test mode
    amplitude_start: float = 0.0
    amplitude_end: float = 0.0
    amplitude_exact: float = 0.0
    max_normalised_error: float = 0.0    # max|dT| / max|T_exact| at t_end
    max_error_over_initial: float = 0.0  # max|dT| / amplitude at t = 0
    decay_rate_error: float = 0.0        # measured k_eff/ (alpha(kx^2+ky^2)) - 1
    sampled_time: list = field(default_factory=list)
    sampled_ratio: list = field(default_factory=list)   # amplitude(t)/amplitude(0)
    exact_ratio: list = field(default_factory=list)
    field_values: list = field(default_factory=list)
    analytic: list = field(default_factory=list)

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        return out


def run_mode_decay(cfg: ModeDecayConfig | None = None) -> ModeDecayResult:
    """Explicit-Euler time loop of the (calibrated) 2-D operator on one mode."""
    from .ice_thermal.sph_thermal import lattice_diffusion_factor

    cfg = cfg or ModeDecayConfig()
    n, length, alpha = int(cfg.n), float(cfg.length), float(cfg.alpha)
    dx = length / n
    h = cfg.h_factor * dx
    kx = 2.0 * math.pi * int(cfg.periods[0]) / length
    ky = 2.0 * math.pi * int(cfg.periods[1]) / length

    if cfg.calibration == "none":
        factor = 1.0
    elif cfg.calibration == "lattice_1d":
        factor = lattice_diffusion_factor(h_factor=cfg.h_factor)[0]
    elif cfg.calibration == "k0_2d":
        factor = lattice_diffusion_factor_2d(cfg.h_factor, dx, cfg.eta_factor, "k0")
    elif cfg.calibration == "mean_2d":
        factor = lattice_diffusion_factor_2d(cfg.h_factor, dx, cfg.eta_factor, "mean")
    elif cfg.calibration == "mode_2d":
        factor = symbol_rate(dx, h, kx, ky, cfg.eta_factor) / (kx * kx + ky * ky)
    else:
        raise ValueError(f"unknown calibration={cfg.calibration!r}")
    alpha_corr = 1.0 / factor if cfg.calibration != "none" else 1.0

    offsets = lattice_offsets(dx, h, cfg.eta_factor)
    temperature = mode_field(n, dx, kx, ky)

    t_end = -math.log(float(cfg.amplitude_end)) / (alpha * (kx * kx + ky * ky))
    dt = cfg.courante * dx * dx / alpha
    steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / steps

    decay0 = alpha * (kx * kx + ky * ky)
    res = ModeDecayResult(
        n=n, dx=dx, dt=dt, steps=steps, time=t_end, kx=kx, ky=ky,
        calibration=cfg.calibration, integrator=cfg.integrator, factor=factor,
        raw_symbol_rate=symbol_rate(dx, h, kx, ky, cfg.eta_factor),
        amplitude_start=mode_amplitude(temperature, n, dx, kx, ky),
    )
    times, ratios, exacts = [], [], []
    for step in range(steps + 1):
        t = step * dt
        if cfg.sample_every and step % cfg.sample_every == 0:
            a = mode_amplitude(temperature, n, dx, kx, ky)
            times.append(t)
            ratios.append(a / res.amplitude_start)
            exacts.append(math.exp(-decay0 * t))
        if step == steps:
            break
        if cfg.integrator == "euler":
            temperature = temperature + dt * alpha_corr * stencil_rhs_2d(
                temperature, offsets)
        elif cfg.integrator == "rk4":
            # classical RK4 on the *same* spatial operator; the time error becomes
            # negligible (O(dt^4)) so the reading is the spatial symbol error.  With
            # forward Euler the first-order time error at dt = 0.2 dx^2/alpha is
            # 1.8 % at n = 32 -- the same size as the spatial error and of the
            # opposite sign, i.e. it can masquerade as accuracy.
            def rhs(state):
                return alpha_corr * stencil_rhs_2d(state, offsets)
            k1 = rhs(temperature)
            k2 = rhs(temperature + 0.5 * dt * k1)
            k3 = rhs(temperature + 0.5 * dt * k2)
            k4 = rhs(temperature + dt * k3)
            temperature = temperature + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"unknown integrator={cfg.integrator!r}")
        if not torch.isfinite(temperature).all():
            raise FloatingPointError(f"2-D conduction diverged at step {step}")

    exact = mode_exact(n, dx, kx, ky, t_end, alpha)
    diff = (temperature - exact).abs()
    res.amplitude_end = mode_amplitude(temperature, n, dx, kx, ky)
    res.amplitude_exact = float(exact.abs().max())
    res.max_normalised_error = float(diff.max() / exact.abs().max())
    res.max_error_over_initial = float(diff.max() / res.amplitude_start)
    res.decay_rate_error = res.amplitude_end / (res.amplitude_start
                                                * math.exp(-decay0 * t_end)) - 1.0
    res.sampled_time, res.sampled_ratio, res.exact_ratio = times, ratios, exacts
    res.field_values = [float(v) for v in temperature.reshape(-1)]
    res.analytic = [float(v) for v in exact.reshape(-1)]
    return res


# --------------------------------------------------- independent reference (FD)
def fd_mode_decay(n: int = 64, length: float = 1.0, alpha: float = 1.0,
                  periods: tuple[int, int] = (1, 1), amplitude_end: float = 0.1,
                  r_target: float = 0.4) -> dict:
    """Crank-Nicolson (periodic, 5-point) reference, sharing no code with the SPH path.

    The initial field is a lattice eigenfunction, so this solver should agree
    with the closed form to its own truncation level -- that is what makes the
    closed form a *checked* target rather than an asserted one.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = int(n)
    dx = float(length) / n
    kx = 2.0 * math.pi * int(periods[0]) / float(length)
    ky = 2.0 * math.pi * int(periods[1]) / float(length)
    t_end = -math.log(float(amplitude_end)) / (alpha * (kx * kx + ky * ky))
    dt = float(r_target) * 2.0 * dx * dx / alpha
    steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / steps
    r = alpha * dt / (2.0 * dx * dx)

    idx = np.arange(n * n).reshape(n, n).reshape(-1)
    rows = [np.arange(n * n)]
    cols = [idx]
    vals = [np.full(n * n, -4.0)]
    for axis in (0, 1):
        for shift in (1, -1):
            rows.append(np.arange(n * n))
            cols.append(np.roll(np.arange(n * n).reshape(n, n), shift, axis=axis).reshape(-1))
            vals.append(np.ones(n * n))
    lap = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n * n, n * n))
    identity = sp.identity(n * n, format="csr")
    a_mat = (identity - r * lap).tocsc()
    b_mat = (identity + r * lap).tocsr()
    solve = spla.factorized(a_mat)

    x = (np.arange(n) + 0.5) * dx
    xg, yg = np.meshgrid(x, x, indexing="ij")
    temp = (np.cos(kx * xg) * np.cos(ky * yg)).reshape(-1)
    for _ in range(steps):
        temp = solve(b_mat @ temp)
    exact = (math.exp(-alpha * (kx * kx + ky * ky) * t_end)
             * np.cos(kx * xg) * np.cos(ky * yg)).reshape(-1)
    return {
        "n": n, "dx": dx, "dt": dt, "steps": steps, "t_end": t_end, "r": r,
        "max_normalised_error_vs_closed_form": float(
            np.abs(temp - exact).max() / np.abs(exact).max()),
    }
