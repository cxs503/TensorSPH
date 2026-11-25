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
from dataclasses import dataclass, field

import torch

from .ice_thermal.sph_thermal import kernel_dwdr

__all__ = [
    "DTYPE",
    "LatticeOffsets",
    "lattice_offsets",
    "stencil_rhs_2d",
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


def solve_steady_slab_2d(heat_source: torch.Tensor, dx: float, x_wall_temperatures: SlabWalls,
                         *, conductivity: float = 1., h_factor: float = 1.3,
                         mean_temperature: float | None = None,
                         x_wall_inward_flux=(0., 0.), return_x_flux: bool = False
                         ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Steady temperature on a uniform lattice, periodic y and reflected x walls.

    Supply a (ny, nx) volumetric heat source in W/m³, a constant conductivity
    in W/(m K), and two wall conditions. None selects a flux wall. Two
    flux walls require a balanced heat input and a prescribed mean_temperature.
    x_wall_inward_flux gives scalar or per-row heat fluxes into the domain in W/m² on
    sides with temperature None. The solution uses an analytical polynomial
    lift for prescribed flux and the calibrated SPH operator for the remainder.
    With return_x_flux, also return the cross-section mean heat flux (W/m²)
    on all nx+1 faces, positive towards increasing x, including the walls.
    """
    return _solve_slab_modes(heat_source, dx, x_wall_temperatures, conductivity, h_factor,
                             mean_temperature, 0., x_wall_inward_flux, return_x_flux)


def solve_harmonic_slab_2d(heat_source: torch.Tensor, dx: float, x_wall_temperatures: SlabWalls,
                           angular_frequency: float, *, conductivity: float = 1.,
                           volumetric_heat_capacity: float = 1., h_factor: float = 1.3,
                           x_wall_inward_flux=(0., 0.), return_x_flux: bool = False
                         ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Temperature phasors under exp(+iωt), for constant isotropic k and capacity.

    Use complex volumetric heating (W/m³) and scalar or row-wise wall phasors
    (K). On each wall with temperature None, x_wall_inward_flux specifies a
    scalar or per-row inward flux phasor (W/m²), default zero. Supply ω in
    rad/s and capacity in J/(m³ K). With return_x_flux, also return nx+1
    section-averaged complex heat fluxes (W/m²), positive towards increasing x.
    """
    storage = angular_frequency*volumetric_heat_capacity
    if not all(math.isfinite(v) and v > 0 for v in (angular_frequency, volumetric_heat_capacity, storage)):
        raise ValueError("frequency and volumetric heat capacity must be positive and finite")
    return _solve_slab_modes(heat_source, dx, x_wall_temperatures, conductivity, h_factor,
                             None, 1j*storage, x_wall_inward_flux, return_x_flux)


def solve_harmonic_layered_slab(heat_source, dx, x_wall_temperatures, angular_frequency,
                                *, conductivity, volumetric_heat_capacity,
                                contact_resistance_x=None, h_factor=1.3, transverse_wavenumber=0.,
                                backend="torch"):
    """Return particle temperature phasors for one transverse Fourier mode of a layered slab.

    Use one source (W/m³) and conductivity (W/(m K)) per x column, a scalar
    or per-column heat capacity (J/(m³ K)), and optional interface resistances
    (m² K/W). Each end has a scalar temperature phasor or None for insulation.
    The dense complex solve uses the calibrated 2-D particle-pair stencil with
    row dependence exp(i β y), with β=transverse_wavenumber in rad/m.
    The inputs and output are modal amplitudes. Supply a periodic wavenumber
    compatible with the transverse period. Alternatively supply a (ny,nx)
    source and scalar or (ny,) end temperatures sampled on periodic rows.
    Then the output has the same grid shape and all Fourier modes are solved.
    Frequency is in rad/s under exp(+iωt). The default backend is dense torch.
    backend="banded" uses SciPy on CPU and stores only the stencil diagonals.
    """
    source = torch.as_tensor(heat_source, dtype=torch.complex128)
    if backend not in ("torch", "banded"):
        raise ValueError("backend must be torch or banded")
    if backend == "banded" and source.device.type != "cpu":
        raise ValueError("banded solves require CPU inputs")
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
        waves = (2*math.pi)*torch.fft.fftfreq(ny, d=dx, dtype=DTYPE)
        rows = [solve_harmonic_layered_slab(load, dx, (west, east), angular_frequency,
                        conductivity=conductivity, volumetric_heat_capacity=volumetric_heat_capacity,
                        contact_resistance_x=contact_resistance_x, h_factor=h_factor,
                        transverse_wavenumber=float(wave), backend=backend)
                for load, west, east, wave in zip(modes, wall_modes(left), wall_modes(right), waves)]
        return torch.fft.ifft(torch.stack(rows), dim=0)
    if source.ndim != 1 or source.numel() == 0 or not torch.isfinite(source).all():
        raise ValueError("provide a finite nonempty vector of column heat sources")
    if not all(math.isfinite(v) and v > 0 for v in (angular_frequency, dx, h_factor)):
        raise ValueError("frequency, dx and h_factor must be positive and finite")
    capacity = torch.as_tensor(volumetric_heat_capacity, dtype=DTYPE, device=source.device)
    if capacity.shape not in ((), source.shape) or not torch.all(torch.isfinite(capacity) & (capacity > 0)):
        raise ValueError("provide a positive finite scalar or column heat capacities")
    left, right = x_wall_temperatures
    for wall in (left, right):
        if wall is not None and (torch.as_tensor(wall).ndim or not cmath.isfinite(wall)):
            raise ValueError("wall phasors must be finite scalars")
    offsets = lattice_offsets(dx, h_factor*dx)
    factor = lattice_diffusion_factor_2d(h_factor, dx, mode="moment")
    if factor <= 0:
        raise ValueError("kernel support must include a conducting neighbour")
    if not math.isfinite(transverse_wavenumber):
        raise ValueError("transverse wavenumber must be finite")
    phase_step = transverse_wavenumber*dx
    homogeneous = tuple(None if wall is None else 0. for wall in (left, right))
    def response(probes, walls):
        rate, _ = _stencil_exchange(probes, offsets, 1., conductivity, walls, True,
                                    dx, contact_resistance_x, None, transverse_phase=phase_step)
        return rate/factor

    load = response(torch.zeros_like(source)[None, :], (left, right))[0]
    storage = 1j*angular_frequency*capacity.expand_as(source)
    size = source.numel()
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
        bands[radius] += storage
        if radius == 0:
            return (source+load)/bands[0]
        result = solve_banded((radius,radius), bands.numpy(), (source+load).numpy())
        return torch.from_numpy(result)

    basis = torch.eye(size, dtype=DTYPE, device=source.device)
    operator = response(basis, homogeneous).T
    matrix = torch.diag(storage)-operator
    return torch.linalg.solve(matrix, source+load)


def solve_harmonic_layered_films(heat_source, dx, ambient_temperatures, angular_frequency,
                                 *, film_coefficients, conductivity, volumetric_heat_capacity,
                                 contact_resistance_x=None, h_factor=1.3, backend="torch",
                                 wall_areal_heat_capacity=(0.,0.)):
    """Solve a transversely uniform layered slab with two convective walls.

    Return (particle temperatures in K, two wall temperatures in K, two inward
    heat fluxes in W/m²), all phasors under exp(+iωt). Film coefficients are
    real and positive in W/(m² K). Ambient phasors and coefficients are ordered
    west, east. Source, material properties and contacts follow
    solve_harmonic_layered_slab's column-vector convention. Each wall can store
    heat with areal capacity in J/(m² K). Returned heat flux enters the slab.
    """
    source = torch.as_tensor(heat_source, dtype=torch.complex128)
    if source.ndim != 1:
        raise ValueError("film solve requires one heat source per x column")
    ambient = torch.as_tensor(ambient_temperatures, dtype=source.dtype, device=source.device)
    films = torch.as_tensor(film_coefficients, device=source.device, dtype=DTYPE)
    if ambient.shape != (2,) or not torch.isfinite(ambient).all():
        raise ValueError("provide two finite ambient temperature phasors")
    if films.shape != (2,) or not torch.all(torch.isfinite(films) & (films > 0)):
        raise ValueError("provide two positive finite film coefficients")

    wall_capacity = torch.as_tensor(wall_areal_heat_capacity, device=source.device, dtype=DTYPE)
    if wall_capacity.shape != (2,) or not torch.all(torch.isfinite(wall_capacity) & (wall_capacity >= 0)):
        raise ValueError("provide two finite nonnegative wall areal heat capacities")
    opts = dict(conductivity=conductivity, contact_resistance_x=contact_resistance_x)
    def response(load, walls):
        return solve_harmonic_layered_slab(load, dx, walls, angular_frequency,
                      volumetric_heat_capacity=volumetric_heat_capacity, h_factor=h_factor,
                      backend=backend, **opts)

    base = response(source, (0., 0.))
    offsets = lattice_offsets(dx, h_factor*dx)
    factor = lattice_diffusion_factor_2d(h_factor, dx, mode="moment")

    def inward(field, walls):
        _, powers = stencil_rhs_with_wall_power_2d(field[None,:], offsets, dx, walls,
                                                   layered_x=True, **opts)
        return powers/(dx*factor)

    # Two unit-temperature solutions give the complete wall admittance,
    # including heat transmitted from one wall to the opposite wall.
    zero = torch.zeros_like(source)
    units = ((1.,0.), (0.,1.))
    fields = [response(zero, walls) for walls in units]
    admittance = torch.stack([inward(u, w) for u,w in zip(fields, units)], dim=1)
    walls = torch.linalg.solve(admittance + torch.diag(films+1j*angular_frequency*wall_capacity),
                               films*ambient-inward(base,(0.,0.)))
    field = base + walls @ torch.stack(fields)
    return field, walls, inward(field, walls)


def step_transient_slab_2d(temperature, dt, dx, x_wall_temperatures, *, heat_source=0.,
                           conductivity=1., volumetric_heat_capacity=1., theta=.5,
                           h_factor=1.3, x_wall_inward_flux=(0.,0.), return_x_flux=False):
    """Advance a homogeneous reflected slab one implicit theta step of duration dt (s).

    Supply real particle temperatures (ny,nx), periodic in y. Supply the
    theta-weighted start and end values of volumetric heating (W/m³), wall
    temperatures (K, scalar or per row), and scalar or per-row inward wall fluxes (W/m²)
    on sides whose wall temperature is None. theta=.5 is Crank–Nicolson and
    theta=1 is backward Euler. Optional section fluxes point right and are
    time-step averages, so dt times the flux difference gives heat per area.
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
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("capacity divided by theta*dt must be finite and positive")
    result = _solve_slab_modes(shift*field+source, dx, x_wall_temperatures,
                               conductivity, h_factor, None, shift, x_wall_inward_flux, return_x_flux)
    stage, flux = result if return_x_flux else (result, None)
    updated = field+(stage-field)/theta
    return (updated,flux) if return_x_flux else updated


def solve_transient_slab_2d(temperature, times, dx, x_wall_temperatures, *,
                            wall_jump_times=(), rannacher=False, heat_source=0.,
                            conductivity=1., volumetric_heat_capacity=1., theta=.5,
                            h_factor=1.3, x_wall_inward_flux=(0.,0.), return_x_heat=False):
    """Advance a reflected slab over requested times with prescribed wall histories.

    Each wall may be a fixed scalar, row profile, callable of time returning
    either, or None for prescribed flux. Volumetric heating (scalar or field)
    and each inward wall flux (scalar or row profile) may also depend on time.
    List discontinuities of any load in wall_jump_times and return the new
    value at the event time.
    Rannacher damping applies two backward Euler half steps at startup and
    immediately after each jump. The default integrator is Crank–Nicolson.
    Return particle temperatures (nt,ny,nx). With return_x_heat also return
    the interval-integrated rightward heat on each section (nt-1,nx+1), J/m².
    """
    grid = [float(t) for t in times]
    jumps = [float(t) for t in wall_jump_times]
    if len(grid)<2 or not all(math.isfinite(t) for t in grid) or any(a>=b for a,b in zip(grid,grid[1:])):
        raise ValueError("times must be finite and strictly increasing")
    if len(set(jumps))!=len(jumps) or not all(grid[0]<t<grid[-1] for t in jumps):
        raise ValueError("wall_jump_times must be distinct interior times")
    if rannacher and theta!=.5:
        raise ValueError("Rannacher damping requires theta=.5")
    walls = tuple(x_wall_temperatures)
    if len(walls)!=2:
        raise ValueError("provide west and east wall temperatures")
    fluxes = tuple(x_wall_inward_flux)
    if len(fluxes)!=2:
        raise ValueError("provide west and east inward fluxes")
    def stage_load(load,start,end,weight):
        if not callable(load):
            return load
        stop_value = load(end)
        if weight==1.:
            return stop_value
        start_value = load(start)
        if any(torch.is_complex(torch.as_tensor(v)) for v in (start_value,stop_value)):
            raise ValueError("transient loads must be real")
        return ((1-weight)*torch.as_tensor(start_value,device=field.device,dtype=field.dtype)
                +weight*torch.as_tensor(stop_value,device=field.device,dtype=field.dtype))
    field = torch.as_tensor(temperature)
    if field.ndim!=2 or min(field.shape)==0 or field.is_complex():
        raise ValueError("provide a real nonempty (ny,nx) temperature field")
    field = field.to(DTYPE)
    history = [field]
    heat = field.new_zeros((len(grid)-1,field.shape[1]+1)) if return_x_heat else None
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
            stage = tuple(stage_load(wall,start,load_end,weight) for wall in walls)
            stage_flux = tuple(stage_load(q,start,load_end,weight) for q in fluxes)
            stage_source = stage_load(heat_source,start,load_end,weight)
            result = step_transient_slab_2d(field,end-start,dx,stage,heat_source=stage_source,
                         conductivity=conductivity,volumetric_heat_capacity=volumetric_heat_capacity,
                         theta=weight,h_factor=h_factor,x_wall_inward_flux=stage_flux,return_x_flux=return_x_heat)
            if return_x_heat:
                field,flux = result
                heat[interval] += (end-start)*flux
            else:
                field = result
            start = end
        if stop==grid[interval+1]:
            history.append(field)
            interval += 1
    fields = torch.stack(history)
    return (fields,heat) if return_x_heat else fields


def _solve_slab_modes(heat_source, dx, x_wall_temperatures, conductivity, h_factor,
                      mean_temperature, shift, x_wall_inward_flux, return_x_flux):
    source = torch.as_tensor(heat_source, dtype=torch.complex128 if isinstance(shift, complex) else DTYPE)
    if source.ndim != 2 or min(source.shape) == 0 or not torch.isfinite(source).all():
        raise ValueError("heat_source must be a finite nonempty (ny, nx) field")
    if not all(math.isfinite(v) and v > 0 for v in (dx, conductivity, h_factor)):
        raise ValueError("dx, conductivity and h_factor must be positive and finite")
    left, right = x_wall_temperatures
    closed = left is None and right is None
    if not source.is_complex() and any(torch.is_complex(torch.as_tensor(q)) for q in x_wall_inward_flux):
        raise ValueError("real slab solves require real wall fluxes")
    west, east = (_wall_column(q, source) for q in x_wall_inward_flux)
    if not all(torch.isfinite(q).all() for q in (west, east)):
        raise ValueError("wall fluxes must be finite")
    west_loaded, east_loaded = bool(west.any()), bool(east.any())
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
        raise ValueError("mean_temperature applies only to two flux walls in a steady solve")
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
    shifted = tuple(None if wall is None else torch.as_tensor(wall, dtype=source.dtype,
                    device=source.device)-reference for wall in (left, right))
    wall_rate = stencil_rhs_2d(torch.zeros_like(source), offsets, x_wall_temperatures=shifted)
    rhs = source + (conductivity/factor)*wall_rate
    if shift:
        rhs = rhs - shift*reference
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
    if source.is_complex():
        modes = torch.fft.fft2(rhs)/(decay + shift)
        correction = reference + torch.fft.ifft2(modes)[:, :source.shape[1]]
    else:
        # The homogeneous correction has zero mean by odd symmetry or by prescription.
        denominator = decay + shift
        if not shift:
            denominator[0, 0] = 1.
        modes = torch.fft.rfft2(rhs)/denominator
        if not shift:
            modes[0, 0] = 0.
        correction = reference + torch.fft.irfft2(modes, s=rhs.shape)[:, :source.shape[1]]
    temperature = lift + correction
    if not return_x_flux:
        return temperature

    # For constant k and periodic y, summing all crossing links over rows
    # commutes with the tangential shift in every diagonal pair.
    profile = correction.mean(dim=0)
    radius = max(abs(n) for (_, n), _ in offsets)
    low, high = profile[:radius].flip((0,)), profile[-radius:].flip((0,))
    if left is not None:
        low = 2*_wall_column(left, correction).mean()-low
    if right is not None:
        high = 2*_wall_column(right, correction).mean()-high
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
    return temperature, flux


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
                      layered_x, dx, contact_resistance_x, specific_heat, collect_power=False, cut=None, power_by_row=False, transverse_phase=None):
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
    for (m, n), a in offsets:
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
                a = a * (2 * k * neighbor / (k + neighbor))
        if transverse_phase is None:
            neighbour_t = torch.roll(t, shifts=(-m, -n), dims=(0, 1))
        else:
            # Rows are independent probes. +/-m links share material and x
            # endpoints, so their complex phase factors average to cos(m β dx).
            neighbour_t = torch.roll(t, shifts=-n, dims=1)*math.cos(m*transverse_phase)
        transfer = a * (t - neighbour_t)
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
                    power += (1-frac)*torch.roll(crossing[:,j], lower)
                    power += frac*torch.roll(crossing[:,j], lower+1)
            else:
                power = power + crossing.sum()
        if collect_power and n and x_wall_temperatures[int(n > 0)] is not None:
            # Only physical particles whose neighbour is a ghost cross a wall.
            start = width if n < 0 else width + nx - n
            crossing = transfer[:, start:start + abs(n)]
            if power_by_row:
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
    k_ij = 2.0 * k[:, None] * k[None, :] / (k[:, None] + k[None, :])
    kernel = k_ij * fv
    return 2.0 / rho_t * (t * kernel.sum(dim=1) - kernel @ t)


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
