"""3-D conduction and latent heat in a sessile fresh-water drop after nucleation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..neighbor_search import cell_linked_pairs
from ..threed.kernel3 import cubic_spline_gradient
from . import properties as p
from .enthalpy import FRESH

# Borosilicate proxy for the slide, in SI units (SCHOTT BOROFLOAT 33).
GLASS_K, GLASS_RHO, GLASS_CP = 1.2, 2230., 830.


def _kernel_weight(separation, dx):
    grad = cubic_spline_gradient(separation, dx)
    return -2*dx**3*(separation*grad).sum(1)/separation.square().sum(1)


def _laplacian(points, dx):
    """Conservative 3-D Brookshaw graph. Its bulk x² moment fixes the scale."""
    i, j = cell_linked_pairs(points, 2*dx).T
    coeff = _kernel_weight(points[i]-points[j], dx)
    offsets = torch.cartesian_prod(*[points.new_tensor([-1., 0., 1.])]*3)*dx
    offsets = offsets[offsets.square().sum(1) > 0]
    moment = (_kernel_weight(offsets, dx)*offsets[:, 0].square()).sum()/2
    coeff = coeff/moment
    row = points.new_zeros(len(points)).index_add_(0, i, coeff).index_add_(0, j, coeff)
    diag = torch.arange(len(points), device=points.device)
    indices = torch.stack((torch.cat((i, j, diag)), torch.cat((j, i, diag))))
    matrix = torch.sparse_coo_tensor(
        indices, torch.cat((coeff, coeff, -row)), (len(points), len(points)),
        check_invariants=True).coalesce().to_sparse_csr()
    return matrix, row


def _heat_potential(celsius):
    """Integral from 0 °C to T of conductivity linear in solid mass fraction."""
    under = (-celsius).clamp(0, FRESH.mushy_dt)
    extra = .5*under.square()/FRESH.mushy_dt + (-celsius-FRESH.mushy_dt).clamp_min(0)
    return p.K_FRESH*celsius - (p.K_ICE-p.K_FRESH)*extra


@dataclass
class DropFreezingResult:
    positions: torch.Tensor
    specific_enthalpy: torch.Tensor
    solid_fraction: torch.Tensor
    section_mask: torch.Tensor
    history: list[dict]
    sections: list[tuple[float, torch.Tensor]]
    metrics: dict


def simulate(*, cells=12, radius=.002, height=.001, plate_c=-20., initial_water_c=0.,
             glass_depth=.001, end_s=30., safety=.8, shape="cap", contact_resistance=0.):
    """Cool a spherical cap or verification slab on an isothermal or finite plate.

    Lengths in m, times in s. Zero glass depth selects an isothermal contact.
    With finite depth the backside is thermostatted and exterior faces insulated.
    contact_resistance is the areal drop/plate thermal resistance (m² K/W).
    Initial liquid below 0 °C is mapped to equal enthalpy at prescribed nucleation.
    The fixed particle mass and shape omit expansion, flow and nucleation delay.
    """
    if not isinstance(cells, int) or cells < 4 or not all(map(math.isfinite, (
            radius, height, plate_c, initial_water_c, glass_depth,
            end_s, safety, contact_resistance))):
        raise ValueError("expected finite parameters and at least four cells per base radius")
    if (min(radius, height, end_s) <= 0 or height > radius
            or glass_depth < 0 or contact_resistance < 0
            or plate_c > 0 or shape not in ("cap", "slab") or not 0 < safety <= 1):
        raise ValueError("invalid cap, plate, time horizon, shape or safety factor")
    dx = radius/cells
    if height < dx or (0 < glass_depth < dx/2):
        raise ValueError("resolve the height of each material with at least one layer")
    # Round the glass domain to complete cells and expose its actual thickness.
    nz_glass = math.floor(glass_depth/dx+.5)
    halfwidth = math.ceil(1.5*cells)
    xy = (torch.arange(-halfwidth, halfwidth, dtype=torch.float64)+.5)*dx
    z = (torch.arange(math.ceil(height/dx), dtype=xy.dtype)+.5)*dx
    lattice = torch.cartesian_prod(xy, xy, z)
    x, y, z = lattice.T
    sphere_r = (radius**2+height**2)/(2*height)
    if shape == "cap":
        wet = x*x+y*y+(z+sphere_r-height)**2 <= sphere_r*sphere_r
    else:
        wet = (x.abs() < radius) & (y.abs() < radius) & (z < height)
    pos = lattice[wet]
    lap, row = _laplacian(pos, dx)
    bottom = torch.where(pos[:, 2] < dx)[0]
    vol = dx**3
    water_mass = p.RHO_FRESH*vol
    initial_h = p.C_FRESH*initial_water_c
    h = pos.new_full((len(pos),), initial_h)
    max_rate = (row.max()+2/dx**2)*p.K_ICE/(p.RHO_FRESH*FRESH.c_solid)
    cold_potential = float(_heat_potential(pos.new_tensor(plate_c)))
    glass_t = pos.new_empty(0)
    if nz_glass:
        gz = -(torch.arange(nz_glass, dtype=xy.dtype)+.5)*dx
        glass = torch.cartesian_prod(xy, xy, gz)
        glass_lap, glass_row = _laplacian(glass, dx)
        # The Cartesian product stores depth as the fastest varying index.
        horizontal = (pos[bottom, :2]/dx + halfwidth - .5).round().long()
        top = (horizontal[:, 0]*len(xy)+horizontal[:, 1])*nz_glass
        back = torch.arange(nz_glass-1, len(glass), nz_glass)
        glass_t = pos.new_full((len(glass),), plate_c)
        # Includes both face resistances even when the slide is only one cell deep.
        glass_rate = (glass_row.max()*GLASS_K + (4*GLASS_K)/dx**2)
        max_rate = max(float(max_rate), float(glass_rate)/(GLASS_RHO*GLASS_CP))
    dt_limit = safety/float(max_rate)
    time = extracted = peak_residual = 0.
    steps = 0
    history, sections = [], []
    thresholds = iter((0., .25, .50, .75, .99))
    next_section = next(thresholds)
    t99 = last_solid = None
    previous_time = previous_fraction = 0.
    section = (pos[:, 1]-.5*dx).abs() < .01*dx

    while True:
        temp = FRESH.temperature(h)-p.T_MELT
        solid = (-temp/FRESH.mushy_dt).clamp(0, 1)
        frozen = float(solid.mean())
        if t99 is None and frozen >= .99:
            rise = frozen-previous_fraction
            t99 = (previous_time + (time-previous_time)*(.99-previous_fraction)/rise
                   if rise > 0 else time)
        if float(solid.min()) >= 1-1e-9:
            last_solid = time
        if next_section is not None and frozen >= next_section:
            sections.append((time, solid[section].clone()))
            while next_section is not None and frozen >= next_section:
                next_section = next(thresholds, None)
        if (not history or time >= history[-1]["t_s"]+.1 or time >= end_s
                or last_solid is not None):
            stored = float((h-initial_h).sum())*water_mass
            stored += float((glass_t-plate_c).sum())*GLASS_RHO*GLASS_CP*vol
            residual = abs(stored + extracted)
            peak_residual = max(peak_residual, residual)
            history.append(dict(t_s=time, frozen_mass_fraction=frozen,
                                min_solid_fraction=float(solid.min()),
                                extracted_J=extracted, enthalpy_change_J=stored))
        if time >= end_s or last_solid is not None:
            break
        dt = min(dt_limit, end_s-time)
        potential = _heat_potential(temp)
        rates = lap @ potential
        if nz_glass:
            # Equal and opposite contact flux using the two half-cell resistances.
            k = p.K_FRESH + (p.K_ICE-p.K_FRESH)*solid[bottom]
            resistance = .5*dx*(1/k + 1/GLASS_K) + contact_resistance
            face = (glass_t[top]-temp[bottom])/(dx*resistance)
            glass_rates = GLASS_K*(glass_lap @ glass_t)
            glass_rates[top] -= face
            back_flux = 2*GLASS_K*(plate_c-glass_t[back])/dx**2
            glass_rates[back] += back_flux
            glass_t += dt/(GLASS_RHO*GLASS_CP)*glass_rates
        else:
            face = 2*(cold_potential-potential[bottom])/dx**2
            if contact_resistance:
                # Secant conductivity of the existing Kirchhoff potential recovers
                # the perfect-contact stencil as the extra resistance tends to zero.
                dtemp = plate_c-temp[bottom]
                k_eff = torch.where(dtemp != 0, (cold_potential-potential[bottom])/dtemp,
                                    p.K_FRESH + (p.K_ICE-p.K_FRESH)*solid[bottom])
                face = dtemp/(dx*(.5*dx/k_eff + contact_resistance))
            back_flux = face
        rates[bottom] += face
        h += dt/p.RHO_FRESH*rates
        extracted -= dt*vol*float(back_flux.sum())
        previous_time, previous_fraction = time, frozen
        time += dt
        steps += 1
    return DropFreezingResult(pos, h, solid, section, history, sections, dict(
        cells_per_radius=cells, dx_m=dx, base_radius_m=radius, height_m=height, shape=shape,
        glass_depth_m=nz_glass*dx, glass_width_m=len(xy)*dx if nz_glass else 0.,
        contact_resistance_m2K_W=contact_resistance, plate_c=plate_c,
        initial_water_c=initial_water_c, safety=safety,
        water_particles=len(pos), glass_particles=len(glass_t), steps=steps, dt_limit_s=dt_limit,
        final_time_s=time, final_frozen_mass_fraction=frozen, water_mass_kg=len(pos)*water_mass,
        sampled_energy_residual_over_latent=peak_residual/(len(pos)*water_mass*FRESH.latent),
        t99_s=t99, last_particle_frozen_s=last_solid))
