"""Single-resolution 2-D δ⁺-SPH (Sun et al., CMAME 315, 2017).

The positional shift uses the symmetric density denominator in Sun et al.'s
water-entry formulation, eq. (5). The interface geometry excludes wall ghosts
as in its section 2.2. See benchmarks/delta_plus/README.md for solver choices.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .density_diffusion import hydrostatic_corrected_rate
from .droplet_impact import wendland_c2_gradient, wendland_c2_kernel
from .neighbor_search import cell_linked_pairs


@dataclass
class DeltaPlusStep:
    position: torch.Tensor
    velocity: torch.Tensor
    density: torch.Tensor
    acceleration: torch.Tensor
    wall_reaction: torch.Tensor
    shift: torch.Tensor


def _both(template, i, j, at_i, at_j):
    """Sum contributions from both ends of an unordered fluid pair."""
    return torch.zeros_like(template).index_add_(0, i, at_i).index_add_(0, j, at_j)


def _fluid_geometry(pos, volume, i, j, sep, grad):
    # First moment M = sum Vj (rj-ri) outer grad_i W. It is symmetric for
    # radial kernels. The eigenvalues of M (not its inverse) measure support.
    outer = -sep.unsqueeze(2) * grad.unsqueeze(1)
    moment = _both(pos.new_empty((len(pos), 2, 2)), i, j,
                   volume[j, None, None]*outer, volume[i, None, None]*outer)
    eig, axes = torch.linalg.eigh(moment)
    floor = (0.1 * eig.mean(1, keepdim=True)).clamp_min(1.e-12)
    inverse = (axes / eig.clamp_min(floor).unsqueeze(1)) @ axes.transpose(1, 2)

    def gradient(values):
        pair_grad = (values[j] - values[i])[:, None]*grad
        summed = _both(pos, i, j, volume[j, None]*pair_grad, volume[i, None]*pair_grad)
        return (inverse @ summed.unsqueeze(2)).squeeze(2)

    return eig[:, 0], gradient


def _boundary_band(pos, i, j, sep, lam, grad_lambda, h):
    """Marrone-style outward empty-cap test and a one-kernel-radius band."""
    magnitude = grad_lambda.norm(dim=1)
    normal = -grad_lambda / magnitude.clamp_min(1.e-12)[:, None]

    def occupied(ray, direction):
        axial = (ray * direction).sum(1)
        tangent = (ray[:, 0]*direction[:, 1] - ray[:, 1]*direction[:, 0]).abs()
        offset = ray - h*direction
        return torch.where(ray.square().sum(1) >= 2*h*h,
                           offset.square().sum(1) < h*h,
                           (axial-h).abs() + tangent < h)

    covered = _both(lam, i, j, occupied(-sep, normal[i]).to(lam),
                    occupied(sep, normal[j]).to(lam))
    surface = (lam < .2) | ((lam < .75) & (covered == 0))
    neighbors = _both(lam, i, j, surface[j].to(lam), surface[i].to(lam))
    return surface | (neighbors > 0), normal, magnitude*h


def _regularize(raw, band, normal, normal_strength, lam):
    tangent = raw - (raw*normal).sum(1, keepdim=True)*normal
    # Freeze sparse sheets and poorly determined normals in the interface band.
    resolved = (lam >= .55) & (normal_strength > 1.e-6)
    return torch.where(band[:, None], torch.where(resolved[:, None], tangent, 0.), raw)


def _wall_tangent(shift, normal, band, f, distance_sq, ghost_normal):
    """Impose the nearest solid tangent, including immersed thin walls.

    In 2-D independent surface and wall normals leave zero allowed shift.
    Equidistant stations with conflicting normals also pin the corner.
    """
    if not f.numel():
        return shift
    nearest = shift.new_full((len(shift),), math.inf)
    nearest.scatter_reduce_(0, f, distance_sq, reduce="amin")
    tied = distance_sq == nearest[f]
    particle, normals = f[tied], ghost_normal[tied]
    # Select a deterministic representative for exact distance ties.
    index = torch.arange(len(particle), device=f.device)
    first = f.new_full((len(shift),), len(particle))
    first.scatter_reduce_(0, particle, index, reduce="amin")
    direction = normals[first.clamp_max(len(particle)-1)]
    tangent = shift - (shift*direction).sum(1, keepdim=True)*direction
    tol = 1.e-6 * shift.norm(dim=1)
    conflict = band & ((tangent*normal).sum(1).abs() > tol)
    other_wall = (tangent[particle]*normals).sum(1).abs() > tol[particle]
    conflict.scatter_reduce_(0, particle, other_wall, reduce="amax")
    return torch.where(nearest.isfinite()[:, None],
                       torch.where(conflict[:, None], 0., tangent), shift)


def delta_plus_step(pos, velocity, density, mass, *, dx, sound_speed, reference_speed,
                    dt, rho0=1000., gravity=9.81, viscosity=1.e-6, diffusion=.1, alpha=.02,
                    wall=None, wall_velocity=None, wall_normal=None, pairs=None):
    """Advance an equal-spacing 2-D liquid with constant particle masses.

    SI units, unit span. The smoothing length is 2 dx. Positions are advanced
    with the new physical velocity plus one separate positional shift. Density
    is a persistent state and must be passed back on the next call. The fluid
    exchanges pressure forces with prescribed wall stations of volume dx².
    Continuity uses a slip ghost reflected in each station normal. The radial
    artificial viscosity has coefficient alpha, in addition to physical viscosity.
    Pairs follow the package convention (unique i < j, fluid before wall);
    cached lists may include skin pairs. Wall reactions are integrated forces.
    """
    if (pos.ndim != 2 or pos.shape[1] != 2 or velocity.shape != pos.shape
            or density.shape != pos.shape[:1] or mass.shape != density.shape
            or not len(pos) or pos.dtype not in (torch.float32, torch.float64)):
        raise ValueError("expected a nonempty 2-D fluid and one density and mass per particle")
    if not all(math.isfinite(x) for x in (dx, sound_speed, reference_speed, dt,
                                          rho0, gravity, viscosity, diffusion, alpha)):
        raise ValueError("solver parameters must be finite")
    if min(dx, sound_speed, dt, rho0) <= 0 or min(reference_speed, viscosity, diffusion, alpha) < 0:
        raise ValueError("invalid length, speed, density, step or transport coefficient")
    h = 2*dx
    radial_viscosity = 8*viscosity + alpha*h*sound_speed
    if (dt*sound_speed > .3*h or dt*radial_viscosity/8 > .125*h*h
            or dt*diffusion*sound_speed > .125*h):
        raise ValueError("time step exceeds an acoustic, viscous or diffusive limit")
    if wall is None:
        wall = pos.new_empty((0, 2))
    if wall_velocity is None:
        wall_velocity = torch.zeros_like(wall)
    if wall.ndim != 2 or wall.shape[1] != 2 or wall_velocity.shape != wall.shape:
        raise ValueError("expected matching (W, 2) wall positions and velocities")
    if wall_normal is None and len(wall) == 0:
        wall_normal = wall
    if wall_normal is None or wall_normal.shape != wall.shape:
        raise ValueError("provide a wall_normal per ghost station for free slip")
    if any(x.dtype != pos.dtype or x.device != pos.device
           for x in (velocity, density, mass, wall, wall_velocity, wall_normal)):
        raise ValueError("states must share a floating dtype and device")
    if not bool(((density > 0) & (mass > 0) & density.isfinite() & mass.isfinite()).all()):
        raise ValueError("density and mass must be positive and finite")
    norms = wall_normal.norm(dim=1)
    if not bool((norms.isfinite() & (norms > 0)).all()):
        raise ValueError("wall normals must be finite and nonzero")
    wall_normal = wall_normal / norms[:, None]
    if not all(bool(x.isfinite().all()) for x in (pos, velocity, wall, wall_velocity)):
        raise ValueError("particle positions and velocities must be finite")
    cloud = torch.cat((pos, wall))
    if pairs is None:
        pairs = cell_linked_pairs(cloud, 2*h)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("expected a (P, 2) long pair list")
    if pairs.device != pos.device:
        raise ValueError("pair list must share the device of the state")
    i, j = pairs.unbind(1)
    if bool(((i < 0) | (i >= j) | (j >= len(cloud))).any()):
        raise ValueError("each pair must have 0 <= i < j < number of particles")
    keep = (i < len(pos)) & ((cloud[i] - cloud[j]).square().sum(1) < (2*h)**2)
    i, j = i[keep], j[keep]
    sep = cloud[i] - cloud[j]
    radius = sep.norm(dim=1)
    weight = wendland_c2_kernel(radius, h)
    grad = wendland_c2_gradient(sep, h)
    ff = j < len(pos)
    a, b, d, g = i[ff], j[ff], sep[ff], grad[ff]
    f, w, dw, gw, ww = i[~ff], j[~ff]-len(pos), sep[~ff], grad[~ff], weight[~ff]
    volume = mass/density
    p = sound_speed**2 * (density-rho0)
    # Fixed-ghost pressure extrapolation with gravity, zero wall acceleration.
    p_sum = pos.new_zeros(len(wall)).index_add_(0, w, ww*(p[f] + density[f]*gravity*dw[:, 1]))
    w_sum = pos.new_zeros(len(wall)).index_add_(0, w, ww)
    wall_p = p_sum / w_sum.clamp_min(torch.finfo(pos.dtype).tiny)
    wall_rho = rho0 + wall_p/sound_speed**2

    if bool((wall_rho[w] <= 0).any()):
        raise FloatingPointError("nonpositive extrapolated wall density")
    force = -(volume[a]*volume[b]*(p[a]+p[b]))[:, None]*g
    if radial_viscosity:
        approach = ((velocity[a]-velocity[b])*d).sum(1)
        # Monaghan's radial viscous stress, 2(D+2) in two dimensions.
        pi_ij = approach / d.square().sum(1).clamp_min(1.e-12*h*h)
        force += (rho0*radial_viscosity*volume[a]*volume[b]*pi_ij)[:, None]*g
    # Orient each impermeable face toward the ghost, including both sides
    # of an immersed strip. The linear acoustic wall pressure damps normal
    # relative motion and does no tangential work in free slip.
    n_wall = wall_normal[w]
    toward_ghost = n_wall * (gw*n_wall).sum(1, keepdim=True).sign()
    closing_speed = ((velocity[f]-wall_velocity[w])*toward_ghost).sum(1)
    normal_grad = (gw*toward_ghost).sum(1, keepdim=True)*toward_ghost
    twice_wall_pressure = p[f] + wall_p[w] + 2*density[f]*sound_speed*closing_speed
    wall_force = -(volume[f]*dx**2*twice_wall_pressure)[:, None]*normal_grad
    net = _both(pos, a, b, force, -force).index_add_(0, f, wall_force)
    reaction = torch.zeros_like(wall).index_add_(0, w, -wall_force)
    acc = net / mass[:, None] + pos.new_tensor([0., -gravity])
    new_velocity = velocity + dt*acc

    lam, gradient = _fluid_geometry(pos, volume, a, b, d, g)
    div_pair = ((new_velocity[b]-new_velocity[a])*g).sum(1)
    div = _both(density, a, b, volume[b]*div_pair, volume[a]*div_pair)
    normal_speed = ((wall_velocity[w]-new_velocity[f])*n_wall).sum(1)
    div.index_add_(0, f, 2*dx**2*normal_speed*(n_wall*gw).sum(1))
    rate = -density*div
    if diffusion:
        rate += hydrostatic_corrected_rate(density, mass, d, -(d*g).sum(1),
                                           d.square().sum(1), (a, b),
                                           2*diffusion*h*sound_speed, gradient(density))
    new_density = density + dt*rate
    if not bool((new_density.isfinite() & (new_density > 0)).all()):
        raise FloatingPointError("δ⁺-SPH density became nonpositive or nonfinite")

    band, normal, strength = _boundary_band(pos, a, b, d, lam, gradient(lam), h)
    w_ref = wendland_c2_kernel(pos.new_tensor(dx), h)
    repulsion = (1 + .2*(weight/w_ref)**4)[:, None]*grad
    q = repulsion[ff] / (density[a]+density[b])[:, None]
    raw = _both(pos, a, b, mass[b, None]*q, -mass[a, None]*q)
    wall_ratio = wall_rho[w]*dx**2 / (density[f]+wall_rho[w])
    raw.index_add_(0, f, wall_ratio[:, None]*repulsion[~ff])
    # Acoustic CFL = dt*c0/h. Apply only once per accepted physical step.
    raw *= -(dt*sound_speed/h)*(reference_speed/sound_speed)*(2*h)**2
    shift = _regularize(raw, band, normal, strength, lam)
    shift = _wall_tangent(shift, normal, band, f, dw.square().sum(1), n_wall)
    return DeltaPlusStep(pos+dt*new_velocity+shift, new_velocity, new_density,
                         acc, reaction, shift)
