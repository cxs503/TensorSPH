"""SPHERIC Test 19 water column with a measured lift gate and a moving thin plate.

Single-phase 2-D WCSPH. The adapter passes the actual pairwise wall reactions
in newtons (one metre of span) to a structural integrator. SI coordinates are
+x downstream, +y upward. The structural API is duck typed so installing the
fluid package does not install TensorFEM.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math

import torch

from .dam_break import DamBreakConfig, make_column
from .delta_plus import delta_plus_step
from .droplet_impact import rhs, stable_dt, wall_lattice
from .neighbor_search import PairListCache


@dataclass(frozen=True)
class GateMotion:
    """Piecewise-linear measured lift in metres, held at its final height."""
    times: tuple[float, ...]
    lifts: tuple[float, ...]

    def __post_init__(self):
        if (len(self.times) < 2 or len(self.times) != len(self.lifts)
                or self.times[0] != 0 or self.lifts[0] != 0
                or not all(math.isfinite(x) for x in (*self.times, *self.lifts))
                or any(b <= a for a, b in zip(self.times, self.times[1:]))
                or any(b < a for a, b in zip(self.lifts, self.lifts[1:]))):
            raise ValueError("expected finite, rising time/lift samples starting at (0,0)")

    @classmethod
    def read(cls, path):
        samples = []
        with open(path, encoding="utf-8-sig") as source:
            for line in source:
                fields = line.partition("#")[0].split()
                if fields:
                    t, h = map(float, fields)
                    samples.append((t, h))
        return cls(tuple(t for t, h in samples), tuple(h for t, h in samples))

    def at(self, t):
        if not math.isfinite(t) or t < 0:
            raise ValueError("gate time must be finite and nonnegative")
        i = bisect_right(self.times, t) - 1
        if i == len(self.times) - 1:
            return self.lifts[-1], 0.
        rate = (self.lifts[i+1] - self.lifts[i]) / (self.times[i+1] - self.times[i])
        return self.lifts[i] + rate * (t - self.times[i]), rate


def tank_contact(previous, pos, velocity, mass, width, height):
    """Inelastic, frictionless contact with floor/side planes; the top is open.

    Correct a trial step in place and return the absolute normal impulse delivered
    to the water [N s per metre of span]. Each component can only lose kinetic
    energy. The impulse belongs to the tank and is excluded from the plate load.
    """
    x, y = pos.unbind(1)
    old_x, old_y = previous.unbind(1)
    moving_x = torch.where(x == old_x, 1., x - old_x)
    # Test height at the crossing, so an overtopped parcel falling outside the
    # tank is never projected back through the side wall.
    left = (old_x >= 0) & (x < 0) & (old_y - old_x / moving_x * (y-old_y) <= height)
    right = ((old_x <= width) & (x > width)
             & (old_y + (width-old_x) / moving_x * (y-old_y) <= height))
    bottom = (y < 0) & ((x >= 0) | left) & ((x <= width) | right)
    vx, vy = velocity.unbind(1)
    stop_x = (left & (vx < 0)) | (right & (vx > 0))
    stop_y = bottom & (vy < 0)
    impulse = (mass * (vx.abs()*stop_x + vy.abs()*stop_y)).sum().item()
    x[left] = 0
    x[right] = width
    y[bottom] = 0
    vx[stop_x] = 0
    vy[stop_y] = 0
    return impulse


def simulate(gate, strip, *, dx=.01, height=.3, end=.46, courant=.22, sound_speed=60.,
             contact=False, method="wcsph"):
    """Advance both solvers and return synchronous marker histories and diagnostics.

    The supplied strip has length 0.09 m and implements station_weights, sample
    and advance. Positions from sample are relative to the clamp. Each force
    belongs to one wall station, already integrated over its tributary area.
    """
    if height not in (.2, .3, .4) or not math.isfinite(dx) or not 0 < dx <= .03:
        raise ValueError("expected H=0.2, 0.3 or 0.4 m and 0 < dx <= 0.03 m")
    if not math.isfinite(end) or end <= 0 or not 0 < courant <= .3:
        raise ValueError("expected positive end time and 0 < courant <= 0.3")
    if not math.isfinite(sound_speed) or sound_speed <= 0:
        raise ValueError("sound speed must be positive and finite")
    if not math.isclose(strip.length, .09):
        raise ValueError("Test 19 requires a 0.09 m strip")
    cfg = DamBreakConfig(dx=dx, pool_width=.8, pool_depth=.6, tank_height=.6,
                         column_width=.2, column_height=height, sound_speed=sound_speed,
                         viscosity=1.e-6, max_courant=courant, density_calibration=True)
    if method not in ("wcsph", "delta-plus"):
        raise ValueError("method must be wcsph or delta-plus")
    reference_speed = math.sqrt(2*cfg.gravity*height)
    if method == "delta-plus":
        cfg.h_factor = 2.
        cfg.wall_layers = 4
    cpu = torch.device("cpu")
    pos, mass = make_column(cfg, cpu)
    pos, mass = pos.double(), mass.double()
    max_density_error = max_shift_dx = 0.
    first_regime_exit = None
    if method == "delta-plus":
        # Exact initial barotropic hydrostatics, p = c0²(rho-rho0).
        density = cfg.rho0*torch.exp(cfg.gravity*(height - pos[:, 1])/sound_speed**2)
        mass = density * dx**2
        max_density_error = float((density/cfg.rho0-1).abs().max())
        if max_density_error > .01:
            first_regime_exit = {"t_s": 0., "mach": 0.,
                                 "relative_density_deviation": max_density_error}
    velocity = torch.zeros_like(pos)
    tank = wall_lattice(cfg, cpu, top=.6).double()
    # The first gate row is one full cell from the last water cell centre.
    y = (torch.arange(math.ceil(.6/dx), dtype=pos.dtype) + .5)*dx
    gate0 = torch.stack((torch.full_like(y, .2 + dx/2), y), dim=1)
    if method == "delta-plus":
        # For slip ghosts the sign of the unit normal is immaterial.
        floor = tank[:, 1] < 0
        tank_normal = torch.stack((~floor, floor), dim=1).to(pos)
        gate_normal = torch.zeros_like(gate0)
        gate_normal[:, 0] = 1
    panels = round(strip.length / dx)
    arcs = (torch.arange(panels, dtype=pos.dtype) + .5) * (strip.length / panels)
    interface = strip.station_weights(arcs)
    markers = strip.station_weights([.0875, .065, .040])
    clamp = pos.new_tensor([.6, 0.])
    cache = PairListCache(2*cfg.h_factor*dx, .3*dx, fluid_count=len(pos))
    t = peak_mach = peak_balance = max_escape = 0.
    wall_work = strip_work = swept_work = 0.
    contact_impulse = peak_trial_escape = 0.
    worst_position = None
    first_large_excursion = None
    escaped = torch.zeros(len(pos), dtype=torch.bool)
    history = [(0., 0., 0., 0.)]
    while t < end:
        lift, rate = gate.at(t)
        plate, plate_v = strip.sample(interface)
        wall = torch.cat((tank, gate0 + pos.new_tensor([0., lift]), plate + clamp))
        wall_v = torch.zeros_like(wall)
        wall_v[len(tank):len(tank)+len(y), 1] = rate
        wall_v[-panels:] = plate_v
        speed = float(velocity.norm(dim=1).max())
        peak_mach = max(peak_mach, speed / cfg.sound_speed)
        dt = min(end - t, stable_dt(cfg, max(speed, abs(rate), float(plate_v.norm(dim=1).max()))))
        pairs, _ = cache.get(torch.cat((pos, wall)))
        previous = pos
        if method == "delta-plus":
            tangent = torch.gradient(plate, dim=0)[0]
            plate_normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
            wall_normal = torch.cat((tank_normal, gate_normal, plate_normal))
            step = delta_plus_step(pos, velocity, density, mass, dx=dx, dt=dt,
                                   sound_speed=sound_speed, reference_speed=reference_speed,
                                   rho0=cfg.rho0, gravity=cfg.gravity, viscosity=cfg.viscosity,
                                   pairs=pairs, wall=wall, wall_velocity=wall_v,
                                   wall_normal=wall_normal)
            pos, velocity, density = step.position, step.velocity, step.density
            acceleration, reactions = step.acceleration, step.wall_reaction
            density_error = float((density/cfg.rho0-1).abs().max())
            step_mach = float(velocity.norm(dim=1).max())/sound_speed
            max_density_error = max(max_density_error, density_error)
            peak_mach = max(peak_mach, step_mach)
            if first_regime_exit is None and (step_mach > .1 or density_error > .01):
                first_regime_exit = {"t_s": t+dt, "mach": step_mach,
                                     "relative_density_deviation": density_error}
            max_shift_dx = max(max_shift_dx, float(step.shift.norm(dim=1).max())/dx)
        else:
            _, _, acceleration, loads = rhs(pos, velocity, mass, cfg, pairs=pairs, wall=wall,
                                            wall_vel=wall_v, dt=dt, split=True)
            reactions = loads["reaction"]
            velocity = velocity + dt * acceleration
            pos = pos + dt * velocity
        # Independent balance of horizontal momentum per step, including all boundaries.
        imbalance = abs(float((acceleration[:, 0]*mass).sum() + reactions[:, 0].sum()))
        peak_balance = max(peak_balance, imbalance)
        panel_force = reactions[-panels:]
        strip.advance(interface, panel_force, dt)
        plate_after, _ = strip.sample(interface)
        # Explicit fluid wall velocity is frozen at the beginning of the step.
        # The subcycled structure traverses its actual station displacements.
        wall_work += dt * float((panel_force*plate_v).sum())
        work = (panel_force*(plate_after-plate)).sum(dim=1)
        strip_work += float(work.sum())
        swept_work += float(work.abs().sum())
        if not bool(pos.isfinite().all() & velocity.isfinite().all()):
            raise FloatingPointError(f"non-finite fluid state at {t:g} s")
        # The experimental tank is open above y=0.60 m. Report envelope excursions
        # and their coordinates so a splash over its rim can be distinguished.
        outside = torch.maximum(-pos[:, 1], torch.maximum(-pos[:, 0], pos[:, 0] - cfg.pool_width))
        if contact:
            peak_trial_escape = max(peak_trial_escape, float(outside.max()))
            contact_impulse += tank_contact(previous, pos, velocity, mass,
                                            cfg.pool_width, cfg.tank_height)
            outside = torch.maximum(-pos[:, 1],
                                    torch.maximum(-pos[:, 0], pos[:, 0] - cfg.pool_width))
        excursion, particle = outside.max(0)
        escaped |= outside > .25*dx
        t += dt
        if float(excursion) > max_escape:
            max_escape = float(excursion)
            worst_position = {"t_s": t, "xy_m": pos[int(particle)].tolist()}
            if first_large_excursion is None and max_escape > .25*dx:
                first_large_excursion = dict(worst_position)
        points, _ = strip.sample(markers)
        history.append((t, *points[:, 0].tolist()))
    method_info = {"method": method, "h_over_dx": cfg.h_factor}
    if method == "delta-plus":
        method_info.update(max_density_relative_deviation=max_density_error,
                           max_shift_over_dx=max_shift_dx, reference_speed_m_s=reference_speed,
                           weak_compressibility_gate=("pass" if first_regime_exit is None
                                                      else "fail"),
                           first_weak_compressibility_exit=first_regime_exit)
    return history, {**method_info, "dx_m": dx, "height_m": height, "courant_limit": courant,
                     "sound_speed_m_per_s": sound_speed,
                     "fluid_particles": len(pos), "steps": len(history)-1,
                     "peak_fluid_mach": max(peak_mach, float(velocity.norm(dim=1).max())
                                            / cfg.sound_speed),
                     "max_horizontal_force_imbalance_N": peak_balance,
                     "plate_work_on_fluid_J": -wall_work,
                     "plate_work_on_structure_J": strip_work,
                     "absolute_station_work_J": swept_work,
                     "partition_work_residual_J": strip_work-wall_work,
                     "partition_work_relative_to_absolute": (
                         abs(strip_work-wall_work)/swept_work if swept_work else None),
                     "tank_contact": contact,
                     "total_absolute_tank_contact_impulse_Ns": contact_impulse,
                     "max_trial_escape_m": max(peak_trial_escape, max_escape),
                     "max_tank_escape_m": max_escape, "worst_excursion": worst_position,
                     "first_excursion_over_quarter_dx": first_large_excursion,
                     "particles_ever_outside_by_quarter_dx": int(escaped.sum()),
                     "escaped_mass_fraction": float(mass[escaped].sum()/mass.sum())}
