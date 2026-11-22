"""Instruments on the SPHERIC Test 2 box (Release 1.1, Figs. 6–7)."""

import math

import torch

# Experimental_data worksheet: peak in [0.35, 2.6] s and 0.02 m upward
# crossing, linearly interpolated on the original acquisition clock.
H2_PEAK_M = .3408
H2_PEAK_TIME_S = 1.7870422657819276
H2_ARRIVAL_S = .3673968228757868


def sensor_positions(cfg, like):
    """Solver coordinates (downstream, up, span). First four taps face upstream."""
    offsets = [0.021 + 0.04 * i for i in range(4)]
    front = [[cfg.obstacle_x, s, cfg.obstacle_z + 0.176] for s in offsets]
    top = [[cfg.obstacle_x + s, cfg.obstacle_height, cfg.obstacle_z1() - 0.176] for s in offsets]
    return like.new_tensor(front + top)


def _cubic(q):
    return torch.where(
        q < 1.0, 1.0 - 1.5 * q.square() + 0.75 * q.pow(3), 0.25 * (2.0 - q).clamp_min(0.0).pow(3)
    )


def centreline_height(pos, volume, cfg):
    """Water volume per horizontal area with a 2-D kernel centred on H2 (m)."""
    h = cfg.h_factor * cfg.dx
    r = torch.hypot(pos[:, 0] - cfg.station_x(), pos[:, 2] - 0.5 * cfg.tank_depth)
    return (volume * _cubic(r / h)).sum() * (10.0 / (7.0 * math.pi * h * h))


def tap_disk_nodes(cfg, taps, diameter):
    """Centre and six azimuthal nodes for a fifth-degree equal-area disk cubature."""
    if not math.isfinite(diameter) or diameter < 0:
        raise ValueError("pressure disk diameter must be finite and nonnegative")
    if diameter == 0:
        return taps[:, None, :]
    radius = diameter / 2
    clearance = torch.cat((taps[:4, 1], cfg.obstacle_height - taps[:4, 1],
                           taps[4:, 0] - cfg.obstacle_x, cfg.obstacle_x1() - taps[4:, 0],
                           taps[:, 2] - cfg.obstacle_z, cfg.obstacle_z1() - taps[:, 2]))
    if radius > float(clearance.min()) + 1e-7:
        raise ValueError("pressure disk extends beyond the instrumented face")
    angle = torch.arange(6, device=taps.device, dtype=taps.dtype) * (math.pi / 3)
    ring = radius * math.sqrt(2 / 3)
    nodes = taps[:, None, :].expand(-1, 7, -1).clone()
    nodes[:4, 1:, 1] += ring * angle.cos()
    nodes[4:, 1:, 0] += ring * angle.cos()
    nodes[:, 1:, 2] += ring * angle.sin()
    return nodes


def tap_node_pressures(pos, pressure, density, volume, cfg, nodes):
    """One-sided SPH reconstruction at each face quadrature node (Pa)."""
    h = cfg.h_factor * cfg.dx
    near = (
        (pos[:, 0] >= cfg.obstacle_x - 2 * h)
        & (pos[:, 0] <= cfg.obstacle_x1() + 2 * h)
        & (pos[:, 1] <= cfg.obstacle_height + 2 * h)
        & (pos[:, 2] >= nodes[..., 2].min() - 2 * h)
        & (pos[:, 2] <= nodes[..., 2].max() + 2 * h)
    )
    delta = pos[near, None, None, :] - nodes
    weights = _cubic(torch.linalg.vector_norm(delta, dim=-1) / h) * volume[near, None, None]
    weights[:, :4] *= delta[:, :4, :, 0] <= 0
    weights[:, 4:] *= delta[:, 4:, :, 1] >= 0
    corrected = pressure[near, None, None] + density[near, None, None] * cfg.gravity * delta[..., 1]
    p = (weights * corrected).sum(0) / weights.sum(0).clamp_min(torch.finfo(pos.dtype).tiny)
    return p.clamp_min(0.0), (weights > 0).sum(0)


def disk_average(node_pressures):
    """Atmospheric gauge pressure on dry nodes contributes zero to the full disk."""
    return .25 * node_pressures[:, 0] + .125 * node_pressures[:, 1:].sum(1)


def tap_pressures(pos, pressure, density, volume, cfg, taps):
    """Point gauge pressure and support at the eight sensor centres."""
    p, count = tap_node_pressures(pos, pressure, density, volume, cfg, taps[:, None, :])
    return p[:, 0], count[:, 0]
