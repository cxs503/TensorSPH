"""m3653: insert the Adami (2012) free-surface pressure closure into TensorSPH.

Run once from the repo root; asserts every anchor so a partial match cannot
silently corrupt the file.
"""
from pathlib import Path

P = Path("src/tensorsph/droplet_impact.py")
src = P.read_text()

# ---------------------------------------------------------------- 1. Config ---
anchor = '''    csf_curvature: str = "fit_quadratic"
'''
assert src.count(anchor) == 1
cfg_block = '''    csf_curvature: str = "fit_quadratic"
    # Free-surface pressure closure, Adami, Ihmsen & Adams (2012), JCP 231:7057,
    # section 3.3. Off by default.
    # ``tait_pressure`` clamps p >= 0, so every particle in the density-deficit
    # shell of a free surface carries p = 0 exactly. A drop in vacuum then has
    # no way to hold the Young-Laplace jump sigma/R across its interface: only
    # the interior settles, at ~0.57 sigma/R (measured in
    # benchmarks/laplace_pressure_floor/, residual -34 %).
    # When on, particles flagged as free surface (rho/rho0 <
    # ``free_surface_rho_ratio``) get their pressure extrapolated from reliable
    # neighbours, exactly as ``rhs`` already extrapolates the *wall* pressure.
    # It is opt-in because it changes every free-surface case, and the A/B record
    # that would justify a default flip lives in
    # docs/verification_free_surface_pressure_closure.md.
    free_surface_pressure: bool = False
    # Deficit ratio that flags a fluid particle as free surface.
    free_surface_rho_ratio: float = 0.99
    # Number of outward layers filled by the extrapolation (Adami's N_s). Each
    # pass may only use neighbours that already carry a trustworthy pressure, so
    # the closure walks inwards out instead of averaging zeros into zeros.
    free_surface_layers: int = 3
'''
src = src.replace(anchor, cfg_block, 1)

# ------------------------------------------------------- 2. closure operator ---
anchor2 = '''def acoustic_courant(dt: float, cfg: Config, max_speed: float = 0.0) -> float:
'''
assert src.count(anchor2) == 1
closure = '''def free_surface_pressure_closure(pos, rho, pressure, mass, cfg: Config,
                                   pairs=None, n=None):
    """Fill the free-surface pressure deficit by kernel extrapolation.

    Implements section 3.3 of Adami, Ihmsen & Adams (2012), JCP 231:7057::

        p_i = max(0, sum_j (p_j + rho_j g.(r_i - r_j)) W_ij) / sum_j W_ij

    summed over neighbours ``j`` that already carry a trustworthy pressure. The
    sum is done one shell layer at a time (``cfg.free_surface_layers`` passes),
    so a particle in the second layer uses the first layer's extrapolated values
    instead of averaging the ``p = 0`` of the particle outside it into its own
    estimate.

    This is the free-surface analogue of the wall-pressure extrapolation that
    ``rhs`` already performs, and it exists because ``tait_pressure`` clamps
    ``p >= 0``: in the density-deficit shell of a free surface the clamp pins
    every particle to exactly zero, so the interface cannot carry the
    Young-Laplace jump (see docs/verification_laplace_pressure_floor.md).

    Parameters
    ----------
    pos, rho, pressure : (n, D), (n,), (n,)
        Fluid particles only. ``pressure`` is the raw Tait pressure; particles
        that the closure cannot reach are returned unchanged.
    mass : (n,)
        Fluid masses. Kept in the signature so callers pass the same state as
        to :func:`rhs`; the kernel average in Adami eq. (11) is mass weighted.
    pairs : (P, 2) long tensor, optional
        Pair list over the fluid cloud. Built with ``cell_linked_pairs`` when
        omitted. Pairs that index past ``n`` (wall particles) are ignored.
    n : int, optional
        Number of fluid particles, when ``pos`` carries wall particles too.

    Returns
    -------
    (n,) tensor of pressures, finite, ``>= 0``.
    """
    if not cfg.free_surface_pressure:
        return pressure
    n = pos.shape[0] if n is None else int(n)
    h = cfg.h_factor * cfg.dx
    rho_f = rho[:n]
    shell = rho_f / cfg.rho0 < cfg.free_surface_rho_ratio
    if not bool(shell.any()):
        return pressure

    if pairs is None:
        pairs = cell_linked_pairs(pos[:n], 2.0 * h)
    i, j = pairs.unbind(dim=1)
    inside = (i < n) & (j < n)
    i, j = i[inside], j[inside]
    delta = pos[i] - pos[j]
    wij, _ = pair_kernels(delta, h)

    # Every pair is stored once, but the closure sums over *all* neighbours of a
    # particle, so both directions are needed. Direction a <- b carries
    # ``y_a - y_b`` (the body-force term) and ``rho_b``.
    a = torch.cat((i, j))
    b = torch.cat((j, i))
    w = torch.cat((wij, wij))
    dy = torch.cat((delta[:, 1], -delta[:, 1]))
    rho_b = torch.cat((rho_f[j], rho_f[i]))

    zeros = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    p = pressure[:n].clone()
    reliable = ~shell
    for _ in range(max(1, int(cfg.free_surface_layers))):
        usable = shell[a] & reliable[b]
        weight = torch.where(usable, w, zeros)
        source = p[b] * weight
        if cfg.gravity != 0.0:
            # g = (0, -gravity), so g.(r_a - r_b) = -gravity * (y_a - y_b). The
            # neighbouring particle's *own* acceleration is neglected, as in the
            # wall treatment above (the drop is in free fall / at rest).
            source = source + (rho_b * (-cfg.gravity) * dy) * weight
        numerator = zeros.index_add(0, a, source)
        denominator = zeros.index_add(0, a, weight)
        filled = shell & (denominator > 0.0) & ~reliable
        if not bool(filled.any()):
            break
        filled_pressure = (numerator / denominator.clamp_min(1.0e-30)).clamp_min(0.0)
        p = torch.where(filled, filled_pressure, p)
        reliable = reliable | filled

    if pressure.shape[0] > n:
        return torch.cat((p, pressure[n:]))
    return p


def acoustic_courant(dt: float, cfg: Config, max_speed: float = 0.0) -> float:
'''
src = src.replace(anchor2, closure, 1)

# ------------------------------------------------------------ 3. rhs wiring ---
anchor3 = '''    pressure = tait_pressure(rho, cfg)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
'''
assert src.count(anchor3) == 1
wired = '''    pressure = tait_pressure(rho, cfg)

    if cfg.free_surface_pressure:
        # Before the wall extrapolation, so the wall pressure is extrapolated
        # from a fluid pressure that already carries the interface jump.
        pressure = free_surface_pressure_closure(
            pos, rho, pressure, mass, cfg, pairs=pairs, n=n)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
'''
src = src.replace(anchor3, wired, 1)

# ------------------------------------------------------- 4. validate_config ---
anchor4 = '''    if not 0.0 < cfg.dt_safety <= 1.0:
        raise ValueError("dt_safety must be in (0, 1]")
'''
assert src.count(anchor4) == 1
src = src.replace(anchor4, anchor4 + '''    if not 0.0 < cfg.free_surface_rho_ratio <= 1.0:
        raise ValueError("free_surface_rho_ratio must be in (0, 1]")
    if cfg.free_surface_layers < 1:
        raise ValueError("free_surface_layers must be at least 1")
''', 1)

P.write_text(src)
print("patched", P)

# ------------------------------------------------------------------ exports ---
Q = Path("src/tensorsph/__init__.py")
s = Q.read_text()
s = s.replace("    drop_wall_wall_pairs,\n", "    drop_wall_wall_pairs,\n    free_surface_pressure_closure,\n", 1)
s = s.replace('    "drop_wall_wall_pairs",\n', '    "drop_wall_wall_pairs", "free_surface_pressure_closure",\n', 1)
Q.write_text(s)
print("patched", Q)