"""Surgical patch: Adami et al. (2012) free-surface pressure closure.

Run once from the worktree root.  Every anchor is asserted to appear exactly
once, so a silent drift in the source stops the patch instead of mangling it.
"""
import pathlib

path = pathlib.Path("src/tensorsph/droplet_impact.py")
src = path.read_text()


def sub(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old[:80])
    return text.replace(old, new)


# ---------------------------------------------------------------- Config ----
src = sub(src, """    density_diffusion: float = 0.0
""", """    density_diffusion: float = 0.0
    # Free-surface pressure closure. "clamp" (default) is the first-order
    # treatment: the Tait law is clamped at p >= 0, so the deficit shell
    # ``rho/rho0 < threshold`` carries no pressure and the interior only builds
    # up a fraction of the pressure that the boundary condition asks for.
    # "shell" extrapolates the pressure onto the deficit shell from its
    # neighbours (Adami, Ihmsen & Adams 2012, eq. 11 with ``a_i`` the
    # non-pressure acceleration), which is the closure the Young-Laplace
    # verification was waiting for: see
    # docs/verification_free_surface_pressure_closure.md.
    free_surface_pressure: str = "clamp"
    free_surface_threshold: float = 0.99
""")

# ------------------------------------------------------------- new helper ---
src = sub(src, """    bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
    ratio = (rho / cfg.rho0).clamp_min(0.01)
    return (bulk * (ratio.pow(cfg.gamma) - 1.0)).clamp_min(0.0)
""", '''    bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
    ratio = (rho / cfg.rho0).clamp_min(0.01)
    return (bulk * (ratio.pow(cfg.gamma) - 1.0)).clamp_min(0.0)


def tait_density(pressure: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Inverse of the Tait law: the density that carries ``pressure``.

    Used wherever a pressure is *extrapolated* onto a particle whose summation
    density does not belong to it -- wall particles below, and the free-surface
    shell in :func:`shell_pressure_extrapolation`. Pairing an extrapolated
    pressure with the (under-counted) summation density would put a spurious
    ``1/rho^2`` factor into the pressure force of exactly the boundary layer the
    closure is meant to fix.
    """
    bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
    return cfg.rho0 * (pressure.clamp_min(0.0) / bulk + 1.0).pow(1.0 / cfg.gamma)


def shell_pressure_extrapolation(pressure, rho, pos, pairs, cfg: Config,
                                 h: float, external=None,
                                 threshold: float | None = None):
    """Extrapolate pressure onto free-surface particles (Adami et al. 2012 §3.3).

    Density summation under-counts neighbours at a free surface, so ``rho``
    falls below ``rho0`` over a shell of thickness ~2h and the clamped Tait law
    assigns ``p = 0`` there. The interior is then bounded by a zero-pressure
    band instead of by the pressure its boundary condition requires, which is
    why a static droplet settles at a fraction of ``sigma/R`` (measured 0.57 +-
    0.06, docs/verification_laplace_pressure_floor.md).

    For every particle with ``rho < threshold * rho0`` this function replaces the
    pressure by the kernel-weighted extrapolation of its neighbours,

        p_i = max(0, sum_j [p_j + rho_j (g - a_i) . (r_i - r_j)] W_ij
                        / sum_j W_ij)

    with ``a_i`` the *non-pressure* acceleration acting on ``i`` (gravity plus
    the continuum surface force, so that the surface-tension jump is carried by
    the body-force term exactly as eq. (11) of Adami et al. 2012 does for a
    wall). ``pairs`` must contain fluid-fluid pairs only.

    Returns ``(pressure, mask)``; ``mask`` marks the particles that were
    closed, so the caller can pair them with :func:`tait_density`.
    """
    n = pressure.shape[0]
    limit = cfg.free_surface_threshold if threshold is None else threshold
    with torch.no_grad():
        mask = rho < limit * cfg.rho0
        if pairs.numel() == 0 or not bool(mask.any()):
            return pressure, mask

        i, j = pairs.unbind(dim=1)
        delta = pos[i] - pos[j]
        wij, _ = pair_kernels(delta, h)
        gravity = torch.zeros_like(pos)
        if pos.shape[1] > 1:
            gravity[:, 1] = -cfg.gravity
        body = gravity if external is None else gravity - external
        source = pressure[j] + rho[j] * (body[i] * delta).sum(dim=1)
        numerator = torch.zeros(n, dtype=pos.dtype,
                                device=pos.device).index_add(
                                    0, i, source * wij)
        denominator = torch.zeros(n, dtype=pos.dtype,
                                  device=pos.device).index_add(0, i, wij)
        extrapolated = (numerator / denominator.clamp_min(1.0e-30)).clamp_min(0.0)
        return torch.where(mask, extrapolated, pressure), mask
''')

# --------------------------------------------- rhs: reorder + closure ------
src = sub(src, """    pressure = tait_pressure(rho, cfg)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
""", """    pressure = tait_pressure(rho, cfg)

    # Surface tension (CSF model) acts on fluid-fluid pairs only. It is
    # evaluated before the pressure because the free-surface pressure closure
    # needs the non-pressure acceleration a_i as an input (Adami et al. 2012,
    # eq. 11 carries g - a_i).
    acc_st_total = torch.zeros((n, pos.shape[1]), dtype=pos.dtype,
                               device=pos.device)
    if cfg.surface_tension_coefficient > 0.0:
        from .surface_tension import compute_surface_tension_force
        fluid_mask = (i < n) & (j < n)
        if fluid_mask.any():
            acc_st_total = compute_surface_tension_force(
                pos, rho, mass, pairs[fluid_mask], cfg, wall=wall)

    density_for_pressure = rho
    if cfg.free_surface_pressure == "shell":
        fluid_pairs = pairs[(i < n) & (j < n)]
        pressure, closed = shell_pressure_extrapolation(
            pressure, rho, pos, fluid_pairs, cfg, h,
            external=acc_st_total + gravity)
        density_for_pressure = torch.where(closed, tait_density(pressure, cfg),
                                          rho)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
""")

# wall branch: the extrapolated wall pressure needs its EOS density; the fluid
# side of the pressure term is now ``density_for_pressure``.
src = sub(src, """        pressure_all = torch.cat((pressure, wall_pressure))
        density_all = torch.cat((rho, wall_density))
    else:
        pressure_all, density_all = pressure, rho
""", """        pressure_all = torch.cat((pressure, wall_pressure))
        density_all = torch.cat((density_for_pressure, wall_density))
    else:
        pressure_all, density_all = pressure, density_for_pressure
    density_visc_all = (torch.cat((rho, wall_density)) if nw else rho)
""")

# viscosity keeps the summation density: the Morris Laplacian needs the volume
src = sub(src, """    visc_i = (-visc_scale * dv
              * (all_mass[j] / density_all[j] * laplace_weight)[:, None])
    visc_j = (visc_scale * dv
              * (all_mass[i] / density_all[i] * laplace_weight)[:, None])
""", """    visc_i = (-visc_scale * dv
              * (all_mass[j] / density_visc_all[j] * laplace_weight)[:, None])
    visc_j = (visc_scale * dv
              * (all_mass[i] / density_visc_all[i] * laplace_weight)[:, None])
""")

# the surface-tension block moved above
src = sub(src, """    # Surface tension (CSF model) - only if coefficient > 0
    acc_st_total = torch.zeros((n, pos.shape[1]), dtype=pos.dtype, device=pos.device)
    if cfg.surface_tension_coefficient > 0.0:
        from .surface_tension import compute_surface_tension_force
        # Use fluid-only pairs for surface tension (wall particles don't have surface)
        fluid_mask = (i < n) & (j < n)
        if fluid_mask.any():
            fluid_pairs = pairs[fluid_mask]
            acc_st_total = compute_surface_tension_force(
                pos, rho, mass, fluid_pairs, cfg, wall=wall)

    return rho, pressure, acc_pressure + acc_visc[:n] + acc_st_total + gravity
""", """    return rho, pressure, acc_pressure + acc_visc[:n] + acc_st_total + gravity
""")

path.write_text(src)
print("patched", len(src.splitlines()), "lines")