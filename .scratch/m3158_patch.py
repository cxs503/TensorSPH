"""m3158: add the Adami/Ihmsen/Adams free-surface pressure closure to rhs."""
from pathlib import Path

SRC = Path("src/tensorsph/droplet_impact.py")
text = SRC.read_text()

# --------------------------------------------------------------------------- #
# 1. Config fields
# --------------------------------------------------------------------------- #
anchor_cfg = '''    csf_curvature: str = "fit_quadratic"


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
'''
cfg_new = '''    csf_curvature: str = "fit_quadratic"
    # Free-surface pressure closure (Adami, Ihmsen & Adams 2012, sec. 3.3).
    # Off by default, so every previously published number stays reproducible.
    # When on, particles whose *sum* density falls below
    # ``free_surface_threshold * rho0`` -- i.e. the one-to-two outer layers where
    # the kernel support is truncated and the density deficit is an artefact --
    # do not take their pressure from the equation of state (which the p >= 0
    # clamp would pin to zero) but from a Shepard-weighted extrapolation of the
    # neighbouring *bulk* pressure.  Without it no pressure can be transmitted
    # through the outer shell, and the drop can only hold a fraction of the
    # Young-Laplace jump sigma/R (measured -34 %, see
    # docs/verification_laplace_pressure_floor.md).
    free_surface_pressure: bool = False
    free_surface_threshold: float = 0.99


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
'''
assert anchor_cfg in text
text = text.replace(anchor_cfg, cfg_new, 1)

# --------------------------------------------------------------------------- #
# 2. the closure itself, inserted before rhs()
# --------------------------------------------------------------------------- #
anchor_fn = '''def rhs(pos, vel, mass, cfg: Config, pairs=None, wall=None, dt=None):
'''
fn = '''def free_surface_pressure_closure(
    pressure: torch.Tensor,
    rho: torch.Tensor,
    delta: torch.Tensor,
    wij: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    n: int,
    cfg: Config,
):
    """Shell-pressure extrapolation for free-surface particles.

    Implements Adami, Ihmsen & Adams (2012), J. Comput. Phys. 231:7057, sec. 3.3
    (also Ihmsen et al. 2014, sec. 4.2).  The particles flagged as free surface
    are those whose *sum* density is deficient by more than
    ``1 - free_surface_threshold``:

        p_i = max(0, sum_bulk (p_j + rho_j g . (x_i - x_j)) W_ij / sum_bulk W_ij)

    Only *bulk* neighbours enter the average: feeding the deficient particles'
    own clamped pressure (p = 0) back into it would dilute the extrapolation
    towards zero exactly where the deficit is largest.  The gravity term is the
    first-order (linear) part of the extrapolation, so a hydrostatic column keeps
    its dp/dy = -rho g slope instead of being flattened to the neighbour mean.
    Particles with no bulk neighbour keep their equation-of-state pressure.

    Returns ``(pressure, free_surface_mask)``; ``pressure`` is unchanged for
    every particle that is not flagged.
    """
    device, dtype = pressure.device, pressure.dtype
    free_surface = rho < cfg.free_surface_threshold * cfg.rho0
    if not bool(free_surface.any()):
        return pressure, free_surface

    fluid = (i < n) & (j < n)
    if not bool(fluid.any()):
        return pressure, free_surface
    ii, jj = i[fluid], j[fluid]
    # Bulk neighbours only (see docstring).
    bulk = ~free_surface[jj]
    if not bool(bulk.any()):
        return pressure, free_surface
    ii, jj = ii[bulk], jj[bulk]
    w = wij[fluid][bulk]
    # g . (x_i - x_j) with g = (0, -gravity) in the (x, y) plane.
    body = -cfg.gravity * delta[fluid][bulk][:, 1]
    source = pressure[jj] + rho[jj] * body

    row = free_surface[ii]
    contrib = torch.where(row, source * w, torch.zeros_like(w))
    weight = torch.where(row, w, torch.zeros_like(w))
    numerator = torch.zeros(n, dtype=dtype, device=device).index_add(0, ii, contrib)
    denominator = torch.zeros(n, dtype=dtype, device=device).index_add(0, ii, weight)

    extrapolated = (numerator / denominator.clamp_min(1.0e-30)).clamp_min(0.0)
    has_support = (denominator > 0.0) & free_surface
    return torch.where(has_support, extrapolated, pressure), free_surface


'''
assert anchor_fn in text
text = text.replace(anchor_fn, fn + anchor_fn, 1)

# --------------------------------------------------------------------------- #
# 3. wire it into rhs
# --------------------------------------------------------------------------- #
anchor_use = '''    pressure = tait_pressure(rho, cfg)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
'''
use_new = '''    pressure = tait_pressure(rho, cfg)
    # Density used by the *momentum* equation.  Identical to the sum density
    # unless the free-surface closure is on, in which case the deficient
    # particles are treated as sitting at the reference density: their deficit
    # is a kernel-completeness artefact, not compression, and letting it into
    # p/rho^2 would add a 1/rho^2 bias on top of the extrapolated pressure.
    rho_mom = rho
    if cfg.free_surface_pressure:
        pressure, free_surface = free_surface_pressure_closure(
            pressure, rho, delta, wij, i, j, n, cfg)
        if bool(free_surface.any()):
            rho_mom = torch.where(free_surface,
                                  torch.full_like(rho, cfg.rho0), rho)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma
'''
assert anchor_use in text
text = text.replace(anchor_use, use_new, 1)

anchor_wall = '''            source = source + torch.cat((rho, zeros))[fluid_idx] * body'''
assert anchor_wall in text
text = text.replace(
    anchor_wall,
    '''            source = source + torch.cat((rho_mom, zeros))[fluid_idx] * body''', 1)

anchor_all = '''        pressure_all, density_all = torch.cat((pressure, wall_pressure)), torch.cat(
            (rho, wall_density))'''
if anchor_all not in text:
    anchor_all = '''        pressure_all, density_all = torch.cat((pressure, wall_pressure)), \\
torch.cat((rho, wall_density))'''
print("wall-all anchor found:", anchor_all in text)

text = text.replace('''        pressure_all = torch.cat((pressure, wall_pressure))
        density_all = torch.cat((rho, wall_density))
    else:
        pressure_all, density_all = pressure, rho''',
'''        pressure_all = torch.cat((pressure, wall_pressure))
        density_all = torch.cat((rho_mom, wall_density))
    else:
        pressure_all, density_all = pressure, rho_mom''', 1)

SRC.write_text(text)

# --------------------------------------------------------------------------- #
# 4. export
# --------------------------------------------------------------------------- #
INIT = Path("src/tensorsph/__init__.py")
it = INIT.read_text()
it = it.replace("    density_from_neighbor_pairs,\n", "    density_from_neighbor_pairs,\n    free_surface_pressure_closure,\n", 1)
it = it.replace('"density_diffusion_rate", "density_from_neighbor_pairs",',
                '"density_diffusion_rate", "density_from_neighbor_pairs",\n    "free_surface_pressure_closure",', 1)
INIT.write_text(it)
print("patched")