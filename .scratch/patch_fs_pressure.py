"""Apply the free-surface pressure closure (P0) to src + benchmark harness."""
from pathlib import Path

SRC = Path("src/tensorsph/droplet_impact.py")
text = SRC.read_text()

# ---------------------------------------------------------------- Config doc
old_doc = '''    scheme, its invariants and what it measured on B1/B2 are documented in
    docs/benchmarks/free-surface-treatment.md.
    """
'''
new_doc = '''    scheme, its invariants and what it measured on B1/B2 are documented in
    docs/benchmarks/free-surface-treatment.md.

    ``free_surface_pressure`` enables the free-surface (shell) pressure closure
    of Adami, Ihmsen & Adams (2012) JCP 231:7057, sec. 3.3. ``tait_pressure``
    clamps ``p >= 0``, so every particle in the density-deficit shell of a drop
    in vacuum sits at ``p = 0`` and the interface cannot hold ``sigma/R``; only
    the interior gradient carries part of the jump. With this switch on, a
    particle whose summation density is below ``free_surface_threshold * rho0``
    takes the kernel-weighted extrapolation of its neighbours' pressure (plus
    the body-force term) whenever that extrapolation exceeds its own EOS value.
    The EOS pressure is never *lowered* here, so interior particles are
    untouched. Off by default: it makes ``p != p(rho)`` for surface particles,
    which is a closure, not part of the equation of state. Measured effect and
    acceptance criteria: docs/verification_free_surface_pressure.md.
    """
'''
assert text.count(old_doc) == 1
text = text.replace(old_doc, new_doc)

# ---------------------------------------------------------------- Config fields
old_field = '''    csf_curvature: str = "fit_quadratic"
'''
new_field = '''    csf_curvature: str = "fit_quadratic"
    # Free-surface (shell) pressure closure, Adami et al. (2012) sec. 3.3.
    free_surface_pressure: bool = False
    # A particle whose summation density is below this fraction of rho0 is a
    # free-surface particle and receives the extrapolated pressure.
    free_surface_threshold: float = 0.99
'''
assert text.count(old_field) == 1
text = text.replace(old_field, new_field)

# ---------------------------------------------------------------- helper
anchor = '''def acoustic_courant(dt: float, cfg: Config, max_speed: float = 0.0) -> float:'''
helper = '''def free_surface_pressure(rho, pressure, i, j, delta, wij, count,
                          cfg: Config, wall_count: int = 0):
    """Shell (free-surface) pressure closure, Adami et al. (2012) sec. 3.3.

    For every fluid particle the kernel-weighted neighbourhood average

        p_i^ext = sum_j (p_j + rho_j (g . r_ji)) W_ij / sum_j W_ij

    is formed over *fluid-fluid* pairs, with ``r_ji = r_i - r_j`` and
    ``g = (0, -gravity)``, and used for particles flagged as free surface
    (``rho_i < free_surface_threshold * rho0``). The result is only accepted
    when it *exceeds* the EOS pressure: the closure supplies the interfacial
    pressure the clamped Tait law cannot carry, it never removes pressure from a
    particle whose density already supports it. Interior particles are returned
    unchanged, so an equilibrium interior is bit-for-bit unaffected.

    ``i``, ``j`` are the unordered pair indices, ``delta = pos[i] - pos[j]`` and
    ``wij`` the pair kernel values; wall particles (indices ``>= count``) are
    excluded from both the sum and the target set. With ``gravity == 0`` the
    body-force term vanishes and the operator reduces to a plain weighted
    average, which is the case the Laplace verification uses.
    """
    if count == 0 or i.numel() == 0:
        return pressure
    fluid_pair = (i < count) & (j < count)
    if not bool(fluid_pair.any()):
        return pressure

    fi = i[fluid_pair]
    fj = j[fluid_pair]
    w = wij[fluid_pair]
    zeros = torch.zeros(count, dtype=pressure.dtype, device=pressure.device)
    source = torch.cat((pressure, torch.zeros(wall_count, dtype=pressure.dtype,
                                             device=pressure.device)))
    source_rho = torch.cat((rho, zeros))

    numerator = torch.zeros_like(zeros)
    denominator = torch.zeros_like(zeros)
    if cfg.gravity != 0.0 and delta.shape[1] > 1:
        d_y = delta[fluid_pair, 1]
        # Pressure rises downward: the source j is lifted to the target i by
        # -g (y_i - y_j) for the i-slot and its mirror for the j-slot.
        weight_i = (source[fj] - cfg.gravity * source_rho[fj] * d_y) * w
        weight_j = (source[fi] + cfg.gravity * source_rho[fi] * d_y) * w
    else:
        weight_i = source[fj] * w
        weight_j = source[fi] * w
    numerator = numerator.index_add(0, fi, weight_i)
    numerator = numerator.index_add(0, fj, weight_j)
    denominator = denominator.index_add(0, fi, w)
    denominator = denominator.index_add(0, fj, w)

    extrapolated = (numerator / denominator.clamp_min(1.0e-30)).clamp_min(0.0)
    free_surface = rho < cfg.free_surface_threshold * cfg.rho0
    use = free_surface & (extrapolated > pressure)
    return torch.where(use, extrapolated, pressure)


'''
assert text.count(anchor) == 1
text = text.replace(anchor, helper + anchor)

# ---------------------------------------------------------------- rhs hook
old_hook = '''    pressure = tait_pressure(rho, cfg)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma'''
new_hook = '''    pressure = tait_pressure(rho, cfg)
    if cfg.free_surface_pressure:
        pressure = free_surface_pressure(rho, pressure, i, j, delta, wij, n, cfg,
                                         wall_count=nw)

    if nw:
        bulk = cfg.rho0 * cfg.sound_speed**2 / cfg.gamma'''
assert text.count(old_hook) == 1
text = text.replace(old_hook, new_hook)

# ---------------------------------------------------------------- dense reference
old_dense = '''    pressure = tait_pressure(rho, cfg)
    grad_w = cubic_spline_gradient(delta, h)'''
new_dense = '''    pressure = tait_pressure(rho, cfg)
    if cfg.free_surface_pressure:
        n = pos.shape[0]
        upper = torch.triu_indices(n, n, offset=1, device=pos.device)
        pressure = free_surface_pressure(
            rho, pressure, upper[0], upper[1],
            pos[upper[0]] - pos[upper[1]], w[upper[0], upper[1]], n, cfg)
    grad_w = cubic_spline_gradient(delta, h)'''
assert text.count(old_dense) == 1
text = text.replace(old_dense, new_dense)

SRC.write_text(text)
print("patched", SRC)

# export in __init__ if the module re-exports names list
INIT = Path("src/tensorsph/__init__.py")
init_text = INIT.read_text()
if "free_surface_pressure" not in init_text and "rhs_all_pairs" in init_text:
    init_text = init_text.replace("rhs_all_pairs", "free_surface_pressure,\n    rhs_all_pairs", 1)
    INIT.write_text(init_text)
    print("patched", INIT)
else:
    print("init exports untouched")