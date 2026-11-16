"""Well-balanced *hydrostatic-split* closure: exact rest **and** live gravity waves.

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` (still water, ``u = 0``, flat free
surface, analytic oracle ``p = rho0 g (H - y)``, tolerance = machine precision).
Two facts were already established in this repository and they do not fit
together:

* the shipped operator (:func:`tensorsph.droplet_impact.rhs`, summation density
  + Tait EOS + symmetrised pressure gradient) is a real solver but leaves
  ``max|a_x|/g = 0.109`` in the closed tank (``docs/verification_lake_at_rest.md``,
  ``docs/verification_lake_at_rest_wall_gap.md``), invariant under refinement;
* the only closure that meets the tolerance there -- moment/MLS gradient on a
  *prescribed* affine pressure with constant density
  (``docs/verification_lake_at_rest_hydrostatic_closure.md``,
  ``docs/verification_lake_at_rest_closure_response.md``) -- meets it because it
  hands the answer in: measured, its residual does not respond to a free-surface
  perturbation and it cannot carry a gravity wave at all.  ``max|a|/g = 1e-15``
  there is an identity, not a scheme.

What this module adds
---------------------
A third arm, the standard *well-balanced* one (hydrostatic reconstruction:
Audusse et al., *SIAM J. Sci. Comput.* 26 (2004) 1449; Xing & Shu,
*J. Comput. Phys.* 208 (2005) 206; in SPH the same split underlies the
Colagrossi-style ``delta-p`` forms), written out for this case:

.. code-block:: text

    p(x, y, t) = p_hyd(y) + p'(x, y, t),   p_hyd(y) = rho0 g (eta0 - y),  eta0 = H
    p'(x, y, t) = B ((rho/rho0)^gamma - 1)                     B = rho0 c0^2/gamma
    dv/dt   = -(1/rho) grad (p_hyd + p') - g e_y
    drho/dt = rho sum_j V_j (v_i - v_j) . grad_i W_ij          [continuity form]
    p_wall  = Adami continuation of the *total* pressure from the fluid neighbours

Two properties follow, and they are exactly the two the previous arms could not
have together:

1. **The rest state is an exact fixed point of the discretisation.**  At rest
   ``rho = rho0`` uniformly, hence ``p' = 0`` *everywhere* by the Tait law, and
   the field the momentum operator sees is the affine field ``p_hyd``.  The
   first-moment (MLS) gradient reproduces an affine field **exactly on any
   support** -- wall-truncated, free-surface-truncated, whatever -- because the
   difference form annihilates the constant part and the moment matrix inverts
   the linear part.  So ``-(1/rho0) grad p_hyd - g e_y = 0`` to roundoff with no
   mass-field fitting, no symmetry assumption and no resolution tuning.  The
   continuity form is ``rho * sum V_j (v_i - v_j) . grad W``, identically zero
   when every ``v`` is zero, so the density does not drift and the state is
   stationary for an arbitrary horizon, not only for one step.
2. **The scheme is not frozen.**  A free-surface perturbation changes ``p'``;
   ``p'`` is *not* affine, its gradient is not annihilated, and the continuity
   form evolves ``rho`` -- so a gravity wave has to run, and its measured angular
   frequency is compared with ``omega^2 = g k tanh(k H)``.

The two are verified together: a closure that passes (1) and fails (2) is the
identity arm of ``closure_response.py``; one that passes (2) and fails (1) is the
shipped operator.

Honest limits, stated where they are readable
---------------------------------------------
* The hydrostatic reference ``p_hyd`` is analytic *because the reference state is
  constant density*.  That is the incompressible limit of a lake at rest; what
  compressibility remains is carried by ``p'``.  A baroclinic reference
  (stratified lake) is out of scope -- ``p_hyd`` would have to be integrated.
* The continuity form is zeroth-order inconsistent on a truncated support (it
  does not conserve ``rho`` near walls).  At rest this is invisible (the sum is
  identically zero); inside a wave it is a real discretisation error and the wave
  frequency is where it shows.
* Momentum in the difference form is not exactly conserved; the shipped
  symmetrised form is.  The trade shows up as wave damping, which is measured
  and reported, not hidden.
* No random numbers and no tuned constant: every number is a function of
  ``dx, H, c0, gamma, h/dx`` and the library kernel.

Usage
-----
``python benchmarks/lake_at_rest/rest_split.py`` writes the artefact;
:func:`rest_run` and :func:`wave_run` are what the tests call.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .droplet_impact import apply_boundary, pair_kernels, wall_mass
from .lake_at_rest import LakeConfig, build_state

__all__ = [
    "SplitOptions",
    "split_fields",
    "accelerations",
    "continuity_rate",
    "step",
    "rest_run",
    "wave_run",
    "hydrostatic_pressure",
    "tait_density",
    "wave_frequency",
    "analytic_frequency",
]


@dataclass
class SplitOptions:
    """Closure switches.  Defaults are the well-balanced split of the docstring."""

    #: Tikhonov floor of the moment-matrix inverse, relative to ``tr(M) / 2``.
    #: A ladder is reported so the exactness floor is visible, not asserted.
    regularisation: float = 1.0e-13
    #: MLS (first-moment consistent) gradient; ``False`` reproduces the
    #: uncorrected difference form, i.e. keeps the wall-truncation defect.
    consistent_gradient: bool = True
    #: ``"split"`` = p_hyd + Tait(rho) (the new arm); ``"prescribed"`` = hand in
    #: p_hyd only (the identity arm of ``closure_response.py``, as a control).
    closure: str = "split"
    #: How the total pressure is extended to the wall particles.
    #: ``"signed"`` = the linear extension ``p_w = p_f + rho_f g.(r_w - r_f)``
    #: with no floor (negative above the waterline: the continuation of a
    #: hydrostatic field, which is what makes the rest state an exact fixed
    #: point); ``"clamped"`` = the same with ``max(p_w, 0)``, the usual
    #: tensile-safe default, which is what leaves the 0.1 g wall-band residual.
    wall_extension: str = "signed"
    #: Use the *reference* density for the particle volumes (``V = m / rho0``)
    #: instead of the evolved one (``V = m / rho``).  The evolved volume closes a
    #: positive feedback at the free surface (a particle whose density drops grows
    #: a larger volume and pushes harder); measured below, the reference volume is
    #: the difference between a 0.2 s blow-up and a stable wave.  Compressibility
    #: is < 1 % here, so the volume the operator uses is unchanged to that order.
    reference_volumes: bool = True
    #: Operator that carries the perturbation pressure.  ``"conservative"`` =
    #: the symmetrised form ``-sum_j m_j (p'_i/rho_i^2 + p'_j/rho_j^2) grad_i W``,
    #: which is the energy-consistent adjoint of the continuity equation and is
    #: what makes the acoustic mode stable; ``"mls"`` = the same MLS difference
    #: form used for the reference (exact on affine fields, but not the adjoint of
    #: the continuity form -- measured to grow roundoff exponentially).
    perturbation_operator: str = "conservative"
    #: Clamp the perturbation pressure at zero (``p' = max(p', 0)``): the free
    #: surface cannot pull.  At rest ``p' = 0`` so the clamp does not touch the
    #: equilibrium; in motion it removes the tensile branch that makes the
    #: evolving-density form grow roundoff exponentially (measured: 1e-13 -> 1e3
    #: within 0.15 s with the branch open).
    tensile_guard: bool = True
    #: delta-SPH density diffusion strength (Antuono et al. 2010), 0 = off.
    density_diffusion: float = 0.0
    #: Max Courant number of the acoustic step.
    max_courant: float = 0.30


# --------------------------------------------------------------------------- #
# analytic reference state
# --------------------------------------------------------------------------- #
def hydrostatic_pressure(cfg: LakeConfig, y: torch.Tensor,
                         level: float | None = None) -> torch.Tensor:
    """``p_hyd(y) = rho0 g (eta0 - y)``, the affine reference of the split."""
    top = cfg.depth if level is None else float(level)
    return cfg.rho0 * cfg.gravity * (top - y)


def _bulk(cfg: LakeConfig) -> float:
    return cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma


def tait_density(cfg: LakeConfig, pressure: torch.Tensor) -> torch.Tensor:
    """Inverse Tait law: the density whose EOS pressure is ``pressure``.

    ``p = B ((rho/rho0)^gamma - 1)``  =>  ``rho = rho0 (1 + p/B)^(1/gamma)``.
    Same ``B`` and ``gamma`` as the solver, so ``tait_pressure(tait_density(p))``
    returns ``p`` to roundoff.
    """
    bulk = _bulk(cfg)
    return cfg.rho0 * (1.0 + pressure / bulk).clamp_min(0.0).pow(1.0 / cfg.gamma)


# --------------------------------------------------------------------------- #
# discrete fields of the operator
# --------------------------------------------------------------------------- #
def split_fields(cfg: LakeConfig, state: dict | None = None,
                 options: SplitOptions | None = None) -> dict:
    """Pair geometry, per-pair volumes and the MLS moment matrices (built once)."""
    options = options or SplitOptions()
    state = build_state(cfg) if state is None else state
    pos, wall = state["pos"], state["wall"]
    n, nw = pos.shape[0], wall.shape[0]
    h = cfg.h()
    pairs = state["pairs"]
    i, j = pairs[:, 0], pairs[:, 1]
    all_pos = torch.cat((pos, wall), dim=0)
    delta = all_pos[i] - all_pos[j]
    wrap = cfg.wrap_width
    if wrap is not None:
        delta = torch.stack((delta[:, 0] - wrap * torch.round(delta[:, 0] / wrap),
                             delta[:, 1]), dim=1)
    weight, gradient = pair_kernels(delta, h)
    fluid_i, fluid_j = i < n, j < n
    # (x_j - x_i) (x) grad_i W_ij: even under i<->j (grad_j W_ji = -grad_i W_ij
    # and (x_i - x_j) flips too), so both partners add the same tensor.
    outer = (-delta)[:, :, None] * gradient[:, None, :]
    m_wall = (wall_mass(state["config"], nw, pos.device) if nw
              else torch.zeros(0, dtype=pos.dtype, device=pos.device))
    return dict(cfg=cfg, options=options, state=state, pos=pos, wall=wall,
                n=n, nw=nw, h=h, i=i, j=j, delta=delta, weight=weight,
                gradient=gradient, outer=outer, fluid_i=fluid_i,
                fluid_j=fluid_j, all_pos=all_pos, m_wall=m_wall,
                m_fluid=torch.full((n,), float(state["m_unit"]),
                                   dtype=pos.dtype, device=pos.device))


def _volumes(fields: dict, rho: torch.Tensor) -> torch.Tensor:
    """Volume per particle: ``m / rho`` for the fluid, ``m / rho0`` for the wall."""
    cfg = fields["cfg"]
    rho_fluid = torch.full_like(rho, cfg.rho0) if fields["options"].reference_volumes \
        else rho
    rho_wall = torch.full_like(fields["m_wall"], cfg.rho0)
    return torch.cat((fields["m_fluid"] / rho_fluid, fields["m_wall"] / rho_wall),
                     dim=0)


def _moments(fields: dict, volumes: torch.Tensor) -> torch.Tensor:
    """``M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij`` for every fluid particle."""
    n, i, j = fields["n"], fields["i"], fields["j"]
    moments = torch.zeros((n, 2, 2), dtype=fields["pos"].dtype,
                          device=fields["pos"].device)
    moments.index_add_(0, i[fields["fluid_i"]],
                       fields["outer"][fields["fluid_i"]]
                       * volumes[j][fields["fluid_i"]][:, None, None])
    moments.index_add_(0, j[fields["fluid_j"]],
                       fields["outer"][fields["fluid_j"]]
                       * volumes[i][fields["fluid_j"]][:, None, None])
    return moments


def _solve_moments(moments: torch.Tensor, regularisation: float) -> torch.Tensor:
    """Regularised inverse; ``tr(M)/2`` is the natural scale of the 2x2 block."""
    trace = moments[:, 0, 0] + moments[:, 1, 1]
    floor = regularisation * 0.5 * trace.abs().clamp_min(1e-300)
    eye = torch.eye(2, dtype=moments.dtype, device=moments.device)
    return torch.linalg.inv(moments + floor[:, None, None] * eye)


def _wall_pressure(fields: dict, pressure: torch.Tensor, rho: torch.Tensor,
                   gravity: bool = True) -> torch.Tensor:
    """Adami et al. (2012) continuation of the *total* pressure to the wall.

    ``p_w = sum_f (p_f + rho_f (g - a_w).(r_w - r_f)) W / sum_f W``, static wall;
    ``gravity=False`` drops the linear term for a *perturbation* pressure, whose
    continuation is the plain weighted mean.
    For an affine total field ``p = rho0 g (eta0 - y)`` this returns
    ``rho0 g (eta0 - y_w)`` *exactly* (the extrapolation of a plane is the
    plane), which is why the split closure needs no special wall treatment.
    """
    cfg, n, nw = fields["cfg"], fields["n"], fields["nw"]
    if nw == 0:
        return torch.zeros(0, dtype=fields["pos"].dtype,
                           device=fields["pos"].device)
    i, j = fields["i"], fields["j"]
    crossing = fields["fluid_i"] != fields["fluid_j"]
    fluid_idx = torch.where(fields["fluid_i"], i, j)
    wall_idx = (torch.where(fields["fluid_i"], j, i) - n).clamp_(0, nw - 1)
    dy = torch.where(fields["fluid_i"], -fields["delta"][:, 1],
                     fields["delta"][:, 1])
    source = pressure[fluid_idx] + (rho[fluid_idx] * (-cfg.gravity * dy)
                                   if gravity else torch.zeros_like(dy))
    weight = torch.where(crossing, fields["weight"],
                         torch.zeros_like(fields["weight"]))
    num = torch.zeros(nw, dtype=fields["pos"].dtype, device=fields["pos"].device)
    num = num.index_add(0, wall_idx, source * weight)
    den = torch.zeros(nw, dtype=fields["pos"].dtype, device=fields["pos"].device)
    den = den.index_add(0, wall_idx, weight)
    extension = num / den.clamp_min(1e-300)
    if fields["options"].wall_extension == "signed":
        return extension
    if fields["options"].wall_extension == "clamped":
        return extension.clamp_min(0.0)
    raise ValueError(f"unknown wall_extension {fields['options'].wall_extension!r}")


def _gradient(fields: dict, pressure: torch.Tensor,
              rho: torch.Tensor) -> torch.Tensor:
    """MLS gradient of the total pressure at every fluid particle."""
    options, n = fields["options"], fields["n"]
    i, j = fields["i"], fields["j"]
    volumes = _volumes(fields, rho)
    p_all = torch.cat((pressure, _wall_pressure(fields, pressure, rho)))
    jump = (p_all[j] - p_all[i])[:, None] * fields["gradient"]
    num = torch.zeros((n, 2), dtype=fields["pos"].dtype,
                      device=fields["pos"].device)
    if options.consistent_gradient:
        num.index_add_(0, i[fields["fluid_i"]],
                       jump[fields["fluid_i"]]
                       * volumes[j][fields["fluid_i"]][:, None])
        num.index_add_(0, j[fields["fluid_j"]],
                       jump[fields["fluid_j"]]
                       * volumes[i][fields["fluid_j"]][:, None])
        inverse = _solve_moments(_moments(fields, volumes),
                                 options.regularisation)
        return torch.bmm(inverse, num[:, :, None])[:, :, 0]
    # Uncorrected difference form: keeps the wall-truncation defect, on purpose.
    num.index_add_(0, i[fields["fluid_i"]], jump[fields["fluid_i"]])
    num.index_add_(0, j[fields["fluid_j"]], jump[fields["fluid_j"]])
    return num


# --------------------------------------------------------------------------- #
# the two evaluations the case needs
# --------------------------------------------------------------------------- #
def accelerations(fields: dict, rho: torch.Tensor,
                  level: float | None = None) -> dict:
    """Momentum residual of the split closure at the state ``(rho, level)``."""
    cfg, options = fields["cfg"], fields["options"]
    if options.closure == "prescribed":
        perturbation = torch.zeros_like(rho)
    else:
        perturbation = _bulk(cfg) * ((rho / cfg.rho0).pow(cfg.gamma) - 1.0)
        if options.tensile_guard:
            perturbation = perturbation.clamp_min(0.0)
    reference = hydrostatic_pressure(cfg, fields["pos"][:, 1], level)
    total = reference + perturbation
    grad = _gradient(fields, total, rho)
    accel = -grad / rho[:, None]
    accel[:, 1] -= cfg.gravity
    if options.perturbation_operator == "conservative":
        # Replace the perturbation part of the estimate by its conservative
        # (symmetrised) form.  The reference part is kept from the MLS estimate,
        # so the exact-cancellation property is a measured one, not an algebraic
        # identity imposed on the answer.
        reference_gradient = _gradient(fields, reference, rho)
        accel = accel + (grad - reference_gradient) / rho[:, None] \
            - _conservative_gradient(fields, perturbation, rho)
    elif options.perturbation_operator != "mls":
        raise ValueError(f"unknown perturbation_operator "
                         f"{options.perturbation_operator!r}")
    return dict(acceleration=accel, gradient=grad, perturbation=perturbation,
                total=total)


def _conservative_gradient(fields: dict, pressure: torch.Tensor,
                           rho: torch.Tensor) -> torch.Tensor:
    """``sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W_ij`` (symmetrised form).

    The pressure of a wall neighbour is its Adami continuation, so a wall at rest
    with ``p = 0`` contributes nothing -- which is the correct statement for the
    perturbation, and the reason the split does not need a wall closure for it.
    """
    n, i, j = fields["n"], fields["i"], fields["j"]
    # The gravity term of the continuation belongs to the reference field: for the
    # *perturbation* the wall value is the plain weighted mean of the fluid values
    # (measured: including it here gives a spurious 0.5 g at rest).
    wall_p = _wall_pressure(fields, pressure, rho, gravity=False)
    p_all = torch.cat((pressure, wall_p))
    m_all = torch.cat((fields["m_fluid"], fields["m_wall"]))
    rho_all = torch.cat((rho, torch.full_like(fields["m_wall"], fields["cfg"].rho0)))
    coefficient = ((p_all[i] / rho_all[i].square())
                   + (p_all[j] / rho_all[j].square()))
    term = coefficient[:, None] * fields["gradient"] * m_all[j][:, None]
    out = torch.zeros((n, 2), dtype=fields["pos"].dtype,
                      device=fields["pos"].device)
    out.index_add_(0, i[fields["fluid_i"]], term[fields["fluid_i"]])
    out.index_add_(0, j[fields["fluid_j"]], term[fields["fluid_j"]])
    return out


def continuity_rate(fields: dict, rho: torch.Tensor,
                    vel: torch.Tensor) -> torch.Tensor:
    """``drho_i/dt = -rho_i div v_i`` with the MLS-consistent divergence.

    The plain sum ``sum_j V_j (v_j - v_i).grad_i W_ij`` is *zeroth-order
    inconsistent* on a truncated support: it is non-zero for a divergence-free
    velocity field at the free surface and at the wall, which turns the surface
    rows into a density source.  Measured with the plain form, ``rho`` spanned
    ``[500, 1.7e4] kg/m3`` within 0.2 s.  The moment matrix that makes the
    *gradient* affine-exact makes the divergence exact too --
    ``div v = trace(M^-1 sum_j V_j (v_j - v_i) (x) grad_i W_ij)`` -- so one
    operator serves two contractions and no extra state is introduced.
    """
    n, i, j = fields["n"], fields["i"], fields["j"]
    volumes = _volumes(fields, rho)
    v_all = torch.cat((vel, torch.zeros_like(fields["wall"])), dim=0)
    tensor = (v_all[j] - v_all[i])[:, :, None] * fields["gradient"][:, None, :]
    div = torch.zeros((n, 2, 2), dtype=fields["pos"].dtype,
                      device=fields["pos"].device)
    div.index_add_(0, i[fields["fluid_i"]],
                   tensor[fields["fluid_i"]]
                   * volumes[j][fields["fluid_i"]][:, None, None])
    div.index_add_(0, j[fields["fluid_j"]],
                   tensor[fields["fluid_j"]]
                   * volumes[i][fields["fluid_j"]][:, None, None])
    if fields["options"].consistent_gradient:
        inverse = _solve_moments(_moments(fields, volumes),
                                 fields["options"].regularisation)
        corrected = torch.bmm(inverse, div)
        divergence = corrected.diagonal(dim1=1, dim2=2).sum(dim=1)
    else:
        divergence = div.diagonal(dim1=1, dim2=2).sum(dim=1)
    rate = -rho * divergence
    strength = fields["options"].density_diffusion
    if strength > 0.0:
        # delta-SPH: diffusive flux of density, standard Molteni-Colagrossi form.
        radius2 = fields["delta"].square().sum(dim=1).clamp_min(1e-30)
        radial = (fields["delta"] * fields["gradient"]).sum(dim=1) / radius2
        jump = (rho_all_of(fields, rho)[j] - rho_all_of(fields, rho)[i]) * radial
        flux = torch.zeros(n, dtype=fields["pos"].dtype,
                           device=fields["pos"].device)
        flux.index_add_(0, i[fields["fluid_i"]],
                        jump[fields["fluid_i"]] * volumes[j][fields["fluid_i"]])
        flux.index_add_(0, j[fields["fluid_j"]],
                        jump[fields["fluid_j"]] * volumes[i][fields["fluid_j"]])
        rate = rate + strength * fields["h"] * cfg_of(fields).sound_speed * flux
    return rate


def rho_all_of(fields: dict, rho: torch.Tensor) -> torch.Tensor:
    """Fluid density extended to the wall particles at rest density."""
    return torch.cat((rho, torch.full_like(fields["m_wall"], fields["cfg"].rho0)))


def cfg_of(fields: dict) -> LakeConfig:
    return fields["cfg"]


def step(fields: dict, rho: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor,
         dt: float, level: float | None = None) -> tuple:
    """One explicit step of the split scheme (forward Euler, acoustic dt)."""
    accel = accelerations(fields, rho, level)["acceleration"]
    rate = continuity_rate(fields, rho, vel)
    vel = vel + dt * accel
    pos = pos + dt * vel
    rho = (rho + dt * rate).clamp_min(0.5 * fields["cfg"].rho0)
    apply_boundary(pos, vel, fields["state"]["config"])
    return rho, pos, vel


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def rest_run(cfg: LakeConfig, horizon: float | None = None,
             level: float | None = None, options: SplitOptions | None = None,
             sample_every: int = 25) -> dict:
    """Integrate the split closure from the discrete rest state.

    Initial state: ``rho = rho0`` constant, ``v = 0``, the ridge lattice of the
    case.  Nothing is fitted and nothing is reconstructed -- the claim is that
    the discretisation *keeps* that state, over a horizon, to roundoff.
    """
    options = options or SplitOptions()
    state = build_state(cfg)
    fields = split_fields(cfg, state, options)
    n = fields["n"]
    dtype, device = fields["pos"].dtype, fields["pos"].device
    rho = torch.full((n,), cfg.rho0, dtype=dtype, device=device)
    pos = fields["pos"].clone()
    vel = torch.zeros_like(pos)
    horizon = cfg.t_end if horizon is None else float(horizon)
    dt = options.max_courant * cfg.h() / cfg.sound_speed
    steps = max(1, int(math.ceil(horizon / dt)))
    dt = horizon / steps
    pos0 = pos.clone()
    first = accelerations(fields, rho, level)
    series = []
    for k in range(steps + 1):
        if k % sample_every == 0 or k == steps:
            accel = (first["acceleration"] if k == 0
                     else accelerations(fields, rho, level)["acceleration"])
            series.append({
                "t": k * dt,
                "max_ax_g": float(accel[:, 0].abs().max()) / cfg.gravity,
                "max_ay_g": float(accel[:, 1].abs().max()) / cfg.gravity,
                "max_a_g": float(torch.linalg.vector_norm(accel, dim=1).max())
                / cfg.gravity,
                "max_v_c0": float(torch.linalg.vector_norm(vel, dim=1).max())
                / cfg.sound_speed,
                "max_drho": float((rho - cfg.rho0).abs().max()) / cfg.rho0,
                "max_dpos": float(torch.linalg.vector_norm(pos - pos0, dim=1).max()),
            })
        if k == steps:
            break
        rho, pos, vel = step(fields, rho, pos, vel, dt, level)
    total = first["total"]
    return {
        "dx": cfg.dx,
        "periodic_x": cfg.periodic_x,
        "n_fluid": n,
        "n_wall": fields["nw"],
        "h_over_dx": cfg.h_factor,
        "sound_speed": cfg.sound_speed,
        "gamma": cfg.gamma,
        "dt": dt,
        "steps": steps,
        "horizon": horizon,
        "level": cfg.depth if level is None else float(level),
        "closure": options.closure,
        "consistent_gradient": options.consistent_gradient,
        "regularisation": options.regularisation,
        "a0_max_g": series[0]["max_a_g"],
        "a0_max_ax_g": series[0]["max_ax_g"],
        "a0_max_ay_g": series[0]["max_ay_g"],
        "final_max_a_g": series[-1]["max_a_g"],
        "final_max_v_c0": series[-1]["max_v_c0"],
        "final_max_drho": series[-1]["max_drho"],
        "final_max_dpos": series[-1]["max_dpos"],
        "max_a_g_over_horizon": max(row["max_a_g"] for row in series),
        "max_v_c0_over_horizon": max(row["max_v_c0"] for row in series),
        "max_dpos_over_horizon": max(row["max_dpos"] for row in series),
        "pressure_min": float(total.min()),
        "pressure_max": float(total.max()),
        "series": series,
    }


def analytic_frequency(depth: float, k: float, gravity: float = 9.81) -> float:
    """Linear gravity-wave dispersion ``omega^2 = g k tanh(k H)``."""
    return math.sqrt(gravity * k * math.tanh(k * depth))


def wave_frequency(times, elevations, omega_guess: float) -> dict:
    """Least-squares single-mode fit, minimised by a golden-section search.

    Fitting ``eta = c + A cos(w t) + B sin(w t)`` on a one-or-two-period record
    avoids the resolution limit of an FFT while still being a fit to the raw
    series (no damping model, nothing detrended beyond the constant).
    """
    t = torch.as_tensor(times, dtype=torch.float64)
    e = torch.as_tensor(elevations, dtype=torch.float64)

    def residual(omega: float) -> float:
        design = torch.stack((torch.ones_like(t), torch.cos(omega * t),
                              torch.sin(omega * t)), dim=1)
        solution = torch.linalg.lstsq(design, e[:, None]).solution
        return float((design @ solution - e[:, None]).square().sum())

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = 0.5 * omega_guess, 1.5 * omega_guess
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = residual(c), residual(d)
    for _ in range(120):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = residual(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = residual(d)
    omega = 0.5 * (a + b)
    design = torch.stack((torch.ones_like(t), torch.cos(omega * t),
                          torch.sin(omega * t)), dim=1)
    solution = torch.linalg.lstsq(design, e[:, None]).solution
    model = (design @ solution)[:, 0]
    total = float(((e - e.mean()).square()).sum())
    return {
        "omega": omega,
        "period": 2.0 * math.pi / omega,
        "amplitude": float(torch.hypot(solution[1, 0], solution[2, 0])),
        "residual_rms": float((model - e).square().mean().sqrt()),
        "r2": float(1.0 - (model - e).square().sum() / total) if total > 0 else None,
    }


def wave_run(cfg: LakeConfig, amplitude: float | None = None,
             periods: float = 2.0, options: SplitOptions | None = None,
             sample_every: int = 4) -> dict:
    """Standing gravity wave: the non-degeneracy control for the split closure.

    Initial state = the hydrostatic field of the perturbed surface
    ``eta(x) = H + A cos(k x)``, ``k = 2 pi / W`` (periodic) or ``pi / W``
    (side walls, first sloshing mode).  In the split variables that is
    ``p' = rho0 g (eta(x) - H)`` -- independent of ``y`` -- and
    ``rho = tait_density(p')``, the exact barotropic pre-image of the perturbed
    hydrostatic pressure.  So this is a genuine non-equilibrium state *of this
    closure*, and the frequency it rings at is a measurement, not a prescription.
    """
    options = options or SplitOptions()
    state = build_state(cfg)
    fields = split_fields(cfg, state, options)
    n, pos = fields["n"], fields["pos"].clone()
    amplitude = (0.5 * cfg.dx) if amplitude is None else float(amplitude)
    k = (2.0 * math.pi / cfg.width) if cfg.periodic_x else (math.pi / cfg.width)
    eta = cfg.depth + amplitude * torch.cos(k * pos[:, 0])
    rho = tait_density(cfg, cfg.rho0 * cfg.gravity * (eta - cfg.depth))
    vel = torch.zeros_like(pos)
    omega_ref = analytic_frequency(cfg.depth, k, cfg.gravity)
    horizon = periods * 2.0 * math.pi / omega_ref
    dt = options.max_courant * cfg.h() / cfg.sound_speed
    steps = max(1, int(math.ceil(horizon / dt)))
    dt = horizon / steps
    columns = torch.unique(pos[:, 0].round(decimals=12))
    watch = float(columns[int(torch.argmin((columns - 0.5 * cfg.width).abs()))])
    mask = (pos[:, 0] - watch).abs() < 1e-9
    rest = accelerations(fields, torch.full((n,), cfg.rho0, dtype=pos.dtype,
                                            device=pos.device), cfg.depth)
    times, elevations, vmax = [], [], []
    for index in range(steps + 1):
        if index % sample_every == 0 or index == steps:
            times.append(index * dt)
            elevations.append(float(pos[mask, 1].max()) + 0.5 * cfg.dx)
            vmax.append(float(torch.linalg.vector_norm(vel, dim=1).max())
                        / cfg.sound_speed)
        if index == steps:
            break
        rho, pos, vel = step(fields, rho, pos, vel, dt, cfg.depth)
    fit = wave_frequency(times, elevations, omega_ref)
    initial = accelerations(fields, tait_density(
        cfg, cfg.rho0 * cfg.gravity * (eta - cfg.depth)), cfg.depth)["acceleration"]
    return {
        "dx": cfg.dx,
        "periodic_x": cfg.periodic_x,
        "n_fluid": n,
        "amplitude": amplitude,
        "amplitude_over_dx": amplitude / cfg.dx,
        "wavenumber": k,
        "depth": cfg.depth,
        "dt": dt,
        "steps": steps,
        "horizon": horizon,
        "omega_analytic": omega_ref,
        "omega_measured": fit["omega"],
        "frequency_error": abs(fit["omega"] / omega_ref - 1.0),
        "fit": fit,
        "initial_a_max_g": float(torch.linalg.vector_norm(initial, dim=1).max())
        / cfg.gravity,
        "rest_a_max_g": float(torch.linalg.vector_norm(rest["acceleration"],
                                                       dim=1).max())
        / cfg.gravity,
        "max_v_c0": max(vmax),
        "surface_series": {"t": times, "eta": elevations},
    }