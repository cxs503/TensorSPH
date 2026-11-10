"""Three-field coupled column: temperature / salinity / thermal stress.

This is the integration face of the package -- the place where the three fields
of the campaign node actually talk to each other, through four links:

1. ``T -> phase``  the enthalpy inversion gives the brine-filled pore fraction
   ``v_b(T, S)`` -- the mushy layer;
2. ``phase -> S``  ``v_b`` sets the permeability and hence the gravity-drainage
   flux of brine (:mod:`tensorsph.ice_thermal.brine`);
3. ``S -> ocean -> T``  the rejected brine raises the salinity of the mixed
   layer under the ice, which makes it denser than it was: the convective
   velocity scale ``w* = sqrt(g' h_o)`` with ``g' = g beta_S dS / rho_o`` raises
   the ice-ocean exchange velocity ``gamma = gamma_0 + a w*``, and the ocean
   heat flux ``Q_oc = rho c_p gamma (T_o - Tf)`` melts the base;
4. ``T -> stress``  the through-thickness temperature profile gives the
   thermo-elastic stress and the cracking verdict
   (:mod:`tensorsph.ice_thermal.thermal_stress`).

The loop is closed (two-way) by melt water: it dilutes the mixed layer, which
lowers ``g'``, which lowers ``gamma``, which lowers the melt -- a negative
feedback.  ``two_way=False`` switches that dilution off, turning the run into a
one-way coupling; :func:`coupling_strength` runs both and reports the
difference, which is the *measured* coupling strength rather than an assertion.

Numerics
--------
* thermal conduction: the SPH operator of
  :mod:`tensorsph.ice_thermal.sph_thermal` assembled into a matrix and advanced
  with a **backward-Euler Picard** scheme on the apparent heat capacity.  The
  latent spike of ``c_eff`` near ``Tf`` (``L/theta_f ~ 5e5 J/kg/K``) is what
  makes an explicit scheme unusable (``dt ~ 20 s``); the implicit one runs at
  ``dt = 900 s``;
* salt: implicit upwind drainage (unconditionally stable, exactly salt
  conserving -- only the bottom face leaves the ice);
* boundaries: Dirichlet surface temperature at the top, Dirichlet freezing point
  ``Tf(S_ocean)`` at the ice base, i.e. the ice-ocean interface sits at the
  seawater freezing point; the melt rate follows from the Stefan flux balance
  ``rho_i L dh/dt = Q_oc - q_cond``.

Stated limitations (not hidden)
-------------------------------
* the ice thickness is *diagnosed*, not ablated: the particle column keeps its
  initial thickness, so the base does not move (a moving-boundary Stefan
  condition in SPH is left to a later node).  Melt still enters the ocean
  salinity budget, so the two-way feedback is real, just not geometric;
* the ocean is one well-mixed layer (no resolved plume or convection cell);
* ``gamma_0``, ``a`` are declared bulk exchange coefficients, not fitted to a
  dataset; the verification targets here are the conservation laws, the
  phase-change / drainage anchors, and the sign and size of the coupling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import darcy
from . import properties as props
from . import thermal_stress as stress_mod
from .brine import (V_IRR, drainage_advance, drainage_velocity,
                    mobile_brine_fraction, relative_permeability)
from .enthalpy import EnthalpyModel
from .sph_thermal import DTYPE, pair_geometry

__all__ = ["ThreeFieldConfig", "ThreeFieldResult", "run_three_field",
           "coupling_strength", "salt_equilibrium", "assemble_conduction"]


@dataclass
class ThreeFieldConfig:
    """Configuration of the coupled sea-ice column (default: spring melt onset)."""

    thickness: float = 0.3           # m, ice thickness
    n_z: int = 30                    # particles through the thickness
    salinity0: float = 12.0          # psu, initial bulk salinity
    t_surface: float = 272.15        # K, imposed surface temperature (-1 C, ablation)
    t_surface0: float = 272.65       # K, initial surface temperature (-0.5 C)
    t_ocean_anomaly: float = 0.05    # K, mixed layer above freezing
    ocean_depth: float = 20.0        # m, well-mixed layer
    ocean_salinity0: float = 32.0    # psu
    days: float = 45.0
    dt: float = 3600.0               # s, target step (auto-refined when needed)
    v_irr: float = V_IRR
    gamma_background: float = 1.0e-6
    gamma_convective: float = 3.0e-4
    flaw_depth: float = 0.01         # m, assumed edge-crack depth for K_I
    two_way: bool = True
    picard: int = 3
    picard_tol: float = 1e-6         # K
    history_every: int = 8
    h_factor: float = 1.3
    spinup: bool = True

    @property
    def dz(self) -> float:
        return self.thickness / self.n_z


@dataclass
class ThreeFieldResult:
    """Everything the report quotes, computed once."""

    time_h: list = field(default_factory=list)
    bulk_salinity: list = field(default_factory=list)
    ocean_salinity: list = field(default_factory=list)
    ocean_heat_flux: list = field(default_factory=list)
    convective_velocity: list = field(default_factory=list)
    melt_rate_mm_day: list = field(default_factory=list)
    cumulative_melt_m: list = field(default_factory=list)
    max_tensile_mpa: list = field(default_factory=list)
    surface_temperature_c: list = field(default_factory=list)
    z: list = field(default_factory=list)
    temperature: list = field(default_factory=list)
    salinity: list = field(default_factory=list)
    brine_fraction: list = field(default_factory=list)
    permeability: list = field(default_factory=list)
    stress_mpa: list = field(default_factory=list)
    stress_free_mpa: list = field(default_factory=list)
    residual_salinity: list = field(default_factory=list)
    salt_error: float = 0.0
    salt_initial: float = 0.0
    salt_final: float = 0.0
    heat_integral_j_m2: float = 0.0
    melt_total_m: float = 0.0
    crack: object = None
    crack_free: object = None
    crack_time_days: float | None = None
    spinup_relaxation: dict = field(default_factory=dict)
    config: ThreeFieldConfig = field(default_factory=ThreeFieldConfig)

    def summary(self) -> dict:
        """Small JSON-safe digest used by the benchmark and the report."""
        out = {
            "bulk_salinity_final_psu": float(self.bulk_salinity[-1]),
            "ocean_salinity_final_psu": float(self.ocean_salinity[-1]),
            "residual_salinity_final_psu": float(self.residual_salinity[-1]),
            "ocean_heat_flux_w_m2": float(self.ocean_heat_flux[-1]),
            "convective_velocity_m_s": float(self.convective_velocity[-1]),
            "melt_total_m": float(self.melt_total_m),
            "melt_rate_mm_day_mean": float(sum(self.melt_rate_mm_day)
                                           / max(len(self.melt_rate_mm_day), 1)),
            "max_tensile_stress_mpa_restrained": float(max(self.max_tensile_mpa)),
            "max_tensile_stress_mpa_free": float(max(map(abs, self.stress_free_mpa))),
            "salt_conservation_rel_error": float(self.salt_error),
            "salt_exported_g_m2": float(self.salt_exported),
            "crack_time_days_restrained": self.crack_time_days,
            "spinup": self.spinup_relaxation,
        }
        for tag, crack in (("restrained", self.crack), ("free", self.crack_free)):
            out[f"crack_{tag}"] = None if crack is None else {
                "strength_ratio": crack.strength_ratio,
                "fracture_ratio": crack.fracture_ratio,
                "cracks": crack.cracks,
                "critical_depth_m": crack.critical_depth_m,
            }
        return out


def assemble_conduction(pos, dx, h, conductivity, t_top, t_bottom):
    """Matrix form of the SPH conduction operator with mirror (Dirichlet) ghosts.

    Returns ``(A, b)`` such that ``dH/dt = A @ T + b`` for the real particles.
    The ghost temperatures are ``T_ghost = 2 T_wall - T_mirror``; they are folded
    analytically into the matrix (the term on the mirrored real particle) and the
    right-hand side (``-2 T_wall``), so no ghost unknown is carried.
    """
    n = pos.numel()
    length = float(pos[-1] + 0.5 * dx)
    pos_ext = torch.cat([-pos.flip(0), pos, 2.0 * length - pos.flip(0)])
    volume_ext = torch.full((3 * n,), dx, dtype=DTYPE)
    fv = pair_geometry(pos_ext, volume_ext, h)
    k_ext = torch.cat([conductivity.flip(0), conductivity, conductivity.flip(0)])
    k_ij = 2.0 * k_ext[:, None] * k_ext[None, :] / (k_ext[:, None] + k_ext[None, :])
    rows = (k_ij * fv)[n:2 * n, :]
    a = torch.diag(rows.sum(dim=1)) - rows[:, n:2 * n]
    rev = torch.arange(n - 1, -1, -1)
    top = rows[:, :n]
    bot = rows[:, 2 * n:]
    a[:, rev] += top + bot
    b = -2.0 * (top.sum(dim=1) * float(t_top) + bot.sum(dim=1) * float(t_bottom))
    scale = 2.0 / props.RHO_ICE
    return a * scale, b * scale


def _implicit_step(h_field, salinity, model, cfg, t_top, t_bottom):
    """One backward-Euler Picard step of the enthalpy column; returns ``(H, T)``."""
    n, dx = cfg.n_z, cfg.dz
    pos = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    temperature = model.temperature(h_field, salinity)
    a = b = None
    for sweep in range(cfg.picard):
        cap = model.heat_capacity(temperature, salinity).clamp_min(1.0)
        if a is None:   # the conductivity blend moves slowly: assemble once
            w_l = model.liquid_fraction(temperature, salinity)
            k_p = props.K_ICE + w_l * (props.K_FRESH - props.K_ICE)
            a, b = assemble_conduction(pos, dx, cfg.h_factor * dx, k_p, t_top, t_bottom)
        # H is a *specific* enthalpy (J/kg), so dH/dt = c dT/dt -- no density in
        # the capacity term.  Carrying rho here (the trap this line removes)
        # suppresses the temperature response by 917x and freezes the column.
        diag = cap / cfg.dt
        mat = torch.diag(diag) - a
        rhs = (diag * temperature
               + (h_field - model.enthalpy(temperature, salinity)) / cfg.dt + b)
        new = torch.linalg.solve(mat, rhs)
        converged = float((new - temperature).abs().max()) < cfg.picard_tol
        temperature = new
        if converged:
            break
    return model.enthalpy(temperature, salinity), temperature


def salt_equilibrium(temperature, v_irr=V_IRR):
    """Bulk salinity (psu) at which drainage stops at ``temperature``."""
    sb = props.brine_salinity(float(temperature))
    return v_irr * props.brine_density(sb) * sb / props.RHO_ICE


def run_three_field(cfg: ThreeFieldConfig | None = None, model: EnthalpyModel | None = None):
    """Integrate the coupled column and return every diagnostic the report needs."""
    cfg = cfg or ThreeFieldConfig()
    model = model or EnthalpyModel(kind="sea", c_liquid=props.C_SEAWATER)
    n, dz = cfg.n_z, cfg.dz
    z = (torch.arange(n, dtype=DTYPE) + 0.5) * dz
    salinity = torch.full((n,), float(cfg.salinity0), dtype=DTYPE)
    tf0 = props.freezing_point(cfg.salinity0)
    temperature0 = torch.linspace(float(cfg.t_surface0), tf0, n, dtype=DTYPE)
    t_ref = float(temperature0.mean())      # stress-free (formation) temperature
    h_field = model.enthalpy(temperature0, salinity)

    s_ocean = float(cfg.ocean_salinity0)
    melt_total = 0.0
    salt0 = (float((props.RHO_ICE * salinity * dz).sum())
             + props.RHO_SEAWATER_REF * cfg.ocean_depth * s_ocean)
    heat_integral = 0.0
    salt_exported = 0.0
    crack_time = None
    res = ThreeFieldResult(config=cfg)
    t_end = cfg.days * 86400.0
    t = 0.0
    next_record = 0.0
    k = 0
    dt_step = min(cfg.dt, 60.0)   # start small: the initial drainage is the fastest
    while True:
        temperature = model.temperature(h_field, salinity)
        tf_ocean = props.freezing_point(s_ocean)
        # --- drainage (fields 1 & 2) --------------------------------------
        if t < t_end:
            salinity, removed, n_sub = drainage_advance(
                salinity, temperature, dz, dt_step, cfg.v_irr)
            s_ocean += removed / (props.RHO_SEAWATER_REF * cfg.ocean_depth)
            # Brine leaves at the local temperature, so the drainage is an
            # isothermal mass change: the *state* moves along S at fixed T, and
            # the specific enthalpy must be re-projected.  Keeping H fixed while
            # S drops would add the (S-dependent) latent term to the ice and
            # spuriously heat it by ~L per kg of salt removed -- the trap this
            # re-projection removes.
            h_field = model.enthalpy(temperature, salinity)
        # --- buoyancy -> ocean heat flux (field 2 -> 3) --------------------
        d_s = s_ocean - cfg.ocean_salinity0
        g_prime = props.GRAVITY * props.BETA_SALINITY * d_s / props.RHO_SEAWATER_REF
        w_star = math.sqrt(max(g_prime, 0.0) * cfg.ocean_depth)
        gamma = cfg.gamma_background + cfg.gamma_convective * w_star
        q_oc = props.RHO_SEAWATER_REF * props.C_SEAWATER * gamma * cfg.t_ocean_anomaly
        # Heat drawn *up* into the ice at the interface.  The resolved gradient is
        # taken over the outer fifth of the column, not across the last cell: the
        # fixed-thickness column develops a sub-kelvin kink in that cell (the base
        # does not ablate while its salinity drains), and a one-cell gradient
        # turns that kink into a spurious +/-100 W/m^2 interface flux.
        k_grad = max(2, n // 5)
        q_up = (props.K_ICE
                * (float(temperature[-1]) - float(temperature[-1 - k_grad]))
                / (k_grad * dz))
        melt_rate = max(q_oc - q_up, 0.0) / (props.RHO_ICE * props.L_FUSION)
        # the column only has ``thickness`` of ice to lose
        melt_rate = min(melt_rate, max(cfg.thickness - melt_total, 0.0) / dt_step)
        melt_total += melt_rate * dt_step
        heat_integral += q_oc * dt_step
        # --- two-way closure: melt dilutes the mixed layer -----------------
        if cfg.two_way:
            # Only the freshwater fraction dilutes the mixed layer: the column
            # does not ablate, so the melted ice keeps its salt in the ice.  The
            # layer depth is fixed, so diluting it means exporting salt at
            # rho_i melt S_o -- that export is bookkept (salt_exported) and is
            # what makes the salt budget exactly closable.
            dilution = (props.RHO_ICE * melt_rate * s_ocean
                        / (props.RHO_SEAWATER_REF * cfg.ocean_depth)) * dt_step
            s_ocean -= dilution
            salt_exported += props.RHO_SEAWATER_REF * cfg.ocean_depth * dilution
        # --- stress (field 3) ---------------------------------------------
        sigma = stress_mod.restrained_stress(temperature, t_ref=t_ref)
        if crack_time is None and stress_mod.max_tensile_stress(sigma) >= props.SIGMA_TENSILE:
            crack_time = t / 86400.0
        if t >= next_record:
            next_record += cfg.history_every * dt_step
            res.time_h.append(t / 3600.0)
            res.bulk_salinity.append(float(salinity.mean()))
            res.ocean_salinity.append(s_ocean)
            res.ocean_heat_flux.append(q_oc)
            res.convective_velocity.append(w_star)
            res.melt_rate_mm_day.append(melt_rate * 1000.0 * 86400.0)
            res.cumulative_melt_m.append(melt_total)
            res.max_tensile_mpa.append(stress_mod.max_tensile_stress(sigma) / 1e6)
            res.surface_temperature_c.append(float(temperature[0]) - props.T_MELT)
            res.residual_salinity.append(salt_equilibrium(float(temperature.mean()),
                                                          cfg.v_irr))
        if t >= t_end:
            break
        _step_cfg = ThreeFieldConfig(**{**cfg.__dict__, "dt": dt_step})
        h_field, _ = _implicit_step(h_field, salinity, model, _step_cfg, cfg.t_surface,
                                    tf_ocean)
        # choose the next step: never let the CFL-limited drainage sub-cycling
        # saturate (w_max dt <= 16 * cfl * dz), but do not go below 60 s
        w_max = 0.0
        if salinity.numel() > 1:
            w_max = float(drainage_velocity(temperature, salinity, cfg.v_irr)[0].max())
        dt_step = cfg.dt if w_max <= 0.0 else min(
            cfg.dt, max(60.0, 0.3 * dz / w_max * 48.0))
        t += dt_step
        k += 1

    temperature = model.temperature(h_field, salinity)
    sigma = stress_mod.restrained_stress(temperature, t_ref=t_ref)
    sigma_free = stress_mod.free_plate_stress(temperature, z)
    v_b = torch.tensor([props.brine_volume_fraction(float(x), float(salinity[i]))
                        for i, x in enumerate(temperature)], dtype=DTYPE)
    perm = props.permeability_vec(v_b)
    res.z = [float(x) for x in z]
    res.temperature = [float(x) for x in temperature]
    res.salinity = [float(x) for x in salinity]
    res.brine_fraction = [float(x) for x in v_b]
    res.permeability = [float(x) for x in perm]
    res.stress_mpa = [float(x) / 1e6 for x in sigma]
    res.stress_free_mpa = [float(x) / 1e6 for x in sigma_free]
    res.crack = stress_mod.crack_assessment(sigma, cfg.flaw_depth)
    res.crack_free = stress_mod.crack_assessment(sigma_free, cfg.flaw_depth)
    res.crack_time_days = crack_time
    res.melt_total_m = melt_total
    res.heat_integral_j_m2 = heat_integral
    res.salt_initial = salt0
    res.salt_final = (float((props.RHO_ICE * salinity * dz).sum())
                      + props.RHO_SEAWATER_REF * cfg.ocean_depth * s_ocean)
    res.salt_exported = salt_exported
    res.salt_error = abs(res.salt_final + salt_exported - salt0) / salt0
    if cfg.spinup:
        res.spinup_relaxation = spinup_diagnostic(cfg, model)
    return res


def spinup_diagnostic(cfg: ThreeFieldConfig, model: EnthalpyModel, target=0.5,
                      max_steps=8000):
    """How the latent heat of the mushy layer slows a cold-snap transient.

    A step change of the surface temperature is applied to an isothermal column
    of sea ice and to the same column of fresh ice; the time to reach ``target``
    of the steady surface-to-base temperature difference is measured for both.
    Their ratio is the latent buffering factor of the mushy layer -- the
    phase-change signature a pure conduction model cannot produce.
    """
    out = {}
    snap = 258.15          # a -15 C cold snap, independent of the run scenario
    for label, mdl in (("sea_ice", model),
                       ("fresh_ice", EnthalpyModel(kind="fresh", mushy_dt=0.05))):
        t_top = snap
        t_bot = (float(props.freezing_point(cfg.salinity0)) if label == "sea_ice"
                 else props.T_MELT)
        sal = torch.full((cfg.n_z,),
                         cfg.salinity0 if label == "sea_ice" else 0.0, dtype=DTYPE)
        h_field = mdl.enthalpy(torch.full((cfg.n_z,), t_bot, dtype=DTYPE), sal)
        dt = 300.0
        cfg_local = ThreeFieldConfig(**{**cfg.__dict__, "dt": dt, "picard": 2})
        target_t = t_bot + target * (t_top - t_bot)
        relax = None
        for m in range(1, max_steps + 1):
            h_field, temperature = _implicit_step(h_field, sal, mdl, cfg_local,
                                                  t_top, t_bot)
            if float(temperature[0]) <= target_t:
                relax = m * dt
                break
        t_cond = cfg.thickness ** 2 / props.DIFF_ICE
        out[label] = {"relaxation_s": relax, "conduction_time_s": t_cond,
                      "ratio": (relax / t_cond) if relax else None}
    if out["fresh_ice"]["relaxation_s"] and out["sea_ice"]["relaxation_s"]:
        out["latent_buffering_factor"] = (out["sea_ice"]["relaxation_s"]
                                          / out["fresh_ice"]["relaxation_s"])
    return out


def coupling_strength(cfg: ThreeFieldConfig | None = None):
    """Run the column one-way and two-way; return the measured coupling strength."""
    cfg = cfg or ThreeFieldConfig()
    one = ThreeFieldConfig(**{**cfg.__dict__, "two_way": False})
    two = ThreeFieldConfig(**{**cfg.__dict__, "two_way": True})
    r1, r2 = run_three_field(one), run_three_field(two)
    m1, m2 = r1.melt_total_m, r2.melt_total_m
    t_mean = float(torch.tensor(r2.temperature).mean())
    v_b = props.brine_volume_fraction(t_mean, cfg.salinity0)
    s_m = mobile_brine_fraction(v_b, cfg.v_irr)
    return {
        "two_way": r2.summary(),
        "one_way": r1.summary(),
        "melt_ratio_two_way_over_one_way": m2 / m1 if m1 > 0 else float("nan"),
        "ocean_heat_flux_ratio": (r2.ocean_heat_flux[-1] / r1.ocean_heat_flux[-1]
                                  if r1.ocean_heat_flux[-1] > 0 else float("nan")),
        "ocean_salinity_shift_psu": r2.ocean_salinity[-1] - r1.ocean_salinity[-1],
        "mushy_rayleigh": darcy.mushy_rayleigh(t_mean, cfg.salinity0,
                                               cfg.thickness, cfg.v_irr),
        "mobile_brine_fraction": float(s_m),
        "relative_permeability": float(relative_permeability(s_m)),
    }
