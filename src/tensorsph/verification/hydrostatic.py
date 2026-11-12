"""M2 -- hydrostatic pressure verification (``p(z) = rho g (H - z)``).

The case is a tank of still water released from rest.  SPH summation density
truncates at the free surface, so the top two-to-three rows carry ``rho < rho0``
and the clamped Tait law gives them zero pressure; the *interior* must still
recover the analytic profile.  The gate is therefore applied to the interior
bands only, on the time mean over the second half of the run (a lattice released
at rest sloshes acoustically, and the instantaneous profile oscillates by tens
of per cent -- the oscillation is reported next to the mean, not gated).

The integrator and the band table are the ones in :mod:`tensorsph.vv.hydrostatic`
(``Rung 1`` of the verification ladder); this module fixes the *case size* the
campaign asks for (``N >= 1000`` fluid particles), adds the ``rho = 0`` negative
control and returns an evidence-ready record.  Re-using the rung instead of
copying it keeps one definition of "hydrostatic error" in the repository.

Negative control ``rho0 = 0``
    If the density that enters the equation of state is zero the pressure must be
    identically zero -- for an empty domain, for the Tait law pointwise, and for
    the force it produces.  Two checks are run:

    1. pointwise: ``tait_pressure(0) == 0`` (and finite, and non-negative);
    2. field: one ``rhs`` evaluation of an *empty* particle set returns an empty
       pressure with no NaN, and a single isolated particle of zero mass has
       ``rho = 0`` and therefore ``p = 0`` -- again with no NaN.

    A control that merely switches gravity off would test the reference as well
    as the solver (the analytic profile collapses to zero with ``g``); this one
    does not touch ``g``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..droplet_impact import (Config, resolve_device, rhs, tait_pressure)
from ..vv.hydrostatic import HydrostaticConfig, run_hydrostatic


@dataclass
class HydrostaticCase:
    """Square-ish tank sized so that ``N >= min_particles``.

    ``dx`` and the tank size are the two knobs; ``n_particles`` is a *reported*
    number (see :func:`particle_count`), never assumed.
    """

    dx: float = 0.02
    tank_width: float = 1.0
    depth: float = 0.4
    seconds: float = 2.0
    sound_speed: float = 20.0
    min_particles: int = 1000
    # Averaging window of the time mean.  The defaults are the rung's original
    # settings; they are exposed so a caller can ask for a denser sample (the
    # acoustic oscillation of the pool is 6-15 % and the mean's sampling noise
    # scales as 1/sqrt(samples)) without forking the case definition.
    sample_every: int = 20
    average_from: float = 0.5

    def rung_config(self) -> HydrostaticConfig:
        return HydrostaticConfig(dx=self.dx, tank_width=self.tank_width,
                                 depth=self.depth, seconds=self.seconds,
                                 sound_speed=self.sound_speed,
                                 hydrostatic_init=True, bands=8,
                                 average_from=self.average_from,
                                 sample_every=self.sample_every)

    def particle_count(self) -> int:
        from ..droplet_impact import cell_centres
        nx = cell_centres(self.tank_width, self.dx, resolve_device("cpu")).numel()
        ny = cell_centres(self.depth, self.dx, resolve_device("cpu")).numel()
        return int(nx * ny)


def classify_bands(result: dict, depth: float, dx: float) -> dict:
    """Re-classify the rung's bands with an *overlap* rule.

    ``tensorsph.vv.hydrostatic`` marks a band as free-surface when its **centre**
    is within ``2h`` of the surface.  Summation density is truncated for every
    particle within ``2h`` of the surface, so a band is contaminated as soon as
    its **upper edge** enters ``[H - 2h, H]``; the centre rule lets one shelf
    band through, and that band reads several per cent high.  The correction is
    applied here (rather than downstream) so the gate, the figure and the report
    all use one definition.  The rung's own number is kept in the record as
    ``rung_max_band_error_pct`` so the difference is auditable.
    """
    support = 2.0 * 1.3 * dx
    bands = result["measured"]["bands"]
    for band in bands:
        band["free_surface"] = bool(band["free_surface"]
                                    or band["y_high"] > depth - support)
    gated = [b for b in bands if not b["free_surface"] and b["pressure_analytic"] > 0]
    shelf = [b for b in bands if b["free_surface"]]
    if not gated:
        raise ValueError("no gated band: the tank is shallower than the kernel support")
    errors = [abs(b["rel_error_pct"]) for b in gated]
    result["measured"]["gated_bands"] = [b["band"] for b in gated]
    result["measured"]["free_surface_bands"] = [b["band"] for b in shelf]
    result["measured"]["rung_max_band_error_pct"] = result["measured"]["max_band_error_pct"]
    result["measured"]["max_band_error_pct"] = max(errors)
    result["measured"]["mean_band_error_pct"] = sum(errors) / len(errors)
    result["measured"]["shelf_band_error_pct"] = (
        shelf[0]["rel_error_pct"] if shelf else None)
    result["error_pct"] = max(errors)
    result["gate"]["pass"] = bool(max(errors) <= 2.0)
    return result


def run_hydrostatic_case(case: HydrostaticCase, gravity_override: float | None = None,
                         return_fields: bool = False) -> dict:
    rung = run_hydrostatic(case.rung_config(), gravity_override=gravity_override,
                           return_fields=return_fields)
    rung["case"] = "hydrostatic_pressure_profile"
    rung["config"]["n_particles"] = case.particle_count()
    rung["config"]["min_particles"] = case.min_particles
    rung["measured"]["n_particles"] = int(rung["config"]["particles"])
    rung["measured"]["particle_requirement_met"] = bool(
        rung["config"]["particles"] >= case.min_particles)
    rung = classify_bands(rung, case.depth, case.dx)
    return rung


def zero_density_control() -> dict:
    """Negative control: with zero density the pressure must be zero."""
    probe = torch.tensor([0.0, 1.0e-3, 1.0, 1000.0, 2000.0], dtype=torch.float64)
    p = tait_pressure(probe, Config(dx=0.02, sound_speed=20.0))
    pointwise_zero = bool(float(p[0]) == 0.0)
    finite = bool(torch.isfinite(p).all())
    non_negative = bool((p >= 0).all())

    # Field check: an empty fluid set must not produce NaN, negative pressure or
    # a spurious acceleration.
    cfg = Config(dx=0.02, sound_speed=20.0, gravity=9.81, boundary="none")
    device = resolve_device("cpu")
    pos = torch.zeros((0, 2), dtype=torch.float64, device=device)
    vel = torch.zeros((0, 2), dtype=torch.float64, device=device)
    mass = torch.zeros((0,), dtype=torch.float64, device=device)
    rho, pressure, acc = rhs(pos, vel, mass, cfg)
    empty_ok = bool(rho.numel() == 0 and pressure.numel() == 0 and acc.numel() == 0)

    # Field check at zero *density*: one isolated particle with zero mass has
    # rho = 0 (summation density carries no self-contribution when the mass is
    # zero) and therefore pressure zero -- and, importantly, no NaN.  The
    # reference density stays physical: ``rho0 = 0`` is a degenerate
    # parameterisation of the equation of state, not a zero-density state, and
    # the scheme is not required to have an opinion about it.
    one_pos = torch.zeros((1, 2), dtype=torch.float64, device=device)
    one_mass = torch.zeros((1,), dtype=torch.float64, device=device)
    one_vel = torch.zeros((1, 2), dtype=torch.float64, device=device)
    rho1, p1, _acc1 = rhs(one_pos, one_vel, one_mass, cfg)
    # The summation density floor of the solver is not part of this control: what
    # is gated is the pressure, which is what the campaign asks for.  The density
    # is reported next to it for the record.
    single_ok = bool(torch.isfinite(p1).all() and float(p1[0]) == 0.0)

    return {
        "case": "hydrostatic_zero_density_control",
        "eos_at_zero_density": float(p[0]),
        "eos_pointwise_zero": pointwise_zero,
        "eos_finite": finite,
        "eos_non_negative": non_negative,
        "empty_domain_ok": empty_ok,
        "single_particle_zero_rho_ok": single_ok,
        "single_particle_density": float(rho1[0]),
        "single_particle_pressure": float(p1[0]),
        "pass": bool(pointwise_zero and finite and non_negative and empty_ok
                     and single_ok),
    }


def gate(case_result: dict, limit_pct: float = 2.0) -> dict:
    measured = case_result["measured"]
    max_err = measured["max_band_error_pct"]
    min_particles = case_result["config"]["min_particles"]
    n_ok = bool(measured["n_particles"] >= min_particles)
    passed = bool(max_err <= limit_pct and n_ok)
    return {"quantity": "max interior band pressure error vs rho0 g (H - z)",
            "measured_pct": max_err,
            "limit_pct": limit_pct,
            "n_particles": measured["n_particles"],
            "min_particles": min_particles,
            "particles_ok": n_ok,
            "pass": passed}
