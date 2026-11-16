"""Round 9 regression for ``CASE-SPH-OSC-DROP-FREQ``.

Pins the round-9 root-cause finding for the dynamic (time-advancing) arm:

1. the CSF surface-tension acceleration in the axisymmetric reduction points
   **inward** (restoring), consistent with the static arm's
   ``force = -sigma * kappa * delta_s * n`` (round 9 defect fix);
2. the runaway is driven by the **pressure** subsystem (the Laplace-compressed
   initial state's sharp surface gradient), not the CSF operator: at ``t = 0``
   ``|a_press|max > |a_csf|max``;
3. the ``sigma -> 0`` control (no Laplace compression) has no pressure or CSF
   acceleration, i.e. the integrator + mirror axis + continuity are clean and
   the runaway needs the Laplace-compressed surface.

These are honest-FAIL pins: the dynamic arm is *not* usable for a frequency
measurement yet (``passes_anchor`` is False), and the tests assert that state
rather than papering over it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import tensorsph.oscillating_drop_dynamic as M  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.wcsph import kernels  # noqa: E402


def _cfg(sigma=1.0, delta_sph=0.1):
    return M.DynamicConfig(radius=1.0, rho0=1.0, sigma=sigma, dx=1.0 / 9.0,
                          mode=2, eps=0.05, c0=5.0, delta_sph=delta_sph,
                          periods=1.0, skip_periods=0.25, alpha_av=0.02,
                          sample_every=10)


def _setup(cfg):
    torch.set_default_dtype(torch.float64)
    geo = M.half_plane_lattice(cfg)
    pos = geo["pos"].clone()
    mass = geo["mass"].clone()
    n = pos.shape[0]
    for _ in range(2):
        pos_m, _, mass_m = M.mirror_cloud(pos, torch.zeros_like(pos), mass)
        rho_m = M.summation_density(pos_m, mass_m, cfg.h, cfg.kernel)
        comp = M.completeness(pos_m, mass_m, rho_m, cfg.h, cfg.kernel)[:n]
        scale = torch.where(comp >= cfg.completeness_gate,
                            cfg.rho0 / rho_m[:n], torch.ones_like(rho_m[:n]))
        mass = mass * scale
    pos_m, _, mass_m = M.mirror_cloud(pos, torch.zeros_like(pos), mass)
    rho = M.initial_density(pos_m, mass_m, cfg.h, cfg.rho0, sigma=cfg.sigma,
                              radius=cfg.radius, c0=cfg.c0, gamma=cfg.gamma)[:n]
    rho = rho.clamp(min=0.5 * cfg.rho0)
    vel = torch.zeros_like(pos)
    return geo, pos, mass, rho, vel


def _decompose(pos, vel, rho, mass, cfg):
    """Return ``(a_press, a_csf)`` on the primary half plane (t = 0 state)."""
    n = pos.shape[0]
    pos2 = torch.cat((pos, torch.stack((-pos[:, 0], pos[:, 1]), dim=1)), dim=0)
    vel2 = torch.cat((vel, torch.stack((-vel[:, 0], vel[:, 1]), dim=1)), dim=0)
    mass2 = torch.cat((mass, mass), dim=0)
    rho2 = torch.cat((rho, rho), dim=0)
    h, c0, rho0, gamma = cfg.h, cfg.c0, cfg.rho0, cfg.gamma
    pairs = cell_linked_pairs(pos2, cfg.support)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos2[i] - pos2[j]
    r = torch.linalg.vector_norm(delta, dim=1)
    _, grad = kernels.kernel_pairs(delta, h, pos2.shape[1], kind=cfg.kernel)
    p = torch.clamp(rho0 * c0 * c0 / gamma * ((rho2 / rho0).pow(gamma) - 1.0),
                    min=0.0)
    por2 = p / rho2.square()
    bracket = por2[i] + por2[j]
    if cfg.alpha_av > 0.0:
        vdotr = (vel2[i] - vel2[j]).mul(delta).sum(dim=1)
        mu = h * vdotr / (r.square() + 0.01 * h * h)
        pi = torch.where(vdotr < 0.0,
                         (-cfg.alpha_av * c0 * mu) / (0.5 * (rho2[i] + rho2[j])),
                         torch.zeros_like(mu))
        bracket = bracket + pi
    a_press = torch.zeros_like(pos2)
    contrib = -bracket[:, None] * grad
    a_press.index_add_(0, i, mass2[j][:, None] * contrib)
    a_press.index_add_(0, j, mass2[i][:, None] * contrib)
    state, _ = M._csf_fields(pos2, mass2, h, cfg)
    x = pos2[:, 0]
    sign = torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))
    rho_axis = x.abs().clamp_min(0.25 * h)
    kappa = state.kappa + state.normal[:, 0] * sign / rho_axis
    band = ((state.color > 0.25) & (state.color < 0.75)).to(kappa.dtype)
    f = cfg.sigma * kappa * state.delta_s * band
    a_csf = -(f / rho2)[:, None] * state.normal
    return a_press[:n], a_csf[:n], state


def test_csf_acceleration_is_inward_restoring():
    """The CSF term must restore (point inward on a convex drop), matching the
    static arm's ``force = -sigma * kappa * delta_s * n``."""
    cfg = _cfg()
    cfg.validate()
    geo, pos, mass, rho, vel = _setup(cfg)
    n = pos.shape[0]
    _, a_csf, state = _decompose(pos, vel, rho, mass, cfg)
    r = torch.linalg.vector_norm(pos, dim=1)
    rn = pos / r[:, None]
    radial = (a_csf * rn).sum(dim=1)
    band = state.delta_s[:n] > 1e-3 * state.delta_s[:n].max()
    assert bool(band.any()), "no interface particles found"
    assert float(radial[band].max()) <= 1e-6, \
        f"CSF acceleration has an outward component {float(radial[band].max())}"


def test_pressure_drives_the_runaway():
    """The runaway is pressure-driven: at t = 0 the pressure acceleration
    already exceeds the CSF acceleration, and the sigma->0 control (no Laplace
    compression) has no pressure or CSF acceleration."""
    cfg = _cfg()
    cfg.validate()
    geo, pos, mass, rho, vel = _setup(cfg)
    a_press, a_csf, _ = _decompose(pos, vel, rho, mass, cfg)
    assert float(a_press.norm(dim=1).max()) > float(a_csf.norm(dim=1).max()), \
        "expected the pressure term to dominate at t = 0"
    cfg0 = M.DynamicConfig(radius=1.0, rho0=1.0, sigma=1e-9, dx=1.0 / 9.0,
                           mode=2, eps=0.05, c0=5.0, delta_sph=0.0,
                           periods=1.0, skip_periods=0.25, alpha_av=0.02,
                           sample_every=10)
    cfg0.validate()
    geo0, pos0, mass0, rho0f, vel0 = _setup(cfg0)
    a_press0, a_csf0, _ = _decompose(pos0, vel0, rho0f, mass0, cfg0)
    assert float(a_press0.norm(dim=1).max()) < 1e-6, \
        "sigma->0 control must have no pressure acceleration"
    assert float(a_csf0.norm(dim=1).max()) < 1e-6, \
        "sigma->0 control must have no CSF acceleration"


def test_dynamic_arm_is_not_usable_recorded_defect():
    """Honest pin: the dynamic arm does not yet yield a trustworthy omega.

    A short capillary trajectory must run away (``max_speed`` far above the
    mode velocity scale) so that a future fix is detected when this starts to
    pass.  This is the round-5/6/9 recorded defect, not a gate pass.
    """
    cfg = _cfg()
    cfg.validate()
    row = M.run_dynamic_drop(cfg)
    omega_ref = row["omega_analytic"]
    mode_v = omega_ref * 0.05  # eps * omega * R
    assert row["max_speed"] > 20.0 * mode_v, \
        "the recorded runaway no longer reproduces; re-audit the dynamic arm"
