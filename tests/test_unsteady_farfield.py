"""Guards for the unsteady (accumulation) term of the far-field force channel.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.  The round-3 channel
implemented the *steady* control-volume identity only; its accumulation term
``-d/dt int rho u dV`` was withheld because in 2-D its uniform part grows like
``R**2`` and cancels against the pressure flux.  These guards pin the
perturbation form that repairs it:

* the ``d/dt`` stencil must be second order (a first-order one biases the
  amplitude of a harmonic signal by ``O(Omega dt)``, which would masquerade as a
  force error),
* ``cv_momentum(frame_vel=...)`` must remove the uniform part exactly (that is
  what keeps the ``R**2`` cancellation out of the sampled sum),
* an accelerating uniform stream with no body must give exactly zero force
  (the wiring null: it fails by orders of magnitude if the ``pi R**2 a_f``
  surface piece or the analytic fluid area is mis-handled),
* a cylinder in an oscillating stream must reproduce ``F_x = 2 rho pi a**2
  u_dot`` (``C_M = 2``: displaced mass ``1`` + added mass ``C_a = 1``), with the
  cross component ``F_y`` at round-off,
* and the steady-only arm -- the estimator the previous round shipped -- must
  *fail* the same case, because the contrast is the reason this module exists.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

from tensorsph.farfield_force import (  # noqa: E402
    control_volume_force,
    control_volume_force_unsteady,
    cv_fluid_area,
    cv_momentum,
    d_dt_periodic,
    unsteady_potential_flow_cylinder,
)

import farfield_force_audit as audit3  # noqa: E402
import unsteady_farfield_audit as audit4  # noqa: E402

DT = torch.float64
A, RHO, U0, OMEGA = audit4.A, audit4.RHO, audit4.U0, audit4.OMEGA


def _field(radius, dx, phase, n_phases=32):
    centre = torch.zeros(2, dtype=DT)
    pos = audit3.annulus_lattice(centre, A + 0.5 * dx, radius + dx, dx)
    dt = (2.0 * math.pi / OMEGA) / n_phases
    t = phase * dt
    u = U0 * math.cos(OMEGA * t)
    udot = -U0 * OMEGA * math.sin(OMEGA * t)
    vel, pres = unsteady_potential_flow_cylinder(pos, A, u, udot, RHO, centre=centre)
    frame_vel = torch.tensor([u, 0.0], dtype=DT)
    frame_acc = torch.tensor([udot, 0.0], dtype=DT)
    return pos, vel, pres, frame_vel, frame_acc, dt


def test_d_dt_is_second_order_on_a_harmonic():
    """Wrap-around central difference: error ~ (omega dt)**2/6, so 4x per halving.

    The series must be *closed* over the sample window (one period) for the
    periodic stencil to make sense -- that is the contract the driver relies on.
    """
    omega, n = 2.0, 32
    errs = []
    for m in (8, 4, 2):                     # dt shrinks by 2 each step (same window)
        dt = 2.0 * math.pi / (omega * n) * m
        t = torch.arange(n, dtype=DT) * dt
        s = torch.sin(omega * t).reshape(-1, 1)
        got = d_dt_periodic(s, dt, periodic=True)[:, 0]
        exact = omega * torch.cos(omega * t)
        errs.append(float((got - exact).abs().max()))
        assert errs[-1] < 0.2 * (omega * dt) ** 2 * omega * 1.05, (m, errs[-1])
    # second order: halving dt must cut the error by ~4 (allow 3.2..5)
    assert 3.2 < errs[0] / errs[1] < 5.0, errs
    assert 3.2 < errs[1] / errs[2] < 5.0, errs
    # non-periodic ends fall back to a second-order one-sided stencil
    dt = 2.0 * math.pi / (omega * n)
    t = torch.arange(n, dtype=DT) * dt
    s = torch.sin(omega * t).reshape(-1, 1)
    got = d_dt_periodic(s, dt, periodic=False)
    assert float((got[:, 0] - omega * torch.cos(omega * t)).abs().max()) < 2.0 * (omega * dt) ** 2 * omega


def test_cv_momentum_removes_the_uniform_part_exactly():
    centre = torch.zeros(2, dtype=DT)
    pos = audit3.annulus_lattice(centre, 0.6, 1.5, 0.05)
    frame = torch.tensor([0.37, -0.21], dtype=DT)
    uniform = frame.expand_as(pos).clone()
    raw = cv_momentum(pos, uniform, RHO, centre, 1.0, 0.05)
    pert = cv_momentum(pos, uniform, RHO, centre, 1.0, 0.05, frame_vel=frame)
    assert float(raw.norm()) > 1e-2            # the uniform part is present
    assert float(pert.norm()) < 1e-14          # and removed to round-off


def test_null_uniform_accelerating_stream_has_no_force():
    """D2: no body, no perturbation -> exactly zero, at every phase."""
    centre = torch.zeros(2, dtype=DT)
    radius, dx, n = 1.0, 0.05, 32
    pos = audit3.annulus_lattice(centre, 0.2, radius + dx, dx)
    dt = (2.0 * math.pi / OMEGA) / n
    mom = []
    for k in range(n):
        u = U0 * math.cos(OMEGA * k * dt)
        vel = torch.tensor([u, 0.0], dtype=DT).expand_as(pos).clone()
        mom.append(cv_momentum(pos, vel, RHO, centre, radius, dx,
                               frame_vel=torch.tensor([u, 0.0], dtype=DT)))
    rate = d_dt_periodic(torch.stack(mom), dt, periodic=True)
    scale = RHO * radius ** 2 * U0 * OMEGA
    worst = 0.0
    cov_complete = True
    for k in range(n):
        u = U0 * math.cos(OMEGA * k * dt)
        udot = -U0 * OMEGA * math.sin(OMEGA * k * dt)
        vel = torch.tensor([u, 0.0], dtype=DT).expand_as(pos).clone()
        pres = -RHO * udot * pos[:, 0]
        out = control_volume_force_unsteady(
            pos, vel, pres, RHO, centre, radius, dx, momentum_rate=rate[k],
            frame_vel=torch.tensor([u, 0.0], dtype=DT),
            frame_acc=torch.tensor([udot, 0.0], dtype=DT), body_area=0.0)
        worst = max(worst, float(out["force"].norm()) / scale)
        cov_complete = cov_complete and out["coverage"] == 1.0
    assert cov_complete, "a dropped surface bin breaks the null (quadrature, not physics)"
    assert worst < 1e-12, worst


def test_dipole_perturbation_has_zero_volume_momentum():
    """Why the decomposition puts all of ``C_a`` on the pressure flux.

    For ``u' = -U (a**2/r**2) (cos 2t, sin 2t)`` the azimuthal integral vanishes,
    so ``int_CV rho u' dV = 0`` identically.  Asserted so that nobody mistakes the
    zero in the decomposition table for an untested code path.
    """
    pos, vel, pres, fv, fa, _ = _field(3.0 * A, 0.05 * A, 5)
    m = cv_momentum(pos, vel, RHO, torch.zeros(2, dtype=DT), 3.0 * A, 0.05 * A,
                    frame_vel=fv)
    assert float(m.norm()) < 1e-9 * (U0 * A ** 2)


def test_accumulation_operator_matches_analytic_derivative():
    """Non-zero accumulation term: ``u' = f(t) (a/r)**2`` -> ``2 pi rho a**2 ln(R/a) f'``.

    This is the leg the oscillating-cylinder case cannot provide (its volume
    momentum vanishes by symmetry), so the ``-d/dt int rho u' dV`` operator is
    scored here against a closed form instead.
    """
    centre = torch.zeros(2, dtype=DT)
    radius, dx, n = 3.0 * A, 0.025 * A, 64
    pos = audit3.annulus_lattice(centre, A + 0.5 * dx, radius + dx, dx)
    r2 = (pos ** 2).sum(dim=1)
    profile = A * A / r2
    dt = (2.0 * math.pi / OMEGA) / n
    series = []
    for k in range(n):
        f = math.cos(OMEGA * k * dt)
        vel = torch.stack((f * profile, torch.zeros_like(profile)), dim=-1)
        series.append(cv_momentum(pos, vel, RHO, centre, radius, dx))
    rate = d_dt_periodic(torch.stack(series), dt, periodic=True)
    exact_amp = RHO * 2.0 * math.pi * A * A * math.log(radius / A) * OMEGA
    amp, _ = audit4.harmonic_amplitude(torch.stack(series)[:, 0], dt, OMEGA)
    # the *operator* claim: d/dt of the integral is Omega * the integral amplitude
    # (the lattice quadrature of ln(R/a) itself is only accurate to O(dx/a) and is
    #  not what this leg scores)
    d_amp, _ = audit4.harmonic_amplitude(-rate[:, 0], dt, OMEGA)
    assert d_amp == pytest.approx(amp * OMEGA, rel=2e-3), (d_amp, amp * OMEGA)
    # and the sampled integral is the closed form to within the lattice quadrature
    assert amp == pytest.approx(RHO * 2.0 * math.pi * A * A * math.log(radius / A), rel=0.05)
    assert exact_amp > 0.0


def test_cv_fluid_area_rejects_body_larger_than_surface():
    assert cv_fluid_area(2.0, math.pi * 0.25) == pytest.approx(math.pi * (4.0 - 0.25))
    with pytest.raises(ValueError):
        cv_fluid_area(1.0, math.pi * 2.0)


def test_cylinder_in_oscillating_stream_reproduces_cm_two():
    """D1 at the coarsest audited resolution: inertia amplitude and phase."""
    radius, dx, n = 3.0 * A, 0.1 * A, 32
    dt = (2.0 * math.pi / OMEGA) / n
    rate_p, rate_t, forces, steady, displaced = [], [], [], [], []
    for k in range(n):
        pos, vel, pres, fv, fa, _ = _field(radius, dx, k, n)
        rate_p.append(cv_momentum(pos, vel, RHO, pos.new_zeros(2), radius, dx,
                                  frame_vel=fv))
        rate_t.append(cv_momentum(pos, vel, RHO, pos.new_zeros(2), radius, dx))
        forces.append(None)
        steady.append((pos, vel, pres))
        displaced.append(0)
    rp = d_dt_periodic(torch.stack(rate_p), dt, periodic=True)
    rt = d_dt_periodic(torch.stack(rate_t), dt, periodic=True)
    fx, fy, fx_naive, fx_steady, disp = [], [], [], [], []
    for k in range(n):
        pos, vel, pres, fv, fa, _ = _field(radius, dx, k, n)
        out = control_volume_force_unsteady(
            pos, vel, pres, RHO, pos.new_zeros(2), radius, dx,
            momentum_rate=rp[k], frame_vel=fv, frame_acc=fa,
            body_area=audit4.BODY_AREA, momentum_rate_total=rt[k])
        assert out["coverage"] == 1.0
        fx.append(out["force"][0])
        fy.append(out["force"][1])
        fx_naive.append(out["force_naive"][0])
        disp.append(out["force_displaced_mass"][0])
        fx_steady.append(control_volume_force(
            pos, vel, pres, RHO, pos.new_zeros(2), radius, dx)["force"][0])
    amp = audit4.harmonic_amplitude
    exact = audit4.EXACT_FX_AMP
    a_fx, ph_fx = amp(torch.stack(fx), dt, OMEGA)
    a_fy, _ = amp(torch.stack(fy), dt, OMEGA)
    a_naive, _ = amp(torch.stack(fx_naive), dt, OMEGA)
    a_steady, _ = amp(torch.stack(fx_steady), dt, OMEGA)
    a_disp, _ = amp(torch.stack(disp), dt, OMEGA)
    scale = RHO * audit4.BODY_AREA * U0 * OMEGA
    # C_M = 2 and no cross component; the target gate is 3 %, asserted at 0.5 %
    assert abs(a_fx - exact) / exact < 5e-3, (a_fx, exact)
    assert a_fx / scale == pytest.approx(2.0, abs=0.01)
    assert a_fy / exact < 1e-3, a_fy
    assert abs(ph_fx) < 0.2, ph_fx                    # in phase with u_dot
    # the displaced-mass piece alone is C_M = 1 -- the added mass is the rest
    assert a_disp / scale == pytest.approx(1.0, abs=1e-9)
    # the arms this channel must beat: no accumulation term, and the unsplit sum
    assert abs(a_steady - exact) / exact > 1.0
    assert abs(a_naive - exact) / exact > 0.01