"""Wiring invariants of the particle-shifting pass in ``capillary_oscillation.run_point``.

Why these tests exist
---------------------
``benchmarks/surface_tension/droplet_shifting_containment.py`` asks whether the
already-merged Lind et al. (2012) operator can rescue the dynamic surface-tension
reading that transport-velocity correction lost
(``docs/xval_method_selection_csf_dynamic_m22526.md``).  For that sweep to mean
anything, three properties of the wiring must hold *independently of the sweep*:

1. **Off is off.** ``shift_strength = 0`` (the default) must leave the historical
   trajectory bit for bit unchanged, otherwise the new knob silently redefines the
   baseline it is compared against.
2. **The cap is a cap.** ``shift_limit`` bounds one pass' displacement in units of
   ``h``; the diagnostic ``shift_peak_over_h`` must respect it.
3. **Direction is the whole argument.** The reason TVC was falsified on a free drop
   is that its displacement points along the outward normal there; shifting
   projects that component away.  This is asserted geometrically on the same drop
   cloud, not taken from the docstring.

Each test is a property of the formulation (identity, bound, or sign), not a fit
to a stored number.
"""
from __future__ import annotations

import torch

from tensorsph.capillary_oscillation import make_droplet, run_point
from tensorsph.droplet_impact import cell_linked_pairs
from tensorsph.particle_shifting import shift_positions
from tensorsph.transport_velocity_correction import transport_velocity_correction

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SIGMA = 0.072
RHO = 1000.0
RADIUS = 0.05
H_FACTOR = 1.3
T_SHORT = 0.05          # a handful of steps: enough to exercise the pass


def _short_run(**kw) -> dict:
    return run_point(10, SIGMA, 0.0, t_end_s=T_SHORT, sample_every=1,
                     radius=RADIUS, **kw)


class TestOffIsOff:
    def test_default_and_explicit_zero_are_identical(self):
        a = _short_run()
        b = _short_run(shift_strength=0.0)
        assert a["steps"] == b["steps"]
        assert a["shift_steps"] == 0
        assert b["shift_steps"] == 0
        assert a["series"]["q"] == b["series"]["q"]
        assert a["series"]["p_interior"] == b["series"]["p_interior"]
        assert a["containment_max_r_over_R"] == b["containment_max_r_over_R"]

    def test_zero_strength_never_touches_positions(self):
        # shift_every = 1 but strength 0: the branch must not run at all.
        r = _short_run(shift_strength=0.0, shift_every=1)
        assert r["shift_steps"] == 0
        assert r["shift_peak_over_h"] == 0.0


class TestKnobsAreRecorded:
    def test_shifting_runs_and_is_reported(self):
        off = _short_run()
        on = _short_run(shift_strength=1.0)
        assert on["shift_steps"] == on["steps"] - 1      # step 0 skipped by contract
        assert on["shift_steps"] > 0
        assert on["shift_peak_over_h"] > 0.0
        assert 0.0 < on["shift_free_surface_fraction_mean"] <= 1.0
        # the pass moves positions, so the trajectory must differ
        assert on["series"]["q"] != off["series"]["q"]

    def test_every_n_is_honoured(self):
        r = _short_run(shift_strength=1.0, shift_every=4)
        expected = max(0, (r["steps"] - 1) // 4)
        assert r["shift_steps"] == expected


class TestCapIsACap:
    def test_peak_displacement_respects_limit(self):
        for limit in (0.05, 0.25):
            r = _short_run(shift_strength=2.0, shift_limit=limit)
            # allow one ulp of slack for the clipping arithmetic
            assert r["shift_peak_over_h"] <= limit + 1.0e-12


class TestFreeSurfaceDirection:
    """TVC pushes the free surface outward; shifting has no normal component."""

    def _drop(self):
        dx = RADIUS / 10.0
        pos, mass = make_droplet(dx, 0.05, torch.device("cpu"), radius=RADIUS,
                                 rho0=RHO)
        h = H_FACTOR * dx
        volume = torch.full_like(mass, dx * dx)      # rho0 dx^2 / rho0
        pairs = cell_linked_pairs(pos, 2.0 * h)
        r_hat = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
        return pos, volume, mass, h, pairs, r_hat

    def test_shifting_normal_component_vanishes_on_the_free_surface(self):
        pos, volume, mass, h, pairs, r_hat = self._drop()
        _, st = shift_positions(pos, volume, h, dt=1.0e-3, v_ref=0.05,
                                strength=1.0, pairs=pairs, return_state=True)
        assert bool(st.free_surface.any())
        radial = (st.shift[st.free_surface] * r_hat[st.free_surface]).sum(dim=1)
        scale = torch.linalg.vector_norm(st.shift, dim=1).max().clamp_min(1.0e-30)
        assert float(radial.abs().max()) <= 1.0e-9 * float(scale)

    def test_tvc_normal_component_is_outward(self):
        pos, volume, mass, h, pairs, r_hat = self._drop()
        res = transport_velocity_correction(pos, volume, h, alpha=0.05, pairs=pairs)
        radial = (res.shift * r_hat).sum(dim=1)
        # the interface particles are the ones that drift out first
        interface = torch.linalg.vector_norm(pos, dim=1) > 0.95 * RADIUS
        assert bool(interface.any())
        assert float(radial[interface].mean()) > 0.0