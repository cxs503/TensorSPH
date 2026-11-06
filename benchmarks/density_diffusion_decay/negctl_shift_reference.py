"""Negative control for the analytic gate in ``tests/test_density_diffusion_decay.py``.

Run the gate with a reference that is deliberately wrong and it must go red:

    PYTHONPATH=src:benchmarks/density_diffusion_decay \\
        python3 -m pytest tests/test_density_diffusion_decay.py -q \\
        -p negctl_shift_reference

This plugin (pytest's ``-p`` loader) shifts the module-level closed form
``theory_decay_rate`` by +10 % after collection.  A gate that accepts it is not a
gate: it would also accept the errors that matter physically, which are larger.

The shift is applied to the *reference*, not to the operator, so the numerics are
untouched and the only thing that changes is the number the assertion compares
against -- exactly the failure mode of a stale or mis-derived tolerance.

Why 10 %: the measured residual of the gate is ~1 % (see the module docstring of
the test, and ``benchmarks/density_diffusion_decay/decay_rate.json``), so a 10 %
shift is ten times the noise.  The structural failures this gate is designed to
catch are larger still (a missing factor 2 is -50 %, ``dx`` for ``h`` is -23 %).

Expected outcome (measured, see ``negctl_red_output.txt``): two failures and two
passes.

* ``test_the_diffusive_term_decays_a_plane_wave_at_the_analytic_rate`` -- red,
  ``41.42 1/s differs from ... 45.16 1/s by 8.29 % (gate 3 %)``: the gate does
  not absorb the shift.
* ``test_the_decay_rate_scales_with_the_square_of_the_wavenumber`` -- red as well,
  because it re-runs the absolute gate at two wavenumbers before taking their
  ratio.  The ratio alone would be blind to a common-mode reference error, which
  is exactly why the absolute gate has to exist.
* ``test_the_mode_amplitude_follows_the_exponential_over_a_finite_time`` -- green,
  and that is the honest reading of its sensitivity: over the integrated time
  ``gamma T = 0.41``, a 10 % rate error moves the amplitude by
  ``1 - exp(-0.10 * 0.41) = 4 %`` at the reference, 1.7 % in the measured
  comparison, i.e. inside its 3.5 % budget.  Its job is the *time integration*
  (a factor-2 error in ``delta`` makes it red); the rate gate above is the one
  that resolves 10 %.
* ``test_the_gate_is_red_when_the_reference_rate_is_perturbed`` -- green; this is
  the shipped negative control, which perturbs the reference itself by x1.10 and
  x0.90 and asserts that the gate raises.
"""
from __future__ import annotations

SHIFT = 1.10
ATTRIBUTE = "theory_decay_rate"


def pytest_collection_finish(session):
    patched = []
    for item in session.items:
        module = getattr(item, "module", None)
        original = getattr(module, ATTRIBUTE, None)
        if original is None or module in patched:
            continue
        module.theory_decay_rate = (
            lambda delta, sound_speed, h, k, _f=SHIFT, _original=original:
            _f * _original(delta, sound_speed, h, k)
        )
        patched.append(module)
    print(f"\n[negctl] shifted {ATTRIBUTE} by x{SHIFT} in "
          f"{[m.__name__ for m in patched]}")
