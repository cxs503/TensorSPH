"""Free-surface air entrainment, bubble transport and aerated drag (P165W2-N04).

Three loosely coupled pieces make up the model chain:

``entrainment``
    Chanson (1997) inception criterion for self-aeration of a free surface on a
    sloping chute, ``V sin(theta) / sqrt(g d cos(theta)) > C_crit``.

``bubble_transport``
    Dilute bubbly-phase transport: bubble rise velocity (Stokes limit, rigid
    sphere drag balance, Mendelson wave theory), an Eulerian advective-diffusion
    solver for the local void fraction, a Lagrangian bubble-cloud integrator, and
    the SPH summation estimate that maps bubble volumes onto a void fraction.

``friction``
    Frictional resistance of the aerated mixture: Chisholm two-phase multiplier
    and a log-law (Wood 1991 form) closure, plus the reduction rate each implies
    at the same mixture flux.
"""

from .entrainment import (
    CAVITY_FRACTION,
    C_CRIT,
    aeration_active,
    chanson_parameter,
    derived_crit_constant,
    entrainment_flux,
    inception_depth,
    inception_froude,
    inception_froude_normal,
    inception_velocity,
    onset_velocity_by_bisection,
)
from .bubble_transport import (
    AIR_20C,
    WATER_20C,
    analytic_void_fraction,
    drag_balance_rise_velocity,
    erf_anchor,
    l2_relative_error,
    mendelson_rise_velocity,
    rise_velocity,
    schiller_naumann_cd,
    solve_void_fraction_field,
    stokes_rise_velocity,
    transport_bubbles,
    void_fraction_from_bubbles,
    void_fraction_histogram,
    y90_for_diffusion,
)
from .friction import (
    aerated_friction_factor,
    chisholm_multiplier,
    chisholm_reduction_rate,
    friction_factor_loglaw,
    lockhart_martinelli_x,
    loglaw_reduction_rate,
    wood_kappa_effective,
)

__all__ = [
    "C_CRIT",
    "CAVITY_FRACTION",
    "aeration_active",
    "chanson_parameter",
    "derived_crit_constant",
    "entrainment_flux",
    "inception_depth",
    "inception_froude",
    "inception_froude_normal",
    "inception_velocity",
    "onset_velocity_by_bisection",
    "AIR_20C",
    "WATER_20C",
    "analytic_void_fraction",
    "drag_balance_rise_velocity",
    "erf_anchor",
    "l2_relative_error",
    "mendelson_rise_velocity",
    "rise_velocity",
    "schiller_naumann_cd",
    "solve_void_fraction_field",
    "stokes_rise_velocity",
    "transport_bubbles",
    "void_fraction_from_bubbles",
    "void_fraction_histogram",
    "y90_for_diffusion",
    "aerated_friction_factor",
    "chisholm_multiplier",
    "chisholm_reduction_rate",
    "friction_factor_loglaw",
    "lockhart_martinelli_x",
    "loglaw_reduction_rate",
    "wood_kappa_effective",
]
