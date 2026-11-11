"""Three-dimensional SPH operators, verified against analytic solutions.

The 2-D solver in :mod:`tensorsph.droplet_impact` carries the 2-D kernel
normalisation ``10 / (7 pi h^2)`` and rejects any cloud whose second axis is not
2, so nothing in the shipped package produces a correct 3-D density or force.
This subpackage adds the 3-D branch -- kernel, lattice calibration, periodic
neighbour search, operators, wall model, conservation meters and two analytic
benchmark drivers -- **without touching the 2-D code**, so every number already
published under ``docs/`` stays reproducible.

What is verified where
----------------------
=====================================  ==========================================
claim                                  artifact
=====================================  ==========================================
``int W dV = 1`` and ``int r^2 lap W dV``  ``benchmarks/threed_kernel/report.json``
                                       ``= 6`` (Gauss + radial quadrature)
SPH gradient / Laplacian convergence   ``benchmarks/threed_kernel/report.json``
on ``sin sin sin``, order and error
lattice kernel-sum bias ``C - 1``      ``benchmarks/threed_kernel/report.json``
viscous decay rate vs ``2 nu k^2``     ``benchmarks/threed_shear/report.json``
linear-momentum drift (periodic)       ``benchmarks/threed_shear/report.json``
pressure profile vs ``rho0 g (H - z)``  ``benchmarks/threed_hydrostatic/report.json``
wall-model effect (Adami on/off)       ``benchmarks/threed_hydrostatic/report.json``
=====================================  ==========================================

Nothing here is a surface-tension, free-surface or multiphase model: those are
2-D only in this repository and are not claimed for 3-D.
"""

from .conservation import (
    angular_momentum,
    centre_of_mass,
    diagnostics,
    internal_energy,
    kinetic_energy,
    mass_total,
    momentum,
    relative_drift,
    total_energy,
)
from .kernel3 import (
    SUPPORT_FACTOR,
    calibrated_particle_mass,
    cubic_spline_gradient,
    cubic_spline_kernel,
    cubic_spline_laplacian,
    kernel_normalisation,
    lattice_bias,
    lattice_kernel_sum,
    pair_kernels,
    sigma3,
    wendland_c2_gradient,
    wendland_c2_kernel,
)
from .neighbors3 import (
    box_pairs,
    brute_force_pairs_min_image,
    minimum_image,
    periodic_pairs,
)
from .operators3 import (
    accelerations,
    analytic_laplacian_of_mode,
    analytic_mode_decay,
    fourth_order_coefficient,
    gradient_of_field,
    laplacian_fourth_moment,
    laplacian_of_field,
    laplacian_second_moment,
    mode_amplitude,
    pair_geometry,
    pressure_acceleration,
    shear_wave_field,
    summation_density,
    tait_density,
    tait_pressure,
    viscous_acceleration,
    wall_pressure_adami,
)
from .solver import (
    HydrostaticResult,
    ShearWaveResult,
    acoustic_dt,
    cubic_lattice,
    fit_decay_rate,
    hydrostatic_density,
    run_hydrostatic,
    run_shear_wave,
    wall_shell,
)

__all__ = [
    "SUPPORT_FACTOR", "calibrated_particle_mass", "cubic_spline_gradient",
    "cubic_spline_kernel", "cubic_spline_laplacian", "kernel_normalisation",
    "wendland_c2_gradient", "wendland_c2_kernel",
    "lattice_bias", "lattice_kernel_sum", "pair_kernels", "sigma3",
    "box_pairs", "brute_force_pairs_min_image", "minimum_image",
    "periodic_pairs", "accelerations", "analytic_laplacian_of_mode",
    "laplacian_fourth_moment", "fourth_order_coefficient",
    "analytic_mode_decay", "gradient_of_field", "laplacian_of_field",
    "laplacian_second_moment", "mode_amplitude", "pair_geometry",
    "pressure_acceleration", "shear_wave_field", "summation_density",
    "tait_density", "tait_pressure", "viscous_acceleration",
    "wall_pressure_adami", "HydrostaticResult", "ShearWaveResult",
    "acoustic_dt", "cubic_lattice", "fit_decay_rate", "hydrostatic_density",
    "run_hydrostatic", "run_shear_wave", "wall_shell", "angular_momentum",
    "centre_of_mass",
    "diagnostics", "internal_energy", "kinetic_energy", "mass_total", "momentum",
    "relative_drift", "total_energy",
]
