"""TensorSPH: PyTorch-native smoothed particle hydrodynamics."""

from .droplet_impact import (
    Config,
    SimulationUnstableError,
    __version__,
    acoustic_courant,
    acoustic_dt_limit,
    apply_boundary,
    boundary_box,
    cubic_spline_gradient,
    cubic_spline_kernel,
    density_diffusion_rate,
    density_from_neighbor_pairs,
    diffusion_dt_limit,
    drop_wall_wall_pairs,
    evaluate_acceptance,
    gravity_dt_limit,
    hydrostatic_density,
    make_particles,
    pair_kernels,
    resolve_device,
    rhs,
    rhs_all_pairs,
    run,
    next_dt,
    stable_dt,
    suggested_dt,
    tait_pressure,
    validate_config,
    viscous_dt_limit,
    wall_lattice,
)
from .dam_break import DamBreakConfig, make_column, run_case
from .neighbor_search import (
    PairListCache,
    brute_force_pairs,
    cell_linked_pairs,
    cell_linked_pairs_host,
)
from .surface_tension import (
    compute_color_gradient,
    compute_curvature,
    compute_normal,
    compute_surface_tension_force,
    capillary_dt_limit,
)

__all__ = [
    "Config", "SimulationUnstableError", "__version__", "acoustic_courant",
    "apply_boundary", "boundary_box", "cubic_spline_gradient",
    "cubic_spline_kernel", "density_diffusion_rate", "density_from_neighbor_pairs",
    "diffusion_dt_limit", "drop_wall_wall_pairs",
    "evaluate_acceptance", "gravity_dt_limit", "hydrostatic_density",
    "make_particles", "pair_kernels",
    "acoustic_dt_limit", "next_dt", "stable_dt",
    "resolve_device", "rhs", "rhs_all_pairs", "run", "suggested_dt",
    "tait_pressure", "validate_config", "viscous_dt_limit", "wall_lattice",
    "PairListCache", "brute_force_pairs", "cell_linked_pairs",
    "cell_linked_pairs_host", "DamBreakConfig", "make_column", "run_case",
    "compute_color_gradient", "compute_curvature", "compute_normal",
    "compute_surface_tension_force", "capillary_dt_limit",
]
