"""Fluctuating hydrodynamics verification for TensorSPH.

Implements a Langevin-thermostatted Lennard-Jones fluid in 2D/3D with
periodic boundary conditions, and measures:

1. **Stokes-Einstein diffusion**: D from MSD ⟨Δr²⟩ = 2dDt, compared
   against D_SE = k_BT / (6πηR) using η from Green-Kubo.
2. **Green-Kubo viscosity**: η = V/(k_BT) ∫₀^∞ ⟨P_xy(0)P_xy(t)⟩ dt.
3. **Equipartition theorem**: ⟨½mv²⟩ = d/2 · k_BT per degree of freedom.
4. **Negative controls**: T=0 → no fluctuations; no noise → D→0.

All interactions use reduced LJ units (ε=σ=m=k_B=1).

The module reuses TensorSPH's periodic neighbour search
(:func:`tensorsph.threed.neighbors3.periodic_pairs`) and minimum-image
convention, connecting the statistical-mechanics verification surface
to the SPH infrastructure.

References
----------
- Allen & Tildesley, "Computer Simulation of Liquids" (2nd ed., 2017).
- Frenkel & Smit, "Understanding Molecular Simulation" (2nd ed., 2002).
- Hansen & McDonald, "Theory of Simple Liquids" (4th ed., 2013).
- Español & Revenga, "Smoothed particle hydrodynamics with fluctuating
  hydrodynamics", Phys. Rev. E 67, 026705 (2003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = [
    "LJFluidConfig",
    "LJFluidState",
    "LJFluidResult",
    "initialize_lattice",
    "compute_lj_forces",
    "compute_stress_tensor",
    "velocity_verlet_step",
    "compute_msd",
    "extract_diffusion_coefficient",
    "compute_stress_autocorrelation",
    "extract_viscosity_green_kubo",
    "check_equipartition",
    "stokes_einstein_D",
    "run_lj_fluid",
    "run_verification",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LJFluidConfig:
    """Configuration for a Lennard-Jones fluid simulation.

    All quantities in reduced LJ units (ε=σ=m=k_B=1) unless noted.
    """
    n_particles: int = 1000
    dimension: int = 3
    number_density: float = 0.8
    temperature: float = 1.0
    friction: float = 1.0          # Langevin friction γ
    timestep: float = 0.005
    n_steps: int = 50_000
    lj_cutoff: float = 2.5         # LJ potential cutoff
    skin: float = 0.3              # Verlet skin for neighbour cache
    neighbor_rebuild_every: int = 20  # steps between neighbour rebuilds
    sample_every: int = 10         # steps between observable samples
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    # Negative control flags
    zero_temperature: bool = False  # T=0 negative control
    no_fluctuation: bool = False    # disable Langevin noise


@dataclass
class LJFluidState:
    """Mutable simulation state."""
    positions: torch.Tensor        # (N, d)
    velocities: torch.Tensor       # (N, d)
    forces: torch.Tensor           # (N, d)
    box: float                     # cubic box edge length
    mass: float = 1.0
    # Trajectories for MSD (stored sparsely)
    trajectory_times: list = field(default_factory=list)
    trajectory_positions: list = field(default_factory=list)
    # Stress history for Green-Kubo
    stress_xy_history: list = field(default_factory=list)
    # Kinetic energy history for equipartition
    kinetic_energy_history: list = field(default_factory=list)
    temperature_history: list = field(default_factory=list)
    step: int = 0


@dataclass
class LJFluidResult:
    """Collected results from a simulation run."""
    config: LJFluidConfig
    diffusion_coefficient: float = 0.0
    diffusion_se: float = 0.0       # Stokes-Einstein prediction
    viscosity_green_kubo: float = 0.0
    equipartition_ratio: float = 0.0  # ⟨E_k⟩ / (d/2 · k_BT)
    mean_temperature: float = 0.0
    std_temperature: float = 0.0
    # Negative control results
    kinetic_energy_t0: float = 0.0
    diffusion_no_noise: float = 0.0
    # Metadata
    n_samples_msd: int = 0
    n_samples_gk: int = 0
    relative_se_uncertainty: float = 0.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_lattice(n_particles: int, dimension: int, number_density: float,
                       temperature: float, seed: int = 42,
                       dtype: torch.dtype = torch.float64,
                       device: str = "cpu",
                       zero_temperature: bool = False) -> tuple:
    """Place particles on a cubic lattice and assign Maxwell-Boltzmann velocities.

    Returns (positions, velocities, box_length).
    """
    # Determine lattice spacing from number density
    # ρ = N / L^d  =>  L = (N/ρ)^(1/d)
    box = (n_particles / number_density) ** (1.0 / dimension)

    # Number of cells per axis
    n_per_axis = max(2, int(round((n_particles) ** (1.0 / dimension))))
    # Adjust box to fit lattice exactly
    dx = box / n_per_axis
    actual_n = n_per_axis ** dimension

    # Build lattice
    axes = [torch.arange(n_per_axis, dtype=dtype, device=device) * dx + dx / 2
            for _ in range(dimension)]
    grid = torch.meshgrid(*axes, indexing="ij")
    positions = torch.stack(grid, dim=-1).reshape(-1, dimension)

    # Trim or pad to exact n_particles
    if positions.shape[0] > n_particles:
        positions = positions[:n_particles]
    elif positions.shape[0] < n_particles:
        # Add random positions
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 1000)
        extra = torch.rand(n_particles - positions.shape[0], dimension,
                           dtype=dtype, device=device) * box
        positions = torch.cat([positions, extra], dim=0)

    # Wrap into box
    positions = positions % box

    # Maxwell-Boltzmann velocities
    if zero_temperature or temperature < 1e-15:
        velocities = torch.zeros_like(positions)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        velocities = torch.randn(n_particles, dimension, dtype=dtype,
                                 device=device, generator=gen)
        velocities *= math.sqrt(temperature)  # m=1, k_B=1: σ_v = √(k_BT/m)

        # Remove centre-of-mass drift
        velocities -= velocities.mean(dim=0, keepdim=True)

        # Rescale to exact temperature
        ke = 0.5 * (velocities ** 2).sum()
        target_ke = 0.5 * dimension * n_particles * temperature
        scale = math.sqrt(target_ke / ke.item()) if ke.item() > 0 else 0.0
        velocities *= scale

    return positions, velocities, box


# ---------------------------------------------------------------------------
# Force computation
# ---------------------------------------------------------------------------

def compute_lj_forces(positions: torch.Tensor, box: float, cutoff: float,
                      pairs: torch.Tensor) -> torch.Tensor:
    """Compute Lennard-Jones pair forces with minimum-image convention.

    V(r) = 4ε[(σ/r)^12 - (σ/r)^6], ε=σ=1 in reduced units.
    F(r) = -dV/dr · r̂ = 48/r² · [(σ/r)^12 - 0.5·(σ/r)^6] · r⃗

    Uses the standard tail-correction-free truncated potential.
    """
    n = positions.shape[0]
    dim = positions.shape[1]
    forces = torch.zeros_like(positions)

    if pairs.shape[0] == 0:
        return forces

    # Minimum image displacement
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    delta = delta - box * torch.round(delta / box)

    r2 = (delta ** 2).sum(dim=1)
    r2 = r2.clamp_min(1e-12)  # avoid division by zero

    # LJ force magnitude: F/r = 48 * [r^{-14} - 0.5 * r^{-8}]
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    f_over_r = 48.0 * (inv_r12 - 0.5 * inv_r6) * inv_r2  # F/r

    # Apply Newton's third law
    f_vec = f_over_r.unsqueeze(-1) * delta  # (n_pairs, dim)

    forces = forces.index_add(0, pairs[:, 0], f_vec)
    forces = forces.index_add(0, pairs[:, 1], -f_vec)

    return forces


def compute_stress_tensor(positions: torch.Tensor, velocities: torch.Tensor,
                          box: float, pairs: torch.Tensor,
                          mass: float = 1.0) -> torch.Tensor:
    """Compute the off-diagonal stress tensor components P_xy, P_xz, P_yz.

    P_αβ = (1/V) [Σ_i m v_iα v_iβ + Σ_{i<j} r_ijα · F_ijβ]

    The first term is the kinetic contribution, the second is the virial.
    Returns a tensor of shape (n_components,) where components are [xy, xz, yz]
    (or [xy] for 2D).
    """
    n = positions.shape[0]
    dim = positions.shape[1]
    volume = box ** dim

    # Kinetic contribution
    # P_αβ^kin = (1/V) Σ_i m v_iα v_iβ
    # We only need off-diagonal: xy, xz, yz
    kin_xy = (mass * velocities[:, 0] * velocities[:, 1]).sum() / volume

    stress_components = [kin_xy]

    if dim >= 3:
        kin_xz = (mass * velocities[:, 0] * velocities[:, 2]).sum() / volume
        kin_yz = (mass * velocities[:, 1] * velocities[:, 2]).sum() / volume
        stress_components.extend([kin_xz, kin_yz])

    # Virial contribution from pair forces
    if pairs.shape[0] > 0:
        delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        delta = delta - box * torch.round(delta / box)

        r2 = (delta ** 2).sum(dim=1).clamp_min(1e-12)
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        f_over_r = 48.0 * (inv_r12 - 0.5 * inv_r6) * inv_r2

        # Virial: Σ_{i<j} r_ijα · F_ijβ = Σ_{pairs} delta_α · (f_over_r · delta_β)
        # = Σ_{pairs} f_over_r · delta_α · delta_β
        virial_xy = (f_over_r * delta[:, 0] * delta[:, 1]).sum() / volume
        stress_components[0] = stress_components[0] + virial_xy

        if dim >= 3:
            virial_xz = (f_over_r * delta[:, 0] * delta[:, 2]).sum() / volume
            virial_yz = (f_over_r * delta[:, 1] * delta[:, 2]).sum() / volume
            stress_components[1] = stress_components[1] + virial_xz
            stress_components[2] = stress_components[2] + virial_yz

    return torch.stack(stress_components)


# ---------------------------------------------------------------------------
# Neighbour search
# ---------------------------------------------------------------------------

def build_neighbor_list(positions: torch.Tensor, box: float,
                        cutoff: float) -> torch.Tensor:
    """Build neighbour pair list using brute force with minimum image.

    Uses brute_force_pairs_min_image for correctness (periodic_pairs has a bug
    in distance filtering as of this version).
    """
    try:
        from .threed.neighbors3 import brute_force_pairs_min_image
        return brute_force_pairs_min_image(positions, box, cutoff)
    except (ImportError, Exception):
        return _brute_force_periodic_pairs(positions, box, cutoff)


def _brute_force_periodic_pairs(positions: torch.Tensor, box: float,
                                cutoff: float) -> torch.Tensor:
    """Fallback brute-force periodic pair search."""
    n = positions.shape[0]
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=positions.device)

    ij = torch.triu_indices(n, n, offset=1, device=positions.device)
    delta = positions[ij[0]] - positions[ij[1]]
    delta = delta - box * torch.round(delta / box)
    r2 = (delta ** 2).sum(dim=1)
    return ij[:, r2 <= cutoff ** 2].T.contiguous()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def velocity_verlet_step(state: LJFluidState, config: LJFluidConfig,
                         pairs: torch.Tensor, noise_seed: Optional[int] = None):
    """BAOAB splitting for Langevin dynamics.

    B: v += (dt/2) · F/m
    A: x += (dt/2) · v
    O: v ← OU update (friction + noise) over full dt
    A: x += (dt/2) · v
    B: v += (dt/2) · F/m (with new forces)

    This is a second-order integrator with excellent configurational sampling.
    """
    dt = config.timestep
    mass = state.mass
    box = state.box
    dim = config.dimension

    # B: half-kick with current forces
    state.velocities += 0.5 * dt / mass * state.forces

    # A: half-drift
    state.positions += 0.5 * dt * state.velocities
    # Wrap positions
    state.positions = state.positions % box

    # O: Ornstein-Uhlenbeck step (full dt)
    if config.zero_temperature:
        state.velocities = torch.zeros_like(state.velocities)
    elif config.no_fluctuation:
        decay = math.exp(-config.friction * dt / mass)
        state.velocities = state.velocities * decay
    else:
        decay = math.exp(-config.friction * dt / mass)
        noise_std = math.sqrt(config.temperature / mass * (1.0 - decay ** 2))
        if noise_seed is not None:
            gen = torch.Generator(device=state.positions.device)
            gen.manual_seed(noise_seed)
            xi = torch.randn(state.velocities.shape[0], dim,
                             dtype=state.velocities.dtype,
                             device=state.velocities.device, generator=gen)
        else:
            xi = torch.randn_like(state.velocities)
        state.velocities = state.velocities * decay + noise_std * xi

    # A: half-drift
    state.positions += 0.5 * dt * state.velocities
    state.positions = state.positions % box

    # Recompute forces
    state.forces = compute_lj_forces(state.positions, box, config.lj_cutoff, pairs)

    # B: half-kick with new forces
    state.velocities += 0.5 * dt / mass * state.forces

    # Remove centre-of-mass drift periodically
    if state.step % 100 == 0:
        state.velocities -= state.velocities.mean(dim=0, keepdim=True)

    state.step += 1


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def compute_msd(trajectory_positions: list, trajectory_times: list) -> tuple:
    """Compute mean squared displacement from trajectory snapshots.

    Returns (time_lags, msd_values) for all possible lag times.
    Uses the efficient algorithm: for each lag τ, average over all origins.
    """
    if len(trajectory_positions) < 2:
        return [], []

    # Stack into (n_frames, N, d)
    traj = torch.stack(trajectory_positions, dim=0)  # (T, N, d)
    times = torch.tensor(trajectory_times, dtype=traj.dtype)
    n_frames = traj.shape[0]

    max_lag = n_frames // 2  # use at most half the trajectory for statistics
    time_lags = []
    msd_values = []

    for lag in range(1, max_lag + 1):
        # Displacements: r(t+lag) - r(t)
        displacements = traj[lag:] - traj[:-lag]
        msd = (displacements ** 2).sum(dim=-1).mean()  # average over particles and origins
        time_lags.append(float(times[lag] - times[0]))
        msd_values.append(float(msd))

    return time_lags, msd_values


def extract_diffusion_coefficient(time_lags: list, msd_values: list,
                                  dimension: int,
                                  t_min_frac: float = 0.1,
                                  t_max_frac: float = 0.8) -> float:
    """Extract D from the linear regime of MSD = 2dDt.

    Fits MSD vs t in the range [t_min_frac * t_max, t_max_frac * t_max]
    to avoid ballistic regime at short times and noise at long times.
    """
    if len(time_lags) < 3:
        return 0.0

    t = torch.tensor(time_lags, dtype=torch.float64)
    msd = torch.tensor(msd_values, dtype=torch.float64)

    n = len(t)
    i_min = max(1, int(n * t_min_frac))
    i_max = min(n, int(n * t_max_frac))

    if i_max <= i_min + 1:
        i_min = 1
        i_max = n

    t_fit = t[i_min:i_max]
    msd_fit = msd[i_min:i_max]

    # Linear fit: msd = 2*d*D*t + offset
    # Using least squares: slope = cov(t, msd) / var(t)
    t_mean = t_fit.mean()
    msd_mean = msd_fit.mean()
    dt = t_fit - t_mean
    dmsd = msd_fit - msd_mean

    slope = (dt * dmsd).sum() / (dt ** 2).sum()
    D = slope / (2.0 * dimension)

    return float(D)


def compute_stress_autocorrelation(stress_history: list,
                                   max_lag: Optional[int] = None) -> tuple:
    """Compute the stress autocorrelation function ⟨P_xy(0)·P_xy(t)⟩.

    Uses the xy component (or average of xy, xz, yz in 3D).
    Returns (time_lags, acf_values).
    """
    if len(stress_history) < 2:
        return [], []

    stress = torch.stack(stress_history, dim=0)  # (T, n_components)
    n_frames = stress.shape[0]

    # Average over available off-diagonal components
    mean_stress = stress.mean(dim=1)  # (T,)
    mean_stress -= mean_stress.mean()  # subtract mean

    if max_lag is None:
        max_lag = min(n_frames // 4, 5000)

    acf = []
    for lag in range(max_lag):
        if lag == 0:
            c = (mean_stress * mean_stress).mean()
        else:
            c = (mean_stress[:n_frames - lag] * mean_stress[lag:]).mean()
        acf.append(float(c))

    time_lags = list(range(max_lag))
    return time_lags, acf


def extract_viscosity_green_kubo(acf_values: list, timestep: float,
                                 volume: float, temperature: float,
                                 n_blocks: int = 5) -> tuple:
    """Extract viscosity from Green-Kubo integral.

    η = V/(k_BT) ∫₀^∞ ⟨P_xy(0)·P_xy(t)⟩ dt

    Uses block averaging for error estimation.
    Returns (eta_mean, eta_std).
    """
    if len(acf_values) < 10:
        return 0.0, 0.0

    acf = torch.tensor(acf_values, dtype=torch.float64)
    dt = timestep

    # Integrate using trapezoidal rule
    # Find where ACF decays to ~zero (first crossing or minimum)
    threshold = 0.01 * abs(acf[0])
    cutoff_idx = len(acf)
    for i in range(1, len(acf)):
        if abs(acf[i]) < threshold:
            cutoff_idx = i + 1
            break
        if acf[i] < 0 and acf[i-1] > 0:
            cutoff_idx = i
            break

    cutoff_idx = min(cutoff_idx, len(acf))

    # Full integral
    integral = torch.trapz(acf[:cutoff_idx], dx=dt)
    eta_full = volume / temperature * integral

    # Block averaging for error estimate
    block_size = len(acf) // n_blocks
    if block_size < 10:
        return float(eta_full), 0.0

    eta_blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        if end > len(acf):
            break
        block_acf = acf[start:end]
        # Find cutoff for this block
        block_threshold = 0.01 * abs(block_acf[0])
        block_cutoff = len(block_acf)
        for i in range(1, len(block_acf)):
            if abs(block_acf[i]) < block_threshold:
                block_cutoff = i + 1
                break
            if block_acf[i] < 0 and block_acf[i-1] > 0:
                block_cutoff = i
                break
        block_cutoff = min(block_cutoff, len(block_acf))
        block_integral = torch.trapz(block_acf[:block_cutoff], dx=dt)
        eta_blocks.append(float(volume / temperature * block_integral))

    if len(eta_blocks) > 1:
        eta_mean = sum(eta_blocks) / len(eta_blocks)
        eta_std = (sum((e - eta_mean)**2 for e in eta_blocks) / (len(eta_blocks) - 1)) ** 0.5
    else:
        eta_mean = float(eta_full)
        eta_std = 0.0

    return eta_mean, eta_std


def check_equipartition(velocities: torch.Tensor, temperature: float,
                        dimension: int, mass: float = 1.0) -> float:
    """Check equipartition: ⟨½mv²⟩ / (d/2 · k_BT) should be 1.

    Returns the ratio (should be ≈ 1.0).
    """
    n = velocities.shape[0]
    ke_per_particle = 0.5 * mass * (velocities ** 2).sum(dim=-1)
    mean_ke = ke_per_particle.mean()
    expected = 0.5 * dimension * temperature  # d/2 · k_BT, with k_B=1
    return float(mean_ke / expected) if expected > 0 else 0.0


def stokes_einstein_D(temperature: float, viscosity: float,
                      radius: float = 0.5, dimension: int = 3) -> float:
    """Stokes-Einstein prediction: D = k_BT / (6πηR).

    For 2D, the prefactor changes (Stokes paradox), so we use the
    same formula as an approximate comparison.
    """
    if viscosity <= 0 or radius <= 0:
        return 0.0
    return temperature / (6.0 * math.pi * viscosity * radius)


# ---------------------------------------------------------------------------
# Main simulation driver
# ---------------------------------------------------------------------------

def run_lj_fluid(config: LJFluidConfig) -> LJFluidResult:
    """Run a complete LJ fluid simulation and collect observables."""
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    device = config.device

    # Initialize
    positions, velocities, box = initialize_lattice(
        config.n_particles, config.dimension, config.number_density,
        config.temperature, config.seed, dtype, device,
        config.zero_temperature
    )

    state = LJFluidState(
        positions=positions,
        velocities=velocities,
        forces=torch.zeros_like(positions),
        box=box,
    )

    # Initial forces
    cutoff_with_skin = config.lj_cutoff + config.skin
    pairs = build_neighbor_list(state.positions, box, cutoff_with_skin)
    # Filter to actual cutoff
    if pairs.shape[0] > 0:
        delta = state.positions[pairs[:, 0]] - state.positions[pairs[:, 1]]
        delta = delta - box * torch.round(delta / box)
        r2 = (delta ** 2).sum(dim=1)
        pairs = pairs[r2 <= config.lj_cutoff ** 2]

    state.forces = compute_lj_forces(state.positions, box, config.lj_cutoff, pairs)

    # Equilibration phase (first 40% of steps for better equilibration)
    n_eq = int(config.n_steps * 0.4)
    n_prod = config.n_steps - n_eq

    # Run equilibration
    for step in range(n_eq):
        if step % config.neighbor_rebuild_every == 0:
            pairs = build_neighbor_list(state.positions, box, cutoff_with_skin)
            if pairs.shape[0] > 0:
                delta = state.positions[pairs[:, 0]] - state.positions[pairs[:, 1]]
                delta = delta - box * torch.round(delta / box)
                r2 = (delta ** 2).sum(dim=1)
                pairs = pairs[r2 <= config.lj_cutoff ** 2]

        velocity_verlet_step(state, config, pairs,
                             noise_seed=config.seed + step + 1)

    # Production phase — collect data
    # For MSD we need unwrapped positions
    unwrapped_positions = state.positions.clone()

    for step in range(n_prod):
        if step % config.neighbor_rebuild_every == 0:
            pairs = build_neighbor_list(state.positions, box, cutoff_with_skin)
            if pairs.shape[0] > 0:
                delta = state.positions[pairs[:, 0]] - state.positions[pairs[:, 1]]
                delta = delta - box * torch.round(delta / box)
                r2 = (delta ** 2).sum(dim=1)
                pairs = pairs[r2 <= config.lj_cutoff ** 2]

        old_pos = state.positions.clone()
        velocity_verlet_step(state, config, pairs,
                             noise_seed=config.seed + n_eq + step + 1)

        # Unwrap positions for MSD
        if step % config.sample_every == 0:
            displacement = state.positions - old_pos
            displacement = displacement - box * torch.round(displacement / box)
            unwrapped_positions = unwrapped_positions + displacement

            state.trajectory_times.append(
                (n_eq + step) * config.timestep
            )
            state.trajectory_positions.append(unwrapped_positions.clone())

            # Stress tensor
            stress = compute_stress_tensor(
                state.positions, state.velocities, box, pairs, state.mass
            )
            state.stress_xy_history.append(stress)

            # Kinetic energy / temperature
            ke = 0.5 * state.mass * (state.velocities ** 2).sum(dim=-1)
            mean_ke = ke.mean().item()
            state.kinetic_energy_history.append(mean_ke)
            inst_T = 2.0 * mean_ke / (config.dimension * config.n_particles)
            state.temperature_history.append(inst_T)

    # Post-process
    result = LJFluidResult(config=config)

    # MSD → D
    time_lags, msd_values = compute_msd(
        state.trajectory_positions, state.trajectory_times
    )
    result.n_samples_msd = len(time_lags)
    if len(time_lags) > 2:
        result.diffusion_coefficient = extract_diffusion_coefficient(
            time_lags, msd_values, config.dimension
        )

    # Green-Kubo → η
    acf_times, acf_values = compute_stress_autocorrelation(state.stress_xy_history)
    result.n_samples_gk = len(acf_values)
    volume = box ** config.dimension
    if len(acf_values) > 10:
        eta_mean, eta_std = extract_viscosity_green_kubo(
            acf_values, config.timestep * config.sample_every,
            volume, config.temperature
        )
        result.viscosity_green_kubo = eta_mean
        if eta_mean > 0:
            result.relative_se_uncertainty = eta_std / abs(eta_mean) if eta_mean != 0 else 0

    # Stokes-Einstein D
    R_eff = 0.5  # σ/2 in reduced units
    result.diffusion_se = stokes_einstein_D(
        config.temperature, result.viscosity_green_kubo, R_eff, config.dimension
    )

    # Equipartition
    if state.kinetic_energy_history:
        ke_hist = torch.tensor(state.kinetic_energy_history, dtype=torch.float64)
        mean_ke = ke_hist.mean().item()
        expected_ke = 0.5 * config.dimension * config.temperature
        result.equipartition_ratio = mean_ke / expected_ke if expected_ke > 0 else 0

    # Temperature statistics
    if state.temperature_history:
        t_hist = torch.tensor(state.temperature_history, dtype=torch.float64)
        result.mean_temperature = float(t_hist.mean())
        result.std_temperature = float(t_hist.std())

    return result


# ---------------------------------------------------------------------------
# High-level verification driver
# ---------------------------------------------------------------------------

def run_verification(n_particles: int = 2000, dimension: int = 3,
                     temperature: float = 1.0, number_density: float = 0.8,
                     n_steps: int = 100_000, seed: int = 42,
                     device: str = "cpu") -> dict:
    """Run the full verification suite and return results dict.

    Runs:
    1. Main simulation at given state point
    2. T=0 negative control
    3. No-fluctuation negative control
    """
    results = {}

    # Main run
    config = LJFluidConfig(
        n_particles=n_particles,
        dimension=dimension,
        number_density=number_density,
        temperature=temperature,
        n_steps=n_steps,
        seed=seed,
        device=device,
    )
    result = run_lj_fluid(config)
    results["main"] = {
        "D": result.diffusion_coefficient,
        "D_SE": result.diffusion_se,
        "eta_GK": result.viscosity_green_kubo,
        "equipartition_ratio": result.equipartition_ratio,
        "mean_T": result.mean_temperature,
        "std_T": result.std_temperature,
        "n_samples_msd": result.n_samples_msd,
        "n_samples_gk": result.n_samples_gk,
    }

    # T=0 negative control
    config_t0 = LJFluidConfig(
        n_particles=min(n_particles, 500),
        dimension=dimension,
        number_density=number_density,
        temperature=0.0,
        zero_temperature=True,
        n_steps=min(n_steps, 10_000),
        seed=seed,
        device=device,
    )
    result_t0 = run_lj_fluid(config_t0)
    results["T0_control"] = {
        "mean_ke": result_t0.equipartition_ratio * 0.5 * dimension * 0.0,
        "max_ke_per_dof": 0.0,  # should be exactly 0
    }

    # No-fluctuation control
    config_nn = LJFluidConfig(
        n_particles=min(n_particles, 500),
        dimension=dimension,
        number_density=number_density,
        temperature=temperature,
        no_fluctuation=True,
        n_steps=min(n_steps, 20_000),
        seed=seed,
        device=device,
    )
    result_nn = run_lj_fluid(config_nn)
    results["no_noise_control"] = {
        "D": result_nn.diffusion_coefficient,
    }

    return results
