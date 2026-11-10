"""Ice-ocean thermal-saline-mechanical multi-physics face of TensorSPH.

Campaign node IN13W2-N07: a verified basis for the *strongly coupled* physics of
a sea-ice sheet, i.e. the three fields that a ship operating in ice has to be
able to reason about --

* **temperature** with phase change (enthalpy method, mushy sea ice, SPH
  conduction operator) -- :mod:`.enthalpy`, :mod:`.sph_thermal`;
* **salinity** with brine drainage (gravity drainage through the pore network)
  -- :mod:`.brine`;
* **stress** from the through-thickness temperature profile, with the cracking
  verdict -- :mod:`.thermal_stress`;

plus the two couplings that make it a system rather than three solvers:
buoyancy-driven convection onset (:mod:`.darcy`) and the three-field integrated
column with the brine-rejection -> ocean-heat-flux -> melt feedback
(:mod:`.coupled`).

Every anchor is analytic or a documented literature band, and the benchmark
``benchmarks/ice_thermal_multiphysics`` writes the numbers:

===============  ==================================================  ==========
anchor           reference                                           tolerance
===============  ==================================================  ==========
Stefan problem   1-D Neumann similarity solution (front + profile)   3 %
lattice           discrete diffusivity factor of the M4 kernel       measured
desalination     equilibrium floor ``v_b = v_irr`` (exact)           <1e-6
desalination     first-year ice salinity band, Cox & Weeks (1983)    3-8 psu
Darcy-Benard     exact onset ``Ra_c = 4 pi^2``                       1 %
thermal stress   Timoshenko restrained plate ``E alpha dT /2(1-nu)`` 1 %
thermal stress   free plate: net force and moment vanish             machine
coupled column   salt conservation                                  0.1 %
===============  ==================================================  ==========
"""

from __future__ import annotations

from . import brine, coupled, darcy, enthalpy, properties, sph_thermal, thermal_stress
from .brine import (V_IRR, band_check, drainage_advance, drainage_step,
                    drainage_step_implicit, drainage_velocity, residual_salinity,
                    run_desalination)
from .coupled import ThreeFieldConfig, ThreeFieldResult, coupling_strength, run_three_field
from .darcy import RA_CRITICAL_EXACT, critical_rayleigh, mushy_rayleigh
from .enthalpy import FRESH, SEA_ICE, EnthalpyModel, neumann_lambda, stefan_front
from .sph_thermal import StefanConfig, StefanResult, run_stefan_sph
from .thermal_stress import (crack_assessment, critical_temperature_drop,
                            free_plate_stress, restrained_stress)

__all__ = [
    "brine", "coupled", "darcy", "enthalpy", "properties", "sph_thermal",
    "thermal_stress",
    "V_IRR", "band_check", "drainage_advance", "drainage_step",
    "drainage_step_implicit", "drainage_velocity", "residual_salinity",
    "run_desalination",
    "ThreeFieldConfig", "ThreeFieldResult", "coupling_strength", "run_three_field",
    "RA_CRITICAL_EXACT", "critical_rayleigh", "mushy_rayleigh",
    "FRESH", "SEA_ICE", "EnthalpyModel", "neumann_lambda", "stefan_front",
    "StefanConfig", "StefanResult", "run_stefan_sph",
    "crack_assessment", "critical_temperature_drop", "free_plate_stress",
    "restrained_stress",
]
