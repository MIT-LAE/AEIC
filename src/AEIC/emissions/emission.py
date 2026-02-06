# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from AEIC.config import config
from AEIC.emissions.types import EmissionsSubset
from AEIC.performance.models import BasePerformanceModel
from AEIC.performance.types import ThrustModeValues
from AEIC.trajectories import Dimension, Dimensions, FieldMetadata, FieldSet, Trajectory
from AEIC.types import Fuel, Species, SpeciesValues

from .apu import get_APU_emissions
from .gse import get_GSE_emissions
from .lto import get_LTO_emissions
from .trajectory import get_trajectory_emissions

# QUESTION: How to handle species? Dimension, coordinate values, variance?

TRAJ_DIMS = Dimensions.from_abbrev('TSP')
LTO_DIMS = Dimensions.from_abbrev('TSM')
OVERALL_DIMS = Dimensions.from_abbrev('TS')

EMISSIONS_FIELDSET_NAME = 'emissions'


def _f(dims, desc):
    return FieldMetadata(dimensions=dims, description=desc, units='g')


def _fi(dims, desc):
    return FieldMetadata(dimensions=dims, description=desc, units='g / kg fuel')


EMISSIONS_FIELDS = FieldSet(
    EMISSIONS_FIELDSET_NAME,
    trajectory_emissions=_f(TRAJ_DIMS, 'Per-segment emissions along trajectory'),
    trajectory_indices=_fi(TRAJ_DIMS, 'Per-segment emission indices along trajectory'),
    lto_emissions=_f(LTO_DIMS, 'Per-thrust mode LTO emissions'),
    lto_indices=_fi(LTO_DIMS, 'Per-thrust mode LTO emission indices'),
    apu_emissions=_f(OVERALL_DIMS, 'APU emissions'),
    apu_indices=_fi(OVERALL_DIMS, 'APU emission indices'),
    gse_emissions=_f(OVERALL_DIMS, 'GSE emissions'),
    total_emissions=_f(OVERALL_DIMS, 'Total emissions'),
    fuel_burn_per_segment=FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY, Dimension.POINT),
        description='Fuel burn per trajectory segment',
        units='kg',
    ),
    total_fuel_burn=FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY),
        description='Total fuel burn',
        units='kg',
    ),
    lifecycle_co2=FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY),
        description='Lifecycle CO2 adjustment',
        units='g',
        required=False,
    ),
)
"""Field set containing emissions data."""


@dataclass
class Emissions:
    """Emissions for a mission broken down by flight phase/components.

    Each flight phase/component attribute is a dictionary mapping species to
    emissions values or indices implemented using the generic `SpeciesValues`
    type. These dictionaries contain different types of values depending on the
    flight phase/component:

    - `trajectory`: NumPy arrays of per-segment emissions [g].
    - `lto`: `ThrustModeValues` objects containing per-thrust mode emissions [g].
    - `apu`: floats containing total APU emissions [g].
    - `gse`: floats containing total GSE emissions [g].
    - `total`: floats containing total emissions [g].

    """

    FIELD_SETS: ClassVar[list[FieldSet]] = [EMISSIONS_FIELDS]

    trajectory_emissions: SpeciesValues[np.ndarray]
    """Per-segment emissions along the trajectory for each species [g]."""

    trajectory_indices: SpeciesValues[np.ndarray]
    """Per-segment emission indices along the trajectory for each species."""

    lto_emissions: SpeciesValues[ThrustModeValues]
    """LTO emissions for each species [g]."""

    lto_indices: SpeciesValues[ThrustModeValues]
    """LTO emission indices for each species."""

    apu_emissions: SpeciesValues[float]
    """APU emissions for each species [g]."""

    apu_indices: SpeciesValues[float]
    """APU emission indices for each species."""

    gse_emissions: SpeciesValues[float]
    """GSE emissions for each species [g]."""

    total_emissions: SpeciesValues[float]
    """Total emissions for each species [g]."""

    fuel_burn_per_segment: np.ndarray
    """Fuel burn per trajectory segment [kg]."""

    total_fuel_burn: float
    """Total fuel burn for the mission [kg]."""

    lifecycle_co2: float = 0.0
    """Lifecycle CO₂ emissions adjustment [g]."""

    @property
    def species(self) -> set[Species]:
        """Set of species included in any emissions."""
        return (
            set(self.trajectory_emissions.keys())
            | set(self.lto_emissions.keys())
            | set(self.apu_emissions.keys())
            | set(self.gse_emissions.keys())
            | set(self.total_emissions.keys())
        )

    def field_set(self) -> FieldSet: ...


def compute_emissions(
    pm: BasePerformanceModel, fuel: Fuel, traj: Trajectory
) -> Emissions:
    """Compute all emissions.

    Calculate emission indices (in g / kg fuel) and emission amounts (in g) for
    each required species and flight phase/component (as determined by
    configuration settings). Flight phases/components are:

    - Trajectory: Per-segment emission indices and amounts based on fuel flow
      and atmospheric conditions.

    - LTO: Emission indices and amounts for each LTO thrust mode based on LTO
      performance data.

    - APU: Emission indices and amounts based on APU type and fuel flow.

    - GSE: Emission amounts based on aircraft class and fuel flow.

    - Total: Overall emission indices and amounts summing contributions from
      all sources.

    The contents of the trajectory and LTO components depend on the
    `climb_descent_mode` setting in the configuration. If `climb_descent_mode`
    is set to 'trajectory', emissions are calculated for the entire trajectory,
    including climb and descent phases. If it is set to 'lto', emissions for
    climb and descent phases are not calculated from the trajectory; instead,
    LTO data is used to estimate emissions for these phases based on
    time-in-mode values.

    If lifecycle CO₂ adjustments are enabled, lifecycle CO₂ emissions are also
    calculated and included in the total emissions.

    Emission results from all sources are returned in a single `Emissions`
    object."""

    # Calculate cruise trajectory emissions (CO₂, H₂O, SOₓ, NOₓ, HC, CO, PM).
    fuel_burn_per_segment = np.zeros_like(traj.fuel_mass)
    fuel_burn_per_segment[1:] = traj.fuel_mass[:-1] - traj.fuel_mass[1:]
    trajectory = get_trajectory_emissions(pm, traj, fuel_burn_per_segment, fuel)
    total_fuel_burn = trajectory.fuel_burn

    # Calculate LTO emissions for ground and approach/climb modes.
    lto = get_LTO_emissions(pm, fuel)
    total_fuel_burn += lto.fuel_burn

    # Calculate APU emissions based on specified APU type.
    apu = EmissionsSubset[float]()
    if config.emissions.apu_enabled and pm.apu is not None:
        apu = get_APU_emissions(lto.indices, pm.apu, fuel)
        total_fuel_burn += apu.fuel_burn

    # Compute Ground Service Equipment (GSE) emissions based on aircraft
    # class.
    gse = EmissionsSubset[float]()
    if config.emissions.gse_enabled:
        gse = get_GSE_emissions(pm.aircraft_class, fuel)
        total_fuel_burn += gse.fuel_burn

    # Sum all emission contributions: trajectory + LTO + APU + GSE.
    total_emissions = sum_total_emissions(
        trajectory=trajectory.emissions,
        lto=lto.emissions,
        apu=apu.emissions,
        gse=gse.emissions,
    )

    emissions = Emissions(
        trajectory_emissions=trajectory.emissions,
        trajectory_indices=trajectory.indices,
        fuel_burn_per_segment=fuel_burn_per_segment,
        total_fuel_burn=float(total_fuel_burn),
        lto_emissions=lto.emissions,
        lto_indices=lto.indices,
        apu_emissions=apu.emissions,
        apu_indices=apu.indices,
        gse_emissions=gse.emissions,
        total_emissions=total_emissions,
    )

    lifecycle_adjustment = None
    if (
        Species.CO2 in config.emissions.enabled_species
        and config.emissions.lifecycle_enabled
    ):
        lifecycle_adjustment = get_lifecycle_emissions(fuel, traj)
        emissions.total_emissions[Species.CO2] += lifecycle_adjustment
        emissions.lifecycle_co2 = lifecycle_adjustment

    return emissions


def sum_total_emissions(
    trajectory: SpeciesValues[np.ndarray],
    lto: SpeciesValues[ThrustModeValues],
    apu: SpeciesValues[float],
    gse: SpeciesValues[float],
) -> SpeciesValues[float]:
    """Aggregate emissions amounts across all sources. Sums pointwise
    trajectory, LTO, APU, and GSE emissions for each species."""
    result = SpeciesValues[float]()
    for species in Species:
        total = 0.0
        if species in trajectory:
            total += np.sum(trajectory[species])
        if species in lto:
            total += lto[species].sum()
        if config.emissions.apu_enabled and species in apu:
            total += apu[species]
        if config.emissions.gse_enabled and species in gse:
            total += gse[species]
        result[species] = total
    return result


def get_lifecycle_emissions(fuel: Fuel, traj) -> float:
    """Calculate lifecycle CO₂ adjustments."""
    if fuel.lifecycle_CO2 is None:
        raise RuntimeError('Lifecycle CO2 data not available for selected fuel.')
    fuel_used = traj.fuel_mass[0] - traj.fuel_mass[-1]
    return float(fuel.lifecycle_CO2 * (fuel_used * fuel.energy_MJ_per_kg))
