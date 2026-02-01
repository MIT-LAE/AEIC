# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from AEIC.config import config
from AEIC.performance.models import BasePerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from AEIC.types import EmissionsDict, EmissionsSubset, Fuel, ModeValues, Species

from .apu import get_APU_emissions
from .gse import get_GSE_emissions
from .lto import get_LTO_emissions
from .trajectory import get_trajectory_emissions


@dataclass
class EmissionsOutput:
    """Emissions for a mission broken down by flight phase/components.

    Each flight phase/component attribute is a dictionary mapping species to
    emissions values or indices implemented using the generic `EmissionsDict`
    type. These dictionaries contain different types of values depending on the
    flight phase/component:

    - `trajectory`: NumPy arrays of per-segment emissions [g].
    - `lto`: `ModeValues` objects containing per-thrust mode emissions [g].
    - `apu`: floats containing total APU emissions [g].
    - `gse`: floats containing total GSE emissions [g].
    - `total`: floats containing total emissions [g].

    """

    trajectory: EmissionsDict[np.ndarray]
    """Per-segment emissions along the trajectory for each species [g]."""

    trajectory_indices: EmissionsDict[np.ndarray]
    """Per-segment emission indices along the trajectory for each species."""

    lto: EmissionsDict[ModeValues]
    """LTO emissions for each species [g]."""

    lto_indices: EmissionsDict[ModeValues]
    """LTO emission indices for each species."""

    apu: EmissionsDict[float]
    """APU emissions for each species [g]."""

    apu_indices: EmissionsDict[float]
    """APU emission indices for each species."""

    gse: EmissionsDict[float]
    """GSE emissions for each species [g]."""

    total: EmissionsDict[float]
    """Total emissions for each species [g]."""

    fuel_burn_per_segment: np.ndarray
    """Fuel burn per trajectory segment [kg]."""

    total_fuel_burn: float
    """Total fuel burn for the mission [kg]."""

    lifecycle_co2_g: float = 0.0
    """Lifecycle CO2 emissions adjustment [g]."""

    @property
    def species(self) -> set[Species]:
        """Set of species included in any emissions."""
        return (
            set(self.trajectory.keys())
            | set(self.lto.keys())
            | set(self.apu.keys())
            | set(self.gse.keys())
            | set(self.total.keys())
        )


def compute_emissions(
    pm: BasePerformanceModel, fuel: Fuel, traj: Trajectory
) -> EmissionsOutput:
    """
    Compute all emissions.
    TODO: Expand docstring.
    """

    # Calculate cruise trajectory emissions (CO2, H2O, SOx, NOx, HC, CO, PM).
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

    emissions = EmissionsOutput(
        trajectory=trajectory.emissions,
        trajectory_indices=trajectory.indices,
        fuel_burn_per_segment=fuel_burn_per_segment,
        total_fuel_burn=float(total_fuel_burn),
        lto=lto.emissions,
        lto_indices=lto.indices,
        apu=apu.emissions,
        apu_indices=apu.indices,
        gse=gse.emissions,
        total=total_emissions,
    )

    lifecycle_adjustment = None
    if (
        Species.CO2 in config.emissions.enabled_species
        and config.emissions.lifecycle_enabled
    ):
        lifecycle_adjustment = get_lifecycle_emissions(fuel, traj)
        emissions.total[Species.CO2] += lifecycle_adjustment
        emissions.lifecycle_co2_g = lifecycle_adjustment

    return emissions


def sum_total_emissions(
    trajectory: EmissionsDict[np.ndarray],
    lto: EmissionsDict[ModeValues],
    apu: EmissionsDict[float],
    gse: EmissionsDict[float],
) -> EmissionsDict[float]:
    """
    Aggregate emissions (g) across all sources into summed_emission_g.
    Sums pointwise trajectory, LTO, APU, and GSE emissions for each species.
    """
    result = EmissionsDict[float]()
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
    """Apply lifecycle CO2 adjustments when requested by the config."""
    if fuel.lifecycle_CO2 is None:
        raise RuntimeError('Lifecycle CO2 data not available for selected fuel.')
    fuel_used = traj.fuel_mass[0] - traj.fuel_mass[-1]
    return float(fuel.lifecycle_CO2 * (fuel_used * fuel.energy_MJ_per_kg))
