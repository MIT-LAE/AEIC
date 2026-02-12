# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from AEIC.constants import R_air, kappa
from AEIC.types import SpeciesValues
from AEIC.utils.standard_atmosphere import (
    pressure_at_altitude_isa_bada4,
    temperature_at_altitude_isa_bada4,
)


@dataclass
class EmissionsSubset[M]:
    indices: SpeciesValues[M] = field(default_factory=lambda: SpeciesValues[M]())
    emissions: SpeciesValues[M] = field(default_factory=lambda: SpeciesValues[M]())
    fuel_burn: float = 0.0


@dataclass
class AtmosphericState:
    """Per-segment atmospheric state along a trajectory (used by EI models)."""

    temperature: np.ndarray
    """Temperature at trajectory points [K]."""

    pressure: np.ndarray
    """Pressure at trajectory points [Pa]."""

    mach: np.ndarray
    """Mach number at trajectory points."""

    def __init__(
        self,
        altitude: np.ndarray,
        tas: np.ndarray,
    ):
        """Compute temperature, pressure, and Mach profiles when needed."""
        self.temperature = temperature_at_altitude_isa_bada4(altitude)
        self.pressure = np.array(pressure_at_altitude_isa_bada4(altitude))
        self.mach = tas / np.sqrt(kappa * R_air * self.temperature)
