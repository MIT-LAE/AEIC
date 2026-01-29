# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from AEIC.constants import R_air, kappa
from AEIC.utils.standard_atmosphere import (
    pressure_at_altitude_isa_bada4,
    temperature_at_altitude_isa_bada4,
)

if TYPE_CHECKING:
    from AEIC.types import Species


class EmissionsDict[M]:
    """Typed mapping of species to emission values."""

    def __init__(self, data: dict[Species, M] | None = None):
        if data is None:
            data = {}
        self._data: dict[Species, M] = data

    def __getitem__(self, key: Species) -> M:
        return self._data[key]

    def __setitem__(self, key: Species, value: M) -> None:
        self._data[key] = value

    def __contains__(self, key: Species) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, other: EmissionsDict[M]):
        self._data.update(other._data)

    def __repr__(self) -> str:
        return (
            '<'
            + self.__class__.__name__
            + ': '
            + ', '.join([str(s.name) for s in self._data.keys()])
            + '>'
        )


@dataclass
class EmissionsSubset[M]:
    indices: EmissionsDict[M] = field(default_factory=lambda: EmissionsDict[M]())
    emissions: EmissionsDict[M] = field(default_factory=lambda: EmissionsDict[M]())
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
