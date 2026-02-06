# TODO: Remove when we move to Python 3.14+.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import ConfigDict

from AEIC.utils.models import CIBaseModel, CIStrEnum


class ThrustMode(CIStrEnum):
    """Flight modes for LTO data.

    The enumeration values here are ordered by the format of LTO files."""

    IDLE = 'idle'
    APPROACH = 'approach'
    CLIMB = 'climb'
    TAKEOFF = 'takeoff'

    # TODO: Is this the best place for this?
    @property
    def thrust_percentage(self) -> float:
        match self:
            case ThrustMode.IDLE:
                return 7.0
            case ThrustMode.APPROACH:
                return 30.0
            case ThrustMode.TAKEOFF | ThrustMode.CLIMB:
                # TODO: Is this right?
                return 85.0


@dataclass(frozen=True)
class ThrustModeArray:
    data: np.ndarray

    def __post_init__(self):
        if not np.isin(self.data, [cat.value for cat in ThrustMode]).all():
            raise ValueError('ThrustModeArray data contains invalid ThrustMode values.')

    def as_enum(self) -> np.ndarray:
        return np.vectorize(ThrustMode)(self.data)

    def __array__(self):
        return self.data

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def modes(cls) -> ThrustModeArray:
        """Create a ThrustModeArray containing all thrust modes."""
        return cls(np.array([m.value for m in ThrustMode]))

    @property
    def shape(self):
        return self.data.shape


class ThrustModeValues(Mapping):
    """Specialized dictionary for LTO values keyed by ThrustMode."""

    _data: dict[ThrustMode, float]

    # These values are used for a lot of configuration data, so we often want
    # to make them immutable to prevent things getting modified
    # unintentionally.
    _mutable: bool

    def __init__(self, *args, mutable: bool = False):
        self._mutable = mutable
        if len(args) == 0:
            self._data = {}
        elif len(args) == 1 and isinstance(args[0], dict | ThrustModeValues):
            self._data = args[0]  # type: ignore (generic dict vs ThrustModeValues)
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self._data = {m: args[0][i] for i, m in enumerate(ThrustMode)}
        elif len(args) == 1 and isinstance(args[0], float | np.floating):
            self._data = {m: float(args[0]) for m in ThrustMode}
        elif len(args) == 4:
            self._data = {m: args[i] for i, m in enumerate(ThrustMode)}
        else:
            raise ValueError('Invalid initialization of ThrustModeValues.')

    def __getitem__(self, mode: ThrustMode) -> float:
        if mode not in self._data:
            return 0.0
        return self._data[mode]

    def __setitem__(self, mode: ThrustMode, value: float | np.floating) -> None:
        if not self._mutable:
            raise TypeError(
                'ThrustModeValues instance is frozen and cannot be modified.'
            )
        self._data[mode] = float(value)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return f'ThrustModeValues({self._data})'

    def __repr__(self):
        return f'ThrustModeValues({repr(self._data)})'

    def __hash__(self):
        return hash(tuple(self._data.items()))

    def freeze(self):
        """Make a ThrustModeValues dictionary immutable."""
        self._mutable = False

    def copy(self, mutable: bool | None = None):
        """Copy a ThrustModeValues dictionary, optionally modifying the mutability.
        This is the only way to go from an immutable to a mutable instance."""
        new_mutable = self._mutable if mutable is None else mutable
        return ThrustModeValues(self._data.copy(), mutable=new_mutable)

    def as_array(self) -> np.ndarray:
        return np.array([self._data[m] for m in ThrustMode])

    def sum(self) -> float:
        return sum(self._data.values())

    def __add__(self, other: ThrustModeValues | float | int) -> ThrustModeValues:
        if isinstance(other, ThrustModeValues):
            return ThrustModeValues(
                {m: v + other[m] for m, v in self._data.items()}, mutable=True
            )
        if isinstance(other, float | int):
            return ThrustModeValues(
                {m: v + other for m, v in self._data.items()}, mutable=True
            )
        return NotImplemented

    def __radd__(self, other) -> ThrustModeValues:
        return self.__add__(other)

    def __mul__(self, other: ThrustModeValues | float | int) -> ThrustModeValues:
        if isinstance(other, ThrustModeValues):
            return ThrustModeValues(
                {m: v * other[m] for m, v in self._data.items()}, mutable=True
            )
        if isinstance(other, float | int):
            return ThrustModeValues(
                {m: v * other for m, v in self._data.items()}, mutable=True
            )
        return NotImplemented

    def __rmul__(self, other) -> ThrustModeValues:
        return self.__mul__(other)

    def __truediv__(self, other: ThrustModeValues | float) -> ThrustModeValues:
        if isinstance(other, ThrustModeValues):
            return ThrustModeValues(
                {m: v / other[m] for m, v in self._data.items()}, mutable=True
            )
        if isinstance(other, float | int):
            return ThrustModeValues(
                {m: v / other for m, v in self._data.items()}, mutable=True
            )
        return NotImplemented

    def __or__(self, other: dict) -> ThrustModeValues:
        return ThrustModeValues(
            {m: v + other.get(m, 0.0) for m, v in self._data.items()}, mutable=True
        )

    def broadcast(self, modes: ThrustModeArray) -> np.ndarray:
        """Broadcast ThrustModeValues to an array according to the provided
        ThrustModeArray."""
        result = np.empty(modes.shape, dtype=float)
        for m in ThrustMode:
            result[modes.data == m.value] = self[m]
        return result

    def isclose(self, other: ThrustModeValues) -> bool:
        """Check if two ThrustModeValues instances are close to each other."""
        for m in ThrustMode:
            if not np.isclose(self[m], other[m]):
                return False
        return True


@dataclass
class AircraftState:
    """Aircraft state container for performance model inputs.

    The fundamental inputs required for performance model evaluation are
    altitude and aircraft mass. Depending on the performance model being used,
    true airspeed and rate of climb/descent may also be required."""

    altitude: float
    """Altitude [m]."""

    aircraft_mass: float | Literal['min', 'max']
    """Aircraft total mass [kg]. Can also be 'min' or 'max' to indicate
    minimum or maximum aircraft mass."""

    true_airspeed: float | None = None
    """True airspeed [m/s]. Whether this needs to be provided depends on the
    performance model being used."""

    rate_of_climb: float | None = None
    """Rate of climb/descent [m/s]. Whether this needs to be provded depends
    on the performance model being used."""


@dataclass
class Performance:
    """Aircraft performance outputs from performance model."""

    true_airspeed: float
    """Actual achievable true airspeed [m/s]."""

    rate_of_climb: float
    """Actual achievable rate of climb/descent [m/s]."""

    fuel_flow: float
    """Fuel flow in [kg/s]."""


class SimpleFlightRules(CIStrEnum):
    """Flight rules class for simplest case, where the only distinction is
    between climb, cruise, and descent."""

    CLIMB = 'climb'
    CRUISE = 'cruise'
    DESCEND = 'descend'


@dataclass
class LTOPerformance:
    """LTO performance data as used internally."""

    source: str
    """Source of LTO data (e.g., 'EDB' or 'BADA LTO file'). For documentation
    only."""

    ICAO_UID: str
    """ICAO engine ID from engine database (EDB). For documentation only."""

    rated_thrust: float
    """Engine rated thrust [N]."""

    thrust_pct: ThrustModeValues
    """Thrust percentage in thrust mode (0-100)."""

    fuel_flow: ThrustModeValues
    """Fuel flow rate in thrust mode [kg/s]."""

    EI_NOx: ThrustModeValues
    """Emission index for NOₓ in thrust mode [g/kg fuel]."""

    EI_HC: ThrustModeValues
    """Emission index for HC in thrust mode [g/kg fuel]."""

    EI_CO: ThrustModeValues
    """Emission index for CO in thrust mode [g/kg fuel]."""


class SpeedData(CIBaseModel):
    """Performance model speed data for different flight phases."""

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    cas_low: float
    """Low speed calibrated airspeed (CAS) [m/s]."""

    cas_high: float
    """High speed calibrated airspeed (CAS) [m/s]."""

    mach: float
    """Mach number."""


class Speeds(CIBaseModel):
    """Speeds for different flight phases."""

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    climb: SpeedData
    """Speed data for climb phase."""

    cruise: SpeedData
    """Speed data for cruise phase."""

    descent: SpeedData
    """Speed data for descent phase."""
