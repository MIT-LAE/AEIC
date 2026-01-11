# TODO: Remove when we move to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from AEIC.utils.models import CIBaseModel, CIStrEnum


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
    # TODO: Units?
    """Rate of climb/descent [m/s]. Whether this needs to be privded depends on
    the performance model being used."""


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


class LTOThrustMode(CIStrEnum):
    """Flight modes for LTO data.

    The enumeration values here are ordered by the format of LTO files."""

    IDLE = 'idle'
    APPROACH = 'approach'
    CLIMB = 'climb'
    TAKEOFF = 'takeoff'


class LTOModeData(CIBaseModel):
    """LTO data for a given flight phase."""

    # TODO: Better docstrings.
    thrust_frac: float
    fuel_kgs: float
    EI_NOx: float
    EI_HC: float
    EI_CO: float


class LTOPerformance(CIBaseModel):
    """LTO performance data."""

    # TODO: Make source an enum? EDB or in-file?
    # TODO: Docstrings.
    source: str
    ICAO_UID: str
    Foo_kN: float
    mode_data: dict[LTOThrustMode, LTOModeData]


@dataclass
class LTOInputs:
    """LTO data container for fuel flow and emission indices.

    Each entry here is an array of length 4 corresponding to the four LTO
    thrust modes: idle, approach, climb, takeoff, *in that order*."""

    fuel_flow: np.ndarray
    thrust_pct: np.ndarray
    EI_NOx: np.ndarray
    EI_HC: np.ndarray
    EI_CO: np.ndarray

    def __post_init__(self):
        if not (
            len(self.fuel_flow)
            == len(self.thrust_pct)
            == len(self.EI_NOx)
            == len(self.EI_HC)
            == len(self.EI_CO)
            == 4
        ):
            raise ValueError('All LTO input arrays must be of length 4.')

    @classmethod
    def from_performance(cls, perf: LTOPerformance) -> LTOInputs:
        """Create LTOInputs from a dict of LTOModeData."""
        ordered = [(m, perf.mode_data[m]) for m in LTOThrustMode]

        return cls(
            fuel_flow=np.array([m.fuel_kgs for _, m in ordered]),
            EI_NOx=np.array([m.EI_NOx for _, m in ordered]),
            EI_HC=np.array([m.EI_HC for _, m in ordered]),
            EI_CO=np.array([m.EI_CO for _, m in ordered]),
            thrust_pct=np.array([m.thrust_frac * 100.0 for _, m in ordered]),
        )


class SpeedData(CIBaseModel):
    """Performance model speed data for different flight phases."""

    cas_low: float
    """Low speed calibrated airspeed (CAS) [m/s]."""

    cas_high: float
    """High speed calibrated airspeed (CAS) [m/s]."""

    mach: float
    """Mach number."""


class Speeds(CIBaseModel):
    """Speeds for different flight phases."""

    climb: SpeedData
    """Speed data for climb phase."""

    cruise: SpeedData
    """Speed data for cruise phase."""

    descent: SpeedData
    """Speed data for descent phase."""
