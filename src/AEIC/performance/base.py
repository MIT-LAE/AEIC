from abc import ABC, abstractmethod
from typing import Self

from pydantic import PositiveInt, model_validator

from AEIC.utils.models import CIBaseModel
from AEIC.utils.types import AircraftClass

from .apu import APU, find_apu
from .types import LTOPerformance, Speeds


class BasePerformanceModel(CIBaseModel, ABC):
    aircraft_name: str
    """Aircraft name ()."""

    aircraft_class: AircraftClass
    """Aircraft class (e.g., wide or narrow body)."""

    maximum_altitude_ft: PositiveInt
    """Aircraft maximum altitude in feet."""

    maximum_payload_kg: PositiveInt
    """Aircraft maximum payload in kilograms."""

    number_of_engines: PositiveInt
    """Number of engines."""

    apu_name: str
    """APU name."""

    speeds: Speeds | None
    """Optional speed data."""

    lto_performance: LTOPerformance | None
    """Optional LTO performance data."""

    @abstractmethod
    def fuel_flow(
        self, altitude: float, mass: float, rocd: float, speed: float
    ) -> float:
        """Calculate fuel flow based on flight parameters.

        Args:
            altitude (float): Altitude in meters.
            mass (float): Aircraft mass in kilograms.
            rocd (float): Rate of climb/descent in feet per minute.
            speed (float): True airspeed in meters per second.

        Returns:
            float: Fuel flow in kilograms per hour.
        """
        ...

    @model_validator(mode='after')
    def load_apu_data(self) -> Self:
        """Load APU data from APU database."""
        self._apu: APU = find_apu(self.apu_name)
        return self

    @property
    def apu(self) -> APU:
        """APU data associated with the performance model."""
        return self._apu
