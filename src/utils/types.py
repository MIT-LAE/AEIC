from dataclasses import dataclass
from enum import Enum
from typing import Self

import numpy as np
import pandas as pd

# from numpy import ndarray as NDArray
from numpy.typing import NDArray

# create a type for Union[float, NDArray]
FloatOrNDArray = float | NDArray[np.float64]


class DayOfWeek(Enum):
    """Days of the week, with Monday=1 through Sunday=7."""

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7

    @classmethod
    def from_pandas(cls, t: pd.Timestamp) -> Self:
        """Extract day of week from a pandas `Timestamp`."""
        return cls(t.isoweekday())


@dataclass
class TimeOfDay:
    """A time of day as hours and minutes."""

    hour: int
    """Hour of day, 0-23."""

    minute: int
    """Minute of hour, 0-59."""


@dataclass
class Location:
    """A geographic location defined by longitude and latitude."""

    longitude: float
    """Longitude in decimal degrees."""

    latitude: float
    """Latitude in decimal degrees."""


@dataclass
class Position:
    """An aircraft position defined by longitude, latitude, and altitude."""

    longitude: float
    """Longitude in decimal degrees."""

    latitude: float
    """Latitude in decimal degrees."""

    altitude: float
    """Altitude in meters above sea level."""

    @property
    def location(self) -> Location:
        """Get the 2D location (longitude and latitude) of this position."""
        return Location(longitude=self.longitude, latitude=self.latitude)
