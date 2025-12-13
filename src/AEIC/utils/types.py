from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, Self, get_args, get_origin

import numpy as np
import pandas as pd

# from numpy import ndarray as NDArray
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator

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


# ------------------------------------------------------------------------------
#
#  TODO: Decide on better places to put the following things.


class CIStrEnum(StrEnum):
    """Case-insensitive string enumeration."""

    def __str__(self):
        """Normalize on output."""
        return self.value.lower()

    @classmethod
    def _missing_(cls, value):
        """Normalize on input."""
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value.lower() == value:
                    return member
        return None


class CIBaseModel(BaseModel):
    """Base model that recursively normalizes input keys to match lower-case
    model field names."""

    @classmethod
    def _normalize_dict(cls, values: dict) -> dict:
        """Recursively normalize keys of a dict to match model fields."""
        normalized = {}
        field_map = {f.lower(): f for f in cls.model_fields}

        for k, v in values.items():
            field_name = field_map.get(k.lower(), k)
            if field_name is None:
                continue
            field_info = cls.model_fields.get(field_name)

            # If the field is itself a Pydantic model, recurse
            if field_info is not None:
                field_type = field_info.annotation
                origin = get_origin(field_type)
                args = get_args(field_type)

                # Nested BaseModel
                if (
                    isinstance(v, dict)
                    and isinstance(field_type, type)
                    and issubclass(field_type, BaseModel)
                ):
                    v = field_type.model_validate(cls._normalize_dict(v))

                # List of BaseModels
                elif (
                    origin is list
                    and args
                    and issubclass(args[0], BaseModel)
                    and isinstance(v, list)
                ):
                    v = [
                        args[0].model_validate(cls._normalize_dict(item))
                        if isinstance(item, dict)
                        else item
                        for item in v
                    ]

            normalized[field_name] = v

        return normalized

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return cls._normalize_dict(values)
        elif isinstance(values, list):
            # Normalize each item if it's a dict
            return [
                cls._normalize_dict(v) if isinstance(v, dict) else v for v in values
            ]
        else:
            return values


#
# ------------------------------------------------------------------------------


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
