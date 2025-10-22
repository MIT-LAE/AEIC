from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from utils import GEOD
from utils.custom_types import Position
from utils.helpers import iso_to_timestamp

from .database import Database  # noqa
from .filter import BoundingBox, Filter  # noqa
from .query import CountQuery, FrequentFlightQuery, Query  # noqa


@dataclass
class Mission:
    dep_airport: str
    arr_airport: str
    dep_position: Position
    arr_position: Position
    dep_datetime: pd.Timestamp
    arr_datetime: pd.Timestamp
    load_factor: float
    ac_code: str

    @cached_property
    def gc_distance(self) -> float:
        """Great circle distance between departure and arrival positions (m)."""
        return GEOD.inv(
            self.dep_position.longitude,
            self.dep_position.latitude,
            self.arr_position.longitude,
            self.arr_position.latitude,
        )[2]

    @classmethod
    def from_toml(cls, data: dict) -> list['Mission']:
        result = []
        for f in data['flight']:
            dep_pos = Position(
                longitude=f['dep_location'][0],
                latitude=f['dep_location'][1],
                altitude=f['dep_location'][2],
            )
            arr_pos = Position(
                longitude=f['arr_location'][0],
                latitude=f['arr_location'][1],
                altitude=f['arr_location'][2],
            )
            result.append(
                cls(
                    dep_airport=f['dep_airport'],
                    arr_airport=f['arr_airport'],
                    dep_position=dep_pos,
                    arr_position=arr_pos,
                    dep_datetime=iso_to_timestamp(f['dep_datetime']),
                    arr_datetime=iso_to_timestamp(f['arr_datetime']),
                    load_factor=f['load_factor'],
                    ac_code=f['ac_code'],
                )
            )
        return result
