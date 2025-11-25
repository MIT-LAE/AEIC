from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from utils.airports import airport
from utils.helpers import great_circle_distance, iso_to_timestamp
from utils.types import Position

from .query import QueryResult


@dataclass
class Mission:
    origin: str
    destination: str
    departure: pd.Timestamp
    arrival: pd.Timestamp
    load_factor: float
    aircraft_type: str
    flight_id: int | None = None

    @staticmethod
    def _airport_position(code: str) -> Position:
        ap = airport(code)
        if ap is None:
            raise ValueError(f'Unknown airport code: {code}')
        return ap.position

    @cached_property
    def origin_position(self) -> Position:
        return self._airport_position(self.origin)

    @cached_property
    def destination_position(self) -> Position:
        return self._airport_position(self.destination)

    @cached_property
    def gc_distance(self) -> float:
        """Great circle distance between departure and arrival positions (m)."""
        return great_circle_distance(
            self.origin_position.latitude,
            self.origin_position.longitude,
            self.destination_position.latitude,
            self.destination_position.longitude,
            degrees=True,
        )

    @classmethod
    def from_toml(cls, data: dict) -> list['Mission']:
        result = []
        for f in data['flight']:
            result.append(
                cls(
                    origin=f['origin'],
                    destination=f['destination'],
                    departure=iso_to_timestamp(f['departure']),
                    arrival=iso_to_timestamp(f['arrival']),
                    load_factor=f['load_factor'],
                    aircraft_type=f['aircraft_type'],
                )
            )
        return result

    @classmethod
    def from_query_result(cls, qr: QueryResult, load_factor: float = 1.0) -> 'Mission':
        return cls(
            origin=qr.origin,
            destination=qr.destination,
            departure=qr.departure,
            arrival=qr.arrival,
            load_factor=load_factor,  # (real load factor not in QueryResult)
            aircraft_type=qr.aircraft_type,
            flight_id=qr.id,  # The schedule ID is unique across the database
        )
