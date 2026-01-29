from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from AEIC.types import Position
from AEIC.utils.airports import airport
from AEIC.utils.helpers import iso_to_timestamp
from AEIC.utils.spatial import great_circle_distance

from .query import QueryResult


@dataclass
class Mission:
    origin: str
    """IATA code of origin airport."""

    destination: str
    """IATA code of destination airport."""

    departure: pd.Timestamp
    """Departure time (UTC)."""

    arrival: pd.Timestamp
    """Arrival time (UTC)."""

    load_factor: float
    """Load factor (between 0 and 1)."""

    aircraft_type: str
    """Aircraft type (ICAO code)."""

    flight_id: int | None = None
    """Unique flight identifier from mission database (if available)."""

    @staticmethod
    def _airport_position(code: str) -> Position:
        ap = airport(code)
        if ap is None:
            raise ValueError(f'Unknown airport code: {code}')
        return ap.position

    @cached_property
    def origin_position(self) -> Position:
        """Spatial position (3-D) of origin airport."""
        return self._airport_position(self.origin)

    @cached_property
    def destination_position(self) -> Position:
        """Spatial position (3-D) of destination airport."""
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
        """Create a list of `Mission` instances from a TOML-like dictionary.

        This is used for parsing sample mission data.
        """

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
        """Create a `Mission` instance from a `QueryResult` instance.

        This is used for generating missions from mission database queries.
        """
        return cls(
            origin=qr.origin,
            destination=qr.destination,
            departure=qr.departure,
            arrival=qr.arrival,
            load_factor=load_factor,  # (real load factor not in QueryResult)
            aircraft_type=qr.aircraft_type,
            flight_id=qr.id,  # The schedule ID is unique across the database
        )
