# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from typing import cast

import pandas as pd

from AEIC.types import Position
from AEIC.utils import GEOD
from AEIC.utils.airports import airport

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
        return GEOD.inv(
            self.origin_position.latitude,
            self.origin_position.longitude,
            self.destination_position.latitude,
            self.destination_position.longitude,
        )[2]

    @property
    def label(self) -> str:
        """Label for mission, based on origin, destination, and aircraft type."""
        return f'{self.origin}_{self.destination}_{self.aircraft_type}'

    @classmethod
    def from_toml(cls, data: dict) -> list[Mission]:
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
    def from_query_result(cls, qr: QueryResult, load_factor: float = 1.0) -> Mission:
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


def iso_to_timestamp(s: str) -> pd.Timestamp:
    """Convert an ISO 8601 string to a UTC Pandas `Timestamp`."""
    ts = cast(pd.Timestamp, pd.Timestamp(datetime.fromisoformat(s)))
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(UTC)
