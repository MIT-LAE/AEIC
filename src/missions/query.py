from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import ClassVar, Generic, TypeVar

import pandas as pd

from utils.helpers import date_to_timestamp

from .filter import Filter

T = TypeVar('T')


@dataclass
class QueryBase(ABC, Generic[T]):
    """Abstract base class for queries against the mission database."""

    filter: Filter | None = None
    """Flight filter to apply, or None for no filtering."""

    start_date: date | None = None
    """Include only flights on or after this date."""

    end_date: date | None = None
    """Include only flights on or before this date."""

    RESULT_TYPE: ClassVar[type]
    """Result type returned by a given query class."""

    PROCESS_RESULT: ClassVar[Callable | None] = None
    """Special processing function for results, or None if not needed."""

    def __init__(self, *_args, **_kwargs):
        if self.__class__ is QueryBase:
            raise NotImplementedError(
                'QueryBase is an abstract base class and '
                'cannot be instantiated directly.'
            )

    def __post_init__(self):
        # We maintain parallel lists of conditions and parameters to use to
        # build the SQL for the query.
        self._conditions = []
        self._params = []

    @abstractmethod
    def to_sql(self) -> tuple[str, list]:
        """Generate the SQL query string and parameters."""
        ...

    def _common_conditions(self):
        """Add conditions common to all queries."""

        if self.filter is not None:
            # Handle all filter conditions in one go here. The filter
            # conditions are on the flights table, which we alias as 'f' in the
            # main query.
            cond, p = self.filter.to_sql(table='f')
            if cond:
                self._conditions.append(cond)
                self._params.extend(p)

        if self.start_date is not None:
            # Midnight UTC of the given date.
            self._conditions.append('s.departure_timestamp >= ?')
            self._params.append(int(date_to_timestamp(self.start_date).timestamp()))
        if self.end_date is not None:
            # Midnight UTC of the day following the given date.
            self._conditions.append('s.departure_timestamp < ?')
            self._params.append(
                int((date_to_timestamp(self.end_date) + timedelta(days=1)).timestamp())
            )

    def _where_clause(self):
        """Generate WHERE clause from accumulated conditions."""
        return ' WHERE ' + ' AND '.join(self._conditions) if self._conditions else ''


@dataclass
class QueryResult:
    """A single flight query result."""

    departure: pd.Timestamp
    """Flight departure timestamp (UTC)."""

    arrival: pd.Timestamp
    """Flight arrival timestamp (UTC)."""

    carrier: str
    """Airline (IATA code)."""

    flight_number: str
    """Flight number."""

    origin: str
    """Origin airport (IATA code)."""

    origin_country: str
    """Origin country (ISO 3166-1 alpha-2 code)."""

    destination: str
    """Destination airport (IATA code)."""

    destination_country: str
    """Destination country (ISO 3166-1 alpha-2 code)."""

    service_type: str
    """Service type (IATA single-letter code, documented `here
    <https://knowledge.oag.com/v1/docs/iata-service-type-codes>`__)."""

    aircraft_type: str
    """Aircraft type (ICAO code)."""

    engine_type: str | None
    """Engine type, or None if not known."""

    distance: int
    """Flight distance in kilometers."""

    seat_capacity: int
    """Seat capacity."""

    flight_id: int
    """Unique flight ID."""

    @classmethod
    def from_row(cls, row: tuple) -> 'QueryResult':
        """Create a QueryResult from a database row."""

        return cls(
            departure=pd.Timestamp.utcfromtimestamp(row[0]),
            arrival=pd.Timestamp.utcfromtimestamp(row[1]),
            carrier=row[3],
            flight_number=row[4],
            origin=row[5],
            origin_country=row[6],
            destination=row[7],
            destination_country=row[8],
            service_type=row[9],
            aircraft_type=row[10],
            engine_type=row[11],
            distance=row[12],
            seat_capacity=row[13],
            flight_id=row[2],
        )


@dataclass
class Query(QueryBase[QueryResult]):
    """Query for scheduled flights."""

    every_nth: int | None = None
    """Include flights only from every nth day."""

    sample: float | None = None
    """Randomly sample a fraction of the results (0.0 < sample <= 1.0)."""

    RESULT_TYPE = QueryResult
    """Result type returned by this query class."""

    def to_sql(self) -> tuple[str, list]:
        """Generate the SQL query string and parameters."""

        # Validate parameters.
        if self.sample is not None and not (0.0 < self.sample <= 1.0):
            raise ValueError('sample frequency must be between 0.0 and 1.0.')
        if self.every_nth is not None and self.every_nth < 1:
            raise ValueError('every_nth must be at least 1.')

        # Handle filter and date conditions.
        self._common_conditions()

        # Random sampling: generate a random number in (0, 1) based on the
        # specification of SQLite's random() function.
        if self.sample is not None:
            self._conditions.append(
                '(random() + 9223372036854775808) / 18446744073709551615.0 < ?'
            )
            self._params.append(self.sample)

        # Return flights only on every nth day.
        if self.every_nth is not None and self.every_nth > 1:
            if self.start_date is None:
                self._conditions.append(
                    '(s.day - (SELECT MIN(day) FROM schedules)) % ? = 0'
                )
                self._params.append(self.every_nth)
            else:
                self._conditions.append('(s.day - ?) % ? = 0')
                self._params += [
                    (self.start_date - date(1970, 1, 1)).days,
                    self.every_nth,
                ]

        sql = (
            'SELECT s.departure_timestamp, s.arrival_timestamp, '
            'f.id as flight_id, f.carrier, f.flight_number, '
            'ao.iata_code AS origin, ao.country AS origin_country, '
            'ad.iata_code AS destination, ad.country AS destination_country, '
            'f.service_type, f.aircraft_type, f.engine_type, '
            'f.distance, f.seat_capacity '
            'FROM schedules s '
            'JOIN flights f ON f.id = s.flight_id '
            'JOIN airports ao ON f.origin = ao.id '
            'JOIN airports ad ON f.destination = ad.id'
            f'{self._where_clause()}'
            ' ORDER BY s.departure_timestamp'
        )

        return sql, self._params


@dataclass
class FrequentFlightQueryResult:
    """A single frequent flight query result."""

    airport1: str
    """First airport (IATA code)."""

    airport2: str
    """Second airport (IATA code)."""

    number_of_flights: int
    """Number of flights between the two airports."""

    @classmethod
    def from_row(cls, row: tuple) -> 'FrequentFlightQueryResult':
        """Create a FrequentFlightQueryResult from a database row."""

        return cls(airport1=row[0], airport2=row[1], number_of_flights=row[2])


@dataclass
class FrequentFlightQuery(QueryBase[FrequentFlightQueryResult]):
    """Query for the most frequent flight routes."""

    limit: int = 20
    """Maximum number of results to return (default 20)."""

    RESULT_TYPE = FrequentFlightQueryResult
    """Result type returned by this query class."""

    def to_sql(self) -> tuple[str, list]:
        """Generate the SQL query string and parameters."""

        # Validate parameters.
        if self.limit < 1:
            raise ValueError('result limit must be greater than zero')

        # Handle filter and date conditions.
        self._common_conditions()

        sql = (
            'WITH '
            'counts AS ('
            'SELECT COUNT(s.id) AS nflights, f.od_pair AS od_pair '
            'FROM schedules s '
            'JOIN flights f ON s.flight_id = f.id'
            f'{self._where_clause()}'
            ' GROUP BY od_pair) '
            'SELECT substring(od_pair, 1, 3) AS airport1, '
            'substring(od_pair, 4) AS airport2, '
            'nflights '
            'FROM counts '
            'ORDER BY nflights DESC '
            f'LIMIT {self.limit}'
        )

        return sql, self._params


@dataclass
class CountQuery(QueryBase[int]):
    """Count scheduled flights."""

    RESULT_TYPE = int
    """Result type returned by this query class."""

    PROCESS_RESULT = lambda _, gen: next(gen)[0]  # noqa

    def to_sql(self) -> tuple[str, list]:
        """Generate the SQL query string and parameters."""

        # Handle filter and date conditions.
        self._common_conditions()

        # Build the SQL query, shortcutting the common case of no conditions to
        # count all flight instances.
        sql = 'SELECT COUNT(s.id) FROM schedules s'
        if len(self._conditions) > 0:
            sql += (
                ' JOIN flights f ON f.id = s.flight_id '
                'JOIN airports ao ON f.origin = ao.id '
                'JOIN airports ad ON f.destination = ad.id'
                f'{self._where_clause()}'
            )

        return sql, self._params
