import contextlib
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

import pandas as pd

import utils.airports as airports
from utils.spatial import great_circle_distance
from utils.types import DayOfWeek, TimeOfDay

from .database import Database

logger = logging.getLogger(__name__)


@dataclass
class AirportInfo:
    """Information about an airport in the database.

    This is a tuple of the airport ID (primary key in the airports table) and
    an Airport instance with the airport data.
    """

    id: int
    """Primary key ID of the airport in the airports table."""

    airport: airports.Airport
    """Airport data."""

    timezone: str
    """Timezone string for the airport."""


@dataclass
class Warning:
    class Type(str, Enum):
        UNKNOWN_AIRPORT = 'unknown airport code'
        TIME_MISORDERING = 'arrival time before departure time'
        SUSPICIOUS_DISTANCE = 'suspicious distance'
        ZERO_DISTANCE = 'zero distance'

    warn_type: Type
    data: dict | None = None

    def __str__(self) -> str:
        details = ''
        data = self.data
        if data is None:
            data = {}
        match self.warn_type:
            case Warning.Type.UNKNOWN_AIRPORT:
                details = data['unknown_airport']
            case Warning.Type.TIME_MISORDERING:
                origin = data['origin']
                destination = data['destination']
                departure_time = data['departure_time']
                arrival_time = data['arrival_time']
                arrival_day_offset = data['arrival_day_offset']
                details = (
                    f'{origin.airport.iata_code} @ {departure_time} '
                    f'({origin.timezone}) => '
                    f'{destination.airport.iata_code} @ {arrival_time} '
                    f'({destination.timezone}) '
                    f'(+{arrival_day_offset})'
                )
            case Warning.Type.SUSPICIOUS_DISTANCE:
                origin = data['origin']
                destination = data['destination']
                given = data['given_distance_km']
                calculated = data['calculated_distance_km']
                delta = 100 * (given - calculated) / calculated
                details = (
                    f'{origin.airport.iata_code} => '
                    f'{destination.airport.iata_code} '
                    f'(given = {given} km, '
                    f'calculated = {calculated} km, '
                    f'delta = {delta:.1f}%)'
                )
            case Warning.Type.ZERO_DISTANCE:
                origin = data['origin']
                destination = data['destination']
                details = (
                    f'{origin.airport.iata_code} => {destination.airport.iata_code}'
                )
        return f'{self.warn_type.value} - {details}'


class WritableDatabase(Database):
    """Writable flight schedule database.

    Represents a database of flight schedule entries, stored in an SQLite
    database file, using a schema optimized for common AEIC query use cases.
    """

    def __init__(self, db_path: str):
        """Open a mission database file for writing.

        Parameters
        ----------

        db_path : str
            Path to the SQLite database file.
        """

        super().__init__(db_path)

        # We maintain in-memory caches of invariant airport and country data to
        # avoid repeated database lookups while adding entries.
        self._airport_cache: dict[str, AirportInfo] = {}
        self._country_cache: set[str] = set()

        # Maintain a record of unknown IATA airport codes that we see for later
        # diagnostic output.
        self.unknown_airports: set[str] = set()

        # Collect warnings about data issues to report later (only one per
        # line of the input file).
        self.warnings: dict[int, Warning] = {}

        # Lazily initialized timezonefinder instance for looking up airport
        # time zones.
        self._timezonefinder = None

        # Create the tables if needed.
        self._ensure_schema(indexes=False)

    def _check_path(self, db_path: str):
        """Check that the database file does not already exist."""
        if os.path.exists(db_path):
            raise RuntimeError(f'Database file {db_path} already exists.')

    def commit(self):
        self._conn.commit()

    def index(self):
        """Generate database indexes and optimize for queries.

        This should be called after all entries have been added to the
        database.
        """

        # Vacuuming the database rebuilds the database file, repacking it into
        # a minimal amount of disk space and optimizing the layout for access.
        logger.info('Vacuuming database')
        cur = self._conn.cursor()
        cur.execute('VACUUM')

        # Generate indexes required to optimize common query patterns.
        index_data = [
            ('flights', 'carrier', False),
            ('flights', 'origin', False),
            ('flights', 'destination', False),
            ('flights', 'aircraft_type', False),
            ('flights', 'aircraft_type', False),
            ('flights', 'distance', False),
            ('flights', 'seat_capacity', False),
            ('flights', 'od_pair', False),
            ('schedules', 'departure_timestamp', False),
            ('schedules', 'flight_id', False),
            ('airports', 'iata_code', True),
            ('airports', 'country', False),
        ]
        for table, column, unique in index_data:
            logger.info('Creating index on %s.%s', table, column)
            cur.execute(f"""
              CREATE {'UNIQUE' if unique else ''} INDEX IF NOT EXISTS
                  {table}_{column}_idx ON {table}({column})""")

        # This helps SQLite optimize queries using the indexes by storing
        # statistical information about the indexes in internal SQLite tables.
        logger.info('Analyzing database')
        cur.execute('ANALYZE')

    def _warn(self, warn_type: Warning.Type, line: int, **kwargs):
        """Record a warning about a data issue."""
        self.warnings[line] = Warning(warn_type, kwargs)

    def _add_flight(
        self,
        cur: sqlite3.Cursor,
        carrier: str,
        flight_number: int,
        origin: AirportInfo,
        destination: AirportInfo,
        effective_from: date,
        effective_to: date,
        days: set[DayOfWeek],
        departure_time: TimeOfDay,
        arrival_time: TimeOfDay,
        arrival_day_offset: int,
        service_type: str,
        aircraft_type: str,
        seat_capacity: int,
        distance_km: float,
    ) -> int:
        """Add a flight entry to the database.

        This fills in the flight entry, but does not add any of the schedule
        entries that correspond to the flight. That is done separately. Filling
        the number_of_flights field has to be deferred until the scheduled
        flights have been added.
        """

        fields = [
            'carrier',
            'flight_number',
            'origin',
            'destination',
            'day_of_week_mask',
            'departure_time',
            'arrival_time',
            'arrival_day_offset',
            'service_type',
            'aircraft_type',
            'engine_type',
            'distance',
            'seat_capacity',
            'effective_from',
            'effective_to',
            'number_of_flights',
            'od_pair',
        ]
        od_pair = min(origin.airport.iata_code, destination.airport.iata_code) + max(
            origin.airport.iata_code, destination.airport.iata_code
        )
        fs = ', '.join(fields)
        qs = ', '.join(['?'] * len(fields))
        cur.execute(
            f'INSERT INTO flights ({fs}) VALUES ({qs}) RETURNING id',
            (
                carrier,
                flight_number,
                # Foreign keys to airports table.
                origin.id,
                destination.id,
                # Bitmask for days of week.
                self._make_dow_mask(days),
                # Both times are minutes since midnight, local time.
                departure_time.hour * 60 + departure_time.minute,
                arrival_time.hour * 60 + arrival_time.minute,
                arrival_day_offset,
                service_type,
                aircraft_type,
                # Engine type not available.
                '',
                # SI units in database.
                distance_km,
                seat_capacity,
                # Effective dates as YYYY-MM-DD strings.
                effective_from.isoformat(),
                effective_to.isoformat(),
                # Number of flights to be computed later from schedule.
                0,
                # Direction independent OD pair.
                od_pair,
            ),
        )
        row = cur.fetchone()
        assert row is not None

        # Return the ID of the newly added flight entry for use in schedules
        # table.
        return row[0]

    def _add_schedule(
        self,
        cur: sqlite3.Cursor,
        line: int,
        flight_id: int,
        origin: AirportInfo,
        destination: AirportInfo,
        effective_from: date | None,
        effective_to: date | None,
        days: set[DayOfWeek],
        departure_time: TimeOfDay,
        arrival_time: TimeOfDay,
        arrival_day_offset: int,
    ) -> int:
        """Add schedule entries for a flight."""

        # Collect the schedule entries to add. (It's more efficient to do this
        # and to use executemany to add them to the database in one go.)
        data = []

        # Run over the (inclusive) effective date range.
        for flight_date in pd.date_range(effective_from, effective_to):
            # Skip days not in the day-of-week set for this flight.
            if DayOfWeek.from_pandas(flight_date) not in days:
                continue

            # Calculate departure and arrival timestamps from current date and
            # given departure and arrival times for flight. The departure and
            # arrival times are in local times at the respective airports.

            # Make departure datetime in the local time zone of the origin
            # airport.
            dep_time = (
                flight_date
                + timedelta(hours=departure_time.hour, minutes=departure_time.minute)
            ).replace(tzinfo=ZoneInfo(origin.timezone))

            # Make arrival datetime in the local time zone of the destination
            # airport. Here, we also need to add the arrival day offset.
            arr_time = (
                flight_date
                + timedelta(
                    days=arrival_day_offset,
                    hours=arrival_time.hour,
                    minutes=arrival_time.minute,
                )
            ).replace(tzinfo=ZoneInfo(destination.timezone))

            # Extract timestamps in seconds since Unix epoch (1970-01-01). Note
            # that these are independent of the local time zones since they are
            # simple intervals from a fixed point in time.
            dep_timestamp = int(dep_time.timestamp())
            arr_timestamp = int(arr_time.timestamp())

            # Since we have ostensibly taken account of the different time
            # zones at the origin and destination and we have also taken
            # account of the arrival day offset, the departure and arrival
            # timestamps should be in the right order! (If they are not, this
            # may indicate a miscoded airport, a problem with the stated
            # departure or arrrival times, or a problem with timezone
            # determination for an airport location.)
            if arr_timestamp < dep_timestamp:
                self._warn(
                    Warning.Type.TIME_MISORDERING,
                    line,
                    origin=origin,
                    destination=destination,
                    departure_time=departure_time,
                    arrival_time=arrival_time,
                    arrival_day_offset=arrival_day_offset,
                )
                continue

            # Calculate day number since Unix epoch (1970-01-01). This is used
            # for "every N days" sampling.
            day = (flight_date - datetime(1970, 1, 1)).days

            data.append((dep_timestamp, arr_timestamp, day, flight_id))

        if len(data) > 0:
            cur.executemany(
                """INSERT INTO schedules (
                    departure_timestamp, arrival_timestamp, day, flight_id
                ) VALUES (?, ?, ?, ?)""",
                data,
            )

        # We return the number of schedule entries added so that we can fill in
        # the "number of flights" field in the flights table.
        return len(data)

    def _set_flight_count(self, cur: sqlite3.Cursor, flight_id: int, num_flights: int):
        cur.execute(
            'UPDATE flights SET number_of_flights = ? WHERE id = ?',
            (num_flights, flight_id),
        )

    def _get_or_add_airport(
        self, cur: sqlite3.Cursor, line: int, iata_code: str
    ) -> AirportInfo | None:
        """Retrieve or add an airport entry."""

        # If we already have this airport in the cache, return it.
        if iata_code in self._airport_cache:
            return self._airport_cache[iata_code]

        # Try to find the airport in the database.
        cur.execute(
            """SELECT id, iata_code, name,
                      latitude, longitude, elevation,
                      country, municipality
                 FROM airports WHERE iata_code = ?""",
            (iata_code,),
        )
        row = cur.fetchone()
        if row:
            airport_id = row[0]
            airport = airports.Airport(
                iata_code=row[1],
                name=row[2],
                latitude=row[3],
                longitude=row[4],
                elevation=row[5],
                country=row[6],
                municipality=row[7],
            )
            return self._cache_airport(airport_id, airport)

        # Not found, so look up the airport in the static data.
        airport = airports.airport(iata_code)
        if not airport:
            self.unknown_airports.add(iata_code)
            self._warn(Warning.Type.UNKNOWN_AIRPORT, line, unknown_airport=iata_code)
            return None
        assert isinstance(airport, airports.Airport)

        # Ensure the country is in the database.
        country = self._get_or_add_country(cur, airport.country)

        # Add the airport entry.
        cur.execute(
            """INSERT INTO airports (
                iata_code, name, municipality, country,
                latitude, longitude, elevation
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id""",
            (
                airport.iata_code,
                airport.name,
                airport.municipality,
                country,
                airport.latitude,
                airport.longitude,
                airport.elevation,
            ),
        )
        airport_id = cur.fetchone()[0]

        # Add airport location to the spatial R-tree index.
        cur.execute(
            """INSERT INTO airport_location_idx (
                id, min_latitude, max_latitude,
                min_longitude, max_longitude
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                airport_id,
                airport.latitude,
                airport.latitude,
                airport.longitude,
                airport.longitude,
            ),
        )

        # Cache the airport data and return the airport ID.
        return self._cache_airport(airport_id, airport)

    def _cache_airport(self, airport_id, airport):
        """Cache airport data after it has been added to the database."""

        tz = self._lookup_timezone(airport)
        self._airport_cache[airport.iata_code] = AirportInfo(airport_id, airport, tz)
        return self._airport_cache[airport.iata_code]

    def _lookup_timezone(self, airport: airports.Airport) -> str:
        """Look up the time zone for an airport."""

        # Lazy setup of timezonefinder instance.
        if self._timezonefinder is None:
            from timezonefinder import TimezoneFinder

            self._timezonefinder = TimezoneFinder()

        # Use most exacting timezone lookup from timezonefinder. This is very
        # reliable, so any errors here are almost certainly due to bad airport
        # data.
        assert self._timezonefinder is not None
        tz = self._timezonefinder.certain_timezone_at(
            lat=airport.latitude, lng=airport.longitude
        )
        if tz is None:
            raise ValueError(
                f'Could not determine timezone for airport {airport.iata_code}'
            )
        return tz

    def _get_or_add_country(self, cur: sqlite3.Cursor, code: str) -> str:
        """Retrieve or add a country entry."""

        # If we already have this country in the cache, return it.
        if code in self._country_cache:
            return code

        # Try to find the country in the database.
        cur.execute('SELECT code FROM countries WHERE code = ?', (code,))
        row = cur.fetchone()
        if row:
            self._country_cache.add(code)
            return row[0]

        # Not found, so look up the country in the static data.
        country = airports.country(code)
        assert isinstance(country, airports.Country)
        if not country:
            raise ValueError(f'Unknown country code: {code}')

        # Add the country entry to the database, cache it, and return the code.
        cur.execute(
            """INSERT INTO countries (code, name, continent)
               VALUES (?, ?, ?)""",
            (country.code, country.name, country.continent),
        )
        self._country_cache.add(country.code)
        return country.code

    @staticmethod
    def _make_dow_mask(days: set[DayOfWeek]) -> int:
        """Create a bitmask for the given set of days of the week."""
        return sum(1 << (day.value - 1) for day in days)

    def _ensure_schema(self, indexes: bool = True):
        """Create database tables if they don't already exist.

        (See "OAG database" page in the wiki for more details of the schema.)
        """

        cur = self._conn.cursor()

        logger.info('Creating mission database tables')

        cur.execute("""
          CREATE TABLE IF NOT EXISTS flights (
            id INTEGER PRIMARY KEY NOT NULL,
            carrier TEXT NOT NULL,
            flight_number TEXT NOT NULL,
            origin INTEGER NOT NULL REFERENCES airports(id),
            destination INTEGER NOT NULL REFERENCES airports(id),
            day_of_week_mask INTEGER NOT NULL,
            departure_time INTEGER NOT NULL,
            arrival_time INTEGER NOT NULL,
            arrival_day_offset INTEGER NOT NULL,
            service_type TEXT NOT NULL,
            aircraft_type TEXT NOT NULL,
            engine_type TEXT,
            distance REAL NOT NULL,
            seat_capacity INTEGER NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to TEXT NOT NULL,
            number_of_flights INTEGER NOT NULL,
            od_pair TEXT NOT NULL
          )""")

        cur.execute("""
          CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY NOT NULL,
            departure_timestamp INTEGER NOT NULL,
            arrival_timestamp INTEGER NOT NULL,
            day INTEGER NOT NULL,
            flight_id INTEGER NOT NULL REFERENCES flights(id)
          )""")

        cur.execute("""
          CREATE TABLE IF NOT EXISTS airports (
            id INTEGER PRIMARY KEY NOT NULL,
            iata_code TEXT NOT NULL,
            name TEXT NOT NULL,
            municipality TEXT,
            country TEXT NOT NULL REFERENCES countries(code),
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            elevation REAL
          )""")

        cur.execute("""
          CREATE VIRTUAL TABLE IF NOT EXISTS airport_location_idx USING rtree(
            id INTEGER PRIMARY KEY,
            min_latitude, max_latitude,
            min_longitude, max_longitude
          )""")

        cur.execute("""
          CREATE TABLE IF NOT EXISTS countries (
            code TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            continent TEXT NOT NULL
          )""")

        if indexes:
            self.index()

    def _distance_check(
        self,
        line: int,
        origin: AirportInfo,
        destination: AirportInfo,
        given_distance_km: float,
        zero_distance_threshold_km: float = 1.0,
        abs_difference_threshold_km: float = 50.0,
        relative_difference_threshold_percent: float = 10.0,
    ) -> bool:
        """Check if the stated distance is plausible for the given airports.

        This checks if the stated distance and the distance we calculate from
        airport positions are within ±10%, ignoring differences less than 50
        km. (This check helps to catch problems with miscoded airports.)

        Returns True if the distance is plausible, False otherwise, recording
        warnings for failure cases.
        """

        gc_distance_km = (
            great_circle_distance(
                origin.airport.latitude,
                origin.airport.longitude,
                destination.airport.latitude,
                destination.airport.longitude,
                degrees=True,
            )
            / 1000.0
        )

        # Origin/destination distances that are too small are suspicious.
        if gc_distance_km < zero_distance_threshold_km:
            self._warn(
                Warning.Type.ZERO_DISTANCE, line, origin=origin, destination=destination
            )
            return False

        # Mismatches between the stated great circle distance in the input
        # schedule data and the great circle distance calculated from origin
        # and destination airport locations are suspicious.
        if given_distance_km > 0:
            abs_diff = abs(given_distance_km - gc_distance_km)
            pct_diff = 100 * abs_diff / gc_distance_km
            if (
                abs_diff > abs_difference_threshold_km
                and pct_diff > relative_difference_threshold_percent
            ):
                self._warn(
                    Warning.Type.SUSPICIOUS_DISTANCE,
                    line,
                    origin=origin,
                    destination=destination,
                    given_distance_km=given_distance_km,
                    calculated_distance_km=gc_distance_km,
                )
                return False

        return True

    def report(self, nentries: int, report_file: str | None = None):
        if len(self.warnings) == 0 and len(self.unknown_airports) == 0:
            return

        report_fp = sys.stdout
        try:
            if report_file is not None:
                report_fp = open(report_file, 'w')
            with contextlib.redirect_stdout(report_fp):
                if len(self.warnings) > 0:
                    print('Warnings during import:')
                    lost: dict[Warning.Type, int] = defaultdict(int)
                    for line in sorted(self.warnings.keys()):
                        lost[self.warnings[line].warn_type] += 1
                        print(f'{line}: {self.warnings[line]}')
                    print(f'Total entries attempted: {nentries}.')
                    for warn_type in Warning.Type:
                        if warn_type in lost:
                            print(
                                f'Lost to {warn_type.value}: {lost[warn_type]} '
                                f'({100 * lost[warn_type] / nentries:.2f}%).'
                            )

                if len(self.unknown_airports) > 0:
                    print(
                        'Unknown airports during '
                        f'import: {", ".join(sorted(self.unknown_airports))}.'
                    )
        finally:
            if report_fp != sys.stdout:
                report_fp.close()
