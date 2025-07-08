import logging
import os
import sqlite3
from collections.abc import Generator
from datetime import datetime, timedelta
from typing import TypeVar

import pandas as pd

import utils.airports as airports
from missions.custom_types import DayOfWeek
from utils.units import STATUTE_MILES_TO_KM

from .data import CSVEntry
from .query import QueryBase

logger = logging.getLogger(__name__)


T = TypeVar('T')


class Database:
    """OAG flight schedule database.

    Represents a database of OAG flight schedule entries, stored in an SQLite
    database file, using a schema optimized for common AEIC query use cases.
    """

    def __init__(self, db_path: str, write_mode: bool = False):
        """Open an OAG database file.

        Opening for writing is only done when creating a new database file.

        Parameters
        ----------

        db_path : str
            Path to the SQLite database file.
        write_mode : bool, optional
            Whether to open the database for writing (creating a new database
            file). Default is False (read-only mode).
        """

        if os.path.exists(db_path):
            if write_mode:
                raise RuntimeError(f'Database file {db_path} already exists.')
        else:
            if not write_mode:
                raise RuntimeError(f'Database file {db_path} does not exist.')

        self._write_mode = write_mode
        self._conn = sqlite3.connect(db_path)

        # We maintain in-memory caches of invariant airport and country data to
        # avoid repeated database lookups while adding entries.
        self._airport_cache: dict[str, int] = {}
        self._country_cache: set[str] = set()

        # Foreign key constraints are enabled at the connection level, so this
        # needs to be done every time we connect to the database.
        self._conn.cursor().execute("PRAGMA foreign_keys = ON")

        # If we're writing a database file, create the tables if needed.
        if self._write_mode:
            self._ensure_schema(indexes=False)

    def __call__(self, query: QueryBase[T]) -> Generator[T] | T:
        """Execute a query against the database.

        Results are returned via a generator that yields instances of the
        result class for the corresponding query type.

        Supported query types are subclasses of `QueryBase`: `Query` is a
        "normal" scheduled flight query, `FrequentFlightQuery` determines the
        most frequently occurring airport origin/destination pairs, and
        `CountQuery` counts the number of scheduled flights matching filter
        conditions.
        """

        sql, params = query.to_sql()
        cur = self._conn.cursor()

        # Sometimes we return a single result (e.g. for count queries), and
        # sometimes we use a generator to yield multiple results. The single
        # result and generator cases need to be split into separate functions
        # because as soon as Python sees a yield statement in a function, it
        # treats the whole function as a generator.
        if query.PROCESS_RESULT is not None:
            return query.PROCESS_RESULT(cur.execute(sql, params))
        else:
            return self._yield_results(cur, sql, params, query.RESULT_TYPE)

    @staticmethod
    def _yield_results(cur, sql, params, result_type) -> Generator:
        for row in cur.execute(sql, params):
            yield result_type.from_row(row)

    def commit(self):
        """Commit the current transaction to the database."""

        if self._write_mode:
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

    def add(self, e: CSVEntry, commit: bool = True):
        """Add a flight to the database.

        This adds a single flight entry to the database, along with all the
        scheduled flights that it represents. Airport and country entries are
        added as needed.

        Parameters
        ----------
        e : CSVEntry
            The flight entry to add. This represents a single row from the OAG
            input CSV file.
        commit : bool, optional
            Whether to commit the transaction after adding the entry.

        """

        if not self._write_mode:
            raise RuntimeError("Database is not in write mode")
        cur = self._conn.cursor()

        # Add airport entries if they don't already exist. (Also adds country
        # entries as needed.)
        origin_id = self._get_or_add_airport(e.depapt, cur)
        destination_id = self._get_or_add_airport(e.arrapt, cur)
        if origin_id is None or destination_id is None:
            logger.warning(
                'Skipping flight with unknown airport: %s -> %s', e.depapt, e.arrapt
            )
            return

        # Add flight entry (number of flights to be computed from schedule).
        flight_id = self._add_flight(e, origin_id, destination_id, cur)

        # Add schedule entries.
        num_flights = self._add_schedule(e, flight_id, cur)

        # Set number of flights in flight entry.
        cur.execute(
            'UPDATE flights SET number_of_flights = ? WHERE id = ?',
            (num_flights, flight_id),
        )

        if commit:
            self._conn.commit()

    def _add_schedule(self, e: CSVEntry, flight_id: int, cur: sqlite3.Cursor) -> int:
        """Add schedule entries for a flight."""

        # Collect the schedule entries to add. (It's more efficient to do this
        # and to use executemany to add them to the database in one go.)
        data = []

        # Run over the (inclusive) effective date range.
        for d in pd.date_range(e.efffrom, e.effto):
            # Skip days not in the day-of-week set for this flight.
            if DayOfWeek.from_pandas(d) not in e.days:
                continue

            # Calculate departure and arrival timestamps from current date and
            # given departure and arrival times for flight.
            departure_timestamp = int(
                (
                    d + timedelta(hours=e.deptim.hour, minutes=e.deptim.minute)
                ).timestamp()
            )
            arrival_timestamp = int(
                (
                    d + timedelta(hours=e.arrtim.hour, minutes=e.arrtim.minute)
                ).timestamp()
            )

            # If the arrival timestamp is before the departure timestamp, then
            # the flight runs overnight and the arrival is on the next day.
            if arrival_timestamp < departure_timestamp:
                arrival_timestamp += 24 * 3600

            # Calculate day number since Unix epoch (1970-01-01). This is used
            # for "every N days" sampling.
            day = (d - datetime(1970, 1, 1)).days

            data.append((departure_timestamp, arrival_timestamp, day, flight_id))

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

    def _add_flight(
        self, e: CSVEntry, origin_id: int, destination_id: int, cur: sqlite3.Cursor
    ) -> int:
        """Add a flight entry to the database.

        This fills in the flight entry, but does not add any of the schedule
        entries that correspond to the flight. That is done separately. Filling
        the number_of_flights field has to be deferred until the scheduled
        flights have been added.
        """

        placeholders = '?, ' * 14 + '?'
        cur.execute(
            f"""
            INSERT INTO flights (
                carrier, flight_number, origin, destination,
                day_of_week_mask, departure_time, arrival_time,
                aircraft_type, engine_type, distance, seat_capacity,
                effective_from, effective_to, number_of_flights, od_pair
            ) VALUES ({placeholders})
            RETURNING id""",
            (
                e.carrier,
                e.fltno,
                origin_id,  # Foreign key to airports table.
                destination_id,  # Foreign key to airports table.
                self._make_dow_mask(e.days),  # Bitmask for days of week.
                e.deptim.hour * 60 + e.deptim.minute,  # Minutes since midnight.
                e.arrtim.hour * 60 + e.arrtim.minute,  # Minutes since midnight.
                e.inpacft,
                '',  # Engine type not available.
                e.distance * STATUTE_MILES_TO_KM,  # SI units.
                e.seats,
                e.efffrom.isoformat(),  # As YYYY-MM-DD string.
                e.effto.isoformat(),  # As YYYY-MM-DD string.
                0,  # Number of flights to be computed from schedule.
                (
                    min(e.depapt, e.arrapt)  # Direction independent OD pair.
                    + max(e.depapt, e.arrapt)
                ),
            ),
        )
        row = cur.fetchone()
        assert row is not None

        # Return the ID of the newly added flight entry for use in schedules
        # table.
        return row[0]

    def _get_or_add_airport(self, iata_code: str, cur: sqlite3.Cursor) -> int | None:
        """Retrieve or add an airport entry."""

        # If we already have this airport in the cache, return it.
        if iata_code in self._airport_cache:
            return self._airport_cache[iata_code]

        # Try to find the airport in the database.
        cur.execute('SELECT id FROM airports WHERE iata_code = ?', (iata_code,))
        row = cur.fetchone()
        if row:
            self._airport_cache[iata_code] = row[0]
            return row[0]

        # Not found, so look up the airport in the static data.
        airport = airports.airports[iata_code]
        if not airport:
            logger.warning('Unknown airport IATA code: %s', iata_code)
            return None
        assert isinstance(airport, airports.Airport)

        # Ensure the country is in the database.
        country = self._get_or_add_country(airport.country, cur)

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
        self._airport_cache[iata_code] = airport_id
        return airport_id

    def _get_or_add_country(self, code: str, cur: sqlite3.Cursor) -> str:
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
        country = airports.countries[code]
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

        logger.info('Creating OAG database tables')

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
