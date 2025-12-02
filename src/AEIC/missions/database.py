import logging
import os
import sqlite3
import weakref
from collections.abc import Generator
from typing import TypeVar

from .query import QueryBase

logger = logging.getLogger(__name__)


T = TypeVar('T')


class Database:
    """Flight schedule database.

    Represents a database of flight schedule entries, stored in an SQLite
    database file, using a schema optimized for common AEIC query use cases.
    """

    def __init__(self, db_path: str):
        """Open a flight database file.

        Parameters
        ----------

        db_path : str
            Path to the SQLite database file.
        """

        # Check that the database file exists if we're opening an existing
        # database in read-only mode. Overridden in derived WriteDatabase
        # class.
        self._check_path(db_path)

        # Create connection and ensure it gets closed when the object is
        # collected. (Better to use explicit close or a context manager if
        # possible!)
        self._conn = sqlite3.connect(db_path)
        self._finalizer = weakref.finalize(self, self.close)

        # Foreign key constraints are enabled at the connection level, so this
        # needs to be done every time we connect to the database.
        self._conn.cursor().execute("PRAGMA foreign_keys = ON")

    def close(self):
        """Close the database connection."""
        if self._finalizer.alive:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _check_path(self, db_path: str):
        """Check that the database file exists."""
        if not os.path.exists(db_path):
            raise RuntimeError(f'Database file {db_path} does not exist.')

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
