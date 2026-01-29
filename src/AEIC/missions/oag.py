import csv
import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import date

from tqdm import tqdm

from AEIC.types import DayOfWeek, TimeOfDay
from AEIC.units import STATUTE_MILES_TO_KM

from .writable_database import WritableDatabase

logger = logging.getLogger(__name__)


# Some information here:
#  https://knowledge.oag.com/docs/schedules-direct-data-fields-explained
#
# Example row:
#
#   carrier:  "VT"
#   fltno:    "124"
#   depapt:   "AAA"
#   depctry:  "PF"
#   arrapt:   "FAC"
#   arrctry:  "PF"
#   deptim:   "1730"
#   arrtim:   "1750"
#   days:     " 2"
#   distance: "47"
#   inpacft:  "AT7"
#   seats:    "68"
#   efffrom:  "20190115"
#   effto:    "20190115"
#   NFlts:    "1"


# These are values that appear in the genafct field that aren't aircraft at
# all!
EXCLUDE_EQUIPMENT = {'BUS', 'HOV', 'LCH', 'LMO', 'RFS', 'TRN'}


@dataclass
class CSVEntry:
    """A single entry in the CSV OAG flight schedule data."""

    line: int
    """Line number in the CSV file (1-based, including header line)."""

    carrier: str
    """Two letter airline code, e.g. "AA"."""

    fltno: int
    """Flight number, e.g. 124."""

    depapt: str
    """Departure airport: three letter IATA airport code, e.g. "JFK"."""

    depctry: str | None
    """Departure country: two letter ISO country code, e.g. "US"."""

    arrapt: str
    """Arrival airport: two letter IATA airport code, e.g. "LAX"."""

    arrctry: str | None
    """Arrival country: two letter ISO country code, e.g. "US"."""

    deptim: TimeOfDay
    """Departure time (local time at departure airport)."""

    arrtim: TimeOfDay
    """Arrival time (local time at arrival airport)."""

    arrday: int
    """Arrival day offset, e.g. 0=same day, 1=next day, -1=previous day."""

    days: set[DayOfWeek]
    """Days of operation, set of DayOfWeek."""

    distance: int
    """Great circle distance in statute miles."""

    inpacft: str
    """Aircraft type (IATA code), e.g. "738"."""

    service: str
    """Service type (IATA single-letter code), e.g. "J"."""

    seats: int
    """Seat capacity."""

    efffrom: date | None
    """Effective from date."""

    effto: date | None
    """Effective to date."""

    stops: int
    """Number of stops (0 for direct flights)."""

    longest: bool
    """True for full flights, False for individual legs of multi-leg flights."""

    @staticmethod
    def is_row_valid(row: dict) -> bool:
        """Check whether a CSV row is valid for inclusion in the database.

        This filters out various conditions that appear in the OAG data that we
        want to work around. In particular, we only want direct flight legs, we
        do not want duplicate flights, and we do not want the non-aviation
        records from the OAG data.
        """

        # - End of file marker in some OAG data files.
        if row['carrier'] == '\x1a':
            return False

        # - Services operated using surface vehicles.
        if row['service'] in ('V', 'U'):
            return False

        # - Entries with stops. (We only want direct flights.)
        if int(row['stops']) != 0:
            return False

        # - Do not include entries for non-operating carrier: this should
        #   eliminate all duplicate flights.
        if row['operating'] == 'N':
            return False

        # - Non-aircraft equipment codes.
        if row['genacft'] in EXCLUDE_EQUIPMENT:
            return False

        return True

    @classmethod
    def from_csv_row(cls, row: dict, line: int = 0) -> 'CSVEntry | None':
        """Create an CSVEntry instance from a CSV row."""

        try:
            # Skip entries that are not of interest for building the mission
            # database.
            if not cls.is_row_valid(row):
                return None

            days = set()
            if row.get('days'):
                for day in range(1, 8):
                    if str(day) in row['days']:
                        days.add(DayOfWeek(day))

            def make_date(t: str) -> date | None:
                # Special values for indeterminate effective from and to dates.
                if t == '00000000' or t == '99999999':
                    return None
                tint = int(t)
                # YYYYMMDD
                return date(tint // 10000, tint % 10000 // 100, tint % 100)

            def make_time(t: str) -> TimeOfDay:
                tint = int(t)
                return TimeOfDay(hour=tint // 100, minute=tint % 100)

            def optional(t: str) -> str | None:
                return t if t else None

            def convert_arrday(t: str) -> int:
                match t:
                    case 'P':
                        return -1
                    case ' ' | '':
                        return 0
                    case _:
                        return int(t)

            fltno = row.get('fltno')
            if fltno is None or fltno == '':
                fltno = 0
            else:
                fltno = int(fltno)

            return cls(
                line=line,
                carrier=row['carrier'],
                fltno=fltno,
                depapt=row['depapt'],
                depctry=optional(row.get('depctry', '')),
                arrapt=row['arrapt'],
                arrctry=optional(row.get('arrctry', '')),
                deptim=make_time(row['deptim']),
                arrtim=make_time(row['arrtim']),
                arrday=convert_arrday(row['arrday']),
                days=days,
                distance=int(row['distance']),
                service=row['service'],
                inpacft=row['inpacft'],
                seats=int(row['seats']),
                efffrom=make_date(row['efffrom']),
                effto=make_date(row['effto']),
                stops=int(row['stops']),
                longest=(row['longest'] == 'L'),
            )
        except Exception:
            logger.exception(f'Failed to convert row: {row}')
            return None

    @classmethod
    def read(cls, file_path: str) -> Generator['CSVEntry']:
        """Read an OAG CSV file and yield CSVEntry instances for each row."""
        with open(file_path, newline='') as csvfile:
            for idx, row in enumerate(csv.DictReader(csvfile)):
                entry = cls.from_csv_row(row, idx + 2)
                if entry is not None:
                    yield entry


class OAGDatabase(WritableDatabase):
    """Writable flight schedule database for OAG data.

    Represents a database of flight schedule entries, stored in an SQLite
    database file, using a schema optimized for common AEIC query use cases.
    """

    def __init__(self, db_path: str, year: int):
        super().__init__(db_path)
        self._year = year

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

        cur = self._conn.cursor()

        # Add airport entries if they don't already exist. (Also adds country
        # entries as needed.)
        origin = self._get_or_add_airport(cur, e.line, e.depapt)
        destination = self._get_or_add_airport(cur, e.line, e.arrapt)

        # Skip entries with unknown airports.
        if origin is None or destination is None:
            return

        # Skip entries where the stated flight distance is not plausible:
        # require the stated distance and the distance we calculate from
        # airport positions to be within ±10%, ignoring differences less than
        # 50 km. (This check helps to catch problems with miscoded airports.)
        if not self._distance_check(
            e.line, origin, destination, e.distance * STATUTE_MILES_TO_KM
        ):
            return

        # Handle cases where the effective from/to dates are not set, implying
        # from beginning of year or to end of year.
        effective_from = e.efffrom or date(self._year, 1, 1)
        effective_to = e.effto or date(self._year, 12, 31)

        # Add flight entry (number of flights to be computed from schedule).
        flight_id = self._add_flight(
            cur,
            e.carrier,
            e.fltno,
            origin,
            destination,
            effective_from,
            effective_to,
            e.days,
            e.deptim,
            e.arrtim,
            e.arrday,
            e.service,
            e.inpacft,
            e.seats,
            e.distance * STATUTE_MILES_TO_KM,
        )

        # Add schedule entries.
        num_flights = self._add_schedule(
            cur,
            e.line,
            flight_id,
            origin,
            destination,
            e.efffrom,
            e.effto,
            e.days,
            e.deptim,
            e.arrtim,
            e.arrday,
        )

        # Set number of flights in flight entry.
        self._set_flight_count(cur, flight_id, num_flights)

        if commit:
            self._conn.commit()


def convert_oag_data(
    in_file: str, year: int, db_file: str, warnings_file: str | None = None
) -> None:
    with OAGDatabase(db_file, year) as db:
        logger.info('Counting entries in input file...')
        nlines = 0
        with open(in_file) as fp:
            for row in csv.DictReader(fp):
                if CSVEntry.is_row_valid(row):
                    nlines += 1

        nentries = 0
        for entry in tqdm(CSVEntry.read(in_file), total=nlines):
            # Skip invalid entries.
            if entry is None:
                continue

            # Add entries to database in batches of 10,000.
            db.add(entry, commit=False)
            nentries += 1
            if nentries % 10000 == 0:
                db.commit()

        db.commit()
        db.index()
        db.report(nentries, warnings_file)
