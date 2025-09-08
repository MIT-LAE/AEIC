import csv
import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, date, time

from missions.custom_types import DayOfWeek

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


@dataclass
class CSVEntry:
    """A single entry in the CSV OAG flight schedule data.

    Note: This class is derived from the pre-processed OAG data provided by
    Carla.
    """

    carrier: str
    fltno: int
    depapt: str
    depctry: str | None
    arrapt: str
    arrctry: str | None
    deptim: time
    arrtim: time
    days: set[DayOfWeek]  # Set of days of week (M=1, S=7)
    distance: int  # Great circle distance in statute miles
    inpacft: str  # TODO: 181 VALUES ⇒ SEPARATE TABLE?
    seats: int
    efffrom: date
    effto: date

    @classmethod
    def from_csv_row(cls, row: dict) -> 'CSVEntry | None':
        """
        Create an CSVEntry instance from a CSV row.
        """

        try:
            days = set()
            if row.get('days'):
                for day in range(1, 8):
                    if str(day) in row['days']:
                        days.add(DayOfWeek(day))

            def make_date(t: str) -> date | None:
                if t == '00000000' or t == '99999999':
                    return None
                tint = int(t)
                # YYYYMMDD
                return date(tint // 10000, tint % 10000 // 100, tint % 100)

            def make_time(t: str) -> time:
                tint = int(t)
                return time(tint // 100, tint % 100, tzinfo=UTC)

            def optional(t: str) -> str | None:
                return t if t else None

            fltno = row.get('fltno')
            if fltno is not None and fltno != '':
                fltno = int(fltno)

            return cls(
                carrier=row['carrier'],
                fltno=fltno,
                depapt=row['depapt'],
                depctry=optional(row.get('depctry')),
                arrapt=row['arrapt'],
                arrctry=optional(row.get('arrctry')),
                deptim=make_time(row['deptim']),
                arrtim=make_time(row['arrtim']),
                days=days,
                distance=int(row['distance']),
                inpacft=row['inpacft'],
                seats=int(row['seats']),
                efffrom=make_date(row['efffrom']),
                effto=make_date(row['effto']),
            )
        except Exception:
            logger.error(f'Failed to convert row: {row}')
            return None

    @classmethod
    def read(cls, file_path: str) -> Generator['CSVEntry', None, None]:
        """
        Read an OAG CSV file and yield CSVEntry instances for each row.
        """
        with open(file_path, newline='') as csvfile:
            first = True
            for row in csv.DictReader(csvfile):
                if first:
                    first = False
                    continue
                yield cls.from_csv_row(row)
