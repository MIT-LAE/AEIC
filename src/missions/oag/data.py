import csv
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, date, time

from missions.custom_types import DayOfWeek

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
    NFlts: int

    @classmethod
    def from_csv_row(cls, row: list[str]) -> 'CSVEntry':
        """
        Create an CSVEntry instance from a CSV row.
        """

        (
            carrier_in,
            fltno_in,
            depapt_in,
            depctry_in,
            arrapt_in,
            arrctry_in,
            deptim_in,
            arrtim_in,
            days_in,
            distance_in,
            inpacft_in,
            seats_in,
            efffrom_in,
            effto_in,
            NFlts_in,
        ) = row[:15]

        days = set()
        if row[12]:
            for day in range(1, 8):
                if str(day) in days_in:
                    days.add(DayOfWeek(day))

        def make_date(t: str) -> date:
            tint = int(t)
            # YYYYMMDD
            return date(tint // 10000, tint % 10000 // 100, tint % 100)

        def make_time(t: str) -> time:
            tint = int(t)
            return time(tint // 100, tint % 100, tzinfo=UTC)

        def optional(t: str) -> str | None:
            return t if t else None

        return cls(
            carrier=carrier_in,
            fltno=int(fltno_in),
            depapt=depapt_in,
            depctry=optional(depctry_in),
            arrapt=arrapt_in,
            arrctry=optional(arrctry_in),
            deptim=make_time(deptim_in),
            arrtim=make_time(arrtim_in),
            days=days,
            distance=int(distance_in),
            inpacft=inpacft_in,
            seats=int(seats_in),
            efffrom=make_date(efffrom_in),
            effto=make_date(effto_in),
            NFlts=int(NFlts_in),
        )

    @classmethod
    def read(cls, file_path: str) -> Generator['CSVEntry', None, None]:
        """
        Read an OAG CSV file and yield CSVEntry instances for each row.
        """
        with open(file_path, newline='') as csvfile:
            first = True
            for row in csv.reader(csvfile):
                if first:
                    first = False
                    continue
                yield cls.from_csv_row(row)
