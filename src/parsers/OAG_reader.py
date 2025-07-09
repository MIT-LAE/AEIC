import csv
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, date, time, timedelta
from enum import Enum
from typing import Any

# TODO: USE PENDULUM FOR DATES AND TIMES?


class DayOfWeek(Enum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7


class DomesticInternational(str, Enum):
    DOMESTIC = 'D'
    INTERNATIONAL = 'I'


DomInt = tuple[DomesticInternational, DomesticInternational]


class ServiceType(str, Enum):
    # TODO: FILL IN MISSING CASES HERE?
    NORMAL_PASSENGER = 'J'
    PASSENGER_CARGO_IN_CABIN = 'Q'
    PASSENGER_SHUTTLE_MODE = 'S'


# Some information here:
#  https://knowledge.oag.com/docs/schedules-direct-data-fields-explained


@dataclass
class OAGEntry:
    """A single entry in the OAG flight schedule data.

    Note: This class is derived from the pre-processed OAG data provided by
    Prateek.
    """

    carrier: str  # [ 1] = COLUMN NUMBER IN INPUT CSV FILE
    fltno: int  # [ 2]
    depapt: str  # [ 3]
    depcity: str  # [ 4]
    depctry: str | None  # [ 5]
    arrapt: str  # [ 6]
    arrcity: str  # [ 7]
    arrctry: str | None  # [ 8]
    deptim: time  # [ 9]
    arrtim: time  # [10]
    arrday: int  # [11] -1, 0, +1, +2  TODO: CHECK MEANING
    elptim: timedelta  # [12]
    days: set[DayOfWeek]  # [13] Set of days of week (M=1, S=7)
    # stops                 # [14] ALL ZERO
    # intapt                # [15] ALL EMPTY
    # acftchange            # [16] ALL EMPTY
    govt_app: bool  # [17] Government approval required (X OR EMPTY)
    comm10_50: int | None  # [18] TODO: WHAT'S THIS? EITHER 10 OR EMPTY
    genacft: str  # [19] TODO: 81 VALUES ⇒ SEPARATE TABLE?
    inpacft: str  # [20] TODO: 181 VALUES ⇒ SEPARATE TABLE?
    service: ServiceType  # [21]
    seats: int  # [22]
    tons: float  # [23]
    restrict: str | None  # [24] TODO: WHAT IS THIS?
    domint: DomInt  # [25]
    efffrom: date  # [26]
    effto: date  # [27]
    routing: str  # [28]
    longest: bool  # [29] "L" or EMPTY
    distance: int  # [30]
    sad: str | None  # [31] TODO: WHAT IS THIS?
    # mcd                   # [32] ALL EMPTY
    # flt_dupe              # [33] ALL EMPTY
    acft_owner: str | None  # [34]
    operating: bool  # [35] "O" OR EMPTY
    # ghost                 # [36] ALL EMPTY
    duplicate: str | None  # [37] TODO: WHAT IS THIS?  "D", "P" OR EMPTY
    NFlts: int  # [38]

    @classmethod
    def from_csv_row(cls, row: list[str]) -> 'OAGEntry':
        """
        Create an OAGEntry instance from a CSV row.
        """

        days = set()
        if row[12]:
            for day in range(1, 8):
                if str(day) in row[12]:
                    days.add(DayOfWeek(day))

        def make_int(t: str) -> int:
            if '.' in t:
                t = t[:-2]
            return int(t)

        def make_date(t: str) -> date:
            tint = make_int(t)
            # YYYYMMDD
            return date(tint // 10000, tint % 10000 // 100, tint % 100)

        def make_time(t: str) -> time:
            tint = make_int(t)
            return time(tint // 100, tint % 100, tzinfo=UTC)

        def make_timedelta(t: str) -> timedelta:
            tint = make_int(t)
            return timedelta(hours=tint // 100, minutes=tint % 100)

        arrday = 0
        if row[10]:
            if row[10] == 'P':
                arrday = -1
            else:
                arrday = make_int(row[10])

        return cls(
            carrier=row[0],
            fltno=make_int(row[1]),
            depapt=row[2],
            depcity=row[3],
            depctry=row[4] if row[4] else None,
            arrapt=row[5],
            arrcity=row[6],
            arrctry=row[7] if row[7] else None,
            deptim=make_time(row[8]),
            arrtim=make_time(row[9]),
            arrday=arrday,
            elptim=make_timedelta(row[11]),
            days=days,
            govt_app=(row[16].strip().upper() == 'X'),
            # TODO: FIGURE OUT WHAT TO DO WITH THIS ONE
            comm10_50=make_int(row[17]) if row[17] else None,
            genacft=row[18],
            inpacft=row[19],
            service=ServiceType(row[20]),
            seats=make_int(row[21]),
            tons=float(row[22]),
            # TODO: USE ENUMERATION HERE?
            restrict=row[23] if row[23] else None,
            domint=(
                DomesticInternational(row[24][0]),
                DomesticInternational(row[24][1]),
            ),
            efffrom=make_date(row[25]),
            effto=make_date(row[26]),
            routing=row[27],
            longest=(row[28].strip().upper() == 'L'),
            distance=make_int(row[29]),
            sad=row[30] if row[30] else None,
            acft_owner=row[33] if row[33] else None,
            operating=(row[34].strip().upper() == 'O'),
            duplicate=row[36] if row[36] else None,
            NFlts=make_int(row[37]),
        )

    @classmethod
    def from_db_row(cls, row: list[Any]) -> 'OAGEntry':
        """
        Create an OAGEntry instance from a database row.
        """

        if len(row) != 33:
            raise ValueError(f'Expected 39 columns, got {len(row)}')

        row = row[1:]  # Skip the ID column

        return cls(
            carrier=row[0],
            fltno=row[1],
            depapt=row[2],
            depcity=row[3],
            depctry=row[4],
            arrapt=row[5],
            arrcity=row[6],
            arrctry=row[7],
            deptim=time.fromisoformat(row[8]),
            arrtim=time.fromisoformat(row[9]),
            arrday=row[10],
            elptim=timedelta(minutes=row[11]),
            days=set(DayOfWeek(int(d)) for d in row[12]),
            govt_app=row[13],
            comm10_50=row[14],
            genacft=row[15],
            inpacft=row[16],
            service=ServiceType(row[17]),
            seats=row[18],
            tons=row[19],
            restrict=row[20],
            domint=(
                DomesticInternational(row[21][0]),
                DomesticInternational(row[21][1]),
            ),
            efffrom=date.fromisoformat(row[22]),
            effto=date.fromisoformat(row[23]),
            routing=row[24],
            longest=row[25],
            distance=row[26],
            sad=row[27],
            acft_owner=row[28],
            operating=row[29],
            duplicate=row[30],
            NFlts=row[31],
        )


def read_oag_file(file_path: str) -> Generator[OAGEntry, None, None]:
    """
    Read an OAG CSV file and yield OAGEntry instances for each row.
    """
    with open(file_path, newline='') as csvfile:
        first = True
        for row in csv.reader(csvfile):
            if first:
                first = False
                continue
            yield OAGEntry.from_csv_row(row)
