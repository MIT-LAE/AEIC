import csv
from dataclasses import dataclass
from enum import Enum
# TODO: USE PENDULUM?
from datetime import date, time, timedelta, timezone
from typing import Generator


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


class ServiceType(str, Enum):
    NORMAL_PASSENGER = 'J'
    PASSENGER_CARGO_IN_CABIN = 'Q'
    PASSENGER_SHUTTLE_MODE = 'S'


# Some information here: https://knowledge.oag.com/docs/schedules-direct-data-fields-explained

@dataclass
class OAGEntry:
    carrier: str # 1
    fltno: int   # 2
    depapt: str  # 3
    depcity: str  # 4
    depctry: str | None # 5
    arrapt: str  # 6
    arrcity: str # 7
    arrctry: str | None # 8
    deptim: time  # 9
    arrtim: time  # 10
    arrday: int # = 0    # -1, 0, +1, +2  TODO: CHECK MEANING # 11
    elptim: timedelta  # 12
    days: set[DayOfWeek]  # Set of days of the week (Mon=1, Sun=7) the flight operates # 13
    # stops       # ALL ZERO  # 14
    # intapt      # ALL EMPTY # 15
    # acftchange  # ALL EMPTY # 16
    govt_app: bool # Whether government approval is required (X OR EMPTY)  # 17
    comm10_50: int | None # TODO: WHAT'S THIS? EITHER 10 OR EMPTY # 18
    genacft: str  # TODO: 81 VALUES ⇒ SEPARATE TABLE?  # 19
    inpacft: str  # TODO: 181 VALUES ⇒ SEPARATE TABLE?  # 20
    service: ServiceType # 21
    seats: int # 22
    tons: float # 23
    restrict: str | None # TODO: WHAT IS THIS? # 24
    domint: tuple[DomesticInternational, DomesticInternational] # 25
    efffrom: date # 26
    effto: date # 27
    routing: str # 28
    longest: bool # "L" or EMPTY # 29
    distance: int # 30
    sad: str | None # TODO: WHAT IS THIS? # 31
    # mcd   # ALL EMPTY # 32
    # flt_dupe  # ALL EMPTY # 33
    acft_owner: str | None # 34
    operating: bool # "O" OR EMPTY  # 35
    # ghost    # ALL EMPTY # 36
    duplicate: str | None  # TODO: WHAT IS THIS?  "D", "P" OR EMPTY # 37
    NFlts: int # 38

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
            return time(tint // 100, tint % 100, tzinfo=timezone.utc)

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
            comm10_50=make_int(row[17]) if row[17] else None,
            genacft=row[18],
            inpacft=row[19],
            service=ServiceType(row[20]),
            seats=make_int(row[21]),
            tons=float(row[22]),
            # TODO: ENUM HERE
            restrict=row[23] if row[23] else None,
            domint=(DomesticInternational(row[24][0]), DomesticInternational(row[24][1])),
            efffrom=make_date(row[25]),
            effto=make_date(row[26]),
            routing=row[27],
            longest=(row[28].strip().upper() == 'L'),
            distance=make_int(row[29]),
            sad=row[30] if row[30] else None,
            acft_owner=row[33] if row[33] else None,
            operating=(row[34].strip().upper() == 'O'),
            duplicate=row[36] if row[36] else None,
            NFlts=make_int(row[37])
        )


def read_oag_file(file_path: str) -> Generator[OAGEntry, None, None]:
    """
    Reads an OAG CSV file and yields OAGEntry instances for each row.
    """
    with open(file_path, newline='') as csvfile:
        first = True
        for row in csv.reader(csvfile):
            if first:
                first = False
                continue
            yield OAGEntry.from_csv_row(row)
