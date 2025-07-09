import sqlite3
from collections.abc import Generator
from dataclasses import dataclass, fields
from enum import Enum

from parsers.OAG_reader import OAGEntry


class Comparison(str, Enum):
    EQUAL = '=='
    NOT_EQUAL = '!='
    GREATER_THAN = '>'
    LESS_THAN = '<'
    GREATER_OR_EQUAL = '>='
    LESS_OR_EQUAL = '<='

    @property
    def sql(self):
        if self == Comparison.EQUAL:
            return '='
        elif self == Comparison.NOT_EQUAL:
            return '<>'
        else:
            return self.value


@dataclass
class Condition:
    field: str
    value: str
    comp: Comparison

    def __str__(self):
        return f'{self.field}{self.comp.value}{self.value}'


class OAGDatabase:
    def __init__(self, db_path: str, write_mode: bool = False):
        self.db_path = db_path
        self.write_mode = write_mode
        self.conn = sqlite3.connect(self.db_path)
        if self.write_mode:
            self._ensure_schema()

    def add(self, e: OAGEntry, commit: bool = True):
        if not self.write_mode:
            raise RuntimeError("Database is not in write mode")
        cur = self.conn.cursor()
        placeholders = '?, ' * 31 + '?'
        cur.execute(f"""
            INSERT INTO entries (
                carrier, fltno, depapt, depcity, depctry, arrapt, arrcity,
                arrctry, deptim, arrtim, arrday, elptim, days, govt_app,
                comm10_50, genacft, inpacft, service, seats, tons,
                restrict, domint, efffrom, effto, routing,
                longest, distance, sad, acft_owner,
                operating, duplicate, NFlts
            ) VALUES ({placeholders})""", (
            e.carrier, e.fltno,
            e.depapt, e.depcity, e.depctry,
            e.arrapt, e.arrcity, e.arrctry,
            e.deptim.isoformat()[:5], e.arrtim.isoformat()[:5], e.arrday,
            e.elptim.total_seconds() // 60,
            ''.join(sorted([str(day.value) for day in e.days])),
            e.govt_app, e.comm10_50, e.genacft, e.inpacft,
            e.service.value, e.seats, e.tons,
            e.restrict, e.domint[0].value + e.domint[1].value,
            e.efffrom, e.effto, e.routing,
            e.longest, e.distance, e.sad, e.acft_owner,
            e.operating, e.duplicate, e.NFlts
        ))
        if commit:
            self.conn.commit()

    def __call__(self, *conds: Condition) -> Generator[OAGEntry, None, None]:
        conditions = []
        parameters = []
        valid_fields = set(f.name for f in fields(OAGEntry))
        for c in conds:
            if c.field not in valid_fields:
                raise ValueError(f'Invalid field: {c.field}')
            conditions.append(f'{c.field} {c.comp.sql} ?')
            parameters.append(c.value)

        sql = 'SELECT * FROM entries'
        if len(conditions) > 0:
            sql += ' WHERE ' + ' AND '.join(conditions)

        cur = self.conn.cursor()
        for row in cur.execute(sql, parameters):
            yield OAGEntry.from_db_row(row)

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                carrier TEXT NOT NULL,
                fltno INTEGER NOT NULL,
                depapt TEXT NOT NULL,
                depcity TEXT NOT NULL,
                depctry TEXT,
                arrapt TEXT NOT NULL,
                arrcity TEXT NOT NULL,
                arrctry TEXT,
                deptim TEXT NOT NULL,
                arrtim TEXT NOT NULL,
                arrday INTEGER NOT NULL,
                elptim INTEGER NOT NULL,
                days TEXT NOT NULL,
                govt_app BOOLEAN NOT NULL,
                comm10_50 INTEGER,
                genacft TEXT NOT NULL,
                inpacft TEXT NOT NULL,
                service TEXT NOT NULL,
                seats INTEGER NOT NULL,
                tons REAL NOT NULL,
                restrict TEXT,
                domint TEXT NOT NULL,
                efffrom TEXT NOT NULL,
                effto TEXT NOT NULL,
                routing TEXT NOT NULL,
                longest BOOLEAN NOT NULL,
                distance INTEGER NOT NULL,
                sad TEXT,
                acft_owner TEXT,
                operating BOOLEAN NOT NULL,
                duplicate TEXT,
                NFlts INTEGER NOT NULL
            )
        """)
