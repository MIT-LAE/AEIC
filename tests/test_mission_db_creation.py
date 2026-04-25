import csv

import pytest

import AEIC.utils.airports as airports
from AEIC.missions import Database
from AEIC.missions.oag import convert_oag_data
from AEIC.missions.writable_database import Warning, WritableDatabase
from AEIC.types import DayOfWeek


def test_dow_mask():
    assert (
        WritableDatabase._make_dow_mask(
            {
                DayOfWeek.MONDAY,
                DayOfWeek.TUESDAY,
                DayOfWeek.WEDNESDAY,
                DayOfWeek.THURSDAY,
                DayOfWeek.FRIDAY,
                DayOfWeek.SATURDAY,
                DayOfWeek.SUNDAY,
            }
        )
        == 0b01111111
    )
    assert (
        WritableDatabase._make_dow_mask(
            {DayOfWeek.MONDAY, DayOfWeek.WEDNESDAY, DayOfWeek.FRIDAY}
        )
        == 0b00010101
    )
    assert (
        WritableDatabase._make_dow_mask({DayOfWeek.TUESDAY, DayOfWeek.THURSDAY})
        == 0b00001010
    )
    assert WritableDatabase._make_dow_mask({DayOfWeek.SUNDAY}) == 0b01000000
    assert WritableDatabase._make_dow_mask(set()) == 0b00000000


def test_airport_handling(tmp_path):
    """White-box unit test: pokes at private helpers (`_lookup_timezone`,
    `_get_or_add_airport`) on `WritableDatabase` because the public surface
    folds them into the bulk OAG ingestion path. A naïve refactor of
    `WritableDatabase` internals will break this test even when the public
    behaviour is fine — the trade-off is intentional, since it pins the
    contract that lookup populates the airports table.
    """
    with WritableDatabase(tmp_path / 'test.sqlite') as db:
        cur = db._conn.cursor()

        ap = airports.airport('LHR')
        assert ap is not None
        tz = db._lookup_timezone(ap)
        assert tz == 'Europe/London'

        # Real airport.
        airport_info = db._get_or_add_airport(cur, 1234, 'CDG')
        assert airport_info is not None
        assert airport_info.airport.iata_code == 'CDG'
        assert airport_info.airport.country == 'FR'
        assert airport_info.airport.municipality is not None
        assert airport_info.airport.municipality.startswith('Paris')
        # CDG is at 49.0097°N — `int(...) == 49` would silently swallow a
        # +49.9 drift, so use a tight tolerance instead.
        assert airport_info.airport.latitude == pytest.approx(49.01, abs=0.01)
        assert airport_info.timezone == 'Europe/Paris'

        # _get_or_add_airport for an unseen IATA must actually persist a
        # row — the returned AirportInfo on its own only confirms the
        # in-memory cache. The 1234 above is the *CSV line number* used for
        # warnings (`line` parameter), not an airport id; the airport id is
        # auto-assigned by sqlite and surfaced via `airport_info.id`.
        cur.execute(
            "SELECT id, iata_code, country FROM airports WHERE iata_code = 'CDG'"
        )
        persisted = cur.fetchone()
        assert persisted == (airport_info.id, 'CDG', 'FR')

        # Not a real airport — must NOT leave a row behind.
        assert db._get_or_add_airport(cur, 1235, 'QPX') is None
        cur.execute("SELECT id FROM airports WHERE iata_code = 'QPX'")
        assert cur.fetchone() is None


def test_oag_conversion(tmp_path, test_data_dir):
    # This extract of the 2019 OAG data contains 8 valid flights (see
    # tests/data/oag/README.md for provenance and expected filtering).
    oag_file = test_data_dir / 'oag/2019-extract.csv'
    warnings_path = tmp_path / 'oag_warnings.txt'

    convert_oag_data(
        oag_file,
        2019,
        tmp_path / 'oag_test.sqlite',
        warnings_file=warnings_path,
    )

    with Database(tmp_path / 'oag_test.sqlite') as db:
        cur = db._conn.cursor()

        cur.execute('SELECT COUNT(*) FROM flights')
        assert cur.fetchone()[0] == 8

        # Full row content for AS 1011 ORD->SEA (row 4 of the CSV; the first
        # row that survives is_row_valid filtering). All expected values are
        # derived from the CSV, not from the SUT:
        #   carrier=AS, fltno=1011, depapt=ORD, arrapt=SEA
        #   days='  34' (positions 3,4 → WED,THU → mask = 2**2 + 2**3 = 12)
        #   deptim=0805 → 8*60+5 = 485 minutes since midnight
        #   arrtim=1053 → 10*60+53 = 653 minutes since midnight
        #   arrday='' → +0 day offset
        #   distance=1715 statute miles → 1715 * 1.609344 = 2760.02 km
        #   service=J, inpacft=320, seats=146
        #   efffrom=20191205, effto=20191211 → stored as ISO dates
        #   2019-12-05 is Thu (day 4), 2019-12-11 is Wed (day 3); the
        #   effective window covers one of each → 2 scheduled flights
        cur.execute(
            """
            SELECT f.carrier, f.flight_number,
                   o.iata_code, d.iata_code,
                   f.day_of_week_mask,
                   f.departure_time, f.arrival_time, f.arrival_day_offset,
                   f.service_type, f.aircraft_type, f.engine_type,
                   f.seat_capacity,
                   f.effective_from, f.effective_to,
                   f.number_of_flights, f.od_pair,
                   f.distance
            FROM flights f
            JOIN airports o ON f.origin = o.id
            JOIN airports d ON f.destination = d.id
            WHERE f.carrier = ? AND f.flight_number = ?
            """,
            ('AS', 1011),
        )
        row = cur.fetchone()
        assert row is not None
        (
            carrier,
            flight_number,
            origin_iata,
            dest_iata,
            dow_mask,
            dep_time,
            arr_time,
            arr_day_offset,
            service_type,
            aircraft_type,
            engine_type,
            seat_capacity,
            eff_from,
            eff_to,
            n_flights,
            od_pair,
            distance,
        ) = row
        assert carrier == 'AS'
        # flight_number column is TEXT — SQLite stores the int-parsed fltno
        # as its string form.
        assert flight_number == '1011'
        assert origin_iata == 'ORD'
        assert dest_iata == 'SEA'
        assert dow_mask == 0b00001100
        assert dep_time == 8 * 60 + 5
        assert arr_time == 10 * 60 + 53
        assert arr_day_offset == 0
        assert service_type == 'J'
        assert aircraft_type == '320'
        assert engine_type == ''
        assert seat_capacity == 146
        assert eff_from == '2019-12-05'
        assert eff_to == '2019-12-11'
        assert n_flights == 2
        assert od_pair == 'ORDSEA'
        assert distance == pytest.approx(1715 * 1.609344)

        # Schedules: every accepted flight produced at least one schedule,
        # and at least some flights generated multiple (long effective
        # windows × daily/weekday day-of-week masks).
        cur.execute('SELECT COUNT(*) FROM schedules')
        assert cur.fetchone()[0] > 8
        cur.execute('SELECT COUNT(DISTINCT flight_id) FROM schedules')
        assert cur.fetchone()[0] == 8

    # All 8 valid flights use major airports with plausible distances, so no
    # UNKNOWN_AIRPORT / SUSPICIOUS_DISTANCE / ZERO_DISTANCE / TIME_MISORDERING
    # warnings fire, and (since rejected rows are filtered silently by
    # is_row_valid before distance/airport checks) report() does not write
    # the warnings file at all. Assert the silence explicitly — a regression
    # that started emitting spurious warnings on valid rows would fail here.
    assert not warnings_path.exists()
    assert Warning.Type  # exercise the import so a future refactor catches the rename


def test_oag_warning_categories(tmp_path):
    """Each `Warning.Type` is exercised by mutating one cell of a known-good
    AS 1011 ORD->SEA template, so the trigger for each category is visible
    inline. Without this, a regression in any warning branch would only
    surface during a real OAG ingestion.
    """
    header = [
        'carrier',
        'fltno',
        'depapt',
        'arrapt',
        'deptim',
        'arrtim',
        'arrday',
        'days',
        'stops',
        'genacft',
        'inpacft',
        'service',
        'seats',
        'efffrom',
        'effto',
        'longest',
        'distance',
        'operating',
    ]
    good = [
        'AS',
        '1011',
        'ORD',
        'SEA',
        '0805',
        '1053',
        '',
        '2',
        '00',
        '320',
        '320',
        'J',
        '0146',
        '20190101',
        '20190101',
        'L',
        '0001715',
        'O',
    ]
    field = {h: i for i, h in enumerate(header)}

    def with_changes(**kw) -> list[str]:
        row = list(good)
        for k, v in kw.items():
            row[field[k]] = v
        return row

    csv_path = tmp_path / 'bad_rows.csv'
    with open(csv_path, 'w', newline='') as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        # UNKNOWN_AIRPORT — bogus arrival IATA.
        writer.writerow(with_changes(fltno='100', arrapt='XYZ'))
        # ZERO_DISTANCE — origin == destination (gc < 1 km trips before any
        # given-distance check).
        writer.writerow(with_changes(fltno='101', arrapt='ORD'))
        # SUSPICIOUS_DISTANCE — valid airports, given distance ~6× the
        # great-circle value.
        writer.writerow(with_changes(fltno='102', distance='0009999'))
        # TIME_MISORDERING — ORD->LAX 0500→0100 same-day; in UTC the arrival
        # (0900Z) precedes the departure (1100Z) so the schedule guard fires.
        writer.writerow(
            with_changes(
                fltno='103',
                arrapt='LAX',
                deptim='0500',
                arrtim='0100',
                distance='0001745',
            )
        )

    warnings_path = tmp_path / 'warnings.txt'
    convert_oag_data(
        csv_path,
        2019,
        tmp_path / 'out.sqlite',
        warnings_file=warnings_path,
    )

    assert warnings_path.exists()
    text = warnings_path.read_text()
    for warn_type in Warning.Type:
        assert warn_type.value in text, (
            f'expected {warn_type.value!r} in warnings file:\n{text}'
        )
