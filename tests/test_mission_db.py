from collections.abc import Generator
from datetime import date

import pandas as pd
import pytest

from AEIC.missions import (
    BoundingBox,
    CountQuery,
    Database,
    Filter,
    FrequentFlightQuery,
    Mission,
    Query,
)
from AEIC.missions.query import QueryResult, date_to_timestamp
from AEIC.utils import GEOD

_AUSTRIA_BBOX = BoundingBox(
    min_longitude=11.343,
    max_longitude=16.570,
    min_latitude=46.642,
    max_latitude=48.234,
)


@pytest.mark.parametrize(
    'name,filter_kwargs,table,expected_cond,expected_params',
    [
        (
            'distance_range',
            dict(min_distance=1000, max_distance=5000),
            None,
            'distance >= ? AND distance <= ?',
            [1000, 5000],
        ),
        (
            'airport_either_end',
            dict(airport='LHR'),
            None,
            '(origin IN (SELECT id FROM airports WHERE iata_code IN (?)) OR '
            'destination IN (SELECT id FROM airports WHERE iata_code IN (?)))',
            ['LHR', 'LHR'],
        ),
        (
            'country_either_end',
            dict(country='US'),
            None,
            '(origin IN (SELECT id FROM airports WHERE country IN (?)) OR '
            'destination IN (SELECT id FROM airports WHERE country IN (?)))',
            ['US', 'US'],
        ),
        (
            'continent_either_end',
            dict(continent='SA'),
            None,
            '(origin IN (SELECT id FROM airports WHERE country IN '
            '(SELECT code FROM countries WHERE continent IN (?))) OR '
            'destination IN (SELECT id FROM airports WHERE country IN '
            '(SELECT code FROM countries WHERE continent IN (?))))',
            ['SA', 'SA'],
        ),
        (
            'origin_airport_only',
            dict(origin_airport='LAX'),
            None,
            'origin IN (SELECT id FROM airports WHERE iata_code IN (?))',
            ['LAX'],
        ),
        (
            'origin_country_only',
            dict(origin_country='US'),
            None,
            'origin IN (SELECT id FROM airports WHERE country IN (?))',
            ['US'],
        ),
        (
            'bounding_box_either_end',
            dict(bounding_box=_AUSTRIA_BBOX),
            None,
            '(origin IN (SELECT id FROM airport_location_idx '
            'WHERE min_latitude >= ? AND max_latitude <= ? AND '
            'min_longitude >= ? AND max_longitude <= ?) OR '
            'destination IN (SELECT id FROM airport_location_idx '
            'WHERE min_latitude >= ? AND max_latitude <= ? AND '
            'min_longitude >= ? AND max_longitude <= ?))',
            [46.642, 48.234, 11.343, 16.57, 46.642, 48.234, 11.343, 16.57],
        ),
        (
            'origin_country_destination_country',
            dict(origin_country='AT', destination_country='DE'),
            None,
            'origin IN (SELECT id FROM airports WHERE country IN (?)) AND '
            'destination IN (SELECT id FROM airports WHERE country IN (?))',
            ['AT', 'DE'],
        ),
        (
            'origin_country_destination_continent',
            dict(origin_country='FR', destination_continent='SA'),
            None,
            'origin IN (SELECT id FROM airports WHERE country IN (?)) AND '
            'destination IN (SELECT id FROM airports WHERE country IN '
            '(SELECT code FROM countries WHERE continent IN (?)))',
            ['FR', 'SA'],
        ),
        (
            'service_type_list',
            dict(service_type=['J', 'S', 'Q']),
            None,
            'service_type IN (?, ?, ?)',
            ['J', 'S', 'Q'],
        ),
        (
            'aircraft_type_list',
            dict(aircraft_type=['B737', '777']),
            None,
            'aircraft_type IN (?, ?)',
            ['B737', '777'],
        ),
        (
            'combined_with_table_alias',
            dict(min_distance=2000, min_seat_capacity=200, country=['US', 'CA']),
            'f',
            'f.distance >= ? AND f.seat_capacity >= ? AND '
            '(f.origin IN (SELECT id FROM airports WHERE country IN (?, ?)) OR '
            'f.destination IN (SELECT id FROM airports WHERE country IN (?, ?)))',
            [2000, 200, 'US', 'CA', 'US', 'CA'],
        ),
    ],
)
def test_filter(name, filter_kwargs, table, expected_cond, expected_params):
    f = Filter(**filter_kwargs)
    cond, params = f.to_sql(table=table) if table else f.to_sql()
    assert cond == expected_cond
    assert params == expected_params


def test_query():
    sql, params = Query().to_sql()
    assert sql == (
        'SELECT s.departure_timestamp, s.arrival_timestamp, '
        's.id as id, f.id as flight_id, f.carrier, f.flight_number, '
        'ao.iata_code AS origin, ao.country AS origin_country, '
        'ad.iata_code AS destination, ad.country AS destination_country, '
        'f.service_type, f.aircraft_type, f.engine_type, '
        'f.distance, f.seat_capacity '
        'FROM schedules s '
        'JOIN flights f ON f.id = s.flight_id '
        'JOIN airports ao ON f.origin = ao.id '
        'JOIN airports ad ON f.destination = ad.id '
        'ORDER BY s.departure_timestamp'
    )
    assert 'WHERE' not in sql
    assert len(params) == 0

    sql, params = Query(filter=Filter(min_distance=1000, max_distance=5000)).to_sql()
    assert 'WHERE f.distance >= ? AND f.distance <= ?' in sql
    assert params == [1000, 5000]

    sql, params = Query(
        filter=Filter(country='US'),
        start_date=date(2024, 3, 1),
        end_date=date(2024, 8, 31),
    ).to_sql()
    assert params == [
        'US',
        'US',
        int(date_to_timestamp(date(2024, 3, 1)).timestamp()),
        int(date_to_timestamp(date(2024, 9, 1)).timestamp()),
    ]

    sql, params = Query(
        filter=Filter(country='US', min_seat_capacity=250), sample=0.05
    ).to_sql()
    assert params == [250, 'US', 'US', 0.05]
    assert 'det_random()' in sql

    sql, params = Query(
        filter=Filter(country='US', min_distance=1000, max_distance=5000),
        every_nth=8,
    ).to_sql()
    assert params == [1000, 5000, 'US', 'US', 8]
    assert 'SELECT MIN(day) FROM schedules' in sql

    # CountQuery: unfiltered shortcuts to a single SELECT against schedules.
    sql, params = CountQuery().to_sql()
    assert sql == 'SELECT COUNT(s.id) FROM schedules s'
    assert params == []

    # CountQuery with a country filter brings in the JOINs and emits the
    # WHERE clause from the filter.
    sql, params = CountQuery(filter=Filter(country='US')).to_sql()
    assert 'COUNT(s.id)' in sql
    assert 'WHERE' in sql
    assert params == ['US', 'US']

    sql, params = FrequentFlightQuery(
        filter=Filter(origin_country='US'), limit=10
    ).to_sql()
    assert params == ['US']
    assert 'GROUP BY od_pair' in sql
    # FrequentFlightQuery inlines the limit into the SQL string rather than
    # parameterizing it (see query.py:253), so the value lands in `sql`,
    # not in `params`.
    assert 'LIMIT 10' in sql


def test_query_result(test_data_dir):
    # These queries were all tested manually in the SQLite shell to determine
    # the correct results using this exact test database.
    test_db = test_data_dir / 'missions/oag-2019-test-subset.sqlite'
    with Database(test_db) as db:
        # All scheduled flights in the test database.
        result = db(Query())
        assert isinstance(result, Generator)
        nflights = len(list(result))
        assert nflights == 1197

        # Simple distance filter.
        nflights = 0
        result = db(Query(Filter(min_distance=3000)))
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.gc_distance >= 3000 * 1000
            nflights += 1
        assert nflights == 99

        # Country filter: either origin or destination in the given country.
        nflights = 0
        result = db(Query(Filter(country='IT')))
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.origin_country == 'IT' or flight.destination_country == 'IT'
            nflights += 1
        assert nflights == 36

        # Combined filter.
        nflights = 0
        q1 = Query(Filter(max_distance=3000, country=['US', 'CA']))
        result = db(q1)
        assert isinstance(result, Generator)
        for flight in result:
            # Use a sloppy comparison here because the distances in the OAG
            # database are not exact great circle distances, but they should be
            # close enough for testing purposes.
            assert flight.gc_distance <= 3005 * 1000
            assert flight.origin_country in (
                'US',
                'CA',
            ) or flight.destination_country in ('US', 'CA')
            nflights += 1
        assert nflights == 307

        # Sampling.
        q2 = Query(Filter(max_distance=3000, country=['US', 'CA']))
        q2.sample = 0.1
        nflights = 0
        result = db(q2)
        assert isinstance(result, Generator)
        for flight in result:
            # Use a sloppy comparison here because the distances in the OAG
            # database are not exact great circle distances, but they should be
            # close enough for testing purposes.
            assert flight.gc_distance <= 3005 * 1000
            assert flight.origin_country in (
                'US',
                'CA',
            ) or flight.destination_country in ('US', 'CA')
            nflights += 1
        # With a 10% sample, we should get between 40 and 90 flights but for
        # testing it's too dodgy to assert an exact count. The loose
        # `< 307` bound rules out the no-filter case; `> 0` is the bound
        # that catches a regression that silently filters everything out.
        assert nflights > 0
        assert nflights < 307

        # "Every nth day" filtering.
        EPOCH = pd.Timestamp('1970-01-01T00:00:00Z')

        def days_since_epoch(t: pd.Timestamp) -> int:
            return int((t - EPOCH).days)

        q3 = Query(Filter(max_distance=3000, country=['US', 'CA']))
        q3.every_nth = 5
        nflights = 0
        last_day: int | None = None
        saw_nonzero_gap = False
        result = db(q3)
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.gc_distance <= 3000 * 1000
            assert flight.origin_country in (
                'US',
                'CA',
            ) or flight.destination_country in ('US', 'CA')
            day = days_since_epoch(flight.departure)
            if last_day is not None:
                gap = day - last_day
                assert gap % 5 == 0
                if gap > 0:
                    saw_nonzero_gap = True
            last_day = day
            nflights += 1
        # `every_nth` is deterministic (no sampling RNG), so the count is
        # reproducible against this exact test database — pinning it
        # catches a regression that, say, drops every nth row instead of
        # including only every nth day.
        assert nflights == 78
        # Without at least one positive gap, the modulo check only ran on
        # same-day flights where `0 % 5 == 0` passes trivially — i.e. the
        # every_nth=5 contract was never actually exercised.
        assert saw_nonzero_gap

        # Frequent flight query.
        result = db(FrequentFlightQuery(Filter(airport='DTW')))
        assert isinstance(result, Generator)
        results = list(result)
        assert results, 'frequent-flight query returned no rows'
        # DTW must appear on every row, not just the first — a regression
        # that returned matching pairs only at the head of the result would
        # otherwise pass.
        for r in results:
            assert 'DTW' in (r.airport1, r.airport2)
        assert sum(r.number_of_flights for r in results) == 13


def test_mission_from_query_result_row():
    """`QueryResult.from_row` is on the critical path from DB query →
    downstream simulation. Cover the timestamp / IATA / load-factor
    placeholder mapping with a synthetic row that uses real airport codes
    so `gc_distance` (and thus `origin_position` / `destination_position`)
    can be checked too."""
    # Row order matches the SELECT in `Query.to_sql`:
    # (departure_ts, arrival_ts, schedule_id, flight_id, carrier,
    #  flight_number, origin_iata, origin_country, destination_iata,
    #  destination_country, service_type, aircraft_type, engine_type,
    #  distance, seat_capacity)
    dep_ts = int(date_to_timestamp(date(2024, 6, 1)).timestamp())
    arr_ts = dep_ts + 6 * 3600
    row = (
        dep_ts,
        arr_ts,
        4242,  # schedule id → flight_id field on Mission
        7,  # flight number id
        'AA',
        '100',
        'BOS',
        'US',
        'LAX',
        'US',
        'J',
        '738',
        'CFM56',
        4170,
        180,
    )

    mission = QueryResult.from_row(row)

    assert mission.origin == 'BOS'
    assert mission.destination == 'LAX'
    assert mission.aircraft_type == '738'
    assert mission.carrier == 'AA'
    assert mission.flight_number == '100'
    assert mission.origin_country == 'US'
    assert mission.destination_country == 'US'
    assert mission.service_type == 'J'
    assert mission.engine_type == 'CFM56'
    assert mission.seat_capacity == 180
    # `flight_id` comes from row[2] (the schedule id), not row[3].
    assert mission.flight_id == 4242
    # OAG data has no load factor, so the SUT inserts 1.0 as a placeholder.
    assert mission.load_factor == 1.0
    # Timestamps survive the int → UTC pd.Timestamp round-trip.
    assert mission.departure == pd.Timestamp(dep_ts, unit='s', tz='UTC')
    assert mission.arrival == pd.Timestamp(arr_ts, unit='s', tz='UTC')
    # `gc_distance` is in metres; BOS-LAX great-circle is ~4170 km.
    assert mission.gc_distance == pytest.approx(4170_000, rel=0.01)
    assert mission.label == 'BOS_LAX_738'


def test_mission_from_toml_minimal():
    """`Mission.from_toml` is the entry point for the sample-mission TOML
    fixture used elsewhere in the suite. Pin the field mapping for a
    minimal two-flight payload, including the geographic identity that
    `gc_distance` should match `GEOD.inv` directly on the airport
    positions.
    """
    data = {
        'flight': [
            {
                'origin': 'BOS',
                'destination': 'LAX',
                'departure': '2024-06-01T08:00:00+00:00',
                'arrival': '2024-06-01T14:00:00+00:00',
                'load_factor': 0.85,
                'aircraft_type': '738',
            },
            {
                'origin': 'JFK',
                'destination': 'ORD',
                'departure': '2024-06-02T12:00:00+00:00',
                'arrival': '2024-06-02T14:00:00+00:00',
                'load_factor': 1.0,
                'aircraft_type': '739',
            },
        ]
    }

    missions = Mission.from_toml(data)
    assert len(missions) == 2
    assert all(isinstance(m, Mission) for m in missions)
    m0, m1 = missions

    assert m0.origin == 'BOS'
    assert m0.destination == 'LAX'
    assert m0.aircraft_type == '738'
    assert m0.load_factor == 0.85
    assert m0.departure == pd.Timestamp('2024-06-01T08:00:00+00:00')
    assert m0.arrival == pd.Timestamp('2024-06-01T14:00:00+00:00')
    # Optional fields default to None when from_toml is given the minimal
    # required set.
    assert m0.carrier is None
    assert m0.flight_id is None

    # Geographic identity: gc_distance must match GEOD directly on the
    # cached airport positions for both flights.
    for m in (m0, m1):
        expected = GEOD.inv(
            m.origin_position.longitude,
            m.origin_position.latitude,
            m.destination_position.longitude,
            m.destination_position.latitude,
        )[2]
        assert m.gc_distance == pytest.approx(expected, rel=1e-9)


def test_set_random_seed_determinism(test_data_dir):
    """`Database.set_random_seed()` is the documented reproducibility entry
    point for sampling queries (see CLAUDE.md). Two databases seeded with
    the same value must yield identical schedule sequences; a different
    seed must yield a different sequence (otherwise sampling is not
    actually using the seeded RNG).
    """
    test_db = test_data_dir / 'missions/oag-2019-test-subset.sqlite'

    def sample_ids(seed: int) -> list[int]:
        with Database(test_db) as db:
            db.set_random_seed(seed)
            return [f.flight_id for f in db(Query(sample=0.1))]

    ids_a = sample_ids(42)
    ids_b = sample_ids(42)
    ids_c = sample_ids(43)

    assert ids_a, 'sampling returned no rows — test cannot validate determinism'
    assert ids_a == ids_b
    assert ids_a != ids_c
