import os
from collections.abc import Generator
from datetime import date

import missions.oag as oag
from missions.custom_types import DayOfWeek
from utils.helpers import date_to_timestamp


def test_dow_mask():
    assert (
        oag.Database._make_dow_mask(
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
        oag.Database._make_dow_mask(
            {DayOfWeek.MONDAY, DayOfWeek.WEDNESDAY, DayOfWeek.FRIDAY}
        )
        == 0b00010101
    )
    assert (
        oag.Database._make_dow_mask({DayOfWeek.TUESDAY, DayOfWeek.THURSDAY})
        == 0b00001010
    )
    assert oag.Database._make_dow_mask({DayOfWeek.SUNDAY}) == 0b01000000
    assert oag.Database._make_dow_mask(set()) == 0b0000000


def test_oag_filter():
    # Simple range filter.
    cond, params = oag.Filter(min_distance=1000, max_distance=5000).to_sql()
    assert cond == 'distance >= ? AND distance <= ?'
    assert params == [1000, 5000]

    # Combined country filter: is either origin or destination (or both) in the
    # given countries?
    cond, params = oag.Filter(country='US').to_sql()
    assert cond == (
        '(origin IN (SELECT id FROM airports WHERE country IN (?)) OR '
        'destination IN (SELECT id FROM airports WHERE country IN (?)))'
    )
    assert params == ['US', 'US']

    # Combined continent filter: is either origin or destination (or both) in
    # the given continents?
    cond, params = oag.Filter(continent='SA').to_sql()
    assert cond == (
        '(origin IN (SELECT id FROM airports WHERE country IN '
        '(SELECT code FROM countries WHERE continent IN (?))) OR '
        'destination IN (SELECT id FROM airports WHERE country IN '
        '(SELECT code FROM countries WHERE continent IN (?))))'
    )
    assert params == ['SA', 'SA']

    # Origin country only.
    cond, params = oag.Filter(origin_country='US').to_sql()
    assert cond == ('origin IN (SELECT id FROM airports WHERE country IN (?))')
    assert params == ['US']

    # Bounding box for Austria: either origin or destination or both.
    bbox = oag.BoundingBox(
        min_longitude=11.343,
        max_longitude=16.570,
        min_latitude=46.642,
        max_latitude=48.234,
    )
    cond, params = oag.Filter(bounding_box=bbox).to_sql()
    assert cond == (
        '(origin IN (SELECT id FROM airport_location_idx '
        'WHERE min_latitude >= ? AND max_latitude <= ? AND '
        'min_longitude >= ? AND max_longitude <= ?) OR '
        'destination IN (SELECT id FROM airport_location_idx '
        'WHERE min_latitude >= ? AND max_latitude <= ? AND '
        'min_longitude >= ? AND max_longitude <= ?))'
    )
    assert params == [46.642, 48.234, 11.343, 16.57, 46.642, 48.234, 11.343, 16.57]

    # Simple aircraft type filter.
    cond, params = oag.Filter(aircraft_type=['B737', '777']).to_sql()
    assert cond == 'aircraft_type IN (?, ?)'
    assert params == ['B737', '777']

    # Combined filter.
    cond, params = oag.Filter(
        min_distance=2000, min_seat_capacity=200, country=['US', 'CA']
    ).to_sql(table='f')
    assert cond == (
        'f.distance >= ? AND f.seat_capacity >= ? AND '
        '(f.origin IN (SELECT id FROM airports WHERE country IN (?, ?)) OR '
        'f.destination IN (SELECT id FROM airports WHERE country IN (?, ?)))'
    )
    assert params == [2000, 200, 'US', 'CA', 'US', 'CA']


def test_oag_query():
    sql, params = oag.Query().to_sql()
    assert sql == (
        'SELECT s.departure_timestamp, s.arrival_timestamp, '
        'f.id as flight_id, f.carrier, f.flight_number, '
        'ao.iata_code AS origin, ao.country AS origin_country, '
        'ad.iata_code AS destination, ad.country AS destination_country, '
        'f.aircraft_type, f.engine_type, f.distance, f.seat_capacity '
        'FROM schedules s '
        'JOIN flights f ON f.id = s.flight_id '
        'JOIN airports ao ON f.origin = ao.id '
        'JOIN airports ad ON f.destination = ad.id '
        'ORDER BY s.departure_timestamp'
    )
    assert 'WHERE' not in sql
    assert len(params) == 0

    sql, params = oag.Query(
        filter=oag.Filter(min_distance=1000, max_distance=5000)
    ).to_sql()
    assert 'WHERE f.distance >= ? AND f.distance <= ?' in sql
    assert params == [1000, 5000]

    sql, params = oag.Query(
        filter=oag.Filter(country='US'),
        start_date=date(2024, 3, 1),
        end_date=date(2024, 8, 31),
    ).to_sql()
    assert params == [
        'US',
        'US',
        int(date_to_timestamp(date(2024, 3, 1)).timestamp()),
        int(date_to_timestamp(date(2024, 9, 1)).timestamp()),
    ]

    sql, params = oag.Query(
        filter=oag.Filter(country='US', min_seat_capacity=250), sample=0.05
    ).to_sql()
    assert params == [250, 'US', 'US', 0.05]
    assert 'random()' in sql

    sql, params = oag.Query(
        filter=oag.Filter(country='US', min_distance=1000, max_distance=5000),
        every_nth=8,
    ).to_sql()
    assert params == [1000, 5000, 'US', 'US', 8]
    assert 'SELECT MIN(day) FROM schedules' in sql

    sql, params = oag.FrequentFlightQuery(
        filter=oag.Filter(origin_country='US'), limit=10
    ).to_sql()
    assert params == ['US']
    assert 'GROUP BY od_pair' in sql


def test_oag_query_result():
    # These queries were all tested manually in the SQLite shell to determine
    # the correct results using this exact test database.
    test_db = os.path.join(os.path.dirname(__file__), 'oag-2019-test-subset.sqlite')
    db = oag.Database(test_db)

    # All scheduled flights in the test database.
    result = db(oag.Query())
    assert isinstance(result, Generator)
    nflights = len(list(result))
    assert nflights == 1561

    # Simple distance filter.
    nflights = 0
    result = db(oag.Query(oag.Filter(min_distance=9100)))
    assert isinstance(result, Generator)
    for flight in result:
        assert flight.distance >= 9100
        nflights += 1
    assert nflights == 465

    # Country filter: either origin or destination in the given country.
    nflights = 0
    result = db(oag.Query(oag.Filter(country='MV')))
    assert isinstance(result, Generator)
    for flight in result:
        assert flight.origin_country == 'MV' or flight.destination_country == 'MV'
        nflights += 1
    assert nflights == 103

    # Combined filter.
    nflights = 0
    q = oag.Query(oag.Filter(max_distance=6900, country=['US', 'CA']))
    result = db(q)
    assert isinstance(result, Generator)
    for flight in result:
        assert flight.distance <= 6900
        assert flight.origin_country in ('US', 'CA') or flight.destination_country in (
            'US',
            'CA',
        )
        nflights += 1
    assert nflights == 523

    # Sampling.
    q2 = q
    q2.sample = 0.1
    nflights = 0
    result = db(q2)
    assert isinstance(result, Generator)
    for flight in result:
        assert flight.distance <= 6900
        assert flight.origin_country in ('US', 'CA') or flight.destination_country in (
            'US',
            'CA',
        )
        nflights += 1
    # With a 10% sample, we should get between 40 and 90 flights but for
    # testing it's too dodgy to assert that. All we can say with complete
    # certainty is that there should be less than the full 523 flights.
    assert nflights < 523

    # "Every nth day" filtering.
    q3 = q
    q3.every_nth = 5
    nflights = 0
    last_day = -1
    result = db(q3)
    assert isinstance(result, Generator)
    for flight in result:
        assert flight.distance <= 6900
        assert flight.origin_country in ('US', 'CA') or flight.destination_country in (
            'US',
            'CA',
        )
        day = flight.departure.dayofyear
        if last_day > 0:
            assert (day - last_day) % 5 == 0
        last_day = day
        nflights += 1
    assert nflights < 523

    # Frequent flight query.
    result = db(oag.FrequentFlightQuery())
    assert isinstance(result, Generator)
    results = list(result)
    assert results[0].airport1 == 'JFK' or results[0].airport2 == 'JFK'
    assert sum(r.number_of_flights for r in results) == 1561
