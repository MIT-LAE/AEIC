import os
from collections.abc import Generator
from datetime import date

from missions import BoundingBox, Database, Filter, FrequentFlightQuery, Query
from utils.helpers import date_to_timestamp


def test_filter():
    # Simple range filter.
    cond, params = Filter(min_distance=1000, max_distance=5000).to_sql()
    assert cond == 'distance >= ? AND distance <= ?'
    assert params == [1000, 5000]

    # Combined airport filter: is either origin or destination a given airport?
    cond, params = Filter(airport='LHR').to_sql()
    assert cond == (
        '(origin IN (SELECT id FROM airports WHERE iata_code IN (?)) OR '
        'destination IN (SELECT id FROM airports WHERE iata_code IN (?)))'
    )
    assert params == ['LHR', 'LHR']

    # Combined country filter: is either origin or destination (or both) in the
    # given countries?
    cond, params = Filter(country='US').to_sql()
    assert cond == (
        '(origin IN (SELECT id FROM airports WHERE country IN (?)) OR '
        'destination IN (SELECT id FROM airports WHERE country IN (?)))'
    )
    assert params == ['US', 'US']

    # Combined continent filter: is either origin or destination (or both) in
    # the given continents?
    cond, params = Filter(continent='SA').to_sql()
    assert cond == (
        '(origin IN (SELECT id FROM airports WHERE country IN '
        '(SELECT code FROM countries WHERE continent IN (?))) OR '
        'destination IN (SELECT id FROM airports WHERE country IN '
        '(SELECT code FROM countries WHERE continent IN (?))))'
    )
    assert params == ['SA', 'SA']

    # Origin airport only.
    cond, params = Filter(origin_airport='LAX').to_sql()
    assert cond == ('origin IN (SELECT id FROM airports WHERE iata_code IN (?))')
    assert params == ['LAX']

    # Origin country only.
    cond, params = Filter(origin_country='US').to_sql()
    assert cond == ('origin IN (SELECT id FROM airports WHERE country IN (?))')
    assert params == ['US']

    # Bounding box for Austria: either origin or destination or both.
    bbox = BoundingBox(
        min_longitude=11.343,
        max_longitude=16.570,
        min_latitude=46.642,
        max_latitude=48.234,
    )
    cond, params = Filter(bounding_box=bbox).to_sql()
    assert cond == (
        '(origin IN (SELECT id FROM airport_location_idx '
        'WHERE min_latitude >= ? AND max_latitude <= ? AND '
        'min_longitude >= ? AND max_longitude <= ?) OR '
        'destination IN (SELECT id FROM airport_location_idx '
        'WHERE min_latitude >= ? AND max_latitude <= ? AND '
        'min_longitude >= ? AND max_longitude <= ?))'
    )
    assert params == [46.642, 48.234, 11.343, 16.57, 46.642, 48.234, 11.343, 16.57]

    # Complex spatial filter #1: origin in Austria, destination in Germany.
    cond, params = Filter(origin_country='AT', destination_country='DE').to_sql()
    assert cond == (
        'origin IN (SELECT id FROM airports WHERE country IN (?)) AND '
        'destination IN (SELECT id FROM airports WHERE country IN (?))'
    )
    assert params == ['AT', 'DE']

    # Complex spatial filter #2: origin in France, destination in South
    # America.
    cond, params = Filter(origin_country='FR', destination_continent='SA').to_sql()
    assert cond == (
        'origin IN (SELECT id FROM airports WHERE country IN (?)) AND '
        'destination IN (SELECT id FROM airports WHERE country IN '
        '(SELECT code FROM countries WHERE continent IN (?)))'
    )
    assert params == ['FR', 'SA']

    # Simple service type filter.
    cond, params = Filter(service_type=['J', 'S', 'Q']).to_sql()
    assert cond == 'service_type IN (?, ?, ?)'
    assert params == ['J', 'S', 'Q']

    # Simple aircraft type filter.
    cond, params = Filter(aircraft_type=['B737', '777']).to_sql()
    assert cond == 'aircraft_type IN (?, ?)'
    assert params == ['B737', '777']

    # Combined filter.
    cond, params = Filter(
        min_distance=2000, min_seat_capacity=200, country=['US', 'CA']
    ).to_sql(table='f')
    assert cond == (
        'f.distance >= ? AND f.seat_capacity >= ? AND '
        '(f.origin IN (SELECT id FROM airports WHERE country IN (?, ?)) OR '
        'f.destination IN (SELECT id FROM airports WHERE country IN (?, ?)))'
    )
    assert params == [2000, 200, 'US', 'CA', 'US', 'CA']


def test_query():
    sql, params = Query().to_sql()
    assert sql == (
        'SELECT s.departure_timestamp, s.arrival_timestamp, '
        'f.id as flight_id, f.carrier, f.flight_number, '
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
    assert 'random()' in sql

    sql, params = Query(
        filter=Filter(country='US', min_distance=1000, max_distance=5000),
        every_nth=8,
    ).to_sql()
    assert params == [1000, 5000, 'US', 'US', 8]
    assert 'SELECT MIN(day) FROM schedules' in sql

    sql, params = FrequentFlightQuery(
        filter=Filter(origin_country='US'), limit=10
    ).to_sql()
    assert params == ['US']
    assert 'GROUP BY od_pair' in sql


def test_query_result():
    # These queries were all tested manually in the SQLite shell to determine
    # the correct results using this exact test database.
    test_db = os.path.join(os.path.dirname(__file__), 'oag-2019-test-subset.sqlite')
    with Database(test_db) as db:
        # All scheduled flights in the test database.
        result = db(Query())
        assert isinstance(result, Generator)
        nflights = len(list(result))
        assert nflights == 1589

        # Simple distance filter.
        nflights = 0
        result = db(Query(Filter(min_distance=3000)))
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.distance >= 3000
            nflights += 1
        assert nflights == 165

        # Country filter: either origin or destination in the given country.
        nflights = 0
        result = db(Query(Filter(country='IT')))
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.origin_country == 'IT' or flight.destination_country == 'IT'
            nflights += 1
        assert nflights == 64

        # Combined filter.
        nflights = 0
        q = Query(Filter(max_distance=3000, country=['US', 'CA']))
        result = db(q)
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.distance <= 3000
            assert flight.origin_country in (
                'US',
                'CA',
            ) or flight.destination_country in ('US', 'CA')
            nflights += 1
        assert nflights == 353

        # Sampling.
        q2 = q
        q2.sample = 0.1
        nflights = 0
        result = db(q2)
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.distance <= 3000
            assert flight.origin_country in (
                'US',
                'CA',
            ) or flight.destination_country in ('US', 'CA')
            nflights += 1
        # With a 10% sample, we should get between 40 and 90 flights but for
        # testing it's too dodgy to assert that. All we can say with complete
        # certainty is that there should be less than the full 523 flights.
        assert nflights < 353

        # "Every nth day" filtering.
        q3 = q
        q3.every_nth = 5
        nflights = 0
        last_day = -1
        result = db(q3)
        assert isinstance(result, Generator)
        for flight in result:
            assert flight.distance <= 3000
            assert flight.origin_country in (
                'US',
                'CA',
            ) or flight.destination_country in ('US', 'CA')
            day = flight.departure.dayofyear
            if last_day > 0:
                assert (day - last_day) % 5 == 0
            last_day = day
            nflights += 1
        assert nflights < 353

        # Frequent flight query.
        result = db(FrequentFlightQuery(Filter(airport='DTW')))
        assert isinstance(result, Generator)
        results = list(result)
        assert results[0].airport1 == 'DTW' or results[0].airport2 == 'DTW'
        assert sum(r.number_of_flights for r in results) == 41
