import utils.airports as airports
from missions.writable_database import WritableDatabase
from utils.custom_types import DayOfWeek


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
    assert WritableDatabase._make_dow_mask(set()) == 0b0000000


def test_airport_handling(tmp_path):
    db = WritableDatabase(tmp_path / 'test.sqlite')
    cur = db._conn.cursor()

    tz = db._lookup_timezone(airports.airports['LHR'])
    assert tz == 'Europe/London'

    # Real airport.
    airport_info = db._get_or_add_airport(cur, 1234, 'CDG')
    print(airport_info)
    assert airport_info.airport.iata_code == 'CDG'
    assert airport_info.airport.country == 'FR'
    assert airport_info.airport.municipality.startswith('Paris')
    assert int(airport_info.airport.latitude) == 49
    assert airport_info.timezone == 'Europe/Paris'

    # Not a real airport.
    assert db._get_or_add_airport(cur, 1235, 'QPX') is None
