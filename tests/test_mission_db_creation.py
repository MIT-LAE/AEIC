import AEIC.utils.airports as airports
from AEIC.missions import Database
from AEIC.missions.oag import convert_oag_data
from AEIC.missions.writable_database import WritableDatabase
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
    assert WritableDatabase._make_dow_mask(set()) == 0b0000000


def test_airport_handling(tmp_path):
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
        assert int(airport_info.airport.latitude) == 49
        assert airport_info.timezone == 'Europe/Paris'

        # Not a real airport.
        assert db._get_or_add_airport(cur, 1235, 'QPX') is None


def test_oag_conversion(tmp_path, test_data_dir):
    # This extract of the 2019 OAG data contains 7 valid flights.
    oag_file = test_data_dir / 'oag/2019-extract.csv'

    convert_oag_data(
        oag_file,
        2019,
        tmp_path / 'oag_test.sqlite',
        warnings_file=tmp_path / 'oag_warnings.txt',
    )

    with Database(tmp_path / 'oag_test.sqlite') as db:
        cur = db._conn.cursor()
        cur.execute('SELECT COUNT(*) FROM flights')
        assert cur.fetchone()[0] == 7
