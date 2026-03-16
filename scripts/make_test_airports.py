"""Create the tests/data/airports/airports.csv data file. This file contains
only the airports used in tests (from the sample missions and some extras). The
reason for doing this is to make sure that we have stable airport data for
testing: the airport data that we use is updated occasionally, and changes in
airport data can break tests."""

import csv
import tomllib
from pathlib import Path

from AEIC.config import Config, config
from AEIC.missions import Mission

Config.load()

TEST_DIR = Path(__file__).parent.parent.parent.parent / 'tests'
TEST_DATA_DIR = TEST_DIR / 'data'

# Extra airports used in tests that don't appear in the sample missions file.
EXTRA_TEST_AIRPORTS = {'LHR', 'CDG'}


def run():
    missions_file = config.file_location('missions/sample_missions_10.toml')
    with open(missions_file, 'rb') as f:
        mission_dict = tomllib.load(f)
    sample_missions = Mission.from_toml(mission_dict)

    airports = (
        set(m.origin for m in sample_missions)
        | set(m.destination for m in sample_missions)
        | EXTRA_TEST_AIRPORTS
    )

    (TEST_DATA_DIR / 'airports').mkdir(parents=True, exist_ok=True)
    with open(TEST_DATA_DIR / 'airports/airports.csv', 'w') as fp:
        print(
            '"id","ident","type","name","latitude_deg","longitude_deg",'
            '"elevation_ft","continent","iso_country","iso_region",'
            '"municipality","scheduled_service","icao_code","iata_code",'
            '"gps_code","local_code","home_link","wikipedia_link","keywords"',
            file=fp,
        )
        with open(
            config.default_data_file_location('airports/airports.csv'),
            newline='',
            encoding='utf-8',
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            assert reader.fieldnames is not None
            for row in reader:
                if row['iata_code'] in airports:
                    print(
                        ','.join(f'"{row[col]}"' for col in reader.fieldnames), file=fp
                    )


if __name__ == '__main__':
    run()
