import argparse
import csv
import tomllib
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MISSIONS = REPO_ROOT / 'tests/data/verification/legacy/missions.toml'
DEFAULT_OUTPUT = REPO_ROOT / 'tests/data/verification/legacy/matlab-schedule.csv'
DEFAULT_AIRPORTS = REPO_ROOT / 'tests/data/airports/airports.csv'
DEFAULT_AIRPORTS_OUTPUT = (
    REPO_ROOT / 'tests/data/verification/legacy/matlab-airports.csv'
)


def compact_time(value: datetime) -> str:
    return value.strftime('%H%M')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--missions', type=Path, default=DEFAULT_MISSIONS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--airports', type=Path, default=DEFAULT_AIRPORTS)
    parser.add_argument('--airports-output', type=Path, default=DEFAULT_AIRPORTS_OUTPUT)
    args = parser.parse_args()

    with args.missions.open('rb') as fp:
        flights = tomllib.load(fp)['flight']

    fieldnames = [
        'depapt',
        'arrapt',
        'deptim',
        'arrtim',
        'days',
        'inpacft',
        'seats',
        'efffrom',
        'effto',
        'NFlts',
    ]
    with args.output.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for flight in flights:
            departure = datetime.fromisoformat(flight['departure'])
            arrival = datetime.fromisoformat(flight['arrival'])
            service_date = departure.strftime('%Y%m%d')
            writer.writerow(
                {
                    'depapt': flight['origin'],
                    'arrapt': flight['destination'],
                    'deptim': compact_time(departure),
                    'arrtim': compact_time(arrival),
                    'days': departure.isoweekday(),
                    'inpacft': flight['aircraft_type'],
                    'seats': flight['seats'],
                    'efffrom': service_date,
                    'effto': service_date,
                    'NFlts': 1,
                }
            )

    airport_codes = {
        code for flight in flights for code in (flight['origin'], flight['destination'])
    }
    with args.airports.open(newline='', encoding='utf-8') as fp:
        airports = {
            row['iata_code']: row
            for row in csv.DictReader(fp)
            if row['iata_code'] in airport_codes
        }
    missing = airport_codes - airports.keys()
    if missing:
        parser.error(f'missing airport data for: {", ".join(sorted(missing))}')

    # Match the format used by MATLAB AEIC textscan call in
    # readAllAirportData.m.
    airport_fields = [
        'Code',
        'ICAO Code',
        'Country Code',
        'Latitude',
        'Longitude',
        'Elev_ft',
        'Nrunways',
        'Longest_runway_ft',
    ]
    with args.airports_output.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=airport_fields, lineterminator='\n')
        writer.writeheader()
        for code in sorted(airport_codes):
            airport = airports[code]
            writer.writerow(
                {
                    'Code': code,
                    'ICAO Code': airport['ident'],
                    'Country Code': airport['iso_country'],
                    'Latitude': airport['latitude_deg'],
                    'Longitude': airport['longitude_deg'],
                    'Elev_ft': airport['elevation_ft'] or 0,
                    'Nrunways': 1,
                    'Longest_runway_ft': 0,
                }
            )


if __name__ == '__main__':
    main()
