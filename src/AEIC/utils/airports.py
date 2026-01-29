import csv
import logging
import os
from dataclasses import dataclass

from AEIC.config import config
from AEIC.types import Position
from AEIC.units import FEET_TO_METERS
from AEIC.utils.files import download

logger = logging.getLogger(__name__)


@dataclass
class Country:
    """Country data, including ISO 3166-1 alpha-2 code, name, and continent."""

    code: str
    """ISO 3166-1 alpha-2 country code."""

    name: str
    """Country name."""

    continent: str
    """Continent code (e.g., 'EU' for Europe)."""


class CountriesData:
    """Access wrapper for OurAirports countries data."""

    def __init__(self):
        with open(_data_file('countries'), newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            self._countries = {
                row['code']: Country(
                    code=row['code'], name=row['name'], continent=row['continent']
                )
                for row in reader
            }

    def __getitem__(self, code: str) -> Country | None:
        return self._countries.get(code)


@dataclass
class Airport:
    """Airport data."""

    iata_code: str
    """IATA airport code."""

    name: str
    """Airport name."""

    latitude: float
    """Latitude in decimal degrees."""

    longitude: float
    """Longitude in decimal degrees."""

    elevation: float | None
    """Elevation in meters above sea level, or None if not available."""

    country: str
    """ISO 3166-1 alpha-2 country code."""

    municipality: str | None
    """Municipality (city) where the airport is located, or None if not
    available."""

    @property
    def position(self) -> Position:
        """Get the geographic position of the airport."""
        return Position(
            longitude=self.longitude,
            latitude=self.latitude,
            altitude=self.elevation or 0.0,
        )


class AirportsData:
    """Access wrapper for OurAirports airports data."""

    def __init__(self):
        self._airports = self._read_file(_data_file('airports'))

        # Some historical airports are missing from the main data file, so add
        # them here from a supplemental CSV file.
        self._airports.update(
            self._read_file(config.data_file_location('airports/airports-patch.csv'))
        )

    def _read_file(self, f: str) -> dict[str, Airport]:
        """Read airport data from an OurAirports CSV file."""

        with open(f, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return {
                row['iata_code']: Airport(
                    iata_code=row['iata_code'],
                    name=row['name'],
                    latitude=float(row['latitude_deg']),
                    longitude=float(row['longitude_deg']),
                    elevation=(
                        float(row['elevation_ft']) * FEET_TO_METERS
                        if row['elevation_ft']
                        else None
                    ),
                    country=row['iso_country'],
                    municipality=row['municipality'] if row['municipality'] else None,
                )
                for row in reader
                if row['iata_code']
            }

    def __getitem__(self, code: str) -> Airport | None:
        return self._airports.get(code)


def _data_file(f: str) -> str:
    """Lazy download of OurAirports data."""

    base_url = 'https://davidmegginson.github.io/ourairports-data'
    data_file = config.default_data_file_location(f'airports/{f}.csv', missing_ok=True)
    if not os.path.exists(data_file):
        logger.info('Downloading %s data file', f)
        download(f'{base_url}/{f}.csv', data_file)
    return data_file


# Package-level private names to allow lazy initialization of data objects.

_countries: CountriesData | None = None
_airports: AirportsData | None = None


def country(code: str) -> Country | None:
    """Retrieve country data by ISO 3166-1 alpha-2 code."""
    global _countries
    if _countries is None:
        _countries = CountriesData()
    return _countries[code]


def airport(code: str) -> Airport | None:
    """Retrieve airport data by IATA code."""
    global _airports
    if _airports is None:
        _airports = AirportsData()
    return _airports[code]
