from dataclasses import dataclass


@dataclass
class BoundingBox:
    """A bounding box defined by latitude and longitude ranges.

    Used for filtering airports within a specific geographic area."""

    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float


@dataclass
class Filter:
    """A filter for narrowing down mission flight schedule entries.

    This supports filtering by various criteria such as distance, seat
    capacity, aircraft type, and by geographic location using country,
    continent, or a bounding box.

    All conditions are combined with **AND** logic.

    Spatial filters come in two flavors, "combined" and "origin/destination",
    and four region types, "airport", "country", "continent", and "bounding
    box".

    Combined spatial filters apply to both the origin and destination of
    flights and express conditions like "flight departs or arrives from an
    airport in Malaysia". Origin/destination spatial filters apply only to
    either the origin or destination of flights. Applying both an origin and a
    destination filter makes it possible to represent conditions like "flight
    departs from an airport in France AND arrives at an airport in South
    America".

    A combined spatial filter cannot be used together with either an origin or
    destination spatial filter. It is valid to specify either:

    - a single combined spatial filter, or

    - one (optional) spatial filter for the origin and/or one (optional) one
      for the destination.

    This means that, for example, you can specify either: `country` **or**
    `origin_country` and/or `destination_country` for country-based filtering,
    but you may not specify both `country` and either of `origin_country` or
    `destination_country`. And similarly for airport, continent and bounding
    box filtering.

    Here are some examples:

    - `country='US'` means flights originating **or** terminating in the US.

    - `origin_country='FR'` means flights originating in France.

    - `destination_country='AT'` means flights terminating in Austria.

    - `origin_country=['US', 'CA'], destination_country='MX'` means flights
      originating in the US or Canada **and** terminating in Mexico.

    - `origin_country=['DE', 'CH', 'AT']`, `destination_continent='SA'` means
      flights originating in either Germany, Switzerland, or Austria **and**
      terminating at any airport in South America.

    """

    min_distance: float | None = None
    """Minimum flight distance in kilometers."""
    max_distance: float | None = None
    """Maximum flight distance in kilometers."""

    min_seat_capacity: int | None = None
    """Minimum seat capacity."""
    max_seat_capacity: int | None = None
    """Maximum seat capacity."""

    airport: str | list[str] | None = None
    """Originating or terminating airport code(s)."""
    origin_airport: str | list[str] | None = None
    """Originating airport code(s)."""
    destination_airport: str | list[str] | None = None
    """Terminating airport code(s)."""

    country: str | list[str] | None = None
    """Originating or terminating country code(s)."""
    origin_country: str | list[str] | None = None
    """Originating country code(s)."""
    destination_country: str | list[str] | None = None
    """Terminating country code(s)."""

    continent: str | list[str] | None = None
    """Originating or terminating continent code(s)."""
    origin_continent: str | list[str] | None = None
    """Originating continent code(s)."""
    destination_continent: str | list[str] | None = None
    """Terminating continent code(s)."""

    bounding_box: BoundingBox | None = None
    """Bounding box for originating or terminating airports."""
    origin_bounding_box: BoundingBox | None = None
    """Bounding box for originating airports."""
    destination_bounding_box: BoundingBox | None = None
    """Bounding box for terminating airports."""

    service_type: str | list[str] | None = None
    """Service type(s) (using IATA single-letter codes, documented `here
    <https://knowledge.oag.com/v1/docs/iata-service-type-codes>`__)."""
    aircraft_type: str | list[str] | None = None
    """Aircraft type(s) (e.g., '737', '320')."""

    def to_sql(self, table: str | None = None) -> tuple[str, list]:
        """Convert the filter to SQL conditions and parameters.

        For each filter attribute that is set, a SQL condition and appropriate
        parameters to match the placeholders in the condition is generated. The
        conditions are combined with AND logic.
        """

        # Prefix table name to column names if given.
        if table is not None:
            table += '.'
        else:
            table = ''

        # Check and normalize filter attributes.
        self._normalize()

        conditions = []

        # Simple numeric range filters.
        def simple(expr, value):
            if value is not None:
                conditions.append((expr, value))

        simple(f'{table}distance >= ?', self.min_distance)
        simple(f'{table}distance <= ?', self.max_distance)
        simple(f'{table}seat_capacity >= ?', self.min_seat_capacity)
        simple(f'{table}seat_capacity <= ?', self.max_seat_capacity)

        # Other simple filters.
        if self.service_type is not None and len(self.service_type) > 0:
            placeholders = ', '.join('?' * len(self.service_type))
            conditions.append(
                (f'{table}service_type IN ({placeholders})', self.service_type)
            )
        if self.aircraft_type is not None and len(self.aircraft_type) > 0:
            placeholders = ', '.join('?' * len(self.aircraft_type))
            conditions.append(
                (f'{table}aircraft_type IN ({placeholders})', self.aircraft_type)
            )

        # Complex filters involving sub-selects.
        conditions += self._airport_condition(table)
        conditions += self._country_condition(table)
        conditions += self._continent_condition(table)
        conditions += self._bounding_box_condition(table)

        conds, params = list(zip(*conditions))
        return (
            ' AND '.join(conds),
            [p for ps in params for p in (ps if isinstance(ps, list) else [ps])],
        )

    def _airport_condition(self, table: str) -> list[tuple[str, list[str]]]:
        """Generate SQL conditions and parameters for airport-based filtering.

        Handles both combined and origin/destination-specific airport filters.
        """

        def sub_select_for(airports: list[str]) -> str:
            return (
                '(SELECT id FROM airports '
                f'WHERE iata_code IN ({", ".join("?" * len(airports))}))'
            )

        # Combined origin/destination airport filter.
        if self.airport is not None:
            assert isinstance(self.airport, list)
            sub_select = sub_select_for(self.airport)
            return [
                (
                    f'({table}origin IN {sub_select} OR '
                    f'{table}destination IN {sub_select})',
                    self.airport + self.airport,
                )
            ]

        # Origin/destination-specific airport filters.
        conds = []
        if self.origin_airport is not None:
            assert isinstance(self.origin_airport, list)
            conds.append(
                (
                    f'{table}origin IN {sub_select_for(self.origin_airport)}',
                    self.origin_airport,
                )
            )
        if self.destination_airport is not None:
            assert isinstance(self.destination_airport, list)
            conds.append(
                (
                    f'{table}destination IN {sub_select_for(self.destination_airport)}',
                    self.destination_airport,
                )
            )
        return conds

    def _country_condition(self, table: str) -> list[tuple[str, list[str]]]:
        """Generate SQL conditions and parameters for country-based filtering.

        Handles both combined and origin/destination-specific country filters.
        """

        def sub_select_for(countries: list[str]) -> str:
            return (
                '(SELECT id FROM airports '
                f'WHERE country IN ({", ".join("?" * len(countries))}))'
            )

        # Combined origin/destination country filter.
        if self.country is not None:
            assert isinstance(self.country, list)
            sub_select = sub_select_for(self.country)
            return [
                (
                    f'({table}origin IN {sub_select} OR '
                    f'{table}destination IN {sub_select})',
                    self.country + self.country,
                )
            ]

        # Origin/destination-specific country filters.
        conds = []
        if self.origin_country is not None:
            assert isinstance(self.origin_country, list)
            conds.append(
                (
                    f'{table}origin IN {sub_select_for(self.origin_country)}',
                    self.origin_country,
                )
            )
        if self.destination_country is not None:
            assert isinstance(self.destination_country, list)
            conds.append(
                (
                    f'{table}destination IN {sub_select_for(self.destination_country)}',
                    self.destination_country,
                )
            )
        return conds

    def _continent_condition(self, table: str) -> list[tuple[str, list[str]]]:
        """Generate SQL conditions and parameters for continent-based
        filtering.

        Handles both combined and origin/destination-specific continent
        filters.
        """

        def sub_select_for(continents: list[str]) -> str:
            return (
                '(SELECT id FROM airports WHERE country IN '
                '(SELECT code FROM countries WHERE continent IN '
                f'({", ".join("?" * len(continents))})))'
            )

        # Combined origin/destination continent filter.
        if self.continent is not None:
            assert isinstance(self.continent, list)
            sub_select = sub_select_for(self.continent)
            return [
                (
                    f'({table}origin IN {sub_select} OR '
                    f'{table}destination IN {sub_select})',
                    self.continent + self.continent,
                )
            ]

        # Origin/destination-specific continent filters.
        conds = []
        if self.origin_continent is not None:
            assert isinstance(self.origin_continent, list)
            conds.append(
                (
                    f'{table}origin IN {sub_select_for(self.origin_continent)}',
                    self.origin_continent,
                )
            )
        if self.destination_continent is not None:
            assert isinstance(self.destination_continent, list)
            conds.append(
                (
                    f'{table}destination IN '
                    + sub_select_for(self.destination_continent),
                    self.destination_continent,
                )
            )
        return conds

    def _bounding_box_condition(self, table: str) -> list[tuple[str, list[float]]]:
        """Generate SQL conditions and parameters for bounding box-based
        filtering.

        Handles both combined and origin/destination-specific bounding box
        filters.
        """

        sub_select = (
            '(SELECT id FROM airport_location_idx '
            'WHERE min_latitude >= ? AND max_latitude <= ? '
            'AND min_longitude >= ? AND max_longitude <= ?)'
        )

        # Combined origin/destination bounding box filter.
        if self.bounding_box is not None:
            return [
                (
                    f'({table}origin IN {sub_select} OR '
                    f'{table}destination IN {sub_select})',
                    [
                        self.bounding_box.min_latitude,
                        self.bounding_box.max_latitude,
                        self.bounding_box.min_longitude,
                        self.bounding_box.max_longitude,
                        self.bounding_box.min_latitude,
                        self.bounding_box.max_latitude,
                        self.bounding_box.min_longitude,
                        self.bounding_box.max_longitude,
                    ],
                )
            ]

        # Origin/destination-specific bounding box filters.
        conds = []
        if self.origin_bounding_box is not None:
            conds.append(
                (
                    f'{table}origin IN {sub_select}',
                    [
                        self.origin_bounding_box.min_latitude,
                        self.origin_bounding_box.max_latitude,
                        self.origin_bounding_box.min_longitude,
                        self.origin_bounding_box.max_longitude,
                    ],
                )
            )
        if self.destination_bounding_box is not None:
            conds.append(
                (
                    f'{table}destination IN {sub_select}',
                    [
                        self.destination_bounding_box.min_latitude,
                        self.destination_bounding_box.max_latitude,
                        self.destination_bounding_box.min_longitude,
                        self.destination_bounding_box.max_longitude,
                    ],
                )
            )
        return conds

    def _normalize(self):
        """Normalize filter attributes and check for consistency."""

        # Convert single string attributes to lists of strings.
        for attr in [
            'airport',
            'origin_airport',
            'destination_airport',
            'country',
            'origin_country',
            'destination_country',
            'continent',
            'origin_continent',
            'destination_continent',
            'service_type',
            'aircraft_type',
        ]:
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, [getattr(self, attr)])

        # Check compatibility of spatial filters.
        combined, origin, destination = tuple(
            map(
                sum,
                zip(
                    self._spatial('airport', lists=True),
                    self._spatial('country', lists=True),
                    self._spatial('continent', lists=True),
                    self._spatial('bounding_box', lists=False),
                ),
            )
        )
        ok = (combined == 1 and origin == 0 and destination == 0) or (
            combined == 0 and origin <= 1 and destination <= 1
        )
        if not ok:
            raise ValueError('An invalid combination of spatial filters was provided. ')

    def _spatial(self, attr: str, lists: bool = False) -> tuple[int, int, int]:
        """Determine which spatial filters of the given type are set."""

        both = getattr(self, attr)
        origin = getattr(self, 'origin_' + attr)
        destination = getattr(self, 'destination_' + attr)

        if not lists:
            return (
                1 if both is not None else 0,
                1 if origin is not None else 0,
                1 if destination is not None else 0,
            )
        else:
            assert both is None or isinstance(both, list)
            assert origin is None or isinstance(origin, list)
            assert destination is None or isinstance(destination, list)
            return (
                1 if (both is not None and len(both) > 0) else 0,
                1 if (origin is not None and len(origin) > 0) else 0,
                1 if (destination is not None and len(destination) > 0) else 0,
            )
