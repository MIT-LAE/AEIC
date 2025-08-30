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
    """A filter for narrowing down OAG flight schedule entries.

    This supports filtering by various criteria such as distance, seat
    capacity, aircraft type, and by geographic location using country,
    continent, or a bounding box.

    All conditions are combined with **AND** logic.

    Only one type of spatial filter (country, continent, or bounding box) may
    be specified at a time. Spatial filters may be applied to the origin, the
    destination or either end of a flight. For example, for country filtering,
    the following interpretations apply:

    - `country='US'` means flights originating **or** terminating in the US.

    - `origin_country='FR'` means flights originating in France.

    - `destination_country='AT'` means flights terminating in Austria.

    - `origin_country=['US', 'CA'], destination_country='MX'` means flights
      originating in the US or Canada **and** terminating in Mexico.

    Similar logic applies to continent and bounding box filters.

    This means that, for example, you can specify either: `country` **or**
    `origin_country` and/or `destination_country` for country-based filtering,
    but you may not specify both `country` and either of `origin_country` or
    `destination_country`.

    """

    min_distance: float | None = None
    """Minimum flight distance in kilometers."""
    max_distance: float | None = None
    """Maximum flight distance in kilometers."""

    min_seat_capacity: int | None = None
    """Minimum seat capacity."""
    max_seat_capacity: int | None = None
    """Maximum seat capacity."""

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
        if self.aircraft_type is not None and len(self.aircraft_type) > 0:
            placeholders = ', '.join('?' * len(self.aircraft_type))
            conditions.append(
                (f'{table}aircraft_type IN ({placeholders})', self.aircraft_type)
            )

        # Complex filters involving sub-selects.
        conditions += self._country_condition(table)
        conditions += self._continent_condition(table)
        conditions += self._bounding_box_condition(table)

        conds, params = list(zip(*conditions))
        return (
            ' AND '.join(conds),
            [p for ps in params for p in (ps if isinstance(ps, list) else [ps])],
        )

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
            'country',
            'origin_country',
            'destination_country',
            'continent',
            'origin_continent',
            'destination_continent',
            'aircraft_type',
        ]:
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, [getattr(self, attr)])

        # Check compatibility of spatial filters.
        spatial = (
            self._check_spatial('country', lists=True)
            + self._check_spatial('continent', lists=True)
            + self._check_spatial('bounding_box', lists=False)
        )
        if spatial > 1:
            raise ValueError(
                'Only one of country, continent, or '
                'bounding_box can be set in OAG filter'
            )

    def _check_spatial(self, attr: str, lists: bool = False) -> int:
        """Check that both combined and origin/destination location filters of
        the given type are not set."""

        both = getattr(self, attr)
        origin = getattr(self, 'origin_' + attr)
        destination = getattr(self, 'destination_' + attr)
        if both is not None and (origin is not None or destination is not None):
            raise ValueError(
                f'Cannot set both {attr} and origin_{attr} or '
                f'destination_{attr} in OAG filter'
            )
        if both is not None or origin is not None or destination is not None:
            if not lists:
                return 1
            return (
                1
                if (len(both or []) + len(origin or []) + len(destination or [])) > 0
                else 0
            )
        else:
            return 0
