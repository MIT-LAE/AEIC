# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from enum import IntEnum, auto


class Dimension(IntEnum):
    """Standard dimension names for NetCDF serialization.

    The sort order here is important: when serializing to NetCDF, dimensions
    should be written in the order defined here."""

    TRAJECTORY = auto()
    """Dimension for the number of trajectories."""

    SPECIES = auto()
    """Dimension for emissions species."""

    POINT = auto()
    """Dimension for points along trajectories."""

    THRUST_MODE = auto()
    """Dimension for LTO thrust modes."""

    @property
    def dim_name(self) -> str:
        """Dimension name for use in NetCDF files."""
        return self.name.lower()

    @classmethod
    def from_dim_name(cls, name: str) -> Dimension:
        """Convert from names used in NetCDF files."""
        try:
            return DIMENSION_NAME_TO_ENUM[name]
        except KeyError:
            raise ValueError(f'Unknown dimension name: {name}')


DIMENSION_NAME_TO_ENUM = {d.dim_name: d for d in Dimension}
"""Conversion dictionary for dimension names used in NetCDF files."""


class Dimensions:
    def __init__(self, *dims: Dimension):
        self.dims = frozenset(dims)
        if Dimension.TRAJECTORY not in self.dims:
            raise ValueError('Dimensions must include the trajectory')
        if Dimension.POINT in self.dims and Dimension.THRUST_MODE in self.dims:
            raise ValueError('Dimensions cannot include both point and thrust mode')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimensions):
            return False
        return self.dims == other.dims

    def __len__(self):
        return len(self.dims)

    def __hash__(self):
        return hash(self.dims)

    def __contains__(self, item: Dimension):
        return item in self.dims

    def add(self, dim: Dimension) -> Dimensions:
        """Add a dimension."""
        if dim == Dimension.POINT and Dimension.THRUST_MODE in self.dims:
            raise ValueError('Dimensions cannot include both point and thrust mode')
        if dim == Dimension.THRUST_MODE and Dimension.POINT in self.dims:
            raise ValueError('Dimensions cannot include both point and thrust mode')
        return Dimensions(dim, *self.dims)

    def remove(self, dim: Dimension) -> Dimensions:
        """Remove a dimension."""
        return Dimensions(*(d for d in self.dims if d != dim))

    @property
    def ordered(self) -> list[Dimension]:
        """Return dimensions in standard order."""
        return sorted(self.dims)

    @property
    def netcdf(self) -> tuple[str, ...]:
        """Return dimension for use in NetCDF files."""

        # Dimension names used in NetCDF files are the lower cased version of
        # the Dimension enum values, and the sort order of the enum values
        # matches the order the dimensions should be used in the NetCDF files.
        return tuple(d.dim_name for d in sorted(self.dims - {Dimension.POINT}))

    @property
    def abbrev(self) -> str:
        """Return an abbreviated string representation of the dimensions.

        (Mostly for convenience when testing.)"""
        return ''.join(DIMENSION_ABBREVS[d] for d in self.ordered)

    def __str__(self) -> str:
        return f'Dimensions({self.abbrev})'

    def __repr__(self) -> str:
        return f'Dimensions({self.abbrev})'

    @classmethod
    def from_abbrev(cls, abbrev: str) -> Dimensions:
        """Create a Dimensions object from an abbreviated string representation.

        (Mostly for convenience when testing.)"""
        dims = set()
        for ch in abbrev:
            if ch not in DIMENSION_ABBREVS_REV:
                raise ValueError(f'Invalid dimension abbreviation: {ch}')
            dims.add(DIMENSION_ABBREVS_REV[ch])
        return cls(*dims)

    @classmethod
    def from_dim_names(cls, *names: str) -> Dimensions:
        """Create a Dimensions object from a list of dimension names used in
        NetCDF files."""
        dims = set()
        for name in names:
            dims.add(Dimension.from_dim_name(name))
        return cls(*dims)


DIMENSION_ABBREVS = {
    Dimension.TRAJECTORY: 'T',
    Dimension.SPECIES: 'S',
    Dimension.POINT: 'P',
    Dimension.THRUST_MODE: 'M',
}
"""Single-character abbreviations for Dimension values."""


DIMENSION_ABBREVS_REV = {v: k for k, v in DIMENSION_ABBREVS.items()}
"""Reverse mapping from single-character abbreviations to Dimension values."""
