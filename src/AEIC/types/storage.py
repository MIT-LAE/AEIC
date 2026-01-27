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


class Dimensions:
    def __init__(self, *dims: Dimension):
        self.dims = set(dims)
        if Dimension.TRAJECTORY not in self.dims:
            raise ValueError('Dimensions must include the trajectory')
        if Dimension.POINT in self.dims and Dimension.THRUST_MODE in self.dims:
            raise ValueError('Dimensions cannot include both point and thrust mode')

    def __len__(self):
        return len(self.dims)

    def __contains__(self, item: Dimension):
        return item in self.dims

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
        return tuple(d.name.lower() for d in sorted(self.dims - {Dimension.POINT}))
