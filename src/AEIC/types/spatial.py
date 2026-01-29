from dataclasses import dataclass


@dataclass
class Location:
    """A geographic location defined by longitude and latitude."""

    longitude: float
    """Longitude in decimal degrees."""

    latitude: float
    """Latitude in decimal degrees."""


@dataclass
class Position:
    """An aircraft position defined by longitude, latitude, and altitude."""

    longitude: float
    """Longitude in decimal degrees."""

    latitude: float
    """Latitude in decimal degrees."""

    altitude: float
    """Altitude in meters above sea level."""

    @property
    def location(self) -> Location:
        """Get the 2D location (longitude and latitude) of this position."""
        return Location(longitude=self.longitude, latitude=self.latitude)
