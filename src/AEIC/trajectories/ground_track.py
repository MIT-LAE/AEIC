from bisect import bisect_left
from dataclasses import dataclass
from typing import Self

import numpy as np

from utils import GEOD
from utils.custom_types import Location


class GroundTrack:
    """Great circle interpolator along a set of waypoints.

    A `GroundTrack` defines a path in longitude/latitude by an ordered set of
    waypoints. Between waypoints the ground track follows great circle paths.

    This abstraction supports both great circle paths (one start point, one end
    point, no intermediate waypoints) and paths with multiple waypoints (for
    example, paths defined by ADS-B data).

    The fundamental operation on a `GroundTrack` is to find the location of a
    point based on its distance from the ground track's start location.
    """

    @dataclass
    class Point:
        """A point along the ground track, defined by a location and azimuth."""

        location: Location
        azimuth: float

    class Exception(Exception):
        """Exception raised for errors in the `GroundTrack` class."""

        ...

    def __init__(self, waypoints: list[Location]) -> None:
        """Initialize ground track from list of waypoints."""

        self.waypoints = waypoints

        # Calculate great circle distances and azimuths between waypoints and
        # save cumulative distance values as waypoint index.
        lons = [wp.longitude for wp in waypoints]
        lats = [wp.latitude for wp in waypoints]
        self.azimuths, _, distances = GEOD.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        self.index = np.cumsum([0] + distances).tolist()

    @classmethod
    def great_circle(cls, start_loc: Location, end_loc: Location) -> Self:
        """Create a great circle ground track from a start and end location."""
        return cls([start_loc, end_loc])

    def __len__(self) -> int:
        """Number of waypoints in ground track."""
        return len(self.waypoints)

    def __contains__(self, distance: float) -> bool:
        """Check if distance is within range of ground track."""
        return distance >= self.index[0] and distance <= self.index[-1]

    def __getitem__(self, idx: int) -> 'GroundTrack.Point':
        """Get waypoint and azimuth at given index."""
        return GroundTrack.Point(self.waypoints[idx], self.azimuths[idx])

    def waypoint_distance(self, idx: int) -> float:
        """Get cumulative distance to waypoint at given index."""
        return self.index[idx]

    @property
    def total_distance(self) -> float:
        """Get total distance of ground track."""
        return self.index[-1]

    def lookup_waypoint(self, distance: float) -> int:
        """Find index of waypoint immediately after or at given distance."""

        # Error cases.
        if distance not in self:
            raise GroundTrack.Exception('distance outside ground track range')

        # Find following waypoint.
        return bisect_left(self.index, distance)

    def location(self, distance: float) -> 'GroundTrack.Point':
        """Calculate location at a given distance from start of ground track."""

        # Find waypoints to interpolate between.
        pos = self.lookup_waypoint(distance)

        # Boundary cases.
        if pos == 0:
            return GroundTrack.Point(self.waypoints[0], self.azimuths[0])
        if pos == len(self.index) - 1:
            return GroundTrack.Point(self.waypoints[-1], self.azimuths[-1])

        # Interpolate along a great circle from the first point in the adjacent
        # pair of waypoints we found containing the required distance.
        wp_before = self.waypoints[pos - 1]
        lon, lat, _ = GEOD.fwd(
            wp_before.longitude,
            wp_before.latitude,
            self.azimuths[pos - 1],
            distance - self.index[pos - 1],
        )

        # Calculate azimuth from interpolated location to following waypoint.
        wp_after = self.waypoints[pos]
        azimuth, _, _ = GEOD.inv(lon, lat, wp_after.longitude, wp_after.latitude)

        return GroundTrack.Point(Location(lon, lat), azimuth)

    def step(self, from_distance: float, distance_step: float) -> 'GroundTrack.Point':
        """Calculate location a given distance step from a starting distance.

        This is a convenience method that calls `location` with the sum of
        `from_distance` and `distance_step`.
        """

        # Throw an exception if the step steps over a waypoint. Trajectory
        # builders using multiple waypoints may need to be aware of when that
        # happens to end flight phases at a particular waypoint (e.g., if we
        # include TOC or BOD waypoints).
        #
        # The second part of the condition here is needed to deal with the
        # degenerate case where `from_distance` is exactly at a waypoint.
        before_pos = self.lookup_waypoint(from_distance)
        after_pos = self.lookup_waypoint(from_distance + distance_step)
        if before_pos != after_pos and from_distance < self.index[before_pos]:
            raise GroundTrack.Exception('step would cross a waypoint')

        return self.location(from_distance + distance_step)
