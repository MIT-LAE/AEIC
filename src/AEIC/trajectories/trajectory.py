# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from AEIC.storage import (
    PHASE_FIELDS,
    Container,
    Dimension,
    Dimensions,
    FieldMetadata,
    FieldSet,
    FlightPhase,
)
from AEIC.types import Species, SpeciesValues
from AEIC.verification.metrics import ComparisonMetrics, ComparisonMetricsCollection

BASE_FIELDSET_NAME = 'base'

BASE_FIELDS = FieldSet(
    BASE_FIELDSET_NAME,
    # Trajectory data fields and their metadata. Each of these fields is
    # defined for each point along the trajectory.
    fuel_flow=FieldMetadata(description='Fuel flow rate', units='kg/s'),
    aircraft_mass=FieldMetadata(description='Aircraft mass', units='kg'),
    fuel_mass=FieldMetadata(description='Fuel mass remaining', units='kg'),
    ground_distance=FieldMetadata(description='Ground distance traveled', units='m'),
    altitude=FieldMetadata(description='Altitude above sea level', units='m'),
    flight_level=FieldMetadata(description='Flight level', units='FL'),
    rate_of_climb=FieldMetadata(description='Rate of climb/descent', units='m/s'),
    flight_time=FieldMetadata(description='Flight time elapsed', units='s'),
    latitude=FieldMetadata(description='Latitude', units='degrees'),
    longitude=FieldMetadata(description='Longitude', units='degrees'),
    azimuth=FieldMetadata(description='Azimuth angle', units='degrees'),
    heading=FieldMetadata(description='Aircraft heading', units='degrees'),
    true_airspeed=FieldMetadata(description='True airspeed', units='m/s'),
    ground_speed=FieldMetadata(description='Ground speed', units='m/s'),
    # Per-trajectory fields and their metadata.
    starting_mass=FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY),
        description='Aircraft mass at start of trajectory',
        units='kg',
    ),
    total_fuel_mass=FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY),
        description='Total fuel mass used during trajectory',
        units='kg',
    ),
    # Trajectory point counts for the different flight phases. (The "type:
    # ignore" is needed because PyRight cannot prove to itself that the
    # computed field names here don't include "registered", which is a possible
    # argument to the `FieldSet` constructor. That makes PyRight very unhappy
    # so we need to appease it a little, just to let it know that we're paying
    # attention.)
    **PHASE_FIELDS,  # type: ignore[call-overload]
    # Special optional metadata fields: `flight_id` is used to make the
    # connection to entries in missions databases, and `name` is an optional
    # textual name for the trajectory.
    flight_id=FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY),
        field_type=np.int64,
        description='Mission database flight identifier',
        required=False,
    ),
    name=FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY),
        field_type=str,
        description='Trajectory name',
        required=False,
    ),
)
"""Base field set included in every trajectory."""


@dataclass
class SubTrajectory:
    """Class representing a portion of a trajectory that does not cross the
    dateline. This is used to handle trajectories that cross the dateline,
    which can cause issues with longitude values jumping from +180 to -180
    degrees. The class contains only those trajectory fields that are needed
    for mapping trajectories to grid cells, which is the main use case for
    sub-trajectories."""

    longitude: np.ndarray
    """Longitude values for the trajectory points, in degrees."""

    latitude: np.ndarray
    """Latitude values for the trajectory points, in degrees."""

    altitude: np.ndarray
    """Altitude values for the trajectory points, in meters."""

    trajectory_emissions: SpeciesValues[np.ndarray]
    """Emissions values for the trajectory points, indexed by chemical
    species."""


# NOTE: If you add a fixed attribute to the Trajectory class, make sure to
# update the FIXED_FIELDS list in the Container base class.


class Trajectory(Container):
    """Class representing a 1-D trajectory with various data fields and
    metadata.

    The "various fields" include a base set of pointwise fields, one value per
    trajectory point and a base set of trajectory fields, one value per
    trajectory, plus optional additional pointwise or per-trajectory fields
    added by adding field sets to the trajectory. (These additional fields may
    be indexed by chemical species and/or LTO thrust mode as well as trajectory
    and point index. This is needed to record emissions information.)"""

    def __init__(
        self,
        npoints: int | None = None,
        name: str | None = None,
        fieldsets: list[str] | None = None,
    ):
        """Initialized either empty for construction be appending points or
        with a fixed number of points, and an optional name.

        The name is used for labelling trajectories within trajectory sets (and
        NetCDF files).

        All pointwise data and per-trajectory fields included in every
        trajectory by default are taken from the `BASE_FIELDS` dictionary
        above. Other fields may be added using the `add_fields` method."""

        # Initialize container with base field set and any additional field
        # sets specified by the caller.
        if fieldsets is None:
            fieldsets = []
        super().__init__(npoints=npoints, fieldsets=[BASE_FIELDSET_NAME] + fieldsets)

        # A trajectory has an optional name.
        if name is not None:
            self._data['name'] = name

        self._current_phase = min(FlightPhase)

    def set_phase(self, phase: FlightPhase):
        """Set the current flight phase for this trajectory.

        This is used by trajectory builders to keep track of which phase of
        flight they're currently building, so that they can update the correct
        phase point count field in the trajectory metadata."""
        if phase < self._current_phase:
            raise ValueError('cannot set flight phase to an earlier phase')
        self._current_phase = phase
        self._data[self._current_phase.field_name] = 0

    def append(self, *args, **kwargs) -> None:
        """Override the append method to also update the current flight phase
        point count."""
        super().append(*args, **kwargs)
        self._data[self._current_phase.field_name] += 1

    @property
    def nbytes(self) -> int:
        """Calculate approximate memory size of the trajectory in bytes.

        (This only needs to be approximate because it's just used for sizing
        the `TrajectoryStore` LRU cache.)"""
        size = 0
        for f in self._data_dictionary.values():
            size += f.nbytes(self._size)
        return size

    def copy_point(self, from_idx: int, to_idx: int):
        """Copy data from one point to another within the trajectory."""
        if from_idx < 0 or from_idx >= self._size:
            raise IndexError('from_idx out of range')
        if to_idx < 0 or to_idx >= self._size:
            raise IndexError('to_idx out of range')
        for name, field in self._data_dictionary.items():
            # Only copy point-wise data.
            if Dimension.POINT in field.dimensions:
                # TODO: Handle species dimension as well? Or will this never be
                # called other than when simulating trajectories, where there
                # are no species-indexed fields?
                self._data[name][to_idx] = self._data[name][from_idx]

    def dateline_split(self) -> list[SubTrajectory]:
        # Check that we have all the required fields for dateline splitting.
        assert hasattr(self, 'longitude')
        assert hasattr(self, 'latitude')
        assert hasattr(self, 'altitude')
        assert hasattr(self, 'trajectory_emissions')
        assert isinstance(self.trajectory_emissions, SpeciesValues)

        # Determine locations of dateline crossings by looking for large jumps
        # in longitude values. The crosses_dateline array has one less element
        # than the longitude array and will have values of -1 for points where
        # the longitude jumps from just below +180 to just above -180, and +1
        # for points where it jumps from just above -180 to just below +180
        # (with those values being on the position of the longitude value just
        # before the jump).
        diff = self.longitude[1:] - self.longitude[:-1]
        cross = (np.abs(diff) > 180.0).astype(int)
        crosses_dateline = np.sign(diff) * cross

        # Deal with no-crossing case right away.
        if np.all(crosses_dateline == 0):
            return [
                SubTrajectory(
                    longitude=self.longitude,
                    latitude=self.latitude,
                    altitude=self.altitude,
                    trajectory_emissions=self.trajectory_emissions,
                )
            ]

        sub_trajectories = []
        last_split: tuple[float, float, float, SpeciesValues[float]] | None = None
        seg_start_idx = 0
        split_idxs = np.nonzero(crosses_dateline)[0].tolist() + [-1]
        for idx in split_idxs:
            seg_lons: list[float] = []
            seg_lats: list[float] = []
            seg_alts: list[float] = []
            seg_emissions: dict[Species, list[float]] = defaultdict(list)

            if last_split is not None:
                seg_lons.append(last_split[0])
                seg_lats.append(last_split[1])
                seg_alts.append(last_split[2])
                for sp, val in last_split[3].items():
                    seg_emissions[sp].append(val)
                last_split = None

            if idx > -1:
                # TODO: Do all the following using great circles instead?

                # Interpolate to find the exact point of crossing.
                cross_lat, cross_alt, frac = self._interpolate_to_dateline(idx)

                # Split the emissions.
                emissionsm, emissionsp = _split_emissions(
                    self.trajectory_emissions, idx, frac
                )

                last_split = (
                    float(180.0 * crosses_dateline[idx]),
                    cross_lat,
                    cross_alt,
                    emissionsp,
                )

                degenerate = np.abs(self.longitude[idx]) == 180.0

                seg_lons += self.longitude[seg_start_idx : idx + 1].tolist()
                if not degenerate:
                    seg_lons.append(-float(180.0 * crosses_dateline[idx]))
                seg_lats += self.latitude[seg_start_idx : idx + 1].tolist()
                if not degenerate:
                    seg_lats.append(cross_lat)
                seg_alts += self.altitude[seg_start_idx : idx + 1].tolist()
                if not degenerate:
                    seg_alts.append(cross_alt)
                for sp in self.trajectory_emissions.keys():
                    seg_emissions[sp] += self.trajectory_emissions[sp][
                        seg_start_idx : idx + 1
                    ].tolist()
                    if not degenerate:
                        seg_emissions[sp][-1] = emissionsm[sp]
                        seg_emissions[sp].append(0.0)
                    else:
                        seg_emissions[sp][-1] = 0.0
            else:
                seg_lons += self.longitude[seg_start_idx:].tolist()
                seg_lats += self.latitude[seg_start_idx:].tolist()
                seg_alts += self.altitude[seg_start_idx:].tolist()
                for sp in self.trajectory_emissions.keys():
                    seg_emissions[sp] += self.trajectory_emissions[sp][
                        seg_start_idx:
                    ].tolist()

            sub_trajectories.append(
                SubTrajectory(
                    longitude=np.array(seg_lons),
                    latitude=np.array(seg_lats),
                    altitude=np.array(seg_alts),
                    trajectory_emissions=SpeciesValues[np.ndarray](
                        {sp: np.array(vals) for sp, vals in seg_emissions.items()}
                    ),
                )
            )

            seg_start_idx = idx + 1

        return sub_trajectories

    def interpolate_time(self, new_time: np.ndarray) -> Trajectory:
        """Interpolate trajectory data to new time points.

        This method assumes that the trajectory has a `flight_time` field
        containing the time points corresponding to the existing data. The new
        time points must be within the range of the existing time points. The
        method returns a new trajectory with all pointwise fields interpolated
        to the new time points."""

        if 'flight_time' not in self._data:
            raise ValueError(
                'Trajectory must have a flight_time field for interpolation'
            )
        orig_time = self._data['flight_time'][: self._size]

        new_traj = Trajectory(len(new_time), fieldsets=list(self._fieldsets))
        for name, field in self._data_dictionary.items():
            if Dimension.POINT in field.dimensions and name in self._data:
                if Dimension.SPECIES in field.dimensions:
                    # For species-indexed fields, we need to interpolate each
                    # species separately.
                    assert isinstance(self._data[name], SpeciesValues)
                    new_species_values = SpeciesValues[np.ndarray]()
                    for sp in self._data[name].keys():
                        new_species_values[sp] = np.interp(
                            new_time,
                            orig_time,
                            self._data[name][sp][: self._size],
                            left=np.nan,
                            right=np.nan,
                        )
                    new_traj._data[name] = new_species_values
                else:
                    # Interpolate pointwise data fields.
                    new_traj._data[name] = np.interp(
                        new_time,
                        orig_time,
                        self._data[name][: self._size],
                        left=np.nan,
                        right=np.nan,
                    )
            elif name in self._data:
                # Copy per-trajectory fields.
                new_traj._data[name] = deepcopy(self._data[name])
        return new_traj

    def compare(
        self,
        other: Trajectory,
        fields: list[str] | None = None,
        skip_final: set[str] | None = None,
    ) -> ComparisonMetricsCollection:
        """Compare this trajectory to another trajectory and compute comparison
        metrics.

        Comparison metrics are computed for each pointwise field in the data
        dictionary and returned in a dictionary mapping field names to metric
        values."""

        metrics = {}

        for name, field in self._data_dictionary.items():
            # Only compute metrics for pointwise fields, and only if the field
            # is present in both trajectories.
            if (
                Dimension.POINT not in field.dimensions
                or name not in other._data_dictionary
                or name not in self._data
                or name not in other._data
                or (fields is not None and name not in fields)
            ):
                continue

            vs = self._data[name]
            vo = other._data[name]
            n = min(len(vs), len(vo))
            if name in skip_final:
                n -= 1
            if Dimension.SPECIES not in field.dimensions:
                # Simple pointwise field: compute metrics directly.
                metrics[name] = ComparisonMetrics.compute(vs[:n], vo[:n])
            else:
                # Per-species pointwise field: compute metrics for each species
                # separately.
                metrics[name] = SpeciesValues[ComparisonMetrics](
                    {
                        sp: ComparisonMetrics.compute(vs[sp][:n], vo[sp][:n])
                        for sp in vs.keys()
                        if sp in vo.keys()
                    }
                )

        return metrics

    def _interpolate_to_dateline(self, idx: int) -> tuple[float, float, float]:
        """Interpolate to find the exact point of dateline crossing."""

        # Normalize longitudes to [0, 360) range to make interpolation easier.
        lons = [
            math.fmod(self.longitude[idx] + 360.0, 360.0),
            math.fmod(self.longitude[idx + 1] + 360.0, 360.0),
        ]

        # Extract latitudes and altitudes for the two points around the
        # crossing.
        lats = [self.latitude[idx], self.latitude[idx + 1]]
        alts = [self.altitude[idx], self.altitude[idx + 1]]

        # The np.interp function requires the x values to be in increasing
        # order, so we need to check the order of the longitudes and reverse
        # the arrays if necessary. We also keep track of which point is the
        # "base" point for interpolation (the first in trajectory order) so
        # that we can compute the fraction along the segment where the crossing
        # occurs.
        base = 0
        if lons[0] > lons[1]:
            lons = lons[::-1]
            lats = lats[::-1]
            alts = alts[::-1]
            base = 1

        # Do the interpolation to find the latitude and altitude at the
        # crossing point.
        cross_latitude = np.interp(180.0, lons, lats)
        cross_altitude = np.interp(180.0, lons, alts)

        # Determine the fraction along the segment where the crossing occurs,
        # so that we can split emissions values into "before split" and "after
        # split".
        fraction = math.hypot(
            abs(180.0 - lons[base]), abs(cross_latitude - self.latitude[idx])
        ) / math.hypot(
            abs(lons[1] - lons[0]), abs(self.latitude[idx] - self.latitude[idx + 1])
        )

        return cross_latitude, cross_altitude, fraction


def _split_emissions(
    emissions: SpeciesValues[np.ndarray],
    idx: int,
    fraction: float,
) -> tuple[SpeciesValues[float], SpeciesValues[float]]:
    """Split trajectory emissions values for a segment into two parts based on
    a given fractional allocation to before and after the split."""

    emissionsm = SpeciesValues[float]()
    emissionsp = SpeciesValues[float]()

    for sp, values in emissions.items():
        emissionsm[sp] = values[idx] * fraction
        emissionsp[sp] = values[idx] * (1.0 - fraction)

    return emissionsm, emissionsp
