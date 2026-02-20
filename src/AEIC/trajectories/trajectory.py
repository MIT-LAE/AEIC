# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from copy import deepcopy

import numpy as np

from AEIC.storage import (
    PHASE_FIELDS,
    Container,
    Dimension,
    Dimensions,
    FieldMetadata,
    FieldSet,
)
from AEIC.types import SpeciesValues
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
            self.X_data['name'] = name

    @property
    def nbytes(self) -> int:
        """Calculate approximate memory size of the trajectory in bytes.

        (This only needs to be approximate because it's just used for sizing
        the `TrajectoryStore` LRU cache.)"""
        size = 0
        for f in self.X_data_dictionary.values():
            size += f.nbytes(self.X_size)
        return size

    def copy_point(self, from_idx: int, to_idx: int):
        """Copy data from one point to another within the trajectory."""
        if from_idx < 0 or from_idx >= self.X_size:
            raise IndexError('from_idx out of range')
        if to_idx < 0 or to_idx >= self.X_size:
            raise IndexError('to_idx out of range')
        print(self.X_data_dictionary)
        for name, field in self.X_data_dictionary.items():
            # Only copy point-wise data.
            if Dimension.POINT in field.dimensions:
                # TODO: Handle species dimension as well? Or will this never be
                # called other than when simulating trajectories, where there
                # are no species-indexed fields?
                self.X_data[name][to_idx] = self.X_data[name][from_idx]

    def interpolate_time(self, new_time: np.ndarray) -> Trajectory:
        """Interpolate trajectory data to new time points.

        This method assumes that the trajectory has a `flight_time` field
        containing the time points corresponding to the existing data. The new
        time points must be within the range of the existing time points. The
        method returns a new trajectory with all pointwise fields interpolated
        to the new time points."""

        if 'flight_time' not in self.X_data:
            raise ValueError(
                'Trajectory must have a flight_time field for interpolation'
            )
        orig_time = self.X_data['flight_time']

        new_traj = Trajectory(len(new_time), fieldsets=list(self.X_fieldsets))
        for name, field in self.X_data_dictionary.items():
            if Dimension.POINT in field.dimensions and name in self.X_data:
                if Dimension.SPECIES in field.dimensions:
                    # For species-indexed fields, we need to interpolate each
                    # species separately.
                    assert isinstance(self.X_data[name], SpeciesValues)
                    new_species_values = SpeciesValues[np.ndarray]()
                    for sp in self.X_data[name].keys():
                        new_species_values[sp] = np.interp(
                            new_time,
                            orig_time,
                            self.X_data[name][sp],
                            left=np.nan,
                            right=np.nan,
                        )
                    new_traj.X_data[name] = new_species_values
                else:
                    # Interpolate pointwise data fields.
                    new_traj.X_data[name] = np.interp(
                        new_time,
                        orig_time,
                        self.X_data[name],
                        left=np.nan,
                        right=np.nan,
                    )
            elif name in self.X_data:
                # Copy per-trajectory fields.
                new_traj.X_data[name] = deepcopy(self.X_data[name])
        return new_traj

    def compare(
        self, other: Trajectory, fields: list[str] | None = None
    ) -> ComparisonMetricsCollection:
        """Compare this trajectory to another trajectory and compute comparison
        metrics.

        Comparison metrics are computed for each pointwise field in the data
        dictionary and returned in a dictionary mapping field names to metric
        values."""

        metrics = {}

        for name, field in self.X_data_dictionary.items():
            # Only compute metrics for pointwise fields, and only if the field
            # is present in both trajectories.
            if (
                Dimension.POINT not in field.dimensions
                or name not in other.X_data_dictionary
                or name not in self.X_data
                or name not in other.X_data
                or (fields is not None and name not in fields)
            ):
                continue

            vs = self.X_data[name]
            vo = other.X_data[name]
            if Dimension.SPECIES not in field.dimensions:
                # Simple pointwise field: compute metrics directly.
                metrics[name] = ComparisonMetrics.compute(vs, vo)
            else:
                # Per-species pointwise field: compute metrics for each species
                # separately.
                metrics[name] = SpeciesValues[ComparisonMetrics](
                    {
                        sp: ComparisonMetrics.compute(vs[sp], vo[sp])
                        for sp in vs.keys()
                        if sp in vo.keys()
                    }
                )

        return metrics
