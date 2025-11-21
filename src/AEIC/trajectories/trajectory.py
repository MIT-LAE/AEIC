import sys
from typing import Any

import numpy as np

from .field_sets import FieldMetadata, FieldSet, HasFieldSets

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
    flight_level_weight=FieldMetadata(
        description='Flight level weight factor', units='dimensionless'
    ),
    # Trajectory metadata fields and their metadata. Each of these fields has a
    # single value per trajectory.
    flight_id=FieldMetadata(
        metadata=True,
        field_type=np.int64,
        description='Mission database flight identifier',
        required=False,
    ),
    name=FieldMetadata(
        metadata=True,
        field_type=str,
        description='Trajectory name',
        required=False,
    ),
    starting_mass=FieldMetadata(
        metadata=True, description='Aircraft mass at start of trajectory', units='kg'
    ),
    total_fuel_mass=FieldMetadata(
        metadata=True, description='Total fuel mass used during trajectory', units='kg'
    ),
    NClm=FieldMetadata(
        metadata=True,
        field_type=np.int32,
        description='Number of climb points in trajectory',
        units='count',
    ),
    NCrz=FieldMetadata(
        metadata=True,
        field_type=np.int32,
        description='Number of cruise points in trajectory',
        units='count',
    ),
    NDes=FieldMetadata(
        metadata=True,
        field_type=np.int32,
        description='Number of descent points in trajectory',
        units='count',
    ),
)


class Trajectory:
    """Class representing a 1D trajectory with various data fields and
    metadata.

    The "various fields" include a base set of trajectory fields, one value per
    trajectory point and a base set of metadata fields, one value per
    trajectory, plus optional additional per-point or metadata fields added by
    adding field sets to the trajectory.
    """

    # The fixed set of attributes used for implementation of the flexible field
    # interface.
    FIXED_FIELDS = set(['data_dictionary', 'fieldsets', 'data', 'metadata', 'npoints'])

    def __init__(
        self, npoints: int, name: str | None = None, fieldsets: list[str] | None = None
    ):
        """Initialized with a fixed number of points and an optional name.

        The name is used for labelling trajectories within trajectory sets (and
        NetCDF files).

        All per-point data and per-trajectory metadata fields included in every
        trajectory by default are taken from the `BASE_FIELDS` dictionary
        above. Other per-point and metadata fields may be added using the
        `add_fields` method.
        """

        # A trajectory has a fixed number of points, known in advance.
        # TODO: Lift this restriction? How could we make it so that you can add
        # points incrementally, in a nice way?
        self.npoints = npoints

        # A trajectory has a set of per-point data fields and per-trajectory
        # metadata fields. Both are defined by a FieldSet, and the total sets
        # of all fields are stored in a data dictionary.
        self.data_dictionary: FieldSet = BASE_FIELDS

        # Keep track of the FieldSets that contributed to this trajectory.
        self.fieldsets = set([BASE_FIELDS.fieldset_name])

        # Data fields.
        self.data: dict[str, np.ndarray[tuple[int], Any]] = {
            n: np.zeros((npoints,), dtype=f.field_type)
            for n, f in self.data_dictionary.items()
            if not f.metadata
        }

        # Metadata fields.
        self.metadata: dict[str, Any] = {
            n: None for n, f in self.data_dictionary.items() if f.metadata
        }

        # A trajectory has an optional name.
        if name is not None:
            self.metadata['name'] = name

        # Add any extra field sets named in the constructor.
        if fieldsets is not None:
            for fs_name in set(fieldsets) - {BASE_FIELDSET_NAME}:
                if fs_name not in FieldSet.REGISTRY:
                    raise ValueError(f'Unknown FieldSet name: {fs_name}')
                self.add_fields(FieldSet.REGISTRY[fs_name])

    def __len__(self):
        """The total number of points in the trajectory."""
        return self.npoints

    @property
    def nbytes(self) -> int:
        """Calculate approximate memory size of the trajectory in bytes.

        (This only needs to be approximate because it's just used for sizing
        the `TrajectoryStore` LRU cache.)
        """
        size = 0
        for array in self.data.values():
            size += array.nbytes
        for value in self.metadata.values():
            if isinstance(value, np.ndarray):
                size += value.nbytes
            else:
                size += sys.getsizeof(value)
        return size

    def __getattr__(self, name: str) -> np.ndarray[tuple[int], Any] | Any:
        """Override attribute retrieval to access data and metadata fields."""

        # Fixed attributes use for implementation: delegate to normal attribute
        # access.
        if name in self.FIXED_FIELDS:
            return super().__getattribute__(name)

        if name in self.data:
            return self.data[name]
        elif name in self.metadata:
            return self.metadata[name]
        else:
            raise AttributeError(f"'Trajectory' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Override attribute setting to handle data and metadata fields.

        The type and length of assigned values are checked to ensure
        consistency with the trajectory's data dictionary. The type checking
        rules used here are the same as those used by NumPy's `np.can_cast`
        with `casting='same_kind'`.

        NOTE: These type checking rules mean that assigning a value of type `int` to a
        field of type `np.int32` will work, but may result in loss of information if the
        integer value is too large to fit in an `np.int32`. Caveat emptor!
        """

        # Fixed attributes use for implementation: delegate to normal attribute
        # access.
        if name in self.FIXED_FIELDS:
            return super().__setattr__(name, value)

        # Assignment for data or metadata fields.
        if name in self.data:
            # The number of points in a trajectory is currently fixed at
            # creation time.
            if len(value) != self.npoints:
                raise ValueError('Assigned length does not match number of points')

            # Check that the type of the assigned value can be safely cast to
            # the field type and cast and assign the value if OK.
            self.data[name] = _convert_types(
                self.data_dictionary[name].field_type, value, 'data', name
            )
        elif name in self.metadata:
            # Check that the type of the assigned value can be safely cast to
            # the field type and cast and assign the value if OK.
            self.metadata[name] = _convert_types(
                self.data_dictionary[name].field_type, value, 'metadata', name
            )
        else:
            raise ValueError(f"'Trajectory' object has no attribute '{name}'")

    def copy_point(self, from_idx: int, to_idx: int):
        """Copy data from one point to another within the trajectory."""
        if from_idx < 0 or from_idx >= self.npoints:
            raise IndexError('from_idx out of range')
        if to_idx < 0 or to_idx >= self.npoints:
            raise IndexError('to_idx out of range')
        for name, field in self.data_dictionary.items():
            if not field.metadata:
                self.data[name][to_idx] = self.data[name][from_idx]

    def _check_fieldset(self, fieldset: FieldSet):
        # Fields may not overlap with fixed implementatin fields.
        if any(name in self.FIXED_FIELDS for name in fieldset):
            raise ValueError('Field name conflicts with Trajectory fixed attribute')

        # Field sets can only be added once.
        if fieldset.fieldset_name in self.fieldsets:
            raise ValueError(
                f'FieldSet with name "{fieldset.fieldset_name}" '
                'already added to Trajectory'
            )

    def add_fields(self, fieldset: FieldSet | HasFieldSets):
        """Add fields from a FieldSet to the trajectory.

        Either just add fields with empty values, or, if the field set is
        attached to a value object using the `HasFieldSet` protocol, try to
        initialize data values too.
        """

        # Set up field set: either passed directly, or attached to another
        # object as a `FIELD_SETS` class attribute. (There are asserts all over
        # the place here because PyRight gets confused about the types. Not
        # sure why. It seems like a fairly clear case.)
        fss = [fieldset]
        try_data = False
        if not isinstance(fieldset, FieldSet):
            assert isinstance(fieldset, HasFieldSets)
            fss = fieldset.FIELD_SETS
            assert isinstance(fss, list)
            try_data = True
        assert all(isinstance(fs, FieldSet) for fs in fss)
        for fs in fss:
            assert isinstance(fs, FieldSet)
            self._check_fieldset(fs)

        # Adjust the trajectory to include the new fields.
        for fs in fss:
            assert isinstance(fs, FieldSet)
            self.fieldsets.add(fs.fieldset_name)
            self.data_dictionary = self.data_dictionary.merge(fs)

        # Add data and metadata fields and set values from the `HasFieldSet`
        # object if there is one.
        for fs in fss:
            assert isinstance(fs, FieldSet)
            for name, metadata in fs.items():
                if metadata.metadata:
                    if try_data:
                        if hasattr(fieldset, name):
                            # Check that the type of the assigned value can be
                            # safely cast to the field type and cast and assign
                            # the value if OK.
                            self.metadata[name] = _convert_types(
                                metadata.field_type,
                                getattr(fieldset, name),
                                'metadata',
                                name,
                            )
                            continue
                    self.metadata[name] = None
                else:
                    if try_data:
                        if hasattr(fieldset, name):
                            value = getattr(fieldset, name)

                            # The number of points in a trajectory is currently
                            # fixed at creation time.
                            if len(value) != self.npoints:
                                raise ValueError(
                                    'Assigned length does not match number of points'
                                )

                            # Check that the type of the assigned value can be
                            # safely cast to the field type and cast and assign
                            # the value if OK.
                            self.data[name] = _convert_types(
                                metadata.field_type, value, 'data', name
                            )
                            continue
                    self.data[name] = np.zeros(
                        (self.npoints,), dtype=metadata.field_type
                    )


def _convert_types(expected_type: type, value: Any, label: str, name: str) -> Any:
    """Check that the type of the assigned value can be safely cast to the
    expected type and return the cast value.

    (Metadata fields are single values, so we convert to a 1-element array for
    the type check and casting.)
    """

    arr = value
    wrapped = False
    if not isinstance(value, np.ndarray):
        arr = np.asarray(value)
        wrapped = True

    # Check that the type of the assigned value can be safely cast to
    # the field type.
    if not np.can_cast(arr, expected_type, casting='same_kind'):
        raise TypeError(
            f'Cannot cast assigned value of type {type(value)} '
            f'to expected type {expected_type} for {label} field {name}'
        )

    # Return cast value.
    cast_value = np.asarray(arr).astype(expected_type, casting='same_kind')
    if wrapped:
        cast_value = cast_value.item()
    return cast_value
