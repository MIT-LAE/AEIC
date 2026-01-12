import sys
from typing import Any

import numpy as np

from .field_sets import FieldMetadata, FieldSet, HasFieldSets
from .phase import PHASE_FIELDS

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
    # Trajectory metadata fields and their metadata. Each of these fields has a
    # single value per trajectory.
    starting_mass=FieldMetadata(
        metadata=True, description='Aircraft mass at start of trajectory', units='kg'
    ),
    total_fuel_mass=FieldMetadata(
        metadata=True, description='Total fuel mass used during trajectory', units='kg'
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
)
"""Base field set included in every trajectory."""


class Trajectory:
    """Class representing a 1-D trajectory with various data fields and
    metadata.

    The "various fields" include a base set of trajectory fields, one value per
    trajectory point and a base set of metadata fields, one value per
    trajectory, plus optional additional per-point or metadata fields added by
    adding field sets to the trajectory.
    """

    # The fixed set of attributes used for implementation of the flexible field
    # interface. Obscure names are used here to reduce the chance of conflicts
    # with data and metadata field names because we cannot have fields with the
    # same names as these fixed infrastructure attributes.
    #
    # (Unfortunately, because of the flexible field definition approach we're
    # using here, we can't make these into Python double-underscore private
    # fields. Using obscure names is the best we can do.)
    FIXED_FIELDS = {
        'X_data_dictionary',
        'X_fieldsets',
        'X_data',
        'X_metadata',
        'X_npoints',
    }

    def __eq__(self, other: object) -> bool:
        """Two trajectories are equal if their data dictionaries are equal and
        all their field values are equal."""
        if not isinstance(other, Trajectory):
            return NotImplemented
        if self.X_data_dictionary != other.X_data_dictionary:
            return False
        for name in self.X_data_dictionary:
            if name in self.X_data:
                if not np.array_equal(self.X_data[name], other.X_data[name]):
                    return False
            else:
                if self.X_metadata[name] != other.X_metadata[name]:
                    return False
        return True

    def approx_eq(self, other: object) -> bool:
        """Two trajectories are approximately equal if their data dictionaries
        are equal and all their field values are approximately equal."""
        if not isinstance(other, Trajectory):
            return NotImplemented
        if self.X_data_dictionary != other.X_data_dictionary:
            return False
        for name in self.X_data_dictionary:
            if name in self.X_data:
                if not np.allclose(self.X_data[name], other.X_data[name]):
                    return False
            else:
                if isinstance(self.X_metadata[name], str | None):
                    if self.X_metadata[name] != other.X_metadata[name]:
                        return False
                else:
                    if not np.isclose(self.X_metadata[name], other.X_metadata[name]):
                        return False
        return True

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
        self.X_npoints = npoints

        # A trajectory has a set of per-point data fields and per-trajectory
        # metadata fields. Both are defined by a FieldSet, and the total sets
        # of all fields are stored in a data dictionary.
        self.X_data_dictionary: FieldSet = BASE_FIELDS

        # Keep track of the FieldSets that contributed to this trajectory.
        self.X_fieldsets = {BASE_FIELDS.fieldset_name}

        # Data fields.
        self.X_data: dict[str, np.ndarray[tuple[int], Any]] = {
            n: np.zeros((npoints,), dtype=f.field_type)
            for n, f in self.X_data_dictionary.items()
            if not f.metadata
        }

        # Metadata fields.
        self.X_metadata: dict[str, Any] = {
            n: f.default for n, f in self.X_data_dictionary.items() if f.metadata
        }

        # A trajectory has an optional name.
        if name is not None:
            self.X_metadata['name'] = name

        # Add any extra field sets named in the constructor.
        if fieldsets is not None:
            for fs_name in set(fieldsets) - {BASE_FIELDSET_NAME}:
                self.add_fields(FieldSet.from_registry(fs_name))

    def __len__(self):
        """The total number of points in the trajectory."""
        return self.X_npoints

    @property
    def nbytes(self) -> int:
        """Calculate approximate memory size of the trajectory in bytes.

        (This only needs to be approximate because it's just used for sizing
        the `TrajectoryStore` LRU cache.)
        """
        size = 0
        for array in self.X_data.values():
            size += array.nbytes
        for value in self.X_metadata.values():
            if isinstance(value, np.ndarray):
                size += value.nbytes
            else:
                size += sys.getsizeof(value)
        return size

    def __hash__(self):
        """The hash of a trajectory is based on its data dictionary."""
        return hash(self.X_data_dictionary)

    def __getattr__(self, name: str) -> np.ndarray[tuple[int], Any] | Any:
        """Override attribute retrieval to access data and metadata fields."""

        # Fixed attributes use for implementation: delegate to normal attribute
        # access.
        if name in self.FIXED_FIELDS:
            return super().__getattribute__(name)

        if name in self.X_data:
            return self.X_data[name]
        elif name in self.X_metadata:
            return self.X_metadata[name]
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
        if name in self.X_data:
            # The number of points in a trajectory is currently fixed at
            # creation time.
            if len(value) != self.X_npoints:
                raise ValueError('Assigned length does not match number of points')

            # Check that the type of the assigned value can be safely cast to
            # the field type and cast and assign the value if OK.
            self.X_data[name] = _convert_types(
                self.X_data_dictionary[name].field_type,
                value,
                'data',
                name,
                self.X_data_dictionary[name].required,
            )
        elif name in self.X_metadata:
            # Check that the type of the assigned value can be safely cast to
            # the field type and cast and assign the value if OK.
            self.X_metadata[name] = _convert_types(
                self.X_data_dictionary[name].field_type,
                value,
                'metadata',
                name,
                self.X_data_dictionary[name].required,
            )
        else:
            raise ValueError(f"'Trajectory' object has no attribute '{name}'")

    def copy_point(self, from_idx: int, to_idx: int):
        """Copy data from one point to another within the trajectory."""
        if from_idx < 0 or from_idx >= self.X_npoints:
            raise IndexError('from_idx out of range')
        if to_idx < 0 or to_idx >= self.X_npoints:
            raise IndexError('to_idx out of range')
        for name, field in self.X_data_dictionary.items():
            if not field.metadata:
                self.X_data[name][to_idx] = self.X_data[name][from_idx]

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
            self.X_fieldsets.add(fs.fieldset_name)
            self.X_data_dictionary = self.X_data_dictionary.merge(fs)

        # Add data and metadata fields and set values from the `HasFieldSet`
        # object if there is one.
        for fs in fss:
            assert isinstance(fs, FieldSet)
            for name, metadata in fs.items():
                if metadata.metadata:
                    if try_data and hasattr(fieldset, name):
                        # Check that the type of the assigned value can be
                        # safely cast to the field type and cast and assign
                        # the value if OK.
                        self.X_metadata[name] = _convert_types(
                            metadata.field_type,
                            getattr(fieldset, name),
                            'metadata',
                            name,
                        )
                        continue
                    self.X_metadata[name] = None
                else:
                    if try_data and hasattr(fieldset, name):
                        value = getattr(fieldset, name)

                        # The number of points in a trajectory is currently
                        # fixed at creation time.
                        if len(value) != self.X_npoints:
                            raise ValueError(
                                'Assigned length does not match number of points'
                            )

                        # Check that the type of the assigned value can be
                        # safely cast to the field type and cast and assign
                        # the value if OK.
                        self.X_data[name] = _convert_types(
                            metadata.field_type, value, 'data', name
                        )
                        continue
                    self.X_data[name] = np.zeros(
                        (self.X_npoints,), dtype=metadata.field_type
                    )

    def _check_fieldset(self, fieldset: FieldSet):
        # Fields may not overlap with fixed implementatin fields.
        if any(name in self.FIXED_FIELDS for name in fieldset):
            raise ValueError('Field name conflicts with Trajectory fixed attribute')

        # Field sets can only be added once.
        if fieldset.fieldset_name in self.X_fieldsets:
            raise ValueError(
                f'FieldSet with name "{fieldset.fieldset_name}" '
                'already added to Trajectory'
            )


def _convert_types(
    expected_type: type, value: Any, label: str, name: str, required: bool = True
) -> Any:
    """Check that the type of the assigned value can be safely cast to the
    expected type and return the cast value.

    (Metadata fields are single values, so we convert to a 1-element array for
    the type check and casting.)
    """

    # Wrap single values.
    arr = value
    wrapped = False
    if not isinstance(value, np.ndarray):
        arr = np.asarray(value)
        wrapped = True

    # Handle missing values for optional fields.
    if not required:
        if (
            value is None
            or hasattr(arr, 'mask')
            and np.all(getattr(arr, 'mask'))
            or arr.size > 1
            and np.all([v is None for v in arr])
        ):
            return None

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
