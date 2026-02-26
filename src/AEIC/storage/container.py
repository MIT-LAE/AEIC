# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from copy import deepcopy
from typing import Any, Self

import numpy as np

from AEIC.performance.types import ThrustModeValues
from AEIC.types import Species, SpeciesValues

from .dimensions import Dimension
from .field_sets import FieldSet, HasFieldSets


class Container:
    """Container class for 1-D trajectory and single point data with various
    data fields and metadata."""

    # The fixed set of attributes used for implementation of the flexible field
    # interface. Names with leading underscores will not conflict with field
    # names from fieldsets, because those are not allowed to have a leading
    # underscore.
    #
    # NOTE: If any fixed attributes are added to classes derived from
    # Container, they should be added here.
    FIXED_FIELDS = {
        # Field definition information.
        '_data_dictionary',
        # Names of field sets in this container.
        '_fieldsets',
        # Data fields.
        '_data',
        # Number of points in pointwise fields.
        '_size',
        # Space allocated for points in pointwise fields.
        '_capacity',
        # Is it possible to extend the container's pointwise fields, or are
        # they fixed in size?
        '_extensible',
        # Field set for a single point in this container.
        '_single_point_fieldset',
        # Additional fixed fields in Trajectory class:
        '_current_phase',
    }

    STARTING_CAPACITY = 50
    """Starting capacity for extensible containers: the initial number of
    points allocated for pointwise fields in a container without a specified
    number of points. The container will extend its capacity as needed when
    points are added."""

    CAPACITY_EXPANSION = 50
    """Number of points by which to expand the container's capacity when adding
    points to an extensible container that has reached its current capacity."""

    def __init__(
        self,
        npoints: int | None = None,
        fieldsets: list[str] | None = None,
        fieldset: FieldSet | None = None,
    ):
        """A container is either a trajectory or a single point of a trajectory.

        A container either has a fixed number of points, known in advance or is
        an extensible container that can grow as needed when points are added."""

        # Can either provide a single complete fieldset for the container, or a
        # nonempty list of names of registered field sets.
        if fieldsets is not None:
            if fieldset is not None:
                raise ValueError(
                    'cannot specify both fieldset name list and single fieldset'
                )
            if len(fieldsets) == 0:
                raise ValueError(
                    'if fieldset name list is specified, it must be non-empty'
                )

        # Extensible containers start empty, fixed containers have a specified
        # size.
        if npoints is None:
            self._extensible = True
            self._capacity = self.STARTING_CAPACITY
            self._size = 0
        else:
            self._extensible = False
            self._size = npoints
            self._capacity = npoints

        # A container has a set of data fields with specified dimensions. All
        # of these are defined by a FieldSet, and the total sets of all fields
        # are stored in a data dictionary. Start with an empty FieldSet unless
        # a specific fieldset is requested.
        self._data_dictionary: FieldSet = FieldSet()

        # Data fields. The types of these are determined by the dimensions and
        # underlying data type of each field.
        self._data: dict[str, Any] = {}

        # Keep track of the FieldSets that contributed to this container. We
        # start with an empty set.
        self._fieldsets = set()

        # This is calculated lazily as needed.
        self._single_point_fieldset = None

        # Add field sets named in the constructor.
        if fieldsets is not None:
            for fs_name in set(fieldsets):
                self.add_fields(FieldSet.from_registry(fs_name))
        if fieldset is not None:
            self.add_fields(fieldset)

    def __len__(self):
        """The length of pointwise fields in the container."""
        return self._size

    def __eq__(self, other: object) -> bool:
        """Two containers are equal if their data dictionaries are equal and
        all their field values are equal."""
        if not isinstance(other, Container):
            return False
        if self._data_dictionary != other._data_dictionary:
            return False
        for name in self._data_dictionary:
            if name in self._data:
                vs = self._data[name]
                vo = other._data[name]
                if isinstance(
                    vs,
                    str
                    | None
                    | int
                    | float
                    | np.floating
                    | SpeciesValues
                    | ThrustModeValues,
                ):
                    if vs != vo:
                        return False
                elif isinstance(vs, np.ndarray):
                    if self._size != other._size:
                        return False
                    if not np.array_equal(vs[: self._size], vo[: self._size]):
                        return False
                else:
                    raise ValueError('unknown type in container comparison')
        return True

    def approx_eq(self, other: object) -> bool:
        """Two containers are approximately equal if their data dictionaries
        are equal and all their field values are approximately equal."""
        if not isinstance(other, Container):
            return False
        if self._data_dictionary != other._data_dictionary:
            return False
        for name in self._data_dictionary:
            if name in self._data:
                vs = self._data[name]
                vo = other._data[name]
                if isinstance(vs, str | None):
                    if vs != vo:
                        return False
                elif isinstance(vs, int | float | np.floating):
                    if not np.isclose(vs, vo):
                        return False
                elif isinstance(vs, SpeciesValues | ThrustModeValues):
                    if not vs.isclose(vo):
                        return False
                elif isinstance(vs, np.ndarray):
                    if self._size != other._size:
                        return False
                    if not np.allclose(vs[: self._size], vo[: self._size]):
                        return False
                else:
                    raise ValueError('unknown type in container comparison')
        return True

    def __hash__(self):
        """The hash of a container is based on its data dictionary."""
        return hash(self._data_dictionary)

    def __getattr__(self, name: str):
        """Override attribute retrieval to access pointwise and per-container
        data fields."""

        # Fixed attributes use for implementation: delegate to normal attribute
        # access.
        if name in self.FIXED_FIELDS:
            return super().__getattribute__(name)

        if name in self._data:
            val = self._data[name]
            if isinstance(val, np.ndarray):
                return val[: self._size]
            else:
                return val
        else:
            raise AttributeError(f"Container has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Override attribute setting to handle pointwise and per-container
        data fields.

        The type and length of assigned values are checked to ensure
        consistency with the container's data dictionary. The type checking
        rules used here are the same as those used by NumPy's `np.can_cast`
        with `casting='same_kind'`.

        NOTE: These type checking rules mean that assigning a value of type `int` to a
        field of type `np.int32` will work, but may result in loss of information if the
        integer value is too large to fit in an `np.int32`. Caveat emptor!"""

        # Fixed attributes use for implementation: delegate to normal attribute
        # access.
        if name in self.FIXED_FIELDS:
            return super().__setattr__(name, value)

        if name in self._data:
            # Check that the type of the assigned value can be safely cast to
            # the field type and cast and assign the value if OK.
            self._data[name] = self._data_dictionary[name].convert_in(
                value, name, self._size
            )
        else:
            raise ValueError(f"Container has no attribute '{name}'")

    def add_fields(self, fieldset: FieldSet | HasFieldSets):
        """Add fields from a FieldSet to the container.

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

        # Adjust the container to include the new fields.
        for fs in fss:
            assert isinstance(fs, FieldSet)
            # if fs.anonymous:
            #     raise ValueError('cannot add anonymous FieldSet to Container')
            # assert fs.fieldset_name is not None
            if fs.fieldset_name is not None:
                self._fieldsets.add(fs.fieldset_name)
            self._data_dictionary = self._data_dictionary.merge(fs)

        # Invalidate this cached value because the data dictionary has changed.
        self._single_point_fieldset = None

        # Add pointwise and per-trajectory data fields and set values from the
        # `HasFieldSet` object if there is one.
        for fs in fss:
            assert isinstance(fs, FieldSet)
            for name, metadata in fs.items():
                if try_data and hasattr(fieldset, name):
                    value = getattr(fieldset, name)

                    # Check that the type of the assigned value can be
                    # safely cast to the field type and cast and assign
                    # the value if OK.
                    self._data[name] = metadata.convert_in(value, name, self._size)
                else:
                    self._data[name] = metadata.empty(self._capacity)

    @property
    def species(self) -> list[Species]:
        """Set of species included in any species-indexed fields in the
        container."""
        species = set()
        for name, field in self._data_dictionary.items():
            if Dimension.SPECIES in field.dimensions:
                assert isinstance(self._data[name], SpeciesValues)
                species.update(self._data[name].keys())
        return sorted(species)

    def _check_fieldset(self, fieldset: FieldSet):
        # Fields may not overlap with fixed implementation fields.
        if any(name in self.FIXED_FIELDS for name in fieldset):
            raise ValueError('Field name conflicts with Container fixed attribute')

        # Field sets can only be added once.
        if fieldset.fieldset_name in self._fieldsets:
            raise ValueError(
                f'FieldSet with name "{fieldset.fieldset_name}" '
                'already added to Container'
            )

    def copy(self) -> Self:
        """Create a deep copy of the container. A copy of an extensible
        container is *not* extensible."""
        new_traj = (self.__class__)(self._size, fieldsets=list(self._fieldsets))
        for name in self._data_dictionary:
            if name in self._data:
                new_traj._data[name] = deepcopy(self._data[name])
        return new_traj

    def fix(self) -> None:
        """Convert an extensible container to a fixed-size container."""
        self._extensible = False

    def make_point(self, idx: int | None = None) -> Container:
        """Make a new container representing a single point of the possible
        multiple points stored in this container. For example, for a
        trajectory, this produces a container representing a single point of
        the trajectory. If `idx` is specified, the field values for that point
        are copied to the new container. If `idx` is not specified, the field
        values in the new container are left uninitialized."""

        # This value is cached. The cache is invalidated if the data dictionary
        # of this container changes.
        if self._single_point_fieldset is None:
            self._single_point_fieldset = self._data_dictionary.single_point()
        pt = Container(npoints=1, fieldset=self._single_point_fieldset)
        if idx is not None:
            if idx < -self._size or idx >= self._size:
                raise IndexError('point index out of range')
            for name in pt._data_dictionary:
                if name in self._data:
                    val = self._data[name]
                    if isinstance(val, np.ndarray):
                        pt._data[name] = val[idx]
        return pt

    def _expand_capacity(self) -> None:
        self._capacity += self.CAPACITY_EXPANSION
        for name in self._data_dictionary:
            if name in self._data:
                if isinstance(self._data[name], np.ndarray):
                    self._data[name] = np.resize(self._data[name], (self._capacity,))

    def append(self, point: Container | None = None, **kwargs) -> None:
        if not self._extensible:
            raise ValueError('cannot append to fixed-size Container')
        if point is not None and len(kwargs) != 0:
            raise ValueError('cannot specify both point and keyword arguments')
        if point is not None:
            self._append_point(point)
        else:
            self._append_from_dict(kwargs)

    def _append_point(self, point: Container) -> None:
        # TODO: Cache single point fieldset.
        if point._data_dictionary != self._data_dictionary.single_point():
            raise ValueError('cannot append point with different data dictionary')
        if len(point) != 1:
            raise ValueError('can only append a single point')
        self._append_from_dict(point._data)

    def _append_from_dict(self, data: dict[str, Any]) -> None:
        # Fields should match those in single point field set.
        field_set = self._data_dictionary.single_point()
        fields = set(field_set.keys())
        data_fields = set(data.keys())
        if fields != data_fields:
            missing = fields - data_fields
            extra = data_fields - fields
            if missing:
                raise ValueError(f'missing fields in appended point: {missing}')
            if extra:
                raise ValueError(f'extra fields in appended point: {extra}')

        # Increase container capacity if needed.
        if self._size == self._capacity:
            self._expand_capacity()

        for name, value in data.items():
            # Check that the type of the assigned value can be safely cast to
            # the field type and cast and assign the value if OK.
            self._data[name][self._size] = field_set[name].convert_in(value, name, 0)

        self._size += 1
