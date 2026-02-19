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
    # interface. Obscure names are used here to reduce the chance of conflicts
    # with data and metadata field names because we cannot have fields with the
    # same names as these fixed infrastructure attributes.
    #
    # (Unfortunately, because of the flexible field definition approach we're
    # using here, we can't make these into Python double-underscore private
    # fields. Using obscure names is the best we can do.)
    FIXED_FIELDS = {
        # Field definition information.
        'X_data_dictionary',
        # Names of field sets in this container.
        'X_fieldsets',
        # Data fields.
        'X_data',
        # Number of points in pointwise fields.
        'X_size',
        # Number of space allocated for points in pointwise fields.
        'X_capacity',
    }

    def __init__(self, npoints: int, fieldsets: list[str] | None = None):
        """Initialized with a fixed number of points."""

        # A container has a fixed number of points, known in advance.
        # TODO: Lift this restriction? How could we make it so that you can add
        # points incrementally, in a nice way?
        self.X_size = npoints

        # A container has a set of data fields with specified dimensions. All
        # of these are defined by a FieldSet, and the total sets of all fields
        # are stored in a data dictionary. Start with an empty FieldSet.
        self.X_data_dictionary: FieldSet = FieldSet()

        # Data fields. The types of these are determined by the dimensions and
        # underlying data type of each field.
        self.X_data: dict[str, Any] = {}

        # Keep track of the FieldSets that contributed to this container. We
        # start with an empty set.
        self.X_fieldsets = set()

        # Add field sets named in the constructor.
        if fieldsets is not None:
            for fs_name in set(fieldsets):
                self.add_fields(FieldSet.from_registry(fs_name))

    def __len__(self):
        """The length of pointwise fields in the container."""
        return self.X_size

    def __eq__(self, other: object) -> bool:
        """Two containers are equal if their data dictionaries are equal and
        all their field values are equal."""
        if not isinstance(other, Container):
            return False
        if self.X_data_dictionary != other.X_data_dictionary:
            return False
        for name in self.X_data_dictionary:
            if name in self.X_data:
                vs = self.X_data[name]
                vo = other.X_data[name]
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
                    if not np.array_equal(vs, vo):
                        return False
                else:
                    raise ValueError('unknown type in container comparison')
        return True

    def approx_eq(self, other: object) -> bool:
        """Two containers are approximately equal if their data dictionaries
        are equal and all their field values are approximately equal."""
        if not isinstance(other, Container):
            return False
        if self.X_data_dictionary != other.X_data_dictionary:
            return False
        for name in self.X_data_dictionary:
            if name in self.X_data:
                vs = self.X_data[name]
                vo = other.X_data[name]
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
                    if not np.allclose(vs, vo):
                        return False
                else:
                    raise ValueError('unknown type in container comparison')
        return True

    def __hash__(self):
        """The hash of a container is based on its data dictionary."""
        return hash(self.X_data_dictionary)

    def __getattr__(self, name: str) -> np.ndarray[tuple[int], Any] | Any:
        """Override attribute retrieval to access pointwise and per-container
        data fields."""

        # Fixed attributes use for implementation: delegate to normal attribute
        # access.
        if name in self.FIXED_FIELDS:
            return super().__getattribute__(name)

        if name in self.X_data:
            return self.X_data[name]
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

        if name in self.X_data:
            # Check that the type of the assigned value can be safely cast to
            # the field type and cast and assign the value if OK.
            self.X_data[name] = self.X_data_dictionary[name].convert_in(
                value, name, self.X_size
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
            if fs.anonymous:
                raise ValueError('cannot add anonymous FieldSet to Container')
            assert fs.fieldset_name is not None
            self.X_fieldsets.add(fs.fieldset_name)
            self.X_data_dictionary = self.X_data_dictionary.merge(fs)

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
                    self.X_data[name] = metadata.convert_in(value, name, self.X_size)
                else:
                    self.X_data[name] = metadata.empty(self.X_size)

    @property
    def species(self) -> list[Species]:
        """Set of species included in any species-indexed fields in the
        container."""
        species = set()
        for name, field in self.X_data_dictionary.items():
            if Dimension.SPECIES in field.dimensions:
                assert isinstance(self.X_data[name], SpeciesValues)
                species.update(self.X_data[name].keys())
        return sorted(species)

    def _check_fieldset(self, fieldset: FieldSet):
        # Fields may not overlap with fixed implementation fields.
        if any(name in self.FIXED_FIELDS for name in fieldset):
            raise ValueError('Field name conflicts with Container fixed attribute')

        # Field sets can only be added once.
        if fieldset.fieldset_name in self.X_fieldsets:
            raise ValueError(
                f'FieldSet with name "{fieldset.fieldset_name}" '
                'already added to Container'
            )

    def copy(self) -> Self:
        """Create a deep copy of the container."""
        new_traj = Container(self.X_size, fieldsets=list(self.X_fieldsets))
        for name in self.X_data_dictionary:
            if name in self.X_data:
                new_traj.X_data[name] = deepcopy(self.X_data[name])
        return new_traj
