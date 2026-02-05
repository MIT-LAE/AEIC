"""Definitions for field metadata and collections of fields ("field sets").

This module defines classes to represent metadata for individual fields and
collections of such fields, known as field sets. Field sets can be registered
for reuse and support merging while ensuring unique field names.

Field sets are used to represent the data and metadata fields associated with
aircraft trajectories, including emissions data.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

import netCDF4 as nc4
import numpy as np

from AEIC.types import Dimension, Dimensions, EmissionsDict, ModeValues, Species


# The options here cause a hash method to be generated for `FieldMetadata`.
@dataclass(eq=True, frozen=True)
class FieldMetadata:
    """Metadata for a single field.

    This is intended to describe the properties of a field in a dataset that
    may be serialized to NetCDF.

    Fields can indexed by a range of different dimensions, represented by the
    `dimensions` field (in the "Type" rows, "scalar" means a Numpy floating
    point or integer type, or Python str):

    - Dimensions: trajectory
    - Use:        per-trajectory individual values
    - Examples:  total fuel burn for a trajectory, number of climb segments, etc.
    - Type:      scalar

    ----------------------------------------------------------------

    - Dimensions: trajectory, point
    - Use:        pointwise values along a trajectory
    - Examples:   trajectory latitude, longitude, altitude
    - Type:       np.ndarray

    ----------------------------------------------------------------

    - Dimensions: trajectory, species
    - Use:        per-species values for a whole trajectory
    - Examples:   APU emissions for a species on a trajectory
    - Type:       EmissionsDict[scalar]

    ----------------------------------------------------------------

    - Dimensions: trajectory, species, point
    - Use:        pointwise values for each species along a trajectory
    - Example:    CO2 emissions at each point along a trajectory
    - Type:       EmissionsDict[np.ndarray]

    ----------------------------------------------------------------

    - Dimensions: trajectory, thrust_mode
    - Use:        per-thrust-mode values for a whole trajectory
    - Examples:   LTO fuel burn per thrust mode for a trajectory
    - Type:       ModeValues

    ----------------------------------------------------------------

    - Dimensions: trajectory, species, thrust_mode
    - Use:        per-species values for each thrust mode for a whole trajectory.
    - Examples:   NOx emiesions per thrust mode for a trajectory
    - Type:       EmissionsDict[ModeValues]

    """

    dimensions: Dimensions = field(default_factory=lambda: Dimensions.from_abbrev('TP'))
    """Dimensions for this field."""

    field_type: type = np.float64
    """Type of the field; should be a Numpy dtype or str for variable-length
    strings. Note that Python int and float are not allowed because we need to
    have a 1-to-1 mapping from Python to Numpy types."""

    description: str = ''
    """Human-readable description of the field (used for the NetCDF
    "description" attribute)."""

    units: str = ''
    """Units of the field (used for the NetCDF "units" attribute)."""

    required: bool = True
    """Is this field required to be present in the dataset?"""

    default: Any | None = None
    """Default value for the field if not present in the dataset."""

    def __post_init__(self):
        # Acceptable field types include numpy dtypes and str, but *not* native
        # Python int and float, because we need to have an unambigous mapping
        # to NetCDF types.
        if self.field_type is int or self.field_type is float:
            raise ValueError(
                'FieldMetadata: field_type cannot be native type '
                f'{self.field_type}; use corresponding Numpy type instead.'
            )

        # The NetCD4 package handles Python strings as a special case.
        if self.field_type is str:
            return

        # Otherwise, check that we have a valid scalar Numpy dtype.
        try:
            dt = np.dtype(self.field_type)
            if dt.shape != ():
                raise ValueError(
                    'FieldMetadata: field_type cannot be an array '
                    f'type: {self.field_type}'
                )
        except TypeError as e:
            raise ValueError(
                f'FieldMetadata: invalid field_type {self.field_type}: {e}'
            ) from e

    def empty(self, npoints: int) -> Any:
        """Create an empty value for this field based on its type and dimensions."""
        # TODO: This is all kind of sketchy, especially the types for Numpy arrays.
        match (
            Dimension.POINT in self.dimensions,
            Dimension.SPECIES in self.dimensions,
            Dimension.THRUST_MODE in self.dimensions,
        ):
            case (False, False, False):
                # Scalar per-trajectory field.
                return self.default
            case (True, False, False):
                # Pointwise field along trajectory.
                return np.zeros(npoints, dtype=self.field_type)
            case (False, True, False):
                # Per-species field for whole trajectory.
                return EmissionsDict()
            case (True, True, False):
                # Pointwise per-species field along trajectory.
                return EmissionsDict()
            case (False, False, True):
                # Per-thrust-mode field for whole trajectory.
                return ModeValues()
            case (False, True, True):
                # Per-species per-thrust-mode field for whole trajectory.
                return EmissionsDict()
            case _:
                raise ValueError(
                    'FieldMetadata.empty: invalid combination of dimensions '
                    f'for field: {self.dimensions.dims}'
                )

    def nbytes(self, npoints: int) -> int:
        """Estimate the number of bytes used by this field per trajectory point.

        This is a rough estimate used for memory usage calculations.
        """
        size = np.dtype(self.field_type).itemsize
        if Dimension.THRUST_MODE in self.dimensions:
            size *= len(ModeValues())
        if Dimension.SPECIES in self.dimensions:
            size *= len(Species)
        if Dimension.POINT in self.dimensions:
            size *= npoints
        return size

    def _cast(self, v: Any, name: str) -> Any:
        """Cast a scalar or array value to this field's base type."""

        # Wrap single values.
        arr = v
        wrapped = False
        if not isinstance(v, np.ndarray):
            arr = np.asarray(v)
            wrapped = True

        # Check that the incoming type can be safely cast to the field type.
        if not np.can_cast(arr, self.field_type, casting='same_kind'):
            raise TypeError(
                f'field {name}: cannot cast assigned value of type {type(v)} '
                f'to field with dimensions {self.dimensions} and '
                f'scalar type {self.field_type}'
            )

        # Return cast value.
        cast_value = arr.astype(self.field_type, casting='same_kind')
        if wrapped:
            cast_value = cast_value.item()
        return cast_value

    def convert_in(self, v: Any, name: str, npoints: int) -> Any:
        """Convert an incoming value to the appropriate type for this field."""

        # Deal with missing optional values first. Missing values read from
        # NetCDF files are often represented as masked arrays with all values
        # masked, so we handle that case explicitly here.
        if not self.required:
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                if (
                    hasattr(v, 'mask')
                    and np.all(getattr(v, 'mask'))
                    or v.size > 1
                    and np.all([e is None for e in v])
                ):
                    return None

        if isinstance(v, np.ma.MaskedArray):
            v = v.filled()

        # Split by dimension cases.
        match (
            Dimension.POINT in self.dimensions,
            Dimension.SPECIES in self.dimensions,
            Dimension.THRUST_MODE in self.dimensions,
        ):
            case (False, False, False):
                # Scalar per-trajectory field.
                return self._cast(v, name)
            case (True, False, False):
                # Pointwise field along trajectory.
                if isinstance(v, np.ndarray):
                    if len(v) != npoints:
                        raise ValueError(
                            f'field {name}: assigned array has length {len(v)}, '
                            f'expected {npoints} for field with point dimension'
                        )
                    return self._cast(v, name)
            case (False, True, False):
                # Per-species field for whole trajectory.
                if isinstance(v, EmissionsDict):
                    return EmissionsDict({s: self._cast(e, name) for s, e in v.items()})
            case (True, True, False):
                # Pointwise per-species field along trajectory.
                if isinstance(v, EmissionsDict) and all(
                    isinstance(e, np.ndarray) for e in v.values()
                ):
                    if any(len(e) != npoints for e in v.values()):
                        raise ValueError(
                            f'field {name}: assigned arrays have inconsistent lengths '
                            f'for field with point dimension'
                        )
                    return EmissionsDict({s: self._cast(e, name) for s, e in v.items()})
            case (False, False, True):
                # Per-thrust-mode field for whole trajectory.
                if isinstance(v, ModeValues):
                    return ModeValues({m: self._cast(x, name) for m, x in v.items()})
            case (False, True, True):
                # Per-species per-thrust-mode field for whole trajectory.
                if isinstance(v, EmissionsDict) and all(
                    isinstance(e, ModeValues) for e in v.values()
                ):
                    return EmissionsDict(
                        {
                            s: ModeValues(
                                {m: self._cast(x, name) for m, x in e.items()}
                            )
                            for s, e in v.items()
                        }
                    )
            case _:
                raise ValueError(
                    f'field {name}: invalid combination of dimensions '
                    f'for field: {self.dimensions.dims}'
                )
        raise TypeError(
            f'field {name}: assigned value of type {type(v)} is not compatible '
            f'with field dimensions {self.dimensions} and '
            f'scalar type {self.field_type}'
        )

    @property
    def digest_info(self) -> str:
        """Generate a string representation of the field metadata for hashing."""
        parts = [
            self.dimensions.abbrev,
            self.field_type.__name__,
            self.description,
            self.units,
            'req' if self.required else 'opt',
            str(self.default) if self.default is not None else 'nodefault',
        ]
        return ','.join(parts)


class FieldSet(Mapping):
    """A collection of field definitions.

    Represented as a mapping from field name to metadata.
    """

    REGISTRY: dict[str, 'FieldSet'] = {}
    """Registry of named field sets for reuse."""

    def __init__(
        self, fieldset_name: str, *, registered: bool = True, **fields: FieldMetadata
    ):
        # Ensure uniqueness in the registry. We do this by comparing hashes of
        # the `FieldSet` contents. (We use the `calc_hash` method for the input
        # fields because we don't yet have a `FieldSet` value to call `hash` on
        # at this point.)
        if registered and fieldset_name in FieldSet.REGISTRY:
            registry_hash = hash(FieldSet.REGISTRY[fieldset_name])
            arg_hash = self.calc_hash(fieldset_name, fields)
            if arg_hash != registry_hash:
                raise ValueError(
                    f'incompatible FieldSet with name "{fieldset_name}" already exists.'
                )

        # Disallow names starting with underscore so that we can use them as
        # internal NetCDF group names.
        if fieldset_name[0] == '_':
            raise ValueError('FieldSet name cannot start with underscore "_"')

        # Use an awkward name here to allow "name" as a field name.
        self.fieldset_name = fieldset_name

        # Save the fields.
        self._fields = dict(fields)

        # Add the `FieldSet` to the registry. "Normal" field sets will be
        # added, but when we merge two field sets, we do not add the merge
        # result to the registry because there is no reasonable name to give it
        # and it's private to whatever code called the `merge` method.
        if registered:
            FieldSet.REGISTRY[fieldset_name] = self

    @classmethod
    def known(cls, name: str) -> bool:
        """Check if a `FieldSet` with the given name exists in the registry."""
        return name in cls.REGISTRY

    @classmethod
    def from_registry(cls, name: str) -> 'FieldSet':
        """Retrieve a `FieldSet` from the registry by name."""
        if not cls.known(name):
            raise KeyError(f'FieldSet with name "{name}" not found in registry.')
        return cls.REGISTRY[name]

    @classmethod
    def from_netcdf_group(cls, group: nc4.Group) -> 'FieldSet':
        """Construct a `FieldSet` from a NetCDF group."""
        fields = {}
        for f, v in group.variables.items():
            # Get a nice Python type from the NetCDF field type.
            field_type = v.dtype
            if hasattr(field_type, 'type'):
                field_type = field_type.type
            assert isinstance(field_type, type)

            # Determine dimensions for variable.
            dimensions = Dimensions.from_dim_names(*v.dimensions)
            if isinstance(v.datatype, nc4.VLType) and field_type is not str:
                dimensions = dimensions.add(Dimension.POINT)

            # Set up the field information.
            metadata = FieldMetadata(
                dimensions=dimensions,
                field_type=field_type,
                description=v.getncattr('description'),
                units=v.getncattr('units'),
                required=v.getncattr('required') == 'true',
                default=v.getncattr('default') if 'default' in v.ncattrs() else None,
            )
            fields[f] = metadata

        # Use the fields in the main `FieldMetadata` constructor. We do not add
        # the `FieldSet` to the registry: only "normal" field sets created
        # directly using the `FieldSet` constructor are registered.
        return cls(group.name, registered=False, **fields)

    def __getitem__(self, key):
        """Look up a field in the `FieldSet`."""
        return self._fields[key]

    def __iter__(self):
        """Iterating over a `FieldSet` iterates over its fields."""
        return iter(self._fields)

    def __len__(self):
        """The length of a `FieldSet` is the number of fields it contains."""
        return len(self._fields)

    def __contains__(self, key):
        """Membership testing for a `FieldSet` is based on field name."""
        return key in self._fields

    @staticmethod
    def calc_hash(name: str, fields: dict[str, FieldMetadata]) -> int:
        """Calculate a hash from a `FieldSet` name and field data."""
        return hash((name, frozenset(fields.items())))

    def __hash__(self):
        """Hash based on name and field definitions."""
        return self.calc_hash(self.fieldset_name, self._fields)

    @cached_property
    def digest(self):
        """Generate persistent hash for `FieldSet`.

        This MD5-based hash is used for identifying field sets within NetCDF
        files and is used to check the integrity of the link between associated
        NetCDF files and base trajectory NetCDF files in the `TrajectoryStore`
        class.
        """
        m = hashlib.md5()
        m.update(self.fieldset_name.encode('utf-8'))
        m.update(b':')
        data = ';'.join(
            name + '=' + self._fields[name].digest_info
            for name in sorted(self._fields.keys())
        )
        m.update(data.encode('utf-8'))
        return m.hexdigest()

    @property
    def fields(self):
        """Return an immutable view of the field definitions."""
        return MappingProxyType(self._fields)

    def merge(self, other):
        """Combine field sets, ensuring unique field names."""
        overlap = self._fields.keys() & other._fields.keys()
        if overlap:
            raise ValueError(f'Overlapping field names: {overlap}')
        merged_fields = dict(self._fields)
        merged_fields.update(other._fields)
        return FieldSet(
            f'{self.fieldset_name}+{other.fieldset_name}',
            registered=False,
            **merged_fields,
        )

    def __repr__(self):
        return f'<FieldSet {self.fieldset_name}: {list(self._fields)}>'

    @property
    def dimensions(self) -> set[Dimension]:
        """Combined dimensions of all fields in the field set."""
        dims = set()
        for f in self._fields.values():
            dims.update(f.dimensions.dims)
        return dims


@runtime_checkable
class HasFieldSets(Protocol):
    """Protocol for objects that have associated field sets."""

    FIELD_SETS = ClassVar[list[FieldSet]]
