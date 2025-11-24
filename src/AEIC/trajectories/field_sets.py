"""Definitions for field metadata and collections of fields ("field sets").

This module defines classes to represent metadata for individual fields and
collections of such fields, known as `FieldSet`s. `FieldSet`s can be registered
for reuse and support merging while ensuring unique field names.

`FieldSet`s are used to represent the data and metadata fields associated with
aircraft trajectories, including emissions data.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

import netCDF4
import numpy as np


# The options here cause a hash method to be generated for `FieldMetadata`.
@dataclass(eq=True, frozen=True)
class FieldMetadata:
    """Metadata for a single field.

    This is intended to describe the properties of a field in a dataset that
    may be serialized to NetCDF. Fields can either be per-data-point variables
    (metadata=False) or per-trajectory metadata (metadata=True).
    """

    metadata: bool = False
    """Is this a metadata field (one value per trajectory) or a data variable
    (one value per point along a trajectory)?"""

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


class FieldSet(Mapping):
    """A collection of field definitions.

    Represented as a mapping from field name to metadata.
    """

    REGISTRY: dict[str, 'FieldSet'] = {}
    """Registry of named `FieldSet`s for reuse."""

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

        # Add the `FieldSet` to the registry. "Normal" `FieldSet`s will be
        # added, but when we merge two `FieldSet`s, we do not add the merge
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
    def from_netcdf_group(cls, group: netCDF4.Group) -> 'FieldSet':
        """Construct a `FieldSet` from a NetCDF group."""
        fields = {}
        for f, v in group.variables.items():
            # Get a nice Python type from the NetCDF field type.
            field_type = v.dtype
            if hasattr(field_type, 'type'):
                field_type = field_type.type
            assert isinstance(field_type, type)

            # Scalar or string fields are metadata fields.
            # TODO: Make this condition better?
            is_metadata = (
                not isinstance(v.datatype, netCDF4.VLType) or field_type is str
            )

            # Set up the field information.
            metadata = FieldMetadata(
                metadata=is_metadata,
                field_type=field_type,
                description=v.getncattr('description'),
                units=v.getncattr('units'),
                required=v.getncattr('required') == 'true',
                default=v.getncattr('default') if 'default' in v.ncattrs() else None,
            )
            fields[f] = metadata

        # Use the fields in the main `FieldMetadata` constructor. We do not add
        # the `FieldSet` to the registry: only "normal" `FieldSet`s created
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

        This MD5-based hash is used for identifying `FieldSet`s within NetCDF
        files and is used to check the integrity of the link between associated
        NetCDF files and base trajectory NetCDF files in the `TrajectoryStore`
        class."""
        m = hashlib.md5()
        m.update(self.fieldset_name.encode('utf-8'))
        m.update(b':')
        m.update(repr(sorted(self._fields.items())).encode('utf-8'))
        return m.hexdigest()

    @property
    def fields(self):
        """Return an immutable view of the field definitions."""
        return MappingProxyType(self._fields)

    def merge(self, other):
        """Combine `FieldSet`s, ensuring unique field names."""
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


@runtime_checkable
class HasFieldSets(Protocol):
    """Protocol for objects that have associated `FieldSet`s."""

    FIELD_SETS = ClassVar[list[FieldSet]]
