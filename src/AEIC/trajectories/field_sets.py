"""Definitions for field metadata and collections of fields.

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
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from netCDF4 import Group, VLType

# The options here cause a hash method to be generated for FieldMetadata.


@dataclass(eq=True, frozen=True)
class FieldMetadata:
    """Metadata for a single field.

    This is intended to describe the properties of a field in a dataset that
    may be serialized to NetCDF. Fields can either be per-data-point variables
    (metadata=False) or per-trajectory metadata (metadata=True).
    """

    metadata: bool = False
    field_type: type = np.float64
    description: str = ''
    units: str = ''
    required: bool = True

    def __post_init__(self):
        # Acceptable field types include numpy dtypes and str, but *not* native
        # Python int and float, because we need to have an unambigous mapping
        # to NetCDF types.
        if self.field_type is int or self.field_type is float:
            raise ValueError(
                'FieldMetadata: field_type cannot be native type '
                f'{self.field_type}; use corresponding Numpy type instead.'
            )
        if self.field_type is str:
            return
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
    """Registry of named FieldSets for reuse."""

    def __init__(
        self, fieldset_name: str, registered: bool = True, **fields: FieldMetadata
    ):
        # Ensure uniqueness in the registry.
        if (
            registered
            and fieldset_name in FieldSet.REGISTRY
            and (
                hash(FieldSet.REGISTRY[fieldset_name])
                != self.calc_hash(fieldset_name, fields)
            )
        ):
            raise ValueError(
                f'incompatible FieldSet with name "{fieldset_name}" already exists.'
            )

        # Disallow names starting with underscore so that we can use them as
        # internal group names.
        if fieldset_name[0] == '_':
            raise ValueError('FieldSet name cannot start with underscore "_"')

        # Use an awkward name here to allow "name" as a field name.
        self.fieldset_name = fieldset_name
        self._fields = dict(fields)
        if registered:
            FieldSet.REGISTRY[fieldset_name] = self

    @classmethod
    def known(cls, name: str) -> bool:
        """Check if a FieldSet with the given name exists in the registry."""
        return name in cls.REGISTRY

    @classmethod
    def from_registry(cls, name: str) -> 'FieldSet':
        """Retrieve a FieldSet from the registry by name."""
        if name not in cls.REGISTRY:
            raise KeyError(f'FieldSet with name "{name}" not found in registry.')
        return cls.REGISTRY[name]

    @classmethod
    def from_netcdf_group(cls, group: Group) -> 'FieldSet':
        """Construct a FieldSet from a NetCDF group."""
        fields = {}
        for field_name, var in group.variables.items():
            field_type = var.dtype
            if hasattr(field_type, 'type'):
                field_type = field_type.type
            assert isinstance(field_type, type)
            metadata = FieldMetadata(
                metadata=not isinstance(var.datatype, VLType) or field_type is str,
                field_type=field_type,
                description=var.getncattr('description'),
                units=var.getncattr('units'),
                required=var.getncattr('required') == 'true',
            )
            fields[field_name] = metadata
        return cls(group.name, registered=False, **fields)

    def __getitem__(self, key):
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def __contains__(self, key):
        return key in self._fields

    @staticmethod
    def calc_hash(name: str, fields: dict[str, FieldMetadata]) -> int:
        return hash((name, frozenset(fields.items())))

    def __hash__(self):
        """Hash based on name and field definitions."""
        return self.calc_hash(self.fieldset_name, self._fields)

    @cached_property
    def digest(self):
        """Generate persistent hash for field set."""
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
        """Combine field sets, ensuring unique field names."""
        overlap = self._fields.keys() & other._fields.keys()
        if overlap:
            raise ValueError(f'Overlapping field names: {overlap}')
        merged_fields = dict(self._fields)
        merged_fields.update(other._fields)
        return FieldSet(
            f'{self.fieldset_name}+{other.fieldset_name}', False, **merged_fields
        )

    def __repr__(self):
        return f'<FieldSet {self.fieldset_name}: {list(self._fields)}>'


@runtime_checkable
class HasFieldSets(Protocol):
    """Protocol for objects that have associated FieldSets."""

    FIELD_SETS = ClassVar[list[FieldSet]]
