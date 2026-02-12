# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from enum import IntEnum, auto

import numpy as np

from AEIC.performance.types import ThrustModeValues


class Species(IntEnum):
    CO2 = auto()
    H2O = auto()
    HC = auto()
    CO = auto()
    NOx = auto()
    NO = auto()
    NO2 = auto()
    HONO = auto()
    PMnvol = auto()
    PMnvolGMD = auto()
    PMvol = auto()
    OCic = auto()
    SOx = auto()
    SO2 = auto()
    SO4 = auto()
    PMnvolN = auto()


class SpeciesValues[M]:
    """Typed mapping of species to emission values."""

    def __init__(self, data: dict[Species, M] | None = None):
        if data is None:
            data = {}
        self._data: dict[Species, M] = data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpeciesValues) or len(self._data) != len(other._data):
            return False
        if len(self._data) == 0:
            return True
        for key in self._data.keys():
            if key not in other._data:
                return False
            vs = self._data[key]
            vo = other._data[key]
            if isinstance(vs, np.ndarray) and isinstance(vo, np.ndarray):
                if not np.array_equal(vs, vo):
                    return False
            else:
                if vs != vo:
                    return False
        return True

    def isclose(self, other: object) -> bool:
        if not isinstance(other, SpeciesValues) or len(self._data) != len(other._data):
            return False
        if len(self._data) == 0:
            return True
        for key in self._data.keys():
            if key not in other._data:
                return False
            vs = self._data[key]
            vo = other._data[key]
            if isinstance(vs, np.ndarray) and isinstance(vo, np.ndarray):
                if not np.allclose(vs, vo):
                    return False
            elif isinstance(vs, int | float | np.floating):
                if not np.isclose(vs, vo):
                    return False
            elif isinstance(vs, ThrustModeValues):
                if not vs.isclose(vo):
                    return False
            else:
                raise ValueError(
                    'Unsupported type for isclose comparison in SpeciesValues'
                )
        return True

    def __getitem__(self, key: Species) -> M:
        return self._data[key]

    def __setitem__(self, key: Species, value: M) -> None:
        self._data[key] = value

    def __contains__(self, key: Species) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, other: SpeciesValues[M]):
        self._data.update(other._data)

    def __repr__(self) -> str:
        return (
            '<'
            + self.__class__.__name__
            + ': '
            + ', '.join([str(s.name) for s in self._data.keys()])
            + '>'
        )
