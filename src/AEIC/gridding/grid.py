from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

import numpy as np
from pydantic import Field

from AEIC.types import SpeciesValues
from AEIC.utils.models import CIBaseModel


def _edges_from_levels(levels: np.ndarray) -> np.ndarray:
    """Synthesize N+1 bin edges from N level centers.

    Interior edges are midpoints between adjacent levels. The outer edges are
    extended symmetrically (i.e. each outermost level sits at the center of its
    bin).
    """
    midpoints = 0.5 * (levels[:-1] + levels[1:])
    lower = 2 * levels[0] - midpoints[0]
    upper = 2 * levels[-1] - midpoints[-1]
    return np.concatenate(([lower], midpoints, [upper]))


class HorizontalGrid(CIBaseModel):
    resolution: float
    offset: float
    range: tuple[float, float]

    @property
    def bins(self) -> int:
        return int((self.range[1] - self.range[0]) / self.resolution)

    def get_indexes(self, value: np.ndarray) -> np.ndarray:
        return np.floor((value - self.offset) / self.resolution).astype(int)


class LatitudeGrid(HorizontalGrid):
    range: tuple[float, float] = Field(default=(-90.0, 90.0))


class LongitudeGrid(HorizontalGrid):
    range: tuple[float, float] = Field(default=(-180.0, 180.0))


class HeightGrid(CIBaseModel):
    mode: Literal['height']

    resolution: float
    range: tuple[float, float]

    @property
    def bins(self) -> int:
        return int((self.range[1] - self.range[0]) / self.resolution)

    @property
    def bottom(self) -> float:
        return self.range[0]

    @property
    def top(self) -> float:
        return self.range[1]

    @property
    def levels(self) -> np.ndarray:
        """Bin center values (meters)."""
        return self.range[0] + (np.arange(self.bins) + 0.5) * self.resolution

    @property
    def edges(self) -> np.ndarray:
        """N+1 bin edge values synthesized from level midpoints, with outer
        boundaries extended symmetrically."""
        return _edges_from_levels(self.levels)


class ISAPressureGrid(CIBaseModel):
    mode: Literal['isa_pressure']

    levels: list[float]

    @property
    def bins(self) -> int:
        return len(self.levels)

    @property
    def bottom(self) -> float:
        return max(self.levels)

    @property
    def top(self) -> float:
        return min(self.levels)

    @property
    def edges(self) -> np.ndarray:
        """N+1 bin edge values in ascending pressure order, synthesized from
        level midpoints with outer boundaries extended symmetrically."""
        return _edges_from_levels(np.sort(self.levels))


AltitudeGrid = Annotated[HeightGrid | ISAPressureGrid, Field(discriminator='mode')]


class TrajectoryLike(Protocol):
    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    trajectory_emissions: SpeciesValues[np.ndarray]


@dataclass(slots=True)
class GridCell:
    lon: int
    lat: int
    alt: int


class Grid(CIBaseModel):
    latitude: LatitudeGrid
    longitude: LongitudeGrid
    altitude: AltitudeGrid

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.latitude.bins, self.longitude.bins, self.altitude.bins)

    @classmethod
    def load(cls, file_path: Path | str) -> Grid:
        with open(file_path, 'rb') as fp:
            d = tomllib.load(fp)
            return cls.model_validate(d)
