from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

import numpy as np
from pydantic import Field

from AEIC.types import SpeciesValues
from AEIC.utils.models import CIBaseModel


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
        return np.arange(self.bottom, self.top, self.resolution)

    def get_indexes(self, value: np.ndarray) -> np.ndarray:
        return np.clip(
            np.floor((value - self.bottom) / self.resolution).astype(int),
            0,
            self.bins - 1,
        )


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
