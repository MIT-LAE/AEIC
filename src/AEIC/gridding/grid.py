from __future__ import annotations

import tomllib
from typing import Annotated, Literal

from pydantic import Field

from AEIC.utils.models import CIBaseModel


class HorizontalGrid(CIBaseModel):
    resolution: float
    offset: float
    range: tuple[float, float]

    @property
    def bins(self) -> int:
        return int((self.range[1] - self.range[0]) / self.resolution)


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


class Grid(CIBaseModel):
    latitude: LatitudeGrid
    longitude: LongitudeGrid
    altitude: AltitudeGrid

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.latitude.bins, self.longitude.bins, self.altitude.bins)

    @classmethod
    def load(cls, file_path: str) -> Grid:
        with open(file_path, 'rb') as fp:
            d = tomllib.load(fp)
            return cls.model_validate(d)

    def get_cell_indices(self, traj) -> tuple[int, int, int]: ...
