import numpy as np
from numpy.typing import NDArray

from AEIC.utils.models import CIStrEnum

from .fuel import Fuel
from .spatial import Location, Position
from .species import Species, SpeciesValues
from .time import DayOfWeek, TimeOfDay

# create a type for Union[float, NDArray]
FloatOrNDArray = float | NDArray[np.float64]


class AircraftClass(CIStrEnum):
    WIDE = 'wide'
    NARROW = 'narrow'
    SMALL = 'small'
    FREIGHT = 'freight'


__all__ = [
    'AircraftClass',
    'DayOfWeek',
    'FloatOrNDArray',
    'Fuel',
    'Location',
    'Position',
    'Species',
    'SpeciesValues',
    'TimeOfDay',
]
