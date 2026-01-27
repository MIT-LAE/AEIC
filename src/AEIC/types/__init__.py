import numpy as np
from numpy.typing import NDArray

from AEIC.utils.models import CIStrEnum

from .emissions import AtmosphericState, EmissionsDict, EmissionsSubset
from .fuel import Fuel
from .performance import (
    AircraftState,
    LTOPerformance,
    ModeValues,
    Performance,
    SimpleFlightRules,
    SpeedData,
    Speeds,
    ThrustLabel,
    ThrustLabelArray,
    ThrustMode,
    ThrustModeArray,
)
from .spatial import Location, Position
from .species import Species
from .storage import Dimension, Dimensions
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
    'AircraftState',
    'AtmosphericState',
    'DayOfWeek',
    'Dimension',
    'Dimensions',
    'EmissionsDict',
    'EmissionsSubset',
    'FloatOrNDArray',
    'Fuel',
    'Location',
    'LTOPerformance',
    'ModeValues',
    'Performance',
    'Position',
    'SimpleFlightRules',
    'Species',
    'SpeedData',
    'Speeds',
    'ThrustLabel',
    'ThrustLabelArray',
    'ThrustMode',
    'ThrustModeArray',
    'TimeOfDay',
]
