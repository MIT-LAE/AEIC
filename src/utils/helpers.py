import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas._libs.tslibs import Timestamp
from pandas.core.indexes.datetimes import DatetimeIndex

from .consts import R_E
from .custom_types import FloatOrNDArray


def great_circle_distance(
    lat1: FloatOrNDArray,
    lon1: FloatOrNDArray,
    lat2: FloatOrNDArray,
    lon2: FloatOrNDArray,
) -> FloatOrNDArray:
    """Calculates the great circle distance between two points

    Args:
        lat1 (Union[NDArray,float]): latitude of the first point in radians
        lon1 (Union[NDArray,float]): longitude of the first point in radians
        lat2 (Union[NDArray,float]): latitude of the second point in radians
        lon2 (Union[NDArray,float]): longitude of the second point in radians

    Returns:
        Union[NDArray,float]: great circle distance between the two points in meters
    """
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)

    alpha = np.asarray(
        np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    )

    alpha[alpha > 1] = 1  # TODO: this is a hack to avoid invalid arccos value
    alpha[alpha < -1] = -1

    return R_E * (np.arccos(alpha))


def knots_to_mps(knots: FloatOrNDArray) -> FloatOrNDArray:
    """Convert knots to meters per second

    Args:
        knots (float or numpy array): Speed in knots

    Returns:
        float or numpy array: Speed in meters per second
    """
    return knots * 0.514444


def mps_to_knots(mps: FloatOrNDArray) -> FloatOrNDArray:
    """Convert meters per second to knots

    Args:
        mps (float or numpy array): Speed in meters per second

    Returns:
        float or numpy array: Speed in knots
    """
    return mps / 0.514444


def meters_to_feet(meters: FloatOrNDArray) -> FloatOrNDArray:
    """Convert meters to feet

    Args:
        meters (float or numpy array): Length in meters

    Returns:
        float or numpy array: Length in feet
    """
    return meters * 3.28084


def feet_to_meters(ft: FloatOrNDArray) -> FloatOrNDArray:
    return ft * 0.3048


def unix_to_datetime_utc(
    unix_time: FloatOrNDArray
) -> Timestamp | DatetimeIndex:
    """Convert unix time to UTC

    Args:
        unix_time (float or numpy array): Unix time

    Returns:
        UTC timestamp(s)

    """
    return pd.to_datetime(unix_time, unit="s")


def calculate_line_parameters(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """
    Calculates the slope and intercept of the lines defined by the points (x, y).

    Parameters
    ----------
    x : NDArray
        The x coordinates of the points.
    y : NDArray
        The y coordinates of the points.

    Returns
    -------
    tuple[NDArray, NDArray]
        The slopes and intercepts of the lines defined by the points (x, y).
    """

    dx = np.diff(x)
    dy = np.diff(y)

    slopes = np.divide(dy, dx, out=np.full_like(dy, np.inf), where=dx != 0)
    # slopes = np.where(dx != 0, dy / dx, np.inf)
    # slopes = dy / dx

    intercepts = y[:-1] - slopes * x[:-1]

    return slopes, intercepts


def crosses_dateline(lon1, lon2):
    diff = lon2 - lon1
    cross = (np.abs(diff) > np.pi).astype(int)
    return np.sign(diff) * cross


def degrees_to_radians(degrees: FloatOrNDArray) -> FloatOrNDArray:
    return np.pi * degrees / 180


def nautmiles_to_meters(nautmiles: FloatOrNDArray) -> FloatOrNDArray:
    return nautmiles * 1852


def filter_order_duplicates(seq):
    ''' Filters duplicate list entries while perserving order '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
