from datetime import UTC, date, datetime
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .consts import R_E
from .custom_types import FloatOrNDArray


def great_circle_distance(
    lat1: FloatOrNDArray,
    lon1: FloatOrNDArray,
    lat2: FloatOrNDArray,
    lon2: FloatOrNDArray,
    degrees: bool = False,
) -> FloatOrNDArray:
    """Calculates the great circle distance between two points

    Args:
        lat1 (Union[NDArray,float]): latitude of the first point in radians
        lon1 (Union[NDArray,float]): longitude of the first point in radians
        lat2 (Union[NDArray,float]): latitude of the second point in radians
        lon2 (Union[NDArray,float]): longitude of the second point in radians
        degrees (bool, optional): If True, the input coordinates are in degrees.

    Returns:
        Union[NDArray,float]: great circle distance between the two points in meters
    """
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    if degrees:
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

    alpha = np.asarray(
        np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    )

    alpha[alpha > 1] = 1  # TODO: this is a hack to avoid invalid arccos value
    alpha[alpha < -1] = -1

    return R_E * (np.arccos(alpha))


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


def filter_order_duplicates(seq):
    '''Filters duplicate list entries while perserving order'''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def date_to_timestamp(d: date) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(d, tzinfo=UTC))


def iso_to_timestamp(s: str) -> pd.Timestamp:
    ts = cast(pd.Timestamp, pd.Timestamp(datetime.fromisoformat(s)))
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(UTC)
