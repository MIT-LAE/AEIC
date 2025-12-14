from datetime import UTC, date, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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
    """Convert a Python `date` to a UTC Pandas `Timestamp` at midnight."""
    return cast(pd.Timestamp, pd.Timestamp(d, tzinfo=UTC))


def iso_to_timestamp(s: str) -> pd.Timestamp:
    """Convert an ISO 8601 string to a UTC Pandas `Timestamp`."""
    ts = cast(pd.Timestamp, pd.Timestamp(datetime.fromisoformat(s)))
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(UTC)


def deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
