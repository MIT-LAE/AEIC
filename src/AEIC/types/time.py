# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class DayOfWeek(Enum):
    """Days of the week, with Monday=1 through Sunday=7."""

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7

    @classmethod
    def from_pandas(cls, t: pd.Timestamp) -> DayOfWeek:
        """Extract day of week from a pandas `Timestamp`."""
        return cls(t.isoweekday())


@dataclass
class TimeOfDay:
    """A time of day as hours and minutes."""

    hour: int
    """Hour of day, 0-23."""

    minute: int
    """Minute of hour, 0-59."""
