# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

from typing import Literal

from pydantic import PrivateAttr, model_validator

from AEIC.performance.types import AircraftState, Performance, SimpleFlightRules
from AEIC.performance.utils.performance_table import (
    PerformanceTable,
    PerformanceTableInput,
    ROCDFilter,
)

from .base import BasePerformanceModel


class TablePerformanceModel(BasePerformanceModel):
    """Table-based performance model."""

    # TODO: Better docstrings.

    model_type: Literal['table']
    """Model type identifier for TOML input files."""

    flight_performance: PerformanceTableInput
    """Main flight performance table."""

    _performance_table: PerformanceTable = PrivateAttr()
    _performance_subsets: dict[ROCDFilter, PerformanceTable] = PrivateAttr()

    @model_validator(mode='after')
    def validate_pm(self, info):
        """Validate performance model after creation."""

        # TODO: Check that everything is filled in, pulling values from the EDB
        # or elsewhere as required. (I think this comment is obsolete now.)
        self._performance_table = PerformanceTable.from_input(self.flight_performance)

        # Make performance table subsets for climb, cruise and descent.
        self._performance_subsets = {
            rocd: self._performance_table.subset(rocd=rocd) for rocd in ROCDFilter
        }

        return self

    @property
    def empty_mass(self) -> float:
        # Empty mass per BADA-3 is lowest mass / 1.2.
        return min(self.performance_table.mass) / 1.2

    @property
    def maximum_mass(self) -> float:
        return max(self.performance_table.mass)

    @property
    def performance_table(self) -> PerformanceTable:
        return self._performance_table

    def evaluate_impl(
        self, state: AircraftState, rules: SimpleFlightRules
    ) -> Performance:
        # Extract performance table subset for the given flight rule.
        match rules:
            case SimpleFlightRules.CLIMB:
                table = self._performance_subsets[ROCDFilter.POSITIVE]
            case SimpleFlightRules.CRUISE:
                table = self._performance_subsets[ROCDFilter.ZERO]
            case SimpleFlightRules.DESCEND:
                table = self._performance_subsets[ROCDFilter.NEGATIVE]

        # Interpolate performance table to get performance values.
        interp = table.interpolate(state=state)

        return Performance(
            true_airspeed=interp.true_airspeed,
            rate_of_climb=interp.rate_of_climb,
            fuel_flow=interp.fuel_flow,
        )
