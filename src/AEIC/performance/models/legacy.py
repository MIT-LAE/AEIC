"""This module implements a performance model intended to replicate the
behavior of the legacy table-based performance model in the original Matlab
AEIC code. The performance data is derived from BADA PTF performance files and
so has the same restrictions in terms of the dependence of fuel flow,
rate-of-climb and airspeed on flight level and aircraft mass.

Performance evaluation is done by bilinear interpolation in flight level and
aircraft mass in the relevant segment of the performance table (climb, cruise
or descent) selected by the flight rules. Performance table data is checked on
construction to ensure that it satisfies the requirements for this
interpolation to work correctly."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, model_validator
from scipy.interpolate import interpn

from AEIC.performance.types import AircraftState, Performance, SimpleFlightRules
from AEIC.units import METERS_TO_FL

from .base import BasePerformanceModel, PerformanceTableInput


class ROCDFilter(Enum):
    """Rate of climb/descent filter for performance table subsetting."""

    NEGATIVE = auto()
    ZERO = auto()
    POSITIVE = auto()


class Interpolator:
    """Grid-based interpolator for performance model data."""

    def __init__(self, df: pd.DataFrame):
        # Requirements:
        #  - Regular FL, regular mass ⇒ rectlinear grid;
        #  - Dense: unique (FL, mass); #rows = #FL × #mass
        #
        # These conditions should be checked in the PerformanceTable
        # constructor, but we check them here for security and testing
        # purposes.
        if len(list(zip(df.fl.values, df.mass.values))) != len(df):
            raise ValueError('Interpolator requires unique (FL, mass) pairs in data')

        # Coordinate values.
        fls = sorted(float(fl) for fl in df.fl.unique())
        self.min_fl = min(fls)
        self.max_fl = max(fls)
        masses = sorted(float(m) for m in df.mass.unique())
        self.min_mass = min(masses)
        self.max_mass = max(masses)

        # If there is only one mass value, we need to do linear interpolation
        # in flight level. Otherwise we will be doing bilinear interpolation in
        # flight level and mass.
        self.n_masses = len(masses)
        if self.n_masses > 1:
            self.xs = (np.array(fls), np.array(masses))

            # Output values.
            shape = (len(fls), len(masses))
            self.tas = np.zeros(shape)
            self.rocd = np.zeros(shape)
            self.fuel_flow = np.zeros(shape)

            # Construct output values.
            for row in df.itertuples():
                i = fls.index(row.fl)  # type: ignore
                j = masses.index(row.mass)  # type: ignore
                self.tas[i, j] = row.tas  # type: ignore
                self.rocd[i, j] = row.rocd  # type: ignore
                self.fuel_flow[i, j] = row.fuel_flow  # type: ignore
        else:
            self.xs = (np.array(fls),)

            # Output values.
            self.tas = df.tas.values
            self.rocd = df.rocd.values
            self.fuel_flow = df.fuel_flow.values

    def __call__(self, fl: float, mass: float) -> Performance:
        """Perform bilinear interpolation to get performance values at given
        flight level and aircraft mass."""

        if self.n_masses > 1:
            x = (
                np.clip(fl, self.min_fl, self.max_fl),
                np.clip(mass, self.min_mass, self.max_mass),
            )
        else:
            x = np.array([np.clip(fl, self.min_fl, self.max_fl)])

        return Performance(
            true_airspeed=float(interpn(self.xs, self.tas, x, method='linear')[0]),
            rate_of_climb=float(interpn(self.xs, self.rocd, x, method='linear')[0]),
            fuel_flow=float(interpn(self.xs, self.fuel_flow, x, method='linear')[0]),
        )


@dataclass
class PerformanceTable:
    """Aircraft performance data table for legacy table-driven performance
    model.

    This class implements performance data interpolation as done in the legacy
    AEIC code. The data in these performance tables is identical to the data
    provided in BADA PTF performance table files. This means that:

    1. The table is divided into three sections, for climb (ROCD>0), cruise
       (ROCD≈0) and descent (ROCD<0).
    2. There are three distinct mass values: low, nominal and high. Only the
       nominal mass value is used in the descent section of the table.
    3. In all sections of the table, TAS depends only on FL.
    4. In the climb section of the table, ROCD depends on FL and mass, and fuel
       flow depends only on FL.
    5. In the cruise section of the table, fuel flow depends on FL and mass.
    6. In the descent section of the table, both ROCD and fuel flow depend only
       on FL.

    On construction, the class checks that the input data satisfies these
    requirements, ensuring that subsequent interpolation in the table data will
    work correctly.

    """

    df: pd.DataFrame
    """Performance table data."""

    fl: list[float]
    """Sorted list of unique flight levels in the table."""

    tas: list[float]
    """Sorted list of unique airspeed values in the table."""

    rocd: list[float]
    """Sorted list of unique ROCD values in the table."""

    mass: list[float]
    """Sorted list of unique mass values in the table."""

    rocd_filter: ROCDFilter
    """ROCD filter for this performance table segment."""

    _interpolator: Interpolator | None = None
    """Interpolator for single flight phase table segment."""

    ZERO_ROCD_TOL: ClassVar[float] = 1.0e-6
    """Tolerance for zero rate of climb/descent comparisons."""

    def __post_init__(self):
        # Check the sign of the ROCD values in the table matches the ROCD
        # filter for this table segment.
        match self.rocd_filter:
            case ROCDFilter.NEGATIVE:
                phase = 'descent'
                if not all(v < -self.ZERO_ROCD_TOL for v in self.rocd):
                    raise ValueError(
                        'ROCD values in descent performance table are not all negative'
                    )
            case ROCDFilter.ZERO:
                phase = 'cruise'
                if not all(abs(v) <= self.ZERO_ROCD_TOL for v in self.rocd):
                    raise ValueError(
                        'ROCD values in cruise performance table are not all zero'
                    )
            case ROCDFilter.POSITIVE:
                phase = 'climb'
                # Condition is different here because climb ROCD values can be
                # zero near the operating ceiling of an aircraft.
                if not all(v >= 0.0 for v in self.rocd):
                    raise ValueError(
                        'some ROCD values in climb performance table are negative'
                    )

        # Check that we have the right number of mass values: two or three for
        # the climb and cruise phases, but one for the descent sub-table.
        mass_ok = True
        if self.rocd_filter == ROCDFilter.NEGATIVE:
            mass_ok = len(self.mass) == 1
        else:
            mass_ok = len(self.mass) in (2, 3)
        if not mass_ok:
            raise ValueError(
                f'Legacy performance table ({phase}) has wrong number of mass values'
            )

        # For each of positive, zero, and negative ROCD, it should be the case
        # that the input data is dense in (FL, mass) values, in the sense that
        # #rows = #FL × #mass.
        if len(self.df.fl.unique()) * len(self.df.mass.unique()) != len(self.df):
            raise ValueError(
                f'Performance data for {phase} does not have full coverage'
            )

        def check_fl_only(var: str):
            if len(self.df.drop_duplicates(subset=['fl', var])) != len(
                self.df.fl.unique()
            ):
                raise ValueError(
                    f'{var} for {phase} phase depends on variables other than FL'
                )

        match self.rocd_filter:
            case ROCDFilter.ZERO:
                # Zero ROC: TAS should depend only on FL.
                check_fl_only('tas')

            case ROCDFilter.POSITIVE:
                # Positive ROC: TAS and fuel flow should depend only on FL.
                check_fl_only('tas')
                check_fl_only('fuel_flow')

            case ROCDFilter.NEGATIVE:
                # Negative ROC: TAS, fuel flow and ROCD should depend only on FL.
                check_fl_only('tas')
                check_fl_only('fuel_flow')
                check_fl_only('rocd')

    @classmethod
    def from_input(cls, ptin: PerformanceTableInput, rocd_type: ROCDFilter) -> Self:
        """Convert performance table data from input format.

        This class holds performance table data in the form needed for
        trajectory and emissions calculations. The constructor converts from
        the input format from the performance model TOML file."""

        # Convert to Pandas DataFrame for easier handling.
        df = pd.DataFrame(
            [row[: len(ptin.cols)] for row in ptin.data], columns=np.array(ptin.cols)
        )

        # Extract column unique values for searching.
        fl = sorted(df.fl.unique().tolist())
        tas = sorted(df.tas.unique().tolist())
        rocd = sorted(df.rocd.unique().tolist())
        mass = sorted(df.mass.unique().tolist())

        return cls(df=df, fl=fl, tas=tas, rocd=rocd, mass=mass, rocd_filter=rocd_type)

    def __len__(self) -> int:
        return len(self.df)

    def interpolate(self, state: AircraftState) -> Performance:
        """Perform bilinear interpolation in flight level and aircraft mass.

        The interpolation is done in the subset of the performance table
        corresponding to the given rate of climb/descent filter."""

        fl = state.altitude * METERS_TO_FL
        mass = state.aircraft_mass
        if mass == 'min':
            mass = min(self.mass)
        elif mass == 'max':
            mass = max(self.mass)

        # Lazily create interpolator for flight phase segment.
        if self._interpolator is None:
            self._interpolator = Interpolator(self.df)

        return self._interpolator(fl, mass)


class LegacyPerformanceModel(BasePerformanceModel[SimpleFlightRules]):
    """Legacy table-based performance model."""

    model_type: Literal['legacy']
    """Model type identifier for TOML input files."""

    climb_flight_performance: PerformanceTableInput
    """Input data for flight performance table in climb phase."""

    cruise_flight_performance: PerformanceTableInput
    """Input data for flight performance table in cruise phase."""

    descent_flight_performance: PerformanceTableInput
    """Input data for flight performance table in descent phase."""

    _climb_performance_table: PerformanceTable = PrivateAttr()
    _cruise_performance_table: PerformanceTable = PrivateAttr()
    _descent_performance_table: PerformanceTable = PrivateAttr()

    @model_validator(mode='after')
    def validate_pm(self, info):
        """Validate performance model after creation."""

        # Create performance tables from "flight_performance" sections.
        self._climb_performance_table = PerformanceTable.from_input(
            self.climb_flight_performance, ROCDFilter.POSITIVE
        )
        self._cruise_performance_table = PerformanceTable.from_input(
            self.cruise_flight_performance, ROCDFilter.ZERO
        )
        self._descent_performance_table = PerformanceTable.from_input(
            self.descent_flight_performance, ROCDFilter.NEGATIVE
        )

        return self

    @property
    def empty_mass(self) -> float:
        """Empty aircraft mass.

        Empty mass per BADA-3 is lowest mass in performance table / 1.2."""
        return (
            min(
                min(self._climb_performance_table.mass),
                min(self._cruise_performance_table.mass),
                min(self._descent_performance_table.mass),
            )
            / 1.2
        )

    @property
    def maximum_mass(self) -> float:
        """Maximum aircraft mass from performance table."""
        return max(
            max(self._climb_performance_table.mass),
            max(self._cruise_performance_table.mass),
            max(self._descent_performance_table.mass),
        )

    @property
    def maximum_rocd(self) -> float:
        return max(
            max(self._climb_performance_table.rocd),
            max(self._cruise_performance_table.rocd),
            max(self._descent_performance_table.rocd),
        )

    @property
    def minimum_tas(self) -> float:
        return min(
            min(self._climb_performance_table.tas),
            min(self._cruise_performance_table.tas),
            min(self._descent_performance_table.tas),
        )

    def performance_table(self, rocd_filter: ROCDFilter) -> PerformanceTable:
        """Performance table accessor."""
        match rocd_filter:
            case ROCDFilter.POSITIVE:
                return self._climb_performance_table
            case ROCDFilter.ZERO:
                return self._cruise_performance_table
            case ROCDFilter.NEGATIVE:
                return self._descent_performance_table

    def evaluate_impl(
        self, state: AircraftState, rules: SimpleFlightRules
    ) -> Performance:
        """Implementation of performance evaluation for legacy table-based
        performance model.

        The performance table is separated into climb, cruise and descent
        segments. The performance evaluation implementation uses bilinear
        interpolation in flight level and aircraft mass in the relevant segment
        of the performance table (selected by the flight rule) to get
        performance values."""

        match rules:
            case SimpleFlightRules.CLIMB:
                return self._climb_performance_table.interpolate(state)
            case SimpleFlightRules.CRUISE:
                return self._cruise_performance_table.interpolate(state)
            case SimpleFlightRules.DESCEND:
                return self._descent_performance_table.interpolate(state)
