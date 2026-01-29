"""TODO: Docstring here!"""

# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, model_validator
from scipy.interpolate import interpn

from AEIC.performance import AircraftState, Performance, SimpleFlightRules
from AEIC.utils.models import CIBaseModel
from AEIC.utils.units import METERS_TO_FL

from .base import BasePerformanceModel


class PerformanceTableInput(CIBaseModel):
    """Performance table data from TOML file."""

    cols: list[str]
    """Performance table column labels."""

    data: list[list[float]]
    """Performance table data."""

    @model_validator(mode='after')
    def validate_names_and_sizes(self) -> Self:
        """Normalize and check input column names and array sizes."""

        self.cols = [c.lower() for c in self.cols]

        # Validate column names.
        if len(self.cols) != len(set(self.cols)):
            raise ValueError('Duplicate column names in performance table')
        for required in ['fuel_flow', 'fl', 'tas', 'rocd', 'mass']:
            if required not in self.cols:
                raise ValueError(
                    f'Missing required "{required}" column in performance table'
                )

        # Validate data table dimensions.
        ncols = len(self.cols)
        ndata = len(self.data[0])
        if ndata < ncols:
            raise ValueError('Not enough data columns in performance table')
        if any(len(row) != ndata for row in self.data):
            raise ValueError('Inconsistent number of data columns in performance table')

        return self


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
        masses = sorted(float(m) for m in df.mass.unique())

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
            x = (fl, mass)
        else:
            x = np.array([fl])

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

    _interpolators: dict[ROCDFilter, Interpolator] = field(default_factory=dict)
    """Interpolators for each flight phase table segment."""

    ZERO_ROCD_TOL: ClassVar[float] = 1.0e-6
    """Tolerance for zero rate of climb/descent comparisons."""

    def __post_init__(self):
        # Check that we have the right number of mass values: three for the
        # whole table and the same for the climb and cruise sub-tables, but one
        # for the descent sub-table.
        sub_table = None
        n_mass_values = 3
        if all(v > self.ZERO_ROCD_TOL for v in self.rocd):
            sub_table = 'positive'
        elif all(abs(v) <= self.ZERO_ROCD_TOL for v in self.rocd):
            sub_table = 'zero'
        elif all(v < -self.ZERO_ROCD_TOL for v in self.rocd):
            sub_table = 'negative'
            n_mass_values = 1
        if len(self.mass) != n_mass_values:
            sub_table_str = '(' + sub_table + ') ' if sub_table is not None else ''
            raise ValueError(
                f'Legacy performance table {sub_table_str}'
                f'has wrong number of mass values (should be {n_mass_values})'
            )

        # Split into sub-tables for checking. (The asserts are to help type
        # checkers.)
        check_zero = self.df[
            (self.df.rocd >= -self.ZERO_ROCD_TOL) & (self.df.rocd <= self.ZERO_ROCD_TOL)
        ]
        assert isinstance(check_zero, pd.DataFrame)
        check_pos = self.df[self.df.rocd > self.ZERO_ROCD_TOL]
        assert isinstance(check_pos, pd.DataFrame)
        check_neg = self.df[self.df.rocd < -self.ZERO_ROCD_TOL]
        assert isinstance(check_neg, pd.DataFrame)

        def check_coverage(df, label):
            if len(df.fl.unique()) * len(df.mass.unique()) != len(df):
                raise ValueError(
                    f'Performance data at {label} ROC does not have full coverage'
                )

        # For each of positive, zero, and negative ROCD, it should be the case
        # that the input data is dense in (FL, mass) values, in the sense that
        # #rows = #FL × #mass.
        check_coverage(check_zero, 'zero')
        check_coverage(check_pos, 'positive')
        check_coverage(check_neg, 'negative')

        def check_fl_only(df: pd.DataFrame, var: str, label: str):
            if len(df.drop_duplicates(subset=['fl', var])) != len(df.fl.unique()):
                raise ValueError(
                    f'{var} at {label} ROC depends on variables other than FL'
                )

        # Zero ROC: TAS should depend only on FL.
        check_fl_only(check_zero, 'tas', 'zero')

        # Positive ROC: TAS and fuel flow should depend only on FL.
        check_fl_only(check_pos, 'tas', 'positive')
        check_fl_only(check_pos, 'fuel_flow', 'positive')

        # Negative ROC: TAS, fuel flow and ROCD should depend only on FL.
        check_fl_only(check_neg, 'tas', 'negative')
        check_fl_only(check_neg, 'fuel_flow', 'negative')
        check_fl_only(check_neg, 'rocd', 'negative')

    @classmethod
    def from_input(cls, ptin: PerformanceTableInput) -> Self:
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

        return cls(df=df, fl=fl, tas=tas, rocd=rocd, mass=mass)

    def __len__(self) -> int:
        return len(self.df)

    def interpolate(self, state: AircraftState, rocd: ROCDFilter) -> Performance:
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
        if rocd not in self._interpolators:
            self._interpolators[rocd] = Interpolator(self.subset(rocd).df)

        return self._interpolators[rocd](fl, mass)

    def subset(self, rocd: ROCDFilter) -> PerformanceTable:
        """Extract subset of performance table for given rate of climb/descent
        filter."""

        df_new = self.df

        match rocd:
            case ROCDFilter.NEGATIVE:
                df_new = df_new[df_new.rocd < -self.ZERO_ROCD_TOL]
            case ROCDFilter.ZERO:
                df_new = df_new[
                    (df_new.rocd >= -self.ZERO_ROCD_TOL)
                    & (df_new.rocd <= self.ZERO_ROCD_TOL)
                ]
            case ROCDFilter.POSITIVE:
                df_new = df_new[df_new.rocd > self.ZERO_ROCD_TOL]
        assert isinstance(df_new, pd.DataFrame)

        fl_new = sorted(df_new.fl.unique().tolist())
        tas_new = sorted(df_new.tas.unique().tolist())
        rocd_new = sorted(df_new.rocd.unique().tolist())
        mass_new = sorted(df_new.mass.unique().tolist())

        return PerformanceTable(
            df=df_new, fl=fl_new, tas=tas_new, rocd=rocd_new, mass=mass_new
        )


class LegacyPerformanceModel(BasePerformanceModel):
    """Legacy table-based performance model."""

    model_type: Literal['legacy']
    """Model type identifier for TOML input files."""

    flight_performance: PerformanceTableInput
    """Input data for flight performance table."""

    _performance_table: PerformanceTable = PrivateAttr()

    @model_validator(mode='after')
    def validate_pm(self, info):
        """Validate performance model after creation."""

        # Create performance table from "flight_performance" section.
        self._performance_table = PerformanceTable.from_input(self.flight_performance)

        return self

    @property
    def empty_mass(self) -> float:
        """Empty aircraft mass.

        Empty mass per BADA-3 is lowest mass in performance table / 1.2."""
        return min(self.performance_table.mass) / 1.2

    @property
    def maximum_mass(self) -> float:
        """Maximum aircraft mass from performance table."""
        return max(self.performance_table.mass)

    @property
    def performance_table(self) -> PerformanceTable:
        """Performance table accessor."""
        return self._performance_table

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
                return self._performance_table.interpolate(state, ROCDFilter.POSITIVE)
            case SimpleFlightRules.CRUISE:
                return self._performance_table.interpolate(state, ROCDFilter.ZERO)
            case SimpleFlightRules.DESCEND:
                return self._performance_table.interpolate(state, ROCDFilter.NEGATIVE)
