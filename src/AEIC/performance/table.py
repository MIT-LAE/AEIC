from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, model_validator

from AEIC.utils.models import CIBaseModel

from .base import BasePerformanceModel

# ------------------------------------------------------------------------------
#
#  Main performance table class
#


class PerformanceTable(CIBaseModel):
    """Performance table data."""

    cols: list[str]
    """Performance table column labels."""

    data: list[list[float]]
    """Performance table data."""

    _df: pd.DataFrame | None = PrivateAttr(default=None)
    """Proof-of-concept representation of performance table as Pandas data
    frame. Intended to replace Numpy array representation eventually."""

    # These are all private attributes because they are derived from the input
    # table data during model validation. These variables support the existing
    # Numpy array-based representation of the performance table. They will
    # eventually be replaced with a representation using a Pandas data frame
    # with a simplified subsetting and interpolation API.
    # TODO: Remove these eventually.
    _performance_table: np.ndarray | None = PrivateAttr(default=None)
    _performance_table_cols: list[list[float]] | None = PrivateAttr(default=None)
    _performance_table_colnames: list[str] | None = PrivateAttr(default=None)
    _columns: dict[str, list[float]] | None = PrivateAttr(default=None)
    _column_idxs: dict[str, int] | None = PrivateAttr(default=None)

    @model_validator(mode='after')
    def process_column_names(self) -> Self:
        """Normalize and check input column names."""

        self.cols = [c.lower() for c in self.cols]

        # Validate column names.
        if len(self.cols) != len(set(self.cols)):
            raise ValueError('Duplicate column names in performance table')
        for required in ['fuel_flow', 'fl', 'tas', 'rocd', 'mass']:
            if required not in self.cols:
                raise ValueError(
                    f'Missing required "{required}" column in performance table'
                )

        return self

    @model_validator(mode='after')
    def make_dataframe(self) -> Self:
        """Convert performance table data to a pandas DataFrame.

        This is a Pydantic model validator that runs whenever a
        PerformanceTable instance is created. It validates the column names and
        data dimensions, and constructs a pandas DataFrame from the data."""

        # Validate data table dimensions.
        ncols = len(self.cols)
        ndata = len(self.data[0])
        if ndata < ncols:
            raise ValueError('Not enough data columns in performance table')
        if any(len(row) != ndata for row in self.data):
            raise ValueError('Inconsistent number of data columns in performance table')
        self._df = pd.DataFrame(
            [row[:ncols] for row in self.data], columns=np.array(self.cols)
        )

        return self

    @property
    def df(self) -> pd.DataFrame:
        """Data frame representation of performance table."""
        assert self._df is not None
        return self._df

    @model_validator(mode='after')
    def old_setup(self) -> Self:
        cols = self.cols
        data = self.data

        # Identify output column (we assume it's the first column or
        # explicitly labeled as fuel flow)
        try:
            output_col_idx = cols.index("fuel_flow")  # Output is fuel flow
        except ValueError:
            raise ValueError("FUEL_FLOW column not found in performance data.")

        input_col_names = [c for i, c in enumerate(cols) if i != output_col_idx]

        # Extract and sort unique values for each input dimension
        input_values = {
            col: sorted(set(row[cols.index(col)] for row in data))
            for col in input_col_names
        }
        input_indices = {
            col: {val: idx for idx, val in enumerate(input_values[col])}
            for col in input_col_names
        }

        # Prepare multidimensional shape and index arrays
        shape = tuple(len(input_values[col]) for col in input_col_names)
        fuel_flow_array = np.zeros(shape)

        # Get index arrays for each input variable
        index_arrays = [
            np.array([input_indices[col][row[cols.index(col)]] for row in data])
            for col in input_col_names
        ]
        index_arrays = tuple(index_arrays)

        # Extract output (fuel flow) values
        fuel_flow = np.array([row[output_col_idx] for row in data])

        # Assign to multidimensional array using advanced indexing
        fuel_flow_array[index_arrays] = fuel_flow

        # Save results
        self._performance_table = fuel_flow_array
        self._performance_table_cols = [input_values[col] for col in input_col_names]
        self._performance_table_colnames = input_col_names
        self._columns = {col: input_values[col] for col in input_col_names}
        self._column_idxs = {col: idx for idx, col in enumerate(input_col_names)}

        return self

    @property
    def performance_table(self) -> np.ndarray:
        assert self._performance_table is not None
        return self._performance_table

    @property
    def performance_table_cols(self) -> list[list[float]]:
        assert self._performance_table_cols is not None
        return self._performance_table_cols

    @property
    def performance_table_colnames(self) -> list[str]:
        assert self._performance_table_colnames is not None
        return self._performance_table_colnames

    @property
    def columns(self) -> dict[str, list[float]]:
        assert self._columns is not None
        return self._columns

    @property
    def column_idxs(self) -> dict[str, int]:
        assert self._column_idxs is not None
        return self._column_idxs

    # TODO: Remove this eventually.
    def search_indexes(
        self, column: str, error_label: str, value: float
    ) -> tuple[int, int]:
        """Searches the valid values in the performance model for the indices
        bounding a known value in a specified column.

        Args:
            column (str): Column name to search.
            value (float): Value of interest.

        Returns:
            (tuple[int, int]) Tuple containing the indices of the values in performance
                data that bound the given value.
        """

        if column not in self.columns:
            raise ValueError(f'Performance data missing required {column} column')

        ind_high = np.searchsorted(self.columns[column], value)

        if ind_high == 0:
            raise ValueError(
                f'Aircraft is trying to operate below minimum {error_label}'
            )
        if ind_high == len(self.columns[column]):
            raise ValueError(
                f'Aircraft is trying to operate above maximum {error_label}'
            )

        return (int(ind_high - 1), int(ind_high))

    def search_mass_ind(self, mass: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given mass value."""
        return self.search_indexes('mass', 'mass', mass)

    def search_flight_levels_ind(self, FL: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given FL value."""
        return self.search_indexes('fl', f'cruise altitude (FL {FL:.2f})', FL)

    def bracketing_fls(self, FL: float) -> tuple[float, float]:
        """Find the bracketing flight levels for a given flight level."""
        ind_low, ind_high = self.search_flight_levels_ind(FL)
        return (self.columns['fl'][ind_low], self.columns['fl'][ind_high])

    def bracketing_mass(self, mass: float) -> tuple[float, float]:
        """Find the bracketing mass values for a given mass."""
        ind_low, ind_high = self.search_mass_ind(mass)
        return (self.columns['mass'][ind_low], self.columns['mass'][ind_high])


# ------------------------------------------------------------------------------
#
#  Performance model class
#


class TablePerformanceModel(BasePerformanceModel):
    """Table-based performance model."""

    # TODO: Better docstrings.

    model_type: Literal['table']
    """Model type identifier for TOML input files."""

    flight_performance: PerformanceTable
    """Main flight performance table."""

    @model_validator(mode='after')
    def validate_pm(self, info):
        """Validate performance model after creation."""

        # TODO: Check that everything is filled in, pulling values from the EDB
        # or elsewhere as required.
        return self

    def fuel_flow(
        self, altitude: float, mass: float, rocd: float, speed: float
    ) -> float:
        """*Fuel flow API not yet implemented.*"""
        raise NotImplementedError(
            'TablePerformanceModel.fuel_flow not implemented yet.'
        )

    def search_mass_ind(self, mass: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given mass value.

        (Forwards to performance table.)"""
        return self.flight_performance.search_indexes('mass', 'mass', mass)

    def search_flight_levels_ind(self, FL: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given FL value.

        (Forwards to performance table.)"""
        return self.flight_performance.search_indexes(
            'fl', f'cruise altitude (FL {FL:.2f})', FL
        )

    def bracketing_fls(self, FL: float) -> tuple[float, float]:
        """Find the bracketing flight levels for a given flight level.

        (Forwards to performance table.)"""
        return self.flight_performance.bracketing_fls(FL)

    def bracketing_mass(self, mass: float) -> tuple[float, float]:
        """Find the bracketing mass values for a given mass.

        (Forwards to performance table.)"""
        return self.flight_performance.bracketing_mass(mass)

    @property
    def performance_table(self) -> np.ndarray:
        """Return performance table."""
        return self.flight_performance.performance_table

    @property
    def performance_table_cols(self) -> list[list[float]]:
        """Return performance table columns.

        (Forwards to performance table.)"""
        return self.flight_performance.performance_table_cols

    @property
    def performance_table_colnames(self) -> list[str]:
        """Return performance table column names.

        (Forwards to performance table.)"""
        return self.flight_performance.performance_table_colnames
