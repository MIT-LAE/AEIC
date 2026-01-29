import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import Polygon

from AEIC.units import METERS_TO_FEET
from AEIC.utils.helpers import calculate_line_parameters, crosses_dateline
from AEIC.utils.spatial import great_circle_distance


@dataclass
class GeospatialGrid:
    grid_latitudes: NDArray  # in radians
    grid_longitudes: NDArray  # in radians
    grid_altitudes: NDArray | None = None  # in meters
    grid_times: NDArray | None = None  # in seconds-

    @property
    def n_latitudes(self) -> int:
        return len(self.grid_latitudes)

    @property
    def n_longitudes(self) -> int:
        return len(self.grid_longitudes)

    @property
    def n_altitudes(self) -> int | None:
        if self.grid_altitudes is None:
            raise ValueError("No altitude grid")
        return len(self.grid_altitudes)

    @property
    def n_times(self) -> int | None:
        if self.grid_times is None:
            return None
        return len(self.grid_times)

    @property
    def n_cells(self) -> int:
        n_cells = self.n_latitudes * self.n_longitudes
        if self.n_altitudes is not None:
            n_cells *= self.n_altitudes
        if self.n_times is not None:
            n_cells *= self.n_times
        return n_cells

    @property
    def shape(self) -> tuple[int | None, ...]:
        return (self.n_latitudes, self.n_longitudes, self.n_altitudes, self.n_times)

    @property
    def grid_latitudes_degrees(self) -> NDArray:
        return np.rad2deg(self.grid_latitudes)

    @property
    def grid_longitudes_degrees(self) -> NDArray:
        return np.rad2deg(self.grid_longitudes)

    @property
    def grid_altitudes_km(self) -> NDArray | None:
        if self.grid_altitudes is None:
            return None
        return self.grid_altitudes / 1000

    @property
    def grid_altitudes_feet(self) -> NDArray | None:
        if self.grid_altitudes is None:
            return None
        return self.grid_altitudes * METERS_TO_FEET

    @property
    def grid_times_datetime(self) -> NDArray | None:
        if self.grid_times is None:
            return None
        return pd.to_datetime(self.grid_times, unit="s")

    @property
    def max_grid_latitude(self) -> float:
        return max(self.grid_latitudes)

    @property
    def min_grid_latitude(self) -> float:
        return min(self.grid_latitudes)

    @property
    def max_grid_longitude(self) -> float:
        return max(self.grid_longitudes)

    @property
    def min_grid_longitude(self) -> float:
        return min(self.grid_longitudes)

    @property
    def max_grid_altitude(self) -> float:
        if self.grid_altitudes is None:
            raise ValueError("No altitude grid")
        return max(self.grid_altitudes)

    @property
    def min_grid_altitude(self) -> float:
        if self.grid_altitudes is None:
            raise ValueError("No altitude grid")
        return min(self.grid_altitudes)

    @property
    def max_grid_time(self) -> float:
        if self.grid_times is None:
            raise ValueError("No time grid")
        return max(self.grid_times)

    @property
    def min_grid_time(self) -> float:
        if self.grid_times is None:
            raise ValueError("No time grid")
        return min(self.grid_times)


@dataclass
class Gridder(GeospatialGrid):
    """Main methods"""

    def grid_polygon(self, polygon_lats, polygon_lons):
        polygon_lons_extended = np.concatenate((polygon_lons, [polygon_lons[0]]))
        dateline_crossing = crosses_dateline(
            polygon_lons_extended[:-1], polygon_lons_extended[1:]
        )

        if np.any(np.abs(dateline_crossing) > 0):
            print("crosses dateline")
            print(50 * "-")

            mask_left = polygon_lons > 0
            mask_right = polygon_lons < 0

            polygon_lons_left = polygon_lons[mask_left]
            polygon_lats_left = polygon_lats[mask_left]

            polygon_lons_right = polygon_lons[mask_right]
            polygon_lats_right = polygon_lats[mask_right]

            touched_cells_left = self._polygon_touched_cells(
                polygon_lats_left, polygon_lons_left
            )
            touched_cells_right = self._polygon_touched_cells(
                polygon_lats_right, polygon_lons_right
            )

            return touched_cells_left | touched_cells_right

        else:
            return self._polygon_touched_cells(polygon_lats, polygon_lons)

    def grid_trajectory(
        self,
        lats: NDArray,
        lons: NDArray,
        altitudes: NDArray | None = None,
        times: NDArray | None = None,
        state_variables: tuple[NDArray, ...] = (),
        integrated_variables: tuple[NDArray, ...] = (),
    ) -> tuple[
        NDArray, NDArray, NDArray, NDArray, tuple[NDArray, ...], tuple[NDArray, ...]
    ]:
        """
        this is a refactored version of the
        cells_touched_by_trajectory_with_state_and_integrated_variables method
        """

        dateline_crossing = crosses_dateline(lons[:-1], lons[1:])
        if np.any(dateline_crossing != 0):
            return self._grid_trajectory_with_dateline_crossing(
                dateline_crossing,
                lats,
                lons,
                altitudes,
                times,
                state_variables,
                integrated_variables,
            )
        else:
            return self._grid_trajectory_without_dateline_crossing(
                lats, lons, altitudes, times, state_variables, integrated_variables
            )

    """Helper methods"""

    def _grid_trajectory_without_dateline_crossing(
        self,
        lats: NDArray,
        lons: NDArray,
        altitudes: NDArray | None,
        times: NDArray | None,
        state_variables: tuple[NDArray, ...],
        integrated_variables: tuple[NDArray, ...],
    ) -> tuple[
        NDArray, NDArray, NDArray, NDArray, tuple[NDArray, ...], tuple[NDArray, ...]
    ]:
        (
            touched_cells_lat_indices,
            touched_cells_lon_indices,
            touched_cells_altitude_indices,
            touched_cells_time_indices,
            state_variable_values,
            integrated_variable_values,
        ) = self._cell_idxs_touched_by_trajectory_with_state_and_integrated_vars(
            lats, lons, altitudes, times, state_variables, integrated_variables
        )

        touched_cells_lats = self.grid_latitudes[touched_cells_lat_indices]
        touched_cells_lons = self.grid_longitudes[touched_cells_lon_indices]
        touched_cells_altitudes = (
            self.grid_altitudes[touched_cells_altitude_indices]
            if altitudes is not None
            else None
        )
        touched_cells_times = (
            self.grid_times[touched_cells_time_indices] if times is not None else None
        )

        return (
            touched_cells_lats,
            touched_cells_lons,
            touched_cells_altitudes,
            touched_cells_times,
            state_variable_values,
            integrated_variable_values,
        )

    def _grid_trajectory_with_dateline_crossing(
        self,
        dateline_crossing: NDArray,
        lats: NDArray,
        lons: NDArray,
        altitudes: NDArray | None = None,
        times: NDArray | None = None,
        state_variables: tuple[NDArray, ...] = (),
        integrated_variables: tuple[NDArray, ...] = (),
    ) -> tuple[
        NDArray, NDArray, NDArray, NDArray, tuple[NDArray, ...], tuple[NDArray, ...]
    ]:
        if np.count_nonzero(dateline_crossing) > 1:
            warnings.warn(
                "Trajectory crosses the dateline more than once, "
                "this isn't implemented, returning None for all outputs"
            )
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                (np.array([]),) * len(state_variables),
                (np.array([]),) * len(integrated_variables),
            )

        dateline_crossing_idx = np.where(dateline_crossing != 0)[0][0]
        dateline_crossing_sign = dateline_crossing[dateline_crossing_idx]

        first_segment_length, second_segment_length, total_segment_length = (
            self._calculate_segment_lengths(
                lats, lons, dateline_crossing_idx, dateline_crossing_sign
            )
        )

        (
            lats_first_part,
            lons_first_part,
            altitudes_first_part,
            times_first_part,
            state_variables_first_parts,
            integrated_variables_first_parts,
        ) = self._dateline_split_first_segment(
            lats,
            lons,
            altitudes,
            times,
            state_variables,
            integrated_variables,
            dateline_crossing_idx,
            dateline_crossing_sign,
            first_segment_length,
            total_segment_length,
        )

        (
            lats_second_part,
            lons_second_part,
            altitudes_second_part,
            times_second_part,
            state_variables_second_parts,
            integrated_variables_second_parts,
        ) = self._dateline_split_second_segment(
            lats,
            lons,
            altitudes,
            times,
            state_variables,
            integrated_variables,
            dateline_crossing_idx,
            dateline_crossing_sign,
            second_segment_length,
            total_segment_length,
        )

        return self._cell_idxs_and_variables_for_dateline_split_trajectory(
            lats_first_part,
            lons_first_part,
            altitudes_first_part,
            times_first_part,
            state_variables_first_parts,
            integrated_variables_first_parts,
            lats_second_part,
            lons_second_part,
            altitudes_second_part,
            times_second_part,
            state_variables_second_parts,
            integrated_variables_second_parts,
        )

    def _calculate_segment_lengths(
        self, lats, lons, dateline_crossing_idx, dateline_crossing_sign
    ):
        first_segment_length = great_circle_distance(
            lats[dateline_crossing_idx],
            lons[dateline_crossing_idx],
            lats[dateline_crossing_idx],
            np.pi if dateline_crossing_sign == -1 else -np.pi,
        )

        second_segment_length = great_circle_distance(
            lats[dateline_crossing_idx],
            -np.pi if dateline_crossing_sign == -1 else np.pi,
            lats[dateline_crossing_idx + 1],
            lons[dateline_crossing_idx + 1],
        )

        total_segment_length = first_segment_length + second_segment_length
        return first_segment_length, second_segment_length, total_segment_length

    def _dateline_split_first_segment(
        self,
        lats,
        lons,
        altitudes,
        times,
        state_variables,
        integrated_variables,
        dateline_crossing_idx,
        dateline_crossing_sign,
        first_segment_length,
        total_segment_length,
    ):
        lons_first_part = np.concatenate(
            (
                lons[: dateline_crossing_idx + 1],
                np.array([np.pi if dateline_crossing_sign == -1 else -np.pi]),
            )
        )
        lats_first_part = np.concatenate(
            (
                lats[: dateline_crossing_idx + 1],
                np.array([lats[dateline_crossing_idx]]),
            )
        )
        altitudes_first_part = (
            np.concatenate(
                (
                    altitudes[: dateline_crossing_idx + 1],
                    np.array([altitudes[dateline_crossing_idx]]),
                )
            )
            if altitudes is not None
            else None
        )
        times_first_part = (
            np.concatenate(
                (
                    times[: dateline_crossing_idx + 1],
                    np.array([times[dateline_crossing_idx]]),
                )
            )
            if times is not None
            else None
        )

        state_variables_first_parts = tuple(
            np.concatenate(
                (
                    variable[: dateline_crossing_idx + 1],
                    np.array([variable[dateline_crossing_idx]]),
                )
            )
            for variable in state_variables
        )

        integrated_variables_first_parts = tuple(
            np.concatenate(
                (
                    variable[:dateline_crossing_idx],
                    np.array(
                        [
                            variable[dateline_crossing_idx]
                            * first_segment_length
                            / total_segment_length
                        ]
                    ),
                )
            )
            for variable in integrated_variables
        )

        return (
            lats_first_part,
            lons_first_part,
            altitudes_first_part,
            times_first_part,
            state_variables_first_parts,
            integrated_variables_first_parts,
        )

    def _dateline_split_second_segment(
        self,
        lats,
        lons,
        altitudes,
        times,
        state_variables,
        integrated_variables,
        dateline_crossing_idx,
        dateline_crossing_sign,
        second_segment_length,
        total_segment_length,
    ):
        lons_second_part = np.concatenate(
            (
                np.array([-np.pi if dateline_crossing_sign == -1 else np.pi]),
                lons[dateline_crossing_idx + 1 :],
            )
        )

        lats_second_part = np.concatenate(
            (
                np.array([lats[dateline_crossing_idx]]),
                lats[dateline_crossing_idx + 1 :],
            )
        )

        altitudes_second_part = (
            np.concatenate(
                (
                    np.array([altitudes[dateline_crossing_idx]]),
                    altitudes[dateline_crossing_idx + 1 :],
                )
            )
            if altitudes is not None
            else None
        )

        times_second_part = (
            np.concatenate(
                (
                    np.array([times[dateline_crossing_idx]]),
                    times[dateline_crossing_idx + 1 :],
                )
            )
            if times is not None
            else None
        )

        state_variables_second_parts = tuple(
            np.concatenate(
                (
                    np.array([var[dateline_crossing_idx]]),
                    var[dateline_crossing_idx + 1 :],
                )
            )
            for var in state_variables
        )

        integrated_variables_second_parts = tuple(
            np.concatenate(
                (
                    np.array(
                        [
                            var[dateline_crossing_idx]
                            * second_segment_length
                            / total_segment_length
                        ]
                    ),
                    var[dateline_crossing_idx + 1 :],
                )
            )
            for var in integrated_variables
        )

        return (
            lats_second_part,
            lons_second_part,
            altitudes_second_part,
            times_second_part,
            state_variables_second_parts,
            integrated_variables_second_parts,
        )

    def _cell_idxs_and_variables_for_dateline_split_trajectory(
        self,
        lats_first_part,
        lons_first_part,
        altitudes_first_part,
        times_first_part,
        state_variables_first_parts,
        integrated_variables_first_parts,
        lats_second_part,
        lons_second_part,
        altitudes_second_part,
        times_second_part,
        state_variables_second_parts,
        integrated_variables_second_parts,
    ):
        (
            touched_cells_lat_indices_first_part,
            touched_cells_lon_indices_first_part,
            touched_cells_altitude_indices_first_part,
            touched_cells_time_indices_first_part,
            subsegment_state_variable_values_first_part,
            subsegment_integrated_variable_values_first_part,
        ) = self._cell_idxs_touched_by_trajectory_with_state_and_integrated_vars(
            lats_first_part,
            lons_first_part,
            altitudes_first_part,
            times_first_part,
            state_variables_first_parts,
            integrated_variables_first_parts,
        )

        touched_cells_lats_first_part = self.grid_latitudes[
            touched_cells_lat_indices_first_part
        ]
        touched_cells_lons_first_part = self.grid_longitudes[
            touched_cells_lon_indices_first_part
        ]
        touched_cells_altitudes_first_part = (
            self.grid_altitudes[touched_cells_altitude_indices_first_part]
            if altitudes_first_part is not None
            else None
        )
        touched_cells_times_first_part = (
            self.grid_times[touched_cells_time_indices_first_part]
            if times_first_part is not None
            else None
        )

        (
            touched_cells_lat_indices_second_part,
            touched_cells_lon_indices_second_part,
            touched_cells_altitude_indices_second_part,
            touched_cells_time_indices_second_part,
            subsegment_state_variable_values_second_part,
            subsegment_integrated_variable_values_second_part,
        ) = self._cell_idxs_touched_by_trajectory_with_state_and_integrated_vars(
            lats_second_part,
            lons_second_part,
            altitudes_second_part,
            times_second_part,
            state_variables_second_parts,
            integrated_variables_second_parts,
        )

        touched_cells_lats_second_part = self.grid_latitudes[
            touched_cells_lat_indices_second_part
        ]
        touched_cells_lons_second_part = self.grid_longitudes[
            touched_cells_lon_indices_second_part
        ]
        touched_cells_altitudes_second_part = (
            self.grid_altitudes[touched_cells_altitude_indices_second_part]
            if altitudes_second_part is not None
            else None
        )
        touched_cells_times_second_part = (
            self.grid_times[touched_cells_time_indices_second_part]
            if times_second_part is not None
            else None
        )

        return (
            np.concatenate(
                [touched_cells_lats_first_part, touched_cells_lats_second_part]
            ),
            np.concatenate(
                [touched_cells_lons_first_part, touched_cells_lons_second_part]
            ),
            (
                np.concatenate(
                    [
                        touched_cells_altitudes_first_part,
                        touched_cells_altitudes_second_part,
                    ]
                )
                if altitudes_first_part is not None
                else None
            ),
            (
                np.concatenate(
                    [
                        touched_cells_times_first_part,
                        touched_cells_times_second_part,
                    ]
                )
                if times_first_part is not None
                else None
            ),
            tuple(
                np.concatenate([first_part, second_part])
                for first_part, second_part in zip(
                    subsegment_state_variable_values_first_part,
                    subsegment_state_variable_values_second_part,
                )
            ),
            tuple(
                np.concatenate([first_part, second_part])
                for first_part, second_part in zip(
                    subsegment_integrated_variable_values_first_part,
                    subsegment_integrated_variable_values_second_part,
                )
            ),
        )

    def _trajectory_intersection_points_and_cells_horizontal(
        self, lats: NDArray, lons: NDArray
    ) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
        """
        For each consecutive pair of points in the trajectory (a segment)
        returns the points where the segment intersects the grid and the
        indexes of the grid cells in which resulting subsegments fall

        Parameters
        ----------
        lats : NDArray
            The latitudes of the trajectory points. IN RADIANS!
        lons : NDArray
            The longitudes of the trajectory points. IN RADIANS!
        grid : GeospatialGrid
            The grid to use. (latitudes and longitudes in radians)

        Returns
        -------
        tuple[tuple[NDArray, NDArray], tuple[NDArray,NDArray]]
            The latitudes and longitudes of the intersection points and the lat and lon
            indices of the grid cells in which the resulting subsegments fall.
            - The shape of the latitudes and longitudes point arrays - the first output-
            is (#of trajectory points - 1, maximum segment index change  + 2),
            the first column are trajectory points and the last column are the
            following trajectory points (i.e. shifted by one).
            The middle columns describe the points where the trajectory segments
            intersect the grid. if there are fewer intersections than the maximum
            index change (i.e. the maximum number of gridlines intersected by a single
            segment in the trajectory) then some columns will be nan.
            The reason for the +2 in the number of columns is that the first and
            following trajectory points are also included in the output.
            - The shape of the lat and lon indices arrays - the second output - is
            (#of trajectory points - 1, maximum segment index change + 1),
            it contains the lat and lon indices of the grid cells that each segment of
            the trajectory intersects, including the ones that the segment starts and
            ends in. If the segment is fully inside a single gridcell, only the first
            column is not nan
            if the segment crosses 1 gridline (index change = 1), it touches two cells.
            If it crosses 2 gridlines (index change = 2), it touches 3 cells, etc.
            Hence why the +1 in the number of columns.
            if the segment has a smaller index change than the maximum index change,
            some columns will be nan.
        """

        # Number of segments is one less than the number of points
        n_segments = len(lats) - 1

        # Calculate sign of change for lat and lon coordinates for each segment
        lat_change_signs = np.sign(np.diff(lats))
        lon_change_signs = np.sign(np.diff(lons))

        # Calculate line properties of trajectory segments
        slopes, intercepts = calculate_line_parameters(lats, lons)

        # Get the indices of the grid cells where trajectory points are located
        lat_grid_indices = np.searchsorted(self.grid_latitudes, lats) - 1
        lon_grid_indices = np.searchsorted(self.grid_longitudes, lons) - 1

        # Get index range for each trajectory segment
        lat_index_ranges = np.column_stack(
            (lat_grid_indices[:-1], lat_grid_indices[1:])
        )
        lon_index_ranges = np.column_stack(
            (lon_grid_indices[:-1], lon_grid_indices[1:])
        )

        # Get index change of each trajectory segment
        lat_index_changes = lat_index_ranges[:, 1] - lat_index_ranges[:, 0]
        lon_index_changes = lon_index_ranges[:, 1] - lon_index_ranges[:, 0]

        absolute_index_changes = np.abs(lat_index_changes) + np.abs(lon_index_changes)
        # maximum index change in lat and lon
        max_lat_index_change = np.max(np.abs(lat_index_changes))
        max_lon_index_change = np.max(np.abs(lon_index_changes))

        # unique index changes in lat and lon
        unique_lat_index_changes = np.unique(lat_index_changes)
        unique_lon_index_changes = np.unique(lon_index_changes)

        # initialize empty arrays for lat and lon lines intersected by the
        # trajectory segments, the lat array should have shape (n_segments,
        # max_lat_index_change) and the lon array should have
        # shape (n_segments, max_lon_index_change)
        lat_lines_intersected = np.empty(
            (n_segments, max_lat_index_change), dtype=float
        )
        lat_lines_intersected[:] = np.nan
        lon_lines_intersected = np.empty(
            (n_segments, max_lon_index_change), dtype=float
        )
        lon_lines_intersected[:] = np.nan

        lons_for_lat_intersections = np.empty(
            (n_segments, max_lat_index_change), dtype=float
        )
        lons_for_lat_intersections[:] = np.nan
        lats_for_lon_intersections = np.empty(
            (n_segments, max_lon_index_change), dtype=float
        )
        lats_for_lon_intersections[:] = np.nan

        # find the lat and lon lines intersected by the segment from the index ranges
        # each row of lat index range contains the lat index of the cell in which
        # the start and end point of the segment are located
        # e.g. if the lat index range is [0,2] then the segment intersects
        # the lat lines at index 1 and 2
        # the same applies to lon index range
        # store the lat and lon lines intersected by the segment in the
        # lat and lon lines intersected arrays
        # do this in a vectorized way

        for _lat_index_change in unique_lat_index_changes:
            _abs_lat_index_change = np.abs(_lat_index_change)
            _mask = lat_index_changes == _lat_index_change
            _lat_index_ranges = lat_index_ranges[_mask]
            _start_lat_index = _lat_index_ranges[:, 0]
            _slopes = slopes[_mask]
            _intercepts = intercepts[_mask]

            _change_range = np.arange(_abs_lat_index_change)
            if _lat_index_change < 0:
                _change_range *= -1
            else:
                _change_range += 1

            # now create _lat_indexes_intersected which has shape
            # (sum(_mask, _lat_index_change) and contains the lat
            # indexes of the lat lines intersected by the segment,
            # keep in mind that _start_lat_index has shape (sum(_mask),)
            _lat_indexes_intersected = (
                _start_lat_index[:, np.newaxis] + _change_range[np.newaxis, :]
            )

            _lat_lines_intersected = self.grid_latitudes[_lat_indexes_intersected]

            _lons_for_lat_intersections = np.multiply(
                np.expand_dims(_slopes, axis=1), _lat_lines_intersected
            ) + np.expand_dims(_intercepts, axis=1)

            # now store the lat lines intersected by the segment in the
            # lat_lines_intersected array
            lat_lines_intersected[_mask, :_abs_lat_index_change] = (
                _lat_lines_intersected
            )
            lons_for_lat_intersections[_mask, :_abs_lat_index_change] = (
                _lons_for_lat_intersections
            )

        for _lon_index_change in unique_lon_index_changes:
            if _lon_index_change == 0:
                continue
            _abs_lon_index_change = np.abs(_lon_index_change)
            _mask = lon_index_changes == _lon_index_change
            _lon_index_ranges = lon_index_ranges[_mask]
            _start_lon_index = _lon_index_ranges[:, 0]
            _slopes = slopes[_mask]
            _intercepts = intercepts[_mask]
            _lats = lats[:-1][_mask]

            _change_range = np.arange(_abs_lon_index_change)
            if _lon_index_change < 0:
                _change_range *= -1
            else:
                _change_range += 1

            # now create _lon_indexes_intersected which has shape\
            # (sum(_mask, _lon_index_change) and contains the lon
            # indexes of the lon lines intersected by the segment,
            # keep in mind that _start_lon_index has shape (sum(_mask),)
            _lon_indexes_intersected = (
                _start_lon_index[:, np.newaxis] + _change_range[np.newaxis, :]
            )

            _lon_lines_intersected = self.grid_longitudes[_lon_indexes_intersected]

            _inf_mask = np.isinf(_slopes)
            _lats_for_lon_intersections = np.empty(
                (_lon_indexes_intersected.shape), dtype=float
            )

            _lats_for_lon_intersections[~_inf_mask] = np.divide(
                _lon_lines_intersected[~_inf_mask]
                - np.expand_dims(_intercepts[~_inf_mask], axis=1),
                np.expand_dims(_slopes[~_inf_mask], axis=1),
            )

            _lats_for_lon_intersections[_inf_mask] = np.expand_dims(
                _lats[_inf_mask], axis=1
            )

            # now store the lon lines intersected by the segment in the
            # lon_lines_intersected array
            lon_lines_intersected[_mask, :_abs_lon_index_change] = (
                _lon_lines_intersected
            )
            lats_for_lon_intersections[_mask, :_abs_lon_index_change] = (
                _lats_for_lon_intersections
            )

        # now we have the lat and lon lines intersected by the trajectory segments,
        # we combine with lats and lons for lon and lat intersections and sort
        intersection_point_lats = np.column_stack(
            (lat_lines_intersected, lats_for_lon_intersections)
        )

        intersection_point_lats[lat_change_signs == -1] = -intersection_point_lats[
            lat_change_signs == -1
        ]
        intersection_point_lats.sort(axis=1)
        intersection_point_lats[lat_change_signs == -1] = -intersection_point_lats[
            lat_change_signs == -1
        ]
        intersection_point_lats = intersection_point_lats[
            :, ~np.all(np.isnan(intersection_point_lats), axis=0)
        ]

        intersection_point_lons = np.column_stack(
            (lon_lines_intersected, lons_for_lat_intersections)
        )

        intersection_point_lons[lon_change_signs == -1] = -intersection_point_lons[
            lon_change_signs == -1
        ]
        intersection_point_lons.sort(axis=1)
        intersection_point_lons[lon_change_signs == -1] = -intersection_point_lons[
            lon_change_signs == -1
        ]
        intersection_point_lons = intersection_point_lons[
            :, ~np.all(np.isnan(intersection_point_lons), axis=0)
        ]

        # find midpoints between neighboring intersections points
        midpoints_lats = (
            intersection_point_lats[:, :-1] + intersection_point_lats[:, 1:]
        ) / 2
        midpoints_lons = (
            intersection_point_lons[:, :-1] + intersection_point_lons[:, 1:]
        ) / 2

        # lat and lon indices of the gridcells where midpoints are located
        midpoint_lat_indices = np.searchsorted(self.grid_latitudes, midpoints_lats) - 1
        midpoint_lat_indices = np.where(
            np.isnan(midpoints_lats), np.nan, midpoint_lat_indices
        )
        midpoint_lon_indices = np.searchsorted(self.grid_longitudes, midpoints_lons) - 1
        midpoint_lon_indices = np.where(
            np.isnan(midpoints_lons), np.nan, midpoint_lon_indices
        )

        all_subsegment_lat_indices = np.column_stack(
            (lat_grid_indices[:-1], midpoint_lat_indices, lat_grid_indices[1:])
        )
        all_subsegment_lon_indices = np.column_stack(
            (lon_grid_indices[:-1], midpoint_lon_indices, lon_grid_indices[1:])
        )

        all_subsegment_lat_indices[absolute_index_changes == 0, -1] = np.nan
        all_subsegment_lon_indices[absolute_index_changes == 0, -1] = np.nan

        all_subsegment_point_lats = np.column_stack(
            (lats[:-1], intersection_point_lats, lats[1:])
        )
        all_subsegment_point_lons = np.column_stack(
            (lons[:-1], intersection_point_lons, lons[1:])
        )

        return (all_subsegment_point_lats, all_subsegment_point_lons), (
            all_subsegment_lat_indices,
            all_subsegment_lon_indices,
        )

    def _trajectory_time_grid_indices(self, times: NDArray) -> NDArray:
        if self.grid_times is None:
            raise ValueError("No time grid")
        return (np.searchsorted(self.grid_times, times) - 1).astype(int)

    def _trajectory_altitude_grid_indices(self, altitudes: NDArray) -> NDArray:
        if self.grid_altitudes is None:
            raise ValueError("No altitude grid")
        return (np.searchsorted(self.grid_altitudes, altitudes) - 1).astype(int)

    def _trajectory_segment_time_grid_indices(self, times: NDArray) -> NDArray:
        if self.grid_times is None:
            raise ValueError("No time grid")
        return (np.searchsorted(self.grid_times, times) - 1)[:-1]

    def _trajectory_segment_altitude_grid_indices(self, altitudes: NDArray) -> NDArray:
        if self.grid_altitudes is None:
            raise ValueError("No altitude grid")
        return (np.searchsorted(self.grid_altitudes, altitudes) - 1)[:-1]

    def _cell_idxs_touched_by_trajectory_with_state_and_integrated_vars(
        self,
        lats: NDArray,
        lons: NDArray,
        altitudes: NDArray | None = None,
        times: NDArray | None = None,
        state_variables: tuple[NDArray, ...] = (),
        integrated_variables: tuple[NDArray, ...] = (),
    ) -> tuple[
        NDArray,
        NDArray,
        NDArray,
        NDArray,
        tuple[NDArray, ...],
        tuple[NDArray, ...],
    ]:
        (
            (all_subsegment_point_lats, all_subsegment_point_lons),
            (
                all_subsegment_lat_indices,
                all_subsegment_lon_indices,
            ),
        ) = self._trajectory_intersection_points_and_cells_horizontal(lats, lons)

        count_subsegments = np.count_nonzero(
            ~np.isnan(all_subsegment_lat_indices), axis=1
        )

        if altitudes is not None:
            segment_altitude_indices = self._trajectory_segment_altitude_grid_indices(
                altitudes
            )
            touched_cells_altitude_indices = np.repeat(
                segment_altitude_indices, count_subsegments
            )
        else:
            touched_cells_altitude_indices = np.array([])

        if times is not None:
            segment_time_indices = self._trajectory_segment_time_grid_indices(times)
            touched_cells_time_indices = np.repeat(
                segment_time_indices, count_subsegments
            )
        else:
            touched_cells_time_indices = np.array([])

        touched_cells_lat_indices = (
            all_subsegment_lat_indices[~np.isnan(all_subsegment_lat_indices)]
            .flatten()
            .astype(int)
        )

        touched_cells_lon_indices = (
            all_subsegment_lon_indices[~np.isnan(all_subsegment_lon_indices)]
            .flatten()
            .astype(int)
        )

        # state_variables handling
        state_variable_values = tuple(
            np.repeat(variable[:-1], count_subsegments) for variable in state_variables
        )

        # integrated_variables handling
        if integrated_variables:
            non_segment_idxs = (np.cumsum(count_subsegments + 1) - 1)[:-1]

            all_segment_point_lats_flat = all_subsegment_point_lats[
                ~np.isnan(all_subsegment_point_lats)
            ].flatten()
            all_segment_point_lons_flat = all_subsegment_point_lons[
                ~np.isnan(all_subsegment_point_lons)
            ].flatten()

            subsegment_distances = great_circle_distance(
                all_segment_point_lats_flat[:-1],
                all_segment_point_lons_flat[:-1],
                all_segment_point_lats_flat[1:],
                all_segment_point_lons_flat[1:],
            )

            subsegment_distances = np.delete(subsegment_distances, non_segment_idxs)

            segment_distances = great_circle_distance(
                lats[:-1], lons[:-1], lats[1:], lons[1:]
            )

            segment_distances_repeated = np.repeat(segment_distances, count_subsegments)

            subsegment_distance_fractions = np.divide(
                subsegment_distances,
                segment_distances_repeated,
                out=np.zeros_like(subsegment_distances),
                where=segment_distances_repeated != 0,
            )

            integrated_variable_values = tuple(
                np.repeat(variable, count_subsegments) * subsegment_distance_fractions
                for variable in integrated_variables
            )
        else:
            integrated_variable_values = ()

        return (
            touched_cells_lat_indices,
            touched_cells_lon_indices,
            touched_cells_altitude_indices,
            touched_cells_time_indices,
            state_variable_values,
            integrated_variable_values,
        )

    def _polygon_touched_cells(self, polygon_lats, polygon_lons):
        polygon = Polygon(zip(polygon_lons, polygon_lats))

        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        min_lat_idx = np.searchsorted(self.grid_latitudes, min_lat) - 1
        max_lat_idx = np.searchsorted(self.grid_latitudes, max_lat) - 1
        min_lon_idx = np.searchsorted(self.grid_longitudes, min_lon) - 1
        max_lon_idx = np.searchsorted(self.grid_longitudes, max_lon) - 1

        touched_cells = set()

        for lat_idx in range(min_lat_idx, max_lat_idx + 1):
            for lon_idx in range(min_lon_idx, max_lon_idx + 1):
                cell = Polygon(
                    [
                        (self.grid_longitudes[lon_idx], self.grid_latitudes[lat_idx]),
                        (
                            self.grid_longitudes[lon_idx + 1],
                            self.grid_latitudes[lat_idx],
                        ),
                        (
                            self.grid_longitudes[lon_idx + 1],
                            self.grid_latitudes[lat_idx + 1],
                        ),
                        (
                            self.grid_longitudes[lon_idx],
                            self.grid_latitudes[lat_idx + 1],
                        ),
                    ]
                )

                if cell.intersects(polygon):
                    touched_cells.add(
                        (self.grid_latitudes[lat_idx], self.grid_longitudes[lon_idx])
                    )

        return touched_cells

    def cells_touched_by_trajectory_with_state_and_integrated_variables(
        self,
        lats: NDArray,
        lons: NDArray,
        altitudes: NDArray | None = None,
        times: NDArray | None = None,
        state_variables: tuple[NDArray, ...] = (),
        integrated_variables: tuple[NDArray, ...] = (),
    ) -> tuple[
        NDArray | None,
        NDArray | None,
        NDArray | None,
        NDArray | None,
        tuple[NDArray | None, ...],
        tuple[NDArray | None, ...],
    ]:
        # does the same as cells_touched_by_trajectory but for
        # both state and integrated variables

        dateline_crossing = crosses_dateline(lons[:-1], lons[1:])
        if np.any(dateline_crossing != 0):
            if np.count_nonzero(dateline_crossing) > 1:
                warnings.warn(
                    "Trajectory crosses the dateline more than once, "
                    "this isn't implemented, returning None for all outputs"
                )
                return None, None, None, None, None, None

            dateline_crossing_idx = np.where(dateline_crossing != 0)[0][0]
            dateline_crossing_sign = dateline_crossing[dateline_crossing_idx]

            first_segment_length = great_circle_distance(
                lats[dateline_crossing_idx],
                lons[dateline_crossing_idx],
                lats[dateline_crossing_idx],
                np.pi if dateline_crossing_sign == -1 else -np.pi,
            )

            second_segment_length = great_circle_distance(
                lats[dateline_crossing_idx],
                -np.pi if dateline_crossing_sign == -1 else np.pi,
                lats[dateline_crossing_idx + 1],
                lons[dateline_crossing_idx + 1],
            )

            total_segment_length = first_segment_length + second_segment_length

            lons_first_part = np.concatenate(
                (
                    lons[: dateline_crossing_idx + 1],
                    np.array([np.pi if dateline_crossing_sign == -1 else -np.pi]),
                )
            )
            lats_first_part = np.concatenate(
                (
                    lats[: dateline_crossing_idx + 1],
                    np.array([lats[dateline_crossing_idx]]),
                )
            )
            altitudes_first_part = (
                np.concatenate(
                    (
                        altitudes[: dateline_crossing_idx + 1],
                        np.array([altitudes[dateline_crossing_idx]]),
                    )
                )
                if altitudes is not None
                else None
            )
            times_first_part = (
                np.concatenate(
                    (
                        times[: dateline_crossing_idx + 1],
                        np.array([times[dateline_crossing_idx]]),
                    )
                )
                if times is not None
                else None
            )

            state_variables_first_parts = tuple(
                np.concatenate(
                    (
                        variable[: dateline_crossing_idx + 1],
                        np.array([variable[dateline_crossing_idx]]),
                    )
                )
                for variable in state_variables
            )

            integrated_variables_first_parts = tuple(
                np.concatenate(
                    (
                        variable[:dateline_crossing_idx],
                        np.array(
                            [
                                variable[dateline_crossing_idx]
                                * first_segment_length
                                / total_segment_length
                            ]
                        ),
                    )
                )
                for variable in integrated_variables
            )

            lons_second_part = np.concatenate(
                (
                    np.array([-np.pi if dateline_crossing_sign == -1 else np.pi]),
                    lons[dateline_crossing_idx + 1 :],
                )
            )

            lats_second_part = np.concatenate(
                (
                    np.array([lats[dateline_crossing_idx]]),
                    lats[dateline_crossing_idx + 1 :],
                )
            )

            altitudes_second_part = (
                np.concatenate(
                    (
                        np.array([altitudes[dateline_crossing_idx]]),
                        altitudes[dateline_crossing_idx + 1 :],
                    )
                )
                if altitudes is not None
                else None
            )

            times_second_part = (
                np.concatenate(
                    (
                        np.array([times[dateline_crossing_idx]]),
                        times[dateline_crossing_idx + 1 :],
                    )
                )
                if times is not None
                else None
            )

            state_variables_second_parts = tuple(
                np.concatenate(
                    (
                        np.array([var[dateline_crossing_idx]]),
                        var[dateline_crossing_idx + 1 :],
                    )
                )
                for var in state_variables
            )

            integrated_variables_second_parts = tuple(
                np.concatenate(
                    (
                        np.array(
                            [
                                var[dateline_crossing_idx]
                                * second_segment_length
                                / total_segment_length
                            ]
                        ),
                        var[dateline_crossing_idx + 1 :],
                    )
                )
                for var in integrated_variables
            )

            (
                touched_cells_lat_indices_first_part,
                touched_cells_lon_indices_first_part,
                touched_cells_altitude_indices_first_part,
                touched_cells_time_indices_first_part,
                subsegment_state_variable_values_first_part,
                subsegment_integrated_variable_values_first_part,
            ) = self._cell_idxs_touched_by_trajectory_with_state_and_integrated_vars(
                lats_first_part,
                lons_first_part,
                altitudes_first_part,
                times_first_part,
                state_variables_first_parts,
                integrated_variables_first_parts,
            )

            touched_cells_lats_first_part = self.grid_latitudes[
                touched_cells_lat_indices_first_part
            ]
            touched_cells_lons_first_part = self.grid_longitudes[
                touched_cells_lon_indices_first_part
            ]
            touched_cells_altitudes_first_part = (
                self.grid_altitudes[touched_cells_altitude_indices_first_part]
                if altitudes_first_part is not None
                else None
            )
            touched_cells_times_first_part = (
                self.grid_times[touched_cells_time_indices_first_part]
                if times_first_part is not None
                else None
            )

            (
                touched_cells_lat_indices_second_part,
                touched_cells_lon_indices_second_part,
                touched_cells_altitude_indices_second_part,
                touched_cells_time_indices_second_part,
                subsegment_state_variable_values_second_part,
                subsegment_integrated_variable_values_second_part,
            ) = self._cell_idxs_touched_by_trajectory_with_state_and_integrated_vars(
                lats_second_part,
                lons_second_part,
                altitudes_second_part,
                times_second_part,
                state_variables_second_parts,
                integrated_variables_second_parts,
            )

            touched_cells_lats_second_part = self.grid_latitudes[
                touched_cells_lat_indices_second_part
            ]
            touched_cells_lons_second_part = self.grid_longitudes[
                touched_cells_lon_indices_second_part
            ]
            touched_cells_altitudes_second_part = (
                self.grid_altitudes[touched_cells_altitude_indices_second_part]
                if altitudes_second_part is not None
                else None
            )
            touched_cells_times_second_part = (
                self.grid_times[touched_cells_time_indices_second_part]
                if times_second_part is not None
                else None
            )

            return (
                np.concatenate(
                    [touched_cells_lats_first_part, touched_cells_lats_second_part]
                ),
                np.concatenate(
                    [touched_cells_lons_first_part, touched_cells_lons_second_part]
                ),
                (
                    np.concatenate(
                        [
                            touched_cells_altitudes_first_part,
                            touched_cells_altitudes_second_part,
                        ]
                    )
                    if altitudes is not None
                    else None
                ),
                (
                    np.concatenate(
                        [
                            touched_cells_times_first_part,
                            touched_cells_times_second_part,
                        ]
                    )
                    if times is not None
                    else None
                ),
                tuple(
                    np.concatenate([first_part, second_part])
                    for first_part, second_part in zip(
                        subsegment_state_variable_values_first_part,
                        subsegment_state_variable_values_second_part,
                    )
                ),
                tuple(
                    np.concatenate([first_part, second_part])
                    for first_part, second_part in zip(
                        subsegment_integrated_variable_values_first_part,
                        subsegment_integrated_variable_values_second_part,
                    )
                ),
            )

        else:
            (
                touched_cells_lat_indices,
                touched_cells_lon_indices,
                touched_cells_altitude_indices,
                touched_cells_time_indices,
                state_variable_values,
                integrated_variable_values,
            ) = self._cell_idxs_touched_by_trajectory_with_state_and_integrated_vars(
                lats, lons, altitudes, times, state_variables, integrated_variables
            )

            touched_cells_lats = self.grid_latitudes[touched_cells_lat_indices]
            touched_cells_lons = self.grid_longitudes[touched_cells_lon_indices]
            touched_cells_altitudes = (
                self.grid_altitudes[touched_cells_altitude_indices]
                if altitudes is not None
                else None
            )
            touched_cells_times = (
                self.grid_times[touched_cells_time_indices]
                if times is not None
                else None
            )

            return (
                touched_cells_lats,
                touched_cells_lons,
                touched_cells_altitudes,
                touched_cells_times,
                state_variable_values,
                integrated_variable_values,
            )
