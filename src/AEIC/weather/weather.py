import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from AEIC.missions import Mission
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.utils.units import FEET_TO_METERS, NAUTICAL_MILES_TO_METERS


class Weather:
    """
    A class to create and query regridded weather data slices along flight paths.
    Produces regridded slices for u, v, temperature, and pressure.

    Parameters
    ----------
    ds : xarray.Dataset | str
        Weather data containing variables: 't', 'u', 'v' and coordinate 'pressure_level'
        or a path to a NetCDF file containing that data.
    mission : Mission
        Mission information with origin, destination location
                as well as missions start time
    ground_track : GroundTrack object with waypoints along mission
    fl_min : int, optional
        Minimum flight level for regridding (default: 4)
    fl_max : int, optional
        Maximum flight level for regridding (default: 518)
    fl_spacing : int, optional
        Flight level spacing for regridding (default: 1)
    """

    def __init__(
        self,
        ds: xr.Dataset | str,
        mission: Mission,
        ground_track: GroundTrack,
        fl_min: int = 4,
        fl_max: int = 518,
        fl_spacing: int = 10,
    ):
        self.fl_min = fl_min
        self.fl_max = fl_max
        self.fl_spacing = fl_spacing

        self.valid_time_index = self._mission_hour(mission)

        # Load dataset (opening file if a path is provided).
        ds_obj, owns_dataset = self._prepare_dataset(ds)
        self.ds = self._select_valid_time(ds_obj)

        self.origin = mission.origin_position.location
        self.destination = mission.destination_position.location

        # Initialize storage for regridded data
        self.t_regridded = None
        self.p_regridded = None
        self.u_regridded = None
        self.v_regridded = None

        # Arc information
        self.arc_info = self._build_arc_info(ground_track)
        self.total_nm = float(self.arc_info['distances_nm'][-1])

        # Generate all regridded maps
        self._initialize_maps()

        # Once regridded maps are built, close the dataset if we opened it
        # from disk so that the NetCDF file handle is released.
        if owns_dataset:
            ds_obj.close()

    def _mission_hour(self, mission: Mission) -> int:
        """Extract the hour-of-day for mission departure."""
        try:
            return int(getattr(mission.departure, 'hour'))
        except Exception:
            try:
                return int(pd.to_datetime(mission.departure).hour)
            except Exception:
                return 0

    def _prepare_dataset(self, ds: xr.Dataset | str) -> tuple[xr.Dataset, bool]:
        """Return dataset object and whether this class owns the open handle."""
        if isinstance(ds, str):
            return xr.open_dataset(ds), True
        return ds, False

    def _select_valid_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Slice dataset to the mission departure hour when a time dimension exists."""
        if ds is None:
            raise ValueError('Weather dataset must be provided')

        if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
            if ds['valid_time'].values.size > 1:
                idx = self.valid_time_index % ds['valid_time'].values.size
                return ds.isel(valid_time=idx)
            return ds

        if 'time' in ds.dims or 'time' in ds.coords:
            if ds['time'].values.size > 1:
                idx = self.valid_time_index % ds['time'].values.size
                return ds.isel(time=idx)
            return ds

        return ds

    def _lat_lon_names(self) -> tuple[str, str]:
        """Return latitude and longitude coordinate names in the dataset."""
        if self.ds is None:
            raise ValueError('Weather dataset is not initialized')

        lat_name = (
            'latitude'
            if 'latitude' in self.ds.coords
            else ('lat' if 'lat' in self.ds.coords else list(self.ds.dims)[-2])
        )
        lon_name = (
            'longitude'
            if 'longitude' in self.ds.coords
            else ('lon' if 'lon' in self.ds.coords else list(self.ds.dims)[-1])
        )
        return lat_name, lon_name

    @staticmethod
    def _smallest_step(coord: np.ndarray) -> float | None:
        """Return the smallest non-zero absolute step in a 1D coordinate array."""
        coord = np.asarray(coord, dtype=float)
        if coord.size < 2:
            return None
        diffs = np.abs(np.diff(coord))
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return None
        return float(np.min(diffs))

    def _native_grid_spacing_m(self) -> float | None:
        """Approximate native horizontal spacing (meters) from weather grid."""
        if self.ds is None:
            return None

        lat_name, lon_name = self._lat_lon_names()
        lat_step = self._smallest_step(self.ds[lat_name].values)
        lon_step = self._smallest_step(self.ds[lon_name].values)

        lat_spacing_nm = lat_step * 60.0 if lat_step else None  # 1 deg lat ~ 60 NM
        mean_lat = float(np.mean(self.ds[lat_name].values))
        lon_spacing_nm = (
            lon_step * 60.0 * max(np.cos(np.deg2rad(mean_lat)), 1e-6)
            if lon_step
            else None
        )

        spacings_nm = [s for s in (lat_spacing_nm, lon_spacing_nm) if s]
        if not spacings_nm:
            return None
        spacing_nm = min(spacings_nm)
        return spacing_nm * NAUTICAL_MILES_TO_METERS

    def _build_arc_info(self, ground_track: GroundTrack) -> dict[str, np.ndarray]:
        """Sample the ground track using the native weather grid spacing."""
        base_distances_m = np.asarray(ground_track.index, dtype=float)
        total_distance_m = float(base_distances_m[-1])

        spacing_m = self._native_grid_spacing_m()
        if spacing_m is None or spacing_m <= 0:
            distances_m = base_distances_m
        else:
            distances_m = np.arange(
                0.0, total_distance_m + spacing_m, spacing_m, dtype=float
            )
            if distances_m[-1] > total_distance_m:
                distances_m[-1] = total_distance_m
            if distances_m[-1] < total_distance_m:
                distances_m = np.append(distances_m, total_distance_m)
            distances_m = np.union1d(distances_m, base_distances_m)

        distances_nm = distances_m / NAUTICAL_MILES_TO_METERS

        lats: list[float] = []
        lons: list[float] = []
        headings: list[float] = []

        for dist_m in distances_m:
            pt = ground_track.location(dist_m)
            lats.append(pt.location.latitude)
            lons.append(pt.location.longitude)
            headings.append(pt.azimuth % 360.0)
        return {
            'lats': np.asarray(lats),
            'lons': np.asarray(lons),
            'headings': np.asarray(headings),
            'distances_nm': distances_nm,
        }

    def _initialize_maps(self):
        """
        Generate all regridded weather maps during initialization.

        """
        # print('Initializing weather slice maps...')

        # 2. Generate regridded maps for t, pressure_level, u, v (without special logic)
        # print('  - Processing temperature (t)...')
        self.t_regridded = self._create_basic_regridded_map('t')

        # print('  - Processing pressure (pressure_level)...')
        self.p_regridded = self._create_basic_regridded_map('pressure_level')

        # print('  - Processing u-wind component...')
        self.u_regridded = self._create_basic_regridded_map('u')

        # print('  - Processing v-wind component...')
        self.v_regridded = self._create_basic_regridded_map('v')

        # print('Initialization complete!')

    def _create_basic_regridded_map(self, var_name: str) -> pd.DataFrame:
        """
        Create a regridded map for a weather variable.

        Parameters
        ----------
        var_name : str
            Name of the variable in the dataset ('t', 'pressure_level', 'u', or 'v')

        Returns
        -------
        pd.DataFrame
            Regridded data with columns for each distance point along the arc
        """
        if self.arc_info is None:
            raise ValueError('Ground track information is not initialized')
        if self.ds is None:
            raise ValueError('Weather dataset is not initialized')

        # Get coordinate names
        lat_name, lon_name = self._lat_lon_names()

        lats = self.ds[lat_name].values
        lons = self.ds[lon_name].values
        pressure_name = self._pressure_coord_name()

        # Get arc points
        arc_lats = self.arc_info['lats']
        arc_lons = self.arc_info['lons']
        arc_distances_nm = self.arc_info['distances_nm']

        # Nearest neighbor indices
        lat_idx = self._nearest_indices_1d(lats, arc_lats)
        lon_idx = self._nearest_indices_1d(lons, arc_lons)

        # Get pressure levels and convert to flight levels
        pressure_levels = self.ds[pressure_name].values
        # flight_levels = [self._pressure_to_fl(p) for p in pressure_levels]

        # Extract variable data along arc
        if var_name == 'pressure_level':
            var_data = np.repeat(
                pressure_levels[:, None], len(arc_distances_nm), axis=1
            )
        else:
            var_data = (
                self.ds[var_name]
                .isel(
                    **{
                        lat_name: xr.DataArray(lat_idx, dims='points'),
                        lon_name: xr.DataArray(lon_idx, dims='points'),
                    }
                )
                .values
            )  # shape: (n_pressure_levels, n_points)

        # Create fine grid of flight levels
        fine_fls = list(range(self.fl_min, self.fl_max + 1, self.fl_spacing))
        fine_pressures = [self._fl_to_pressure(fl) for fl in fine_fls]

        # Interpolate to fine grid
        regridded_data = np.zeros((len(fine_fls), len(arc_distances_nm)))

        for point_idx in range(len(arc_distances_nm)):
            # Interpolate vertically at this point
            interpolator = np.interp(
                fine_pressures, pressure_levels[::-1], var_data[::-1, point_idx]
            )
            regridded_data[:, point_idx] = interpolator

        if var_name == 'pressure_level':
            regridded_data *= 100.0  # convert from hPa to Pa for outputs

        # Create DataFrame
        # columns = ['Flight_Level'] + [f'NM_{d:.1f}' for d in arc_distances_nm]
        rows = []
        for i, fl in enumerate(fine_fls):
            row = {'Flight_Level': fl}
            for j, d in enumerate(arc_distances_nm):
                row[f'NM_{d:.1f}'] = regridded_data[i, j]
            rows.append(row)

        return pd.DataFrame(rows)

    def _pressure_coord_name(self) -> str:
        """Return the name of the pressure coordinate in the dataset."""
        for name in ('pressure_level', 'isobaricInhPa', 'level'):
            if self.ds is not None and name in self.ds:
                return name
        raise KeyError('Weather dataset missing pressure level coordinate')

    def _fl_to_pressure(self, fl: int) -> float:
        """Convert flight level to pressure."""
        altitude_ft = fl * 100
        altitude_m = altitude_ft / 3.28084
        pressure_hPa = 1013.25 * (1.0 - altitude_m / 44330.0) ** 5.255
        return pressure_hPa

    def _validate_altitudes(self, alt_array: np.ndarray, *, context: str) -> np.ndarray:
        """
        Convert altitudes in feet to flight levels and ensure they reside within the
        regridded flight level range.

        Parameters
        ----------
        altitudes : np.ndarray
            Altitudes in feet to validate.
        context : str
            Name of the public method requesting validation for improved error messages.

        Returns
        -------
        np.ndarray
            Flight levels derived from the provided altitudes.

        Raises
        ------
        ValueError
            If any altitude falls outside the supported range.
        """
        fls = np.asarray(alt_array, dtype=float) / 100.0

        min_fl = int(self.fl_min)
        max_fl = int(self.fl_max)
        out_of_range = (fls < min_fl) | (fls > max_fl)

        if np.any(out_of_range):
            min_alt_ft = min_fl * 100
            max_alt_ft = max_fl * 100
            invalid_alts = ', '.join(f'{alt:.0f}' for alt in alt_array[out_of_range])
            print(
                f'{context} received altitude(s) outside the supported range '
                f'{min_alt_ft}–{max_alt_ft} ft (flight levels {min_fl}–{max_fl}). '
                f'Invalid altitude(s): {invalid_alts} ft. '
                'Returning interpolated altitudes'
            )
            return np.clip(fls, min_fl, max_fl)

        return fls

    def get_supported_altitude_range_ft(self) -> tuple[int, int]:
        """
        Return the inclusive altitude range (in feet) covered by the regridded data.

        Returns
        -------
        tuple[int, int]
            Minimum and maximum supported altitudes in feet.
        """
        return (int(self.fl_min) * 100, int(self.fl_max) * 100)

    def get_heading_from_gd(self, query):
        ground_distances = self.arc_info['distances_nm']
        headings = self.arc_info['headings']

        unwrapped = np.unwrap(np.deg2rad(headings))
        interp = np.interp(
            query, ground_distances, unwrapped, left=unwrapped[0], right=unwrapped[-1]
        )
        headings = (np.rad2deg(interp) + 360.0) % 360.0
        return headings

    # Below code is from different usage of Weather class (AACES)
    # Keeping it in case newer trajectories need it (weather/altitude optimized traj)
    # def project_wind_to_aircraft_frame(self, u, v, heading_deg):
    #     """
    #     Rotate ERA5 wind components (east=u, north=v) into aircraft body axes.

    #     Parameters
    #     ----------
    #     u : array_like
    #         Eastward wind component [m/s] (ERA5 'u'), positive toward +east.
    #     v : array_like
    #         Northward wind component [m/s] (ERA5 'v'), positive toward +north.
    #     heading_deg : array_like
    #         Aircraft heading [deg], clockwise from *true* north.

    #     Returns
    #     -------
    #     u_new : ndarray
    #         Along-track wind [m/s]; + = tailwind (boosts groundspeed), - = headwind.
    #     v_new : ndarray
    #         Cross-track wind [m/s];
    #               + = to aircraft's right (starboard), - = to left (port).
    #     """
    #     u = np.asarray(u, dtype=float)
    #     v = np.asarray(v, dtype=float)
    #     psi = np.deg2rad(np.asarray(heading_deg, dtype=float))

    #     sinψ = np.sin(psi)
    #     cosψ = np.cos(psi)

    #     # Forward/right components
    #     u_new = u * sinψ + v * cosψ  # tailwind (+) / headwind (-)
    #     v_new = u * cosψ - v * sinψ  # rightward (+) / leftward (-)

    #     return u_new, v_new

    # def get_weather_slice(
    #     self, ground_track: np.ndarray, altitude: np.ndarray,
    #     project_wind_to_ac_frame: bool = False
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Get weather variables at specific ground track and altitude coordinates.
    #     `project_wind_to_ac_frame` Projects the wind vector into the local aircraft
    #     frame, so `u_new` would be in the direction of aircraft velocity
    #     (i..e. headwind/tailwind component) and `v_new` being lateral flow component.

    #     ----------
    #     ground_track : np.ndarray
    #         Ground track positions in nautical miles
    #     altitude : np.ndarray
    #         Altitudes in feet. All altitudes must lie within the regridded range
    #         defined by
    #           ``fl_min``/``fl_max`` (see ``get_supported_altitude_range_ft``).

    #     """
    #     if len(ground_track) != len(altitude):
    #         raise ValueError('ground_track and altitude must have the same length')

    #     fls = self._validate_altitudes(altitude, context='get_weather_slice')

    #     n_points = len(ground_track)
    #     u_out = np.zeros(n_points)
    #     v_out = np.zeros(n_points)
    #     t_out = np.zeros(n_points)
    #     p_out = np.zeros(n_points)

    #     for i, (nm, fl) in enumerate(zip(ground_track, fls)):
    #         # Interpolate each variable
    #         u_out[i] = self._interpolate_value(self.u_regridded, nm, fl)
    #         v_out[i] = self._interpolate_value(self.v_regridded, nm, fl)
    #         t_out[i] = self._interpolate_value(self.t_regridded, nm, fl)
    #         p_out[i] = self._interpolate_value(self.p_regridded, nm, fl)

    #     if project_wind_to_ac_frame:
    #         heading = self.get_heading_from_gd(ground_track)
    #         u_new, v_new = self.project_wind_to_aircraft_frame(u_out, v_out, heading)
    #     else:
    #         u_new, v_new = u_out, v_out
    #     return u_new, v_new, t_out, p_out

    def compute_ground_speed(
        self,
        ground_distance_m: float,
        altitude_m: float,
        tas_ms: float,
        track_deg: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Compute groundspeed at a point along the mission using regridded winds.

        Parameters
        ----------
        ground_distance_m : float
            Distance flown along the ground track from the origin [meters].
        altitude_m : float
            Altitude above sea level [meters].
        tas_ms : float
            True airspeed [m/s].
        track_deg : float, optional
            Track heading [degrees].
            If omitted, use the precomputed ground-track heading.

        Returns
        -------
        tuple
            Ground speed [m/s], heading used [deg],
            eastward wind [m/s], northward wind [m/s]
        """
        if self.u_regridded is None or self.v_regridded is None:
            raise ValueError('Weather maps have not been initialized')

        nm = ground_distance_m / NAUTICAL_MILES_TO_METERS
        alt_ft = altitude_m / FEET_TO_METERS
        fl = float(
            self._validate_altitudes(
                np.array([alt_ft]), context='compute_ground_speed'
            )[0]
        )

        heading_deg = (
            float(track_deg)
            if track_deg is not None
            else float(self.get_heading_from_gd(nm))
        )

        wind_u = self._interpolate_value(self.u_regridded, nm, fl)
        wind_v = self._interpolate_value(self.v_regridded, nm, fl)

        heading_rad = np.deg2rad(heading_deg)
        u_air = tas_ms * np.cos(heading_rad)
        v_air = tas_ms * np.sin(heading_rad)

        gs_ms = float(np.hypot(u_air + wind_u, v_air + wind_v))
        return gs_ms, heading_deg, wind_u, wind_v

    def _interpolate_value(self, df: pd.DataFrame, nm: float, fl: int) -> float:
        """
        Interpolate a value from a regridded dataframe at a given arc distance and
        flight level.

        Parameters
        ----------
        df : pandas.DataFrame
            Regridded output including a ``Flight_Level`` column and ``NM_*`` columns.
        nm : float
            Ground-track distance in nautical miles to sample.
        fl : int
            Flight level (hundreds of feet) to sample.

        Returns
        -------
        float
            Bilinearly interpolated value at the requested location.
        """
        # Get available distance columns
        dist_cols = [c for c in df.columns if c.startswith('NM_')]
        distances = np.array([float(c.split('_')[1]) for c in dist_cols])

        # Get flight levels
        flight_levels = df['Flight_Level'].values

        # Create 2D grid values
        values = df[dist_cols].values  # shape: (n_fls, n_distances)

        # Clamp query point to grid bounds
        nm_clamped = np.clip(nm, distances.min(), distances.max())
        fl_clamped = np.clip(fl, flight_levels.min(), flight_levels.max())

        # Bilinear interpolation
        interpolator = RegularGridInterpolator(
            (flight_levels, distances),
            values,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        return float(interpolator([fl_clamped, nm_clamped])[0])

    def plot_variable_map(
        self,
        variable: str,
        save_path: str | None = None,
        cmap: str = 'viridis',
        title: str | None = None,
    ):
        """
        Plot a 2D map of a weather variable along the arc.

        Parameters
        ----------
        variable : str
            Variable to plot: 'T', 'P', 'pressure_level', 'u', or 'v'.
        save_path : str, optional
            Path to save the figure
        cmap : str, optional
            Colormap to use
        title : str, optional
            Plot title
        """
        # Select the appropriate dataframe
        var_map = {
            'T': self.t_regridded,
            'P': self.p_regridded,
            'pressure_level': self.p_regridded,
            'u': self.u_regridded,
            'v': self.v_regridded,
        }

        if variable not in var_map:
            raise ValueError(f'Variable must be one of {list(var_map.keys())}')

        df = var_map[variable]
        if df is None:
            raise ValueError(f'{variable} map has not been initialized')

        self._plot_continuous_map(df, variable, save_path, cmap, title)

    def _plot_continuous_map(
        self,
        df: pd.DataFrame,
        variable: str,
        save_path: str | None,
        cmap: str,
        title: str | None,
    ):
        """Plot continuous weather variable map."""
        # Extract data
        dist_cols = [c for c in df.columns if c.startswith('NM_')]
        distances = np.array([float(c.split('_')[1]) for c in dist_cols])
        fls = df['Flight_Level'].values
        values = df[dist_cols].values

        # Plot
        plt.figure(figsize=(15, 8))
        im = plt.pcolormesh(distances, fls, values, cmap=cmap, shading='nearest')
        plt.xlim(left=max(0.0, distances.min()), right=distances.max())
        plt.xlabel('Distance Along Arc (NM)', fontsize=14)
        plt.ylabel('Flight Level', fontsize=14)

        var_labels = {
            'T': 'Temperature (K)',
            'P': 'Pressure (Pa)',
            'pressure_level': 'Pressure (Pa)',
            'u': 'U-Wind Component (m/s)',
            'v': 'V-Wind Component (m/s)',
        }
        plt.title(title or var_labels.get(variable, variable), fontsize=16)
        plt.colorbar(im, label=var_labels.get(variable, variable))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        plt.close()

    def _nearest_indices_1d(self, coord, targets):
        """
        Fast nearest-neighbor indices on a 1D grid (ascending or descending).
        coord: (n,) array of grid coords
        targets: (m,) array of query coords
        returns: (m,) int indices into coord.
        """
        coord = np.asarray(coord)
        targets = np.asarray(targets)

        if coord[0] <= coord[-1]:  # ascending
            idx = np.searchsorted(coord, targets, side='left')
            idx = np.clip(idx, 1, len(coord) - 1)
            left = coord[idx - 1]
            right = coord[idx]
            choose_right = (targets - left) > (right - targets)
            return idx - 1 + choose_right.astype(np.intp)
        else:  # descending (e.g., ERA5 latitude)
            c = coord[::-1]
            idx = np.searchsorted(c, targets, side='left')
            idx = np.clip(idx, 1, len(c) - 1)
            left = c[idx - 1]
            right = c[idx]
            choose_right = (targets - left) > (right - targets)
            idx_asc = idx - 1 + choose_right.astype(np.intp)
            return (len(coord) - 1) - idx_asc
