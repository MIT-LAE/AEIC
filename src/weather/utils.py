from pyproj import Geod
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

def altitude_to_pressure_hpa(alt_ft):
    """Convert altitude in feet to pressure in hPa using ISA approximation."""
    alt_m = np.array(alt_ft) * 0.3048
    pressure = 1013.25 * (1 - (0.0065 * alt_m) / 288.15) ** 5.2561
    return pressure
    

def get_wind_at_points(mission_data, era5_path):
    """
    Interpolates U and V wind components from ERA5 at each lat/lon/alt point.

    Parameters
    ----------
    mission_data : dict
        Output from get_mission_points(); must contain 'lons', 'lats', and 'H'.
    era5_path : str
        Path to ERA5 GRIB file.

    Returns
    -------
    u_vals : np.ndarray
        Zonal wind (eastward) component at each point [m/s]
    v_vals : np.ndarray
        Meridional wind (northward) component at each point [m/s]
    wind_speed : np.ndarray
        Wind speed magnitude [m/s]
    """
    
    # Load ERA4 GRIB file (single time slice)
    ds = xr.open_dataset(era5_path, engine="cfgrib")
    
    # Ensure proper dimension order
    lats_era = ds.latitude.values
    lons_era = ds.longitude.values
    levels_era = ds.isobaricInhPa.values
    
    # Sort lattitude in ascending order
    if lats_era[0] > lats_era[-1]:
        lats_era = lats_era[::-1]
        ds['u'] = ds['u'][::-1, :, :]
        ds['v'] = ds['v'][::-1, :, :]

    # Prepare 3D interpolators
    u_interp = RegularGridInterpolator(
        (levels_era, lats_era, lons_era),
        ds['u'].values,
        bounds_error=False,
        fill_value=np.nan
    )
    
    v_interp = RegularGridInterpolator(
        (levels_era, lats_era, lons_era),
        ds['v'].values,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Convert altitude (ft) to pressure (hPa)
    pressures = altitude_to_pressure_hpa(mission_data['H'])
    
    # Convert lon to [0, 360] for ERA5 grid compatibility
    lons_360 = [(lon + 360) if lon < 0 else lon for lon in mission_data['lons']]
    
    # Form (pressure, lat, lon) points for interpolation
    points = np.array([
        (p, lat, lon) for p, lat, lon in zip(pressures, mission_data['lats'], lons_360)
    ])
    
    # Interpolate
    u_vals = u_interp(points)
    v_vals = v_interp(points)
    
    wind_speed = np.sqrt(u_vals**2 + v_vals**2)
    
    return u_vals, v_vals, wind_speed

def get_tas(mission_data, era5_path):
    """
    Computes the true airspeed (TAS), heading, and drift angle along a flight path using geodesic 
    track angles and interpolated wind components from ERA5 reanalysis data.

    Parameters
    ----------
    mission_data : dict
        Dictionary containing flight path data with the following keys:
            - 'lats': array-like, latitudes of flight path points [degrees]
            - 'lons': array-like, longitudes of flight path points [degrees]
            - 'GS': array-like, ground speed at each point [knots]

    era5_path : str
        File path to the ERA5 dataset used for wind interpolation.

    Returns
    -------
    track_angles : np.ndarray
        Array of geodesic track angles (course) between consecutive lat-lon points [degrees].

    headings : np.ndarray
        Array of aircraft headings (direction of motion through the air) [degrees].

    drifts : np.ndarray
        Array of drift angles (heading - track), indicating crosswind correction required [degrees].

    tas_vals : np.ndarray
        Array of computed true airspeeds (TAS) along the path [knots].

    u_vals : np.ndarray
        Interpolated eastward wind components (u-wind) at each flight point [m/s].

    v_vals : np.ndarray
        Interpolated northward wind components (v-wind) at each flight point [m/s].

    wind_speed : np.ndarray
        Magnitude of interpolated wind speed at each point [m/s].

    Notes
    -----
    - TAS is computed using the wind triangle from the vector difference between ground speed vector and wind vector.
    - Drift angle is signed and bounded to the range [-180, 180] degrees.
    - Requires `pyproj.Geod` and NumPy.
    """

    # Get wind components based on path
    u_vals, v_vals, wind_speed = get_wind_at_points(mission_data, era5_path)
    
    # Compute geodesic track angles between consecutive flight segments
    geod = Geod(ellps="WGS84")
    lons = mission_data["lons"]
    lats = mission_data["lats"]
    gs = mission_data["GS"]
    
    track_angles = []
    headings = []
    drifts = []
    tas_vals = []
    
    # Loop over all points along the arc
    for i in range(len(lons) - 1):
        # Compute track angle for the two lat-lon pairs
        lon1, lat1 = lons[i], lats[i]
        lon2, lat2 = lons[i + 1], lats[i + 1]
        az_fwd, _, _ = geod.inv(lon1, lat1, lon2, lat2)
        track = az_fwd % 360
        track_angles.append(track)
        
    # Use wind triangle to compute heading and drift
        track_rad = np.deg2rad(track)
        vgx = gs[i] * np.cos(track_rad)  # ground speed-x along track
        vgy = gs[i] * np.sin(track_rad)  # ground speed-y along track
        
        
        # Substract wind vector (souce ERA-5 dummy weather)
        vax = vgx - u_vals[i]   # TAS x-component
        vay = vgy - v_vals[i]   # TAS y-component
        
        tas_mps = np.sqrt(vax**2 + vay**2)
        tas_knots = tas_mps / 0.514444  # m/s -> knots
        
        tas_vals.append(tas_knots)
        
        heading_rad = np.arctan2(vay, vax)
        heading_deg = np.rad2deg(heading_rad) % 360
        
        # Compute path drift
        drift = (heading_deg - track + 360) % 360
        if drift > 180:
            drift -= 360  # Convert to signed drift
        headings.append(heading_deg)
        drifts.append(drift)
        
    return (
        np.array(track_angles),
        np.array(headings),
        np.array(drifts),
        np.array(tas_vals),
        u_vals,
        v_vals,
        wind_speed
    )
    

def apply_drift_to_mission_points(mission_data, drift_angles):
    """
    Computes drifted lat/lon positions by adjusting the heading from the track angle.

    Parameters
    ----------
    mission_data : dict
        Output from get_mission_points(), containing 'lons', 'lats'.
    drift_angles : list or np.ndarray
        Drift angles in degrees for each segment.

    Returns
    -------
    drifted_lons : list
        Adjusted longitude values after drift.
    drifted_lats : list
        Adjusted latitude values after drift.
    """
    geod = Geod(ellps="WGS84")
    lons = mission_data["lons"]
    lats = mission_data["lats"]

    drifted_lons = [lons[0]]
    drifted_lats = [lats[0]]

    for i in range(len(drift_angles)):
        lon1 = drifted_lons[-1]
        lat1 = drifted_lats[-1]
        lon2 = lons[i + 1]
        lat2 = lats[i + 1]

        # Compute track azimuth and distance from original points
        fwd_azimuth, _, distance_m = geod.inv(lon1, lat1, lon2, lat2)

        # Apply drift to adjust heading
        drifted_heading = (fwd_azimuth + drift_angles[i]) % 360

        # Project new position using drifted heading and same distance
        lon_drifted, lat_drifted, _ = geod.fwd(lon1, lat1, drifted_heading, distance_m)

        drifted_lons.append(lon_drifted)
        drifted_lats.append(lat_drifted)

    return drifted_lons, drifted_lats


def build_era5_interpolators(era5_path):
    """
    Load GRIB file and build interpolators for u/v wind components.

    Parameters
    ----------
    era5_path : str
        Path to ERA5 GRIB file.

    Returns
    -------
    tuple: (u_interp, v_interp, metadata)
        Interpolators and grid metadata (lats, lons, levels).
    """
    ds = xr.open_dataset(era5_path, engine="cfgrib")

    # ERA5 longitudes are in [0, 360]; convert to [-180, 180]
    if (ds.longitude > 180).any():
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
        ds = ds.sortby("longitude")

    # Ensure latitudes are increasing
    if np.any(np.diff(ds.latitude.values) < 0):
        ds = ds.sortby("latitude")

    # Extract dimensions
    lats = ds.latitude.values
    lons = ds.longitude.values
    levels = ds.isobaricInhPa.values  # Pressure levels

    # Extract u and v wind components (assumes 3D: level x lat x lon)
    u_data = ds["u"].values
    v_data = ds["v"].values

    # Interpolator over (level, lat, lon) (Filling with NaNs for safety)
    u_interp = RegularGridInterpolator((levels, lats, lons), u_data, bounds_error=False, fill_value=4.0)
    v_interp = RegularGridInterpolator((levels, lats, lons), v_data, bounds_error=False, fill_value=4.0)

    return u_interp, v_interp, {"levels": levels, "lats": lats, "lons": lons}

def compute_ground_speed(lon, lat, lon_next, lat_next, alt_ft, tas_kts, u_interp, v_interp):
    
    """
    Computes ground speed for a single point using TAS, heading, and interpolated winds.

    Parameters
    ----------
    lon, lat : float
        Longitude and latitude of the point [degrees]
    lon_next, lat_next : float
        Next point [deg] to compute heading
    alt_ft : float
        Altitude in feet
    tas_kts : float
        True airspeed in knots
    heading_rad : float
        Aircraft heading (radians from true north)
    u_interp, v_interp : interpolator functions
        ERA-5 wind interpolators (from build_era5_interpolators)

    Returns
    -------
    gs_kts : float
        Ground speed at this point [knots]
    u, v : float
        Interpolated wind components [m/s]
    """
    
    # Convert altitude and TAS to S.I units
    pressure_level = float(altitude_to_pressure_hpa(alt_ft))  # hPa
    tas_ms = tas_kts * 0.514444                       # m/s
    
    
    # Compute heading in radians from current to next point
    geod = Geod(ellps="WGS84")
    azimuth_deg, _, _ = geod.inv(lon, lat, lon_next, lat_next)
    heading_rad = np.deg2rad(azimuth_deg)
    
    # Interpolate wind at this location
    u = u_interp([[pressure_level, lat, lon]])[0]
    v = v_interp([[pressure_level, lat, lon]])[0]
    
    
    
    # Airspeed vector in Earth frame
    u_air = tas_ms * np.cos(heading_rad)
    v_air = tas_ms * np.sin(heading_rad)
    

    
    # Ground speed = air vector + wind vector
    u_ground = u_air + u
    v_ground = v_air + v
    
    gs_ms = np.sqrt(u_ground**2 + v_ground**2)
    gs_kts = gs_ms / 0.514444
    
    return gs_kts, heading_rad, u, v
                 