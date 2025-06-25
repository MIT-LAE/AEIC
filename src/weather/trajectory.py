import numpy as np
from pyproj import Geod


def get_mission_points(mission):
    """
    Generates a discretized set of latitude and longitude points along a great-circle
    path between departure and arrival locations, assuming constant cruise altitude
    and ground speed.

    Parameters
    ----------
    mission : dict
        Dictionary containing origin and destination coordinates
        with the following keys:
            - 'dep_location': tuple of (longitude, latitude, altitude) for the
                departure point [degrees, degrees, feet]
            - 'arr_location': tuple of (longitude, latitude, altitude) for the
                arrival point [degrees, degrees, feet]

    Returns
    -------
    dict
        A dictionary containing:
            - 'lons' : list of longitudes along the flight path [degrees]
            - 'lats' : list of latitudes along the flight path [degrees]
            - 'GS'   : list of assumed constant ground speed at each point [knots]
            - 'H'    : list of assumed constant cruise altitude at each point [feet]

    Notes
    -----
    - The path is discretized using 100 intermediate points (plus endpoints),
        resulting in 102 total waypoints.
    - This function uses `pyproj.Geod` with the WGS84 ellipsoid to compute
        a geodesic path.
    - All values for speed and altitude are placeholders; replace them with
        mission-specific data in actual use.
    """

    # Instantiate WGS84 ellipsoid
    geod = Geod(ellps ="WGS84")

    # Extract OD lat-lon
    lon_dep, lat_dep, _ = mission["dep_location"]
    lon_arr, lat_arr, _ = mission["arr_location"]

    # Assume 100 points for discretization (for demo only)
    # Note: This will change when flying actual missions

    points = geod.npts(lon_dep, lat_dep, lon_arr, lat_arr, 100)

    lons = [lon_dep] + [pt[0] for pt in points] + [lon_arr]
    lats = [lat_dep] + [pt[1] for pt in points] + [lat_arr]

    # Assign a dummy ground speed
    ground_speeds = [450] * len(lons)

    # Assign a dummy cruise altitude
    altitude_ft = [35000] * len(lons)

    return {
        "lons": lons,
        "lats": lats,
        "GS": ground_speeds,
        "H": altitude_ft
    }

def create_dummy_traj(mission):
    """
    Generates a discretized trapezoidal trajectory profile (altitude and speed)
    along a geodesic path between departure and arrival locations.

    Parameters
    ----------
    mission : dict
        Dictionary with:
            - 'dep_location': (longitude, latitude, altitude) of departure
                            [degrees, degrees, feet]
            - 'arr_location': (longitude, latitude, altitude) of arrival
                            [degrees, degrees, feet]

    Returns
    -------
    dict
        Dictionary containing:
            - 'lons': list of longitudes [degrees]
            - 'lats': list of latitudes [degrees]
            - 'TAS'  : list of True Air Speed [knots]
            - 'H'   : list of altitudes [feet]
    """
    # Instantiate GGS84 ellipsoid for geodesic calculation
    geod = Geod(ellps="WGS84")

    # Extract dept + Arrival lat-lon
    lon_dep, lat_dep, _ = mission["dep_location"]
    lon_arr, lat_arr, _ = mission["arr_location"]

    # Assume 100 points for trajectory discretization (demo only)
    n_total = 100
    points = geod.npts(lon_dep, lat_dep, lon_arr, lat_arr, n_total)
    lons = [lon_dep] + [pt[0] for pt in points] + [lon_arr]
    lats = [lat_dep] + [pt[1] for pt in points] + [lat_arr]

    # Define climb and descent length as 25 % of total trajectory
    n_climb = n_descent = n_total // 4
    n_cruise = len(lons) - n_climb - n_descent

    # Define end points for speed and altitude
    alt_start, alt_cruise = 1000, 35000  # feet
    spd_start, spd_cruise = 140, 450 # knots

    # Define climb profile
    climb_alt = np.linspace(alt_start, alt_cruise, n_climb)
    climb_spd = np.linspace(spd_start, spd_cruise, n_climb)

    # Define cruise profile
    cruise_alt = np.full(n_cruise, alt_cruise)
    cruise_spd = np.full(n_cruise, spd_cruise)

    # Define descent profile
    descent_alt = np.linspace(alt_cruise, alt_start, n_descent)
    descent_spd = np.linspace(spd_cruise, spd_start, n_descent)



    # Collect height and TAS
    H = np.concatenate([climb_alt, cruise_alt, descent_alt])
    TAS = np.concatenate([climb_spd, cruise_spd, descent_spd])

    H = H[:len(lons)]
    TAS = TAS[:len(lons)]

    return {
        "lons": lons,
        "lats": lats,
        "TAS": TAS,
        "H": H
    }

