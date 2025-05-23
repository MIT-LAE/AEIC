from pyproj import Geod
import numpy as np



def get_mission_points(mission):
    """
    Generates a discretized set of latitude and longitude points along a great-circle path
    between departure and arrival locations, assuming constant cruise altitude and ground speed.

    Parameters
    ----------
    mission : dict
        Dictionary containing origin and destination coordinates with the following keys:
            - 'dep_location': tuple of (longitude, latitude, altitude) for the departure point [degrees, degrees, feet]
            - 'arr_location': tuple of (longitude, latitude, altitude) for the arrival point [degrees, degrees, feet]

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
    - The path is discretized using 100 intermediate points (plus endpoints), resulting in 102 total waypoints.
    - This function uses `pyproj.Geod` with the WGS84 ellipsoid to compute a geodesic path.
    - All values for speed and altitude are placeholders; replace them with mission-specific data in actual use.
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
    
    