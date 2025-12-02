from pyproj import Geod

from AEIC.utils.types import FloatOrNDArray

GEOD = Geod(ellps="WGS84")
"""WGS84 ellipsoid geodetic calculator."""


def great_circle_distance(
    lat1: FloatOrNDArray,
    lon1: FloatOrNDArray,
    lat2: FloatOrNDArray,
    lon2: FloatOrNDArray,
    degrees: bool = False,
):
    """Calculates the great circle distance between two points. **Note that the
    latitude and longitude inputs are in radians by default; set degrees=True
    if using degrees.**

    Args:
        lat1 (Union[NDArray,float]): latitude of the first point in radians
        lon1 (Union[NDArray,float]): longitude of the first point in radians
        lat2 (Union[NDArray,float]): latitude of the second point in radians
        lon2 (Union[NDArray,float]): longitude of the second point in radians
        degrees (bool, optional): If True, the input coordinates are in degrees.

    Returns:
        Union[NDArray,float]: great circle distance between the two points in meters

    """
    return GEOD.inv(lon1, lat1, lon2, lat2, radians=not degrees)[2]
