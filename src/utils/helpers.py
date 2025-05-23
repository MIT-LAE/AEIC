from .custom_types import FloatOrNDArray


def knots_to_mps(knots: FloatOrNDArray) -> FloatOrNDArray:
    """Convert knots to meters per second

    Args:
        knots (float or numpy array): Speed in knots

    Returns:
        float or numpy array: Speed in meters per second
    """
    return knots * 0.514444


def mps_to_knots(mps: FloatOrNDArray) -> FloatOrNDArray:
    """Convert meters per second to knots

    Args:
        mps (float or numpy array): Speed in meters per second

    Returns:
        float or numpy array: Speed in knots
    """
    return mps / 0.514444


def meters_to_feet(meters: FloatOrNDArray) -> FloatOrNDArray:
    """Convert meters to feet

    Args:
        meters (float or numpy array): Length in meters

    Returns:
        float or numpy array: Length in feet
    """
    return meters * 3.28084

def feet_to_meters(ft: FloatOrNDArray) -> FloatOrNDArray:
    return ft * 0.3048
