from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on the Earth given their latitude and longitude.
    Returns the distance in nautical miles (NM).
    """
    R = 3440.0  # Earth's radius in nautical miles
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in NM

def get_great_circle_distance(flight_data):
    """
    Computes the great-circle distance between departure and arrival airports in nautical miles.
    """
    if not flight_data or "dep_lat" not in flight_data or "arr_lat" not in flight_data:
        return "Invalid flight data."
    
    return haversine(flight_data["dep_lat"], flight_data["dep_lon"],
                     flight_data["arr_lat"], flight_data["arr_lon"])
    
    

