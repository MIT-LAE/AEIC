import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
import numpy as np


def plot_flight_arc(mission):
    
    # Set up map
    
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-130, -65, 22, 50], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    
    # Extract OD lat lon position
    lon_dep, lat_dep, _ = mission["dep_location"]
    lon_arr, lat_arr, _ = mission["arr_location"]
    
    # Use WGS84 ellipse definition
    geod = Geod(ellps="WGS84")
    
    # Get 100 equally spaced points along the arc
    points = geod.npts(lon_dep, lat_dep, lon_arr, lat_arr, 100)
    lons = [lon_dep] + [pt[0] for pt in points] + [lon_arr]
    lats = [lat_dep] + [pt[1] for pt in points] + [lat_arr]

    ax.plot(lons, lats, 'k-', linewidth=0.8, alpha=0.7)
    
    # Mark endpoints
    ax.plot(lon_dep, lat_dep, 'go', markersize=6, label="Departure")
    ax.plot(lon_arr, lat_arr, 'ro', markersize=6, label="Arrival")
    
    # Add legend and title
    ax.legend(loc='lower left')
    dep_code = mission['dep_airport']
    arr_code = mission['arr_airport']
    ax.set_title(f"Flight Arc: {dep_code} → {arr_code}")
    
    plt.tight_layout()
    plt.savefig("sample.png", dpi=300)