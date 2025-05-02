import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from airportsdata import load
from pyproj import Geod
import numpy as np
import matplotlib.ticker as mticker


# Load airport database
airports = load('IATA')

# Load missions
with open('sample_missions_AACES.json', 'r') as f:
    missions = json.load(f)

# Helper: Get coordinates
def get_coords(code):
    info = airports.get(code)
    return (info['lat'], info['lon']) if info else (None, None)

# Get O-D pairs
od_pairs = []
for m in missions:
    lat1, lon1 = get_coords(m['dep_airport'])
    lat2, lon2 = get_coords(m['arr_airport'])
    if None not in (lat1, lon1, lat2, lon2):
        od_pairs.append(((lon1, lat1), (lon2, lat2)))

# Setup CONUS map
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())  # use rectangular projection for extent clipping
ax.set_extent([-125, -66.5, 24.5, 49.5], crs=ccrs.Geodetic())  # CONUS extent

# Map features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.1)
#ax.add_feature(cfeature.LAND, facecolor='lightgray')
#ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Plot arcs
geod = Geod(ellps="WGS84")
for (lon1, lat1), (lon2, lat2) in od_pairs:
    points = geod.npts(lon1, lat1, lon2, lat2, 100)
    lons = [lon1] + [pt[0] for pt in points] + [lon2]
    lats = [lat1] + [pt[1] for pt in points] + [lat2]
    ax.plot(lons, lats, transform=ccrs.Geodetic(), color='darkred', linewidth=2, alpha=0.7)
    
    
# Add lat/lon gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.05, color='gray', alpha=1.0, linestyle='--')
gl.xlocator = mticker.MultipleLocator(0.25)
gl.ylocator = mticker.MultipleLocator(0.25)
gl.top_labels = False
gl.right_labels = False

# Only label every 10°
gl.xformatter = mticker.FuncFormatter(lambda lon, pos: f"{int(round(lon))}°" if round(lon) % 5 == 0 else "")
gl.yformatter = mticker.FuncFormatter(lambda lat, pos: f"{int(round(lat))}°" if round(lat) % 5 == 0 else "")


gl.xlabel_style = {'size': 10, 'fontname': 'Times New Roman'}
gl.ylabel_style = {'size': 10, 'fontname': 'Times New Roman'}

# Plot square markers at each O-D airport
airport_coords = set()
for (lon1, lat1), (lon2, lat2) in od_pairs:
    airport_coords.add((lon1, lat1))
    airport_coords.add((lon2, lat2))

for lon, lat in airport_coords:
    ax.plot(lon, lat, marker='s', markersize=5, color='black',
            transform=ccrs.Geodetic(), zorder=5)


plt.tight_layout()
plt.show()
