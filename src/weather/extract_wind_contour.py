import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from airportsdata import load
from pyproj import Geod
import numpy as np
import matplotlib.ticker as mticker
import xarray as xr
from scipy.interpolate import RegularGridInterpolator



# Return airport (ICAO) lat & lon
def get_coords(code):
    info = airports.get(code)
    return (info['lat'], info['lon']) if info else (None, None)

# --- Extract U-wind along arcs ---
def extract_u_along_arc(ds, od_pairs, npts=100):
    geod = Geod(ellps="WGS84")
    results = []
    arc_coords_all = []

    for (lon1, lat1), (lon2, lat2) in od_pairs:
        spacing_km = 25
        # Compute total arc distance in meters
        _, _, total_distance_m = geod.inv(lon1, lat1, lon2, lat2)
        
        # Compute number of intermediate points (~25 km spacing)
        npts = max(1, int(total_distance_m // (spacing_km * 1000)))
        
        # Build arc coordinates
        arc = [(lon1, lat1)] + geod.npts(lon1, lat1, lon2, lat2, npts) + [(lon2, lat2)]
        arc_coords_all.append(arc)
        u_arc = []

        # ERA-5 data
        for lon, lat in arc:
            lon = (lon + 360) % 360  # Convert to 0–360
            u_at_levels = []
            for ilev in range(len(levels)):
                interp_func = RegularGridInterpolator(
                    (lats, lons), u_data[ilev, :, :],
                    bounds_error=False, fill_value=np.nan
                )
                u_val = interp_func([[lat, lon]])[0]
                u_at_levels.append(u_val)
            u_arc.append(u_at_levels)

        results.append(np.array(u_arc))  # (npts+2, n_levels)

    return results, arc_coords_all

# --- Extract U-wind along arcs ---
def extract_v_along_arc(ds, od_pairs, npts=100):
    geod = Geod(ellps="WGS84")
    results = []
    arc_coords_all = []

    for (lon1, lat1), (lon2, lat2) in od_pairs:
        spacing_km = 25
        # Compute total arc distance in meters
        _, _, total_distance_m = geod.inv(lon1, lat1, lon2, lat2)
        
        # Compute number of intermediate points (~25 km spacing)
        npts = max(1, int(total_distance_m // (spacing_km * 1000)))
        
        # Build arc coordinates
        arc = [(lon1, lat1)] + geod.npts(lon1, lat1, lon2, lat2, npts) + [(lon2, lat2)]
        arc_coords_all.append(arc)
        v_arc = []

        # ERA-5 data
        for lon, lat in arc:
            lon = (lon + 360) % 360  # Convert to 0–360
            v_at_levels = []
            for ilev in range(len(levels)):
                interp_func = RegularGridInterpolator(
                    (lats, lons), v_data[ilev, :, :],
                    bounds_error=False, fill_value=np.nan
                )
                v_val = interp_func([[lat, lon]])[0]
                v_at_levels.append(v_val)
            v_arc.append(v_at_levels)

        results.append(np.array(v_arc))  # (npts+2, n_levels)

    return results, arc_coords_all


# --- Plot U-wind profile along arc ---
def plot_u_wind_contour(u_wind, arc_coords, pressure_levels, arc_title=None):
    geod = Geod(ellps="WGS84")
    npts = len(arc_coords)
    distances = [0.0]

    for i in range(1, npts):
        lon1, lat1 = arc_coords[i-1]
        lon2, lat2 = arc_coords[i]
        _, _, d = geod.inv(lon1, lat1, lon2, lat2)
        distances.append(distances[-1] + d / 1000)

    pressure = np.array(pressure_levels)
    altitude_m = 44330.0 * (1 - (pressure / 1013.25) ** (1 / 5.255))

    if pressure[0] < pressure[-1]:
        pressure = pressure[::-1]
        u_wind = u_wind[:, ::-1]

    #fig, ax = plt.subplots(figsize=(10, 5))
    # Create figure and axis
    fig = plt.figure()
    ax = fig.gca()
    # Filter to altitudes ≤ 15000 m
    alt_cap = 15000
    mask = altitude_m <= alt_cap
    altitude_m_cap = altitude_m[mask]
    u_wind_cap = u_wind[:, mask]
    X, Y = np.meshgrid(distances, altitude_m_cap)
    cs = ax.contourf(X, Y, u_wind_cap.T, levels=50, cmap="viridis", extend='both')
    
    #cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.1, shrink = 0.7)
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical')
    cbar.set_label("U-wind (m/s)", fontname='Times New Roman', fontsize=18)
    
    ax.set_xlabel("Ground Track Distance (km)", fontname='Times New Roman', fontsize=18)
    ax.set_ylabel("Altitude (m)", fontname='Times New Roman', fontsize=18)
    ax.set_title(arc_title or "U-wind Profile Along Arc", fontname='Times New Roman', fontsize=16)
    
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname('Times New Roman')

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_fontsize(14)
    
    
    ax.grid(False)
    
    # Adjust figure size
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    
    
    plt.tight_layout()
    plt.savefig(f"Plots/{od_name}_u_wind_contour.png", dpi=300, bbox_inches='tight')
    plt.show()
    
# --- Plot V-wind profile along arc ---
def plot_v_wind_contour(v_wind, arc_coords, pressure_levels, arc_title=None):
    geod = Geod(ellps="WGS84")
    npts = len(arc_coords)
    distances = [0.0]

    for i in range(1, npts):
        lon1, lat1 = arc_coords[i-1]
        lon2, lat2 = arc_coords[i]
        _, _, d = geod.inv(lon1, lat1, lon2, lat2)
        distances.append(distances[-1] + d / 1000)

    pressure = np.array(pressure_levels)
    altitude_m = 44330.0 * (1 - (pressure / 1013.25) ** (1 / 5.255))

    if pressure[0] < pressure[-1]:
        pressure = pressure[::-1]
        v_wind = v_wind[:, ::-1]

    #fig, ax = plt.subplots(figsize=(10, 5))
    # Create figure and axis
    fig = plt.figure()
    ax = fig.gca()
    # Filter to altitudes ≤ 15000 m
    alt_cap = 15000
    mask = altitude_m <= alt_cap
    altitude_m_cap = altitude_m[mask]
    u_wind_cap = v_wind[:, mask]
    X, Y = np.meshgrid(distances, altitude_m_cap)
    cs = ax.contourf(X, Y, u_wind_cap.T, levels=50, cmap="viridis", extend='both')
    
    #cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.1, shrink = 0.7)
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical')
    cbar.set_label("V-wind (m/s)", fontname='Times New Roman', fontsize=18)
    
    ax.set_xlabel("Ground Track Distance (km)", fontname='Times New Roman', fontsize=18)
    ax.set_ylabel("Altitude (m)", fontname='Times New Roman', fontsize=18)
    ax.set_title(arc_title or "V-wind Profile Along Arc", fontname='Times New Roman', fontsize=16)
    
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname('Times New Roman')

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_fontsize(14)
    
    
    ax.grid(False)
    
    # Adjust figure size
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    
    
    plt.tight_layout()
    plt.savefig(f"Plots/{od_name}_v_wind_contour.png", dpi=300, bbox_inches='tight')
    plt.show()

# --- Load ERA-5 U wind data ---
ds = xr.open_dataset("data.grib", engine="cfgrib") # <-- Path to your NetCDF file
ds = ds.isel(time=0)  # pick the first time slice

# --- Load O-D data ---
airports = load('IATA')
with open('sample_missions_AACES.json', 'r') as f:
    missions = json.load(f)

# Lat & lon + pressure levels
lats = ds.latitude.values
lons = ds.longitude.values
levels = ds.level.values if 'level' in ds.coords else ds['isobaricInhPa'].values

# Exract u component (raw)
u_data = ds['u'].values  # shape: (level, lat, lon)

# Exract v component (raw)
v_data = ds['v'].values  # shape: (level, lat, lon)

# Ensure latitude is increasing
if lats[0] > lats[-1]:
    lats = lats[::-1]
    u_data = u_data[:, ::-1, :]
    v_data = v_data[:, ::-1, :]
    
    
# Get lat and lon of the O-D pairs    
od_pairs = []
for m in missions:
    lat1, lon1 = get_coords(m['dep_airport'])
    lat2, lon2 = get_coords(m['arr_airport'])
    if None not in (lat1, lon1, lat2, lon2):
        od_pairs.append(((lon1, lat1), (lon2, lat2)))
        

# Discretize arc for every 25 km interval + extract u-v-component for all pressure levels    
u_wind_along_arcs, arc_coords_all = extract_u_along_arc(ds, od_pairs, npts=100)
v_wind_along_arcs, arc_coords_all = extract_v_along_arc(ds, od_pairs, npts=100)

# --- Example usage for arc 0 ---
i = 0
od_name = f"{missions[i]['dep_airport']} → {missions[i]['arr_airport']}"


# Plot u-component across multiple altutudes
plot_u_wind_contour(u_wind_along_arcs[i], arc_coords_all[i], levels, arc_title=f"{od_name}")

# Plot v-component across multiple altutudes
plot_v_wind_contour(v_wind_along_arcs[i], arc_coords_all[i], levels, arc_title=f"{od_name}")