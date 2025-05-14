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


def inspect_grib_fields(grib_file_path):
    """
    Reads a GRIB file and prints available data fields, dimensions, and attributes.

    Parameters:
    - grib_file_path: Path to the .grib or .grib2 file (e.g., 'data/era5_u_v.grib')
    """
    import xarray as xr
    import numpy as np

    print(f"Opening GRIB file: {grib_file_path}")
    try:
        ds = xr.open_dataset(grib_file_path, engine="cfgrib")
    except Exception as e:
        print("❌ Failed to open GRIB file:", e)
        return

    print("\n✅ Successfully loaded dataset.\n")

    print("📌 **Data Variables:**")
    for var in ds.data_vars:
        print(f"  - {var}: {ds[var].attrs.get('long_name', '')} | shape = {ds[var].shape}")

    print("\n📏 **Dimensions:**")
    for dim in ds.dims:
        print(f"  - {dim}: size = {ds.dims[dim]}")

    print("\n🧭 **Coordinates:**")
    for coord in ds.coords:
        values = ds[coord].values
        if np.isscalar(values) or values.ndim == 0:
            print(f"  - {coord}: scalar value = {values}")
        else:
            preview = values[:min(3, len(values))]  # show up to 3 values
            print(f"  - {coord}: values = {preview}... (total: {ds[coord].size})")

    print("\n📝 **Global Attributes:**")
    for key, value in ds.attrs.items():
        print(f"  - {key}: {value}")

    print("\n📋 Dataset Summary:")
    print(ds)


import xarray as xr
import numpy as np
from pyproj import Geod
import os

def write_uv_profile_to_netcdf(i, missions, arc_coords_all, u_wind_along_arcs, v_wind_along_arcs, levels, output_dir="Profiles"):
    """
    Writes U, V wind components and ground track distance for mission index `i` to a NetCDF file.
    
    Parameters:
    - i: index of the mission in `missions`
    - missions: list of mission dictionaries
    - arc_coords_all: list of arc coordinates for all missions
    - u_wind_along_arcs: list of U-wind arrays (npts, nlevels) for each arc
    - v_wind_along_arcs: list of V-wind arrays (npts, nlevels) for each arc
    - levels: 1D array of pressure levels
    - output_dir: directory to save the NetCDF file
    """
    geod = Geod(ellps="WGS84")
    arc_coords = arc_coords_all[i]
    npts = len(arc_coords)
    
    # Compute ground track distances (km)
    distances = [0.0]
    for j in range(1, npts):
        lon1, lat1 = arc_coords[j - 1]
        lon2, lat2 = arc_coords[j]
        _, _, d = geod.inv(lon1, lat1, lon2, lat2)
        distances.append(distances[-1] + d / 1000)

    distances = np.array(distances)  # (npts,)
    u_wind = np.array(u_wind_along_arcs[i])  # shape (npts, nlevels)
    v_wind = np.array(v_wind_along_arcs[i])  # shape (npts, nlevels)

    # Create xarray Dataset
    ds_out = xr.Dataset(
        {
            "u_component": (("distance_km", "pressure_hPa"), u_wind),
            "v_component": (("distance_km", "pressure_hPa"), v_wind),
        },
        coords={
            "distance_km": distances,
            "pressure_hPa": levels
        },
        attrs={
            "description": "U and V wind components along great-circle mission trajectory",
            "mission": f"{missions[i]['dep_airport']} → {missions[i]['arr_airport']}"
        }
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{missions[i]['dep_airport']}_{missions[i]['arr_airport']}_wind_profile.nc"
    filepath = os.path.join(output_dir, filename)

    # Write to NetCDF
    ds_out.to_netcdf(filepath)
    print(f"✅ Written wind profile to: {filepath}")



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


def plot_global_wind_magnitude(u_data, v_data, lons, lats, level_index=0, title=None):
    """
    Plots wind magnitude over the globe centered on CONUS using orthographic projection.
    
    Parameters:
    - u_data: ndarray of shape (levels, lat, lon), u-wind component
    - v_data: ndarray of shape (levels, lat, lon), v-wind component
    - lons: 1D array of longitudes (0–360 or -180–180)
    - lats: 1D array of latitudes
    - level_index: integer index for pressure level to plot
    - title: optional string for the plot title
    """
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    import matplotlib.ticker as mticker
    import cartopy.mpl.ticker as cticker

    # Compute wind magnitude and mask invalids
    wind_magnitude = np.sqrt(u_data[level_index]**2 + v_data[level_index]**2)
    wind_magnitude = ma.masked_invalid(wind_magnitude)

    # Convert lons to [-180, 180] and sort
    lons_wrapped = np.where(lons > 180, lons - 360, lons)
    sort_idx = np.argsort(lons_wrapped)
    lons_sorted = lons_wrapped[sort_idx]
    wind_magnitude_sorted = wind_magnitude[:, sort_idx]

    # Add cyclic point
    wind_cyclic, lon_cyclic = add_cyclic_point(wind_magnitude_sorted, coord=lons_sorted)
    lon_grid, lat_grid = np.meshgrid(lon_cyclic, lats)

    # Set up figure and projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-95.0, central_latitude=37.5))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    # Contour plot of wind magnitude
    cs = ax.contourf(
        lon_grid, lat_grid, wind_cyclic,
        transform=ccrs.PlateCarree(),
        cmap="viridis", levels=100, extend='both'
    )

    # Colorbar
    cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label("Wind Speed Magnitude (m/s)", fontname='Times New Roman', fontsize=16)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_fontsize(14)

    # Gridlines with labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 15))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    # Title
    if title:
        plt.title(title, fontsize=14, fontname='Times New Roman')

    # Save and show
    plt.tight_layout()
    plt.savefig("Plots/wind_field_magnitude.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_global_relative_humidity(rh_data, lons, lats, level_index=0, title=None):
    """
    Plots relative humidity over the globe centered on CONUS using orthographic projection.
    
    Parameters:
    - rh_data: ndarray of shape (levels, lat, lon), relative humidity (%)
    - lons: 1D array of longitudes (0–360 or -180–180)
    - lats: 1D array of latitudes
    - level_index: integer index for pressure level to plot
    - title: optional string for the plot title
    """
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    import matplotlib.ticker as mticker
    import cartopy.mpl.ticker as cticker

    # Mask invalid values
    rh_level = rh_data[level_index]
    rh_level = ma.masked_invalid(rh_level)

    # Convert longitudes from [0, 360] to [-180, 180] and sort
    lons_wrapped = np.where(lons > 180, lons - 360, lons)
    sort_idx = np.argsort(lons_wrapped)
    lons_sorted = lons_wrapped[sort_idx]
    rh_sorted = rh_level[:, sort_idx]

    # Add cyclic point for smooth wraparound
    rh_cyclic, lon_cyclic = add_cyclic_point(rh_sorted, coord=lons_sorted)
    lon_grid, lat_grid = np.meshgrid(lon_cyclic, lats)

    # Set up figure and projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-95.0, central_latitude=37.5))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    # Contour plot
    cs = ax.contourf(
        lon_grid, lat_grid, rh_cyclic,
        transform=ccrs.PlateCarree(),
        cmap="YlGnBu", levels=np.linspace(0, 100, 21), extend='both'
    )

    # Colorbar
    cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label("Relative Humidity (%)", fontname='Times New Roman', fontsize=16)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_fontsize(14)

    # Gridlines with labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 15))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    # Title
    if title:
        plt.title(title, fontsize=14, fontname='Times New Roman')

    # Save and show
    plt.tight_layout()
    plt.savefig("Plots/relative_humidity.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_global_temperature(temp_data, lons, lats, level_index=0, title=None):
    """
    Plots global temperature at a given pressure level using orthographic projection centered on CONUS.

    Parameters:
    - temp_data: ndarray of shape (levels, lat, lon), temperature data (K)
    - lons: 1D array of longitudes (0–360 or -180–180)
    - lats: 1D array of latitudes
    - level_index: integer index for pressure level to plot
    - title: optional string for the plot title
    """
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    import matplotlib.ticker as mticker
    import cartopy.mpl.ticker as cticker

    # Mask invalid values
    temp_level = temp_data[level_index]
    temp_level = ma.masked_invalid(temp_level)

    # Convert longitudes to [-180, 180] and sort
    lons_wrapped = np.where(lons > 180, lons - 360, lons)
    sort_idx = np.argsort(lons_wrapped)
    lons_sorted = lons_wrapped[sort_idx]
    temp_sorted = temp_level[:, sort_idx]

    # Add cyclic point for wraparound
    temp_cyclic, lon_cyclic = add_cyclic_point(temp_sorted, coord=lons_sorted)
    lon_grid, lat_grid = np.meshgrid(lon_cyclic, lats)

    # Set up figure and map projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-95.0, central_latitude=37.5))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    # Contour plot
    cs = ax.contourf(
        lon_grid, lat_grid, temp_cyclic,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm", levels=np.linspace(np.min(temp_cyclic), np.max(temp_cyclic), 100), extend='both'
    )

    # Colorbar
    cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label("Temperature (K)", fontname='Times New Roman', fontsize=16)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_fontsize(14)

    # Gridlines with labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 15))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    # Title
    if title:
        plt.title(title, fontsize=14, fontname='Times New Roman')

    # Save and show
    plt.tight_layout()
    plt.savefig("Plots/global_temperature.png", dpi=300, bbox_inches='tight')
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
#plot_u_wind_contour(u_wind_along_arcs[i], arc_coords_all[i], levels, arc_title=f"{od_name}")

# Plot v-component across multiple altutudes
#plot_v_wind_contour(v_wind_along_arcs[i], arc_coords_all[i], levels, arc_title=f"{od_name}")


# Plot global wind field magnitude
#plot_global_wind_magnitude(u_data, v_data, lons, lats, level_index=10)


# Relative humidity
#plot_global_relative_humidity(rh_data=ds['r'].values, lons=ds.longitude.values, lats=ds.latitude.values,
#                              level_index=5, title="Global Relative Humidity at Level 5")


# Temperature
#plot_global_temperature(temp_data=ds['t'].values,
#                        lons=ds.longitude.values,
#                        lats=ds.latitude.values,
#                        level_index=3,
#                        title="Global Temperature at Level 3 (ERA5)")


#inspect_grib_fields("data.grib")


write_uv_profile_to_netcdf(
    i=4,
    missions=missions,
    arc_coords_all=arc_coords_all,
    u_wind_along_arcs=u_wind_along_arcs,
    v_wind_along_arcs=v_wind_along_arcs,
    levels=levels
)
