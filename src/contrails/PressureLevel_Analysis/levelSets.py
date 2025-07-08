import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Geod
import xarray as xr
import csv
from matplotlib.colors import ListedColormap, BoundaryNorm

# (Data source: Fig. 9) Rädel, G. and Shine, K.P., 2008. Radiative forcing by persistent contrails and its dependence on cruise altitudes. Journal of Geophysical Research: Atmospheres, 113(D7).
def interpolate_rf_from_altitude(alt_km, rf_df):
    """Interpolate RF (mW/m²) from altitude in km using Radel_Shine.csv data."""
    return np.interp(alt_km, rf_df["height (km)"], rf_df["RF (mW/m^2)"])


def write_issr_rf_to_csv(total_issr_by_km, total_rf_by_km, output_path="issr_rf_by_altitude.csv"):
    """
    Write ISSR extent and radiative forcing data by altitude into a CSV file.

    Parameters:
        total_issr_by_km (dict): Dictionary of {altitude_km: issr_extent_km}
        total_rf_by_km (dict): Dictionary of {altitude_km: rf_mW_per_m2}
        output_path (str): Path to save the CSV file
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Altitude_km", "ISSR_extent_km", "RF_mW_per_m2"])
        for alt in sorted(total_rf_by_km.keys()):
            writer.writerow([
                f"{alt:.2f}",
                f"{total_issr_by_km.get(alt, 0):.3f}",
                f"{total_rf_by_km.get(alt, 0):.11f}"
            ])
    print(f"Saved ISSR extent and RF data to: {output_path}")
    
    
    
def plot_issr_along_geodesic(ds, rf_df, origin, destination, fileName, valid_time_index=0):
    """
    Compute and plot ISSR intersection along the geodesic arc between two points.

    Parameters:
        ds (xarray.Dataset): ERA5 ISSR dataset
        rf_df (pd.DataFrame): Radiative forcing lookup table
        origin (tuple): (lat, lon)
        destination (tuple): (lat, lon)
        fileName (str): Basename for output
        valid_time_index (int): Time index into the dataset
        npts (int): Number of points to discretize geodesic arc
    """

    geod = Geod(ellps='WGS84')
    
    # Compute number of points at ~10 km resolution
    total_distance_km = geodesic(origin, destination).km
    npts = max(int(total_distance_km // 10) - 1, 1)
    
    
    arc_coords = geod.npts(origin[1], origin[0], destination[1], destination[0], npts)
    arc_coords.insert(0, (origin[1], origin[0]))
    arc_coords.append((destination[1], destination[0]))

    arc_lats = [lat for lon, lat in arc_coords]
    arc_lons = [lon for lon, lat in arc_coords]

    ds_t = ds.isel(valid_time=valid_time_index)

    pressure_levels = ds.pressure_level.values
    arc_spacing_km = geodesic(origin, destination).km / len(arc_coords)

    total_issr_by_km = {}
    total_rf_by_km = {}

    print("=== ISSR Segment Lengths by Altitude (based on arc sampling) ===")
    for p in pressure_levels:
        issr_flags = []

        for lat, lon in zip(arc_lats, arc_lons):
            try:
                val = ds_t['ISSR_flag'].sel(
                    pressure_level=p, latitude=lat, longitude=lon, method='nearest'
                ).values.item()
                issr_flags.append(val)
            except:
                issr_flags.append(0)

        issr_flags = np.array(issr_flags)
        issr_mask = issr_flags == 1

        segments = []
        current = []

        for i, flag in enumerate(issr_mask):
            if flag:
                current.append(i)
            else:
                if len(current) > 1:
                    segments.append(current)
                current = []
        if len(current) > 1:
            segments.append(current)

        total_len = sum((len(seg) - 1) * arc_spacing_km for seg in segments)

        alt_km = pressure_to_altitude_km(p)
        total_issr_by_km[alt_km] = total_len
        rf_per_km = interpolate_rf_from_altitude(alt_km, rf_df) * 1e-9
        total_rf_by_km[alt_km] = total_len * rf_per_km

        print(f"{alt_km:.2f} km: {total_len:.1f} km → {total_rf_by_km[alt_km]:.11f} mW/m²")

    # Save data
    write_issr_rf_to_csv(total_issr_by_km, total_rf_by_km, output_path=f"Raw/{fileName}.csv")
    #plot_rf_extent_altitude_contour(total_issr_by_km, total_rf_by_km, save_path=f"Plots/Slices/{fileName}_contour.png")




def plot_rf_extent_altitude_contour(total_issr_by_km, total_rf_by_km, save_path="rf_extent_altitude_contour.png"):
    """
    Plot 2D contour of RF (mW/m²) as a function of horizontal ISSR extent and altitude.

    Parameters:
        total_issr_by_km (dict): {altitude_km: issr_length_km}
        total_rf_by_km (dict): {altitude_km: rf_mW_per_m2}
        save_path (str): Path to save the output PNG
    """
    # Prepare data
    altitudes = np.array(sorted(total_rf_by_km.keys()))
    extents = np.array([total_issr_by_km.get(alt, 0) for alt in altitudes])
    rfs = np.array([total_rf_by_km.get(alt, 0) for alt in altitudes])

    # Create 2D grid using simple outer product model (alt vs extent)
    X, Y = np.meshgrid(np.linspace(extents.min(), extents.max(), 100),
                       np.linspace(altitudes.min(), altitudes.max(), 100))
    Z = np.interp(Y, altitudes, rfs) * np.interp(X, extents, np.ones_like(extents))

    # Plot contour
    plt.figure(figsize=(8, 6), dpi=300)
    contour = plt.contourf(X, Y, Z, levels=20, cmap='plasma')
    cbar = plt.colorbar(contour)
    cbar.set_label("Radiative Forcing (mW/m²)", fontsize=11)

    plt.xlabel("ISSR Horizontal Extent (km)", fontsize=12)
    plt.ylabel("Altitude (km)", fontsize=12)
    plt.title("2D Contour of Radiative Forcing", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved RF contour plot to: {save_path}")


def pressure_to_altitude_km(pressure_hPa):
    """Convert pressure to altitude in kilometers using the barometric formula."""
    altitude_m = 44330.0 * (1.0 - (pressure_hPa / 1013.25)**(1.0 / 5.255))
    return altitude_m / 1000.0  # meters to kilometers


def plot_issr_flag_slice_norm(ds_RHI, filename, origin, destination, valid_time_index=0):
    """
    Plot a 2D ISSR flag slice along the geodesic arc for a given origin-destination pair,
    with ground track distance normalized to [0,1]

    Parameters:
        ds_RHI (xarray.Dataset): Dataset containing 'ISSR_flag' at pressure levels
        origin, destination (tuple): (lat, lon) of start and end points
        valid_time_index (int): Index into the time dimension
        filename (str): Output plot file path
    """
    
    # Build the geodesic NOTE: Assuming 60 KM separaton for now
    geod = Geod(ellps='WGS84')
    
    # Compute geodesic length
    total_km = geodesic(origin, destination).km
    total_nm = total_km / 1.852
    
    # Compute discretization length based on x KM spacing
    npts = max(int(total_km // 60) - 1, 1)

    # Compute points along geodesic
    arc_coords = geod.npts(origin[1], origin[0], destination[1], destination[0], npts)
    
    # Add O-D coordinates 
    arc_coords.insert(0, (origin[1], origin[0]))
    arc_coords.append((destination[1], destination[0]))

    # Compute lat and lon values along arc
    arc_lats = [lat for lon, lat in arc_coords]
    arc_lons = [lon for lon, lat in arc_coords]
    
    # Arc spacing in nm
    spacing_nm = total_nm / len(arc_coords)
    arc_distances_nm = np.arange(len(arc_coords)) * spacing_nm

    # Slice the ERA5 dataset along time dimension
    ds_t = ds_RHI.isel(valid_time=valid_time_index)
    
    # Get pressure level
    pressure_levels = ds_RHI.pressure_level.values
    
    # Build distance bins
    dx = spacing_nm
    arc_edges_nm = np.concatenate([[arc_distances_nm[0] - dx / 2], arc_distances_nm + dx / 2])
    
    # Normalize distance to [0,1] ==>
    # Normaluze arc distances with total nautical miles
    arc_distances_norm = arc_distances_nm / total_nm
    
    # Construct arc edges for pcolor mesh in non-dimensional space
    dx_norm = arc_distances_norm[1] - arc_distances_norm[0]
    arc_edges_norm = np.concatenate([[arc_distances_norm[0] - dx_norm / 2], arc_distances_norm + dx_norm / 2])
    
    # Clip the arc edges norm to 0
    arc_edges_norm[0] = max(arc_edges_norm[0], 0.0)
    
    issr_matrix = []
    flight_levels = []
    
    # Consider ISSR extents that are only "limit" naitcal miles long
    limit = 100

    for p in pressure_levels:
        row = []
        for lat, lon in zip(arc_lats, arc_lons):
            try:
                val = ds_t['ISSR_flag'].sel(
                    pressure_level=p, latitude=lat, longitude=lon, method='nearest'
                ).values.item()
                row.append(val)
            except:
                row.append(np.nan)
        #issr_matrix.append(row)

        alt_ft = pressure_to_altitude_km(p) * 3280.84
        fl = int(round(alt_ft, -2) / 100)  # Convert to FL
        flight_levels.append(fl)
        
        # Compute comulative ISSR length with length >=limit nm
        
        # Replace any non values with 0
        issr_flags = np.nan_to_num(np.array(row), nan=0)
        
        # Accumulate the sum of all qualifying streaks >= limit
        length_nm = 0.0
        
        # Accumulate continous segment length while scanning
        current_streak = 0.0
        
        j = 0
        while j < len(issr_flags):
            if issr_flags[j] == 1:
                start = j
                streak_length = 0.0
                while j < len(issr_flags) and issr_flags[j] == 1:
                    seg_length = arc_edges_nm[j+1] - arc_edges_nm[j]
                    streak_length += seg_length
                    j += 1
                if streak_length < limit:
                    issr_flags[start:j] = 0
            else:
                j = j +1
         # Set the corresponding rows to zero
        issr_matrix.append(issr_flags)     
        
    issr_matrix = np.array(issr_matrix)
    flight_levels = np.array(flight_levels)
    
    # Altitudes in ft from ERA5 dataset
    altitudes_ft = np.array([20815, 23577, 26635, 30069, 34004, 36216,
                         38636, 41316, 44326, 47774, 51834])
    
    
    #flight_edges_ft = np.concatenate([[first_edge], midpoints, [last_edge]])
    flight_edges_ft = np.concatenate([altitudes_ft - 00, [altitudes_ft[-1] + 00]])
    
    # Plot
    cmap = ListedColormap(['white', 'C0'])
    norm = BoundaryNorm([0, 0.5, 1], ncolors=2)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(arc_edges_norm, flight_edges_ft, issr_matrix, cmap=cmap, norm=norm, shading='flat')
    

    # Vertical lines at 10 % from origin and destination
    plt.axvline(x=0.1, color='green', linestyle=':', linewidth=2.5, label='TOC')
    plt.axvline(x= 0.9, color='red', linestyle=':', linewidth=2.5, label='TOD')

    
    # Draw indicators for ERA5 altitudes
    #for y in altitudes_ft:
    #    plt.axhline(y=y, color='black', linewidth=1.1, zorder=1)
    
    # Draw indicators for flight edges used for bin
    for y in flight_edges_ft:
        plt.axhline(y=y, color='whitesmoke', linewidth=0.5, zorder=1)
    
    
    plt.xlabel("Arc fraction", fontsize = 22, fontname="Times New Roman")
    plt.ylabel("Altitude (ft)", fontsize = 22, fontname="Times New Roman")
    plt.xticks(np.linspace(0, 1, 11), labels=[f"{int(x*100)}%" for x in np.linspace(0,1,11)])
    plt.yticks(altitudes_ft)
    plt.xticks(fontname = "Times New Roman", fontsize  = 20)
    plt.yticks(fontname = "Times New Roman", fontsize = 20)
    
    #plt.ylim(flight_levels[0], flight_levels[-1])
    plt.legend(frameon=False, loc='upper right', prop={'size': 22, 'family': 'Times New Roman'}, ncol=2)
    
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*1.5, Size[1]*1.5, forward=True) # Set forward to True to resize window along with plot in figure.
    
    
    plt.tight_layout()
    plt.savefig(f"Plots/Slices/norm_{filename}", dpi=300)
    plt.close()


def plot_issr_flag_slice(ds_RHI, filename, origin, destination, valid_time_index=0):
    """
    Plot a 2D ISSR flag slice along the geodesic arc for a given origin-destination pair.

    Parameters:
        ds_RHI (xarray.Dataset): Dataset containing 'ISSR_flag' at pressure levels
        origin, destination (tuple): (lat, lon) of start and end points
        valid_time_index (int): Index into the time dimension
        filename (str): Output plot file path
    """
    geod = Geod(ellps='WGS84')
    total_km = geodesic(origin, destination).km
    total_nm = total_km / 1.852
    npts = max(int(total_km // 60) - 1, 1)

    # Compute points along geodesic
    arc_coords = geod.npts(origin[1], origin[0], destination[1], destination[0], npts)
    arc_coords.insert(0, (origin[1], origin[0]))
    arc_coords.append((destination[1], destination[0]))

    arc_lats = [lat for lon, lat in arc_coords]
    arc_lons = [lon for lon, lat in arc_coords]
    spacing_nm = total_nm / len(arc_coords)
    arc_distances_nm = np.arange(len(arc_coords)) * spacing_nm

    ds_t = ds_RHI.isel(valid_time=valid_time_index)
    pressure_levels = ds_RHI.pressure_level.values
    
    # Build distance bins
    dx = spacing_nm
    arc_edges_nm = np.concatenate([[arc_distances_nm[0] - dx / 2], arc_distances_nm + dx / 2])
    
    issr_matrix = []
    flight_levels = []
    limit = 100
    issr_lengths_nm = [] # Store cumulative ISSR length at each level
    issr_segment_counts = []    # Number of ISSR segments per level
    issr_segment_details = []   # List of segment lengths per level
    

    for p in pressure_levels:
        row = []
        for lat, lon in zip(arc_lats, arc_lons):
            try:
                val = ds_t['ISSR_flag'].sel(
                    pressure_level=p, latitude=lat, longitude=lon, method='nearest'
                ).values.item()
                row.append(val)
            except:
                row.append(np.nan)
        #issr_matrix.append(row)

        alt_ft = pressure_to_altitude_km(p) * 3280.84
        fl = int(round(alt_ft, -2) / 100)  # Convert to FL
        flight_levels.append(fl)
        
        # Compute comulative ISSR length with length >=limit nm
        
        # Replace any non values with 0
        issr_flags = np.nan_to_num(np.array(row), nan=0)
        
        # Accumulate the sum of all qualifying streaks >= limit
        length_nm = 0.0
        
        # Accumulate continous segment length while scanning
        current_streak = 0.0
        
        segment_lengths = []
        j = 0
        while j < len(issr_flags):
            if issr_flags[j] == 1:
                start = j
                # Initialize this streak length to zero
                streak_length = 0.0
                
                # Stay within the bounds of ISSR 
                while j < len(issr_flags) and issr_flags[j] == 1:
                    seg_length = arc_edges_nm[j+1] - arc_edges_nm[j]
                    streak_length += seg_length
                    j += 1
                if streak_length < limit:
                    issr_flags[start:j] = 0
                else:
                    segment_lengths.append(streak_length)    
            else:
                j = j + 1
                
        
         # Set the corresponding rows to zero
        issr_matrix.append(issr_flags)
        
        # Store per-level total and per-gement lengths
        issr_lengths_nm.append(np.sum(segment_lengths))
        issr_segment_counts.append(len(segment_lengths))
        issr_segment_details.append(segment_lengths)    
                
    issr_matrix = np.array(issr_matrix)
    flight_levels = np.array(flight_levels)
    
 

    # Build edges for pcolormesh
    dx = spacing_nm
    
    # Altitudes in ft from your dataset
    altitudes_ft = np.array([20815, 23577, 26635, 30069, 34004, 36216,
                         38636, 41316, 44326, 47774, 51834])
    
    
    #flight_edges_ft = np.concatenate([[first_edge], midpoints, [last_edge]])
    flight_edges_ft = np.concatenate([altitudes_ft - 00, [altitudes_ft[-1] + 00]])
    
    
    # Plot
    cmap = ListedColormap(['white', 'C0'])
    norm = BoundaryNorm([0, 0.5, 1], ncolors=2)
    
   # print("Flight Level (FL) | Total ISSR Length (NM)")
   # for fl, L in zip(flight_levels, issr_lengths_nm):
   #     print(f"FL{fl:03d}             | {L:.1f}")

    
    # Determine max number of segments across all levels
    max_segs = max(len(segments) for segments in issr_segment_details)
    
    # Header row
    header = f"{'FL':>6} | {'#Segs':>6} | {'Total(NM)':>10} | " + " | ".join([f"Seg{i+1}" for i in range(max_segs)])
    print("\nPer-Level ISSR Segment Report (One Segment Per Column)")
    print("-" * (len(header) + 5))
    print(header)
    print("-" * (len(header) + 5))
    
    # Data rows
    for fl, seg_count, total_len, segments in zip(flight_levels, issr_segment_counts, issr_lengths_nm, issr_segment_details):
        # Fill missing segments with blanks for alignment
        padded_segments = [f"{s:.1f}" for s in segments] + [""] * (max_segs - len(segments))
        segment_cols = " | ".join(f"{s:>6}" for s in padded_segments)
        print(f"FL{fl:03d} | {seg_count:6d} | {total_len:10.1f} | {segment_cols}")
    
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(arc_edges_nm, flight_edges_ft, issr_matrix, cmap=cmap, norm=norm, shading='flat')
    

    # Vertical lines at 250 NM from origin and destination
    plt.axvline(x=250, color='red', linestyle='--', linewidth=1.2, label='250 NM from origin')
    plt.axvline(x=total_nm - 250, color='green', linestyle='--', linewidth=1.2, label='250 NM from destination')

    
    # Draw indicators for flight edges used for bin
    for y in flight_edges_ft:
        plt.axhline(y=y, color='gray', linewidth=1.1, zorder=1)
    
    
    plt.xlabel("Distance Along Arc (NM)", fontsize = 22, fontname="Times New Roman")
    plt.ylabel("Altitude (ft)", fontsize = 22, fontname="Times New Roman")
    plt.xticks(np.arange(0, total_nm + 1, 50))
    plt.yticks(altitudes_ft)
    plt.xticks(fontname = "Times New Roman", fontsize  = 20)
    plt.yticks(fontname = "Times New Roman", fontsize = 20)
    
    plt.legend(frameon=False, loc='upper right', prop={'size': 22, 'family': 'Times New Roman'}, ncol=2)
    
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*1.5, Size[1]*1.5, forward=True) # Set forward to True to resize window along with plot in figure.
   
    plt.tight_layout()
    plt.savefig(f"Plots/Slices/{filename}", dpi=300)
    plt.close()




# === RUN SCRIPT ===
fileName = "20241229"
file_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/PROCESSED/12_2024/RHi_{fileName}.nc"
rf_table_path = "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/PressureLevel_Analysis/Radel_Shine.csv"

# Get R.F per unit kilometer (incorrect for now, fix later)
rf_df = pd.read_csv(rf_table_path)

# Get RHI contours from post-processed ERA-5 data
ds_RHI = xr.open_dataset(file_path)

# Fix to just one arc for now
fileName = f"BOS-DAL_{fileName}.png"
origin = (42.3656, -71.0096)    # Boston
destination = (32.8968, -97.0370)  # Dallas
#destination = (34.0522, -118.2437)  # Los Angeles
#destination = (47.4502, -122.3088)  # Seattle
#destination = (25.7933, -80.2906)  # Miami


#plot_issr_flag_slice_norm(ds_RHI, fileName, origin=origin, destination=destination)

plot_issr_flag_slice(ds_RHI, fileName, origin=origin, destination=destination)


#plot_issr_along_geodesic(ds_RHI, rf_df, origin, destination, fileName2, valid_time_index=0)




