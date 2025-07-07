import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_avg_bin(df, month):
    
    # Filter data for aircraft type 614, SEATS > 0, PASSENGERS > 0 for a specific month
    aircraft_614_df = df[(df["AIRCRAFT_TYPE"] == 614) & (df["SEATS"] > 0) & (df["MONTH"] == month) & (df["PASSENGERS"] > 0) ].copy()
    
    # Compute load factor
    aircraft_614_df["LOAD_FACTOR"] = aircraft_614_df["PASSENGERS"] / aircraft_614_df["SEATS"]

    # Create bins
    bins = np.arange(0, aircraft_614_df["DISTANCE"].max() + 100, 100)
    labels = bins[:-1] + 50  # bin centers

    # Bin the data and compute average load factor
    aircraft_614_df["DISTANCE_BIN"] = pd.cut(aircraft_614_df["DISTANCE"], bins=bins)
    binned = aircraft_614_df.groupby("DISTANCE_BIN")["LOAD_FACTOR"].mean()
    
    
    # Create scatter plot
    fig1 = plt.figure()
    ax1 = fig1.gca()

    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False) 
    ax1.tick_params(which='major', length=10, width=1.2, direction='in')
    ax1.tick_params(which='minor', length=5, width=1.2, direction='in')
    
    
    # Plot

    plt.plot(labels[:len(binned)], binned.values, marker='o')
    plt.xlabel("Distance (nm)", fontsize=22, fontname="Times New Roman")
    plt.ylabel("Average Load Factor", fontsize=22, fontname="Times New Roman")
    
    # Plot grid properties
    ax1.grid(which='major', color='black', linestyle=':', linewidth='0.05')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='black', linestyle=':', linewidth='0.05')
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)

    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    plt.tight_layout()
    plt.savefig(f"Plots/{month}_2024_avg_binned_load_factor_vs_distance.png")
    
    
def plot_loess(df):
    
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    # Filter for specified aircraft type and valid SEATS
    filtered = df[(df["AIRCRAFT_TYPE"] == 614) & (df["SEATS"] > 0)].copy()
    filtered["LOAD_FACTOR"] = filtered["PASSENGERS"] / filtered["SEATS"]

    # Extract variables
    x = filtered["DISTANCE"]
    y = filtered["LOAD_FACTOR"]
    
    # Apply LOESS smoothing
    smoothed = lowess(y, x, frac=0.1, return_sorted=True)
    
    # Plot
    fig1 = plt.figure()
    ax1 = fig1.gca()
    
    plt.scatter(x, y, alpha=0.3, s=10, label="Raw Data")
    plt.plot(smoothed[:, 0], smoothed[:, 1], color="green", linewidth=2, label=f"LOESS Smoothed (frac={0.1})")
    plt.xlabel("Distance (nm)", fontsize=22, fontname="Times New Roman")
    plt.ylabel("Load Factor", fontsize=22, fontname="Times New Roman")

            # Modify axis tick properties
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax1.tick_params(which='major', length=10, width=1.2, direction='in')
    ax1.tick_params(which='minor', length=5, width=1.2, direction='in')
    
    # Plot grid properties
    ax1.grid(which='major', color='black', linestyle=':', linewidth='0.05')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='black', linestyle=':', linewidth='0.05')
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.tight_layout()
    
    plt.savefig('012024_loess_load_factor_vs_distance.png')
    
    
    
def plot_raw_scatter(df, month):
    # Filter data for aircraft type 614 and SEATS > 0 to avoid division by zero
    aircraft_614_df = df[(df["AIRCRAFT_TYPE"] == 614) & (df["SEATS"] > 10) & (df["PASSENGERS"] > 10) & (df["MONTH"] == month) ].copy()

    # Compute load factor
    aircraft_614_df["LOAD_FACTOR"] = aircraft_614_df["PASSENGERS"] / aircraft_614_df["SEATS"]

    # Create scatter plot
    fig1 = plt.figure()
    ax1 = fig1.gca()

    plt.scatter(aircraft_614_df["DISTANCE"], aircraft_614_df["LOAD_FACTOR"], alpha=0.5, s=10)

    plt.xlabel("Distance (nm)", fontsize=22, fontname="Times New Roman")
    plt.ylabel("Load Factor", fontsize=22, fontname="Times New Roman")


    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    
    
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False) 
    ax1.tick_params(which='major', length=10, width=1.2, direction='in')
    ax1.tick_params(which='minor', length=5, width=1.2, direction='in')
    
    # Plot grid properties
    ax1.grid(which='major', color='black', linestyle=':', linewidth='0.05')
    ax1.minorticks_on()
    ax1.grid(which='minor', color='black', linestyle=':', linewidth='0.05')

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)

    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300


    plt.tight_layout()
    plt.savefig(f"Plots/{month}_2024_load_factor_vs_distance.png")



# Load the CSV file
file_path = "data/T_T100D_SEGMENT_ALL_CARRIER.csv"
df = pd.read_csv(file_path)

month_index = 1

plot_avg_bin(df, month_index)

#plot_raw_scatter(df, month_index)

#plot_avg_bin(df)

#plot_curve_fit(df)

#plot_loess(df)