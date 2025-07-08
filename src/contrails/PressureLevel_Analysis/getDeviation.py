import pandas as pd

def compare_with_deviated_flight_level(filtered_df, deviation_ft=4000):
    """
    For the first flight level in the filtered_df, adds deviation_ft to it and finds
    the nearest available flight level in filtered_df. Prints ISSR segment lengths and
    start positions at the deviated level.
    
    Parameters:
        filtered_df (pd.DataFrame): Filtered DataFrame of ISSR segments
        deviation_ft (float): Deviation altitude in feet
    """
    if filtered_df.empty:
        print("Empty DataFrame.")
        return

    # Step 1: Get the first flight level
    fl_ref = filtered_df.iloc[0]['Flight_Level']
    target_alt = fl_ref * 100 + deviation_ft

    # Step 2: Find the nearest flight level
    fl_values = filtered_df['Flight_Level'].values * 100  # Convert to feet
    nearest_idx = (abs(fl_values - target_alt)).argmin()
    nearest_row = filtered_df.iloc[nearest_idx]
    fl_nearest = nearest_row['Flight_Level']

    print(f"Reference FL: {fl_ref} (Altitude: {fl_ref * 100} ft)")
    print(f"Target Altitude: {target_alt:.0f} ft")
    print(f"Nearest FL: {fl_nearest} (Altitude: {fl_nearest * 100} ft)\n")

    print(f"ISSR Segments at FL{int(fl_nearest):03d}:")
    num_segments = int(nearest_row['Num_Segments'])

    for i in range(1, num_segments + 1):
        seg_len = nearest_row.get(f'Segment_{i:02d}_Length_NM')
        seg_pos = nearest_row.get(f'Segment_{i:02d}_Start_NM')
        print(f"  Segment {i}: Length = {seg_len:.1f} NM, Start = {seg_pos:.1f} NM")



# Load the original CSV file
input_csv = "BOS-DAL_20241229.csv"
df = pd.read_csv(input_csv)

# Filter out rows where ISSR intersections are non-zero
filtered_df = df[df['Num_Segments'] > 0].reset_index(drop=True)


info = compare_with_deviated_flight_level(filtered_df)

print(info)

