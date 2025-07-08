import pandas as pd
import numpy as np

def evaluate_filtered_deviation_and_write_csv(df_filtered, deviation_ft=4000, output_file="deviation_report_filtered.csv"):
    """
    Evaluate ISSR segments at deviated flight levels and write results to CSV,
    only including deviated segments that start after the first segment position
    at the reference altitude.

    Parameters:
        df_filtered (pd.DataFrame): Filtered DataFrame with ISSR segments
        deviation_ft (float): Altitude deviation in feet
        output_file (str): Path to output CSV file
    """
    results = []
    
    for _, row in df_filtered.iterrows():
        fl_ref = row['Flight_Level']
        alt_ref = fl_ref * 100
        target_alt = alt_ref + deviation_ft
        
        # Find nearest FL in df_filtered
        fl_values = df_filtered['Flight_Level'].values * 100
        nearest_idx = (np.abs(fl_values - target_alt)).argmin()
        nearest_row = df_filtered.iloc[nearest_idx]
        alt_nearest = nearest_row['Flight_Level'] * 100
        
        # Extract reference and new segment lengths and positions (up to 3)
        ref_lengths = [row.get(f'Segment_{i:02d}_Length_NM', np.nan) for i in range(1, 4)]
        ref_positions = [row.get(f'Segment_{i:02d}_Start_NM', np.nan) for i in range(1, 4)]

        # First segment start position from reference altitude
        ref_threshold_pos = ref_positions[0] if not pd.isna(ref_positions[0]) else -np.inf

        # Filter new lengths to only include segments starting after reference threshold
        new_lengths_filtered = []
        for i in range(1, 4):
            pos = nearest_row.get(f'Segment_{i:02d}_Start_NM', np.nan)
            if not pd.isna(pos) and pos > ref_threshold_pos:
                new_len = nearest_row.get(f'Segment_{i:02d}_Length_NM', np.nan)
            else:
                new_len = np.nan
            new_lengths_filtered.append(new_len)

        results.append({
            "Ref Alt (ft)": alt_ref,
            "Deviation Alt (ft)": target_alt,
            "Nearest Alt (ft)": alt_nearest,
            "Ref Seg Len 1": ref_lengths[0],
            "Ref Seg Len 2": ref_lengths[1],
            "Ref Seg Len 3": ref_lengths[2],
            "Ref Start Pos 1": ref_positions[0],
            "Ref Start Pos 2": ref_positions[1],
            "Ref Start Pos 3": ref_positions[2],
            "New Seg Len 1": new_lengths_filtered[0],
            "New Seg Len 2": new_lengths_filtered[1],
            "New Seg Len 3": new_lengths_filtered[2],
        })
    
    # Create DataFrame and write to CSV
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False)
    return output_file





# Load the original CSV file
input_csv = "BOS-DAL_20241229.csv"
df = pd.read_csv(input_csv)

# Filter out rows where ISSR intersections are non-zero
filtered_df = df[df['Num_Segments'] > 0].reset_index(drop=True)

# Run this version of the function
csv_out_path = "BOS-DAL-deviation_report.csv"
evaluate_filtered_deviation_and_write_csv(filtered_df, output_file=csv_out_path)
