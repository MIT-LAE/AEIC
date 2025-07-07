import pandas as pd

def filter_south_asia_flights(csv_path, output_path):
    """
    Filters flights where both departure and arrival countries are in South Asia,
    and writes the result to a CSV.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the filtered CSV.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    south_asian_countries = {'IN', 'PK', 'AF', 'LK', 'BD', 'NP'}
    
     # Common IATA aircraft codes for Boeing 737 variants
    #b737_codes = {'733', '734', '735', '736', '737', '738', '739',
    #              '73G', '73H', '73J', '73K', '73L', '73M', '73P', '73W'}
    
    
    b737_codes = {'738'}
    

    
    df = pd.read_csv(csv_path)
    
    filtered_df = df[
        df['depctry'].isin(south_asian_countries) &
        df['arrctry'].isin(south_asian_countries) &
        df['inpacft'].isin(b737_codes)
    ]
    
    #filtered_df = df[
    #    df['depctry'].isin(south_asian_countries) &
    #    df['arrctry'].isin(south_asian_countries)
    #]
    
   
    
    filtered_df.to_csv(output_path, index=False)
    
    return filtered_df



filtered_df = filter_south_asia_flights(
    "data/OAG_2024_processed_AACES_2050.csv",
    "data/south_asia_flights.csv"
)