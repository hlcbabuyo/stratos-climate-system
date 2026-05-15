import requests
import pandas as pd
import os

def download_nasa_power_data(lat, lon, start_date, end_date, output_path):
    """
    Downloads daily meteorological data from NASA POWER API.
    Parameters requested:
    - T2M: Temperature at 2 Meters (C)
    - RH2M: Relative Humidity at 2 Meters (%)
    - ALLSKY_SFC_SW_DWN: All Sky Surface Shortwave Downward Irradiance (Solar Radiation) (kW-hr/m^2/day)
    - WS10M: Wind Speed at 10 Meters (m/s)
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        "parameters": "T2M,RH2M,ALLSKY_SFC_SW_DWN,WS10M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    print(f"Fetching NASA POWER data for Lat: {lat}, Lon: {lon} from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # The actual data is inside properties.parameter
    features = data.get("properties", {}).get("parameter", {})
    
    if not features:
        raise ValueError("Unexpected response format or no data available.")
        
    df = pd.DataFrame(features)
    
    # NASA POWER returns dicts with YYYYMMDD as keys. 
    # Pandas DataFrame from dict makes the dates the index.
    df = df.reset_index()
    df = df.rename(columns={
        "index": "date", 
        "T2M": "temperature_2m", 
        "RH2M": "relative_humidity_2m", 
        "ALLSKY_SFC_SW_DWN": "solar_radiation", 
        "WS10M": "wind_speed_10m"
    })
    
    # Format date
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")
    print(f"Total records downloaded: {len(df)}")

if __name__ == "__main__":
    # Misamis Oriental Coordinates
    LATITUDE = 8.5342
    LONGITUDE = 124.7554
    
    # 10 years of data
    START_DATE = "20140101"
    END_DATE = "20231231"
    
    # Determine the output path (data/raw/stratos_climate_10yr.csv)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_csv_path = os.path.join(project_root, "data", "raw", "stratos_climate_10yr.csv")
    
    download_nasa_power_data(LATITUDE, LONGITUDE, START_DATE, END_DATE, output_csv_path)
