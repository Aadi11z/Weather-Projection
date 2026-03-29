import os
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np

# 1. Define the Timeframe
start_year = 1950
end_year = 1989

# 2. Define the Location (The optimized UAE bounding box)
# This perfectly matches the 17x17 ERA5 grid and 3x3 MIROC6 grid
uae_area = [26.2, 52.0, 22.0, 56.2] 

# Force the library to look in your current working directory
os.environ['CDSAPI_RC'] = os.path.join(os.getcwd(), '.cdsapirc')

print("Connecting to the Copernicus Climate Data Store (CDS)...")
c = cdsapi.Client()

# 3. Fetch the Hourly Data Month-by-Month
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        month_str = f"{month:02d}"
        output_nc = f"ERA5_UAE_{year}_{month_str}.nc"
        
        # Skip downloading if the file already exists on your hard drive
        if not os.path.exists(output_nc):
            print(f"Downloading ERA5 data for {year}-{month_str}...")
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        '2m_temperature',          
                        'total_precipitation',     
                        'surface_pressure',        
                        '2m_dewpoint_temperature', 
                        '10m_u_component_of_wind', 
                        '10m_v_component_of_wind', 
                    ],
                    'year': str(year),
                    'month': month_str, 
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': [f"{h:02d}:00" for h in range(24)],
                    'area': uae_area,
                },
                output_nc
            )
            
# 4. Process and Clean the Data
print("Loading downloaded files and calculating daily statistics...")
ds = xr.open_mfdataset("ERA5_UAE_*.nc", combine='by_coords')

# Convert to Pandas DataFrame directly to preserve the Lat/Lon grid!
df_hourly = ds.to_dataframe().reset_index()

# 5. Meteorological Conversions
T_kelvin = df_hourly['t2m']
Td_kelvin = df_hourly['d2m']
u_wind = df_hourly['u10']
v_wind = df_hourly['v10']

df_hourly['T_Celsius'] = T_kelvin - 273.15
df_hourly['RH'] = 100 * (np.exp((17.625 * (Td_kelvin - 273.15)) / (243.04 + (Td_kelvin - 273.15))) / 
                         np.exp((17.625 * (T_kelvin - 273.15)) / (243.04 + (T_kelvin - 273.15))))
df_hourly['WS'] = np.sqrt(u_wind**2 + v_wind**2) 
df_hourly['PCP'] = df_hourly['tp'] * 1000 
df_hourly['AP'] = df_hourly['sp'] / 100 

# 6. Downsample Hourly Data to Daily Data (Preserving the Spatial Grid)
print("Resampling hourly data to daily metrics for the CNN...")
time_col = 'valid_time' if 'valid_time' in df_hourly.columns else 'time'

# Group by the specific pixel (Lat, Lon) AND the day
df_daily = df_hourly.groupby([pd.Grouper(key=time_col, freq='D'), 'latitude', 'longitude']).agg({
    'T_Celsius': ['mean', 'min', 'max'],
    'PCP': 'sum',
    'RH': 'mean',
    'AP': 'mean',
    'WS': 'mean'
}).reset_index()

# Flatten the multi-level column names
df_daily.columns = ['Date', 'Lat', 'Lon', 'T_avg', 'T_min', 'T_max', 'PCP', 'RH', 'AP', 'WS']

# Handle rare missing data by grouping by pixel first so we don't blur coordinates together
print("Interpolating any missing values...")
df_daily = df_daily.groupby(['Lat', 'Lon'], group_keys=False).apply(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)

# 7. View the Results and Save
print("\n--- ERA5 Spatiotemporal Data successfully loaded and cleaned! ---")
print(df_daily.head(10)) 

output_filename = "UAE_ERA5_Spatial_Baseline_1950_2014.csv"
df_daily.to_csv(output_filename, index=False) # index=False prevents writing the old index numbers
print(f"\nSaved Gridded Ground Truth data to {output_filename}")