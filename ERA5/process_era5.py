import xarray as xr
import pandas as pd
import numpy as np

print("Loading unified NetCDF files and calculating daily statistics...")

# 1. Load the REAL .nc files from the new UAE extraction folder
ds = xr.open_mfdataset("ERA5_UAE_Unified_Data/*.nc", combine='by_coords')

# 2. Convert to Pandas WITHOUT taking the spatial mean
# We keep latitude and longitude to preserve the high-resolution grid
df_hourly = ds.to_dataframe().reset_index()

# 3. Meteorological Conversions
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

# 4. Downsample Hourly to Daily (Grouped by Lat/Lon)
print("Resampling hourly data to daily metrics, preserving the spatial grid...")

# We group by the specific pixel (lat, lon) AND the day, so each pixel gets its own daily average
time_col = 'valid_time' if 'valid_time' in df_hourly.columns else 'time'

df_daily = df_hourly.groupby([pd.Grouper(key=time_col, freq='D'), 'latitude', 'longitude']).agg({
    'T_Celsius': ['mean', 'min', 'max'],
    'PCP': 'sum',
    'RH': 'mean',
    'AP': 'mean',
    'WS': 'mean'
}).reset_index()

# Flatten the multi-level column names created by the agg() function
df_daily.columns = ['Date', 'Lat', 'Lon', 'T_avg', 'T_min', 'T_max', 'PCP', 'RH', 'AP', 'WS']

# CRITICAL FIX: Handle missing data safely
# Group by pixel first so we interpolate purely through time, not across the physical map
print("Interpolating missing values safely across time...")
df_daily = df_daily.groupby(['Lat', 'Lon'], group_keys=False).apply(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)

print("\n--- ERA5 Spatiotemporal Data successfully loaded and cleaned! ---")
print(df_daily.head(10)) 

# 5. Save to CSV
# I updated the filename to end in 2014 to match the timeframe of the logs you just shared
output_filename = "UAE_ERA5_Spatial_Baseline_1977_2014.csv"
df_daily.to_csv(output_filename, index=False)
print(f"\nSaved Gridded Ground Truth data to {output_filename}")