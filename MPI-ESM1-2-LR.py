import intake
import xarray as xr
import pandas as pd
import numpy as np
import dask
import asyncio
import sys

# Configure dask to use multithreading for faster network I/O
dask.config.set(scheduler='threads')

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print("Connecting to the Google Cloud CMIP6 catalog...")
catalog_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(catalog_url)

variables = ['tas', 'pr', 'psl', 'hursmax', 'hursmin', 'uas', 'vas']

def fetch_mpi_data(experiment_name, start_date, end_date):
    """Fetches and slices data for the MPI-ESM model."""
    query = dict(
        source_id='MPI-ESM1-2-LR', # The German MPI Model
        experiment_id=experiment_name,
        table_id='day',
        variable_id=variables,
        member_id='r1i1p1f1'
    )
    
    search_results = col.search(**query)
    
    if len(search_results.df) == 0:
        print(f"Warning: No data found for {experiment_name}.")
        return None
        
    dsets = search_results.to_dataset_dict(
        zarr_kwargs={'consolidated': True},
        xarray_combine_by_coords_kwargs={'compat': 'override'}
    )
    
    dataset_key = list(dsets.keys())[0]
    ds_global = dsets[dataset_key]
    
    # --- SPATIAL SLICING: The 3x3 UAE Bounding Box ---
    print(f"Slicing global {experiment_name} data to the UAE 3x3 grid...")
    ds_uae = ds_global.sel(lat=slice(22.0, 26.2), lon=slice(52.0, 56.2))
    ds_timeframe = ds_uae.sel(time=slice(start_date, end_date)).compute()
    
    df = ds_timeframe.to_dataframe().reset_index()
    return df[['time', 'lat', 'lon', 'tas', 'pr', 'psl', 'hursmax', 'hursmin', 'uas', 'vas']]

def process_and_save(df_raw, filename):
    """Handles meteorological conversions and saves to CSV."""
    if df_raw is None:
        return
        
    print("Converting units to match the ERA5 baseline...")
    df = df_raw.copy()
    
    df['T_avg'] = df['tas'] - 273.15 
    df['PCP'] = df['pr'] * 86400
    df['AP'] = df['psl'] / 100
    df['RH'] = (df['hursmax'] + df['hursmin']) / 2
    df['WS'] = np.sqrt(df['uas']**2 + df['vas']**2)

    df_final = df[['time', 'lat', 'lon', 'T_avg', 'PCP', 'AP', 'RH', 'WS']]
    df_final.rename(columns={'time': 'Date', 'lat': 'Lat', 'lon': 'Lon'}, inplace=True)

    # FIX: Convert CMIP6 custom calendar formats into standard Datetime
    df_final['Date'] = pd.to_datetime(df_final['Date'].astype(str).str[:10])

    # Ensure the spatial grid remains intact while dropping dimensional duplicates
    df_final = df_final.drop_duplicates(subset=['Date', 'Lat', 'Lon'], keep='first')

    df_final.to_csv(filename, index=False)
    print(f"--> Successfully saved grid to {filename}\n")

# --- 1. Historical Baseline (1950 - 2014) ---
print("\n[Phase 1/3] Fetching Historical Data")
df_hist = fetch_mpi_data('historical', '1950-01-01', '2014-12-31')
process_and_save(df_hist, "MPI_ESM_UAE_Spatial_Historical_1950_2014.csv")

# --- 2. Middle-of-the-Road Future (2015 - 2100) ---
print("[Phase 2/3] Fetching Future SSP245 Data")
df_ssp245 = fetch_mpi_data('ssp245', '2015-01-01', '2100-12-31')
process_and_save(df_ssp245, "MPI_ESM_UAE_Spatial_SSP245_2015_2100.csv")

# --- 3. High-Emission Future (2015 - 2100) ---
print("[Phase 3/3] Fetching Future SSP585 Data")
df_ssp585 = fetch_mpi_data('ssp585', '2015-01-01', '2100-12-31')
process_and_save(df_ssp585, "MPI_ESM_UAE_Spatial_SSP585_2015_2100.csv")

print("All MPI-ESM data extraction and formatting is fully complete.")

# Clean up to prevent asyncio errors at exit
try:
    import fsspec
    fsspec.clear_instance_cache()
except Exception:
    pass