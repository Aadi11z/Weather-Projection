"""
fetch_future_miroc6.py - Fetch MIROC6 Future SSP Projections
=============================================================
Downloads MIROC6 daily climate data for future scenarios (2015-2100)
under selected SSP pathways from the Google Cloud CMIP6 catalog.

Available SSP Scenarios:
  - ssp126: SSP1-2.6 (Sustainability / Low Emissions)
  - ssp245: SSP2-4.5 (Middle of the Road)
  - ssp370: SSP3-7.0 (Regional Rivalry / High Emissions)
  - ssp585: SSP5-8.5 (Fossil-Fueled Development / Very High Emissions)

Usage:
  python fetch_future_miroc6.py
  
  By default fetches SSP2-4.5 and SSP5-8.5. Edit the SSP_SCENARIOS
  list below to include/exclude scenarios as needed.
"""
import intake
import xarray as xr
import pandas as pd
import numpy as np
import dask
import asyncio
import sys
import os

# Configure dask to use multithreading for faster network I/O
dask.config.set(scheduler='threads')

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─── CONFIGURATION ───────────────────────────────────────
# Select which SSP scenarios to download
# Uncomment/add the ones you need for your research papers
SSP_SCENARIOS = [
    'ssp245',   # SSP2-4.5: Middle of the Road
    # 'ssp585',   # SSP5-8.5: Fossil-Fueled (worst case)
    # 'ssp126', # SSP1-2.6: Sustainability (uncomment if needed)
    # 'ssp370', # SSP3-7.0: Regional Rivalry (uncomment if needed)
]

START_DATE = '2015-01-01'
END_DATE   = '2100-12-31'

variables = ['tas', 'pr', 'psl', 'hursmax', 'hursmin', 'uas', 'vas']

# ─── CONNECT TO CATALOG ──────────────────────────────────
print("Connecting to the Google Cloud CMIP6 catalog...")
catalog_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(catalog_url)

def fetch_miroc6_ssp(experiment_name, start_date, end_date):
    """Fetch and process MIROC6 data for a given SSP experiment."""
    print(f"\n{'='*60}")
    print(f"Fetching {experiment_name.upper()} data ({start_date} to {end_date})")
    print(f"{'='*60}")
    
    query = dict(
        source_id='MIROC6',
        experiment_id=experiment_name,
        table_id='day',
        variable_id=variables,
        member_id='r1i1p1f1'
    )
    
    search_results = col.search(**query)
    print(f"  Found {len(search_results.df)} dataset entries in catalog.")
    
    if len(search_results.df) == 0:
        print(f"  WARNING: No data found for {experiment_name}! Skipping.")
        return None
    
    dsets = search_results.to_dataset_dict(
        zarr_kwargs={'consolidated': True},
        xarray_combine_by_coords_kwargs={'compat': 'override'}
    )
    
    dataset_key = list(dsets.keys())[0]
    ds_global = dsets[dataset_key]
    
    # Spatial slicing: same 3x3 UAE bounding box as historical
    print("  Slicing to UAE 3x3 grid (lat: 22.0-26.2, lon: 52.0-56.2)...")
    ds_uae = ds_global.sel(lat=slice(22.0, 26.2), lon=slice(52.0, 56.2))
    ds_timeframe = ds_uae.sel(time=slice(start_date, end_date)).compute()
    
    df = ds_timeframe.to_dataframe().reset_index()
    return df[['time', 'lat', 'lon', 'tas', 'pr', 'psl', 'hursmax', 'hursmin', 'uas', 'vas']]


def process_and_save(df_raw, experiment_name, start_year, end_year):
    """Apply meteorological conversions and save to CSV."""
    # Unit conversions (identical to historical MIROC6.py)
    df_raw['T_avg'] = df_raw['tas'] - 273.15 
    df_raw['PCP']   = df_raw['pr'] * 86400
    df_raw['AP']    = df_raw['psl'] / 100
    df_raw['RH']    = (df_raw['hursmax'] + df_raw['hursmin']) / 2
    df_raw['WS']    = np.sqrt(df_raw['uas']**2 + df_raw['vas']**2)
    
    df_final = df_raw[['time', 'lat', 'lon', 'T_avg', 'PCP', 'AP', 'RH', 'WS']].copy()
    df_final.rename(columns={'time': 'Date', 'lat': 'Lat', 'lon': 'Lon'}, inplace=True)
    
    # Drop duplicates from hidden height dimensions
    df_final = df_final.drop_duplicates(subset=['Date', 'Lat', 'Lon'], keep='first')
    
    # Validate grid structure
    dates = df_final['Date'].unique()
    rows_per_date = df_final.groupby('Date').size()
    n_bad = (rows_per_date != 9).sum()
    
    output_filename = f"MIROC6_UAE_{experiment_name.upper()}_{start_year}_{end_year}.csv"
    df_final.to_csv(output_filename, index=False)
    
    print(f"\n  --- {experiment_name.upper()} Summary ---")
    print(f"  Total rows:     {len(df_final):,}")
    print(f"  Date range:     {df_final['Date'].min()} --> {df_final['Date'].max()}")
    print(f"  Total dates:    {len(dates):,}")
    print(f"  Grid integrity: {'ALL OK (9 rows/date)' if n_bad == 0 else f'{n_bad} dates have wrong row count!'}")
    print(f"  Saved to:       {output_filename}")
    
    return output_filename


# ─── MAIN LOOP ────────────────────────────────────────────
saved_files = []

for ssp in SSP_SCENARIOS:
    try:
        df_raw = fetch_miroc6_ssp(ssp, START_DATE, END_DATE)
        if df_raw is not None:
            fname = process_and_save(df_raw, ssp, 2015, 2100)
            saved_files.append((ssp, fname))
    except Exception as e:
        print(f"\n  ERROR fetching {ssp}: {e}")
        print("  Skipping this scenario and continuing...")
        continue

# ─── SUMMARY ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("ALL DONE")
print(f"{'='*60}")
print(f"  Successfully downloaded {len(saved_files)}/{len(SSP_SCENARIOS)} scenarios:")
for ssp, fname in saved_files:
    size_mb = os.path.getsize(fname) / (1024 * 1024)
    print(f"    {ssp.upper():>8s} --> {fname} ({size_mb:.1f} MB)")
print(f"\n  Next step: Run  python project_future.py  to downscale these projections.")

# Clean up to prevent asyncio errors at exit
try:
    import fsspec
    fsspec.clear_instance_cache()
except Exception as e:
    pass  # Harmless cleanup — data is already saved
