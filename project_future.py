"""
project_future.py - Downscale Future SSP Projections (Optimized Multi-GCM)
================================================================================
Uses batched GPU inference and vectorized fuzzy lookup tables to downscale
future SSP scenarios across multiple Global Climate Models (GCMs).

Pipeline:
  1. Load specific model weights and norm stats for a given GCM.
  2. Run CNN-LSTM downscaling to generate 17x17 future grids.
  3. Apply Fuzzy UHI to Thermal (T), Moisture (RH), Friction (WS).
     *CRITICAL:* PCP and AP remain raw to preserve extremes.
  4. Run WMO 30-year TFPW-MK and SQ-MK statistical sub-period analysis.
"""
import torch
import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import CNNLSTM_Downscaler
from data_loader import ClimateDataset
from fuzzy_uhi import UHIFuzzyAdjuster
from statistical_analysis import ClimateTrendAnalyzer

# ─── CONFIG ───────────────────────────────────────────────
SEQ_LENGTH      = 14
HIDDEN_CHANNELS = 64
BATCH_SIZE      = 32
FEATURES        = ['T_avg', 'PCP', 'AP', 'RH', 'WS']
RESULTS_BASE    = "results_future"

# ─── MULTI-GCM CONFIGURATION ──────────────────────────────
GCM_CONFIGS = {
    'MIROC6': {
        'ssp245': 'MIROC6_UAE_SSP245_2015_2100.csv',
        'ssp585': 'MIROC6_UAE_SSP585_2015_2100.csv'
    },
    'MPI-ESM': {
        'ssp245': 'MPI_ESM_UAE_Spatial_SSP245_2015_2100.csv',
        'ssp585': 'MPI_ESM_UAE_Spatial_SSP585_2015_2100.csv'
    }
}

os.makedirs(RESULTS_BASE, exist_ok=True)

# ─── SHARED UTILITIES ────────────────────────────────────
def run_inference(dataset, model, device, means, stds):
    """Run the CNN-LSTM on a dataset with batched GPU inference."""
    # Collect prediction dates
    prediction_dates = []
    for idx in range(len(dataset)):
        start_idx = dataset.sequence_indices[idx]
        pred_date = dataset.valid_dates[start_idx + SEQ_LENGTH - 1]
        prediction_dates.append(pred_date)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_predictions = []
    processed = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            last_frames = output[:, -1, :, :, :].cpu().numpy()
            denorm_batch = last_frames * (stds + 1e-8) + means
            all_predictions.append(denorm_batch)
            processed += len(batch)
            if processed % 5000 < BATCH_SIZE:
                print(f"      Processed {processed}/{len(dataset)} sequences...")
    
    predictions = np.concatenate(all_predictions, axis=0)
    return predictions, prediction_dates


def build_annual_cube(daily_grids, dates, agg='mean'):
    """Aggregate daily 17x17 grids into annual cubes."""
    years = [d.year for d in dates]
    unique_yrs = sorted(set(years))
    cube = np.zeros((len(unique_yrs), 17, 17))
    for yi, yr in enumerate(unique_yrs):
        mask = [i for i, y in enumerate(years) if y == yr]
        if agg == 'mean':
            cube[yi] = daily_grids[mask].mean(axis=0)
        elif agg == 'sum':
            cube[yi] = daily_grids[mask].sum(axis=0)
    return cube, unique_yrs

def export_npy_to_spatial_csv(npy_array, prediction_dates, features_list, output_csv):
    """
    Recombines the 17x17 output grids with ERA5 Lat/Lon coordinates 
    and exports as a flat spatial CSV for GIS software.
    """
    import pandas as pd
    print(f"      Converting .npy to GIS spatial CSV: {os.path.basename(output_csv)}")
    
    T_len = len(prediction_dates)
    lats = np.linspace(22.0, 26.2, 17)
    lons = np.linspace(52.0, 56.2, 17)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    dates_rep = np.repeat(prediction_dates, 17*17)
    lats_rep = np.tile(lat_grid.flatten(), T_len)
    lons_rep = np.tile(lon_grid.flatten(), T_len)
    
    df_dict = {
        'Date': dates_rep,
        'Lat': lats_rep,
        'Lon': lons_rep
    }
    
    for c_idx, feat in enumerate(features_list):
        df_dict[feat] = npy_array[:, c_idx, :, :].reshape(-1)
        
    df = pd.DataFrame(df_dict)
    df.to_csv(output_csv, index=False)


# ─── MAIN PIPELINE ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("UAE CLIMATE FUTURE PROJECTION PIPELINE (MULTI-GCM)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Compute Device: {device}\n")

    uhi = UHIFuzzyAdjuster()
    analyzer = ClimateTrendAnalyzer()
    
    # Store all WMO results for the final cross-GCM summary
    # Format: { 'Temperature': { 'MIROC6_ssp245': [...], ... }, 'Precipitation': { ... } }
    wmo_summary = {
        'Temperature': {},
        'Precipitation': {}
    }

    for gcm_name, ssp_dict in GCM_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f" INITIALIZING GCM: {gcm_name}")
        print(f"{'='*60}")
        
        # 1. Load GCM-specific model and stats
        model_path = f"best_downscaler_{gcm_name}.pth"
        norm_stats_path = f"{gcm_name}_norm_stats.json"
        
        if not os.path.exists(model_path) or not os.path.exists(norm_stats_path):
            print(f"  [SKIP] Missing weights or stats for {gcm_name}. Did you run train.py?")
            continue
            
        print(f"  Loading model weights: {model_path}")
        print(f"  Loading normalisation: {norm_stats_path}")
        
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
            
        model = CNNLSTM_Downscaler(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # Precompute denormalization vectors
        means = np.array([norm_stats['y'][f][0] for f in FEATURES]).reshape(5, 1, 1)
        stds  = np.array([norm_stats['y'][f][1] for f in FEATURES]).reshape(5, 1, 1)

        # 2. Process each SSP Scenario
        for ssp_name, ssp_file in ssp_dict.items():
            if not os.path.exists(ssp_file):
                print(f"\n  [SKIP] Missing SSP data file: {ssp_file}")
                continue
                
            results_dir = os.path.join(RESULTS_BASE, gcm_name, ssp_name)
            os.makedirs(results_dir, exist_ok=True)
            run_id = f"{gcm_name}_{ssp_name}"
            
            print(f"\n  --- PROCESSING: {gcm_name} | {ssp_name.upper()} ---")
            
            # Step A: CNN-LSTM Inference
            print(f"  [1/3] Running batched CNN-LSTM inference...")
            dataset = ClimateDataset(ssp_file, era5_path=None, sequence_length=SEQ_LENGTH, 
                                     norm_stats=norm_stats, gcm_name=gcm_name)
            predictions, prediction_dates = run_inference(dataset, model, device, means, stds)
            
            np.save(os.path.join(results_dir, f"raw_predictions_{run_id}.npy"), predictions)
            
            # Step B: Fuzzy UHI Routing (CRITICAL)
            print(f"  [2/3] Applying Multi-Variable Urban Physics (Fuzzy UHI)...")
            print(f"        -> Adjusting: T_avg, RH, WS")
            print(f"        -> Bypassing: PCP, AP (Raw)")
            
            uhi_predictions = np.zeros_like(predictions)
            for i in range(len(predictions)):
                uhi_predictions[i] = uhi.adjust_full_grid(predictions[i])
                
            np.save(os.path.join(results_dir, f"fuzzy_adjusted_{run_id}.npy"), uhi_predictions)
            
            # Export to GIS CSV
            csv_path = os.path.join(results_dir, f"spatial_gis_export_{run_id}.csv")
            export_npy_to_spatial_csv(uhi_predictions, prediction_dates, FEATURES, csv_path)
            
            # Extract routing streams for statistical analysis
            t_idx = FEATURES.index('T_avg')
            pcp_idx = FEATURES.index('PCP')
            
            # Use FUZZY adjusted for Temperature
            annual_temp, year_list = build_annual_cube(uhi_predictions[:, t_idx, :, :], prediction_dates, agg='mean')
            # Use RAW predictions for Precipitation
            annual_pcp, _ = build_annual_cube(predictions[:, pcp_idx, :, :], prediction_dates, agg='sum')

            # Step C: WMO 30-Year Statistical Analysis
            print(f"  [3/3] Running WMO 30-Year TFPW-MK & SQ-MK Analysis...")
            
            print(f"        > Temperature (Fuzzy-Adjusted)")
            temp_results = analyzer.run_wmo_period_analysis(
                annual_cube=annual_temp, 
                year_list=year_list, 
                variable_name='Temperature', 
                gcm_name=run_id, 
                output_dir=results_dir
            )
            wmo_summary['Temperature'][run_id] = temp_results
            
            print(f"        > Precipitation (Raw)")
            pcp_results = analyzer.run_wmo_period_analysis(
                annual_cube=annual_pcp, 
                year_list=year_list, 
                variable_name='Precipitation', 
                gcm_name=run_id, 
                output_dir=results_dir
            )
            wmo_summary['Precipitation'][run_id] = pcp_results
            
            # Generate overall timeline plot for visual check
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(year_list, annual_temp.mean(axis=(1, 2)), color='orange', linewidth=2, label='UAE Mean')
            ax.plot(year_list, annual_temp[:, 12, 10], color='red', linewidth=2, label='Dubai Core')
            ax.plot(year_list, annual_temp[:, 2, 2], color='blue', linewidth=2, alpha=0.7, label='Desert Pixel [2,2]')
            ax.set_title(f'{run_id.upper()} - Annual Mean Temperature Projection (2015-2100)', fontsize=13)
            ax.set_xlabel('Year')
            ax.set_ylabel('Temperature (C)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"temp_timeline_{run_id}.png"), dpi=150)
            plt.close()

    # ─── FINAL CROSS-GCM SUMMARY ─────────────────────────────
    if wmo_summary['Temperature']:
        analyzer.print_multi_gcm_comparison(
            wmo_summary['Temperature'], 
            variable_name="Temperature (Fuzzy-Adjusted)",
            output_csv=os.path.join(RESULTS_BASE, "UHI_Temperature_WMO_Summary.csv")
        )
        analyzer.print_multi_gcm_comparison(
            wmo_summary['Precipitation'], 
            variable_name="Precipitation (Raw)",
            output_csv=os.path.join(RESULTS_BASE, "Precipitation_WMO_Summary.csv")
        )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  All outputs saved to: {RESULTS_BASE}/")
