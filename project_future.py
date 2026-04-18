"""
project_future.py - Downscale Future SSP Projections (Optimized)
=================================================================
Uses batched GPU inference and vectorized fuzzy lookup tables
for fast processing of future MIROC6 SSP scenarios.

Prerequisites:
  - best_downscaler_model.pth  (from train.py)
  - norm_stats.json            (from training data_loader)
  - MIROC6_UAE_SSP*_2015_2100.csv files (from fetch_future_miroc6.py)
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
MODEL_PATH      = "best_downscaler_model.pth"
NORM_STATS      = "norm_stats.json"
SEQ_LENGTH      = 14
HIDDEN_CHANNELS = 64
BATCH_SIZE      = 32
FEATURES        = ['T_avg', 'PCP', 'AP', 'RH', 'WS']
RESULTS_BASE    = "results_future"

# ─── DETECT AVAILABLE SSP DATASETS ───────────────────────
ssp_files = sorted(glob.glob("MIROC6_UAE_SSP*_2015_2100.csv"))

if not ssp_files:
    print("ERROR: No future SSP datasets found!")
    print("  Expected files like: MIROC6_UAE_SSP245_2015_2100.csv")
    print("  Run  python fetch_future_miroc6.py  first.")
    exit(1)

print("=" * 60)
print("UAE CLIMATE FUTURE PROJECTION PIPELINE (OPTIMIZED)")
print("=" * 60)
print(f"  Found {len(ssp_files)} SSP scenario(s):")
for f in ssp_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"    {f} ({size_mb:.1f} MB)")

# ─── LOAD MODEL ──────────────────────────────────────────
print(f"\n  Loading model from {MODEL_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

with open(NORM_STATS, 'r') as f:
    norm_stats = json.load(f)

model = CNNLSTM_Downscaler(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("  Model loaded successfully.")

# Precompute denormalization vectors
means = np.array([norm_stats['y'][f][0] for f in FEATURES]).reshape(5, 1, 1)
stds  = np.array([norm_stats['y'][f][1] for f in FEATURES]).reshape(5, 1, 1)

# ─── SHARED UTILITIES ────────────────────────────────────
def run_inference(dataset, model, device):
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


# Latitude/Longitude tick labels for the 17x17 UAE grid
_LAT_TICKS = [0, 4, 8, 12, 16]
_LAT_LABELS = ['22.0°N', '23.0°N', '24.0°N', '25.0°N', '26.0°N']
_LON_TICKS = [0, 4, 8, 12, 16]
_LON_LABELS = ['52.0°E', '53.0°E', '54.0°E', '55.0°E', '56.0°E']

def _apply_geo_axes(ax):
    """Apply geographic tick labels to a 17x17 heatmap axis."""
    ax.set_yticks(_LAT_TICKS)
    ax.set_yticklabels(_LAT_LABELS)
    ax.set_xticks(_LON_TICKS)
    ax.set_xticklabels(_LON_LABELS)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

def generate_plots(ssp_name, results_dir, predictions, prediction_dates,
                   uhi_adjusted, analyzer):
    """Generate all MK/SQ-MK plots and save results for one SSP scenario."""
    
    # --- Temperature Analysis (UHI-adjusted) ---
    print(f"\n    [Temperature] Running MK spatial analysis...")
    annual_temp, year_list = build_annual_cube(uhi_adjusted, prediction_dates, agg='mean')
    z_temp, p_temp = analyzer.analyze_spatial_grid(annual_temp)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{ssp_name.upper()} - Temperature Trend Analysis (2015-2100)', fontsize=14)
    
    im1 = axes[0].imshow(z_temp, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[0].set_title('Z-statistic (MK Test)')
    _apply_geo_axes(axes[0])
    plt.colorbar(im1, ax=axes[0], label='Z-stat (+ve = warming)')
    
    im2 = axes[1].imshow(p_temp, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=0.1, origin='lower')
    axes[1].set_title('p-value (MK Test)')
    _apply_geo_axes(axes[1])
    plt.colorbar(im2, ax=axes[1], label='p-value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"mk_temperature_{ssp_name}.png"), dpi=150)
    plt.close()
    
    # Dubai core MK
    dubai_ts = annual_temp[:, 12, 10]
    mk_temp = analyzer.run_mk_test(dubai_ts)
    print(f"    Dubai Core: trend={mk_temp['trend']}, Z={mk_temp['z_stat']:.3f}, "
          f"p={mk_temp['p_value']:.4f}, slope={mk_temp['slope']:.4f} C/year")
    
    # SQ-MK Temperature
    sqmk_temp = analyzer.run_sqmk_test(
        dubai_ts, year_list,
        title=f"SQ-MK: Dubai Core Temperature ({ssp_name.upper()})"
    )
    sqmk_temp['plot_fig'].savefig(os.path.join(results_dir, f"sqmk_temperature_{ssp_name}.png"), dpi=150)
    plt.close(sqmk_temp['plot_fig'])
    print(f"    Temperature mutation years: {sqmk_temp['significant_mutation_years']}")
    
    # --- Precipitation Analysis ---
    print(f"\n    [Precipitation] Running MK spatial analysis...")
    pcp_grids = predictions[:, FEATURES.index('PCP'), :, :]
    annual_pcp, _ = build_annual_cube(pcp_grids, prediction_dates, agg='sum')
    z_pcp, p_pcp = analyzer.analyze_spatial_grid(annual_pcp)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{ssp_name.upper()} - Precipitation Trend Analysis (2015-2100)', fontsize=14)
    
    im1 = axes[0].imshow(z_pcp, cmap='BrBG', aspect='auto', origin='lower')
    axes[0].set_title('Z-statistic (MK Test)')
    _apply_geo_axes(axes[0])
    plt.colorbar(im1, ax=axes[0], label='Z-stat (+ve = wetting)')
    
    im2 = axes[1].imshow(p_pcp, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=0.1, origin='lower')
    axes[1].set_title('p-value (MK Test)')
    _apply_geo_axes(axes[1])
    plt.colorbar(im2, ax=axes[1], label='p-value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"mk_precipitation_{ssp_name}.png"), dpi=150)
    plt.close()
    
    # UAE mean precipitation SQ-MK
    uae_pcp = annual_pcp.mean(axis=(1, 2))
    mk_pcp = analyzer.run_mk_test(uae_pcp)
    sqmk_pcp = analyzer.run_sqmk_test(
        uae_pcp, year_list,
        title=f"SQ-MK: UAE Mean Precipitation ({ssp_name.upper()})"
    )
    sqmk_pcp['plot_fig'].savefig(os.path.join(results_dir, f"sqmk_precipitation_{ssp_name}.png"), dpi=150)
    plt.close(sqmk_pcp['plot_fig'])
    print(f"    UAE PCP: trend={mk_pcp['trend']}, Z={mk_pcp['z_stat']:.3f}, p={mk_pcp['p_value']:.4f}")
    print(f"    Precipitation mutation years: {sqmk_pcp['significant_mutation_years']}")
    
    # --- Decadal Temperature Projection Plot ---
    print(f"\n    [Decadal] Generating temperature projection timeline...")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    uae_temp = annual_temp.mean(axis=(1, 2))
    ax.plot(year_list, uae_temp, color='orange', linewidth=2, label='UAE Mean')
    ax.plot(year_list, dubai_ts, color='red', linewidth=2, label='Dubai Core')
    
    desert_ts = annual_temp[:, 2, 2]
    ax.plot(year_list, desert_ts, color='blue', linewidth=2, alpha=0.7, label='Desert Pixel [2,2]')
    
    ax.set_title(f'{ssp_name.upper()} - Annual Mean Temperature Projection (2015-2100)', fontsize=13)
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"temp_timeline_{ssp_name}.png"), dpi=150)
    plt.close()
    
    return {
        'mk_temp': mk_temp,
        'mk_pcp': mk_pcp,
        'sqmk_temp_mutations': sqmk_temp['significant_mutation_years'],
        'sqmk_pcp_mutations': sqmk_pcp['significant_mutation_years'],
    }


# ─── PROCESS EACH SSP SCENARIO ───────────────────────────
uhi = UHIFuzzyAdjuster()
analyzer = ClimateTrendAnalyzer()
all_results = {}

for ssp_file in ssp_files:
    ssp_name = ssp_file.split("_")[2].lower()
    results_dir = os.path.join(RESULTS_BASE, ssp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {ssp_name.upper()} ({ssp_file})")
    print(f"{'='*60}")
    
    # Step 1: Batched GPU inference
    print(f"\n  [1/3] Running batched CNN-LSTM inference (batch_size={BATCH_SIZE})...")
    dataset = ClimateDataset(ssp_file, era5_path=None, sequence_length=SEQ_LENGTH, norm_stats=norm_stats)
    predictions, prediction_dates = run_inference(dataset, model, device)
    print(f"    Output shape: {predictions.shape}")
    print(f"    Date range: {prediction_dates[0]} --> {prediction_dates[-1]}")
    
    np.save(os.path.join(results_dir, f"downscaled_{ssp_name}.npy"), predictions)
    
    # Step 2: Multi-Variable Urban Physics (vectorized LUT)
    print(f"\n  [2/3] Applying Multi-Variable Urban Physics Engine (vectorized)...")
    print(f"        (Thermal + Moisture + Friction | PCP & AP untouched)")
    uhi_predictions = np.zeros_like(predictions)
    for i in range(len(predictions)):
        uhi_predictions[i] = uhi.adjust_full_grid(predictions[i])
        if (i + 1) % 5000 == 0:
            print(f"      Adjusted {i + 1}/{len(predictions)} grids...")
    
    np.save(os.path.join(results_dir, f"uhi_adjusted_full_{ssp_name}.npy"), uhi_predictions)
    
    uhi_adjusted = uhi_predictions[:, FEATURES.index('T_avg'), :, :]
    np.save(os.path.join(results_dir, f"uhi_adjusted_{ssp_name}.npy"), uhi_adjusted)
    
    # Step 3: Statistical Analysis & Plots
    print(f"\n  [3/3] Running statistical analysis & generating plots...")
    results = generate_plots(ssp_name, results_dir, predictions, prediction_dates,
                            uhi_adjusted, analyzer)
    all_results[ssp_name] = results

# ─── CROSS-SSP COMPARISON SUMMARY ────────────────────────
print(f"\n{'='*60}")
print("CROSS-SCENARIO COMPARISON")
print(f"{'='*60}")
print(f"\n  {'Scenario':>10s} | {'Temp Trend':>12s} | {'Temp Z':>8s} | {'Temp Slope':>12s} | {'PCP Trend':>10s} | {'PCP Z':>8s}")
print(f"  {'-'*10} | {'-'*12} | {'-'*8} | {'-'*12} | {'-'*10} | {'-'*8}")

for ssp, res in all_results.items():
    print(f"  {ssp.upper():>10s} | {res['mk_temp']['trend']:>12s} | {res['mk_temp']['z_stat']:>8.3f} | "
          f"{res['mk_temp']['slope']:>10.4f} C/yr | {res['mk_pcp']['trend']:>10s} | {res['mk_pcp']['z_stat']:>8.3f}")

print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"{'='*60}")
print(f"  All outputs saved to: {RESULTS_BASE}/")
for ssp_name in all_results:
    rdir = os.path.join(RESULTS_BASE, ssp_name)
    files = os.listdir(rdir)
    print(f"\n  {ssp_name.upper()}/ ({len(files)} files):")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(rdir, f))
        print(f"    {f:45s} ({size/1024:.1f} KB)")
