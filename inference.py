"""
inference.py - Full Post-Training Pipeline (Optimized)
======================================================
Loads the trained CNN-LSTM model, generates downscaled 17x17 predictions
using batched GPU inference, applies vectorized UHI fuzzy adjustment,
and runs Mann-Kendall statistical analysis.
"""
import torch
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import CNNLSTM_Downscaler
from data_loader import ClimateDataset
from fuzzy_uhi import UHIFuzzyAdjuster
from statistical_analysis import ClimateTrendAnalyzer

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH     = "best_downscaler_model.pth"
NORM_STATS     = "norm_stats.json"
MIROC6_PATH    = "MIROC6_UAE_Spatial_Input_1950_2014.csv"
RESULTS_DIR    = "results"
SEQ_LENGTH     = 14
HIDDEN_CHANNELS = 64
BATCH_SIZE     = 32
FEATURES       = ['T_avg', 'PCP', 'AP', 'RH', 'WS']

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── STEP 1: Load Model & Normalization Stats ────────────
print("=" * 60)
print("STEP 1: Loading trained model and normalization stats")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

with open(NORM_STATS, 'r') as f:
    norm_stats = json.load(f)
print(f"  Loaded normalization stats from {NORM_STATS}")

model = CNNLSTM_Downscaler(in_channels=5, hidden_channels=HIDDEN_CHANNELS, out_channels=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"  Loaded model weights from {MODEL_PATH}")

# ─── STEP 2: Batched GPU Inference ───────────────────────
print()
print("=" * 60)
print("STEP 2: Generating downscaled 17x17 predictions (batched GPU)")
print("=" * 60)

dataset = ClimateDataset(MIROC6_PATH, era5_path=None, sequence_length=SEQ_LENGTH, norm_stats=norm_stats)

# Precompute denormalization vectors
means = np.array([norm_stats['y'][f][0] for f in FEATURES]).reshape(5, 1, 1)
stds  = np.array([norm_stats['y'][f][1] for f in FEATURES]).reshape(5, 1, 1)

# Collect prediction dates
prediction_dates = []
for idx in range(len(dataset)):
    start_idx = dataset.sequence_indices[idx]
    pred_date = dataset.valid_dates[start_idx + SEQ_LENGTH - 1]
    prediction_dates.append(pred_date)

# Batched inference
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_predictions = []
processed = 0

print(f"  Total sequences: {len(dataset)} | Batch size: {BATCH_SIZE}")

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        last_frames = output[:, -1, :, :, :].cpu().numpy()
        denorm_batch = last_frames * (stds + 1e-8) + means
        all_predictions.append(denorm_batch)
        processed += len(batch)
        if processed % 5000 < BATCH_SIZE:
            print(f"    Processed {processed}/{len(dataset)} sequences...")

predictions = np.concatenate(all_predictions, axis=0)
print(f"  Predictions shape: {predictions.shape}")
print(f"  Date range: {prediction_dates[0]} --> {prediction_dates[-1]}")

np.save(os.path.join(RESULTS_DIR, "downscaled_predictions.npy"), predictions)
print(f"  Saved predictions to {RESULTS_DIR}/downscaled_predictions.npy")

# ─── STEP 3: Apply Multi-Variable Urban Physics ──────────
print()
print("=" * 60)
print("STEP 3: Applying Multi-Variable Urban Physics Engine (vectorized)")
print("  - Thermal:  +UHI on Temperature")
print("  - Moisture:  -Urban Dry Island on RH")
print("  - Friction:  -Surface Roughness on Wind Speed")
print("  - PCP & AP:  UNTOUCHED")
print("=" * 60)

uhi = UHIFuzzyAdjuster()

uhi_predictions = np.zeros_like(predictions)

print(f"  Processing {len(predictions)} daily 5-channel grids...")
for i in range(len(predictions)):
    uhi_predictions[i] = uhi.adjust_full_grid(predictions[i])
    if (i + 1) % 5000 == 0:
        print(f"    Adjusted {i + 1}/{len(predictions)} grids...")

np.save(os.path.join(RESULTS_DIR, "uhi_adjusted_predictions.npy"), uhi_predictions)
print(f"  Saved full UHI-adjusted predictions to {RESULTS_DIR}/uhi_adjusted_predictions.npy")

uhi_adjusted = uhi_predictions[:, FEATURES.index('T_avg'), :, :]
np.save(os.path.join(RESULTS_DIR, "uhi_adjusted_temperatures.npy"), uhi_adjusted)

# Quick sample comparison
sample_idx = len(predictions) // 2
print(f"\n  Sample pixel [12,10] (Dubai Core) on {prediction_dates[sample_idx]}:")
print(f"    T_avg: {predictions[sample_idx, 0, 12, 10]:.2f} --> {uhi_predictions[sample_idx, 0, 12, 10]:.2f} C   (delta: +{uhi_predictions[sample_idx, 0, 12, 10] - predictions[sample_idx, 0, 12, 10]:.2f})")
print(f"    PCP:   {predictions[sample_idx, 1, 12, 10]:.2f} --> {uhi_predictions[sample_idx, 1, 12, 10]:.2f} mm  [UNTOUCHED]")
print(f"    AP:    {predictions[sample_idx, 2, 12, 10]:.2f} --> {uhi_predictions[sample_idx, 2, 12, 10]:.2f} hPa [UNTOUCHED]")
print(f"    RH:    {predictions[sample_idx, 3, 12, 10]:.2f} --> {uhi_predictions[sample_idx, 3, 12, 10]:.2f} %    (delta: {uhi_predictions[sample_idx, 3, 12, 10] - predictions[sample_idx, 3, 12, 10]:.2f})")
print(f"    WS:    {predictions[sample_idx, 4, 12, 10]:.2f} --> {uhi_predictions[sample_idx, 4, 12, 10]:.2f} m/s  (delta: {uhi_predictions[sample_idx, 4, 12, 10] - predictions[sample_idx, 4, 12, 10]:.2f})")

# ─── STEP 4: Statistical Trend Analysis ──────────────────
print()
print("=" * 60)
print("STEP 4: Running Mann-Kendall trend analysis")
print("=" * 60)

analyzer = ClimateTrendAnalyzer()

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

years_per_date = [d.year for d in prediction_dates]
unique_years = sorted(set(years_per_date))
print(f"  Year range: {unique_years[0]} - {unique_years[-1]} ({len(unique_years)} years)")

def build_annual_cube(daily_grids, dates, agg='mean'):
    """Aggregate daily 17x17 grids into annual (Years, 17, 17) cubes."""
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

# --- 4A: Temperature Trend (UHI-adjusted) ---
print("\n  [4A] Temperature trend analysis (UHI-adjusted)...")
annual_temp, year_list = build_annual_cube(uhi_adjusted, prediction_dates, agg='mean')

z_temp, p_temp = analyzer.analyze_spatial_grid(annual_temp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im1 = axes[0].imshow(z_temp, cmap='RdBu_r', aspect='auto', origin='lower')
axes[0].set_title('Temperature Z-statistic (MK Test)')
_apply_geo_axes(axes[0])
plt.colorbar(im1, ax=axes[0], label='Z-stat')

im2 = axes[1].imshow(p_temp, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=0.1, origin='lower')
axes[1].set_title('Temperature p-value (MK Test)')
_apply_geo_axes(axes[1])
plt.colorbar(im2, ax=axes[1], label='p-value')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mk_temperature_spatial.png"), dpi=150)
plt.close()
print("    Saved: mk_temperature_spatial.png")

dubai_ts = annual_temp[:, 12, 10]
mk_result = analyzer.run_mk_test(dubai_ts)
print(f"    Dubai Core MK: trend={mk_result['trend']}, Z={mk_result['z_stat']:.3f}, p={mk_result['p_value']:.4f}, slope={mk_result['slope']:.4f} C/year")

sqmk = analyzer.run_sqmk_test(dubai_ts, year_list, title="SQ-MK: Dubai Core Temperature Mutation Point")
sqmk['plot_fig'].savefig(os.path.join(RESULTS_DIR, "sqmk_dubai_temperature.png"), dpi=150)
plt.close(sqmk['plot_fig'])
print(f"    Mutation years detected: {sqmk['significant_mutation_years']}")
print("    Saved: sqmk_dubai_temperature.png")

# --- 4B: Precipitation Trend ---
print("\n  [4B] Precipitation trend analysis...")
pcp_grids = predictions[:, FEATURES.index('PCP'), :, :]
annual_pcp, _ = build_annual_cube(pcp_grids, prediction_dates, agg='sum')

z_pcp, p_pcp = analyzer.analyze_spatial_grid(annual_pcp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im1 = axes[0].imshow(z_pcp, cmap='BrBG', aspect='auto', origin='lower')
axes[0].set_title('Precipitation Z-statistic (MK Test)')
_apply_geo_axes(axes[0])
plt.colorbar(im1, ax=axes[0], label='Z-stat')

im2 = axes[1].imshow(p_pcp, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=0.1, origin='lower')
axes[1].set_title('Precipitation p-value (MK Test)')
_apply_geo_axes(axes[1])
plt.colorbar(im2, ax=axes[1], label='p-value')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mk_precipitation_spatial.png"), dpi=150)
plt.close()
print("    Saved: mk_precipitation_spatial.png")

uae_mean_pcp = annual_pcp.mean(axis=(1, 2))
sqmk_pcp = analyzer.run_sqmk_test(uae_mean_pcp, year_list, title="SQ-MK: UAE Mean Annual Precipitation Mutation Point")
sqmk_pcp['plot_fig'].savefig(os.path.join(RESULTS_DIR, "sqmk_uae_precipitation.png"), dpi=150)
plt.close(sqmk_pcp['plot_fig'])
mk_pcp = analyzer.run_mk_test(uae_mean_pcp)
print(f"    UAE Mean PCP MK: trend={mk_pcp['trend']}, Z={mk_pcp['z_stat']:.3f}, p={mk_pcp['p_value']:.4f}")
print(f"    Mutation years: {sqmk_pcp['significant_mutation_years']}")
print("    Saved: sqmk_uae_precipitation.png")

# ─── SUMMARY ─────────────────────────────────────────────
print()
print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Output directory: {RESULTS_DIR}/")
print(f"  Files generated:")
for f in sorted(os.listdir(RESULTS_DIR)):
    size = os.path.getsize(os.path.join(RESULTS_DIR, f))
    print(f"    {f:45s} ({size/1024:.1f} KB)")
print()
print("  Next steps:")
print("  - Use downscaled_predictions.npy for further spatial analysis")
print("  - Use uhi_adjusted_temperatures.npy for the UHI paper")
print("  - Review the MK/SQ-MK plots in results/ for your papers")
