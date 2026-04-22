import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("======================================================")
print("  EVALUATING MODEL ACCURACY: DOWNSCALER vs ERA5 BASELINE")
print("======================================================")

# Ensure evaluation directory exists
os.makedirs('evaluation', exist_ok=True)

# ─── 1. LOAD AND ALIGN DATES ───────────────────────────────────────────────
print("\n[1/3] Loading Dates and Aligning Temporal Sequences...")

# Load ERA5 dates
era5_csv = 'UAE_ERA5_Spatial_Baseline_2015_2025.csv'
print(f"  Loading ERA5 dates from {era5_csv}...")
df_era5 = pd.read_csv(era5_csv)
df_era5['Date'] = pd.to_datetime(df_era5['Date']).dt.date
era5_dates = sorted(df_era5['Date'].unique())

# Load MIROC6 dates (to reconstruct the sequence valid dates)
miroc_csv = 'MIROC6_UAE_SSP245_2015_2025.csv'
print(f"  Loading MIROC6 dates from {miroc_csv}...")
df_miroc = pd.read_csv(miroc_csv, usecols=['Date', 'Lat', 'Lon'])
df_miroc['Date'] = pd.to_datetime(df_miroc['Date']).dt.date
miroc_dates = sorted(df_miroc['Date'].unique())

# Since SEQ_LENGTH = 14, the first predicted output is index 13
prediction_dates = miroc_dates[13:]

# Find intersection of dates (2015-2025 overlapping window)
valid_dates = sorted(list(set(era5_dates).intersection(set(prediction_dates))))
print(f"  Found {len(valid_dates)} exact matching days between 2015 and 2025.")

era5_indices = [era5_dates.index(d) for d in valid_dates]
miroc_indices = [prediction_dates.index(d) for d in valid_dates]

# ─── 2. LOAD AND SLICE ARRAYS ──────────────────────────────────────────────
print("\n[2/3] Loading Full Arrays and Slicing to Target Window...")

FEATURES = ['T_avg', 'PCP', 'AP', 'RH', 'WS']

raw_ml_full = np.load('results_future/ssp245/downscaled_ssp245.npy')
fuzzy_ml_full = np.load('results_future/ssp245/uhi_adjusted_full_ssp245.npy')

raw_ml_sub = raw_ml_full[miroc_indices]
fuzzy_ml_sub = fuzzy_ml_full[miroc_indices]

# Reconstruct ERA5 subset into (Time, Channel, H, W)
print("  Reconstructing ERA5 17x17 daily grids...")
era5_data_dict = {}
df_era5_sub = df_era5[df_era5['Date'].isin(valid_dates)].sort_values(by=['Date', 'Lat', 'Lon'])
grouped = df_era5_sub.groupby('Date')

for date, group in grouped:
    spatial_tensor = []
    for feat in FEATURES:
        mat = group[feat].values.reshape(17, 17)
        spatial_tensor.append(mat)
    era5_data_dict[date] = np.stack(spatial_tensor, axis=0)

era5_sub = np.stack([era5_data_dict[d] for d in valid_dates], axis=0)

# Check Shapes
print(f"  ERA5 Ground Truth Shape: {era5_sub.shape}")
print(f"  Raw ML Projection Shape: {raw_ml_sub.shape}")
print(f"  Fuzzy Adjusted Shape:    {fuzzy_ml_sub.shape}")

# Extract Temperature channel (Index 0 in FEATURES) for detailed analysis
idx_t = FEATURES.index('T_avg')
era5_temp = era5_sub[:, idx_t, :, :]
raw_temp = raw_ml_sub[:, idx_t, :, :]
fuzzy_temp = fuzzy_ml_sub[:, idx_t, :, :]

# extract Precipitation (Index 1)
idx_p = FEATURES.index('PCP')
era5_pcp = era5_sub[:, idx_p, :, :]
raw_pcp = raw_ml_sub[:, idx_p, :, :]
fuzzy_pcp = fuzzy_ml_sub[:, idx_p, :, :]


# ─── 3. STATISTICAL EVALUATION & PLOTTING ──────────────────────────────────
print("\n[3/3] Running Statistical Evaluations...")

_LAT_TICKS = [0, 4, 8, 12, 16]
_LAT_LABELS = ['22.0°N', '23.0°N', '24.0°N', '25.0°N', '26.0°N']
_LON_TICKS = [0, 4, 8, 12, 16]
_LON_LABELS = ['52.0°E', '53.0°E', '54.0°E', '55.0°E', '56.0°E']

def apply_geo_axes(ax):
    ax.set_yticks(_LAT_TICKS)
    ax.set_yticklabels(_LAT_LABELS)
    ax.set_xticks(_LON_TICKS)
    ax.set_xticklabels(_LON_LABELS)
    ax.tick_params(axis='both', which='major', labelsize=8)

# --- Analysis A: Spatial Climatology (Grid Integrity) ---
print("  > Generating Spatial Climatologies (Mean pixel temperatures over 11 years)")
mean_era5 = era5_temp.mean(axis=0)
mean_raw = raw_temp.mean(axis=0)
mean_fuz = fuzzy_temp.mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Spatial Climatology - 11 Year Baseline Average Temperature (2015-2025)', fontsize=14)

vmin = min(mean_era5.min(), mean_raw.min(), mean_fuz.min())
vmax = max(mean_era5.max(), mean_raw.max(), mean_fuz.max())

im0 = axes[0].imshow(mean_era5, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
axes[0].set_title('Ground Truth (ERA5 Observations)')
apply_geo_axes(axes[0])
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(mean_raw, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
axes[1].set_title('Raw ML Projector Output')
apply_geo_axes(axes[1])
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(mean_fuz, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
axes[2].set_title('Fuzzy-Adjusted Physics Engine')
apply_geo_axes(axes[2])
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('evaluation/A_Spatial_Climatologies.png', dpi=150)
plt.close()

# Error Grids
mae_raw = np.abs(mean_raw - mean_era5)
mae_fuz = np.abs(mean_fuz - mean_era5)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Spatial Absolute Error vs ERA5 Climatology (2015-2025)', fontsize=14)
imx = axes[0].imshow(mae_raw, cmap='Reds', origin='lower', vmin=0, vmax=mae_raw.max())
axes[0].set_title(f'Raw ML vs ERA5 (Avg Error: {mae_raw.mean():.2f}°C)')
apply_geo_axes(axes[0])
plt.colorbar(imx, ax=axes[0], fraction=0.046, pad=0.04)

imy = axes[1].imshow(mae_fuz, cmap='Reds', origin='lower', vmin=0, vmax=mae_raw.max())
axes[1].set_title(f'Fuzzy Adjusted vs ERA5 (Avg Error: {mae_fuz.mean():.2f}°C)')
apply_geo_axes(axes[1])
plt.colorbar(imy, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('evaluation/B_Spatial_Errors.png', dpi=150)
plt.close()

# Print metrics
print(f"      Raw ML Mean Absolute Error:   {mae_raw.mean():.3f} °C")
print(f"      Fuzzy ML Mean Absolute Error: {mae_fuz.mean():.3f} °C")
print(f"      Max Error on Dubai Core (Raw):   {mae_raw[12, 10]:.3f} °C")
print(f"      Max Error on Dubai Core (Fuzzy): {mae_fuz[12, 10]:.3f} °C")


# --- Analysis B: Monthly Seasonality (Temporal Bounding) ---
print("  > Generating Seasonality Curves (Monthly Averages)")
months = np.array([d.month for d in valid_dates])
monthly_era5, monthly_raw, monthly_fuz = [], [], []

for m in range(1, 13):
    mask = (months == m)
    # Average across all spatial pixels, and across all days belonging to month m
    monthly_era5.append(era5_temp[mask].mean())
    monthly_raw.append(raw_temp[mask].mean())
    monthly_fuz.append(fuzzy_temp[mask].mean())

fig, ax = plt.subplots(figsize=(10, 6))
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.plot(month_labels, monthly_era5, 'k-o', linewidth=3, label='ERA5 Ground Truth')
ax.plot(month_labels, monthly_raw, 'b--o', linewidth=2, label='Raw ML Downscaled')
ax.plot(month_labels, monthly_fuz, 'r--o', linewidth=2, label='Fuzzy-Adjusted Output')

ax.set_title('Seasonal Climatology (UAE Spatial Mean, 2015-2025)', fontsize=14)
ax.set_ylabel('Mean Temperature (°C)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('evaluation/C_Seasonality_Curve.png', dpi=150)
plt.close()

# --- Analysis C: Probability Density Distributions (Extremes) ---
# Flatten the Dubai core pixel over all time to compare distributions
print("  > Generating Probability Density Distributions for Dubai Core Extremes...")
dubai_era5 = era5_temp[:, 12, 10]
dubai_raw = raw_temp[:, 12, 10]
dubai_fuz = fuzzy_temp[:, 12, 10]

fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(dubai_era5, color='black', linewidth=3, label='ERA5 Dubai Core', ax=ax)
sns.kdeplot(dubai_raw, color='blue', linestyle='--', linewidth=2, label='Raw ML Core Projection', ax=ax)
sns.kdeplot(dubai_fuz, color='red', linestyle='--', linewidth=2, label='Fuzzy-Adjusted Core', ax=ax)

ax.set_title('Temperature Distribution Probability Density (Dubai Core, 2015-2025)', fontsize=14)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Density (Frequency of Occurrence)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('evaluation/D_Probability_Density.png', dpi=150)
plt.close()

print("\n======================================================")
print("  EVALUATION COMPLETE - RESULTS SAVED TO /evaluation/")
print("======================================================")
