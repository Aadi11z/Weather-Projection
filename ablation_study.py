"""
ablation_study.py  —  Fuzzy UHI Physics Engine Ablation
========================================================
Quantifies the contribution of the Fuzzy UHI Urban Physics Engine by comparing
two model variants against ERA5 ground truth over the 2015-2025 validation window:

  Variant A  (Baseline)  : CNN-LSTM raw output       — NO fuzzy adjustment
  Variant B  (Full Model): CNN-LSTM + Fuzzy Physics  — WITH UHI / Dry Island / Friction

Evaluation Metrics (per channel, per variant):
  • MAE   — Mean Absolute Error
  • RMSE  — Root Mean Squared Error
  • Bias  — Mean signed error  (positive = model warm/wet bias)
  • R²    — Pearson coefficient of determination
  • SSIM  — Spatial Structural Similarity Index (per-frame, averaged)
  • PeakErr  — 95th-percentile absolute error (captures extremes)
  • PSNR  — Peak Signal-to-Noise Ratio (spatial fidelity proxy)

Outputs  (all written to  ablation_results/):
  • ablation_metrics_table.csv           — full numeric comparison
  • ablation_bar_charts.png              — grouped bar chart per metric / feature
  • ablation_spatial_error_maps.png      — MAE grid: raw vs fuzzy vs delta
  • ablation_seasonal_bias.png           — monthly bias curve raw vs fuzzy
  • ablation_extreme_qq.png              — Q-Q plot of tails (Dubai core)
  • ablation_summary_report.txt          — human-readable narrative summary
  • ablation_delta_improvement.png       — % improvement heatmap (fuzzy vs raw)

Run:
    python ablation_study.py

Prerequisites (same as evaluate_model.py):
    UAE_ERA5_Spatial_Baseline_2015_2025.csv
    MIROC6_UAE_SSP245_2015_2025.csv
    results_future/ssp245/downscaled_ssp245.npy
    results_future/ssp245/uhi_adjusted_full_ssp245.npy
"""
"""
ablation_study.py  —  Fuzzy UHI Physics Engine Ablation
========================================================
Quantifies the contribution of the Fuzzy UHI Urban Physics Engine by comparing
two model variants against ERA5 ground truth over the 2015-2025 validation window.

FIXED: Added safety logic for TwoSlopeNorm to handle cases with zero delta.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
import seaborn as sns
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ERA5_CSV        = 'UAE_ERA5_Spatial_Baseline_2015_2025.csv'
MIROC_CSV       = 'MIROC6_UAE_SSP245_2015_2025.csv'
RAW_NPY         = 'results_future/ssp245/downscaled_ssp245.npy'
FUZZY_NPY       = 'results_future/ssp245/uhi_adjusted_full_ssp245.npy'
SEQ_LENGTH      = 14
FEATURES        = ['T_avg', 'PCP', 'AP', 'RH', 'WS']
UNITS           = ['°C',    'mm',  'hPa','%',  'm/s']
DUBAI_CORE      = (12, 10)   # pixel index of Dubai urban core in the 17×17 grid
OUT_DIR         = 'ablation_results'

_LAT_TICKS      = [0, 4, 8, 12, 16]
_LAT_LABELS     = ['22°N', '23°N', '24°N', '25°N', '26°N']
_LON_TICKS      = [0, 4, 8, 12, 16]
_LON_LABELS     = ['52°E', '53°E', '54°E', '55°E', '56°E']

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def apply_geo_axes(ax, fontsize=7):
    ax.set_yticks(_LAT_TICKS);  ax.set_yticklabels(_LAT_LABELS, fontsize=fontsize)
    ax.set_xticks(_LON_TICKS);  ax.set_xticklabels(_LON_LABELS, fontsize=fontsize)

def mae(pred, truth):
    return float(np.nanmean(np.abs(pred - truth)))

def rmse(pred, truth):
    return float(np.sqrt(np.nanmean((pred - truth) ** 2)))

def bias(pred, truth):
    return float(np.nanmean(pred - truth))

def r_squared(pred, truth):
    p = pred.ravel();  t = truth.ravel()
    mask = ~(np.isnan(p) | np.isnan(t))
    if mask.sum() < 2: return np.nan
    corr = np.corrcoef(p[mask], t[mask])[0, 1]
    return float(corr ** 2)

def peak_error(pred, truth, q=0.95):
    return float(np.nanquantile(np.abs(pred - truth), q))

def psnr(pred, truth):
    data_range = float(np.nanmax(truth) - np.nanmin(truth))
    if data_range == 0: return np.nan
    mse_val = np.nanmean((pred - truth) ** 2)
    if mse_val == 0: return np.inf
    return float(10 * np.log10(data_range ** 2 / mse_val))

def ssim_spatial(pred_stack, truth_stack, data_range=None):
    if data_range is None:
        data_range = float(np.nanmax(truth_stack) - np.nanmin(truth_stack))
    if data_range == 0: return np.nan
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    ssim_vals = []
    for t in range(pred_stack.shape[0]):
        p, g = pred_stack[t].astype(float), truth_stack[t].astype(float)
        mu_p, mu_g = np.nanmean(p), np.nanmean(g)
        sig_p, sig_g = np.nanstd(p), np.nanstd(g)
        sig_pg = np.nanmean((p - mu_p) * (g - mu_g))
        num = (2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)
        den = (mu_p**2 + mu_g**2 + C1) * (sig_p**2 + sig_g**2 + C2)
        ssim_vals.append(num / den if den != 0 else 0.0)
    return float(np.mean(ssim_vals))

def compute_all_metrics(pred, truth, label):
    results = {}
    for ci, (feat, unit) in enumerate(zip(FEATURES, UNITS)):
        p, t = pred[:, ci, :, :], truth[:, ci, :, :]
        results[feat] = {
            'variant': label, 'feature': feat, 'unit': unit,
            'MAE': mae(p, t), 'RMSE': rmse(p, t), 'Bias': bias(p, t),
            'R2': r_squared(p, t), 'PeakErr95': peak_error(p, t),
            'PSNR_dB': psnr(p, t), 'SSIM': ssim_spatial(p, t),
        }
    return results

# ─── STEP 0-2: SETUP & DATA LOADING ──────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
print("=" * 65); print("  ABLATION STUDY: Raw CNN-LSTM vs. Fuzzy Physics"); print("=" * 65)

df_era5 = pd.read_csv(ERA5_CSV); df_era5['Date'] = pd.to_datetime(df_era5['Date']).dt.date
era5_dates = sorted(df_era5['Date'].unique())
df_miroc = pd.read_csv(MIROC_CSV, usecols=['Date']); df_miroc['Date'] = pd.to_datetime(df_miroc['Date']).dt.date
miroc_dates = sorted(df_miroc['Date'].unique())
prediction_dates = miroc_dates[SEQ_LENGTH - 1:]
valid_dates = sorted(set(era5_dates).intersection(set(prediction_dates)))
era5_indices = [era5_dates.index(d) for d in valid_dates]
miroc_indices = [prediction_dates.index(d) for d in valid_dates]

raw_sub = np.load(RAW_NPY)[miroc_indices]
fuzzy_sub = np.load(FUZZY_NPY)[miroc_indices]

df_era5_sub = df_era5[df_era5['Date'].isin(valid_dates)].sort_values(['Date', 'Lat', 'Lon'])
era5_data_dict = {date: np.stack([grp[f].values.reshape(17, 17) for f in FEATURES], axis=0) 
                  for date, grp in df_era5_sub.groupby('Date')}
era5_sub = np.stack([era5_data_dict[d] for d in valid_dates], axis=0)

# ─── STEP 3: COMPUTE METRICS ──────────────────────────────────────────────────
metrics_raw = compute_all_metrics(raw_sub, era5_sub, 'Raw CNN-LSTM')
metrics_fuzzy = compute_all_metrics(fuzzy_sub, era5_sub, 'Fuzzy-Adjusted')

rows = []
for feat in FEATURES:
    rows.append(metrics_raw[feat]); rows.append(metrics_fuzzy[feat])
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'ablation_metrics_table.csv'), index=False)

# ─── STEP 4: VISUALISATIONS (WITH FIXES) ───────────────────────────────────────
PALETTE = {'raw': '#4A90D9', 'fuzzy': '#E8643A', 'era5': '#2C2C2C', 'delta': '#27AE60'}

# Fig 1: Grouped Bar Charts
METRICS_TO_PLOT = ['MAE', 'RMSE', 'PeakErr95', 'R2', 'SSIM', 'PSNR_dB']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for mi, metric in enumerate(METRICS_TO_PLOT):
    ax = axes.flatten()[mi]
    x, w = np.arange(len(FEATURES)), 0.35
    v_raw = [metrics_raw[f][metric] for f in FEATURES]
    v_fuz = [metrics_fuzzy[f][metric] for f in FEATURES]
    ax.bar(x - w/2, v_raw, w, label='Raw', color=PALETTE['raw'], alpha=0.8)
    ax.bar(x + w/2, v_fuz, w, label='Fuzzy', color=PALETTE['fuzzy'], alpha=0.8)
    ax.set_title(metric); ax.set_xticks(x); ax.set_xticklabels(FEATURES); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'ablation_bar_charts.png'))

# Fig 2: Spatial Error Maps (FIXED TwoSlopeNorm)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for row_i, (ci, feat_label, unit, cmap_err) in enumerate([(0, 'T_avg', '°C', 'Reds'), (3, 'RH', '%', 'Blues')]):
    m_raw = np.abs(raw_sub[:, ci, :, :] - era5_sub[:, ci, :, :]).mean(axis=0)
    m_fuz = np.abs(fuzzy_sub[:, ci, :, :] - era5_sub[:, ci, :, :]).mean(axis=0)
    delta = m_raw - m_fuz
    vmax = max(m_raw.max(), m_fuz.max())
    
    axes[row_i, 0].imshow(m_raw, cmap=cmap_err, vmin=0, vmax=vmax, origin='lower')
    axes[row_i, 1].imshow(m_fuz, cmap=cmap_err, vmin=0, vmax=vmax, origin='lower')
    
    # FIX: Safety check for TwoSlopeNorm
    abs_max = np.abs(delta).max()
    if abs_max > 1e-7:
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = Normalize(vmin=-1, vmax=1)
    
    im = axes[row_i, 2].imshow(delta, cmap='RdYlGn', norm=norm, origin='lower')
    plt.colorbar(im, ax=axes[row_i, 2])
    for ax in axes[row_i]: apply_geo_axes(ax)

plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'ablation_spatial_error_maps.png'))

# Fig 3: Seasonal Bias
months = np.array([d.month for d in valid_dates])
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (ci, feat_label) in zip(axes, [(0, 'T_avg'), (3, 'RH')]):
    b_raw, b_fuz = [], []
    for m in range(1, 13):
        mask = (months == m)
        e_m = era5_sub[mask, ci].mean()
        b_raw.append(raw_sub[mask, ci].mean() - e_m)
        b_fuz.append(fuzzy_sub[mask, ci].mean() - e_m)
    ax.plot(b_raw, 'o-', color=PALETTE['raw'], label='Raw Bias')
    ax.plot(b_fuz, 's--', color=PALETTE['fuzzy'], label='Fuzzy Bias')
    ax.axhline(0, color='k', alpha=0.5); ax.legend(); ax.set_title(feat_label)
plt.savefig(os.path.join(OUT_DIR, 'ablation_seasonal_bias.png'))

# Fig 4: Q-Q Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (ci, feat_label) in zip(axes, [(0, 'T_avg'), (1, 'PCP')]):
    r, c = DUBAI_CORE
    q_grid = np.linspace(0.01, 0.99, 100)
    q_e = np.quantile(era5_sub[:, ci, r, c], q_grid)
    ax.scatter(q_e, np.quantile(raw_sub[:, ci, r, c], q_grid), s=10, color=PALETTE['raw'], label='Raw')
    ax.scatter(q_e, np.quantile(fuzzy_sub[:, ci, r, c], q_grid), s=10, color=PALETTE['fuzzy'], label='Fuzzy')
    ax.plot(q_e, q_e, 'k--', alpha=0.5); ax.set_title(feat_label); ax.legend()
plt.savefig(os.path.join(OUT_DIR, 'ablation_extreme_qq.png'))

# Fig 5: % Improvement Heatmap (FIXED TwoSlopeNorm)
m_lower, m_higher = ['MAE', 'RMSE', 'PeakErr95'], ['R2', 'SSIM', 'PSNR_dB']
improve_mat = np.zeros((len(FEATURES), len(m_lower) + len(m_higher)))
for fi, feat in enumerate(FEATURES):
    for mi, m in enumerate(m_lower):
        r, f = metrics_raw[feat][m], metrics_fuzzy[feat][m]
        improve_mat[fi, mi] = (r - f) / (abs(r) + 1e-9) * 100
    for mi, m in enumerate(m_higher):
        r, f = metrics_raw[feat][m], metrics_fuzzy[feat][m]
        improve_mat[fi, mi + len(m_lower)] = (f - r) / (abs(r) + 1e-9) * 100

fig, ax = plt.subplots(figsize=(12, 6))
v_min, v_max = improve_mat.min(), improve_mat.max()
# FIX: Handle cases where data doesn't cross zero
if v_min < 0 < v_max:
    norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
else:
    norm = Normalize(vmin=v_min, vmax=v_max)

im = ax.imshow(improve_mat, cmap='RdYlGn', norm=norm, aspect='auto')
plt.colorbar(im); ax.set_title("Percent Improvement (Fuzzy vs Raw)")
plt.savefig(os.path.join(OUT_DIR, 'ablation_delta_improvement.png'))

# ─── STEP 5-6: FINAL REPORT ───────────────────────────────────────────────────
print("\n[5/6] Writing report..."); print("\n[6/6] Done.")
with open(os.path.join(OUT_DIR, 'ablation_summary_report.txt'), 'w') as f:
    f.write(f"Ablation Study Summary\nGenerated: {datetime.now()}\n")
    for feat in FEATURES:
        f.write(f"\n{feat}: Raw MAE={metrics_raw[feat]['MAE']:.3f}, Fuzzy MAE={metrics_fuzzy[feat]['MAE']:.3f}")
