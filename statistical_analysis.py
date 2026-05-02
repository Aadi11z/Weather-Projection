"""
statistical_analysis.py - Climate Trend Analysis with TFPW-MK and WMO Normals
================================================================================
Provides non-parametric statistical tools for validating long-term trends in
climate downscaling outputs.

Key upgrades (committee feedback):
  1. Trend-Free Pre-Whitening (TFPW) for all MK tests to eliminate Type I
     errors from serial correlation in daily/annual climate data.
  2. WMO 30-year climate normal sub-period analysis for structurally valid
     trend comparisons across multiple GCMs and time horizons.
"""
import pymannkendall as mk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class ClimateTrendAnalyzer:
    def __init__(self):
        """
        Analyzer for validating long-term trends in climate data using non-parametric statistics.
        Designed to support Extreme Precipitation and Urban Heat Island research papers.

        All MK tests use the Trend-Free Pre-Whitening Modification (TFPW) to
        remove lag-1 serial autocorrelation before computing the test statistic,
        preventing inflated Type I error rates on serially correlated climate
        time series.
        """
        pass

    # ─── CORE TESTS ───────────────────────────────────────────

    def run_mk_test(self, time_series_data, alpha=0.05):
        """
        Runs the TFPW Mann-Kendall test on a 1D time series to detect monotonic trends.

        The TFPW variant removes the estimated Sen slope, computes the lag-1
        autoregressive coefficient on the detrended residuals, pre-whitens the
        original series, re-adds the trend, and then applies the standard MK
        statistic.  This eliminates Type I error inflation caused by persistent
        climate autocorrelation.

        Args:
            time_series_data: list or numpy array of values over time.
            alpha: Significance level (default 0.05 for 95% confidence).

        Returns:
            Dictionary containing trend direction, p-value, Z-statistic, and Sen slope.
        """
        result = mk.trend_free_pre_whitening_modification_test(time_series_data, alpha=alpha)
        return {
            'trend': result.trend,
            'p_value': result.p,
            'z_stat': result.z,
            'slope': result.slope
        }

    def run_sqmk_test(self, time_series_data, years_list, title="Sequential MK Trend Mutation Analysis"):
        """
        Runs the Sequential Mann-Kendall (SQ-MK) test to identify chronological mutation points
        (abrupt structural shifts in the climate data).

        Args:
            time_series_data: list or 1D numpy array of values (e.g., annual precipitation peaks).
            years_list: list of years corresponding to the data for plotting.
            title: Title for the generated output plot.

        Returns:
            Dictionary containing forward stats, backward stats, identified mutation years, and the plot figure.
        """
        n = len(time_series_data)

        def _calc_seq_mk(data):
            """Helper function to calculate the sequential standard normal deviate u(t)."""
            sk = 0
            u = np.zeros(len(data))
            for i in range(1, len(data)):
                for j in range(i):
                    if data[i] > data[j]:
                        sk += 1
                    elif data[i] < data[j]:
                        sk -= 1

                # Expected value and variance of sk based on MK theory
                E_sk = i * (i + 1) / 4.0
                Var_sk = i * (i + 1) * (2 * i + 5) / 72.0

                # Standard normal deviate
                if Var_sk > 0:
                    u[i] = (sk - E_sk) / np.sqrt(Var_sk)
                else:
                    u[i] = 0
            return u

        # Forward Sequence u(t)
        u_f = _calc_seq_mk(time_series_data)

        # Backward Sequence u'(t) (calculated logically from end to beginning)
        u_b_rev = _calc_seq_mk(time_series_data[::-1])
        u_b = -1 * u_b_rev[::-1]

        # --- Plotting the Sequence ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years_list, u_f, label="Forward Sequence $u(t)$", color='blue', linewidth=2)
        ax.plot(years_list, u_b, label="Backward Sequence $u'(t)$", color='red', linestyle='--', linewidth=2)

        # 95% Confidence Limits (alpha=0.05 -> z_critical = +/- 1.96)
        ax.axhline(y=1.96, color='black', linestyle=':', label='95% Confidence Bounds')
        ax.axhline(y=-1.96, color='black', linestyle=':')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Sequential Statistic Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.tight_layout()

        # --- Mutation Point Detection ---
        # A significant structural shift occurs where the forward and backward lines cross
        mutation_points = []
        for i in range(1, len(u_f)):
            if (u_f[i-1] - u_b[i-1]) * (u_f[i] - u_b[i]) < 0:  # Intersection detected
                # Check if intersection happens outside the confidence bounds (significance)
                if abs(u_f[i]) > 1.96 or abs(u_b[i]) > 1.96:
                    mutation_points.append(years_list[i])

        return {
            'forward_stats': u_f,
            'backward_stats': u_b,
            'significant_mutation_years': mutation_points,
            'plot_fig': fig
        }

    def analyze_spatial_grid(self, spatiotemporal_cube):
        """
        Runs the TFPW-MK trend test for every individual pixel in a 17x17 grid over time,
        producing a complete map of statistical warming/drying gradients.

        Args:
            spatiotemporal_cube: numpy array of shape (Time_Years, Height, Width)
                                 e.g., (30, 17, 17) containing annual averages/sums.
        Returns:
            z_grid: Map of Z-statistics.
            p_grid: Map of p-values (significance).
        """
        T, H, W = spatiotemporal_cube.shape
        z_grid = np.zeros((H, W))
        p_grid = np.zeros((H, W))

        for i in range(H):
            for j in range(W):
                ts = spatiotemporal_cube[:, i, j]
                # Small safety check for NaN clusters
                if np.isnan(ts).all():
                    z_grid[i, j], p_grid[i, j] = np.nan, np.nan
                    continue

                result = mk.trend_free_pre_whitening_modification_test(ts)
                z_grid[i, j] = result.z
                p_grid[i, j] = result.p

        return z_grid, p_grid

    # ─── WMO 30-YEAR CLIMATE NORMAL ANALYSIS ─────────────────

    # Standard WMO sub-periods for CMIP6 future projections
    WMO_PERIODS = [
        ('2015-2044', 2015, 2044),
        ('2045-2074', 2045, 2074),
        ('2075-2100', 2075, 2100),  # 26 years — maximum available
    ]

    def _slice_annual_cube(self, annual_cube, year_list, start_year, end_year):
        """
        Slice an annual cube and corresponding year list to a [start_year, end_year] window.

        Args:
            annual_cube: numpy array of shape (N_years, 17, 17).
            year_list: list of years corresponding to axis 0.
            start_year: inclusive start year.
            end_year: inclusive end year.

        Returns:
            (sub_cube, sub_years) — sliced arrays.
        """
        mask = [(y >= start_year and y <= end_year) for y in year_list]
        indices = [i for i, m in enumerate(mask) if m]
        if len(indices) == 0:
            return None, []
        sub_cube = annual_cube[indices]
        sub_years = [year_list[i] for i in indices]
        return sub_cube, sub_years

    def run_wmo_period_analysis(self, annual_cube, year_list, variable_name,
                                gcm_name, output_dir, agg_axes=(1, 2)):
        """
        Analyse an annual climate cube across WMO 30-year sub-periods.

        For each sub-period this method:
          1. Slices the annual cube.
          2. Computes the spatial mean time series (averaging over the 17x17 grid).
          3. Runs the TFPW-MK test for trend detection.
          4. Runs the SQ-MK test for mutation point detection.
          5. Generates per-period SQ-MK plots saved to output_dir.

        Args:
            annual_cube: numpy array (N_years, 17, 17) — annual mean/sum values.
            year_list: list of years corresponding to axis 0.
            variable_name: e.g. 'Temperature' or 'Precipitation' — for labels.
            gcm_name: e.g. 'MIROC6' — used in titles and filenames.
            output_dir: directory to save plots.
            agg_axes: axes over which to spatially average (default (1,2) = 17x17).

        Returns:
            List of dicts, one per WMO period, each containing:
              period_label, years, mk_result, sqmk_mutation_years.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for label, y_start, y_end in self.WMO_PERIODS:
            sub_cube, sub_years = self._slice_annual_cube(annual_cube, year_list, y_start, y_end)
            if sub_cube is None or len(sub_years) < 5:
                print(f"    [{label}] Insufficient data ({len(sub_years) if sub_years else 0} years), skipping.")
                results.append({
                    'period_label': label, 'years': sub_years,
                    'mk_result': None, 'sqmk_mutation_years': []
                })
                continue

            # Spatial mean time series
            ts = sub_cube.mean(axis=tuple(agg_axes))

            # TFPW-MK trend test
            mk_res = self.run_mk_test(ts)

            # SQ-MK mutation test
            sqmk_title = f"SQ-MK: {gcm_name} {variable_name} ({label})"
            sqmk_res = self.run_sqmk_test(ts, sub_years, title=sqmk_title)
            sqmk_fname = f"sqmk_{gcm_name}_{variable_name.lower()}_{label}.png"
            sqmk_res['plot_fig'].savefig(os.path.join(output_dir, sqmk_fname), dpi=150)
            plt.close(sqmk_res['plot_fig'])

            print(f"    [{label}] {variable_name}: trend={mk_res['trend']}, "
                  f"Z={mk_res['z_stat']:.3f}, p={mk_res['p_value']:.4f}, "
                  f"slope={mk_res['slope']:.5f}/yr  |  mutations={sqmk_res['significant_mutation_years']}")

            # --- Pixel-by-pixel Spatial MK Heatmaps ---
            z_grid, p_grid = self.analyze_spatial_grid(sub_cube)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'{gcm_name} {variable_name} TFPW-MK Trends ({label})', fontsize=14)
            
            lat_ticks = [0, 4, 8, 12, 16]
            lat_labels = ['22.0°N', '23.0°N', '24.0°N', '25.0°N', '26.0°N']
            lon_ticks = [0, 4, 8, 12, 16]
            lon_labels = ['52.0°E', '53.0°E', '54.0°E', '55.0°E', '56.0°E']
            
            cmap_z = 'RdBu_r' if variable_name.lower() == 'temperature' else 'BrBG'
            cbar_label = 'Z-stat (+ve = warming)' if variable_name.lower() == 'temperature' else 'Z-stat (+ve = wetting)'
            
            im1 = axes[0].imshow(z_grid, cmap=cmap_z, aspect='auto', origin='lower')
            axes[0].set_title('Z-statistic (MK Test)')
            axes[0].set_yticks(lat_ticks)
            axes[0].set_yticklabels(lat_labels)
            axes[0].set_xticks(lon_ticks)
            axes[0].set_xticklabels(lon_labels)
            axes[0].set_ylabel('Latitude')
            axes[0].set_xlabel('Longitude')
            plt.colorbar(im1, ax=axes[0], label=cbar_label)
            
            im2 = axes[1].imshow(p_grid, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=0.1, origin='lower')
            axes[1].set_title('p-value (MK Test)')
            axes[1].set_yticks(lat_ticks)
            axes[1].set_yticklabels(lat_labels)
            axes[1].set_xticks(lon_ticks)
            axes[1].set_xticklabels(lon_labels)
            axes[1].set_ylabel('Latitude')
            axes[1].set_xlabel('Longitude')
            plt.colorbar(im2, ax=axes[1], label='p-value')
            
            plt.tight_layout()
            spatial_fname = f"spatial_mk_{gcm_name}_{variable_name.lower().replace(' ', '_')}_{label}.png"
            plt.savefig(os.path.join(output_dir, spatial_fname), dpi=150)
            plt.close(fig)

            results.append({
                'period_label': label,
                'years': sub_years,
                'mk_result': mk_res,
                'sqmk_mutation_years': sqmk_res['significant_mutation_years']
            })

        return results

    @staticmethod
    def print_multi_gcm_comparison(all_results, variable_name, output_csv=None):
        """
        Print a formatted cross-GCM comparison table to the console, and optionally export to CSV.

        Args:
            all_results: dict  {gcm_name: list of period result dicts}
                         (as returned by run_wmo_period_analysis).
            variable_name: e.g. 'Temperature' — for the table header.
            output_csv: Optional filepath to export the summary table.
        """
        print(f"\n{'=' * 85}")
        print(f"  CROSS-GCM WMO-PERIOD COMPARISON: {variable_name}")
        print(f"{'=' * 85}")
        header = f"  {'GCM':>10s} | {'Period':>11s} | {'Trend':>12s} | {'Z':>8s} | {'p':>8s} | {'Slope':>12s} | Mutations"
        print(header)
        print(f"  {'-'*10} | {'-'*11} | {'-'*12} | {'-'*8} | {'-'*8} | {'-'*12} | {'-'*10}")

        for gcm_name, period_results in all_results.items():
            for pr in period_results:
                mk = pr['mk_result']
                if mk is None:
                    print(f"  {gcm_name:>10s} | {pr['period_label']:>11s} | {'N/A':>12s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>12s} | N/A")
                else:
                    print(f"  {gcm_name:>10s} | {pr['period_label']:>11s} | {mk['trend']:>12s} | "
                          f"{mk['z_stat']:>8.3f} | {mk['p_value']:>8.4f} | {mk['slope']:>10.5f}/yr | "
                          f"{pr['sqmk_mutation_years']}")

        print(f"{'=' * 85}\n")
        
        if output_csv:
            import csv
            with open(output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['GCM', 'Period', 'Trend', 'Z', 'p', 'Slope_per_yr', 'Mutations'])
                for gcm_name, period_results in all_results.items():
                    for pr in period_results:
                        mk = pr['mk_result']
                        if mk is None:
                            writer.writerow([gcm_name, pr['period_label'], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
                        else:
                            writer.writerow([
                                gcm_name, 
                                pr['period_label'], 
                                mk['trend'], 
                                round(mk['z_stat'], 3), 
                                round(mk['p_value'], 4), 
                                round(mk['slope'], 5), 
                                str(pr['sqmk_mutation_years'])
                            ])
            print(f"  --> Saved summary table to {output_csv}")


if __name__ == "__main__":
    analyzer = ClimateTrendAnalyzer()
    print("Statistical Trend Analyzer initialized.")
    print("  - MK method: Trend-Free Pre-Whitening Modification (TFPW)")
    print("  - WMO sub-periods: 2015-2044, 2045-2074, 2075-2100")
    print("Ready to process AI-downscaled temporal arrays for TFPW-MK and SQ-MK validation.")
