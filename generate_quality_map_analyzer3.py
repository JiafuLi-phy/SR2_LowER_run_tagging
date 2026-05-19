import argparse
import pandas as pd
import numpy as np
import os
import sys

# Force headless backend for remote servers (e.g., Midway3) before importing pyplot
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker  
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches 
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==========================================
# Scientific Plotting Global Settings
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 3.0,       
    'axes.edgecolor': 'black',   
    'axes.labelsize': 16,        
    'axes.titlesize': 22,        
    'xtick.labelsize': 14,       
    'ytick.labelsize': 14,
    'xtick.major.width': 3.0,    
    'ytick.major.width': 3.0,
    'xtick.color': 'black',      
    'ytick.color': 'black',
    'xtick.direction': 'in',     
    'ytick.direction': 'in',
    'figure.dpi': 300
})

def merge_intervals(intervals):
    """Helper function to merge overlapping intervals and prevent alpha-stacking in plots."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

class SR2QualityAnalyzer:
    def __init__(self, run_info_path, rates_path):
        self.run_info_path = run_info_path
        self.rates_path = rates_path
        self.df = None
        self.df_raw_info = None 
        self.feature_cols = []
        self.scaler = StandardScaler()

    def load_and_merge_data(self):
        print(f"Loading data...\n - Info: {self.run_info_path}\n - Rates: {self.rates_path}")
        df_run = pd.read_csv(self.run_info_path)
        df_rates = pd.read_csv(self.rates_path)
        
        for df_tmp in [df_run, df_rates]:
            if 'number' not in df_tmp.columns:
                for possible_name in ['name', 'run_id', 'RunID', 'Run_ID']:
                    if possible_name in df_tmp.columns:
                        df_tmp.rename(columns={possible_name: 'number'}, inplace=True)
                        break
        
        df_run['number'] = df_run['number'].astype(str).str.zfill(6)
        df_rates['number'] = df_rates['number'].astype(str).str.zfill(6)

        if 'start' in df_run.columns and 'end' in df_run.columns:
            df_run['start'] = pd.to_datetime(df_run['start'], errors='coerce')
            df_run['end'] = pd.to_datetime(df_run['end'], errors='coerce')
        
        self.df_raw_info = df_run.copy() 
        self.df = pd.merge(df_run, df_rates, on='number', how='inner', suffixes=('', '_rates'))
            
        print(f"Raw runs in Info file: {len(df_run)}")
        print(f"Runs with valid Rate features (Merged): {len(self.df)}")
        return self.df

    def _extract_machine_learning_features(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_keywords = ['number', 'Run_ID', 'id', 'time', 'duration', 'Count', 'x_bin', 'y_bin', 'peak_positions', 'peak_basics']
        features = [col for col in numeric_cols if not any(kw.lower() in col.lower() for kw in exclude_keywords)]
        df_features = df[features].fillna(df[features].median())
        std_check = df_features.std()
        valid_features = std_check[std_check > 0].index.tolist()
        print(f"Selected {len(valid_features)} physical rate features for ML modeling.")
        
        print("Features used for Isolation Forest training:")
        for feat in valid_features:
            print(f"  - {feat}")
        
        return valid_features

    def _train_logic(self, data_subset):
        X = data_subset.fillna(data_subset.median())
        X_scaled = self.scaler.fit_transform(X)
        model = IsolationForest(n_estimators=150, contamination='auto', random_state=42)
        model.fit(X_scaled)
        raw_scores = model.decision_function(X_scaled)
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            return 100 * (raw_scores - s_min) / (s_max - s_min)
        return np.full(len(raw_scores), 100.0)

    def calculate_consensus_quality(self, window_sizes=[1, 2, 4, 8, 10], k_mad=4.5, vote_ratio=0.5):
        if not self.feature_cols:
            self.feature_cols = self._extract_machine_learning_features(self.df)

        print(f"\n--- Calculating Consensus Quality across windows: {window_sizes} ---")
        consensus_df = self.df.copy()
        flag_columns = []
        score_columns = []

        for w in window_sizes:
            X_batch = consensus_df[self.feature_cols].rolling(window=w, min_periods=1, center=True).mean()
            score_col = f'score_w{w}'
            consensus_df[score_col] = self._train_logic(X_batch)
            score_columns.append(score_col)

            scores = consensus_df[score_col].values
            median_score = np.median(scores)
            mad = np.median(np.abs(scores - median_score))
            mad = max(mad, 1e-6) 
            threshold = median_score - (k_mad * mad)

            flag_col = f'is_anomaly_w{w}'
            consensus_df[flag_col] = consensus_df[score_col] < threshold
            flag_columns.append(flag_col)

            anomalies_found = consensus_df[flag_col].sum()
            print(f" - Window {w:>2}: Median={median_score:>5.1f}, MAD={mad:>4.1f}, Cut Threshold={threshold:>5.1f} -> Found {anomalies_found:>3} anomalies")

        consensus_df['anomaly_votes'] = consensus_df[flag_columns].sum(axis=1)
        required_votes = int(len(window_sizes) * vote_ratio) + 1
        consensus_df['is_consensus_bad'] = consensus_df['anomaly_votes'] >= required_votes
        consensus_df['quality_score'] = consensus_df[score_columns].mean(axis=1)

        total_bad = consensus_df['is_consensus_bad'].sum()
        print(f"✅ Consensus voting complete. Identified {total_bad} definitively anomalous runs.")
        return consensus_df

    def save_results(self, df_to_save, base_path):
        directory = os.path.dirname(base_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if base_path.endswith('.h5'):
            cols_to_keep = df_to_save.select_dtypes(include=[np.number, 'bool', 'object', 'datetime']).columns
            df_clean = df_to_save[cols_to_keep].copy()
            for c in df_clean.select_dtypes(include=['object']).columns:
                df_clean[c] = df_clean[c].astype(str)
            with pd.HDFStore(base_path, mode='w') as store:
                store.put('run_data', df_clean, format='table', data_columns=True)
        else:
            df_to_save.to_csv(base_path, index=False)

    def _generate_anomaly_plots(self, df_full, bad_subset, window_sizes, save_prefix, title_suffix="", mode_col='mode'):
        """Helper to generate standard anomaly plots for any given subset of runs."""
        
        # 1. Export CSV summary
        export_cols = ['number', 'quality_score', 'anomaly_votes', 'is_consensus_bad']
        if 'start' in bad_subset.columns: export_cols.append('start')
        if mode_col in bad_subset.columns: export_cols.append(mode_col)
        
        bad_runs_summary = bad_subset[export_cols].sort_values('quality_score')
        bad_runs_summary.to_csv(f"{save_prefix}.csv", index=False)

        # 2. Voting Heatmap
        flag_cols = [f'is_anomaly_w{w}' for w in window_sizes]
        heatmap_data = bad_subset.set_index('number')[flag_cols].astype(int)
        heatmap_data['total'] = heatmap_data.sum(axis=1)
        heatmap_data = heatmap_data.sort_values(['total', 'number'], ascending=[False, True]).drop(columns=['total'])
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.3)))
        cmap = sns.color_palette(["#f1f2f6", "#e74c3c"]) 
        sns.heatmap(heatmap_data, cmap=cmap, cbar=False, linewidths=1.0, linecolor='black', ax=ax)
        
        ax.set_title(f"Consensus Voting Heatmap{title_suffix}", pad=20, fontsize=20, weight='bold')
        ax.set_xlabel("Window Size Evaluation", fontsize=16, weight='bold')
        ax.set_ylabel("Run Number", fontsize=16, weight='bold')
        ax.set_xticklabels([f"Window={w}" for w in window_sizes], rotation=45, ha='right')
        
        pass_patch = mpatches.Patch(color='#f1f2f6', label='Pass (Normal)')
        fail_patch = mpatches.Patch(color='#e74c3c', label='Fail (Anomaly)')
        ax.legend(handles=[pass_patch, fail_patch], loc='upper right', bbox_to_anchor=(1.4, 1))

        plt.savefig(f"{save_prefix}_voting_heatmap.png", bbox_inches='tight')
        plt.savefig(f"{save_prefix}_voting_heatmap.pdf", bbox_inches='tight') # Optimization: added pdf
        plt.close()

        # 3. Top Deviating Features Boxplot
        baseline_mean = df_full[self.feature_cols].mean()
        baseline_std = df_full[self.feature_cols].std().replace(0, 1e-9)
        z_scores = (bad_subset[self.feature_cols] - baseline_mean) / baseline_std
        mean_abs_z = z_scores.abs().mean().sort_values(ascending=False)
        top_15_features = mean_abs_z.head(15).index.tolist()
        
        plot_data = z_scores[top_15_features]
        fig, ax = plt.subplots(figsize=(16, 11))
        
        sns.boxplot(data=plot_data, orient='h', ax=ax, palette='Spectral',
                    flierprops=dict(marker='D', markerfacecolor='#e74c3c', markersize=8),
                    boxprops=dict(edgecolor='black', linewidth=2.5),
                    medianprops=dict(color='#f1c40f', linewidth=3.5))
                    
        ax.axvline(0, color='black', linewidth=4.0)
        ax.axvline(3, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.8) 
        ax.axvline(-3, color='#2980b9', linestyle='--', linewidth=2.5, alpha=0.8) 
        ax.set_title(f"Top 15 Feature Deviations{title_suffix}", pad=20, fontsize=22, weight='bold')
        ax.set_xlabel("Z-Score (Standard Deviations from Dataset Mean)", fontsize=18, weight='bold')
        ax.set_ylabel("Detector Features", fontsize=18, weight='bold')
        
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_color('black')
            
        plt.savefig(f"{save_prefix}_feature_deviations.png", bbox_inches='tight')
        plt.savefig(f"{save_prefix}_feature_deviations.pdf", bbox_inches='tight') # Optimization: added pdf
        plt.close()

        # 4. Temporal Distribution (Histogram of anomalies over time)
        if 'start' in bad_subset.columns:
            valid_dates = bad_subset['start'].dropna()
            if not valid_dates.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Robust bin calculation preventing zero-division
                delta_days = (valid_dates.max() - valid_dates.min()).days
                num_bins = min(max(10, delta_days // 7), 50) if delta_days > 0 else 10
                
                sns.histplot(data=valid_dates, bins=num_bins, color="#e74c3c", edgecolor='black', linewidth=2, ax=ax)
                ax.set_title(f"Temporal Distribution of Bad Runs{title_suffix}", pad=20, fontsize=20, weight='bold')
                ax.set_xlabel("Time", fontsize=16, weight='bold')
                ax.set_ylabel("Count of Anomalous Runs", fontsize=16, weight='bold')
                
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right', weight='bold', fontsize=14)
                
                for spine in ax.spines.values():
                    spine.set_linewidth(3.0)
                    spine.set_color('black')
                    
                plt.savefig(f"{save_prefix}_time_distribution.png", bbox_inches='tight')
                plt.savefig(f"{save_prefix}_time_distribution.pdf", bbox_inches='tight') # Optimization: added pdf
                plt.close()

    def analyze_consensus_anomalies(self, df, window_sizes, save_prefix=""):
        print(f"\n--- Generating Consensus Anomaly Diagnostics ---")
        bad_runs = df[df['is_consensus_bad']].copy()
        
        if bad_runs.empty:
            print("✅ No runs flagged by consensus voting.")
            return

        mode_col = 'mode'
        if self.df_raw_info is not None and len(self.df_raw_info.columns) > 1:
            mode_col = self.df_raw_info.columns[1] 

        # ========================================================
        # Phase 1: Global Analysis (All run types included)
        # ========================================================
        global_prefix = f"{save_prefix}_all_modes"
        self._generate_anomaly_plots(df, bad_runs, window_sizes, global_prefix, title_suffix=" (All Modes)", mode_col=mode_col)
        
        if mode_col in bad_runs.columns:
            type_counts = bad_runs[mode_col].value_counts()
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(x=type_counts.index, y=type_counts.values, palette='Set1', edgecolor='black', linewidth=2.5, ax=ax)
            for i, v in enumerate(type_counts.values):
                ax.text(i, v + 0.02 * max(type_counts.values), str(v), ha='center', va='bottom', fontweight='bold', fontsize=16, color='black')
            ax.set_title(f"Distribution of Consensus Anomalous Runs by Type", pad=20, fontsize=20, weight='bold')
            ax.set_ylabel("Number of Anomalous Runs", fontsize=16, weight='bold')
            ax.set_xlabel("Run Type", fontsize=16, weight='bold')
            plt.xticks(rotation=30, ha='right', fontsize=14, weight='bold')
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_color('black')
            plt.savefig(f"{global_prefix}_type_distribution.png", bbox_inches='tight')
            plt.savefig(f"{global_prefix}_type_distribution.pdf", bbox_inches='tight') # Optimization: added pdf
            plt.close()

        # ========================================================
        # Phase 2: Science Run Exclusive Analysis
        # ========================================================
        if mode_col in bad_runs.columns:
            science_mask = bad_runs[mode_col].astype(str).str.lower().str.contains('bkg|background|science', na=False)
            science_bad_runs = bad_runs[science_mask].copy()
            
            if not science_bad_runs.empty:
                print(f"✅ Found {len(science_bad_runs)} anomalous Science Runs. Generating exclusive diagnostics...")
                science_prefix = f"{save_prefix}_science_only"
                self._generate_anomaly_plots(df, science_bad_runs, window_sizes, science_prefix, title_suffix="\n(Science Runs Only)", mode_col=mode_col)
            else:
                print("✅ Awesome! No Science Run anomalies were found among the flagged data.")


    def plot_quality_trend(self, df, title="Detector Quality Trend", save_prefix=""):
        """Plot the trend and explicitly overlay red markers for consensus bad runs."""
        fig = plt.figure(figsize=(20, 12)) 
        gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], hspace=0.17, wspace=0.02)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharey=ax1)
        cax = fig.add_subplot(gs[:, 1]) 
        
        df_sorted = df.sort_values('number').copy()
        
        # Split normal and consensus bad runs
        df_normal = df_sorted[~df_sorted['is_consensus_bad']]
        df_bad = df_sorted[df_sorted['is_consensus_bad']]
        
        mode_col = 'mode'
        if self.df_raw_info is not None and len(self.df_raw_info.columns) > 1:
            mode_col = self.df_raw_info.columns[1]
        
        calib_styles = {
            'kr83m':      {'label': 'Kr-83m',      'color': "#FF0026"}, 
            'ambe':       {'label': 'AmBe',        'color': '#32CD32'}, 
            'rn':         {'label': 'Radon',       'color': '#FF00FF'}, 
            'radon':      {'label': 'Radon',       'color': '#FF00FF'}, 
            'ar37':       {'label': 'Ar-37',       'color': '#FF8C00'}, 
            'neutron':    {'label': 'Neutron',     'color': '#8B4513'}, 
            'th232':      {'label': 'Th-232',      'color': "#100CCE"}, 
            'bkg':        {'label': 'Science Run', 'color': "#E8E8E8"}, 
            'background': {'label': 'Science Run', 'color': "#E8E8E8"}  
        }
        
        run_intervals_dict = {style['label']: [] for style in calib_styles.values()}
        time_intervals_dict = {style['label']: [] for style in calib_styles.values()}
        color_map = {style['label']: style['color'] for style in calib_styles.values()}
        
        if self.df_raw_info is not None and mode_col in self.df_raw_info.columns:
            run_min, run_max = df_sorted['number'].astype(int).min(), df_sorted['number'].astype(int).max()
            
            df_run_axis = self.df_raw_info.dropna(subset=['number', mode_col]).copy()
            df_run_axis['run_int'] = df_run_axis['number'].astype(int)
            df_run_axis = df_run_axis[(df_run_axis['run_int'] >= run_min) & (df_run_axis['run_int'] <= run_max)].sort_values('run_int')
            
            mode_series_run = df_run_axis[mode_col].astype(str).str.lower()
            mode_changed_run = mode_series_run != mode_series_run.shift(1)
            is_large_run_gap = (df_run_axis['run_int'] - df_run_axis['run_int'].shift(1)) > 50 
            block_id_run = (mode_changed_run | is_large_run_gap).cumsum()
            
            for _, group in df_run_axis.groupby(block_id_run):
                current_mode = group[mode_col].iloc[0].lower()
                for key, style in calib_styles.items():
                    if key in current_mode:
                        s_run = group['run_int'].iloc[0] - 0.5
                        e_run = group['run_int'].iloc[-1] + 0.5
                        if (e_run - s_run) < 80:
                            mid = (s_run + e_run) / 2
                            s_run, e_run = mid - 40, mid + 40
                        run_intervals_dict[style['label']].append((s_run, e_run))
                        break

            if 'start' in self.df_raw_info.columns and 'start' in df_sorted.columns:
                time_min, time_max = df_sorted['start'].min(), df_sorted['start'].max()
                df_time_axis = self.df_raw_info.dropna(subset=['start', mode_col]).sort_values('start').copy()
                df_time_axis = df_time_axis[(df_time_axis['start'] >= time_min) & (df_time_axis['start'] <= time_max)]
                
                mode_series_time = df_time_axis[mode_col].astype(str).str.lower()
                mode_changed_time = mode_series_time != mode_series_time.shift(1)
                is_large_time_gap = (df_time_axis['start'] - df_time_axis['start'].shift(1)) > pd.Timedelta(days=1.0)
                block_id_time = (mode_changed_time | is_large_time_gap).cumsum()
                
                for _, group in df_time_axis.groupby(block_id_time):
                    current_mode = group[mode_col].iloc[0].lower()
                    for key, style in calib_styles.items():
                        if key in current_mode:
                            s_time = group['start'].iloc[0]
                            e_time = group['end'].iloc[-1] if 'end' in group.columns and pd.notnull(group['end'].iloc[-1]) else group['start'].iloc[-1]
                            delta = e_time - s_time
                            if delta < pd.Timedelta(days=4):
                                mid_t = s_time + delta / 2
                                s_time, e_time = mid_t - pd.Timedelta(days=2), mid_t + pd.Timedelta(days=2)
                            time_intervals_dict[style['label']].append((s_time, e_time))
                            break
        
        BG_ALPHA = 0.35 
        plotted_labels = set()
        legend_handles = []

        for label, intervals in run_intervals_dict.items():
            if intervals:
                merged_runs = merge_intervals(intervals)
                color = color_map[label]
                plotted_labels.add(label)
                legend_handles.append(mpatches.Patch(facecolor=color, edgecolor='black', linewidth=1.0, alpha=BG_ALPHA, label=label))
                for s, e in merged_runs:
                    ax1.axvspan(s, e, facecolor=color, edgecolor='none', alpha=BG_ALPHA, zorder=0)

        for label, intervals in time_intervals_dict.items():
            if intervals:
                for s, e in merge_intervals(intervals):
                    ax2.axvspan(s, e, facecolor=color_map[label], edgecolor='none', alpha=BG_ALPHA, zorder=0)

        x_all_run = df_sorted['number'].astype(int)
        y_all = df_sorted['quality_score']
        ax1.plot(x_all_run, y_all, alpha=0.4, color='gray', linewidth=1.5, zorder=1)
        
        scatter1 = ax1.scatter(df_normal['number'].astype(int), df_normal['quality_score'], 
                               c=df_normal['quality_score'], cmap='RdYlGn', s=45, edgecolors='k', linewidth=0.5, alpha=0.9, zorder=2)
        if not df_bad.empty:
            ax1.scatter(df_bad['number'].astype(int), df_bad['quality_score'], 
                        c='#e74c3c', marker='X', s=120, edgecolors='black', linewidth=1.0, alpha=1.0, zorder=3, label="Consensus Bad")
        
        title_pad = 55 if plotted_labels else 20
        ax1.set_title(title, pad=title_pad, fontsize=22)
        ax1.set_xlabel("Run Number", fontsize=16, labelpad=15, weight='bold')
        ax1.set_ylabel("Global Quality Score", fontsize=16, weight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
        
        if 'start' in df_sorted.columns:
            df_time_sorted = df_sorted.sort_values('start')
            df_time_normal = df_time_sorted[~df_time_sorted['is_consensus_bad']]
            df_time_bad = df_time_sorted[df_time_sorted['is_consensus_bad']]
            
            x_time_all = df_time_sorted['start']
            ax2.plot(x_time_all, df_time_sorted['quality_score'], alpha=0.4, color='gray', linewidth=1.5, zorder=1)
            ax2.scatter(df_time_normal['start'], df_time_normal['quality_score'], 
                        c=df_time_normal['quality_score'], cmap='RdYlGn', s=45, edgecolors='k', linewidth=0.5, alpha=0.9, zorder=2)
            
            if not df_time_bad.empty:
                ax2.scatter(df_time_bad['start'], df_time_bad['quality_score'], 
                            c='#e74c3c', marker='X', s=120, edgecolors='black', linewidth=1.0, alpha=1.0, zorder=3)
            
            ax2.set_xlabel("Time (Year-Month)", fontsize=16, labelpad=15, weight='bold')
            ax2.set_ylabel("Global Quality Score", fontsize=16, weight='bold')
            ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)
            ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.get_xticklabels(), rotation=0, ha='center', weight='bold', fontsize=14)

        if not df_bad.empty:
            legend_handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c', 
                                             markeredgecolor='black', markersize=12, label='Consensus Bad'))

        if legend_handles:
            ax1.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.18), 
                       ncol=min(len(legend_handles), 6), framealpha=1.0, edgecolor='black', 
                       fontsize=14, fancybox=False)

        cbar = fig.colorbar(scatter1, cax=cax)
        cbar.set_label('Average Window Quality Score', weight='bold', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        cbar.outline.set_linewidth(3.0)   
        cbar.outline.set_edgecolor('black') 
        
        for ax in [ax1, ax2, cax]:
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_color('black')
            
        plt.savefig(f"{save_prefix}_trend3.png", bbox_inches='tight')
        plt.savefig(f"{save_prefix}_trend3.pdf", bbox_inches='tight')
        print(f"📊 Consensus trend plots saved to: {save_prefix}_trend3.[png|pdf]")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SR2 Detector Consensus Quality Evaluation Tool")
    parser.add_argument("--run_info", type=str, required=True, help="Path to Run Info CSV")
    parser.add_argument("--rates", type=str, required=True, help="Path to Rates CSV")
    parser.add_argument("--output", type=str, default="results/sr2_quality_consensus.h5")
    parser.add_argument("--plot_dir", type=str, default="results/plots")
    
    parser.add_argument("--windows", type=int, nargs='+', default=[1, 2, 4, 8, 10], help="List of window sizes for evaluation")
    parser.add_argument("--k_mad", type=float, default=4.5, help="Multiplier for MAD threshold calculation (4.5 approx 3-sigma)")
    parser.add_argument("--vote_ratio", type=float, default=0.5, help="Fraction of windows a run must fail to be flagged bad")
    
    parser.add_argument("--plot", action="store_true", help="Generate overall trend plots")
    parser.add_argument("--start_date", type=str, default="")
    parser.add_argument("--end_date", type=str, default="")
    parser.add_argument("--analyze_bad", action="store_true", help="Generate detailed diagnostic plots for consensus bad runs")

    args = parser.parse_args()
    
    if args.plot or args.analyze_bad:
        os.makedirs(args.plot_dir, exist_ok=True)

    analyzer = SR2QualityAnalyzer(args.run_info, args.rates)
    df_base = analyzer.load_and_merge_data()

    if args.start_date:
        start_dt = pd.to_datetime(args.start_date)
        df_base = df_base[df_base['start'] >= start_dt]
    if args.end_date:
        end_dt = pd.to_datetime(args.end_date + " 23:59:59")
        df_base = df_base[df_base['start'] <= end_dt]
        
    if len(df_base) == 0:
        print("❌ Error: No data left after applying the date filter.")
        sys.exit(1)
        
    analyzer.df = df_base.reset_index(drop=True)

    # ---------------------------------------------------------
    # Core Consensus Evaluation
    # ---------------------------------------------------------
    df_evaluated = analyzer.calculate_consensus_quality(
        window_sizes=args.windows, 
        k_mad=args.k_mad, 
        vote_ratio=args.vote_ratio
    )
    
    analyzer.save_results(df_evaluated, args.output)
    
    # ---------------------------------------------------------
    # Visualization & Diagnostics
    # ---------------------------------------------------------
    time_suffix = f" ({args.start_date} to {args.end_date})" if args.start_date or args.end_date else ""
    
    if args.plot:
        plot_prefix = os.path.join(args.plot_dir, "consensus_run")
        analyzer.plot_quality_trend(df_evaluated, f"Consensus Model Quality Trend{time_suffix}", save_prefix=plot_prefix)
        
        mode_col = 'mode'
        if analyzer.df_raw_info is not None and len(analyzer.df_raw_info.columns) > 1:
            mode_col = analyzer.df_raw_info.columns[1]
            
        if mode_col in df_evaluated.columns:
            science_mask = df_evaluated[mode_col].astype(str).str.lower().str.contains('bkg|background|science', na=False)
            df_science_only = df_evaluated[science_mask].copy()
            
            if not df_science_only.empty:
                print(f"Generating exclusive Science-Only Trend plot...")
                analyzer.plot_quality_trend(
                    df_science_only, 
                    f"Science Runs Quality Trend{time_suffix}", 
                    save_prefix=f"{plot_prefix}_science_only"
                )

    if args.analyze_bad:
        anomaly_prefix = os.path.join(args.plot_dir, "consensus_anomaly")
        analyzer.analyze_consensus_anomalies(df_evaluated, window_sizes=args.windows, save_prefix=anomaly_prefix)