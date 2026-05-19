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
    'axes.linewidth': 3.0,       # Thicker axes borders
    'axes.edgecolor': 'black',   # Force solid black edge color
    'axes.labelsize': 16,        
    'axes.titlesize': 22,        
    'xtick.labelsize': 14,       
    'ytick.labelsize': 14,
    'xtick.major.width': 3.0,    # Thicker major tick marks
    'ytick.major.width': 3.0,
    'xtick.color': 'black',      # Black tick marks
    'ytick.color': 'black',
    'xtick.direction': 'in',     
    'ytick.direction': 'in',
    'figure.dpi': 300
})

class SR2QualityAnalyzer:
    def __init__(self, run_info_path, deadtime_path, rates_path):
        self.run_info_path = run_info_path
        self.deadtime_path = deadtime_path
        self.rates_path = rates_path
        self.df = None
        self.feature_cols = []
        self.scaler = StandardScaler()

    def load_and_merge_data(self):
        print(f"Loading data...\n - Info: {self.run_info_path}\n - Deadtime: {self.deadtime_path}\n - Rates: {self.rates_path}")
        
        df_run = pd.read_csv(self.run_info_path)
        df_dt = pd.read_csv(self.deadtime_path)
        df_rates = pd.read_csv(self.rates_path)
        
        # Standardize run ID column names
        for df_tmp in [df_run, df_dt, df_rates]:
            if 'number' not in df_tmp.columns:
                for possible_name in ['name', 'run_id', 'RunID', 'Run_ID']:
                    if possible_name in df_tmp.columns:
                        df_tmp.rename(columns={possible_name: 'number'}, inplace=True)
                        break
        
        df_run['number'] = df_run['number'].astype(str).str.zfill(6)
        df_dt['number'] = df_dt['number'].astype(str).str.zfill(6)
        df_rates['number'] = df_rates['number'].astype(str).str.zfill(6)
        
        self.df = pd.merge(df_run, df_dt, on='number', how='inner', suffixes=('', '_dt'))
        self.df = pd.merge(self.df, df_rates, on='number', how='inner', suffixes=('', '_rates'))
        
        if 'start' in self.df.columns and 'end' in self.df.columns:
            self.df['start'] = pd.to_datetime(self.df['start'])
            self.df['end'] = pd.to_datetime(self.df['end'])
            
        print(f"Data merged. Total overlapping records: {len(self.df)}")
        return self.df

    def _extract_machine_learning_features(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        exclude_keywords = [
            'number', 'Run_ID', 'id',              
            'time', 'duration',                    
            'Count',                               
            'x_bin', 'y_bin',                      
            'peak_positions', 'peak_basics'        
        ]
        
        features = []
        for col in numeric_cols:
            is_blacklisted = any(kw.lower() in col.lower() for kw in exclude_keywords)
            if not is_blacklisted:
                features.append(col)
                
        df_features = df[features].fillna(df[features].median())
        std_check = df_features.std()
        valid_features = std_check[std_check > 0].index.tolist()
        
        print(f"Selected {len(valid_features)} physical features for ML modeling.")
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

    def calculate_single_quality(self):
        self.feature_cols = self._extract_machine_learning_features(self.df)
        print("Calculating [Independent Single Run Quality Score]...")
        self.df['quality_score'] = self._train_logic(self.df[self.feature_cols])
        return self.df

    def calculate_batch_quality(self, n):
        if not self.feature_cols:
            self.feature_cols = self._extract_machine_learning_features(self.df)
        print(f"Calculating [Rolling Batch Quality Score] with window size N={n}...")
        batch_df = self.df.copy()
        X_batch = batch_df[self.feature_cols].rolling(window=n, min_periods=1, center=True).mean()
        batch_df['quality_score'] = self._train_logic(X_batch)
        return batch_df

    def save_results(self, df_to_save, base_path):
        directory = os.path.dirname(base_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        if base_path.endswith('.h5'):
            print(f"Saving HDF5: {base_path}")
            cols_to_keep = df_to_save.select_dtypes(include=[np.number, 'bool', 'object', 'datetime']).columns
            df_clean = df_to_save[cols_to_keep].copy()
            for c in df_clean.select_dtypes(include=['object']).columns:
                df_clean[c] = df_clean[c].astype(str)
            with pd.HDFStore(base_path, mode='w') as store:
                store.put('run_data', df_clean, format='table', data_columns=True)
        else:
            print(f"Saving CSV: {base_path}")
            df_to_save.to_csv(base_path, index=False)

    # ==========================================
    # UPDATED: Bad Run Analysis & Diagnostics (Vibrant & High Contrast)
    # ==========================================
    def analyze_anomalies(self, df, threshold=20.0, save_prefix=""):
        print(f"\n--- Analyzing Anomalous Runs (Quality Score < {threshold}) ---")
        bad_runs = df[df['quality_score'] < threshold].copy()
        
        if bad_runs.empty:
            print("✅ No runs fall below the anomaly threshold.")
            return

        print(f"⚠️ Found {len(bad_runs)} anomalous runs out of {len(df)} total.")

        # 1. Dynamically extract the second column name from the run_info CSV (for 'mode')
        try:
            run_info_headers = pd.read_csv(self.run_info_path, nrows=0).columns
            mode_col = run_info_headers[1] if len(run_info_headers) > 1 else 'mode'
            print(f"🔍 Extracted 'mode' representation from column 2: '{mode_col}'")
        except Exception as e:
            print(f"⚠️ Could not read 2nd column from {self.run_info_path}. Defaulting to 'mode'. Error: {e}")
            mode_col = 'mode'
                
        # 2. Export Bad Runs List (Adding the extracted mode column)
        export_cols = ['number', 'quality_score']
        if 'start' in bad_runs.columns: export_cols.append('start')
        if mode_col in bad_runs.columns: export_cols.append(mode_col)
        
        bad_runs_summary = bad_runs[export_cols].sort_values('quality_score')
        csv_path = f"{save_prefix}_bad_runs_list.csv"
        bad_runs_summary.to_csv(csv_path, index=False)
        print(f"📄 Saved bad runs list (with '{mode_col}') to: {csv_path}")

        # 3. Plot Anomalous Run Types Distribution (Vibrant & High Contrast)
        if mode_col in bad_runs.columns:
            type_counts = bad_runs[mode_col].value_counts()
            print(f"\nAnomalous Runs by Data Type ({mode_col}):")
            print(type_counts.to_string())
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Using a highly vibrant palette (Set1) for distinct categorical differences
            sns.barplot(x=type_counts.index, y=type_counts.values, 
                        palette='Set1', edgecolor='black', linewidth=2.5, ax=ax)
            
            # Add value annotations on top of each bar for clarity
            for i, v in enumerate(type_counts.values):
                ax.text(i, v + 0.02 * max(type_counts.values), str(v), 
                        ha='center', va='bottom', fontweight='bold', fontsize=16, color='black')
                
            ax.set_title(f"Distribution of Anomalous Runs by Type ({mode_col})", pad=20, fontsize=20, weight='bold')
            ax.set_ylabel("Number of Anomalous Runs", fontsize=16, weight='bold')
            ax.set_xlabel(f"Run {mode_col.capitalize()}", fontsize=16, weight='bold')
            plt.xticks(rotation=30, ha='right', fontsize=14, weight='bold')
            
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_color('black')
                
            plt.savefig(f"{save_prefix}_bad_run_types.png", bbox_inches='tight')
            plt.close()
        else:
            print(f"\n⚠️ Column '{mode_col}' not found in merged data. Skipping type distribution plot.")

        # 4. Analyze which variables are abnormal across the bad runs
        print("\nCalculating global feature deviations for anomalous runs...")
        baseline_mean = df[self.feature_cols].mean()
        baseline_std = df[self.feature_cols].std().replace(0, 1e-9)
        z_scores = (bad_runs[self.feature_cols] - baseline_mean) / baseline_std
        
        # Rank features by average absolute deviation
        mean_abs_z = z_scores.abs().mean().sort_values(ascending=False)
        top_15_features = mean_abs_z.head(15).index.tolist()
        
        # 5. Plot Boxplot of Top 15 Abnormal Variables (Vibrant, Clear Outliers)
        plot_data = z_scores[top_15_features]
        fig, ax = plt.subplots(figsize=(16, 11))
        
        # Scientific & Vibrant Aesthetics for Boxplot
        flierprops = dict(marker='D', markerfacecolor='#e74c3c', markersize=8, markeredgecolor='black', alpha=0.9)
        medianprops = dict(color='#f1c40f', linewidth=3.5) # Distinct yellow median line
        boxprops = dict(edgecolor='black', linewidth=2.5)
        whiskerprops = dict(color='black', linewidth=2.5)
        capprops = dict(color='black', linewidth=2.5)
        
        sns.boxplot(data=plot_data, orient='h', ax=ax, palette='Spectral', 
                    flierprops=flierprops, boxprops=boxprops, 
                    medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
        
        # Thicker baseline and reference lines
        ax.axvline(0, color='black', linewidth=4.0)
        ax.axvline(3, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.8) # Red-ish warning line
        ax.axvline(-3, color='#2980b9', linestyle='--', linewidth=2.5, alpha=0.8) # Blue-ish warning line
        
        ax.set_title("Top 15 Most Deviating Features in Anomalous Runs\n(Z-Score distribution relative to all runs)", pad=20, fontsize=22, weight='bold')
        ax.set_xlabel("Z-Score (Standard Deviations from Dataset Mean)", fontsize=18, weight='bold')
        ax.set_ylabel("Detector Features", fontsize=18, weight='bold')
        
        ax.xaxis.grid(True, linestyle='--', alpha=0.7, color='gray') # Added faint vertical grid lines
        ax.yaxis.grid(False) # Turn off horizontal grid lines to keep it clean
        
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_color('black')
            
        plt.savefig(f"{save_prefix}_bad_run_features_boxplot.png", bbox_inches='tight')
        plt.savefig(f"{save_prefix}_bad_run_features_boxplot.pdf", bbox_inches='tight')
        plt.close()
        print(f"📊 Bad run feature diagnostic plots saved to: {save_prefix}_bad_run_features_boxplot.[png|pdf]")
        print("-" * 50)


    def plot_quality_trend(self, df, title="Detector Quality Trend", save_prefix=""):
        fig = plt.figure(figsize=(20, 12)) 
        gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], hspace=0.2, wspace=0.02)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharey=ax1)
        cax = fig.add_subplot(gs[:, 1]) 
        
        df_sorted = df.sort_values('number') 
        y_data = df_sorted['quality_score']
        
        # --- Top Panel: Run Number ---
        x_run = df_sorted['number'].astype(int)
        scatter1 = ax1.scatter(x_run, y_data, c=y_data, cmap='RdYlGn', s=45, edgecolors='k', linewidth=0.5, alpha=0.9, zorder=2)
        ax1.plot(x_run, y_data, alpha=0.4, color='gray', linewidth=1.5, zorder=1)
        
        ax1.set_title(title, pad=20, fontsize=22)
        ax1.set_xlabel("Run Number", fontsize=16)
        ax1.set_ylabel("Quality Score (0-100)", fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
        
        # --- Bottom Panel: Time (Start Date) ---
        if 'start' in df_sorted.columns:
            df_time_sorted = df_sorted.sort_values('start')
            x_time = df_time_sorted['start']
            y_time = df_time_sorted['quality_score']
            
            ax2.scatter(x_time, y_time, c=y_time, cmap='RdYlGn', s=45, edgecolors='k', linewidth=0.5, alpha=0.9, zorder=2)
            ax2.plot(x_time, y_time, alpha=0.4, color='gray', linewidth=1.5, zorder=1)
            
            ax2.set_xlabel("Time (Year-Month)", fontsize=16)
            ax2.set_ylabel("Quality Score (0-100)", fontsize=16)
            ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)
            
            ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            plt.setp(ax2.get_xticklabels(), rotation=0, ha='center', weight='bold', fontsize=14)
        else:
            ax2.text(0.5, 0.5, "Time 'start' data not available", ha='center', va='center', fontsize=16)

        # --- Shared Elements ---
        cbar = fig.colorbar(scatter1, cax=cax)
        cbar.set_label('Quality Score', weight='bold', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        cbar.outline.set_linewidth(3.0)   
        cbar.outline.set_edgecolor('black') 
        
        for ax in [ax1, ax2, cax]:
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_color('black')
            
        plt.savefig(f"{save_prefix}.png", bbox_inches='tight')
        plt.savefig(f"{save_prefix}.pdf", bbox_inches='tight')
        print(f"📊 Dual-panel trend plots saved to: {save_prefix}.[png|pdf]")
        plt.close()

    def plot_run_diagnostic(self, df, run_id, save_prefix=""):
        run_id_str = str(run_id).zfill(6)
        row = df[df['number'] == run_id_str]
        if row.empty:
            print(f"Run {run_id_str} not found in the dataset.")
            return
        
        mean_values = df[self.feature_cols].mean()
        std_values = df[self.feature_cols].std().replace(0, 1e-9)
        
        z_scores = (row[self.feature_cols].iloc[0] - mean_values) / std_values
        top_deviations = z_scores.abs().sort_values(ascending=False).head(10)
        plot_data = z_scores[top_deviations.index]

        fig, ax = plt.subplots(figsize=(16, 7))
        colors = ['#B22222' if abs(x) > 3 else '#4682B4' for x in plot_data[::-1]]
        
        plot_data[::-1].plot(kind='barh', color=colors, edgecolor='black', linewidth=2.0, ax=ax)
        ax.axvline(0, color='black', linewidth=3.0) 
        ax.axvline(3, color='black', linestyle='--', linewidth=1.5, alpha=0.8) 
        ax.axvline(-3, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax.set_title(f"Diagnostic: Top Feature Deviations for Run {run_id_str}\n(Z-Score vs Average)", pad=20, fontsize=18)
        ax.set_xlabel("Standard Deviations from Mean (Z-Score)", fontsize=16)
        
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_color('black')
            
        plt.savefig(f"{save_prefix}_diag_{run_id_str}.png", bbox_inches='tight')
        plt.savefig(f"{save_prefix}_diag_{run_id_str}.pdf", bbox_inches='tight')
        print(f"📊 Diagnostic plots saved to: {save_prefix}_diag_{run_id_str}.[png|pdf]")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SR2 Detector Quality Refined Evaluation Tool")
    parser.add_argument("--run_info", type=str, required=True)
    parser.add_argument("--deadtime", type=str, required=True)
    parser.add_argument("--rates", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/sr2_quality_master.h5")
    parser.add_argument("--plot_dir", type=str, default="results/plots")
    parser.add_argument("--batch_n", type=int, default=10)
    parser.add_argument("--enable_batch", action="store_true")
    parser.add_argument("--inspect_id", type=str, default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--start_date", type=str, default="")
    parser.add_argument("--end_date", type=str, default="")
    
    # NEW ARGUMENTS FOR BAD RUN ANALYSIS
    parser.add_argument("--analyze_bad", action="store_true", help="Enable aggregate anomaly analysis for bad runs")
    parser.add_argument("--bad_threshold", type=float, default=20.0, help="Quality score threshold below which a run is considered 'bad'")

    args = parser.parse_args()
    
    if args.plot or args.inspect_id or args.analyze_bad:
        os.makedirs(args.plot_dir, exist_ok=True)

    analyzer = SR2QualityAnalyzer(args.run_info, args.deadtime, args.rates)
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

    df_single = analyzer.calculate_single_quality()
    analyzer.save_results(df_single, args.output)
    
    if args.plot:
        time_suffix = f" ({args.start_date} to {args.end_date})" if args.start_date or args.end_date else ""
        single_plot_prefix = os.path.join(args.plot_dir, "single_run_trend")
        analyzer.plot_quality_trend(df_single, f"Single Run Quality Trend{time_suffix}", save_prefix=single_plot_prefix)

    # Trigger Global Anomaly Analysis
    if args.analyze_bad:
        anomaly_prefix = os.path.join(args.plot_dir, "anomaly_analysis")
        # Explicitly passing the threshold, type_col_hint is removed as it's extracted automatically
        analyzer.analyze_anomalies(df_single, threshold=args.bad_threshold, save_prefix=anomaly_prefix)

    if args.enable_batch:
        batch_output = os.path.splitext(args.output)[0] + f"_batch_n{args.batch_n}{os.path.splitext(args.output)[1]}"
        df_batch = analyzer.calculate_batch_quality(args.batch_n)
        analyzer.save_results(df_batch, batch_output)
        
        if args.plot:
            batch_plot_prefix = os.path.join(args.plot_dir, f"batch_n{args.batch_n}_trend")
            analyzer.plot_quality_trend(df_batch, f"Batch Stability Trend (N={args.batch_n}){time_suffix}", save_prefix=batch_plot_prefix)

    if args.inspect_id:
        target_df = df_batch if args.enable_batch else df_single
        diag_prefix = os.path.join(args.plot_dir, "run")
        analyzer.plot_run_diagnostic(target_df, args.inspect_id, save_prefix=diag_prefix)