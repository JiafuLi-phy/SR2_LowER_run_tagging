#!/usr/bin/env python
"""
Compute per-run detector quality scores via Isolation Forest anomaly detection.

Merges the run-information database with event-rate measurements, extracts
physical features, and trains an Isolation Forest to assign each run a 0-100
quality score.  Supports three scoring modes:

  single    One IF model on per-run features (fast, independent).
  batch     Rolling-window mean features → IF (captures local context).
  consensus Multi-window voting with MAD-based threshold (most robust).

Also flags anomalous runs and exports summary CSV tables.

Usage:
    # Single-run scoring (default)
    python compute_quality_scores.py \
        --run-info sr2_run_tagging_info.csv --rates sr2_master_run_rates.csv

    # Consensus voting across multiple window sizes
    python compute_quality_scores.py --mode consensus \
        --run-info sr2_run_tagging_info.csv --rates sr2_master_run_rates.csv \
        --windows 1 2 4 8 10 --k-mad 4.5 --vote-ratio 0.5

    # With date filtering
    python compute_quality_scores.py \
        --run-info ... --rates ... \
        --start-date 2024-01-01 --end-date 2024-06-30
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Quality analyzer
# ---------------------------------------------------------------------------

class QualityAnalyzer:
    """Merge run metadata with event rates and compute Isolation Forest scores.

    Parameters
    ----------
    run_info_path : str
        Path to the run-tagging info CSV.
    rates_path : str
        Path to the master event-rates CSV.
    deadtime_path : str or None
        Optional path to deadtime CSV (backward-compatible with older data).
    """

    def __init__(self, run_info_path, rates_path, deadtime_path=None):
        self.run_info_path = run_info_path
        self.rates_path = rates_path
        self.deadtime_path = deadtime_path
        self.df = None           # merged dataset (only runs with rates)
        self.df_raw = None       # full run_info (all runs, including calibrations)
        self.feature_cols = []
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_and_merge(self):
        """Load CSVs, standardize column names, merge on run number."""
        print(f"Loading data ...\n"
              f"  Run info : {self.run_info_path}\n"
              f"  Rates    : {self.rates_path}")
        if self.deadtime_path:
            print(f"  Deadtime : {self.deadtime_path}")

        df_info = pd.read_csv(self.run_info_path)
        df_rates = pd.read_csv(self.rates_path)

        # Standardize run-ID column to 'number'
        for df_tmp in [df_info, df_rates]:
            if 'number' not in df_tmp.columns:
                for candidate in ['name', 'run_id', 'RunID', 'Run_ID']:
                    if candidate in df_tmp.columns:
                        df_tmp.rename(columns={candidate: 'number'}, inplace=True)
                        break

        df_info['number'] = df_info['number'].astype(str).str.zfill(6)
        df_rates['number'] = df_rates['number'].astype(str).str.zfill(6)

        # Parse timestamps
        if 'start' in df_info.columns and 'end' in df_info.columns:
            df_info['start'] = pd.to_datetime(df_info['start'], errors='coerce')
            df_info['end'] = pd.to_datetime(df_info['end'], errors='coerce')

        # Merge deadtime if provided
        if self.deadtime_path:
            df_dt = pd.read_csv(self.deadtime_path)
            if 'number' not in df_dt.columns:
                for candidate in ['name', 'run_id', 'RunID', 'Run_ID']:
                    if candidate in df_dt.columns:
                        df_dt.rename(columns={candidate: 'number'}, inplace=True)
                        break
            df_dt['number'] = df_dt['number'].astype(str).str.zfill(6)
            df_info = pd.merge(df_info, df_dt, on='number', how='inner',
                               suffixes=('', '_dt'))

        self.df_raw = df_info.copy()
        self.df = pd.merge(df_info, df_rates, on='number', how='inner',
                           suffixes=('', '_rates'))

        # Detect mode column from raw info (second column by convention)
        self.mode_col = (
            self.df_raw.columns[1]
            if self.df_raw is not None and len(self.df_raw.columns) > 1
            else 'mode'
        )

        print(f"  Runs in info DB : {len(df_info)}")
        print(f"  Runs with rates : {len(self.df)}")
        if len(df_info) > len(self.df):
            missing = set(df_info['number']) - set(df_rates['number'])
            print(f"  Dropped (no rates): {len(missing)}")
            if self.mode_col in df_info.columns:
                ar37 = df_info[
                    df_info['number'].isin(missing)
                    & df_info[self.mode_col].astype(str).str.contains(
                        'ar37', case=False, na=False
                    )
                ]
                if len(ar37) > 0:
                    print(f"    └ {len(ar37)} Ar-37 runs missing from rates")

        return self.df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    EXCLUDE_KEYWORDS = [
        'number', 'run_id', 'id',
        'time', 'duration',
        'count',
        'x_bin', 'y_bin',
        'peak_positions', 'peak_basics',
    ]

    def extract_features(self, df):
        """Select numeric columns suitable for ML, dropping metadata columns.

        Returns a list of valid column names with non-zero variance.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [
            col for col in numeric_cols
            if not any(kw in col.lower() for kw in self.EXCLUDE_KEYWORDS)
        ]
        df_filled = df[features].fillna(df[features].median())
        valid = df_filled.std()
        valid = valid[valid > 0].index.tolist()
        print(f"  Selected {len(valid)} physical features for IF training")
        return valid

    # ------------------------------------------------------------------
    # Core IF training
    # ------------------------------------------------------------------

    def _fit_if(self, data):
        """Train Isolation Forest and return 0-100 normalized scores."""
        X = data.fillna(data.median())
        X_scaled = self.scaler.fit_transform(X)
        model = IsolationForest(
            n_estimators=150, contamination='auto', random_state=42,
        )
        model.fit(X_scaled)
        raw = model.decision_function(X_scaled)
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            return 100.0 * (raw - lo) / (hi - lo)
        return np.full(len(raw), 100.0)

    # ------------------------------------------------------------------
    # Scoring modes
    # ------------------------------------------------------------------

    def score_single(self):
        """Per-run IF scoring (no temporal context)."""
        self.feature_cols = self.extract_features(self.df)
        print("Computing single-run quality scores ...")
        self.df['quality_score'] = self._fit_if(self.df[self.feature_cols])
        return self.df

    def score_batch(self, window_size=10):
        """Rolling-window IF scoring using window-mean features."""
        if not self.feature_cols:
            self.feature_cols = self.extract_features(self.df)
        print(f"Computing batch quality scores (window N={window_size}) ...")
        rolled = self.df[self.feature_cols].rolling(
            window=window_size, min_periods=1, center=True,
        ).mean()
        self.df['quality_score'] = self._fit_if(rolled)
        return self.df

    def score_consensus(self, window_sizes=(1, 2, 4, 8, 10),
                        k_mad=4.5, vote_ratio=0.5):
        """Multi-window consensus scoring with MAD-based anomaly threshold.

        Each window size gets its own IF model.  A run is flagged as anomalous
        if the fraction of windows voting "bad" exceeds *vote_ratio*.

        Parameters
        ----------
        window_sizes : tuple of int
            Rolling-mean window sizes for the ensemble.
        k_mad : float
            Multiplier on the median absolute deviation for the cut threshold.
            Default 4.5 approximates a 3-sigma cut for Gaussian residuals.
        vote_ratio : float
            Fraction of windows that must flag a run for it to be marked bad.
        """
        if not self.feature_cols:
            self.feature_cols = self.extract_features(self.df)

        print(f"\nConsensus scoring across windows: {list(window_sizes)}")
        print(f"  k_MAD = {k_mad}, vote_ratio = {vote_ratio}")

        df = self.df.copy()
        flag_cols = []
        score_cols = []

        for w in window_sizes:
            X_rolled = df[self.feature_cols].rolling(
                window=w, min_periods=1, center=True,
            ).mean()
            score_col = f'score_w{w}'
            df[score_col] = self._fit_if(X_rolled)
            score_cols.append(score_col)

            scores = df[score_col].values
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            mad = max(mad, 1e-6)
            threshold = median - k_mad * mad

            flag_col = f'is_anomaly_w{w}'
            df[flag_col] = df[score_col] < threshold
            flag_cols.append(flag_col)

            n_bad = df[flag_col].sum()
            print(f"  Window {w:>2}: median={median:6.1f}  MAD={mad:5.1f}  "
                  f"cut={threshold:6.1f}  → {n_bad:>4} flagged")

        # Voting
        df['anomaly_votes'] = df[flag_cols].sum(axis=1)
        required = int(len(window_sizes) * vote_ratio) + 1
        df['is_consensus_bad'] = df['anomaly_votes'] >= required
        df['quality_score'] = df[score_cols].mean(axis=1)

        n_bad = df['is_consensus_bad'].sum()
        print(f"\n  Consensus result: {n_bad} runs flagged as anomalous "
              f"(≥ {required} / {len(window_sizes)} windows)")

        self.df = df
        return df

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, df, path):
        """Save dataframe to HDF5 (.h5) or CSV depending on file extension."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if path.endswith('.h5'):
            print(f"Saving HDF5 → {path}")
            cols = df.select_dtypes(
                include=[np.number, 'bool', 'object', 'datetime'],
            ).columns
            clean = df[cols].copy()
            for c in clean.select_dtypes(include=['object']).columns:
                clean[c] = clean[c].astype(str)
            with pd.HDFStore(path, mode='w') as store:
                store.put('run_data', clean, format='table', data_columns=True)
        else:
            print(f"Saving CSV → {path}")
            df.to_csv(path, index=False)

    def export_anomaly_list(self, df, threshold=20.0, output_dir="results"):
        """Export a ranked list of anomalous runs to CSV."""
        bad = df[df['quality_score'] < threshold].copy()
        if bad.empty:
            print("No runs below the anomaly threshold.")
            return None

        cols = ['number', 'quality_score']
        if 'start' in bad.columns:
            cols.append('start')
        if self.mode_col in bad.columns:
            cols.append(self.mode_col)
        if 'is_consensus_bad' in bad.columns:
            cols.append('anomaly_votes')
            cols.append('is_consensus_bad')

        available = [c for c in cols if c in bad.columns]
        summary = bad[available].sort_values('quality_score')

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "anomalous_runs.csv")
        summary.to_csv(csv_path, index=False)
        print(f"\n{len(bad)} anomalous runs exported to {csv_path}")

        # Print summary by mode
        if self.mode_col in bad.columns:
            print("\nAnomalous runs by type:")
            for mode, count in bad[self.mode_col].value_counts().items():
                print(f"  {mode:<20} {count:>4}")

        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SR2 LowER detector quality scoring via Isolation Forest",
    )
    parser.add_argument(
        '--run-info', type=str, required=True,
        help="Path to run-tagging info CSV",
    )
    parser.add_argument(
        '--rates', type=str, required=True,
        help="Path to master event-rates CSV",
    )
    parser.add_argument(
        '--deadtime', type=str, default=None,
        help="Optional path to deadtime CSV (backward-compatible)",
    )
    parser.add_argument(
        '--output', type=str, default='results/sr2_quality.h5',
        help="Output path (.h5 or .csv)",
    )
    parser.add_argument(
        '--mode', type=str, default='single',
        choices=['single', 'batch', 'consensus'],
        help="Scoring mode (default: single)",
    )
    # Batch / consensus parameters
    parser.add_argument(
        '--batch-n', type=int, default=10,
        help="Window size for batch mode",
    )
    parser.add_argument(
        '--windows', type=int, nargs='+', default=[1, 2, 4, 8, 10],
        help="Window sizes for consensus mode",
    )
    parser.add_argument(
        '--k-mad', type=float, default=4.5,
        help="MAD multiplier for consensus anomaly cut",
    )
    parser.add_argument(
        '--vote-ratio', type=float, default=0.5,
        help="Fraction of windows required to flag a run as bad",
    )
    # Filtering
    parser.add_argument(
        '--start-date', type=str, default='',
        help="Filter: earliest date (YYYY-MM-DD)",
    )
    parser.add_argument(
        '--end-date', type=str, default='',
        help="Filter: latest date (YYYY-MM-DD)",
    )
    # Anomaly export
    parser.add_argument(
        '--anomaly-threshold', type=float, default=20.0,
        help="Quality-score threshold for anomaly list export",
    )
    parser.add_argument(
        '--export-anomalies', action='store_true',
        help="Export ranked list of anomalous runs to CSV",
    )
    args = parser.parse_args()

    analyzer = QualityAnalyzer(args.run_info, args.rates, args.deadtime)
    df_base = analyzer.load_and_merge()

    # Date filtering
    if args.start_date:
        start_dt = pd.to_datetime(args.start_date)
        df_base = df_base[df_base['start'] >= start_dt]
    if args.end_date:
        end_dt = pd.to_datetime(args.end_date + " 23:59:59")
        df_base = df_base[df_base['start'] <= end_dt]
    if len(df_base) == 0:
        sys.exit("Error: no data remaining after date filter.")
    analyzer.df = df_base.reset_index(drop=True)

    # Score
    if args.mode == 'single':
        df_result = analyzer.score_single()
    elif args.mode == 'batch':
        df_result = analyzer.score_batch(window_size=args.batch_n)
    else:  # consensus
        df_result = analyzer.score_consensus(
            window_sizes=args.windows,
            k_mad=args.k_mad,
            vote_ratio=args.vote_ratio,
        )

    analyzer.save_results(df_result, args.output)

    if args.export_anomalies:
        analyzer.export_anomaly_list(df_result, threshold=args.anomaly_threshold)

    print("Done.")


if __name__ == "__main__":
    main()
