#!/usr/bin/env python
"""
Compute per-run event rates for XENONnT SR2 LowER analysis.

Processes a single run through the XENONnT strax framework, extracts event_info
(and event_shadow if available), then computes livetime-normalized rates for
five physics event categories: Gate, Cathode, S1-only, S2-only, and Wall events.

Output is appended atomically (with file locking) to a shared CSV, making this
safe for parallel Slurm job-array execution.

Usage:
    python compute_event_rates.py -r 054585
    python compute_event_rates.py -r 054585 -c /path/to/run_info.csv -o output.csv
"""

import argparse
import fcntl
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Suppress verbose warnings in batch environments
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cutax
    HAS_CUTAX = True
except ImportError:
    HAS_CUTAX = False

import strax
import straxen


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_run_metadata(csv_path, run_id):
    """Retrieve livetime (seconds) and start date for a given run from the CSV.

    Parameters
    ----------
    csv_path : str
        Path to the run-tagging info CSV.
    run_id : str
        Zero-padded 6-digit run identifier.

    Returns
    -------
    livetime : float
        Run livetime in seconds.
    start_date : str
        Run start timestamp string.
    """
    if not os.path.exists(csv_path):
        sys.exit(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Detect the run-ID column
    run_col = next(
        (c for c in ['number', 'name', 'run_id', 'RunID'] if c in df.columns),
        None,
    )
    if run_col is None:
        sys.exit("No run-ID column found in CSV.")

    df[run_col] = df[run_col].astype(str).str.zfill(6)
    row = df[df[run_col] == run_id]
    if row.empty:
        sys.exit(f"Run {run_id} not found in CSV.")

    # Livetime
    val = row['livetime'].values[0]
    try:
        livetime = pd.to_timedelta(val).total_seconds()
    except (ValueError, TypeError):
        livetime = float(val)

    # Start date
    start_date = row['start'].values[0] if 'start' in df.columns else "Unknown"

    return livetime, start_date


# ---------------------------------------------------------------------------
# Event-rate computation
# ---------------------------------------------------------------------------

def compute_rates(run_id, csv_path, output_path):
    """Main processing: load strax data, apply physics masks, write results.

    Parameters
    ----------
    run_id : str
        Zero-padded 6-digit run ID.
    csv_path : str
        Path to the run info CSV (for livetime / metadata).
    output_path : str
        Destination CSV path (appended with file locking).
    """
    livetime, start_date = load_run_metadata(csv_path, run_id)

    # --- Strax context ------------------------------------------------------
    if HAS_CUTAX:
        st = cutax.contexts.xenonnt_offline()
    else:
        st = straxen.contexts.xenonnt_online()

    st.storage += [
        strax.DataDirectory(p, readonly=True)
        for p in [
            "/project2/lgrandi/xenonnt/processed/",
            "/project/lgrandi/xenonnt/processed/",
        ]
    ]

    # --- Load data ----------------------------------------------------------
    targets = ['event_info']
    if st.is_stored(run_id, 'event_shadow'):
        targets.append('event_shadow')

    try:
        df = st.get_df(run_id, targets=tuple(targets))
    except Exception as exc:
        sys.exit(f"Failed to load strax data for run {run_id}: {exc}")

    # Derived quantities
    df['r2'] = df['x'] ** 2 + df['y'] ** 2
    s1_raw = df['s1_area'].fillna(0)
    s2_raw = df['s2_area'].fillna(0)

    # Physics category masks
    categories = {
        'Gate_Event': {
            'mask': (df['drift_time'] > 0) & (df['drift_time'] < 8e3),
        },
        'Cathode_Event': {
            'mask': (
                ((df['drift_time'] > 1.8e6) & (df['drift_time'] < 2.5e6))
                | ((df['z'] > -150) & (df['z'] < -145))
                | ((s1_raw > 1000) & (s2_raw < 200))
            ),
        },
        'S1_Only_Heavy': {
            'mask': (s1_raw < 100) & (s2_raw < 100),
        },
        'S2_Only_SE': {
            'mask': (s1_raw < 10) & (s2_raw < 200),
        },
        'Wall_Event': {
            'mask': (df['r2'] > 3800),
        },
    }

    # Build result record
    record = {'Run_ID': run_id, 'Start_Date': start_date, 'Livetime_sec': livetime}
    print(f"\n[{run_id}]  Processing category rates ...")

    for name, cfg in categories.items():
        count = int(cfg['mask'].sum())
        rate = count / livetime
        record[f'{name}_Count'] = count
        record[f'{name}_Rate_Hz'] = rate
        print(f"  - {name:<15}: {rate:10.4f} Hz  (N = {count})")

    # --- Atomic append to shared CSV ---------------------------------------
    df_out = pd.DataFrame([record])
    output_path = os.path.abspath(output_path)

    max_retries = 15
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(1.0, 5.0))

            file_exists = os.path.isfile(output_path)
            with open(output_path, 'a') as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                df_out.to_csv(fh, header=not file_exists, index=False)
                fh.flush()
                os.fsync(fh.fileno())
                fcntl.flock(fh, fcntl.LOCK_UN)

            print(f"\nDone. Output appended to {output_path}")
            break
        except Exception as io_err:
            print(
                f"  [Attempt {attempt + 1}/{max_retries}] Locked or busy "
                f"for run {run_id}: {io_err}"
            )
            if attempt == max_retries - 1:
                sys.exit(
                    f"FATAL: Failed to write run {run_id} "
                    f"after {max_retries} attempts."
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-run event rates for XENONnT SR2 LowER"
    )
    parser.add_argument(
        '-r', '--run-id', type=str, required=True, help="Run ID (e.g. 054585)"
    )
    parser.add_argument(
        '-c', '--csv-path', type=str,
        default=(
            '/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/'
            'data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv'
        ),
        help="Path to the run-info CSV with livetime and metadata",
    )
    parser.add_argument(
        '-o', '--output', type=str,
        default='sr2_master_run_rates.csv',
        help="Output CSV path (appended atomically)",
    )
    args = parser.parse_args()

    run_id = args.run_id.zfill(6)
    compute_rates(run_id, args.csv_path, args.output)


if __name__ == "__main__":
    main()
