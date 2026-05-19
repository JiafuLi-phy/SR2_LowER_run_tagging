import argparse
import os
import sys
import warnings
import time
import random
import fcntl
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
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

# ==========================================
# Helper Functions
# ==========================================
def get_run_metadata(csv_path, run_id):
    """
    Retrieves livetime (converted to seconds) and the start date from the CSV.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Missing CSV: {csv_path}"); sys.exit(1)
    df_info = pd.read_csv(csv_path)
    
    # Find the correct run ID column
    run_col = next((c for c in ['number', 'name', 'run_id', 'RunID'] if c in df_info.columns), None)
    if not run_col: print("❌ No RunID column"); sys.exit(1)
    
    # Format and search for the run
    df_info[run_col] = df_info[run_col].astype(str).str.zfill(6)
    run_row = df_info[df_info[run_col] == run_id]
    if run_row.empty: print(f"❌ Run {run_id} not in CSV"); sys.exit(1)
    
    # 1. Extract Livetime
    val = run_row['livetime'].values[0]
    try:
        livetime = pd.to_timedelta(val).total_seconds()
    except:
        livetime = float(val)
        
    # 2. Extract Start Date (Column 'start')
    start_date = run_row['start'].values[0] if 'start' in df_info.columns else "Unknown"
    
    return livetime, start_date

# ==========================================
# Main Processing Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_id', type=str, required=True)
    parser.add_argument('-c', '--csv_path', type=str, default='/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv')
    parser.add_argument('-o', '--output', type=str, default='sr2_master_run_rates.csv')
    args = parser.parse_args()
    run_id = args.run_id.zfill(6)

    # 💡 Modified to unpack both livetime and start_date
    livetime, start_date = get_run_metadata(args.csv_path, run_id)

    if HAS_CUTAX:
        st = cutax.contexts.xenonnt_offline()
    else:
        st = straxen.contexts.xenonnt_online()
    
    st.storage += [strax.DataDirectory(p, readonly=True) for p in ["/project2/lgrandi/xenonnt/processed/", "/project/lgrandi/xenonnt/processed/"]]
    
    target_list = ['event_info']
    if st.is_stored(run_id, 'event_shadow'): target_list.append('event_shadow')
    
    try:
        df = st.get_df(run_id, targets=tuple(target_list))
    except Exception as e:
        print(f"❌ Failed: {e}"); sys.exit(1)

    # Derived Variables
    df['r2'] = df['x']**2 + df['y']**2
    s1_raw = df['s1_area'].fillna(0)
    s2_raw = df['s2_area'].fillna(0)

    # Categories Configuration
    configs = {
        'Gate_Event': {'mask': (df['drift_time'] > 0) & (df['drift_time'] < 8e3)},
        'Cathode_Event': {'mask': ((df['drift_time'] > 1.8e6) & (df['drift_time'] < 2.5e6)) | ((df['z'] > -150) & (df['z'] < -145)) | ((s1_raw > 1000) & (s2_raw < 200))},
        'S1_Only_Heavy': {'mask': (s1_raw < 100) & (s2_raw < 100)},
        'S2_Only_SE': {'mask': (s1_raw < 10) & (s2_raw < 200)},
        'Wall_Event': {'mask': (df['r2'] > 3800)}
    }

    # 💡 Added Start_Date as the second element in the dictionary
    run_data = {'Run_ID': run_id, 'Start_Date': start_date, 'Livetime_sec': livetime}
    print(f"\n[{run_id}] 📊 Processing Categories Rates...")

    for name, cfg in configs.items():
        df_sub = df[cfg['mask']]
        count = len(df_sub)
        rate = count / livetime
        
        run_data[f'{name}_Count'] = count
        run_data[f'{name}_Rate_Hz'] = rate
        
        print(f" - {name:<15}: {rate:10.4f} Hz (N = {count})")

    # ==========================================
    # Safe Concurrent Single CSV Appending
    # ==========================================
    output_path = os.path.abspath(args.output)
    df_out = pd.DataFrame([run_data])

    max_retries = 15
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(1.0, 5.0))
            
            file_exists = os.path.isfile(output_path)
            with open(output_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                df_out.to_csv(f, header=not file_exists, index=False)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f, fcntl.LOCK_UN)
            
            print(f"\n✅ Done. Output safely appended to {output_path}")
            break
        except Exception as io_err:
            print(f"⚠️ [Attempt {attempt+1}/{max_retries}] Locked or busy for Run {run_id}: {io_err}")
            if attempt == max_retries - 1:
                print(f"🔥 FATAL: Failed to write Run {run_id} after {max_retries} attempts.")
                sys.exit(1)

if __name__ == "__main__":
    main()
    