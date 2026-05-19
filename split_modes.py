import pandas as pd
import os
import re

# ==========================================
# 1. Define File Paths
# ==========================================
rates_path = "/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/sr2_master_run_rates.csv"
info_path = "/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv"
output_dir = "/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/split_modes"

merged_output_path = os.path.join(output_dir, "sr2_master_run_rates_with_mode.csv")
intervals_output_path = os.path.join(output_dir, "mode_intervals_summary.csv")
calib_intervals_output_path = os.path.join(output_dir, "calibration_intervals_summary.csv")

def merge_and_split_time_based():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Reading input files...")
    df_rates = pd.read_csv(rates_path, dtype=str)
    df_info = pd.read_csv(info_path, dtype=str)

    # ==========================================
    # 🚨 BULLET-PROOF CLEANING SECTION
    # ==========================================
    df_rates.columns = df_rates.columns.str.strip()
    df_info.columns = df_info.columns.str.strip()

    if 'number' in df_info.columns:
        df_info = df_info.rename(columns={'number': 'Run_ID'})
    elif 'name' in df_info.columns:
        df_info = df_info.rename(columns={'name': 'Run_ID'})

    def clean_run_ids(series):
        return series.str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(6)

    df_rates['Run_ID'] = clean_run_ids(df_rates['Run_ID'])
    df_info['Run_ID'] = clean_run_ids(df_info['Run_ID'])

    mode_col = 'mode'
    if mode_col in df_info.columns:
        df_info[mode_col] = df_info[mode_col].str.strip()

    # ==========================================
    # Merge and Process
    # ==========================================
    print("Merging data...")
    time_cols = [c for c in ['start', 'end'] if c in df_info.columns]
    cols_to_select = ['Run_ID', mode_col] + time_cols
    
    df_merged = pd.merge(df_rates, df_info[cols_to_select], on='Run_ID', how='left')

    cols = list(df_merged.columns)
    cols.insert(1, cols.pop(cols.index(mode_col)))
    df_merged = df_merged[cols]

    df_merged.to_csv(merged_output_path, index=False)
    print(f"✅ Master merged file saved to: {merged_output_path}")

    # ==========================================
    # Extract Time-Based Campaigns (OPTIMIZED)
    # ==========================================
    print("\nCalculating time-based campaign intervals (Grouping by Mode + < 1 day gap)...")
    
    # 1. Ensure time columns exist and convert to datetime for math
    if 'start' in time_cols:
        df_merged['start_dt'] = pd.to_datetime(df_merged['start'], errors='coerce')
    else:
        raise ValueError("Error: 'start' column is missing. Cannot calculate time gaps.")

    # 2. Sort by Mode FIRST, then chronologically. 
    # This groups all identical modes together, ignoring intervening different modes.
    df_sorted = df_merged.sort_values([mode_col, 'start_dt']).reset_index(drop=True)
    
    # 3. Define block boundary logic
    MAX_GAP_DAYS = 1.0  # <--- Change this to 2.0 or 3.0 if you want to span across weekends
    
    mode_changed = df_sorted[mode_col] != df_sorted[mode_col].shift(1)
    time_gap = df_sorted['start_dt'] - df_sorted['start_dt'].shift(1)
    is_large_gap = time_gap > pd.Timedelta(days=MAX_GAP_DAYS)
    
    # A new block starts if the mode changes OR the time gap is too large
    df_sorted['block_id'] = (mode_changed | is_large_gap).cumsum()
    
    # 4. Aggregate the blocks
    agg_kwargs = {
        'Mode': (mode_col, 'first'),
        'Start_Run_ID': ('Run_ID', 'first'),
        'End_Run_ID': ('Run_ID', 'last'),
        'Start_Time': ('start', 'first'),
        'End_Time': ('end', 'last') if 'end' in time_cols else ('start', 'last'),
        'Total_Runs_In_Block': ('Run_ID', 'count') 
    }
    
    df_intervals = df_sorted.groupby('block_id', as_index=False).agg(**agg_kwargs)
    df_intervals = df_intervals.drop(columns=['block_id'])
    
    # 5. Re-sort the final summary chronologically so it's easy to read
    df_intervals['Sort_Time'] = pd.to_datetime(df_intervals['Start_Time'], errors='coerce')
    df_intervals = df_intervals.sort_values('Sort_Time').drop(columns=['Sort_Time'])

    df_intervals.to_csv(intervals_output_path, index=False)
    print(f"✅ Full campaign intervals summary saved to: {intervals_output_path}")

    # ==========================================
    # Filter & Save Target Data Only (Calibrations + Science Runs)
    # ==========================================
    # UPDATED: Added 'bkg' and 'background' to the keywords list
    target_keywords = ['kr83m', 'ambe', 'radon', 'ar37', 'neutron', 'th232', 'bkg', 'background']
    target_pattern = '|'.join(target_keywords)

    # Filter rows where 'Mode' contains any of the keywords (case-insensitive)
    df_target = df_intervals[df_intervals['Mode'].str.contains(target_pattern, case=False, na=False)]
    df_target.to_csv(calib_intervals_output_path, index=False)
    print(f"🎯 Target campaigns (Calibrations & Science Runs) saved to: {calib_intervals_output_path}")
    print(f"   -> Found {len(df_target)} targeted campaigns.")

    # ==========================================
    # Split by mode
    # ==========================================
    print("\nSplitting master data by individual modes...")
    unique_modes = df_merged[mode_col].dropna().unique()

    for m in unique_modes:
        df_subset = df_merged[df_merged[mode_col] == m]
        safe_name = str(m).replace("/", "_").replace(" ", "_")
        sub_path = os.path.join(output_dir, f"{safe_name}.csv")
        # Drop the temporary datetime column before saving
        if 'start_dt' in df_subset.columns:
            df_subset = df_subset.drop(columns=['start_dt'])
        df_subset.to_csv(sub_path, index=False)
        print(f" - [{m}] -> {len(df_subset)} runs saved.")

    # Check for Unknowns
    df_nan = df_merged[df_merged[mode_col].isna()]
    if not df_nan.empty:
        print(f"\n⚠️ Found {len(df_nan)} runs missing a mode. Saving to unknown_mode.csv")
        if 'start_dt' in df_nan.columns:
            df_nan = df_nan.drop(columns=['start_dt'])
        df_nan.to_csv(os.path.join(output_dir, "unknown_mode.csv"), index=False)
    else:
        print("\n🎉 PERFECT MATCH! No unknown runs were found.")

if __name__ == "__main__":
    merge_and_split_time_based()