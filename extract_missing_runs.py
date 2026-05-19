import pandas as pd
import os

# ==========================================
# 1. Define file paths
# ==========================================
missing_txt_path = "/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/missing_runs.txt"
info_path = "/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv"
output_path = "/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/missing_runs_full_info.csv"

def extract_missing_info():
    print(">>> Starting missing runs extraction process...")

    # ==========================================
    # 2. Read and clean the missing runs list (.txt)
    # ==========================================
    if not os.path.exists(missing_txt_path):
        print(f"❌ Error: Missing runs file not found at {missing_txt_path}")
        return

    with open(missing_txt_path, 'r') as f:
        raw_lines = f.readlines()

    missing_runs = set()
    for line in raw_lines:
        clean_line = line.strip()
        # Skip empty lines or header strings (like "Run_ID")
        if not clean_line or not clean_line.replace('.', '').isdigit():
            continue
        
        # Format strictly to 6 digits with leading zeros
        formatted_id = clean_line.replace('.0', '').zfill(6)
        missing_runs.add(formatted_id)

    print(f"📄 Loaded {len(missing_runs)} valid Run IDs from missing_runs.txt")

    # ==========================================
    # 3. Read and clean the master info database (.csv)
    # ==========================================
    print(f"📚 Loading master info database (this may take a few seconds)...")
    # Read as string to prevent pandas from dropping leading zeros
    df_info = pd.read_csv(info_path, dtype=str)
    
    # Strip hidden whitespaces from column headers
    df_info.columns = df_info.columns.str.strip()

    # Identify the Run ID column ('number' or 'name')
    id_col = 'number' if 'number' in df_info.columns else 'name'
    if id_col not in df_info.columns:
        print("❌ Error: Could not find 'number' or 'name' column in info CSV.")
        return

    # Clean the Run ID column in the dataframe just like we did for the txt file
    df_info[id_col] = df_info[id_col].str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(6)

    # ==========================================
    # 4. Extract the matching runs
    # ==========================================
    print("🔍 Searching for matches...")
    # Filter the dataframe to keep only the rows where the ID is in our missing_runs set
    df_extracted = df_info[df_info[id_col].isin(missing_runs)]

    # ==========================================
    # 5. Save results and print summary
    # ==========================================
    df_extracted.to_csv(output_path, index=False)
    
    print("\n==========================================")
    print(f"✅ Extraction Complete!")
    print(f" - Expected to find: {len(missing_runs)} runs")
    print(f" - Actually found:   {len(df_extracted)} runs")
    print(f"💾 Saved full information to: {output_path}")
    print("==========================================")
    
    # Quick sanity check for runs that were in the .txt but NOT in the .csv
    found_runs = set(df_extracted[id_col])
    not_found = missing_runs - found_runs
    if not_found:
        print(f"⚠️ Warning: {len(not_found)} runs from your .txt file were NOT FOUND in the info.csv database.")
        print("First few missing IDs:", list(not_found)[:5])

if __name__ == "__main__":
    extract_missing_info()