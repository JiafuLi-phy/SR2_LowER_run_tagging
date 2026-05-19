#!/bin/bash

# ======================================================================
# XENONnT SR2 Batch Submitter (Synchronized for Single Master File)
# ======================================================================

# --- Configuration ---
RUNLIST="${1:-/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv}"
# RUNLIST="${1:-/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/test_list.csv}"
RESULT_CSV="sr2_master_run_rates.csv" 
MISSING_FULL_CSV="missing_runs_full.csv" # 💡 NEW: Used to save the missing list containing all metadata

# Slurm Settings
MAX_JOBS=200
CHUNK_SIZE=10 
USER_NAME=$USER

# Directories
mkdir -p ./logs
mkdir -p ./strax_data

# 1. Extract Run IDs for Processing
RUN_IDS=$(awk -F',' 'NR>1 {print $1}' "$RUNLIST" | grep -E '^[0-9]+$')

if [ -z "$RUN_IDS" ]; then
    echo "❌ Error: No valid Run IDs found in $RUNLIST"
    exit 1
fi

# Create a sorted temp file of intended runs for later comparison
INTENDED_SORTED=$(mktemp)
echo "$RUN_IDS" | awk '{printf "%06d\n", $1}' | sort > "$INTENDED_SORTED"

echo "✅ Successfully loaded runs from: $RUNLIST"

# 2. Iteration and Submission Loop
counter=0
chunk=""

for run in $RUN_IDS; do
    run_formatted=$(printf "%06d" $run)
    chunk="$chunk $run_formatted"
    ((counter++))

    if [[ $counter -eq $CHUNK_SIZE ]]; then
        # Throttling
        while true; do
            current_jobs=$(squeue -u "$USER_NAME" -h | wc -l)
            if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
                break
            fi
            echo "⏳ Queue full ($current_jobs jobs). Sleeping 30s..."
            sleep 30
        done

        # Submission
        job_name="xnt_rate_$(date +%s)"
        sbatch --job-name="$job_name" \
               --partition=lgrandi \
               --account=pi-lgrandi \
               --qos=lgrandi \
               --mem=32G \
               --cpus-per-task=1 \
               --output="./logs/job_%j.log" \
               --wrap="bash ./run_analysis.sh $chunk"
        
        echo "🚀 Submitted chunk ($CHUNK_SIZE runs). User queue: $current_jobs"
        sleep 0.5 
        counter=0
        chunk=""
    fi
done

# Final Chunk
if [[ -n "$chunk" ]]; then
    sbatch --job-name="last_chunk" --partition=lgrandi --account=pi-lgrandi --qos=lgrandi --mem=32G --output="./logs/job_%j.log" --wrap="bash ./run_analysis.sh $chunk"
fi

echo "🎉 All jobs dispatched."

# ======================================================================
# 3. Post-Analysis Logic
# ======================================================================
function generate_missing_list {
    if [ -f "$RESULT_CSV" ]; then
        echo "🔍 Scanning $RESULT_CSV for completed runs..."
        COMPLETED_SORTED=$(mktemp)
        
        # Extract the first column (Run ID) of the generated master CSV, skip header, and sort
        awk -F',' 'NR>1 {print $1}' "$RESULT_CSV" | awk '{printf "%06d\n", $1}' | sort > "$COMPLETED_SORTED"
        
        # Find the missing IDs
        MISSING_IDS=$(comm -23 "$INTENDED_SORTED" "$COMPLETED_SORTED")
        MISSING_COUNT=$(echo "$MISSING_IDS" | grep -v '^$' | wc -l)
        
        if [ "$MISSING_COUNT" -gt 0 ]; then
            # 💡 1. Extract the header of the original file and write to the new CSV
            head -n 1 "$RUNLIST" > "$MISSING_FULL_CSV"
            
            # 💡 2. Extract the entire row from the original file based on the missing IDs
            for id in $MISSING_IDS; do
                # Remove zero-padding (e.g., 054575 -> 54575) for exact matching in the original file
                unpadded_id=$((10#$id)) 
                # Use regex to match only data where the first column at the beginning of the line is the ID
                grep -E "^${unpadded_id}," "$RUNLIST" >> "$MISSING_FULL_CSV"
            done
        else
            # If there are no missing runs, create an empty record file
            > "$MISSING_FULL_CSV" 
        fi
        
        echo "------------------------------------------------"
        echo "📊 Summary:"
        echo "   - Intended: $(wc -l < "$INTENDED_SORTED")"
        echo "   - Completed: $(wc -l < "$COMPLETED_SORTED")"
        echo "   - Missing: $MISSING_COUNT"
        if [ "$MISSING_COUNT" -gt 0 ]; then
            echo "   - 📝 Missing details saved to: $MISSING_FULL_CSV"
        fi
        echo "------------------------------------------------"
        
        rm "$COMPLETED_SORTED"
    else
        echo "⚠️ $RESULT_CSV not found yet. Cannot check for missing runs."
    fi
}