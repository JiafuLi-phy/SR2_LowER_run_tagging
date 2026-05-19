#!/bin/bash

# =================================================================
# SR2 Data Quality Analysis & Diagnostic Automation
# =================================================================

# Default file paths
DEFAULT_RUN_INFO="/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv"
DEFAULT_DEADTIME="/scratch/midway3/jiafu/SR2_LowER/data_organization/deadtime/deadtime_selection_sr2.csv"
DEFAULT_RATES="/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/sr2_master_run_rates.csv"
OUTPUT_H5="results/sr2_quality_master.h5"
PLOT_DIR="results/plots"

BATCH_N=10
ENABLE_BATCH=false
VISUALIZE=false
INSPECT_ID=""
START_DATE=""
END_DATE=""

usage() {
    echo "Usage: $0 [Options]"
    echo "Options:"
    echo "  -i PATH    Run info CSV"
    echo "  -d PATH    Deadtime CSV"
    echo "  -r PATH    Rates CSV"
    echo "  -o PATH    Output filename (.h5, .csv, .xlsx)"
    echo "  -n INT     Window size for rolling batch mode (Default: 10)"
    echo "  -b         Enable sequential batch processing mode"
    echo "  -v         Enable result visualization (IMPORTANT: Required for plots)"
    echo "  -p ID      Inspect/Diagnose a specific Run ID (e.g., 54585)"
    echo "  -s DATE    Start Date filter (Format: YYYY-MM-DD, e.g., 2023-10-14)"
    echo "  -e DATE    End Date filter   (Format: YYYY-MM-DD, e.g., 2023-10-16)"
    echo "  -h         Show help menu"
    exit 1
}

# Added 's:' and 'e:' to getopts
while getopts "i:d:r:o:n:bvp:s:e:h" opt; do
    case $opt in
        i) RUN_INFO=$OPTARG ;;
        d) DEADTIME=$OPTARG ;;
        r) RATES=$OPTARG ;;
        o) OUTPUT_H5=$OPTARG ;;
        n) BATCH_N=$OPTARG ;;
        b) ENABLE_BATCH=true ;;
        v) VISUALIZE=true ;;
        p) INSPECT_ID=$OPTARG ;;
        s) START_DATE=$OPTARG ;;
        e) END_DATE=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

RUN_INFO=${RUN_INFO:-$DEFAULT_RUN_INFO}
DEADTIME=${DEADTIME:-$DEFAULT_DEADTIME}
RATES=${RATES:-$DEFAULT_RATES}

# Best Practice: Wrap directory paths in quotes to prevent word-splitting errors
mkdir -p "$(dirname "$OUTPUT_H5")"
mkdir -p "$PLOT_DIR"

echo ">>> Analysis started at: $(date)"

# Construct Python arguments dynamically
PY_ARGS="--run_info $RUN_INFO --deadtime $DEADTIME --rates $RATES --output $OUTPUT_H5 --plot_dir $PLOT_DIR"

if [ "$ENABLE_BATCH" = true ]; then PY_ARGS="$PY_ARGS --enable_batch --batch_n $BATCH_N"; fi
if [ "$VISUALIZE" = true ]; then PY_ARGS="$PY_ARGS --plot"; fi
if [[ -n "$INSPECT_ID" ]]; then PY_ARGS="$PY_ARGS --inspect_id $INSPECT_ID"; fi
if [[ -n "$START_DATE" ]]; then PY_ARGS="$PY_ARGS --start_date $START_DATE"; fi
if [[ -n "$END_DATE" ]]; then PY_ARGS="$PY_ARGS --end_date $END_DATE"; fi

# Execute Python (PY_ARGS is intentionally unquoted here so bash passes them as separate arguments)
python generate_quality_map_plot.py $PY_ARGS

if [ $? -eq 0 ]; then
    echo "✅ >>> Analysis successful. Check the 'results/' folder for outputs."
else
    echo "❌ >>> Analysis failed. Please review the Python traceback above."
    exit 1
fi