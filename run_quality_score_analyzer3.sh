#!/bin/bash

# =================================================================
# SR2 Data Quality Analysis & Consensus Diagnostic Automation
# =================================================================

# Default file paths
DEFAULT_RUN_INFO="/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv"
DEFAULT_RATES="/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/sr2_master_run_rates_with_mode.csv"
OUTPUT_H5="results/sr2_quality_consensus.h5"
PLOT_DIR="results/plots"

# Default Consensus Parameters
WINDOWS="1 2 4 8 10"
K_MAD=""
VOTE_RATIO=""

# Feature flags
VISUALIZE=false
ANALYZE_BAD=false
START_DATE=""
END_DATE=""

usage() {
    echo "Usage: $0 [Options]"
    echo "Options:"
    echo "  -i PATH    Run info CSV"
    echo "  -r PATH    Rates CSV"
    echo "  -o PATH    Output filename (Default: results/sr2_quality_consensus.h5)"
    echo "  -w LIST    Window sizes for consensus evaluation (Default: \"1 2 4 8 10\", enclose in quotes)"
    echo "  -k FLOAT   MAD multiplier for anomaly threshold (Default: 4.5, approx 3-sigma cut)"
    echo "  -f FLOAT   Vote ratio required to flag a run as bad (Default: 0.5, means >50% failure)"
    echo "  -v         Enable general trend visualization (--plot)"
    echo "  -a         Enable Consensus Bad Run / Anomaly Analysis (--analyze_bad)"
    echo "  -s DATE    Start Date filter (Format: YYYY-MM-DD, e.g., 2026-03-01)"
    echo "  -e DATE    End Date filter   (Format: YYYY-MM-DD, e.g., 2026-03-31)"
    echo "  -h         Show help menu"
    exit 1
}

# Parse command line options
while getopts "i:r:o:w:k:f:vas:e:h" opt; do
    case $opt in
        i) RUN_INFO=$OPTARG ;;
        r) RATES=$OPTARG ;;
        o) OUTPUT_H5=$OPTARG ;;
        w) WINDOWS=$OPTARG ;;
        k) K_MAD=$OPTARG ;;
        f) VOTE_RATIO=$OPTARG ;;
        v) VISUALIZE=true ;;
        a) ANALYZE_BAD=true ;;
        s) START_DATE=$OPTARG ;;
        e) END_DATE=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Apply defaults if not provided
RUN_INFO=${RUN_INFO:-$DEFAULT_RUN_INFO}
RATES=${RATES:-$DEFAULT_RATES}

# Ensure directories exist
mkdir -p "$(dirname "$OUTPUT_H5")"
mkdir -p "$PLOT_DIR"

echo ">>> Consensus Analysis started at: $(date)"

# Construct Python arguments (ensure your python script is named generate_quality_map_analyzer3.py)
PY_ARGS="--run_info $RUN_INFO --rates $RATES --output $OUTPUT_H5 --plot_dir $PLOT_DIR"

if [[ -n "$WINDOWS" ]]; then PY_ARGS="$PY_ARGS --windows $WINDOWS"; fi
if [[ -n "$K_MAD" ]]; then PY_ARGS="$PY_ARGS --k_mad $K_MAD"; fi
if [[ -n "$VOTE_RATIO" ]]; then PY_ARGS="$PY_ARGS --vote_ratio $VOTE_RATIO"; fi
if [ "$VISUALIZE" = true ]; then PY_ARGS="$PY_ARGS --plot"; fi
if [ "$ANALYZE_BAD" = true ]; then PY_ARGS="$PY_ARGS --analyze_bad"; fi
if [[ -n "$START_DATE" ]]; then PY_ARGS="$PY_ARGS --start_date $START_DATE"; fi
if [[ -n "$END_DATE" ]]; then PY_ARGS="$PY_ARGS --end_date $END_DATE"; fi

# Execute Python
python generate_quality_map_analyzer3.py $PY_ARGS

if [ $? -eq 0 ]; then
    echo "✅ >>> Analysis successful. Check the '$(dirname "$OUTPUT_H5")' and '$PLOT_DIR' folders for outputs."
else
    echo "❌ >>> Analysis failed. Please review the Python traceback above."
    exit 1
fi