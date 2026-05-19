#!/bin/bash

# ======================================================================
# XENONnT Run Rate & Plot (CVMFS Ultimate Sandbox Edition)
# Automatically routes runs to their historical software environments
# ======================================================================

# Obtain the absolute path where the current script is located to prevent path confusion
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/compute_event_rates.py"

# add missing record file
MISSING_FILE="${SCRIPT_DIR}/missing_runs.txt"

# 1. Check user inputs
if [ "$#" -eq 0 ]; then
    echo "❌ Error: No Run ID provided."
    echo "👉 Usage: $0 <run_id_1> [run_id_2] [run_id_3] ..."
    exit 1
fi

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Cannot find Python script '$PYTHON_SCRIPT'"
    exit 1
fi

# ======================================================================
# 2. Loop through all provided Run IDs
# ======================================================================
for RUN_ID in "$@"
do
    echo ""
    echo "🚀 ========================================="
    
    # Convert to base-10 to prevent octal evaluation errors (e.g., "08", "09")
    RUN_NUM=$((10#$RUN_ID))
    # Format back to a 6-digit string for Python arguments
    FORMATTED_RUN=$(printf "%06d" $RUN_NUM)
    
    echo "🚀 Starting processing for Run ID: $FORMATTED_RUN"
    echo "🚀 ========================================="

    # ------------------------------------------------------------------
    # 🌟 Determine the era and assign the appropriate CVMFS Release
    # ------------------------------------------------------------------
    if [ "$RUN_NUM" -lt 47614 ]; then
        TARGET_RELEASE="2022.09.1"
    elif [ "$RUN_NUM" -lt 52315 ]; then
        TARGET_RELEASE="2023.05.2"
    elif [ "$RUN_NUM" -lt 57693 ]; then
        TARGET_RELEASE="2023.07.1"
    elif [ "$RUN_NUM" -lt 65000 ]; then
        # SR2 Mid (e.g., 059642). Uses 2024.04.1 to match online hashes.
        TARGET_RELEASE="2024.04.1"
    else
        # SR2 Late/Current (e.g., 067497). Midway3 uses el7 containers.
        TARGET_RELEASE="el7.2025.07.2"
    fi
    
    echo "🔍 Era Routing: Run $FORMATTED_RUN mapped to $TARGET_RELEASE environment."

    # ------------------------------------------------------------------
    # 🌟 Execute in a Subshell to guarantee strict environment isolation
    # ------------------------------------------------------------------
    (
        # 💡 CRITICAL FIX: Smart cutax Isolation
        if [ "$TARGET_RELEASE" == "el7.2025.07.2" ]; then
            # Modern runs NEED the cluster's dynamic cutax injection
            echo "🧩 Modern run detected: Permitting dynamic cutax injection."
            unset INSTALL_CUTAX
        else
            # Historical runs MUST BLOCK the new cutax to prevent 'PeakSEScore' crash
            echo "🛡️ Legacy run detected: Blocking new cutax to enforce native environment."
            export INSTALL_CUTAX=0
            unset PYTHONPATH
        fi

        # Dynamically construct the path to the specific release's setup script
        SETUP_SCRIPT="/cvmfs/xenon.opensciencegrid.org/releases/nT/$TARGET_RELEASE/setup.sh"
        
        if [ ! -f "$SETUP_SCRIPT" ]; then
            echo "❌ Error: CVMFS setup script not found at: $SETUP_SCRIPT"
            echo "$FORMATTED_RUN" >> "$MISSING_FILE"
            exit 1
        fi

        echo "🔄 Initializing CVMFS Sandbox Environment: $TARGET_RELEASE ..."
        
        # Source the environment (suppress output to keep the terminal clean)
        source "$SETUP_SCRIPT" > /dev/null 2>&1
        
        # Execute the Python analysis script
        python "$PYTHON_SCRIPT" -r "$FORMATTED_RUN"
        
        #  Missing Runs
        if [ $? -eq 0 ]; then
            echo "✅ Run $FORMATTED_RUN successfully processed!"
        else
            echo "❌ Run $FORMATTED_RUN processing failed! Logging to missing_runs.txt"
            echo "$FORMATTED_RUN" >> "$MISSING_FILE"
        fi
    )
done

echo ""
echo "🎉 All requested runs processed! Results saved in run_event_rate/ directory."