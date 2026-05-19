# SR2 LowER Run Tagging & Quality Analysis

Automated pipeline for XENONnT **Science Run 2 (SR2)** low-energy region (LowER) run-by-run event-rate computation, detector quality scoring via unsupervised machine learning (Isolation Forest), and publication-grade time-evolution visualization.

## Overview

This repository performs three core tasks:

1. **Event-rate extraction** — processes raw XENONnT data for each run, computing physics event rates (Gate, Cathode, S1-only, S2-only, Wall) normalized by livetime.
2. **Detector quality scoring** — merges run metadata, deadtime, and rate information, then applies Isolation Forest to assign a 0–100 quality score to each run, enabling anomaly detection and bad-run flagging.
3. **Evolution visualization** — generates publication-ready plots of all detector parameters and event rates over time, with calibration-campaign backgrounds.

The project targets XENONnT SR2 data (Oct 2023 – Apr 2025) and is designed to run on the **University of Chicago Midway3** cluster with Slurm scheduling and CVMFS-based software environments.

---

## Directory Structure

```
run_tagging_lower/
├── run_quality_raw_event.py          # Core: single-run event-rate computation (called by Slurm jobs)
├── run_analysis.sh                   # Slurm job wrapper — routes each run to correct CVMFS release
├── submit_batch.sh                   # Batch-submits all SR2 runs as Slurm job array chunks
│
├── split_modes.py                    # Merges rates + run metadata, splits by calibration mode
├── extract_missing_runs.py           # Cross-references missing runs against the master run database
│
├── generate_quality_map_analyzer.py  # Quality analyzer v1: run_info + deadtime + rates → IF score
├── generate_quality_map_analyzer2.py # Quality analyzer v2: run_info + rates (no deadtime dependency)
├── generate_quality_map_analyzer3.py # Quality analyzer v3: consensus scoring across window sizes
├── generate_quality_map_plot.py      # Lightweight version: plotting-only, same core class
│
├── plot_quality_evo.py               # Event-rate evolution plot (basic, color-line style)
├── plot_quality_evo_v2.py            # Event-rate evolution plot (styled, scatter + colorbar)
├── plot_quality_evo_v3.py            # Event-rate evolution plot (adds calibration background spans)
├── plot_all_evolutions.py            # Grid of ALL physical features over time → single PDF
│
├── run_quality_score_analyzer.sh     # Shell driver for analyzer v1
├── run_quality_score_analyzer2.sh    # Shell driver for analyzer v2
├── run_quality_score_analyzer3.sh    # Shell driver for analyzer v3 (consensus)
├── run_quality_score_plot.sh         # Shell driver for plot-only analyzer
│
├── results/                          # Output data (HDF5, CSV) and generated plots (PNG, PDF)
│   └── plots/                        # Diagnostic plots: trend lines, anomaly heatmaps, boxplots
├── split_modes/                      # Per-mode CSV exports + calibration interval summaries
├── resource_cache/                   # Cached resources for strax data processing
├── strax_data/                       # Stored strax data
│
├── sr2_master_run_rates.csv          # Primary output: event rates for every processed SR2 run
├── sr2_master_run_rates_with_mode.csv# Merged rates with mode/source labels from run database
├── missing_runs.txt                  # Log of runs that failed processing
└── test_list.csv                     # Small test subset for development
```

---

## Pipeline Workflow

### Step 1: Batch Event-Rate Extraction

```bash
bash submit_batch.sh [runlist.csv]
```

Submits Slurm job chunks (`--partition=lgrandi`). Each chunk calls `run_analysis.sh`, which:
1. Determines the correct CVMFS software release for each run's era (5 historical releases from 2022.09.1 to el7.2025.07.2).
2. Sources the isolated CVMFS environment (`strax` + `straxen` + `cutax`).
3. Executes `run_quality_raw_event.py -r <run_id>`.

**What it computes per run:**
- Loads `event_info` (and `event_shadow` if available) via strax.
- Derives `r² = x² + y²`.
- Applies physics masks:
  | Category | Selection |
  |---|---|
  | Gate Events | 0 < drift_time < 8 µs |
  | Cathode Events | drift_time 1.8–2.5 ms OR z near cathode OR high S1 / low S2 |
  | S1-Only (Heavy) | S1 < 100 PE, S2 < 100 PE |
  | S2-Only (SE) | S1 < 10 PE, S2 < 200 PE |
  | Wall Events | r² > 3800 cm² |
- Normalizes each count by the run livetime → rate in Hz.
- Appends result atomically (with file locking) to `sr2_master_run_rates.csv`.

### Step 2: Data Organization

```bash
python split_modes.py
```

Merges `sr2_master_run_rates.csv` with the master run database (`sr2_run_tagging_info_0.0.5.csv`), then:
- Sorts runs by mode and chronologically.
- Identifies contiguous calibration/science-run time intervals (gap ≤ 1 day groups consecutive runs).
- Exports per-mode CSV files to `split_modes/`.
- Outputs `mode_intervals_summary.csv` and `calibration_intervals_summary.csv`.

```bash
python extract_missing_runs.py
```

Cross-references `missing_runs.txt` against the master database and exports `missing_runs_full_info.csv` containing full metadata for runs that failed during batch processing.

### Step 3: Quality Scoring

Three progressive analyzer versions are provided:

**v1 (`generate_quality_map_analyzer.py`):**
```bash
bash run_quality_score_analyzer.sh -v -a
```
- Merges 3 sources: run_info CSV + deadtime CSV + rates CSV.
- Extracts numeric physical features (excluding IDs, times, counts, binning metadata).
- Trains an **Isolation Forest** (150 estimators, `contamination='auto'`) on the features.
- Rescales decision function output to 0–100 quality score.
- Optional `-b` flag enables rolling-window batch quality scoring.

**v2 (`generate_quality_map_analyzer2.py`):**
```bash
bash run_quality_score_analyzer2.sh -v -a
```
- Same as v1 but **drops the deadtime dependency** — uses only run_info + rates (with mode labels).
- Equivalent ML pipeline with 150-tree Isolation Forest.

**v3 (`generate_quality_map_analyzer3.py`) — Consensus:**
```bash
bash run_quality_score_analyzer3.sh -w "1 2 4 8 10" -v -a
```
- Trains **multiple** Isolation Forest models across independent window sizes (default: 1, 2, 4, 8, 10).
- Each run receives a binary vote from each window model (MAD-based threshold with configurable multiplier `-k`).
- Final quality determined by **voting**: a run is flagged anomalous if the fraction of models voting "bad" exceeds `vote_ratio` (default 0.5).
- Produces consensus HDF5 output and ensemble voting heatmaps.

**Diagnostic outputs** (when `-a` is enabled):
- `*_bad_runs_list.csv` — ranked list of anomalous runs with mode labels.
- `*_bad_run_types.png` — distribution of anomalous runs by data type.
- `*_bad_run_features_boxplot.png/pdf` — top 15 most deviating features across anomalous runs (Z-score boxplot).
- Per-run diagnostic plots (`run_diag_<id>.png/pdf`) for any run via `-p <id>`.

### Step 4: Evolution Visualization

**Event rates over time:**
```bash
python plot_quality_evo.py          # Basic
python plot_quality_evo_v2.py       # Styled with scatter + RdYlGn colormap
python plot_quality_evo_v3.py       # Adds calibration/source background spans
```
Each generates a 5-panel stacked timeseries (Gate, Cathode, S1-only, S2-only, Wall) saved as high-DPI PNG and vector PDF.

**All physical features over time:**
```bash
python plot_all_evolutions.py
```
Automatically identifies all numerical columns (excluding metadata) from the run database, generates individual scatter+colorbar evolution plots with calibration backgrounds, and compiles them into a single `all_evolution_plots_colored.pdf`. Each run is color-coded by its Z-axis variable (default: run number) and uses distinct markers per data type.

---

## Key Output Files

| File | Description |
|---|---|
| `sr2_master_run_rates.csv` | Event rates (Hz) per run: Gate, Cathode, S1-only, S2-only, Wall |
| `sr2_master_run_rates_with_mode.csv` | Above + mode/source labels merged from run database |
| `results/sr2_quality_master.h5` | HDF5 store of all runs with single-run quality scores (v1/v2) |
| `results/sr2_quality_consensus.h5` | HDF5 store of all runs with consensus quality scores (v3) |
| `results/plots/single_run_trend.png/pdf` | Dual-panel quality trend (by run number and by time) |
| `results/plots/consensus_anomaly_*.png/pdf` | Consensus anomaly heatmaps, time/type distributions, feature deviations |
| `split_modes/calibration_intervals_summary.csv` | Time intervals for each calibration/science campaign |
| `split_modes/*.csv` | Per-mode data subsets (tpc_bkg, tpc_kr83m, tpc_ambe, etc.) |
| `all_evolution_plots_colored.pdf` | Multi-page PDF of all detector parameters over time |

---

## Dependencies

- **Python 3** with: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- **XENONnT software stack** (on Midway3 via CVMFS): `strax`, `straxen`, optionally `cutax`
- **Slurm** workload manager (for `submit_batch.sh`)

---

## Environment Notes

- All scripts force the **Agg** matplotlib backend for headless cluster execution.
- The batch submitter targets partition `lgrandi` with account `pi-lgrandi` and QoS `lgrandi`.
- Legacy runs (before run 65000) block `cutax` injection to avoid `PeakSEScore` compatibility issues; modern runs permit it.
- File writes from parallel Slurm jobs are protected by `fcntl.flock` with retry logic (up to 15 attempts with random backoff).
