# SR2 LowER Run Tagging & Quality Analysis

Automated pipeline for XENONnT **Science Run 2 (SR2)** low-energy region (LowER) run-by-run event-rate computation, detector quality scoring via unsupervised machine learning (Isolation Forest), and publication-grade time-evolution visualization.

## Overview

Three core scripts implement the full workflow:

1. **`compute_event_rates.py`** — extracts per-run physics event rates (Gate, Cathode, S1-only, S2-only, Wall) from raw XENONnT data, normalized by livetime.
2. **`compute_quality_scores.py`** — merges detector parameters and event rates, then applies Isolation Forest to assign each run a 0–100 quality score with three scoring modes (single, batch, consensus).
3. **`plot_evolution.py`** — generates all publication-quality plots: event-rate timeseries, quality-score trends, all-feature evolution PDFs, anomaly diagnostics, and per-run Z-score charts.

The project targets XENONnT SR2 data (Oct 2023 – Apr 2025) and is designed to run on the **University of Chicago Midway3** cluster with Slurm scheduling and CVMFS-based software environments.

---

## Directory Structure

```
run_tagging_lower/
├── compute_event_rates.py             # [1] Per-run event-rate extraction
├── compute_quality_scores.py          # [2] Isolation Forest quality scoring
├── plot_evolution.py                  # [3] Publication-grade visualization
│
├── run_analysis.sh                    # Slurm job wrapper — CVMFS routing per run
├── submit_batch.sh                    # Batch-submits all SR2 runs as Slurm job chunks
│
├── split_modes.py                     # Utility: merge rates + metadata, split by mode
├── extract_missing_runs.py            # Utility: cross-reference missing runs against DB
│
├── results/                           # Output data (HDF5, CSV)
│   └── plots/                         # Generated plots (PNG, PDF)
├── split_modes/                       # Per-mode CSV exports + interval summaries
├── resource_cache/                    # Cached resources for strax data processing
├── strax_data/                        # Stored strax data
│
├── sr2_master_run_rates.csv           # Primary output: event rates per SR2 run
├── sr2_master_run_rates_with_mode.csv # Merged rates with mode/source labels
├── missing_runs.txt                   # Log of runs that failed batch processing
└── test_list.csv                      # Small test subset for development
```

---

## Workflow

### Step 1 — Event-Rate Extraction

```bash
# Single run
bash run_analysis.sh 054585

# Batch (Slurm)
bash submit_batch.sh [runlist.csv]
```

`run_analysis.sh` routes each run to the correct CVMFS software release (5 historical eras from `2022.09.1` to `el7.2025.07.2`), sources the isolated environment (`strax` + `straxen` ± `cutax`), then calls:

```bash
python compute_event_rates.py -r 054585
```

**Per-run computation:**
- Loads `event_info` (and `event_shadow` if available) via strax.
- Derives `r² = x² + y²`.
- Applies physics masks:
  | Category | Selection |
  |---|---|
  | Gate Events | 0 < drift_time < 8 µs |
  | Cathode Events | drift_time 1.8–2.5 ms, or z near cathode, or high S1 / low S2 |
  | S1-Only (Heavy) | S1 < 100 PE, S2 < 100 PE |
  | S2-Only (SE) | S1 < 10 PE, S2 < 200 PE |
  | Wall Events | r² > 3800 cm² |
- Normalizes each count by livetime → rate in Hz.
- Appends result atomically (file-locked with retry) to `sr2_master_run_rates.csv`.

### Step 2 — Quality Scoring

```bash
# Single-run scoring (default — one IF model on per-run features)
python compute_quality_scores.py \
    --run-info sr2_run_tagging_info.csv --rates sr2_master_run_rates.csv

# Batch scoring (rolling-window mean features)
python compute_quality_scores.py --mode batch --batch-n 10 \
    --run-info sr2_run_tagging_info.csv --rates sr2_master_run_rates.csv

# Consensus scoring (multi-window voting — most robust)
python compute_quality_scores.py --mode consensus \
    --run-info sr2_run_tagging_info.csv --rates sr2_master_run_rates.csv \
    --windows 1 2 4 8 10 --k-mad 4.5 --vote-ratio 0.5

# With date filtering and anomaly export
python compute_quality_scores.py --mode consensus \
    --run-info sr2_run_tagging_info.csv --rates sr2_master_run_rates.csv \
    --start-date 2024-01-01 --end-date 2024-06-30 --export-anomalies \
    --anomaly-threshold 20.0
```

**Scoring modes:**

| Mode | Description |
|---|---|
| `single` | One Isolation Forest (150 trees) on per-run features. Fast, independent. |
| `batch` | Rolling-window mean features → IF. Captures local temporal context. |
| `consensus` | Multiple IF models across window sizes. MAD-based anomaly threshold with voting. Most robust against isolated outliers. |

**Consensus details:**
- Each window size trains an independent IF model.
- Anomaly threshold per window: `median(score) − k_mad × MAD(score)`.
- A run is flagged bad if the fraction of windows voting "bad" exceeds `vote_ratio`.
- Output includes per-window scores, anomaly flags, vote counts, and a final consensus flag.

### Step 3 — Visualization

All plotting is done via subcommands of `plot_evolution.py`:

```bash
# Event-rate evolution (5-panel timeseries)
python plot_evolution.py rates \
    --rates sr2_master_run_rates.csv \
    --calib-intervals split_modes/calibration_intervals_summary.csv

# Quality-score trend (dual-panel: run number + time, with calibration backgrounds)
python plot_evolution.py quality \
    --quality results/sr2_quality.h5 --run-info sr2_run_tagging_info.csv

# All detector features over time → single multi-page PDF
python plot_evolution.py features --run-info sr2_run_tagging_info.csv

# Consensus anomaly diagnostics (heatmaps, distributions, boxplots)
python plot_evolution.py anomaly \
    --quality results/sr2_quality_consensus.h5 \
    --run-info sr2_run_tagging_info.csv --windows 1 2 4 8 10

# Per-run Z-score diagnostic bar chart
python plot_evolution.py diagnose --run-id 054585 \
    --quality results/sr2_quality.h5 --run-info sr2_run_tagging_info.csv
```

All plot types produce both high-DPI PNG and vector PDF output.

**Plot types produced:**

| Subcommand | Output |
|---|---|
| `rates` | 5-panel stacked event-rate timeseries with calibration backgrounds and RdYlGn colormap |
| `quality` | Dual-panel quality trend (vs run number + vs time), mode-colored background spans, consensus-bad X markers |
| `features` | Every numerical detector parameter as a scatter plot over time, compiled into one multi-page PDF |
| `anomaly` | Voting heatmap, type distribution bar chart, time histogram, feature-deviation Z-score boxplot (all-modes + science-only subsets) |
| `diagnose` | Horizontal bar chart of top-10 Z-score deviations for a single run (±3σ reference lines) |

---

## Key Output Files

| File | Description |
|---|---|
| `sr2_master_run_rates.csv` | Event rates (Hz) per run: Gate, Cathode, S1, S2, Wall |
| `sr2_master_run_rates_with_mode.csv` | Above + mode/source labels from run database |
| `results/sr2_quality.h5` | All runs with quality scores, flags, and per-window details |
| `results/plots/*.png / *.pdf` | Generated plots from `plot_evolution.py` |
| `results/anomalous_runs.csv` | Ranked list of runs below the quality threshold |
| `split_modes/calibration_intervals_summary.csv` | Time intervals per calibration/science campaign |
| `all_features_evolution.pdf` | Multi-page PDF of every detector parameter over time |

---

## Dependencies

- **Python 3** with: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- **XENONnT software stack** (on Midway3 via CVMFS): `strax`, `straxen`, optionally `cutax`
- **Slurm** workload manager (for `submit_batch.sh`)

---

## Environment Notes

- All scripts force the **Agg** matplotlib backend for headless cluster execution.
- The batch submitter targets partition `lgrandi` with account `pi-lgrandi` and QoS `lgrandi`.
- `run_analysis.sh` routes legacy runs (before run 65000) to older CVMFS releases and blocks `cutax` injection to avoid `PeakSEScore` compatibility issues.
- File writes from parallel Slurm jobs are protected by `fcntl.flock` with retry logic (up to 15 attempts with random backoff).
- The consensus voting threshold `k_mad` defaults to 4.5, which approximates a 3σ cut for Gaussian-distributed residuals.
