#!/usr/bin/env python
"""
Publication-grade evolution plots for XENONnT SR2 LowER analysis.

Generates five types of visualizations:

  1. Event-rate evolution    — 5-panel timeseries with calibration backgrounds.
  2. Quality-score trend     — dual-panel (run number / time) with mode spans.
  3. All-features evolution  — every numerical detector parameter over time
                                compiled into a single multi-page PDF.
  4. Anomaly diagnostics     — voting heatmaps, type/time distributions,
                                feature-deviation boxplots.
  5. Per-run diagnostic      — Z-score bar chart for a single run.

Usage:
    # Event-rate evolution
    python plot_evolution.py rates \
        --rates sr2_master_run_rates.csv \
        --calib-intervals split_modes/calibration_intervals_summary.csv

    # Quality-score trend
    python plot_evolution.py quality \
        --quality results/sr2_quality.h5 --run-info sr2_run_tagging_info.csv

    # All features evolution
    python plot_evolution.py features --run-info sr2_run_tagging_info.csv

    # Anomaly diagnostics (consensus mode)
    python plot_evolution.py anomaly \
        --quality results/sr2_quality_consensus.h5 \
        --run-info sr2_run_tagging_info.csv \
        --windows 1 2 4 8 10

    # Single-run diagnostic
    python plot_evolution.py diagnose --run-id 054585 \
        --quality results/sr2_quality.h5 --run-info sr2_run_tagging_info.csv
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# Global plot style
# ===========================================================================

def set_publication_style():
    """Apply consistent publication-quality matplotlib rcParams."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 3.0,
        'axes.edgecolor': 'black',
        'axes.labelsize': 16,
        'axes.titlesize': 22,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.major.width': 3.0,
        'ytick.major.width': 3.0,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300,
    })


# ===========================================================================
# Calibration / source colour and marker definitions
# ===========================================================================

CALIB_STYLES = {
    'bkg':        {'label': 'Science Run', 'color': "#F3F3EF", 'marker': 'o'},
    'background': {'label': 'Science Run', 'color': "#F3F3EF", 'marker': 'o'},
    'kr83m':      {'label': 'Kr-83m',      'color': "#FF0026", 'marker': 's'},
    'rn220':      {'label': 'Rn-220',      'color': "#FF00FF", 'marker': '^'},
    'radon':      {'label': 'Radon',       'color': "#FF00FF", 'marker': '^'},
    'rn':         {'label': 'Radon',       'color': "#FF00FF", 'marker': '^'},
    'ambe':       {'label': 'AmBe',        'color': "#32CD32", 'marker': 'v'},
    'neutron':    {'label': 'Neutron',     'color': "#8B4513", 'marker': 'd'},
    'ar37':       {'label': 'Ar-37',       'color': "#FF8C00", 'marker': 'p'},
    'th232':      {'label': 'Th-232',      'color': "#100CCE", 'marker': '*'},
}

BACKGROUND_ALPHA = 0.35
SCATTER_SIZE = 45


# ===========================================================================
# Utilities
# ===========================================================================

def merge_intervals(intervals):
    """Merge overlapping (start, end) intervals to prevent alpha-stacking."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for cur in intervals[1:]:
        last = merged[-1]
        if cur[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], cur[1]))
        else:
            merged.append(cur)
    return merged


def _detect_mode_column(df):
    """Return the name of the column likely containing the run mode / source."""
    for col in ['mode', 'source', 'data_type']:
        if col in df.columns:
            return col
    return df.columns[1] if len(df.columns) > 1 else 'mode'


def _read_quality_file(path):
    """Read quality-score output, supporting HDF5 and CSV formats."""
    if path.endswith('.h5') or path.endswith('.hdf5'):
        return pd.read_hdf(path, key='run_data')
    return pd.read_csv(path, dtype={'number': str})


def _build_mode_spans(df_info, mode_col, df_scored=None,
                      x_min=None, x_max=None, time_min=None, time_max=None):
    """Build calibration-background intervals for run-number and time axes.

    Returns (run_intervals, time_intervals, color_map, legend_handles).
    Each dict is {label: [(start, end), ...]}.
    """
    run_intervals = {s['label']: [] for s in CALIB_STYLES.values()}
    time_intervals = {s['label']: [] for s in CALIB_STYLES.values()}
    color_map = {s['label']: s['color'] for s in CALIB_STYLES.values()}

    df = df_info.dropna(subset=[mode_col]).copy()
    if 'number' in df.columns:
        df['run_int'] = df['number'].astype(int)

    # ---- Run-number axis ----
    if x_min is not None and x_max is not None and 'run_int' in df.columns:
        df_run = df[(df['run_int'] >= x_min) & (df['run_int'] <= x_max)] \
                 .sort_values('run_int')
        mode_s = df_run[mode_col].astype(str).str.lower()
        block = ((mode_s != mode_s.shift(1))
                 | ((df_run['run_int'] - df_run['run_int'].shift(1)) > 50)).cumsum()
        for _, grp in df_run.groupby(block):
            cur_mode = grp[mode_col].iloc[0].lower()
            for key, style in CALIB_STYLES.items():
                if key in cur_mode:
                    s, e = grp['run_int'].iloc[0] - 0.5, grp['run_int'].iloc[-1] + 0.5
                    if (e - s) < 80:
                        mid = (s + e) / 2
                        s, e = mid - 40, mid + 40
                    run_intervals[style['label']].append((s, e))
                    break

    # ---- Time axis ----
    if time_min is not None and time_max is not None \
            and 'start' in df.columns:
        df_time = df[(df['start'] >= time_min) & (df['start'] <= time_max)] \
                  .sort_values('start')
        mode_s = df_time[mode_col].astype(str).str.lower()
        block = ((mode_s != mode_s.shift(1))
                 | ((df_time['start'] - df_time['start'].shift(1))
                    > pd.Timedelta(days=1.0))).cumsum()
        for _, grp in df_time.groupby(block):
            cur_mode = grp[mode_col].iloc[0].lower()
            for key, style in CALIB_STYLES.items():
                if key in cur_mode:
                    s = grp['start'].iloc[0]
                    e = (grp['end'].iloc[-1]
                         if 'end' in grp.columns and pd.notnull(grp['end'].iloc[-1])
                         else grp['start'].iloc[-1])
                    delta = e - s
                    if delta < pd.Timedelta(days=4):
                        mid = s + delta / 2
                        s, e = mid - pd.Timedelta(days=2), mid + pd.Timedelta(days=2)
                    time_intervals[style['label']].append((s, e))
                    break

    # Build legend handles
    legend_handles = []
    seen = set()
    for label, intervals in run_intervals.items():
        if intervals and label not in seen:
            seen.add(label)
            legend_handles.append(
                mpatches.Patch(facecolor=color_map[label], edgecolor='black',
                               linewidth=1.0, alpha=BACKGROUND_ALPHA, label=label)
            )
    for label, intervals in time_intervals.items():
        if intervals and label not in seen:
            seen.add(label)
            legend_handles.append(
                mpatches.Patch(facecolor=color_map[label], edgecolor='black',
                               linewidth=1.0, alpha=BACKGROUND_ALPHA, label=label)
            )

    return run_intervals, time_intervals, color_map, legend_handles


def _paint_backgrounds(ax1, ax2, run_intervals, time_intervals, color_map):
    """Draw merged interval backgrounds on run-number (ax1) and time (ax2) axes."""
    for label, intervals in run_intervals.items():
        for s, e in merge_intervals(intervals):
            ax1.axvspan(s, e, facecolor=color_map[label], edgecolor='none',
                        alpha=BACKGROUND_ALPHA, zorder=0)
    for label, intervals in time_intervals.items():
        for s, e in merge_intervals(intervals):
            ax2.axvspan(s, e, facecolor=color_map[label], edgecolor='none',
                        alpha=BACKGROUND_ALPHA, zorder=0)


def _add_legend(ax, handles, ncol_max=6):
    """Add a legend above a top-panel axis, with up to *ncol_max* columns."""
    if not handles:
        return
    ncol = min(len(handles), ncol_max)
    ax.legend(handles=handles, loc='upper center',
              bbox_to_anchor=(0.5, 1.18), ncol=ncol,
              framealpha=1.0, edgecolor='black', fontsize=14, fancybox=False)


def _thicken_spines(*axes):
    """Set thick black borders on all given axes."""
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_color('black')


def _save_and_close(fig, prefix, fmts=('png', 'pdf')):
    """Save figure in each format and close."""
    for fmt in fmts:
        fig.savefig(f"{prefix}.{fmt}", bbox_inches='tight')
    print(f"  Saved → {prefix}.{{{','.join(fmts)}}}")
    plt.close(fig)


# ===========================================================================
# 1. Event-rate evolution
# ===========================================================================

RATE_COLUMNS = {
    'Gate_Event_Rate_Hz':     'Gate Events',
    'Cathode_Event_Rate_Hz':  'Cathode Events',
    'S1_Only_Heavy_Rate_Hz':  'S1-only (Heavy)',
    'S2_Only_SE_Rate_Hz':     'S2-only (SE)',
    'Wall_Event_Rate_Hz':     'Wall Events',
}


def plot_rate_evolution(rates_csv, calib_csv=None, output_prefix="rate_evolution"):
    """5-panel stacked timeseries of event rates with calibration backgrounds.

    Parameters
    ----------
    rates_csv : str
        Path to sr2_master_run_rates.csv.
    calib_csv : str or None
        Path to calibration_intervals_summary.csv (from split_modes.py).
    output_prefix : str
        Prefix for output PNG / PDF.
    """
    df = pd.read_csv(rates_csv)
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    df = df.sort_values('Start_Date').reset_index(drop=True)

    # Load calibration spans
    calib_spans = []
    if calib_csv and os.path.exists(calib_csv):
        calib_spans = _load_calibration_spans(calib_csv)

    fig, axes = plt.subplots(len(RATE_COLUMNS), 1, figsize=(12, 20), sharex=True)
    plotted_labels = set()

    for i, (col, label) in enumerate(RATE_COLUMNS.items()):
        ax = axes[i]
        x, y = df['Start_Date'], df[col]

        if calib_spans:
            for span in calib_spans:
                lbl = span['label'] if span['label'] not in plotted_labels else None
                if lbl:
                    plotted_labels.add(lbl)
                ax.axvspan(span['start'], span['end'], color=span['color'],
                           alpha=0.5, zorder=0, label=lbl)

        ax.plot(x, y, color='#cccccc', linestyle='-', linewidth=1.2, zorder=1)
        sc = ax.scatter(x, y, c=y, cmap='RdYlGn', s=35,
                        edgecolors='black', linewidths=0.6, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label('Rate [Hz]', fontweight='bold')
        cbar.outline.set_linewidth(2.0)

        ax.set_ylabel('Rate [Hz]', fontweight='bold', color='black')
        ax.set_title(label, loc='center', fontsize=16, fontweight='bold',
                     pad=12, color='black')
        ax.set_facecolor('#ffffff')
        ax.tick_params(axis='both', which='major', width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)

        if i == 0 and plotted_labels:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40),
                      ncol=min(len(plotted_labels), 5), framealpha=1.0,
                      edgecolor='black', fontsize=12, fancybox=False)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), fontweight='bold', color='black')
    plt.setp(ax.get_yticklabels(), fontweight='bold', color='black')
    plt.xlabel('Time (Year-Month)', fontsize=16, labelpad=15,
               fontweight='bold', color='black')
    plt.suptitle('Evolution of XENONnT SR2 LowER Event Rates',
                 y=0.995, fontsize=22, fontweight='bold', color='black')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    base = output_prefix if output_prefix else "sr2_rate_evolution"
    _save_and_close(fig, base)


def _load_calibration_spans(csv_path):
    """Parse calibration_intervals_summary.csv into plottable spans."""
    df = pd.read_csv(csv_path)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df = df.dropna(subset=['Start_Time', 'End_Time'])

    spans = []
    for _, row in df.iterrows():
        mode_str = str(row['Mode']).lower()
        for key, style in CALIB_STYLES.items():
            if key in mode_str:
                spans.append({
                    'start': row['Start_Time'], 'end': row['End_Time'],
                    'label': style['label'], 'color': style['color'],
                })
                break
    return spans


# ===========================================================================
# 2. Quality-score trend
# ===========================================================================

def plot_quality_trend(quality_path, run_info_path,
                       output_prefix="quality_trend", title=None):
    """Dual-panel trend: quality score vs run number / time.

    Also overlays red X markers for consensus-bad runs when available.
    """
    df_q = _read_quality_file(quality_path)
    df_info = pd.read_csv(run_info_path)
    df_info['number'] = df_info['number'].astype(str).str.zfill(6)

    if 'start' in df_info.columns and 'end' in df_info.columns:
        df_info['start'] = pd.to_datetime(df_info['start'], errors='coerce')
        df_info['end'] = pd.to_datetime(df_info['end'], errors='coerce')

    # Merge to get start times for the quality-scored runs
    if 'start' not in df_q.columns or df_q['start'].isna().all():
        if 'start' in df_info.columns:
            df_q = df_q.merge(
                df_info[['number', 'start']], on='number', how='left',
            )

    mode_col = _detect_mode_column(df_info)

    df_sorted = df_q.sort_values('number').copy()

    has_consensus = 'is_consensus_bad' in df_sorted.columns
    df_normal = df_sorted[~df_sorted['is_consensus_bad']] if has_consensus else df_sorted
    df_bad = df_sorted[df_sorted['is_consensus_bad']] if has_consensus else pd.DataFrame()

    x_run_min = df_sorted['number'].astype(int).min()
    x_run_max = df_sorted['number'].astype(int).max()

    time_min = df_sorted['start'].min() if 'start' in df_sorted.columns else None
    time_max = df_sorted['start'].max() if 'start' in df_sorted.columns else None

    run_intervals, time_intervals, color_map, legend_handles = \
        _build_mode_spans(df_info, mode_col,
                          x_min=x_run_min, x_max=x_run_max,
                          time_min=time_min, time_max=time_max)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], hspace=0.17, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])   # run-number axis
    ax2 = fig.add_subplot(gs[1, 0], sharey=ax1)  # time axis
    cax = fig.add_subplot(gs[:, 1])   # colorbar

    _paint_backgrounds(ax1, ax2, run_intervals, time_intervals, color_map)

    # Data
    x_run_all = df_sorted['number'].astype(int)
    y_all = df_sorted['quality_score']

    ax1.plot(x_run_all, y_all, alpha=0.4, color='gray', linewidth=1.5, zorder=1)
    sc = ax1.scatter(df_normal['number'].astype(int), df_normal['quality_score'],
                     c=df_normal['quality_score'], cmap='RdYlGn', s=SCATTER_SIZE,
                     edgecolors='k', linewidth=0.5, alpha=0.9, zorder=2)
    if not df_bad.empty:
        ax1.scatter(df_bad['number'].astype(int), df_bad['quality_score'],
                    c='#e74c3c', marker='X', s=120, edgecolors='black',
                    linewidth=1.0, alpha=1.0, zorder=3)

    ax1.set_title(title or "Detector Quality Trend", pad=(55 if legend_handles else 20),
                  fontsize=22)
    ax1.set_xlabel("Run Number", fontsize=16, labelpad=15, weight='bold')
    ax1.set_ylabel("Quality Score (0-100)", fontsize=16, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)

    # Time axis
    if 'start' in df_sorted.columns:
        df_time = df_sorted.sort_values('start')
        df_tn = df_time[~df_time['is_consensus_bad']] if has_consensus else df_time
        df_tb = df_time[df_time['is_consensus_bad']] if has_consensus else pd.DataFrame()

        ax2.plot(df_time['start'], df_time['quality_score'],
                 alpha=0.4, color='gray', linewidth=1.5, zorder=1)
        ax2.scatter(df_tn['start'], df_tn['quality_score'],
                    c=df_tn['quality_score'], cmap='RdYlGn', s=SCATTER_SIZE,
                    edgecolors='k', linewidth=0.5, alpha=0.9, zorder=2)
        if not df_tb.empty:
            ax2.scatter(df_tb['start'], df_tb['quality_score'],
                        c='#e74c3c', marker='X', s=120, edgecolors='black',
                        linewidth=1.0, alpha=1.0, zorder=3)
        ax2.set_xlabel("Time (Year-Month)", fontsize=16, labelpad=15, weight='bold')
        ax2.set_ylabel("Quality Score (0-100)", fontsize=16, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.get_xticklabels(), rotation=0, ha='center',
                 weight='bold', fontsize=14)
    else:
        ax2.text(0.5, 0.5, "Start-time data unavailable",
                 ha='center', va='center', fontsize=16)

    # Legend
    if has_consensus and not df_bad.empty:
        legend_handles.append(
            plt.Line2D([0], [0], marker='X', color='w',
                       markerfacecolor='#e74c3c', markeredgecolor='black',
                       markersize=12, label='Consensus Bad')
        )
    _add_legend(ax1, legend_handles)

    # Colorbar
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Quality Score', weight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(3.0)
    cbar.outline.set_edgecolor('black')
    _thicken_spines(ax1, ax2, cax)

    _save_and_close(fig, output_prefix)


# ===========================================================================
# 3. All-features evolution
# ===========================================================================

def plot_all_features(run_info_csv, output_pdf="all_features_evolution.pdf",
                      calib_csv=None,
                      z_col='number', x_min='2023-10-01', x_max='2025-04-01'):
    """Plot every numerical detector parameter vs time, compiled into one PDF.

    Parameters
    ----------
    run_info_csv : str
        Path to run-tagging info CSV.
    output_pdf : str
        Output PDF filename.
    calib_csv : str or None
        Optional calibration intervals CSV for background shading.
    """
    df = pd.read_csv(run_info_csv)
    if 'start' in df.columns and 'end' in df.columns:
        df['start'] = pd.to_datetime(df['start'], errors='coerce')
        df['end'] = pd.to_datetime(df['end'], errors='coerce')
    df = df.sort_values('start').reset_index(drop=True)

    # Determine data-type column
    if 'source' in df.columns and 'mode' in df.columns:
        df['data_type'] = df['source'].fillna(df['mode'])
    elif 'mode' in df.columns:
        df['data_type'] = df['mode']
    else:
        sys.exit("No 'mode' or 'source' column found in CSV.")

    df['data_type'] = df['data_type'].astype(str).str.lower().str.strip()
    name_map = {
        'none': 'bkg', 'background': 'bkg', 'bkg run': 'bkg',
        'kr-83m': 'kr83m', 'kr83m': 'kr83m',
        'rn-220': 'rn220', 'rn220': 'rn220', 'radon': 'rn220',
        'ambe': 'ambe', 'neutron': 'neutron', 'ar37': 'ar37',
        'th232': 'th232',
    }
    df['data_type'] = df['data_type'].replace(name_map)
    valid_types = [k for k in CALIB_STYLES if k in ['bkg', 'kr83m', 'rn220',
                                                     'ambe', 'neutron', 'ar37', 'th232']]
    df = df[df['data_type'].isin(valid_types)]

    # Identify physical feature columns
    exclude = ['number', 'mode', 'source', 'start', 'end', 'tags', 'livetime',
               'peak_positions_mlp_available', 'peak_basics_available', 'data_type']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_targets = [c for c in numeric_cols if c not in exclude]
    print(f"  {len(plot_targets)} numerical features to plot")

    # Background blocks
    bg_blocks = []
    df_bg = df.dropna(subset=['start', 'data_type']).copy()
    mode_s = df_bg['data_type']
    block_id = ((mode_s != mode_s.shift(1))
                | ((df_bg['start'] - df_bg['start'].shift(1))
                   > pd.Timedelta(days=1.0))).cumsum()

    for _, grp in df_bg.groupby(block_id):
        dtype = grp['data_type'].iloc[0]
        if dtype in CALIB_STYLES:
            s, e = grp['start'].iloc[0], (grp['end'].iloc[-1]
                                          if 'end' in grp.columns
                                          and pd.notnull(grp['end'].iloc[-1])
                                          else grp['start'].iloc[-1])
            delta = e - s
            if delta < pd.Timedelta(days=3):
                mid = s + delta / 2
                s, e = mid - pd.Timedelta(days=1.5), mid + pd.Timedelta(days=1.5)
            bg_blocks.append((s, e, dtype))

    bg_dict = {dt: [] for dt in valid_types}
    for s, e, dtype in bg_blocks:
        bg_dict[dtype].append((s, e))

    unique_types = [t for t in df['data_type'].unique() if t in valid_types]

    with PdfPages(output_pdf) as pdf:
        for i, y_col in enumerate(plot_targets):
            df_plot = df.dropna(subset=[y_col, 'start', z_col]).copy()
            if df_plot.empty:
                continue
            if y_col == 'elife_mean':
                df_plot = df_plot[(df_plot[y_col] > 0) & (df_plot[y_col] < 100000)]

            fig, ax = plt.subplots(figsize=(18, 6))

            for dtype, intervals in bg_dict.items():
                for s, e in merge_intervals(intervals):
                    ax.axvspan(s, e, facecolor=CALIB_STYLES[dtype]['color'],
                               edgecolor='none', alpha=BACKGROUND_ALPHA, zorder=0)

            z_min, z_max = df_plot[z_col].min(), df_plot[z_col].max()
            scatter = None
            for dtype in unique_types:
                sub = df_plot[df_plot['data_type'] == dtype]
                if not sub.empty:
                    scatter = ax.scatter(
                        sub['start'], sub[y_col],
                        c=sub[z_col], cmap='viridis', vmin=z_min, vmax=z_max,
                        marker=CALIB_STYLES[dtype]['marker'],
                        s=SCATTER_SIZE, alpha=0.85, edgecolors='black',
                        linewidth=0.5, zorder=3,
                    )

            # Y-axis range
            if y_col == 'gate_mean':
                ax.set_ylim(299.9, 300.1)
            elif y_col == 'elife_mean':
                ax.set_ylim(0, 100000)
            else:
                q_lo, q_hi = df_plot[y_col].quantile(0.01), df_plot[y_col].quantile(0.99)
                iqr = q_hi - q_lo
                if iqr > 0:
                    ax.set_ylim(q_lo - 0.2 * iqr, q_hi + 0.2 * iqr)

            ax.set_xlim(pd.to_datetime(x_min), pd.to_datetime(x_max))
            ax.set_facecolor('#ffffff')
            ax.grid(True, alpha=0.3, linestyle='--', color='gray', zorder=0)
            clean = y_col.replace('_', ' ').title()
            ax.set_ylabel(clean, color='black', fontweight='bold', fontsize=20)
            ax.set_title(f'Evolution of {clean}', color='black',
                         fontweight='bold', fontsize=20, pad=20)
            ax.set_xlabel('Time (Year-Month)', color='black',
                          fontweight='bold', fontsize=20, labelpad=15)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            _thicken_spines(ax)
            ax.tick_params(direction='in', top=True, right=True,
                           width=2.0, labelsize=15, colors='black')
            for lbl in ax.get_yticklabels():
                lbl.set_fontweight('bold')
            for lbl in ax.get_xticklabels():
                lbl.set_fontweight('bold')

            if i == 0:
                handles = []
                for dtype in unique_types:
                    st = CALIB_STYLES[dtype]
                    handles.append(
                        mpatches.Patch(facecolor=st['color'], edgecolor='black',
                                       linewidth=1.5, alpha=BACKGROUND_ALPHA + 0.15,
                                       label=st['label']))
                ax.legend(handles=handles, loc='upper center',
                          bbox_to_anchor=(0.5, 1.35),
                          ncol=min(len(unique_types), 7), framealpha=0,
                          edgecolor='white', fancybox=False,
                          prop={'size': 20, 'weight': 'bold'})

            if scatter is not None:
                cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                cbar.ax.tick_params(labelsize=12)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  → {len(plot_targets)} plots saved to {output_pdf}")


# ===========================================================================
# 4. Anomaly diagnostics
# ===========================================================================

def plot_anomaly_diagnostics(quality_path, run_info_path, window_sizes,
                             output_prefix="anomaly"):
    """Generate consensus anomaly diagnostic plots.

    Creates: voting heatmap, type distribution, time distribution,
             feature-deviation boxplot.  Splits into 'all modes' and
             'science only' versions when applicable.
    """
    df_q = _read_quality_file(quality_path)
    df_info = pd.read_csv(run_info_path)
    df_info['number'] = df_info['number'].astype(str).str.zfill(6)

    mode_col = _detect_mode_column(df_info)

    # Merge mode info if missing from quality file
    if mode_col not in df_q.columns:
        df_q = df_q.merge(df_info[['number', mode_col]], on='number', how='left')
    if 'start' not in df_q.columns and 'start' in df_info.columns:
        df_info['start'] = pd.to_datetime(df_info['start'], errors='coerce')
        df_q = df_q.merge(df_info[['number', 'start']], on='number', how='left')

    has_consensus = 'is_consensus_bad' in df_q.columns
    if not has_consensus:
        print("No consensus columns found — skipping anomaly diagnostics.")
        return

    bad = df_q[df_q['is_consensus_bad']].copy()
    if bad.empty:
        print("No anomalous runs found by consensus.")
        return

    # Determine feature columns (stored in analyzer output re: actual features)
    # Use the per-window score columns to infer features
    score_cols = [c for c in df_q.columns if c.startswith('score_w')]
    if not score_cols:
        # Single-run mode has a single quality_score — no per-feature diagnostics
        print("No per-window scores found; skipping feature-deviation plots.")
        feature_cols = []
    else:
        # The original features aren't stored in the output, skip deep feature diag
        feature_cols = []

    # ---- Export CSV ----
    export_cols = ['number', 'quality_score', 'anomaly_votes', 'is_consensus_bad']
    if 'start' in bad.columns:
        export_cols.append('start')
    if mode_col in bad.columns:
        export_cols.append(mode_col)
    available = [c for c in export_cols if c in bad.columns]
    bad[available].sort_values('quality_score').to_csv(
        f"{output_prefix}_bad_runs.csv", index=False,
    )

    # ---- Voting heatmap ----
    flag_cols = [c for c in df_q.columns if c.startswith('is_anomaly_w')]
    if flag_cols:
        _plot_voting_heatmap(bad, flag_cols, window_sizes, output_prefix)

    # ---- Type distribution ----
    if mode_col in bad.columns:
        _plot_type_distribution(bad, mode_col, output_prefix)

    # ---- Time distribution ----
    if 'start' in bad.columns:
        _plot_time_distribution(bad, output_prefix)

    # ---- Feature deviations (if feature columns identifiable) ----
    if feature_cols:
        # Build feature list from rate columns
        rate_features = [c for c in df_q.columns
                         if c.endswith('_Rate_Hz') or c.endswith('_Count')]
        if rate_features:
            _plot_feature_deviations(df_q, bad, rate_features, output_prefix,
                                     title_suffix=" (All Modes)")

    # ---- Science-only subset ----
    if mode_col in bad.columns:
        sci_mask = bad[mode_col].astype(str).str.lower().str.contains(
            'bkg|background|science', na=False,
        )
        sci_bad = bad[sci_mask]
        if not sci_bad.empty:
            print(f"  Science-only anomalies: {len(sci_bad)}")
            # Voting heatmap for science only
            if flag_cols:
                _plot_voting_heatmap(sci_bad, flag_cols, window_sizes,
                                     f"{output_prefix}_science_only",
                                     title_suffix=" (Science Runs Only)")
            if rate_features:
                _plot_feature_deviations(df_q, sci_bad, rate_features,
                                         f"{output_prefix}_science_only",
                                         title_suffix=" (Science Runs Only)")
            if 'start' in sci_bad.columns:
                _plot_time_distribution(sci_bad,
                                        f"{output_prefix}_science_only",
                                        title_suffix=" (Science Runs Only)")

    print(f"  Anomaly diagnostics saved with prefix '{output_prefix}'")


def _plot_voting_heatmap(bad_df, flag_cols, window_sizes, prefix,
                         title_suffix=""):
    """Consensus voting heatmap for anomalous runs."""
    hm = bad_df.set_index('number')[flag_cols].astype(int)
    hm['total'] = hm.sum(axis=1)
    hm = hm.sort_values(['total', 'number'], ascending=[False, True]) \
           .drop(columns=['total'])

    fig, ax = plt.subplots(figsize=(10, max(4, len(hm) * 0.3)))
    cmap = sns.color_palette(["#f1f2f6", "#e74c3c"])
    sns.heatmap(hm, cmap=cmap, cbar=False, linewidths=1.0,
                linecolor='black', ax=ax)
    ax.set_title(f"Consensus Voting Heatmap{title_suffix}", pad=20,
                 fontsize=20, weight='bold')
    ax.set_xlabel("Window Size Evaluation", fontsize=16, weight='bold')
    ax.set_ylabel("Run Number", fontsize=16, weight='bold')
    ax.set_xticklabels([f"Window={w}" for w in window_sizes],
                       rotation=45, ha='right')
    pass_p = mpatches.Patch(color='#f1f2f6', label='Pass (Normal)')
    fail_p = mpatches.Patch(color='#e74c3c', label='Fail (Anomaly)')
    ax.legend(handles=[pass_p, fail_p], loc='upper right',
              bbox_to_anchor=(1.4, 1))
    _save_and_close(fig, f"{prefix}_voting_heatmap")


def _plot_type_distribution(bad_df, mode_col, prefix):
    """Bar chart of anomalous runs by calibration type."""
    counts = bad_df[mode_col].value_counts()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=counts.index, y=counts.values, palette='Set1',
                edgecolor='black', linewidth=2.5, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.02 * max(counts.values), str(v),
                ha='center', va='bottom', fontweight='bold',
                fontsize=16, color='black')
    ax.set_title("Distribution of Consensus-Anomalous Runs by Type",
                 pad=20, fontsize=20, weight='bold')
    ax.set_ylabel("Number of Anomalous Runs", fontsize=16, weight='bold')
    ax.set_xlabel("Run Type", fontsize=16, weight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=14, weight='bold')
    _thicken_spines(ax)
    _save_and_close(fig, f"{prefix}_type_distribution")


def _plot_time_distribution(bad_df, prefix, title_suffix=""):
    """Histogram of anomalous runs over time."""
    valid = bad_df['start'].dropna()
    if valid.empty:
        return
    delta_days = (valid.max() - valid.min()).days
    num_bins = min(max(10, delta_days // 7), 50) if delta_days > 0 else 10

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=valid, bins=num_bins, color="#e74c3c",
                 edgecolor='black', linewidth=2, ax=ax)
    ax.set_title(f"Temporal Distribution of Anomalous Runs{title_suffix}",
                 pad=20, fontsize=20, weight='bold')
    ax.set_xlabel("Time", fontsize=16, weight='bold')
    ax.set_ylabel("Count of Anomalous Runs", fontsize=16, weight='bold')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right',
             weight='bold', fontsize=14)
    _thicken_spines(ax)
    _save_and_close(fig, f"{prefix}_time_distribution")


def _plot_feature_deviations(df_full, bad_df, feature_cols, prefix,
                             title_suffix=""):
    """Boxplot of Z-score deviations for top-15 most deviating features."""
    feat = [c for c in feature_cols if c in df_full.columns and c in bad_df.columns]
    if not feat:
        return
    mean = df_full[feat].mean()
    std = df_full[feat].std().replace(0, 1e-9)
    z = (bad_df[feat] - mean) / std
    top15 = z.abs().mean().sort_values(ascending=False).head(15).index.tolist()

    fig, ax = plt.subplots(figsize=(16, 11))
    sns.boxplot(data=z[top15], orient='h', ax=ax, palette='Spectral',
                flierprops=dict(marker='D', markerfacecolor='#e74c3c', markersize=8),
                boxprops=dict(edgecolor='black', linewidth=2.5),
                medianprops=dict(color='#f1c40f', linewidth=3.5))
    ax.axvline(0, color='black', linewidth=4.0)
    ax.axvline(3, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(-3, color='#2980b9', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.set_title(f"Top 15 Feature Deviations in Anomalous Runs{title_suffix}",
                 pad=20, fontsize=22, weight='bold')
    ax.set_xlabel("Z-Score (Standard Deviations from Mean)",
                  fontsize=18, weight='bold')
    ax.set_ylabel("Detector Features", fontsize=18, weight='bold')
    _thicken_spines(ax)
    _save_and_close(fig, f"{prefix}_feature_deviations")


# ===========================================================================
# 5. Per-run diagnostic
# ===========================================================================

def plot_run_diagnostic(run_id, quality_path, run_info_path,
                        output_prefix="run_diag"):
    """Z-score bar chart showing the top-10 most deviating features for one run.

    Requires that the quality HDF5/CSV contain rate features (e.g. from
    the single or batch scoring modes that merge rate CSV columns).
    """
    df_q = _read_quality_file(quality_path)
    df_q['number'] = df_q['number'].astype(str).str.zfill(6)

    # Identify rate-based feature columns
    rate_features = [c for c in df_q.columns
                     if c.endswith('_Rate_Hz') or c.endswith('_Count')]
    if not rate_features:
        print("No rate feature columns found — cannot compute Z-scores.")
        return

    row = df_q[df_q['number'] == run_id]
    if row.empty:
        print(f"Run {run_id} not found in quality dataset.")
        return

    mean = df_q[rate_features].mean()
    std = df_q[rate_features].std().replace(0, 1e-9)
    z = (row[rate_features].iloc[0] - mean) / std
    top10 = z.abs().sort_values(ascending=False).head(10)
    plot_data = z[top10.index]

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ['#B22222' if abs(x) > 3 else '#4682B4' for x in plot_data[::-1]]
    plot_data[::-1].plot(kind='barh', color=colors, edgecolor='black',
                         linewidth=2.0, ax=ax)
    ax.axvline(0, color='black', linewidth=3.0)
    ax.axvline(3, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(-3, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_title(f"Diagnostic: Top Feature Deviations for Run {run_id}",
                 pad=20, fontsize=18)
    ax.set_xlabel("Standard Deviations from Mean (Z-Score)", fontsize=16)
    _thicken_spines(ax)
    _save_and_close(fig, f"{output_prefix}_{run_id}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    set_publication_style()

    parser = argparse.ArgumentParser(
        description="Publication-grade evolution plots for SR2 LowER",
    )
    sub = parser.add_subparsers(dest='command', help='Plot type')

    # ---- rates ----
    p_rates = sub.add_parser('rates', help='Event-rate evolution timeseries')
    p_rates.add_argument('--rates', type=str, required=True,
                         help="Path to sr2_master_run_rates.csv")
    p_rates.add_argument('--calib-intervals', type=str, default=None,
                         help="Path to calibration_intervals_summary.csv")
    p_rates.add_argument('--output', type=str, default='rate_evolution',
                         help="Output filename prefix")

    # ---- quality ----
    p_qual = sub.add_parser('quality', help='Quality-score trend plots')
    p_qual.add_argument('--quality', type=str, required=True,
                        help="Path to quality HDF5 or CSV file")
    p_qual.add_argument('--run-info', type=str, required=True,
                        help="Path to run-info CSV")
    p_qual.add_argument('--output', type=str, default='quality_trend',
                        help="Output filename prefix")
    p_qual.add_argument('--title', type=str, default=None,
                        help="Optional plot title override")

    # ---- features ----
    p_feat = sub.add_parser('features',
                            help='All-features evolution (multi-page PDF)')
    p_feat.add_argument('--run-info', type=str, required=True,
                        help="Path to run-info CSV")
    p_feat.add_argument('--output', type=str,
                        default='all_features_evolution.pdf',
                        help="Output PDF path")
    p_feat.add_argument('--calib-intervals', type=str, default=None,
                        help="Path to calibration_intervals_summary.csv")

    # ---- anomaly ----
    p_anom = sub.add_parser('anomaly',
                            help='Consensus anomaly diagnostic plots')
    p_anom.add_argument('--quality', type=str, required=True,
                        help="Path to quality HDF5 or CSV")
    p_anom.add_argument('--run-info', type=str, required=True,
                        help="Path to run-info CSV")
    p_anom.add_argument('--windows', type=int, nargs='+',
                        default=[1, 2, 4, 8, 10],
                        help="Window sizes used for consensus scoring")
    p_anom.add_argument('--output', type=str, default='anomaly',
                        help="Output filename prefix")

    # ---- diagnose ----
    p_diag = sub.add_parser('diagnose',
                            help='Per-run Z-score diagnostic bar chart')
    p_diag.add_argument('--run-id', type=str, required=True,
                        help="Run ID to diagnose (e.g. 054585)")
    p_diag.add_argument('--quality', type=str, required=True,
                        help="Path to quality HDF5 or CSV")
    p_diag.add_argument('--run-info', type=str, required=True,
                        help="Path to run-info CSV")
    p_diag.add_argument('--output', type=str, default='run_diag',
                        help="Output filename prefix")

    args = parser.parse_args()

    if args.command == 'rates':
        plot_rate_evolution(args.rates, args.calib_intervals, args.output)

    elif args.command == 'quality':
        plot_quality_trend(args.quality, args.run_info,
                           args.output, args.title)

    elif args.command == 'features':
        plot_all_features(args.run_info, args.output, args.calib_intervals)

    elif args.command == 'anomaly':
        plot_anomaly_diagnostics(args.quality, args.run_info,
                                 args.windows, args.output)

    elif args.command == 'diagnose':
        plot_run_diagnostic(args.run_id, args.quality, args.run_info,
                            args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
