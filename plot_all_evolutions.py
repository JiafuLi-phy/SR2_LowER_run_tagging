import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

# [New 1] Import PdfPages for single PDF output
from matplotlib.backends.backend_pdf import PdfPages

# Force headless backend for remote servers
import matplotlib
matplotlib.use('Agg')

# ==========================================
# [Section 1] Master Styling Configurations 🌟
# ==========================================
color_map = {
    'bkg':                     '#F3F3EF',  
    'kr83m':                   '#FF0026',  
    'rn220':                   '#FF00FF',  
    'ambe':                    '#32CD32',  
    'neutron':                 '#8B4513',  
    'ar37':                    '#FF8C00',  
    'th232':                   '#100CCE',  
}

legend_labels = {
    'bkg': 'Science Run',
    'kr83m': 'Kr-83m',
    'rn220': 'Rn-220',
    'ambe': 'AmBe',
    'neutron': 'Neutron',
    'ar37': 'Ar-37',
    'th232': 'Th-232'
}

marker_map = {
    'bkg': 'o', 
    'kr83m': 's', 
    'rn220': '^', 
    'ambe': 'v', 
    'neutron': 'd', 
    'ar37': 'p',
    'th232': '*',
}

FONT_WEIGHT_BOLD = 'bold'
TITLE_COLOR = 'black'
TITLE_FONT_SIZE = 20    
LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20   # Increased size for the single large legend
AXIS_SPINE_WIDTH = 2.5  
TICK_WIDTH = 2.0
TICK_DIRECTION = 'in'

SCATTER_MARKER_SIZE = 45 
BACKGROUND_ALPHA = 0.5 

# [New 2] Specify the Z-axis variable (for color mapping) and PDF output path
z_col = 'number' # <--- Replace 'number' with your actual column name for the Z-axis color mapping
output_pdf_file = 'all_evolution_plots_colored.pdf' 

# ==========================================
# [Section 2] Configure File Paths & Data Loading
# ==========================================
csv_input_file = '/scratch/midway3/jiafu/SR2_LowER/SRs_Analysis_Hub/SR2/data_organization/run_tagging/results/sr2_run_tagging_info_0.0.5.csv'

print(f"Loading data from: {csv_input_file} ...")
try:
    df = pd.read_csv(csv_input_file)
except Exception as e:
    print(f"❌ Failed to load CSV. Error: {e}")
    sys.exit(1)

if 'start' in df.columns and 'end' in df.columns:
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
df = df.sort_values('start').reset_index(drop=True)

# ==========================================
# [Section 3] Data Type Cleaning & Tagging
# ==========================================
if 'source' in df.columns and 'mode' in df.columns:
    df['data_type'] = df['source'].fillna(df['mode'])
elif 'mode' in df.columns:
    df['data_type'] = df['mode']
else:
    print("❌ Error: Neither 'source' nor 'mode' columns found in CSV.")
    sys.exit(1)

df['data_type'] = df['data_type'].astype(str).str.lower().str.strip()

name_mapping = {
    'none': 'bkg',
    'background': 'bkg',
    'bkg run': 'bkg',
    'kr-83m': 'kr83m',
    'kr83m': 'kr83m',
    'rn-220': 'rn220',
    'rn220': 'rn220',
    'radon': 'rn220',
    'ambe': 'ambe',
    'neutron': 'neutron',
    'ar37': 'ar37',
    'th232': 'th232'
}
df['data_type'] = df['data_type'].replace(name_mapping)

valid_types = list(color_map.keys())
df = df[df['data_type'].isin(valid_types)].copy()
print(f"Filtered data. Kept {len(df)} runs matching the target mode/source list.")
unique_types_present = [t for t in df['data_type'].unique() if t in valid_types]

# ==========================================
# [Section 4] Automatically Identify Target Features
# ==========================================
exclude_cols = ['number', 'mode', 'source', 'start', 'end', 'tags', 'livetime', 
                'peak_positions_mlp_available', 'peak_basics_available', 'data_type']

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plot_targets = [col for col in numeric_cols if col not in exclude_cols]

print(f"🎯 Automatically identified {len(plot_targets)} physical features to plot:")
print(plot_targets)

if not plot_targets:
    print("❌ No valid numerical features found to plot. Exiting.")
    sys.exit(0)

# Global X-Axis Limits
x_min = pd.to_datetime('2023-10-01') 
x_max = pd.to_datetime('2025-04-01') 

# ==========================================
# [Section 5] Calculate Smart Background Intervals
# ==========================================
print("\nCalculating contiguous background blocks...")
bg_blocks = []
df_bg = df.dropna(subset=['start', 'data_type']).copy()
mode_series = df_bg['data_type']

mode_changed = mode_series != mode_series.shift(1)
time_gap = df_bg['start'] - df_bg['start'].shift(1)
is_large_gap = time_gap > pd.Timedelta(days=1.0)
block_id = (mode_changed | is_large_gap).cumsum()

for _, group in df_bg.groupby(block_id):
    dtype = group['data_type'].iloc[0]
    if dtype in color_map:
        s_time = group['start'].iloc[0]
        e_time = group['end'].iloc[-1] if 'end' in group.columns and pd.notnull(group['end'].iloc[-1]) else group['start'].iloc[-1]
        
        delta = e_time - s_time
        if delta < pd.Timedelta(days=3):
            mid = s_time + delta / 2
            s_time = mid - pd.Timedelta(days=1.5)
            e_time = mid + pd.Timedelta(days=1.5)
        
        bg_blocks.append((s_time, e_time, dtype))

def merge_intervals(intervals):
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

bg_dict = {dtype: [] for dtype in valid_types}
for s, e, dtype in bg_blocks:
    bg_dict[dtype].append((s, e))

# ==========================================
# [Section 6] Batch Processing: Exporting to Single PDF
# ==========================================
# [New 3] Open PdfPages context manager to save multiple pages into one file
with PdfPages(output_pdf_file) as pdf:
    # Use enumerate to track the plot index
    for i, y_col in enumerate(plot_targets):
        print(f"➡️ Generating plot for: {y_col} ...")
        
        # Filter valid data (must have y_col, start time, and z_col)
        df_plot = df.dropna(subset=[y_col, 'start', z_col]).copy()
        if df_plot.empty:
            print(f"   ⚠️ Skipping {y_col}: No valid data points.")
            continue

        if y_col == 'elife_mean':
            df_plot = df_plot[(df_plot[y_col] > 0) & (df_plot[y_col] < 100000)]
        
        fig, ax = plt.subplots(figsize=(18, 6))

        # Draw backgrounds
        for dtype, intervals in bg_dict.items():
            if intervals:
                merged_times = merge_intervals(intervals)
                for s, e in merged_times:
                    ax.axvspan(s, e, facecolor=color_map[dtype], edgecolor='none', alpha=BACKGROUND_ALPHA, zorder=0)

        # [New 4] Calculate global min/max for the Z-axis to keep color mapping consistent
        z_min = df_plot[z_col].min()
        z_max = df_plot[z_col].max()
        scatter_plot = None # Store the scatter object for the colorbar

        # Draw scatter points
        for dtype in unique_types_present:
            df_sub = df_plot[df_plot['data_type'] == dtype]
            if not df_sub.empty:
                # [Modified] Use c=df_sub[z_col] and cmap='viridis' instead of a fixed color
                scatter_plot = ax.scatter(df_sub['start'], df_sub[y_col], 
                                          c=df_sub[z_col], cmap='viridis', vmin=z_min, vmax=z_max,
                                          marker=marker_map[dtype], 
                                          s=SCATTER_MARKER_SIZE, alpha=0.85, edgecolors='black', linewidth=0.5, zorder=3)

        # Axis limits logic
        if y_col == 'gate_mean':
            ax.set_ylim(299.9, 300.1)
        elif y_col == 'elife_mean':
            ax.set_ylim(0, 100000)
        else:
            q_low = df_plot[y_col].quantile(0.01)
            q_hi  = df_plot[y_col].quantile(0.99)
            iqr = q_hi - q_low
            if iqr > 0:
                ax.set_ylim(q_low - 0.2 * iqr, q_hi + 0.2 * iqr)

        # Aesthetics
        ax.set_xlim(x_min, x_max)
        ax.set_facecolor('#ffffff') 
        ax.grid(True, alpha=0.3, linestyle='--', color='gray', zorder=0)

        clean_title = y_col.replace('_', ' ').title()
        ax.set_ylabel(clean_title, color=TITLE_COLOR, fontweight=FONT_WEIGHT_BOLD, fontsize=LABEL_FONT_SIZE)
        ax.set_title(f'Evolution of {clean_title}', color=TITLE_COLOR, fontweight=FONT_WEIGHT_BOLD, fontsize=TITLE_FONT_SIZE, pad=20)
        
        ax.set_xlabel('Time (Year-Month)', color=TITLE_COLOR, fontweight=FONT_WEIGHT_BOLD, fontsize=LABEL_FONT_SIZE, labelpad=15)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_SPINE_WIDTH)
            spine.set_color(TITLE_COLOR)
        
        ax.tick_params(direction=TICK_DIRECTION, top=True, right=True, width=TICK_WIDTH, labelsize=15, colors=TITLE_COLOR)
        for label in ax.get_yticklabels(): label.set_fontweight(FONT_WEIGHT_BOLD)
        for label in ax.get_xticklabels(): label.set_fontweight(FONT_WEIGHT_BOLD)

        # ==========================================
        # [Modified] Add Legend ONLY to the first plot
        # ==========================================
        if i == 0:
            legend_handles = []
            for dtype in unique_types_present:
                bg_patch = mpatches.Patch(facecolor=color_map[dtype], edgecolor='black', linewidth=1.5, alpha=BACKGROUND_ALPHA + 0.15, label=legend_labels[dtype])
                legend_handles.append(bg_patch)

            # Pushed higher up (bbox_to_anchor 1.35) and using a larger font to leave space
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.35), 
                      ncol=min(len(unique_types_present), 7), framealpha=0, edgecolor='white', 
                      fancybox=False, prop={'size': LEGEND_FONT_SIZE, 'weight': FONT_WEIGHT_BOLD},
                      handlelength=2.5, handleheight=1.8)


        # [New 5] Add Colorbar on the right to represent the Z values
        if scatter_plot is not None:
            cbar = plt.colorbar(scatter_plot, ax=ax, pad=0.02)
          #  cbar.set_label(z_col.replace('_', ' ').title(), fontweight=FONT_WEIGHT_BOLD, fontsize=16)
            cbar.ax.tick_params(labelsize=12)

        # Save plot to PDF
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig) # Free memory

print(f"\n✅ All {len(plot_targets)} single-variable evolution plots successfully compiled into '{output_pdf_file}'")