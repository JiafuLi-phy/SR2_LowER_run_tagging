import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ==========================================
# 0. CALIBRATION BACKGROUND LIBRARY
# ==========================================
def load_calibration_spans(calib_csv_path):
    """
    Reads the calibration summary CSV and categorizes time intervals based on keywords.
    Assigns specific high-quality, vibrant background colors for each type.
    """
    if not os.path.exists(calib_csv_path):
        print(f"Warning: Calibration file not found at {calib_csv_path}")
        return []

    df_calib = pd.read_csv(calib_csv_path)
    
    # Ensure time columns are proper datetime objects
    df_calib['Start_Time'] = pd.to_datetime(df_calib['Start_Time'], errors='coerce')
    df_calib['End_Time'] = pd.to_datetime(df_calib['End_Time'], errors='coerce')
    df_calib = df_calib.dropna(subset=['Start_Time', 'End_Time'])

    # UPDATED: Added Science Run and made kr83m, ambe, and radon much more vibrant
    calib_styles = {
        'bkg':        {'label': 'Science Run', 'color': "#F3F3EF"}, # Distinct Gold/Yellow
        'background': {'label': 'Science Run', 'color': "#F3F3EF"},  # Distinct Gold/Yellow
        'kr83m':      {'label': 'Kr-83m',      'color': "#FF0026"}, # Vibrant Deep Sky Blue
        'ambe':       {'label': 'AmBe',        'color': '#32CD32'}, # Vibrant Lime Green
        'rn':         {'label': 'Radon',       'color': '#FF00FF'}, # Vibrant Magenta/Fuchsia
        'radon':      {'label': 'Radon',       'color': '#FF00FF'}, # Fallback for radon
        'ar37':       {'label': 'Ar-37',       'color': '#FF8C00'}, # Dark Orange
        'neutron':    {'label': 'Neutron',     'color': '#8B4513'}, # Saddle Brown
        'th232':      {'label': 'Th-232',      'color': "#100CCE"} # Deep Pink
    }

    spans_to_plot = []
    for _, row in df_calib.iterrows():
        mode_str = str(row['Mode']).lower()
        
        # Check which keyword exists in the mode string
        for key, style in calib_styles.items():
            if key in mode_str:
                spans_to_plot.append({
                    'start': row['Start_Time'],
                    'end': row['End_Time'],
                    'label': style['label'],
                    'color': style['color']
                })
                break # Stop searching once a match is found for this block
                
    return spans_to_plot

def apply_calib_backgrounds(ax, spans_to_plot, plotted_labels_set):
    """
    Draws the background color spans on the provided matplotlib Axis.
    Uses a Set to ensure we don't create duplicate legend entries.
    """
    for span in spans_to_plot:
        # Only assign a label if it hasn't been added to the legend yet
        label = span['label'] if span['label'] not in plotted_labels_set else None
        
        if label:
            plotted_labels_set.add(label)

        # Plot the background span (zorder=0 puts it behind the grid and data)
        # alpha=0.35 keeps it colorful but transparent enough to see the data
        ax.axvspan(span['start'], span['end'], color=span['color'], alpha=0.5, zorder=0, label=label)


# ==========================================
# 1. GLOBAL PLOT STYLE SETTINGS
# ==========================================
plt.style.use('default') 
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'sans-serif',     
    'font.weight': 'bold',           
    'axes.labelweight': 'bold',      
    'axes.titleweight': 'bold',      
    'axes.grid': True,               
    'grid.alpha': 0.6,               
    'grid.linestyle': ':',           
    'axes.linewidth': 2.0,           
    'axes.edgecolor': 'black',       
    'xtick.color': 'black',          
    'ytick.color': 'black',
})

# ==========================================
# 2. FILE PATHS & DATA LOADING
# ==========================================
file_path = '/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/sr2_master_run_rates.csv'
calib_file_path = '/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/split_modes/calibration_intervals_summary.csv'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    df = pd.read_csv(file_path)
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    df = df.sort_values('Start_Date').reset_index(drop=True)

    # Pre-load calibration and science run backgrounds
    calib_spans = load_calibration_spans(calib_file_path)

    rate_columns = {
        'Gate_Event_Rate_Hz': 'Gate Events',
        'Cathode_Event_Rate_Hz': 'Cathode Events',
        'S1_Only_Heavy_Rate_Hz': 'S1-only (Heavy)',
        'S2_Only_SE_Rate_Hz': 'S2-only (SE)',
        'Wall_Event_Rate_Hz': 'Wall Events'
    }

    # ==========================================
    # 3. INITIALIZE PLOT
    # ==========================================
    fig, axes = plt.subplots(len(rate_columns), 1, figsize=(12, 20), sharex=True)
    
    # Track which calibration labels have been added to prevent a massive legend
    global_plotted_labels = set()

    for i, (col, label) in enumerate(rate_columns.items()):
        ax = axes[i]
        x = df['Start_Date']
        y = df[col]
        
        # --- Apply Backgrounds FIRST ---
        if calib_spans:
            apply_calib_backgrounds(ax, calib_spans, global_plotted_labels)
        
        # --- Core Styling Implementation ---
        ax.plot(x, y, color='#cccccc', linestyle='-', linewidth=1.2, zorder=1)
        sc = ax.scatter(x, y, c=y, cmap='RdYlGn', s=35, 
                        edgecolors='black', linewidths=0.6, zorder=2)
        
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label('Rate [Hz]', fontweight='bold')
        cbar.outline.set_linewidth(2.0) 
        
        ax.set_ylabel('Rate [Hz]', fontweight='bold', color='black')
        ax.set_title(label, loc='center', fontsize=16, fontweight='bold', pad=12, color='black')
        ax.set_facecolor('#ffffff')
        
        ax.tick_params(axis='both', which='major', width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        
        # Add legend to the TOP subplot only
        if i == 0 and global_plotted_labels:
            # Dynamically calculate legend columns (max 5 per row to avoid crowding)
            legend_cols = min(len(global_plotted_labels), 5)
            # Increased Y-anchor slightly (1.40) to make sure 2 rows of legend fit if needed
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40), 
                      ncol=legend_cols, framealpha=1.0, edgecolor='black', 
                      fontsize=12, fancybox=False)

    # ==========================================
    # 4. AXIS FORMATTING & SAVING
    # ==========================================
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.setp(ax.get_xticklabels(), fontweight='bold', color='black')
    plt.setp(ax.get_yticklabels(), fontweight='bold', color='black')

    plt.xlabel('Time (Year-Month)', fontsize=16, labelpad=15, fontweight='bold', color='black')
    
    # Raised title slightly higher to accommodate the potentially thicker legend
    plt.suptitle('Evolution of XENONnT SR2 LowER Event Rates', y=0.995, fontsize=22, fontweight='bold', color='black')

    # Adjusted top margin slightly to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    base_name = 'sr2_lowER_evolution_rates_styled'
    plt.savefig(f"{base_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{base_name}.pdf", dpi=300, bbox_inches='tight')

    print(f"Styled publication plots successfully saved as:\n - {base_name}.png\n - {base_name}.pdf")