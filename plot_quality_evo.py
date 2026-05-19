import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 22,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.labelweight': 'bold',      # Make axis titles bold
    'axes.labelcolor': 'black',      # Make axis titles black
    'xtick.color': 'black',          # Make tick labels black
    'ytick.color': 'black',          # Make tick labels black
})

# Define the file path
file_path = '/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/sr2_master_run_rates.csv'

# Check if the data file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load the dataset
    df = pd.read_csv(file_path)

    # 1. Data Preprocessing
    # Convert 'Start_Date' column to datetime objects for accurate time plotting
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    
    # Sort by date to ensure the line plots connect chronologically
    df = df.sort_values('Start_Date').reset_index(drop=True)

    # Map the CSV columns to professional English labels
    rate_columns = {
        'Gate_Event_Rate_Hz': 'Gate Events',
        'Cathode_Event_Rate_Hz': 'Cathode Events',
        'S1_Only_Heavy_Rate_Hz': 'S1-only (Heavy)',
        'S2_Only_SE_Rate_Hz': 'S2-only (SE)',
        'Wall_Event_Rate_Hz': 'Wall Events'
    }

    # 2. Initialize the figure and subplots
    # Create a vertical stack of subplots to handle different orders of magnitude
    fig, axes = plt.subplots(len(rate_columns), 1, figsize=(12, 20), sharex=True)
    colors = sns.color_palette("magma", len(rate_columns))

    for i, (col, label) in enumerate(rate_columns.items()):
        ax = axes[i]
        
        # Plot the evolution curve with markers
        ax.plot(df['Start_Date'], df[col], marker='o', linestyle='-', 
                color=colors[i], markersize=6, linewidth=2, alpha=0.9, label=label)
        
        # Set individual subplot Y-axis label and Title
        ax.set_ylabel('Rate [Hz]', fontweight='bold', color='black')
        ax.set_title(label, loc='left', fontsize=15, fontweight='bold', pad=12, color='black')
        
        # Format Borders (Spines): Make them thick and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)
        
        # Light grey background for subplots to enhance contrast
        ax.set_facecolor('#fdfdfd')

    # 3. Optimize X-axis Time Formatting
    ax = plt.gca()
    # Locate major ticks automatically and format as "YYYY-MM-DD HH:MM"
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # --- NEW: Bold the tick labels ---
    # Bold X-axis time labels
    plt.setp(ax.get_xticklabels(), fontweight='bold', color='black')
    # Bold Y-axis rate labels
    plt.setp(ax.get_yticklabels(), fontweight='bold', color='black')

    # 4. Final Global Customization
    plt.xlabel('Time (UTC)', fontsize=16, labelpad=15, fontweight='bold', color='black')
    plt.suptitle('Evolution of XENONnT SR2 LowER Event Rates', y=0.96, fontweight='bold', color='black')

    # Adjust layout to prevent label clipping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 5. Save the output in multiple formats
    base_name = 'sr2_lowER_evolution_rates'
    
    # Save as high-resolution PNG for presentations/web
    plt.savefig(f"{base_name}.png", dpi=600, bbox_inches='tight')
    
    # Save as PDF for high-quality scientific publication/vector format
    plt.savefig(f"{base_name}.pdf", dpi=300, bbox_inches='tight')
    
    plt.show()

    print(f"Publication-quality plots successfully saved as:\n - {base_name}.png\n - {base_name}.pdf")