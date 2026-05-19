import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 1. Adjust global plot style to closely match the target visualization
plt.style.use('default') # Reset to base style before customizing
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'sans-serif',     # Use sans-serif font for a cleaner look
    'font.weight': 'bold',           # Apply bold weight globally
    'axes.labelweight': 'bold',      # Bold axis labels
    'axes.titleweight': 'bold',      # Bold subplot titles
    'axes.grid': True,               # Enable grid
    'grid.alpha': 0.6,               # Set grid transparency
    'grid.linestyle': ':',           # Use dotted lines for the grid
    'axes.linewidth': 2.0,           # Thicken the axes borders (spines)
    'axes.edgecolor': 'black',       # Set border color to black
    'xtick.color': 'black',          # Set tick colors to black
    'ytick.color': 'black',
})

# Define the file path
file_path = '/scratch/midway3/jiafu/SR2_LowER/run_tagging_lower/sr2_master_run_rates.csv'

# Check if the data file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'Start_Date' column to datetime objects for accurate time plotting
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    
    # Sort by date to ensure chronological connection of data points
    df = df.sort_values('Start_Date').reset_index(drop=True)

    # Map the CSV column names to professional display labels
    rate_columns = {
        'Gate_Event_Rate_Hz': 'Gate Events',
        'Cathode_Event_Rate_Hz': 'Cathode Events',
        'S1_Only_Heavy_Rate_Hz': 'S1-only (Heavy)',
        'S2_Only_SE_Rate_Hz': 'S2-only (SE)',
        'Wall_Event_Rate_Hz': 'Wall Events'
    }

    # 2. Initialize the figure with subplots sharing the X-axis
    fig, axes = plt.subplots(len(rate_columns), 1, figsize=(12, 20), sharex=True)

    for i, (col, label) in enumerate(rate_columns.items()):
        ax = axes[i]
        
        x = df['Start_Date']
        y = df[col]
        
        # --- Core Styling Implementation ---
        
        # Step A: Plot the underlying connecting line
        # Uses a thin, light gray line. zorder=1 pushes it to the background.
        ax.plot(x, y, color='#cccccc', linestyle='-', linewidth=1.2, zorder=1)
        
        # Step B: Plot the colored scatter points on top
        # cmap='RdYlGn' applies a Red-Yellow-Green colormap dynamically based on the Y-value (c=y).
        # zorder=2 ensures points are drawn over the connecting line.
        sc = ax.scatter(x, y, c=y, cmap='RdYlGn', s=35, 
                        edgecolors='black', linewidths=0.6, zorder=2)
        
        # Step C: Add a Colorbar to the right of each subplot
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label('Rate [Hz]', fontweight='bold')
        cbar.outline.set_linewidth(2.0) # Thicken colorbar border to match plot borders
        
        # --- End Core Styling ---

        # Set individual subplot labels and titles
        ax.set_ylabel('Rate [Hz]', fontweight='bold', color='black')
        ax.set_title(label, loc='center', fontsize=16, fontweight='bold', pad=12, color='black')
        
        # Set subplot background color to pure white
        ax.set_facecolor('#ffffff')
        
        # Customize tick mark thickness and length for a bolder appearance
        ax.tick_params(axis='both', which='major', width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)

    # 3. Optimize X-axis Time Formatting
    ax = plt.gca()
    # Format X-axis tick labels as Year-Month (e.g., 2023-11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Apply bold font weight and black color specifically to tick labels
    plt.setp(ax.get_xticklabels(), fontweight='bold', color='black')
    plt.setp(ax.get_yticklabels(), fontweight='bold', color='black')

    # 4. Final Global Customization
    plt.xlabel('Time (Year-Month)', fontsize=16, labelpad=15, fontweight='bold', color='black')
    
    # Add the main figure title, adjusting the 'y' parameter to position it properly
    plt.suptitle('Evolution of XENONnT SR2 LowER Event Rates', y=0.98, fontsize=22, fontweight='bold', color='black')

    # Adjust layout to prevent label clipping and accommodate colorbars
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # 5. Save the output in multiple formats
    base_name = 'sr2_lowER_evolution_rates_styled'
    
    # Save as high-resolution PNG
    plt.savefig(f"{base_name}.png", dpi=600, bbox_inches='tight')
    # Save as PDF for vector graphics quality
    plt.savefig(f"{base_name}.pdf", dpi=300, bbox_inches='tight')
    
    plt.show()

    print(f"Styled publication plots successfully saved as:\n - {base_name}.png\n - {base_name}.pdf")