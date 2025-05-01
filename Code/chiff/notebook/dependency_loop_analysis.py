"""
Dependency Loop Analysis for African Exports
============================================
This script analyzes the dependency loop in African trade patterns:
1. High freight rates make exporting finished products uneconomical
2. Value addition happens abroad (mainly in developed countries)
3. Africa imports processed goods at marked-up prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Create output directory
output_dir = "dependency_loop_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of African countries (ISO3 codes)
african_countries = [
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD", "COM",
    "COD", "DJI", "EGY", "GNQ", "ERI", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB",
    "CIV", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MYT",
    "MAR", "MOZ", "NAM", "NER", "NGA", "REU", "RWA", "STP", "SEN", "SYC", "SLE",
    "SOM", "ZAF", "SSD", "SDN", "SWZ", "TZA", "TGO", "TUN", "UGA", "ESH", "ZMB", "ZWE"
]

print("Starting Dependency Loop Analysis for African Exports...")

# Load the dataset
try:
    df = pd.read_csv("imputed_full_matrix_at_centroid.csv")
except FileNotFoundError:
    df = pd.read_csv("../../../../../data/imputed_full_matrix_at_centroid.csv")

print(f"Full dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Filter for exports from African countries only
df_africa = df[df['origin_ISO'].isin(african_countries)]
print(f"African exports dataset: {df_africa.shape[0]} rows ({df_africa.shape[0]/df.shape[0]:.1%} of total data)")

# Define North American countries
north_american_countries = ['USA', 'CAN', 'MEX']

# Define raw vs processed goods categories
raw_commodities = ['Crude oil', 'Coal', 'Gas', 'Rice_crops', 'Other_mining', 'Other_minerals', 'Food']
processed_commodities = ['Refined_oil', 'Transport_equipment', 'Electronic_devices', 'Chemicals_plastic', 
                        'Machinery', 'Iron_steel', 'Other_metals', 'Livestock', 'Wood_products']

# Add classification of raw vs processed
df_africa['commodity_type'] = df_africa['IFM_HS'].apply(
    lambda x: 'Raw material' if x in raw_commodities else 'Processed good')

# Filter for imports to Africa (destinations in Africa)
df_africa_imports = df[df['destination_ISO'].isin(african_countries)]

#############################
# PART 1: HIGH FREIGHT RATES FOR AFRICAN EXPORTS
#############################
print("\n\n===== 1. HIGH FREIGHT RATES FOR AFRICAN EXPORTS =====")

# Compare freight rates between Africa and other regions
regions = {
    'Africa': african_countries,
    'North America': north_american_countries,
    'Asia': ['CHN', 'JPN', 'KOR', 'IND', 'IDN', 'THA', 'MYS', 'SGP', 'VNM', 'PHL', 'PAK', 'BGD'],
    'Europe': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'BEL', 'CHE', 'AUT', 'PRT', 'SWE', 'NOR', 'FIN', 'DNK', 'IRL', 'LUX'],
}

# Add region classifications
df['origin_region'] = df['origin_ISO'].apply(
    lambda x: next((region for region, countries in regions.items() if x in countries), 'Other'))

# Calculate average freight rates by origin region
freight_rates_by_region = df.groupby('origin_region')['Unit logistics costs ($/ton)'].agg(['mean', 'median']).reset_index()
freight_rates_by_region = freight_rates_by_region.sort_values('mean', ascending=False)

print("\nAverage freight rates by region:")
print(freight_rates_by_region)

# Plot freight rates comparison by region
plt.figure(figsize=(10, 6))
sns.barplot(x='origin_region', y='mean', data=freight_rates_by_region, 
           palette=['#d73027' if x == 'Africa' else '#4575b4' for x in freight_rates_by_region['origin_region']])
plt.title('Average Freight Rates by Origin Region', fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Average Logistics Costs ($/ton)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(freight_rates_by_region['mean']):
    plt.text(i, v + 100, f"${v:.2f}", ha='center')
plt.tight_layout()
plt.savefig(f"{output_dir}/freight_rates_by_region.png")
plt.close()

# Compare freight rates for raw vs processed goods from Africa
freight_rates_by_commodity_type = df_africa.groupby(['commodity_type', 'Mode_name'])[
    'Unit logistics costs ($/ton)'].mean().reset_index()

print("\nFreight rates for raw vs processed goods from Africa:")
print(freight_rates_by_commodity_type)

# Plot freight rates for raw vs processed goods
plt.figure(figsize=(10, 6))
sns.barplot(x='commodity_type', y='Unit logistics costs ($/ton)', 
           hue='Mode_name', data=freight_rates_by_commodity_type)
plt.title('Freight Rates: Raw Materials vs Processed Goods from Africa', fontsize=16)
plt.xlabel('Commodity Type', fontsize=14)
plt.ylabel('Average Logistics Costs ($/ton)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/freight_rates_raw_vs_processed.png")
plt.close()

#############################
# PART 2: VALUE ADDITION HAPPENS ABROAD
#############################
print("\n\n===== 2. VALUE ADDITION HAPPENS ABROAD =====")

# Analyze the proportion of raw vs processed goods in African exports
raw_vs_processed_exports = df_africa.groupby('commodity_type').agg({
    'flow(tonne)': 'sum'
}).reset_index()
raw_vs_processed_exports['percentage'] = (raw_vs_processed_exports['flow(tonne)'] / 
                                         raw_vs_processed_exports['flow(tonne)'].sum() * 100)

print("\nProportion of raw vs processed goods in African exports:")
print(raw_vs_processed_exports)

# Pie chart of raw vs processed goods exports
plt.figure(figsize=(10, 7))
plt.pie(raw_vs_processed_exports['percentage'], 
       labels=raw_vs_processed_exports['commodity_type'], 
       autopct='%1.1f%%', 
       startangle=90, 
       colors=['#d73027', '#4575b4'])
plt.title('African Exports: Raw Materials vs Processed Goods', fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/raw_vs_processed_exports_pie.png")
plt.close()

# Top export destinations for raw materials from Africa
top_raw_destinations = df_africa[df_africa['commodity_type'] == 'Raw material'].groupby(
    'destination_ISO')['flow(tonne)'].sum().sort_values(ascending=False).head(10)

print("\nTop destinations for raw materials from Africa:")
print(top_raw_destinations)

# Destinations for raw materials (highlighting North America)
raw_exports_by_dest_region = df_africa[df_africa['commodity_type'] == 'Raw material'].copy()
raw_exports_by_dest_region['destination_region'] = raw_exports_by_dest_region['destination_ISO'].apply(
    lambda x: next((region for region, countries in regions.items() if x in countries), 'Other'))

raw_exports_by_region = raw_exports_by_dest_region.groupby('destination_region')['flow(tonne)'].sum().reset_index()
raw_exports_by_region['percentage'] = (raw_exports_by_region['flow(tonne)'] / 
                                     raw_exports_by_region['flow(tonne)'].sum() * 100)
raw_exports_by_region = raw_exports_by_region.sort_values('flow(tonne)', ascending=False)

print("\nDestination regions for raw materials from Africa:")
print(raw_exports_by_region)

# Plot destinations for raw materials
plt.figure(figsize=(10, 6))
bars = plt.bar(raw_exports_by_region['destination_region'], raw_exports_by_region['flow(tonne)'] / 1_000_000,
             color=['#fc8d59' if x == 'North America' else '#91bfdb' for x in raw_exports_by_region['destination_region']])
plt.title('Destinations for Raw Materials from Africa', fontsize=16)
plt.xlabel('Destination Region', fontsize=14)
plt.ylabel('Volume (Million Tonnes)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f"{height:.1f}M", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f"{output_dir}/raw_exports_destinations.png")
plt.close()

#############################
# PART 3: COSTLY TO BUY BACK PROCESSED GOODS
#############################
print("\n\n===== 3. COSTLY TO BUY BACK PROCESSED GOODS =====")

# Filter for processed goods imported to Africa
processed_imports_to_africa = df_africa_imports[
    (df_africa_imports['IFM_HS'].isin(processed_commodities))]

# Filter for exports originating from North America
processed_imports_from_na = processed_imports_to_africa[
    processed_imports_to_africa['origin_ISO'].isin(north_american_countries)]

# Compare costs of processed goods by origin
processed_imports_cost_by_origin = processed_imports_to_africa.groupby(
    'origin_region')['Unit logistics costs ($/ton)'].mean().reset_index()
processed_imports_cost_by_origin = processed_imports_cost_by_origin.sort_values(
    'Unit logistics costs ($/ton)', ascending=False)

print("\nAverage cost of processed goods imported to Africa by origin region:")
print(processed_imports_cost_by_origin)

# Plot import costs of processed goods by origin
plt.figure(figsize=(10, 6))
bars = plt.bar(processed_imports_cost_by_origin['origin_region'], 
             processed_imports_cost_by_origin['Unit logistics costs ($/ton)'],
             color=['#fc8d59' if x == 'North America' else '#91bfdb' for x in processed_imports_cost_by_origin['origin_region']])
plt.title('Cost of Processed Goods Imported to Africa by Origin', fontsize=16)
plt.xlabel('Origin Region', fontsize=14)
plt.ylabel('Average Logistics Costs ($/ton)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 100,
             f"${height:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f"{output_dir}/processed_imports_cost_by_origin.png")
plt.close()

# Compare raw export vs processed import prices for specific commodities
# For example: Crude oil exports vs Refined oil imports
try:
    crude_exports = df_africa[df_africa['IFM_HS'] == 'Crude oil']['Unit logistics costs ($/ton)'].mean()
    refined_imports = df_africa_imports[df_africa_imports['IFM_HS'] == 'Refined_oil']['Unit logistics costs ($/ton)'].mean()
    
    print(f"\nAverage logistics cost for crude oil exports: ${crude_exports:.2f} per ton")
    print(f"Average logistics cost for refined oil imports: ${refined_imports:.2f} per ton")
    print(f"Price markup ratio: {refined_imports/crude_exports:.2f}x")
    
    # Add more commodity pairs as needed
    commodity_pairs = [
        ('Crude oil', 'Refined_oil'),
        ('Other_mining', 'Iron_steel'),
        ('Food', 'Chemicals_plastic')
    ]
    
    export_import_comparison = []
    for raw, processed in commodity_pairs:
        if df_africa[df_africa['IFM_HS'] == raw].shape[0] > 0 and df_africa_imports[df_africa_imports['IFM_HS'] == processed].shape[0] > 0:
            export_cost = df_africa[df_africa['IFM_HS'] == raw]['Unit logistics costs ($/ton)'].mean()
            import_cost = df_africa_imports[df_africa_imports['IFM_HS'] == processed]['Unit logistics costs ($/ton)'].mean()
            markup = import_cost / export_cost
            export_import_comparison.append({
                'Raw Material': raw,
                'Processed Good': processed,
                'Export Cost ($/ton)': export_cost,
                'Import Cost ($/ton)': import_cost,
                'Markup Ratio': markup
            })
    
    comparison_df = pd.DataFrame(export_import_comparison)
    print("\nComparison of export vs import logistics costs:")
    print(comparison_df)
    
    # Plot the comparison
    plt.figure(figsize=(12, 7))
    x = range(len(comparison_df))
    width = 0.35
    plt.bar([i - width/2 for i in x], comparison_df['Export Cost ($/ton)'], width, label='Raw Material Export Cost', color='#91bfdb')
    plt.bar([i + width/2 for i in x], comparison_df['Import Cost ($/ton)'], width, label='Processed Good Import Cost', color='#fc8d59')
    
    plt.xlabel('Commodity Pairs', fontsize=14)
    plt.ylabel('Logistics Costs ($/ton)', fontsize=14)
    plt.title('Export vs Import Costs: Raw Materials vs Processed Goods', fontsize=16)
    plt.xticks(x, [f"{r} → {p}" for r, p in zip(comparison_df['Raw Material'], comparison_df['Processed Good'])], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add markup ratios
    for i, markup in enumerate(comparison_df['Markup Ratio']):
        plt.text(i, max(comparison_df.iloc[i]['Export Cost ($/ton)'], comparison_df.iloc[i]['Import Cost ($/ton)']) + 500,
                f"{markup:.1f}x", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/export_import_cost_comparison.png")
    plt.close()
except Exception as e:
    print(f"Error creating export-import comparison: {e}")

#############################
# PART 4: THE DEPENDENCY LOOP - COMBINED VISUALIZATION
#############################
print("\n\n===== 4. THE DEPENDENCY LOOP - COMBINED VISUALIZATION =====")

# Create a circular flow diagram to visualize the dependency loop
try:
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define the circular layout
    circle_radius = 4
    center = (5, 5)
    
    # Node positions (on a circle)
    angles = [0, 2*np.pi/3, 4*np.pi/3]  # 120 degree spacing
    nodes = [(center[0] + circle_radius*np.cos(angle), center[1] + circle_radius*np.sin(angle)) for angle in angles]
    
    # Node labels and values
    node_labels = ['African\nRaw Materials', 'Foreign\nProcessing', 'African\nConsumers']
    
    # Calculate values for edges (from actual data)
    raw_export_volume = df_africa[df_africa['commodity_type'] == 'Raw material']['flow(tonne)'].sum() / 1_000_000
    raw_export_cost = df_africa[df_africa['commodity_type'] == 'Raw material']['Unit logistics costs ($/ton)'].mean()
    
    processed_import_volume = df_africa_imports[df_africa_imports['IFM_HS'].isin(processed_commodities)]['flow(tonne)'].sum() / 1_000_000
    processed_import_cost = df_africa_imports[df_africa_imports['IFM_HS'].isin(processed_commodities)]['Unit logistics costs ($/ton)'].mean()
    
    # Edge labels with actual data
    edge_labels = [
        f"Raw exports\n{raw_export_volume:.1f}M tons\n${raw_export_cost:.2f}/ton",
        f"Value addition\nabroad",
        f"Processed imports\n{processed_import_volume:.1f}M tons\n${processed_import_cost:.2f}/ton"
    ]
    
    # Draw nodes
    node_colors = ['#abdda4', '#fdae61', '#d7191c']
    for i, (pos, label, color) in enumerate(zip(nodes, node_labels, node_colors)):
        circle = plt.Circle(pos, 0.8, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Calculate control points for curved edges
    def get_control_points(start, end, curvature=0.2):
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
        # Calculate perpendicular vector
        dx, dy = end[0] - start[0], end[1] - start[1]
        perp = (-dy, dx)  # Perpendicular
        norm = np.sqrt(perp[0]**2 + perp[1]**2)
        perp = (perp[0]/norm, perp[1]/norm)
        
        # Control point is perpendicular to midpoint
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        control = (mid[0] + curvature*dist*perp[0], mid[1] + curvature*dist*perp[1])
        return control
    
    # Draw edges
    edge_colors = ['#1a9641', '#dfc27d', '#d73027']
    for i in range(3):
        start = nodes[i]
        end = nodes[(i + 1) % 3]
        
        # Create curved paths
        control = get_control_points(start, end)
        
        path = Path([start, control, end], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        patch = PathPatch(path, facecolor='none', edgecolor=edge_colors[i], linewidth=2, alpha=0.8, arrowstyle='->')
        ax.add_patch(patch)
        
        # Add edge labels at control points
        ax.text(control[0], control[1], edge_labels[i], ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    plt.title("The African Trade Dependency Loop", fontsize=18, pad=20)
    
    # Add annotations explaining the cycle
    cycle_text = """THE DEPENDENCY CYCLE:
    1. High shipping costs make exporting finished goods uneconomical
    2. Raw materials are exported at relatively low value
    3. Value-addition occurs in foreign countries
    4. Processed goods are imported back at marked-up prices
    5. African industries remain underdeveloped"""
    
    plt.figtext(0.5, 0.05, cycle_text, ha='center', va='center', fontsize=12, 
               bbox=dict(facecolor='#f7f7f7', edgecolor='#cccccc', boxstyle='round,pad=1'))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dependency_loop_diagram.png")
    plt.close()
except Exception as e:
    print(f"Error creating dependency loop diagram: {e}")

print(f"\nDependency loop analysis completed. Results saved in '{output_dir}' directory.") 