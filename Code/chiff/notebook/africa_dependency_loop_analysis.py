"""
African Trade Dependency Loop Analysis
======================================
This script analyzes the dependency loop in African trade described as:
1. Expensive to ship out: High freight rates make exporting finished products uneconomical
2. Value-add happens abroad: Raw materials exported, processed elsewhere
3. Costly to buy back: Processed goods imported at marked-up prices

Using data from the exports analysis to validate these patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

# Create output directories
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

print("Starting African Dependency Loop Analysis...")

# Load the dataset
try:
    df = pd.read_csv("imputed_full_matrix_at_centroid.csv")
except FileNotFoundError:
    df = pd.read_csv("../../../../../data/imputed_full_matrix_at_centroid.csv")

print(f"Full dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Filter for exports from African countries
df_africa = df[df['origin_ISO'].isin(african_countries)]
print(f"African exports dataset: {df_africa.shape[0]} rows ({df_africa.shape[0]/df.shape[0]:.1%} of total data)")

# Filter for imports to African countries
df_africa_imports = df[df['destination_ISO'].isin(african_countries)]
print(f"African imports dataset: {df_africa_imports.shape[0]} rows ({df_africa_imports.shape[0]/df.shape[0]:.1%} of total data)")

#############################
# 1. EXPENSIVE TO SHIP OUT - ANALYZE AFRICAN EXPORT COSTS
#############################
print("\n\n===== PART 1: EXPENSIVE TO SHIP OUT =====")

# Calculate average export costs ($/ton) by origin region
# First define regions for all countries
def get_region(country_code):
    regions = {
        'Africa': african_countries,
        'Europe': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'BEL', 'CHE', 'AUT', 'PRT', 'SWE', 'NOR', 'FIN', 'DNK', 'IRL', 'LUX'],
        'North America': ['USA', 'CAN', 'MEX'],
        'South America': ['BRA', 'ARG', 'CHL', 'COL', 'PER', 'VEN', 'ECU', 'BOL', 'PRY', 'URY'],
        'Asia': ['CHN', 'JPN', 'KOR', 'IND', 'IDN', 'THA', 'MYS', 'SGP', 'VNM', 'PHL', 'PAK', 'BGD', 'SAU', 'ARE', 'IRN', 'TUR'],
        'Oceania': ['AUS', 'NZL', 'PNG', 'FJI']
    }
    
    for region, countries in regions.items():
        if country_code in countries:
            return region
    return 'Other'

# Add region to the dataframe
df['origin_region'] = df['origin_ISO'].apply(get_region)

# Calculate average export costs by region
export_costs_by_region = df.groupby('origin_region')['Unit logistics costs ($/ton)'].mean().reset_index()
export_costs_by_region = export_costs_by_region.sort_values('Unit logistics costs ($/ton)')

# Visualize export costs by region
plt.figure(figsize=(12, 6))
sns.barplot(x='origin_region', y='Unit logistics costs ($/ton)', data=export_costs_by_region)
plt.title('Average Export Logistics Costs by Region', fontsize=16)
plt.xlabel('Origin Region', fontsize=14)
plt.ylabel('Average Unit Logistics Costs ($/ton)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/export_costs_by_region.png")
plt.close()

# Calculate cost per distance by region (to account for geographic disadvantages)
df['cost_per_km'] = df['Unit logistics costs ($/ton)'] / df['distance(km)']
cost_per_km_by_region = df.groupby('origin_region')['cost_per_km'].mean().reset_index()
cost_per_km_by_region = cost_per_km_by_region.sort_values('cost_per_km')

# Visualize cost per km by region
plt.figure(figsize=(12, 6))
sns.barplot(x='origin_region', y='cost_per_km', data=cost_per_km_by_region)
plt.title('Average Export Logistics Costs per km by Region', fontsize=16)
plt.xlabel('Origin Region', fontsize=14)
plt.ylabel('Average Cost per km ($/ton-km)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/export_costs_per_km_by_region.png")
plt.close()

# Classify commodities as raw materials or processed goods
# This is a simplified classification based on the IFM_HS categories in the data
raw_materials = ['Crude oil', 'Coal', 'Gas', 'Rice_crops', 'Other_minerals', 'Other_mining', 'Livestock', 'Vegetables']
processed_goods = ['Refined_oil', 'Food', 'Chemicals_plastic', 'Electronic_devices', 'Transport_equipment', 'Iron_steel', 'Other_metals']

# Add commodity type to the dataframe
def classify_commodity(commodity):
    if commodity in raw_materials:
        return 'Raw Materials'
    elif commodity in processed_goods:
        return 'Processed Goods'
    else:
        return 'Other'

df_africa['commodity_type'] = df_africa['IFM_HS'].apply(classify_commodity)

# Calculate the composition of African exports
african_exports_composition = df_africa.groupby('commodity_type')['flow(tonne)'].sum()
african_exports_composition = african_exports_composition / african_exports_composition.sum() * 100

# Visualize the composition of African exports
plt.figure(figsize=(10, 6))
african_exports_composition.plot(kind='pie', autopct='%1.1f%%')
plt.title('Composition of African Exports by Commodity Type', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.savefig(f"{output_dir}/african_exports_composition.png")
plt.close()

# Compare export costs for raw materials vs processed goods from Africa
export_costs_by_type = df_africa.groupby('commodity_type')['Unit logistics costs ($/ton)'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='commodity_type', y='Unit logistics costs ($/ton)', data=export_costs_by_type)
plt.title('African Export Costs by Commodity Type', fontsize=16)
plt.xlabel('Commodity Type', fontsize=14)
plt.ylabel('Average Unit Logistics Costs ($/ton)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/african_export_costs_by_type.png")
plt.close()

#############################
# 2. VALUE-ADD HAPPENS ABROAD - ANALYZE EXPORT DESTINATIONS AND COMMODITIES
#############################
print("\n\n===== PART 2: VALUE-ADD HAPPENS ABROAD =====")

# Analyze top destinations for African raw material exports
raw_exports = df_africa[df_africa['commodity_type'] == 'Raw Materials']
top_raw_destinations = raw_exports.groupby('destination_ISO')['flow(tonne)'].sum().reset_index()
top_raw_destinations = top_raw_destinations.sort_values('flow(tonne)', ascending=False).head(10)

# Add region information to the top destinations
top_raw_destinations['destination_region'] = top_raw_destinations['destination_ISO'].apply(get_region)

# Visualize top destinations for raw materials
plt.figure(figsize=(12, 6))
bars = plt.bar(top_raw_destinations['destination_ISO'], top_raw_destinations['flow(tonne)'] / 1e6)

# Color bars by region
for i, bar in enumerate(bars):
    region = top_raw_destinations.iloc[i]['destination_region']
    if region == 'Asia':
        bar.set_color('royalblue')
    elif region == 'Europe':
        bar.set_color('forestgreen')
    elif region == 'North America':
        bar.set_color('tomato')
    elif region == 'Africa':
        bar.set_color('purple')
    else:
        bar.set_color('gray')

plt.title('Top 10 Destinations for African Raw Material Exports', fontsize=16)
plt.xlabel('Destination Country', fontsize=14)
plt.ylabel('Total Volume (Million Tonnes)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='royalblue', label='Asia'),
    Patch(facecolor='forestgreen', label='Europe'),
    Patch(facecolor='tomato', label='North America'),
    Patch(facecolor='purple', label='Africa'),
    Patch(facecolor='gray', label='Other')
]
plt.legend(handles=legend_elements, title='Region')

plt.tight_layout()
plt.savefig(f"{output_dir}/top_raw_destinations.png")
plt.close()

# Analyze top raw material exports
top_raw_commodities = raw_exports.groupby('IFM_HS')['flow(tonne)'].sum().reset_index()
top_raw_commodities = top_raw_commodities.sort_values('flow(tonne)', ascending=False)

# Visualize top raw material exports
plt.figure(figsize=(12, 6))
plt.bar(top_raw_commodities['IFM_HS'], top_raw_commodities['flow(tonne)'] / 1e6, color='darkblue')
plt.title('African Raw Material Exports by Commodity', fontsize=16)
plt.xlabel('Commodity', fontsize=14)
plt.ylabel('Total Volume (Million Tonnes)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/raw_exports_by_commodity.png")
plt.close()

#############################
# 3. COSTLY TO BUY BACK - ANALYZE IMPORT COSTS FOR PROCESSED GOODS
#############################
print("\n\n===== PART 3: COSTLY TO BUY BACK =====")

# Add commodity type to import data
df_africa_imports['commodity_type'] = df_africa_imports['IFM_HS'].apply(classify_commodity)

# Filter for processed goods imports to Africa
processed_imports = df_africa_imports[df_africa_imports['commodity_type'] == 'Processed Goods']

# Calculate average import costs for processed goods by origin region
processed_imports['origin_region'] = processed_imports['origin_ISO'].apply(get_region)
import_costs_by_region = processed_imports.groupby('origin_region')[['Unit logistics costs ($/ton)', 'flow(tonne)']].agg({
    'Unit logistics costs ($/ton)': 'mean',
    'flow(tonne)': 'sum'
}).reset_index()
import_costs_by_region = import_costs_by_region.sort_values('Unit logistics costs ($/ton)')

# Visualize import costs for processed goods by origin region
plt.figure(figsize=(12, 6))
bars = plt.bar(import_costs_by_region['origin_region'], import_costs_by_region['Unit logistics costs ($/ton)'])

# Add volume as text labels
for i, bar in enumerate(bars):
    volume_millions = import_costs_by_region.iloc[i]['flow(tonne)'] / 1e6
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f"{volume_millions:.1f}M", ha='center')

plt.title('Average Import Costs for Processed Goods to Africa by Origin Region', fontsize=16)
plt.xlabel('Origin Region', fontsize=14)
plt.ylabel('Average Unit Logistics Costs ($/ton)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/processed_import_costs_by_region.png")
plt.close()

# Calculate the top processed goods being imported to Africa
top_processed_imports = processed_imports.groupby('IFM_HS')['flow(tonne)'].sum().reset_index()
top_processed_imports = top_processed_imports.sort_values('flow(tonne)', ascending=False)

# Visualize top processed goods imports
plt.figure(figsize=(12, 6))
plt.bar(top_processed_imports['IFM_HS'], top_processed_imports['flow(tonne)'] / 1e6, color='darkred')
plt.title('Top Processed Goods Imported to Africa', fontsize=16)
plt.xlabel('Commodity', fontsize=14)
plt.ylabel('Total Volume (Million Tonnes)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/top_processed_imports.png")
plt.close()

#############################
# 4. THE DEPENDENCY LOOP - VISUALIZE THE CYCLE
#############################
print("\n\n===== PART 4: THE DEPENDENCY LOOP VISUALIZATION =====")

# Create a visualization that captures the entire dependency loop
# First get the necessary data
raw_export_volume = african_exports_composition['Raw Materials'] / 100 * df_africa['flow(tonne)'].sum()
processed_export_volume = african_exports_composition['Processed Goods'] / 100 * df_africa['flow(tonne)'].sum()

processed_import_volume = processed_imports['flow(tonne)'].sum()

# Ratio of processed goods imported vs exported
if processed_export_volume > 0:
    import_export_ratio = processed_import_volume / processed_export_volume
else:
    import_export_ratio = float('inf')

# Average costs
raw_export_cost = df_africa[df_africa['commodity_type'] == 'Raw Materials']['Unit logistics costs ($/ton)'].mean()
processed_export_cost = df_africa[df_africa['commodity_type'] == 'Processed Goods']['Unit logistics costs ($/ton)'].mean()
processed_import_cost = processed_imports['Unit logistics costs ($/ton)'].mean()

# Create a Sankey diagram representing the dependency loop
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create figure
    fig = go.Figure()
    
    # Define nodes
    nodes = ['African Raw Materials', 'Global Markets', 'Processed Goods', 'African Markets']
    
    # Define links
    source = [0, 1, 2]  # from African Raw Materials, Global Markets, Processed Goods
    target = [1, 2, 3]  # to Global Markets, Processed Goods, African Markets
    value = [raw_export_volume / 1e6, raw_export_volume / 1e6 * 0.8, processed_import_volume / 1e6]  # Adjusted for readability
    
    # Cost labels
    label = [f"Export cost: ${raw_export_cost:.2f}/ton",
             f"Value addition abroad",
             f"Import cost: ${processed_import_cost:.2f}/ton"]
    
    # Node colors
    node_color = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 
                  'rgba(44, 160, 44, 0.8)', 'rgba(214, 39, 40, 0.8)']
    
    # Link colors
    link_color = ['rgba(31, 119, 180, 0.5)', 'rgba(255, 127, 14, 0.5)', 'rgba(44, 160, 44, 0.5)']
    
    # Add Sankey diagram
    fig.add_trace(go.Sankey(
        arrangement = "snap",
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = nodes,
            color = node_color
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            label = label,
            color = link_color
        )
    ))
    
    # Update layout
    fig.update_layout(
        title_text="African Trade Dependency Loop",
        font_size=14,
        width=1000,
        height=600
    )
    
    # Save as HTML file
    fig.write_html(f"{output_dir}/dependency_loop_sankey.html")
    
    # Also save as image if orca is available
    try:
        fig.write_image(f"{output_dir}/dependency_loop_sankey.png")
    except:
        print("Could not save Sankey diagram as PNG. Consider installing 'orca' or using the HTML version.")
    
except ImportError:
    print("Plotly not available for Sankey diagram. Creating an alternative visualization.")
    
    # Create an alternative circular flow diagram
    plt.figure(figsize=(12, 8))
    
    # Draw a circle divided into segments
    ax = plt.subplot(111, polar=True)
    
    # Define data points for the circle
    theta = np.linspace(0, 2*np.pi, 4, endpoint=False)
    radii = [1, 1, 1, 1]
    width = np.pi/2
    
    # Create the bars
    bars = ax.bar(theta, radii, width=width, bottom=0.3)
    
    # Set colors and labels
    bars[0].set_color('darkblue')
    bars[0].set_label(f'African Raw Material Exports\n{raw_export_volume/1e6:.1f}M tons\n${raw_export_cost:.2f}/ton')
    
    bars[1].set_color('darkgreen')
    bars[1].set_label('Value Addition Abroad')
    
    bars[2].set_color('darkred')
    bars[2].set_label(f'Processed Goods Import\n{processed_import_volume/1e6:.1f}M tons\n${processed_import_cost:.2f}/ton')
    
    bars[3].set_color('purple')
    bars[3].set_label('High Costs to African Consumers\nLimited Local Industry')
    
    # Add arrows between segments
    arrow_kwargs = dict(arrowstyle='->', linewidth=2, color='black')
    
    ax.annotate("", xy=(np.pi/4, 0.8), xytext=(3*np.pi/4, 0.8), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(5*np.pi/4, 0.8), xytext=(7*np.pi/4, 0.8), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(2*np.pi, 0.2), xytext=(0, 0.2), arrowprops=arrow_kwargs)
    
    # Remove radial labels and grid
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    
    # Add title and custom legend
    plt.title('African Trade Dependency Loop', fontsize=16, pad=20)
    plt.legend(handles=bars, loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dependency_loop_circle.png")
    plt.close()

# Create a summary text file
with open(f"{output_dir}/dependency_loop_summary.txt", "w") as f:
    f.write("AFRICAN TRADE DEPENDENCY LOOP ANALYSIS\n")
    f.write("======================================\n\n")
    
    f.write("1. EXPENSIVE TO SHIP OUT\n")
    f.write(f"   - Average export cost from Africa: ${df_africa['Unit logistics costs ($/ton)'].mean():.2f}/ton\n")
    f.write(f"   - African exports are {african_exports_composition['Raw Materials']:.1f}% raw materials\n")
    f.write(f"   - Raw material export cost: ${raw_export_cost:.2f}/ton\n")
    f.write(f"   - Processed goods export cost: ${processed_export_cost:.2f}/ton\n\n")
    
    f.write("2. VALUE-ADD HAPPENS ABROAD\n")
    f.write(f"   - Top raw material exports: {', '.join(top_raw_commodities['IFM_HS'].head(3).tolist())}\n")
    f.write(f"   - Top destinations: {', '.join(top_raw_destinations['destination_ISO'].head(3).tolist())}\n\n")
    
    f.write("3. COSTLY TO BUY BACK\n")
    f.write(f"   - Processed goods import cost: ${processed_import_cost:.2f}/ton\n")
    f.write(f"   - Top processed imports: {', '.join(top_processed_imports['IFM_HS'].head(3).tolist())}\n")
    if processed_export_volume > 0:
        f.write(f"   - Africa imports {import_export_ratio:.1f} times more processed goods than it exports\n\n")
    
    f.write("THE DEPENDENCY LOOP\n")
    f.write("   - High shipping costs make it uneconomical to export processed goods\n")
    f.write("   - Instead, Africa primarily exports raw materials\n")
    f.write("   - These materials are processed abroad, adding significant value\n")
    f.write("   - The processed goods are then imported back to Africa at higher costs\n")
    f.write("   - This reinforces Africa's dependency on raw material exports and\n")
    f.write("     limits development of local processing industries\n")

print(f"\nAnalysis results saved in the '{output_dir}' directory.") 