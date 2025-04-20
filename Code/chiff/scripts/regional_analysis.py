import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
data_path = '../data/wb_data_all_fields_final.csv'
region_data_path = '../data/Final_Country_Classification.csv'

print("Loading country classification data...")
region_df = pd.read_csv(region_data_path)
print(f"Loaded {len(region_df)} country classifications")

# Create mapping dictionaries from country code to fixed_region
country_to_region = dict(zip(region_df['country_code'], region_df['fixed_region']))

print("Loading main dataset...")
# Load only necessary columns to save memory with such a large file
df = pd.read_csv(data_path, usecols=[
    'origin_ISO', 'destination_ISO', 'origin_continent', 'destination_continent',
    'origin_region', 'destination_region', 'income_group', 'Unit logistics costs ($/ton)',
    'flow(tonne)', 'distance(km)', 'year'
])
print(f"Loaded {len(df)} records")

# Filter out empty or zero values
print("\nFiltering out empty or invalid values...")
print(f"Original data shape: {df.shape}")
df_filtered = df[
    (df['flow(tonne)'] > 0) &
    # (df['distance(km)'] > 0) &  # Commented out as per user's example
    (df['Unit logistics costs ($/ton)'] > 0)
]
print(f"Filtered data shape: {df_filtered.shape}")
print(f"Removed {len(df) - len(df_filtered)} records ({(len(df) - len(df_filtered)) / len(df) * 100:.2f}% of data)")

# Use the filtered dataframe for analysis
df = df_filtered

# Map countries to their fixed regions based on ISO codes
df['fixed_origin_region'] = df['origin_ISO'].map(country_to_region)
df['fixed_destination_region'] = df['destination_ISO'].map(country_to_region)

# Fill NaN values with "Unclassified" for any countries not in the mapping
df['fixed_origin_region'] = df['fixed_origin_region'].fillna('Unclassified')
df['fixed_destination_region'] = df['fixed_destination_region'].fillna('Unclassified')

# Basic info
print("\nData Overview:")
print(f"Number of unique origin countries: {df['origin_ISO'].nunique()}")
print(f"Number of unique destination countries: {df['destination_ISO'].nunique()}")
print(f"Years in dataset: {df['year'].unique()}")

# 1. Analysis by Continent (Original)
print("\n1. CONTINENT ANALYSIS (ORIGINAL)")
print("=============================")

# Calculate average costs by origin continent
continent_costs = df.groupby('origin_continent')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
continent_costs = continent_costs.sort_values('mean', ascending=False)

print("\nAverage Logistics Costs by Origin Continent:")
print(continent_costs)

# Continental flow analysis
continent_flow = pd.crosstab(df['origin_continent'], df['destination_continent'], 
                          values=df['Unit logistics costs ($/ton)'], aggfunc='mean')
print("\nAverage Logistics Costs by Origin-Destination Continent Pairs:")
print(continent_flow)

# 2. Analysis by Fixed Region
print("\n2. FIXED REGIONAL ANALYSIS")
print("=======================")

# Calculate average costs by fixed origin region
fixed_region_costs = df.groupby('fixed_origin_region')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
fixed_region_costs = fixed_region_costs.sort_values('mean', ascending=False)

print("\nAverage Logistics Costs by Fixed Origin Region:")
print(fixed_region_costs)

# Fixed regional flow analysis
fixed_region_flow = pd.crosstab(df['fixed_origin_region'], df['fixed_destination_region'], 
                          values=df['Unit logistics costs ($/ton)'], aggfunc='mean')
print("\nAverage Logistics Costs by Fixed Origin-Destination Region Pairs:")
print(fixed_region_flow)

# 3. Analysis by Original Region (For Comparison)
print("\n3. ORIGINAL REGIONAL ANALYSIS")
print("==========================")

# Calculate average costs by origin region
region_costs = df.groupby('origin_region')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
region_costs = region_costs.sort_values('mean', ascending=False)

print("\nAverage Logistics Costs by Original Origin Region:")
print(region_costs)

# Regional flow analysis
region_flow = pd.crosstab(df['origin_region'], df['destination_region'], 
                         values=df['Unit logistics costs ($/ton)'], aggfunc='mean')
print("\nAverage Logistics Costs by Original Origin-Destination Region Pairs:")
print(region_flow)

# 4. Analysis by Income Group
print("\n4. INCOME GROUP ANALYSIS")
print("======================")

# Calculate average costs by origin income group
income_costs = df.groupby('income_group')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
income_costs = income_costs.sort_values('mean', ascending=False)

print(f"\nAverage Logistics Costs by Income Group:")
print(income_costs)

# 5. Additional Analysis - Distribution by Flow Volume
print("\n5. ANALYSIS BY FLOW VOLUME")
print("=======================")

# Create volume bins
df['flow_bin'] = pd.cut(df['flow(tonne)'], 
                        bins=[0, 10, 100, 1000, 10000, 100000, float('inf')],
                        labels=['0-10', '10-100', '100-1000', '1K-10K', '10K-100K', '>100K'])

# Calculate average costs by flow volume
flow_costs = df.groupby('flow_bin')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
print("\nAverage Logistics Costs by Flow Volume (tonnes):")
print(flow_costs)

# 6. Fixed Region and Flow Volume Analysis
print("\n6. FIXED REGION AND FLOW VOLUME ANALYSIS")
print("=====================================")

# Regional costs by flow volume
region_flow_costs = df.groupby(['fixed_origin_region', 'flow_bin'])['Unit logistics costs ($/ton)'].agg(['mean', 'count'])
print("\nTop 20 Region-Flow Volume Combinations by Cost:")
print(region_flow_costs.sort_values('mean', ascending=False).head(20))

# Save results to CSV files
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# continent_costs.to_csv(f"{output_dir}/continent_costs.csv")
# continent_flow.to_csv(f"{output_dir}/continent_flow.csv")
# fixed_region_costs.to_csv(f"{output_dir}/fixed_region_costs.csv")
# fixed_region_flow.to_csv(f"{output_dir}/fixed_region_flow.csv")
# region_costs.to_csv(f"{output_dir}/original_region_costs.csv")
# region_flow.to_csv(f"{output_dir}/original_region_flow.csv")
# income_costs.to_csv(f"{output_dir}/income_costs.csv")
# flow_costs.to_csv(f"{output_dir}/flow_volume_costs.csv")
# region_flow_costs.to_csv(f"{output_dir}/region_flow_volume_costs.csv")

print("\nAnalysis complete. Results saved to CSV files in the output directory.")

# Create visualizations
try:
    plt.figure(figsize=(12, 8))
    sns.barplot(x=continent_costs.index, y=continent_costs['mean'])
    plt.title('Average Logistics Costs by Origin Continent')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/continent_costs.png")
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(continent_flow, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Average Logistics Costs by Continent Pairs')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/continent_flow_heatmap.png")
    
    # Fixed region visualizations
    plt.figure(figsize=(14, 8))
    sns.barplot(x=fixed_region_costs.index, y=fixed_region_costs['mean'])
    plt.title('Average Logistics Costs by Fixed Origin Region')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fixed_region_costs.png")
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(fixed_region_flow, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Average Logistics Costs by Fixed Region Pairs')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fixed_region_flow_heatmap.png")
    
    # Flow volume visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=flow_costs.index, y=flow_costs['mean'])
    plt.title('Average Logistics Costs by Flow Volume')
    plt.ylabel('Unit Logistics Costs ($/ton)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/flow_volume_costs.png")
    
    print("Visualizations saved to output directory.")
except Exception as e:
    print(f"Error creating visualizations: {e}") 