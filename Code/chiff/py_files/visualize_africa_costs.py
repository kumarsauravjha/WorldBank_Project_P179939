import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Path to the generated dataset
file_path = "/Users/abishekchiffon/Documents/Technical/Masters/sem 4/captsone/data/africa_avg_cost_by_commodity.csv"

print("Reading the dataset...")
df = pd.read_csv(file_path)

# Output directory for visualizations
output_dir = "/Users/abishekchiffon/Documents/Technical/Masters/sem 4/captsone/data/visualizations/"
import os
os.makedirs(output_dir, exist_ok=True)

# ----- Visualization 1: Average cost by commodity -----
print("Creating visualization: Average cost by commodity")
plt.figure(figsize=(12, 8))
commodity_costs = df.groupby('IFM_HS')['avg_unit_cost_per_ton'].mean().sort_values(ascending=False)
commodity_costs.plot(kind='bar', color=sns.color_palette('viridis', len(commodity_costs)))
plt.title('Average Logistics Cost by Commodity Type', fontsize=16)
plt.xlabel('Commodity Type', fontsize=14)
plt.ylabel('Average Cost ($/ton)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}avg_cost_by_commodity.png", dpi=300)
plt.close()

# ----- Visualization 2: Top 15 African countries by average cost -----
print("Creating visualization: Top/Bottom 15 countries by average cost")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Top 15 countries with highest costs
top_countries = df.groupby('origin_ISO')['avg_unit_cost_per_ton'].mean().sort_values(ascending=False).head(15)
top_countries.plot(kind='barh', color=sns.color_palette('Reds_r', len(top_countries)), ax=ax1)
ax1.set_title('African Countries with Highest Average Logistics Costs', fontsize=16)
ax1.set_xlabel('Average Cost ($/ton)', fontsize=14)
ax1.set_ylabel('Country ISO Code', fontsize=14)

# Bottom 15 countries with lowest costs
bottom_countries = df.groupby('origin_ISO')['avg_unit_cost_per_ton'].mean().sort_values().head(15)
bottom_countries.plot(kind='barh', color=sns.color_palette('Greens_r', len(bottom_countries)), ax=ax2)
ax2.set_title('African Countries with Lowest Average Logistics Costs', fontsize=16)
ax2.set_xlabel('Average Cost ($/ton)', fontsize=14)
ax2.set_ylabel('Country ISO Code', fontsize=14)

plt.tight_layout()
plt.savefig(f"{output_dir}top_bottom_countries_by_cost.png", dpi=300)
plt.close()

# ----- Visualization 3: Cost distribution for each commodity -----
print("Creating visualization: Cost distribution by commodity")
plt.figure(figsize=(14, 10))
sns.boxplot(x='IFM_HS', y='avg_unit_cost_per_ton', data=df.sample(min(50000, len(df))), palette='viridis')
plt.title('Distribution of Logistics Costs by Commodity Type', fontsize=16)
plt.xlabel('Commodity Type', fontsize=14)
plt.ylabel('Cost ($/ton)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, df['avg_unit_cost_per_ton'].quantile(0.99))  # Limit y-axis to 99th percentile for better visualization
plt.tight_layout()
plt.savefig(f"{output_dir}cost_distribution_by_commodity.png", dpi=300)
plt.close()

# ----- Visualization 4: Heatmap of costs for top 10 origin and destination pairs -----
print("Creating visualization: Heatmap of costs for top pairs")
# Get top 10 origin and destination countries by frequency
top_origins = df['origin_ISO'].value_counts().head(10).index
top_destinations = df['destination_ISO'].value_counts().head(10).index

# Filter data for these countries and get average cost
heatmap_data = df[df['origin_ISO'].isin(top_origins) & df['destination_ISO'].isin(top_destinations)]
heatmap_pivot = heatmap_data.pivot_table(
    values='avg_unit_cost_per_ton', 
    index='origin_ISO', 
    columns='destination_ISO', 
    aggfunc='mean'
)

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_pivot, annot=True, fmt='.1f', cmap='viridis', linewidths=0.5)
plt.title('Average Logistics Cost Between Top African Origins and Global Destinations', fontsize=16)
plt.xlabel('Destination Country', fontsize=14)
plt.ylabel('Origin Country (Africa)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}cost_heatmap_top_pairs.png", dpi=300)
plt.close()

# ----- Visualization 5: Cost comparison for selected commodities across top 5 countries -----
print("Creating visualization: Cost comparison for selected commodities")
# Select a few interesting commodities
selected_commodities = ['Electronic_devices', 'Food', 'Textile', 'Refined_oil', 'Coal']
# Get top 5 countries by total shipping volume
top5_countries = top_origins[:5]

# Filter data
comparison_data = df[df['origin_ISO'].isin(top5_countries) & df['IFM_HS'].isin(selected_commodities)]
comparison_pivot = comparison_data.pivot_table(
    values='avg_unit_cost_per_ton', 
    index='origin_ISO', 
    columns='IFM_HS', 
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
comparison_pivot.plot(kind='bar', figsize=(12, 8))
plt.title('Comparison of Logistics Costs for Key Commodities Across Top African Countries', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Average Cost ($/ton)', fontsize=14)
plt.legend(title='Commodity Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}commodity_cost_comparison.png", dpi=300)
plt.close()

print(f"All visualizations have been saved to {output_dir}") 