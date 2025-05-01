import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150

# List of African countries (ISO3 codes)
african_countries = [
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD", "COM",
    "COD", "DJI", "EGY", "GNQ", "ERI", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB",
    "CIV", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MYT",
    "MAR", "MOZ", "NAM", "NER", "NGA", "REU", "RWA", "STP", "SEN", "SYC", "SLE",
    "SOM", "ZAF", "SSD", "SDN", "SWZ", "TZA", "TGO", "TUN", "UGA", "ESH", "ZMB", "ZWE"
]

# Data from the original notebook
# Using direct data extraction to create visualizations
# Normally would read from the full dataset, but using summary data here

# Export volumes by commodity type (from notebook)
raw_materials_volume = 608397300  # tons (approximate from the notebook)
processed_goods_volume = 137651000  # tons (approximate)
total_volume = raw_materials_volume + processed_goods_volume

# Top export destinations (from notebook)
destinations = ['CHN', 'IND', 'USA', 'DEU', 'GBR', 'FRA', 'ITA', 'NGA', 'ESP', 'HKG']
volumes = [175895900, 159521500, 95919040, 39083800, 30111030, 
           27416470, 23333550, 20131500, 19014200, 18715770]

# Export costs by commodity type (from notebook)
commodities = ['Crude oil', 'Food', 'Other_mining', 'Coal', 'Gas', 
               'Rice_crops', 'Iron_steel', 'Other_metals', 'Other_minerals', 'Refined_oil']
commodity_volumes = [387391300, 91159410, 90053360, 86582440, 70247880, 
                    59829100, 41454310, 38266080, 26336330, 20272600]
commodity_costs = [3211.26, 2907.31, 11081.70, 711.77, 4814.66, 
                  2691.38, 4315.69, 5070.99, 5338.83, 16390.73]

# Classify commodities
raw_materials = ['Crude oil', 'Coal', 'Gas', 'Rice_crops', 'Other_minerals', 'Other_mining']
processed_goods = ['Refined_oil', 'Food', 'Iron_steel', 'Other_metals']

# Transportation costs by region
regions = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
export_costs = [3270, 5741, 3568, 8692, 12191, 8897]  # $ per ton

# 1. COMMODITY COMPOSITION CHART
plt.figure(figsize=(12, 6))
raw_pct = raw_materials_volume / total_volume * 100
processed_pct = processed_goods_volume / total_volume * 100

plt.pie([raw_pct, processed_pct], 
        labels=['Raw Materials', 'Processed Goods'],
        autopct='%1.1f%%', 
        colors=['lightblue', 'lightgreen'],
        explode=(0.05, 0),
        startangle=90,
        shadow=True)
plt.title('African Export Composition', fontsize=16)
plt.tight_layout()
plt.savefig('african_export_composition.png')
plt.close()

# 2. TOP DESTINATIONS FOR AFRICAN EXPORTS
plt.figure(figsize=(14, 7))
colors = ['darkblue' if vol > 100000000 else 'royalblue' for vol in volumes]
bars = plt.bar(destinations, [v/1000000 for v in volumes], color=colors)

# Highlight top 3
for i in range(3):
    bars[i].set_hatch('///')

plt.title('Top 10 Destinations for African Exports', fontsize=16)
plt.xlabel('Destination Country', fontsize=14)
plt.ylabel('Export Volume (Million Tonnes)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Add legend
legend_elements = [
    Patch(facecolor='darkblue', hatch='///', label='Major Value Addition Countries'),
    Patch(facecolor='royalblue', label='Other Destinations')
]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('african_export_destinations.png')
plt.close()

# 3. EXPORT COSTS BY COMMODITY TYPE
plt.figure(figsize=(14, 7))
colors = ['darkred' if commodity in processed_goods else 'steelblue' for commodity in commodities]
plt.bar(commodities, commodity_costs, color=colors)
plt.axhline(y=np.mean(commodity_costs), color='black', linestyle='--', label='Average')

plt.title('African Export Costs by Commodity Type', fontsize=16)
plt.xlabel('Commodity', fontsize=14)
plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Add legend
legend_elements = [
    Patch(facecolor='darkred', label='Processed Goods'),
    Patch(facecolor='steelblue', label='Raw Materials'),
    Patch(facecolor='black', label='Average Cost')
]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('export_costs_by_commodity.png')
plt.close()

# 4. COMPARATIVE EXPORT COSTS BY REGION
plt.figure(figsize=(14, 7))
colors = ['darkred' if region == 'Africa' else 'lightgray' for region in regions]
bars = plt.bar(regions, export_costs, color=colors)

for i, bar in enumerate(bars):
    if regions[i] == 'Africa':
        bar.set_hatch('///')

plt.title('Export Logistics Costs by Region', fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Average Export Cost ($/ton)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Add Annotations
for i, cost in enumerate(export_costs):
    if regions[i] == 'Africa':
        plt.text(i, cost + 300, f"${cost}", ha='center', fontweight='bold')
    else:
        plt.text(i, cost + 300, f"${cost}", ha='center')

plt.tight_layout()
plt.savefig('export_costs_by_region.png')
plt.close()

# 5. COMBINED DEPENDENCY LOOP DATA CHART
plt.figure(figsize=(14, 8))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Export Composition (left)
ax1.pie([raw_pct, processed_pct], 
      labels=['Raw Materials', 'Processed Goods'],
      autopct='%1.1f%%', 
      colors=['lightblue', 'lightgreen'],
      startangle=90)
ax1.set_title('African Export Composition', fontsize=14)

# Freight Rate Comparison (right)
indices = [i for i, r in enumerate(regions) if r in ['Africa', 'Asia', 'Europe', 'North America']]
selected_regions = [regions[i] for i in indices]
selected_costs = [export_costs[i] for i in indices]

colors = ['darkred' if region == 'Africa' else 'lightgray' for region in selected_regions]
ax2.bar(selected_regions, selected_costs, color=colors)
ax2.set_title('Export Logistics Costs Comparison', fontsize=14)
ax2.set_ylabel('Average Export Cost ($/ton)', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add overall title
fig.suptitle('African Trade Dependency Loop - Key Evidence', fontsize=16)
plt.tight_layout()
plt.savefig('dependency_loop_evidence.png')
plt.close()

print("Analysis complete. Visualizations saved to current directory.") 