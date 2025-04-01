#!/usr/bin/env python
# coding: utf-8

"""
Transport Anomalies Analysis
============================
This script investigates unusual patterns and anomalies in the transport data, including:
1. RoRo classification in air transport
2. High-cost countries/routes (SDN, MOZ, MDG, COD, ZWE) and routes to American Samoa (ASM)
3. Comoros (COM) air cost spike
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

print("Starting Transport Anomalies Analysis...")

# Load the dataset
try:
    df = pd.read_csv("imputed_full_matrix_at_centroid.csv")
except FileNotFoundError:
    df = pd.read_csv("../../../data/imputed_full_matrix_at_centroid.csv")

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Create a report directory if it doesn't exist
import os
report_dir = "anomalies_report"
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

#############################
# 1. RoRo in Air Transport Analysis
#############################
print("\n\n===== RoRo in Air Transport Analysis =====")

# Filter for RoRo shipments via air
roro_air = df[(df['Mode_name'] == 'Air') & (df['ship_type'] == 'RoRo')]

print(f"Number of RoRo shipments via air: {len(roro_air)}")

if len(roro_air) > 0:
    # Basic statistics
    print("\nBasic statistics for RoRo air shipments:")
    print(f"Average cost: ${roro_air['Unit logistics costs ($/ton)'].mean():.2f} per ton")
    print(f"Average distance: {roro_air['distance(km)'].mean():.2f} km")
    print(f"Average flow: {roro_air['flow(tonne)'].mean():.2f} tonnes")
    
    # Origin-destination analysis
    print("\nTop origin-destination pairs for RoRo air shipments:")
    origin_dest_counts = roro_air.groupby(['origin_ISO', 'destination_ISO']).size().reset_index(name='count')
    origin_dest_costs = roro_air.groupby(['origin_ISO', 'destination_ISO'])['Unit logistics costs ($/ton)'].mean().reset_index(name='avg_cost')
    origin_dest = pd.merge(origin_dest_counts, origin_dest_costs, on=['origin_ISO', 'destination_ISO'])
    origin_dest = origin_dest.sort_values('count', ascending=False)
    print(origin_dest.head(10))
    
    # Commodity analysis
    print("\nTop commodities for RoRo air shipments:")
    commodity_counts = roro_air['IFM_HS'].value_counts()
    print(commodity_counts.head(10))
    
    # Cost comparison with regular air shipments
    regular_air = df[(df['Mode_name'] == 'Air') & (df['ship_type'] != 'RoRo')]
    print("\nCost comparison - RoRo air vs Regular air:")
    print(f"RoRo air - Average cost: ${roro_air['Unit logistics costs ($/ton)'].mean():.2f} per ton")
    print(f"Regular air - Average cost: ${regular_air['Unit logistics costs ($/ton)'].mean():.2f} per ton")
    
    # Visualization - Cost comparison boxplot
    plt.figure(figsize=(10, 6))
    comparison_data = pd.DataFrame({
        'RoRo Air': roro_air['Unit logistics costs ($/ton)'].sample(min(1000, len(roro_air)), replace=True).values,
        'Regular Air': regular_air['Unit logistics costs ($/ton)'].sample(1000, replace=True).values
    })
    sns.boxplot(data=comparison_data)
    plt.title('Cost Comparison: RoRo Air vs Regular Air', fontsize=16)
    plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{report_dir}/roro_air_cost_comparison.png")
    plt.close()
    
    # Sample of actual records for inspection
    print("\nSample of RoRo air shipment records:")
    sample_cols = ['origin_ISO', 'destination_ISO', 'IFM_HS', 'distance(km)', 
                  'flow(tonne)', 'Unit logistics costs ($/ton)', 'container_type']
    print(roro_air[sample_cols].head(10))
    
    # Save detailed data for further analysis
    roro_air.to_csv(f"{report_dir}/roro_air_shipments.csv", index=False)
    
    # Possible explanation
    print("\nPossible explanations for RoRo in air transport:")
    print("1. Data miscategorization - RoRo is typically associated with sea transport")
    print("2. Special cargo (e.g., vehicles, racing cars) that require roll-on/roll-off loading onto cargo planes")
    print("3. Multi-modal transport where the main leg is by air but with RoRo components")
    print("4. Specialized military or humanitarian missions involving vehicles transported by air")

else:
    print("No RoRo shipments via air found in the dataset.")

#############################
# 2. High-Cost Countries Analysis
#############################
print("\n\n===== High-Cost Countries Analysis =====")

# List of high-cost countries to analyze
high_cost_countries = ['SDN', 'MOZ', 'MDG', 'COD', 'ZWE']
print(f"Analyzing high-cost countries: {', '.join(high_cost_countries)}")

# Filter data for these countries (as origin)
high_cost_data = df[df['origin_ISO'].isin(high_cost_countries)]
print(f"Number of shipments from high-cost countries: {len(high_cost_data)}")

# Basic statistics by country
print("\nAverage costs by country:")
country_stats = high_cost_data.groupby('origin_ISO')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'min', 'max', 'count'])
print(country_stats)

# Transport mode breakdown by country
print("\nTransport mode distribution by country (%):")
mode_country = pd.crosstab(high_cost_data['origin_ISO'], high_cost_data['Mode_name'], normalize='index') * 100
print(mode_country.round(2))

# Commodity analysis for each country
print("\nTop expensive commodities by country:")
for country in high_cost_countries:
    country_data = high_cost_data[high_cost_data['origin_ISO'] == country]
    if len(country_data) > 0:
        top_commodities = country_data.groupby('IFM_HS')['Unit logistics costs ($/ton)'].mean().sort_values(ascending=False)
        print(f"\n{country}:")
        print(top_commodities.head(5))

# Top trading partners (destinations) for each country
print("\nTop trading partners by country:")
for country in high_cost_countries:
    country_data = high_cost_data[high_cost_data['origin_ISO'] == country]
    if len(country_data) > 0:
        top_partners = country_data.groupby('destination_ISO').agg({
            'Unit logistics costs ($/ton)': 'mean',
            'flow(tonne)': 'sum'
        }).sort_values('flow(tonne)', ascending=False)
        print(f"\n{country} - Top Partners by Volume:")
        print(top_partners.head(5))

# Visualization - Cost by transport mode for each country
plt.figure(figsize=(14, 8))
sns.boxplot(x='origin_ISO', y='Unit logistics costs ($/ton)', hue='Mode_name', data=high_cost_data)
plt.title('Transport Costs by Mode for High-Cost Countries', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.yscale('log')  # Log scale to handle extreme values
plt.legend(title='Transport Mode')
plt.tight_layout()
plt.savefig(f"{report_dir}/high_cost_countries_by_mode.png")
plt.close()

# Visualization - Distance vs Cost scatter plot
plt.figure(figsize=(12, 8))
for country in high_cost_countries:
    country_data = high_cost_data[high_cost_data['origin_ISO'] == country]
    if len(country_data) > 0:
        plt.scatter(country_data['distance(km)'], country_data['Unit logistics costs ($/ton)'], 
                   alpha=0.7, label=country)
plt.title('Distance vs. Cost for High-Cost Countries', fontsize=16)
plt.xlabel('Distance (km)', fontsize=14)
plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{report_dir}/high_cost_countries_distance_vs_cost.png")
plt.close()

#############################
# 3. American Samoa (ASM) Routes Analysis
#############################
print("\n\n===== American Samoa (ASM) Routes Analysis =====")

# Filter for routes to American Samoa
asm_routes = df[df['destination_ISO'] == 'ASM']
print(f"Number of routes to American Samoa: {len(asm_routes)}")

if len(asm_routes) > 0:
    # Basic statistics
    print("\nBasic statistics for routes to American Samoa:")
    print(f"Average cost: ${asm_routes['Unit logistics costs ($/ton)'].mean():.2f} per ton")
    print(f"Median cost: ${asm_routes['Unit logistics costs ($/ton)'].median():.2f} per ton")
    print(f"Maximum cost: ${asm_routes['Unit logistics costs ($/ton)'].max():.2f} per ton")
    
    # Origin countries analysis
    print("\nTop origin countries for shipments to American Samoa:")
    origin_stats = asm_routes.groupby('origin_ISO').agg({
        'Unit logistics costs ($/ton)': ['mean', 'count'],
        'flow(tonne)': 'sum'
    }).sort_values(('Unit logistics costs ($/ton)', 'mean'), ascending=False)
    print(origin_stats.head(10))
    
    # Analyze the most expensive route: SHN (Saint Helena) to ASM
    shn_asm = asm_routes[asm_routes['origin_ISO'] == 'SHN']
    if len(shn_asm) > 0:
        print("\nAnalysis of Saint Helena (SHN) to American Samoa (ASM) route:")
        print(f"Number of shipments: {len(shn_asm)}")
        print(f"Average cost: ${shn_asm['Unit logistics costs ($/ton)'].mean():.2f} per ton")
        print(f"Average distance: {shn_asm['distance(km)'].mean():.2f} km")
        print(f"Transport modes used: {shn_asm['Mode_name'].value_counts().to_dict()}")
        print(f"Commodities shipped: {shn_asm['IFM_HS'].value_counts().to_dict()}")
    
    # Visualization - Top 10 most expensive routes to ASM
    top_routes = asm_routes.groupby('origin_ISO')['Unit logistics costs ($/ton)'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    top_routes.plot(kind='bar', color='darkblue')
    plt.title('Top 10 Most Expensive Routes to American Samoa', fontsize=16)
    plt.xlabel('Origin Country', fontsize=14)
    plt.ylabel('Average Unit Logistics Costs ($/ton)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{report_dir}/asm_top_expensive_routes.png")
    plt.close()
    
    # Save detailed data for further analysis
    asm_routes.to_csv(f"{report_dir}/american_samoa_routes.csv", index=False)

else:
    print("No routes to American Samoa found in the dataset.")

#############################
# 4. Comoros (COM) Air Cost Spike Analysis
#############################
print("\n\n===== Comoros (COM) Air Cost Spike Analysis =====")

# Filter for Comoros air shipments
com_air = df[(df['origin_ISO'] == 'COM') & (df['Mode_name'] == 'Air')]
print(f"Number of air shipments from Comoros: {len(com_air)}")

if len(com_air) > 0:
    # Basic statistics
    print("\nBasic statistics for Comoros air shipments:")
    print(f"Average cost: ${com_air['Unit logistics costs ($/ton)'].mean():.2f} per ton")
    print(f"Median cost: ${com_air['Unit logistics costs ($/ton)'].median():.2f} per ton")
    print(f"Maximum cost: ${com_air['Unit logistics costs ($/ton)'].max():.2f} per ton")
    
    # Distance analysis
    print("\nCost by distance range:")
    # Create distance bins
    com_air['distance_bin'] = pd.cut(com_air['distance(km)'], 
                                    bins=[0, 2000, 5000, 10000, float('inf')],
                                    labels=['0-2000 km', '2000-5000 km', '5000-10000 km', '10000+ km'])
    distance_stats = com_air.groupby('distance_bin').agg({
        'Unit logistics costs ($/ton)': ['mean', 'count'],
        'flow(tonne)': 'mean'
    })
    print(distance_stats)
    
    # Destination analysis
    print("\nTop destinations for Comoros air shipments:")
    dest_stats = com_air.groupby('destination_ISO').agg({
        'Unit logistics costs ($/ton)': ['mean', 'count'],
        'distance(km)': 'mean',
        'flow(tonne)': 'mean'
    }).sort_values(('Unit logistics costs ($/ton)', 'mean'), ascending=False)
    print(dest_stats.head(10))
    
    # Commodity analysis
    print("\nCosts by commodity for Comoros air shipments:")
    commodity_stats = com_air.groupby('IFM_HS').agg({
        'Unit logistics costs ($/ton)': ['mean', 'count'],
        'flow(tonne)': 'mean'
    }).sort_values(('Unit logistics costs ($/ton)', 'mean'), ascending=False)
    print(commodity_stats.head(10))
    
    # Compare with other island nations
    island_nations = ['MUS', 'SYC', 'MDG']  # Mauritius, Seychelles, Madagascar
    island_air = df[(df['origin_ISO'].isin(island_nations)) & (df['Mode_name'] == 'Air')]
    
    print("\nComparison with other island nations (Air transport):")
    island_comparison = pd.DataFrame({
        'COM': com_air['Unit logistics costs ($/ton)'].mean(),
        'MUS': island_air[island_air['origin_ISO'] == 'MUS']['Unit logistics costs ($/ton)'].mean(),
        'SYC': island_air[island_air['origin_ISO'] == 'SYC']['Unit logistics costs ($/ton)'].mean(),
        'MDG': island_air[island_air['origin_ISO'] == 'MDG']['Unit logistics costs ($/ton)'].mean()
    }, index=['Average Cost ($/ton)'])
    print(island_comparison)
    
    # Visualization - Costs by distance for Comoros vs other islands
    plt.figure(figsize=(12, 8))
    plt.scatter(com_air['distance(km)'], com_air['Unit logistics costs ($/ton)'], 
               alpha=0.7, label='Comoros (COM)', color='red')
    
    for country in island_nations:
        country_data = island_air[island_air['origin_ISO'] == country]
        if len(country_data) > 0:
            plt.scatter(country_data['distance(km)'], country_data['Unit logistics costs ($/ton)'], 
                       alpha=0.5, label=country)
    
    plt.title('Air Transport Costs vs. Distance: Comoros vs. Other Islands', fontsize=16)
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{report_dir}/comoros_air_costs_comparison.png")
    plt.close()
    
    # Save detailed data for further analysis
    com_air.to_csv(f"{report_dir}/comoros_air_shipments.csv", index=False)
    
    # Possible explanations
    print("\nPossible explanations for Comoros air cost spike:")
    print("1. Limited air cargo capacity requiring charter flights")
    print("2. Low trade volumes leading to higher per-unit costs")
    print("3. Lack of competition among air cargo providers")
    print("4. High-value or specialized cargo requiring special handling")
    print("5. Multi-leg transport with transshipment costs")

else:
    print("No air shipments from Comoros found in the dataset.")

#############################
# Summary of findings
#############################
print("\n\n===== Summary of Findings =====")

print("""
1. RoRo in Air Transport:
   - [Summary will be populated based on actual data findings]

2. High-Cost Countries (SDN, MOZ, MDG, COD, ZWE):
   - [Summary will be populated based on actual data findings]

3. American Samoa (ASM) Routes:
   - [Summary will be populated based on actual data findings]

4. Comoros (COM) Air Cost Spike:
   - [Summary will be populated based on actual data findings]
""")

print(f"\nDetailed analysis files saved in the '{report_dir}' directory.") 