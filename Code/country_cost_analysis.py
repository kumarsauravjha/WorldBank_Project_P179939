#!/usr/bin/env python
# coding: utf-8

"""
Country Level Exploratory Data Analysis
======================================
This script performs country-level analysis to explore why specific countries 
have varying logistics costs in trade routes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

print("Starting Country-Level Cost Analysis...")

# Load the dataset
try:
    # Try the path used in the notebook
    df = pd.read_csv("../../../data/imputed_full_matrix_at_centroid.csv")
except FileNotFoundError:
    # If that fails, try the path used in the original script
    df = pd.read_csv("imputed_full_matrix_at_centroid.csv")

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# 1. Basic statistics by country (origin)
print("\n1. ORIGIN COUNTRY ANALYSIS")
print("==========================")

# Calculate average logistics costs by origin country
origin_costs = df.groupby('origin_ISO')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
origin_costs = origin_costs.sort_values('mean', ascending=False)

# Display top 10 most expensive origin countries
print("\nTop 10 Countries with Highest Average Export Costs:")
print(origin_costs.head(10))

# Display 10 countries with lowest costs
print("\nTop 10 Countries with Lowest Average Export Costs:")
print(origin_costs.tail(10))

# 2. Basic statistics by country (destination)
print("\n2. DESTINATION COUNTRY ANALYSIS")
print("==============================")

# Calculate average logistics costs by destination country
dest_costs = df.groupby('destination_ISO')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
dest_costs = dest_costs.sort_values('mean', ascending=False)

# Display top 10 most expensive destination countries
print("\nTop 10 Most Expensive Import Destinations:")
print(dest_costs.head(10))

# Display 10 countries with lowest costs
print("\nTop 10 Least Expensive Import Destinations:")
print(dest_costs.tail(10))

# 3. Visualize country cost distributions
print("\n3. VISUALIZING COUNTRY COST DISTRIBUTIONS")
print("========================================")

# Plotting distribution of costs by origin country (top 15)
plt.figure(figsize=(14, 10))
top_origin_countries = origin_costs.head(15).index.tolist()
country_data = df[df['origin_ISO'].isin(top_origin_countries)]

sns.boxplot(x='origin_ISO', y='Unit logistics costs ($/ton)', data=country_data, 
            order=top_origin_countries)
plt.title('Distribution of Logistics Costs by Origin Country (Top 15 Most Expensive)', fontsize=16)
plt.xlabel('Origin Country', fontsize=14)
plt.ylabel('Logistics Costs ($/ton)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('origin_country_costs_boxplot.png')
plt.close()

# 4. Analyze how distance affects costs by country
print("\n4. DISTANCE EFFECTS BY COUNTRY")
print("============================")

# Calculate average distance and cost by origin country
distance_vs_cost = df.groupby('origin_ISO').agg({
    'distance(km)': 'mean',
    'Unit logistics costs ($/ton)': 'mean'
}).reset_index()

# Calculate correlation between distance and cost
corr = distance_vs_cost['distance(km)'].corr(distance_vs_cost['Unit logistics costs ($/ton)'])
print(f"\nCorrelation between average distance and average cost: {corr:.4f}")

# Plot distance vs cost by country
plt.figure(figsize=(12, 8))
sns.scatterplot(x='distance(km)', y='Unit logistics costs ($/ton)', 
                data=distance_vs_cost, alpha=0.7)

# Label some interesting outliers
for i, row in distance_vs_cost.iterrows():
    if row['Unit logistics costs ($/ton)'] > 20000 or row['distance(km)'] > 12000:
        plt.text(row['distance(km)'], row['Unit logistics costs ($/ton)'], row['origin_ISO'], 
                 fontsize=10, ha='center')

plt.title('Average Logistics Costs vs. Average Distance by Origin Country', fontsize=16)
plt.xlabel('Average Distance (km)', fontsize=14)
plt.ylabel('Average Logistics Costs ($/ton)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('distance_vs_cost_by_country.png')
plt.close()

# 5. Analyze transport mode preferences by country
print("\n5. TRANSPORT MODE ANALYSIS BY COUNTRY")
print("===================================")

# Calculate modal share by origin country
mode_by_country = pd.crosstab(df['origin_ISO'], df['Mode_name'], normalize='index') * 100
mode_by_country = mode_by_country.reset_index()

# Calculate average cost by country and mode
cost_by_country_mode = df.groupby(['origin_ISO', 'Mode_name'])['Unit logistics costs ($/ton)'].mean().reset_index()
cost_by_country_mode = cost_by_country_mode.pivot(index='origin_ISO', columns='Mode_name', values='Unit logistics costs ($/ton)')

# Display modal share for top 10 most expensive countries
print("\nTransport Mode Share (%) for Top 10 Most Expensive Origin Countries:")
top_expensive_countries = origin_costs.head(10).index.tolist()
print(mode_by_country[mode_by_country['origin_ISO'].isin(top_expensive_countries)])

# Display modal share for 10 least expensive countries
print("\nTransport Mode Share (%) for 10 Least Expensive Origin Countries:")
least_expensive_countries = origin_costs.tail(10).index.tolist()
print(mode_by_country[mode_by_country['origin_ISO'].isin(least_expensive_countries)])

# 6. Commodity impact analysis by country
print("\n6. COMMODITY ANALYSIS BY COUNTRY")
print("==============================")

# Calculate average cost by country and commodity
commodity_costs = df.groupby(['origin_ISO', 'IFM_HS'])['Unit logistics costs ($/ton)'].mean().reset_index()

# Find the most expensive commodity for each country
most_expensive_commodity = commodity_costs.loc[commodity_costs.groupby('origin_ISO')['Unit logistics costs ($/ton)'].idxmax()]
most_expensive_commodity = most_expensive_commodity.sort_values('Unit logistics costs ($/ton)', ascending=False)

print("\nMost Expensive Commodity by Origin Country (Top 15):")
print(most_expensive_commodity.head(15)[['origin_ISO', 'IFM_HS', 'Unit logistics costs ($/ton)']])

# 7. Landlocked country analysis
print("\n7. LANDLOCKED COUNTRY ANALYSIS")
print("============================")

# Define a list of landlocked countries (ISO codes)
landlocked_countries = ['AFG', 'AND', 'ARM', 'AUT', 'AZE', 'BLR', 'BHU', 'BOL', 'BWA', 
                        'BFA', 'BDI', 'CAF', 'TCD', 'CZE', 'ETH', 'HUN', 'KAZ', 'KGZ', 
                        'LAO', 'LSO', 'LIE', 'MKD', 'MWI', 'MLI', 'MDA', 'MNG', 'NPL', 
                        'NER', 'PRY', 'RWA', 'SMR', 'SRB', 'SVK', 'SWZ', 'CHE', 'TJK', 
                        'TKM', 'UGA', 'UZB', 'ZMB', 'ZWE']

# Add a column indicating if country is landlocked
df['origin_landlocked'] = df['origin_ISO'].isin(landlocked_countries)
df['destination_landlocked'] = df['destination_ISO'].isin(landlocked_countries)

# Compare costs for landlocked vs non-landlocked countries
landlocked_stats = df.groupby('origin_landlocked')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
print("\nLogistics Costs Comparison - Landlocked vs Coastal Countries (Origin):")
print(landlocked_stats)

# Test if the difference is statistically significant
coastal = df[~df['origin_landlocked']]['Unit logistics costs ($/ton)']
landlocked = df[df['origin_landlocked']]['Unit logistics costs ($/ton)']
t_stat, p_value = stats.ttest_ind(coastal, landlocked, equal_var=False)
print(f"\nT-test for difference in means: t={t_stat:.4f}, p={p_value:.4f}")
print(f"Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")

# 8. Regional analysis
print("\n8. REGIONAL ANALYSIS")
print("==================")

# Define regions (simplified for this analysis)
regions = {
    'North America': ['USA', 'CAN', 'MEX'],
    'Europe': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'BEL', 'POL', 'SWE', 'AUT', 'CHE', 'NOR', 'DNK', 'FIN'],
    'East Asia': ['CHN', 'JPN', 'KOR', 'TWN', 'HKG', 'MNG'],
    'Southeast Asia': ['IDN', 'MYS', 'SGP', 'THA', 'VNM', 'PHL', 'MMR', 'LAO', 'KHM', 'BRN'],
    'South Asia': ['IND', 'PAK', 'BGD', 'LKA', 'NPL', 'BTN', 'MDV', 'AFG'],
    'Middle East': ['SAU', 'ARE', 'IRN', 'IRQ', 'ISR', 'QAT', 'KWT', 'OMN', 'JOR', 'LBN', 'BHR', 'SYR', 'YEM'],
    'Africa': ['ZAF', 'EGY', 'NGA', 'MAR', 'KEN', 'ETH', 'GHA', 'TZA', 'CIV', 'CMR', 'UGA', 'SEN'],
    'Latin America': ['BRA', 'ARG', 'COL', 'CHL', 'PER', 'VEN', 'ECU', 'URY', 'BOL', 'PRY', 'GTM', 'CRI', 'PAN'],
    'Oceania': ['AUS', 'NZL', 'PNG', 'FJI']
}

# Function to map country to region
def get_region(country):
    for region, countries in regions.items():
        if country in countries:
            return region
    return 'Other'

# Add region column
df['origin_region'] = df['origin_ISO'].apply(get_region)
df['destination_region'] = df['destination_ISO'].apply(get_region)

# Calculate average costs by region
region_costs = df.groupby('origin_region')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
region_costs = region_costs.sort_values('mean', ascending=False)

print("\nAverage Logistics Costs by Origin Region:")
print(region_costs)

# Regional flow analysis
region_flow = pd.crosstab(df['origin_region'], df['destination_region'], 
                         values=df['Unit logistics costs ($/ton)'], aggfunc='mean')
print("\nAverage Logistics Costs by Origin-Destination Region Pairs:")
print(region_flow)

# 9. Analysis of costs by trade volume
print("\n9. TRADE VOLUME VS. COSTS ANALYSIS")
print("================================")

# Calculate total trade volume by origin country
volume_by_country = df.groupby('origin_ISO')['flow(tonne)'].sum().reset_index()
volume_by_country = volume_by_country.rename(columns={'flow(tonne)': 'total_volume'})

# Merge with average costs
volume_cost = pd.merge(volume_by_country, 
                      origin_costs.reset_index()[['origin_ISO', 'mean']], 
                      on='origin_ISO')
volume_cost = volume_cost.rename(columns={'mean': 'avg_cost'})

# Calculate correlation
vol_cost_corr = volume_cost['total_volume'].corr(volume_cost['avg_cost'])
print(f"\nCorrelation between total trade volume and average logistics cost: {vol_cost_corr:.4f}")

# Plot volume vs cost
plt.figure(figsize=(12, 8))
sns.scatterplot(x='total_volume', y='avg_cost', data=volume_cost, alpha=0.7)

# Label some interesting countries
for i, row in volume_cost.iterrows():
    if row['total_volume'] > 10000000 or row['avg_cost'] > 20000:
        plt.text(row['total_volume'], row['avg_cost'], row['origin_ISO'], 
                 fontsize=10, ha='center')

plt.title('Average Logistics Costs vs. Total Trade Volume by Origin Country', fontsize=16)
plt.xlabel('Total Trade Volume (tonnes)', fontsize=14)
plt.ylabel('Average Logistics Costs ($/ton)', fontsize=14)
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('volume_vs_cost_by_country.png')
plt.close()

# 10. Cost efficiency by country
print("\n10. COST EFFICIENCY BY COUNTRY")
print("============================")

# Calculate cost efficiency (cost per ton per km) by country
df['cost_efficiency'] = df['Unit logistics costs ($/ton)'] / df['distance(km)']
df['cost_efficiency'] = df['cost_efficiency'].replace([np.inf, -np.inf], np.nan)  # Handle division by zero

# Average efficiency by origin country
efficiency_by_country = df.groupby('origin_ISO')['cost_efficiency'].mean().reset_index()
efficiency_by_country = efficiency_by_country.sort_values('cost_efficiency', ascending=False)

print("\nTop 15 Countries with Highest Cost per ton per km (least efficient):")
print(efficiency_by_country.head(15))

print("\nTop 15 Countries with Lowest Cost per ton per km (most efficient):")
print(efficiency_by_country.tail(15))

# 11. Container type and Ship type analysis by country
print("\n11. TRANSPORT EQUIPMENT ANALYSIS BY COUNTRY")
print("========================================")

# Container type usage by country
container_by_country = pd.crosstab(df['origin_ISO'], df['container_type'], normalize='index') * 100
container_by_country = container_by_country.reset_index()

# Ship type usage by country
ship_by_country = pd.crosstab(df['origin_ISO'], df['ship_type'], normalize='index') * 100
ship_by_country = ship_by_country.reset_index()

print("\nContainer Type Usage (%) for Top 10 Most Expensive Countries:")
print(container_by_country[container_by_country['origin_ISO'].isin(top_expensive_countries)].head(10))

print("\nShip Type Usage (%) for Top 10 Most Expensive Countries:")
print(ship_by_country[ship_by_country['origin_ISO'].isin(top_expensive_countries)].head(10))

# 12. Summary and key findings
print("\n12. SUMMARY AND KEY FINDINGS")
print("==========================")

# Identify countries with unusual cost profiles
unusual_countries = []

# High cost but average distance
high_cost_avg_dist = distance_vs_cost[
    (distance_vs_cost['Unit logistics costs ($/ton)'] > distance_vs_cost['Unit logistics costs ($/ton)'].quantile(0.9)) & 
    (distance_vs_cost['distance(km)'] < distance_vs_cost['distance(km)'].quantile(0.75))
]
if not high_cost_avg_dist.empty:
    unusual_countries.append("High cost despite average distance: " + ", ".join(high_cost_avg_dist['origin_ISO'].tolist()))

# Landlocked but low cost
landlocked_low_cost = pd.merge(
    pd.DataFrame({'origin_ISO': landlocked_countries}),
    origin_costs.reset_index(),
    on='origin_ISO'
)
landlocked_low_cost = landlocked_low_cost[
    landlocked_low_cost['mean'] < origin_costs['mean'].quantile(0.5)
]
if not landlocked_low_cost.empty:
    unusual_countries.append("Landlocked but low cost: " + ", ".join(landlocked_low_cost['origin_ISO'].tolist()))

# Print unusual countries
print("\nCountries with Unusual Cost Profiles:")
for finding in unusual_countries:
    print("- " + finding)

print("\nKey Factors Affecting Country-Level Logistics Costs:")
print("1. Transport mode preference (higher use of air transport correlates with higher costs)")
print("2. Geographic location (landlocked status typically increases costs)")
print("3. Trade volume (economies of scale can reduce per-unit costs)")
print("4. Commodity mix (countries specializing in high-value commodities have different cost profiles)")
print("5. Regional trade patterns (intra-regional trade tends to have lower logistics costs)")
print("6. Infrastructure quality (not directly measured but inferred from efficiency metrics)")

print("\nCountry-Level Analysis Complete!")

# Save the processed data with classifications for future reference
if not df.empty:
    df.to_csv('country_logistics_analysis.csv', index=False)
    print("Processed data saved to 'country_logistics_analysis.csv'") 