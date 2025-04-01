#!/usr/bin/env python
# coding: utf-8

"""
African Exports Analysis
=======================
This script analyzes transport costs and logistics patterns specifically for 
African countries' exports, combining predictive modeling and anomaly detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from category_encoders import TargetEncoder
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Create output directories
output_dir = "african_exports"
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

print("Starting African Exports Analysis...")

# Load the dataset
try:
    df = pd.read_csv("imputed_full_matrix_at_centroid.csv")
except FileNotFoundError:
    df = pd.read_csv("../../../data/imputed_full_matrix_at_centroid.csv")

print(f"Full dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Filter for exports from African countries only
df_africa = df[df['origin_ISO'].isin(african_countries)]
print(f"African exports dataset: {df_africa.shape[0]} rows ({df_africa.shape[0]/df.shape[0]:.1%} of total data)")

#############################
# 1. OVERVIEW OF AFRICAN EXPORTS
#############################
print("\n\n===== OVERVIEW OF AFRICAN EXPORTS =====")

# Top exporting African countries by volume
top_exporters_volume = df_africa.groupby('origin_ISO')['flow(tonne)'].sum().sort_values(ascending=False)
print("\nTop 10 African Exporters by Volume (tonnes):")
print(top_exporters_volume.head(10))

# Top exporting African countries by trade routes
top_exporters_routes = df_africa.groupby('origin_ISO').size().sort_values(ascending=False)
print("\nTop 10 African Exporters by Number of Trade Routes:")
print(top_exporters_routes.head(10))

# Export costs by country
export_costs = df_africa.groupby('origin_ISO')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'min', 'max', 'count']).sort_values('mean', ascending=False)
print("\nAfrican Countries by Average Export Costs ($/ton):")
print(export_costs.head(10))
print("\nAfrican Countries with Lowest Export Costs ($/ton):")
print(export_costs.tail(10))

# Visualization - Export costs by country
plt.figure(figsize=(14, 8))
export_costs_plot = export_costs.sort_values('mean')
plt.bar(export_costs_plot.index, export_costs_plot['mean'], yerr=export_costs_plot['mean'].std())
plt.title('Average Export Costs by African Country', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Average Unit Logistics Costs ($/ton)', fontsize=14)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/african_export_costs.png")
plt.close()

# Transport mode distribution for African exports
mode_dist = pd.crosstab(df_africa['origin_ISO'], df_africa['Mode_name'], normalize='index') * 100
print("\nTransport Mode Distribution by Country (%):")
print(mode_dist.head(10))

# Overall mode distribution
overall_mode_dist = df_africa['Mode_name'].value_counts(normalize=True) * 100
print("\nOverall Transport Mode Distribution for African Exports:")
print(overall_mode_dist)

# Visualization - Mode distribution
plt.figure(figsize=(10, 6))
overall_mode_dist.plot(kind='bar', color='darkblue')
plt.title('Transport Mode Distribution for African Exports', fontsize=16)
plt.xlabel('Transport Mode', fontsize=14)
plt.ylabel('Percentage of Exports (%)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/african_mode_distribution.png")
plt.close()

# Top export destinations
top_destinations = df_africa.groupby('destination_ISO').agg({
    'flow(tonne)': 'sum',
    'Unit logistics costs ($/ton)': 'mean'
}).sort_values('flow(tonne)', ascending=False)
print("\nTop 10 Destinations for African Exports:")
print(top_destinations.head(10))

# Destination regions
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

df_africa['destination_region'] = df_africa['destination_ISO'].apply(get_region)
region_dist = df_africa.groupby('destination_region').agg({
    'flow(tonne)': 'sum',
    'Unit logistics costs ($/ton)': 'mean'
}).sort_values('flow(tonne)', ascending=False)

print("\nAfrican Exports by Destination Region:")
print(region_dist)

# Visualization - Exports by destination region
plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots(figsize=(12, 6))

x = region_dist.index
y1 = region_dist['flow(tonne)'] / 1_000_000  # Convert to millions of tonnes

ax1.bar(x, y1, color='blue', alpha=0.7)
ax1.set_xlabel('Destination Region', fontsize=14)
ax1.set_ylabel('Total Volume (Million Tonnes)', color='blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(x, rotation=45)

ax2 = ax1.twinx()
y2 = region_dist['Unit logistics costs ($/ton)']
ax2.plot(x, y2, 'ro-', linewidth=2, markersize=8)
ax2.set_ylabel('Average Logistics Cost ($/ton)', color='red', fontsize=14)
ax2.tick_params(axis='y', labelcolor='red')

plt.title('African Exports by Destination Region: Volume and Cost', fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/african_exports_by_region.png")
plt.close()

# Top commodities
top_commodities = df_africa.groupby('IFM_HS').agg({
    'flow(tonne)': 'sum',
    'Unit logistics costs ($/ton)': 'mean'
}).sort_values('flow(tonne)', ascending=False)
print("\nTop 10 Export Commodities from Africa:")
print(top_commodities.head(10))

#############################
# 2. TRANSPORT COST PREDICTION MODEL FOR AFRICAN EXPORTS
#############################
print("\n\n===== TRANSPORT COST PREDICTION MODEL FOR AFRICAN EXPORTS =====")

# Remove extreme outliers for modeling purposes
q1 = df_africa['Unit logistics costs ($/ton)'].quantile(0.01)
q3 = df_africa['Unit logistics costs ($/ton)'].quantile(0.99)
df_africa_model = df_africa[(df_africa['Unit logistics costs ($/ton)'] >= q1) & 
                           (df_africa['Unit logistics costs ($/ton)'] <= q3)]

print(f"Removed {df_africa.shape[0] - df_africa_model.shape[0]} outliers for modeling")

# Define features for modeling
numeric_features = ['distance(km)', 'flow(tonne)']
categorical_features = ['Mode_name', 'destination_ISO', 'IFM_HS', 'ship_type', 'container_type']

# Prepare features and target variable
X = df_africa_model[numeric_features + categorical_features].copy()
y = df_africa_model['Unit logistics costs ($/ton)']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('target_encoder', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(random_state=42))
])

# Train the model
print("\nTraining XGBoost model for African export costs...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model performance:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  R²: {r2:.4f}")

# Feature importance analysis
print("\nAnalyzing feature importance...")

# Extract feature importances
xgb_model = model.named_steps['model']
importances = xgb_model.feature_importances_

# Get feature names (approximate because of preprocessing transformations)
feature_names = numeric_features + categorical_features

# Sort importances
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title('Feature Importance for African Export Costs', fontsize=16)
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png")
plt.close()

# Print top 10 features
print("\nTop 10 most important features for African export costs:")
for i in range(min(10, len(indices))):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# SHAP analysis
print("\nPerforming SHAP analysis...")
try:
    # Sample a subset of the test data for SHAP analysis
    X_sample = shap.sample(X_test, 1000, random_state=42)
    
    # Create explainer
    explainer = shap.Explainer(model.named_steps['model'])
    
    # Calculate SHAP values
    shap_values = explainer(preprocessor.transform(X_sample))
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance for African Export Costs', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance.png")
    plt.close()
    
    # Detailed SHAP summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, show=False)
    plt.title('SHAP Summary Plot for African Export Costs', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png")
    plt.close()
    
except Exception as e:
    print(f"SHAP analysis error: {e}")
    print("Skipping detailed SHAP analysis.")

#############################
# 3. ANOMALY ANALYSIS IN AFRICAN EXPORTS
#############################
print("\n\n===== ANOMALY ANALYSIS IN AFRICAN EXPORTS =====")

# 3.1 Analyze high-cost countries
print("\n3.1 High-Cost African Exporters")
high_cost_countries = export_costs.head(5).index.tolist()
print(f"Analyzing high-cost African exporters: {', '.join(high_cost_countries)}")

# Filter data for these countries
high_cost_data = df_africa[df_africa['origin_ISO'].isin(high_cost_countries)]

# Transport mode breakdown for high-cost countries
print("\nTransport mode distribution for high-cost countries (%):")
high_cost_modes = pd.crosstab(high_cost_data['origin_ISO'], high_cost_data['Mode_name'], normalize='index') * 100
print(high_cost_modes.round(2))

# Commodity analysis for high-cost countries
print("\nTop expensive commodities by high-cost country:")
for country in high_cost_countries:
    country_data = high_cost_data[high_cost_data['origin_ISO'] == country]
    if len(country_data) > 0:
        top_commodities = country_data.groupby('IFM_HS').agg({
            'Unit logistics costs ($/ton)': 'mean',
            'flow(tonne)': 'sum'
        }).sort_values('Unit logistics costs ($/ton)', ascending=False)
        print(f"\n{country} - Top 5 Most Expensive Commodities:")
        print(top_commodities.head(5)[['Unit logistics costs ($/ton)', 'flow(tonne)']])

# Visualization - Cost by transport mode for high-cost countries
plt.figure(figsize=(14, 8))
sns.boxplot(x='origin_ISO', y='Unit logistics costs ($/ton)', hue='Mode_name', data=high_cost_data)
plt.title('Transport Costs by Mode for High-Cost African Exporters', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.yscale('log')  # Log scale to handle extreme values
plt.legend(title='Transport Mode')
plt.tight_layout()
plt.savefig(f"{output_dir}/high_cost_countries_by_mode.png")
plt.close()

# 3.2 Analyze landlocked vs. coastal countries
print("\n3.2 Landlocked vs. Coastal African Countries")

# Define African landlocked countries
african_landlocked = ['BDI', 'BFA', 'CAF', 'ETH', 'LSO', 'MWI', 'MLI', 'NER', 'RWA', 'SSD', 'SWZ', 'UGA', 'ZMB', 'ZWE', 'BWA', 'TCD']
df_africa['is_landlocked'] = df_africa['origin_ISO'].isin(african_landlocked)

# Compare costs
landlocked_stats = df_africa.groupby('is_landlocked')['Unit logistics costs ($/ton)'].agg(['mean', 'median', 'std', 'count'])
print("\nLogistics Costs: Landlocked vs. Coastal African Countries:")
print(landlocked_stats)

# Compare transport modes
landlocked_modes = pd.crosstab(df_africa['is_landlocked'], df_africa['Mode_name'], normalize='index') * 100
print("\nTransport Mode Distribution - Landlocked vs. Coastal (%):")
print(landlocked_modes.round(2))

# Visualization - Costs by landlocked status
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_landlocked', y='Unit logistics costs ($/ton)', data=df_africa)
plt.title('Export Costs: Landlocked vs. Coastal African Countries', fontsize=16)
plt.xlabel('Landlocked Status', fontsize=14)
plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.xticks([0, 1], ['Coastal', 'Landlocked'])
plt.tight_layout()
plt.savefig(f"{output_dir}/landlocked_vs_coastal_costs.png")
plt.close()

# 3.3 Analyze RoRo in air transport for African countries
print("\n3.3 RoRo in Air Transport for African Countries")

# Filter for RoRo shipments via air from African countries
roro_air_africa = df_africa[(df_africa['Mode_name'] == 'Air') & (df_africa['ship_type'] == 'RoRo')]
print(f"Number of RoRo air shipments from African countries: {len(roro_air_africa)}")

if len(roro_air_africa) > 0:
    # Basic statistics
    print("\nBasic statistics for African RoRo air shipments:")
    print(f"Average cost: ${roro_air_africa['Unit logistics costs ($/ton)'].mean():.2f} per ton")
    print(f"Average distance: {roro_air_africa['distance(km)'].mean():.2f} km")
    
    # Top origin countries
    print("\nAfrican countries with RoRo air shipments:")
    print(roro_air_africa['origin_ISO'].value_counts())
    
    # Commodity analysis
    print("\nTop commodities for African RoRo air shipments:")
    print(roro_air_africa['IFM_HS'].value_counts().head(10))
    
    # Save for further analysis
    roro_air_africa.to_csv(f"{output_dir}/african_roro_air_shipments.csv", index=False)

# 3.4 Regional exports analysis
print("\n3.4 Regional Exports Analysis for African Countries")

# Add origin region (North Africa, East Africa, etc.)
region_mapping = {
    # North Africa
    'DZA': 'North Africa', 'EGY': 'North Africa', 'LBY': 'North Africa', 
    'MAR': 'North Africa', 'TUN': 'North Africa', 'ESH': 'North Africa',
    
    # West Africa
    'BEN': 'West Africa', 'BFA': 'West Africa', 'CPV': 'West Africa', 
    'CIV': 'West Africa', 'GMB': 'West Africa', 'GHA': 'West Africa', 
    'GIN': 'West Africa', 'GNB': 'West Africa', 'LBR': 'West Africa', 
    'MLI': 'West Africa', 'MRT': 'West Africa', 'NER': 'West Africa', 
    'NGA': 'West Africa', 'SEN': 'West Africa', 'SLE': 'West Africa', 
    'TGO': 'West Africa',
    
    # Central Africa
    'AGO': 'Central Africa', 'CMR': 'Central Africa', 'CAF': 'Central Africa', 
    'TCD': 'Central Africa', 'COD': 'Central Africa', 'GNQ': 'Central Africa', 
    'GAB': 'Central Africa', 'STP': 'Central Africa',
    
    # East Africa
    'BDI': 'East Africa', 'COM': 'East Africa', 'DJI': 'East Africa', 
    'ERI': 'East Africa', 'ETH': 'East Africa', 'KEN': 'East Africa', 
    'MDG': 'East Africa', 'MWI': 'East Africa', 'MUS': 'East Africa', 
    'MYT': 'East Africa', 'MOZ': 'East Africa', 'REU': 'East Africa', 
    'RWA': 'East Africa', 'SYC': 'East Africa', 'SOM': 'East Africa', 
    'SSD': 'East Africa', 'SDN': 'East Africa', 'TZA': 'East Africa', 
    'UGA': 'East Africa',
    
    # Southern Africa
    'BWA': 'Southern Africa', 'LSO': 'Southern Africa', 'NAM': 'Southern Africa', 
    'ZAF': 'Southern Africa', 'SWZ': 'Southern Africa', 'ZMB': 'Southern Africa', 
    'ZWE': 'Southern Africa'
}

df_africa['origin_region'] = df_africa['origin_ISO'].map(region_mapping)

# Analyze by African region
region_stats = df_africa.groupby('origin_region').agg({
    'flow(tonne)': 'sum',
    'Unit logistics costs ($/ton)': 'mean'
}).sort_values('flow(tonne)', ascending=False)

print("\nExport Statistics by African Region:")
print(region_stats)

# Transport mode by African region
region_modes = pd.crosstab(df_africa['origin_region'], df_africa['Mode_name'], normalize='index') * 100
print("\nTransport Mode Distribution by African Region (%):")
print(region_modes.round(2))

# Visualization - Export volume by African region
plt.figure(figsize=(12, 6))
region_stats['flow(tonne)'].plot(kind='bar', color='darkblue')
plt.title('Export Volume by African Region', fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Total Volume (tonnes)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/export_volume_by_region.png")
plt.close()

# Visualization - Export costs by African region
plt.figure(figsize=(12, 6))
region_stats['Unit logistics costs ($/ton)'].plot(kind='bar', color='darkred')
plt.title('Average Export Costs by African Region', fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Average Unit Logistics Costs ($/ton)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/export_costs_by_region.png")
plt.close()

#############################
# 4. COST REDUCTION OPPORTUNITIES
#############################
print("\n\n===== COST REDUCTION OPPORTUNITIES FOR AFRICAN EXPORTS =====")

# 4.1 Mode optimization opportunities
print("\n4.1 Mode Optimization Opportunities")

# Analyze trade routes where mode shift could reduce costs
# For each country-destination pair, compare costs across modes
country_dest_mode_costs = df_africa.groupby(['origin_ISO', 'destination_ISO', 'Mode_name'])['Unit logistics costs ($/ton)'].agg(['mean', 'count']).reset_index()

# Find pairs where multiple modes exist
mode_counts = country_dest_mode_costs.groupby(['origin_ISO', 'destination_ISO']).size()
multi_mode_pairs = mode_counts[mode_counts > 1].index

# For these pairs, find where cost difference between modes is significant
if len(multi_mode_pairs) > 0:
    print("\nTrade routes with significant cost differences between transport modes:")
    
    opportunity_rows = []
    
    for origin, dest in multi_mode_pairs:
        pair_data = country_dest_mode_costs[(country_dest_mode_costs['origin_ISO'] == origin) & 
                                          (country_dest_mode_costs['destination_ISO'] == dest)]
        
        # Find cheapest and most expensive modes
        cheapest_mode = pair_data.loc[pair_data['mean'].idxmin()]
        most_exp_mode = pair_data.loc[pair_data['mean'].idxmax()]
        
        # Calculate potential savings
        savings_pct = (most_exp_mode['mean'] - cheapest_mode['mean']) / most_exp_mode['mean'] * 100
        
        # Only include if savings are substantial (>30%)
        if savings_pct > 30 and most_exp_mode['count'] >= 10:  # Ensure sufficient data points
            opportunity_rows.append({
                'Origin': origin,
                'Destination': dest,
                'Current Mode': most_exp_mode['Mode_name'],
                'Suggested Mode': cheapest_mode['Mode_name'],
                'Current Cost': most_exp_mode['mean'],
                'Optimal Cost': cheapest_mode['mean'],
                'Savings (%)': savings_pct,
                'Number of Shipments': most_exp_mode['count']
            })
    
    if opportunity_rows:
        opportunities_df = pd.DataFrame(opportunity_rows)
        opportunities_df = opportunities_df.sort_values('Savings (%)', ascending=False)
        print(opportunities_df.head(10))
        
        # Save opportunities for further analysis
        opportunities_df.to_csv(f"{output_dir}/mode_optimization_opportunities.csv", index=False)
        
        # Visualization - Top mode optimization opportunities
        plt.figure(figsize=(14, 8))
        top_opps = opportunities_df.head(10)
        bars = plt.bar(range(len(top_opps)), top_opps['Savings (%)'])
        
        # Add route labels
        plt.xticks(range(len(top_opps)), [f"{o} → {d}" for o, d in zip(top_opps['Origin'], top_opps['Destination'])], rotation=45)
        
        # Add mode information
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, 5, 
                     f"{top_opps.iloc[i]['Current Mode']} → {top_opps.iloc[i]['Suggested Mode']}", 
                     ha='center', rotation=90, color='white', fontweight='bold')
        
        plt.title('Top 10 Mode Optimization Opportunities for African Exports', fontsize=16)
        plt.ylabel('Potential Cost Savings (%)', fontsize=14)
        plt.xlabel('Trade Route', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mode_optimization_opportunities.png")
        plt.close()

# 4.2 Corridor development opportunities
print("\n4.2 Corridor Development Opportunities")

# For landlocked countries, identify high-traffic, high-cost routes that could benefit from corridor development
if len(african_landlocked) > 0:
    landlocked_exports = df_africa[df_africa['origin_ISO'].isin(african_landlocked)]
    
    # Identify major trade corridors
    corridors = landlocked_exports.groupby(['origin_ISO', 'destination_ISO']).agg({
        'flow(tonne)': 'sum',
        'Unit logistics costs ($/ton)': 'mean',
        'distance(km)': 'mean'
    }).reset_index()
    
    # Calculate cost per ton-km
    corridors['cost_per_ton_km'] = corridors['Unit logistics costs ($/ton)'] / corridors['distance(km)']
    
    # Sort by volume and cost
    high_volume_corridors = corridors[corridors['flow(tonne)'] > corridors['flow(tonne)'].quantile(0.75)]
    high_volume_corridors = high_volume_corridors.sort_values('cost_per_ton_km', ascending=False)
    
    print("\nHigh-traffic corridors from landlocked countries with high costs per ton-km:")
    print(high_volume_corridors.head(10))
    
    # Save for further analysis
    high_volume_corridors.to_csv(f"{output_dir}/corridor_development_opportunities.csv", index=False)
    
    # Visualization - Corridor development opportunities
    plt.figure(figsize=(14, 8))
    plt.scatter(high_volume_corridors['flow(tonne)'], 
               high_volume_corridors['cost_per_ton_km'], 
               s=high_volume_corridors['distance(km)']/50, 
               alpha=0.7)
    
    # Add labels for top corridors
    for i, row in high_volume_corridors.head(5).iterrows():
        plt.annotate(f"{row['origin_ISO']} → {row['destination_ISO']}", 
                    (row['flow(tonne)'], row['cost_per_ton_km']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Corridor Development Opportunities for Landlocked African Countries', fontsize=16)
    plt.xlabel('Total Flow Volume (tonnes)', fontsize=14)
    plt.ylabel('Cost per ton-km ($/ton-km)', fontsize=14)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/corridor_development_opportunities.png")
    plt.close()

# 4.3 Port efficiency analysis
print("\n4.3 Port Efficiency Analysis for African Exports")

# Analyze which ports/coastal countries provide most cost-effective export routes for African goods
# Focus on exports to same destination regions but via different ports

# For non-African destinations, analyze which coastal countries offer cheapest routes
non_african_dests = df_africa[~df_africa['destination_ISO'].isin(african_countries)]
coastal_routes = non_african_dests[~non_african_dests['origin_ISO'].isin(african_landlocked)]

# Group by destination region and origin (coastal African country)
port_efficiency = coastal_routes.groupby(['destination_region', 'origin_ISO']).agg({
    'Unit logistics costs ($/ton)': 'mean',
    'flow(tonne)': 'sum'
}).reset_index()

# For each destination region, rank African coastal countries by cost-effectiveness
region_rankings = {}
for region in port_efficiency['destination_region'].unique():
    region_data = port_efficiency[port_efficiency['destination_region'] == region]
    if len(region_data) > 1:  # Only include if multiple coastal countries serve this region
        region_rankings[region] = region_data.sort_values('Unit logistics costs ($/ton)')[['origin_ISO', 'Unit logistics costs ($/ton)', 'flow(tonne)']]

print("\nMost cost-effective African ports/coastal countries by destination region:")
for region, ranking in region_rankings.items():
    if not region_rankings[region].empty:
        print(f"\nDestination: {region}")
        print(region_rankings[region].head(3))  # Top 3 most cost-effective

# Save port efficiency data
port_efficiency.to_csv(f"{output_dir}/port_efficiency.csv", index=False)

#############################
# 5. CONCLUSIONS AND RECOMMENDATIONS
#############################
print("\n\n===== CONCLUSIONS AND RECOMMENDATIONS FOR AFRICAN EXPORTS =====")

print("""
Key Findings:
1. [To be populated based on actual analysis results]

Recommendations for Cost Reduction:
1. [To be populated based on actual analysis results]

Policy Implications:
1. [To be populated based on actual analysis results]
""")

print(f"\nAnalysis results saved in the '{output_dir}' directory.") 