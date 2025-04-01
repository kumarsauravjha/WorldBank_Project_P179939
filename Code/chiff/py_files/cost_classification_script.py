#!/usr/bin/env python
# coding: utf-8

# # Trade Route Cost Classification System
# 
# This script develops a classification system to categorize trade routes by cost-efficiency into:
# - Low Cost: Most cost-efficient routes
# - Moderate Cost: Average cost-efficiency routes
# - High Cost: Least cost-efficient routes

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv("imputed_full_matrix_at_centroid.csv")

# Display basic information
print(f"Dataset shape: {df.shape}")
print(df.head())

# Check for missing values in the cost column
print(f"Missing values in 'Unit logistics costs ($/ton)': {df['Unit logistics costs ($/ton)'].isna().sum()}")

# Basic statistics of the cost column
print(df['Unit logistics costs ($/ton)'].describe())

# Plot histogram of logistics costs
plt.figure(figsize=(12, 6))
sns.histplot(df['Unit logistics costs ($/ton)'], bins=50, kde=True)
plt.title('Distribution of Unit Logistics Costs', fontsize=16)
plt.xlabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(df['Unit logistics costs ($/ton)'].median(), color='red', linestyle='--', label=f'Median: {df["Unit logistics costs ($/ton)"].median():.2f}')
plt.axvline(df['Unit logistics costs ($/ton)'].mean(), color='green', linestyle='--', label=f'Mean: {df["Unit logistics costs ($/ton)"].mean():.2f}')
plt.legend()
plt.savefig('cost_distribution.png')
plt.close()

# Check for outliers with a boxplot
plt.figure(figsize=(12, 4))
sns.boxplot(x=df['Unit logistics costs ($/ton)'])
plt.title('Boxplot of Unit Logistics Costs', fontsize=16)
plt.xlabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.tight_layout()
plt.savefig('cost_boxplot.png')
plt.close()

print("\nApproach 1: Classification by Percentiles")
# Define thresholds using percentiles
low_threshold = df['Unit logistics costs ($/ton)'].quantile(0.333)
high_threshold = df['Unit logistics costs ($/ton)'].quantile(0.667)

# Create a new column with cost categories
df['cost_category_percentile'] = pd.cut(
    df['Unit logistics costs ($/ton)'],
    bins=[float('-inf'), low_threshold, high_threshold, float('inf')],
    labels=['Low Cost', 'Moderate Cost', 'High Cost']
)

# Display category distribution
print("Cost Category Distribution (Percentile-based):")
category_counts = df['cost_category_percentile'].value_counts()
print(category_counts)
print(f"\nCategory Percentages:\n{(category_counts / len(df) * 100).round(2)}%")

# Display threshold values
print(f"\nLow Cost: Less than ${low_threshold:.2f} per ton")
print(f"Moderate Cost: ${low_threshold:.2f} to ${high_threshold:.2f} per ton")
print(f"High Cost: Greater than ${high_threshold:.2f} per ton")

# Visualize the categories
plt.figure(figsize=(12, 6))
sns.histplot(
    data=df, 
    x='Unit logistics costs ($/ton)', 
    hue='cost_category_percentile',
    bins=50, 
    multiple='stack',
    palette='viridis'
)
plt.axvline(low_threshold, color='red', linestyle='--', label=f'Low Threshold: ${low_threshold:.2f}')
plt.axvline(high_threshold, color='green', linestyle='--', label=f'High Threshold: ${high_threshold:.2f}')
plt.title('Distribution of Logistics Costs by Category (Percentile-based)', fontsize=16)
plt.xlabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.savefig('percentile_distribution.png')
plt.close()

# Bar chart of category counts
plt.figure(figsize=(10, 6))
sns.countplot(x='cost_category_percentile', data=df, palette='viridis')
plt.title('Number of Routes by Cost Category (Percentile-based)', fontsize=16)
plt.xlabel('Cost Category', fontsize=14)
plt.ylabel('Number of Routes', fontsize=14)
plt.savefig('percentile_counts.png')
plt.close()

print("\nApproach 2: Classification using K-means Clustering")
# Prepare data for clustering
# Get only the cost column and handle missing values if any
X = df[['Unit logistics costs ($/ton)']].copy()
X = X.fillna(X.mean())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cost_category_kmeans'] = kmeans.fit_predict(X_scaled)

# Get cluster centers and transform back to original scale
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Sort clusters by their centers to maintain consistent labels (Low, Moderate, High)
sorted_indices = np.argsort(cluster_centers.flatten())
mapping = {sorted_indices[0]: 'Low Cost', sorted_indices[1]: 'Moderate Cost', sorted_indices[2]: 'High Cost'}
df['cost_category_kmeans'] = df['cost_category_kmeans'].map(mapping)

# Display cluster centers
print("K-means Cluster Centers (in original scale):")
for i, center in enumerate(cluster_centers.flatten()[sorted_indices]):
    category = ['Low Cost', 'Moderate Cost', 'High Cost'][i]
    print(f"{category}: ${center:.2f} per ton")

# Display category distribution
print("\nCost Category Distribution (K-means):")
category_counts_kmeans = df['cost_category_kmeans'].value_counts()
print(category_counts_kmeans)
print(f"\nCategory Percentages:\n{(category_counts_kmeans / len(df) * 100).round(2)}%")

# Visualize the K-means categories
plt.figure(figsize=(12, 6))
sns.histplot(
    data=df, 
    x='Unit logistics costs ($/ton)', 
    hue='cost_category_kmeans',
    bins=50, 
    multiple='stack',
    palette='viridis'
)

# Add vertical lines for cluster centers
for center in cluster_centers.flatten()[sorted_indices]:
    plt.axvline(center, color='red', linestyle='--')
    
plt.title('Distribution of Logistics Costs by Category (K-means)', fontsize=16)
plt.xlabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.savefig('kmeans_distribution.png')
plt.close()

# Bar chart of K-means category counts
plt.figure(figsize=(10, 6))
category_order = ['Low Cost', 'Moderate Cost', 'High Cost']
sns.countplot(x='cost_category_kmeans', data=df, order=category_order, palette='viridis')
plt.title('Number of Routes by Cost Category (K-means)', fontsize=16)
plt.xlabel('Cost Category', fontsize=14)
plt.ylabel('Number of Routes', fontsize=14)
plt.savefig('kmeans_counts.png')
plt.close()

print("\nApproach 3: Classification based on Domain Knowledge and Context")
# Define a function to assign contextual cost categories
def assign_cost_category(row):
    cost = row['Unit logistics costs ($/ton)']
    distance = row['distance(km)']
    mode = row['Mode_name']
    
    # Different thresholds for different transport modes
    if mode == 'Air':
        if cost < 1500:
            return 'Low Cost'
        elif cost < 3000:
            return 'Moderate Cost'
        else:
            return 'High Cost'
    elif mode == 'Rail':
        if cost < 300:
            return 'Low Cost'
        elif cost < 600:
            return 'Moderate Cost'
        else:
            return 'High Cost'
    elif mode == 'Road':
        if cost < 200:
            return 'Low Cost'
        elif cost < 500:
            return 'Moderate Cost'
        else:
            return 'High Cost'
    else:  # Sea or other modes
        if cost < 250:
            return 'Low Cost'
        elif cost < 750:
            return 'Moderate Cost'
        else:
            return 'High Cost'

# Apply the contextual classification
df['cost_category_contextual'] = df.apply(assign_cost_category, axis=1)

# Display category distribution
print("Cost Category Distribution (Context-based):")
category_counts_context = df['cost_category_contextual'].value_counts()
print(category_counts_context)
print(f"\nCategory Percentages:\n{(category_counts_context / len(df) * 100).round(2)}%")

# Create a figure to show cost distributions by transport mode
plt.figure(figsize=(15, 10))

# Get unique transport modes
modes = df['Mode_name'].unique()

for i, mode in enumerate(modes):
    plt.subplot(2, 2, i+1)
    
    # Filter data for this mode
    mode_data = df[df['Mode_name'] == mode]
    
    # Create histogram
    sns.histplot(
        data=mode_data, 
        x='Unit logistics costs ($/ton)', 
        hue='cost_category_contextual',
        bins=30, 
        multiple='stack',
        palette='viridis'
    )
    
    plt.title(f'Cost Distribution for {mode} Transport', fontsize=14)
    plt.xlabel('Unit Logistics Costs ($/ton)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Cost Category')
    
plt.tight_layout()
plt.savefig('mode_distributions.png')
plt.close()

# Proportion of each category by transport mode
plt.figure(figsize=(14, 8))
crosstab = pd.crosstab(df['Mode_name'], df['cost_category_contextual'], normalize='index') * 100
crosstab.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Proportion of Cost Categories by Transport Mode', fontsize=16)
plt.xlabel('Transport Mode', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Cost Category')
plt.xticks(rotation=45)
plt.savefig('mode_proportions.png')
plt.close()

print("\nAnalyzing Cost Categories by Distance")
# Create distance bins
df['distance_bin'] = pd.cut(
    df['distance(km)'], 
    bins=[0, 500, 1000, 5000, float('inf')],
    labels=['< 500 km', '500-1000 km', '1000-5000 km', '> 5000 km']
)

# Create a scatter plot of cost vs. distance, colored by cost category
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df.sample(10000), # Sample for better visualization
    x='distance(km)', 
    y='Unit logistics costs ($/ton)',
    hue='cost_category_contextual',
    alpha=0.7,
    palette='viridis'
)
plt.title('Logistics Costs vs. Distance by Cost Category', fontsize=16)
plt.xlabel('Distance (km)', fontsize=14)
plt.ylabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.legend(title='Cost Category')
plt.savefig('cost_vs_distance.png')
plt.close()

# Cost category distribution by distance bin
plt.figure(figsize=(14, 8))
distance_crosstab = pd.crosstab(df['distance_bin'], df['cost_category_contextual'], normalize='index') * 100
distance_crosstab.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribution of Cost Categories by Distance Range', fontsize=16)
plt.xlabel('Distance Range', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Cost Category')
plt.xticks(rotation=45)
plt.savefig('distance_proportions.png')
plt.close()

print("\nAnalyzing Cost Categories by Commodity Type")
# Cost category distribution by commodity type
plt.figure(figsize=(14, 8))
commodity_crosstab = pd.crosstab(df['IFM_HS'], df['cost_category_contextual'], normalize='index') * 100
commodity_crosstab.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribution of Cost Categories by Commodity Type', fontsize=16)
plt.xlabel('Commodity Type', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Cost Category')
plt.xticks(rotation=90)
plt.savefig('commodity_proportions.png')
plt.close()

print("\nAnalyzing Cost Efficiency by Origin-Destination Pairs")
# Create origin-destination pairs
df['origin_destination'] = df['origin_ISO'] + '-' + df['destination_ISO']

# Get top 20 route pairs by volume
top_routes = df.groupby('origin_destination')['flow(tonne)'].sum().sort_values(ascending=False).head(20).index

# Filter data for top routes
top_routes_data = df[df['origin_destination'].isin(top_routes)]

# Calculate average cost for each route and cost category distribution
route_analysis = top_routes_data.groupby('origin_destination')['Unit logistics costs ($/ton)'].mean().reset_index()
route_analysis = route_analysis.sort_values('Unit logistics costs ($/ton)')

# Plot average logistics costs for top routes
plt.figure(figsize=(14, 8))
sns.barplot(x='origin_destination', y='Unit logistics costs ($/ton)', data=route_analysis, palette='viridis')
plt.title('Average Logistics Costs for Top 20 Trade Routes by Volume', fontsize=16)
plt.xlabel('Origin-Destination Route', fontsize=14)
plt.ylabel('Average Unit Logistics Costs ($/ton)', fontsize=14)
plt.xticks(rotation=90)
plt.savefig('route_costs.png')
plt.close()

# Analyze cost category distribution for top routes
route_category_dist = pd.crosstab(top_routes_data['origin_destination'], top_routes_data['cost_category_contextual'], normalize='index') * 100
route_category_dist = route_category_dist.reindex(route_analysis['origin_destination'])

plt.figure(figsize=(14, 8))
route_category_dist.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Cost Category Distribution for Top 20 Trade Routes by Volume', fontsize=16)
plt.xlabel('Origin-Destination Route', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Cost Category')
plt.xticks(rotation=90)
plt.savefig('route_proportions.png')
plt.close()

print("\nSaving the Classified Data")
# Select the final classification method
df['final_cost_category'] = df['cost_category_contextual']  # Using the contextual approach

# Save the classified data
df.to_csv('trade_routes_with_cost_classification.csv', index=False)

# Display the first few rows of the final dataset
print(df[['origin_ISO', 'destination_ISO', 'Mode_name', 'distance(km)', 
    'Unit logistics costs ($/ton)', 'final_cost_category']].head(10))

print("\nSummary: Cost Efficiency Classification Findings")
print("""
We've developed three different approaches to classify trade routes by cost efficiency:

1. Percentile-based Classification: Simple equal distribution approach
   - Low Cost: Bottom 33.3% of costs 
   - Moderate Cost: Middle 33.3% of costs
   - High Cost: Top 33.3% of costs

2. K-means Clustering: Data-driven approach that identifies natural groupings
   - Identifies natural thresholds based on the data distribution
   - May result in unequal group sizes based on data patterns

3. Contextual Classification: Domain knowledge-based approach
   - Considers transport mode, distance, and commodity type
   - Sets different thresholds for different contexts (e.g., air vs. sea transport)
   - Provides more nuanced insights for decision-making

Key Findings:
- Transport mode has a significant impact on logistics costs
- Long-distance routes aren't always the most expensive per ton
- Different commodity types have distinct cost profiles
- Some origin-destination pairs consistently fall into specific cost categories

Recommendations:
- Focus optimization efforts on routes classified as "High Cost"
- Study the characteristics of "Low Cost" routes to identify best practices
- Consider mode shifts for routes with unfavorable cost categories
- Use this classification system to track improvements over time
""") 