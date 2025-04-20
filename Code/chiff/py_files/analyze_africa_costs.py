import pandas as pd
import numpy as np

# Path to the generated dataset
file_path = "/Users/abishekchiffon/Documents/Technical/Masters/sem 4/captsone/data/africa_avg_cost_by_commodity.csv"

print("Reading the dataset...")
df = pd.read_csv(file_path)

print("\n--- Dataset Overview ---")
print(f"Total records: {df.shape[0]}")
print(f"Columns: {', '.join(df.columns)}")

print("\n--- Origin Countries (African) ---")
origin_counts = df['origin_ISO'].value_counts()
print(f"Number of origin countries: {len(origin_counts)}")
print("Top 10 origin countries by record count:")
print(origin_counts.head(10))

print("\n--- Destination Countries ---")
dest_counts = df['destination_ISO'].value_counts()
print(f"Number of destination countries: {len(dest_counts)}")
print("Top 10 destination countries by record count:")
print(dest_counts.head(10))

print("\n--- Commodity Types ---")
commodity_counts = df['IFM_HS'].value_counts()
print("Commodity types and counts:")
print(commodity_counts)

print("\n--- Cost Statistics by Commodity Type ---")
cost_by_commodity = df.groupby('IFM_HS')['avg_unit_cost_per_ton'].agg(['mean', 'min', 'max', 'std'])
print(cost_by_commodity.sort_values('mean', ascending=False))

print("\n--- Lowest Cost Routes ---")
lowest_cost = df.sort_values('avg_unit_cost_per_ton').head(10)
print(lowest_cost[['origin_ISO', 'destination_ISO', 'IFM_HS', 'avg_unit_cost_per_ton']])

print("\n--- Highest Cost Routes ---")
highest_cost = df.sort_values('avg_unit_cost_per_ton', ascending=False).head(10)
print(highest_cost[['origin_ISO', 'destination_ISO', 'IFM_HS', 'avg_unit_cost_per_ton']])

# Calculate average cost per origin country
print("\n--- Average Cost by Origin Country ---")
avg_by_origin = df.groupby('origin_ISO')['avg_unit_cost_per_ton'].mean().sort_values()
print("Countries with lowest average shipping costs:")
print(avg_by_origin.head(10))
print("\nCountries with highest average shipping costs:")
print(avg_by_origin.tail(10)) 