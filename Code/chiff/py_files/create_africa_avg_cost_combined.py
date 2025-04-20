import pandas as pd
import numpy as np

# Define African countries ISO codes
african_countries = ["DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD", 
                     "COM", "COG", "COD", "DJI", "EGY", "GNQ", "ERI", "ETH", "GAB", "GMB",
                     "GHA", "GIN", "GNB", "CIV", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI",
                     "MLI", "MRT", "MUS", "MYT", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
                     "REU", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TGO",
                     "TUN", "UGA", "TZA", "ZMB", "ZWE"]

# Path to the data file
file_path = "/Users/abishekchiffon/Documents/Technical/Masters/sem 4/captsone/data/imputed_full_matrix_at_centroid.csv"

# Output file path
output_path = "/Users/abishekchiffon/Documents/Technical/Masters/sem 4/captsone/data/africa_transport_costs.csv"

print("Reading the dataset...")
# Read the CSV file
# Use low_memory=False because the dataset is large with mixed data types
df = pd.read_csv(file_path, low_memory=False)

print("Filtering for African countries as origin...")
# Filter the dataset to include only African countries as origin
africa_df = df[df['origin_ISO'].isin(african_countries)]

print("Calculating average unit costs by country pair, commodity, and transport mode...")
# Group by origin country, destination country, commodity type, and transport mode
# Calculate the average unit logistics cost for each group
avg_cost_df = africa_df.groupby(['origin_ISO', 'destination_ISO', 'IFM_HS', 'Mode_name'])['Unit logistics costs ($/ton)'].mean().reset_index()

# Rename columns for clarity
avg_cost_df.rename(columns={
    'Unit logistics costs ($/ton)': 'avg_unit_cost_per_ton'
}, inplace=True)

print(f"Saving results to {output_path}...")
# Save the results to a CSV file
avg_cost_df.to_csv(output_path, index=False)

# Copy file to the chiff directory for web server access
import shutil
import os

chiff_data_dir = "chiff/data"
os.makedirs(chiff_data_dir, exist_ok=True)

print(f"Copying file to {chiff_data_dir} for web server access...")
shutil.copy(output_path, os.path.join(chiff_data_dir, "africa_transport_costs.csv"))

print("Done! Summary statistics:")
print(f"Total country pairs: {avg_cost_df[['origin_ISO', 'destination_ISO']].drop_duplicates().shape[0]}")
print(f"Total unique commodities: {avg_cost_df['IFM_HS'].nunique()}")
print(f"Total unique transport modes: {avg_cost_df['Mode_name'].nunique()}")
print(f"Total records: {avg_cost_df.shape[0]}")

# Print sample of the data
print("\nSample data (first 5 rows):")
print(avg_cost_df.head())

# Show average costs by transport mode
print("\nAverage costs by transport mode:")
mode_avg = avg_cost_df.groupby('Mode_name')['avg_unit_cost_per_ton'].mean().reset_index().sort_values('avg_unit_cost_per_ton')
print(mode_avg)

# Show count of routes by transport mode
print("\nCount of routes by transport mode:")
mode_count = avg_cost_df.groupby('Mode_name').size().reset_index(name='count').sort_values('count', ascending=False)
print(mode_count) 