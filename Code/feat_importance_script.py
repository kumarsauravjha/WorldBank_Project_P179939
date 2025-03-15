#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,  classification_report
from xgboost import XGBClassifier
import shap
import xgboost as xgb
from category_encoders import TargetEncoder
import pickle
# %%
df = pd.read_csv("imputed_full_matrix_at_centroid.csv")
df.head()
# %%
# Convert relevant columns to numeric
numeric_cols = ['flow(tonne)', 'distance(km)', 'Unit logistics costs ($/ton)', 'Model']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Mode_name', y='Unit logistics costs ($/ton)')
plt.xticks(rotation=45)
plt.title('Mode-Specific Transport Costs')
plt.ylabel('Unit Logistics Costs ($/ton)')
plt.xlabel('Transport Mode')
plt.show()
# %%
african_countries = [
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD", "COM",
    "COD", "DJI", "EGY", "GNQ", "ERI", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB",
    "CIV", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MYT",
    "MOZ", "NAM", "NER", "NGA", "REU", "RWA", "SHN", "STP", "SEN", "SYC", "SLE",
    "SOM", "ZAF", "SSD", "SDN", "SWZ", "TZA", "TGO", "TUN", "UGA", "ESH", "ZMB", "ZWE"
]

print(len(african_countries))

#%%
'''considering only export data'''
df_africa = df[df['origin_ISO'].isin(african_countries)]
df_africa.shape


# %%
#understanding the Model field
df.groupby('Model')['Unit logistics costs ($/ton)'].describe()
# %%
df.groupby(['IFM_HS','commodity_index']).nunique()
# %%
df.groupby(['IFM_HS','commodity_index'])['Model'].nunique()
# %%
grouped_model = df.groupby(['IFM_HS','commodity_index'])['Model']
#%%
print(grouped_model.head())
# %%
df[["commodity_index", "Model"]].drop_duplicates().sort_values("commodity_index")

# %%
df.groupby('commodity_index')['Model'].nunique()
# %%
df['ship_type'].head()
# %%
df['ship_type'].value_counts()
# %%
df['container_type'].value_counts()
# %%
df['Mode_name'].unique()
# %%
df[df.Mode_name == 'Sea']['ship_type'].head(10)
# %%
df[df.Mode_name == 'Rail']['ship_type'].head(10)
# %%
df.ship_type.unique()
# %%
df.container_type.unique()
# %%
cols = df.dtypes
cols
# for row in 
# %%
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Unique values of {col} are {df[col].unique()}")
# %%
print(df_africa.select_dtypes(include='object').columns)
# %%
print(df_africa.select_dtypes(include='number').columns)
# %%
# df_africa_model['origin_ISO']
X = df_africa[['origin_ISO', 'destination_ISO', 'distance(km)','flow(tonne)',
            #    'commodity_index',
               'ship_type',
               'IFM_HS'
               ]].copy()
y = df_africa['mode']
# Concatenate origin & destination
X['origin_destination'] = X['origin_ISO'].astype(str) + "_" + X['destination_ISO'].astype(str)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Define encoders
iso_encoder = TargetEncoder()
ship_type_encoder = TargetEncoder()
IFM_HS_encoder = TargetEncoder()

# Perform Target Encoding using temporary variables
X_train['origin_destination_encoded'] = iso_encoder.fit_transform(X_train[['origin_destination']], y_train)
X_test['origin_destination_encoded'] = iso_encoder.transform(X_test[['origin_destination']]) 

X_train['ship_type_encoded'] = ship_type_encoder.fit_transform(X_train[['ship_type']], y_train) 
X_test['ship_type_encoded'] = ship_type_encoder.transform(X_test[['ship_type']])

X_train['IHM_HS_encoded'] = IFM_HS_encoder.fit_transform(X_train[['IFM_HS']], y_train) 
X_test['IHM_HS_encoded'] = IFM_HS_encoder.transform(X_test[['IFM_HS']])
# %%
# Drop original categorical columns
X_train.drop(columns=['origin_ISO', 'destination_ISO', 'origin_destination', 'ship_type', 'IFM_HS'], inplace=True)
X_test.drop(columns=['origin_ISO', 'destination_ISO', 'origin_destination', 'ship_type','IFM_HS'], inplace=True)


#%%
from sklearn.model_selection import RandomizedSearchCV
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [6, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5]
}

# Initialize model
xgb_model = XGBClassifier(eval_metric='logloss')

# RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=10, cv=3, scoring='f1_weighted', verbose=2, n_jobs=-1)
random_search.fit(X_train, y_train)

# Best model
best_xgb = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Evaluate
y_pred = best_xgb.predict(X_test)
print(classification_report(y_test, y_pred))
# %%
# Feature importance
importances = best_xgb.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importances)[::-1]

for idx in sorted_idx[:10]:  # Top 10 features
    print(feature_names[idx], importances[idx])

#%%
#saving the model with best parameters
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

#%%
# X_sample = X_test.sample(n=5000, random_state=42)  # Adjust sample size for performance
X_sample = shap.sample(X_test, 10000, random_state=42)
# %%
'''SHAP for every mode'''
explainer = shap.TreeExplainer(best_xgb, approximate=True)


# Compute SHAP values
shap_values_sample = explainer(X_sample, check_additivity=False)

# Handle multi-class case
if len(shap_values_sample.values.shape) == 3:
    shap_values_processed = shap_values_sample.values.mean(axis=2)
else:
    shap_values_processed = shap_values_sample.values

# Plot SHAP summary
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_processed, X_sample, show=False) #overall summary plot
plt.title("SHAP Summary for All Modes", fontsize=14, y=1.05)
plt.show()
# %%
#Summary plot of individual modes
mode_names = ["air", "rail", "road", "sea"]  # Updated order

for i, mode_name in enumerate(mode_names):
    plt.figure(figsize=(8, 6))  # Create a new figure
    
    shap.summary_plot(shap_values_sample.values[:, :, i], X_sample, feature_names=X_sample.columns, show=False)
    plt.title(f"SHAP Summary for Mode: {mode_name}", fontsize=14, y=1.05)  # Add title
    plt.show()

# %%
#force plot
# from IPython.display import display

# # Pick a single sample from X_test
# sample_idx = 10  # Choose any row index
# X_single = X_test.iloc[sample_idx:sample_idx+1]  # Keep it as a DataFrame

# # Compute SHAP values for this single sample
# explainer2 = shap.Explainer(best_xgb)
# shap_values_single = explainer2(X_single)

# # Define mode names
# mode_names = ["air", "rail", "road", "sea"]  # Ensure correct order

# # Set light background for better contrast
# shap.initjs()
# plt.style.use("default")  # Reset to default style

# # Generate force plots with better tick display
# for i, mode_name in enumerate(mode_names):
#     print(f"\nSHAP Force Plot for Mode: {mode_name}")

#     fig, ax = plt.subplots(figsize=(10, 3))  # Adjust figure size
#     shap.plots.force(
#         explainer.expected_value[i], 
#         shap_values_single.values[:, :, i], 
#         X_single, 
#         matplotlib=True,  # Use Matplotlib rendering
#         show=True
#     )
#     plt.xticks(fontsize=12)  # Make ticks visible
#     plt.xlabel("Feature Contribution", fontsize=12)  # Label x-axis
#     plt.title(f"SHAP Force Plot for Mode: {mode_name}", fontsize=14)
#     plt.show()

# %%
# Plot SHAP summary
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_processed, X_sample, plot_type="bar",show=False) #overall summary plot
plt.title("SHAP Summary for All Modes", fontsize=14, y=1.05)
plt.show()
# %%
