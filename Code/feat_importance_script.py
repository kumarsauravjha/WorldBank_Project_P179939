#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# %%
df = pd.read_csv("../Data/imputed_full_matrix_at_centroid.csv")
# %%
df.info()
#%%
# List of African country codes (based on UN country codes)
african_countries = [
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD", "COM",
    "COD", "DJI", "EGY", "GNQ", "ERI", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB",
    "CIV", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MYT",
    "MOZ", "NAM", "NER", "NGA", "REU", "RWA", "SHN", "STP", "SEN", "SYC", "SLE",
    "SOM", "ZAF", "SSD", "SDN", "SWZ", "TZA", "TGO", "TUN", "UGA", "ESH", "ZMB", "ZWE"
]

# Filter rows where origin_ISO is an African country
df_africa = df[df['origin_ISO'].isin(african_countries)]

# Display the first few rows
print(df_africa.head())

#%%
df_africa['ship_type'].value_counts()
# %%
df["Unit logistics costs ($/ton)"].hist(bins=5)
plt.show()
# %%
df.isna().sum()
#%%
df['origin_ISO'].value_counts()
#%%
print(df_africa.groupby("Mode_name")["Unit logistics costs ($/ton)"].mean())

#%%
df_africa["Mode_name"].value_counts().plot(kind='bar')
#%%
# sns.heatmap(df.isnull(), cbar=False)
print(df.groupby("ship_type")["Unit logistics costs ($/ton)"].mean())
#%%
#calculating total cost for each row

df_africa['Total_cost'] = np.where(
    df_africa['flow(tonne)'] == 0,
    df_africa['Unit logistics costs ($/ton)'],
    df_africa['Unit logistics costs ($/ton)'] * df_africa['flow(tonne)']
)
#%%
df_africa.head()
#%%
df_africa['Unit logistics cost ($/km)'] = df_africa['Total_cost'] / df_africa['distance(km)']
df_africa['Unit logistics cost ($/km)'].describe()
#%%
'''Trade Flow Analysis'''
#routes having highest trade volume
trade_flow = df_africa.groupby(["origin_ISO", "destination_ISO"])["flow(tonne)"].sum().sort_values(ascending=False).head(10)
print(trade_flow)
#%%
# %%
print(len(df[df["flow(tonne)" == "0"]]))
# %%
df.groupby('Mode_name')['flow(tonne)'].describe()
#%%
df.groupby('ship_type')['Unit logistics costs ($/ton)'].describe()

# %%
numeric_cols = ['flow(tonne)', 'distance(km)', 'Unit logistics costs ($/ton)']

# %%
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
# %%
plt.figure(figsize=(16,10))
sns.histplot(df['Unit logistics costs ($/ton)'], kde=True,bins=10, color='blue')
plt.show()
# %%

#%%
mode_counts = df['Mode_name'].value_counts()
plt.bar(mode_counts.index,mode_counts.values)
plt.title('Ship Mode counts')
plt.show()
# %%
mode_by_distance = df.groupby('Mode_name')['distance(km)'].sum().sort_values(ascending=False)
# %%
mode_by_distance
# %%
plt.bar(mode_by_distance.index,mode_by_distance.values, alpha=0.7,)
plt.xlabel("Mode Name")
plt.ylabel("Total Distance (km)")
plt.title("Total Distance Traveled by Transport Mode")
plt.show()
# %%
mode_by_cost = df.groupby('Mode_name')['Unit logistics costs ($/ton)'].sum().sort_values(ascending=False)
mode_by_cost
# %%
plt.bar(mode_by_cost.index,mode_by_cost.values, alpha=0.7,)
plt.xlabel("Mode Name")
plt.ylabel("Unit logistics costs ($/ton)")
plt.title("unit logistics costs by Transport Mode")
plt.show()
# %%
mode_by_weight = df.groupby('Mode_name')['flow(tonne)'].sum().sort_values(ascending=False)
mode_by_weight

plt.bar(mode_by_weight.index,mode_by_weight.values, alpha=0.7,)
plt.xlabel("Mode Name")
plt.ylabel("Flow (tonne)")
plt.title("Flow (tonne) by Transport Mode")
plt.show()
# %%
'''predictive modelling'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,  classification_report
from xgboost import XGBClassifier
import shap
import xgboost as xgb
#%%
df_africa_model = df_africa.copy()

#%%
df_africa['year'] = df_africa['year'].astype('category')

#%%
df_africa['year'].value_counts()        #so only 2020 data available
#%%
numerical_cols = df_africa.select_dtypes(include=['number'])
numerical_cols.info()
#%%
categorical_cols = df_africa.select_dtypes(exclude=['number'])
categorical_cols.info()
#%%

# df_africa_model['origin_ISO']
X = df_africa[['origin_ISO', 'destination_ISO', 'distance(km)','flow(tonne)',
            #    'Unit logistics costs ($/ton)',
               'commodity_index',
               'ship_type',
               'IFM_HS']]
y = df_africa['mode']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
le = LabelEncoder()
X_train['origin_ISO_encoded'] = le.fit_transform(X_train['origin_ISO'])
X_test['origin_ISO_encoded'] = le.transform(X_test['origin_ISO'])

le2 = LabelEncoder()
X_train['destination_ISO_encoded'] = le2.fit_transform(X_train['destination_ISO'])
X_test['destination_ISO_encoded'] = le2.transform(X_test['destination_ISO'])

#%%
le_commodity = LabelEncoder()
le_ship_type = LabelEncoder()
le_ifm_hs = LabelEncoder()

# Encode 'commodity_index'
X_train['commodity_index_encoded'] = le_commodity.fit_transform(X_train['commodity_index'])
X_test['commodity_index_encoded'] = le_commodity.transform(X_test['commodity_index'])  # Only transform

# Encode 'ship_type'
X_train['ship_type_encoded'] = le_ship_type.fit_transform(X_train['ship_type'])
X_test['ship_type_encoded'] = le_ship_type.transform(X_test['ship_type'])  # Only transform

# Encode 'IFM_HS'
X_train['IFM_HS_encoded'] = le_ifm_hs.fit_transform(X_train['IFM_HS'])
X_test['IFM_HS_encoded'] = le_ifm_hs.transform(X_test['IFM_HS'])  # Only transform

#%%
X_train.drop(['origin_ISO', 'destination_ISO', 'commodity_index', 'ship_type', 'IFM_HS'], axis=1, inplace=True)
X_test.drop(['origin_ISO', 'destination_ISO', 'commodity_index', 'ship_type', 'IFM_HS'], axis=1, inplace=True)
# %%
clf_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
clf_rf.fit(X_train, y_train)
# %%
y_pred = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred))
# %%
#feature importance
importances = clf_rf.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importances)[::-1]

for idx in sorted_idx[:10]:
    print(feature_names[idx], importances[idx])

# %%
'''XGBoost'''
clf_xgb = XGBClassifier(n_estimators=50, max_depth=10, random_state=42, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, y_train)

# Make predictions
y_pred = clf_xgb.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# Feature importance
importances = clf_xgb.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importances)[::-1]

for idx in sorted_idx[:10]:  # Top 10 features
    print(feature_names[idx], importances[idx])

#%%
corr_matrix = X_train.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
# %%
# Save the XGBoost model in binary format
clf_xgb.save_model('model.json')

#%%
# Load the model from the saved binary file
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('model.json')

# SHAP Explainer
explainer = shap.Explainer(loaded_model)
shap_values = explainer(X_test)

# Initialize the SHAP JavaScript library
shap.initjs()
# %%
shap_values_mean = np.abs(shap_values).mean(axis=2)

#%%
shap.summary_plot(shap_values_mean, X_test, plot_type="bar")

# %%
shap.dependence_plot("feature_name", shap_values, X_test)
# %%
# Reduce sample size for faster SHAP computation
X_sample = X_test.sample(n=500, random_state=42)  # Adjust n based on dataset size

# Compute SHAP values only for the sample
shap_values_sample = explainer(X_sample)

# Aggregate SHAP values across classes
shap_values_mean_sample = np.abs(shap_values_sample.values).mean(axis=2)

# Summary Plot
shap.summary_plot(shap_values_mean_sample, X_sample, plot_type="bar")
# %%
shap.summary_plot(shap_values_mean_sample, X_sample)

# %%
