#!/usr/bin/env python
# coding: utf-8

"""
Transport Cost Prediction Model
===============================
This script builds regression models to predict transport costs based on various features
such as distance, flow, mode, origin/destination, commodity type, and equipment type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from category_encoders import TargetEncoder
import shap
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

print("Starting Transport Cost Prediction Modeling...")

# Load the dataset
try:
    df = pd.read_csv("imputed_full_matrix_at_centroid.csv")
except FileNotFoundError:
    df = pd.read_csv("../../../data/imputed_full_matrix_at_centroid.csv")

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Display basic statistics of the target variable
print("\nBasic statistics of transport costs:")
print(df['Unit logistics costs ($/ton)'].describe())

# Check for outliers and visualize the distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Unit logistics costs ($/ton)'], bins=50, kde=True)
plt.title('Distribution of Unit Logistics Costs', fontsize=16)
plt.xlabel('Unit Logistics Costs ($/ton)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(df['Unit logistics costs ($/ton)'].median(), color='red', linestyle='--', 
           label=f'Median: {df["Unit logistics costs ($/ton)"].median():.2f}')
plt.axvline(df['Unit logistics costs ($/ton)'].mean(), color='green', linestyle='--', 
           label=f'Mean: {df["Unit logistics costs ($/ton)"].mean():.2f}')
plt.legend()
plt.savefig('cost_distribution_prediction.png')
plt.close()

# Remove extreme outliers for modeling purposes
# (can be adjusted based on domain knowledge)
q1 = df['Unit logistics costs ($/ton)'].quantile(0.01)
q3 = df['Unit logistics costs ($/ton)'].quantile(0.99)
df_model = df[(df['Unit logistics costs ($/ton)'] >= q1) & 
              (df['Unit logistics costs ($/ton)'] <= q3)]

print(f"Removed {df.shape[0] - df_model.shape[0]} outliers ({(1 - df_model.shape[0]/df.shape[0])*100:.2f}%)")

# Define features for modeling
numeric_features = ['distance(km)', 'flow(tonne)']
categorical_features = ['Mode_name', 'origin_ISO', 'destination_ISO', 'IFM_HS', 
                       'ship_type', 'container_type']

# Prepare features and target variable
X = df_model[numeric_features + categorical_features].copy()
y = df_model['Unit logistics costs ($/ton)']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create preprocessing pipeline with different strategies for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Target encoding is often more effective than one-hot for high-cardinality features
categorical_transformer = Pipeline(steps=[
    ('target_encoder', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models to test
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'pipeline': pipeline
    }
    
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")

# Identify the best model based on RMSE
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['pipeline']
print(f"\nBest model: {best_model_name} with RMSE = {results[best_model_name]['RMSE']:.2f}")

# Fine-tune the best model (if it's a tree-based model)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    print(f"\nFine-tuning {best_model_name}...")
    
    param_grid = {}
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5, 10]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5, 10],
            'model__subsample': [0.8, 1.0]
        }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        best_model,
        param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best parameters and model
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluate the tuned model
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Tuned model performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")

# Feature importance analysis for the best model
print("\nAnalyzing feature importance...")

if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    # For tree-based models, we can extract feature importance directly
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        
        # Get feature names after preprocessing
        feature_names = (numeric_features + 
                        [f"{col}_{val}" for col in categorical_features for val in df_model[col].unique()])
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances ({best_model_name})', fontsize=16)
        plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance_transport_cost.png')
        plt.close()
        
        # Print top 10 features
        print("\nTop 10 most important features:")
        for i in range(min(10, len(indices))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # SHAP analysis for more detailed feature importance
    try:
        # Sample a subset of the test data for SHAP analysis (for performance)
        X_sample = shap.sample(X_test, 1000, random_state=42)
        
        # Create explainer
        if best_model_name == 'XGBoost':
            explainer = shap.TreeExplainer(best_model.named_steps['model'])
        else:
            explainer = shap.Explainer(best_model.named_steps['model'])
        
        # Calculate SHAP values
        shap_values = explainer(preprocessor.transform(X_sample))
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance ({best_model_name})', fontsize=16)
        plt.tight_layout()
        plt.savefig('shap_importance_transport_cost.png')
        plt.close()
        
        # Detailed SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Summary Plot ({best_model_name})', fontsize=16)
        plt.tight_layout()
        plt.savefig('shap_summary_transport_cost.png')
        plt.close()
        
    except Exception as e:
        print(f"SHAP analysis error: {e}")
        print("Skipping SHAP analysis.")
        
# Create cost predictions for specific scenarios
print("\nGenerating cost predictions for specific scenarios...")

# Example 1: Compare costs across modes for the same origin-destination and commodity
def compare_modes_costs(model, origin, destination, commodity, distance, flow):
    """Compare predicted costs across different transport modes."""
    modes = ['Air', 'Rail', 'Road', 'Sea']
    results = []
    
    # Create test data for each mode
    for mode in modes:
        # Create a sample with the specified parameters
        sample = pd.DataFrame({
            'distance(km)': [distance],
            'flow(tonne)': [flow],
            'Mode_name': [mode],
            'origin_ISO': [origin],
            'destination_ISO': [destination],
            'IFM_HS': [commodity],
            'ship_type': ['Container'] if mode == 'Sea' else ['NA'],
            'container_type': ['Container_20ft'] if mode in ['Sea', 'Rail'] else ['NA']
        })
        
        # Predict cost
        try:
            cost = model.predict(sample)[0]
            results.append((mode, cost))
        except:
            results.append((mode, "Prediction failed"))
    
    return pd.DataFrame(results, columns=['Mode', 'Predicted Cost ($/ton)'])

# Example scenarios
try:
    # Scenario 1: USA to China with Electronic Devices
    print("\nScenario 1: USA to China with Electronic Devices (5000 km, 10 tonnes)")
    usa_china = compare_modes_costs(
        best_model, 'USA', 'CHN', 'Electronic_devices', 10000, 10
    )
    print(usa_china)
    
    # Scenario 2: Kenya to South Africa with Agricultural Products
    print("\nScenario 2: Kenya to South Africa with Agricultural Products (3000 km, 100 tonnes)")
    kenya_sa = compare_modes_costs(
        best_model, 'KEN', 'ZAF', 'Agricultural_products', 3000, 100
    )
    print(kenya_sa)
    
    # Scenario 3: Germany to France with Machinery
    print("\nScenario 3: Germany to France with Machinery (1000 km, 50 tonnes)")
    germany_france = compare_modes_costs(
        best_model, 'DEU', 'FRA', 'Machinery', 1000, 50
    )
    print(germany_france)
except Exception as e:
    print(f"Scenario testing error: {e}")
    print("Make sure all required sample values exist in the training data.")

# Print conclusions
print("\nConclusions:")
print(f"1. Best prediction model: {best_model_name} with R² = {results[best_model_name]['R2']:.4f}")
print(f"2. Most important factors in determining transport costs:")
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost'] and hasattr(best_model.named_steps['model'], 'feature_importances_'):
    for i in range(min(3, len(indices))):
        print(f"   - {feature_names[indices[i]]}")
print("3. The model can be used to compare transport costs across modes, origins, destinations, and commodities")
print("4. Possible applications include route optimization, mode selection, and cost forecasting")

# Save the model for future use
import pickle
try:
    with open('transport_cost_prediction_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("\nModel saved successfully as 'transport_cost_prediction_model.pkl'")
except Exception as e:
    print(f"Error saving model: {e}") 