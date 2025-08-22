import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import glob


sns.set(style="whitegrid", context="talk")

results_folder = "Model results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# -------------------------------
# 1. Load Simulation Outputs for ML
# -------------------------------
sample_files = glob.glob(os.path.join(results_folder, "*_property_results_sample.csv"))
dfs = [pd.read_csv(f) for f in sample_files]
df_ml = pd.concat(dfs, ignore_index=True)
print(f"Combined ML dataset has {df_ml.shape[0]} rows.")

# Optionally downsample if dataset is very large
if df_ml.shape[0] > 2000000:
    print("Downsampling ML dataset for memory efficiency.")
    df_ml = df_ml.sample(n=2000000, random_state=42)

# -------------------------------
# 2. Prepare Data for ML
# -------------------------------
# One-hot encode the "Scenario" column
df_ml["Scenario_orig"] = df_ml["Scenario"]
df_ml_encoded = pd.get_dummies(df_ml, columns=["Scenario"], drop_first=False)

# Define features for ML training
scenario_cols = [col for col in df_ml_encoded.columns if col.startswith("Scenario_")]
features = [
    "res_price",         # Baseline property value                         
    "Flood_Risk",        # Flood risk
    "wildfire_risk",     # Wildfire risk
    "Earthquake_Risk",   # Earthquake risk
    "Year",              # Year of observation
    "loan_amount"
] + scenario_cols       # Scenario-based features

X = df_ml_encoded[features]
y = df_ml_encoded["LGD"]

# Ensure all features are numeric using .loc to avoid SettingWithCopyWarning
for col in features:
    X.loc[:, col] = pd.to_numeric(X.loc[:, col], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Instead of dropping rows with NaN, fill missing values with the column mean
X = X.fillna(X.mean())
y = y.fillna(y.mean())

print("Shape of X after filling NaNs:", X.shape)

# -------------------------------
# 3. Scale Risk Features Using Standardization
# -------------------------------
# Standardize risk features to ensure similar ranges
scaler = StandardScaler()
risk_features = ['Flood_Risk', 'wildfire_risk', 'Earthquake_Risk']
X.loc[:, risk_features] = scaler.fit_transform(X.loc[:, risk_features])

# -------------------------------
# 4. Cross-Validation and Model Training
# -------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10,
                           random_state=42, n_jobs=1)
rf_cv_scores = cross_val_score(rf, X, y, cv=kf, scoring="r2", n_jobs=1)
print("Random Forest CV R^2 scores:", rf_cv_scores)
print("Random Forest CV Mean R^2:", np.mean(rf_cv_scores))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_y_pred)
rf_mse = mean_squared_error(y_test, rf_y_pred)
print("Random Forest Regression Performance:")
print("R^2 score:", rf_r2)
print("MSE:", rf_mse)
feature_importances_rf = rf.feature_importances_
for feature, importance in zip(features, feature_importances_rf):
    print(f"{feature}: {importance:.4f}")

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.2, max_depth=10, random_state=42)
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring="r2")
print("XGBoost CV R^2 scores:", xgb_cv_scores)
print("XGBoost CV Mean R^2:", np.mean(xgb_cv_scores))

xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_y_pred)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
print("XGBoost Regression Performance:")
print("R^2 score:", xgb_r2)
print("MSE:", xgb_mse)
importance_dict = xgb_model.get_booster().get_score(importance_type='gain')
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importance:
    print(f"{feature}: {importance:.4f}")

# -------------------------------
# 5. Save Evaluation Plots per Scenario
# -------------------------------
test_data = df_ml_encoded.loc[X_test.index].copy()
test_data["RF_Predicted_LGD"] = rf_y_pred
test_data["XGB_Predicted_LGD"] = xgb_model.predict(X_test)

scenarios_unique_ml = df_ml["Scenario_orig"].unique()

for scenario in scenarios_unique_ml:
    scenario_data = test_data[test_data["Scenario_orig"] == scenario]
    
    # Random Forest plot
    plt.figure(figsize=(10, 8))
    plt.scatter(scenario_data["LGD"], scenario_data["RF_Predicted_LGD"], alpha=0.6, color='tab:blue', edgecolor='k')
    plt.xlabel("Actual LGD", fontsize=16)
    plt.ylabel("RF Predicted LGD", fontsize=16)
    plt.title(f"Random Forest: Actual vs. Predicted LGD ({scenario} Scenario)", fontsize=20)
    plt.plot([scenario_data["LGD"].min(), scenario_data["LGD"].max()],
             [scenario_data["LGD"].min(), scenario_data["LGD"].max()], 'r--', linewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"rf_actual_vs_predicted_{scenario}.png"), dpi=300)
    plt.close()
    
    # XGBoost plot
    plt.figure(figsize=(10, 8))
    plt.scatter(scenario_data["LGD"], scenario_data["XGB_Predicted_LGD"], alpha=0.6, color='tab:green', edgecolor='k')
    plt.xlabel("Actual LGD", fontsize=16)
    plt.ylabel("XGBoost Predicted LGD", fontsize=16)
    plt.title(f"XGBoost: Actual vs. Predicted LGD ({scenario} Scenario)", fontsize=20)
    plt.plot([scenario_data["LGD"].min(), scenario_data["LGD"].max()],
             [scenario_data["LGD"].min(), scenario_data["LGD"].max()], 'r--', linewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"xgb_actual_vs_predicted_{scenario}.png"), dpi=300)
    plt.close()

print("\nFinal Model Comparison:")
print(f"Random Forest - Mean CV R^2: {np.mean(rf_cv_scores):.6f}, Test MSE: {rf_mse:.2e}")
print(f"XGBoost       - Mean CV R^2: {np.mean(xgb_cv_scores):.6f}, Test MSE: {xgb_mse:.2e}")

