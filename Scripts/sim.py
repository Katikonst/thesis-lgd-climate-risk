import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set seaborn style for clarity
sns.set(style="whitegrid", context="talk")

#########################################
# 1. Data Loading and Preparation
#########################################
data_path = r"C:\Users\katid_7ngm4sv\OneDrive\Desktop\augmscrip\augmented_properties_with_loans.csv"
df = pd.read_csv(data_path)

common_columns = [
    'location_name', 'location_region', 'res_date', 'res_type', 'res_price', 
    'res_price_sqr', 'res_sqr', 'construction_year', 'levels', 'bedrooms', 
    'bathrooms', 'energyclass', 'auto_heating', 'solar', 'cooling', 'safe_door', 
    'gas', 'fireplace', 'furniture', 'student', 'parking', 'LAT_DEG', 'LONG_DEG', 
    'MEAN_TEMP', 'HIGH_TEMP', 'LOW_TEMP', 'HEAT_DEG_DAYS', 'COOL_DEG_DAYS', 
    'RAIN', 'AVG_WIND_SPEED', 'HIGHEST_WIND_SPEED', 'WIND_DIR', 'ELEVATION', 
    'geometry', 'Property_ID', 'LTV', 'loan_amount', 'property_age'
]
earthquake_columns = [
    'Earthquake_Frequency', 'Normalized_Earthquake_Frequency', 
    'Normalized_Earthquake_Magnitude', 'Earthquake_Risk', 'Earthquake_Damage'
]
wildfire_columns = [
    'Frequency', 'Avg_Fire_Duration', 'Normalized_Frequency', 
    'Normalized_Duration', 'wildfire_risk', 'wildfire_damage'
]
flood_columns = [
    'Normalized_Rainfall_Frequency', 'Normalized_Rainfall_Intensity', 
    'Rainfall_Frequency', 'Rainfall_Intensity', 'Flood_Risk', 'Flood_Damage'
]

df = df[common_columns + earthquake_columns + wildfire_columns + flood_columns]

# Fill missing risk values with the column mean
for col in ["Flood_Risk", "wildfire_risk", "Earthquake_Risk"]:
    df[col] = df[col].fillna(df[col].mean())

# Convert numeric columns to float32 for memory efficiency
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype(np.float32)

# Save a pristine copy for simulation runs
df_orig = df.copy()

#########################################
# 2. Simulation Parameters and Scenario Setup
#########################################
NUM_SIMULATIONS = 100  # iterations per year
years = np.arange(2025, 2051)  # 26 years: 2025-2050

# Base growth rates for risk factors (annual means) with small variation
growth_flood_mean = 0.02
growth_wildfire_mean = 0.03
growth_earthquake_mean = 0.01
growth_std = 0.005

# Economic shock: adjust property prices each year (mean 0, std 0.02)
econ_shock_std = 0.02

# Gamma exponent to introduce non-linearity in combined risk (gamma > 1 exaggerates simultaneous risks)
gamma = 1.3

# Noise standard deviation for risk multipliers
noise_std = 0.05

# Define simulation scenarios
scenarios = {
    "Baseline": {
         "flood_multiplier_mean": 1.0, "flood_multiplier_std": 0.1,
         "wildfire_multiplier_mean": 1.0, "wildfire_multiplier_std": 0.1,
         "earthquake_multiplier_mean": 1.0, "earthquake_multiplier_std": 0.1
    },
    "High Flood Risk": {
         "flood_multiplier_mean": 1.5, "flood_multiplier_std": 0.1,
         "wildfire_multiplier_mean": 1.0, "wildfire_multiplier_std": 0.1,
         "earthquake_multiplier_mean": 1.0, "earthquake_multiplier_std": 0.1
    },
    "High Wildfire Risk": {
         "flood_multiplier_mean": 1.0, "flood_multiplier_std": 0.1,
         "wildfire_multiplier_mean": 1.5, "wildfire_multiplier_std": 0.1,
         "earthquake_multiplier_mean": 1.0, "earthquake_multiplier_std": 0.1
    },
    "High Earthquake Risk": {
         "flood_multiplier_mean": 1.0, "flood_multiplier_std": 0.1,
         "wildfire_multiplier_mean": 1.0, "wildfire_multiplier_std": 0.1,
         "earthquake_multiplier_mean": 1.5, "earthquake_multiplier_std": 0.1
    },
    "Flood + Wildfire": {
         "flood_multiplier_mean": 1.5, "flood_multiplier_std": 0.1,
         "wildfire_multiplier_mean": 1.5, "wildfire_multiplier_std": 0.1,
         "earthquake_multiplier_mean": 1.0, "earthquake_multiplier_std": 0.1
    },
}

results_folder = "Model results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

#########################################
# 3. Monte Carlo Simulation Over Time and Scenarios
#########################################
# Use a sample fraction to limit memory use
SAMPLE_FRACTION = 0.02
all_property_samples = []  # For potential future analysis

for scenario_name, params in scenarios.items():
    print(f"Processing scenario: {scenario_name}")
    regional_results = []
    property_sample_list = []
    
    for year in years:
        # Sample annual growth rates for risk factors
        growth_flood = np.random.normal(growth_flood_mean, growth_std)
        growth_wildfire = np.random.normal(growth_wildfire_mean, growth_std)
        growth_earthquake = np.random.normal(growth_earthquake_mean, growth_std)
        
        # Sample an economic shock for property values and compute multiplier
        econ_shock = np.random.normal(0, econ_shock_std)
        property_multiplier = 1 + econ_shock
        
        # Generate noise factors for risk multipliers
        noise_flood = np.random.normal(1.0, noise_std)
        noise_wildfire = np.random.normal(1.0, noise_std)
        noise_earthquake = np.random.normal(1.0, noise_std)
        year_index = year - years[0]
        multiplier_flood_time = (1 + growth_flood) ** year_index * noise_flood
        multiplier_wildfire_time = (1 + growth_wildfire) ** year_index * noise_wildfire
        multiplier_earthquake_time = (1 + growth_earthquake) ** year_index * noise_earthquake
        
        for sim in range(NUM_SIMULATIONS):
            df_sim = df_orig.copy()
            
            # Adjust property price with economic shock factor
            df_sim["res_price"] = df_sim["res_price"] * property_multiplier
            
            # For the "Flood + Wildfire" scenario, correlate flood and wildfire multipliers
            if scenario_name == "Flood + Wildfire":
                means = [params["flood_multiplier_mean"], params["wildfire_multiplier_mean"]]
                std_flood = params["flood_multiplier_std"]
                std_wildfire = params["wildfire_multiplier_std"]
                corr = 0.5  # chosen correlation coefficient
                cov = [[std_flood**2, corr*std_flood*std_wildfire],
                       [corr*std_flood*std_wildfire, std_wildfire**2]]
                sample = np.random.multivariate_normal(means, cov)
                sampled_flood = max(sample[0], 0.5)
                sampled_wildfire = max(sample[1], 0.5)
                sampled_earthquake = max(np.random.normal(params["earthquake_multiplier_mean"],
                                                          params["earthquake_multiplier_std"]), 0.5)
            else:
                sampled_flood = max(np.random.normal(params["flood_multiplier_mean"], params["flood_multiplier_std"]), 0.5)
                sampled_wildfire = max(np.random.normal(params["wildfire_multiplier_mean"], params["wildfire_multiplier_std"]), 0.5)
                sampled_earthquake = max(np.random.normal(params["earthquake_multiplier_mean"], params["earthquake_multiplier_std"]), 0.5)
            
            flood_multiplier = sampled_flood * multiplier_flood_time
            wildfire_multiplier = sampled_wildfire * multiplier_wildfire_time
            earthquake_multiplier = sampled_earthquake * multiplier_earthquake_time
            
            # Adjust risk factors based on multipliers and cap at 1.0
            df_sim["adj_Flood_Risk"] = (df_sim["Flood_Risk"] * flood_multiplier).clip(upper=1.0)
            df_sim["adj_Wildfire_Risk"] = (df_sim["wildfire_risk"] * wildfire_multiplier).clip(upper=1.0)
            df_sim["adj_Earthquake_Risk"] = (df_sim["Earthquake_Risk"] * earthquake_multiplier).clip(upper=1.0)
            
            # Combine risks with non-linear interaction using gamma exponent
            df_sim["combined_risk"] = 1 - (((1 - df_sim["adj_Flood_Risk"]) *
                                            (1 - df_sim["adj_Wildfire_Risk"]) *
                                            (1 - df_sim["adj_Earthquake_Risk"])) ** gamma)
            
            # Compute property value after risk impact
            df_sim["Property_Value_After_Impact"] = df_sim["res_price"] * (1 - df_sim["combined_risk"])
            
            # Incorporate property vulnerability based on age
            vulnerability_multiplier = 1 + df_sim["property_age"] / 100.0
            df_sim["Property_Value_After_Impact"] *= vulnerability_multiplier
            
            # Sample a stochastic haircut from 7% to 15% and compute effective property value
            current_haircut = np.random.uniform(0.07, 0.15)
            df_sim["Effective_Property_Value"] = df_sim["Property_Value_After_Impact"] * (1 - current_haircut)
            df_sim["Haircut"] = current_haircut
            
            # Calculate LGD, VtL_original, VtL_after, and VtL_change
            df_sim["LGD"] = (1 - (df_sim["Effective_Property_Value"] / df_sim["res_price"])) * df_sim["LTV"]
            # If LGD exceeds 1, set it to 0.95
            df_sim["LGD"] = np.where(df_sim["LGD"] > 1, 0.95, df_sim["LGD"])
            df_sim["VtL_original"] = df_sim["res_price"] / df_sim["loan_amount"]
            df_sim["VtL_after"] = df_sim["Property_Value_After_Impact"] / df_sim["loan_amount"]
            df_sim["VtL_change"] = (df_sim["VtL_after"] - df_sim["VtL_original"]) / df_sim["VtL_original"]
            
            df_sim["Scenario"] = scenario_name
            df_sim["Year"] = year
            
            # Sample a fraction of properties per region to limit memory use
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")
                df_sample = df_sim.groupby('location_region', group_keys=False).apply(
                    lambda x: x.sample(frac=SAMPLE_FRACTION, random_state=sim)
                ).reset_index(drop=True)
            property_sample_list.append(df_sample)
            
            # Regional aggregation: compute mean LGD, VtL_change and risk measures per region
            region_means = df_sim.groupby("location_region").agg({
                "LGD": "mean",
                "VtL_change": "mean",
                "adj_Flood_Risk": "mean",
                "adj_Wildfire_Risk": "mean",
                "adj_Earthquake_Risk": "mean"
            }).reset_index()
            region_means["Scenario"] = scenario_name
            region_means["Year"] = year
            regional_results.append(region_means)
            
            del df_sim
            gc.collect()
    
    # Save regional results for the scenario (including risk measures)
    regional_results_df = pd.concat(regional_results, ignore_index=True)
    regional_summary = regional_results_df.groupby(["location_region", "Scenario", "Year"]).agg({
        "LGD": "mean",
        "VtL_change": "mean",
        "adj_Flood_Risk": "mean",
        "adj_Wildfire_Risk": "mean",
        "adj_Earthquake_Risk": "mean"
    }).reset_index()
    # Calculate LGD change relative to base year (2025)
    baseline_vals = regional_summary[regional_summary["Year"] == years[0]].copy()
    baseline_vals = baseline_vals.set_index(["location_region", "Scenario"])["LGD"]
    regional_summary["LGD_change"] = regional_summary.apply(
        lambda row: row["LGD"] - baseline_vals.get((row["location_region"], row["Scenario"]), np.nan),
        axis=1
    )
    # Save the CSV with all aggregated measures
    regional_output_path = os.path.join(results_folder, f"{scenario_name}_regional_results.csv")
    regional_summary.to_csv(regional_output_path, index=False)
    print(f"Saved regional results for scenario '{scenario_name}' to {regional_output_path}")
    
    # Save property-level sample results for the scenario
    df_ml_scenario = pd.concat(property_sample_list, ignore_index=True)
    sample_output_path = os.path.join(results_folder, f"{scenario_name}_property_results_sample.csv")
    df_ml_scenario.to_csv(sample_output_path, index=False)
    print(f"Saved property-level sample results for scenario '{scenario_name}' to {sample_output_path}")
    
    all_property_samples.append(df_ml_scenario)
    del regional_results, property_sample_list, regional_results_df, regional_summary, df_ml_scenario
    gc.collect()
