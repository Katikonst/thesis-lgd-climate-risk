import pandas as pd
import numpy as np

# ---------------------------
# Step 1: Calculate Composite Risk
# ---------------------------
def calculate_composite_risk(data, weights=None):
    data['composite_risk'] = (
        data['wildfire_risk'] +
        data['Earthquake_Risk'] +
        data['Flood_Risk']
    )
    return data

# ---------------------------
# Step 2: Assign Loan Percentage (LTV) Based on Risk Level
# ---------------------------
def assign_loan_percentage(data, noise_scale=0.02):
    def get_ltv(risk):
        if risk < 0.5:  # Low risk
            return 0.5  # 50% LTV
        elif 0.5 <= risk <= 0.7:  # Medium risk
            return 0.6  # 60% LTV
        else:  # High risk
            return 0.7  # 70% LTV
    
    # Assign base LTV
    data['LTV'] = data['composite_risk'].apply(get_ltv)
    
    # Add noise to LTV
    data['LTV'] += np.random.normal(0, noise_scale, len(data))
    data['LTV'] = data['LTV'].clip(lower=0, upper=1)  # Ensure LTV stays within realistic bounds (0% to 100%)
    return data

# ---------------------------
# Step 3: Calculate Loan Amount
# ---------------------------
def calculate_loan_amount(data, price_column, noise_scale=0.01):
    # Base loan amount
    data['loan_amount'] = data[price_column] * data['LTV']
    
    # Add noise to loan amount
    noise = np.random.normal(0, noise_scale * data[price_column], len(data))
    data['loan_amount'] += noise
    data['loan_amount'] = data['loan_amount'].clip(lower=0, upper=data[price_column])  # Ensure loan doesn't exceed property price
    return data

# ---------------------------
# Step 4: Calculate Property Age
# ---------------------------
def calculate_property_age(data, res_date_column, construction_year_column):
    data[res_date_column] = pd.to_datetime(data[res_date_column])
    data['property_age'] = data[res_date_column].dt.year - data[construction_year_column]
    data['property_age'] = data['property_age'].clip(lower=0)
    return data

# ---------------------------
# Main Workflow
# ---------------------------
def process_data(data, price_column='res_price', res_date_column='res_date', construction_year_column='construction_year'):
    # Step 1: Calculate composite risk
    data = calculate_composite_risk(data)
    
    # Step 2: Assign loan percentage (LTV) based on composite risk
    data = assign_loan_percentage(data)
    
    # Step 3: Calculate loan amount
    data = calculate_loan_amount(data, price_column)
    
    # Step 4: Calculate property age
    data = calculate_property_age(data, res_date_column, construction_year_column)
    
    return data

# ---------------------------
# Load Dataset
# ---------------------------
file_path = r"C:\Users\katid_7ngm4sv\OneDrive\Desktop\augmscrip\properties_with_all_risks.csv"
augmented_data = pd.read_csv(file_path)

# Process the data
augmented_data = process_data(
    augmented_data,
    price_column='res_price',
    res_date_column='res_date',
    construction_year_column='construction_year'
)

# Save the updated dataset
output_path = r"C:\Users\katid_7ngm4sv\OneDrive\Desktop\augmscrip\augmented_properties_with_loans.csv"
augmented_data.to_csv(output_path, index=False)

print(f"Augmented data saved to: {output_path}")
