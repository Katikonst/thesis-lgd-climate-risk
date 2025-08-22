import pandas as pd
import numpy as np
from geopy.distance import distance

# Load datasets
properties = pd.read_csv("C:/Users/katid_7ngm4sv/OneDrive/Desktop/updated_dataset.csv")  
earthquakes = pd.read_csv("C:/Users/katid_7ngm4sv/OneDrive/Desktop/earthquake/Earthquakes_v3.csv")

# Rename earthquake columns
earthquakes.rename(columns={"DATETIME": "Datetime", "LAT": "Earthquake_LAT", "LONG": "Earthquake_LONG", 
                            "DEPTH": "Depth", "MAGNITUDE": "Magnitude"}, inplace=True)

# Define a function to filter earthquakes based on a latitude and longitude range
def get_nearby_earthquakes(property_row, earthquakes_df, lat_range=0.5, lon_range=0.5):
    property_lat = property_row['LAT_DEG']
    property_lon = property_row['LONG_DEG']
    
    # Define the bounds for the filtering range
    lat_min = property_lat - lat_range
    lat_max = property_lat + lat_range
    lon_min = property_lon - lon_range
    lon_max = property_lon + lon_range
    
    # Filter earthquakes within this bounding box
    nearby_eq = earthquakes_df[(
        earthquakes_df['Earthquake_LAT'] >= lat_min) & 
        (earthquakes_df['Earthquake_LAT'] <= lat_max) & 
        (earthquakes_df['Earthquake_LONG'] >= lon_min) & 
        (earthquakes_df['Earthquake_LONG'] <= lon_max)
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    return nearby_eq

# Define a function to process a batch of properties
def process_batch(properties_batch):
    filtered_earthquakes = []
    for _, property_row in properties_batch.iterrows():
        nearby_eq = get_nearby_earthquakes(property_row, earthquakes)
        
        # Use .loc to avoid SettingWithCopyWarning
        nearby_eq.loc[:, 'Property_ID'] = property_row.name  # Add Property ID for tracking
        filtered_earthquakes.append(nearby_eq)

    # Concatenate the filtered results for the batch
    filtered_earthquakes_df = pd.concat(filtered_earthquakes, ignore_index=True)

    # Aggregate risk metrics by property
    earthquake_risk = filtered_earthquakes_df.groupby('Property_ID').agg(
        Earthquake_Frequency=('Magnitude', 'size'),
        Avg_Earthquake_Magnitude=('Magnitude', 'mean')
    ).reset_index()

    # Normalize and calculate risk
    earthquake_risk['Normalized_Earthquake_Frequency'] = earthquake_risk['Earthquake_Frequency'] / earthquake_risk['Earthquake_Frequency'].max()
    earthquake_risk['Normalized_Earthquake_Magnitude'] = earthquake_risk['Avg_Earthquake_Magnitude'] / 10
    earthquake_risk['Earthquake_Risk'] = earthquake_risk['Normalized_Earthquake_Frequency'] * earthquake_risk['Normalized_Earthquake_Magnitude']
    earthquake_risk['Earthquake_Damage'] = earthquake_risk['Earthquake_Risk'] * properties['res_price']  # Or use another column for Property Value Factor


    # Return the calculated earthquake risk for this batch
    return earthquake_risk[['Property_ID', 'Earthquake_Frequency', 'Normalized_Earthquake_Frequency', 
                             'Normalized_Earthquake_Magnitude', 'Earthquake_Risk', 'Earthquake_Damage']]

# Batch processing function
def process_in_batches(properties_df, batch_size=1000):
    all_earthquake_risk = []
    total_properties = len(properties_df)
    
    for start in range(0, total_properties, batch_size):
        end = min(start + batch_size, total_properties)
        properties_batch = properties_df[start:end]
        earthquake_risk_batch = process_batch(properties_batch)
        all_earthquake_risk.append(earthquake_risk_batch)

    # Concatenate all the results
    all_earthquake_risk_df = pd.concat(all_earthquake_risk, ignore_index=True)

    return all_earthquake_risk_df

# Process the properties in batches
earthquake_risk_all = process_in_batches(properties)

# Merge the new earthquake risk data back into properties
properties = properties.merge(earthquake_risk_all, how='left', left_index=True, right_on='Property_ID')
# Remove the old earthquake risk and earthquake damage columns
properties.drop(columns=['earthquake_risk', 'earthquake_damage'], inplace=True, errors='ignore')

# Save the updated dataset
properties.to_csv('updated_properties_with_earthquake_risk.csv', index=False)
print("Earthquake risk calculation completed in batches. Updated dataset saved.")
