import pandas as pd
import numpy as np

# Load real property data
property_data = pd.read_csv(r'path\to\greece_listings.csv')
additional_data_1 = pd.read_excel(r'path\to\A1602_SAM05_TB_DC_00_2021_A03_F_GR.xlsx', skiprows=5)
additional_data_2 = pd.read_excel(r'path\to\A1602_SAM05_TB_DC_00_2021_A02_F_GR.xlsx', skiprows=5)

# Clean and standardize column names for size distributions
additional_data_1.columns = [f'col_{i}' for i in range(len(additional_data_1.columns))]
size_data = additional_data_1[['col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7']].dropna()
size_data.columns = ['Region', '30-39', '40-49', '50-59', '60-79', '80+']

# Define regions and corresponding cities with coordinates
region_to_address = {
    "Πελοπόννησος": {
        "cities": ["Πάτρα", "Αγρίνιο", "Σπάρτη", "Ναύπακτος"],
        "coordinates": (37.508, 22.379)
    },
    "Κρήτη": {
        "cities": ["Ηράκλειο", "Χανιά", "Ρέθυμνο", "Λασίθι"],
        "coordinates": (35.240, 24.809)
    },
    "Ήπειρος": {
        "cities": ["Ιωάννινα", "Άρτα", "Πρέβεζα", "Ηγουμενίτσα"],
        "coordinates": (39.665, 20.853)
    },
    "Θεσσαλία": {
        "cities": ["Λάρισα", "Βόλος", "Τρίκαλα", "Καρδίτσα"],
        "coordinates": (39.625, 22.382)
    },
    "Μακεδονία": {
        "cities": ["Κιλκίς", "Καβάλα", "Βέροια", "Κοζάνη", "Σέρρες"],
        "coordinates": (40.620, 22.970)
    },
    "Ιόνια Νησιά": {
        "cities": ["Κέρκυρα", "Λευκάδα", "Κεφαλονιά", "Ζάκυνθος", "Ιθάκη"],
        "coordinates": (38.200, 20.600)
    },
    "Στερεά Ελλάδα": {
        "cities": ["Λαμία", "Χαλκίδα", "Θήβα", "Άμφισσα"],
        "coordinates": (38.500, 23.000)
    },
    "Δυτικό Αιγαίο": {
        "cities": ["Ρόδος", "Κως", "Σάμος", "Λέσβος"],
        "coordinates": (37.950, 26.350)
    },
    "Βόρειο Αιγαίο": {
        "cities": ["Χίος", "Λήμνος", "Ικαρία", "Σκύρος"],
        "coordinates": (39.200, 25.900)
    },
    "Νότιο Αιγαίο": {
        "cities": ["Μύκονος", "Σαντορίνη", "Πάρος", "Νάξος"],
        "coordinates": (36.850, 25.300)
    }
}

# Number of samples
num_samples = 20000

# Synthetic location names based on cities in the regions
location_names = np.random.choice(
    [city for region in region_to_address.values() for city in region['cities']],
    num_samples
)

# Synthetic location regions and coordinates
location_regions = []
location_lat = []
location_long = []
for location_name in location_names:
    for region, data in region_to_address.items():
        if location_name in data['cities']:
            location_regions.append(region)
            location_lat.append(data['coordinates'][0])
            location_long.append(data['coordinates'][1])
            break

# Function to generate a valid date if NaT is encountered
def generate_valid_date():
    while True:
        random_date = np.random.choice(pd.date_range("2023-01-01", "2023-12-31"))
        if pd.notnull(random_date):  # Ensure the date is not NaT
            return random_date

# Generate the rest of the synthetic data
synthetic_data = {
    "location_name": location_names,
    "location_region": location_regions,
    "LAT": location_lat,
    "LONG": location_long,
    "res_date": [generate_valid_date() for _ in range(num_samples)],
    "res_type": np.random.choice(["Διαμέρισμα", "Μεζονέτα", "Μονοκατοικία"], num_samples),
    "res_address": location_names,
    "res_price": np.random.normal(120000, 50000, num_samples).clip(20000, 500000),
    "res_sqr": np.random.randint(50, 150, num_samples),
    "construction_year": np.random.randint(1970, 2022, num_samples),
    "levels": np.random.choice(["1ος", "2ος", "3ος", "4ος", "5ος", "Υπερυψωμένο", "Υπόγειο", "Ισόγειο", "Ημιυπόγειο"], num_samples),
    "bathrooms": np.random.choice([1, 2], num_samples),
    "energyclass": np.random.choice(["A", "B", "C", "D", "E"], num_samples),
    "furniture": np.random.choice([0, 1], num_samples),
    "student": np.random.choice([0, 1], num_samples),
    "parking": np.random.choice(["Open parking", "Closed parking", "No parking"], num_samples),
    "auto_heating": np.random.choice([0, 1], num_samples),
    "solar": np.random.choice([0, 1], num_samples),
    "cooling": np.random.choice([0, 1], num_samples),
    "safe_door": np.random.choice([0, 1], num_samples),
    "gas": np.random.choice([0, 1], num_samples),
    "fireplace": np.random.choice([0, 1], num_samples),
}

# Generate bedrooms based on square footage
def get_bedrooms_from_size(size):
    if size < 50:
        return 1
    elif 50 <= size < 80:
        return 2
    elif 80 <= size < 120:
        return 3
    else:
        return 4

synthetic_data["bedrooms"] = [get_bedrooms_from_size(size) for size in synthetic_data["res_sqr"]]

# Format dates
default_date = pd.Timestamp('2023-05-01')
synthetic_data["res_date"] = [
    pd.Timestamp(date).strftime('%d/%m/%Y') if pd.notnull(date) else default_date.strftime('%d/%m/%Y')
    for date in synthetic_data["res_date"]
]

# Create DataFrame from synthetic data
synthetic_df = pd.DataFrame(synthetic_data)

# Calculate price per square meter
synthetic_df['res_price_sqr'] = synthetic_df['res_price'] / synthetic_df['res_sqr']

# Convert 'res_price' and 'res_price_sqr' to integers
synthetic_df['res_price'] = synthetic_df['res_price'].round().astype(int)
synthetic_df['res_price_sqr'] = synthetic_df['res_price_sqr'].round().astype(int)

# Select columns to keep
columns_to_keep = [
    "location_name", "location_region", "res_date", "res_type", "res_address", "res_price", 
    "res_price_sqr", "res_sqr", "construction_year", "levels", "bedrooms", "bathrooms", 
    "energyclass", "auto_heating", "solar", "cooling", "safe_door", "gas", 
    "fireplace", "furniture", "student", "parking"
]

# Keep only the selected columns
synthetic_df = synthetic_df[columns_to_keep]

# Set 'levels' to 1 for 'Μονοκατοικία' (Monokatoikia) entries
synthetic_df.loc[synthetic_df['res_type'] == 'Μονοκατοικία', 'levels'] = 'Ισόγειο'

# Combine synthetic data with real property data
combined_df = pd.concat([property_data, synthetic_df], ignore_index=True)

print(f"Columns before dropping: {combined_df.columns.tolist()}")

combined_df = combined_df.drop(columns=['status', 'deleted', 'deleted_at'], errors='ignore')

# Print column names after dropping
print(f"Columns after dropping: {combined_df.columns.tolist()}")

# Save the combined dataset
combined_df.to_csv(r'path\to\combined_property_data.csv', index=False)

print("Combined property data with cities, regions, and real data saved successfully.")

