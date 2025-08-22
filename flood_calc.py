import os
import pandas as pd
import numpy as np
import random

# Load the properties dataset
properties = pd.read_csv("C:/Users/katid_7ngm4sv/OneDrive/Desktop/augmscrip/updated_dataset_with_wildfire.csv")

# Weights for flood risk calculation
w1 = 0.6  # Weight for Rainfall Frequency
w2 = 0.4  # Weight for Rainfall Intensity

# Function to calculate flood risk
def calculate_flood_risk(Normalized_Rainfall_Frequency, Normalized_Rainfall_Intensity):
    return (w1 * Normalized_Rainfall_Frequency) + (w2 * Normalized_Rainfall_Intensity)

# Weather data folder
weather_data_folder = "C:/Users/katid_7ngm4sv/OneDrive/Desktop/weather_final"

# City name mapping
city_name_mapping = {
    "aghiosnikolaos": "Λασίθι",
    "alexandroupolis": "Αλεξανδρούπολη",
    "athens": "Αθήνα",
    "chania": "Χανιά",
    "drama": "Δράμα",
    "florina": "Φλώρινα",
    "heraclion": "Ηράκλειο",
    "ioannina": "Ιωάννινα",
    "ithaki": "Ιθάκη",
    "kilkis": "Κιλκίς",
    "larissa": "Λάρισα",
    "lefkada": "Λευκάδα",
    "lemnos": "Λήμνος",
    "livadeia": "Άμφισσα",
    "patra": "Πάτρα",
    "pirgos": "Πύργος",
    "polygyros": "Πολύγυρος",
    "rethymno": "Ρέθυμνο",
    "samos": "Σάμος",
    "sparti": "Σπάρτη",
    "thessaloniki": "Θεσσαλονίκη",
    "tinos": "Τήνος",
    "tripoli": "Τρίπολη",
    "trikala": "Τρίκαλα",
    "veroia": "Βέροια",
    "volos": "Βόλος",
    "xanthi": "Ξάνθη",
    "kerkyra":"Κέρκυρα"
}

# Region-to-City mapping
region_to_address = {
    "Πελοπόννησος": ["Πάτρα", "Αγρίνιο", "Σπάρτη", "Ναύπακτος", "Τρίπολη", "Πύργος"],
    "Κρήτη": ["Ηράκλειο", "Χανιά", "Ρέθυμνο", "Λασίθι"],
    "Ήπειρος": ["Ιωάννινα", "Άρτα", "Πρέβεζα", "Ηγουμενίτσα"],
    "Θεσσαλία": ["Λάρισα", "Βόλος", "Τρίκαλα", "Καρδίτσα"],
    "Μακεδονία": ["Κιλκίς", "Καβάλα", "Βέροια", "Κοζάνη", "Σέρρες", "Δράμα", "Φλώρινα", "Πολύγυρος"],
    "Ιόνια Νησιά": ["Κέρκυρα", "Λευκάδα", "Κεφαλονιά", "Ζάκυνθος", "Ιθάκη"],
    "Στερεά Ελλάδα": ["Λαμία", "Χαλκίδα", "Θήβα", "Άμφισσα"],
    "Θράκη": ["Αλεξανδρούπολη", "Κομοτηνή", "Ξάνθη"],
    "Αιγαίο": ["Μύκονος", "Σαντορίνη", "Πάρος", "Νάξος", "Τήνος", "Χίος", "Λήμνος", "Ικαρία", "Σκύρος", "Ρόδος", "Κως", "Σάμος", "Λέσβος"],
    "Αττική": [
        "Παλαιό Φάληρο", "Αχαρνές", "Παγκράτι", "Αθήνα", "Ίλιον", "Γαλάτσι", 
        "Δάφνη", "Χαϊδάρι", "Καλλιθέα", "Αργυρούπολη", "Περιστέρι", "Κρωπία", 
        "Νέα Ιωνία", "Μοσχάτο - Ταύρος", "Βάρη - Βούλα - Βουλιαγμένη", "Νίκαια", 
        "Πειραιάς", "Ζωγράφου", "Χαλάνδρι", "Ηλιούπολη", "Διόνυσος", "Άλιμος", 
        "Μαρούσι", "Νέα Σμύρνη", "Αιγάλεω", "Αγία Παρασκευή", "Γλυφάδα", "Σπάτα", 
        "Κερατσίνι-Δραπετσώνα", "Άγιοι Ανάργυροι", "Βριλήσσια", "Παιανία", 
        "Άγιος Δημήτριος", "Ραφήνα Πικέρμι", "Σαλαμίνα", "Λαυρεωτική", 
        "Λυκόβρυση Πεύκη", "Δάφνη-Υμηττός", "Ελληνικό - Αργυρούπολη", "Βύρωνας", 
        "Παλλήνη", "Κηφισιά", "Πετρούπολη", "Μέγαρα", "Παπάγου", "Πεντέλη", 
        "Χολαργός", "Αγιά Βαρβάρα", "Κορυδαλλός", "Πέραμα", "Φιλοθέη - Ψυχικό", 
        "Αίγινα", "Μάνδρα - Ειδυλλία", "Ασπρόπυργος", "Ελευσίνα", "Καισαριανή", 
        "Φυλή"
    ],
    "Θεσσαλονίκη": [
        "Λαγκάδας", "Θεσσαλονίκη", "Δήμος Θερμαϊκού", "Δήμος Εχεδώρου", 
        "Καλαμαριά", "Νεάπολη", "Επανομή", "Μυγδονία", "Θέρμη", "Μηχανιώνα", 
        "Άγιος Παύλος", "Ευκαρπία", "Δήμος Καλλινδοίων", "Άσσηρος", "Βασιλικα"
    ] 
}  

# Aggregate weather data
weather_data = []
for filename in os.listdir(weather_data_folder):
    if filename.endswith(".csv"):
        city_name = os.path.splitext(filename)[0]
        weather_df = pd.read_csv(os.path.join(weather_data_folder, filename))
        weather_df['city'] = city_name
        weather_data.append(weather_df)

# Combine all weather data
weather_data = pd.concat(weather_data, ignore_index=True)

# Normalize the city names in weather data to match location_name
weather_data['city'] = weather_data['city'].map(city_name_mapping).fillna(weather_data['city'])

# Convert 'Ac_R' to numeric (forcing errors to NaN for invalid entries)
weather_data['Ac_R'] = pd.to_numeric(weather_data['Ac_R'], errors='coerce')

# Calculate rainfall frequency and intensity for each city
weather_summary = weather_data.groupby('city').agg(
    Rainfall_Frequency=('Ac_R', lambda x: (x > 0).sum()),  # Non-zero rainfall days
    Rainfall_Intensity=('Ac_R', 'max')  # Maximum rainfall intensity
).reset_index()

# Normalize frequency and intensity
weather_summary['Normalized_Rainfall_Frequency'] = (weather_summary['Rainfall_Frequency'] - weather_summary['Rainfall_Frequency'].min()) / (weather_summary['Rainfall_Frequency'].max() - weather_summary['Rainfall_Frequency'].min())
weather_summary['Normalized_Rainfall_Intensity'] = (weather_summary['Rainfall_Intensity'] - weather_summary['Rainfall_Intensity'].min()) / (weather_summary['Rainfall_Intensity'].max() - weather_summary['Rainfall_Intensity'].min())

# Debug: Ensure normalization columns are created
assert 'Normalized_Rainfall_Frequency' in weather_summary.columns, "Normalized_Rainfall_Frequency column is missing in weather_summary."
assert 'Normalized_Rainfall_Intensity' in weather_summary.columns, "Normalized_Rainfall_Intensity column is missing in weather_summary."

# Merge weather data with properties dataset
properties = properties.merge(
    weather_summary[['city', 'Normalized_Rainfall_Frequency', 'Normalized_Rainfall_Intensity']], 
    how='left', left_on='location_name', right_on='city'
)
# Add Rainfall_Frequency and Rainfall_Intensity to properties if missing
if 'Rainfall_Frequency' not in properties.columns:
    properties['Rainfall_Frequency'] = weather_data.groupby('city')['Ac_R'].apply(lambda x: (x > 0).sum()).reindex(properties['location_name']).values

if 'Rainfall_Intensity' not in properties.columns:
    properties['Rainfall_Intensity'] = weather_data.groupby('city')['Ac_R'].max().reindex(properties['location_name']).values

# Check if the normalized columns are created
if 'Normalized_Rainfall_Frequency' not in properties.columns:
    # Create the Normalized_Rainfall_Frequency column manually
    properties['Normalized_Rainfall_Frequency'] = (properties['Rainfall_Frequency'] - properties['Rainfall_Frequency'].min()) / (properties['Rainfall_Frequency'].max() - properties['Rainfall_Frequency'].min())

if 'Normalized_Rainfall_Intensity' not in properties.columns:
    # Create the Normalized_Rainfall_Intensity column manually
    properties['Normalized_Rainfall_Intensity'] = (properties['Rainfall_Intensity'] - properties['Rainfall_Intensity'].min()) / (properties['Rainfall_Intensity'].max() - properties['Rainfall_Intensity'].min())

# Handle missing data for both raw and normalized rainfall values
global_fallback = weather_summary[['Rainfall_Frequency', 'Rainfall_Intensity', 
                                   'Normalized_Rainfall_Frequency', 'Normalized_Rainfall_Intensity']].mean()
# List of missing cities and regions
missing_cities = {
    "Πελοπόννησος": ["Αγρίνιο", "Ναύπακτος"],
    "Ήπειρος": ["Άρτα", "Πρέβεζα", "Ηγουμενίτσα"],
    "Θεσσαλία": ["Καρδίτσα"],
    "Μακεδονία": ["Καβάλα", "Κοζάνη", "Σέρρες", "Δράμα"],
    "Ιόνια Νησιά": ["Κεφαλονιά", "Ζάκυνθος"],
    "Στερεά Ελλάδα": ["Λαμία", "Χαλκίδα", "Θήβα"],
    "Θράκη": ["Κομοτηνή"],
    "Αιγαίο": ["Μύκονος", "Σαντορίνη", "Νάξος", "Χίος", "Ικαρία", "Σκύρος", "Πάρος", "Ρόδος", "Κως", "Λέσβος"],
    "Αττική": [
        "Παλαιό Φάληρο", "Αχαρνές", "Παγκράτι", "Ίλιον", "Γαλάτσι", 
        "Δάφνη", "Χαϊδάρι", "Καλλιθέα", "Αργυρούπολη", "Περιστέρι", "Κρωπία", 
        "Νέα Ιωνία", "Μοσχάτο - Ταύρος", "Βάρη - Βούλα - Βουλιαγμένη", "Νίκαια", 
        "Πειραιάς", "Ζωγράφου", "Χαλάνδρι", "Ηλιούπολη", "Διόνυσος", "Άλιμος", 
        "Μαρούσι", "Νέα Σμύρνη", "Αιγάλεω", "Αγία Παρασκευή", "Γλυφάδα", "Σπάτα", 
        "Κερατσίνι-Δραπετσώνα", "Άγιοι Ανάργυροι", "Βριλήσσια", "Παιανία", 
        "Άγιος Δημήτριος", "Ραφήνα Πικέρμι", "Σαλαμίνα", "Λαυρεωτική", 
        "Λυκόβρυση Πεύκη", "Δάφνη-Υμηττός", "Ελληνικό - Αργυρούπολη", "Βύρωνας", 
        "Παλλήνη", "Κηφισιά", "Πετρούπολη", "Μέγαρα", "Παπάγου", "Πεντέλη", 
        "Χολαργός", "Αγιά Βαρβάρα", "Κορυδαλλός", "Πέραμα", "Φιλοθέη - Ψυχικό", 
        "Αίγινα", "Μάνδρα - Ειδυλλία", "Ασπρόπυργος", "Ελευσίνα", "Καισαριανή", 
        "Φυλή"
    ],
    "Θεσσαλονίκη": [
        "Λαγκάδας", "Δήμος Θερμαϊκού", "Δήμος Εχεδώρου", 
        "Καλαμαριά", "Νεάπολη", "Επανομή", "Μυγδονία", "Θέρμη", "Μηχανιώνα", 
        "Άγιος Παύλος", "Ευκαρπία", "Δήμος Καλλινδοίων", "Άσσηρος", "Βασιλικα"
    ] 
}

for region, cities in missing_cities.items():
    for city in cities:
        # Get all cities in the region with weather data
        region_cities = region_to_address.get(region, [])
        region_weather_data = weather_summary[weather_summary['city'].isin(region_cities)]
        
        if not region_weather_data.empty:
            # Iterate through region cities until valid data is found
            valid_city_found = False
            for region_city in region_cities:
                random_city_data = region_weather_data[region_weather_data['city'] == region_city]
                if not random_city_data.empty:
                    # Valid city data found
                    rainfall_frequency = random_city_data['Rainfall_Frequency'].values[0]
                    rainfall_intensity = random_city_data['Rainfall_Intensity'].values[0]
                    normalized_rainfall_frequency = random_city_data['Normalized_Rainfall_Frequency'].values[0]
                    normalized_rainfall_intensity = random_city_data['Normalized_Rainfall_Intensity'].values[0]
                    
                    noise_factor = 0.02
                    rainfall_frequency += np.random.normal(0, noise_factor * rainfall_frequency)
                    rainfall_intensity += np.random.normal(0, noise_factor * rainfall_intensity)
                    normalized_rainfall_frequency += np.random.normal(0, noise_factor * normalized_rainfall_frequency)
                    normalized_rainfall_intensity += np.random.normal(0, noise_factor * normalized_rainfall_intensity)

                    # Convert rainfall frequency to an integer
                    rainfall_frequency = int(round(rainfall_frequency))

                    # Assign values to the missing city
                    properties.loc[properties['location_name'] == city, 'Rainfall_Frequency'] = rainfall_frequency
                    properties.loc[properties['location_name'] == city, 'Rainfall_Intensity'] = rainfall_intensity
                    properties.loc[properties['location_name'] == city, 'Normalized_Rainfall_Frequency'] = normalized_rainfall_frequency
                    properties.loc[properties['location_name'] == city, 'Normalized_Rainfall_Intensity'] = normalized_rainfall_intensity
                    valid_city_found = True
                    break  # Stop searching once valid data is found
            
            
            if not valid_city_found:
                properties.loc[properties['location_name'] == city, 'Rainfall_Frequency'] = global_fallback['Rainfall_Frequency']
                properties.loc[properties['location_name'] == city, 'Rainfall_Intensity'] = global_fallback['Rainfall_Intensity']
                properties.loc[properties['location_name'] == city, 'Normalized_Rainfall_Frequency'] = global_fallback['Normalized_Rainfall_Frequency']
                properties.loc[properties['location_name'] == city, 'Normalized_Rainfall_Intensity'] = global_fallback['Normalized_Rainfall_Intensity']
        else:
            
            properties.loc[properties['location_name'] == city, 'Rainfall_Frequency'] = global_fallback['Rainfall_Frequency']
            properties.loc[properties['location_name'] == city, 'Rainfall_Intensity'] = global_fallback['Rainfall_Intensity']
            properties.loc[properties['location_name'] == city, 'Normalized_Rainfall_Frequency'] = global_fallback['Normalized_Rainfall_Frequency']
            properties.loc[properties['location_name'] == city, 'Normalized_Rainfall_Intensity'] = global_fallback['Normalized_Rainfall_Intensity']



# Drop the existing flood_risk and flood_damage columns
properties.drop(columns=['flood_risk', 'flood_damage', 'res_address', 'city'], inplace=True, errors='ignore')

# Calculate flood risk (ensure calculate_flood_risk is well-defined)
properties['Flood_Risk'] = calculate_flood_risk(
    properties['Normalized_Rainfall_Frequency'], properties['Normalized_Rainfall_Intensity']
)

# Calculate flood damage as price
properties['Flood_Damage'] = properties['Flood_Risk'] * properties['res_price']

# Save the updated dataset
properties.to_csv("C:/Users/katid_7ngm4sv/OneDrive/Desktop/augmscrip/updated_dataset_with_flood.csv", index=False)
print("Updated dataset with flood risk (normalized) and damage (as price) saved.")