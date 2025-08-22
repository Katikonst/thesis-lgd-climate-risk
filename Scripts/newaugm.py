import pandas as pd
import numpy as np

# Load your augmented dataset
augmented_data = pd.read_csv('C:/Users/katid_7ngm4sv/OneDrive/Desktop/augmscrip/updated_properties_with_earthquake_risk.csv')

# ---------------------------
# Step 1: Region-Specific Temperature Data+Add Wind and Rain Data
# ---------------------------
location_temperature_data = {
    "Λασίθι": {"mean_temp": 20.21, "max_temp": 42.30, "min_temp": 1.80},
    "Αττική": {"mean_temp": 19.35, "max_temp": 42.00, "min_temp": -1.10},
    "Χανιά": {"mean_temp": 19.03, "max_temp": 43.20, "min_temp": 0.70},
    "Ηράκλειο": {"mean_temp": 19.30, "max_temp": 40.80, "min_temp": 0.50},
    "Ιωάννινα": {"mean_temp": 13.57, "max_temp": 41.40, "min_temp": -13.60},
    "Ιθάκη": {"mean_temp": 19.11, "max_temp": 41.40, "min_temp": -1.20},
    "Κέρκυρα": {"mean_temp": 17.99, "max_temp": 39.70, "min_temp": -2.60},
    "Κιλκίς": {"mean_temp": 15.10, "max_temp": 41.60, "min_temp": -12.30},
    "Λάρισα": {"mean_temp": 17.43, "max_temp": 45.50, "min_temp": -12.70},
    "Λευκάδα": {"mean_temp": 18.58, "max_temp": 39.20, "min_temp": -1.90},
    "Λήμνος": {"mean_temp": 17.08, "max_temp": 40.80, "min_temp": -4.60},
    "Πρέβεζα": {"mean_temp": 16.61, "max_temp": 43.90, "min_temp": -12.80},
    "Πάρος": {"mean_temp": 19.02, "max_temp": 37.30, "min_temp": 0.30},
    "Πάτρα": {"mean_temp": 19.14, "max_temp": 38.30, "min_temp": -2.20},
    "Ρέθυμνο": {"mean_temp": 19.65, "max_temp": 40.20, "min_temp": 0.40},
    "Σάμος": {"mean_temp": 18.26, "max_temp": 38.40, "min_temp": -1.20},
    "Σπάρτη": {"mean_temp": 17.32, "max_temp": 45.70, "min_temp": -5.30},
    "Θεσσαλονίκη": {"mean_temp": 17.16, "max_temp": 39.70, "min_temp": -8.40},
    "Τρίκαλα": {"mean_temp": 16.76, "max_temp": 44.00, "min_temp": -10.70},
    "Βέροια": {"mean_temp": 16.18, "max_temp": 42.10, "min_temp": -13.90},
    "Βόλος": {"mean_temp": 18.29, "max_temp": 44.40, "min_temp": -6.10},
    "Αλεξανδρούπολη": {"mean_temp": 15.42, "max_temp": 40.10, "min_temp": -11.70},
    "Ξάνθη": {"mean_temp": 16.03, "max_temp": 40.90, "min_temp": -10.30},
    "Τρίπολη": {"mean_temp": 13.37, "max_temp": 42.80, "min_temp": -12.50},
    "Τήνος": {"mean_temp": 19.02, "max_temp": 39.90, "min_temp": 0.30},
    "Πολύγυρος": {"mean_temp": 14.44, "max_temp": 38.20, "min_temp": -14.70},
    "Πύργος": {"mean_temp": 19.17, "max_temp": 42.70, "min_temp": -2.10},
    "Φλώρινα": {"mean_temp": 12.02, "max_temp": 39.00, "min_temp": -3.20},
    "Δράμα": {"mean_temp": 14.71, "max_temp": 40.10, "min_temp": -13.80},

}

# Calculate median values for temperatures
median_max_temp = np.median([temp["max_temp"] for temp in location_temperature_data.values()])
median_min_temp = np.median([temp["min_temp"] for temp in location_temperature_data.values()])
median_mean_temp = np.median([temp["mean_temp"] for temp in location_temperature_data.values()])

# Define regional temperature averages (for missing data)
region_temperature_averages = {
    "Πελοπόννησος": {"mean_temp": 17.5, "max_temp": 42.0, "min_temp": -3.0},
    "Κρήτη": {"mean_temp": 18.5, "max_temp": 42.0, "min_temp": 0.0},
    "Ήπειρος": {"mean_temp": 13.5, "max_temp": 41.0, "min_temp": -10.0},
    "Θεσσαλία": {"mean_temp": 16.5, "max_temp": 44.0, "min_temp": -8.0},
    "Μακεδονία": {"mean_temp": 15.0, "max_temp": 42.0, "min_temp": -10.0},
    "Ιόνια Νησιά": {"mean_temp": 18.0, "max_temp": 40.0, "min_temp": -1.0},
    "Στερεά Ελλάδα": {"mean_temp": 17.0, "max_temp": 43.0, "min_temp": -5.0},
    "Αιγαίο": {"mean_temp": 19.0, "max_temp": 40.0, "min_temp": 1.0},
    "Θράκη": {"mean_temp": 15.5, "max_temp": 40.0, "min_temp": -11.0}
}

region_to_address = {
    "Πελοπόννησος": ["Πάτρα", "Αγρίνιο", "Σπάρτη", "Ναύπακτος", "Τρίπολη", "Πύργος"],
    "Κρήτη": ["Ηράκλειο", "Χανιά", "Ρέθυμνο", "Λασίθι"],
    "Ήπειρος": ["Ιωάννινα", "Άρτα", "Πρέβεζα", "Ηγουμενίτσα"],
    "Θεσσαλία": ["Λάρισα", "Βόλος", "Τρίκαλα", "Καρδίτσα"],
    "Μακεδονία": ["Κιλκίς", "Καβάλα", "Βέροια", "Κοζάνη", "Σέρρες", "Δράμα", "Φλώρινα", "Πολύγυρος"],
    "Ιόνια Νησιά": ["Κέρκυρα", "Λευκάδα", "Κεφαλονιά", "Ζάκυνθος", "Ιθάκη"],
    "Στερεά Ελλάδα": ["Λαμία", "Χαλκίδα", "Θήβα", "Άμφισσα"],
    "Θράκη": ["Αλεξανδρούπολη", "Κομοτηνή", "Ξάνθη"],
    "Αιγαίο": ["Μύκονος", "Σαντορίνη", "Πάρος", "Νάξος", "Τήνος", "Χίος", "Λήμνος", "Ικαρία", "Σκύρος", "Ρόδος", "Κως", "Σάμος", "Λέσβος"]
}

def adjust_temperatures_by_location(data, noise_factor=0.1):
    high_temps, low_temps, mean_temps = [], [], []
    for location in data['location_name']:
        if location in location_temperature_data:
            temp_data = location_temperature_data[location]
        else:
            region = next(
                (reg for reg, cities in region_to_address.items() if location in cities), None)
            if region and region in region_temperature_averages:
                temp_data = region_temperature_averages[region]
            else:
                temp_data = {"mean_temp": median_mean_temp,
                             "max_temp": median_max_temp,
                             "min_temp": median_min_temp}

        high_temp_with_noise = temp_data['max_temp'] + np.random.normal(0, noise_factor)
        low_temp_with_noise = temp_data['min_temp'] + np.random.normal(0, noise_factor)
        mean_temp_with_noise = temp_data['mean_temp'] + np.random.normal(0, noise_factor)
        
        high_temps.append(high_temp_with_noise)
        low_temps.append(low_temp_with_noise)
        mean_temps.append(mean_temp_with_noise)

    data['HIGH_TEMP'] = high_temps
    data['LOW_TEMP'] = low_temps
    data['MEAN_TEMP'] = mean_temps
    return data

augmented_data = adjust_temperatures_by_location(augmented_data, noise_factor=0.1)
location_wind_data = {
    "Λασίθι": {"max_wind_speed": 15.60, "mean_wind_speed": 4.32},
    "Αττική": {"max_wind_speed": 15.60, "mean_wind_speed": 5.36},
    "Χανιά": {"max_wind_speed": 25.10, "mean_wind_speed": 7.19},
    "Ηράκλειο": {"max_wind_speed": 19.40, "mean_wind_speed": 5.50},
    "Ιωάννινα": {"max_wind_speed": 14.00, "mean_wind_speed": 4.30},
    "Ιθάκη": {"max_wind_speed": 23.00, "mean_wind_speed": 6.50},
    "Κέρκυρα": {"max_wind_speed": 27.30, "mean_wind_speed": 8.00},
    "Κιλκίς": {"max_wind_speed": 17.50, "mean_wind_speed": 5.00},
    "Λάρισα": {"max_wind_speed": 18.00, "mean_wind_speed": 5.20},
    "Λευκάδα": {"max_wind_speed": 22.50, "mean_wind_speed": 7.00},
    "Λήμνος": {"max_wind_speed": 23.40, "mean_wind_speed": 6.80},
    "Πρέβεζα": {"max_wind_speed": 21.30, "mean_wind_speed": 6.30},
    "Πάρος": {"max_wind_speed": 28.70, "mean_wind_speed": 9.10},
    "Πάτρα": {"max_wind_speed": 24.80, "mean_wind_speed": 7.20},
    "Ρέθυμνο": {"max_wind_speed": 26.10, "mean_wind_speed": 8.40},
    "Σάμος": {"max_wind_speed": 20.90, "mean_wind_speed": 6.00},
    "Σπάρτη": {"max_wind_speed": 18.80, "mean_wind_speed": 5.10},
    "Θεσσαλονίκη": {"max_wind_speed": 21.60, "mean_wind_speed": 6.60},
    "Τρίκαλα": {"max_wind_speed": 16.50, "mean_wind_speed": 4.70},
    "Βέροια": {"max_wind_speed": 20.00, "mean_wind_speed": 5.70},
    "Αλεξανδρούπολη": {"max_wind_speed": 27.60, "mean_wind_speed": 5.80},
    "Ξάνθη": {"max_wind_speed": 18.40, "mean_wind_speed": 4.80},
    "Τρίπολη": {"max_wind_speed": 11.60, "mean_wind_speed": 4.00},
    "Τήνος": {"max_wind_speed": 31.80, "mean_wind_speed": 6.10},
    "Πολύγυρος": {"max_wind_speed": 21.90, "mean_wind_speed": 5.30},
    "Πύργος": {"max_wind_speed": 13.10, "mean_wind_speed": 4.10},
    "Φλώρινα": {"max_wind_speed": 13.30, "mean_wind_speed": 4.20},
    "Δράμα": {"max_wind_speed": 13.30, "mean_wind_speed": 4.20},

}
region_rainfall_data = {
        'Λασίθι': 1.25, 'Αττική': 1.14, 'Χανιά': 1.66, 'Ηράκλειο': 1.37, 
        'Ιωάννινα': 3.75, 'Ιθάκη': 2.72, 'Κέρκυρα': 3.08, 'Κιλκίς': 1.69, 
        'Λάρισα': 1.19, 'Λευκάδα': 3.26, 'Λήμνος': 1.34, 'Πρέβεζα': 1.87, 
        'Πάρος': 0.96, 'Πάτρα': 1.91, 'Ρέθυμνο': 1.43, 'Σάμος': 2.23, 
        'Σπάρτη': 1.84, 'Θεσσαλονίκη': 1.18, 'Τρίκαλα': 2.00, 'Βέροια': 1.87, 
        'Βόλος': 1.81, 'Αλεξανδρούπολη': 1.55, 'Ξάνθη': 1.76, 'Τρίπολη': 1.98, 'Τήνος': 0.92, 'Πολύγυρος': 1.79, 'Πύργος': 2.46, 'Φλώρινα': 1.58, 'Δράμα': 1.91,
    }

# Function to adjust rainfall data and add noise
def adjust_rainfall_by_location(data, noise_factor=0.1):
    rainfall_values = []
    for location in data['location_name']:
        if location in region_rainfall_data:
            rainfall = region_rainfall_data[location]
        else:
            region = next(
                (reg for reg, cities in region_to_address.items() if location in cities), None)
            if region and region in region_rainfall_data:
                rainfall = region_rainfall_data[region]
            else:
                # Default to 1.5mm rainfall if no data is available
                rainfall = 1.5

        # Add noise to the rainfall values
        rainfall_with_noise = rainfall + np.random.normal(0, noise_factor)
        rainfall_values.append(rainfall_with_noise)

    data['RAIN'] = rainfall_values
    return data



def adjust_wind_speeds_by_location(data, noise_factor=0.1):
    max_wind_speeds, mean_wind_speeds = [], []
    for location in data['location_name']:
        if location in location_wind_data:
            wind_data = location_wind_data[location]
        else:
            region = next(
                (reg for reg, cities in region_to_address.items() if location in cities), None)
            if region and region in region_temperature_averages:
                wind_data = location_wind_data.get(region, {"max_wind_speed": 20.0, "mean_wind_speed": 5.0})
            else:
                wind_data = {"max_wind_speed": 20.0, "mean_wind_speed": 5.0}

        # Add noise to wind speeds
        max_wind_speed_with_noise = wind_data['max_wind_speed'] + np.random.normal(0, noise_factor)
        mean_wind_speed_with_noise = wind_data['mean_wind_speed'] + np.random.normal(0, noise_factor)

        max_wind_speeds.append(max_wind_speed_with_noise)
        mean_wind_speeds.append(mean_wind_speed_with_noise)

    data['HIGHEST_WIND_SPEED'] = max_wind_speeds
    data['AVG_WIND_SPEED'] = mean_wind_speeds
    return data

augmented_data = adjust_wind_speeds_by_location(augmented_data)
augmented_data =adjust_rainfall_by_location(augmented_data)

# ---------------------------
# Step 6: Remove Old Risk Columns (heatwave and storm risk)
# ---------------------------
augmented_data.drop(columns=['storm_risk', 'storm_damage', 'total_damage'], inplace=True)

# Save the updated dataset
augmented_data.to_csv('C:/Users/katid_7ngm4sv/OneDrive/Desktop/augmented_data_with_updated_risks.csv', index=False)

print("Final normalized dataset saved as 'C:/Users/katid_7ngm4sv/OneDrive/Desktop/augmented_data_with_updated_risks.csv'")