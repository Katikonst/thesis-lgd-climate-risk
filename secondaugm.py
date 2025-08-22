import pandas as pd
import numpy as np

# Define the regions and corresponding cities
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

# Define latitude and longitude for cities
city_coordinates = {
    "Πάτρα": (38.2466, 21.7346), "Αγρίνιο": (38.6219, 21.4073), "Σπάρτη": (37.0731, 22.4298),
    "Ναύπακτος": (38.3927, 21.8275), "Τρίπολη": (37.5089, 22.3794), "Πύργος": (37.6750, 21.4419),
    "Ηράκλειο": (35.3387, 25.1442), "Χανιά": (35.5138, 24.0180), "Ρέθυμνο": (35.3647, 24.4822),
    "Λασίθι": (35.1517, 25.7159), "Ιωάννινα": (39.6676, 20.8509), "Άρτα": (39.1597, 20.9855),
    "Πρέβεζα": (38.9554, 20.7508), "Ηγουμενίτσα": (39.5034, 20.2622), "Λάρισα": (39.6413, 22.4170),
    "Βόλος": (39.3610, 22.9426), "Τρίκαλα": (39.5546, 21.7679), "Καρδίτσα": (39.3643, 21.9210),
    "Κιλκίς": (40.9930, 22.8714), "Καβάλα": (40.9396, 24.4065), "Βέροια": (40.5241, 22.2024),
    "Κοζάνη": (40.3003, 21.7889), "Σέρρες": (41.0850, 23.5480), "Δράμα": (41.1515, 24.1517),
    "Φλώρινα": (40.7814, 21.4093), "Κέρκυρα": (39.6243, 19.9217), "Λευκάδα": (38.8305, 20.7078),
    "Κεφαλονιά": (38.2003, 20.4572), "Ζάκυνθος": (37.7870, 20.8994), "Ιθάκη": (38.3676, 20.7205),
    "Λαμία": (38.8995, 22.4349), "Χαλκίδα": (38.4636, 23.5982), "Θήβα": (38.3256, 23.3186),
    "Άμφισσα": (38.5251, 22.3848), "Ρόδος": (36.4349, 28.2176), "Κως": (36.8920, 27.2877),
    "Σάμος": (37.7544, 26.9778), "Λέσβος": (39.1002, 26.5529), "Χίος": (38.3678, 26.1353),
    "Λήμνος": (39.8732, 25.0644), "Ικαρία": (37.5968, 26.1025), "Σκύρος": (38.9076, 24.5652), "Αλεξανδρούπολη": (40.8484, 25.8747),
    "Κομοτηνή": (41.1171, 25.4059), "Ξάνθη": (41.1347, 24.8837),
    "Μύκονος": (37.4467, 25.3289), "Σαντορίνη": (36.3932, 25.4615), "Πάρος": (37.0855, 25.1481),
    "Νάξος": (37.1043, 25.3775), "Τήνος": (37.5423, 25.1610), "Πολύγυρος": (40.3811, 23.4436)
}

# Load the dataset
combined_df = pd.read_csv('C:/Users/katid_7ngm4sv/OneDrive/Desktop/augmented_dataset.csv') 
# Step 1: Split the dataset
real_data = combined_df.iloc[:20000]  # First 20k rows (real data)
augmented_data = combined_df.iloc[20000:]  # Last 20k rows (augmented data)

# Step 2: Generate new location names, regions, and coordinates for the augmented dataset
new_location_names = np.random.choice(
    [city for cities in region_to_address.values() for city in cities], 
    len(augmented_data)
)
new_location_regions = [
    region for location_name in new_location_names
    for region, cities in region_to_address.items()
    if location_name in cities
]
new_coordinates = [city_coordinates[city] for city in new_location_names]

# Step 3: Update the augmented dataset
augmented_data.loc[:, "location_name"] = new_location_names
augmented_data.loc[:, "location_region"] = new_location_regions
augmented_data.loc[:, "LAT_DEG"] = [coord[0] for coord in new_coordinates]  # Update latitude
augmented_data.loc[:, "LONG_DEG"] = [coord[1] for coord in new_coordinates]  # Update longitude


# Step 4: Combine the datasets back
updated_combined_df = pd.concat([real_data, augmented_data], ignore_index=True)

# Save or inspect the updated dataset
updated_combined_df.to_csv("updated_dataset.csv", index=False)
print("Updated dataset saved as 'updated_dataset.csv'")
