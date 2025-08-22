import os
import pandas as pd
import numpy as np


main_dataset = pd.read_csv("C:/Users/katid_7ngm4sv/OneDrive/Desktop/augmented_data_with_updated_risks.csv")  

# Step 1: Define mappings
nomos_to_city = {
    "ΑΙΤΩΛΟΑΚΑΡΝΑΝΙΑΣ": ["Αγρίνιο", "Ναύπακτος"],
    "ΑΡΚΑΔΙΑΣ": ["Τρίπολη"],
    "ΑΧΑΙΑΣ": ["Πάτρα"],
    "ΗΛΕΙΑΣ": ["Πύργος"],
    "ΛΑΚΩΝΙΑΣ": ["Σπάρτη"],
    "ΑΡΤΑΣ": ["Άρτα"],
    "ΘΕΣΠΡΩΤΙΑΣ": ["Ηγουμενίτσα"],
    "ΙΩΑΝΝΙΝΩΝ": ["Ιωάννινα"],
    "ΠΡΕΒΕΖΗΣ": ["Πρέβεζα"],
    "ΗΡΑΚΛΕΙΟΥ": ["Ηράκλειο"],
    "ΧΑΝΙΩΝ": ["Χανιά"],
    "ΡΕΘΥΜΝΟΥ": ["Ρέθυμνο"],
    "ΛΑΣΙΘΙΟΥ": ["Λασίθι"],
    "ΛΑΡΙΣΑΣ": ["Λάρισα"],
    "ΜΑΓΝΗΣΙΑΣ": ["Βόλος"],
    "ΤΡΙΚΑΛΩΝ": ["Τρίκαλα"],
    "ΚΑΡΔΙΤΣΑΣ": ["Καρδίτσα"],
    "ΘΕΣΣΑΛΟΝΙΚΗΣ": ["Θεσσαλονίκη"],
    "ΚΙΛΚΙΣ": ["Κιλκίς"],
    "ΚΑΒΑΛΑΣ": ["Καβάλα"],
    "ΗΜΑΘΙΑΣ": ["Βέροια"],
    "ΚΟΖΑΝΗΣ": ["Κοζάνη"],
    "ΣΕΡΡΩΝ": ["Σέρρες"],
    "ΔΡΑΜΑΣ": ["Δράμα"],
    "ΦΛΩΡΙΝΑΣ": ["Φλώρινα"],
    "ΧΑΛΚΙΔΙΚΗΣ": ["Πολύγυρος"],
    "ΚΕΡΚΥΡΑΣ": ["Κέρκυρα"],
    "ΛΕΥΚΑΔΟΣ": ["Λευκάδα"],
    "ΚΕΦΑΛΛΟΝΙΑΣ": ["Κεφαλονιά"],
    "ΖΑΚΥΝΘΟΥ": ["Ζάκυνθος"],
    "ΦΘΙΩΤΙΔΑΣ": ["Λαμία"],
    "ΕΥΒΟΙΑΣ": ["Χαλκίδα"],
    "ΒΟΙΩΤΙΑΣ": ["Θήβα"],
    "ΦΩΚΙΔΟΣ": ["Άμφισσα"],
    "ΡΟΔΟΠΗΣ": ["Κομοτηνή"],
    "ΕΒΡΟΥ": ["Αλεξανδρούπολη"],
    "ΞΑΝΘΗΣ": ["Ξάνθη"],
    "ΚΥΚΛΑΔΩΝ": ["Μύκονος", "Σαντορίνη", "Πάρος", "Νάξος", "Τήνος"],
    "ΧΙΟΥ": ["Χίος"],
    "ΛΕΣΒΟΥ": ["Λέσβος"],
    "ΛΗΜΝΟΥ": ["Λήμνος"],
    "ΙΚΑΡΙΑΣ": ["Ικαρία"],
    "ΣΑΜΟΥ": ["Σάμος"],
    "ΔΩΔΕΚΑΝΗΣΩΝ": ["Ρόδος", "Κως"],
    "ΣΚΥΡΟΥ": ["Σκύρος"]
}


nomos_to_region = {
    "ΑΙΤΩΛΟΑΚΑΡΝΑΝΙΑΣ": "Πελοπόννησος",
    "ΑΡΚΑΔΙΑΣ": "Πελοπόννησος",
    "ΑΧΑΙΑΣ": "Πελοπόννησος",
    "ΗΛΕΙΑΣ": "Πελοπόννησος",
    "ΛΑΚΩΝΙΑΣ": "Πελοπόννησος",
    "ΑΡΤΑΣ": "Ήπειρος",
    "ΘΕΣΠΡΩΤΙΑΣ": "Ήπειρος",
    "ΙΩΑΝΝΙΝΩΝ": "Ήπειρος",
    "ΠΡΕΒΕΖΗΣ": "Ήπειρος",
    "ΗΡΑΚΛΕΙΟΥ": "Κρήτη",
    "ΧΑΝΙΩΝ": "Κρήτη",
    "ΡΕΘΥΜΝΟΥ": "Κρήτη",
    "ΛΑΣΙΘΙΟΥ": "Κρήτη",
    "ΛΑΡΙΣΑΣ": "Θεσσαλία",
    "ΜΑΓΝΗΣΙΑΣ": "Θεσσαλία",
    "ΤΡΙΚΑΛΩΝ": "Θεσσαλία",
    "ΚΑΡΔΙΤΣΑΣ": "Θεσσαλία",
    "ΘΕΣΣΑΛΟΝΙΚΗΣ": "Μακεδονία",
    "ΚΙΛΚΙΣ": "Μακεδονία",
    "ΚΑΒΑΛΑΣ": "Μακεδονία",
    "ΗΜΑΘΙΑΣ": "Μακεδονία",
    "ΚΟΖΑΝΗΣ": "Μακεδονία",
    "ΣΕΡΡΩΝ": "Μακεδονία",
    "ΔΡΑΜΑΣ": "Μακεδονία",
    "ΦΛΩΡΙΝΑΣ": "Μακεδονία",
    "ΧΑΛΚΙΔΙΚΗΣ": "Μακεδονία",
    "ΚΕΡΚΥΡΑΣ": "Ιόνια Νησιά",
    "ΛΕΥΚΑΔΟΣ": "Ιόνια Νησιά",
    "ΚΕΦΑΛΛΟΝΙΑΣ": "Ιόνια Νησιά",
    "ΖΑΚΥΝΘΟΥ": "Ιόνια Νησιά",
    "ΦΘΙΩΤΙΔΑΣ": "Στερεά Ελλάδα",
    "ΕΥΒΟΙΑΣ": "Στερεά Ελλάδα",
    "ΒΟΙΩΤΙΑΣ": "Στερεά Ελλάδα",
    "ΦΩΚΙΔΟΣ": "Στερεά Ελλάδα",
    "ΡΟΔΟΠΗΣ": "Θράκη",
    "ΕΒΡΟΥ": "Θράκη",
    "ΞΑΝΘΗΣ": "Θράκη",
    "ΚΥΚΛΑΔΩΝ": "Αιγαίο",
    "ΧΙΟΥ": "Αιγαίο",
    "ΛΕΣΒΟΥ": "Αιγαίο",
    "ΛΗΜΝΟΥ": "Αιγαίο",
    "ΙΚΑΡΙΑΣ": "Αιγαίο",
    "ΣΑΜΟΥ": "Αιγαίο",
    "ΔΩΔΕΚΑΝΗΣΩΝ": "Αιγαίο",
    "ΣΚΥΡΟΥ": "Αιγαίο"
}


# Step 2: Load wildfire data
wildfire_folder = "C:/Users/katid_7ngm4sv/OneDrive/Desktop/wildfire"  
wildfire_files = [f for f in os.listdir(wildfire_folder) if f.endswith(".xls")]

wildfire_data = pd.DataFrame()


for file in wildfire_files:
    file_path = os.path.join(wildfire_folder, file)
    df = pd.read_excel(file_path)
    wildfire_data = pd.concat([wildfire_data, df], ignore_index=True)

# Preprocess wildfire data
wildfire_data['Ημερ/νία Έναρξης'] = pd.to_datetime(wildfire_data['Ημερ/νία Έναρξης'], errors='coerce')
wildfire_data['Ημερ/νία Κατασβεσης'] = pd.to_datetime(wildfire_data['Ημερ/νία Κατασβεσης'], errors='coerce')
wildfire_data['Fire_Duration'] = (wildfire_data['Ημερ/νία Κατασβεσης'] - wildfire_data['Ημερ/νία Έναρξης']).dt.total_seconds() / 3600

# Step 3: Match Νομός to Cities
wildfire_data['Matched_City'] = wildfire_data['Νομός'].apply(
    lambda nomos: next((city for city in nomos_to_city.get(nomos, []) if city in main_dataset['location_name'].values), None)
)

# Step 4: Match Remaining Νομός to Regions
wildfire_data['Matched_Region'] = wildfire_data['Νομός'].apply(
    lambda nomos: nomos_to_region.get(nomos) if nomos_to_region.get(nomos) in main_dataset['location_region'].values else None
)

# Step 5: Filter unmatched data
wildfire_data = wildfire_data.dropna(subset=['Matched_City', 'Matched_Region'], how='all')

# Add a unified match column
wildfire_data['Match'] = wildfire_data['Matched_City'].combine_first(wildfire_data['Matched_Region'])

# Step 6: Aggregate wildfire data by Match (city or region)
wildfire_agg = wildfire_data.groupby('Match').agg(
    Frequency=('Ημερ/νία Έναρξης', 'count'),
    Avg_Fire_Duration=('Fire_Duration', 'mean')
).reset_index()

# Normalize wildfire metrics
wildfire_agg['Normalized_Frequency'] = wildfire_agg['Frequency'] / wildfire_agg['Frequency'].max()
wildfire_agg['Normalized_Duration'] = wildfire_agg['Avg_Fire_Duration'] / wildfire_agg['Avg_Fire_Duration'].max()

# Step 7: Merge with main dataset
main_dataset['Match'] = main_dataset['location_name'].combine_first(main_dataset['location_region'])
main_dataset = main_dataset.merge(wildfire_agg, on='Match', how='left')

# Fill missing values for regions without wildfire data
main_dataset[['Frequency', 'Avg_Fire_Duration', 'Normalized_Frequency', 'Normalized_Duration']] = main_dataset[['Frequency', 'Avg_Fire_Duration', 'Normalized_Frequency', 'Normalized_Duration']].fillna(0)

# Step 8: Calculate wildfire risk and damage
ALPHA = 0.5
BETA = 0.3
GAMMA = 0.2

main_dataset['wildfire_risk'] = (
    ALPHA * main_dataset['Normalized_Frequency'] +
    BETA * main_dataset['Normalized_Duration'] +
    GAMMA * (main_dataset['HIGH_TEMP'] / main_dataset['HIGH_TEMP'].max() + main_dataset['AVG_WIND_SPEED'] / main_dataset['AVG_WIND_SPEED'].max()) / 2
)

main_dataset['wildfire_damage'] = main_dataset['wildfire_risk'] * main_dataset['res_price']

# Drop old columns and save
main_dataset = main_dataset.drop(columns=['heat_wave_risk', 'heatwave_damage'], errors='ignore')


main_dataset.to_csv("updated_dataset_with_wildfire.csv", index=False)

print("Updated dataset with wildfire risk and damage saved.")
