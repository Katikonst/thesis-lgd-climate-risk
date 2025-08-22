import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for clarity
sns.set(style="whitegrid", context="talk")

results_folder = r"path/to/Model results"
scenarios = ["Baseline", "High Flood Risk", "High Wildfire Risk", "High Earthquake Risk", "Flood + Wildfire"]
years = range(2025, 2051)

# Function to annotate final value for each region
def annotate_final_values(ax, df, measure, year):
    # Get the final year values by group
    final_vals = df[df["Year"] == year].groupby("location_region")[measure].mean().reset_index()
    # For each region, annotate near the last point
    for _, row in final_vals.iterrows():
        region = row["location_region"]
        val = row[measure]
        # Find the x-coordinate (year) and y-coordinate (val) for this region in the plot data
        ax.text(year + 0.2, val, f"{val:.2f}", fontsize=10, color='black', 
                verticalalignment='center')

# ================================
# Generate Graphs from Regional CSVs (not maps)
# ================================
for scenario in scenarios:
    regional_file = os.path.join(results_folder, f"{scenario}_regional_results.csv")
    if not os.path.exists(regional_file):
        print(f"Regional file for scenario '{scenario}' not found. Skipping scenario.")
        continue

    regional_summary = pd.read_csv(regional_file)
    
  
    if "location_region" in regional_summary.columns:
        regional_summary["location_region"] = regional_summary["location_region"].str.upper()

    # --- Plot 1: Relative LGD Change (compared to 2025) ---
    plt.figure(figsize=(12, 8))
    ax1 = sns.lineplot(
        data=regional_summary,
        x="Year",
        y="LGD_change",
        hue="location_region",
        marker="o",
        dashes=False,
        palette="tab10"
    )
    plt.title(f"Regional LGD Change (Relative to 2025) - {scenario}", fontsize=20)
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("LGD Change (Percentage Points)", fontsize=16)
    plt.axhline(0, color="grey", linestyle="--", linewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Region", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc=2)
    annotate_final_values(ax1, regional_summary, "LGD_change", max(years))
    plt.tight_layout()
    out_path1 = os.path.join(results_folder, f"{scenario}_LGD_change.png")
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved visualization for Relative LGD Change in scenario '{scenario}' to {out_path1}")

    # --- Plot 2: Absolute LGD Values ---
    plt.figure(figsize=(12, 8))
    ax2 = sns.lineplot(
        data=regional_summary,
        x="Year",
        y="LGD",
        hue="location_region",
        marker="o",
        dashes=False,
        palette="tab10"
    )
    plt.title(f"Regional Absolute LGD - {scenario}", fontsize=20)
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Absolute LGD", fontsize=16)
    plt.axhline(0, color="grey", linestyle="--", linewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Region", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc=2)
    annotate_final_values(ax2, regional_summary, "LGD", max(years))
    plt.tight_layout()
    out_path2 = os.path.join(results_folder, f"{scenario}_Absolute_LGD.png")
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved visualization for Absolute LGD in scenario '{scenario}' to {out_path2}")

    # --- Plot 3: Regional VtL Change Over Time ---
    plt.figure(figsize=(12, 8))
    ax3 = sns.lineplot(
        data=regional_summary,
        x="Year",
        y="VtL_change",
        hue="location_region",
        marker="o",
        dashes=False,
        palette="tab10"
    )
    plt.title(f"Regional VtL Change Over Time - {scenario}", fontsize=20)
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("VtL Change", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Region", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc=2)
    annotate_final_values(ax3, regional_summary, "VtL_change", max(years))
    plt.tight_layout()
    out_path3 = os.path.join(results_folder, f"{scenario}_VtL_change.png")
    plt.savefig(out_path3, dpi=300)
    plt.close()
    print(f"Saved VtL change visualization in scenario '{scenario}' to {out_path3}")

    # --- Plot 4: Regional Adjusted Flood Risk Over Time ---
    if "adj_Flood_Risk" in regional_summary.columns:
        plt.figure(figsize=(12, 8))
        ax4 = sns.lineplot(
            data=regional_summary,
            x="Year",
            y="adj_Flood_Risk",
            hue="location_region",
            marker="o",
            dashes=False,
            palette="tab10"
        )
        plt.title(f"Regional Adjusted Flood Risk Over Time - {scenario}", fontsize=20)
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("Adjusted Flood Risk", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title="Region", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc=2)
        annotate_final_values(ax4, regional_summary, "adj_Flood_Risk", max(years))
        plt.tight_layout()
        out_path4 = os.path.join(results_folder, f"{scenario}_Adj_Flood_Risk.png")
        plt.savefig(out_path4, dpi=300)
        plt.close()
        print(f"Saved Adjusted Flood Risk visualization in scenario '{scenario}' to {out_path4}")
    else:
        print(f"'adj_Flood_Risk' column not found in {scenario} regional data.")

    # --- Plot 5: Regional Adjusted Wildfire Risk Over Time ---
    if "adj_Wildfire_Risk" in regional_summary.columns:
        plt.figure(figsize=(12, 8))
        ax5 = sns.lineplot(
            data=regional_summary,
            x="Year",
            y="adj_Wildfire_Risk",
            hue="location_region",
            marker="o",
            dashes=False,
            palette="tab10"
        )
        plt.title(f"Regional Adjusted Wildfire Risk Over Time - {scenario}", fontsize=20)
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("Adjusted Wildfire Risk", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title="Region", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc=2)
        annotate_final_values(ax5, regional_summary, "adj_Wildfire_Risk", max(years))
        plt.tight_layout()
        out_path5 = os.path.join(results_folder, f"{scenario}_Adj_Wildfire_Risk.png")
        plt.savefig(out_path5, dpi=300)
        plt.close()
        print(f"Saved Adjusted Wildfire Risk visualization in scenario '{scenario}' to {out_path5}")
    else:
        print(f"'adj_Wildfire_Risk' column not found in {scenario} regional data.")
        
    # --- Plot 6: Regional Adjusted Earthquake Risk Over Time ---
    if "adj_Earthquake_Risk" in regional_summary.columns:
        plt.figure(figsize=(12, 8))
        ax6 = sns.lineplot(
            data=regional_summary,
            x="Year",
            y="adj_Earthquake_Risk",
            hue="location_region",
            marker="o",
            dashes=False,
            palette="tab10"
        )
        plt.title(f"Regional Adjusted Earthquake Risk Over Time - {scenario}", fontsize=20)
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("Adjusted Earthquake Risk", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title="Region", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc=2)
        annotate_final_values(ax6, regional_summary, "adj_Earthquake_Risk", max(years))
        plt.tight_layout()
        out_path6 = os.path.join(results_folder, f"{scenario}_Adj_Earthquake_Risk.png")
        plt.savefig(out_path6, dpi=300)
        plt.close()
        print(f"Saved Adjusted Earthquake Risk visualization in scenario '{scenario}' to {out_path6}")
    else:
        print(f"'adj_Earthquake_Risk' column not found in {scenario} regional data.")

print("All regional graphs saved successfully.")

