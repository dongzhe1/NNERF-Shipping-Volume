import os
import numpy as np
import pandas as pd
from tqdm import tqdm

input_dir = "../data/ballast/p_intro/"
output_dir = "../data/ballast/p_port/"

os.makedirs(output_dir, exist_ok=True)

combined_output_file = f"{output_dir}port_risk.csv"

all_results = []

for scenario in range(0, 9):
    input_file = f"{input_dir}Ballast_env_Eco_s{scenario}.csv"

    print(f"Processing scenario {scenario}...")

    data = pd.read_csv(input_file, sep=" ", header=None, names=["source", "destination", "probability"])

    data["source"] = data["source"].astype(int)
    data["destination"] = data["destination"].astype(int)

    grouped_data = data.groupby(["source", "destination"])

    scenario_results = []

    for (source, dest), group in tqdm(grouped_data):
        probabilities = group["probability"].values

        product_term = np.prod(1.0 - probabilities)

        p_ij_baseline = 1.0 - product_term

        scenario_results.append([scenario, source, dest, p_ij_baseline])

    all_results.extend(scenario_results)

    print(f"Completed scenario {scenario}.")

combined_df = pd.DataFrame(all_results, columns=["Scenario", "Source", "Destination", "p_ij_baseline"])
combined_df = combined_df.sort_values(by=["Scenario", "Source", "Destination"])

combined_df.to_csv(combined_output_file, sep=",", index=False)

print(f"All scenarios processed. Combined results saved to {combined_output_file}")