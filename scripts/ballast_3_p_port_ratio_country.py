import os
import pandas as pd
import gc

port_baseline_file = "../data/ballast/p_port/port_risk.csv"
growth_ratio_file = "../data/ballast/predict/all_ratios.csv"
port_country_mapping_file = "../data/raw/places.csv"
port_output_file = "../data/ballast/p_port_ratio/port_risk_ratio.csv"
country_output_file = "../data/ballast/p_country/ballast_country_aggregated.csv"

os.makedirs(os.path.dirname(port_output_file), exist_ok=True)
os.makedirs(os.path.dirname(country_output_file), exist_ok=True)

print("Loading port to country mapping...")
port_country = pd.read_csv(port_country_mapping_file)
port_to_country = dict(zip(port_country['D_Id'], port_country['D_Country']))

print("Loading growth ratio data...")
ratio_df = pd.read_csv(growth_ratio_file, dtype={
    'OCountry': 'category',
    'DCountry': 'category',
    'SSP': 'category',
    'VesselType': 'category',
    'Year': 'category',
    'Ratio': 'float32',
    'Ratio_Lower': 'float32',
    'Ratio_Upper': 'float32'
})

unique_ssps = ratio_df['SSP'].unique()
unique_years = ratio_df['Year'].unique()
unique_vessel_types = ratio_df['VesselType'].unique()

ratio_df.set_index(['OCountry', 'DCountry', 'SSP', 'VesselType', 'Year'], inplace=True)
print("Building ratio lookup dictionary...")
ratio_dict = {
    (row['OCountry'], row['DCountry'], row['SSP'], row['VesselType'], row['Year']):
        (row['Ratio'], row['Ratio_Lower'], row['Ratio_Upper'])
    for _, row in ratio_df.reset_index().iterrows()
}

print(f"Processing data with {len(unique_ssps)} SSPs, {len(unique_years)} Years, and {len(unique_vessel_types)} Vessel Types...")

gc.collect()

print("Loading port baseline data...")
baseline_data = pd.read_csv(port_baseline_file, dtype={
    'Scenario': 'category',
    'Source': 'object',
    'Destination': 'object',
    'p_ij_baseline': 'float32'
})

target_scenarios = [0, 3, 4, 5]
print(f"Filtering baseline data for target scenarios: {target_scenarios}...")
# astype(int) ensures safe matching whether the CSV loaded them as strings or ints
baseline_data = baseline_data[baseline_data['Scenario'].astype(int).isin(target_scenarios)].copy()
print(f"Filtered baseline data to {len(baseline_data)} records.")

print("Adding country mappings to baseline data...")
baseline_data['SourceCountry'] = baseline_data['Source'].map(lambda x: port_to_country.get(int(x)))
baseline_data['DestinationCountry'] = baseline_data['Destination'].map(lambda x: port_to_country.get(int(x)))

chunk_size = 10000
port_file_initialized = False

def process_chunk(chunk_df, chunk_id, file_initialized):
    all_combinations = []

    for _, row in chunk_df.iterrows():
        for ssp in unique_ssps:
            for year in unique_years:
                for vessel_type in unique_vessel_types:
                    key = (row['SourceCountry'], row['DestinationCountry'], ssp, vessel_type, year)
                    if key in ratio_dict:
                        r_mean, r_lower, r_upper = ratio_dict[key]
                        all_combinations.append({
                            'Scenario': row['Scenario'],
                            'Source': row['Source'],
                            'Destination': row['Destination'],
                            'SourceCountry': row['SourceCountry'],
                            'DestinationCountry': row['DestinationCountry'],
                            'p_ij_baseline': row['p_ij_baseline'],
                            'SSP': ssp,
                            'Year': year,
                            'VesselType': vessel_type,
                            'growth_factor': r_mean,
                            'growth_factor_lower': r_lower,
                            'growth_factor_upper': r_upper
                        })

    if not all_combinations:
        return file_initialized, pd.DataFrame()

    combinations_df = pd.DataFrame(all_combinations)

    baseline_probs = combinations_df['p_ij_baseline'].values

    g_factors = combinations_df['growth_factor'].values
    combinations_df['prob'] = 1 - (1 - baseline_probs) ** g_factors

    g_factors_low = combinations_df['growth_factor_lower'].values
    combinations_df['prob_lower'] = 1 - (1 - baseline_probs) ** g_factors_low

    g_factors_up = combinations_df['growth_factor_upper'].values
    combinations_df['prob_upper'] = 1 - (1 - baseline_probs) ** g_factors_up

    result_df = combinations_df[[
        'Scenario', 'Source', 'Destination', 'SourceCountry', 'DestinationCountry',
        'SSP', 'Year', 'VesselType', 'prob', 'prob_lower', 'prob_upper'
    ]]

    mode = 'a' if file_initialized else 'w'
    header = not file_initialized
    result_df.to_csv(port_output_file, mode=mode, header=header, index=False)

    return True, result_df[['Scenario', 'SourceCountry', 'DestinationCountry', 'SSP', 'Year', 'VesselType', 'prob', 'prob_lower', 'prob_upper']]

print("Processing baseline data in chunks...")
all_country_dfs = []

for scenario, scenario_data in baseline_data.groupby('Scenario'):
    print(f"Processing scenario {scenario}...")

    for i in range(0, len(scenario_data), chunk_size):
        chunk = scenario_data.iloc[i:i + chunk_size].copy()

        port_file_initialized, port_results = process_chunk(chunk, f"{scenario}_{i // chunk_size + 1}",
                                                            port_file_initialized)

        if not port_results.empty:
            country_cols = ['Scenario', 'SourceCountry', 'DestinationCountry', 'SSP', 'Year', 'VesselType']
            country_agg = port_results.groupby(country_cols)[['prob', 'prob_lower', 'prob_upper']].mean().reset_index()
            all_country_dfs.append(country_agg)

        del port_results
        gc.collect()

print("Combining all country-level data...")
combined_country_df = pd.concat(all_country_dfs, ignore_index=True)
del all_country_dfs
gc.collect()

print("Final country-level aggregation...")
country_cols = ['Scenario', 'SourceCountry', 'DestinationCountry', 'SSP', 'Year', 'VesselType']
final_country_df = combined_country_df.groupby(country_cols)[['prob', 'prob_lower', 'prob_upper']].mean().reset_index()

max_prob_value = final_country_df['prob'].max()
print(f"Maximum probability value for normalization: {max_prob_value}")

final_country_df['prob'] = final_country_df['prob'] / max_prob_value
final_country_df['prob_lower'] = final_country_df['prob_lower'] / max_prob_value
final_country_df['prob_upper'] = final_country_df['prob_upper'] / max_prob_value

print(f"Saving country-level results to {country_output_file}...")
final_country_df.to_csv(country_output_file, index=False)

print("All processing completed successfully!")