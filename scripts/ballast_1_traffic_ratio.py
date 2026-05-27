import pandas as pd
import os

MACRO_PI_RATE = 0.105

os.makedirs('../data/ballast/predict', exist_ok=True)

print("Reading country mapping...")
country_map = pd.read_csv('../data/predict/country.csv')
country_dict = dict(zip(country_map['REGION_NAME'], country_map['COUNTRY_CODE']))

print("Reading prediction data...")
try:
    merged_data = pd.read_csv('../out/merged_predictions.csv')
except FileNotFoundError:
    print("File out/merged_predictions.csv not found.")
    exit(1)

print(f"Total rows in merged data: {len(merged_data)}")

merged_data['OCountry_Code'] = merged_data['OCountry'].map(country_dict)
merged_data['DCountry_Code'] = merged_data['DCountry'].map(country_dict)

unmapped_o = merged_data[merged_data['OCountry_Code'].isna()]['OCountry'].unique()
unmapped_d = merged_data[merged_data['DCountry_Code'].isna()]['DCountry'].unique()
if len(unmapped_o) > 0 or len(unmapped_d) > 0:
    print(f"Warning: {len(unmapped_o)} origin and {len(unmapped_d)} destination countries could not be mapped")

merged_data = merged_data.dropna(subset=['OCountry_Code', 'DCountry_Code'])
print(f"Rows after filtering unmapped countries: {len(merged_data)}")

print("Creating baseline index...")
baseline_data = merged_data[merged_data['Year'] == 2018].copy()
print(f"Baseline data rows: {len(baseline_data)}")

merged_data['baseline_key'] = merged_data['OCountry'] + '|' + merged_data['DCountry'] + '|' + merged_data[
    'VesselType'] + '|' + merged_data['SSP'].astype(str)
baseline_data['baseline_key'] = baseline_data['OCountry'] + '|' + baseline_data['DCountry'] + '|' + baseline_data[
    'VesselType'] + '|' + baseline_data['SSP'].astype(str)

baseline_dict = dict(zip(baseline_data['baseline_key'], baseline_data['RF']))
print(f"Unique baseline combinations: {len(baseline_dict)}")

print("Calculating ratios with 10.5% macro uncertainty...")
merged_data['baseline_RF'] = merged_data['baseline_key'].map(baseline_dict)

merged_data = merged_data.dropna(subset=['baseline_RF'])
merged_data = merged_data[merged_data['baseline_RF'] > 0]
print(f"Rows with valid baseline values: {len(merged_data)}")

merged_data['Ratio'] = merged_data['RF'] / merged_data['baseline_RF']
merged_data['Ratio_Lower'] = merged_data['Ratio'] * (1 - MACRO_PI_RATE)
merged_data['Ratio_Upper'] = merged_data['Ratio'] * (1 + MACRO_PI_RATE)

result_df = merged_data[['OCountry_Code', 'DCountry_Code', 'SSP', 'VesselType', 'Year', 'Ratio', 'Ratio_Lower', 'Ratio_Upper']].copy()
result_df = result_df.rename(columns={'OCountry_Code': 'OCountry', 'DCountry_Code': 'DCountry'})

print(f"Writing {len(result_df)} rows to output file...")
output_file = '../data/ballast/predict/all_ratios.csv'
result_df.to_csv(output_file, index=False)
print(f"Ratio data saved to {output_file}")

print("Processing complete!")