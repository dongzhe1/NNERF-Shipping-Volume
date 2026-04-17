import pandas as pd
import numpy as np
import os

RECORDS_PER_YEAR = 1000


def generate_fake_data():
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data/' directory")

    years = [2002, 2005, 2008, 2012, 2015, 2018]

    countries = [
        'USA', 'CAN', 'CHN', 'GBR', 'FRA', 'DEU', 'JPN', 'KOR', 'IND', 'BRA',
        'AUS', 'RUS', 'ZAF', 'MEX', 'ITA', 'ESP', 'NLD', 'SGP', 'VNM', 'TUR',
        'ARG', 'IDN', 'SAU', 'CHE', 'SWE', 'POL', 'BEL', 'THA', 'AUT', 'NOR',
        'ARE', 'ISR', 'IRL', 'PHL', 'MYS', 'DNK', 'CHL', 'COL', 'FIN', 'BGD',
        'EGY', 'PKR', 'KAZ', 'ROU', 'PER', 'CZE', 'GRC', 'QAT', 'PRT', 'NZL'
    ]

    vessel_types = ["Chemical", "Bulk", "Container", "Oil", "General",
                    "Liquified-Gas"]

    full_data = []
    vessel_data = []

    for year in years:
        usa_can_base = {
            'Year': year,
            'OCountry': 'USA',
            'DCountry': 'CAN',
            'OCentrality': 0.85,
            'DCentrality': 0.75,
            'OGDP': 1.8e13,
            'DGDP': 1.6e12,
            'OPOP': 3.3e8,
            'DPOP': 3.8e7,
            'contig': 1,
            'comlang_off': 1,
            'comcol': 0,
            'col45': 0,
            'fta_wto': 1,
            'Distance': 3500.0
        }

        usa_can_full = usa_can_base.copy()
        usa_can_full['RouteCount'] = 1500
        full_data.append(usa_can_full)

        for vt in vessel_types:
            v_row = usa_can_base.copy()
            v_row['VesselType'] = vt
            v_row['RouteCount'] = np.random.randint(100, 500)
            vessel_data.append(v_row)

        for _ in range(RECORDS_PER_YEAR - 1):
            o_c, d_c = np.random.choice(countries, 2, replace=False)

            base_row = {
                'Year': year,
                'OCountry': o_c,
                'DCountry': d_c,
                'OCentrality': np.random.uniform(0.01, 1.0),
                'DCentrality': np.random.uniform(0.01, 1.0),
                'OGDP': np.random.uniform(1e10, 2e13),
                'DGDP': np.random.uniform(1e10, 2e13),
                'OPOP': np.random.uniform(1e6, 1.4e9),
                'DPOP': np.random.uniform(1e6, 1.4e9),
                'contig': np.random.choice([0, 1], p=[0.9, 0.1]),
                'comlang_off': np.random.choice([0, 1]),
                'comcol': np.random.choice([0, 1], p=[0.9, 0.1]),
                'col45': np.random.choice([0, 1], p=[0.95, 0.05]),
                'fta_wto': np.random.choice([0, 1]),
                'Distance': np.random.uniform(500, 18000)
            }

            f_row = base_row.copy()
            f_row['RouteCount'] = np.random.randint(1, 4000)
            full_data.append(f_row)

            v_row = base_row.copy()
            v_row['VesselType'] = np.random.choice(vessel_types)
            v_row['RouteCount'] = np.random.randint(1, 1000)
            vessel_data.append(v_row)

    pd.DataFrame(full_data).to_csv('data/voyages_grouped_country.csv',
                                   index=False)
    pd.DataFrame(vessel_data).to_csv('data/voyages_grouped_country_vessel.csv',
                                     index=False)

    print(f"Demo data generated successfully!")
    print(f" - Countries: {len(countries)}")
    print(f" - Records per year: {RECORDS_PER_YEAR}")
    print(f" - Total records in full data: {len(full_data)}")
    print(f" - Total records in vessel data: {len(vessel_data)}")


if __name__ == "__main__":
    generate_fake_data()