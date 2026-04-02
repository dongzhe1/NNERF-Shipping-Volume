# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import joblib
import os
from scipy.stats import t
import json

from common import EmbeddingModel, engineer_features

def calculate_sardain_pi(data_path='data/voyages_grouped_country.csv', model_dir='model/All'):
    print(f"Loading historical data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return

    print(f"Loading model artifacts from {model_dir}...")
    try:
        config_path = os.path.join(model_dir, 'training_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        country_to_idx = joblib.load(os.path.join(model_dir, 'country_to_idx.pkl'))
        imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        rf_model = joblib.load(os.path.join(model_dir, 'random_forest_with_embeddings.pkl'))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_model = EmbeddingModel(
            config['n_countries'],
            config['embedding_dim'],
            config['num_numeric_features_nn_input']
        ).to(device)
        nn_model.load_state_dict(torch.load(os.path.join(model_dir, 'embedding_model.pth'), map_location=device))
        nn_model.eval()
    except Exception as e:
        print(f"Failed to load artifacts: {e}")
        return

    print("Preprocessing historical data for inference...")
    df = df.dropna(subset=['RouteCount']).copy()

    df = engineer_features(df)

    df['OCountry'] = df['OCountry'].astype(str).fillna('Unknown')
    df['DCountry'] = df['DCountry'].astype(str).fillna('Unknown')
    df['OCountry_idx'] = df['OCountry'].map(country_to_idx).fillna(0).astype(int)
    df['DCountry_idx'] = df['DCountry'].map(country_to_idx).fillna(0).astype(int)

    numeric_cols = config['numeric_feature_cols_order']
    X_numeric = df[numeric_cols].copy()
    X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

    X_imputed = pd.DataFrame(imputer.transform(X_numeric), columns=numeric_cols)
    X_scaled = scaler.transform(X_imputed)

    print("Running inference on historical data...")
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        o_idx_tensor = torch.tensor(df['OCountry_idx'].values, dtype=torch.long).to(device)
        d_idx_tensor = torch.tensor(df['DCountry_idx'].values, dtype=torch.long).to(device)

        embeddings = nn_model.get_embeddings(X_tensor, o_idx_tensor, d_idx_tensor).cpu().numpy()

    predictions = rf_model.predict(embeddings)
    df['RF_Prediction'] = np.maximum(0, predictions)

    print("\n--- Aggregating by Year ---")
    yearly_agg = df.groupby('Year').agg(
        Actual_Total=('RouteCount', 'sum'),
        Predicted_Total=('RF_Prediction', 'sum')
    ).reset_index()

    yearly_agg['Relative_Error'] = (yearly_agg['Actual_Total'] - yearly_agg['Predicted_Total']) / yearly_agg['Predicted_Total']

    print(yearly_agg.to_string(index=False))

    print("\n--- Calculating Sardain's Macro Prediction Interval ---")
    n = len(yearly_agg)
    if n < 2:
        print("Not enough years to calculate standard deviation.")
        return

    s = np.std(yearly_agg['Relative_Error'], ddof=1)
    t_val = t.ppf(0.975, df=n-1)
    pi_rate = s * t_val * np.sqrt(1 + 1.0/n)

    print(f"Number of historical years (n) : {n}")
    print(f"Standard deviation of error (s): {s:.4f}")
    print(f"Critical t-value (df={n-1})      : {t_val:.4f}")
    print(f"--------------------------------------------------")
    print(f"CALCULATED MACRO PI RATE       : {pi_rate:.4f} (approx {pi_rate*100:.2f}%)")
    print(f"--------------------------------------------------")

if __name__ == "__main__":
    calculate_sardain_pi()