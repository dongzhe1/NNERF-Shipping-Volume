# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import json
import torch.nn.functional as F

AllType = "All"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.isnan(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def engineer_features(df):
    required_cols = [
        'OGDP', 'DGDP', 'OPOP', 'DPOP',
        'OCentrality', 'DCentrality',
        'contig', 'comlang_off', 'comcol', 'col45', 'fta_wto'
    ]

    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing one or more required columns for feature engineering.")
        engineered_cols = [
            'GDPRatio', 'POPRatio', 'OGDPPerCapita', 'DGDPPerCapita',
            'GDPProduct', 'POPProduct', 'CentralityDiff', 'CentralityProduct'
        ]
        for col in engineered_cols:
            if col not in df.columns:
                 df[col] = np.nan

    numeric_bases = ['OGDP', 'DGDP', 'OPOP', 'DPOP',
                     'OCentrality', 'DCentrality']
    for col in numeric_bases:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['GDPRatio'] = np.nan
    mask = df['OGDP'].notna() & df['DGDP'].notna() & (df['DGDP'] != 0)
    if 'OGDP' in df.columns and 'DGDP' in df.columns:
        df.loc[mask, 'GDPRatio'] = df.loc[mask, 'OGDP'] / df.loc[mask, 'DGDP']

    df['POPRatio'] = np.nan
    mask = df['OPOP'].notna() & df['DPOP'].notna() & (df['DPOP'] != 0)
    if 'OPOP' in df.columns and 'DPOP' in df.columns:
        df.loc[mask, 'POPRatio'] = df.loc[mask, 'OPOP'] / df.loc[mask, 'DPOP']

    df['OGDPPerCapita'] = np.nan
    mask = df['OGDP'].notna() & df['OPOP'].notna() & (df['OPOP'] != 0)
    if 'OGDP' in df.columns and 'OPOP' in df.columns:
        df.loc[mask, 'OGDPPerCapita'] = df.loc[mask, 'OGDP'] / df.loc[mask, 'OPOP']

    df['DGDPPerCapita'] = np.nan
    mask = df['DGDP'].notna() & df['DPOP'].notna() & (df['DPOP'] != 0)
    if 'DGDP' in df.columns and 'DPOP' in df.columns:
        df.loc[mask, 'DGDPPerCapita'] = df.loc[mask, 'DGDP'] / df.loc[mask, 'DPOP']

    df['GDPProduct'] = np.nan
    mask = df['OGDP'].notna() & df['DGDP'].notna()
    if 'OGDP' in df.columns and 'DGDP' in df.columns:
       df.loc[mask, 'GDPProduct'] = df.loc[mask, 'OGDP'] * df.loc[mask, 'DGDP']

    df['POPProduct'] = np.nan
    mask = df['OPOP'].notna() & df['DPOP'].notna()
    if 'OPOP' in df.columns and 'DPOP' in df.columns:
        df.loc[mask, 'POPProduct'] = df.loc[mask, 'OPOP'] * df.loc[mask, 'DPOP']

    df['CentralityDiff'] = np.nan
    mask = df['OCentrality'].notna() & df['DCentrality'].notna()
    if 'OCentrality' in df.columns and 'DCentrality' in df.columns:
        df.loc[mask, 'CentralityDiff'] = df.loc[mask, 'OCentrality'] - df.loc[mask, 'DCentrality']

    df['CentralityProduct'] = np.nan
    mask = df['OCentrality'].notna() & df['DCentrality'].notna()
    if 'OCentrality' in df.columns and 'DCentrality' in df.columns:
       df.loc[mask, 'CentralityProduct'] = df.loc[mask, 'OCentrality'] * df.loc[mask, 'DCentrality']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

class EmbeddingModel(nn.Module):
    def __init__(self, num_countries, embedding_dim, num_numeric_features,
                 hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super().__init__()

        self.country_embedding = nn.Embedding(num_countries, embedding_dim // 2)
        input_size = num_numeric_features + embedding_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ])

        self.has_residual = input_size == hidden_dims[0]
        if not self.has_residual:
            self.residual_proj = nn.Linear(input_size, hidden_dims[0])

        self.dropout = nn.Dropout(dropout_rate)
        self.linear_extrapolation = nn.Linear(num_numeric_features, 1)
        self.extrapolation_weight = nn.Parameter(torch.tensor([0.5]))
        self.feature_extraction_layer = nn.Linear(hidden_dims[-1], embedding_dim)

    def forward(self, numeric_features, origin_idx, dest_idx, return_embeddings=False):
        origin_emb = self.country_embedding(origin_idx)
        dest_emb = self.country_embedding(dest_idx)
        x = torch.cat((numeric_features, origin_emb, dest_emb), dim=1)
        original_x = x

        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            if i == 0:
                x = layer(x)
                if self.has_residual:
                    x = x + original_x
                else:
                    x = x + self.residual_proj(original_x)
            else:
                residual = x
                x = layer(x)
                if x.shape == residual.shape:
                    x = x + residual

            x = bn(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)

        if return_embeddings:
            rf_features = self.feature_extraction_layer(x)
            return rf_features

        linear_pred = self.linear_extrapolation(numeric_features)
        nn_pred = self.output_layer(x).squeeze(-1)
        nn_pred = torch.clamp(nn_pred, min=-10.0, max=None)
        combined_pred = 0.1 * nn_pred + 0.9 * linear_pred.squeeze(-1)
        final_pred = F.softplus(combined_pred, beta=1.0)

        return final_pred

    def get_embeddings(self, numeric_features, origin_idx, dest_idx):
        with torch.no_grad():
            origin_emb = self.country_embedding(origin_idx)
            dest_emb = self.country_embedding(dest_idx)
            x = torch.cat((numeric_features, origin_emb, dest_emb), dim=1)

            for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
                x = layer(x)
                x = bn(x)
                x = F.leaky_relu(x, negative_slope=0.1)

            rf_features = self.feature_extraction_layer(x)
            return torch.cat([rf_features, numeric_features], dim=1)

def combine_feature_importance(rf_importance_df, numeric_feature_cols, nn_model, device='cpu'):
    nn_learned_features = [col for col in rf_importance_df['Feature'] if 'nn_learned_feature_' in col]
    direct_features = [col for col in rf_importance_df['Feature'] if 'nn_learned_feature_' not in col]

    nn_learned_importance_total = rf_importance_df[rf_importance_df['Feature'].isin(nn_learned_features)][
        'Importance'].sum()

    direct_importance_df = rf_importance_df[rf_importance_df['Feature'].isin(direct_features)].copy()

    country_identity_ratio = 0.6
    feature_via_embedding_ratio = 1.0 - country_identity_ratio

    nn_model.to(device)
    nn_model.eval()

    num_features = len(numeric_feature_cols)
    unit_vectors = torch.eye(num_features).to(device)
    country_idx = torch.zeros(num_features, dtype=torch.long).to(device)

    feature_contributions = {}

    with torch.no_grad():
        if hasattr(nn_model, 'layers') and len(nn_model.layers) > 0:
            first_layer_weights = nn_model.layers[0].weight.data

            if first_layer_weights.shape[1] >= num_features:
                numeric_weights = first_layer_weights[:, :num_features]
                feature_importance_raw = torch.abs(numeric_weights).sum(dim=0).cpu().numpy()

                total_importance = np.sum(feature_importance_raw)
                if total_importance > 0:
                    feature_importance_norm = feature_importance_raw / total_importance

                    for i, feat in enumerate(numeric_feature_cols):
                        if i < len(feature_importance_norm):
                            feature_contributions[feat] = feature_importance_norm[
                                                              i] * nn_learned_importance_total * feature_via_embedding_ratio

    country_importance = nn_learned_importance_total * country_identity_ratio
    origin_country_importance = country_importance / 2
    dest_country_importance = country_importance / 2

    redistributed_df = pd.DataFrame({
        'Feature': list(feature_contributions.keys()) + ['OCountry_Identity', 'DCountry_Identity'],
        'NN_Attributed_Importance': list(feature_contributions.values()) + [origin_country_importance,
                                                                            dest_country_importance]
    })

    combined_df = pd.merge(direct_importance_df, redistributed_df, on='Feature', how='outer').fillna(0)
    combined_df['Final_Importance'] = combined_df['Importance'] + combined_df['NN_Attributed_Importance']

    result_df = combined_df[['Feature', 'Final_Importance']].copy()
    result_df.columns = ['Feature', 'Importance']
    result_df = result_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    return result_df

def combine_feature_importance_with_feature_extraction_analysis(rf_importance_df, numeric_feature_cols, nn_model,
                                                                device='cpu'):
    nn_learned_features = [col for col in rf_importance_df['Feature'] if 'nn_learned_feature_' in col]
    direct_features = [col for col in rf_importance_df['Feature'] if 'nn_learned_feature_' not in col]

    nn_learned_importance_total = rf_importance_df[rf_importance_df['Feature'].isin(nn_learned_features)][
        'Importance'].sum()

    direct_importance_df = rf_importance_df[rf_importance_df['Feature'].isin(direct_features)].copy()

    country_identity_ratio = 0.6
    feature_via_embedding_ratio = 1.0 - country_identity_ratio

    nn_model.to(device)
    nn_model.eval()

    feature_contributions = {}

    with torch.no_grad():
        if hasattr(nn_model, 'feature_extraction_layer') and hasattr(nn_model, 'layers'):
            last_hidden_weights = nn_model.layers[-1].weight.data
            feature_extraction_weights = nn_model.feature_extraction_layer.weight.data

            combined_weight_impact = torch.abs(feature_extraction_weights).mean(dim=0)

            first_layer_weights = nn_model.layers[0].weight.data
            if first_layer_weights.shape[1] >= len(numeric_feature_cols):
                numeric_weights = first_layer_weights[:, :len(numeric_feature_cols)]
                feature_importance_raw = torch.abs(numeric_weights).sum(dim=0).cpu().numpy()

                total_importance = np.sum(feature_importance_raw)
                if total_importance > 0:
                    feature_importance_norm = feature_importance_raw / total_importance

                    for i, feat in enumerate(numeric_feature_cols):
                        if i < len(feature_importance_norm):
                            feature_contributions[feat] = feature_importance_norm[
                                                              i] * nn_learned_importance_total * feature_via_embedding_ratio

    country_importance = nn_learned_importance_total * country_identity_ratio
    origin_country_importance = country_importance / 2
    dest_country_importance = country_importance / 2

    redistributed_df = pd.DataFrame({
        'Feature': list(feature_contributions.keys()) + ['OCountry_Identity', 'DCountry_Identity'],
        'NN_Attributed_Importance': list(feature_contributions.values()) + [origin_country_importance,
                                                                            dest_country_importance]
    })

    combined_df = pd.merge(direct_importance_df, redistributed_df, on='Feature', how='outer').fillna(0)
    combined_df['Final_Importance'] = combined_df['Importance'] + combined_df['NN_Attributed_Importance']

    result_df = combined_df[['Feature', 'Final_Importance']].copy()
    result_df.columns = ['Feature', 'Importance']
    result_df = result_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    return result_df

def combine_feature_importance_advanced(rf_importance_df, numeric_feature_cols, nn_model, device='cpu',
                                        num_samples=100, country_identity_ratio=0.6):
    nn_learned_features = [col for col in rf_importance_df['Feature'] if 'nn_learned_feature_' in col]
    direct_features = [col for col in rf_importance_df['Feature'] if 'nn_learned_feature_' not in col]

    nn_learned_importance_total = rf_importance_df[rf_importance_df['Feature'].isin(nn_learned_features)][
        'Importance'].sum()

    direct_importance_df = rf_importance_df[rf_importance_df['Feature'].isin(direct_features)].copy()

    feature_via_embedding_ratio = 1.0 - country_identity_ratio

    nn_model.to(device)
    nn_model.eval()

    feature_sensitivities = {}

    with torch.no_grad():
        random_samples = torch.randn(num_samples, len(numeric_feature_cols)).to(device)
        random_origin = torch.randint(0, 50, (num_samples,)).to(device)
        random_dest = torch.randint(0, 50, (num_samples,)).to(device)

        embeddings = nn_model.get_embeddings(random_samples, random_origin, random_dest)

        sensitivities = []
        for i in range(len(numeric_feature_cols)):
            perturbed_samples = random_samples.clone()
            perturbed_samples[:, i] += 1.0

            perturbed_embeddings = nn_model.get_embeddings(perturbed_samples, random_origin, random_dest)
            embedding_diff = torch.norm(perturbed_embeddings - embeddings, dim=1).mean().item()
            sensitivities.append(embedding_diff)

        total_sensitivity = sum(sensitivities)
        if total_sensitivity > 0:
            normalized_sensitivities = [s / total_sensitivity for s in sensitivities]

            for i, feat in enumerate(numeric_feature_cols):
                feature_sensitivities[feat] = normalized_sensitivities[
                                                  i] * nn_learned_importance_total * feature_via_embedding_ratio

    country_importance = nn_learned_importance_total * country_identity_ratio

    origin_sensitivity = 0
    dest_sensitivity = 0

    with torch.no_grad():
        feature_samples = random_samples[:20]
        base_origin = torch.ones(20, dtype=torch.long).to(device) * 0
        base_dest = torch.ones(20, dtype=torch.long).to(device) * 0
        base_embeddings = nn_model.get_embeddings(feature_samples, base_origin, base_dest)

        for country in range(1, 10):
            test_origin = torch.ones(20, dtype=torch.long).to(device) * country
            origin_embeddings = nn_model.get_embeddings(feature_samples, test_origin, base_dest)
            origin_diff = torch.norm(origin_embeddings - base_embeddings, dim=1).mean().item()
            origin_sensitivity += origin_diff

            test_dest = torch.ones(20, dtype=torch.long).to(device) * country
            dest_embeddings = nn_model.get_embeddings(feature_samples, base_origin, test_dest)
            dest_diff = torch.norm(dest_embeddings - base_embeddings, dim=1).mean().item()
            dest_sensitivity += dest_diff

    total_country_sensitivity = origin_sensitivity + dest_sensitivity
    if total_country_sensitivity > 0:
        origin_country_importance = (origin_sensitivity / total_country_sensitivity) * country_importance
        dest_country_importance = (dest_sensitivity / total_country_sensitivity) * country_importance
    else:
        origin_country_importance = country_importance / 2
        dest_country_importance = country_importance / 2

    redistributed_df = pd.DataFrame({
        'Feature': list(feature_sensitivities.keys()) + ['OCountry_Identity', 'DCountry_Identity'],
        'NN_Attributed_Importance': list(feature_sensitivities.values()) + [origin_country_importance,
                                                                            dest_country_importance]
    })

    combined_df = pd.merge(direct_importance_df, redistributed_df, on='Feature', how='outer').fillna(0)
    combined_df['Final_Importance'] = combined_df['Importance'] + combined_df['NN_Attributed_Importance']

    result_df = combined_df[['Feature', 'Final_Importance']].copy()
    result_df.columns = ['Feature', 'Importance']
    result_df = result_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    return result_df