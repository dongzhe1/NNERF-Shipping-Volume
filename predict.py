# -*- coding: utf-8 -*-
import joblib
import os
import types
from common import *
import time
import traceback

model_artifact_cache = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def patch_model_forward(model):
    original_forward = model.forward

    def fixed_forward(self, numeric_features, origin_idx, dest_idx, return_embeddings=False):
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

        linear_pred = self.linear_extrapolation(numeric_features).squeeze(-1)
        nn_pred = self.output_layer(x).squeeze(-1)

        nn_pred_clamped = torch.clamp(nn_pred, min=-10.0, max=None)
        combined_pred = 0.1 * nn_pred_clamped + 0.9 * linear_pred
        final_pred = F.softplus(combined_pred, beta=1.0)

        return final_pred

    model.forward = types.MethodType(fixed_forward, model)
    return model

def _load_artifacts(model_dir, verbose=False):
    if verbose: print(f"Executing _load_artifacts for: {model_dir}")
    artifacts = {}
    try:
        config_path = os.path.join(model_dir, 'training_config.json')
        with open(config_path, 'r') as f: config = json.load(f)
        artifacts['config'] = config;
        artifacts['country_to_idx'] = joblib.load(os.path.join(model_dir, 'country_to_idx.pkl'));
        artifacts['static_pair_features'] = joblib.load(os.path.join(model_dir, 'static_pair_features.pkl'));
        artifacts['imputer'] = joblib.load(os.path.join(model_dir, 'imputer.pkl'));
        artifacts['scaler'] = joblib.load(os.path.join(model_dir, 'scaler.pkl'));
        artifacts['rf_model'] = joblib.load(os.path.join(model_dir, 'random_forest_with_embeddings.pkl'));
        nn_model_path = os.path.join(model_dir, 'embedding_model.pth')
        n_countries = config['n_countries']; embedding_dim = config['embedding_dim']; num_numeric_features_nn_input = config['num_numeric_features_nn_input']
        nn_model = EmbeddingModel(n_countries, embedding_dim, num_numeric_features_nn_input).to(device)
        nn_model.load_state_dict(torch.load(nn_model_path, map_location=device))
        nn_model = patch_model_forward(nn_model)

        artifacts['nn_model'] = nn_model
        if verbose: print(f" All artifacts loaded successfully for {model_dir}.")
        return artifacts
    except FileNotFoundError as e:
        print(f"  Error loading artifacts: Missing file - {e.filename}")
        raise
    except Exception as e:
        print(f"  Error loading artifacts: {e}")
        raise

def _get_or_load_artifacts(model_dir, verbose=False):
    global model_artifact_cache
    if model_dir not in model_artifact_cache:
        if verbose: print(f"Cache miss for '{model_dir}'. Loading artifacts...")
        try:
            model_artifact_cache[model_dir] = _load_artifacts(model_dir, verbose=verbose)
            if verbose: print(f"Artifacts for '{model_dir}' loaded and cached.")
        except Exception as e:
            print(f"Failed to load artifacts for {model_dir}. Error: {e}")
            return None
    else:
        if verbose: print(f"Cache hit for '{model_dir}'. Using cached artifacts.")
    return model_artifact_cache.get(model_dir)

def predict_batch(batch_input_data, model_type=AllType, verbose=False):
    if verbose: print(f"\n--- Received batch prediction request for {len(batch_input_data)} items (Model Type: {model_type}) ---")
    batch_start_time = time.time()
    base_model_dir = 'model'
    results = []

    if model_type.lower() == AllType: model_subdir = AllType
    else: model_subdir = model_type.replace(" ", "_").replace("-", "_")
    model_dir = os.path.join(base_model_dir, model_subdir)

    artifacts = _get_or_load_artifacts(model_dir, verbose=verbose)

    if artifacts is None:
        error_msg = f"Failed to load artifacts for model type '{model_type}' from {model_dir}."
        print(error_msg)
        return [{'Error': error_msg} for _ in batch_input_data]

    if verbose: print(" Preparing features for the batch...")
    prep_start_time = time.time()
    try:
        config=artifacts['config']; country_to_idx=artifacts['country_to_idx']; static_pair_features_map=artifacts['static_pair_features']
        imputer=artifacts['imputer']; scaler=artifacts['scaler']; rf_model=artifacts['rf_model']; nn_model=artifacts['nn_model']
        numeric_feature_cols_order=config['numeric_feature_cols_order']; embedding_dim=config['embedding_dim']

        input_df = pd.DataFrame(batch_input_data)
        if input_df.empty: return []
        input_df['RF_Prediction'] = np.nan
        input_df['Error'] = None
        input_df['Processed'] = False

        def get_static(row): return static_pair_features_map.get((row['o_country'], row['d_country']), {})
        static_features_batch = input_df.apply(get_static, axis=1)
        static_features_df = pd.json_normalize(static_features_batch)
        input_df = pd.concat([input_df, static_features_df.set_index(input_df.index)], axis=1)

        loaded_static_cols = []
        if static_pair_features_map:
             first_key = next(iter(static_pair_features_map), None)
             if first_key:
                 loaded_static_cols = list(static_pair_features_map[first_key].keys())

        if loaded_static_cols:
             input_df['Static_OK'] = input_df[loaded_static_cols].notna().all(axis=1)
             input_df.loc[~input_df['Static_OK'], 'Error'] = 'O-D pair not found in static features map.'
        else:
             input_df['Static_OK'] = True

        input_df['o_idx'] = input_df['o_country'].map(country_to_idx)
        input_df['d_idx'] = input_df['d_country'].map(country_to_idx)
        input_df['Indices_OK'] = input_df['o_idx'].notna() & input_df['d_idx'].notna()
        input_df.loc[~input_df['Indices_OK'] & input_df['Error'].isna(), 'Error'] = 'Unknown O or D country name.'

        valid_rows_mask = input_df['Error'].isna()
        if not valid_rows_mask.any():
            if verbose: print(" No valid rows in the batch after initial checks.")
            return input_df[['o_country', 'd_country', 'origin_gdp', 'dest_gdp', 'origin_pop', 'dest_pop', 'RF_Prediction', 'Error']].to_dict('records')

        valid_input_df = input_df[valid_rows_mask].copy()
        rename_map = {'o_country': 'OCountry', 'd_country': 'DCountry','origin_gdp': 'OGDP', 'dest_gdp': 'DGDP','origin_pop': 'OPOP', 'dest_pop': 'DPOP'}
        valid_input_df.rename(columns=rename_map, inplace=True)
        valid_engineered_df = engineer_features(valid_input_df)

        try:
            X_numeric_batch = valid_engineered_df[numeric_feature_cols_order].copy()
        except KeyError as e:
            error_msg = f"Missing required numeric columns after engineering: {e}. Cannot proceed with batch."
            print(error_msg)
            input_df.loc[valid_rows_mask, 'Error'] = error_msg
            return input_df[['o_country', 'd_country', 'origin_gdp', 'dest_gdp', 'origin_pop', 'dest_pop', 'RF_Prediction', 'Error']].to_dict('records')

        X_numeric_batch.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_numeric_imputed_array = imputer.transform(X_numeric_batch)
        X_numeric_imputed_df = pd.DataFrame(X_numeric_imputed_array, columns=numeric_feature_cols_order, index=X_numeric_batch.index)
        X_numeric_scaled_array = scaler.transform(X_numeric_imputed_df)

        if verbose: print(f" Feature preparation took {time.time() - prep_start_time:.3f} seconds.")

        if verbose: print(" Performing batch inference...")
        inference_start_time = time.time()

        with torch.no_grad():
            X_numeric_tensor = torch.tensor(X_numeric_scaled_array, dtype=torch.float32).to(device)
            o_idx_batch_tensor = torch.tensor(valid_engineered_df['o_idx'].values, dtype=torch.long).to(device)
            d_idx_batch_tensor = torch.tensor(valid_engineered_df['d_idx'].values, dtype=torch.long).to(device)

        nn_model.eval()
        with torch.no_grad():
            embeddings_tensor = nn_model.get_embeddings(X_numeric_tensor, o_idx_batch_tensor, d_idx_batch_tensor)
            embeddings_numpy = embeddings_tensor.cpu().numpy()

            nn_preds_tensor = nn_model(X_numeric_tensor, o_idx_batch_tensor, d_idx_batch_tensor)

        rf_preds_batch = rf_model.predict(embeddings_numpy)
        input_df.loc[valid_rows_mask, 'RF_Prediction'] = rf_preds_batch
        residual_std = config.get('residual_std', 0)
        z_score = 1.96
        input_df.loc[valid_rows_mask, 'Traffic_Lower'] = rf_preds_batch - (
                    z_score * residual_std)
        input_df.loc[valid_rows_mask, 'Traffic_Upper'] = rf_preds_batch + (
                    z_score * residual_std)
        input_df.loc[input_df['Traffic_Lower'] < 0, 'Traffic_Lower'] = 0

        input_df.loc[valid_rows_mask, 'Processed'] = True
        if verbose: print(f" Batch inference took {time.time() - inference_start_time:.3f} seconds.")

    except Exception as e:
        error_msg = f"Error during batch feature prep or inference: {e}"
        print(error_msg)
        traceback.print_exc()
        input_df.loc[input_df['Error'].isna(), 'Error'] = error_msg

    output_cols = ['o_country', 'd_country', 'origin_gdp', 'dest_gdp', 'origin_pop', 'dest_pop', 'RF_Prediction', 'Traffic_Lower', 'Traffic_Upper', 'NN_Prediction', 'Error']
    for col in output_cols:
        if col not in input_df.columns: input_df[col] = None if col != 'Error' else 'Processing Error'
    results = input_df[output_cols].to_dict('records')

    batch_end_time = time.time()
    if verbose: print(f"--- Batch prediction finished in {batch_end_time - batch_start_time:.3f} seconds ---")
    return results

def get_prediction(o_country, d_country, orgin_gdp, dest_gdp, origin_pop, dest_pop, model_type=AllType, verbose=True):
    if verbose: print(f"--- Requesting single prediction for {o_country} -> {d_country} (Model Type: {model_type}) ---")
    batch_input = [{'o_country': o_country, 'd_country': d_country, 'origin_gdp': orgin_gdp, 'dest_gdp': dest_gdp, 'origin_pop': origin_pop, 'dest_pop': dest_pop}]
    results = predict_batch(batch_input, model_type=model_type, verbose=verbose)
    if results:
        single_result = results[0]
        return {'rf_prediction': single_result.get('RF_Prediction'), 'error': single_result.get('Error')}
    else: return {'error': 'Batch prediction returned empty results.'}

if __name__ == "__main__":
    result = get_prediction(
        o_country='CIV', d_country='GHA',
        orgin_gdp=45815005169, dest_gdp=45815005169,
        origin_pop=25246342, dest_pop=28696068,
        model_type=AllType
    )
    print(result)