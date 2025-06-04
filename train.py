# -*- coding: utf-8 -*-
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from common import *
import time

SEED = 42
BASE_OUTPUT_DIR = 'model'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
NN_EPOCHS = 100
NN_BATCH_SIZE = 64
NN_LEARNING_RATE = 0.001
NN_WEIGHT_DECAY = 1e-5
RF_N_ESTIMATORS = 500
NN_LEARNED_FEATURE_EMBEDDING_DIM = 32

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class VoyageDataset(Dataset):
    def __init__(self, X_numeric, X_cat, y=None):
        self.X_numeric = torch.tensor(np.asarray(X_numeric), dtype=torch.float32)
        self.X_cat = torch.tensor(np.asarray(X_cat), dtype=torch.long)
        if y is not None:
            self.y = torch.tensor(np.asarray(y), dtype=torch.float32)
        else:
            self.y = None
    def __len__(self):
        return len(self.X_numeric)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_numeric[idx], self.X_cat[idx, 0], self.X_cat[idx, 1], self.y[idx]
        else:
            return self.X_numeric[idx], self.X_cat[idx, 0], self.X_cat[idx, 1]

def train_nn_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    print(f"Starting NN training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_numeric_batch, origin_idx_batch, dest_idx_batch, y_batch in train_loader:
            X_numeric_batch, origin_idx_batch, dest_idx_batch, y_batch = \
                X_numeric_batch.to(device), origin_idx_batch.to(device), dest_idx_batch.to(device), y_batch.to(device)
            outputs = model(X_numeric_batch, origin_idx_batch, dest_idx_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f'  Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
    print("NN Training finished.")

def evaluate_model(model, data_loader, criterion, device, model_type="NN"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            X_numeric_batch, origin_idx_batch, dest_idx_batch, y_batch = batch
            X_numeric_batch, origin_idx_batch, dest_idx_batch, y_batch = \
                X_numeric_batch.to(device), origin_idx_batch.to(device), dest_idx_batch.to(device), y_batch.to(device)
            outputs = model(X_numeric_batch, origin_idx_batch, dest_idx_batch)
            if criterion:
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(data_loader) if criterion else None
    all_preds = np.nan_to_num(np.array(all_preds))
    all_targets = np.nan_to_num(np.array(all_targets))
    finite_mask = np.isfinite(all_targets) & np.isfinite(all_preds)
    if not np.all(finite_mask):
        print(f"Warning: Non-finite values found in targets or predictions. Count: {np.sum(~finite_mask)}")
        all_targets = all_targets[finite_mask]
        all_preds = all_preds[finite_mask]
    if len(all_targets) == 0:
        print("Error: No valid pairs for evaluation.")
        return {"RMSE": np.nan, "MAE": np.nan, "R²": np.nan, "Loss": avg_loss,
                "NRMSE_Range": np.nan, "NRMSE_Mean": np.nan, "MAPE": np.nan, "SMAPE": np.nan, "CV_RMSE": np.nan}

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    if np.var(all_targets) < 1e-9:
        r2 = np.nan if len(np.unique(all_targets)) <= 1 else r2_score(all_targets, all_preds)
    else:
        r2 = r2_score(all_targets, all_preds)

    target_range = np.max(all_targets) - np.min(all_targets)
    target_mean = np.mean(all_targets)
    nrmse_range = rmse / target_range if target_range > 0 else np.nan
    nrmse_mean = rmse / target_mean if target_mean > 0 else np.nan

    non_zero_mask = all_targets != 0
    if np.any(non_zero_mask):
        mape = np.mean(
            np.abs((all_targets[non_zero_mask] - all_preds[non_zero_mask]) / all_targets[non_zero_mask])) * 100
    else:
        mape = np.nan

    denominator = np.abs(all_targets) + np.abs(all_preds)
    valid_denom = denominator > 0
    if np.any(valid_denom):
        smape = np.mean(2 * np.abs(all_preds[valid_denom] - all_targets[valid_denom]) / denominator[valid_denom]) * 100
    else:
        smape = np.nan

    cv_rmse = rmse / target_mean if target_mean > 0 else np.nan

    print(
        f"  Evaluation ({model_type}) - Loss: {avg_loss if avg_loss else 'N/A'}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"  Additional Metrics - NRMSE (Range): {nrmse_range:.4f}, NRMSE (Mean): {nrmse_mean:.4f}")
    print(f"  Additional Metrics - MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%, CV(RMSE): {cv_rmse:.4f}")

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R²": float(r2),
        "Loss": float(avg_loss) if avg_loss is not None else None,
        "NRMSE_Range": float(nrmse_range) if not np.isnan(nrmse_range) else None,
        "NRMSE_Mean": float(nrmse_mean) if not np.isnan(nrmse_mean) else None,
        "MAPE": float(mape) if not np.isnan(mape) else None,
        "SMAPE": float(smape) if not np.isnan(smape) else None,
        "CV_RMSE": float(cv_rmse) if not np.isnan(cv_rmse) else None
    }

def run_training(output_dir, input_csv=None, input_df=None, vessel_type_filter=None):
    print("-" * 50)
    run_identifier = vessel_type_filter if vessel_type_filter else "full"
    print(f"Starting training run: {run_identifier}")
    print(f"Output Directory: {output_dir}")
    start_time_run = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if input_df is not None:
        print("Using pre-loaded DataFrame.")
        df = input_df.copy()
    elif input_csv is not None:
        print(f"Loading data from CSV: {input_csv}")
        try:
            df_loaded = pd.read_csv(input_csv)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_csv}")
            return None
        if vessel_type_filter:
            print(f"Filtering for VesselType: {vessel_type_filter}")
            df = df_loaded[df_loaded['VesselType'] == vessel_type_filter].copy()
        else:
            df = df_loaded.copy()
    else:
        print("Error: Neither input_csv nor input_df provided.")
        return None

    if df.empty:
        print(f"Warning: No data available for run '{run_identifier}'. Skipping.")
        return None
    print(f"Initial data shape for this run: {df.shape}")

    base_feature_cols = ['OCentrality', 'DCentrality',
                         'OGDP', 'DGDP', 'OPOP', 'DPOP',
                         'contig', 'comlang_off', 'comcol', 'col45', 'fta_wto', 'Distance']
    static_pair_cols = ['OCentrality', 'DCentrality', 'Distance',
                        'contig', 'comlang_off', 'comcol', 'col45', 'fta_wto']
    target_col = 'RouteCount'

    numeric_cols_to_convert = base_feature_cols + [target_col]
    print("Converting columns to numeric...")
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found during numeric conversion.")

    if df.empty:
        print("Error: No data remaining before feature engineering. Cannot proceed.")
        return None

    print("Performing feature engineering...")
    df = engineer_features(df)

    final_numeric_feature_cols = [
        'OGDP', 'DGDP', 'OPOP', 'DPOP',
        'OCentrality', 'DCentrality', 'Distance',
        'contig', 'comlang_off', 'comcol', 'col45', 'fta_wto',
        'GDPRatio', 'POPRatio', 'OGDPPerCapita', 'DGDPPerCapita',
        'GDPProduct', 'POPProduct', 'CentralityDiff', 'CentralityProduct'
    ]
    final_numeric_feature_cols = [col for col in final_numeric_feature_cols if col in df.columns]
    print(f"Final numeric features ({len(final_numeric_feature_cols)}): {final_numeric_feature_cols}")

    missing_counts = df[final_numeric_feature_cols].isnull().sum()
    missing_report = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if not missing_report.empty:
        print("\nMissing values in numeric features (to be imputed):")
        print(missing_report)
    else:
         print("\nNo missing values found in numeric features before imputation.")

    print("Creating country to index mapping...")
    df['OCountry'] = df['OCountry'].astype(str).fillna('Unknown')
    df['DCountry'] = df['DCountry'].astype(str).fillna('Unknown')
    origin_countries = df['OCountry'].unique()
    dest_countries = df['DCountry'].unique()
    all_countries = np.union1d(origin_countries, dest_countries)
    n_countries = len(all_countries)
    print(f"Total unique countries: {n_countries}")
    country_to_idx = {country: idx for idx, country in enumerate(all_countries)}
    joblib.dump(country_to_idx, os.path.join(output_dir, 'country_to_idx.pkl'))
    print("Saved country_to_idx mapping.")
    df['OCountry_idx'] = df['OCountry'].map(country_to_idx)
    df['DCountry_idx'] = df['DCountry'].map(country_to_idx)

    print("Creating static feature mapping for O-D pairs...")
    static_features_map = {}
    cols_for_map = ['OCountry', 'DCountry'] + [col for col in static_pair_cols if col in df.columns]
    if len(cols_for_map) > 2:
        unique_pairs_df = df[cols_for_map].drop_duplicates(subset=['OCountry', 'DCountry'], keep='first').copy()
        for col in static_pair_cols:
            if col in unique_pairs_df.columns:
                 unique_pairs_df[col] = pd.to_numeric(unique_pairs_df[col], errors='coerce').fillna(0)
        for _, row in unique_pairs_df.iterrows():
            key = (row['OCountry'], row['DCountry'])
            static_features_map[key] = {col: row[col] for col in static_pair_cols if col in row}
        joblib.dump(static_features_map, os.path.join(output_dir, 'static_pair_features.pkl'))
        print(f"Saved static features map for {len(static_features_map)} O-D pairs.")
    else:
        print("Warning: Not enough static columns found to create static_pair_features.pkl.")

    print("Preprocessing data...")

    rows_before_target_drop = len(df)
    df_processed = df.dropna(subset=[target_col]).copy()
    rows_after_target_drop = len(df_processed)
    print(f"Dropped {rows_before_target_drop - rows_after_target_drop} rows due to NaN in target column '{target_col}'.")
    print(f"Data shape after dropping rows with NaN target: {df_processed.shape}")
    if df_processed.empty:
        print("Error: No data remaining after dropping rows with NaN target. Cannot proceed.")
        return None

    X_numeric_df = df_processed[final_numeric_feature_cols].copy()
    X_cat_df = df_processed[['OCountry_idx', 'DCountry_idx']].copy()
    y_series = df_processed[target_col].copy()

    X_numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Applying Imputation (Median)...")
    imputer = SimpleImputer(strategy='median')
    X_numeric_imputed = imputer.fit_transform(X_numeric_df)
    X_numeric_imputed_df = pd.DataFrame(X_numeric_imputed, columns=final_numeric_feature_cols, index=X_numeric_df.index)
    if X_numeric_imputed_df.isnull().sum().sum() > 0:
        print("Warning: NaNs found AFTER imputation. Check data.")

    print("Applying Scaling...")
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric_imputed_df)
    X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=final_numeric_feature_cols, index=X_numeric_imputed_df.index)

    joblib.dump(imputer, os.path.join(output_dir, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    training_config = {
        'numeric_feature_cols_order': final_numeric_feature_cols,
        'n_countries': n_countries,
        'embedding_dim': NN_LEARNED_FEATURE_EMBEDDING_DIM,
        'num_numeric_features_nn_input': X_numeric_scaled_df.shape[1]
    }
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, cls=NumpyEncoder)
    print("Saved imputer, scaler, and training config.")

    print("Splitting data into Train/Test sets...")
    X_train_numeric, X_test_numeric, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
        X_numeric_scaled_df, X_cat_df, y_series, test_size=0.2, random_state=SEED
    )
    print(f"Final Train set size: {len(X_train_numeric)}, Final Test set size: {len(X_test_numeric)}")

    train_dataset = VoyageDataset(X_train_numeric, X_train_cat, y_train)
    test_dataset = VoyageDataset(X_test_numeric, X_test_cat, y_test)
    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NN_BATCH_SIZE, shuffle=False)

    print("\n--- Training Neural Network ---")
    num_numeric_features_nn = X_train_numeric.shape[1]
    nn_model = EmbeddingModel(n_countries, NN_LEARNED_FEATURE_EMBEDDING_DIM, num_numeric_features_nn).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=NN_LEARNING_RATE, weight_decay=NN_WEIGHT_DECAY)

    train_nn_model(nn_model, train_loader, criterion, optimizer, device, epochs=NN_EPOCHS)
    nn_results = evaluate_model(nn_model, test_loader, criterion, device, model_type="NN")

    torch.save(nn_model.state_dict(), os.path.join(output_dir, 'embedding_model.pth'))

    print("\n--- Preparing data for Random Forest ---")
    nn_model.eval()
    nn_model.to('cpu')
    X_train_numeric_tensor = torch.tensor(X_train_numeric.values, dtype=torch.float32)
    X_train_origin_tensor = torch.tensor(X_train_cat['OCountry_idx'].values, dtype=torch.long)
    X_train_dest_tensor = torch.tensor(X_train_cat['DCountry_idx'].values, dtype=torch.long)

    X_test_numeric_tensor = torch.tensor(X_test_numeric.values, dtype=torch.float32)
    X_test_origin_tensor = torch.tensor(X_test_cat['OCountry_idx'].values, dtype=torch.long)
    X_test_dest_tensor = torch.tensor(X_test_cat['DCountry_idx'].values, dtype=torch.long)

    X_train_final_rf = nn_model.get_embeddings(X_train_numeric_tensor, X_train_origin_tensor,
                                               X_train_dest_tensor).numpy()
    X_test_final_rf = nn_model.get_embeddings(X_test_numeric_tensor, X_test_origin_tensor, X_test_dest_tensor).numpy()

    y_train_np = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)

    print("\n--- Training Random Forest ---")
    rf_model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train_final_rf, y_train_np)
    print("Evaluating Random Forest...")
    y_pred_rf = rf_model.predict(X_test_final_rf)
    y_pred_rf = np.nan_to_num(y_pred_rf)
    finite_mask_rf = np.isfinite(y_test_np) & np.isfinite(y_pred_rf)
    y_test_rf_filt = y_test_np[finite_mask_rf]
    y_pred_rf_filt = y_pred_rf[finite_mask_rf]

    if len(y_test_rf_filt) == 0:
        print("Error: No valid pairs for RF evaluation.")
        rf_results = {
            "RMSE": np.nan, "MAE": np.nan, "R²": np.nan,
            "NRMSE_Range": np.nan, "NRMSE_Mean": np.nan,
            "MAPE": np.nan, "SMAPE": np.nan, "CV_RMSE": np.nan
        }
    else:
        rf_rmse = np.sqrt(mean_squared_error(y_test_rf_filt, y_pred_rf_filt))
        rf_mae = mean_absolute_error(y_test_rf_filt, y_pred_rf_filt)
        if np.var(y_test_rf_filt) < 1e-9:
            rf_r2 = np.nan if len(np.unique(y_test_rf_filt)) <= 1 else r2_score(y_test_rf_filt, y_pred_rf_filt)
        else:
            rf_r2 = r2_score(y_test_rf_filt, y_pred_rf_filt)

        target_range = np.max(y_test_rf_filt) - np.min(y_test_rf_filt)
        target_mean = np.mean(y_test_rf_filt)
        rf_nrmse_range = rf_rmse / target_range if target_range > 0 else np.nan
        rf_nrmse_mean = rf_rmse / target_mean if target_mean > 0 else np.nan

        non_zero_mask = y_test_rf_filt != 0
        if np.any(non_zero_mask):
            rf_mape = np.mean(np.abs((y_test_rf_filt[non_zero_mask] - y_pred_rf_filt[non_zero_mask]) /
                                     y_test_rf_filt[non_zero_mask])) * 100
        else:
            rf_mape = np.nan

        denominator = np.abs(y_test_rf_filt) + np.abs(y_pred_rf_filt)
        valid_denom = denominator > 0
        if np.any(valid_denom):
            rf_smape = np.mean(2 * np.abs(y_pred_rf_filt[valid_denom] - y_test_rf_filt[valid_denom]) /
                               denominator[valid_denom]) * 100
        else:
            rf_smape = np.nan

        rf_cv_rmse = rf_rmse / target_mean if target_mean > 0 else np.nan

        rf_results = {
            "RMSE": float(rf_rmse),
            "MAE": float(rf_mae),
            "R²": float(rf_r2),
            "NRMSE_Range": float(rf_nrmse_range) if not np.isnan(rf_nrmse_range) else None,
            "NRMSE_Mean": float(rf_nrmse_mean) if not np.isnan(rf_nrmse_mean) else None,
            "MAPE": float(rf_mape) if not np.isnan(rf_mape) else None,
            "SMAPE": float(rf_smape) if not np.isnan(rf_smape) else None,
            "CV_RMSE": float(rf_cv_rmse) if not np.isnan(rf_cv_rmse) else None
        }
        print(f"  Evaluation (RF) - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
        print(f"  Additional Metrics - NRMSE (Range): {rf_nrmse_range:.4f}, NRMSE (Mean): {rf_nrmse_mean:.4f}")
        print(f"  Additional Metrics - MAPE: {rf_mape:.2f}%, SMAPE: {rf_smape:.2f}%, CV(RMSE): {rf_cv_rmse:.4f}")

    joblib.dump(rf_model, os.path.join(output_dir, 'random_forest_with_embeddings.pkl'))
    print("Saved RF model.")
    print("Calculating and saving RF feature importance...")
    numeric_feature_names_rf = final_numeric_feature_cols
    nn_feature_names = [f'nn_learned_feature_{i}' for i in range(NN_LEARNED_FEATURE_EMBEDDING_DIM)]
    all_rf_feature_names = numeric_feature_names_rf + nn_feature_names
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        if len(all_rf_feature_names) == len(importances):
            feature_importance_df = pd.DataFrame({'Feature': all_rf_feature_names,'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
            importance_filepath = os.path.join(output_dir, 'random_forest_feature_importance.csv')
            try:
                feature_importance_df.to_csv(importance_filepath, index=False)
                print(f"Saved RF feature importance to {importance_filepath}")
            except Exception as e:
                print(f"Error saving feature importance: {e}")
        else:
            print(f"Warning: Mismatch between number of feature names ({len(all_rf_feature_names)}) and importances ({len(importances)}). Skipping importance saving.")
    else:
        print("Warning: Trained RF model does not have 'feature_importances_' attribute.")

    print("\n--- Calculating Combined Feature Importance ---")
    from common import combine_feature_importance, combine_feature_importance_with_feature_extraction_analysis, \
        combine_feature_importance_advanced

    nn_model.to('cpu')

    if hasattr(rf_model, 'feature_importances_') and len(all_rf_feature_names) == len(importances):
        try:
            rf_importance_df = feature_importance_df

            print("Calculating basic combined feature importance...")
            basic_result = combine_feature_importance(
                rf_importance_df=rf_importance_df,
                numeric_feature_cols=numeric_feature_names_rf,
                nn_model=nn_model
            )
            basic_importance_filepath = os.path.join(output_dir, 'combined_feature_importance_basic.csv')
            basic_result.to_csv(basic_importance_filepath, index=False)
            print(f"Saved basic combined feature importance to {basic_importance_filepath}")

            print("\nCalculating feature extraction based combined feature importance...")
            extraction_result = combine_feature_importance_with_feature_extraction_analysis(
                rf_importance_df=rf_importance_df,
                numeric_feature_cols=numeric_feature_names_rf,
                nn_model=nn_model
            )
            extraction_importance_filepath = os.path.join(output_dir, 'combined_feature_importance_extraction.csv')
            extraction_result.to_csv(extraction_importance_filepath, index=False)
            print(f"Saved extraction-based combined feature importance to {extraction_importance_filepath}")

            advanced_samples = 50
            print(
                f"\nCalculating advanced sensitivity-based combined feature importance (with {advanced_samples} samples)...")
            advanced_result = combine_feature_importance_advanced(
                rf_importance_df=rf_importance_df,
                numeric_feature_cols=numeric_feature_names_rf,
                nn_model=nn_model,
                num_samples=advanced_samples,
                country_identity_ratio=0.6
            )
            advanced_importance_filepath = os.path.join(output_dir, 'combined_feature_importance_advanced.csv')
            advanced_result.to_csv(advanced_importance_filepath, index=False)
            print(f"Saved advanced combined feature importance to {advanced_importance_filepath}")

        except Exception as e:
            print(f"Error calculating combined feature importance: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Combined feature importance calculation skipped due to missing RF feature importance.")

    final_results = {
        "Neural Network": nn_results,
        "Random Forest": rf_results
    }
    results_filepath = os.path.join(output_dir, 'results.json')
    with open(results_filepath, 'w') as f:
        json.dump(final_results, f, indent=4, cls=NumpyEncoder)
    end_time_run = time.time()
    print(f"\nTraining run '{run_identifier}' completed in {end_time_run - start_time_run:.2f} seconds.")
    print(f"Results saved to {results_filepath}")
    print("-" * 50)
    return final_results

if __name__ == "__main__":
    total_start_time = time.time()
    full_data_csv = 'data/voyages_grouped_country.csv'
    vessel_data_csv = 'data/voyages_grouped_country_vessel.csv'
    vessel_types = ["Chemical", "Bulk", "Container", "Oil", "General", "Liquified-Gas"]

    print("--- Starting Full Dataset Training Run ---")
    run_training(input_csv=full_data_csv,
                 output_dir=os.path.join(BASE_OUTPUT_DIR, AllType),
                 input_df=None,
                 vessel_type_filter=None)
    print("\n--- Starting Vessel-Specific Training Runs ---")
    if not os.path.exists(vessel_data_csv):
         print(f"\nWarning: Vessel-specific data file not found at '{vessel_data_csv}'. Skipping vessel type runs.")
    else:
        print(f"Reading vessel data file once: {vessel_data_csv} ...")
        try:
            vessel_df_all = pd.read_csv(vessel_data_csv)
            print(f"Successfully read vessel data. Shape: {vessel_df_all.shape}")
            for vt in vessel_types:
                output_sub_dir = os.path.join(BASE_OUTPUT_DIR, vt.replace(" ", "_").replace("-", "_"))
                print(f"\nFiltering data for VesselType: {vt}...")
                filtered_df = vessel_df_all[vessel_df_all['VesselType'] == vt].copy()
                if filtered_df.empty:
                    print(f"Warning: No data found for VesselType '{vt}' after filtering. Skipping run.")
                    continue
                print(f"Data shape for {vt}: {filtered_df.shape}")
                run_training(
                    output_dir=output_sub_dir,
                    input_df=filtered_df,
                    input_csv=None,
                    vessel_type_filter=vt
                )
        except Exception as e:
             print(f"Error reading or processing {vessel_data_csv}: {e}")
    total_end_time = time.time()
    print(f"\nAll training runs finished in {total_end_time - total_start_time:.2f} seconds.")