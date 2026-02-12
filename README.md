# Global Shipping Volume Prediction Model (NNERF)

This project implements a hybrid **Neural Network-Enhanced Random Forest (NNERF)** model to predict global shipping volumes. The architecture leverages deep learning for country-level entity embeddings and Random Forest for robust regression.

---

## Installation

### Prerequisites
* Python 3.8+
* GPU with CUDA support (recommended)

### Setup

```bash
# Install GPU-optimized PyTorch (Adjust 'cu124' to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

**Note**: Training results may slightly vary between CPU and GPU due to non-deterministic CUDA kernels. For reproducibility, maintain a consistent hardware environment.

---

## Data Requirements

Place the following CSV files in a `data/` subdirectory:

### 1. `voyages_grouped_country.csv`
Aggregated shipping data at the country-pair level:
```
Year, OCountry, DCountry, OCentrality, DCentrality, OGDP, DGDP, OPOP, DPOP, 
contig, comlang_off, comcol, col45, fta_wto, Distance, RouteCount
```

### 2. `voyages_grouped_country_vessel.csv`
Shipping data at the country-pair and vessel-type level:
```
Year, OCountry, DCountry, OCentrality, DCentrality, OGDP, DGDP, OPOP, DPOP, 
contig, comlang_off, comcol, col45, fta_wto, Distance, RouteCount, VesselType
```

**Key Fields**:
* **OCountry, DCountry**: Origin/Destination country codes (ISO 3166-1 alpha-3)
* **OCentrality, DCentrality**: Network betweenness centrality scores
* **OGDP, DGDP**: Gross Domestic Product (current USD)
* **OPOP, DPOP**: Population (total count)
* **contig, comlang_off, comcol, col45, fta_wto**: Binary geopolitical/trade indicators
* **Distance**: Weighted average inter-port distance (km)
* **RouteCount**: Target variable (number of voyages/routes)
* **VesselType**: Vessel category (Chemical, Bulk, Container, Oil, General, Liquified-Gas)

---

## Usage

### Training

```bash
# Basic training (single seed)
# Set RUN_ANALYSIS = False in train.py
python train.py

# Multi-seed analysis (for reproducibility)
# Set RUN_ANALYSIS = True in train.py
python train.py
```

### Prediction

```python
from predict import get_prediction, predict_batch

# Single prediction with uncertainty
result = get_prediction(
    o_country='USA',
    d_country='CHN',
    orgin_gdp=21000000000000,
    dest_gdp=14000000000000,
    origin_pop=331000000,
    dest_pop=1400000000,
    model_type='All'
)

# Batch predictions
results = predict_batch(batch_input, model_type='All')
```

**Output includes**:
* `RF_Prediction`: Point estimate
* `Traffic_Lower` / `Traffic_Upper`: 95% prediction interval bounds

---

## Statistical Reporting (Reviewer Requirements)

This section addresses the specific statistical items flagged for clarification in the Nature Portfolio Statistics Checklist.

### 1. Sample Size (n)

**Exact sample size**: n = 50,000 country-pair-year observations

**Unit of measurement**: The analytical unit is the **country-pair-year**. Each observation represents the aggregate number of maritime voyages between an origin country and a destination country in a specific year.

**Data structure**: 
* Temporal coverage: 6 non-consecutive years (2002, 2005, 2008, 2012, 2015, 2018)
* Source: City-to-city maritime routes aggregated to country-pair level
* Geographic coverage: Global (all active maritime trading nations)

**Data partitioning**:
* Training set: 80% (n = 40,000)
* Independent test set: 20% (n = 10,000)

### 2. Repeated Measurements

**Measurement structure**: The dataset contains **repeated annual measurements** of the same country pairs across the six sample years. This represents a longitudinal panel design with temporal replication.

**Key characteristics**:
* Each country pair (e.g., USA→China) is observed multiple times across different years
* Observations are **not strictly independent** within a country pair across time
* However, the sparse temporal resolution (6 non-consecutive years spanning 16 years) and high inter-annual volatility in shipping volumes reduce autocorrelation concerns

**Data splitting approach**: Random 80/20 split (rather than temporal holdout)

**Rationale for random splitting**:

1. **Sparse temporal coverage**: With only 6 time points, holding out an entire year would sacrifice 16.7% of temporal anchor points, severely limiting the model's ability to learn temporal trends

2. **Long-term extrapolation requirement**: The model includes a linear extrapolation layer designed to project shipping volumes to 2100. This component requires capturing the full range of GDP-driven variance across the 2002-2018 period. Removing any year would create gaps in the economic growth trajectory essential for century-scale forecasting

3. **Limited data leakage risk**: 
   * Country identity contributes only ~10% to final Random Forest predictions (verified through feature importance analysis in `combined_feature_importance_advanced.csv`)
   * Shipping volumes between country pairs exhibit substantial year-to-year fluctuations (high variance), making it difficult for the model to "memorize" country-specific patterns
   * The model relies primarily on dynamic economic indicators (GDP, population) and network topology (centrality), which exhibit strong temporal trends that benefit from complete temporal coverage

4. **Methodological precedent**: Random splits are standard practice in economic time series modeling when temporal resolution is sparse and the goal is long-term structural forecasting rather than short-term sequential prediction

### 3. Statistical Parameters: Central Tendency and Variation

**Model evaluation**: Performance is assessed using regression metrics on the independent test set (n = 10,000).

**Reproducibility analysis**: To quantify model variability and ensure reproducibility, we trained the model across **5 independent random seeds** (42, 1, 10, 100, 2026) using the `run_multi_seed_analysis()` function in `train.py`.

**Performance metrics (Mean ± SD)**:

| Metric | Test Set Performance |
|--------|---------------------|
| **R²** (Coefficient of Determination) | **0.9316 ± 0.0047** |
| **RMSE** (Root Mean Squared Error) | **228.93 ± 7.86** voyages |
| **MAE** (Mean Absolute Error) | **48.86 ± 1.02** voyages |

**Interpretation**:
* The model explains 93.16% of variance in shipping volumes (R² = 0.9316)
* Standard deviation across seeds is small (σ_R² = 0.0047), indicating robust reproducibility
* RMSE and MAE are reported in original units (number of voyages per country-pair-year)

**Prediction uncertainty quantification**:

For individual predictions, the model provides 95% prediction intervals using residual-based estimation:

**Formula**: 
```
Prediction Interval = ŷ ± 1.96 × σ_residual
```

where:
* **ŷ** = point prediction from Random Forest
* **σ_residual** = standard deviation of residuals on the test set
* **1.96** = z-score for 95% confidence level

**Implementation**: The `predict_batch()` function in `predict.py` (lines 387-391) automatically calculates:
* `Traffic_Lower` = max(0, ŷ - 1.96 × σ_residual)
* `Traffic_Upper` = ŷ + 1.96 × σ_residual

These bounds are used for visualizing forecast uncertainty (e.g., shaded regions in projections to 2100).

**Additional metrics** (full details in `model/All/results.json`):
* NRMSE (Normalized RMSE by range and mean)
* MAPE (Mean Absolute Percentage Error)
* SMAPE (Symmetric MAPE)
* CV(RMSE) (Coefficient of Variation)

### 4. Hierarchical and Complex Designs

**Data hierarchy**: The global shipping network has inherent multi-level structure:
* **City-level routes** → aggregated to **country-level pairs** → embedded within **global maritime network**

**How the model addresses hierarchical structure**:

#### A. Country-Level Fixed Effects (Entity Embeddings)

Each country (origin and destination) is assigned a learnable embedding vector (dimension = 32 in the neural network). These embeddings function analogously to **country-level random effects** in hierarchical linear models:

* **Purpose**: Capture latent, country-specific characteristics not fully represented by observable features (GDP, population, centrality)
* **Examples**: Maritime infrastructure quality, port efficiency, regulatory environment, historical trading patterns
* **Implementation**: `EmbeddingModel.country_embedding` layer in `common.py`
* **Training**: Embeddings are jointly optimized with other model parameters via gradient descent

This approach addresses the **nested structure** where routes are nested within country pairs, and country pairs are nested within the global network.

#### B. Network-Level Topology (Centrality Metrics)

The model incorporates **betweenness centrality** scores for both origin and destination countries:

* **Purpose**: Quantify each country's position in the global maritime network topology
* **Interpretation**: 
  - High centrality = "hub" countries (e.g., Singapore, Netherlands) that serve as transshipment centers
  - Low centrality = "peripheral" countries dependent on hubs for connectivity
* **Hierarchical role**: Captures the **topological hierarchy** of global trade routes, where certain countries occupy structurally advantageous positions

This differs from simple pairwise features (e.g., GDP, distance) by incorporating the **global network context** into country-pair predictions.

#### C. Appropriate Level for Statistical Evaluation

**Analysis level**: All statistical tests and performance metrics are computed at the **country-pair-year level** (the unit of observation).

**Residual structure**: 
* Residuals are calculated for each individual observation (n = 10,000 in test set)
* No aggregation or clustering corrections applied
* The Random Forest and neural network implicitly account for hierarchical dependencies through learned embeddings and centrality features

**Feature importance analysis**: 
* Country identity (via embeddings): ~10% contribution to predictions
* Economic variables (GDP, population): ~50-60% contribution
* Network centrality: ~15-20% contribution
* Geographic/political features: ~10-15% contribution

These importance scores are derived from Random Forest's mean decrease in impurity and are saved in:
* `model/All/random_forest_feature_importance.csv` (raw RF importances)
* `model/All/combined_feature_importance_advanced.csv` (accounting for neural network feature interactions)

---

## Reproducibility Guide

To reproduce the multi-seed analysis results:

1. **Edit `train.py`**: Set `RUN_ANALYSIS = True`
2. **Run training**:
   ```bash
   python train.py
   ```
3. **Output**: The script will display:
   ```
   ========================================
   FINAL STATISTICAL RESULTS (Mean ± SD)
   ========================================
   R²   : 0.9316 ± 0.0047
   RMSE : 228.9297 ± 7.8621
   MAE  : 48.8581 ± 1.0228
   ========================================
   ```

**Seeds used**: [42, 1, 10, 100, 2026]

**Sources of variability**:
* Random Forest bootstrap sampling
* Neural network weight initialization
* Stochastic gradient descent
* Train-test split randomization
* GPU kernel non-determinism (minor)

---

## Model Architecture

**Hybrid Neural Network-Enhanced Random Forest (NNERF)**:

1. **Neural Network Component**:
   - Entity embeddings for countries (capture latent features)
   - Deep feature extraction layers (learn representations)
   - Linear extrapolation layer (support long-term forecasting to 2100)

2. **Random Forest Component**:
   - Uses neural network's learned representations + original features
   - Provides robust, non-linear regression
   - Generates feature importance scores

This architecture combines deep learning's representation learning with ensemble methods' robustness and interpretability.

---

## Repository Structure

```
NNERF-Shipping-Volume/
├── data/
│   ├── voyages_grouped_country.csv
│   └── voyages_grouped_country_vessel.csv
├── model/                                # Generated after training
│   ├── All/                             # Overall shipping model
│   │   ├── results.json                 # Performance metrics
│   │   ├── training_config.json         # Config including σ_residual
│   │   ├── random_forest_feature_importance.csv
│   │   └── combined_feature_importance_advanced.csv
│   └── [Vessel_Type]/                   # Vessel-specific models
├── train.py                             # Training script
├── predict.py                           # Prediction with uncertainty
├── common.py                            # Shared utilities
└── requirements.txt
```

---
