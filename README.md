# Global Shipping Volume Prediction Model

This project implements a hybrid machine learning model to predict global shipping volumes based on economic, geographic, and network centrality factors. The core model leverages a neural network for advanced feature engineering, feeding learned representations into a Random Forest regressor for final prediction.

## Data Requirements

Place the following CSV files in a `data/` subdirectory within the project root:

1.  **`voyages_grouped_country.csv`**: This file should contain aggregated shipping data at the country-pair level. Expected columns are:
    `Year,OCountry,DCountry,OCentrality,DCentrality,OGDP,DGDP,OPOP,DPOP,contig,comlang_off,comcol,col45,fta_wto,Distance,RouteCount`
    The `RouteCount` column serves as the target variable for overall predictions.

2.  **`voyages_grouped_country_vessel.csv`**: This file should contain shipping data aggregated at the country-pair and vessel-type level. Expected columns are:
    `Year,OCountry,DCountry,OCentrality,DCentrality,OGDP,DGDP,OPOP,DPOP,contig,comlang_off,comcol,col45,fta_wto,Distance,RouteCount,VesselType`
    The `RouteCount` column is the target for vessel-specific predictions, and the `VesselType` column is used for filtering.

**Key Data Fields Description:**
* `OCountry`, `DCountry`: Origin/Destination country codes.
* `OCentrality`, `DCentrality`: Network centrality scores.
* `OGDP`, `DGDP`: Gross Domestic Product.
* `OPOP`, `DPOP`: Population.
* `contig`, `comlang_off`, `comcol`, `col45`, `fta_wto`: Binary geopolitical/trade agreement indicators.
* `Distance`: Weighted average inter-port distance.
* `RouteCount`: Target variable (number of voyages/routes).

## Setup

### Prerequisites

* Python 3.8+
* PIP (Python package installer)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/dongzhe1/NNERF-Shipping-Volume
    cd NNERF-Shipping-Volume
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

The `train.py` script handles the training of the hybrid Neural Network-Enhanced Random Forest (NNERF) model. It will train a model for overall shipping volume ("All" types) using `voyages_grouped_country.csv` and separate models for each vessel type using `voyages_grouped_country_vessel.csv`.

### Making Predictions

Use the functions in `predict.py` (e.g., `get_prediction`, `predict_batch`) to load trained models and generate predictions.
