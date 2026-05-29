"""
Enterprise Transport Demand Forecasting Engine

File:
    src/forecasting/transport_demand_forecasting.py

Purpose:
- Enterprise transport demand forecasting
- Time-series aware validation
- Multi-model benchmarking
- Leakage-resistant forecasting
- Operational demand prediction

Forecast Target:
- daily_transport_load

Models:
- Linear Regression
- Random Forest
- XGBoost
- LightGBM
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import (
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    TimeSeriesSplit,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

FEATURE_DIR = (
    BASE_DIR / "feature_data"
)

FORECAST_DIR = (
    BASE_DIR / "forecast_output"
)

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_FILE = (
    FEATURE_DIR
    / "transport_features.parquet"
)

METRICS_OUTPUT = (
    FORECAST_DIR
    / "forecasting_metrics.csv"
)

PREDICTION_OUTPUT = (
    FORECAST_DIR
    / "demand_predictions.parquet"
)

BEST_MODEL_OUTPUT = (
    FORECAST_DIR
    / "best_model.pkl"
)

LOG_FILE = (
    FORECAST_DIR
    / "forecasting.log"
)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="w",
            encoding="utf-8",
        ),
        logging.StreamHandler(
            sys.stdout,
        ),
    ],
)

logger = logging.getLogger(__name__)

# =========================================================
# DATA LOADING
# =========================================================


def load_feature_store() -> pd.DataFrame:
    """
    Load enterprise feature store.
    """

    logger.info(
        "=" * 60
    )

    logger.info(
        "Loading transport feature store..."
    )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    logger.info(
        f"Loaded {len(df):,} rows."
    )

    return df

# =========================================================
# DATA PREPARATION
# =========================================================


def prepare_training_data(
    df: pd.DataFrame,
):
    """
    Enterprise forecasting dataset preparation.

    Features:
    - daily aggregation
    - time-aware ordering
    - leakage prevention
    - operational forecasting alignment
    """

    logger.info(
        "Preparing enterprise forecasting dataset..."
    )

    # -----------------------------------------------------
    # DATE NORMALIZATION
    # -----------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # -----------------------------------------------------
    # DAILY OPERATIONAL AGGREGATION
    # -----------------------------------------------------

    logger.info(
        "Aggregating transport demand..."
    )

    aggregation_columns = [
        "date",
        "transport_shift",
        "hub",
    ]

    forecast_df = (
        df.groupby(
            aggregation_columns,
            as_index=False,
        )
        .agg({

            # TARGET
            "daily_transport_load": "mean",

            # TEMPORAL
            "day_of_week": "first",
            "month": "first",
            "week_of_year": "first",
            "week_of_month": "first",
            "is_weekend": "first",
            "is_month_end": "first",

            # OPERATIONAL
            "vendor_delay_flag": "mean",
            "heavy_rain_flag": "mean",
            "month_end_surge_flag": "mean",
            "system_outage_flag": "mean",

            # WORKFORCE
            "extension_flag": "mean",
            "high_extension_flag": "mean",

            # GEOSPATIAL
            "hub_density": "mean",
            "home_distance_km": "mean",

            # FORECASTING FEATURES
            "prev_day_transport_load": "mean",
            "prev_week_transport_load": "mean",
            "rolling_7d_avg_load": "mean",
            "rolling_14d_avg_load": "mean",

            # intentionally excluded initially
            # to reduce leakage risk
            # "rolling_30d_avg_load"

            "load_growth_rate": "mean",

            "rolling_vendor_delay_rate": "mean",
            "rolling_rain_rate": "mean",
            "rolling_extension_rate": "mean",
        })
    )

    logger.info(
        f"Aggregated forecasting rows: "
        f"{len(forecast_df):,}"
    )

    # -----------------------------------------------------
    # CHRONOLOGICAL SORTING
    # -----------------------------------------------------

    forecast_df = (
        forecast_df
        .sort_values("date")
    )

    # -----------------------------------------------------
    # FEATURE SELECTION
    # -----------------------------------------------------

    feature_columns = [

        # TEMPORAL
        "day_of_week",
        "month",
        "week_of_year",
        "week_of_month",
        "is_weekend",
        "is_month_end",

        # OPERATIONAL
        "vendor_delay_flag",
        "heavy_rain_flag",
        "month_end_surge_flag",
        "system_outage_flag",

        # WORKFORCE
        "extension_flag",
        "high_extension_flag",

        # GEOSPATIAL
        "hub_density",
        "home_distance_km",

        # FORECASTING
        "prev_day_transport_load",
        "prev_week_transport_load",
        "rolling_7d_avg_load",
        "rolling_14d_avg_load",

        "load_growth_rate",

        "rolling_vendor_delay_rate",
        "rolling_rain_rate",
        "rolling_extension_rate",
    ]

    target_column = (
        "daily_transport_load"
    )

    # -----------------------------------------------------
    # REMOVE NULLS
    # -----------------------------------------------------

    forecast_df = (
        forecast_df
        .dropna(
            subset=feature_columns
        )
    )

    # -----------------------------------------------------
    # STRICT TYPE ENFORCEMENT
    # -----------------------------------------------------

    X = (
        forecast_df[
            feature_columns
        ]
        .astype(np.float32)
    )
    y = (
        forecast_df[
            target_column
        ]
        .astype(np.float32)
    )
    logger.info(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    return (
        forecast_df,
        X,
        y,
    )

# =========================================================
# TIME SERIES BENCHMARKING
# =========================================================


def run_timeseries_benchmark(
    X,
    y,
):
    """
    Enterprise time-series model benchmarking.
    """

    logger.info(
        "Running TimeSeriesSplit benchmarking..."
    )

    models = {

        "LinearRegression":
        Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]),

        "RandomForest":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost":
        xgb.XGBRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),

        "LightGBM":
        lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
    }

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    metrics = []

    best_model = None

    best_rmse = np.inf

    best_predictions = None

    best_actuals = None

    for model_name, model in models.items():

        logger.info(
            f"Benchmarking: {model_name}"
        )

        fold_rmse = []
        fold_mae = []
        fold_r2 = []

        all_predictions = []
        all_actuals = []

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(tscv.split(X)):

            logger.info(
                f"{model_name} | Fold {fold + 1}"
            )

            X_train = X.iloc[
                train_idx
            ]

            X_test = X.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            y_test = y.iloc[
                test_idx
            ]

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )

            r2 = r2_score(
                y_test,
                predictions,
            )

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)

            all_predictions.extend(
                predictions
            )

            all_actuals.extend(
                y_test.values
            )

        avg_mae = np.mean(
            fold_mae
        )

        avg_rmse = np.mean(
            fold_rmse
        )

        avg_r2 = np.mean(
            fold_r2
        )

        metrics.append({
            "model": model_name,
            "MAE": round(avg_mae, 4),
            "RMSE": round(avg_rmse, 4),
            "R2": round(avg_r2, 4),
        })

        logger.info(
            f"{model_name} | "
            f"MAE={avg_mae:.4f} | "
            f"RMSE={avg_rmse:.4f} | "
            f"R2={avg_r2:.4f}"
        )

        # -------------------------------------------------
        # BEST MODEL TRACKING
        # -------------------------------------------------

        if avg_rmse < best_rmse:

            best_rmse = avg_rmse

            best_model = (
                model_name,
                model,
            )

            best_predictions = (
                np.array(all_predictions)
            )

            best_actuals = (
                np.array(all_actuals)
            )

    metrics_df = pd.DataFrame(
        metrics
    )

    return (
        metrics_df,
        best_model,
        best_predictions,
        best_actuals,
    )

# =========================================================
# EXPORT RESULTS
# =========================================================


def export_results(
    metrics_df,
    best_model,
    predictions,
    actuals,
):
    """
    Export enterprise forecasting artifacts.
    """

    logger.info(
        "Exporting forecasting artifacts..."
    )

    # -----------------------------------------------------
    # METRICS EXPORT
    # -----------------------------------------------------

    metrics_df.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # PREDICTIONS EXPORT
    # -----------------------------------------------------

    prediction_df = pd.DataFrame({

        "actual_transport_load":
        actuals,

        "predicted_transport_load":
        predictions,
    })

    prediction_df.to_parquet(
        PREDICTION_OUTPUT,
        index=False,
    )

    prediction_df.to_csv(
        PREDICTION_OUTPUT.with_suffix(
            ".csv"
        ),
        index=False,
    )

    # -----------------------------------------------------
    # BEST MODEL EXPORT
    # -----------------------------------------------------

    model_name, model = best_model

    with open(
        BEST_MODEL_OUTPUT,
        "wb",
    ) as f:

        pickle.dump(
            model,
            f,
        )

    logger.info(
        f"Best model: {model_name}"
    )

    logger.info(
        f"Model exported: "
        f"{BEST_MODEL_OUTPUT.name}"
    )

# =========================================================
# MAIN PIPELINE
# =========================================================


def run_forecasting_pipeline():
    """
    Enterprise forecasting pipeline.
    """

    logger.info(
        "=" * 60
    )

    logger.info(
        "Starting enterprise "
        "transport forecasting..."
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_feature_store()

    # -----------------------------------------------------
    # DATA PREPARATION
    # -----------------------------------------------------

    (
        forecast_df,
        X,
        y,
    ) = prepare_training_data(df)

    # -----------------------------------------------------
    # MODEL BENCHMARKING
    # -----------------------------------------------------

    (
        metrics_df,
        best_model,
        predictions,
        actuals,
    ) = run_timeseries_benchmark(
        X,
        y,
    )

    # -----------------------------------------------------
    # EXPORT RESULTS
    # -----------------------------------------------------

    export_results(
        metrics_df,
        best_model,
        predictions,
        actuals,
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        "Forecasting pipeline completed "
        "successfully."
    )

# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    run_forecasting_pipeline()
