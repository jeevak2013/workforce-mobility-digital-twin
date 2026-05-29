"""
Enterprise Workforce Demand Forecasting

Predicts:

1. Login Demand
   (date, hub, login_shift)

2. Logout Demand
   (date, hub, transport_shift)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

import joblib
import numpy as np
import re

from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.ensemble import (
    RandomForestRegressor,
)

from sklearn.model_selection import (
    TimeSeriesSplit,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import (
    XGBRegressor,
)

from lightgbm import (
    LGBMRegressor,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR: Final = (
    Path(__file__)
    .resolve()
    .parents[2]
)

FEATURE_DIR: Final = (
    BASE_DIR
    / "feature_data"
)

MODEL_DIR: Final = (
    BASE_DIR
    / "models"
)

OUTPUT_DIR: Final = (
    BASE_DIR
    / "forecast_output"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_STORE: Final = (
    FEATURE_DIR
    / "transport_features.parquet"
)

LOG_FILE: Final = (
    OUTPUT_DIR
    / "workforce_forecasting.log"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# DATA LOADING
# ============================================================

def load_feature_store() -> pd.DataFrame:

    logger.info(
        "Loading feature store..."
    )

    df = pd.read_parquet(
        FEATURE_STORE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    logger.info(
        "Rows loaded: %s",
        f"{len(df):,}",
    )

    return df


# ============================================================
# LOGIN DEMAND DATASET
# ============================================================

def build_login_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Building login demand dataset..."
    )

    login_df = (
        df.groupby(
            [
                "date",
                "hub",
                "login_shift",
            ],
            as_index=False,
        )
        .agg(
            daily_login_demand=(
                "employee_id",
                "count",
            ),

            day_of_week=(
                "day_of_week",
                "first",
            ),

            month=(
                "month",
                "first",
            ),

            week_of_year=(
                "week_of_year",
                "first",
            ),

            week_of_month=(
                "week_of_month",
                "first",
            ),

            is_weekend=(
                "is_weekend",
                "first",
            ),

            is_month_end=(
                "is_month_end",
                "first",
            ),

            vendor_delay_flag=(
                "vendor_delay_flag",
                "mean",
            ),

            heavy_rain_flag=(
                "heavy_rain_flag",
                "mean",
            ),

            month_end_surge_flag=(
                "month_end_surge_flag",
                "mean",
            ),

            system_outage_flag=(
                "system_outage_flag",
                "mean",
            ),

            hub_operational_weight=(
                "hub_operational_weight",
                "mean",
            ),
        )
    )

    logger.info(
        "Login forecast rows: %s",
        f"{len(login_df):,}",
    )

    return login_df


# ============================================================
# LOGOUT DEMAND DATASET
# ============================================================

def build_logout_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Building logout demand dataset..."
    )

    logout_df = (
        df.groupby(
            [
                "date",
                "hub",
                "transport_shift",
            ],
            as_index=False,
        )
        .agg(
            daily_logout_demand=(
                "employee_id",
                "count",
            ),

            day_of_week=(
                "day_of_week",
                "first",
            ),

            month=(
                "month",
                "first",
            ),

            week_of_year=(
                "week_of_year",
                "first",
            ),

            week_of_month=(
                "week_of_month",
                "first",
            ),

            is_weekend=(
                "is_weekend",
                "first",
            ),

            is_month_end=(
                "is_month_end",
                "first",
            ),

            vendor_delay_flag=(
                "vendor_delay_flag",
                "mean",
            ),

            heavy_rain_flag=(
                "heavy_rain_flag",
                "mean",
            ),

            month_end_surge_flag=(
                "month_end_surge_flag",
                "mean",
            ),

            system_outage_flag=(
                "system_outage_flag",
                "mean",
            ),

            hub_operational_weight=(
                "hub_operational_weight",
                "mean",
            ),
        )
    )

    logger.info(
        "Logout forecast rows: %s",
        f"{len(logout_df):,}",
    )

    return logout_df

def create_forecast_features(
    df: pd.DataFrame,
    target_column: str,
    grouping_columns: list[str],
) -> pd.DataFrame:

    logger.info(
        "Creating forecasting features..."
    )

    df = df.sort_values(
        grouping_columns + ["date"]
    ).copy()

    grouped = df.groupby(
        grouping_columns
    )

    # ==========================================
    # LAG FEATURES
    # ==========================================

    df["lag_1"] = (
        grouped[target_column]
        .shift(1)
    )

    df["lag_7"] = (
        grouped[target_column]
        .shift(7)
    )

    df["lag_14"] = (
        grouped[target_column]
        .shift(14)
    )

    # ==========================================
    # ROLLING FEATURES
    # ==========================================

    shifted_target = (
        grouped[target_column]
        .shift(1)
    )

    df["rolling_7d_avg"] = (
        grouped[target_column]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=7,
                min_periods=1,
            )
            .mean()
        )
    )

    df["rolling_14d_avg"] = (
        grouped[target_column]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=14,
                min_periods=1,
            )
            .mean()
        )
    )

    # ==========================================
    # TREND FEATURES
    # ==========================================

    df["weekly_growth_rate"] = (
        (
            df["lag_1"]
            - df["lag_7"]
        )
        /
        (
            df["lag_7"]
            + 1
        )
    )

    df["monthly_growth_rate"] = (
        (
            df["lag_1"]
            - df["lag_14"]
        )
        /
        (
            df["lag_14"]
            + 1
        )
    )

    # ==========================================
    # CLEANUP
    # ==========================================

    df = df.fillna(0)

    logger.info(
        "Forecast features created."
    )

    return df

def benchmark_models(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str],
) -> tuple:

    logger.info(
        "Starting model benchmarking..."
    )

    df = df.copy()

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False,
    )

    df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            str(col),
        )
        for col in df.columns
    ]

    feature_columns = [

        col
        for col in df.columns
        if col
        not in [
            "date",
            target_column,
        ]
    ]

    X = df[
        feature_columns
    ].astype(
        np.float32
    )

    y = df[
        target_column
    ]

    models = {

        "LinearRegression":
        LinearRegression(),

        "RandomForest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost":
        XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),

        "LightGBM":
        LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        ),
    }

    splitter = (
        TimeSeriesSplit(
            n_splits=5
        )
    )

    metrics = []

    best_model = None

    best_rmse = float(
        "inf"
    )

    best_predictions = None

    best_actuals = None

    for model_name, model in (
        models.items()
    ):

        logger.info(
            f"Benchmarking: {model_name}"
        )

        fold_predictions = []

        fold_actuals = []

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(
            splitter.split(X),
            start=1,
        ):

            logger.info(
                f"{model_name} | Fold {fold}"
            )

            X_train = (
                X.iloc[train_idx]
            )

            X_test = (
                X.iloc[test_idx]
            )

            y_train = (
                y.iloc[train_idx]
            )

            y_test = (
                y.iloc[test_idx]
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = (
                model.predict(
                    X_test
                )
            )

            fold_predictions.extend(
                predictions
            )

            fold_actuals.extend(
                y_test
            )

        mae = mean_absolute_error(
            fold_actuals,
            fold_predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                fold_actuals,
                fold_predictions,
            )
        )

        r2 = r2_score(
            fold_actuals,
            fold_predictions,
        )

        metrics.append({

            "model":
            model_name,

            "MAE":
            round(mae, 4),

            "RMSE":
            round(rmse, 4),

            "R2":
            round(r2, 4),
        })

        logger.info(
            f"{model_name} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:.4f}"
        )

        if rmse < best_rmse:

            best_rmse = rmse

            best_model = model

            best_predictions = (
                fold_predictions
            )

            best_actuals = (
                fold_actuals
            )

    metrics_df = (
        pd.DataFrame(
            metrics
        )
    )

    return (

        best_model,

        metrics_df,

        best_predictions,

        best_actuals,

        feature_columns,
    )

def train_login_forecaster():

    logger.info(
        "=" * 60
    )

    logger.info(
        "TRAINING LOGIN FORECASTER"
    )

    df = load_feature_store()

    login_df = build_login_dataset(
        df
    )

    login_df = (
        create_forecast_features(
            df=login_df,
            target_column=
            "daily_login_demand",
            grouping_columns=[
                "hub",
                "login_shift",
            ],
        )
    )

    (
        best_model,
        metrics_df,
        predictions,
        actuals,
        feature_columns,
    ) = benchmark_models(
        df=login_df,
        target_column=
        "daily_login_demand",
        categorical_columns=[
            "hub",
            "login_shift",
        ],
    )

    metrics_df.to_csv(
        OUTPUT_DIR
        / "login_metrics.csv",
        index=False,
    )

    logger.info(
        "Training final login model..."
    )

    model_df = pd.get_dummies(
        login_df.copy(),
        columns=[
            "hub",
            "login_shift",
        ],
        drop_first=False,
    )

    import re

    model_df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            str(col),
        )
        for col in model_df.columns
    ]

    feature_columns = [

        col
        for col in model_df.columns
        if col
        not in [
            "date",
            "daily_login_demand",
        ]
    ]

    X = model_df[
        feature_columns
    ]

    y = model_df[
        "daily_login_demand"
    ]

    best_model.fit(
        X,
        y,
    )

    joblib.dump(
        best_model,
        MODEL_DIR
        / "workforce_login_model.pkl",
    )

    prediction_df = pd.DataFrame({

        "actual":
        actuals,

        "predicted":
        predictions,
    })

    prediction_df[
        "error"
    ] = (
        prediction_df["actual"]
        -
        prediction_df["predicted"]
    )

    prediction_df.to_parquet(
        OUTPUT_DIR
        / "login_forecast.parquet",
        index=False,
    )

    if hasattr(
        best_model,
        "coef_",
    ):

        importance_df = (
            pd.DataFrame({
                "feature":
                feature_columns,

                "importance":
                best_model.coef_,
            })
            .sort_values(
                "importance",
                key=abs,
                ascending=False,
            )
        )

        importance_df.to_csv(
            OUTPUT_DIR
            / "login_feature_importance.csv",
            index=False,
        )

    logger.info(
        "Login forecaster completed."
    )

def train_logout_forecaster():

    logger.info(
        "=" * 60
    )

    logger.info(
        "TRAINING LOGOUT FORECASTER"
    )

    df = load_feature_store()

    logout_df = build_logout_dataset(
        df
    )

    logout_df = (
        create_forecast_features(
            df=logout_df,
            target_column=
            "daily_logout_demand",
            grouping_columns=[
                "hub",
                "transport_shift",
            ],
        )
    )

    (
        best_model,
        metrics_df,
        predictions,
        actuals,
        feature_columns,
    ) = benchmark_models(
        df=logout_df,
        target_column=
        "daily_logout_demand",
        categorical_columns=[
            "hub",
            "transport_shift",
        ],
    )

    metrics_df.to_csv(
        OUTPUT_DIR
        / "logout_metrics.csv",
        index=False,
    )

    model_df = pd.get_dummies(
        logout_df.copy(),
        columns=[
            "hub",
            "transport_shift",
        ],
        drop_first=False,
    )

    import re

    model_df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            str(col),
        )
        for col in model_df.columns
    ]

    feature_columns = [

        col
        for col in model_df.columns
        if col
        not in [
            "date",
            "daily_logout_demand",
        ]
    ]

    X = model_df[
        feature_columns
    ]

    y = model_df[
        "daily_logout_demand"
    ]

    best_model.fit(
        X,
        y,
    )

    joblib.dump(
        best_model,
        MODEL_DIR
        / "workforce_logout_model.pkl",
    )

    prediction_df = pd.DataFrame({

        "actual":
        actuals,

        "predicted":
        predictions,
    })

    prediction_df[
        "error"
    ] = (
        prediction_df["actual"]
        -
        prediction_df["predicted"]
    )

    prediction_df.to_parquet(
        OUTPUT_DIR
        / "logout_forecast.parquet",
        index=False,
    )

    if hasattr(
        best_model,
        "coef_",
    ):

        importance_df = (
            pd.DataFrame({
                "feature":
                feature_columns,

                "importance":
                best_model.coef_,
            })
            .sort_values(
                "importance",
                key=abs,
                ascending=False,
            )
        )

        importance_df.to_csv(
            OUTPUT_DIR
            / "logout_feature_importance.csv",
            index=False,
        )

    logger.info(
        "Logout forecaster completed."
    )

def main():

    logger.info(
        "=" * 60
    )

    logger.info(
        "Starting workforce demand forecasting..."
    )

    train_login_forecaster()

    train_logout_forecaster()

    logger.info(
        "=" * 60
    )

    logger.info(
        "Workforce forecasting completed."
    )


if __name__ == "__main__":
    main()