from pathlib import Path
from datetime import timedelta
import logging
import joblib
import pandas as pd
import numpy as np

import re

from src.forecasting.transport_demand_forecasting import (
    load_feature_store,
    build_login_dataset,
    build_logout_dataset,
    create_forecast_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "forecast_output"

OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_models():

    logger.info("Loading forecast models...")

    login_model = joblib.load(MODEL_DIR / "workforce_login_model.pkl")

    logout_model = joblib.load(MODEL_DIR / "workforce_logout_model.pkl")

    login_features = joblib.load(MODEL_DIR / "workforce_login_features.pkl")

    logout_features = joblib.load(MODEL_DIR / "workforce_logout_features.pkl")

    return (
        login_model,
        logout_model,
        login_features,
        logout_features,
    )


def prepare_login_inference_data():

    logger.info("Preparing login inference data...")

    df = load_feature_store()

    login_df = build_login_dataset(df)

    login_df = create_forecast_features(
        df=login_df,
        target_column="daily_login_demand",
        grouping_columns=[
            "hub",
            "login_shift",
        ],
    )

    return login_df


def get_latest_login_records():

    login_df = prepare_login_inference_data()

    latest_login_df = (
        login_df.sort_values("date")
        .groupby(
            [
                "hub",
                "login_shift",
            ],
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_login_df


# Build Tomorrow Login Features
def build_tomorrow_login_features():

    latest_login_df = get_latest_login_records()

    tomorrow_df = latest_login_df.copy()

    tomorrow_df["forecast_date"] = tomorrow_df["date"] + timedelta(days=1)

    # Update Calendar Features
    tomorrow_df["day_of_week"] = tomorrow_df["forecast_date"].dt.dayofweek

    tomorrow_df["month"] = tomorrow_df["forecast_date"].dt.month

    tomorrow_df["week_of_year"] = (
        tomorrow_df["forecast_date"].dt.isocalendar().week.astype(int)
    )

    tomorrow_df["week_of_month"] = (tomorrow_df["forecast_date"].dt.day - 1) // 7 + 1

    tomorrow_df["is_weekend"] = (tomorrow_df["day_of_week"] >= 5).astype(int)

    tomorrow_df["is_month_end"] = (tomorrow_df["forecast_date"].dt.is_month_end).astype(
        int
    )

    # Operational Flags
    tomorrow_df["vendor_delay_flag"] = 0

    tomorrow_df["heavy_rain_flag"] = 0

    tomorrow_df["system_outage_flag"] = 0

    # Month-End Surge
    tomorrow_df["month_end_surge_flag"] = (
        tomorrow_df["forecast_date"].dt.day >= 25
    ).astype(float)

    return tomorrow_df


def predict_tomorrow_login_demand():

    logger.info("Predicting tomorrow login demand...")

    (
        login_model,
        _,
        login_features,
        _,
    ) = load_models()

    tomorrow_df = build_tomorrow_login_features()

    # One-Hot Encoding
    model_df = pd.get_dummies(
        tomorrow_df.copy(),
        columns=[
            "hub",
            "login_shift",
        ],
        drop_first=False,
    )

    # Same Column Cleaning
    model_df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            str(col),
        )
        for col in model_df.columns
    ]

    # Feature Alignment
    for col in login_features:
        if col not in model_df.columns:
            model_df[col] = 0

    # Same Order As Training
    X_future = model_df[login_features]

    predictions = login_model.predict(X_future)

    predictions = np.maximum(predictions, 0).round().astype(int)

    forecast_df = pd.DataFrame(
        {
            "forecast_date": tomorrow_df["forecast_date"],
            "hub": tomorrow_df["hub"],
            "login_shift": tomorrow_df["login_shift"],
            "predicted_demand": predictions,
        }
    )

    assert forecast_df["predicted_demand"].min() >= 0

    logger.info(
        "Generated %s login forecasts",
        len(forecast_df),
    )

    return forecast_df


def prepare_logout_inference_data():

    logger.info("Preparing logout inference data...")

    df = load_feature_store()

    logout_df = build_logout_dataset(df)

    logout_df = create_forecast_features(
        df=logout_df,
        target_column="daily_logout_demand",
        grouping_columns=[
            "hub",
            "transport_shift",
        ],
    )

    return logout_df


def get_latest_logout_records():

    logout_df = prepare_logout_inference_data()

    latest_logout_df = (
        logout_df.sort_values("date")
        .groupby(
            [
                "hub",
                "transport_shift",
            ],
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_logout_df


def build_tomorrow_logout_features():

    latest_logout_df = get_latest_logout_records()

    tomorrow_df = latest_logout_df.copy()

    tomorrow_df["forecast_date"] = tomorrow_df["date"] + timedelta(days=1)

    # Update Calendar Features
    tomorrow_df["day_of_week"] = tomorrow_df["forecast_date"].dt.dayofweek

    tomorrow_df["month"] = tomorrow_df["forecast_date"].dt.month

    tomorrow_df["week_of_year"] = (
        tomorrow_df["forecast_date"].dt.isocalendar().week.astype(int)
    )

    tomorrow_df["week_of_month"] = (tomorrow_df["forecast_date"].dt.day - 1) // 7 + 1

    tomorrow_df["is_weekend"] = (tomorrow_df["day_of_week"] >= 5).astype(int)

    tomorrow_df["is_month_end"] = (tomorrow_df["forecast_date"].dt.is_month_end).astype(
        int
    )

    # Operational
    tomorrow_df["vendor_delay_flag"] = 0

    tomorrow_df["heavy_rain_flag"] = 0

    tomorrow_df["system_outage_flag"] = 0

    # Month End Surge
    tomorrow_df["month_end_surge_flag"] = (
        tomorrow_df["forecast_date"].dt.day >= 25
    ).astype(float)

    return tomorrow_df


def predict_tomorrow_logout_demand():

    logger.info("Predicting tomorrow logout demand...")

    (
        _,
        logout_model,
        _,
        logout_features,
    ) = load_models()

    tomorrow_df = build_tomorrow_logout_features()

    # One-Hot Encoding
    model_df = pd.get_dummies(
        tomorrow_df.copy(),
        columns=[
            "hub",
            "transport_shift",
        ],
        drop_first=False,
    )

    # Same Column Cleaning
    model_df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            str(col),
        )
        for col in model_df.columns
    ]

    # Feature Alignment
    for col in logout_features:
        if col not in model_df.columns:
            model_df[col] = 0

    # Same Feature Order
    X_future = model_df[logout_features]

    # Predict
    predictions = logout_model.predict(X_future)

    predictions = np.maximum(predictions, 0).round().astype(int)

    # Build output
    forecast_df = pd.DataFrame(
        {
            "forecast_date": tomorrow_df["forecast_date"],
            "hub": tomorrow_df["hub"],
            "transport_shift": tomorrow_df["transport_shift"],
            "predicted_demand": predictions,
        }
    )

    # Validation
    assert forecast_df["predicted_demand"].min() >= 0

    logger.info(
        "Generated %s logout forecasts",
        len(forecast_df),
    )

    return forecast_df


def export_forecasts():

    login_forecast_df = predict_tomorrow_login_demand()

    logout_forecast_df = predict_tomorrow_logout_demand()

    login_forecast_df.to_csv(
        OUTPUT_DIR / "tomorrow_login_forecast.csv",
        index=False,
    )

    login_forecast_df.to_parquet(
        OUTPUT_DIR / "tomorrow_login_forecast.parquet",
        index=False,
    )

    logout_forecast_df.to_csv(
        OUTPUT_DIR / "tomorrow_logout_forecast.csv",
        index=False,
    )

    logout_forecast_df.to_parquet(
        OUTPUT_DIR / "tomorrow_logout_forecast.parquet",
        index=False,
    )

    logger.info("Forecast files exported successfully.")


def main():

    logger.info("=" * 60)

    logger.info("Starting forecast generation...")

    export_forecasts()

    logger.info("Forecast generation completed.")


if __name__ == "__main__":
    main()
