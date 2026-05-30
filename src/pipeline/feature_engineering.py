"""
Enterprise Feature Engineering Pipeline

File:
    src/pipeline/feature_engineering.py

Purpose:
- Transform activity logs into forecasting-ready datasets
- Engineer temporal lag features
- Generate rolling statistical signals
- Create train/test chronological splits
- Support enterprise demand forecasting pipelines

Forecasting Objective:
----------------------
Predict:
    Daily 04:30 overtime employee demand

Key Improvements:
-----------------
- Weekend operational handling
- Defensive pct_change engineering
- Temporal robustness
- Enterprise-safe rolling windows
- Better observability

Outputs:
--------
- train_features.csv
- test_features.csv
- feature_metadata.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 📁 PROJECT PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "synthetic_data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# 📝 LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(message)s"),
)

logger = logging.getLogger(__name__)

# =========================================================
# 📊 FEATURE ENGINEERING ENGINE
# =========================================================


def build_forecasting_features() -> pd.DataFrame:
    """
    Creates enterprise-grade forecasting features
    from historical activity logs.
    """

    logger.info("Loading historical activity logs...")

    df = pd.read_csv(DATA_DIR / "historical_roster_logs.csv")

    # -----------------------------------------------------
    # DATE NORMALIZATION
    # -----------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])

    # =====================================================
    # TARGET ENGINEERING
    # =====================================================

    logger.info("Generating daily overtime targets...")

    overtime_df = (
        df[df["actual_logout"] == "04:30:00"]
        .groupby("date")
        .size()
        .reset_index(name="Y")
    )

    # =====================================================
    # CALENDAR FEATURES
    # =====================================================

    logger.info("Generating calendar features...")

    overtime_df["day_of_week"] = overtime_df["date"].dt.dayofweek

    overtime_df["day_of_month"] = overtime_df["date"].dt.day

    overtime_df["week_of_year"] = overtime_df["date"].dt.isocalendar().week.astype(int)

    overtime_df["month"] = overtime_df["date"].dt.month

    overtime_df["is_month_end"] = overtime_df["date"].dt.is_month_end.astype(int)

    # =====================================================
    # WEEKEND / HOLIDAY STYLE FEATURES
    # =====================================================

    logger.info("Generating weekend operational features...")

    overtime_df["is_weekend"] = overtime_df["day_of_week"].isin([5, 6]).astype(int)

    # Weekend staffing reduction proxy
    overtime_df["expected_staffing_pressure"] = np.where(
        overtime_df["is_weekend"] == 1,
        0.25,  # ~50-100 employees
        1.0,  # ~1000 employees
    )

    # =====================================================
    # CYCLICAL ENCODING
    # =====================================================

    logger.info("Generating cyclical temporal features...")

    overtime_df["dow_sin"] = np.sin(2 * np.pi * overtime_df["day_of_week"] / 7)

    overtime_df["dow_cos"] = np.cos(2 * np.pi * overtime_df["day_of_week"] / 7)

    overtime_df["month_sin"] = np.sin(2 * np.pi * overtime_df["month"] / 12)

    overtime_df["month_cos"] = np.cos(2 * np.pi * overtime_df["month"] / 12)

    # =====================================================
    # LAG FEATURES
    # =====================================================

    logger.info("Generating lag features...")

    overtime_df["Y_lag_1"] = overtime_df["Y"].shift(1)

    overtime_df["Y_lag_3"] = overtime_df["Y"].shift(3)

    overtime_df["Y_lag_7"] = overtime_df["Y"].shift(7)

    # =====================================================
    # ROLLING WINDOW FEATURES
    # =====================================================

    logger.info("Generating rolling statistics...")

    overtime_df["Y_3d_mean"] = (
        overtime_df["Y"]
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
    )

    overtime_df["Y_7d_mean"] = (
        overtime_df["Y"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .mean()
    )

    overtime_df["Y_7d_max"] = (
        overtime_df["Y"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .max()
    )

    overtime_df["Y_7d_min"] = (
        overtime_df["Y"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .min()
    )

    overtime_df["Y_7d_std"] = (
        overtime_df["Y"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .std()
    )

    # =====================================================
    # TREND FEATURES
    # =====================================================

    logger.info("Generating trend features...")

    overtime_df["Y_diff_1"] = overtime_df["Y"].diff(1)

    # -----------------------------------------------------
    # Defensive percentage change engineering
    # -----------------------------------------------------

    epsilon = 1e-6

    overtime_df["Y_pct_change"] = overtime_df["Y"].add(epsilon).pct_change()

    overtime_df["Y_pct_change"] = overtime_df["Y_pct_change"].replace(
        [np.inf, -np.inf],
        0,
    )

    # =====================================================
    # IMPUTATION / CLEANUP
    # =====================================================

    logger.info("Applying defensive imputations...")

    numeric_columns = overtime_df.select_dtypes(include=[np.number]).columns

    overtime_df[numeric_columns] = overtime_df[numeric_columns].bfill().fillna(0)

    overtime_df = overtime_df.reset_index(drop=True)

    logger.info(f"Final feature rows: {len(overtime_df)}")

    logger.info(f"Final feature columns: {len(overtime_df.columns)}")

    return overtime_df


# =========================================================
# ✂ CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================


def split_train_test(
    feature_df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological train/test splitting.
    """

    split_index = int(len(feature_df) * train_ratio)

    train_df = feature_df.iloc[:split_index].copy()

    test_df = feature_df.iloc[split_index:].copy()

    logger.info(f"Train rows: {len(train_df)}")

    logger.info(f"Test rows: {len(test_df)}")

    return train_df, test_df


# =========================================================
# 📤 EXPORT ENGINE
# =========================================================


def export_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Exports forecasting datasets.
    """

    logger.info("Exporting engineered datasets...")

    train_df.to_csv(
        DATA_DIR / "train_features.csv",
        index=False,
    )

    test_df.to_csv(
        DATA_DIR / "test_features.csv",
        index=False,
    )

    metadata = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(train_df.columns)),
        "target_column": "Y",
        "forecasting_goal": "Predict overtime demand",
        "weekend_modeling": "Enabled",
        "cyclical_encoding": "Enabled",
        "rolling_statistics": "Enabled",
        "defensive_imputation": "Enabled",
    }

    with open(
        DATA_DIR / "feature_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    logger.info("Feature exports completed.")


# =========================================================
# 🚀 MAIN PIPELINE
# =========================================================


def main() -> None:

    logger.info("=" * 60)

    logger.info("Starting enterprise feature engineering...")

    feature_df = build_forecasting_features()

    train_df, test_df = split_train_test(feature_df)

    export_datasets(
        train_df,
        test_df,
    )

    logger.info("Feature engineering completed successfully.")

    print("\n🚀 Forecasting feature datasets generated successfully.\n")


# =========================================================
# 🚀 ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()
