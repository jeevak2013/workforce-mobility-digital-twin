"""
Enterprise Employee Extension Feature Engineering Pipeline

File:
    src/pipeline/employee_extension_features.py

Purpose:
--------
Generate employee-level behavioral forecasting features
for predicting WHICH employees will extend overtime.

Forecasting Objective:
----------------------
Binary Classification:
    extended = 1 if employee logout == 04:30:00
    extended = 0 otherwise

Outputs:
--------
- employee_extension_train.csv
- employee_extension_test.csv
- employee_extension_metadata.json

Business Value:
---------------
Supports:
- cab allocation
- route optimization
- female safety routing
- dynamic dispatch planning
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

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

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
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)

# =========================================================
# 📥 DATA LOADING
# =========================================================

def load_datasets():

    logger.info(
        "Loading datasets..."
    )

    logs_df = pd.read_csv(
        DATA_DIR / "historical_roster_logs.csv"
    )

    logs_df.columns = [

        col.strip()
        .lower()
        .replace(" ", "_")

        for col in logs_df.columns
    ]    

    employee_df = pd.read_csv(
        DATA_DIR / "employees.csv"
    )

    # -----------------------------------------------------
    # Standardize column names
    # -----------------------------------------------------

    employee_df.columns = [

        col.strip()
        .lower()
        .replace(" ", "_")

        for col in employee_df.columns
    ]

    logs_df["date"] = pd.to_datetime(
        logs_df["date"]
    )

    logger.info(
        f"Activity logs rows: "
        f"{len(logs_df)}"
    )

    logger.info(
        f"Employee master rows: "
        f"{len(employee_df)}"
    )

    return logs_df, employee_df


# =========================================================
# 🎯 TARGET ENGINEERING
# =========================================================

def create_target_variable(
    logs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates employee-level extension target.
    """

    logger.info(
        "Generating extension target..."
    )

    logs_df["extended"] = np.where(
        logs_df["actual_logout"]
        == "04:30:00",
        1,
        0,
    )

    extension_rate = (
        logs_df["extended"]
        .mean()
    )

    logger.info(
        f"Extension rate: "
        f"{extension_rate:.2%}"
    )

    return logs_df


# =========================================================
# 📅 TEMPORAL FEATURES
# =========================================================

def build_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates calendar-based operational features.
    """

    logger.info(
        "Generating temporal features..."
    )

    df["day_of_week"] = (
        df["date"]
        .dt.dayofweek
    )

    df["day_of_month"] = (
        df["date"]
        .dt.day
    )

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["month"] = (
        df["date"]
        .dt.month
    )

    df["is_month_end"] = (
        df["date"]
        .dt.is_month_end
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    return df


# =========================================================
# 🔄 CYCLICAL ENCODING
# =========================================================

def build_cyclical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates cyclical time encodings.
    """

    logger.info(
        "Generating cyclical features..."
    )

    df["dow_sin"] = np.sin(
        2 * np.pi
        * df["day_of_week"]
        / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi
        * df["day_of_week"]
        / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi
        * df["month"]
        / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi
        * df["month"]
        / 12
    )

    return df


# =========================================================
# 👤 EMPLOYEE BEHAVIOR FEATURES
# =========================================================

def build_employee_behavior_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates employee historical behavior signals.
    """

    logger.info(
        "Generating employee behavior features..."
    )

    df = df.sort_values(
        by=[
            "employee_id",
            "date",
        ]
    )

    # -----------------------------------------------------
    # Historical extension persistence
    # -----------------------------------------------------

    df["employee_last_3d_extension_rate"] = (

        df.groupby("employee_id")[
            "extended"
        ]

        .transform(

            lambda x:
            x.shift(1)
            .rolling(
                window=3,
                min_periods=1,
            )
            .mean()
        )
    )

    df["employee_last_7d_extension_rate"] = (

        df.groupby("employee_id")[
            "extended"
        ]

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

    df["employee_last_7d_extension_count"] = (

        df.groupby("employee_id")[
            "extended"
        ]

        .transform(

            lambda x:
            x.shift(1)
            .rolling(
                window=7,
                min_periods=1,
            )
            .sum()
        )
    )

    return df


# =========================================================
# 🏢 OPERATIONAL PRESSURE FEATURES
# =========================================================

def build_operational_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates operational workload indicators.
    """

    logger.info(
        "Generating operational features..."
    )

    # Daily overtime load
    daily_extension_counts = (

        df.groupby("date")[
            "extended"
        ]

        .sum()

        .sort_index()
    )

    daily_extension_counts = (

        daily_extension_counts
        .shift(1)
        .fillna(0)
    )

    daily_extension_counts = (
        daily_extension_counts
        .rename(
            "previous_day_extension_load"
        )
    )

    df = df.merge(
        daily_extension_counts,
        on="date",
        how="left",
    )
    # Weekend staffing proxy
    df["expected_staffing_pressure"] = np.where(
        df["is_weekend"] == 1,
        0.25,
        1.0,
    )

    return df


# =========================================================
# 👥 EMPLOYEE MASTER ENRICHMENT
# =========================================================

def merge_employee_master(
    df: pd.DataFrame,
    employee_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges employee enrichment features
    without creating duplicate columns.
    """

    logger.info(
        "Merging employee master data..."
    )

    # -------------------------------------------------
    # Select ONLY enrichment fields
    # -------------------------------------------------

    employee_features = [

        "employee_id",

        "gender",

        "tenure_days",

        "safety_priority_score",

        "requires_security_escort",
    ]

    employee_df = employee_df[
        employee_features
    ]

    # -------------------------------------------------
    # Merge enrichment data
    # -------------------------------------------------

    df = df.merge(
        employee_df,
        on="employee_id",
        how="left",
    )

    return df

# =========================================================
# 🧹 CLEANING / IMPUTATION
# =========================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applies enterprise-safe imputations.
    """

    logger.info(
        "Cleaning feature matrix..."
    )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (

        df[numeric_columns]

        .fillna(0)
    )

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    df[categorical_columns] = (
        df[categorical_columns]
        .fillna("Unknown")
    )

    logger.info(
        f"Final rows: {len(df)}"
    )

    logger.info(
        f"Final columns: {len(df.columns)}"
    )

    return df


# =========================================================
# ✂ CHRONOLOGICAL SPLIT
# =========================================================

def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
):
    """
    Chronological split preserving temporal order.
    """

    logger.info(
        "Performing chronological split..."
    )

    unique_dates = sorted(
        df["date"].unique()
    )

    split_index = int(
        len(unique_dates)
        * train_ratio
    )

    train_dates = unique_dates[
        :split_index
    ]

    test_dates = unique_dates[
        split_index:
    ]

    train_df = df[
        df["date"].isin(train_dates)
    ].copy()

    test_df = df[
        df["date"].isin(test_dates)
    ].copy()

    logger.info(
        f"Train rows: {len(train_df)}"
    )

    logger.info(
        f"Test rows: {len(test_df)}"
    )

    return train_df, test_df


# =========================================================
# 📤 EXPORT ENGINE
# =========================================================

def export_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Exports employee-level classification datasets.
    """

    logger.info(
        "Exporting employee extension datasets..."
    )

    train_df.to_csv(
        DATA_DIR
        / "employee_extension_train.csv",
        index=False,
    )

    test_df.to_csv(
        DATA_DIR
        / "employee_extension_test.csv",
        index=False,
    )

    metadata = {

        "problem_type":
            "binary_classification",

        "target_column":
            "extended",

        "train_rows":
            int(len(train_df)),

        "test_rows":
            int(len(test_df)),

        "feature_count":
            int(len(train_df.columns)),

        "objective":
            (
                "Predict which employees "
                "will extend overtime"
            ),
    }

    with open(
        DATA_DIR
        / "employee_extension_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    logger.info(
        "Employee extension datasets exported."
    )


# =========================================================
# 🚀 MAIN PIPELINE
# =========================================================

def main() -> None:

    logger.info("=" * 60)

    logger.info(
        "Starting employee extension "
        "feature engineering..."
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    logs_df, employee_df = (
        load_datasets()
    )

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    logs_df = create_target_variable(
        logs_df
    )

    # -----------------------------------------------------
    # Temporal
    # -----------------------------------------------------

    logs_df = build_temporal_features(
        logs_df
    )

    # -----------------------------------------------------
    # Cyclical
    # -----------------------------------------------------

    logs_df = build_cyclical_features(
        logs_df
    )

    # -----------------------------------------------------
    # Employee behavior
    # -----------------------------------------------------

    logs_df = (
        build_employee_behavior_features(
            logs_df
        )
    )

    # -----------------------------------------------------
    # Operational
    # -----------------------------------------------------

    logs_df = build_operational_features(
        logs_df
    )

    # -----------------------------------------------------
    # Employee enrichment
    # -----------------------------------------------------

    logs_df = merge_employee_master(
        logs_df,
        employee_df,
    )

    # -----------------------------------------------------
    # Cleaning
    # -----------------------------------------------------

    logs_df = clean_dataset(
        logs_df
    )

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    train_df, test_df = split_train_test(
        logs_df
    )

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------

    export_datasets(
        train_df,
        test_df,
    )

    logger.info(
        "Employee extension feature "
        "engineering completed."
    )

    print(
        "\n🚀 Employee extension feature "
        "datasets generated successfully.\n"
    )


# =========================================================
# 🚀 ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()