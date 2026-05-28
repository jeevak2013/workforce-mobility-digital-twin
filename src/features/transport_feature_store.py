"""
Enterprise Transport Feature Store

File:
    src/features/transport_feature_store.py

Purpose:
- Centralized enterprise feature engineering layer
- ML-ready transport intelligence dataset generation
- Workforce mobility feature derivation
- Temporal + operational + geospatial feature creation

Outputs:
- transport_features.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

INPUT_DATA_DIR = (
    BASE_DIR / "data"
)

OUTPUT_FEATURE_DIR = (
    BASE_DIR / "feature_data"
)

OUTPUT_FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_OUTPUT = (
    OUTPUT_FEATURE_DIR
    / "transport_features.parquet"
)

LOG_FILE = (
    OUTPUT_FEATURE_DIR
    / "feature_store.log"
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
# ENCODING MAPS
# =========================================================

TRANSPORT_SHIFT_ENCODING = {
    "03:30": 1,
    "04:30": 2,
    "NON_TRANSPORT": 0,
}

ROUTE_RISK_ENCODING = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}

EXTENSION_ENCODING = {
    "NO_EXTENSION": 0,
    "EXTEND_TO_0430": 1,
    "EXTEND_BEYOND_0630": 2,
}

# =========================================================
# FEATURE ENGINEERING
# =========================================================


def create_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enterprise temporal feature derivation.
    """

    logger.info(
        "Creating temporal features..."
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["actual_logout"] = pd.to_datetime(
        df["actual_logout"]
    )

    # -----------------------------------------------------
    # Calendar Features
    # -----------------------------------------------------

    df["day_of_week"] = (
        df["date"]
        .dt.dayofweek
    )

    df["day_name"] = (
        df["date"]
        .dt.day_name()
    )

    df["month"] = (
        df["date"]
        .dt.month
    )

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["week_of_month"] = (
        (
            df["date"]
            .dt.day
            - 1
        ) // 7
    ) + 1

    # -----------------------------------------------------
    # Weekend / Month-End Intelligence
    # -----------------------------------------------------

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_month_end"] = (
        df["date"]
        .dt.is_month_end
    ).astype(int)

    # -----------------------------------------------------
    # Time Intelligence
    # -----------------------------------------------------

    df["logout_hour"] = (
        df["actual_logout"]
        .dt.hour
    )

    df["logout_minute"] = (
        df["actual_logout"]
        .dt.minute
    )

    df["minutes_since_midnight"] = (
        (
            df["logout_hour"] * 60
        )
        + df["logout_minute"]
    )

    return df

# =========================================================
# WORKFORCE FEATURES
# =========================================================


def create_workforce_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Workforce behavioral intelligence.
    """

    logger.info(
        "Creating workforce features..."
    )

    # -----------------------------------------------------
    # Binary Flags
    # -----------------------------------------------------

    df["is_transport_user"] = (
        df["uses_company_transport"]
        .astype(int)
    )

    df["escort_required"] = (
        df["requires_security_escort"]
        .astype(int)
    )

    df["is_female"] = (
        df["gender"]
        == "Female"
    ).astype(int)

    # -----------------------------------------------------
    # Extension Intelligence
    # -----------------------------------------------------

    df["extension_flag"] = (
        df["extension_category"]
        != "NO_EXTENSION"
    ).astype(int)

    df["high_extension_flag"] = (
        df["extension_category"]
        == "EXTEND_BEYOND_0630"
    ).astype(int)

    # -----------------------------------------------------
    # Shift Encoding
    # -----------------------------------------------------

    df["transport_shift_encoded"] = (
        df["transport_shift"]
        .map(
            TRANSPORT_SHIFT_ENCODING
        )
    )

    df["extension_category_encoded"] = (
        df["extension_category"]
        .map(
            EXTENSION_ENCODING
        )
    )

    return df

# =========================================================
# OPERATIONAL FEATURES
# =========================================================


def create_operational_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Operational transport intelligence.
    """

    logger.info(
        "Creating operational features..."
    )

    # -----------------------------------------------------
    # Operational Event Flags
    # -----------------------------------------------------

    df["vendor_delay_flag"] = (
        df["operational_event"]
        == "VENDOR_DELAY"
    ).astype(int)

    df["heavy_rain_flag"] = (
        df["operational_event"]
        == "HEAVY_RAIN"
    ).astype(int)

    df["month_end_surge_flag"] = (
        df["operational_event"]
        == "MONTH_END_SURGE"
    ).astype(int)

    df["system_outage_flag"] = (
        df["operational_event"]
        == "SYSTEM_OUTAGE"
    ).astype(int)

    # -----------------------------------------------------
    # Route Risk
    # -----------------------------------------------------

    df["route_risk_encoded"] = (
        df["route_risk_level"]
        .map(
            ROUTE_RISK_ENCODING
        )
    )

    return df

# =========================================================
# GEOSPATIAL FEATURES
# =========================================================


def create_geospatial_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Geospatial mobility intelligence.
    """

    logger.info(
        "Creating geospatial features..."
    )

    # -----------------------------------------------------
    # Distance Intelligence
    # -----------------------------------------------------

    df["long_distance_flag"] = (
        df["home_distance_km"]
        > 18
    ).astype(int)

    df["extreme_distance_flag"] = (
        df["home_distance_km"]
        > 25
    ).astype(int)

    # -----------------------------------------------------
    # Hub Density Features
    # -----------------------------------------------------

    hub_density = (
        df.groupby("hub")
        ["employee_id"]
        .transform("count")
    )

    df["hub_density"] = (
        hub_density
    )

    return df

# =========================================================
# TRANSPORT LOAD FEATURES
# =========================================================


def create_transport_load_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enterprise dispatch-load intelligence.
    """

    logger.info(
        "Creating transport load features..."
    )

    # -----------------------------------------------------
    # Daily Transport Demand
    # -----------------------------------------------------

    daily_transport_load = (
        df.groupby(
            [
                "date",
                "transport_shift",
            ]
        )
        ["employee_id"]
        .transform("count")
    )

    df["daily_transport_load"] = (
        daily_transport_load
    )

    # -----------------------------------------------------
    # Hub + Shift Load
    # -----------------------------------------------------

    hub_shift_load = (
        df.groupby(
            [
                "date",
                "hub",
                "transport_shift",
            ]
        )
        ["employee_id"]
        .transform("count")
    )

    df["hub_shift_load"] = (
        hub_shift_load
    )

    return df

# =========================================================
# FORECASTING INTELLIGENCE FEATURES
# =========================================================

def create_forecasting_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enterprise forecasting intelligence layer.

    Adds:
    - lag features
    - rolling averages
    - growth trends
    - historical operational persistence
    """

    logger.info(
        "Creating forecasting intelligence features..."
    )

    # -----------------------------------------------------
    # SORTING
    # -----------------------------------------------------

    df = df.sort_values(
        by=[
            "date",
            "transport_shift",
            "hub",
        ]
    )

    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------

    logger.info(
        "Creating lag features..."
    )

    group_cols = [
        "transport_shift",
        "hub",
    ]

    # Previous day transport load
    df["prev_day_transport_load"] = (
        df.groupby(group_cols)[
            "daily_transport_load"
        ]
        .shift(1)
    )

    # Previous week same corridor load
    df["prev_week_transport_load"] = (
        df.groupby(group_cols)[
            "daily_transport_load"
        ]
        .shift(7)
    )

    # Previous hub shift load
    df["prev_hub_shift_load"] = (
        df.groupby(group_cols)[
            "hub_shift_load"
        ]
        .shift(1)
    )

    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------

    logger.info(
        "Creating rolling window features..."
    )

    df["rolling_7d_avg_load"] = (
        df.groupby(group_cols)[
            "daily_transport_load"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    df["rolling_14d_avg_load"] = (
        df.groupby(group_cols)[
            "daily_transport_load"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=14,
                min_periods=1,
            ).mean()
        )
    )

    df["rolling_30d_avg_load"] = (
        df.groupby(group_cols)[
            "daily_transport_load"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=30,
                min_periods=1,
            ).mean()
        )
    )

    # -----------------------------------------------------
    # TREND FEATURES
    # -----------------------------------------------------

    logger.info(
        "Creating trend intelligence..."
    )

    df["load_growth_rate"] = (
        (
            df["daily_transport_load"]
            - df[
                "prev_day_transport_load"
            ]
        )
        / (
            df[
                "prev_day_transport_load"
            ]
            + 1
        )
    )

    df["hub_load_growth_rate"] = (
        (
            df["hub_shift_load"]
            - df[
                "prev_hub_shift_load"
            ]
        )
        / (
            df[
                "prev_hub_shift_load"
            ]
            + 1
        )
    )

    # -----------------------------------------------------
    # HISTORICAL OPERATIONAL FEATURES
    # -----------------------------------------------------

    logger.info(
        "Creating operational persistence features..."
    )

    # Historical vendor delay persistence
    df["rolling_vendor_delay_rate"] = (
        df.groupby(group_cols)[
            "vendor_delay_flag"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    # Historical heavy rain persistence
    df["rolling_rain_rate"] = (
        df.groupby(group_cols)[
            "heavy_rain_flag"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    # Historical overtime persistence
    df["rolling_extension_rate"] = (
        df.groupby(group_cols)[
            "extension_flag"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    # -----------------------------------------------------
    # NULL HANDLING
    # -----------------------------------------------------

    forecasting_cols = [

        "prev_day_transport_load",

        "prev_week_transport_load",

        "prev_hub_shift_load",

        "rolling_7d_avg_load",

        "rolling_14d_avg_load",

        "rolling_30d_avg_load",

        "load_growth_rate",

        "hub_load_growth_rate",

        "rolling_vendor_delay_rate",

        "rolling_rain_rate",

        "rolling_extension_rate",
    ]

    for col in forecasting_cols:

        df[col] = (
            df[col]
            .fillna(0)
        )

    return df

# =========================================================
# MAIN FEATURE PIPELINE
# =========================================================


def build_transport_feature_store(
) -> pd.DataFrame:
    """
    Enterprise transport feature store builder.
    """

    logger.info("=" * 60)

    logger.info(
        "Building enterprise "
        "transport feature store..."
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    activity_logs_path = (
        INPUT_DATA_DIR
        / "activity_logs.parquet"
    )

    logger.info(
        "Loading activity logs..."
    )

    df = pd.read_parquet(
        activity_logs_path
    )

    # -----------------------------------------------------
    # LOAD EMPLOYEE MASTER DATA
    # -----------------------------------------------------

    employees_path = (
        INPUT_DATA_DIR
        / "employees_after_churn.csv"
    )

    logger.info(
        "Loading employee master data..."
    )

    employees_df = pd.read_csv(
        employees_path
    )

    # -----------------------------------------------------
    # ENTERPRISE DATA ENRICHMENT
    # -----------------------------------------------------

    logger.info(
        "Merging workforce attributes..."
    )

    employee_columns = [

        "employee_id",

        "gender",

        "hub",

        "pickup_hub",

        "safety_priority_score",

        "home_lat",

        "home_lon",
    ]

    df = df.merge(

        employees_df[
            employee_columns
        ],

        on="employee_id",

        how="left",
    )

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    df = create_temporal_features(df)

    df = create_workforce_features(df)

    df = create_operational_features(df)

    df = create_geospatial_features(df)

    df = create_transport_load_features(df)

    df = create_forecasting_features(df)

    # -----------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------

    logger.info(
        "Optimizing dataframe..."
    )

    # Fill missing encoded values safely
    encoded_columns = [

        "transport_shift_encoded",

        "extension_category_encoded",

        "route_risk_encoded",
    ]

    for col in encoded_columns:

        df[col] = (
            df[col]
            .fillna(0)
            .astype(int)
        )

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    logger.info(
        "Exporting feature store..."
    )

    df.to_parquet(
        FEATURE_OUTPUT,
        index=False,
    )

    df.to_csv(
        FEATURE_OUTPUT.with_suffix(".csv"),
        index=False,
        encoding="utf-8",
    )    

    logger.info(
        f"Feature store exported: "
        f"{FEATURE_OUTPUT.name}"
    )

    logger.info(
        f"Total engineered features: "
        f"{len(df.columns)}"
    )

    logger.info("=" * 60)

    return df

# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    build_transport_feature_store()