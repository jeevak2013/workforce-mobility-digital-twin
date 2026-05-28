"""
Enterprise Dispatch Wave Builder

File:
    src/pipeline/dispatch_wave_builder.py

Purpose:
--------
Separates workforce into transport dispatch waves:

    • 03:30 AM Standard Logout Wave
    • 04:30 AM Overtime Extension Wave

Enterprise Enhancements:
------------------------
✅ Female safety conservative thresholding
✅ Defensive NaN prediction handling
✅ Operational resiliency safeguards
✅ Safety-first dispatch orchestration

Outputs:
--------
- wave_0330.csv
- wave_0430.csv
- dispatch_summary.json
- wave_distribution.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# 📁 PATH CONFIGURATION
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
# ⚙ ENTERPRISE DISPATCH THRESHOLDS
# =========================================================

DEFAULT_EXTENSION_THRESHOLD = 0.50

# Conservative female safety threshold
FEMALE_EXTENSION_THRESHOLD = 0.35

# =========================================================
# 📥 LOAD DATASETS
# =========================================================

def load_datasets():

    logger.info(
        "Loading dispatch datasets..."
    )

    prediction_df = pd.read_csv(
        DATA_DIR
        / "employee_extension_predictions.csv"
    )

    employee_df = pd.read_csv(
        DATA_DIR
        / "employee_extension_test.csv"
    )

    logger.info(
        f"Prediction rows: "
        f"{len(prediction_df)}"
    )

    logger.info(
        f"Employee rows: "
        f"{len(employee_df)}"
    )

    return (
        prediction_df,
        employee_df,
    )

# =========================================================
# 🔗 MERGE PREDICTIONS
# =========================================================

def merge_predictions(

    prediction_df: pd.DataFrame,
    employee_df: pd.DataFrame,

) -> pd.DataFrame:
    """
    Merges prediction outputs with
    employee operational data.

    Includes defensive fallback handling
    for missing predictions.
    """

    logger.info(
        "Merging predictions with "
        "employee operational data..."
    )

    merged_df = employee_df.copy()

    # -----------------------------------------------------
    # Merge prediction columns
    # -----------------------------------------------------

    merged_df[
        "prediction_probability"
    ] = prediction_df[
        "prediction_probability"
    ]

    merged_df[
        "predicted_extension"
    ] = prediction_df[
        "predicted_extension"
    ]

    # -----------------------------------------------------
    # Defensive fallback handling
    # -----------------------------------------------------

    logger.info(
        "Applying defensive prediction "
        "fallback handling..."
    )

    # Track missing predictions
    merged_df[
        "prediction_missing_flag"
    ] = (
        merged_df[
            "prediction_probability"
        ].isna()
    )

    missing_count = int(
        merged_df[
            "prediction_missing_flag"
        ].sum()
    )

    logger.info(
        f"Missing predictions detected: "
        f"{missing_count}"
    )

    # Safe operational fallback
    # Default employees to 03:30 wave
    merged_df[
        "prediction_probability"
    ] = (
        merged_df[
            "prediction_probability"
        ]
        .fillna(0.0)
    )

    merged_df[
        "predicted_extension"
    ] = (
        merged_df[
            "predicted_extension"
        ]
        .fillna(0)
        .astype(int)
    )

    logger.info(
        f"Merged rows: "
        f"{len(merged_df)}"
    )

    return merged_df

# =========================================================
# 🌊 BUILD DISPATCH WAVES
# =========================================================

def build_dispatch_waves(
    merged_df: pd.DataFrame,
):
    """
    Creates enterprise workforce
    dispatch pools with:

    ✅ Female safety guardrails
    ✅ Conservative late-night routing
    ✅ Dynamic thresholding
    """

    logger.info(
        "Building enterprise dispatch waves..."
    )

    # -----------------------------------------------------
    # DAILY OPERATIONAL DISPATCH FILTER
    # -----------------------------------------------------

    latest_date = (
        merged_df["date"].max()
    )

    merged_df = merged_df[
        merged_df["date"] == latest_date
    ].copy()

    logger.info(
        f"Building dispatch for date: "
        f"{latest_date}"
    )

    logger.info(
        f"Daily workforce size: "
        f"{len(merged_df)}"
    )

    # -----------------------------------------------------
    # Female Safety Conservative Logic
    # -----------------------------------------------------

    female_extension_mask = (

        (
            merged_df["gender"]
            == "Female"
        )

        &

        (
            merged_df[
                "prediction_probability"
            ]
            >= FEMALE_EXTENSION_THRESHOLD
        )
    )

    # -----------------------------------------------------
    # Standard Threshold Logic
    # -----------------------------------------------------

    non_female_extension_mask = (

        (
            merged_df["gender"]
            != "Female"
        )

        &

        (
            merged_df[
                "prediction_probability"
            ]
            >= DEFAULT_EXTENSION_THRESHOLD
        )
    )

    # -----------------------------------------------------
    # Combined Extension Pool
    # -----------------------------------------------------

    extension_mask = (

        female_extension_mask

        |

        non_female_extension_mask
    )

    # -----------------------------------------------------
    # 04:30 Overtime Dispatch Wave
    # -----------------------------------------------------

    wave_0430 = merged_df[
        extension_mask
    ].copy()

    wave_0430[
        "dispatch_wave"
    ] = "04:30"

    # -----------------------------------------------------
    # 03:30 Standard Dispatch Wave
    # -----------------------------------------------------

    wave_0330 = merged_df.drop(
        wave_0430.index
    ).copy()

    wave_0330[
        "dispatch_wave"
    ] = "03:30"

    # -----------------------------------------------------
    # Safety Analytics
    # -----------------------------------------------------

    female_0430_count = int(
        (
            wave_0430["gender"]
            == "Female"
        ).sum()
    )

    logger.info(
        f"Female employees routed "
        f"to 04:30 wave: "
        f"{female_0430_count}"
    )

    logger.info(
        f"03:30 wave employees: "
        f"{len(wave_0330)}"
    )

    logger.info(
        f"04:30 wave employees: "
        f"{len(wave_0430)}"
    )

    return (
        wave_0330,
        wave_0430,
    )

# =========================================================
# 📊 OPERATIONAL SUMMARY
# =========================================================

def generate_operational_summary(
    wave_0330: pd.DataFrame,
    wave_0430: pd.DataFrame,
):
    """
    Generates enterprise dispatch KPIs.
    """

    logger.info(
        "Generating operational summary..."
    )

    summary = {

        # -------------------------------------------------
        # Workforce Volumes
        # -------------------------------------------------

        "wave_0330_employee_count":
            int(len(wave_0330)),

        "wave_0430_employee_count":
            int(len(wave_0430)),

        # -------------------------------------------------
        # Female Workforce Ratios
        # -------------------------------------------------

        "wave_0330_female_ratio":
            round(
                (
                    wave_0330["gender"]
                    == "Female"
                ).mean(),
                4,
            ),

        "wave_0430_female_ratio":
            round(
                (
                    wave_0430["gender"]
                    == "Female"
                ).mean(),
                4,
            ),

        # -------------------------------------------------
        # Escort Requirements
        # -------------------------------------------------

        "wave_0330_escort_required":
            int(
                wave_0330[
                    "requires_security_escort"
                ].sum()
            ),

        "wave_0430_escort_required":
            int(
                wave_0430[
                    "requires_security_escort"
                ].sum()
            ),

        # -------------------------------------------------
        # Missing Prediction Monitoring
        # -------------------------------------------------

        "wave_0330_missing_predictions":
            int(
                wave_0330[
                    "prediction_missing_flag"
                ].sum()
            ),

        "wave_0430_missing_predictions":
            int(
                wave_0430[
                    "prediction_missing_flag"
                ].sum()
            ),

        # -------------------------------------------------
        # Operational Fleet Planning
        # -------------------------------------------------

        "estimated_cabs_wave_0330":
            int(
                len(wave_0330) / 4
            ) + 1,

        "estimated_cabs_wave_0430":
            int(
                len(wave_0430) / 4
            ) + 1,
    }

    return summary

# =========================================================
# 📈 VISUALIZATION
# =========================================================

def export_visualizations(
    wave_0330: pd.DataFrame,
    wave_0430: pd.DataFrame,
):
    """
    Exports workforce wave visualization.
    """

    logger.info(
        "Exporting wave visualization..."
    )

    labels = [
        "03:30 Wave",
        "04:30 Wave",
    ]

    values = [
        len(wave_0330),
        len(wave_0430),
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Employee Count"
    )

    plt.title(
        "Enterprise Dispatch Wave Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        DATA_DIR
        / "wave_distribution.png"
    )

    plt.close()

# =========================================================
# 📤 EXPORT OUTPUTS
# =========================================================

def export_outputs(
    wave_0330: pd.DataFrame,
    wave_0430: pd.DataFrame,
    summary: dict,
):
    """
    Exports dispatch orchestration outputs.
    """

    logger.info(
        "Exporting dispatch outputs..."
    )

    # -----------------------------------------------------
    # Export Wave Files
    # -----------------------------------------------------

    wave_0330.to_csv(
        DATA_DIR
        / "wave_0330.csv",
        index=False,
    )

    wave_0430.to_csv(
        DATA_DIR
        / "wave_0430.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Export Operational Summary
    # -----------------------------------------------------

    with open(
        DATA_DIR
        / "dispatch_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    logger.info(
        "Dispatch outputs exported."
    )

# =========================================================
# 🚀 MAIN PIPELINE
# =========================================================

def main():

    logger.info("=" * 60)

    logger.info(
        "Starting enterprise dispatch "
        "wave builder..."
    )

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    (
        prediction_df,
        employee_df,
    ) = load_datasets()

    # -----------------------------------------------------
    # Merge predictions
    # -----------------------------------------------------

    merged_df = merge_predictions(
        prediction_df,
        employee_df,
    )

    # -----------------------------------------------------
    # Build dispatch waves
    # -----------------------------------------------------

    (
        wave_0330,
        wave_0430,
    ) = build_dispatch_waves(
        merged_df
    )

    # -----------------------------------------------------
    # Generate operational summary
    # -----------------------------------------------------

    summary = (
        generate_operational_summary(
            wave_0330,
            wave_0430,
        )
    )

    # -----------------------------------------------------
    # Export visualization
    # -----------------------------------------------------

    export_visualizations(
        wave_0330,
        wave_0430,
    )

    # -----------------------------------------------------
    # Export outputs
    # -----------------------------------------------------

    export_outputs(
        wave_0330,
        wave_0430,
        summary,
    )

    logger.info(
        "Enterprise dispatch wave "
        "builder completed."
    )

    print(
        "\n🚀 Enterprise dispatch wave "
        "generation completed successfully.\n"
    )

# =========================================================
# 🚀 ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()