"""
Enterprise Workforce Mobility Simulation Pipeline

File:
    src/pipeline/generate_synthetic.py

Purpose:
- Enterprise workforce orchestration
- Mobility digital twin generation
- Transport activity simulation
- Churn-aware workforce evolution
- Pricing and fleet generation
- Dispatch-ready dataset generation
- Enterprise validation orchestration
"""

from __future__ import annotations

import gc
import json
import logging
import random
import sys
import time

from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# ENTERPRISE MODULE IMPORTS
# =========================================================

from src.data_generation.employee_generator import (
    generate_initial_employees,
)

from src.simulation.churn_simulator import (
    simulate_employee_churn,
)

from src.data_generation.activity_logs import (
    generate_activity_logs,
)

from src.optimization.pricing_engine import (
    fleet_registry_dataframe,
    vendor_registry_dataframe,
)

from src.pipeline.verify import (
    verify_activity_logs,
    verify_employees,
    verify_pricing,
)

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = (
    BASE_DIR / "data"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = (
    DATA_DIR / "pipeline.log"
)

# =========================================================
# ENTERPRISE LOGGING
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
            sys.stdout
        ),
    ],
)

logger = logging.getLogger(__name__)

# =========================================================
# GLOBAL SEED
# =========================================================

SEED = 42

# =========================================================
# PIPELINE TIMER
# =========================================================


def timed_stage(stage_name: str):

    class _Timer:

        def __enter__(self):

            self.start = time.time()

            logger.info(
                f"STARTED: {stage_name}"
            )

            return self

        def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb,
        ):

            duration = (
                time.time()
                - self.start
            )

            logger.info(
                f"COMPLETED: {stage_name} "
                f"({duration:.2f}s)"
            )

    return _Timer()

# =========================================================
# DATAFRAME VALIDATION
# =========================================================


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
) -> None:
    """
    Enterprise dataframe validation.
    """

    if df.empty:

        raise ValueError(
            f"{name} dataframe is empty."
        )

    if (
        df.isnull()
        .sum()
        .sum()
        > 0
    ):

        logger.warning(
            f"{name} contains null values."
        )

# =========================================================
# EXPORT ENGINE
# =========================================================


def export_dataframe(
    df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Enterprise CSV export utility.
    """

    out_path = (
        DATA_DIR / filename
    )

    df.to_csv(
        out_path,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        f"Exported: {filename}"
    )

# =========================================================
# PIPELINE METADATA
# =========================================================


def generate_metadata(
    employees_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
) -> dict:
    """
    Enterprise pipeline metadata.
    """

    return {

        "generated_at":
            datetime.now(UTC)
            .isoformat(),

        "seed":
            SEED,

        "employee_count":
            int(
                len(employees_df)
            ),

        "transport_users":
            int(
                employees_df[
                    "uses_company_transport"
                ].sum()
            ),

        "activity_log_rows":
            int(
                len(activity_df)
            ),

        "fleet_count":
            int(
                len(fleet_df)
            ),

        "vendor_count":
            int(
                len(pricing_df)
            ),

        "dispatch_waves":

            activity_df[
                "actual_dispatch_wave"
            ]
            .value_counts()
            .to_dict(),
    }

# =========================================================
# MAIN ORCHESTRATOR
# =========================================================


def main() -> None:

    logger.info("=" * 60)

    logger.info(
        "Initializing Enterprise "
        "Mobility Simulation Pipeline"
    )

    random.seed(SEED)

    np.random.seed(SEED)

    pipeline_start = time.time()

    try:

        # =================================================
        # EMPLOYEE GENERATION
        # =================================================

        with timed_stage(
            "Employee Generation"
        ):

            employees_df = (
                generate_initial_employees(
                    total_employees=2000,
                    transport_users=1000,
                    seed=SEED,
                )
            )

            validate_dataframe(
                employees_df,
                "employees",
            )

            export_dataframe(
                employees_df,
                "employees.csv",
            )

        # =================================================
        # CHURN SIMULATION
        # =================================================

        with timed_stage(
            "Churn Simulation"
        ):

            employees_after_churn = (
                simulate_employee_churn(
                    employees_df=employees_df,
                    monthly_churn_rate=0.08,
                    seed=SEED,
                )
            )

            validate_dataframe(
                employees_after_churn,
                "employees_after_churn",
            )

            export_dataframe(
                employees_after_churn,
                "employees_after_churn.csv",
            )

        # =================================================
        # ACTIVITY STREAM GENERATION
        # =================================================

        with timed_stage(
            "Activity Stream Generation"
        ):

            generate_activity_logs(

                employees_csv=str(
                    DATA_DIR
                    / "employees_after_churn.csv"
                ),

                output_csv=str(
                    DATA_DIR
                    / "activity_logs.csv"
                ),

                start_date="2025-01-01",

                days=365,

                seed=SEED,
            )

            activity_logs_df = (
                pd.read_csv(
                    DATA_DIR
                    / "activity_logs.csv"
                )
            )

            validate_dataframe(
                activity_logs_df,
                "activity_logs",
            )

        # =================================================
        # FLEET REGISTRY
        # =================================================

        with timed_stage(
            "Fleet Registry Generation"
        ):

            fleet_df = (
                fleet_registry_dataframe()
            )

            export_dataframe(
                fleet_df,
                "fleet_registry.csv",
            )

        # =================================================
        # VENDOR PRICING
        # =================================================

        with timed_stage(
            "Vendor Pricing Generation"
        ):

            pricing_df = (
                vendor_registry_dataframe()
            )

            export_dataframe(
                pricing_df,
                "vendor_pricing.csv",
            )

        # =================================================
        # ENTERPRISE VERIFICATION
        # =================================================

        with timed_stage(
            "Enterprise Verification"
        ):

            verify_employees()

            verify_activity_logs()

            verify_pricing()

        # =================================================
        # PIPELINE METADATA
        # =================================================

        metadata = generate_metadata(

            employees_df=
                employees_after_churn,

            activity_df=
                activity_logs_df,

            fleet_df=
                fleet_df,

            pricing_df=
                pricing_df,
        )

        metadata_path = (
            DATA_DIR
            / "pipeline_metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        # =================================================
        # MEMORY CLEANUP
        # =================================================

        del employees_df
        del employees_after_churn
        del activity_logs_df
        del fleet_df
        del pricing_df

        gc.collect()

        # =================================================
        # PIPELINE SUCCESS
        # =================================================

        duration = (
            time.time()
            - pipeline_start
        )

        logger.info(
            f"Pipeline completed successfully "
            f"in {duration:.2f} seconds."
        )

        print(
            "\nEnterprise workforce mobility "
            "pipeline completed successfully.\n"
        )

    except Exception as exc:

        logger.exception(
            "Pipeline execution failed."
        )

        print(
            "\nPipeline execution failed.\n"
            f"Check logs: {LOG_FILE}\n"
        )

        raise exc

# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()