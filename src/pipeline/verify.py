"""
Enterprise Workforce Mobility Verification Pipeline

File:
    src/pipeline/verify.py

Purpose:
- Enterprise dataset validation
- Mobility orchestration verification
- Transport policy enforcement
- Dispatch realism validation
- Safety integrity checks
- Pricing validation
- Geospatial compliance
- CI/CD verification gate
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config.geography_config import (
    BPO,
)

from src.utils.geo_math import (
    haversine_distance,
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

REPORT_FILE = (
    DATA_DIR
    / "verification_report.json"
)

LOG_FILE = (
    DATA_DIR
    / "verification.log"
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
            sys.stdout
        ),
    ],
)

logger = logging.getLogger(__name__)

# =========================================================
# REPORT CONTAINER
# =========================================================

verification_report = {

    "status": "PASSED",

    "checks": [],

    "failures": [],
}

# =========================================================
# REPORT HELPERS
# =========================================================


def record_success(
    message: str,
) -> None:

    logger.info(message)

    verification_report[
        "checks"
    ].append(message)


def record_failure(
    message: str,
) -> None:

    logger.error(message)

    verification_report[
        "status"
    ] = "FAILED"

    verification_report[
        "failures"
    ].append(message)

# =========================================================
# EMPLOYEE VALIDATION
# =========================================================


def verify_employees() -> bool:

    try:

        df = pd.read_csv(
            DATA_DIR
            / "employees.csv"
        )

        # -------------------------------------------------
        # Schema validation
        # -------------------------------------------------

        required_columns = [

            "employee_id",

            "gender",

            "home_lat",

            "home_lon",

            "hub",

            "pickup_hub",

            "uses_company_transport",

            "transport_shift",

            "transport_eligibility",

            "extension_category",

            "home_distance_km",
        ]

        missing = [

            c

            for c in required_columns

            if c not in df.columns
        ]

        if missing:

            record_failure(
                f"employees.csv missing columns: "
                f"{missing}"
            )

            return False

        # -------------------------------------------------
        # Duplicate validation
        # -------------------------------------------------

        if (
            df["employee_id"]
            .duplicated()
            .any()
        ):

            record_failure(
                "Duplicate employee IDs detected."
            )

            return False

        # -------------------------------------------------
        # Transport user validation
        # -------------------------------------------------

        transport_users = int(
            df[
                "uses_company_transport"
            ].sum()
        )

        if not (
            950
            <= transport_users
            <= 1050
        ):

            record_failure(
                f"Transport user count invalid: "
                f"{transport_users}"
            )

            return False

        # -------------------------------------------------
        # Shift-wave validation
        # -------------------------------------------------

        transport_df = df[
            df[
                "uses_company_transport"
            ]
            == True
        ]

        shift_dist = (
            transport_df[
                "transport_shift"
            ]
            .value_counts(
                normalize=True
            )
        )

        ratio_0330 = shift_dist.get(
            "03:30",
            0.0,
        )

        ratio_0430 = shift_dist.get(
            "04:30",
            0.0,
        )

        if not (
            0.35 <= ratio_0330 <= 0.45
        ):

            record_failure(
                f"03:30 ratio invalid: "
                f"{ratio_0330:.2%}"
            )

            return False

        if not (
            0.55 <= ratio_0430 <= 0.65
        ):

            record_failure(
                f"04:30 ratio invalid: "
                f"{ratio_0430:.2%}"
            )

            return False

        # -------------------------------------------------
        # Geofence validation
        # -------------------------------------------------

        for row in df.itertuples():

            home_lat: float = float(
                str(row.home_lat)
            )

            home_lon: float = float(
                str(row.home_lon)
            )

            distance = (
                haversine_distance(
                    home_lat,
                    home_lon,

                    BPO.location.latitude,
                    BPO.location.longitude,
                )
            )

            if (
                distance
                > (
                    BPO.operational_radius_km
                    + 10
                )
            ):

                record_failure(
                    f"Employee "
                    f"{row.employee_id} "
                    f"outside operational radius."
                )

                return False

        record_success(
            "Employee dataset verification passed."
        )

        return True

    except Exception as exc:

        record_failure(
            f"Employee verification failed: "
            f"{exc}"
        )

        return False

# =========================================================
# ACTIVITY LOG VALIDATION
# =========================================================


def verify_activity_logs() -> bool:

    try:

        df = pd.read_csv(
            DATA_DIR
            / "activity_logs.csv"
        )

        # -------------------------------------------------
        # Dispatch validation
        # -------------------------------------------------

        extended_0330 = df[
            (
                df[
                    "transport_shift"
                ]
                == "03:30"
            )
            &
            (
                df[
                    "extension_category"
                ]
                == "EXTEND_TO_0430"
            )
        ]

        invalid_dispatch = (
            extended_0330[
                extended_0330[
                    "actual_dispatch_wave"
                ]
                != "04:30"
            ]
        )

        if len(invalid_dispatch) > 0:

            record_failure(
                "03:30 extension dispatch "
                "logic violated."
            )

            return False

        # -------------------------------------------------
        # Beyond 06:30 transport-tail validation
        # -------------------------------------------------

        transport_df = df[
            df[
                "uses_company_transport"
            ]
            == True
        ]

        transport_logout_times = (
            pd.to_datetime(
                transport_df[
                    "actual_logout"
                ]
            )
        )

        extreme_tail = transport_df[
            (
                transport_logout_times.dt.time
                > pd.to_datetime(
                    "06:30:00"
                ).time()
            )
        ]

        tail_percentage = (
            len(extreme_tail)
            / max(len(transport_df), 1)
        ) * 100

        print(
            f"Beyond-06:30 transport tail: "
            f"{tail_percentage:.2f}%"
        )

        if (
            len(extreme_tail)
            > len(transport_df) * 0.08
        ):

            record_failure(
                "Excessive beyond-06:30 "
                "transport tail detected."
            )

            return False

        record_success(
            "Activity log verification passed."
        )

        return True

    except Exception as exc:

        record_failure(
            f"Activity verification failed: "
            f"{exc}"
        )

        return False

# =========================================================
# PRICING VALIDATION
# =========================================================


def verify_pricing() -> bool:

    try:

        pricing_file = (
            DATA_DIR
            / "vendor_pricing.csv"
        )

        if not pricing_file.exists():

            record_success(
                "Pricing dataset skipped."
            )

            return True

        df = pd.read_csv(
            pricing_file
        )

        if (
            df.select_dtypes(
                include=["number"]
            ) < 0
        ).any().any():

            record_failure(
                "Negative pricing values detected."
            )

            return False

        record_success(
            "Pricing verification passed."
        )

        return True

    except Exception as exc:

        record_failure(
            f"Pricing verification failed: "
            f"{exc}"
        )

        return False

# =========================================================
# MAIN ORCHESTRATOR
# =========================================================


def main() -> None:

    start = time.time()

    logger.info("=" * 60)

    logger.info(
        "Starting enterprise verification pipeline..."
    )

    checks = [

        verify_employees(),

        verify_activity_logs(),

        verify_pricing(),
    ]

    duration = (
        time.time() - start
    )

    verification_report[
        "duration_seconds"
    ] = round(duration, 2)

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            verification_report,
            f,
            indent=4,
        )

    if all(checks):

        print(
            "\nSUCCESS: "
            "Enterprise verification passed.\n"
        )

        sys.exit(0)

    print(
        "\nFAILED: "
        "Verification pipeline failed.\n"
    )

    sys.exit(1)

# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()