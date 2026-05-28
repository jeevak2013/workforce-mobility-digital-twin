"""
Enterprise Workforce Activity Stream Generator

File:
    src/data_generation/activity_logs.py

Purpose:
- Enterprise operational activity simulation
- Transport-wave event generation
- Dispatch-ready workforce streams
- Overtime realism
- Attendance simulation
- Female safety operations
- Transport orchestration support
- Enterprise mobility digital twin
"""

from __future__ import annotations

import random
from datetime import (
    datetime,
    timedelta,
)
from typing import Dict, List, Literal, cast

import pandas as pd


# =========================================================
# TYPE CONTRACTS
# =========================================================

AttendanceStatusType = Literal[
    "PRESENT",
    "SICK_LEAVE",
    "PLANNED_LEAVE",
    "NO_SHOW",
]

OperationalEventType = Literal[
    "NORMAL_OPERATIONS",
    "HEAVY_RAIN",
    "VENDOR_DELAY",
    "SYSTEM_OUTAGE",
    "MONTH_END_SURGE",
]

DispatchWaveType = Literal[
    "03:30",
    "04:30",
    "NO_TRANSPORT",
]

# =========================================================
# ENTERPRISE CONFIGURATION
# =========================================================

ABSENTEEISM_CONFIG = {

    "SICK_LEAVE": 0.04,

    "PLANNED_LEAVE": 0.03,

    "NO_SHOW": 0.01,
}

OPERATIONAL_EVENT_CONFIG = {

    "HEAVY_RAIN": 0.02,

    "VENDOR_DELAY": 0.03,

    "SYSTEM_OUTAGE": 0.01,

    "MONTH_END_SURGE": 0.08,
}

WEEKEND_WORKFORCE_FACTOR = 0.65

SUNDAY_WORKFORCE_FACTOR = 0.45

# =========================================================
# ATTENDANCE ENGINE
# =========================================================


def generate_attendance_status(
    rng: random.Random,
) -> AttendanceStatusType:
    """
    Enterprise attendance simulation.
    """

    value = rng.random()

    cumulative = 0.0

    for status, probability in (
        ABSENTEEISM_CONFIG.items()
    ):

        cumulative += probability

        if value < cumulative:

            return cast(
                AttendanceStatusType,
                status,
            )

    return "PRESENT"


# =========================================================
# OPERATIONAL EVENT ENGINE
# =========================================================


def generate_operational_event(
    rng: random.Random,
) -> OperationalEventType:
    """
    Enterprise operational anomaly simulation.
    """

    value = rng.random()

    cumulative = 0.0

    for event, probability in (
        OPERATIONAL_EVENT_CONFIG.items()
    ):

        cumulative += probability

        if value < cumulative:

            return cast(
            OperationalEventType,
            event,
        )

    return "NORMAL_OPERATIONS"


# =========================================================
# WEEKEND REDUCTION ENGINE
# =========================================================


def workforce_factor_for_day(
    current_date: datetime,
) -> float:
    """
    Weekend workforce reduction modeling.
    """

    weekday = current_date.weekday()

    # Saturday
    if weekday == 5:
        return WEEKEND_WORKFORCE_FACTOR

    # Sunday
    if weekday == 6:
        return SUNDAY_WORKFORCE_FACTOR

    return 1.0


# =========================================================
# EXTENSION ENGINE
# =========================================================


def compute_actual_logout(
    scheduled_logout: datetime,
    extension_minutes: int,
) -> datetime:
    """
    Enterprise overtime calculation.
    """

    return (
        scheduled_logout
        + timedelta(
            minutes=extension_minutes
        )
    )


# =========================================================
# DISPATCH WAVE ENGINE
# =========================================================


def compute_dispatch_wave(
    transport_shift: str,
    extension_category: str,
    uses_company_transport: bool,
) -> DispatchWaveType:
    """
    Enterprise dispatch orchestration logic.

    Critical Business Rules:

    03:30 dispatch:
    - only non-extended 03:30 employees

    04:30 dispatch:
    - native 04:30 employees
    - extended 03:30 employees
    """

    if not uses_company_transport:

        return "NO_TRANSPORT"

    # -----------------------------------------------------
    # 03:30 employees
    # -----------------------------------------------------

    if transport_shift == "03:30":

        if (
            extension_category
            == "EXTEND_TO_0430"
        ):

            return "04:30"

        return "03:30"

    # -----------------------------------------------------
    # Native 04:30 employees
    # -----------------------------------------------------

    if transport_shift == "04:30":

        return "04:30"

    return "NO_TRANSPORT"


# =========================================================
# TRANSPORT REQUIREMENT ENGINE
# =========================================================


def transport_required(
    actual_logout: datetime,
    uses_company_transport: bool,
) -> bool:
    """
    Enterprise transport eligibility.

    Company transport not mandatory
    after 06:30.
    """

    if not uses_company_transport:
        return False

    return actual_logout.hour < 6 or (
        actual_logout.hour == 6
        and actual_logout.minute <= 30
    )


# =========================================================
# ROUTE RISK ENGINE
# =========================================================


def route_risk_level(
    home_distance_km: float,
    requires_security_escort: bool,
    operational_event: str,
) -> str:
    """
    Enterprise route risk analytics.
    """

    score = 0

    if home_distance_km > 25:
        score += 2

    elif home_distance_km > 18:
        score += 1

    if requires_security_escort:
        score += 2

    if operational_event == "HEAVY_RAIN":
        score += 2

    if operational_event == "SYSTEM_OUTAGE":
        score += 1

    if score >= 4:
        return "HIGH"

    if score >= 2:
        return "MEDIUM"

    return "LOW"


# =========================================================
# MAIN ACTIVITY GENERATOR
# =========================================================


def generate_activity_logs(
    employees_csv: str,
    output_csv: str,
    start_date: str = "2025-01-01",
    days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enterprise workforce activity stream generator.
    """

    rng = random.Random(seed)

    employees_df = pd.read_csv(
        employees_csv
    )

    # =====================================================
    # ENTERPRISE VALIDATION
    # =====================================================

    required_columns = [

        "employee_id",

        "transport_shift",

        "extension_category",

        "uses_company_transport",

        "transport_eligibility",

        "requires_security_escort",

        "predicted_extension_minutes",

        "shift_logout",

        "home_distance_km",
    ]

    missing = [

        c

        for c in required_columns

        if c not in employees_df.columns
    ]

    if missing:

        raise ValueError(
            f"employees.csv missing columns: "
            f"{missing}"
        )

    # =====================================================
    # ACTIVITY GENERATION
    # =====================================================

    records: List[
        Dict
    ] = []

    start_dt = pd.to_datetime(
        start_date
    )

    for day_offset in range(days):

        current_date = (
            start_dt
            + timedelta(
                days=day_offset
            )
        )

        # -------------------------------------------------
        # Weekend workforce reduction
        # -------------------------------------------------

        workforce_factor = (
            workforce_factor_for_day(
                current_date
            )
        )

        daily_df = employees_df.sample(
            frac=workforce_factor,
            random_state=(
                seed + day_offset
            ),
        )

        # -------------------------------------------------
        # Daily operational event
        # -------------------------------------------------

        operational_event = (
            generate_operational_event(
                rng
            )
        )

        # -------------------------------------------------
        # Employee activity generation
        # -------------------------------------------------

        for _, row in (
            daily_df.iterrows()
        ):

            attendance_status = (
                generate_attendance_status(
                    rng
                )
            )

            # ---------------------------------------------
            # Skip absentees
            # ---------------------------------------------

            if (
                attendance_status
                != "PRESENT"
            ):

                continue

            scheduled_logout = (
                pd.to_datetime(
                    row[
                        "shift_logout"
                    ]
                )
            )

            extension_minutes = int(
                row[
                    "predicted_extension_minutes"
                ]
            )

            # ---------------------------------------------
            # Operational randomness
            # ---------------------------------------------

            variability = rng.randint(
                -10,
                20,
            )

            actual_extension = max(
                extension_minutes
                + variability,
                0,
            )

            actual_logout = (
                compute_actual_logout(
                    scheduled_logout,
                    actual_extension,
                )
            )

            # ---------------------------------------------
            # Dispatch logic
            # ---------------------------------------------

            dispatch_wave = (
                compute_dispatch_wave(
                    transport_shift=row[
                        "transport_shift"
                    ],

                    extension_category=row[
                        "extension_category"
                    ],

                    uses_company_transport=bool(
                        row[
                            "uses_company_transport"
                        ]
                    ),
                )
            )

            # ---------------------------------------------
            # Transport requirement
            # ---------------------------------------------

            dispatch_required = (
                transport_required(
                    actual_logout,
                    bool(
                        row[
                            "uses_company_transport"
                        ]
                    ),
                )
            )

            # ---------------------------------------------
            # Risk analytics
            # ---------------------------------------------

            risk_level = (
                route_risk_level(
                    home_distance_km=float(
                        row[
                            "home_distance_km"
                        ]
                    ),

                    requires_security_escort=bool(
                        row[
                            "requires_security_escort"
                        ]
                    ),

                    operational_event=(
                        operational_event
                    ),
                )
            )

            # ---------------------------------------------
            # Activity record
            # ---------------------------------------------

            records.append({

                "date": current_date.date(),

                "employee_id": int(
                    row[
                        "employee_id"
                    ]
                ),

                "attendance_status": (
                    attendance_status
                ),

                "transport_shift": row[
                    "transport_shift"
                ],

                "scheduled_logout": (
                    scheduled_logout
                ),

                "actual_logout": (
                    actual_logout
                ),

                "extension_category": row[
                    "extension_category"
                ],

                "overtime_minutes": (
                    actual_extension
                ),

                "actual_dispatch_wave": (
                    dispatch_wave
                ),

                "uses_company_transport": bool(
                    row[
                        "uses_company_transport"
                    ]
                ),

                "dispatch_required": (
                    dispatch_required
                ),

                "transport_eligibility": row[
                    "transport_eligibility"
                ],

                "requires_security_escort": bool(
                    row[
                        "requires_security_escort"
                    ]
                ),

                "operational_event": (
                    operational_event
                ),

                "route_risk_level": (
                    risk_level
                ),

                "home_distance_km": float(
                    row[
                        "home_distance_km"
                    ]
                ),
            })

    # =====================================================
    # DATAFRAME
    # =====================================================

    logs_df = pd.DataFrame(
        records
    )

    logs_df.to_csv(
        output_csv,
        index=False,
    )

    # =====================================================
    # ENTERPRISE ANALYTICS
    # =====================================================

    print(
        "\n================================================="
    )

    print(
        "ENTERPRISE ACTIVITY LOG GENERATION"
    )

    print(
        "=================================================\n"
    )

    print(
        f"Generated records: "
        f"{len(logs_df)}"
    )

    print(
        "\nDispatch wave distribution:"
    )

    print(
        logs_df[
            "actual_dispatch_wave"
        ].value_counts()
    )

    print(
        "\nOperational events:"
    )

    print(
        logs_df[
            "operational_event"
        ].value_counts()
    )

    print(
        "\nRoute risk distribution:"
    )

    print(
        logs_df[
            "route_risk_level"
        ].value_counts()
    )

    print(
        "\n=================================================\n"
    )

    return logs_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    generate_activity_logs(

        employees_csv=(
            "data/employees.csv"
        ),

        output_csv=(
            "data/activity_logs.csv"
        ),

        start_date="2025-01-01",

        days=30,

        seed=42,
    )