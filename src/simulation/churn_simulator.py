"""
Enterprise Workforce Churn Simulator

File:
    src/simulation/churn_simulator.py

Purpose:
- Enterprise workforce attrition simulation
- Transport-aware churn intelligence
- Operational fatigue modeling
- Shift-wave preservation
- Workforce continuity simulation
- Mobility demand recalibration
- Transport optimization support
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, List, cast

import pandas as pd

from src.domain.schemas import (
    Coordinate,
    Employee,
)

# =========================================================
# ENTERPRISE CHURN CONFIGURATION
# =========================================================

BASE_MONTHLY_CHURN_RATE = 0.08

HIGH_DISTANCE_THRESHOLD = 25.0

MODERATE_DISTANCE_THRESHOLD = 18.0

HIGH_FATIGUE_EXTENSION = (
    "EXTEND_BEYOND_0630"
)

MODERATE_FATIGUE_EXTENSION = (
    "EXTEND_TO_0430"
)

# =========================================================
# CHURN PROBABILITY ENGINE
# =========================================================


def compute_churn_probability(
    employee: Employee,
) -> float:
    """
    Enterprise churn intelligence model.

    Factors:
    - long commute fatigue
    - overtime burnout
    - transport dissatisfaction
    - female night-shift stress
    - operational fatigue
    """

    probability = (
        BASE_MONTHLY_CHURN_RATE
    )

    # -----------------------------------------------------
    # Distance fatigue
    # -----------------------------------------------------

    if (
        employee.home_distance_km
        > HIGH_DISTANCE_THRESHOLD
    ):

        probability += 0.08

    elif (
        employee.home_distance_km
        > MODERATE_DISTANCE_THRESHOLD
    ):

        probability += 0.04

    # -----------------------------------------------------
    # Overtime fatigue
    # -----------------------------------------------------

    if (
        employee.extension_category
        == HIGH_FATIGUE_EXTENSION
    ):

        probability += 0.10

    elif (
        employee.extension_category
        == MODERATE_FATIGUE_EXTENSION
    ):

        probability += 0.05

    # -----------------------------------------------------
    # Transport dissatisfaction
    # -----------------------------------------------------

    if (
        employee.transport_eligibility
        == "HUB_DROP_ONLY"
    ):

        probability += 0.07

    elif (
        employee.transport_eligibility
        == "CONDITIONAL_APPROVAL"
    ):

        probability += 0.03

    # -----------------------------------------------------
    # Female night-shift fatigue
    # -----------------------------------------------------

    if (
        employee.gender == "Female"
        and employee.transport_shift
        in ["03:30", "04:30"]
    ):

        probability += 0.03

    # -----------------------------------------------------
    # Heavy overtime penalty
    # -----------------------------------------------------

    if (
        employee.predicted_extension_minutes
        >= 90
    ):

        probability += 0.05

    return min(
        round(probability, 4),
        0.45,
    )


# =========================================================
# EMPLOYEE RECONSTRUCTION
# =========================================================


def reconstruct_employee(
    row: pd.Series,
) -> Employee:
    """
    Reconstructs enterprise Employee
    schema from analytics row.
    """

    shift_logout = row[
        "shift_logout"
    ]

    if isinstance(
        shift_logout,
        str,
    ):

        shift_logout = (
            pd.to_datetime(
                shift_logout
            )
        )

    return Employee(

        employee_id=int(
            row["employee_id"]
        ),

        gender=row["gender"],

        home=Coordinate(
            latitude=float(
                row["home_lat"]
            ),
            longitude=float(
                row["home_lon"]
            ),
        ),

        hub=row["hub"],

        pickup_hub=row.get(
            "pickup_hub",
            row["hub"],
        ),

        uses_company_transport=bool(
            row.get(
                "uses_company_transport",
                True,
            )
        ),

        transport_shift=row.get(
            "transport_shift",
            "NON_TRANSPORT",
        ),

        transport_eligibility=row.get(
            "transport_eligibility",
            "FULL_HOME_DROP",
        ),

        extension_category=row.get(
            "extension_category",
            "NO_EXTENSION",
        ),

        shift_logout=shift_logout,

        predicted_extension_minutes=int(
            row.get(
                "predicted_extension_minutes",
                0,
            )
        ),

        safety_priority_score=float(
            row.get(
                "safety_priority_score",
                1.0,
            )
        ),

        requires_security_escort=bool(
            row.get(
                "requires_security_escort",
                False,
            )
        ),

        home_distance_km=float(
            row.get(
                "home_distance_km",
                0.0,
            )
        ),
    )


# =========================================================
# TRANSPORT DISTRIBUTION ANALYTICS
# =========================================================


def transport_distribution_summary(
    df: pd.DataFrame,
) -> Dict[str, int]:
    """
    Enterprise transport-wave analytics.
    """

    return cast(
        Dict[str, int],
        (
            df[
                "transport_shift"
            ]
            .value_counts()
            .to_dict()
        ),
    )


# =========================================================
# ENTERPRISE CHURN SIMULATION
# =========================================================


def simulate_employee_churn(
    employees_df: pd.DataFrame,
    monthly_churn_rate: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enterprise workforce attrition engine.

    Features:
    - transport-aware churn
    - operational fatigue modeling
    - enterprise workforce realism
    - stochastic human behavior
    """

    rng = random.Random(seed)

    # =====================================================
    # ENTERPRISE VALIDATION
    # =====================================================

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
        if c not in employees_df.columns
    ]

    if missing:

        raise ValueError(
            f"employees.csv missing columns: "
            f"{missing}"
        )

    # =====================================================
    # EMPLOYEE RECONSTRUCTION
    # =====================================================

    employees: List[Employee] = [

        reconstruct_employee(row)

        for _, row in (
            employees_df.iterrows()
        )
    ]

    retained = []

    churned = []

    # =====================================================
    # ENTERPRISE CHURN ENGINE
    # =====================================================

    for employee in employees:

        churn_probability = (
            compute_churn_probability(
                employee
            )
        )

        # -------------------------------------------------
        # Dynamic scaling
        # -------------------------------------------------

        churn_probability *= (
            monthly_churn_rate
            / BASE_MONTHLY_CHURN_RATE
        )

        churn_probability = min(
            churn_probability,
            0.60,
        )

        # -------------------------------------------------
        # Stochastic churn decision
        # -------------------------------------------------

        is_churned = (
            rng.random()
            < churn_probability
        )

        if is_churned:

            churned.append(
                employee.model_dump()
            )

        else:

            retained.append(
                employee.model_dump()
            )

    # =====================================================
    # DATAFRAME RECONSTRUCTION
    # =====================================================

    retained_df = pd.DataFrame(
        retained
    )

    churned_df = pd.DataFrame(
        churned
    )

    # -----------------------------------------------------
    # Flatten nested coordinates
    # -----------------------------------------------------

    for df in [
        retained_df,
        churned_df,
    ]:

        if len(df) == 0:
            continue

        df["home_lat"] = (
            df["home"]
            .apply(
                lambda x: x["latitude"]
            )
        )

        df["home_lon"] = (
            df["home"]
            .apply(
                lambda x: x["longitude"]
            )
        )

        df.drop(
            columns=["home"],
            inplace=True,
        )

    # =====================================================
    # ENTERPRISE ANALYTICS
    # =====================================================

    print(
        "\n================================================="
    )

    print(
        "ENTERPRISE CHURN SIMULATION SUMMARY"
    )

    print(
        "=================================================\n"
    )

    print(
        f"Original workforce: "
        f"{len(employees_df)}"
    )

    print(
        f"Employees churned: "
        f"{len(churned_df)}"
    )

    print(
        f"Remaining workforce: "
        f"{len(retained_df)}"
    )

    # -----------------------------------------------------
    # Transport analytics
    # -----------------------------------------------------

    remaining_transport = (
        retained_df[
            retained_df[
                "uses_company_transport"
            ]
            == True
        ]
    )

    print(
        f"\nTransport users remaining: "
        f"{len(remaining_transport)}"
    )

    print(
        "\nTransport shift distribution:"
    )

    print(
        transport_distribution_summary(
            remaining_transport
        )
    )

    print(
        "\nExtension distribution:"
    )

    print(
        retained_df[
            "extension_category"
        ].value_counts()
    )

    print(
        "\nTransport eligibility:"
    )

    print(
        retained_df[
            "transport_eligibility"
        ].value_counts()
    )

    print(
        "\n=================================================\n"
    )

    return retained_df


# =========================================================
# EXPORT PIPELINE
# =========================================================


def simulate_churn_and_export(
    input_csv: str,
    output_csv: str,
    monthly_churn_rate: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enterprise churn orchestration pipeline.
    """

    employees_df = pd.read_csv(
        input_csv
    )

    churned_df = (
        simulate_employee_churn(
            employees_df=employees_df,
            monthly_churn_rate=(
                monthly_churn_rate
            ),
            seed=seed,
        )
    )

    churned_df.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"Exported churn-adjusted workforce:"
    )

    print(
        output_csv
    )

    return churned_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    simulate_churn_and_export(
        input_csv="data/employees.csv",
        output_csv=(
            "data/employees_after_churn.csv"
        ),
        monthly_churn_rate=0.08,
        seed=42,
    )