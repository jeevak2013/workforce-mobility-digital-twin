"""
Enterprise Workforce Transportation Simulation Engine

File:
    src/data_generation/employee_generator.py

Purpose:
- Enterprise workforce generation
- Transport enrollment modeling
- Shift-wave simulation
- Overtime extension realism
- Spatial workforce distribution
- Transport policy initialization
- Safety-aware transport simulation
- Enterprise mobility digital twin generation
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Dict, List, Literal, cast

import pandas as pd

from src.config.geography_config import (
    BPO,
    COIMBATORE_HUBS,
    HubLiteral,
    TransportShiftLiteral,
)

from src.data_generation.geography_generator import (
    generate_random_home_drop,
)

from src.domain.schemas import (
    Coordinate,
    Employee,
)

# =========================================================
# TYPE CONTRACTS
# =========================================================

GenderType = Literal[
    "Female",
    "Male",
]

TransportEligibilityType = Literal[
    "FULL_HOME_DROP",
    "CONDITIONAL_APPROVAL",
    "HUB_DROP_ONLY",
]

ExtensionCategoryType = Literal[
    "NO_EXTENSION",
    "EXTEND_TO_0430",
    "EXTEND_BEYOND_0630",
]

# =========================================================
# ENTERPRISE DISTRIBUTIONS
# =========================================================

HUB_DISTRIBUTION: Dict[
    HubLiteral,
    float,
] = {
    "Saravanampatti": 0.32,
    "Singanallur": 0.25,
    "Thudiyalur": 0.18,
    "Hopes": 0.15,
    "Ganapathy": 0.10,
}

TRANSPORT_SHIFT_OPTIONS = [
    "03:30",
    "04:30",
]

TRANSPORT_SHIFT_WEIGHTS = [
    0.40,
    0.60,
]

NON_TRANSPORT_SHIFTS = [
    "18:30",
    "21:30",
    "00:30",
]

# =========================================================
# GEOSPATIAL DISTANCE ENGINE
# =========================================================


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Computes geodesic distance
    between employee home and BPO.
    """

    radius = 6371.0

    dlat = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return round(
        radius * c,
        2,
    )


# =========================================================
# GENDER DISTRIBUTION
# =========================================================


def generate_gender_pool(
    count: int,
) -> List[GenderType]:
    """
    Enterprise workforce gender ratio.
    """

    female_count = int(
        round(
            count
            * BPO.gender_female_target
        )
    )

    male_count = (
        count - female_count
    )

    pool = (
        ["Female"] * female_count
        + ["Male"] * male_count
    )

    return cast(
        List[GenderType],
        pool,
    )


# =========================================================
# HUB ASSIGNMENT
# =========================================================


def choose_hub(
    rng: random.Random,
) -> HubLiteral:
    """
    Weighted enterprise corridor allocation.
    """

    return cast(
        HubLiteral,
        rng.choices(
            population=list(
                HUB_DISTRIBUTION.keys()
            ),
            weights=list(
                HUB_DISTRIBUTION.values()
            ),
            k=1,
        )[0],
    )


# =========================================================
# TRANSPORT SHIFT ENGINE
# =========================================================


def assign_transport_shift(
    rng: random.Random,
    uses_company_transport: bool,
) -> TransportShiftLiteral:
    """
    Enterprise transport-wave assignment.
    """

    if not uses_company_transport:
        return "NON_TRANSPORT"

    return cast(
        TransportShiftLiteral,
        rng.choices(
            TRANSPORT_SHIFT_OPTIONS,
            weights=TRANSPORT_SHIFT_WEIGHTS,
            k=1,
        )[0],
    )


# =========================================================
# SHIFT LOGOUT ENGINE
# =========================================================


def generate_shift_logout(
    rng: random.Random,
    transport_shift: TransportShiftLiteral,
) -> datetime:
    """
    Enterprise logout generation.
    """

    if transport_shift == "NON_TRANSPORT":

        shift = rng.choice(
            NON_TRANSPORT_SHIFTS
        )

    else:

        shift = transport_shift

    hour, minute = map(
        int,
        shift.split(":"),
    )

    return datetime(
        2025,
        1,
        1,
        hour,
        minute,
    )


# =========================================================
# EXTENSION CATEGORY ENGINE
# =========================================================


def assign_extension_category(
    rng: random.Random,
    transport_shift: TransportShiftLiteral,
) -> ExtensionCategoryType:
    """
    Enterprise overtime modeling.
    """

    # -----------------------------------------------------
    # 03:30 extensions
    # -----------------------------------------------------

    if transport_shift == "03:30":

        if rng.random() < 0.25:

            return "EXTEND_TO_0430"

    # -----------------------------------------------------
    # 04:30 extensions
    # -----------------------------------------------------

    elif transport_shift == "04:30":

        if rng.random() < 0.10:

            return "EXTEND_BEYOND_0630"

    return "NO_EXTENSION"


# =========================================================
# EXTENSION MINUTES ENGINE
# =========================================================


def generate_extension_minutes(
    rng: random.Random,
    extension_category: ExtensionCategoryType,
) -> int:
    """
    Generates realistic overtime duration.
    """

    if (
        extension_category
        == "EXTEND_TO_0430"
    ):

        return rng.choice(
            [30, 60]
        )

    if (
        extension_category
        == "EXTEND_BEYOND_0630"
    ):

        return rng.choice(
            [60, 90, 120]
        )

    return 0


# =========================================================
# TRANSPORT POLICY ENGINE
# =========================================================


def assign_transport_eligibility(
    distance_km: float,
) -> TransportEligibilityType:
    """
    Enterprise transport eligibility policy.
    """

    if distance_km <= 18:

        return "FULL_HOME_DROP"

    if distance_km <= 25:

        return "CONDITIONAL_APPROVAL"

    return "HUB_DROP_ONLY"


# =========================================================
# SAFETY PRIORITY ENGINE
# =========================================================


def generate_safety_priority(
    gender: GenderType,
    shift_logout: datetime,
) -> float:
    """
    Enterprise safety-aware routing priority.
    """

    score = 1.0

    if gender == "Female":

        score += 3.0

    if shift_logout.hour in [0, 3, 4]:

        score += 3.0

    return min(score, 10.0)


# =========================================================
# SECURITY ESCORT ENGINE
# =========================================================


def requires_security_escort(
    gender: GenderType,
    shift_logout: datetime,
) -> bool:
    """
    Enterprise female escort enforcement.
    """

    return (
        gender == "Female"
        and shift_logout.hour
        in [0, 3, 4]
    )


# =========================================================
# EMPLOYEE SERIALIZER
# =========================================================


def serialize_employee(
    employee: Employee,
    status: str = "Active",
) -> dict:
    """
    Converts Employee schema
    into analytics-friendly row.
    """

    data = employee.model_dump()

    data["home_lat"] = (
        data["home"]["latitude"]
    )

    data["home_lon"] = (
        data["home"]["longitude"]
    )

    del data["home"]

    data["status"] = status

    return data


# =========================================================
# ENTERPRISE EMPLOYEE GENERATOR
# =========================================================


def generate_initial_employees(
    total_employees: int = 2000,
    transport_users: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enterprise workforce generation engine.
    """

    rng = random.Random(seed)

    # -----------------------------------------------------
    # Gender distribution
    # -----------------------------------------------------

    genders = generate_gender_pool(
        total_employees
    )

    rng.shuffle(genders)

    # -----------------------------------------------------
    # Transport enrollment
    # -----------------------------------------------------

    transport_employee_ids = set(
        rng.sample(
            range(
                1,
                total_employees + 1,
            ),
            transport_users,
        )
    )

    records = []

    # -----------------------------------------------------
    # Employee generation
    # -----------------------------------------------------

    for employee_id in range(
        1,
        total_employees + 1,
    ):

        gender = genders[
            employee_id - 1
        ]

        uses_company_transport = (
            employee_id
            in transport_employee_ids
        )

        # -------------------------------------------------
        # Operational hub assignment
        # -------------------------------------------------

        assigned_hub = choose_hub(
            rng
        )

        hub_config = (
            COIMBATORE_HUBS[
                assigned_hub
            ]
        )

        # -------------------------------------------------
        # Long-distance outliers
        # -------------------------------------------------

        if rng.random() < 0.07:

            max_scatter_km = (
                rng.uniform(
                    20,
                    35,
                )
            )

        else:

            max_scatter_km = (
                rng.uniform(
                    3,
                    hub_config.operational_radius_km,
                )
            )

        # -------------------------------------------------
        # Employee home generation
        # -------------------------------------------------

        lat, lon = (
            generate_random_home_drop(
                hub_center=hub_config,
                max_scatter_km=max_scatter_km,
                rng=rng,
            )
        )

        # -------------------------------------------------
        # Distance from BPO
        # -------------------------------------------------

        home_distance_km = (
            haversine_distance_km(
                lat,
                lon,
                BPO.location.latitude,
                BPO.location.longitude,
            )
        )

        # -------------------------------------------------
        # Shift-wave assignment
        # -------------------------------------------------

        transport_shift = (
            assign_transport_shift(
                rng,
                uses_company_transport,
            )
        )

        shift_logout = (
            generate_shift_logout(
                rng,
                transport_shift,
            )
        )

        # -------------------------------------------------
        # Overtime generation
        # -------------------------------------------------

        extension_category = (
            assign_extension_category(
                rng,
                transport_shift,
            )
        )

        extension_minutes = (
            generate_extension_minutes(
                rng,
                extension_category,
            )
        )

        # -------------------------------------------------
        # Transport policy
        # -------------------------------------------------

        transport_eligibility = (
            assign_transport_eligibility(
                home_distance_km,
            )
        )

        # -------------------------------------------------
        # Safety analytics
        # -------------------------------------------------

        safety_score = (
            generate_safety_priority(
                gender,
                shift_logout,
            )
        )

        escort_flag = (
            requires_security_escort(
                gender,
                shift_logout,
            )
        )

        # -------------------------------------------------
        # Enterprise Employee Schema
        # -------------------------------------------------

        employee = Employee(
            employee_id=employee_id,

            gender=gender,

            home=Coordinate(
                latitude=lat,
                longitude=lon,
            ),

            hub=assigned_hub,

            pickup_hub=assigned_hub,

            uses_company_transport=(
                uses_company_transport
            ),

            transport_shift=(
                transport_shift
            ),

            transport_eligibility=(
                transport_eligibility
            ),

            extension_category=(
                extension_category
            ),

            shift_logout=shift_logout,

            predicted_extension_minutes=(
                extension_minutes
            ),

            safety_priority_score=(
                safety_score
            ),

            requires_security_escort=(
                escort_flag
            ),

            home_distance_km=(
                home_distance_km
            ),
        )

        records.append(
            serialize_employee(
                employee
            )
        )

    employees_df = pd.DataFrame(
        records
    )

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
            f"Generated employees "
            f"missing columns: {missing}"
        )

    return employees_df


# =========================================================
# CSV EXPORT
# =========================================================


def export_employees_csv(
    output_path: str,
    total_employees: int = 2000,
    transport_users: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates enterprise workforce CSV.
    """

    employees_df = (
        generate_initial_employees(
            total_employees=(
                total_employees
            ),
            transport_users=(
                transport_users
            ),
            seed=seed,
        )
    )

    employees_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n================================================="
    )

    print(
        "ENTERPRISE EMPLOYEE GENERATION COMPLETE"
    )

    print(
        "=================================================\n"
    )

    print(
        f"Total employees: "
        f"{len(employees_df)}"
    )

    print(
        f"Transport users: "
        f"{employees_df['uses_company_transport'].sum()}"
    )

    print(
        "\nTransport shift distribution:"
    )

    print(
        employees_df[
            "transport_shift"
        ].value_counts()
    )

    print(
        "\nTransport eligibility:"
    )

    print(
        employees_df[
            "transport_eligibility"
        ].value_counts()
    )

    print(
        "\nExtension categories:"
    )

    print(
        employees_df[
            "extension_category"
        ].value_counts()
    )

    print(
        "\n=================================================\n"
    )

    return employees_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    export_employees_csv(
        output_path="data/employees.csv",
        total_employees=2000,
        transport_users=1000,
        seed=42,
    )