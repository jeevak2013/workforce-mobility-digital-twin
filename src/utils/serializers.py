"""
Enterprise Workforce Serialization Utilities

File:
    src/data_generation/serializers.py

Purpose:
- Enterprise workforce serialization
- Mobility orchestration normalization
- Analytics-ready transport flattening
- Dispatch pipeline compatibility
- ML feature standardization
- Route optimization readiness
- Operational audit preservation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from src.domain.schemas import Employee

# =========================================================
# ENTERPRISE TEMPORAL NORMALIZER
# =========================================================


def serialize_datetime(
    value: datetime | None,
) -> str | None:
    """
    Standardized datetime serialization.
    """

    if value is None:
        return None

    return value.isoformat()


# =========================================================
# ENTERPRISE EMPLOYEE SERIALIZER
# =========================================================


def serialize_employee(
    employee: Employee,
    status: str = "Active",
) -> Dict[str, Any]:
    """
    Enterprise workforce serializer.

    Features:
    - geographic flattening
    - transport normalization
    - temporal standardization
    - analytics-ready schema
    - mobility optimization support
    """

    data = employee.model_dump()

    # =====================================================
    # FLATTEN GEOGRAPHIC STRUCTURES
    # =====================================================

    home = data.pop("home")

    data["home_lat"] = home["latitude"]

    data["home_lon"] = home["longitude"]

    # =====================================================
    # TEMPORAL NORMALIZATION
    # =====================================================

    data["shift_logout"] = serialize_datetime(data.get("shift_logout"))

    if "effective_logout_time" in data:
        data["effective_logout_time"] = serialize_datetime(
            data.get("effective_logout_time")
        )

    # =====================================================
    # BOOLEAN NORMALIZATION
    # =====================================================

    data["uses_company_transport"] = bool(
        data.get(
            "uses_company_transport",
            False,
        )
    )

    data["requires_security_escort"] = bool(
        data.get(
            "requires_security_escort",
            False,
        )
    )

    # =====================================================
    # NUMERIC NORMALIZATION
    # =====================================================

    data["home_distance_km"] = round(
        float(
            data.get(
                "home_distance_km",
                0.0,
            )
        ),
        2,
    )

    data["safety_priority_score"] = round(
        float(
            data.get(
                "safety_priority_score",
                1.0,
            )
        ),
        2,
    )

    data["predicted_extension_minutes"] = int(
        data.get(
            "predicted_extension_minutes",
            0,
        )
    )

    # =====================================================
    # OPERATIONAL METADATA
    # =====================================================

    data["status"] = status

    # =====================================================
    # ENTERPRISE DERIVED ANALYTICS
    # =====================================================

    data["is_transport_user"] = data["uses_company_transport"]

    data["is_night_shift"] = data["transport_shift"] in ["03:30", "04:30"]

    data["is_transport_login_shift"] = data["login_shift"] in ["18:30", "19:30"]

    data["is_long_distance_employee"] = data["home_distance_km"] > 25

    data["requires_special_approval"] = (
        data["transport_eligibility"] == "CONDITIONAL_APPROVAL"
    )

    data["is_high_risk_route"] = (
        data["requires_security_escort"] or data["home_distance_km"] > 25
    )

    # =====================================================
    # WORKFORCE FATIGUE ANALYTICS
    # =====================================================

    data["fatigue_risk_score"] = compute_fatigue_score(data)

    return data


# =========================================================
# FATIGUE ANALYTICS ENGINE
# =========================================================


def compute_fatigue_score(
    data: Dict[str, Any],
) -> float:
    """
    Enterprise workforce fatigue scoring.
    """

    score = 1.0

    if data["extension_category"] == "EXTEND_TO_0430":
        score += 2.0

    elif data["extension_category"] == "EXTEND_BEYOND_0630":
        score += 4.0

    if data["home_distance_km"] > 25:
        score += 2.5

    if data["transport_shift"] == "03:30":
        score += 1.5

    return round(
        min(score, 10.0),
        2,
    )


# =========================================================
# BULK SERIALIZATION ENGINE
# =========================================================


def serialize_employee_batch(
    employees: List[Employee],
    status: str = "Active",
) -> List[Dict[str, Any]]:
    """
    Bulk enterprise workforce serialization.
    """

    return [
        serialize_employee(
            employee=employee,
            status=status,
        )
        for employee in employees
    ]


# =========================================================
# ENTERPRISE DATAFRAME VALIDATION
# =========================================================


def validate_serialized_employee(
    data: Dict[str, Any],
) -> None:
    """
    Enterprise serialization validator.
    """

    required_fields = [
        "employee_id",
        "gender",
        "home_lat",
        "home_lon",
        "hub",
        "pickup_hub",
        "uses_company_transport",
        "transport_shift",
        "login_shift",
        "transport_eligibility",
        "extension_category",
        "home_distance_km",
    ]

    missing = [field for field in required_fields if field not in data]

    if missing:
        raise ValueError(f"Serialized employee missing fields: {missing}")


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":
    print("\nEnterprise serializer ready.")
