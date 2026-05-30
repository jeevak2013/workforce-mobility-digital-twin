"""
Enterprise Geography Configuration Layer

File:
    src/config/geography_config.py

Purpose:
- Enterprise transport geography contracts
- Coimbatore operational intelligence
- Hub definitions
- Shift transport policy constants
- Mobility optimization configuration
- Routing thresholds
- Safety radius controls
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, Literal

# =========================================================
# TYPE DEFINITIONS
# =========================================================

HubLiteral = Literal[
    "Thudiyalur",
    "Saravanampatti",
    "Ganapathy",
    "Singanallur",
    "Hopes",
]

TransportEligibilityType = Literal[
    "FULL_HOME_DROP",
    "CONDITIONAL_APPROVAL",
    "HUB_DROP_ONLY",
]

TransportShiftLiteral = Literal[
    "03:30",
    "04:30",
    "NON_TRANSPORT",
]

LoginShiftLiteral = Literal[
    "18:30",
    "19:30",
    "NON_TRANSPORT",
]

# =========================================================
# LOGIN ↔ LOGOUT MAPPING
# =========================================================

LOGIN_TO_LOGOUT_MAPPING: Final[
    dict[
        LoginShiftLiteral,
        TransportShiftLiteral,
    ]
] = {
    "18:30": "03:30",
    "19:30": "04:30",
    "NON_TRANSPORT": "NON_TRANSPORT",
}

LOGOUT_TO_LOGIN_MAPPING: Final[
    dict[
        TransportShiftLiteral,
        LoginShiftLiteral,
    ]
] = {
    "03:30": "18:30",
    "04:30": "19:30",
    "NON_TRANSPORT": "NON_TRANSPORT",
}

# =========================================================
# ENTERPRISE GEO COORDINATE
# =========================================================


@dataclass(frozen=True)
class GeoCoordinate:
    """
    Immutable geographic coordinate.
    """

    latitude: float

    longitude: float


# =========================================================
# ENTERPRISE HUB CONFIG
# =========================================================


@dataclass(frozen=True)
class HubConfig:
    """
    Enterprise transport hub definition.
    """

    name: HubLiteral

    latitude: float

    longitude: float

    max_capacity: int

    operational_radius_km: float

    vendor_priority: int

    active: bool = True


# =========================================================
# ENTERPRISE BPO CONFIG
# =========================================================


@dataclass(frozen=True)
class BPOConfig:
    """
    Enterprise operational headquarters.
    """

    name: str

    location: GeoCoordinate

    operational_radius_km: float

    safe_home_drop_radius_km: float

    conditional_approval_radius_km: float

    female_safety_radius_km: float

    average_night_speed_kmph: float

    max_pooling_radius_km: float

    gender_female_target: float


# =========================================================
# ENTERPRISE BPO HEADQUARTERS
# =========================================================

BPO: Final[BPOConfig] = BPOConfig(
    name="Healthcare BPO Coimbatore",
    location=GeoCoordinate(
        latitude=11.0168,
        longitude=76.9558,
    ),
    operational_radius_km=40.0,
    safe_home_drop_radius_km=18.0,
    conditional_approval_radius_km=25.0,
    female_safety_radius_km=18.0,
    average_night_speed_kmph=24.0,
    max_pooling_radius_km=6.0,
    gender_female_target=0.42,
)

# =========================================================
# ENTERPRISE TRANSPORT HUBS
# =========================================================

COIMBATORE_HUBS: Final[
    Dict[
        HubLiteral,
        HubConfig,
    ]
] = {
    "Thudiyalur": HubConfig(
        name="Thudiyalur",
        latitude=11.0824,
        longitude=76.9410,
        max_capacity=220,
        operational_radius_km=12.0,
        vendor_priority=2,
    ),
    "Saravanampatti": HubConfig(
        name="Saravanampatti",
        latitude=11.0797,
        longitude=77.0030,
        max_capacity=350,
        operational_radius_km=14.0,
        vendor_priority=1,
    ),
    "Ganapathy": HubConfig(
        name="Ganapathy",
        latitude=11.0421,
        longitude=76.9860,
        max_capacity=180,
        operational_radius_km=10.0,
        vendor_priority=3,
    ),
    "Singanallur": HubConfig(
        name="Singanallur",
        latitude=11.0012,
        longitude=77.0265,
        max_capacity=320,
        operational_radius_km=15.0,
        vendor_priority=1,
    ),
    "Hopes": HubConfig(
        name="Hopes",
        latitude=11.0270,
        longitude=77.0442,
        max_capacity=250,
        operational_radius_km=11.0,
        vendor_priority=2,
    ),
}

# =========================================================
# SHIFT-WAVE DISTRIBUTION
# =========================================================

SHIFT_DISTRIBUTION: Final[
    Dict[
        TransportShiftLiteral,
        float,
    ]
] = {
    "03:30": 0.40,
    "04:30": 0.60,
    "NON_TRANSPORT": 0.00,
}

# =========================================================
# EXTENSION OPERATIONAL RATIOS
# =========================================================

EXTENSION_RATIOS: Final = {
    "03:30_EXTENSION_RATE": 0.25,
    "04:30_EXTENSION_RATE": 0.04,
    "MAX_EXTENSION_MINUTES": 120,
}

# =========================================================
# ENTERPRISE TRANSPORT POLICY
# =========================================================

TRANSPORT_POLICY: Final = {
    "FULL_HOME_DROP_RADIUS_KM": 18.0,
    "CONDITIONAL_APPROVAL_RADIUS_KM": 25.0,
    "MAX_OPERATIONAL_RADIUS_KM": 40.0,
    "MAX_POOLING_RADIUS_KM": 6.0,
    "FEMALE_ESCORT_START_HOUR": 0,
    "FEMALE_ESCORT_END_HOUR": 5,
}

# =========================================================
# ENTERPRISE VENDOR CONFIG
# =========================================================

VENDOR_OPTIMIZATION: Final = {
    "DEFAULT_CAB_CAPACITY": 4,
    "MAX_CAB_CAPACITY": 6,
    "MAX_ROUTE_DISTANCE_KM": 42.0,
    "MAX_ROUTE_DURATION_MINUTES": 120,
    "TARGET_POOLING_UTILIZATION": 0.82,
}

# =========================================================
# ENTERPRISE CLUSTERING CONFIG
# =========================================================

CLUSTERING_CONFIG: Final = {
    "MAX_CLUSTER_RADIUS_KM": 5.5,
    "MIN_CLUSTER_SIZE": 2,
    "MAX_CLUSTER_SIZE": 6,
    "DENSITY_THRESHOLD": 4.0,
}

# =========================================================
# SAFETY CONFIGURATION
# =========================================================

SAFETY_POLICY: Final = {
    "ENABLE_FEMALE_ESCORT_POLICY": True,
    "ENABLE_NIGHT_RISK_SCORING": True,
    "HIGH_RISK_DISTANCE_KM": 25.0,
    "MAX_ALLOWED_DETOUR_PERCENTAGE": 35.0,
}

# =========================================================
# DEMAND FORECASTING CONFIG
# =========================================================

FORECASTING_CONFIG: Final = {
    "WEEKEND_WORKFORCE_FACTOR": 0.65,
    "SUNDAY_WORKFORCE_FACTOR": 0.45,
    "MONTHLY_CHURN_RATE": 0.08,
    "OVERTIME_VARIABILITY_FACTOR": 0.20,
}

# =========================================================
# ENTERPRISE OPERATIONAL WAVES
# =========================================================

DISPATCH_WAVES: Final = {
    "03:30": {
        "target_count": 400,
        "dispatch_window_minutes": 20,
    },
    "04:30": {
        "target_count": 600,
        "dispatch_window_minutes": 25,
    },
}

# =========================================================
# HUB LOOKUP UTILITIES
# =========================================================


def get_hub_coordinate(
    hub_name: HubLiteral,
) -> GeoCoordinate:
    """
    Returns enterprise hub coordinate.
    """

    hub = COIMBATORE_HUBS[hub_name]

    return GeoCoordinate(
        latitude=hub.latitude,
        longitude=hub.longitude,
    )


# =========================================================
# HUB CAPACITY LOOKUP
# =========================================================


def get_hub_capacity(
    hub_name: HubLiteral,
) -> int:
    """
    Returns operational hub capacity.
    """

    return COIMBATORE_HUBS[hub_name].max_capacity


# =========================================================
# ENTERPRISE POLICY LOOKUPS
# =========================================================


def transport_eligibility_from_distance(
    distance_km: float,
) -> TransportEligibilityType:
    """
    Enterprise transport policy engine.
    """

    if distance_km <= BPO.safe_home_drop_radius_km:
        return "FULL_HOME_DROP"

    if distance_km <= BPO.conditional_approval_radius_km:
        return "CONDITIONAL_APPROVAL"

    return "HUB_DROP_ONLY"


# =========================================================
# NIGHT SHIFT SAFETY
# =========================================================


def requires_night_security_escort(
    shift_hour: int,
    gender: str,
) -> bool:
    """
    Enterprise female safety enforcement.
    """

    return gender == "Female" and (
        TRANSPORT_POLICY["FEMALE_ESCORT_START_HOUR"]
        <= shift_hour
        <= TRANSPORT_POLICY["FEMALE_ESCORT_END_HOUR"]
    )


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":
    print("\n=================================================")

    print("ENTERPRISE GEOGRAPHY CONFIG")

    print("=================================================\n")

    print(f"BPO: {BPO.name}")

    print(f"Location: {BPO.location.latitude}, {BPO.location.longitude}")

    print(f"\nOperational hubs: {len(COIMBATORE_HUBS)}")

    for hub_name, hub in COIMBATORE_HUBS.items():
        print(f"\n{hub_name}")

        print(f"Capacity: {hub.max_capacity}")

        print(f"Radius: {hub.operational_radius_km} km")
