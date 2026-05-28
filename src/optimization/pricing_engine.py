"""
Enterprise Mobility Pricing Engine

File:
    src/optimization/pricing_engine.py

Purpose:
- Enterprise transport economics engine
- Vendor pricing intelligence
- Route-level trip costing
- Fleet utilization analytics
- Safety-aware pricing
- Shift-wave pricing optimization
- Transport FinOps support
- Dispatch cost forecasting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import pandas as pd

# =========================================================
# TYPE CONTRACTS
# =========================================================

ShiftWaveType = Literal[
    "03:30",
    "04:30",
    "NON_TRANSPORT",
]

VehicleType = Literal[
    "SEDAN",
    "SUV",
    "VAN",
]

FuelType = Literal[
    "PETROL",
    "DIESEL",
    "CNG",
    "EV",
]

# =========================================================
# ENTERPRISE VENDOR MODEL
# =========================================================


@dataclass(frozen=True)
class VendorContract:
    """
    Enterprise transport vendor contract.
    """

    vendor_name: str

    base_km_rate: float

    night_shift_multiplier: float

    escort_charge: float

    weekend_multiplier: float

    fuel_surge_multiplier: float

    dead_km_rate: float

    cancellation_penalty: float

    safety_compliance_score: float

    SLA_score: float

    fleet_availability_score: float


# =========================================================
# ENTERPRISE FLEET MODEL
# =========================================================


@dataclass(frozen=True)
class FleetVehicle:
    """
    Enterprise fleet asset.
    """

    vehicle_id: str

    vendor_name: str

    vehicle_type: VehicleType

    seating_capacity: int

    fuel_type: FuelType

    gps_enabled: bool

    panic_button_enabled: bool

    safety_rating: float

    active: bool = True


# =========================================================
# ROUTE COST RESULT
# =========================================================


@dataclass(frozen=True)
class RouteCostResult:
    """
    Enterprise route pricing output.
    """

    total_cost: float

    cost_per_employee: float

    route_km: float

    dead_km: float

    fuel_adjustment: float

    safety_premium: float

    utilization_score: float

    vendor_name: str


# =========================================================
# ENTERPRISE VENDOR REGISTRY
# =========================================================

VENDOR_REGISTRY: Dict[
    str,
    VendorContract,
] = {

    "FastTrack Mobility": VendorContract(
        vendor_name="FastTrack Mobility",

        base_km_rate=24.0,

        night_shift_multiplier=1.18,

        escort_charge=350.0,

        weekend_multiplier=1.10,

        fuel_surge_multiplier=1.05,

        dead_km_rate=10.0,

        cancellation_penalty=500.0,

        safety_compliance_score=9.1,

        SLA_score=8.8,

        fleet_availability_score=8.5,
    ),

    "SecureRide Logistics": VendorContract(
        vendor_name="SecureRide Logistics",

        base_km_rate=26.5,

        night_shift_multiplier=1.22,

        escort_charge=400.0,

        weekend_multiplier=1.15,

        fuel_surge_multiplier=1.08,

        dead_km_rate=12.0,

        cancellation_penalty=700.0,

        safety_compliance_score=9.5,

        SLA_score=9.0,

        fleet_availability_score=8.2,
    ),

    "CityCommute Services": VendorContract(
        vendor_name="CityCommute Services",

        base_km_rate=22.0,

        night_shift_multiplier=1.12,

        escort_charge=300.0,

        weekend_multiplier=1.08,

        fuel_surge_multiplier=1.04,

        dead_km_rate=9.0,

        cancellation_penalty=450.0,

        safety_compliance_score=8.3,

        SLA_score=8.1,

        fleet_availability_score=8.8,
    ),
}

# =========================================================
# ENTERPRISE FLEET REGISTRY
# =========================================================

FLEET_REGISTRY: List[
    FleetVehicle
] = [

    FleetVehicle(
        vehicle_id="FT-SDN-001",

        vendor_name="FastTrack Mobility",

        vehicle_type="SEDAN",

        seating_capacity=4,

        fuel_type="DIESEL",

        gps_enabled=True,

        panic_button_enabled=True,

        safety_rating=8.8,
    ),

    FleetVehicle(
        vehicle_id="SR-SUV-101",

        vendor_name="SecureRide Logistics",

        vehicle_type="SUV",

        seating_capacity=6,

        fuel_type="DIESEL",

        gps_enabled=True,

        panic_button_enabled=True,

        safety_rating=9.5,
    ),

    FleetVehicle(
        vehicle_id="CC-VAN-220",

        vendor_name="CityCommute Services",

        vehicle_type="VAN",

        seating_capacity=8,

        fuel_type="CNG",

        gps_enabled=True,

        panic_button_enabled=False,

        safety_rating=8.1,
    ),
]

# =========================================================
# SHIFT-WAVE MULTIPLIER ENGINE
# =========================================================


def shift_wave_multiplier(
    shift_wave: ShiftWaveType,
) -> float:
    """
    Enterprise shift-wave pricing logic.
    """

    if shift_wave == "03:30":
        return 1.20

    if shift_wave == "04:30":
        return 1.10

    return 1.0


# =========================================================
# HUB COST ENGINE
# =========================================================


def hub_cost_multiplier(
    hub_name: str,
) -> float:
    """
    Hub-based operational cost modeling.
    """

    multipliers = {

        "Saravanampatti": 1.00,

        "Singanallur": 1.05,

        "Thudiyalur": 1.08,

        "Hopes": 1.03,

        "Ganapathy": 1.06,
    }

    return multipliers.get(
        hub_name,
        1.0,
    )


# =========================================================
# UTILIZATION ANALYTICS
# =========================================================


def utilization_score(
    employee_count: int,
    vehicle_capacity: int,
) -> float:
    """
    Enterprise fleet utilization efficiency.
    """

    if vehicle_capacity <= 0:
        return 0.0

    return round(
        employee_count
        / vehicle_capacity,
        2,
    )


# =========================================================
# ROUTE COST ENGINE
# =========================================================


def compute_route_trip_cost(
    vendor_name: str,
    route_distance_km: float,
    dead_km: float,
    employee_count: int,
    shift_wave: ShiftWaveType,
    hub_name: str,
    escort_required: bool = False,
    weekend_trip: bool = False,
    fuel_price_index: float = 1.0,
    vehicle_capacity: int = 4,
) -> RouteCostResult:
    """
    Enterprise route-level pricing engine.
    """

    vendor = VENDOR_REGISTRY[
        vendor_name
    ]

    # =====================================================
    # BASE DISTANCE COST
    # =====================================================

    total_km = (
        route_distance_km
        + dead_km
    )

    base_cost = (
        total_km
        * vendor.base_km_rate
    )

    # =====================================================
    # SHIFT-WAVE SURGE
    # =====================================================

    base_cost *= (
        shift_wave_multiplier(
            shift_wave
        )
    )

    # =====================================================
    # HUB MULTIPLIER
    # =====================================================

    base_cost *= (
        hub_cost_multiplier(
            hub_name
        )
    )

    # =====================================================
    # WEEKEND SURGE
    # =====================================================

    if weekend_trip:

        base_cost *= (
            vendor.weekend_multiplier
        )

    # =====================================================
    # FUEL INFLATION
    # =====================================================

    fuel_adjustment = (
        base_cost
        * (
            vendor.fuel_surge_multiplier
            - 1
        )
        * fuel_price_index
    )

    total_cost = (
        base_cost
        + fuel_adjustment
    )

    # =====================================================
    # ESCORT PREMIUM
    # =====================================================

    safety_premium = 0.0

    if escort_required:

        safety_premium = (
            vendor.escort_charge
        )

        total_cost += (
            safety_premium
        )

    # =====================================================
    # DEAD KM COST
    # =====================================================

    total_cost += (
        dead_km
        * vendor.dead_km_rate
    )

    # =====================================================
    # UTILIZATION ANALYTICS
    # =====================================================

    utilization = (
        utilization_score(
            employee_count,
            vehicle_capacity,
        )
    )

    # Underutilized routes are expensive
    if utilization < 0.50:

        total_cost *= 1.15

    # =====================================================
    # COST PER EMPLOYEE
    # =====================================================

    cost_per_employee = (
        total_cost
        / max(employee_count, 1)
    )

    return RouteCostResult(

        total_cost=round(
            total_cost,
            2,
        ),

        cost_per_employee=round(
            cost_per_employee,
            2,
        ),

        route_km=round(
            route_distance_km,
            2,
        ),

        dead_km=round(
            dead_km,
            2,
        ),

        fuel_adjustment=round(
            fuel_adjustment,
            2,
        ),

        safety_premium=round(
            safety_premium,
            2,
        ),

        utilization_score=utilization,

        vendor_name=vendor_name,
    )


# =========================================================
# VENDOR OPTIMIZATION ENGINE
# =========================================================


def select_optimal_vendor(
    route_distance_km: float,
    shift_wave: ShiftWaveType,
    escort_required: bool,
    employee_count: int,
    hub_name: str,
) -> RouteCostResult:
    """
    Enterprise vendor optimization engine.
    """

    best_result = None

    lowest_cost = float("inf")

    for vendor_name in (
        VENDOR_REGISTRY.keys()
    ):

        result = (
            compute_route_trip_cost(
                vendor_name=vendor_name,

                route_distance_km=(
                    route_distance_km
                ),

                dead_km=5.0,

                employee_count=(
                    employee_count
                ),

                shift_wave=shift_wave,

                hub_name=hub_name,

                escort_required=(
                    escort_required
                ),
            )
        )

        if (
            result.total_cost
            < lowest_cost
        ):

            lowest_cost = (
                result.total_cost
            )

            best_result = result

    if best_result is None:

        raise ValueError(
            "No vendor pricing result generated."
        )

    return best_result


# =========================================================
# FLEET ANALYTICS
# =========================================================


def fleet_registry_dataframe(
) -> pd.DataFrame:
    """
    Fleet analytics export.
    """

    return pd.DataFrame([

        vars(vehicle)

        for vehicle in FLEET_REGISTRY
    ])


# =========================================================
# VENDOR ANALYTICS
# =========================================================


def vendor_registry_dataframe(
) -> pd.DataFrame:
    """
    Vendor analytics export.
    """

    return pd.DataFrame([

        vars(vendor)

        for vendor in (
            VENDOR_REGISTRY.values()
        )
    ])


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    result = (
        compute_route_trip_cost(
            vendor_name="FastTrack Mobility",

            route_distance_km=28.0,

            dead_km=6.0,

            employee_count=4,

            shift_wave="03:30",

            hub_name="Saravanampatti",

            escort_required=True,

            weekend_trip=False,

            fuel_price_index=1.1,

            vehicle_capacity=4,
        )
    )

    print(
        "\n================================================="
    )

    print(
        "ENTERPRISE ROUTE COST SUMMARY"
    )

    print(
        "=================================================\n"
    )

    print(result)

    print(
        "\n=================================================\n"
    )

    optimal = (
        select_optimal_vendor(
            route_distance_km=32,

            shift_wave="04:30",

            escort_required=False,

            employee_count=5,

            hub_name="Singanallur",
        )
    )

    print(
        "OPTIMAL VENDOR SELECTION"
    )

    print(optimal)