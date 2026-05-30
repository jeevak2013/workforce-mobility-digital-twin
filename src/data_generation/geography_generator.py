"""
Enterprise Geography Generator

File:
    src/data_generation/geography_generator.py

Purpose:
- Enterprise workforce geospatial simulation
- Realistic employee home generation
- Hub-centric population modeling
- Transport corridor simulation
- Long-distance outlier generation
- Workforce density simulation
- Operational geography intelligence
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from src.config.geography_config import (
    BPO,
    COIMBATORE_HUBS,
    HubConfig,
    HubLiteral,
)

from src.utils.geo_math import (
    haversine_distance,
)

from typing import cast

# =========================================================
# EARTH CONSTANTS
# =========================================================

KM_PER_DEGREE_LATITUDE = 111.0

# =========================================================
# RANDOM GEO OFFSET ENGINE
# =========================================================


def generate_geo_offset(
    distance_km: float,
    angle_radians: float,
) -> Tuple[float, float]:
    """
    Converts radial distance into
    latitude/longitude offset.
    """

    delta_lat = distance_km * math.cos(angle_radians) / KM_PER_DEGREE_LATITUDE

    delta_lon = (
        distance_km
        * math.sin(angle_radians)
        / (KM_PER_DEGREE_LATITUDE * math.cos(math.radians(BPO.location.latitude)))
    )

    return delta_lat, delta_lon


# =========================================================
# RANDOM HOME DROP GENERATOR
# =========================================================


def generate_random_home_drop(
    hub_center: HubConfig,
    max_scatter_km: float,
    rng: random.Random,
) -> Tuple[float, float]:
    """
    Generates realistic employee home location
    around operational hub.

    Features:
    - radial distribution
    - realistic workforce spread
    - operational corridor simulation
    """

    distance = rng.uniform(
        0.5,
        max_scatter_km,
    )

    angle = rng.uniform(
        0,
        2 * math.pi,
    )

    delta_lat, delta_lon = generate_geo_offset(
        distance,
        angle,
    )

    latitude = hub_center.latitude + delta_lat

    longitude = hub_center.longitude + delta_lon

    return (
        round(latitude, 6),
        round(longitude, 6),
    )


# =========================================================
# LONG-DISTANCE OUTLIER ENGINE
# =========================================================


def generate_long_distance_outlier(
    rng: random.Random,
) -> Tuple[float, float]:
    """
    Generates realistic long-distance
    employee outliers.

    Used for:
    - HUB_DROP_ONLY scenarios
    - routing stress testing
    - vendor optimization
    - dispatch realism
    """

    hub_name = rng.choice(list(COIMBATORE_HUBS.keys()))

    hub = COIMBATORE_HUBS[hub_name]

    distance = rng.uniform(
        22,
        38,
    )

    angle = rng.uniform(
        0,
        2 * math.pi,
    )

    delta_lat, delta_lon = generate_geo_offset(
        distance,
        angle,
    )

    latitude = hub.latitude + delta_lat

    longitude = hub.longitude + delta_lon

    return (
        round(latitude, 6),
        round(longitude, 6),
    )


# =========================================================
# ENTERPRISE OPERATIONAL BOUNDARY
# =========================================================


def within_operational_boundary(
    latitude: float,
    longitude: float,
) -> bool:
    """
    Validates whether coordinate
    falls inside enterprise transport radius.
    """

    distance = haversine_distance(
        latitude,
        longitude,
        BPO.location.latitude,
        BPO.location.longitude,
    )

    return distance <= BPO.operational_radius_km


# =========================================================
# ENTERPRISE WORKFORCE LOCATION ENGINE
# =========================================================


def generate_employee_location(
    rng: random.Random,
    outlier_probability: float = 0.07,
) -> Tuple[
    float,
    float,
    HubLiteral,
]:
    """
    Enterprise workforce spatial simulator.

    Features:
    - realistic corridor distributions
    - hub-centric geography
    - operational outliers
    - transport-aware generation
    """

    # -----------------------------------------------------
    # Weighted operational corridors
    # -----------------------------------------------------

    hub_weights = {
        "Saravanampatti": 0.32,
        "Singanallur": 0.25,
        "Thudiyalur": 0.18,
        "Hopes": 0.15,
        "Ganapathy": 0.10,
    }

    hub_name = cast(
        HubLiteral,
        rng.choices(
            population=list(hub_weights.keys()),
            weights=list(hub_weights.values()),
            k=1,
        )[0],
    )

    hub = COIMBATORE_HUBS[hub_name]

    # -----------------------------------------------------
    # Long-distance outlier logic
    # -----------------------------------------------------

    if rng.random() < outlier_probability:
        latitude, longitude = generate_long_distance_outlier(rng)

    else:
        max_scatter = rng.uniform(
            3,
            hub.operational_radius_km,
        )

        latitude, longitude = generate_random_home_drop(
            hub_center=hub,
            max_scatter_km=max_scatter,
            rng=rng,
        )

    return (
        latitude,
        longitude,
        hub_name,
    )


# =========================================================
# BULK LOCATION GENERATOR
# =========================================================


def generate_bulk_employee_locations(
    employee_count: int,
    seed: int = 42,
) -> List[
    Tuple[
        float,
        float,
        HubLiteral,
    ]
]:
    """
    Bulk enterprise workforce geography generation.
    """

    rng = random.Random(seed)

    locations = []

    for _ in range(employee_count):
        location = generate_employee_location(rng)

        locations.append(location)

    return locations


# =========================================================
# WORKFORCE DENSITY ENGINE
# =========================================================


def generate_density_cluster(
    hub_name: HubLiteral,
    cluster_size: int,
    cluster_radius_km: float,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    """
    Generates dense employee clusters.

    Useful for:
    - cab pooling simulation
    - route optimization
    - demand hotspots
    """

    hub = COIMBATORE_HUBS[hub_name]

    cluster = []

    for _ in range(cluster_size):
        latitude, longitude = generate_random_home_drop(
            hub_center=hub,
            max_scatter_km=cluster_radius_km,
            rng=rng,
        )

        cluster.append(
            (
                latitude,
                longitude,
            )
        )

    return cluster


# =========================================================
# SAFETY RISK GEO ENGINE
# =========================================================


def generate_high_risk_zone_employee(
    rng: random.Random,
) -> Tuple[
    float,
    float,
]:
    """
    Generates employees in difficult
    operational zones.

    Used for:
    - escort testing
    - dispatch edge cases
    - routing optimization stress tests
    """

    distance = rng.uniform(
        25,
        40,
    )

    angle = rng.uniform(
        0,
        2 * math.pi,
    )

    delta_lat, delta_lon = generate_geo_offset(
        distance,
        angle,
    )

    latitude = BPO.location.latitude + delta_lat

    longitude = BPO.location.longitude + delta_lon

    return (
        round(latitude, 6),
        round(longitude, 6),
    )


# =========================================================
# ROUTE SCATTER ANALYTICS
# =========================================================


def average_distance_from_bpo(
    coordinates: List[Tuple[float, float]],
) -> float:
    """
    Computes workforce spread
    from enterprise headquarters.
    """

    if not coordinates:
        return 0.0

    distances = []

    for lat, lon in coordinates:
        distances.append(
            haversine_distance(
                lat,
                lon,
                BPO.location.latitude,
                BPO.location.longitude,
            )
        )

    return round(
        sum(distances) / len(distances),
        2,
    )


# =========================================================
# OPERATIONAL HEAT ZONE DETECTOR
# =========================================================


def identify_operational_zone(
    distance_km: float,
) -> str:
    """
    Classifies workforce geography zone.
    """

    if distance_km <= 8:
        return "CORE_ZONE"

    if distance_km <= 18:
        return "NORMAL_OPERATION_ZONE"

    if distance_km <= 25:
        return "EXTENDED_OPERATION_ZONE"

    return "HIGH_COST_ZONE"


# =========================================================
# ENTERPRISE GEO VALIDATION
# =========================================================


def validate_generated_coordinate(
    latitude: float,
    longitude: float,
) -> bool:
    """
    Enterprise coordinate validator.
    """

    if not (10.5 <= latitude <= 11.5):
        return False

    if not (76.5 <= longitude <= 77.5):
        return False

    return True


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":
    rng = random.Random(42)

    print("\n=================================================")

    print("ENTERPRISE GEOGRAPHY GENERATOR")

    print("=================================================\n")

    for i in range(5):
        lat, lon, hub = generate_employee_location(rng)

        distance = haversine_distance(
            lat,
            lon,
            BPO.location.latitude,
            BPO.location.longitude,
        )

        print(f"Employee {i + 1}")

        print(f"Hub: {hub}")

        print(f"Coordinate: {lat}, {lon}")

        print(f"Distance from BPO: {distance:.2f} km")

        print(f"Valid: {validate_generated_coordinate(lat, lon)}")

        print()
