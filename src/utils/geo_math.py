"""
Enterprise Geospatial Intelligence Engine

File:
    src/utils/geo_math.py

Purpose:
- Enterprise workforce mobility mathematics
- Transport routing utilities
- Spatial optimization support
- Cab pooling analytics
- Dispatch intelligence
- Hub allocation support
- Safety-aware geospatial computations
"""

from __future__ import annotations

import math
from typing import List, Tuple

# =========================================================
# EARTH CONSTANTS
# =========================================================

EARTH_RADIUS_KM = 6371.0088

# =========================================================
# CORE HAVERSINE ENGINE
# =========================================================


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Computes geodesic distance between
    two coordinates using Haversine formula.

    Returns:
        Distance in kilometers.
    """

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
        EARTH_RADIUS_KM * c,
        3,
    )


# =========================================================
# DISTANCE MATRIX ENGINE
# =========================================================


def build_distance_matrix_km(
    coordinates: List[
        Tuple[float, float]
    ],
) -> List[List[float]]:
    """
    Builds enterprise distance matrix.

    Used for:
    - OR-Tools VRP
    - Route optimization
    - Cab pooling
    - Vendor allocation
    """

    n = len(coordinates)

    matrix = []

    for i in range(n):

        row = []

        lat1, lon1 = coordinates[i]

        for j in range(n):

            lat2, lon2 = coordinates[j]

            dist = haversine_distance(
                lat1,
                lon1,
                lat2,
                lon2,
            )

            row.append(dist)

        matrix.append(row)

    return matrix


# =========================================================
# HUB DISTANCE ENGINE
# =========================================================


def nearest_hub_distance(
    employee_lat: float,
    employee_lon: float,
    hub_lat: float,
    hub_lon: float,
) -> float:
    """
    Computes employee-to-hub distance.
    """

    return haversine_distance(
        employee_lat,
        employee_lon,
        hub_lat,
        hub_lon,
    )


# =========================================================
# SAFE DROP VALIDATION
# =========================================================


def is_within_safe_home_drop_radius(
    distance_km: float,
    threshold_km: float = 18.0,
) -> bool:
    """
    Enterprise home-drop policy engine.

    Default:
        <=18km → full home drop eligible
    """

    return distance_km <= threshold_km


# =========================================================
# TRANSPORT ELIGIBILITY ENGINE
# =========================================================


def compute_transport_eligibility(
    distance_km: float,
) -> str:
    """
    Enterprise transport policy classification.
    """

    if distance_km <= 18:
        return "FULL_HOME_DROP"

    if distance_km <= 25:
        return "CONDITIONAL_APPROVAL"

    return "HUB_DROP_ONLY"


# =========================================================
# CLUSTER COMPACTNESS
# =========================================================


def cluster_compactness_score(
    coordinates: List[
        Tuple[float, float]
    ],
) -> float:
    """
    Computes average intra-cluster distance.

    Lower score:
    - better pooling
    - lower fuel cost
    - higher routing efficiency
    """

    if len(coordinates) <= 1:
        return 0.0

    total_distance = 0.0

    pair_count = 0

    for i in range(len(coordinates)):

        for j in range(
            i + 1,
            len(coordinates),
        ):

            total_distance += (
                haversine_distance(
                    coordinates[i][0],
                    coordinates[i][1],
                    coordinates[j][0],
                    coordinates[j][1],
                )
            )

            pair_count += 1

    if pair_count == 0:
        return 0.0

    return round(
        total_distance / pair_count,
        3,
    )


# =========================================================
# EMPLOYEE DENSITY ENGINE
# =========================================================


def employee_density_per_sq_km(
    employee_count: int,
    radius_km: float,
) -> float:
    """
    Workforce density estimator.

    Useful for:
    - hub planning
    - demand forecasting
    - vendor optimization
    """

    if radius_km <= 0:
        return 0.0

    area = (
        math.pi
        * (radius_km ** 2)
    )

    return round(
        employee_count / area,
        2,
    )


# =========================================================
# ENTERPRISE TRAVEL TIME ENGINE
# =========================================================


def estimate_travel_time_minutes(
    distance_km: float,
    avg_speed_kmph: float = 24.0,
) -> float:
    """
    Enterprise travel-time estimator.

    Coimbatore night-shift routing assumption:
        ~24 kmph effective average speed
    """

    if avg_speed_kmph <= 0:
        return 0.0

    return round(
        (distance_km / avg_speed_kmph)
        * 60,
        1,
    )


# =========================================================
# CAB POOLING FEASIBILITY
# =========================================================


def feasible_pooling_distance(
    max_employee_separation_km: float,
    threshold_km: float = 6.0,
) -> bool:
    """
    Determines whether employees
    can realistically share a cab.
    """

    return (
        max_employee_separation_km
        <= threshold_km
    )


# =========================================================
# SPATIAL RISK SCORING
# =========================================================


def spatial_risk_score(
    distance_km: float,
    night_shift: bool,
    female_employee: bool,
) -> float:
    """
    Enterprise transport risk score.

    Used for:
    - safety prioritization
    - escort planning
    - dispatch optimization
    """

    score = 1.0

    if distance_km > 18:
        score += 2.0

    if distance_km > 25:
        score += 1.5

    if night_shift:
        score += 2.0

    if female_employee:
        score += 3.0

    return round(
        min(score, 10.0),
        2,
    )


# =========================================================
# ROUTE EFFICIENCY SCORE
# =========================================================


def route_efficiency_score(
    total_route_distance_km: float,
    employee_count: int,
) -> float:
    """
    Computes enterprise route efficiency.

    Lower distance per employee
    indicates better optimization.
    """

    if employee_count <= 0:
        return 0.0

    return round(
        total_route_distance_km
        / employee_count,
        2,
    )


# =========================================================
# DETOUR ANALYSIS
# =========================================================


def detour_percentage(
    direct_distance_km: float,
    actual_route_distance_km: float,
) -> float:
    """
    Computes routing detour overhead.
    """

    if direct_distance_km <= 0:
        return 0.0

    extra = (
        actual_route_distance_km
        - direct_distance_km
    )

    return round(
        (extra / direct_distance_km)
        * 100,
        2,
    )


# =========================================================
# ENTERPRISE HUB LOAD BALANCING
# =========================================================


def hub_load_score(
    employee_count: int,
    optimal_capacity: int,
) -> float:
    """
    Computes hub utilization ratio.

    Useful for:
    - operational balancing
    - vendor planning
    - dynamic hub expansion
    """

    if optimal_capacity <= 0:
        return 0.0

    return round(
        employee_count
        / optimal_capacity,
        2,
    )


# =========================================================
# GEO-FENCE VALIDATION
# =========================================================


def within_geofence(
    employee_lat: float,
    employee_lon: float,
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> bool:
    """
    Validates whether employee lies
    within operational geofence.
    """

    distance = haversine_distance(
        employee_lat,
        employee_lon,
        center_lat,
        center_lon,
    )

    return distance <= radius_km


# =========================================================
# OPERATIONAL SCATTER SCORE
# =========================================================


def operational_scatter_score(
    coordinates: List[
        Tuple[float, float]
    ],
) -> float:
    """
    Measures geographic spread.

    Higher spread:
    - harder routing
    - higher fuel cost
    - lower pooling efficiency
    """

    if len(coordinates) <= 1:
        return 0.0

    center_lat = sum(
        c[0]
        for c in coordinates
    ) / len(coordinates)

    center_lon = sum(
        c[1]
        for c in coordinates
    ) / len(coordinates)

    distances = []

    for lat, lon in coordinates:

        distances.append(
            haversine_distance(
                lat,
                lon,
                center_lat,
                center_lon,
            )
        )

    return round(
        sum(distances)
        / len(distances),
        3,
    )


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    dist = haversine_distance(
        11.0168,
        76.9558,
        11.0270,
        76.9650,
    )

    print(
        f"Distance: {dist} km"
    )

    print(
        compute_transport_eligibility(
            dist
        )
    )