"""
Enterprise Pickup Hub Allocator

File:
    src/pipeline/pickup_hub_allocator.py

Purpose:
--------
Enterprise-grade pickup and drop allocation engine.

Capabilities:
-------------
✅ KDTree nearest hub allocation
✅ Vectorized geospatial indexing
✅ Pickup hub assignment
✅ 3-tier drop eligibility engine
✅ Long-distance transport governance
✅ Escort-aware routing metadata
✅ Enterprise scalability architecture

Business Rules:
----------------
Tier 1:
    <=18 km
    FULL_HOME_DROP

Tier 2:
    18–25 km
    CONDITIONAL_APPROVAL

Tier 3:
    >25 km
    HUB_DROP_ONLY

Outputs:
--------
- pickup_allocations.csv
- pickup_hub_summary.csv
- drop_eligibility_summary.csv
- pickup_hub_distribution.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree

from src.utils.geo_math import (
    haversine_distance,
    to_local_projected_xy,
)

from src.config.geography_config import (
    BPO,
    COIMBATORE_HUBS,
)

# =========================================================
# 📁 PATH CONFIGURATION
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = (
    BASE_DIR
    / "synthetic_data"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# 📝 LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)

# =========================================================
# 🚦 TRANSPORT ELIGIBILITY TIERS
# =========================================================

FULL_HOME_DROP_RADIUS_KM = 18.0

CONDITIONAL_APPROVAL_RADIUS_KM = 25.0

# =========================================================
# 📥 LOAD EMPLOYEE DATA
# =========================================================

def load_employee_data() -> pd.DataFrame:

    logger.info(
        "Loading employee master dataset..."
    )

    employee_df = pd.read_csv(
        DATA_DIR / "employees.csv"
    )

    logger.info(
        f"Loaded employees: "
        f"{len(employee_df)}"
    )

    return employee_df

# =========================================================
# 🚕 FILTER TRANSPORT USERS
# =========================================================

def filter_transport_users(
    employee_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filters only company cab users.
    """

    logger.info(
        "Filtering company transport users..."
    )

    transport_df = employee_df[
        employee_df[
            "uses_company_transport"
        ] == 1
    ].copy()

    logger.info(
        f"Cab users: "
        f"{len(transport_df)}"
    )

    return transport_df

# =========================================================
# 🌍 BUILD HUB KDTREE
# =========================================================

def build_hub_kdtree():
    """
    Builds enterprise KDTree index
    for fast nearest-hub lookup.
    """

    logger.info(
        "Building pickup hub KDTree..."
    )

    hub_names: List[str] = []

    hub_xy = []

    for (
        hub_name,
        (
            lat,
            lon,
        ),
    ) in COIMBATORE_HUBS.items():

        x, y = to_local_projected_xy(
            lat,
            lon,
        )

        hub_names.append(
            hub_name
        )

        hub_xy.append(
            [x, y]
        )

    hub_xy = np.array(
        hub_xy
    )

    tree = KDTree(
        hub_xy
    )

    logger.info(
        "KDTree initialized successfully."
    )

    return (
        tree,
        hub_names,
        hub_xy,
    )

# =========================================================
# 📍 KDTree HUB ALLOCATION
# =========================================================

def allocate_pickup_hubs(
    transport_df: pd.DataFrame,
):
    """
    Allocates nearest pickup hubs
    using vectorized KDTree search.
    """

    logger.info(
        "Allocating pickup hubs..."
    )

    (
        tree,
        hub_names,
        hub_xy,
    ) = build_hub_kdtree()

    # -----------------------------------------------------
    # Project employee coordinates
    # -----------------------------------------------------

    projected_points = []

    for _, row in transport_df.iterrows():

        x, y = to_local_projected_xy(
            row["home_lat"],
            row["home_lon"],
        )

        projected_points.append(
            [x, y]
        )

    projected_points = np.array(
        projected_points
    )

    # -----------------------------------------------------
    # Vectorized nearest-neighbor search
    # -----------------------------------------------------

    distances, indices = tree.query(
        projected_points,
        k=1,
    )

    # -----------------------------------------------------
    # Assign hubs
    # -----------------------------------------------------

    transport_df[
        "pickup_hub"
    ] = [

        hub_names[idx[0]]

        for idx in indices
    ]

    # KDTree distance in meters
    transport_df[
        "pickup_distance_km"
    ] = (
        distances.flatten()
        / 1000.0
    ).round(2)

    logger.info(
        "Pickup hub allocation completed."
    )

    return transport_df

# =========================================================
# 🏠 DROP ELIGIBILITY ENGINE
# =========================================================

def assign_drop_eligibility(
    transport_df: pd.DataFrame,
):
    """
    Assigns enterprise transport
    eligibility tiers.
    """

    logger.info(
        "Assigning drop eligibility tiers..."
    )

    drop_distances = []

    eligibility_labels = []

    for _, row in transport_df.iterrows():

        distance_from_bpo = haversine_distance(

            BPO.lat,
            BPO.lon,

            row["home_lat"],
            row["home_lon"],
        )

        drop_distances.append(
            round(
                distance_from_bpo,
                2,
            )
        )

        # -------------------------------------------------
        # Tier 1
        # -------------------------------------------------

        if (
            distance_from_bpo
            <= FULL_HOME_DROP_RADIUS_KM
        ):

            eligibility_labels.append(
                "FULL_HOME_DROP"
            )

        # -------------------------------------------------
        # Tier 2
        # -------------------------------------------------

        elif (
            distance_from_bpo
            <= CONDITIONAL_APPROVAL_RADIUS_KM
        ):

            eligibility_labels.append(
                "CONDITIONAL_APPROVAL"
            )

        # -------------------------------------------------
        # Tier 3
        # -------------------------------------------------

        else:

            eligibility_labels.append(
                "HUB_DROP_ONLY"
            )

    transport_df[
        "distance_from_bpo_km"
    ] = drop_distances

    transport_df[
        "transport_eligibility"
    ] = eligibility_labels

    return transport_df

# =========================================================
# 📊 PICKUP HUB SUMMARY
# =========================================================

def generate_hub_summary(
    transport_df: pd.DataFrame,
):
    """
    Generates operational hub KPIs.
    """

    logger.info(
        "Generating pickup hub summary..."
    )

    summary_df = (

        transport_df

        .groupby("pickup_hub")

        .agg({

            "employee_id":
                "count",

            "pickup_distance_km":
                "mean",

            "requires_security_escort":
                "sum",
        })

        .reset_index()
    )

    summary_df = summary_df.rename(
        columns={

            "employee_id":
                "employee_count",

            "pickup_distance_km":
                "avg_pickup_distance_km",

            "requires_security_escort":
                "escort_required_count",
        }
    )

    summary_df[
        "avg_pickup_distance_km"
    ] = (
        summary_df[
            "avg_pickup_distance_km"
        ]
        .round(2)
    )

    return summary_df

# =========================================================
# 📊 DROP ELIGIBILITY SUMMARY
# =========================================================

def generate_drop_summary(
    transport_df: pd.DataFrame,
):
    """
    Generates drop eligibility metrics.
    """

    logger.info(
        "Generating drop eligibility summary..."
    )

    summary_df = (

        transport_df

        .groupby(
            "transport_eligibility"
        )

        .agg({

            "employee_id":
                "count",

            "distance_from_bpo_km":
                "mean",
        })

        .reset_index()
    )

    summary_df = summary_df.rename(
        columns={

            "employee_id":
                "employee_count",

            "distance_from_bpo_km":
                "avg_distance_from_bpo_km",
        }
    )

    summary_df[
        "avg_distance_from_bpo_km"
    ] = (
        summary_df[
            "avg_distance_from_bpo_km"
        ]
        .round(2)
    )

    return summary_df

# =========================================================
# 📈 EXPORT VISUALIZATION
# =========================================================

def export_visualization(
    hub_summary_df: pd.DataFrame,
):
    """
    Exports pickup hub distribution chart.
    """

    logger.info(
        "Exporting pickup visualization..."
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        hub_summary_df["pickup_hub"],
        hub_summary_df["employee_count"],
    )

    plt.xlabel(
        "Pickup Hub"
    )

    plt.ylabel(
        "Employee Count"
    )

    plt.title(
        "Enterprise Pickup Hub Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        DATA_DIR
        / "pickup_hub_distribution.png"
    )

    plt.close()

# =========================================================
# 📤 EXPORT OUTPUTS
# =========================================================

def export_outputs(
    transport_df: pd.DataFrame,
    hub_summary_df: pd.DataFrame,
    drop_summary_df: pd.DataFrame,
):
    """
    Exports enterprise allocation outputs.
    """

    logger.info(
        "Exporting allocation outputs..."
    )

    transport_df.to_csv(
        DATA_DIR
        / "pickup_allocations.csv",
        index=False,
    )

    hub_summary_df.to_csv(
        DATA_DIR
        / "pickup_hub_summary.csv",
        index=False,
    )

    drop_summary_df.to_csv(
        DATA_DIR
        / "drop_eligibility_summary.csv",
        index=False,
    )

    logger.info(
        "Allocation outputs exported."
    )

# =========================================================
# 🚀 MAIN PIPELINE
# =========================================================

def main():

    logger.info("=" * 60)

    logger.info(
        "Starting enterprise pickup "
        "hub allocation engine..."
    )

    # -----------------------------------------------------
    # Load employee data
    # -----------------------------------------------------

    employee_df = load_employee_data()

    # -----------------------------------------------------
    # Filter cab users
    # -----------------------------------------------------

    transport_df = filter_transport_users(
        employee_df
    )

    # -----------------------------------------------------
    # Allocate pickup hubs
    # -----------------------------------------------------

    transport_df = allocate_pickup_hubs(
        transport_df
    )

    # -----------------------------------------------------
    # Assign drop eligibility
    # -----------------------------------------------------

    transport_df = assign_drop_eligibility(
        transport_df
    )

    # -----------------------------------------------------
    # Generate summaries
    # -----------------------------------------------------

    hub_summary_df = generate_hub_summary(
        transport_df
    )

    drop_summary_df = generate_drop_summary(
        transport_df
    )

    # -----------------------------------------------------
    # Export visualization
    # -----------------------------------------------------

    export_visualization(
        hub_summary_df
    )

    # -----------------------------------------------------
    # Export outputs
    # -----------------------------------------------------

    export_outputs(
        transport_df,
        hub_summary_df,
        drop_summary_df,
    )

    logger.info(
        "Enterprise pickup hub allocator "
        "completed successfully."
    )

    print(
        "\n🚀 Enterprise pickup allocation "
        "completed successfully.\n"
    )

# =========================================================
# 🚀 ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()