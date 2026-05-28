"""
Enterprise Geospatial + Workforce Transportation Domain Schemas
Coimbatore Healthcare BPO Fleet Optimization Platform

Architectural Goals:
- Immutable domain contracts
- Centralized geospatial mathematics
- Enterprise-safe validation
- Spatial ML readiness
- HDBSCAN / OR-Tools compatibility
- Female safety optimization readiness
- Temporal forecasting compatibility
- Enterprise transport orchestration readiness
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Literal,
    List,
    Dict,
    Tuple,
    TYPE_CHECKING,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

if TYPE_CHECKING:
    import pandas as pd

# =========================================================
# 🌍 CENTRALIZED GEOSPATIAL MATHEMATICAL ENGINE
# =========================================================

EARTH_RADIUS_KM: float = 6371.0088

CENTRAL_LATITUDE: float = 11.045706877864594

CENTRAL_LONGITUDE: float = 77.01213631503671

FALSE_EASTING: float = 500000.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Computes numerically stable geodesic distance.
    """

    phi1 = math.radians(lat1)

    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)

    dlambda = math.radians(lon2 - lon1)

    a = (

        math.sin(dphi / 2) ** 2

        +

        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    a = min(
        1.0,
        max(0.0, a),
    )

    return (
        2
        * EARTH_RADIUS_KM
        * math.asin(math.sqrt(a))
    )


def to_local_projected_xy(
    lat: float,
    lon: float,
) -> Tuple[float, float]:
    """
    Converts WGS84 coordinates into
    localized projected Cartesian coordinates.
    """

    r_meters = (
        EARTH_RADIUS_KM
        * 1000.0
    )

    lat_diff_rad = math.radians(
        lat - CENTRAL_LATITUDE
    )

    lon_diff_rad = math.radians(
        lon - CENTRAL_LONGITUDE
    )

    center_lat_rad = math.radians(
        CENTRAL_LATITUDE
    )

    x_meters = (

        r_meters
        * lon_diff_rad
        * math.cos(center_lat_rad)

        +

        FALSE_EASTING
    )

    y_meters = (
        r_meters
        * lat_diff_rad
    )

    return (
        x_meters,
        y_meters,
    )

# =========================================================
# 📊 CENTRALIZED CONFIGURATION REGISTRY
# =========================================================

@dataclass(frozen=True)
class BPOConfig:
    """
    Centralized enterprise deployment configuration.
    """

    lat: float = CENTRAL_LATITUDE

    lon: float = CENTRAL_LONGITUDE

    max_radius_km: float = 18.0

    gender_female_target: float = 0.80

    gender_tolerance: float = 0.02

    max_ride_time_minutes: int = 60

    cab_capacity_default: int = 4


BPO = BPOConfig()

# =========================================================
# 🏢 FIXED GEOGRAPHIC HUB REGISTRY
# =========================================================

HUB_COORDS: Dict[
    str,
    Tuple[float, float],
] = {

    "Thudiyalur":
        (11.0850, 76.9840),

    "Saravanampatti":
        (11.0730, 77.0600),

    "Singanallur":
        (11.0010, 77.0450),

    "Hopes":
        (10.9970, 77.0100),

    "Ganapathy":
        (11.0370, 77.0030),
}

HubLiteral = Literal[
    "Thudiyalur",
    "Saravanampatti",
    "Singanallur",
    "Hopes",
    "Ganapathy",
]

# =========================================================
# 📍 CORE GEOGRAPHIC VALUE OBJECTS
# =========================================================

class Coordinate(BaseModel):
    """
    Immutable geographic coordinate representation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    latitude: float

    longitude: float

    # -----------------------------------------------------
    # VALIDATORS
    # -----------------------------------------------------

    @field_validator("latitude")
    @classmethod
    def validate_latitude(
        cls,
        value: float,
    ) -> float:

        if not -90.0 <= value <= 90.0:

            raise ValueError(
                "Latitude must remain between -90 and 90."
            )

        return value

    @field_validator("longitude")
    @classmethod
    def validate_longitude(
        cls,
        value: float,
    ) -> float:

        if not -180.0 <= value <= 180.0:

            raise ValueError(
                "Longitude must remain between -180 and 180."
            )

        return value

    # -----------------------------------------------------
    # GEOSPATIAL UTILITIES
    # -----------------------------------------------------

    def distance_to_bpo(self) -> float:

        return haversine_distance(

            self.latitude,
            self.longitude,

            BPO.lat,
            BPO.lon,
        )

    def projected_xy(
        self,
    ) -> Tuple[float, float]:

        return to_local_projected_xy(

            self.latitude,
            self.longitude,
        )

# =========================================================
# 🏢 LOGISTICS HUB DOMAIN OBJECT
# =========================================================

class Hub(BaseModel):
    """
    Immutable logistics routing corridor node.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    name: HubLiteral

    location: Coordinate

    @model_validator(mode="after")
    def validate_hub_within_radius(
        self,
    ) -> "Hub":

        dist = (
            self.location.distance_to_bpo()
        )

        if dist > BPO.max_radius_km:

            raise ValueError(
                f"Hub '{self.name}' violates "
                f"enterprise operational radius."
            )

        return self

# =========================================================
# 👤 EMPLOYEE DOMAIN ENTITY
# =========================================================

class Employee(BaseModel):
    """
    Immutable enterprise workforce entity.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # -----------------------------------------------------
    # CORE IDENTITY
    # -----------------------------------------------------

    employee_id: int = Field(
        ...,
        description="Unique employee identifier",
    )

    gender: Literal[
        "Female",
        "Male",
    ]

    # -----------------------------------------------------
    # GEOSPATIAL ATTRIBUTES
    # -----------------------------------------------------

    home: Coordinate

    hub: HubLiteral

    pickup_hub: HubLiteral

    # -----------------------------------------------------
    # TRANSPORT ATTRIBUTES
    # -----------------------------------------------------

    uses_company_transport: bool = True

    transport_shift: Literal[
        "03:30",
        "04:30",
        "NON_TRANSPORT",
    ]

    transport_eligibility: Literal[
        "FULL_HOME_DROP",
        "CONDITIONAL_APPROVAL",
        "HUB_DROP_ONLY",
    ] = "FULL_HOME_DROP"

    extension_category: Literal[
        "NO_EXTENSION",
        "EXTEND_TO_0430",
        "EXTEND_BEYOND_0630",
    ] = "NO_EXTENSION"

    # -----------------------------------------------------
    # TEMPORAL ATTRIBUTES
    # -----------------------------------------------------

    shift_logout: datetime

    predicted_extension_minutes: int = Field(
        default=0,
        ge=0,
        le=240,
    )

    # -----------------------------------------------------
    # SAFETY ATTRIBUTES
    # -----------------------------------------------------

    safety_priority_score: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
    )

    requires_security_escort: bool = False

    home_distance_km: float

    # -----------------------------------------------------
    # VALIDATORS
    # -----------------------------------------------------

    @field_validator(
        "home",
        mode="after",
    )
    @classmethod
    def validate_employee_radius(
        cls,
        value: Coordinate,
    ) -> Coordinate:
        """
        Enterprise routing boundary governance.
        """

        dist = value.distance_to_bpo()

        if dist > 40:

            raise ValueError(
                f"Employee home coordinate exceeds "
                f"maximum simulation boundary "
                f"({dist:.2f} km from BPO)."
            )

        return value

    # -----------------------------------------------------
    # GEOSPATIAL UTILITIES
    # -----------------------------------------------------

    @property
    def hub_coordinate(
        self,
    ) -> Coordinate:

        lat, lon = HUB_COORDS[
            self.hub
        ]

        return Coordinate(
            latitude=lat,
            longitude=lon,
        )

    def distance_to_hub(
        self,
    ) -> float:

        lat, lon = HUB_COORDS[
            self.hub
        ]

        return haversine_distance(

            self.home.latitude,
            self.home.longitude,

            lat,
            lon,
        )

    def projected_xy(
        self,
    ) -> Tuple[float, float]:

        return self.home.projected_xy()

    # -----------------------------------------------------
    # TEMPORAL UTILITIES
    # -----------------------------------------------------

    @property
    def effective_logout_time(
        self,
    ) -> datetime:

        from datetime import timedelta

        return (
            self.shift_logout
            + timedelta(
                minutes=self.predicted_extension_minutes
            )
        )

    # -----------------------------------------------------
    # SAFETY UTILITIES
    # -----------------------------------------------------

    @property
    def female_safety_flag(
        self,
    ) -> int:

        return (
            1
            if self.gender == "Female"
            else 0
        )

    # -----------------------------------------------------
    # TRANSPORT UTILITIES
    # -----------------------------------------------------

    @property
    def requires_home_drop(
        self,
    ) -> bool:

        return (
            self.transport_eligibility
            in [
                "FULL_HOME_DROP",
                "CONDITIONAL_APPROVAL",
            ]
        )

    @property
    def is_transport_user(
        self,
    ) -> int:

        return (
            1
            if self.uses_company_transport
            else 0
        )

    # -----------------------------------------------------
    # ML FEATURE VECTOR
    # -----------------------------------------------------

    def clustering_feature_vector(
        self,
    ) -> List[float]:
        """
        Enterprise clustering embedding.
        """

        x, y = self.projected_xy()

        logout_minutes = (

            self.effective_logout_time.hour
            * 60

            +

            self.effective_logout_time.minute
        )

        return [

            # Spatial
            x,
            y,

            # Temporal
            float(logout_minutes),

            # Overtime
            float(
                self.predicted_extension_minutes
            ),

            # Safety
            float(
                self.safety_priority_score
            ),

            float(
                self.female_safety_flag
            ),

            # Transport
            float(
                self.is_transport_user
            ),
        ]

# =========================================================
# 👥 EMPLOYEE POOL AGGREGATE ROOT
# =========================================================

class EmployeePool(BaseModel):
    """
    Enterprise aggregate workforce abstraction.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    employees: List[Employee]

    # -----------------------------------------------------
    # VALIDATORS
    # -----------------------------------------------------

    @model_validator(mode="after")
    def validate_gender_ratio(
        self,
    ) -> "EmployeePool":

        if not self.employees:
            return self

        female_count = sum(

            1

            for emp in self.employees

            if emp.gender == "Female"
        )

        female_ratio = (
            female_count
            / len(self.employees)
        )

        lower = (
            BPO.gender_female_target
            - BPO.gender_tolerance
        )

        upper = (
            BPO.gender_female_target
            + BPO.gender_tolerance
        )

        if not lower <= female_ratio <= upper:

            raise ValueError(
                f"Female ratio "
                f"({female_ratio:.1%}) violates "
                f"enterprise staffing policy."
            )

        return self

    # -----------------------------------------------------
    # INGESTION UTILITIES
    # -----------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
    ) -> "EmployeePool":

        employees = []

        for _, row in df.iterrows():

            employees.append(

                Employee(

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

                    shift_logout=row[
                        "shift_logout"
                    ],

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
            )

        return cls(
            employees=employees
        )

    # -----------------------------------------------------
    # QUERY UTILITIES
    # -----------------------------------------------------

    def by_hub(
        self,
        hub: HubLiteral,
    ) -> List[Employee]:

        return [

            emp

            for emp in self.employees

            if emp.hub == hub
        ]

    def female_employees(
        self,
    ) -> List[Employee]:

        return [

            emp

            for emp in self.employees

            if emp.gender == "Female"
        ]

    def escort_required_employees(
        self,
    ) -> List[Employee]:

        return [

            emp

            for emp in self.employees

            if emp.requires_security_escort
        ]

    # -----------------------------------------------------
    # ANALYTICS UTILITIES
    # -----------------------------------------------------

    def average_extension_minutes(
        self,
    ) -> float:

        if not self.employees:
            return 0.0

        return (

            sum(
                emp.predicted_extension_minutes
                for emp in self.employees
            )

            /

            len(self.employees)
        )

    def total_high_priority_employees(
        self,
        threshold: float = 7.0,
    ) -> int:

        return sum(

            1

            for emp in self.employees

            if (
                emp.safety_priority_score
                >= threshold
            )
        )

    # -----------------------------------------------------
    # CONTAINER UTILITIES
    # -----------------------------------------------------

    def __len__(self):

        return len(self.employees)

    def __iter__(self):

        return iter(self.employees)

    def __getitem__(
        self,
        item,
    ):

        return self.employees[item]