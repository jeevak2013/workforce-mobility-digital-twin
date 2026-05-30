from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import numpy as np

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    DBSCAN,
    AgglomerativeClustering,
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "cluster_output"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(message)s"),
)

logger = logging.getLogger(__name__)

EMPLOYEE_FILE = DATA_DIR / "employees_after_churn.csv"


def load_employee_data() -> pd.DataFrame:

    logger.info("Loading employees...")

    df = pd.read_csv(EMPLOYEE_FILE)

    logger.info(
        "Rows loaded: %s",
        f"{len(df):,}",
    )

    return df


def filter_transport_users(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Filtering transport users...")

    df = df[
        (df["uses_company_transport"] == True)
        & (df["transport_shift"] != "NON_TRANSPORT")
    ].copy()

    logger.info(
        "Transport users: %s",
        f"{len(df):,}",
    )

    return df


def prepare_clustering_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Preparing clustering dataset...")

    cluster_df = df[
        [
            "employee_id",
            "hub",
            "transport_shift",
            "home_lat",
            "home_lon",
        ]
    ].copy()

    logger.info(
        "Clustering rows: %s",
        f"{len(cluster_df):,}",
    )

    return cluster_df


# ============================================================
# CLUSTERING BENCHMARK
# ============================================================


def create_feature_matrix(
    df: pd.DataFrame,
) -> np.ndarray:

    return df[
        [
            "home_lat",
            "home_lon",
        ]
    ].values


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict:

    unique_clusters = len(set(labels))

    if unique_clusters <= 1:
        return {
            "silhouette": -1,
            "davies_bouldin": np.inf,
            "calinski_harabasz": 0,
            "cluster_count": unique_clusters,
        }

    return {
        "silhouette": silhouette_score(
            X,
            labels,
        ),
        "davies_bouldin": davies_bouldin_score(
            X,
            labels,
        ),
        "calinski_harabasz": calinski_harabasz_score(
            X,
            labels,
        ),
        "cluster_count": unique_clusters,
    }


def benchmark_kmeans(
    X: np.ndarray,
) -> tuple:

    results = []

    best_score = -1

    best_model = None

    best_labels = None

    for k in range(2, 15):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X)

        metrics = evaluate_clustering(
            X,
            labels,
        )

        metrics["model"] = "KMeans"

        metrics["k"] = k

        results.append(metrics)

        if metrics["silhouette"] > best_score:
            best_score = metrics["silhouette"]

            best_model = model

            best_labels = labels

    return (
        pd.DataFrame(results),
        best_model,
        best_labels,
    )


def benchmark_minibatch_kmeans(
    X: np.ndarray,
) -> tuple:

    results = []

    best_score = -1

    best_model = None

    best_labels = None

    for k in range(2, 15):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=64,
            n_init=10,
        )

        labels = model.fit_predict(X)

        metrics = evaluate_clustering(
            X,
            labels,
        )

        metrics["model"] = "MiniBatchKMeans"

        metrics["k"] = k

        results.append(metrics)

        if metrics["silhouette"] > best_score:
            best_score = metrics["silhouette"]

            best_model = model

            best_labels = labels

    return (
        pd.DataFrame(results),
        best_model,
        best_labels,
    )


def benchmark_agglomerative(
    X: np.ndarray,
) -> tuple:

    results = []

    best_score = -1

    best_model = None

    best_labels = None

    for k in range(2, 15):
        model = AgglomerativeClustering(n_clusters=k)

        labels = model.fit_predict(X)

        metrics = evaluate_clustering(
            X,
            labels,
        )

        metrics["model"] = "Agglomerative"

        metrics["k"] = k

        results.append(metrics)

        if metrics["silhouette"] > best_score:
            best_score = metrics["silhouette"]

            best_model = model

            best_labels = labels

    return (
        pd.DataFrame(results),
        best_model,
        best_labels,
    )


def benchmark_dbscan(
    X: np.ndarray,
) -> tuple:

    results = []

    best_score = -1

    best_labels = None

    best_eps = None

    for eps in [
        0.003,
        0.005,
        0.007,
        0.01,
        0.015,
    ]:
        model = DBSCAN(
            eps=eps,
            min_samples=5,
        )

        labels = model.fit_predict(X)

        metrics = evaluate_clustering(
            X,
            labels,
        )

        metrics["model"] = "DBSCAN"

        metrics["eps"] = eps

        results.append(metrics)

        if metrics["silhouette"] > best_score:
            best_score = metrics["silhouette"]

            best_labels = labels

            best_eps = eps

    return (
        pd.DataFrame(results),
        best_eps,
        best_labels,
    )
