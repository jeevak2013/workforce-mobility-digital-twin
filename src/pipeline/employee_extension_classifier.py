"""
Enterprise Employee Extension Classification Pipeline

File:
    src/pipeline/employee_extension_classifier.py

Purpose:
--------
Predict WHICH employees are likely to extend overtime.

Business Goal:
--------------
Support:
- dynamic cab allocation
- route optimization
- safety-aware dispatch planning
- workforce mobility intelligence

Outputs:
--------
- employee_extension_predictions.csv
- classifier_metrics.json
- employee_extension_model.json
- classifier_feature_importance.csv
- classifier_feature_importance.png
- confusion_matrix.png
- roc_curve.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)

from sklearn.model_selection import (
    TimeSeriesSplit,
)

from xgboost import XGBClassifier


# =========================================================
# 📁 PATH CONFIGURATION
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = BASE_DIR / "synthetic_data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# 📝 LOGGING
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
# 🎯 TARGET
# =========================================================

TARGET_COLUMN = "extended"

# =========================================================
# 📥 LOAD DATASETS
# =========================================================

def load_datasets():

    logger.info(
        "Loading employee extension datasets..."
    )

    train_df = pd.read_csv(
        DATA_DIR
        / "employee_extension_train.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR
        / "employee_extension_test.csv"
    )

    logger.info(
        f"Train rows: {len(train_df)}"
    )

    logger.info(
        f"Test rows: {len(test_df)}"
    )

    return train_df, test_df


# =========================================================
# 🧠 FEATURE PREPARATION
# =========================================================

def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    logger.info(
        "Preparing classification features..."
    )

    feature_columns = [

        # ---------------------------------------------
        # Temporal
        # ---------------------------------------------

        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "is_month_end",
        "is_weekend",

        # ---------------------------------------------
        # Cyclical
        # ---------------------------------------------

        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",

        # ---------------------------------------------
        # Employee Behavior
        # ---------------------------------------------

        "employee_last_3d_extension_rate",
        "employee_last_7d_extension_rate",
        "employee_last_7d_extension_count",

        # ---------------------------------------------
        # Operational
        # ---------------------------------------------

        "previous_day_extension_load",
        "expected_staffing_pressure",

        # ---------------------------------------------
        # Employee Metadata
        # ---------------------------------------------

        "tenure_days",
        "safety_priority_score",
        "requires_security_escort",

        # ---------------------------------------------
        # Categoricals
        # ---------------------------------------------

        "gender",
        "hub",
        "shift",
    ]

    train_df = train_df[
        feature_columns
        + [TARGET_COLUMN]
    ].copy()

    test_df = test_df[
        feature_columns
        + [TARGET_COLUMN]
    ].copy()

    # -------------------------------------------------
    # One-hot encoding
    # -------------------------------------------------

    combined = pd.concat(
        [train_df, test_df],
        axis=0,
    )

    combined = pd.get_dummies(
        combined,
        columns=[
            "gender",
            "hub",
            "shift",
        ],
        drop_first=True,
    )

    train_encoded = combined.iloc[
        : len(train_df)
    ]

    test_encoded = combined.iloc[
        len(train_df) :
    ]

    X_train = train_encoded.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_encoded[
        TARGET_COLUMN
    ]

    X_test = test_encoded.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_encoded[
        TARGET_COLUMN
    ]

    logger.info(
        f"Training feature shape: "
        f"{X_train.shape}"
    )

    logger.info(
        f"Testing feature shape: "
        f"{X_test.shape}"
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )

def compute_scale_pos_weight(
    y_train: pd.Series,
) -> float:
    """
    Computes imbalance compensation weight.
    """

    negative_count = (
        (y_train == 0).sum()
    )

    positive_count = (
        (y_train == 1).sum()
    )

    if positive_count == 0:
        return 1.0

    return (
        negative_count
        / positive_count
    )

# =========================================================
# 🔁 TIME SERIES CV
# =========================================================

def run_time_series_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    logger.info(
        "Running TimeSeriesSplit CV..."
    )

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    auc_scores = []

    recall_scores = []

    fold = 1

    for train_idx, val_idx in tscv.split(X_train):

        logger.info(
            f"Processing fold {fold}"
        )

        X_fold_train = (
            X_train.iloc[train_idx]
        )

        y_fold_train = (
            y_train.iloc[train_idx]
        )

        X_fold_val = (
            X_train.iloc[val_idx]
        )

        y_fold_val = (
            y_train.iloc[val_idx]
        )

        scale_pos_weight = compute_scale_pos_weight(
            y_fold_train
        )

        model = XGBClassifier(

            random_state=42,

            n_estimators=500,

            max_depth=5,

            learning_rate=0.05,

            subsample=0.90,

            colsample_bytree=0.90,

            reg_alpha=0.10,

            reg_lambda=1.0,

            eval_metric="logloss",

            early_stopping_rounds=20,

            scale_pos_weight=scale_pos_weight,
        )

        model.fit(

            X_fold_train,
            y_fold_train,

            eval_set=[
                (
                    X_fold_val,
                    y_fold_val,
                )
            ],

            verbose=False,
        )

        probabilities = (
            model.predict_proba(
                X_fold_val
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        fold_auc = roc_auc_score(
            y_fold_val,
            probabilities,
        )

        fold_recall = recall_score(
            y_fold_val,
            predictions,
        )

        auc_scores.append(fold_auc)

        recall_scores.append(
            fold_recall
        )

        logger.info(
            f"Fold {fold} | "
            f"AUC={fold_auc:.4f} | "
            f"Recall={fold_recall:.4f}"
        )

        fold += 1

    cv_results = {

        "cv_mean_auc":
            float(np.mean(auc_scores)),

        "cv_std_auc":
            float(np.std(auc_scores)),

        "cv_mean_recall":
            float(
                np.mean(recall_scores)
            ),
    }

    return cv_results


# =========================================================
# 🤖 TRAIN FINAL MODEL
# =========================================================

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):

    logger.info(
        "Training final classifier..."
    )

    scale_pos_weight = compute_scale_pos_weight(
        y_train
    )

    model = XGBClassifier(

        random_state=42,

        n_estimators=500,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.90,

        colsample_bytree=0.90,

        reg_alpha=0.10,

        reg_lambda=1.0,

        eval_metric="logloss",

        early_stopping_rounds=20,

        scale_pos_weight=scale_pos_weight,
    )

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (
                X_test,
                y_test,
            )
        ],

        verbose=False,
    )

    logger.info(
        f"Best iteration: "
        f"{model.best_iteration}"
    )

    return model


# =========================================================
# 📈 EVALUATION
# =========================================================

def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):

    logger.info(
        "Evaluating classifier..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions,
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
            ),

        "f1_score":
            f1_score(
                y_test,
                predictions,
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities,
            ),
    }

    for k, v in metrics.items():

        logger.info(
            f"{k}: {v:.4f}"
        )

    return (
        probabilities,
        predictions,
        metrics,
    )


# =========================================================
# 📊 VISUALIZATION EXPORTS
# =========================================================

def export_visualizations(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities,
    predictions,
):

    logger.info(
        "Exporting visualizations..."
    )

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    importance_df = pd.DataFrame({

        "feature":
            X_test.columns,

        "importance":
            model.feature_importances_,
    })

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False,
        )
    )

    importance_df.to_csv(
        DATA_DIR
        / "classifier_feature_importance.csv",
        index=False,
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Employee Extension Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        DATA_DIR
        / "classifier_feature_importance.png"
    )

    plt.close()

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(
        DATA_DIR
        / "confusion_matrix.png"
    )

    plt.close()

    # -------------------------------------------------
    # ROC Curve
    # -------------------------------------------------

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
    )

    plt.tight_layout()

    plt.savefig(
        DATA_DIR
        / "roc_curve.png"
    )

    plt.close()


# =========================================================
# 📤 EXPORT OUTPUTS
# =========================================================

def export_outputs(
    model: XGBClassifier,
    probabilities,
    predictions,
    metrics: dict,
    cv_results: dict,
):

    logger.info(
        "Exporting classifier artifacts..."
    )

    prediction_df = pd.DataFrame({

        "prediction_probability":
            probabilities,

        "predicted_extension":
            predictions,
    })

    prediction_df.to_csv(
        DATA_DIR
        / "employee_extension_predictions.csv",
        index=False,
    )

    full_metrics = {
        **metrics,
        **cv_results,
    }

    with open(
        DATA_DIR
        / "classifier_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            full_metrics,
            f,
            indent=4,
        )

    model.save_model(
        str(
            DATA_DIR
            / "employee_extension_model.json"
        )
    )

    logger.info(
        "Classifier artifacts exported."
    )


# =========================================================
# 🚀 MAIN
# =========================================================

def main():

    logger.info("=" * 60)

    logger.info(
        "Starting employee extension classifier..."
    )

    # -------------------------------------------------
    # Load
    # -------------------------------------------------

    train_df, test_df = (
        load_datasets()
    )

    # -------------------------------------------------
    # Features
    # -------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_features(
        train_df,
        test_df,
    )

    # -------------------------------------------------
    # CV
    # -------------------------------------------------

    cv_results = run_time_series_cv(
        X_train,
        y_train,
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    model = train_final_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------

    (
        probabilities,
        predictions,
        metrics,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -------------------------------------------------
    # Visuals
    # -------------------------------------------------

    export_visualizations(
        model,
        X_test,
        y_test,
        probabilities,
        predictions,
    )

    # -------------------------------------------------
    # Outputs
    # -------------------------------------------------

    export_outputs(
        model,
        probabilities,
        predictions,
        metrics,
        cv_results,
    )

    logger.info(
        "Employee extension classifier completed."
    )

    print(
        "\n🚀 Employee extension "
        "classification pipeline completed.\n"
    )


# =========================================================
# 🚀 ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()