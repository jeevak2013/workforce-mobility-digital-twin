"""
Enterprise Demand Forecasting Pipeline

File:
    src/pipeline/forecasting.py

Purpose:
- Train enterprise overtime demand forecasting model
- Evaluate predictive performance
- Perform Time-Series Cross Validation
- Prevent overfitting using Early Stopping
- Export explainable forecasting artifacts

Outputs:
--------
- evaluation.json
- predicted_demand.csv
- xgboost_forecaster.json
- feature_importance.csv
- feature_importance.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    TimeSeriesSplit,
)

from xgboost import XGBRegressor


# =========================================================
# 📁 PROJECT PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "synthetic_data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# 📝 LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(message)s"),
)

logger = logging.getLogger(__name__)

# =========================================================
# 🎯 TARGET COLUMN
# =========================================================

TARGET_COLUMN = "Y"

# =========================================================
# 🧠 FEATURE SELECTION
# =========================================================

FEATURE_COLUMNS = [
    # Calendar
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "is_month_end",
    # Weekend
    "is_weekend",
    "expected_staffing_pressure",
    # Lag
    "Y_lag_1",
    "Y_lag_3",
    "Y_lag_7",
    # Rolling
    "Y_3d_mean",
    "Y_7d_mean",
    "Y_7d_max",
    "Y_7d_min",
    "Y_7d_std",
    # Cyclical
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

# =========================================================
# 📥 DATA LOADING
# =========================================================


def load_datasets():

    logger.info("Loading forecasting datasets...")

    train_df = pd.read_csv(DATA_DIR / "train_features.csv")

    test_df = pd.read_csv(DATA_DIR / "test_features.csv")

    logger.info(f"Train rows: {len(train_df)}")

    logger.info(f"Test rows: {len(test_df)}")

    return train_df, test_df


# =========================================================
# ✂ FEATURE/TARGET SPLIT
# =========================================================


def prepare_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    X_train = train_df[FEATURE_COLUMNS]

    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]

    y_test = test_df[TARGET_COLUMN]

    logger.info(f"Training feature shape: {X_train.shape}")

    logger.info(f"Testing feature shape: {X_test.shape}")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


# =========================================================
# 🔁 TIME-SERIES CROSS VALIDATION
# =========================================================


def run_time_series_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    logger.info("Running TimeSeriesSplit cross-validation...")

    tscv = TimeSeriesSplit(n_splits=5)

    r2_scores = []

    rmse_scores = []

    fold = 1

    for train_idx, val_idx in tscv.split(X_train):
        logger.info(f"Processing CV Fold {fold}")

        X_fold_train = X_train.iloc[train_idx]

        y_fold_train = y_train.iloc[train_idx]

        X_fold_val = X_train.iloc[val_idx]

        y_fold_val = y_train.iloc[val_idx]

        model = XGBRegressor(
            random_state=42,
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            objective="reg:squarederror",
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=1.0,
            early_stopping_rounds=20,
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

        predictions = model.predict(X_fold_val)

        fold_r2 = r2_score(
            y_fold_val,
            predictions,
        )

        fold_rmse = np.sqrt(
            mean_squared_error(
                y_fold_val,
                predictions,
            )
        )

        r2_scores.append(fold_r2)

        rmse_scores.append(fold_rmse)

        logger.info(f"Fold {fold} | R²={fold_r2:.4f} | RMSE={fold_rmse:.4f}")

        fold += 1

    cv_results = {
        "cv_mean_r2": float(np.mean(r2_scores)),
        "cv_std_r2": float(np.std(r2_scores)),
        "cv_mean_rmse": float(np.mean(rmse_scores)),
    }

    logger.info(f"CV Mean R²: {cv_results['cv_mean_r2']:.4f}")

    logger.info(f"CV Mean RMSE: {cv_results['cv_mean_rmse']:.4f}")

    return cv_results


# =========================================================
# 🤖 MODEL TRAINING
# =========================================================


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):

    logger.info("Training final forecasting model...")

    model = XGBRegressor(
        random_state=42,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        reg_lambda=1.0,
        early_stopping_rounds=20,
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

    logger.info(f"Best iteration: {model.best_iteration}")

    return model


# =========================================================
# 📈 MODEL EVALUATION
# =========================================================


def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):

    logger.info("Evaluating final model...")

    predictions = model.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    logger.info(f"Final R² Score: {r2:.4f}")

    logger.info(f"Final RMSE: {rmse:.4f}")

    return predictions, r2, rmse


# =========================================================
# 📊 FEATURE IMPORTANCE VISUALIZATION
# =========================================================


def export_feature_importance(
    model: XGBRegressor,
):

    logger.info("Exporting feature importance...")

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    importance_df.to_csv(
        DATA_DIR / "feature_importance.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.xlabel("Importance Score")

    plt.ylabel("Features")

    plt.title("XGBoost Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(DATA_DIR / "feature_importance.png")

    plt.close()

    logger.info("Feature importance artifacts exported.")


# =========================================================
# 📤 EXPORT ENGINE
# =========================================================


def export_outputs(
    model: XGBRegressor,
    test_df: pd.DataFrame,
    predictions,
    r2: float,
    rmse: float,
    cv_results: dict,
):

    logger.info("Exporting forecasting artifacts...")

    evaluation = {
        "r2_score": round(float(r2), 6),
        "rmse": round(float(rmse), 6),
        "cv_mean_r2": round(
            cv_results["cv_mean_r2"],
            6,
        ),
        "cv_std_r2": round(
            cv_results["cv_std_r2"],
            6,
        ),
        "cv_mean_rmse": round(
            cv_results["cv_mean_rmse"],
            6,
        ),
    }

    with open(
        DATA_DIR / "evaluation.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evaluation,
            f,
            indent=4,
        )

    prediction_df = pd.DataFrame(
        {
            "date": test_df["date"],
            "actual": test_df[TARGET_COLUMN],
            "predicted": predictions,
        }
    )

    prediction_df.to_csv(
        DATA_DIR / "predicted_demand.csv",
        index=False,
    )

    model.save_model(str(DATA_DIR / "xgboost_forecaster.json"))

    logger.info("Forecasting artifacts exported successfully.")


# =========================================================
# 🚀 MAIN PIPELINE
# =========================================================


def main() -> None:

    logger.info("=" * 60)

    logger.info("Starting enterprise forecasting pipeline...")

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    train_df, test_df = load_datasets()

    # -----------------------------------------------------
    # Prepare matrices
    # -----------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_datasets(
        train_df,
        test_df,
    )

    # -----------------------------------------------------
    # Cross Validation
    # -----------------------------------------------------

    cv_results = run_time_series_cv(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Final Model
    # -----------------------------------------------------

    model = train_final_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Final Evaluation
    # -----------------------------------------------------

    (
        predictions,
        r2,
        rmse,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Feature Importance
    # -----------------------------------------------------

    export_feature_importance(model)

    # -----------------------------------------------------
    # Export Outputs
    # -----------------------------------------------------

    export_outputs(
        model=model,
        test_df=test_df,
        predictions=predictions,
        r2=r2,
        rmse=rmse,
        cv_results=cv_results,
    )

    logger.info("Forecasting pipeline completed successfully.")

    print("\n🚀 Enterprise forecasting pipeline completed successfully.\n")


# =========================================================
# 🚀 ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()
