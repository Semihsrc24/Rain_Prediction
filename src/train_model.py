"""Train, evaluate, and export the RainCast AI 5-Model Ensemble on 15,344 Turkish weather records.

Uses physics-informed meteorological features (src/features.py) derived from
data/turkey_weather_dataset.csv (6 years across Turkey's 7 climate regions).
Evaluates with BOTH:
  1. Strict temporal holdout (Train: 2020-2024 -> Test: unseen year 2025)
  2. 5-fold stratified cross-validation for threshold tuning & overall metrics
Then fits the final deployment model on 100% of available data and saves models/rain_model.pkl.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import FEATURE_NAMES, engineer_features  # noqa: E402


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load turkey_weather_dataset.csv and compute engineered features matching the UI."""
    df = pd.read_csv(csv_path)
    required = [
        "date",
        "Max_Temperature",
        "Min_Temperature",
        "WindSpeed",
        "CloudCover",
        "Humidity",
        "Pressure",
        "Uvi",
        "Precipitation",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing}")

    df = df.dropna(subset=required).copy()
    date_dt = pd.to_datetime(df["date"])
    df["Month"] = date_dt.dt.month
    years = date_dt.dt.year

    X = engineer_features(df)
    y = (df["Precipitation"] > 0.2).astype(int)
    return X, y, years


def build_ensemble(random_state: int = 42) -> VotingClassifier:
    """Build a diverse, well-calibrated 5-model soft-voting ensemble."""
    hgb = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=random_state,
    )
    gb = GradientBoostingClassifier(
        n_estimators=260,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(96, 48),
            alpha=0.05,
            max_iter=300,
            early_stopping=True,
            random_state=random_state,
        ),
    )

    return VotingClassifier(
        estimators=[
            ("hgb", hgb),
            ("gb", gb),
            ("rf", rf),
            ("et", et),
            ("mlp", mlp),
        ],
        voting="soft",
        weights=[3, 2, 2, 2, 2],
    )


def find_best_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    """Select the probability threshold maximizing accuracy (tie-breaking on F1)."""
    best_thresh = 0.50
    best_score = (-1.0, -1.0)
    for thresh in np.linspace(0.35, 0.65, 61):
        preds = (probs >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        score = (round(acc, 5), f1)
        if score > best_score:
            best_score = score
            best_thresh = float(round(thresh, 3))
    return best_thresh


def train_model(use_mlflow: bool = False) -> dict:
    csv_path = os.path.join(PROJECT_ROOT, "data", "turkey_weather_dataset.csv")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "rain_model.pkl")

    X, y, years = load_dataset(csv_path)
    if y.nunique() < 2:
        raise ValueError("Dataset contains only one target class.")

    # 1) Strict temporal holdout evaluation: Train 2020-2024 -> Test unseen 2025
    train_mask = years < 2025
    test_mask = years == 2025
    temp_model = build_ensemble(random_state=42)
    temp_model.fit(X[train_mask], y[train_mask])
    temp_probs = temp_model.predict_proba(X[test_mask])[:, 1]

    # 2) Honest 5-fold stratified out-of-fold evaluation across all 15,344 records
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(
        build_ensemble(random_state=42),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    best_threshold = find_best_threshold(y, oof_probs)
    oof_preds = (oof_probs >= best_threshold).astype(int)
    temp_preds = (temp_probs >= best_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, oof_preds)),
        "roc_auc": float(roc_auc_score(y, oof_probs)),
        "precision_rain": float(precision_score(y, oof_preds, zero_division=0)),
        "recall_rain": float(recall_score(y, oof_preds, zero_division=0)),
        "f1_rain": float(f1_score(y, oof_preds, zero_division=0)),
        "brier_score": float(brier_score_loss(y, oof_probs)),
        "temporal_2025_accuracy": float(accuracy_score(y[test_mask], temp_preds)),
        "temporal_2025_roc_auc": float(roc_auc_score(y[test_mask], temp_probs)),
        "temporal_2025_f1": float(f1_score(y[test_mask], temp_preds, zero_division=0)),
        "cv_folds": 5,
        "total_samples": int(len(X)),
    }

    # 3) Fit final ensemble on 100% of available historical records
    final_model = build_ensemble(random_state=42)
    final_model.fit(X, y)

    rf_imp = final_model.named_estimators_["rf"].feature_importances_
    et_imp = final_model.named_estimators_["et"].feature_importances_
    gb_imp = final_model.named_estimators_["gb"].feature_importances_
    avg_imp = (rf_imp + et_imp + gb_imp) / 3.0
    importances = {
        name: round(float(score), 4) for name, score in zip(FEATURE_NAMES, avg_imp)
    }

    bundle = {
        "model": final_model,
        "model_name": "Ensemble (HGB + GB + RF + ET + MLP)",
        "feature_names": FEATURE_NAMES,
        "threshold": best_threshold,
        "metrics": metrics,
        "feature_importances": importances,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": "turkey_weather_dataset.csv",
    }

    joblib.dump(bundle, model_path, compress=3)

    print(f"Model                  : {bundle['model_name']}")
    print(f"5-Fold CV Accuracy     : {metrics['accuracy'] * 100:.2f}%")
    print(f"5-Fold CV ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"5-Fold CV Rain F1      : {metrics['f1_rain']:.4f}")
    print(f"5-Fold CV Brier Score  : {metrics['brier_score']:.4f}")
    print(f"2025 Temporal Accuracy : {metrics['temporal_2025_accuracy'] * 100:.2f}%")
    print(f"2025 Temporal ROC-AUC  : {metrics['temporal_2025_roc_auc']:.4f}")
    print(f"Optimal Threshold      : {best_threshold:.3f}")
    print("\n5-Fold Out-of-Fold Classification Report:\n")
    print(classification_report(y, oof_preds, target_names=["No Rain", "Rain"], zero_division=0))
    print(f"Model bundle saved to: {model_path}")

    if use_mlflow:
        try:
            import mlflow

            mlruns_dir = os.path.join(PROJECT_ROOT, "mlruns")
            mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")
            mlflow.set_experiment("RainPrediction Turkey Ensemble")
            with mlflow.start_run():
                mlflow.log_param("model_name", bundle["model_name"])
                mlflow.log_param("threshold", best_threshold)
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                mlflow.log_artifact(model_path)
            print("Logged run to MLflow.")
        except ImportError:
            print("MLflow is not installed; skipping MLflow logging.")

    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RainCast AI Turkey Ensemble model.")
    parser.add_argument("--mlflow", action="store_true", help="Log experiment metrics to MLflow.")
    args = parser.parse_args()
    train_model(use_mlflow=args.mlflow)
