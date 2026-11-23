"""Chronological ensemble training and evaluation for LankaWeather AI."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight
from ensemble_estimators import ExpandingWindowStackingClassifier, TrainingWeightedGradientBoosting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "weather_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TARGET_PATH = MODEL_DIR / "target_column.txt"
DATE_NAME_HINTS = ("date", "time", "timestamp")
IDENTIFIER_NAME_HINTS = ("id", "identifier", "index")
FUTURE_NAME_HINTS = ("next_", "future", "tomorrow", "lead", "target")


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, str, pd.Series]:
    """Discover the single binary target and predictors from the supplied table."""
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("Dataset must contain rows, a target, and at least one predictor.")
    dates = [column for column in frame if any(hint in column.lower() for hint in DATE_NAME_HINTS)]
    if len(dates) != 1:
        raise ValueError(f"Expected exactly one date/time column for chronological ordering; found {dates}.")
    date_column = dates[0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if frame[date_column].isna().any():
        raise ValueError(f"Date column {date_column!r} contains invalid values.")
    frame = frame.sort_values(date_column, kind="stable").reset_index(drop=True)

    binary_candidates: list[str] = []
    for column in frame.columns:
        if column == date_column:
            continue
        values = frame[column].dropna().unique()
        if len(values) == 2 and set(values).issubset({0, 1, False, True}):
            binary_candidates.append(column)
    if len(binary_candidates) != 1:
        raise ValueError(
            "Could not identify exactly one binary 0/1 target from the data; "
            f"candidates were {binary_candidates}. Set up a single binary target first."
        )
    target_name = binary_candidates[0]
    y = frame[target_name].astype("int8")
    excluded = {date_column, target_name}
    excluded.update(
        column
        for column in frame.columns
        if any(hint in column.lower() for hint in IDENTIFIER_NAME_HINTS + FUTURE_NAME_HINTS)
    )
    predictors = [column for column in frame.columns if column not in excluded]
    if not predictors:
        raise ValueError("No predictors remain after excluding dates, identifiers, target, and future fields.")
    X = frame[predictors].copy()
    if X.isna().any().any():
        raise ValueError("Predictors contain missing values; resolve them before model training.")
    return X, y, target_name, frame[date_column]


def _encode_predictors(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical predictors using training categories only."""
    categories = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    train_encoded = pd.get_dummies(X_train, columns=categories, dtype="int8")
    test_encoded = pd.get_dummies(X_test, columns=categories, dtype="int8")
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded.astype(float), test_encoded.astype(float)


def _base_estimators() -> tuple[RandomForestClassifier, GradientBoostingClassifier]:
    """Return deterministic base estimators with fixed documented parameters."""
    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    boosting = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    return forest, boosting


@dataclass
class ModelRun:
    """Fitted model and measured predictions from the held-out chronological test set."""

    name: str
    model: Any
    predictions: np.ndarray
    probabilities: np.ndarray
    training_seconds: float
    prediction_seconds: float


def _evaluate(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
              train_seconds: float, prediction_seconds: float) -> dict[str, Any]:
    """Calculate test metrics and retain the full report and confusion matrix."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "confusion_matrix": json.dumps(matrix.tolist()),
        "training_time_seconds": train_seconds,
        "prediction_time_seconds": prediction_seconds,
        "classification_report": json.dumps(
            classification_report(y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0)
        ),
    }


def _save_figures(runs: list[ModelRun], y_test: pd.Series, X_test: pd.DataFrame,
                  forest: RandomForestClassifier, boosting: GradientBoostingClassifier) -> None:
    """Write confusion matrix, ROC, PR, and tree feature importance figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for run in runs:
        matrix = confusion_matrix(y_test, run.predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(title=f"{run.name} confusion matrix", xlabel="Predicted", ylabel="Actual")
        ax.set_xticks([0, 1], labels=["No rain", "Rain"])
        ax.set_yticks([0, 1], labels=["No rain", "Rain"])
        for (row, col), count in np.ndenumerate(matrix):
            ax.text(col, row, str(count), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{run.name}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        fpr, tpr, _ = roc_curve(y_test, run.probabilities)
        ax.plot(fpr, tpr, label=f"{run.name} (AUC={roc_auc_score(y_test, run.probabilities):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        precision, recall, _ = precision_recall_curve(y_test, run.probabilities)
        ax.plot(recall, precision, label=run.name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=160)
    plt.close(fig)

    importances = pd.DataFrame({"feature": X_test.columns,
                                "Random Forest": forest.feature_importances_,
                                "Gradient Boosting": boosting.feature_importances_})
    for model_name in ("Random Forest", "Gradient Boosting"):
        top = importances.nlargest(15, model_name).sort_values(model_name)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top[model_name])
        ax.set(title=f"{model_name} feature importance", xlabel="Importance")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(fig)


def run_pipeline() -> pd.DataFrame:
    """Train baseline and ensembles, evaluate chronologically, and save artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X, y, target_name, dates = load_dataset()
    split_at = int(len(X) * 0.8)
    if split_at < 2 or split_at >= len(X):
        raise ValueError("Dataset is too small for a chronological 80/20 split.")
    X_train_raw, X_test_raw = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    X_train, X_test = _encode_predictors(X_train_raw, X_test_raw)
    print(f"Dataset shape: {len(X):,} rows; target={target_name!r}; predictors={list(X.columns)}")
    print(f"Chronological split: train={len(X_train):,} ({dates.iloc[0].date()}–{dates.iloc[split_at-1].date()}), "
          f"test={len(X_test):,} ({dates.iloc[split_at].date()}–{dates.iloc[-1].date()})")
    print(f"Training class counts: {y_train.value_counts().sort_index().to_dict()}; "
          f"test class counts: {y_test.value_counts().sort_index().to_dict()}")

    forest, boosting = _base_estimators()
    stack = ExpandingWindowStackingClassifier(forest, boosting, n_splits=5, random_state=42)
    hybrid = VotingClassifier(
        estimators=[("rf", clone(forest)), ("gb", TrainingWeightedGradientBoosting(boosting))],
        voting="soft",
    )
    specs: list[tuple[str, Any, dict[str, Any]]] = [
        ("Dummy Baseline", DummyClassifier(strategy="most_frequent"), {}),
        ("Random Forest", clone(forest), {}),
        ("Gradient Boosting", clone(boosting), {"sample_weight": compute_sample_weight("balanced", y_train)}),
        ("Stacking", stack, {}),
        ("Hybrid Soft Voting", hybrid, {}),
    ]
    runs: list[ModelRun] = []
    metric_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, model, fit_params in specs:
        start = time.perf_counter()
        model.fit(X_train, y_train, **fit_params)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        predict_seconds = time.perf_counter() - start
        runs.append(ModelRun(name, model, predictions, probabilities, train_seconds, predict_seconds))
        metric_rows.append(_evaluate(name, model, X_test, y_test, train_seconds, predict_seconds))
        fitted[name] = model
        print(f"Trained and evaluated {name}.")

    # Persist four requested deployable ensembles; retain metadata for predictor order.
    artifacts = {
        "random_forest.pkl": fitted["Random Forest"],
        "gradient_boosting.pkl": fitted["Gradient Boosting"],
        "stacking_model.pkl": fitted["Stacking"],
        "hybrid_model.pkl": fitted["Hybrid Soft Voting"],
    }
    for filename, model in artifacts.items():
        joblib.dump(model, MODEL_DIR / filename)
    metadata = {"target": target_name, "predictors": list(X.columns),
                "encoded_predictors": list(X_train.columns), "date_order_column": "date",
                "train_rows": len(X_train), "test_rows": len(X_test), "random_state": 42}
    (MODEL_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    TARGET_PATH.write_text(target_name + "\n", encoding="utf-8")

    results = pd.DataFrame(metric_rows)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    _save_figures(runs, y_test, X_test, fitted["Random Forest"], fitted["Gradient Boosting"])
    from evaluation import write_analysis

    write_analysis(results, y_train, y_test, dates.iloc[:split_at], dates.iloc[split_at:])
    print("\nHeld-out chronological test metrics:")
    print(results[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "balanced_accuracy"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved metrics, figures, and analysis under {RESULTS_DIR}")
    return results


def main() -> None:
    """Run the end-to-end model training and evaluation workflow."""
    try:
        run_pipeline()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"Training pipeline stopped: {exc}") from exc


if __name__ == "__main__":
    main()
