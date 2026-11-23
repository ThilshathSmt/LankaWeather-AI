"""Serializable custom estimators used by the LankaWeather ensemble pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


class ExpandingWindowStackingClassifier(ClassifierMixin, BaseEstimator):
    """Stack RF and GB probabilities from forward-only chronological OOF folds."""

    def __init__(self, forest: Any, boosting: Any, n_splits: int = 5, random_state: int = 42):
        self.forest = forest
        self.boosting = boosting
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExpandingWindowStackingClassifier":
        """Fit logistic meta model on expanding folds; refit bases on full training data."""
        n_samples = len(y)
        initial = n_samples // (self.n_splits + 1)
        if initial < 2:
            raise ValueError("Not enough training rows for expanding-window stacking.")
        boundaries = np.linspace(initial, n_samples, self.n_splits + 1, dtype=int)
        oof_features: list[np.ndarray] = []
        oof_targets: list[np.ndarray] = []
        for fold in range(self.n_splits):
            train_end, valid_end = int(boundaries[fold]), int(boundaries[fold + 1])
            if valid_end <= train_end:
                continue
            X_fold_train, X_fold_valid = X.iloc[:train_end], X.iloc[train_end:valid_end]
            y_fold_train, y_fold_valid = y.iloc[:train_end], y.iloc[train_end:valid_end]
            if y_fold_train.nunique() < 2:
                continue
            rf_fold, gb_fold = clone(self.forest), clone(self.boosting)
            rf_fold.fit(X_fold_train, y_fold_train)
            gb_fold.fit(X_fold_train, y_fold_train,
                        sample_weight=compute_sample_weight("balanced", y_fold_train))
            oof_features.append(np.column_stack([
                rf_fold.predict_proba(X_fold_valid)[:, 1],
                gb_fold.predict_proba(X_fold_valid)[:, 1],
            ]))
            oof_targets.append(y_fold_valid.to_numpy())
        if not oof_features:
            raise ValueError("No valid chronological OOF folds could be constructed.")
        self.meta_ = LogisticRegression(class_weight="balanced", random_state=self.random_state, max_iter=1000)
        self.meta_.fit(np.vstack(oof_features), np.concatenate(oof_targets))
        self.forest_ = clone(self.forest).fit(X, y)
        self.boosting_ = clone(self.boosting)
        self.boosting_.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.meta_.classes_
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return meta-learner probabilities from both fitted base classifiers."""
        meta_X = np.column_stack([
            self.forest_.predict_proba(X)[:, 1], self.boosting_.predict_proba(X)[:, 1]
        ])
        return self.meta_.predict_proba(meta_X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the most likely target class."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TrainingWeightedGradientBoosting(ClassifierMixin, BaseEstimator):
    """Gradient boosting wrapper with balanced weights calculated in fit only."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainingWeightedGradientBoosting":
        """Fit the wrapped gradient boosting model using only the supplied labels."""
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return wrapped model predictions."""
        return self.estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return wrapped model class probabilities."""
        return self.estimator_.predict_proba(X)
