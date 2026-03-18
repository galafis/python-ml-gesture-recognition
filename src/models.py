"""Gesture Classification Models Module.

Implements multiple ML classifiers for hand gesture recognition:
- Random Forest
- SVM (Support Vector Machine)
- Gradient Boosting
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class GestureModelTrainer:
    """Trains and evaluates gesture classification models."""

    MODEL_REGISTRY = {
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "gradient_boosting": GradientBoostingClassifier,
    }

    MODEL_PARAMS = {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
        "svm": {
            "kernel": "rbf",
            "C": 10.0,
            "gamma": "scale",
            "random_state": 42,
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
    }

    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model_type: str = "all",
    ) -> Dict[str, Dict[str, Any]]:
        """Train models and return evaluation metrics."""
        X, y = self._prepare_data(df)
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

        models_to_train = (
            list(self.MODEL_REGISTRY.keys())
            if model_type == "all"
            else [model_type]
        )

        results = {}
        for name in models_to_train:
            logger.info(f"Training: {name}")
            metrics = self._cross_validate(X, y, name)
            results[name] = metrics
            logger.info(
                f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1_macro']:.4f}"
            )

        return results

    def _prepare_data(
        self, df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and encoded labels."""
        y = self.label_encoder.fit_transform(df["gesture"])
        feature_cols = [
            col for col in df.columns
            if col != "gesture"
            and df[col].dtype in ["float64", "int64", "float32"]
        ]
        X = df[feature_cols].values
        return X, y

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """Stratified K-Fold cross-validation."""
        model_class = self.MODEL_REGISTRY[model_name]
        params = self.MODEL_PARAMS[model_name]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics: List[Dict[str, float]] = []

        for train_idx, val_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            model = model_class(**params)
            model.fit(X_train, y[train_idx])
            preds = model.predict(X_val)

            fold_metrics.append({
                "accuracy": accuracy_score(y[val_idx], preds),
                "f1_macro": f1_score(y[val_idx], preds, average="macro"),
            })

        avg = {
            key: np.mean([m[key] for m in fold_metrics])
            for key in fold_metrics[0]
        }
        avg["model_name"] = model_name
        avg["n_features"] = X.shape[1]
        avg["n_classes"] = len(np.unique(y))

        return avg
