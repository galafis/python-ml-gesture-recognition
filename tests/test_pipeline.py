"""Unit tests for gesture recognition pipeline."""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_gesture_df() -> pd.DataFrame:
    """Create sample gesture landmark data."""
    np.random.seed(42)
    n_samples = 120
    gestures = ["open_palm", "fist", "thumbs_up", "peace", "pointing", "ok_sign"]
    records = []
    for gesture in gestures:
        for _ in range(n_samples // len(gestures)):
            row = {"gesture": gesture}
            for i in range(21):
                for dim in ["x", "y", "z"]:
                    row[f"landmark_{dim}_{i}"] = np.random.uniform(0.2, 0.8)
            records.append(row)
    return pd.DataFrame(records)


class TestGestureDataCollector:
    """Tests for data collector module."""

    def test_generate_synthetic_data(self):
        from src.data_collector import GestureDataCollector
        collector = GestureDataCollector()
        df = collector.generate_synthetic_data(n_samples_per_class=50)
        assert not df.empty
        assert "gesture" in df.columns
        assert df["gesture"].nunique() == 6

    def test_validate_missing_gesture(self):
        from src.data_collector import GestureDataCollector
        collector = GestureDataCollector()
        bad_df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(ValueError, match="Missing 'gesture' column"):
            collector._validate(bad_df)


class TestGestureFeatureEngineer:
    """Tests for feature engineering module."""

    def test_create_features(self, sample_gesture_df):
        from src.feature_engineering import GestureFeatureEngineer
        engineer = GestureFeatureEngineer()
        result = engineer.create_features(sample_gesture_df)
        assert len(result.columns) > len(sample_gesture_df.columns)

    def test_distance_features(self, sample_gesture_df):
        from src.feature_engineering import GestureFeatureEngineer
        engineer = GestureFeatureEngineer()
        result = engineer._add_distance_features(sample_gesture_df.copy())
        assert "dist_wrist_to_8" in result.columns


class TestGestureModelTrainer:
    """Tests for model training module."""

    def test_train_random_forest(self, sample_gesture_df):
        from src.feature_engineering import GestureFeatureEngineer
        from src.models import GestureModelTrainer
        engineer = GestureFeatureEngineer()
        df = engineer.create_features(sample_gesture_df)
        trainer = GestureModelTrainer()
        results = trainer.train_and_evaluate(df, model_type="random_forest")
        assert "random_forest" in results
        assert "accuracy" in results["random_forest"]
        assert results["random_forest"]["accuracy"] >= 0
