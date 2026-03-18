"""Gesture Data Collection Module.

Handles loading gesture landmark data from CSV files
and optional real-time collection via MediaPipe Hands.
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

GESTURE_CLASSES = [
    "open_palm", "fist", "thumbs_up", "peace",
    "pointing", "ok_sign",
]

NUM_LANDMARKS = 21
DIMENSIONS = ["x", "y", "z"]


class GestureDataCollector:
    """Manages gesture landmark data acquisition."""

    def load_dataset(
        self, filepath: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load gesture dataset from CSV file."""
        if filepath is None:
            filepath = DATA_DIR / "sample_gestures.csv"

        logger.info(f"Loading gesture data from {filepath}")
        df = pd.read_csv(filepath)
        self._validate(df)
        logger.info(
            f"Loaded {len(df)} samples across "
            f"{df['gesture'].nunique()} gesture classes"
        )
        return df

    def generate_synthetic_data(
        self,
        n_samples_per_class: int = 200,
        noise_level: float = 0.05,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic gesture landmark data."""
        np.random.seed(seed)
        logger.info(f"Generating synthetic data: {n_samples_per_class} per class")

        gesture_prototypes = self._define_prototypes()
        records: List[dict] = []

        for gesture, prototype in gesture_prototypes.items():
            for _ in range(n_samples_per_class):
                noise = np.random.normal(0, noise_level, prototype.shape)
                sample = prototype + noise
                sample = np.clip(sample, 0.0, 1.0)

                row = {"gesture": gesture}
                for i in range(NUM_LANDMARKS):
                    for j, dim in enumerate(DIMENSIONS):
                        row[f"landmark_{dim}_{i}"] = sample[i, j]
                records.append(row)

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} synthetic samples")
        return df

    def _define_prototypes(self) -> dict:
        """Define prototype landmark positions for each gesture."""
        np.random.seed(42)
        prototypes = {}
        for gesture in GESTURE_CLASSES:
            prototypes[gesture] = np.random.uniform(
                0.2, 0.8, size=(NUM_LANDMARKS, len(DIMENSIONS)),
            )
        return prototypes

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate gesture dataset structure."""
        if "gesture" not in df.columns:
            raise ValueError("Missing 'gesture' column")
        if df.empty:
            raise ValueError("Dataset is empty")

        landmark_cols = [
            col for col in df.columns if col.startswith("landmark_")
        ]
        if len(landmark_cols) < NUM_LANDMARKS * len(DIMENSIONS):
            raise ValueError(
                f"Expected at least {NUM_LANDMARKS * len(DIMENSIONS)} "
                f"landmark columns, found {len(landmark_cols)}"
            )
