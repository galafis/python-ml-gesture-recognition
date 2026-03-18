"""Gesture Feature Engineering Module.

Extracts geometric, angular, and statistical features
from hand landmark coordinates for gesture classification.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 21


class GestureFeatureEngineer:
    """Generates discriminative features from hand landmarks."""

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_BASES = [2, 5, 9, 13, 17]
    WRIST = 0

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature engineering pipeline."""
        logger.info("Starting gesture feature engineering")
        df = df.copy()

        df = self._add_distance_features(df)
        df = self._add_angle_features(df)
        df = self._add_normalized_features(df)
        df = self._add_statistical_features(df)

        logger.info(f"Total features: {len(df.columns)}")
        return df

    def _get_landmark(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """Extract x,y,z coordinates for a landmark."""
        return np.column_stack([
            df[f"landmark_x_{idx}"].values,
            df[f"landmark_y_{idx}"].values,
            df[f"landmark_z_{idx}"].values,
        ])

    def _add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distances between key landmarks."""
        wrist = self._get_landmark(df, self.WRIST)

        for tip_idx in self.FINGER_TIPS:
            tip = self._get_landmark(df, tip_idx)
            dist = np.linalg.norm(tip - wrist, axis=1)
            df[f"dist_wrist_to_{tip_idx}"] = dist

        for i, tip_a in enumerate(self.FINGER_TIPS):
            for tip_b in self.FINGER_TIPS[i + 1:]:
                a = self._get_landmark(df, tip_a)
                b = self._get_landmark(df, tip_b)
                df[f"dist_{tip_a}_to_{tip_b}"] = np.linalg.norm(a - b, axis=1)

        return df

    def _add_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate angles between finger joints."""
        for tip, base in zip(self.FINGER_TIPS, self.FINGER_BASES):
            t = self._get_landmark(df, tip)
            b = self._get_landmark(df, base)
            w = self._get_landmark(df, self.WRIST)

            v1 = b - w
            v2 = t - b

            cos_angle = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
            )
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            df[f"angle_finger_{tip}"] = np.degrees(angle)

        return df

    def _add_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize landmarks relative to wrist and hand size."""
        wrist = self._get_landmark(df, self.WRIST)
        middle_tip = self._get_landmark(df, 12)
        hand_size = np.linalg.norm(middle_tip - wrist, axis=1) + 1e-8

        for i in range(NUM_LANDMARKS):
            lm = self._get_landmark(df, i)
            relative = lm - wrist
            for j, dim in enumerate(["x", "y", "z"]):
                df[f"norm_{dim}_{i}"] = relative[:, j] / hand_size

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add aggregate statistics across all landmarks."""
        for dim in ["x", "y", "z"]:
            cols = [f"landmark_{dim}_{i}" for i in range(NUM_LANDMARKS)]
            values = df[cols].values
            df[f"mean_{dim}"] = values.mean(axis=1)
            df[f"std_{dim}"] = values.std(axis=1)
            df[f"range_{dim}"] = values.max(axis=1) - values.min(axis=1)

        return df
