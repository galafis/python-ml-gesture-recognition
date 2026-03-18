"""Gesture Recognition Pipeline - End-to-end orchestration."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd

from src.data_collector import GestureDataCollector
from src.feature_engineering import GestureFeatureEngineer
from src.models import GestureModelTrainer

logger = logging.getLogger(__name__)


class GestureRecognitionPipeline:
    """End-to-end gesture recognition pipeline."""

    def __init__(
        self,
        model_type: str = "all",
        output_dir: str = "output",
    ) -> None:
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collector = GestureDataCollector()
        self.engineer = GestureFeatureEngineer()
        self.trainer = GestureModelTrainer()

    def run(
        self, data_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute the full recognition pipeline."""
        logger.info("Starting gesture recognition pipeline")

        # Step 1: Data Collection
        logger.info("Step 1/3: Data Collection")
        if data_path:
            df = self.collector.load_dataset(data_path)
        else:
            df = self.collector.generate_synthetic_data()
        logger.info(f"Loaded {len(df)} samples")

        # Step 2: Feature Engineering
        logger.info("Step 2/3: Feature Engineering")
        df = self.engineer.create_features(df)
        logger.info(f"Generated {len(df.columns)} features")

        # Step 3: Model Training
        logger.info("Step 3/3: Model Training")
        results = self.trainer.train_and_evaluate(
            df, model_type=self.model_type,
        )

        # Save results
        self._save_results(df, results)
        return results

    def evaluate(
        self, data_path: Optional[str] = None,
    ) -> None:
        """Evaluate pipeline on given dataset."""
        results = self.run(data_path=data_path)
        for name, metrics in results.items():
            logger.info(f"\n{name}: {metrics}")

    def _save_results(
        self,
        df: pd.DataFrame,
        results: Dict[str, Dict[str, Any]],
    ) -> None:
        """Save pipeline outputs."""
        metrics_df = pd.DataFrame(results).T
        metrics_path = self.output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
