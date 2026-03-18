"""Hand Gesture Recognition - CLI Entry Point.

Orchestrates the gesture recognition pipeline:
collect data, train models, and run predictions.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import GestureRecognitionPipeline


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition with ML",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        default="train",
        help="Pipeline mode (default: train)",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "svm", "gradient_boosting", "all"],
        default="all",
        help="Model type to train (default: all)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to gesture dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the gesture recognition pipeline."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Hand Gesture Recognition Pipeline")
    logger.info(f"Mode: {args.mode} | Model: {args.model_type}")

    pipeline = GestureRecognitionPipeline(
        model_type=args.model_type,
        output_dir=args.output_dir,
    )

    if args.mode == "train":
        results = pipeline.run(data_path=args.data_path)
        logger.info("Training complete")
        for model_name, metrics in results.items():
            logger.info(
                f"{model_name}: Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1_macro']:.4f}"
            )
    elif args.mode == "evaluate":
        pipeline.evaluate(data_path=args.data_path)
    else:
        logger.error(f"Mode '{args.mode}' not fully implemented yet")
        sys.exit(1)

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
