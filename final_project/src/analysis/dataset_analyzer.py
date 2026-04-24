"""Reusable dataset analysis utilities for notebook-driven reporting."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import SpectrogramDatasetBuilder
from src.utils.logger import get_logger


class SpectrogramDatasetAnalyzer:
    """Analyze dataset structure, label balance, and image statistics."""

    def __init__(self, cfg: Any, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the dataset analyzer.

        Args:
            cfg: Project configuration object.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        self.dataset_builder = SpectrogramDatasetBuilder(cfg=cfg, logger=self.logger)

    def load_train_metadata(self) -> pd.DataFrame:
        """Load train metadata as a DataFrame.

        Returns:
            A DataFrame with ``image_id``, ``image_path``, and ``target`` columns.
        """
        train_samples = self.dataset_builder.load_train_samples()
        return pd.DataFrame(
            [
                {
                    "image_id": sample.image_id,
                    "image_path": sample.image_path,
                    "target": sample.target,
                }
                for sample in train_samples
            ]
        )

    def load_test_metadata(self) -> pd.DataFrame:
        """Load test metadata as a DataFrame.

        Returns:
            A DataFrame with ``image_id`` and ``image_path`` columns.
        """
        test_samples = self.dataset_builder.load_test_samples()
        return pd.DataFrame(
            [
                {
                    "image_id": sample.image_id,
                    "image_path": sample.image_path,
                }
                for sample in test_samples
            ]
        )

    def compute_label_distribution(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Compute label counts and proportions.

        Args:
            train_df: Training metadata DataFrame.

        Returns:
            A sorted DataFrame with label counts and proportions.
        """
        distribution_df = (
            train_df.groupby("target")
            .size()
            .reset_index(name="count")
            .sort_values("target")
            .reset_index(drop=True)
        )
        distribution_df["ratio"] = distribution_df["count"] / distribution_df["count"].sum()
        return distribution_df

    def compute_image_statistics(
        self,
        metadata_df: pd.DataFrame,
        sample_size: Optional[int] = 256,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute per-image statistics on a sample of images.

        Args:
            metadata_df: Metadata DataFrame containing image paths.
            sample_size: Optional number of images to analyze.
            random_state: Random seed for sampling rows.

        Returns:
            A DataFrame with width, height, and intensity statistics.
        """
        if sample_size is not None and len(metadata_df) > sample_size:
            target_df = metadata_df.sample(n=sample_size, random_state=random_state)
        else:
            target_df = metadata_df.copy()

        statistics = []
        for row in target_df.itertuples(index=False):
            image_array = np.asarray(Image.open(row.image_path).convert("RGB"), dtype=np.float32)
            grayscale = image_array.mean(axis=2)

            statistics.append(
                {
                    "image_id": row.image_id,
                    "width": image_array.shape[1],
                    "height": image_array.shape[0],
                    "pixel_mean": float(grayscale.mean()),
                    "pixel_std": float(grayscale.std()),
                    "pixel_min": float(grayscale.min()),
                    "pixel_max": float(grayscale.max()),
                }
            )

        return pd.DataFrame(statistics)

    def sample_images_by_class(
        self,
        train_df: pd.DataFrame,
        samples_per_class: int = 4,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Sample a small, class-balanced subset for visualization.

        Args:
            train_df: Training metadata DataFrame.
            samples_per_class: Number of examples to draw per class.
            random_state: Random seed for deterministic sampling.

        Returns:
            A DataFrame containing sampled rows from each class.
        """
        sampled_frames = []
        for target_value, group_df in train_df.groupby("target"):
            sampled_frames.append(
                group_df.sample(
                    n=min(samples_per_class, len(group_df)),
                    random_state=random_state + int(target_value),
                )
            )

        return pd.concat(sampled_frames, axis=0).sort_values(["target", "image_id"])

    def build_dataset_summary(
        self,
        image_stat_sample_size: Optional[int] = 256,
    ) -> Dict[str, pd.DataFrame]:
        """Build the main analysis tables for notebook usage.

        Args:
            image_stat_sample_size: Number of images used for image-level statistics.

        Returns:
            A dictionary of DataFrames ready for notebook display.
        """
        train_df = self.load_train_metadata()
        test_df = self.load_test_metadata()
        label_distribution_df = self.compute_label_distribution(train_df)
        train_image_stats_df = self.compute_image_statistics(
            metadata_df=train_df,
            sample_size=image_stat_sample_size,
        )
        test_image_stats_df = self.compute_image_statistics(
            metadata_df=test_df,
            sample_size=image_stat_sample_size,
        )
        sample_grid_df = self.sample_images_by_class(train_df=train_df)

        summary_df = pd.DataFrame(
            [
                {"metric": "num_train_images", "value": len(train_df)},
                {"metric": "num_test_images", "value": len(test_df)},
                {"metric": "num_classes", "value": train_df["target"].nunique()},
                {
                    "metric": "mean_train_pixel_mean",
                    "value": train_image_stats_df["pixel_mean"].mean(),
                },
                {
                    "metric": "mean_train_pixel_std",
                    "value": train_image_stats_df["pixel_std"].mean(),
                },
            ]
        )

        self.logger.info("Built dataset summary tables for notebook usage.")
        return {
            "summary": summary_df,
            "train_metadata": train_df,
            "test_metadata": test_df,
            "label_distribution": label_distribution_df,
            "train_image_stats": train_image_stats_df,
            "test_image_stats": test_image_stats_df,
            "sample_grid": sample_grid_df.reset_index(drop=True),
        }
