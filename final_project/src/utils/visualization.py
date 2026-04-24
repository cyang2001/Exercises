"""Visualization helpers for notebook-based reporting."""

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


class SpectrogramVisualizer:
    """Build reusable figures for dataset exploration and reporting."""

    def plot_label_distribution(
        self,
        distribution_df: pd.DataFrame,
        title: str = "Label Distribution",
    ):
        """Plot the class distribution as a bar chart.

        Args:
            distribution_df: DataFrame with ``target`` and ``count`` columns.
            title: Figure title.

        Returns:
            The created matplotlib figure and axis.
        """
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            distribution_df["target"].astype(str),
            distribution_df["count"],
            color="steelblue",
        )
        axis.set_title(title)
        axis.set_xlabel("Target")
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        return figure, axis

    def plot_stat_histogram(
        self,
        stats_df: pd.DataFrame,
        column: str,
        bins: int = 30,
        title: Optional[str] = None,
    ):
        """Plot a histogram for one numeric image statistic.

        Args:
            stats_df: DataFrame containing image statistics.
            column: Numeric column to visualize.
            bins: Histogram bin count.
            title: Optional custom figure title.

        Returns:
            The created matplotlib figure and axis.
        """
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(stats_df[column].dropna(), bins=bins, color="mediumpurple", alpha=0.8)
        axis.set_title(title or f"{column} Distribution")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        return figure, axis

    def plot_sample_grid(
        self,
        sample_df: pd.DataFrame,
        title: str = "Sample Spectrograms",
    ):
        """Plot a grid of spectrogram samples from file paths.

        Args:
            sample_df: DataFrame containing ``image_path`` and optional ``target`` columns.
            title: Figure title.

        Returns:
            The created matplotlib figure and axes array.
        """
        sample_count = len(sample_df)
        columns = min(4, max(sample_count, 1))
        rows = int(np.ceil(sample_count / columns))

        figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
        axes_array = np.array(axes).reshape(-1)

        for axis in axes_array:
            axis.axis("off")

        for axis, row in zip(axes_array, sample_df.itertuples(index=False)):
            image = Image.open(row.image_path).convert("RGB")
            axis.imshow(image)
            label_text = f" | target={row.target}" if hasattr(row, "target") else ""
            axis.set_title(f"id={row.image_id}{label_text}")
            axis.axis("off")

        figure.suptitle(title)
        figure.tight_layout()
        return figure, axes

    def plot_channel_profiles(
        self,
        image_paths: Iterable[str],
        title: str = "Mean Intensity Profile",
    ):
        """Plot the average horizontal intensity profile across images.

        Args:
            image_paths: Collection of image paths used to compute the profile.
            title: Figure title.

        Returns:
            The created matplotlib figure and axis.
        """
        profiles = []
        for image_path in image_paths:
            image_array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
            grayscale = image_array.mean(axis=2)
            profiles.append(grayscale.mean(axis=0))

        stacked_profiles = np.stack(profiles, axis=0)
        mean_profile = stacked_profiles.mean(axis=0)

        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(mean_profile, color="darkorange")
        axis.set_title(title)
        axis.set_xlabel("Time Axis")
        axis.set_ylabel("Mean Intensity")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        return figure, axis
