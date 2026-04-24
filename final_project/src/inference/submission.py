"""Submission file generation utilities."""

import csv
import logging
import os
from typing import Iterable, List, Optional

from src.data.dataset import SpectrogramSample
from src.utils.logger import get_logger


class SubmissionBuilder:
    """Build Kaggle submission files from predicted labels."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the submission builder.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or get_logger(__name__)

    def build(
        self,
        samples: Iterable[SpectrogramSample],
        predictions: List[int],
        output_file: str,
    ) -> None:
        """Write a Kaggle submission CSV file.

        Args:
            samples: Test samples in prediction order.
            predictions: Predicted class indices.
            output_file: Destination submission file path.

        Raises:
            ValueError: If sample count and prediction count differ.
        """
        sample_list = list(samples)
        if len(sample_list) != len(predictions):
            raise ValueError(
                "Number of predictions does not match the number of provided samples."
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["id", "target"])
            for sample, prediction in zip(sample_list, predictions):
                writer.writerow([sample.image_id, prediction])

        self.logger.info("Submission saved to %s", output_file)
