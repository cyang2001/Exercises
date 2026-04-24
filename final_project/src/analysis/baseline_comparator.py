"""Baseline comparison helpers for validation summaries."""

import csv
import logging
import os
from typing import Any, Dict, List

from src.utils.logger import get_logger


class BaselineComparator:
    """Aggregate and rank validation summaries across baseline models."""

    def __init__(self, cfg: Any, logger: logging.Logger = None) -> None:
        """Initialize the baseline comparator.

        Args:
            cfg: Project configuration object.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        self.experiment_root_dir = cfg.paths.experiment_root_dir
        self.output_file = cfg.paths.baseline_comparison_file

    def compare(self, model_names: List[str], aug_name: str) -> List[Dict[str, Any]]:
        """Compare validation summaries for a list of baseline models.

        Args:
            model_names: Model names whose summaries should be compared.
            aug_name: Augmentation preset name shared by the compared runs.

        Returns:
            Ranked comparison rows sorted by validation macro F1 and accuracy.

        Raises:
            FileNotFoundError: If none of the requested model summaries exist.
        """
        comparison_rows: List[Dict[str, Any]] = []

        for model_name in model_names:
            summary_file = self._build_summary_file(
                model_name=model_name,
                aug_name=aug_name,
            )
            if not os.path.exists(summary_file):
                self.logger.warning(
                    "Missing validation summary for model=%s aug=%s at %s",
                    model_name,
                    aug_name,
                    summary_file,
                )
                continue

            comparison_rows.append(self._read_summary(summary_file))

        if not comparison_rows:
            raise FileNotFoundError(
                "No validation summary files were found for the requested baselines."
            )

        ranked_rows = self._rank_rows(comparison_rows)
        self._write_rows(ranked_rows)
        return ranked_rows

    def _build_summary_file(self, model_name: str, aug_name: str) -> str:
        """Build the expected validation summary path for a model run.

        Args:
            model_name: Model name used in the training run.
            aug_name: Augmentation preset name used in the training run.

        Returns:
            Absolute path to the validation summary ``csv`` file.
        """
        return os.path.join(
            self.experiment_root_dir,
            model_name,
            aug_name,
            "validation_summary.csv",
        )

    def _read_summary(self, summary_file: str) -> Dict[str, Any]:
        """Read and normalize a single validation summary row.

        Args:
            summary_file: Path to the validation summary ``csv`` file.

        Returns:
            Normalized summary row with numeric metric fields converted.
        """
        with open(summary_file, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            row = next(reader)

        row["best_epoch"] = int(row["best_epoch"])
        row["valid_loss"] = float(row["valid_loss"])
        row["valid_accuracy"] = float(row["valid_accuracy"])
        row["valid_macro_f1"] = float(row["valid_macro_f1"])
        row["summary_file"] = summary_file
        return row

    def _rank_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank comparison rows from strongest to weakest validation result.

        Args:
            rows: Raw summary rows loaded from baseline runs.

        Returns:
            Sorted rows with a ``rank`` field added.
        """
        sorted_rows = sorted(
            rows,
            key=lambda row: (row["valid_macro_f1"], row["valid_accuracy"]),
            reverse=True,
        )

        ranked_rows: List[Dict[str, Any]] = []
        for rank, row in enumerate(sorted_rows, start=1):
            ranked_row = dict(row)
            ranked_row["rank"] = rank
            ranked_rows.append(ranked_row)

        return ranked_rows

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Write ranked baseline comparison rows to the configured output file.

        Args:
            rows: Ranked comparison rows to persist.
        """
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info("Baseline comparison saved to %s", self.output_file)
