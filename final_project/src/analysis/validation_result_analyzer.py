"""Validation artifact analysis helpers for holdout experiments."""

import csv
import logging
import os
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.dataset import SpectrogramDatasetBuilder
from src.utils.logger import get_logger


class ValidationResultAnalyzer:
    """Analyze validation predictions and export readable error-analysis artifacts."""

    def __init__(
        self,
        cfg: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the validation result analyzer.

        Args:
            cfg: Project configuration object.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        self.dataset_builder = SpectrogramDatasetBuilder(cfg=cfg, logger=self.logger)
        self.num_classes = int(cfg.data.num_classes)
        self.top_k_errors = int(cfg.analysis.top_k_errors)

    def analyze(self) -> Dict[str, Any]:
        """Analyze the saved validation predictions for the current experiment.

        Returns:
            A compact dictionary with the main validation analysis summary.
        """
        prediction_rows = self._load_prediction_rows()
        image_path_mapping = self._load_image_path_mapping()
        targets = [int(row["y_true"]) for row in prediction_rows]
        predictions = [int(row["y_pred"]) for row in prediction_rows]

        confusion_rows = self._build_confusion_matrix_rows(targets=targets, predictions=predictions)
        class_metric_rows = self._build_class_metric_rows(targets=targets, predictions=predictions)
        misclassified_rows = self._build_misclassified_rows(
            prediction_rows=prediction_rows,
            image_path_mapping=image_path_mapping,
        )
        summary_rows = self._build_summary_rows(
            prediction_rows=prediction_rows,
            targets=targets,
            predictions=predictions,
            misclassified_rows=misclassified_rows,
        )

        self._write_csv(self.cfg.paths.validation_confusion_matrix_file, confusion_rows)
        self._write_csv(self.cfg.paths.validation_class_metrics_file, class_metric_rows)
        self._write_csv(self.cfg.paths.validation_misclassified_file, misclassified_rows)
        self._write_csv(self.cfg.paths.validation_analysis_summary_file, summary_rows)

        summary_map = {row["metric"]: row["value"] for row in summary_rows}
        self.logger.info(
            "Validation analysis finished for model=%s aug=%s with accuracy=%s and macro_f1=%s",
            self.cfg.model.name,
            self.cfg.aug.name,
            summary_map["accuracy"],
            summary_map["macro_f1"],
        )
        return summary_map

    def _load_prediction_rows(self) -> List[Dict[str, Any]]:
        """Load validation prediction records from the saved CSV artifact.

        Returns:
            Parsed validation prediction rows.

        Raises:
            FileNotFoundError: If the configured prediction file does not exist.
        """
        prediction_file = self.cfg.paths.validation_predictions_file
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"Validation prediction file not found: {prediction_file}")

        with open(prediction_file, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = [self._normalize_prediction_row(row) for row in reader]

        self.logger.info("Loaded %s validation prediction rows.", len(rows))
        return rows

    def _normalize_prediction_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Normalize one prediction row from CSV text into typed values.

        Args:
            row: Raw CSV row.

        Returns:
            Row with numeric values converted to Python numeric types.
        """
        normalized_row: Dict[str, Any] = {
            "image_id": int(row["image_id"]),
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        }
        for key, value in row.items():
            if key.startswith("prob_"):
                normalized_row[key] = float(value)
        return normalized_row

    def _load_image_path_mapping(self) -> Dict[int, str]:
        """Build a mapping from image id to training image path.

        Returns:
            Mapping from image id to image path.
        """
        train_samples = self.dataset_builder.load_train_samples()
        return {sample.image_id: sample.image_path for sample in train_samples}

    def _build_confusion_matrix_rows(
        self,
        targets: List[int],
        predictions: List[int],
    ) -> List[Dict[str, Any]]:
        """Build confusion-matrix rows with totals for CSV export.

        Args:
            targets: Ground-truth labels.
            predictions: Predicted labels.

        Returns:
            CSV-ready confusion matrix rows.
        """
        labels = list(range(self.num_classes))
        matrix = confusion_matrix(targets, predictions, labels=labels)
        rows: List[Dict[str, Any]] = []

        for target_index, matrix_row in enumerate(matrix):
            row: Dict[str, Any] = {"true_label": target_index}
            for predicted_index, cell_value in enumerate(matrix_row):
                row[f"pred_{predicted_index}"] = int(cell_value)
            row["row_total"] = int(sum(matrix_row))
            rows.append(row)

        totals_row: Dict[str, Any] = {"true_label": "all"}
        for predicted_index in labels:
            totals_row[f"pred_{predicted_index}"] = int(matrix[:, predicted_index].sum())
        totals_row["row_total"] = int(matrix.sum())
        rows.append(totals_row)
        return rows

    def _build_class_metric_rows(
        self,
        targets: List[int],
        predictions: List[int],
    ) -> List[Dict[str, Any]]:
        """Build per-class precision, recall, and F1 metrics.

        Args:
            targets: Ground-truth labels.
            predictions: Predicted labels.

        Returns:
            CSV-ready per-class metric rows.
        """
        labels = list(range(self.num_classes))
        precision, recall, f1_score_values, supports = precision_recall_fscore_support(
            targets,
            predictions,
            labels=labels,
            zero_division=0,
        )
        target_counter = Counter(targets)
        prediction_counter = Counter(predictions)
        confusion = confusion_matrix(targets, predictions, labels=labels)

        rows: List[Dict[str, Any]] = []
        for class_id in labels:
            rows.append(
                {
                    "class_id": class_id,
                    "support": int(supports[class_id]),
                    "predicted_count": int(prediction_counter.get(class_id, 0)),
                    "correct_count": int(confusion[class_id, class_id]),
                    "precision": float(precision[class_id]),
                    "recall": float(recall[class_id]),
                    "f1_score": float(f1_score_values[class_id]),
                    "target_count": int(target_counter.get(class_id, 0)),
                }
            )
        return rows

    def _build_misclassified_rows(
        self,
        prediction_rows: List[Dict[str, Any]],
        image_path_mapping: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        """Build a ranked list of misclassified validation samples.

        Args:
            prediction_rows: Validation prediction rows.
            image_path_mapping: Mapping from image id to source file path.

        Returns:
            Sorted misclassified sample rows.
        """
        misclassified_rows: List[Dict[str, Any]] = []

        for row in prediction_rows:
            if row["y_true"] == row["y_pred"]:
                continue

            predicted_probability = float(row[f"prob_{row['y_pred']}"])
            true_probability = float(row[f"prob_{row['y_true']}"])
            misclassified_row: Dict[str, Any] = {
                "image_id": row["image_id"],
                "image_path": image_path_mapping.get(row["image_id"], ""),
                "y_true": row["y_true"],
                "y_pred": row["y_pred"],
                "predicted_confidence": predicted_probability,
                "true_class_probability": true_probability,
                "confidence_gap": predicted_probability - true_probability,
            }
            for class_id in range(self.num_classes):
                misclassified_row[f"prob_{class_id}"] = row.get(f"prob_{class_id}", 0.0)
            misclassified_rows.append(misclassified_row)

        misclassified_rows.sort(
            key=lambda row: (
                row["predicted_confidence"],
                row["confidence_gap"],
            ),
            reverse=True,
        )
        return misclassified_rows[: self.top_k_errors]

    def _build_summary_rows(
        self,
        prediction_rows: List[Dict[str, Any]],
        targets: List[int],
        predictions: List[int],
        misclassified_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build compact validation summary rows for quick inspection.

        Args:
            prediction_rows: Validation prediction rows.
            targets: Ground-truth labels.
            predictions: Predicted labels.
            misclassified_rows: Ranked misclassified rows.

        Returns:
            Metric-value rows for summary CSV export.
        """
        error_rows = [row for row in prediction_rows if row["y_true"] != row["y_pred"]]
        prediction_confidences = [
            float(row[f"prob_{row['y_pred']}"])
            for row in prediction_rows
        ]
        correct_confidences = [
            float(row[f"prob_{row['y_pred']}"])
            for row in prediction_rows
            if row["y_true"] == row["y_pred"]
        ]
        error_confidences = [
            float(row[f"prob_{row['y_pred']}"])
            for row in error_rows
        ]
        confusion_counter = Counter(
            (row["y_true"], row["y_pred"])
            for row in error_rows
        )
        most_common_confusion = confusion_counter.most_common(1)
        confusion_name = ""
        confusion_count = 0
        if most_common_confusion:
            confusion_pair, confusion_count = most_common_confusion[0]
            confusion_name = f"{confusion_pair[0]}->{confusion_pair[1]}"

        return [
            {"metric": "num_validation_samples", "value": len(prediction_rows)},
            {"metric": "num_errors", "value": len(error_rows)},
            {"metric": "accuracy", "value": float(accuracy_score(targets, predictions))},
            {"metric": "macro_f1", "value": float(f1_score(targets, predictions, average="macro"))},
            {"metric": "mean_prediction_confidence", "value": self._safe_mean(prediction_confidences)},
            {"metric": "mean_correct_confidence", "value": self._safe_mean(correct_confidences)},
            {"metric": "mean_error_confidence", "value": self._safe_mean(error_confidences)},
            {"metric": "max_error_confidence", "value": max(error_confidences) if error_confidences else 0.0},
            {"metric": "top_k_errors_exported", "value": len(misclassified_rows)},
            {"metric": "most_common_confusion", "value": confusion_name},
            {"metric": "most_common_confusion_count", "value": confusion_count},
        ]

    def _safe_mean(self, values: List[float]) -> float:
        """Compute a mean value with an empty-list fallback.

        Args:
            values: Numeric values to average.

        Returns:
            Mean value or ``0.0`` if the list is empty.
        """
        if not values:
            return 0.0
        return float(mean(values))

    def _write_csv(self, output_file: str, rows: List[Dict[str, Any]]) -> None:
        """Write rows to a CSV file when data is available.

        Args:
            output_file: Destination CSV path.
            rows: Rows to write.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if not rows:
            self.logger.warning("Skip writing empty CSV to %s", output_file)
            return

        with open(output_file, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info("Saved analysis artifact to %s", output_file)
