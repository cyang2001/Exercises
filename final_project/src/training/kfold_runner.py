"""K-fold training utilities for final model evaluation."""

import csv
import logging
import os
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data.dataset import SpectrogramDataset, SpectrogramDatasetBuilder, SpectrogramSample
from src.data.split import TrainValidationSplitter
from src.data.transforms import TransformFactory
from src.models.factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.logger import get_logger


class StratifiedKFoldRunner:
    """Run stratified k-fold evaluation and export aggregate artifacts."""

    def __init__(
        self,
        cfg: Any,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the k-fold runner.

        Args:
            cfg: Project configuration object.
            device: Target torch device.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.device = device
        self.logger = logger or get_logger(__name__)
        self.dataset_builder = SpectrogramDatasetBuilder(cfg=cfg, logger=self.logger)
        self.splitter = TrainValidationSplitter(cfg=cfg, logger=self.logger)
        self.transform_factory = TransformFactory(cfg=cfg)

    def run(self) -> Dict[str, Any]:
        """Run full stratified k-fold training and write aggregate artifacts.

        Returns:
            A compact dictionary with aggregate k-fold metrics.
        """
        train_samples = self.dataset_builder.load_train_samples()
        self.dataset_builder.validate_train_samples(train_samples)
        fold_splits = self.splitter.build_kfold_splits(
            samples=train_samples,
            n_splits=int(self.cfg.train.n_splits),
        )

        fold_rows: List[Dict[str, Any]] = []
        oof_rows: List[Dict[str, Any]] = []

        for fold_index, (fold_train_samples, fold_valid_samples) in enumerate(fold_splits, start=1):
            self.logger.info(
                "Starting fold %s/%s with %s train samples and %s valid samples.",
                fold_index,
                len(fold_splits),
                len(fold_train_samples),
                len(fold_valid_samples),
            )
            artifact_paths = self._build_fold_artifact_paths(fold_index=fold_index)
            trainer = self._build_trainer(artifact_paths=artifact_paths)
            train_loader, valid_loader = self._build_data_loaders(
                train_samples=fold_train_samples,
                valid_samples=fold_valid_samples,
            )
            class_weights = self._build_class_weights(samples=fold_train_samples)
            fold_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                class_weights=class_weights,
            )

            fold_rows.append(
                {
                    "fold_id": fold_index,
                    "train_samples": len(fold_train_samples),
                    "valid_samples": len(fold_valid_samples),
                    "best_epoch": int(fold_result["best_epoch"]),
                    "valid_loss": float(fold_result["best_valid_loss"]),
                    "valid_accuracy": float(fold_result["best_valid_accuracy"]),
                    "valid_macro_f1": float(fold_result["best_valid_macro_f1"]),
                    "checkpoint_file": artifact_paths["best_model_file"],
                    "history_file": artifact_paths["training_history_file"],
                    "validation_prediction_file": artifact_paths["validation_predictions_file"],
                    "validation_summary_file": artifact_paths["validation_summary_file"],
                }
            )
            oof_rows.extend(
                self._attach_fold_metadata(
                    fold_id=fold_index,
                    prediction_rows=fold_result["best_prediction_rows"],
                )
            )

        summary_rows = self._build_summary_rows(fold_rows=fold_rows)
        confusion_rows = self._build_confusion_rows(oof_rows=oof_rows)
        class_metric_rows = self._build_class_metric_rows(oof_rows=oof_rows)

        self._write_csv(self.cfg.paths.kfold_fold_metrics_file, fold_rows)
        self._write_csv(self.cfg.paths.kfold_oof_predictions_file, oof_rows)
        self._write_csv(self.cfg.paths.kfold_summary_file, summary_rows)
        self._write_csv(self.cfg.paths.kfold_confusion_matrix_file, confusion_rows)
        self._write_csv(self.cfg.paths.kfold_class_metrics_file, class_metric_rows)

        summary_map = {row["metric"]: row["value"] for row in summary_rows}
        self.logger.info(
            "K-fold evaluation finished with mean_macro_f1=%s and mean_accuracy=%s",
            summary_map["mean_valid_macro_f1"],
            summary_map["mean_valid_accuracy"],
        )
        return summary_map

    def _build_trainer(self, artifact_paths: Dict[str, str]) -> Trainer:
        """Build a trainer instance for one fold.

        Args:
            artifact_paths: Artifact paths scoped to one fold.

        Returns:
            A configured trainer instance.
        """
        model = ModelFactory(cfg=self.cfg, logger=self.logger).build_model()
        return Trainer(
            cfg=self.cfg,
            model=model,
            device=self.device,
            artifact_paths=artifact_paths,
            logger=self.logger,
        )

    def _build_data_loaders(
        self,
        train_samples: List[SpectrogramSample],
        valid_samples: List[SpectrogramSample],
    ) -> List[DataLoader]:
        """Build train and validation data loaders for one fold.

        Args:
            train_samples: Fold training samples.
            valid_samples: Fold validation samples.

        Returns:
            A list containing train and validation loaders.
        """
        train_dataset = SpectrogramDataset(
            samples=train_samples,
            transform=self.transform_factory.build_train_transform(),
            image_mode=self._resolve_pil_image_mode(),
            logger=self.logger,
        )
        valid_dataset = SpectrogramDataset(
            samples=valid_samples,
            transform=self.transform_factory.build_eval_transform(),
            image_mode=self._resolve_pil_image_mode(),
            logger=self.logger,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )
        return [train_loader, valid_loader]

    def _build_class_weights(self, samples: List[SpectrogramSample]) -> Optional[torch.Tensor]:
        """Build inverse-frequency class weights for one fold.

        Args:
            samples: Fold training samples.

        Returns:
            Class-weight tensor or ``None`` if disabled.
        """
        if not self.cfg.train.use_class_weights:
            return None

        label_distribution: Dict[int, int] = {}
        for sample in samples:
            label_distribution[sample.target] = label_distribution.get(sample.target, 0) + 1

        total_count = sum(label_distribution.values())
        class_weights = []
        for class_id in range(self.cfg.data.num_classes):
            class_count = label_distribution.get(class_id, 1)
            class_weights.append(total_count / (self.cfg.data.num_classes * class_count))

        return torch.tensor(class_weights, dtype=torch.float32, device=self.device)

    def _build_fold_artifact_paths(self, fold_index: int) -> Dict[str, str]:
        """Build artifact paths for one fold.

        Args:
            fold_index: One-based fold index.

        Returns:
            Mapping of artifact path names to absolute file paths.
        """
        fold_dir = os.path.join(self.cfg.paths.kfold_folds_dir, f"fold_{fold_index}")
        checkpoint_dir = os.path.join(fold_dir, "checkpoints")
        return {
            "checkpoint_dir": checkpoint_dir,
            "best_model_file": os.path.join(checkpoint_dir, "best_model.pt"),
            "training_history_file": os.path.join(fold_dir, "training_history.csv"),
            "validation_predictions_file": os.path.join(fold_dir, "validation_predictions.csv"),
            "validation_summary_file": os.path.join(fold_dir, "validation_summary.csv"),
        }

    def _attach_fold_metadata(
        self,
        fold_id: int,
        prediction_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Attach fold metadata to exported validation predictions.

        Args:
            fold_id: One-based fold index.
            prediction_rows: Best-epoch prediction rows for the fold.

        Returns:
            Prediction rows with the fold identifier added.
        """
        rows_with_fold: List[Dict[str, Any]] = []
        for row in prediction_rows:
            row_with_fold = dict(row)
            row_with_fold["fold_id"] = fold_id
            rows_with_fold.append(row_with_fold)
        return rows_with_fold

    def _build_summary_rows(self, fold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build aggregate summary rows from per-fold metrics.

        Args:
            fold_rows: Per-fold result rows.

        Returns:
            Summary rows with mean and standard deviation metrics.
        """
        losses = [float(row["valid_loss"]) for row in fold_rows]
        accuracies = [float(row["valid_accuracy"]) for row in fold_rows]
        macro_f1_scores = [float(row["valid_macro_f1"]) for row in fold_rows]
        best_row = max(fold_rows, key=lambda row: row["valid_macro_f1"])

        return [
            {"metric": "num_folds", "value": len(fold_rows)},
            {"metric": "mean_valid_loss", "value": float(mean(losses))},
            {"metric": "std_valid_loss", "value": self._safe_std(losses)},
            {"metric": "mean_valid_accuracy", "value": float(mean(accuracies))},
            {"metric": "std_valid_accuracy", "value": self._safe_std(accuracies)},
            {"metric": "mean_valid_macro_f1", "value": float(mean(macro_f1_scores))},
            {"metric": "std_valid_macro_f1", "value": self._safe_std(macro_f1_scores)},
            {"metric": "best_fold_id", "value": int(best_row["fold_id"])},
            {"metric": "best_fold_macro_f1", "value": float(best_row["valid_macro_f1"])},
            {"metric": "best_fold_accuracy", "value": float(best_row["valid_accuracy"])},
        ]

    def _build_confusion_rows(self, oof_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build confusion-matrix rows from out-of-fold predictions.

        Args:
            oof_rows: Combined out-of-fold prediction rows.

        Returns:
            CSV-ready confusion matrix rows.
        """
        labels = list(range(self.cfg.data.num_classes))
        targets = [int(row["y_true"]) for row in oof_rows]
        predictions = [int(row["y_pred"]) for row in oof_rows]
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

    def _build_class_metric_rows(self, oof_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build class-level metrics from out-of-fold predictions.

        Args:
            oof_rows: Combined out-of-fold prediction rows.

        Returns:
            CSV-ready class metric rows.
        """
        labels = list(range(self.cfg.data.num_classes))
        targets = [int(row["y_true"]) for row in oof_rows]
        predictions = [int(row["y_pred"]) for row in oof_rows]
        precision, recall, f1_score_values, supports = precision_recall_fscore_support(
            targets,
            predictions,
            labels=labels,
            zero_division=0,
        )

        rows: List[Dict[str, Any]] = []
        for class_id in labels:
            rows.append(
                {
                    "class_id": class_id,
                    "support": int(supports[class_id]),
                    "precision": float(precision[class_id]),
                    "recall": float(recall[class_id]),
                    "f1_score": float(f1_score_values[class_id]),
                }
            )
        return rows

    def _resolve_pil_image_mode(self) -> str:
        """Resolve the Pillow image mode used by dataset loading.

        Returns:
            A Pillow image mode string such as ``"L"`` or ``"RGB"``.
        """
        configured_mode = str(self.cfg.data.get("image_mode", "rgb")).lower()
        if configured_mode in {"gray", "grayscale", "l"} or self.cfg.data.num_channels == 1:
            return "L"
        return "RGB"

    def _safe_std(self, values: List[float]) -> float:
        """Compute population standard deviation with short-list fallback.

        Args:
            values: Numeric values to summarize.

        Returns:
            Population standard deviation or ``0.0`` when unavailable.
        """
        if len(values) <= 1:
            return 0.0
        return float(pstdev(values))

    def _write_csv(self, output_file: str, rows: List[Dict[str, Any]]) -> None:
        """Write rows to a CSV file.

        Args:
            output_file: Destination CSV path.
            rows: Rows to persist.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if not rows:
            self.logger.warning("Skip writing empty CSV to %s", output_file)
            return

        with open(output_file, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info("Saved k-fold artifact to %s", output_file)
