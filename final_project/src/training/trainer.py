"""Training loop implementation for the baseline classifier."""

import csv
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.loss import LossFactory
from src.training.metrics import ClassificationMetrics
from src.training.scheduler import SchedulerFactory
from src.utils.logger import get_logger


class Trainer:
    """Train and validate a spectrogram classification model."""

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        device: torch.device,
        artifact_paths: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            cfg: Project configuration object.
            model: Model to train.
            device: Target torch device.
            artifact_paths: Optional override paths for training artifacts.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.artifact_paths = artifact_paths or {}
        self.logger = logger or get_logger(__name__)
        self.metrics = ClassificationMetrics()
        self.enable_progress_bar = self.cfg.train.get("enable_progress_bar", True)
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.scheduler = SchedulerFactory().build(self.optimizer, self.cfg)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Train the model and return the best validation summary.

        Args:
            train_loader: DataLoader for training batches.
            valid_loader: DataLoader for validation batches.
            class_weights: Optional class weights tensor.

        Returns:
            Best validation metrics and checkpoint metadata.
        """
        criterion = LossFactory().build_cross_entropy(class_weights=class_weights)
        best_macro_f1 = float("-inf")
        best_epoch = -1
        best_valid_result: Optional[Dict[str, Any]] = None
        best_prediction_rows: List[Dict[str, Any]] = []
        history_rows: List[Dict[str, float]] = []
        epochs_without_improvement = 0

        for epoch in range(1, self.cfg.train.epochs + 1):
            train_result = self._run_one_epoch(
                data_loader=train_loader,
                criterion=criterion,
                training=True,
                max_batches=self.cfg.train.max_train_batches,
                epoch=epoch,
            )
            valid_result = self._run_one_epoch(
                data_loader=valid_loader,
                criterion=criterion,
                training=False,
                max_batches=self.cfg.train.max_valid_batches,
                epoch=epoch,
            )

            if self.scheduler is not None:
                self.scheduler.step(valid_result["macro_f1"])

            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_accuracy": train_result["accuracy"],
                "train_macro_f1": train_result["macro_f1"],
                "valid_loss": valid_result["loss"],
                "valid_accuracy": valid_result["accuracy"],
                "valid_macro_f1": valid_result["macro_f1"],
            }
            history_rows.append(epoch_summary)
            self.logger.info(
                (
                    "Epoch %s | train_loss=%.4f | train_acc=%.4f | train_f1=%.4f | "
                    "valid_loss=%.4f | valid_acc=%.4f | valid_f1=%.4f"
                ),
                epoch,
                train_result["loss"],
                train_result["accuracy"],
                train_result["macro_f1"],
                valid_result["loss"],
                valid_result["accuracy"],
                valid_result["macro_f1"],
            )

            if valid_result["macro_f1"] > best_macro_f1:
                best_macro_f1 = valid_result["macro_f1"]
                best_epoch = epoch
                best_valid_result = valid_result
                best_prediction_rows = valid_result["prediction_rows"]
                epochs_without_improvement = 0
                self._save_checkpoint(
                    epoch=epoch,
                    valid_result=self._extract_metric_summary(valid_result),
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.cfg.train.early_stopping_patience:
                self.logger.info(
                    "Early stopping triggered after %s epochs without improvement.",
                    epochs_without_improvement,
                )
                break

        self._write_history(history_rows)
        if best_valid_result is not None:
            self._write_validation_predictions(best_prediction_rows)
            self._write_validation_summary(
                epoch=best_epoch,
                valid_result=best_valid_result,
            )

        return {
            "best_epoch": best_epoch,
            "best_valid_macro_f1": best_macro_f1,
            "best_valid_accuracy": (
                best_valid_result["accuracy"] if best_valid_result is not None else None
            ),
            "best_valid_loss": (
                best_valid_result["loss"] if best_valid_result is not None else None
            ),
            "history_rows": history_rows,
            "best_prediction_rows": best_prediction_rows,
        }

    def _get_artifact_path(self, path_name: str) -> str:
        """Resolve one artifact path with optional runtime overrides.

        Args:
            path_name: Attribute name inside ``cfg.paths``.

        Returns:
            Final artifact path for the current trainer instance.
        """
        if path_name in self.artifact_paths:
            return self.artifact_paths[path_name]
        return str(getattr(self.cfg.paths, path_name))

    def _run_one_epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        training: bool,
        max_batches: Optional[int],
        epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one training or validation epoch.

        Args:
            data_loader: Loader that yields mini-batches.
            criterion: Loss function instance.
            training: Whether the epoch is a training epoch.
            max_batches: Optional upper bound for processed batches.
            epoch: Optional epoch index for progress bar labeling.

        Returns:
            Aggregated loss and metric values.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_targets: List[int] = []
        all_predictions: List[int] = []
        prediction_rows: List[Dict[str, Any]] = []
        progress_bar = self._build_progress_bar(
            data_loader=data_loader,
            training=training,
            max_batches=max_batches,
            epoch=epoch,
        )

        for batch_index, batch in enumerate(data_loader, start=1):
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            image_ids = batch["image_id"]

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = criterion(logits, targets)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            if not training:
                prediction_rows.extend(
                    self._build_prediction_rows(
                        image_ids=image_ids,
                        targets=targets,
                        predictions=predictions,
                        probabilities=probabilities,
                    )
                )

            if progress_bar is not None:
                average_loss = total_loss / max(total_samples, 1)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "avg_loss": f"{average_loss:.4f}",
                        "batch": batch_index,
                    }
                )

            if max_batches is not None and batch_index >= max_batches:
                break

        if progress_bar is not None:
            progress_bar.close()

        metric_values = self.metrics.compute(
            targets=all_targets,
            predictions=all_predictions,
        )
        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": metric_values["accuracy"],
            "macro_f1": metric_values["macro_f1"],
            "prediction_rows": prediction_rows,
        }

    def _build_progress_bar(
        self,
        data_loader: DataLoader,
        training: bool,
        max_batches: Optional[int],
        epoch: Optional[int],
    ) -> Optional[tqdm]:
        """Build a progress bar for one epoch when enabled.

        Args:
            data_loader: Loader used for the current epoch.
            training: Whether the current loop is training or validation.
            max_batches: Optional maximum number of processed batches.
            epoch: Optional epoch index for display.

        Returns:
            A configured tqdm progress bar or ``None`` when disabled.
        """
        if not self.enable_progress_bar:
            return None

        total_batches = self._resolve_total_batches(
            data_loader=data_loader,
            max_batches=max_batches,
        )
        phase_name = "train" if training else "valid"
        description = f"Epoch {epoch or '?'} [{phase_name}]"
        return tqdm(
            total=total_batches,
            desc=description,
            leave=False,
            dynamic_ncols=True,
        )

    def _resolve_total_batches(
        self,
        data_loader: DataLoader,
        max_batches: Optional[int],
    ) -> Optional[int]:
        """Resolve the number of batches shown in the progress bar.

        Args:
            data_loader: Loader used for the current epoch.
            max_batches: Optional maximum number of processed batches.

        Returns:
            Total batch count for tqdm or ``None`` when unavailable.
        """
        try:
            total_batches = len(data_loader)
        except TypeError:
            total_batches = None

        if total_batches is None:
            return max_batches
        if max_batches is None:
            return total_batches
        return min(total_batches, max_batches)

    def _build_prediction_rows(
        self,
        image_ids: Any,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Build row-wise validation prediction records for artifact export.

        Args:
            image_ids: Batch image identifiers from the dataloader.
            targets: Ground-truth target tensor.
            predictions: Predicted class tensor.
            probabilities: Per-class probability tensor.

        Returns:
            A list of row dictionaries ready to be written as ``csv``.
        """
        if isinstance(image_ids, torch.Tensor):
            image_id_values = image_ids.detach().cpu().tolist()
        else:
            image_id_values = list(image_ids)

        target_values = targets.detach().cpu().tolist()
        prediction_values = predictions.detach().cpu().tolist()
        probability_values = probabilities.detach().cpu().tolist()
        prediction_rows: List[Dict[str, Any]] = []

        for index, image_id in enumerate(image_id_values):
            row: Dict[str, Any] = {
                "image_id": int(image_id),
                "y_true": int(target_values[index]),
                "y_pred": int(prediction_values[index]),
            }
            for class_index, probability in enumerate(probability_values[index]):
                row[f"prob_{class_index}"] = float(probability)
            prediction_rows.append(row)

        return prediction_rows

    def _extract_metric_summary(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract compact metric fields from an epoch result dictionary.

        Args:
            result: Epoch result dictionary that may include artifact payloads.

        Returns:
            A dictionary containing scalar metric values only.
        """
        return {
            "loss": float(result["loss"]),
            "accuracy": float(result["accuracy"]),
            "macro_f1": float(result["macro_f1"]),
        }

    def _save_checkpoint(self, epoch: int, valid_result: Dict[str, float]) -> None:
        """Save the current best model checkpoint.

        Args:
            epoch: Epoch index associated with the checkpoint.
            valid_result: Validation metrics for the checkpoint.
        """
        checkpoint_dir = self._get_artifact_path("checkpoint_dir")
        best_model_file = self._get_artifact_path("best_model_file")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "valid_metrics": valid_result,
                "model_name": self.cfg.model.name,
            },
            best_model_file,
        )
        self.logger.info("Saved best checkpoint to %s", best_model_file)

    def _write_history(self, history_rows: List[Dict[str, float]]) -> None:
        """Write epoch history to the configured CSV file.

        Args:
            history_rows: Per-epoch metric records.
        """
        if not history_rows:
            return

        training_history_file = self._get_artifact_path("training_history_file")
        os.makedirs(os.path.dirname(training_history_file), exist_ok=True)
        field_names = list(history_rows[0].keys())

        with open(
            training_history_file,
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(history_rows)

        self.logger.info(
            "Training history saved to %s",
            training_history_file,
        )

    def _write_validation_predictions(
        self,
        prediction_rows: List[Dict[str, Any]],
    ) -> None:
        """Write best-epoch validation predictions to the configured CSV file.

        Args:
            prediction_rows: Validation prediction records from the best epoch.
        """
        if not prediction_rows:
            return

        validation_predictions_file = self._get_artifact_path("validation_predictions_file")
        os.makedirs(
            os.path.dirname(validation_predictions_file),
            exist_ok=True,
        )
        field_names = list(prediction_rows[0].keys())

        with open(
            validation_predictions_file,
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(prediction_rows)

        self.logger.info(
            "Validation predictions saved to %s",
            validation_predictions_file,
        )

    def _write_validation_summary(
        self,
        epoch: int,
        valid_result: Dict[str, Any],
    ) -> None:
        """Write the best validation summary for downstream baseline comparison.

        Args:
            epoch: Best validation epoch index.
            valid_result: Best validation metrics and prediction payload.
        """
        summary_row = {
            "model_name": self.cfg.model.name,
            "aug_name": self.cfg.aug.name,
            "train_strategy": self.cfg.train.strategy,
            "pretrained": self.cfg.model.pretrained,
            "best_epoch": epoch,
            "valid_loss": float(valid_result["loss"]),
            "valid_accuracy": float(valid_result["accuracy"]),
            "valid_macro_f1": float(valid_result["macro_f1"]),
            "checkpoint_file": self._get_artifact_path("best_model_file"),
            "history_file": self._get_artifact_path("training_history_file"),
            "validation_prediction_file": self._get_artifact_path("validation_predictions_file"),
        }

        validation_summary_file = self._get_artifact_path("validation_summary_file")
        os.makedirs(
            os.path.dirname(validation_summary_file),
            exist_ok=True,
        )
        with open(
            validation_summary_file,
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)

        self.logger.info(
            "Validation summary saved to %s",
            validation_summary_file,
        )
