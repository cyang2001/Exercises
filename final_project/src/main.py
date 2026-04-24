"""Main pipeline orchestration for the final project."""

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.analysis.baseline_comparator import BaselineComparator
from src.analysis.validation_result_analyzer import ValidationResultAnalyzer
from src.data.dataset import SpectrogramDataset, SpectrogramDatasetBuilder, SpectrogramSample
from src.data.split import TrainValidationSplitter
from src.data.transforms import TransformFactory
from src.inference.predict import Predictor
from src.inference.submission import SubmissionBuilder
from src.models.factory import ModelFactory
from src.training.kfold_runner import StratifiedKFoldRunner
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


class PipelineManager:
    """Coordinate data preparation, training, and inference workflows."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize the pipeline manager.

        Args:
            cfg: Project configuration object.
            logger: Optional logger instance.
        """
        self.logger = logger or get_logger(__name__)
        self._init_config(cfg)

    def _init_config(self, cfg: DictConfig) -> None:
        """Initialize configuration fields used by the pipeline.

        Args:
            cfg: Project configuration object.

        Raises:
            ValueError: If a required configuration field is missing.
        """
        self.cfg = cfg
        required_fields = [
            "project",
            "paths",
            "data",
            "mode",
            "train",
            "model",
            "aug",
            "analysis",
        ]

        for field in required_fields:
            if cfg.get(field) is None:
                raise ValueError(f"Missing required configuration field: {field}")

        self.logger.info("PipelineManager initialized successfully.")

    def run(self) -> None:
        """Run the configured project pipeline."""
        set_global_seed(self.cfg.project.random_seed)
        self._ensure_directories()
        pipeline_name = self.cfg.mode.pipeline_name
        self._validate_input_paths(pipeline_name=pipeline_name)

        if pipeline_name == "prepare_data":
            self._prepare_data_overview()
            return

        if pipeline_name == "train_baseline":
            self._run_training_pipeline()
            return

        if pipeline_name == "predict_submission":
            self._run_prediction_pipeline()
            return

        if pipeline_name == "compare_baselines":
            self._run_baseline_comparison()
            return

        if pipeline_name == "analyze_validation":
            self._run_validation_analysis_pipeline()
            return

        if pipeline_name == "run_kfold_evaluation":
            self._run_kfold_evaluation_pipeline()
            return

        raise ValueError(f"Unsupported pipeline name: {pipeline_name}")

    def _ensure_directories(self) -> None:
        """Ensure output directories exist before execution."""
        os.makedirs(self.cfg.paths.experiment_dir, exist_ok=True)
        os.makedirs(self.cfg.paths.submission_dir, exist_ok=True)
        os.makedirs(self.cfg.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.paths.comparison_dir, exist_ok=True)
        os.makedirs(self.cfg.paths.analysis_dir, exist_ok=True)
        os.makedirs(self.cfg.paths.kfold_dir, exist_ok=True)
        os.makedirs(self.cfg.paths.kfold_folds_dir, exist_ok=True)

    def _validate_input_paths(self, pipeline_name: str) -> None:
        """Validate required input paths for the selected pipeline.

        Args:
            pipeline_name: Name of the pipeline to be executed.

        Raises:
            FileNotFoundError: If a required file or directory is missing.
        """
        pipeline_required_paths = {
            "prepare_data": [
                self.cfg.paths.train_image_dir,
                self.cfg.paths.test_image_dir,
                self.cfg.paths.train_label_file,
                self.cfg.paths.sample_submission_file,
            ],
            "train_baseline": [
                self.cfg.paths.train_image_dir,
                self.cfg.paths.train_label_file,
            ],
            "predict_submission": [
                self.cfg.paths.test_image_dir,
                self.cfg.paths.best_model_file,
            ],
            "compare_baselines": [],
            "analyze_validation": [
                self.cfg.paths.train_image_dir,
                self.cfg.paths.train_label_file,
                self.cfg.paths.validation_predictions_file,
            ],
            "run_kfold_evaluation": [
                self.cfg.paths.train_image_dir,
                self.cfg.paths.train_label_file,
            ],
        }
        required_paths = pipeline_required_paths.get(pipeline_name, [])

        missing_paths = [path for path in required_paths if not os.path.exists(path)]
        if missing_paths:
            raise FileNotFoundError(
                f"Missing required input path. First missing path: {missing_paths[0]}"
            )

    def _prepare_data_overview(self) -> None:
        """Build sample objects, validate splits, and log basic dataset facts."""
        dataset_builder = SpectrogramDatasetBuilder(cfg=self.cfg, logger=self.logger)
        splitter = TrainValidationSplitter(cfg=self.cfg, logger=self.logger)
        transform_factory = TransformFactory(cfg=self.cfg)

        train_samples = dataset_builder.load_train_samples()
        test_samples = dataset_builder.load_test_samples()
        dataset_builder.validate_train_samples(train_samples)

        train_split, valid_split = splitter.split_holdout(train_samples)
        train_dataset = SpectrogramDataset(
            samples=train_split,
            transform=transform_factory.build_train_transform(),
            image_mode=self._resolve_pil_image_mode(),
            logger=self.logger,
        )
        valid_dataset = SpectrogramDataset(
            samples=valid_split,
            transform=transform_factory.build_eval_transform(),
            image_mode=self._resolve_pil_image_mode(),
            logger=self.logger,
        )

        label_distribution = self._compute_label_distribution(train_samples)
        first_batch_sample = train_dataset[0]

        self.logger.info("Project: %s", self.cfg.project.name)
        self.logger.info("Configured model baseline: %s", self.cfg.model.name)
        self.logger.info("Augmentation preset: %s", self.cfg.aug.name)
        self.logger.info("Total train samples: %s", len(train_samples))
        self.logger.info("Total test samples: %s", len(test_samples))
        self.logger.info("Holdout train size: %s", len(train_dataset))
        self.logger.info("Holdout valid size: %s", len(valid_dataset))
        self.logger.info("Training label distribution: %s", label_distribution)
        self.logger.info(
            "First transformed sample tensor shape: %s",
            tuple(first_batch_sample["image"].shape),
        )

    def _run_training_pipeline(self) -> None:
        """Train the configured baseline model on the holdout split."""
        dataset_builder = SpectrogramDatasetBuilder(cfg=self.cfg, logger=self.logger)
        splitter = TrainValidationSplitter(cfg=self.cfg, logger=self.logger)
        transform_factory = TransformFactory(cfg=self.cfg)

        train_samples = dataset_builder.load_train_samples()
        dataset_builder.validate_train_samples(train_samples)
        train_split, valid_split = splitter.split_holdout(train_samples)

        train_dataset = SpectrogramDataset(
            samples=train_split,
            transform=transform_factory.build_train_transform(),
            image_mode=self._resolve_pil_image_mode(),
            logger=self.logger,
        )
        valid_dataset = SpectrogramDataset(
            samples=valid_split,
            transform=transform_factory.build_eval_transform(),
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

        device = self._resolve_device()
        model = ModelFactory(cfg=self.cfg, logger=self.logger).build_model()
        trainer = Trainer(
            cfg=self.cfg,
            model=model,
            device=device,
            logger=self.logger,
        )
        class_weights = self._build_class_weights(train_split, device)
        best_result = trainer.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            class_weights=class_weights,
        )

        self.logger.info("Training finished on device: %s", device)
        self.logger.info("Best epoch: %s", best_result["best_epoch"])
        self.logger.info(
            "Best validation accuracy: %.4f",
            best_result["best_valid_accuracy"],
        )
        self.logger.info(
            "Best validation macro F1: %.4f",
            best_result["best_valid_macro_f1"],
        )

    def _run_prediction_pipeline(self) -> None:
        """Load the best checkpoint and generate a Kaggle submission file."""
        if not os.path.exists(self.cfg.paths.best_model_file):
            raise FileNotFoundError(
                f"Best model file not found: {self.cfg.paths.best_model_file}"
            )

        dataset_builder = SpectrogramDatasetBuilder(cfg=self.cfg, logger=self.logger)
        transform_factory = TransformFactory(cfg=self.cfg)
        test_samples = dataset_builder.load_test_samples()
        test_dataset = SpectrogramDataset(
            samples=test_samples,
            transform=transform_factory.build_eval_transform(),
            image_mode=self._resolve_pil_image_mode(),
            logger=self.logger,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

        device = self._resolve_device()
        model = ModelFactory(cfg=self.cfg, logger=self.logger).build_model()
        checkpoint = torch.load(self.cfg.paths.best_model_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        predictions = Predictor(
            model=model,
            device=device,
            logger=self.logger,
        ).predict(test_loader)

        submission_file = os.path.join(
            self.cfg.paths.submission_dir,
            f"submission_{self.cfg.model.name}.csv",
        )
        SubmissionBuilder(logger=self.logger).build(
            samples=test_samples,
            predictions=predictions,
            output_file=submission_file,
        )

    def _run_baseline_comparison(self) -> None:
        """Compare saved validation summaries for the configured baseline models."""
        comparison_rows = BaselineComparator(
            cfg=self.cfg,
            logger=self.logger,
        ).compare(
            model_names=list(self.cfg.compare.baseline_models),
            aug_name=self.cfg.aug.name,
        )

        top_row = comparison_rows[0]
        self.logger.info(
            "Best baseline for aug=%s is %s with valid_macro_f1=%.4f and valid_accuracy=%.4f",
            self.cfg.aug.name,
            top_row["model_name"],
            top_row["valid_macro_f1"],
            top_row["valid_accuracy"],
        )

    def _run_validation_analysis_pipeline(self) -> None:
        """Analyze saved validation predictions for the current experiment."""
        summary_map = ValidationResultAnalyzer(
            cfg=self.cfg,
            logger=self.logger,
        ).analyze()
        self.logger.info(
            "Validation analysis summary | accuracy=%s | macro_f1=%s | errors=%s",
            summary_map["accuracy"],
            summary_map["macro_f1"],
            summary_map["num_errors"],
        )

    def _run_kfold_evaluation_pipeline(self) -> None:
        """Run stratified k-fold evaluation for the current configuration."""
        if self.cfg.train.strategy != "stratified_kfold":
            raise ValueError(
                "run_kfold_evaluation requires train.strategy=stratified_kfold."
            )

        summary_map = StratifiedKFoldRunner(
            cfg=self.cfg,
            device=self._resolve_device(),
            logger=self.logger,
        ).run()
        self.logger.info(
            "K-fold summary | mean_accuracy=%s | mean_macro_f1=%s | best_fold=%s",
            summary_map["mean_valid_accuracy"],
            summary_map["mean_valid_macro_f1"],
            summary_map["best_fold_id"],
        )

    def _resolve_device(self) -> torch.device:
        """Resolve the torch device from configuration and hardware.

        Returns:
            A torch device instance.
        """
        runtime_device = self.cfg.runtime.device
        if runtime_device != "auto":
            return torch.device(runtime_device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_pil_image_mode(self) -> str:
        """Resolve the Pillow image mode used by dataset loading.

        Returns:
            A Pillow image mode string such as ``"L"`` or ``"RGB"``.
        """
        configured_mode = str(self.cfg.data.get("image_mode", "rgb")).lower()
        if configured_mode in {"gray", "grayscale", "l"} or self.cfg.data.num_channels == 1:
            return "L"
        return "RGB"

    def _build_class_weights(
        self,
        samples: List[SpectrogramSample],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build inverse-frequency class weights from training samples.

        Args:
            samples: Training split samples.
            device: Target torch device.

        Returns:
            Class weights tensor or ``None`` when disabled.
        """
        if not self.cfg.train.use_class_weights:
            return None

        label_distribution = self._compute_label_distribution(samples)
        total_count = sum(label_distribution.values())
        class_weights = []
        for class_id in range(self.cfg.data.num_classes):
            class_count = label_distribution.get(class_id, 1)
            class_weights.append(total_count / (self.cfg.data.num_classes * class_count))

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.logger.info("Using class weights: %s", class_weights)
        return weight_tensor

    def _compute_label_distribution(self, samples: Any) -> Dict[int, int]:
        """Compute class counts from labeled samples.

        Args:
            samples: Sequence of labeled samples.

        Returns:
            Mapping from class id to sample count.
        """
        counter = Counter(sample.target for sample in samples)
        return dict(sorted(counter.items()))


def main(cfg: DictConfig) -> None:
    """Run the project pipeline through the manager class.

    Args:
        cfg: Project configuration object.
    """
    manager = PipelineManager(cfg=cfg)
    manager.run()
