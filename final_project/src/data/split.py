"""Data split helpers for training and validation."""

import logging
from typing import Any, List, Sequence, Tuple

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.data.dataset import SpectrogramSample
from src.utils.logger import get_logger


class TrainValidationSplitter:
    """Create stratified holdout or k-fold splits for training data."""

    def __init__(self, cfg: Any, logger: logging.Logger = None) -> None:
        """Initialize the splitter.

        Args:
            cfg: Project configuration object.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        self.valid_size = cfg.data.valid_size
        self.random_seed = cfg.project.random_seed

    def split_holdout(
        self,
        samples: Sequence[SpectrogramSample],
    ) -> Tuple[List[SpectrogramSample], List[SpectrogramSample]]:
        """Create one stratified train/validation split.

        Args:
            samples: Full labeled training sample sequence.

        Returns:
            A tuple containing train samples and validation samples.
        """
        labels = [sample.target for sample in samples]
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.valid_size,
            random_state=self.random_seed,
        )

        train_indices, valid_indices = next(splitter.split(samples, labels))
        train_samples = self._gather_samples(samples, train_indices)
        valid_samples = self._gather_samples(samples, valid_indices)

        self.logger.info(
            "Created holdout split with %s training samples and %s validation samples.",
            len(train_samples),
            len(valid_samples),
        )
        return train_samples, valid_samples

    def build_kfold_splits(
        self,
        samples: Sequence[SpectrogramSample],
        n_splits: int,
    ) -> List[Tuple[List[SpectrogramSample], List[SpectrogramSample]]]:
        """Create stratified k-fold splits.

        Args:
            samples: Full labeled training sample sequence.
            n_splits: Number of folds to generate.

        Returns:
            A list of train/validation sample tuples.
        """
        labels = [sample.target for sample in samples]
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_seed,
        )

        fold_splits = []
        for train_indices, valid_indices in splitter.split(samples, labels):
            fold_splits.append(
                (
                    self._gather_samples(samples, train_indices),
                    self._gather_samples(samples, valid_indices),
                )
            )

        self.logger.info("Created %s stratified k-fold splits.", len(fold_splits))
        return fold_splits

    def _gather_samples(
        self,
        samples: Sequence[SpectrogramSample],
        indices: Sequence[int],
    ) -> List[SpectrogramSample]:
        """Gather samples based on a sequence of indices.

        Args:
            samples: Complete sample sequence.
            indices: Indices to extract.

        Returns:
            Selected samples in index order.
        """
        return [samples[index] for index in indices]
