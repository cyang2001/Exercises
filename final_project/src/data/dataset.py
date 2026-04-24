"""Dataset definitions for the spectrogram classification task."""

import csv
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from src.utils.logger import get_logger


@dataclass
class SpectrogramSample:
    """Container for a single spectrogram sample."""

    image_id: int
    image_path: str
    target: Optional[int] = None


class SpectrogramDataset(Dataset):
    """PyTorch dataset for room occupancy spectrogram images."""

    def __init__(
        self,
        samples: List[SpectrogramSample],
        transform: Any = None,
        image_mode: str = "RGB",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            samples: List of spectrogram sample metadata.
            transform: Optional transform pipeline applied to each image.
            image_mode: Pillow image mode used when loading each sample.
            logger: Optional logger instance.
        """
        self.samples = samples
        self.transform = transform
        self.image_mode = image_mode
        self.logger = logger or get_logger(__name__)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Dataset size.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load a single sample by index.

        Args:
            index: Sample index.

        Returns:
            A dictionary containing image tensor, image id, and target.
        """
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert(self.image_mode)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_id": sample.image_id,
            "target": -1 if sample.target is None else sample.target,
        }


class SpectrogramDatasetBuilder:
    """Build dataset sample lists from the current project structure."""

    def __init__(self, cfg: Any, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the dataset builder.

        Args:
            cfg: Project configuration object.
            logger: Optional logger instance.
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        self.train_image_dir = cfg.paths.train_image_dir
        self.test_image_dir = cfg.paths.test_image_dir
        self.train_label_file = cfg.paths.train_label_file

    def load_train_samples(self) -> List[SpectrogramSample]:
        """Load all labeled training samples.

        Returns:
            A list of training samples with labels.
        """
        samples: List[SpectrogramSample] = []

        with open(self.train_label_file, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                image_id = int(row["id"])
                file_id = image_id + 1
                image_path = os.path.join(self.train_image_dir, f"img_{file_id}.png")

                samples.append(
                    SpectrogramSample(
                        image_id=image_id,
                        image_path=image_path,
                        target=int(row["target"]),
                    )
                )

        self.logger.info("Loaded %s training samples.", len(samples))
        return samples

    def load_test_samples(self) -> List[SpectrogramSample]:
        """Load all unlabeled test samples.

        Returns:
            A list of test samples without labels.
        """
        image_files = sorted(
            [
                file_name
                for file_name in os.listdir(self.test_image_dir)
                if file_name.endswith(".png")
            ],
            key=self._extract_file_id,
        )

        samples = [
            SpectrogramSample(
                image_id=self._extract_sample_id(file_name),
                image_path=os.path.join(self.test_image_dir, file_name),
            )
            for file_name in image_files
        ]

        self.logger.info("Loaded %s test samples.", len(samples))
        return samples

    def validate_train_samples(self, samples: List[SpectrogramSample]) -> None:
        """Validate that all training sample files exist.

        Args:
            samples: Training samples to validate.

        Raises:
            FileNotFoundError: If at least one image file is missing.
        """
        missing_files = [
            sample.image_path for sample in samples if not os.path.exists(sample.image_path)
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Missing {len(missing_files)} training image files. "
                f"First missing file: {missing_files[0]}"
            )

    def _extract_file_id(self, file_name: str) -> int:
        """Extract the numeric file id from an image file name.

        Args:
            file_name: Image file name such as ``img_42.png``.

        Returns:
            Parsed integer file id from the file name.
        """
        return int(file_name.replace("img_", "").replace(".png", ""))

    def _extract_sample_id(self, file_name: str) -> int:
        """Convert an image file name into the corresponding sample id.

        Args:
            file_name: Image file name such as ``img_42.png``.

        Returns:
            Sample id aligned with ``csv`` and submission files.
        """
        return self._extract_file_id(file_name) - 1
