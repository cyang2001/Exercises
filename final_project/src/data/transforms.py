"""Transform builders for spectrogram images."""

import random
from typing import Any

import torch
from torchvision import transforms


class RandomTimeMask:
    """Mask a random continuous band on the time axis."""

    def __init__(self, prob: float, max_width: int) -> None:
        """Initialize the time mask transform.

        Args:
            prob: Probability of applying the transform.
            max_width: Maximum masked width in pixels.
        """
        self.prob = prob
        self.max_width = max_width

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply time masking to a tensor image.

        Args:
            tensor: Input tensor with shape ``(channels, height, width)``.

        Returns:
            The augmented tensor.
        """
        if random.random() > self.prob or self.max_width <= 0:
            return tensor

        width = tensor.shape[2]
        mask_width = min(random.randint(1, self.max_width), width)
        start = random.randint(0, max(width - mask_width, 0))
        tensor[:, :, start : start + mask_width] = 0.0
        return tensor


class RandomFrequencyMask:
    """Mask a random continuous band on the frequency axis."""

    def __init__(self, prob: float, max_height: int) -> None:
        """Initialize the frequency mask transform.

        Args:
            prob: Probability of applying the transform.
            max_height: Maximum masked height in pixels.
        """
        self.prob = prob
        self.max_height = max_height

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to a tensor image.

        Args:
            tensor: Input tensor with shape ``(channels, height, width)``.

        Returns:
            The augmented tensor.
        """
        if random.random() > self.prob or self.max_height <= 0:
            return tensor

        height = tensor.shape[1]
        mask_height = min(random.randint(1, self.max_height), height)
        start = random.randint(0, max(height - mask_height, 0))
        tensor[:, start : start + mask_height, :] = 0.0
        return tensor


class RandomAmplitudeScaling:
    """Scale spectrogram intensity by a random multiplicative factor."""

    def __init__(self, prob: float, min_scale: float, max_scale: float) -> None:
        """Initialize the amplitude scaling transform.

        Args:
            prob: Probability of applying the transform.
            min_scale: Lower bound of the scaling factor.
            max_scale: Upper bound of the scaling factor.
        """
        self.prob = prob
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply amplitude scaling to a tensor image.

        Args:
            tensor: Input tensor with shape ``(channels, height, width)``.

        Returns:
            The augmented tensor.
        """
        if random.random() > self.prob:
            return tensor

        scale = random.uniform(self.min_scale, self.max_scale)
        return torch.clamp(tensor * scale, min=0.0, max=1.0)


class RandomGaussianNoise:
    """Add Gaussian noise to a tensor spectrogram."""

    def __init__(self, prob: float, std: float) -> None:
        """Initialize the Gaussian noise transform.

        Args:
            prob: Probability of applying the transform.
            std: Standard deviation of the Gaussian noise.
        """
        self.prob = prob
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to a tensor image.

        Args:
            tensor: Input tensor with shape ``(channels, height, width)``.

        Returns:
            The augmented tensor.
        """
        if random.random() > self.prob or self.std <= 0:
            return tensor

        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, min=0.0, max=1.0)


class TransformFactory:
    """Build torchvision transforms from the current configuration."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the transform factory.

        Args:
            cfg: Project configuration object.
        """
        self.cfg = cfg
        self.image_size = cfg.data.image_size
        self.num_channels = cfg.data.num_channels
        self.aug_cfg = cfg.aug

    def build_train_transform(self) -> transforms.Compose:
        """Build the transform pipeline for training samples.

        Returns:
            A composed torchvision transform for training.
        """
        transform_steps = [
            transforms.Resize((self.image_size, self.image_size)),
        ]

        if self._get_aug_value("horizontal_flip_prob", 0.0) > 0:
            transform_steps.append(
                transforms.RandomHorizontalFlip(
                    p=self._get_aug_value("horizontal_flip_prob", 0.0),
                )
            )

        if self._get_aug_value("vertical_flip_prob", 0.0) > 0:
            transform_steps.append(
                transforms.RandomVerticalFlip(
                    p=self._get_aug_value("vertical_flip_prob", 0.0),
                )
            )

        if self._get_aug_value("color_jitter_prob", 0.0) > 0:
            transform_steps.append(
                transforms.RandomApply(
                    [
                        self._build_color_jitter()
                    ],
                    p=self._get_aug_value("color_jitter_prob", 0.0),
                )
            )

        transform_steps.extend(
            [
                transforms.ToTensor(),
                RandomTimeMask(
                    prob=self._get_aug_value("time_mask_prob", 0.0),
                    max_width=self._get_aug_value("time_mask_max_width", 0),
                ),
                RandomFrequencyMask(
                    prob=self._get_aug_value("frequency_mask_prob", 0.0),
                    max_height=self._get_aug_value("frequency_mask_max_height", 0),
                ),
                RandomAmplitudeScaling(
                    prob=self._get_aug_value("amplitude_scale_prob", 0.0),
                    min_scale=self._get_aug_value("amplitude_scale_min", 1.0),
                    max_scale=self._get_aug_value("amplitude_scale_max", 1.0),
                ),
                RandomGaussianNoise(
                    prob=self._get_aug_value("gaussian_noise_prob", 0.0),
                    std=self._get_aug_value("gaussian_noise_std", 0.0),
                ),
            ]
        )

        return transforms.Compose(transform_steps)

    def build_eval_transform(self) -> transforms.Compose:
        """Build the transform pipeline for validation or test samples.

        Returns:
            A composed torchvision transform for evaluation.
        """
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def _get_aug_value(self, key: str, default_value: Any) -> Any:
        """Safely retrieve an augmentation parameter from the config.

        Args:
            key: Augmentation config key.
            default_value: Fallback value when the key is missing.

        Returns:
            The configured value or the provided default.
        """
        return self.aug_cfg.get(key, default_value)

    def _build_color_jitter(self) -> transforms.ColorJitter:
        """Build a color jitter transform compatible with the current channel count.

        Returns:
            A torchvision color jitter transform.
        """
        if self.num_channels == 1:
            return transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            )

        return transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        )
