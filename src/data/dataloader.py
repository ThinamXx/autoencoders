"""
MNIST Data Loader Module.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    batch_size: int = 64
    num_workers: int = 4
    train_data: bool = True
    shuffle_train: bool = True
    shuffle_test: bool = False
    data_root: Union[str, Path] = "./data"
    image_size: int = 32
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    download: bool = True


class MNISTDataLoader:
    """
    MNIST Data Loader.

    Example:
        >>> config = DataLoaderConfig(batch_size=128, num_workers=8)
        >>> data_loader = MNISTDataLoader(config)
        >>> train_loader, test_loader = data_loader.get_data_loaders()
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        self.config = config or DataLoaderConfig()
        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None

        logger.info(f"Initialized MNISTDataLoader with config: {self.config}")

    def _create_transforms(self) -> Compose:
        """
        Create the transformation pipeline for MNIST images.

        Returns:
            Composed transformation pipeline.
        """
        transform_list = [
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[self.config.normalize_mean], std=[self.config.normalize_std]
            ),
        ]

        return transforms.Compose(transform_list)

    def _ensure_data_directory(self) -> None:
        try:
            Path(self.config.data_root).mkdir(parents=True, exist_ok=True)
            logger.info(f"Data directory ensured at: {self.config.data_root}")

        except OSError as e:
            logger.error(f"Failed to create data directory: {e}")
            raise

    def _load_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Load MNIST train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset).

        Raises:
            RuntimeError: If dataset loading fails.
        """
        try:
            self._ensure_data_directory()
            transform = self._create_transforms()

            logger.info("Loading MNIST datasets...")

            train_dataset = datasets.MNIST(
                root=str(self.config.data_root),
                train=self.config.train_data,
                transform=transform,
                download=self.config.download,
            )

            test_dataset = datasets.MNIST(
                root=str(self.config.data_root),
                train=False,
                transform=transform,
                download=self.config.download,
            )

            logger.info(
                f"Loaded {len(train_dataset)} training samples and "
                f"{len(test_dataset)} test samples"
            )

            return train_dataset, test_dataset

        except Exception as e:
            logger.error(f"Failed to load MNIST datasets: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}") from e

    def _create_data_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        try:
            if self._train_dataset is None or self._test_dataset is None:
                self._train_dataset, self._test_dataset = self._load_datasets()

            train_loader = self._create_data_loader(
                self._train_dataset,
                shuffle=self.config.shuffle_train,
            )

            test_loader = self._create_data_loader(
                self._test_dataset,
                shuffle=self.config.shuffle_test,
            )

            logger.info(
                f"Created data loaders - Train batches: {len(train_loader)}, "
                f"Test batches: {len(test_loader)}"
            )

            return train_loader, test_loader

        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            raise RuntimeError(f"Data loader creation failed: {e}") from e
