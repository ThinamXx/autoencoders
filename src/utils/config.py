"""
Configuration utilities for loading YAML config files.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union, List


@dataclass
class ConfigBase:
    """Base configuration class."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class ModelConfig(ConfigBase):
    """Model configuration."""

    type: str
    hidden_dim: List[int]
    latent_dim: int
    in_channels: int
    image_size: int
    beta: float = 1.0  # For VAE.


@dataclass
class DataConfig(ConfigBase):
    """Data configuration."""

    dataset: str
    batch_size: int
    num_workers: int
    image_size: int
    normalize_mean: float
    normalize_std: float
    download: bool
    data_root: str


@dataclass
class TrainingConfig(ConfigBase):
    """Training configuration."""

    learning_rate: float
    num_epochs: int
    device: str
    save_interval: int
    log_dir: str
    noise_level: float = 0.0  # For denoising autoencoder.


@dataclass
class OptimizerConfig(ConfigBase):
    """Optimizer configuration."""

    type: str
    weight_decay: float = 0.0


@dataclass
class LossConfig(ConfigBase):
    """Loss configuration."""

    type: str


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    loss: LossConfig

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig.from_dict(config_dict["model"]),
            data=DataConfig.from_dict(config_dict["data"]),
            training=TrainingConfig.from_dict(config_dict["training"]),
            optimizer=OptimizerConfig.from_dict(config_dict["optimizer"]),
            loss=LossConfig.from_dict(config_dict["loss"]),
        )

    def to_yaml(self, output_path: Union[str, Path]):
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "optimizer": self.optimizer.__dict__,
            "loss": self.loss.__dict__,
        }

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(config_path)
