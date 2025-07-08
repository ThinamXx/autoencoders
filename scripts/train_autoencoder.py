#!/usr/bin/env python3

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.models.autoencoder import ModelConfig
from src.data.dataloader import DataLoaderConfig
from src.training.train import train_ae, Config


def main():
    """Main training function."""
    config_path = "configs/autoencoder.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print(f"Loading configuration from: {config_path}")
    
    yaml_config = load_config(config_path)
    
    training_config = Config(
        model_config=ModelConfig(
            hidden_dim=yaml_config.model.hidden_dim,
            latent_dim=yaml_config.model.latent_dim,
            in_channels=yaml_config.model.in_channels,
            image_size=yaml_config.model.image_size
        ),
        dataloader_config=DataLoaderConfig(
            batch_size=yaml_config.data.batch_size,
            num_workers=yaml_config.data.num_workers,
            image_size=yaml_config.data.image_size,
            normalize_mean=yaml_config.data.normalize_mean,
            normalize_std=yaml_config.data.normalize_std,
            download=yaml_config.data.download,
            data_root=yaml_config.data.data_root
        ),
        learning_rate=yaml_config.training.learning_rate,
        num_epochs=yaml_config.training.num_epochs,
        noise_level=yaml_config.training.noise_level,
        device=yaml_config.training.device if yaml_config.training.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    train_ae(training_config)


if __name__ == "__main__":
    main() 