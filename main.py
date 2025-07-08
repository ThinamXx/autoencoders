from src.models.autoencoder import ModelConfig
from src.data.dataloader import DataLoaderConfig
from src.training.train import train_ae, Config


def main():
    """Main entry point for training."""
    config = Config(
        model_config=ModelConfig(
            hidden_dim=[32, 64, 128, 256],
            latent_dim=128,
            in_channels=1,
            image_size=32,
        ),
        dataloader_config=DataLoaderConfig(
            batch_size=16, num_workers=4, image_size=32, download=True
        ),
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=10,
        noise_level=0.3,
    )

    print("Training standard autoencoder...")
    train_ae(config)


if __name__ == "__main__":
    main()
