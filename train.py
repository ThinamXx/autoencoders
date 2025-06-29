import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm

from autoencoder import AutoEncoder, ModelConfig
from dataloader import MNISTDataLoader, DataLoaderConfig


@dataclass
class Config:
    model_config: ModelConfig
    dataloader_config: DataLoaderConfig
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    noise_level: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_ae(config: Config):
    # 1. Initialize the configurations:
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    noise_level = config.noise_level
    device = config.device

    print(f"Training on device: {device}")
    print(f"Configuration: {config}")

    # 2. Initialize the dataloader:
    dataloader = MNISTDataLoader(config.dataloader_config)
    train_loader, test_loader = dataloader.get_data_loaders()

    # 8x throughput if GPU supports TF32.
    torch.set_float32_matmul_precision("high")

    # 3. Initialize the model:
    model = AutoEncoder(config.model_config)
    model.to(device)

    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # 4. Initialize the optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Initialize the loss function:
    criterion = nn.MSELoss()

    # Saving the checkpoints and logs:
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Training Configuration:\n{config}\n\n")

    # 6. Training the model:
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)

            # Add noise to the image for training (denoising autoencoder):
            noise = (
                torch.bernoulli((1 - noise_level) * torch.ones_like(images)) * 2
            ) - 1
            noisy_images = images * noise

            # Forward pass.
            recon_images, _ = model(noisy_images)
            loss = criterion(recon_images, images)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Calculate average loss for the epoch.
        avg_train_loss = train_loss / len(train_loader)

        epoch_log = f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_train_loss:.4f}"
        print(epoch_log)

        with open(log_file, "a") as f:
            f.write(f"{epoch_log}\n")

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(log_dir, f"autoencoder_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training completed!")

    # 7. Save final model.
    final_model_path = os.path.join(log_dir, "autoencoder_final.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config}, final_model_path
    )
    print(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
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

    train_ae(config)
