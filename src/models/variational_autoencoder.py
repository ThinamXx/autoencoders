"""
Variational Autoencoder (VAE).

Key differences from standard autoencoder:
1. Encoder outputs mean and log-variance instead of direct latent code.
2. Uses reparameterization trick for sampling.
3. Loss includes reconstruction + KL divergence.
4. Can generate new samples from prior distribution.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    hidden_dim: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    latent_dim: int = 128
    in_channels: int = 1
    image_size: int = 32
    beta: float = 1.0  # KL divergence weight (β-VAE parameter).


class VAEEncoder(nn.Module):
    """
    VAE Encoder that outputs mean and log-variance for latent distribution.

    Instead of outputting a single latent vector, it outputs parameters (μ, log σ²) of a Gaussian distribution.
    """

    def __init__(self, config: VAEConfig):
        super(VAEEncoder, self).__init__()
        self.config = config

        modules = []
        in_channels = config.in_channels

        for i, hidden_dim in enumerate(config.hidden_dim):
            modules.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = hidden_dim

        self.conv_layers = nn.Sequential(*modules)

        # Calculate the size after convolutions: each conv layer
        # with stride=2 halves the spatial dimensions i.e. height and width.
        self.final_size = config.image_size // (2 ** len(config.hidden_dim))
        self.final_channels = config.hidden_dim[-1]
        self.flatten_size = self.final_channels * self.final_size * self.final_size

        # Two separate linear layers for mean and log-variance.
        self.fc_mean = nn.Linear(self.flatten_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and log-variance.

        Returns:
            mean: μ (batch_size, latent_dim)
            logvar: log σ² (batch_size, latent_dim)
        """
        x = self.conv_layers(x)

        # Flatten.
        x = x.view(x.size(0), -1)

        # Get mean and log-variance.
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder reconstructs images from latent samples.
    Similar to standard autoencoder decoder.
    """

    def __init__(self, config: VAEConfig):
        super(VAEDecoder, self).__init__()
        self.config = config

        self.final_size = config.image_size // (2 ** len(config.hidden_dim))
        self.final_channels = config.hidden_dim[-1]

        # Linear layer from latent space.
        self.fc = nn.Linear(
            config.latent_dim, self.final_channels * self.final_size * self.final_size
        )

        # Transpose convolutional layers.
        modules = []
        hidden_dims = config.hidden_dim[::-1]  # Reverse order.

        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                out_channels = config.in_channels
                modules.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels=hidden_dims[i],
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        ),
                        nn.Sigmoid(),  # For normalized images [0,1].
                    ]
                )
            else:
                out_channels = hidden_dims[i + 1]
                modules.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels=hidden_dims[i],
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )

        self.conv_layers = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent sample to reconstruction.

        Args:
            z: Latent sample (batch_size, latent_dim)

        Returns:
            reconstruction: (batch_size, channels, height, width)
        """
        # Project to feature maps.
        x = self.fc(z)

        # Reshape to feature maps.
        x = x.view(x.size(0), self.final_channels, self.final_size, self.final_size)

        # Apply transpose convolutions.
        x = self.conv_layers(x)

        return x


class VariationalAutoEncoder(nn.Module):
    """
    Complete Variational Autoencoder.

    Key innovations:
    1. Probabilistic latent space.
    2. Reparameterization trick.
    3. KL divergence regularization.
    4. Generative sampling capability.
    """

    def __init__(self, config: VAEConfig):
        super(VariationalAutoEncoder, self).__init__()
        self.config = config
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"VAE created with {total_params:,} parameters")

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)

        This allows gradients to flow through the sampling process.
        Instead of sampling z directly from N(μ, σ²), we sample ε from N(0,1)
        and transform it deterministically.

        Args:
            mean: μ (batch_size, latent_dim)
            logvar: log σ² (batch_size, latent_dim)

        Returns:
            z: Sampled latent vector (batch_size, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log σ²).
            eps = torch.randn_like(std)  # ε ~ N(0,1).
            return mean + eps * std  # z = μ + σ * ε.
        else:
            return mean

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Returns:
            reconstruction: Decoded output.
            mean: Encoder mean output.
            logvar: Encoder log-variance output.
        """
        mean, logvar = self.encoder(x)

        # Pass through reparameterization mentioned in 2.4 of the paper.
        z = self.reparameterize(mean, logvar)

        reconstruction = self.decoder(z)

        return reconstruction, mean, logvar

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample to reconstruction."""
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate new samples by sampling from prior N(0,I).

        This is the key generative capability of VAEs!
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from standard normal distribution.
        z = torch.randn(num_samples, self.config.latent_dim, device=device)

        # Decode to generate images.
        with torch.no_grad():
            samples = self.decode(z)

        return samples

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input (encode mean then decode)."""
        mean, _ = self.encode(x)
        return self.decode(mean)


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss function: Reconstruction Loss + β * KL Divergence

    Args:
        recon_x: Reconstructed images.
        x: Original images.
        mean: Encoder mean output.
        logvar: Encoder log-variance output.
        beta: Weight for KL divergence (β-VAE parameter).

    Returns:
        total_loss: Combined loss.
        recon_loss: Reconstruction loss.
        kl_loss: KL divergence loss.
    """
    # Reconstruction loss: use MSE for more flexibility with input ranges.
    # For images normalized to [0,1], you can use binary_cross_entropy.
    # For other ranges, MSE works better.
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # Alternative: For [0,1] normalized images, use binary cross entropy.
    # recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum').

    # KL divergence loss.
    # KL(q(z|x) || p(z)) where q(z|x) = N(μ, σ²) and p(z) = N(0, I).
    # KL = -0.5 * Σ(1 + log σ² - μ² - σ²).
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss.
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


class VAETrainer:
    """Training utilities for VAE."""

    def __init__(self, model: VariationalAutoEncoder, config: VAEConfig):
        self.model = model
        self.config = config

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute VAE loss and return metrics."""
        recon_x, mean, logvar = self.model(x)

        total_loss, recon_loss, kl_loss = vae_loss_function(
            recon_x, x, mean, logvar, self.config.beta
        )

        # Normalize by batch size.
        batch_size = x.size(0)
        metrics = {
            "total_loss": total_loss / batch_size,
            "recon_loss": recon_loss / batch_size,
            "kl_loss": kl_loss / batch_size,
            "beta": self.config.beta,
        }

        return total_loss / batch_size, metrics


if __name__ == "__main__":
    print("=== VAE vs Standard Autoencoder Comparison ===\n")

    vae_config = VAEConfig(
        hidden_dim=[32, 64, 128, 256],
        latent_dim=128,
        in_channels=1,
        image_size=32,
        beta=1.0,
    )

    vae = VariationalAutoEncoder(vae_config)

    batch_size = 4
    test_input_raw = torch.randn(batch_size, 1, 32, 32)
    test_input = torch.sigmoid(test_input_raw)  # Normalize to [0,1].

    print(f"Input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")

    with torch.no_grad():
        recon, mean, logvar = vae(test_input)
        print(f"Reconstruction shape: {recon.shape}")
        print(f"Reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
        print(f"Mean shape: {mean.shape}")
        print(f"Log-variance shape: {logvar.shape}")

        # Loss computation.
        trainer = VAETrainer(vae, vae_config)
        loss, metrics = trainer.compute_loss(test_input)
        print("\nLoss components:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Demonstrate different loss functions.
        print("\n=== Loss Function Comparison ===")
        mse_loss, _, _ = vae_loss_function(recon, test_input, mean, logvar)
        print(f"MSE-based loss: {mse_loss/batch_size:.4f}")

        # For binary cross entropy, both inputs need to be in [0,1].
        bce_loss = F.binary_cross_entropy(recon, test_input, reduction="sum")
        print(f"BCE-based loss: {bce_loss/batch_size:.4f}")

        # Generate new samples.
        print("\n=== Generative Sampling ===")
        generated_samples = vae.sample(num_samples=8)
        print(f"Generated samples shape: {generated_samples.shape}")
        print(
            f"Generated range: [{generated_samples.min():.3f}, {generated_samples.max():.3f}]"
        )
