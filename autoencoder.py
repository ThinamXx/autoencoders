import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    hidden_dim: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    latent_dim: int = 128
    in_channels: int = 1
    image_size: int = 32


class Encoder(nn.Module):
    """Encoder that compresses input to latent representation."""

    def __init__(self, config: ModelConfig):
        super(Encoder, self).__init__()
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

        # Linear layer to project from feature maps to latent space.
        self.fc_out = nn.Linear(
            self.final_channels * self.final_size * self.final_size, config.latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)

        # Flatten for linear layer.
        x = x.view(x.size(0), -1)

        # Project to latent space.
        x = self.fc_out(x)

        return x


class Decoder(nn.Module):
    """Decoder that reconstructs input from latent representation."""

    def __init__(self, config: ModelConfig):
        super(Decoder, self).__init__()
        self.config = config

        # Calculate the size after convolutions same as encoder.
        self.final_size = config.image_size // (2 ** len(config.hidden_dim))
        self.final_channels = config.hidden_dim[-1]

        # Linear layer from latent space to feature maps.
        self.fc_in = nn.Linear(
            config.latent_dim, self.final_channels * self.final_size * self.final_size
        )

        modules = []
        hidden_dims = config.hidden_dim[::-1]

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
                        nn.Tanh(),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project from latent space to feature maps.
        x = self.fc_in(x)

        # Reshape to feature maps.
        x = x.view(x.size(0), self.final_channels, self.final_size, self.final_size)

        # Apply transpose convolutional layers.
        x = self.conv_layers(x)

        return x


class AutoEncoder(nn.Module):
    """Complete Autoencoder model combining Encoder and Decoder."""

    def __init__(self, config: ModelConfig):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Calculate total parameters.
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"AutoEncoder created with {total_params:,} parameters")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both latent representation and reconstruction."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(latent)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input (encode then decode)."""
        latent = self.encode(x)
        return self.decode(latent)

if __name__ == "__main__":
    config = ModelConfig(
        hidden_dim=[32, 64, 128, 256], latent_dim=128, in_channels=1, image_size=32
    )

    model = AutoEncoder(config)

    batch_size = 4
    test_input = torch.randn(
        batch_size, config.in_channels, config.image_size, config.image_size
    )

    print(f"Input shape: {test_input.shape}")

    with torch.no_grad():
        reconstruction, latent = model(test_input)
        print(f"Latent shape: {latent.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")