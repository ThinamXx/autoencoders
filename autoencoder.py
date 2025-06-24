"""
Autoencoder Model.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    feat_dim: int = 32
    latent_dim: int = 128
    channels: int = 1


class Encoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super(Encoder, self).__init__()
        self.config = config

        channels = self.config.channels
        feat_dim = self.config.feat_dim
        latent_dim = self.config.latent_dim

        # (b, 3, 32, 32) -> (b, 32, 16, 16)
        # (b, c, h, w) -> (b, f_dim, h/2, w/2)
        self.conv1 = nn.Conv2d(channels, feat_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(feat_dim)

        # (b, 32, 16, 16) -> (b, 64, 8, 8)
        # (b, f_dim, h/2, w/2) -> (b, 2*f_dim, h/4, w/4)
        self.conv2 = nn.Conv2d(
            feat_dim, 2 * feat_dim, kernel_size=3, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(2 * feat_dim)

        # (b, 64, 8, 8) -> (b, 128, 4, 4)
        # (b, 2*f_dim, h/4, w/4) -> (b, 4*f_dim, h/8, w/8)
        self.conv3 = nn.Conv2d(
            2 * feat_dim, 4 * feat_dim, kernel_size=3, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(4 * feat_dim)

        # (b, 128, 4, 4) -> (b, 64, 1, 1)
        # (b, 4*f_dim, h/8, w/8) -> (b, l_dim, 1, 1)
        self.conv_out = nn.Conv2d(
            4 * feat_dim, latent_dim, kernel_size=4, stride=1, padding=0
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.conv_out(x)
        return x


class Decoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super(Decoder, self).__init__()
        self.config = config

        channels = self.config.channels
        feat_dim = self.config.feat_dim
        latent_dim = self.config.latent_dim

        # (b, 64, 1, 1) -> (b, 128, 4, 4)
        # (b, l_dim, 1, 1) -> (b, 4*f_dim, 4, 4)
        self.conv1 = nn.ConvTranspose2d(
            latent_dim,
            4 * feat_dim,
            kernel_size=4,
            stride=1,
            padding=0,
            output_padding=0,
        )
        self.bn1 = nn.BatchNorm2d(4 * feat_dim)

        # (b, 128, 4, 4) -> (b, 64, 8, 8)
        # (b, 4*f_dim, 4, 4) -> (b, 2*f_dim, 8, 8)
        self.conv2 = nn.ConvTranspose2d(
            4 * feat_dim,
            2 * feat_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn2 = nn.BatchNorm2d(2 * feat_dim)

        # (b, 64, 8, 8) -> (b, 32, 16, 16)
        # (b, 2*f_dim, 8, 8) -> (b, f_dim, 16, 16)
        self.conv3 = nn.ConvTranspose2d(
            2 * feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(feat_dim)

        # (b, 32, 16, 16) -> (b, 32, 32, 32)
        # (b, f_dim, 16, 16) -> (b, f_dim, 32, 32)
        self.conv4 = nn.ConvTranspose2d(
            feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(feat_dim)

        # (b, 32, 32, 32) -> (b, 3, 32, 32)
        # (b, f_dim, 32, 32) -> (b, c, 32, 32)
        self.conv_out = nn.ConvTranspose2d(
            feat_dim, channels, kernel_size=3, stride=1, padding=1, output_padding=0
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.tanh(self.conv_out(x))
        return x


class AutoEncoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        encoding = self.encoder(x)
        x_hat = self.decoder(encoding)
        return x_hat, encoding
