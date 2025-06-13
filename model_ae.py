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
class Config:
    feat_dim: int = 32
    latent_dim: int = 32
    channels: int = 3


class Encoder(nn.Module):

    def __init__(self, config: Config):
        super(Encoder, self).__init__()
        self.config = config

        channels = self.config.channels
        feat_dim = self.config.feat_dim
        latent_dim = self.config.latent_dim

        # (b, 3, 32, 32) -> (b, 32, 16, 16)
        self.conv1 = nn.Conv2d(channels, feat_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(feat_dim)
        # (b, 32, 16, 16) -> (b, 64, 8, 8)
        self.conv2 = nn.Conv2d(
            feat_dim, 2 * feat_dim, kernel_size=3, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(2 * feat_dim)
        # (b, 64, 8, 8) -> (b, 64, 4, 4)
        self.conv3 = nn.Conv2d(
            2 * feat_dim, 4 * feat_dim, kernel_size=3, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(4 * feat_dim)

        # (b, 128, 4, 4) -> (b, 32, 1, 1)
        self.conv_out = nn.Conv2d(
            4 * feat_dim, latent_dim, kernel_size=4, stride=1, padding=0
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.conv_out(x)
        return x
