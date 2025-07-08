# Autoencoders - PyTorch Implementation

## 🚀 Features

- **Standard Autoencoder**: Basic encoder-decoder architecture for dimensionality reduction.
- **Variational Autoencoder (VAE)**: Probabilistic autoencoder with reparameterization trick.

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration).

## 🎯 Quick Start

### Training a Standard Autoencoder

```bash
python main.py train --config configs/autoencoder.yaml
```

### Training a Variational Autoencoder

```bash
python main.py train --config configs/vae.yaml
```

### Evaluating a Trained Model

```bash
python main.py evaluate --config configs/autoencoder.yaml --model logs/autoencoder_final.pth
```

## 🔬 Models [WIP]

### Standard Autoencoder.

The standard autoencoder consists of:
- **Encoder**: Compresses input images to latent representations. 
- **Decoder**: Reconstructs images from latent representations.
- **Loss**: Mean Squared Error (MSE) between input and reconstruction.

### Variational Autoencoder (VAE).

The VAE extends the standard autoencoder with:
- **Probabilistic Encoder**: Outputs mean and log-variance of latent distribution.
- **Reparameterization Trick**: Enables gradient flow through stochastic sampling.
- **VAE Loss**: Reconstruction loss + KL divergence regularization.
- **Generative Capability**: Can generate new samples from the prior distribution.