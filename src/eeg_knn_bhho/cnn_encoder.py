# src/eeg_knn_bhho/cnn_encoder.py
"""
CNN-based encoder for EEG epochs.

Implements a simple 1D‐CNN that takes raw EEG epochs of shape
(n_channels, n_samples) and outputs a fixed‐length embedding.

Includes:
 - EEGCNNEncoder: a PyTorch nn.Module
 - CNNEncoderTransformer: sklearn-style transformer that wraps EEGCNNEncoder

Usage:
    transformer = CNNEncoderTransformer(
        input_channels=n_channels,
        input_samples=n_samples,
        embedding_dim=128,
        hidden_channels=[32, 64],
        kernel_sizes=[7, 5],
        pool_sizes=[2, 2],
        device='cpu'
    )
    embeddings = transformer.transform(X_epochs)  # X_epochs: np.ndarray of shape (n_epochs, n_channels, n_samples)
    # embeddings.shape == (n_epochs, embedding_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


class EEGCNNEncoder(nn.Module):
    """
    1D‐CNN encoder for EEG epochs.

    Architecture:

      Input: (batch_size, input_channels, input_samples)

      For i in range(len(hidden_channels)):
        - Conv1d(in_channels=prev_channels, out_channels=hidden_channels[i], kernel_size=kernel_sizes[i], padding='same')
        - BatchNorm1d(hidden_channels[i])
        - ReLU
        - MaxPool1d(pool_sizes[i])

      After final conv block, flatten and apply a fully‐connected layer to get `embedding_dim`.

    Parameters
    ----------
    input_channels : int
        Number of EEG channels (e.g., 2, 16, 32).
    input_samples : int
        Number of time‐samples per epoch (e.g., 128, 256).
    hidden_channels : List[int]
        List of output channels for each Conv1d block (e.g., [32, 64]).
    kernel_sizes : List[int]
        Kernel sizes for each Conv1d block (e.g., [7, 5]).
    pool_sizes : List[int]
        Pool sizes for MaxPool1d after each block (e.g., [2, 2]).
    embedding_dim : int
        Dimension of the final embedding vector (e.g., 128).
    """

    def __init__(
        self,
        input_channels: int,
        input_samples: int,
        hidden_channels: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[int],
        embedding_dim: int
    ):
        super().__init__()

        assert len(hidden_channels) == len(kernel_sizes) == len(pool_sizes), (
            "hidden_channels, kernel_sizes, pool_sizes must have the same length"
        )

        layers = []
        prev_channels = input_channels
        curr_samples = input_samples

        # Build convolutional blocks
        for out_ch, k_sz, p_sz in zip(hidden_channels, kernel_sizes, pool_sizes):
            # Conv1d with padding='same' to preserve length; use padding=(kernel_size//2)
            pad = k_sz // 2
            layers.append(nn.Conv1d(prev_channels, out_ch, kernel_size=k_sz, padding=pad))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=p_sz))
            # After pooling, update sample count:
            curr_samples = (curr_samples + 0) // p_sz  # integer division
            prev_channels = out_ch

        self.conv = nn.Sequential(*layers)

        # Final linear layer: flatten (prev_channels * curr_samples) → embedding_dim
        self.flatten_dim = prev_channels * curr_samples
        self.embedding = nn.Linear(self.flatten_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_channels, input_samples)

        Returns
        -------
        emb : torch.Tensor, shape (batch_size, embedding_dim)
        """
        # x → conv blocks
        z = self.conv(x)               # shape: (batch_size, last_hidden_ch, reduced_samples)
        z = z.view(z.size(0), -1)      # shape: (batch_size, flatten_dim)
        emb = self.embedding(z)        # shape: (batch_size, embedding_dim)
        return emb


class CNNEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    scikit-learn transformer that wraps EEGCNNEncoder.

    Transforms a NumPy array of shape (n_epochs, n_channels, n_samples)
    into embeddings of shape (n_epochs, embedding_dim).

    Parameters
    ----------
    input_channels : int
        Number of EEG channels per epoch.
    input_samples : int
        Number of time‐samples per epoch.
    hidden_channels : List[int]
        Hidden channels for each Conv1d block.
    kernel_sizes : List[int]
        Kernel sizes for each Conv1d block.
    pool_sizes : List[int]
        Pool sizes for MaxPool1d after each block.
    embedding_dim : int
        Dimensionality of output embedding.
    device : str, default='cpu'
        Torch device to run on ('cpu' or 'cuda' if available).
    batch_size : int, default=32
        Batch size when processing multiple epochs.
    """

    def __init__(
        self,
        input_channels: int,
        input_samples: int,
        hidden_channels: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[int],
        embedding_dim: int,
        device: str = "cpu",
        batch_size: int = 32
    ):
        self.input_channels = input_channels
        self.input_samples = input_samples
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Initialize the PyTorch model
        self._model = EEGCNNEncoder(
            input_channels=input_channels,
            input_samples=input_samples,
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            pool_sizes=pool_sizes,
            embedding_dim=embedding_dim
        ).to(self.device)

        # Ensure eval mode (no dropout/batchnorm updates)
        self._model.eval()

    def fit(self, X: np.ndarray, y=None):
        """
        No training here; the encoder is used as a fixed feature extractor.
        If you want to train the CNN, you would override this to include a training loop.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the CNN encoder to each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_samples)

        Returns
        -------
        embeddings : np.ndarray, shape (n_epochs, embedding_dim)
        """
        n_epochs, n_ch, n_samp = X.shape
        assert n_ch == self.input_channels, f"Expected {self.input_channels} channels, got {n_ch}"
        assert n_samp == self.input_samples, f"Expected {self.input_samples} samples, got {n_samp}"

        embeddings = []
        with torch.no_grad():
            for start in range(0, n_epochs, self.batch_size):
                end = min(start + self.batch_size, n_epochs)
                batch = X[start:end]  # shape: (batch_size, n_ch, n_samp)
                # Convert to torch.Tensor on correct device
                batch_t = torch.from_numpy(batch.astype(np.float32)).to(self.device)
                # Forward pass
                emb_t = self._model(batch_t)           # shape: (batch_size, embedding_dim)
                emb_np = emb_t.cpu().numpy()           # shape: (batch_size, embedding_dim)
                embeddings.append(emb_np)

        return np.vstack(embeddings)  # shape: (n_epochs, embedding_dim)
