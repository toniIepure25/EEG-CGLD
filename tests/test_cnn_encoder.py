# tests/test_cnn_encoder.py
import numpy as np
import torch
from eeg_knn_bhho.cnn_encoder import EEGCNNEncoder, CNNEncoderTransformer

def test_eeg_cnn_encoder_forward():
    # Input: batch_size=2, channels=3, samples=128
    batch_size, n_ch, n_samp = 2, 3, 128
    hidden_channels = [16, 32]
    kernel_sizes = [7, 5]
    pool_sizes = [2, 2]
    embedding_dim = 64

    model = EEGCNNEncoder(
        input_channels=n_ch,
        input_samples=n_samp,
        hidden_channels=hidden_channels,
        kernel_sizes=kernel_sizes,
        pool_sizes=pool_sizes,
        embedding_dim=embedding_dim
    )
    model.eval()

    x = torch.randn(batch_size, n_ch, n_samp, dtype=torch.float32)
    with torch.no_grad():
        emb = model(x)

    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (batch_size, embedding_dim)

def test_cnn_encoder_transformer_shape():
    # Create random data: 5 epochs, 3 channels, 128 samples each
    n_epochs, n_ch, n_samp = 5, 3, 128
    X = np.random.randn(n_epochs, n_ch, n_samp).astype(np.float32)

    transformer = CNNEncoderTransformer(
        input_channels=n_ch,
        input_samples=n_samp,
        hidden_channels=[16, 32],
        kernel_sizes=[7, 5],
        pool_sizes=[2, 2],
        embedding_dim=64,
        device="cpu",
        batch_size=2
    )
    embeddings = transformer.transform(X)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (n_epochs, 64)
