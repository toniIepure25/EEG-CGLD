# tests/test_connectivity.py
import numpy as np
from eeg_knn_bhho.connectivity import ConnectivityExtractor

def test_connectivity_extractor_shape():
    n_epochs, n_ch, n_samp = 5, 3, 128
    rng = np.random.RandomState(0)
    X = rng.randn(n_epochs, n_ch, n_samp)
    ext = ConnectivityExtractor(sfreq=128.0, nperseg=64, n_jobs=1)
    feats = ext.transform(X)
    n_pairs = n_ch * (n_ch - 1) // 2
    assert feats.shape == (n_epochs, n_pairs)
