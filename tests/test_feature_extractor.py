# tests/test_feature_extractor.py
import numpy as np
from eeg_knn_bhho.features import CiSSALightFeatureExtractor

def test_feature_extraction_shapes():
    n_epochs = 3
    n_comps = 2        # e.g., 2 IMFs per epoch
    n_samples = 64     # 64 samples per IMF
    # Create random data
    rng = np.random.RandomState(0)
    X_cissa = rng.randn(n_epochs, n_comps, n_samples)
    # Define two bands: 1–4 Hz and 4–8 Hz
    bands = [(1, 4), (4, 8)]
    extractor = CiSSALightFeatureExtractor(sfreq=128.0, bands=bands, ds=2, nperseg=32, n_jobs=1)
    feats = extractor.transform(X_cissa)
    # Each IMF yields: len(bands)=2 band powers + 4 time-stats + 3 Hjorth + 3 entropies = 12 features
    assert feats.shape == (n_epochs, n_comps * 12)
    # No NaNs
    assert not np.isnan(feats).any()
