# tests/test_feature_extractor.py
import numpy as np
from eeg_knn_bhho.features import CiSSALightFeatureExtractor

def test_feature_extraction_shapes_and_values():
    n_epochs = 2
    n_comps = 1         # single IMF for simplicity
    n_samples = 128     # 1 second @128 Hz
    rng = np.random.RandomState(0)

    # Create a pure 10 Hz sine wave for 2 epochs, 1 component
    t = np.arange(n_samples) / 128.0
    sine = np.sin(2 * np.pi * 10 * t)
    X_cissa = np.stack([np.vstack([sine]), np.vstack([sine])], axis=0)
    # X_cissa shape: (2 epochs, 1 comp, 128 samples)

    bands = [(4, 8), (8, 13)]  # theta and alpha
    extractor = CiSSALightFeatureExtractor(sfreq=128.0, bands=bands, ds=4, nperseg=64, n_jobs=1)
    feats = extractor.transform(X_cissa)

    # Features per IMF = len(bands)=2 + 4(time) + 3(hjorth) + 3(entropy) + 2(ratios) = 14
    assert feats.shape == (2, 1 * 14)

    # Since it’s a pure 10 Hz sine, alpha power >> theta power, so ratio_theta_alpha ≈ 0,
    # ratio_beta_alpha should also be near zero (no beta activity).
    ratio_idx = len(bands) + 4 + 3 + 3  # index of first spectral_ratio in vector
    for epoch_feats in feats:
        theta_alpha = epoch_feats[ratio_idx]
        beta_alpha  = epoch_feats[ratio_idx + 1]
        assert theta_alpha < 0.1
        assert beta_alpha < 0.1
