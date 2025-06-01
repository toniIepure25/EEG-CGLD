# tests/test_cissa_decomposer.py
import numpy as np
from eeg_knn_bhho.decomposition import CISSADecomposer

def test_cissa_decomposer_sine():
    # Create a pure sine wave at 5 Hz, sampled at 128 Hz, length=128 samples => 1 second
    sfreq = 128.0
    t = np.arange(128) / sfreq
    sine = np.sin(2 * np.pi * 5 * t)
    # Form a 2-channel epoch: both channels identical
    epoch = np.vstack([sine, sine])  # shape (2,128)
    X = np.stack([epoch, epoch])     # 2 epochs, each (2,128)
    decomposer = CISSADecomposer(n_imfs=2, n_jobs=1)
    X_decomp = decomposer.transform(X) 
    # Output shape should be (2 epochs, 2 channels * 2 imfs = 4, 128 samples)
    assert X_decomp.shape == (2, 4, 128)
    # For each IMF, one should capture the 5 Hz component. Check that the FFT of IMF0 has a peak near bin=5 Hz.
    imf0 = X_decomp[0, 0]  # first epoch, first IMF
    freqs = np.fft.rfftfreq(128, 1/sfreq)
    mag = np.abs(np.fft.rfft(imf0))
    peak_idx = np.argmax(mag)
    # Expect peak_idx ~ 5 Hz (i.e., bin index where freq≈5)
    assert abs(freqs[peak_idx] - 5) < 1.0  # within 1 Hz
