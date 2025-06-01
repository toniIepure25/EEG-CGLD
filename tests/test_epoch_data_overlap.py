# tests/test_epoch_data_overlap.py
import numpy as np
from eeg_knn_bhho.data_loading import epoch_data_overlap

def test_epoch_data_overlap_basic():
    # Create a 2-channel signal of 50 samples at 10 Hz (so epoch_length=2s => 20 samples, overlap=0.5 => stride=10)
    data = np.vstack([np.arange(50), np.arange(50)])  # shape (2,50)
    epochs = epoch_data_overlap(data, sfreq=10.0, epoch_length=2.0, overlap=0.5)
    # Should get starts at 0, 10, 20, 30 => 4 epochs, each (2,20)
    assert epochs.shape == (4, 2, 20)
    # First epoch should equal data[:, 0:20]
    np.testing.assert_array_equal(epochs[0], data[:, 0:20])
