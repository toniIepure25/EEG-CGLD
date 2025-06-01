# sanity_check.py
import os
import numpy as np
from eeg_knn_bhho.data_loading import load_mat_epochs, load_stew_epochs
from eeg_knn_bhho.preprocessing import BandpassNotchFilter, EpochNormalizer

def main():
    # Adjust these paths to your environment
    mat_dir = os.environ.get("PROJECT_DIR", ".") + "/data/raw/MAT"
    stew_dir = os.environ.get("PROJECT_DIR", ".") + "/data/raw/STEW"

    print("Loading MAT epochs...")
    X_mat, y_mat = load_mat_epochs(mat_dir, epoch_length=1.0, resample_sfreq=128.0, overlap=0.5)
    print(f"MAT: X_mat shape = {X_mat.shape}, y_mat distribution = {np.bincount(y_mat)}")

    print("Loading STEW epochs...")
    X_stew, y_stew = load_stew_epochs(stew_dir, epoch_length=1.0, sfreq=256.0, overlap=0.5)
    print(f"STEW: X_stew shape = {X_stew.shape}, y_stew distribution = {np.bincount(y_stew)}")

    # Apply preprocessing to the first 5 epochs of MAT
    bp_filter = BandpassNotchFilter(
        band_low=1.0, band_high=40.0,
        notch_freq=50.0, notch_Q=30.0,
        sfreq=128.0, order=5
    )
    normalizer = EpochNormalizer()
    X_sample = X_mat[:5]  # take 5 epochs
    print("Applying Bandpass+Notch filter on 5 MAT epochs...")
    X_filt = bp_filter.transform(X_sample)
    print("Bandpass+Notch output shape:", X_filt.shape)
    print("Applying normalization on filtered epochs...")
    X_norm = normalizer.transform(X_filt)
    print("Normalized shape:", X_norm.shape)
    # Check no NaNs
    print("Any NaNs in normalized data?", np.isnan(X_norm).any())

if __name__ == "__main__":
    main()
