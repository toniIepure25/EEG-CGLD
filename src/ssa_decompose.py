import numpy as np
# from pyssa import ssa

def decompose_epoch(epoch, L=30, n_imfs=3):
    """
    Apply Singular Spectrum Analysis (SSA) to a single EEG epoch.

    Parameters
    ----------
    epoch : np.ndarray
        Shape (n_channels, n_samples)
    L : int
        Window size for SSA decomposition
    n_imfs : int
        Number of components (IMFs) to retain per channel

    Returns
    -------
    imf_features : np.ndarray
        Array of shape (n_channels * n_imfs, n_samples)
    """
    all_imfs = []
    for ch in epoch:
        result = ssa(ch, L=L)
        # Take first `n_imfs` components
        comp = result.reconstruct(range(n_imfs))
        all_imfs.append(comp)
    return np.vstack(all_imfs)


def decompose_epochs(X, L=30, n_imfs=3):
    """
    Apply SSA to all EEG epochs.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_epochs, n_channels, n_samples)
    L : int
        SSA window length
    n_imfs : int
        Number of components per channel to keep

    Returns
    -------
    X_ssa : np.ndarray
        Decomposed components, shape (n_epochs, n_channels * n_imfs, n_samples)
    """
    return np.array([decompose_epoch(epoch, L=L, n_imfs=n_imfs) for epoch in X])
