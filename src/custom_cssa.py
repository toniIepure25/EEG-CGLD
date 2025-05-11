import numpy as np
from sklearn.utils.extmath import randomized_svd
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit
from joblib import Parallel, delayed


def circulant_embed(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Embed a 1D signal into a circulant trajectory matrix for C-SSA.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Input time series.
    window_size : int
        Embedding dimension L.

    Returns
    -------
    X : np.ndarray, shape (L, n_samples)
        Circulant-embedded matrix.
    """
    N = signal.shape[0]
    # Extend the signal to wrap around
    c = np.concatenate([signal, signal[:window_size - 1]])
    # Sliding windows of length L, total N windows
    X_windows = sliding_window_view(c, window_size)
    X = X_windows[:N]      # shape (N, L)
    return X.T            # shape (L, N)


@njit
def hankelize_numba(X_elem, N, L, K):
    """
    Numba-optimized Hankelization (diagonal averaging).
    """
    comp = np.empty(N, dtype=np.float64)
    for k in range(N):
        total = 0.0
        count = 0
        offset = k - (L - 1)
        for i in range(L):
            j = offset + i
            if 0 <= j < K:
                total += X_elem[L - 1 - i, j]
                count += 1
        comp[k] = total / count
    return comp


def ssa_1d_cssa(
    signal: np.ndarray,
    window_size: int = 30,
    energy_thresh: float = 0.90,
    max_comps: int = 10
) -> np.ndarray:
    """
    Circulant SSA with energy-based component selection.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    window_size : int
        Embedding dimension L.
    energy_thresh : float
        Fraction of total singular values energy to retain.
    max_comps : int
        Maximum number of components to compute.

    Returns
    -------
    comps : np.ndarray, shape (k, n_samples)
        Top k IMFs where k is determined by energy_thresh.
    """
    N = signal.shape[0]
    L = window_size
    # 1) Circulant embedding
    X = circulant_embed(signal, window_size=L)  # shape (L, N)

    # 2) Truncated SVD
    U, S, Vt = randomized_svd(
        X, n_components=max_comps, n_iter=5, random_state=0
    )

    # 3) Energy-based k selection
    energy = np.cumsum(S) / np.sum(S)
    k = int(np.searchsorted(energy, energy_thresh) + 1)

    # 4) Reconstruct & hankelize each component
    comps = np.empty((k, N), dtype=np.float64)
    for i in range(k):
        Xc = S[i] * np.outer(U[:, i], Vt[i])  # rank-1
        comps[i] = hankelize_numba(Xc, N, L, N)
    return comps


def decompose_epoch_fast(
    epoch: np.ndarray,
    window_size: int = 30,
    energy_thresh: float = 0.90,
    max_comps: int = 10,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Apply C-SSA to each channel in parallel and stack IMFs.

    Parameters
    ----------
    epoch : np.ndarray, shape (n_channels, n_samples)
    window_size : int
    energy_thresh : float
    max_comps : int
    n_jobs : int

    Returns
    -------
    stacked : np.ndarray, shape (sum(k_i), n_samples)
        Where k_i varies per channel.
    """
    def _process(ch):
        return ssa_1d_cssa(
            signal=ch,
            window_size=window_size,
            energy_thresh=energy_thresh,
            max_comps=max_comps
        )

    # Parallel decomposition
    results = Parallel(n_jobs=n_jobs)(delayed(_process)(ch) for ch in epoch)
    return np.vstack(results)  # shape (total_imfs, n_samples)
